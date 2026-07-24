import re
import threading
import time
from xml.sax.handler import feature_validation
import numpy as np
import copy
from typing import List, Dict, Optional

from vio_pipeline_v2.core.feature import Feature

class FeatureDatabase:

    def __init__(self):
        self.features_idlookup: Dict[int, Feature] = {}
        self.mtx = threading.RLock()

    def get_internal_data(self):
        return self.features_idlookup
    
    def get_feature(self, feat_id, remove: bool = False) -> Optional[Feature]:
        with self.mtx:
            if feat_id in self.features_idlookup:
                temp = self.features_idlookup[feat_id]
                if remove:
                    del self.features_idlookup[feat_id]
                return temp
            else:
                return None
    
    # def get_feature_ids(self, feat_id , remove: bool = False) -> List[int]:
    #     with self.mtx:
    #         if feat_id in self.features_idlookup:
    #             temp= self.features_idlookup[feat_id]
    #             if remove:
    #                 del self.features_idlookup[feat_id]
    #             return temp
    #         else:
    #             return None
    def get_feature_clone(self, feat_id, feat_out):
        with self.mtx:
            if feat_id not in self.features_idlookup:
                return False

            temp = self.features_idlookup[feat_id]
            feat_out.featid = temp.featid
            feat_out.to_delete = temp.to_delete
            feat_out.uvs = copy.deepcopy(temp.uvs)
            feat_out.uvs_norm = copy.deepcopy(temp.uvs_norm)
            feat_out.timestamps = copy.deepcopy(temp.timestamps)

            feat_out.anchor_cam_id = temp.anchor_cam_id
            feat_out.anchor_clone_timestamp = temp.anchor_clone_timestamp
            feat_out.p_FinA = copy.deepcopy(temp.p_FinA)

            feat_out.p_FinG = copy.deepcopy(temp.p_FinG)
            return True
    
    def update_feature(self, feat_id, timestamp, cam_id, u, v, u_n, v_n):

        with self.mtx:
            if feat_id in self.features_idlookup:
                feat = self.features_idlookup[feat_id]

                if cam_id not in feat.timestamps:
                    feat.timestamps[cam_id] = []
                    feat.uvs[cam_id] = []
                    feat.uvs_norm[cam_id] = []
                
                feat.uvs[cam_id].append(np.array([u, v]))
                feat.uvs_norm[cam_id].append(np.array([u_n, v_n]))
                feat.timestamps[cam_id].append(timestamp)
                
                return
            feat = Feature()
            feat.featid = feat_id
            feat.timestamps[cam_id] = [timestamp]
            feat.uvs[cam_id] = [np.array([u, v], dtype=np.float32)]
            feat.uvs_norm[cam_id] = [np.array([u_n, v_n], dtype=np.float32)]
            self.features_idlookup[feat_id] = feat
    
    def features_not_containing_newer(self, timestamp, remove, skip_deleted):

        feats_old = []
        keys_to_remove = []

        with self.mtx:
            for feat_id, feat in self.features_idlookup.items():
                if skip_deleted and feat.to_delete:
                    continue

                has_newer_measurement = False

                for cam_id, times in feat.timestamps.items():
                    if times and times[-1] >= timestamp:
                        has_newer_measurement = True
                        break
                if not has_newer_measurement:
                    feats_old.append(feat)
                    if remove:
                        keys_to_remove.append(feat_id)
            
            for key in keys_to_remove:
                del self.features_idlookup[key]
        return feats_old
    
    def features_containing_older(self,timestamp, remove, skip_deleted):

        feats_old = []
        keys_to_remove = []

        with self.mtx:
            for feat_id, feat in self.features_idlookup.items():
                if skip_deleted and feat.to_delete:
                    continue

                found_containing_older = False

                for cam_id, times in feat.timestamps.items():
                    if times and times[0] < timestamp:
                        found_containing_older = True
                        break
                if found_containing_older:
                    feats_old.append(feat)
                    if remove:
                        keys_to_remove.append(feat_id)  
            for key in keys_to_remove:
                del self.features_idlookup[key]
        return feats_old
    
    def features_containing(self, timestamp, remove, skip_deleted):

        feat_has_timestamp = []
        keys_to_remove = []

        with self.mtx:
            for feat_id, feat in self.features_idlookup.items():
                if skip_deleted and feat.to_delete:
                    continue

                has_timestamps = False

                for cam_id, times in feat.timestamps.items():
                    if timestamp in times:
                        has_timestamps = True
                        break
                if has_timestamps:
                    feat_has_timestamp.append(feat)
                    if remove:
                        keys_to_remove.append(feat_id)  
            for key in keys_to_remove:
                del self.features_idlookup[key]
        return feat_has_timestamp
    
    def cleanup(self):
        keys_to_remove =[]
        with self.mtx:
            for feat_id, feat in self.features_idlookup.items():
                if feat.to_delete:
                    keys_to_remove.append(feat_id)
            for key in keys_to_remove:
                del self.features_idlookup[key]

    def cleanup_measurements(self, timestamp):
        keys_to_remove =[]
        with self.mtx:
            for feat_id, feat in self.features_idlookup.items():
                feat.clear_older_measurements(timestamp)
                
                ct_meas = sum(len(times) for times in feat.timestamps.values())

                if ct_meas <1:
                    keys_to_remove.append(feat_id)
            
            for key in keys_to_remove:
                del self.features_idlookup[key]

    def cleanup_measurements_exact(self, timestamp):
        keys_to_remove = []
        timestamp_list =[timestamp]

        with self.mtx:
            for feat_id, feat in self.features_idlookup.items():
                feat.clean_invalid_measurements(timestamp_list)

                ct_meas = sum(len(times) for times in feat.timestamps.values())

                if ct_meas < 1:
                    keys_to_remove.append(feat_id)

            for key in keys_to_remove:
                del self.features_idlookup[key]

    def get_oldest_timestamp(self):
        oldest_time = -1.0

        with self.mtx:
            for feat in self.features_idlookup.values():
                for times in feat.timestamps.values():
                    if times:
                        t = times[0]
                        if oldest_time ==-1.0 or t < oldest_time:
                            oldest_time = t
        return oldest_time
    
    def append_new_measurements(self, database):
        with self.mtx:
            other_data = database.get_internal_data()
            for feat_id, other_feat in other_data.items():
                if feat_id in self.features_idlookup:
                    temp = self.features_idlookup[feat_id]

                    for cam_id, other_times in other_feat.timestamps.items():

                        if cam_id not in temp.timestamps:
                            temp.timestamps[cam_id] = copy.deepcopy(other_times)
                            temp.uvs[cam_id] = copy.deepcopy(other_feat.uvs[cam_id])
                            temp.uvs_norm[cam_id] = copy.deepcopy(other_feat.uvs_norm[cam_id])
                        else:

                            our_times = temp.timestamps[cam_id]
                            for i, t_val in enumerate(other_times):
                                if t_val not in our_times:
                                    temp.timestamps[cam_id].append(t_val)
                                    temp.uvs[cam_id].append(other_feat.uvs[cam_id][i])
                                    temp.uvs_norm[cam_id].append(other_feat.uvs_norm[cam_id][i])
                else:
                    temp = Feature()
                    temp.featid = other_feat.featid
                    temp.timestamps = copy.deepcopy(other_feat.timestamps)
                    temp.uvs = copy.deepcopy(other_feat.uvs)
                    temp.uvs_norm = copy.deepcopy(other_feat.uvs_norm)
                    self.features_idlookup[feat_id] = temp
