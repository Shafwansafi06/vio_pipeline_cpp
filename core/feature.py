import time
# from typing import assert_never
import numpy as np

class Feature:
    def __init__(self):
        self.featid =-1
        self.timestamps = {}
        self.uvs = {}
        self.uvs_norm = {}
        self.p_FinG = np.zeros(3)
        self.anchor_cam_id = -1
        self.anchor_clone_timestamp = 0.0
        self.p_FinA = np.zeros(3)
        self.to_delete= False

    def num_measurements(self):
        return sum(len(times) for times in self.timestamps.values())

    def clean_old_measurements(self, valid_times):

        valid_set = set(valid_times)    
        for cam_id in list(self.timestamps.keys()):
            ts_list = self.timestamps[cam_id]
            uvs_list = self.uvs[cam_id]
            uvn_list = self.uvs_norm[cam_id]

            assert len(ts_list) == len(uvs_list) == len(uvn_list)

            filtered_data = [(ts, uv, uvn) for ts, uv, uvn in zip(ts_list, uvs_list, uvn_list) if ts in valid_set]

            if not filtered_data:
                self.timestamps[cam_id] = []
                self.uvs[cam_id] = []
                self.uvs_norm[cam_id] = []
            else:
                ts_new, uv_new, uvn_new = zip(*filtered_data)
                self.timestamps[cam_id] = list(ts_new)
                self.uvs[cam_id] = list(uv_new)
                self.uvs_norm[cam_id] = list(uvn_new)
    
    def  clean_invalid_measurements(self, invalid_times):

        invalid_set = set(invalid_times)
        for cam_id in list(self.timestamps.keys()):
            ts_list = self.timestamps[cam_id]
            uv_list = self.uvs[cam_id]
            uvn_list = self.uvs_norm[cam_id]

            assert len(ts_list) == len(uv_list) == len(uvn_list)

            filtered_data = [(ts, uv, uvn) for ts, uv, uvn in zip(ts_list, uv_list, uvn_list) if ts not in invalid_set] 

            if not filtered_data:
                self.timestamps[cam_id] = []
                self.uvs[cam_id] = []
                self.uvs_norm[cam_id] = []
            else:
                ts_new, uv_new, uvn_new = zip(*filtered_data)
                self.timestamps[cam_id] = list(ts_new)
                self.uvs[cam_id] = list(uv_new)
                self.uvs_norm[cam_id] = list(uvn_new)
        
    def clear_older_measurements(self, timestamp):

        for cam_id in list(self.timestamps.keys()):
            ts_list = self.timestamps[cam_id]
            uv_list = self.uvs[cam_id]
            uvn_list = self.uvs_norm[cam_id]

            assert len(ts_list) == len(uv_list) == len(uvn_list)

            filtered_data = [(ts, uv, uvn) for ts, uv, uvn in zip(ts_list, uv_list, uvn_list) if ts > timestamp] 

            if not filtered_data:
                self.timestamps[cam_id] = []
                self.uvs[cam_id] = []
                self.uvs_norm[cam_id] = []
            else:
                ts_new, uv_new, uvn_new = zip(*filtered_data)
                self.timestamps[cam_id] = list(ts_new)
                self.uvs[cam_id] = list(uv_new)
                self.uvs_norm[cam_id] = list(uvn_new)


