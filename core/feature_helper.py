import numpy as np
import math
from typing import Tuple , Optional

from vio_pipeline_v2.core.feature_database import FeatureDatabase
from vio_pipeline_v2.core.feature import Feature



class FeatureHelper:
    """
    Contains helper functions for features.
    These functions only depend on feature and the feature database.
    """

    @staticmethod
    def compute_disparity_two_frames(db: FeatureDatabase, 
                                     time0: float, 
                                     time1: float) -> Tuple[float, float, int]:
        """
        Computes the disparity between common features in the two specific frames.
        
        :param db: Feature database object
        :param time0: First camera frame timestamp
        :param time1: Second camera frame timestamp
        :return: Tuple containing (disp_mean, disp_var, total_feats)
        """
        
        # Get features seen from the first image
        # In Python we assume features_containing returns a list of Feature objects
        feats0 = db.features_containing(time0, remove=False, skip_deleted=True)

        disparities = []

        for feat in feats0:
            # Iterate through each camera's timeline in this feature
            for cam_id, timestamps in feat.timestamps.items():
                
                # Check if both timestamps exist in this camera's observations
                # We can use direct list lookup or iteration.
                # Assuming simple list scan for correctness (or set intersection for speed if huge)
                
                try:
                    idx0 = timestamps.index(time0)
                    idx1 = timestamps.index(time1)
                except ValueError:
                    # One or both timestamps not found for this camera
                    continue

                # Calculate disparity using raw UV coordinates
                # uvs is a list of numpy arrays or vectors
                uv0 = feat.uvs[cam_id][idx0]
                uv1 = feat.uvs[cam_id][idx1]
                
                # Compute Euclidean norm (L2 norm)
                disp = np.linalg.norm(uv1 - uv0)
                disparities.append(disp)

        # Statistical calculations
        total_feats = len(disparities)
        
        if total_feats < 2:
            return -1.0, -1.0, 0

        # Calculate Mean and Standard Deviation (Sample Std Dev)
        # Using ddof=1 to match C++ (N-1 denominator)
        disp_mean = np.mean(disparities)
        disp_var = np.std(disparities, ddof=1) 

        return float(disp_mean), float(disp_var), int(total_feats)

    @staticmethod
    def compute_disparity(db: FeatureDatabase, 
                          newest_time: float = -1, 
                          oldest_time: float = -1) -> Tuple[float, float, int]:
        """
        Computes the disparity over all features within a time window.
        
        :param db: Feature database object
        :param newest_time: Only compute disparity for measurements older than this (-1 to disable)
        :param oldest_time: Only compute disparity for measurements newer than this (-1 to disable)
        :return: Tuple containing (disp_mean, disp_var, total_feats)
        """

        disparities = []
        
        # Access internal dictionary directly or via getter
        internal_data = db.get_internal_data()

        for feat_id, feat in internal_data.items():
            
            for cam_id, timestamps in feat.timestamps.items():
                
                # Skip if only one observation (cannot compute disparity)
                if len(timestamps) < 2:
                    continue

                found0 = False
                found1 = False
                uv0 = None
                uv1 = None
                
                # Timestamps are typically monotonic (append-only), so we iterate
                for idx, time in enumerate(timestamps):
                    
                    # Find the "start" point (oldest valid point)
                    if (oldest_time == -1 or time > oldest_time) and not found0:
                        uv0 = feat.uvs[cam_id][idx]
                        found0 = True
                        continue
                    
                    # Find the "end" point (newest valid point)
                    # We keep updating uv1 as long as it satisfies the condition
                    # This logic finds the *last* valid point before newest_time
                    if (newest_time == -1 or time < newest_time) and found0:
                        uv1 = feat.uvs[cam_id][idx]
                        found1 = True
                        continue
                
                # If we found both valid points, compute disparity
                if found0 and found1:
                    disp = np.linalg.norm(uv1 - uv0)
                    disparities.append(disp)

        # Statistical calculations
        total_feats = len(disparities)
        
        if total_feats < 2:
            return -1.0, -1.0, 0

        disp_mean = np.mean(disparities)
        disp_var = np.std(disparities, ddof=1)

        return float(disp_mean), float(disp_var), int(total_feats)
