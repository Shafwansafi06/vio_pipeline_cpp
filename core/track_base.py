import cv2
import numpy as np
import threading
import copy
from typing import Dict, List, Optional, Tuple

# Assumed imports based on previous context
from vio_pipeline_v2.core.cam_base import CamBase
from vio_pipeline_v2.core.feature import Feature
from vio_pipeline_v2.core.feature_database import FeatureDatabase

class TrackBase:
    """
    Base class for feature trackers.
    Manages camera calibration, feature database, and visualization.
    """
    def __init__(self, cameras: Dict[int, CamBase], numfeats: int, numaruco: int, stereo: bool, histmethod: int):
        
        self.camera_calib = cameras
        self.database = FeatureDatabase()
        self.num_features = numfeats
        self.use_stereo = stereo
        self.histogram_method = histmethod

        # Our current feature ID should be larger then the number of aruco tags we have (each has 4 corners)
        self.currid = 4 * numaruco + 1

        # Create our mutex array based on the number of cameras we have
        # List of Reentrant Locks
        self.mtx_feeds = [threading.RLock() for _ in range(len(cameras))]
        
        # Mutex for last variables (caching for display)
        self.mtx_last_vars = threading.RLock()

        # State variables for visualization (To be populated by derived classes)
        # Maps Camera ID -> Data
        self.img_last: Dict[int, np.ndarray] = {}
        self.img_mask_last: Dict[int, np.ndarray] = {}
        self.pts_last: Dict[int, List[cv2.KeyPoint]] = {} # List of cv2.KeyPoint
        self.ids_last: Dict[int, List[int]] = {}
    
    
    def get_num_features(self) -> int:
        """Getter method for number of active features"""
        return self.num_features

    def set_num_features(self, num_features: int):
        """Setter method for number of active features"""
        self.num_features = num_features

    def get_feature_database(self) -> FeatureDatabase:
        """
        Get the feature database with all the track information.
        Returns FeatureDatabase pointer that one can query for features.
        """
        return self.database

    def get_last_obs(self) -> Dict[int, List[cv2.KeyPoint]]:
        """Getter method for active features in the last frame (observations per camera)"""
        with self.mtx_last_vars:
            return self.pts_last

    def get_last_ids(self) -> Dict[int, List[int]]:
        """Getter method for active features in the last frame (ids per camera)"""
        with self.mtx_last_vars:
            return self.ids_last

    def display_active(self, img_out: np.ndarray, r1: int, g1: int, b1: int, r2: int, g2: int, b2: int, overlay: str) -> np.ndarray:
        """
        Draws the currently tracked active features on the image.
        """
        
        # Cache the images to prevent other threads from editing while we viz
        with self.mtx_last_vars:
            # Shallow copy of the dictionaries is usually sufficient if the arrays are replaced, not modified in place
            img_last_cache = self.img_last.copy()
            img_mask_last_cache = self.img_mask_last.copy()
            pts_last_cache = self.pts_last.copy()

        # Get the largest width and height
        max_width = -1
        max_height = -1
        for img in img_last_cache.values():
            h, w = img.shape[:2]
            if max_width < w: max_width = w
            if max_height < h: max_height = h

        # Return if we didn't have a last image
        if not img_last_cache or max_width == -1 or max_height == -1:
            return img_out

        # If the image is "small" thus we should use smaller display codes
        is_small = (min(max_width, max_height) < 400)

        # If the image is "new" or size mismatch, reallocate
        # Note: img_out might be None initially
        current_rows = img_out.shape[0] if img_out is not None else 0
        current_cols = img_out.shape[1] if img_out is not None else 0
        expected_cols = len(img_last_cache) * max_width

        image_new = (expected_cols != current_cols or max_height != current_rows)

        if image_new:
            img_out = np.zeros((max_height, expected_cols, 3), dtype=np.uint8)

        # Loop through each image, and draw
        # Sort keys to ensure consistent order (CAM 0, CAM 1...)
        sorted_cam_ids = sorted(img_last_cache.keys())
        
        for index_cam, cam_id in enumerate(sorted_cam_ids):
            # Select the subset of the image
            if image_new:
                # Convert grayscale to RGB
                img_temp = cv2.cvtColor(img_last_cache[cam_id], cv2.COLOR_GRAY2RGB)
            else:
                # ROI (Region of Interest)
                x_start = max_width * index_cam
                img_temp = img_out[0:max_height, x_start:x_start+max_width]
            
            # Draw, loop through all keypoints
            if cam_id in pts_last_cache:
                for i, kp in enumerate(pts_last_cache[cam_id]):
                    pt_l = (int(kp.pt[0]), int(kp.pt[1]))
                    
                    # Draw the extracted points
                    radius = 1 if is_small else 2
                    cv2.circle(img_temp, pt_l, radius, (b1, g1, r1), cv2.FILLED) # OpenCV uses BGR
                    
                    # Draw rectangle around the point
                    pt_l_top = (pt_l[0] - 3, pt_l[1] - 3)
                    pt_l_bot = (pt_l[0] + 3, pt_l[1] + 3)
                    cv2.rectangle(img_temp, pt_l_top, pt_l_bot, (b2, g2, r2), 1)

            # Draw what camera this is
            txtpt = (10, 30) if is_small else (30, 60)
            font_scale = 1.5 if is_small else 3.0
            
            text = f"CAM:{cam_id}" if overlay == "" else overlay
            color = (0, 255, 0) if overlay == "" else (0, 0, 255) # Green or Red
            
            cv2.putText(img_temp, text, txtpt, cv2.FONT_HERSHEY_COMPLEX_SMALL, font_scale, color, 3)

            # Overlay the mask
            if cam_id in img_mask_last_cache:
                mask_src = img_mask_last_cache[cam_id]
                # Create a red mask overlay
                mask_colored = np.zeros_like(img_temp)
                mask_colored[:, :, 2] = mask_src # Set Red channel
                
                # Add weighted overlay
                cv2.addWeighted(mask_colored, 0.1, img_temp, 1.0, 0.0, img_temp)

            # Replace the output image section
            x_start = max_width * index_cam
            h, w = img_last_cache[cam_id].shape[:2]
            img_out[0:h, x_start:x_start+w] = img_temp

        return img_out

    def display_history(self, img_out: np.ndarray, r1: int, g1: int, b1: int, r2: int, g2: int, b2: int, 
                        highlighted: List[int], overlay: str) -> np.ndarray:
        """
        Draws the historical tracks of features on the image.
        """
        
        # Cache variables
        with self.mtx_last_vars:
            img_last_cache = self.img_last.copy()
            img_mask_last_cache = self.img_mask_last.copy()
            pts_last_cache = self.pts_last.copy()
            ids_last_cache = self.ids_last.copy()

        # Dimensions check
        max_width = -1
        max_height = -1
        for img in img_last_cache.values():
            h, w = img.shape[:2]
            if max_width < w: max_width = w
            if max_height < h: max_height = h

        if not img_last_cache or max_width == -1 or max_height == -1:
            return img_out

        is_small = (min(max_width, max_height) < 400)

        # Image Allocation logic
        current_rows = img_out.shape[0] if img_out is not None else 0
        current_cols = img_out.shape[1] if img_out is not None else 0
        expected_cols = len(img_last_cache) * max_width

        image_new = (expected_cols != current_cols or max_height != current_rows)

        if image_new:
            img_out = np.zeros((max_height, expected_cols, 3), dtype=np.uint8)

        maxtracks = 50
        sorted_cam_ids = sorted(img_last_cache.keys())

        for index_cam, cam_id in enumerate(sorted_cam_ids):
            # Prep Image
            if image_new:
                img_temp = cv2.cvtColor(img_last_cache[cam_id], cv2.COLOR_GRAY2RGB)
            else:
                x_start = max_width * index_cam
                img_temp = img_out[0:max_height, x_start:x_start+max_width]

            # Draw tracks
            if cam_id in ids_last_cache:
                for i in range(len(ids_last_cache[cam_id])):
                    feat_id = ids_last_cache[cam_id][i]
                    
                    # Highlighted Feature Box
                    if feat_id in highlighted:
                        pt_c = pts_last_cache[cam_id][i].pt
                        pt_c_int = (int(pt_c[0]), int(pt_c[1]))
                        offset = 3 if is_small else 5
                        
                        pt_l_top = (pt_c_int[0] - offset, pt_c_int[1] - offset)
                        pt_l_bot = (pt_c_int[0] + offset, pt_c_int[1] + offset)
                        
                        cv2.rectangle(img_temp, pt_l_top, pt_l_bot, (0, 255, 0), 1)
                        radius = 1 if is_small else 2
                        cv2.circle(img_temp, pt_c_int, radius, (0, 255, 0), cv2.FILLED)

                    # Get Feature History
                    # We create a dummy feature to hold the clone data
                    feat = Feature()
                    if not self.database.get_feature_clone(feat_id, feat):
                        continue
                    
                    if cam_id not in feat.uvs or not feat.uvs[cam_id] or feat.to_delete:
                        continue
                    
                    # Draw History (Iterate backwards)
                    # Note: Python's range(start, stop, step). 
                    # C++: size()-1 down to >0. 
                    hist_size = len(feat.uvs[cam_id])
                    
                    for z in range(hist_size - 1, 0, -1):
                        if (hist_size - z) > maxtracks:
                            break
                        
                        # 

                        # Color Gradient Calculation
                        is_stereo = (len(feat.uvs) > 1)
                        ratio = (1.0 * z) / hist_size
                        
                        # Note: OpenCV uses BGR. C++ params are R, G, B.
                        # Logic adapted from C++:
                        # int color_r = (is_stereo ? b2 : r2) - (int)(1.0 * (is_stereo ? b1 : r1) / size * z);
                        
                        c_r1 = b1 if is_stereo else r1
                        c_r2 = b2 if is_stereo else r2
                        color_r = int(c_r2 - (c_r1 / hist_size * z))
                        
                        c_g1 = r1 if is_stereo else g1 # C++ Logic seems to swap channels for stereo
                        c_g2 = r2 if is_stereo else g2
                        color_g = int(c_g2 - (c_g1 / hist_size * z))
                        
                        c_b1 = g1 if is_stereo else b1
                        c_b2 = g2 if is_stereo else b2
                        color_b = int(c_b2 - (c_b1 / hist_size * z))
                        
                        color = (color_b, color_g, color_r) # BGR for OpenCV

                        # Draw Point
                        uv_curr = feat.uvs[cam_id][z]
                        pt_c = (int(uv_curr[0]), int(uv_curr[1]))
                        
                        radius = 1 if is_small else 2
                        cv2.circle(img_temp, pt_c, radius, color, cv2.FILLED)

                        # Draw Line to Next Point
                        if z + 1 < hist_size:
                            uv_next = feat.uvs[cam_id][z+1]
                            pt_n = (int(uv_next[0]), int(uv_next[1]))
                            cv2.line(img_temp, pt_c, pt_n, color)

            # Draw Text
            txtpt = (10, 30) if is_small else (30, 60)
            font_scale = 1.5 if is_small else 3.0
            
            text = f"CAM:{cam_id}" if overlay == "" else overlay
            color = (0, 255, 0) if overlay == "" else (0, 0, 255)
            
            cv2.putText(img_temp, text, txtpt, cv2.FONT_HERSHEY_COMPLEX_SMALL, font_scale, color, 3)

            # Overlay Mask
            if cam_id in img_mask_last_cache:
                mask_src = img_mask_last_cache[cam_id]
                mask_colored = np.zeros_like(img_temp)
                mask_colored[:, :, 2] = mask_src 
                cv2.addWeighted(mask_colored, 0.1, img_temp, 1.0, 0.0, img_temp)

            # Copy to output
            x_start = max_width * index_cam
            h, w = img_last_cache[cam_id].shape[:2]
            img_out[0:h, x_start:x_start+w] = img_temp

        return img_out

    def change_feat_id(self, id_old: int, id_new: int):
        """
        Updates the ID of a feature in the database and the current tracking state.
        """
        # Update Database
        # Note: get_internal_data() returns the dictionary directly
        db_data = self.database.get_internal_data()
        
        if id_old in db_data:
            feat = db_data[id_old]
            del db_data[id_old]
            feat.featid = id_new
            db_data[id_new] = feat
        
        # Update current track IDs (ids_last)
        for cam_id, id_list in self.ids_last.items():
            for i in range(len(id_list)):
                if id_list[i] == id_old:
                    id_list[i] = id_new