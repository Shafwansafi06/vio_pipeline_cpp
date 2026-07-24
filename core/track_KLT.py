import cv2
import numpy as np
import threading
import time
import math
from typing import List, Dict, Tuple, Optional

# Imports
from vio_pipeline_v2.core.track_base import TrackBase
from vio_pipeline_v2.core.cam_base import CamBase
from vio_pipeline_v2.core.sensor_data import CameraData


class TrackKLT(TrackBase):
    def __init__(self, cameras: Dict[int, CamBase], numfeats: int, numaruco: int, stereo: bool, histmethod: int,
                 fast_threshold: int, grid_x: int, grid_y: int, min_px_dist: int):

        super().__init__(cameras, numfeats, numaruco, stereo, histmethod)

        self.threshold = fast_threshold
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.min_px_dist = min_px_dist

        # KLT Parameters (match C++ defaults in TrackKLT.h)
        self.win_size = (15, 15)
        self.pyr_levels = 5

        # Storage for current frame data
        self.img_curr: Dict[int, np.ndarray] = {}
        self.img_pyramid_curr: Dict[int, List[np.ndarray]] = {}

        # Storage for last frame data (persisted between calls)
        self.img_pyramid_last: Dict[int, List[np.ndarray]] = {}

        # Initialize empty lists for each camera (match C++ default construction)
        for cam_id in cameras:
            self.pts_last[cam_id] = []
            self.ids_last[cam_id] = []

    def feed_new_camera(self, message: CameraData):
        """
        Main entry point for tracking.
        Matches C++ TrackKLT::feed_new_camera exactly.
        """
        # Error check
        if (not message.sensor_ids or
                len(message.sensor_ids) != len(message.images) or
                len(message.images) != len(message.masks)):
            print(f"[ERROR]: MESSAGE DATA SIZES DO NOT MATCH OR EMPTY!!!")
            print(f"[ERROR]:   - message.sensor_ids.size() = {len(message.sensor_ids)}")
            print(f"[ERROR]:   - message.images.size() = {len(message.images)}")
            print(f"[ERROR]:   - message.masks.size() = {len(message.masks)}")
            exit(1)

        # Preprocessing steps that we do not parallelize
        # NOTE: DO NOT PARALLELIZE THESE!
        rT1 = time.time()
        num_images = len(message.images)

        for msg_id in range(num_images):
            # Lock this data feed for this camera
            cam_id = message.sensor_ids[msg_id]
            with self.mtx_feeds[cam_id]:

                # Histogram equalize
                img = None
                if self.histogram_method == 1:  # HISTOGRAM
                    img = cv2.equalizeHist(message.images[msg_id])
                elif self.histogram_method == 2:  # CLAHE
                    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
                    img = clahe.apply(message.images[msg_id])
                else:
                    img = message.images[msg_id]

                # Extract image pyramid
                _, imgpyr = cv2.buildOpticalFlowPyramid(img, self.win_size, self.pyr_levels)

                # Save!
                self.img_curr[cam_id] = img
                self.img_pyramid_curr[cam_id] = imgpyr

        # Either call our stereo or monocular version
        if num_images == 1:
            self.feed_monocular(message, 0)
        elif num_images == 2 and self.use_stereo:
            self.feed_stereo(message, 0, 1)
        elif not self.use_stereo:
            # Parallel processing (sequential in Python due to GIL, but use threads for API consistency)
            threads = []
            for i in range(num_images):
                t = threading.Thread(target=self.feed_monocular, args=(message, i))
                threads.append(t)
                t.start()
            for t in threads:
                t.join()
        else:
            print(f"[ERROR]: invalid number of images passed {num_images}, we only support mono or stereo tracking")
            exit(1)

    def feed_monocular(self, message: CameraData, msg_id: int):
        """
        Process a new monocular image.
        Matches C++ TrackKLT::feed_monocular exactly.
        """
        # Lock this data feed for this camera
        cam_id = message.sensor_ids[msg_id]

        with self.mtx_feeds[cam_id]:
            rT1 = time.time()

            # Get our image objects for this image
            img = self.img_curr[cam_id]
            imgpyr = self.img_pyramid_curr[cam_id]
            mask = message.masks[msg_id]
            rT2 = time.time()

            # If we didn't have any successful tracks last time, just extract this time
            # This also handles the tracking initialization on the first call to this extractor
            if not self.pts_last[cam_id]:
                # Detect new features
                good_left = []
                good_ids_left = []
                self.perform_detection_monocular(imgpyr, mask, good_left, good_ids_left)
                # Save the current image and pyramid
                with self.mtx_last_vars:
                    self.img_last[cam_id] = img
                    self.img_pyramid_last[cam_id] = imgpyr
                    self.img_mask_last[cam_id] = mask
                    self.pts_last[cam_id] = good_left
                    self.ids_last[cam_id] = good_ids_left
                return

            # First we should make that the last images have enough features so we can do KLT
            # This will "top-off" our number of tracks so always have a constant number
            pts_before_detect = len(self.pts_last[cam_id])
            pts_left_old = list(self.pts_last[cam_id])
            ids_left_old = list(self.ids_last[cam_id])
            self.perform_detection_monocular(self.img_pyramid_last[cam_id], self.img_mask_last[cam_id],
                                             pts_left_old, ids_left_old)
            rT3 = time.time()

            # Our return success masks, and predicted new features
            # C++: std::vector<cv::KeyPoint> pts_left_new = pts_left_old;
            pts_left_new = _copy_keypoints(pts_left_old)

            # Lets track temporally
            mask_ll = self.perform_matching(self.img_pyramid_last[cam_id], imgpyr,
                                            pts_left_old, pts_left_new, cam_id, cam_id)
            assert len(pts_left_new) == len(ids_left_old)
            rT4 = time.time()

            # If any of our mask is empty, that means we didn't have enough to do ransac, so just return
            if not mask_ll:
                with self.mtx_last_vars:
                    self.img_last[cam_id] = img
                    self.img_pyramid_last[cam_id] = imgpyr
                    self.img_mask_last[cam_id] = mask
                    self.pts_last[cam_id] = []
                    self.ids_last[cam_id] = []
                print(f"[KLT-EXTRACTOR]: Failed to get enough points to do RANSAC, resetting.....")
                return

            # Get our "good tracks"
            good_left = []
            good_ids_left = []

            h, w = img.shape[:2]

            # Loop through all left points
            for i in range(len(pts_left_new)):
                # Ensure we do not have any bad KLT tracks (i.e., points are negative)
                pt = pts_left_new[i].pt
                if pt[0] < 0 or pt[1] < 0 or int(pt[0]) >= w or int(pt[1]) >= h:
                    continue
                # Check if it is in the mask
                # NOTE: mask has max value of 255 (white) if it should be
                if message.masks[msg_id][int(pt[1]), int(pt[0])] > 127:
                    continue
                # If it is a good track
                if mask_ll[i]:
                    good_left.append(pts_left_new[i])
                    good_ids_left.append(ids_left_old[i])

            # Update our feature database, with these new observations
            for i in range(len(good_left)):
                pt = good_left[i].pt
                npt_l = self.camera_calib[cam_id].undistort_cv(pt)
                self.database.update_feature(good_ids_left[i], message.timestamp, cam_id,
                                             pt[0], pt[1], npt_l[0], npt_l[1])

            # Move forward in time
            with self.mtx_last_vars:
                self.img_last[cam_id] = img
                self.img_pyramid_last[cam_id] = imgpyr
                self.img_mask_last[cam_id] = mask
                self.pts_last[cam_id] = good_left
                self.ids_last[cam_id] = good_ids_left

            rT5 = time.time()

    def feed_stereo(self, message: CameraData, msg_id_left: int, msg_id_right: int):
        """
        Process new stereo pair of images.
        Matches C++ TrackKLT::feed_stereo exactly.
        """
        # Lock this data feed for both cameras
        cam_id_left = message.sensor_ids[msg_id_left]
        cam_id_right = message.sensor_ids[msg_id_right]

        with self.mtx_feeds[cam_id_left]:
            with self.mtx_feeds[cam_id_right]:

                img_left = self.img_curr[cam_id_left]
                img_right = self.img_curr[cam_id_right]
                imgpyr_left = self.img_pyramid_curr[cam_id_left]
                imgpyr_right = self.img_pyramid_curr[cam_id_right]
                mask_left = message.masks[msg_id_left]
                mask_right = message.masks[msg_id_right]
                rT2 = time.time()

                # If we didn't have any successful tracks last time, just extract this time
                if not self.pts_last[cam_id_left] and not self.pts_last[cam_id_right]:
                    good_left = []
                    good_right = []
                    good_ids_left = []
                    good_ids_right = []
                    self.perform_detection_stereo(
                        imgpyr_left, imgpyr_right, mask_left, mask_right,
                        cam_id_left, cam_id_right,
                        good_left, good_right, good_ids_left, good_ids_right
                    )
                    with self.mtx_last_vars:
                        self.img_last[cam_id_left] = img_left
                        self.img_last[cam_id_right] = img_right
                        self.img_pyramid_last[cam_id_left] = imgpyr_left
                        self.img_pyramid_last[cam_id_right] = imgpyr_right
                        self.img_mask_last[cam_id_left] = mask_left
                        self.img_mask_last[cam_id_right] = mask_right
                        self.pts_last[cam_id_left] = good_left
                        self.pts_last[cam_id_right] = good_right
                        self.ids_last[cam_id_left] = good_ids_left
                        self.ids_last[cam_id_right] = good_ids_right
                    return

                # First we should make that the last images have enough features so we can do KLT
                pts_before_detect = len(self.pts_last[cam_id_left])
                pts_left_old = list(self.pts_last[cam_id_left])
                pts_right_old = list(self.pts_last[cam_id_right])
                ids_left_old = list(self.ids_last[cam_id_left])
                ids_right_old = list(self.ids_last[cam_id_right])

                self.perform_detection_stereo(
                    self.img_pyramid_last[cam_id_left],
                    self.img_pyramid_last[cam_id_right],
                    self.img_mask_last[cam_id_left],
                    self.img_mask_last[cam_id_right],
                    cam_id_left, cam_id_right,
                    pts_left_old, pts_right_old, ids_left_old, ids_right_old
                )
                rT3 = time.time()

                # Our return success masks, and predicted new features
                # C++: std::vector<cv::KeyPoint> pts_left_new = pts_left_old; (copy)
                pts_left_new = _copy_keypoints(pts_left_old)
                pts_right_new = _copy_keypoints(pts_right_old)

                # Lets track temporally (C++ does parallel_for_ on left and right)
                t_left = threading.Thread(
                    target=lambda: self._perform_matching_into(
                        self.img_pyramid_last[cam_id_left], imgpyr_left,
                        pts_left_old, pts_left_new, cam_id_left, cam_id_left, '_mask_ll'))
                t_right = threading.Thread(
                    target=lambda: self._perform_matching_into(
                        self.img_pyramid_last[cam_id_right], imgpyr_right,
                        pts_right_old, pts_right_new, cam_id_right, cam_id_right, '_mask_rr'))

                self._mask_ll = []
                self._mask_rr = []
                t_left.start()
                t_right.start()
                t_left.join()
                t_right.join()
                mask_ll = self._mask_ll
                mask_rr = self._mask_rr

                rT4 = time.time()

                # left to right matching
                # TODO: we should probably still do this to reject outliers
                rT5 = time.time()

                # If any of our masks are empty, that means we didn't have enough to do ransac
                if not mask_ll and not mask_rr:
                    with self.mtx_last_vars:
                        self.img_last[cam_id_left] = img_left
                        self.img_last[cam_id_right] = img_right
                        self.img_pyramid_last[cam_id_left] = imgpyr_left
                        self.img_pyramid_last[cam_id_right] = imgpyr_right
                        self.img_mask_last[cam_id_left] = mask_left
                        self.img_mask_last[cam_id_right] = mask_right
                        self.pts_last[cam_id_left] = []
                        self.pts_last[cam_id_right] = []
                        self.ids_last[cam_id_left] = []
                        self.ids_last[cam_id_right] = []
                    print("[KLT-EXTRACTOR]: Failed to get enough points to do RANSAC, resetting.....")
                    return

                # Get our "good tracks"
                good_left = []
                good_right = []
                good_ids_left = []
                good_ids_right = []

                h_l, w_l = img_left.shape[:2]
                h_r, w_r = img_right.shape[:2]

                # Loop through all left points
                for i in range(len(pts_left_new)):
                    # Ensure we do not have any bad KLT tracks (i.e., points are negative)
                    pt = pts_left_new[i].pt
                    if pt[0] < 0 or pt[1] < 0 or int(pt[0]) > w_l or int(pt[1]) > h_l:
                        continue

                    # See if we have the same feature in the right
                    found_right = False
                    index_right = 0
                    for n in range(len(ids_right_old)):
                        if ids_left_old[i] == ids_right_old[n]:
                            found_right = True
                            index_right = n
                            break

                    # If it is a good track, and also tracked from left to right
                    # Else track it as a mono feature in just the left image
                    if mask_ll and mask_ll[i] and found_right and mask_rr and mask_rr[index_right]:
                        # Ensure we do not have any bad KLT tracks (i.e., points are negative)
                        pt_r = pts_right_new[index_right].pt
                        if (pt_r[0] < 0 or pt_r[1] < 0 or
                                int(pt_r[0]) >= w_r or int(pt_r[1]) >= h_r):
                            continue
                        good_left.append(pts_left_new[i])
                        good_right.append(pts_right_new[index_right])
                        good_ids_left.append(ids_left_old[i])
                        good_ids_right.append(ids_right_old[index_right])
                    elif mask_ll and mask_ll[i]:
                        good_left.append(pts_left_new[i])
                        good_ids_left.append(ids_left_old[i])

                # Loop through all right points
                for i in range(len(pts_right_new)):
                    # Ensure we do not have any bad KLT tracks (i.e., points are negative)
                    pt = pts_right_new[i].pt
                    if pt[0] < 0 or pt[1] < 0 or int(pt[0]) >= w_r or int(pt[1]) >= h_r:
                        continue
                    # See if we have the same feature in the right
                    added_already = ids_right_old[i] in good_ids_right
                    # If it has not already been added as a good feature, add it as a mono track
                    if mask_rr and mask_rr[i] and not added_already:
                        good_right.append(pts_right_new[i])
                        good_ids_right.append(ids_right_old[i])

                # Update our feature database, with these new observations
                for i in range(len(good_left)):
                    pt = good_left[i].pt
                    npt_l = self.camera_calib[cam_id_left].undistort_cv(pt)
                    self.database.update_feature(good_ids_left[i], message.timestamp, cam_id_left,
                                                 pt[0], pt[1], npt_l[0], npt_l[1])
                for i in range(len(good_right)):
                    pt = good_right[i].pt
                    npt_r = self.camera_calib[cam_id_right].undistort_cv(pt)
                    self.database.update_feature(good_ids_right[i], message.timestamp, cam_id_right,
                                                 pt[0], pt[1], npt_r[0], npt_r[1])

                # Move forward in time
                with self.mtx_last_vars:
                    self.img_last[cam_id_left] = img_left
                    self.img_last[cam_id_right] = img_right
                    self.img_pyramid_last[cam_id_left] = imgpyr_left
                    self.img_pyramid_last[cam_id_right] = imgpyr_right
                    self.img_mask_last[cam_id_left] = mask_left
                    self.img_mask_last[cam_id_right] = mask_right
                    self.pts_last[cam_id_left] = good_left
                    self.pts_last[cam_id_right] = good_right
                    self.ids_last[cam_id_left] = good_ids_left
                    self.ids_last[cam_id_right] = good_ids_right

                rT6 = time.time()

    def _perform_matching_into(self, img0pyr, img1pyr, kpts0, kpts1, id0, id1, attr_name):
        """Helper for parallel matching in feed_stereo - stores result on self."""
        result = self.perform_matching(img0pyr, img1pyr, kpts0, kpts1, id0, id1)
        setattr(self, attr_name, result)

    def perform_matching(self, img0pyr, img1pyr, kpts0, kpts1, id0, id1) -> List[int]:
        """
        KLT track between two images, and do RANSAC afterwards.
        Matches C++ TrackKLT::perform_matching exactly.
        """
        # We must have equal vectors
        assert len(kpts0) == len(kpts1)

        # Return if we don't have any points
        if not kpts0 or not kpts1:
            return []

        # Convert keypoints into points
        pts0 = np.array([kp.pt for kp in kpts0], dtype=np.float32)
        pts1 = np.array([kp.pt for kp in kpts1], dtype=np.float32)

        # If we don't have enough points for ransac just return empty
        # We set the mask to be all zeros since all points failed RANSAC
        if len(pts0) < 10:
            return [0] * len(pts0)

        # Now do KLT tracking to get the valid new points
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        # NOTE: Python cv2.calcOpticalFlowPyrLK does NOT accept a pre-built pyramid list
        # unlike C++ which accepts InputArray (vector<Mat>). Pass base images instead.
        next_pts, mask_klt, error = cv2.calcOpticalFlowPyrLK(
            img0pyr[0], img1pyr[0],
            pts0, pts1,
            winSize=self.win_size,
            maxLevel=self.pyr_levels,
            criteria=criteria,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        )

        mask_klt = mask_klt.flatten()

        # Normalize these points, so we can then do ransac
        # We don't want to do ransac on distorted image uvs since the mapping is nonlinear
        pts0_n = []
        pts1_n = []
        for i in range(len(pts0)):
            pts0_n.append(self.camera_calib[id0].undistort_cv(pts0[i]))
            pts1_n.append(self.camera_calib[id1].undistort_cv(next_pts[i]))

        pts0_n = np.array(pts0_n, dtype=np.float64)
        pts1_n = np.array(pts1_n, dtype=np.float64)

        # Do RANSAC outlier rejection (note since we normalized the max pixel error is now in the normalized cords)
        max_focallength_img0 = max(self.camera_calib[id0].get_K()[0, 0], self.camera_calib[id0].get_K()[1, 1])
        max_focallength_img1 = max(self.camera_calib[id1].get_K()[0, 0], self.camera_calib[id1].get_K()[1, 1])
        max_focallength = max(max_focallength_img0, max_focallength_img1)

        mask_rsc = None
        try:
            F, mask_rsc = cv2.findFundamentalMat(pts0_n, pts1_n, cv2.FM_RANSAC, 2.0 / max_focallength, 0.999)
            if mask_rsc is not None:
                mask_rsc = mask_rsc.ravel()
        except Exception:
            pass

        # Loop through and record only ones that are valid
        mask_out = []
        for i in range(len(mask_klt)):
            if (mask_klt[i] and mask_rsc is not None and i < len(mask_rsc) and mask_rsc[i]):
                mask_out.append(1)
            else:
                mask_out.append(0)

        # Copy back the updated positions
        for i in range(len(pts0)):
            kpts0[i].pt = (float(pts0[i][0]), float(pts0[i][1]))
            kpts1[i].pt = (float(next_pts[i][0]), float(next_pts[i][1]))

        return mask_out

    def perform_detection_monocular(self, img0pyr, mask0, pts0: list, ids0: list):
        """
        Detects new features in the current image.
        Matches C++ TrackKLT::perform_detection_monocular exactly.

        NOTE: pts0 and ids0 are modified IN-PLACE (passed by reference in C++).
        """
        img = img0pyr[0]
        h, w = img.shape[:2]

        # Create a 2D occupancy grid for this current image
        # Note that we scale this down, so that each grid point is equal to a set of pixels
        size_close = (int(w / self.min_px_dist), int(h / self.min_px_dist))  # (width, height)
        grid_2d_close = np.zeros((size_close[1], size_close[0]), dtype=np.uint8)  # rows x cols

        size_x = float(w) / float(self.grid_x)
        size_y = float(h) / float(self.grid_y)
        grid_2d_grid = np.zeros((self.grid_y, self.grid_x), dtype=np.uint8)  # rows x cols

        mask0_updated = mask0.copy()

        # Filter existing points (C++ uses erase-while-iterating pattern)
        i = 0
        while i < len(pts0):
            kpt = pts0[i]
            x = int(kpt.pt[0])
            y = int(kpt.pt[1])
            edge = 10

            # Check that it is in bounds
            if x < edge or x >= w - edge or y < edge or y >= h - edge:
                pts0.pop(i)
                ids0.pop(i)
                continue

            # Calculate mask coordinates for close points
            x_close = int(kpt.pt[0] / self.min_px_dist)
            y_close = int(kpt.pt[1] / self.min_px_dist)
            if x_close < 0 or x_close >= size_close[0] or y_close < 0 or y_close >= size_close[1]:
                pts0.pop(i)
                ids0.pop(i)
                continue

            # Calculate what grid cell this feature is in
            x_grid = int(math.floor(kpt.pt[0] / size_x))
            y_grid = int(math.floor(kpt.pt[1] / size_y))
            if x_grid < 0 or x_grid >= self.grid_x or y_grid < 0 or y_grid >= self.grid_y:
                pts0.pop(i)
                ids0.pop(i)
                continue

            # Check if this keypoint is near another point
            if grid_2d_close[y_close, x_close] > 127:
                pts0.pop(i)
                ids0.pop(i)
                continue

            # Now check if it is in a mask area or not
            if mask0[y, x] > 127:
                pts0.pop(i)
                ids0.pop(i)
                continue

            # Else we are good, move forward to the next point
            grid_2d_close[y_close, x_close] = 255
            if grid_2d_grid[y_grid, x_grid] < 255:
                grid_2d_grid[y_grid, x_grid] += 1

            # Append this to the local mask of the image
            if (x - self.min_px_dist >= 0 and x + self.min_px_dist < w and
                    y - self.min_px_dist >= 0 and y + self.min_px_dist < h):
                pt1 = (x - self.min_px_dist, y - self.min_px_dist)
                pt2 = (x + self.min_px_dist, y + self.min_px_dist)
                cv2.rectangle(mask0_updated, pt1, pt2, 255, -1)

            i += 1

        # First compute how many more features we need to extract from this image
        min_feat_percent = 0.50
        num_featsneeded = self.num_features - len(pts0)
        if num_featsneeded < min(20, int(min_feat_percent * self.num_features)):
            return

        # We also check a downsampled mask such that we don't extract in areas where it is all masked!
        mask0_grid = cv2.resize(mask0, (self.grid_x, self.grid_y), interpolation=cv2.INTER_NEAREST)

        # Create grids we need to extract from and then extract our features
        num_features_grid = int(float(self.num_features) / float(self.grid_x * self.grid_y)) + 1
        num_features_grid_req = max(1, int(min_feat_percent * num_features_grid))
        valid_locs = []
        for x_loc in range(self.grid_x):
            for y_loc in range(self.grid_y):
                if (int(grid_2d_grid[y_loc, x_loc]) < num_features_grid_req and
                        int(mask0_grid[y_loc, x_loc]) != 255):
                    valid_locs.append((x_loc, y_loc))

        # Perform grid-based FAST extraction (inline Grider_GRID::perform_griding)
        pts0_ext = self._perform_griding(img, mask0_updated, valid_locs, self.num_features,
                                         self.grid_x, self.grid_y, self.threshold, True)

        # Now, reject features that are close to a current feature
        kpts0_new = []
        pts0_new = []
        for kpt in pts0_ext:
            # Check that it is in bounds
            x_g = int(kpt.pt[0] / self.min_px_dist)
            y_g = int(kpt.pt[1] / self.min_px_dist)
            if x_g < 0 or x_g >= size_close[0] or y_g < 0 or y_g >= size_close[1]:
                continue
            # See if there is a point at this location
            if grid_2d_close[y_g, x_g] > 127:
                continue
            # Else lets add it!
            kpts0_new.append(kpt)
            pts0_new.append(kpt.pt)
            grid_2d_close[y_g, x_g] = 255

        # Loop through and record only ones that are valid
        for i in range(len(pts0_new)):
            # update the uv coordinates
            kpts0_new[i].pt = pts0_new[i]
            # append the new uv coordinate
            pts0.append(kpts0_new[i])
            # move id forward and append this new point
            self.currid += 1
            ids0.append(self.currid)

    def perform_detection_stereo(self, img0pyr, img1pyr, mask0, mask1,
                                 cam_id_left, cam_id_right,
                                 pts0: list, pts1: list, ids0: list, ids1: list):
        """
        Detects new features in the current stereo pair.
        Matches C++ TrackKLT::perform_detection_stereo exactly.

        NOTE: pts0, pts1, ids0, ids1 are modified IN-PLACE.
        """
        img0 = img0pyr[0]
        img1 = img1pyr[0]
        h0, w0 = img0.shape[:2]
        h1, w1 = img1.shape[:2]

        # =====================================================================
        # LEFT IMAGE: Create a 2D occupancy grid
        # =====================================================================
        size_close0 = (int(w0 / self.min_px_dist), int(h0 / self.min_px_dist))
        grid_2d_close0 = np.zeros((size_close0[1], size_close0[0]), dtype=np.uint8)
        size_x0 = float(w0) / float(self.grid_x)
        size_y0 = float(h0) / float(self.grid_y)
        grid_2d_grid0 = np.zeros((self.grid_y, self.grid_x), dtype=np.uint8)
        mask0_updated = mask0.copy()

        # Filter existing LEFT points
        i = 0
        while i < len(pts0):
            kpt = pts0[i]
            x = int(kpt.pt[0])
            y = int(kpt.pt[1])
            edge = 10
            if x < edge or x >= w0 - edge or y < edge or y >= h0 - edge:
                pts0.pop(i)
                ids0.pop(i)
                continue
            x_close = int(kpt.pt[0] / self.min_px_dist)
            y_close = int(kpt.pt[1] / self.min_px_dist)
            if x_close < 0 or x_close >= size_close0[0] or y_close < 0 or y_close >= size_close0[1]:
                pts0.pop(i)
                ids0.pop(i)
                continue
            x_grid = int(math.floor(kpt.pt[0] / size_x0))
            y_grid = int(math.floor(kpt.pt[1] / size_y0))
            if x_grid < 0 or x_grid >= self.grid_x or y_grid < 0 or y_grid >= self.grid_y:
                pts0.pop(i)
                ids0.pop(i)
                continue
            if grid_2d_close0[y_close, x_close] > 127:
                pts0.pop(i)
                ids0.pop(i)
                continue
            if mask0[y, x] > 127:
                pts0.pop(i)
                ids0.pop(i)
                continue
            grid_2d_close0[y_close, x_close] = 255
            if grid_2d_grid0[y_grid, x_grid] < 255:
                grid_2d_grid0[y_grid, x_grid] += 1
            if (x - self.min_px_dist >= 0 and x + self.min_px_dist < w0 and
                    y - self.min_px_dist >= 0 and y + self.min_px_dist < h0):
                pt1 = (x - self.min_px_dist, y - self.min_px_dist)
                pt2 = (x + self.min_px_dist, y + self.min_px_dist)
                cv2.rectangle(mask0_updated, pt1, pt2, 255, -1)
            i += 1

        # First compute how many more features we need to extract from this image
        min_feat_percent = 0.50
        num_featsneeded_0 = self.num_features - len(pts0)

        # LEFT: if we need features we should extract them in the current frame
        # LEFT: we will also try to track them from this frame over to the right frame
        if num_featsneeded_0 > min(20, int(min_feat_percent * self.num_features)):

            # We also check a downsampled mask
            mask0_grid = cv2.resize(mask0, (self.grid_x, self.grid_y), interpolation=cv2.INTER_NEAREST)

            # Create grids we need to extract from
            num_features_grid = int(float(self.num_features) / float(self.grid_x * self.grid_y)) + 1
            num_features_grid_req = max(1, int(min_feat_percent * num_features_grid))
            valid_locs = []
            for x_loc in range(self.grid_x):
                for y_loc in range(self.grid_y):
                    if (int(grid_2d_grid0[y_loc, x_loc]) < num_features_grid_req and
                            int(mask0_grid[y_loc, x_loc]) != 255):
                        valid_locs.append((x_loc, y_loc))

            pts0_ext = self._perform_griding(img0, mask0_updated, valid_locs, self.num_features,
                                             self.grid_x, self.grid_y, self.threshold, True)

            # Now, reject features that are close a current feature
            kpts0_new = []
            pts0_new = []
            for kpt in pts0_ext:
                x_g = int(kpt.pt[0] / self.min_px_dist)
                y_g = int(kpt.pt[1] / self.min_px_dist)
                if x_g < 0 or x_g >= size_close0[0] or y_g < 0 or y_g >= size_close0[1]:
                    continue
                if grid_2d_close0[y_g, x_g] > 127:
                    continue
                grid_2d_close0[y_g, x_g] = 255
                kpts0_new.append(kpt)
                pts0_new.append(kpt.pt)

            # TODO: Project points from the left frame into the right frame
            # TODO: This will not work for large baseline systems.....
            kpts1_new = _copy_keypoints(kpts0_new)
            pts1_new = list(pts0_new)

            # If we have points, do KLT tracking to get the valid projections into the right image
            if pts0_new:
                # Do our KLT tracking from the left to the right frame of reference
                # NOTE: C++ calls calcOpticalFlowPyrLK directly here (NOT perform_matching)
                pts0_arr = np.array(pts0_new, dtype=np.float32)
                pts1_arr = np.array(pts1_new, dtype=np.float32)

                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
                next_pts, mask_lr, error = cv2.calcOpticalFlowPyrLK(
                    img0pyr[0], img1pyr[0],
                    pts0_arr, pts1_arr,
                    winSize=self.win_size,
                    maxLevel=self.pyr_levels,
                    criteria=criteria,
                    flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
                )

                mask_lr = mask_lr.flatten()

                # Loop through and record only ones that are valid
                for k in range(len(pts0_new)):
                    # Check to see if the feature is out of bounds (oob) in either image
                    oob_left = (int(pts0_arr[k][0]) < 0 or int(pts0_arr[k][0]) >= w0 or
                                int(pts0_arr[k][1]) < 0 or int(pts0_arr[k][1]) >= h0)
                    oob_right = (int(next_pts[k][0]) < 0 or int(next_pts[k][0]) >= w1 or
                                 int(next_pts[k][1]) < 0 or int(next_pts[k][1]) >= h1)

                    if not oob_left and not oob_right and mask_lr[k] == 1:
                        # update the uv coordinates
                        kpts0_new[k].pt = (float(pts0_arr[k][0]), float(pts0_arr[k][1]))
                        kpts1_new[k].pt = (float(next_pts[k][0]), float(next_pts[k][1]))
                        # append the new uv coordinate
                        pts0.append(kpts0_new[k])
                        pts1.append(kpts1_new[k])
                        # move id forward and append this new point
                        self.currid += 1
                        temp = self.currid
                        ids0.append(temp)
                        ids1.append(temp)
                    elif not oob_left:
                        # update the uv coordinates
                        kpts0_new[k].pt = (float(pts0_arr[k][0]), float(pts0_arr[k][1]))
                        # append the new uv coordinate
                        pts0.append(kpts0_new[k])
                        # move id forward and append this new point
                        self.currid += 1
                        temp = self.currid
                        ids0.append(temp)

        # =====================================================================
        # RIGHT IMAGE: Now summarize the number of tracks in the right image
        # =====================================================================
        size_close1 = (int(w1 / self.min_px_dist), int(h1 / self.min_px_dist))
        grid_2d_close1 = np.zeros((size_close1[1], size_close1[0]), dtype=np.uint8)
        size_x1 = float(w1) / float(self.grid_x)
        size_y1 = float(h1) / float(self.grid_y)
        grid_2d_grid1 = np.zeros((self.grid_y, self.grid_x), dtype=np.uint8)
        mask1_updated = mask0.copy()  # C++ uses mask0.clone() here (intentional in the original code)

        # Filter existing RIGHT points
        i = 0
        while i < len(pts1):
            kpt = pts1[i]
            x = int(kpt.pt[0])
            y = int(kpt.pt[1])
            edge = 10
            if x < edge or x >= w1 - edge or y < edge or y >= h1 - edge:
                pts1.pop(i)
                ids1.pop(i)
                continue
            x_close = int(kpt.pt[0] / self.min_px_dist)
            y_close = int(kpt.pt[1] / self.min_px_dist)
            if x_close < 0 or x_close >= size_close1[0] or y_close < 0 or y_close >= size_close1[1]:
                pts1.pop(i)
                ids1.pop(i)
                continue
            x_grid = int(math.floor(kpt.pt[0] / size_x1))
            y_grid = int(math.floor(kpt.pt[1] / size_y1))
            if x_grid < 0 or x_grid >= self.grid_x or y_grid < 0 or y_grid >= self.grid_y:
                pts1.pop(i)
                ids1.pop(i)
                continue
            # Check if this keypoint is near another point
            # NOTE: if it is *not* a stereo point, then we will not delete the feature
            # NOTE: this means we might have a mono and stereo feature near each other, but that is ok
            is_stereo = ids1[i] in ids0
            if grid_2d_close1[y_close, x_close] > 127 and not is_stereo:
                pts1.pop(i)
                ids1.pop(i)
                continue
            if mask1[y, x] > 127:
                pts1.pop(i)
                ids1.pop(i)
                continue
            grid_2d_close1[y_close, x_close] = 255
            if grid_2d_grid1[y_grid, x_grid] < 255:
                grid_2d_grid1[y_grid, x_grid] += 1
            if (x - self.min_px_dist >= 0 and x + self.min_px_dist < w1 and
                    y - self.min_px_dist >= 0 and y + self.min_px_dist < h1):
                pt1_coord = (x - self.min_px_dist, y - self.min_px_dist)
                pt2_coord = (x + self.min_px_dist, y + self.min_px_dist)
                cv2.rectangle(mask1_updated, pt1_coord, pt2_coord, 255, -1)
            i += 1

        # RIGHT: if we need features we should extract them in the current frame
        num_featsneeded_1 = self.num_features - len(pts1)
        if num_featsneeded_1 > min(20, int(min_feat_percent * self.num_features)):
            mask1_grid = cv2.resize(mask1, (self.grid_x, self.grid_y), interpolation=cv2.INTER_NEAREST)

            num_features_grid = int(float(self.num_features) / float(self.grid_x * self.grid_y)) + 1
            num_features_grid_req = max(1, int(min_feat_percent * num_features_grid))
            valid_locs = []
            for x_loc in range(self.grid_x):
                for y_loc in range(self.grid_y):
                    if (int(grid_2d_grid1[y_loc, x_loc]) < num_features_grid_req and
                            int(mask1_grid[y_loc, x_loc]) != 255):
                        valid_locs.append((x_loc, y_loc))

            pts1_ext = self._perform_griding(img1, mask1_updated, valid_locs, self.num_features,
                                             self.grid_x, self.grid_y, self.threshold, True)

            # Now, reject features that are close a current feature
            for kpt in pts1_ext:
                x_g = int(kpt.pt[0] / self.min_px_dist)
                y_g = int(kpt.pt[1] / self.min_px_dist)
                if x_g < 0 or x_g >= size_close1[0] or y_g < 0 or y_g >= size_close1[1]:
                    continue
                if grid_2d_close1[y_g, x_g] > 127:
                    continue
                # Else lets add it!
                pts1.append(kpt)
                self.currid += 1
                ids1.append(self.currid)
                grid_2d_close1[y_g, x_g] = 255

    def _perform_griding(self, img, mask, valid_locs, num_features, grid_x, grid_y,
                         threshold, nonmax_suppression) -> List[cv2.KeyPoint]:
        """
        Inline implementation of Grider_GRID::perform_griding.
        Extracts FAST features in a grid pattern with sub-pixel refinement.
        """
        if not valid_locs:
            return []

        # We want to have equally distributed features
        # If we have more grids than features, recalculate grid dims
        if num_features < grid_x * grid_y:
            ratio = float(grid_x) / float(grid_y)
            grid_y = int(math.ceil(math.sqrt(num_features / ratio)))
            grid_x = int(math.ceil(grid_y * ratio))

        num_features_grid = int(float(num_features) / float(grid_x * grid_y)) + 1
        assert grid_x > 0
        assert grid_y > 0
        assert num_features_grid > 0

        # Calculate the size our extraction boxes should be
        size_x = img.shape[1] // grid_x
        size_y = img.shape[0] // grid_y
        assert size_x > 0
        assert size_y > 0

        h, w = img.shape[:2]

        fast = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=nonmax_suppression)

        pts = []

        for grid_loc in valid_locs:
            gx, gy = grid_loc
            x = gx * size_x
            y = gy * size_y

            # Skip if we are out of bounds
            if x + size_x > w or y + size_y > h:
                continue

            img_roi = img[y:y + size_y, x:x + size_x]
            mask_roi = mask[y:y + size_y, x:x + size_x]

            # Extract FAST features for this part of the image
            # C++ cv::FAST ignores mask; the mask filtering is done after extraction in the C++ code
            # But we pass inverted mask for efficiency
            inverted_mask = cv2.bitwise_not(mask_roi)
            pts_new = fast.detect(img_roi, mask=inverted_mask)

            # Now lets get the top number from this
            pts_new = sorted(pts_new, key=lambda k: k.response, reverse=True)

            # Append the "best" ones to our vector
            for j in range(min(num_features_grid, len(pts_new))):
                pt_cor = pts_new[j]
                # Correct coordinates for ROI offset
                pt_cor.pt = (pt_cor.pt[0] + float(x), pt_cor.pt[1] + float(y))

                # Reject if out of bounds
                if (int(pt_cor.pt[0]) < 0 or int(pt_cor.pt[0]) > w or
                        int(pt_cor.pt[1]) < 0 or int(pt_cor.pt[1]) > h):
                    continue

                # Check if it is in the mask region
                if mask[int(pt_cor.pt[1]), int(pt_cor.pt[0])] > 127:
                    continue

                pts.append(pt_cor)

        if not pts:
            return []

        # Sub-pixel refinement (matches C++ Grider_GRID exactly)
        pts_refined = np.array([kp.pt for kp in pts], dtype=np.float32)
        win_sp = (5, 5)
        zero_zone = (-1, -1)
        term_crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 20, 0.001)
        cv2.cornerSubPix(img, pts_refined, win_sp, zero_zone, term_crit)

        # Save the refined points
        for i in range(len(pts)):
            pts[i].pt = (float(pts_refined[i][0]), float(pts_refined[i][1]))

        return pts


def _copy_keypoints(kpts: List[cv2.KeyPoint]) -> List[cv2.KeyPoint]:
    """
    Create a copy of a list of cv2.KeyPoint objects.
    cv2.KeyPoint cannot be deepcopied, so we manually reconstruct each one.
    Note: OpenCV 4.2 requires positional args (x, y, _size, _angle, _response, _octave, _class_id).
    """
    result = []
    for kp in kpts:
        result.append(cv2.KeyPoint(
            kp.pt[0], kp.pt[1], kp.size,
            kp.angle, kp.response,
            kp.octave, kp.class_id
        ))
    return result
