import numpy as np
from vio_pipeline_v2.type import IMU, Vec, JPLQuat, PoseJPL
from vio_pipeline_v2.msckf.StateOptions import StateOptions

class State:
    def __init__(self, options):
        self._options = options
        self._variables = []
        current_id = 0

        # ======================================================================
        # 1. Initialize IMU State
        # ======================================================================
        self._imu = IMU()
        self._imu.set_local_id(current_id)
        self._variables.append(self._imu)
        current_id += self._imu.size()

        # ======================================================================
        # 2. Initialize IMU Intrinsics (dw, da, tg)
        # ======================================================================
        # Initialize containers
        self._calib_imu_dw = Vec(6)
        self._calib_imu_da = Vec(6)
        self._calib_imu_tg = Vec(9) # Gravity sensitivity defaults to zeros (Correct)

        # --- SETUP DEFAULT VALUES (IDENTITY) ---
        # We need to create an "Identity" matrix representation for the intrinsics.
        # If these remain zero, the Propagator will multiply all IMU readings by 0.
        
        _imu_default = np.zeros(6)

        # Check Model Type to set correct diagonals
        if options.imu_model == StateOptions.ImuModel.KALIBR:
            # KALIBR (Lower Triangular): Diagonals are at indices 0, 3, 5
            # D = [v0  0  0]
            #     [v1 v3  0]
            #     [v2 v4 v5]
            _imu_default[0] = 1.0
            _imu_default[3] = 1.0
            _imu_default[5] = 1.0
        else:
            # RPNG (Upper Triangular): Diagonals are at indices 0, 2, 5
            # D = [v0 v1 v3]
            #     [ 0 v2 v4]
            #     [ 0  0 v5]
            _imu_default[0] = 1.0
            _imu_default[2] = 1.0
            _imu_default[5] = 1.0

        # Force set the values immediately
        # Using .copy() ensures memory contiguity which helps C++ bindings
        self._calib_imu_dw.set_value(_imu_default.copy())
        self._calib_imu_dw.set_fej(_imu_default.copy())
        
        self._calib_imu_da.set_value(_imu_default.copy())
        self._calib_imu_da.set_fej(_imu_default.copy())

        # Debug Print to Verify Initialization
        print(f"[State Init] Default Intrinsics set to:\n{_imu_default}")
        print(f"[State Init] dw value internal: {self._calib_imu_dw.value()}")

        # Initialize Rotation Calibrations
        self._calib_imu_GYROtoIMU = JPLQuat()
        self._calib_imu_ACCtoIMU = JPLQuat()

        # ======================================================================
        # 3. Add to State Vector (If Calibration Enabled)
        # ======================================================================
        if options.do_calib_imu_intrinsics:
            # Add Gyroscope dw
            self._calib_imu_dw.set_local_id(current_id)
            self._variables.append(self._calib_imu_dw)
            current_id += self._calib_imu_dw.size()
        
            # Add Accelerometer da
            self._calib_imu_da.set_local_id(current_id)
            self._variables.append(self._calib_imu_da)
            current_id += self._calib_imu_da.size()

            # Add Gravity Sensitivity
            if options.do_calib_imu_g_sensitivity:
                self._calib_imu_tg.set_local_id(current_id)
                self._variables.append(self._calib_imu_tg)
                current_id += self._calib_imu_tg.size()

            # Add Rotational Misalignment (Model Dependent)
            if options.imu_model == StateOptions.ImuModel.KALIBR:
                self._calib_imu_GYROtoIMU.set_local_id(current_id)
                self._variables.append(self._calib_imu_GYROtoIMU)
                current_id += self._calib_imu_GYROtoIMU.size()
            else:
                self._calib_imu_ACCtoIMU.set_local_id(current_id)
                self._variables.append(self._calib_imu_ACCtoIMU)
                current_id += self._calib_imu_ACCtoIMU.size()
        
        # ======================================================================
        # 4. Camera Time Offset
        # ======================================================================
        self._calib_dt_CAMtoIMU = Vec(1)
        if options.do_calib_camera_timeoffset:
            self._calib_dt_CAMtoIMU.set_local_id(current_id)
            self._variables.append(self._calib_dt_CAMtoIMU)
            # FIX: Added .size()
            current_id += self._calib_dt_CAMtoIMU.size()

        # ======================================================================
        # 5. Camera Extrinsics & Intrinsics
        # ======================================================================
        self._calib_IMUtoCAM = {}
        self._cam_intrinsics = {}
        # self._cam_intrinsics_cameras = options.camera_intrinsics # Not used in state init directly

        for i in range(options.num_cameras):
            pose = PoseJPL()
            intrin = Vec(8)

            self._calib_IMUtoCAM[i] = pose
            self._cam_intrinsics[i] = intrin

            if options.do_calib_camera_pose:
                pose.set_local_id(current_id)
                self._variables.append(pose)
                current_id += pose.size()

            if options.do_calib_camera_intrinsics:
                intrin.set_local_id(current_id)
                self._variables.append(intrin)
                current_id += intrin.size()

        # ======================================================================
        # 6. Initialize Covariance
        # ======================================================================
        self._Cov = (1e-3**2) * np.eye(current_id)

        if options.do_calib_imu_intrinsics:
            def set_diag(var, noise_std):
                idx = var.id()
                dim = var.size()
                if idx != -1:
                    self._Cov[idx:idx+dim, idx:idx+dim] = (noise_std**2) * np.eye(dim)
            
            # FIX: Typo _calib_imu_dwm -> _calib_imu_dw
            set_diag(self._calib_imu_dw, 0.005)
            set_diag(self._calib_imu_da, 0.008)

            if options.do_calib_imu_g_sensitivity:
                set_diag(self._calib_imu_tg, 0.005)

            # FIX: Use Enum comparison
            if options.imu_model == StateOptions.ImuModel.KALIBR:
                set_diag(self._calib_imu_GYROtoIMU, 0.005)
            else:
                set_diag(self._calib_imu_ACCtoIMU, 0.005)

        if options.do_calib_camera_timeoffset:
            idx = self._calib_dt_CAMtoIMU.id()
            if idx != -1:
                self._Cov[idx, idx] = (0.01**2)

        if options.do_calib_camera_pose:
            for i in range(options.num_cameras):
                pose = self._calib_IMUtoCAM[i]
                idx = pose.id()
                if idx != -1:
                    self._Cov[idx:idx+3, idx:idx+3] = (0.005**2) * np.eye(3) 
                    self._Cov[idx+3:idx+6, idx+3:idx+6] = (0.015**2) * np.eye(3)

        if options.do_calib_camera_intrinsics:
            for i in range(options.num_cameras):
                intrin = self._cam_intrinsics[i]
                idx = intrin.id()
                if idx != -1:
                    self._Cov[idx:idx+4, idx:idx+4] = (1.0**2) * np.eye(4) 
                    self._Cov[idx+4:idx+8, idx+4:idx+8] = (0.005**2) * np.eye(4) 

        # ======================================================================
        # 7. Dynamic Storage Containers
        # ======================================================================
        self._timestamp = 0.0
        self._clones_IMU = {}   
        self._features_SLAM = {} 

    def max_covariance_size(self) -> int:
        return self._Cov.shape[0]

    # ... (Keep Dm, Tg, imu_intrinsic_size, margtimestep as is) ...
    @staticmethod
    def Dm(imu_model, vec: np.ndarray) -> np.ndarray:
        # Note: Ensure this function supports the logic above
        assert vec.shape == (6,) or vec.shape == (6, 1)
        D_matrix = np.eye(3)
        v = vec.flatten()
        
        # Use robust checking or Enum
        if imu_model == StateOptions.ImuModel.KALIBR or str(imu_model) == "KALIBR":
            D_matrix[0, 0] = v[0]
            D_matrix[1, 0] = v[1]
            D_matrix[2, 0] = v[2]
            D_matrix[1, 1] = v[3]
            D_matrix[2, 1] = v[4]
            D_matrix[2, 2] = v[5]
        else:
            D_matrix[0, 0] = v[0]
            D_matrix[0, 1] = v[1]
            D_matrix[1, 1] = v[2]
            D_matrix[0, 2] = v[3]
            D_matrix[1, 2] = v[4]
            D_matrix[2, 2] = v[5]
        return D_matrix
    
    @staticmethod
    def Tg(vec: np.ndarray) -> np.ndarray:
        assert vec.shape == (9,) or vec.shape == (9, 1)
        v = vec.flatten()
        Tg_mat = np.zeros((3, 3))
        Tg_mat[:, 0] = v[0:3]
        Tg_mat[:, 1] = v[3:6]
        Tg_mat[:, 2] = v[6:9]
        return Tg_mat

    def imu_intrinsic_size(self) -> int:
        sz = 0
        if self._options.do_calib_imu_intrinsics:
            sz += 15
            if self._options.do_calib_imu_g_sensitivity:
                sz += 9 
        return sz
        
    def margtimestep(self):
        if not self._clones_IMU:
            return -1.0
        return min(self._clones_IMU.keys())