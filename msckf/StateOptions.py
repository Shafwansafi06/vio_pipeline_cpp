import sys
from enum import Enum, auto
from typing import Optional

# Imports
from vio_pipeline_v2.type import LandmarkRepresentation
# Assuming YamlParser is handled via simple dictionary or external logic, 
# but here we keep the structure compatible with the C++ logic.

class StateOptions:
    """
    Struct which stores all our filter options.
    """
    
    class IntegrationMethod(Enum):
        DISCRETE = auto()
        RK4 = auto()
        ANALYTICAL = auto()

    class ImuModel(Enum):
        KALIBR = auto()
        RPNG = auto()

    def __init__(self):
        # Bool to determine whether or not to do first estimate Jacobians
        self.do_fej = True

        # What type of numerical integration is used during propagation
        self.integration_method = StateOptions.IntegrationMethod.RK4

        # Calibration Flags
        self.do_calib_camera_pose = False
        self.do_calib_camera_intrinsics = False
        self.do_calib_camera_timeoffset = False
        self.do_calib_imu_intrinsics = False
        self.do_calib_imu_g_sensitivity = False

        # IMU Model
        self.imu_model = StateOptions.ImuModel.KALIBR

        # Limits
        self.max_clone_size = 11
        self.max_slam_features = 25
        self.max_slam_in_update = 1000
        self.max_msckf_in_update = 1000
        self.max_aruco_features = 1024
        self.num_cameras = 1

        # Feature Representations
        self.feat_rep_msckf = LandmarkRepresentation.Representation.GLOBAL_3D
        self.feat_rep_slam = LandmarkRepresentation.Representation.GLOBAL_3D
        self.feat_rep_aruco = LandmarkRepresentation.Representation.GLOBAL_3D

    def print(self, parser=None):
        """
        Loads parameters from parser (if provided) and prints current configuration.
        parser: Dictionary-like object or specific wrapper matching C++ YamlParser interface.
        """
        if parser is not None:
            # Helper to get value if exists
            def get(key, default):
                return parser.get(key, default) if isinstance(parser, dict) else default

            self.do_fej = get("use_fej", self.do_fej)

            # Integration Method
            int_str = get("integration", "rk4").lower()
            if int_str == "discrete":
                self.integration_method = StateOptions.IntegrationMethod.DISCRETE
            elif int_str == "rk4":
                self.integration_method = StateOptions.IntegrationMethod.RK4
            elif int_str == "analytical":
                self.integration_method = StateOptions.IntegrationMethod.ANALYTICAL
            else:
                print(f"[ERROR] Invalid imu integration model: {int_str}")
                print("[ERROR] Please select: discrete, rk4, analytical")
                sys.exit(1)

            # Calibration
            self.do_calib_camera_pose = get("calib_cam_extrinsics", self.do_calib_camera_pose)
            self.do_calib_camera_intrinsics = get("calib_cam_intrinsics", self.do_calib_camera_intrinsics)
            self.do_calib_camera_timeoffset = get("calib_cam_timeoffset", self.do_calib_camera_timeoffset)
            self.do_calib_imu_intrinsics = get("calib_imu_intrinsics", self.do_calib_imu_intrinsics)
            self.do_calib_imu_g_sensitivity = get("calib_imu_g_sensitivity", self.do_calib_imu_g_sensitivity)

            # Limits
            self.max_clone_size = get("max_clones", self.max_clone_size)
            self.max_slam_features = get("max_slam", self.max_slam_features)
            self.max_slam_in_update = get("max_slam_in_update", self.max_slam_in_update)
            self.max_msckf_in_update = get("max_msckf_in_update", self.max_msckf_in_update)
            self.max_aruco_features = get("num_aruco", self.max_aruco_features)
            self.num_cameras = get("max_cameras", self.num_cameras)

            # Representations
            rep1 = LandmarkRepresentation.as_string(self.feat_rep_msckf)
            rep1 = get("feat_rep_msckf", rep1)
            self.feat_rep_msckf = LandmarkRepresentation.from_string(rep1)

            rep2 = LandmarkRepresentation.as_string(self.feat_rep_slam)
            rep2 = get("feat_rep_slam", rep2)
            self.feat_rep_slam = LandmarkRepresentation.from_string(rep2)

            rep3 = LandmarkRepresentation.as_string(self.feat_rep_aruco)
            rep3 = get("feat_rep_aruco", rep3)
            self.feat_rep_aruco = LandmarkRepresentation.from_string(rep3)

            # IMU Model
            # Assuming parser has method to get external/relative configs or we flatten the dict
            # For simplicity, assuming flattened keys or manual extraction logic:
            imu_str = get("imu_model", "kalibr").lower() 
            # In C++ it parses "relative_config_imu", "imu0", "model"
            
            if imu_str in ["kalibr", "calibrated"]:
                self.imu_model = StateOptions.ImuModel.KALIBR
            elif imu_str == "rpng":
                self.imu_model = StateOptions.ImuModel.RPNG
            else:
                print(f"[ERROR] Invalid IMU model: {imu_str}")
                sys.exit(1)

            if imu_str == "calibrated" and (self.do_calib_imu_intrinsics or self.do_calib_imu_g_sensitivity):
                print("[ERROR] 'calibrated' IMU model selected but calibration requested!")
                sys.exit(1)

        # Print Configuration
        print(f"  - use_fej: {self.do_fej}")
        print(f"  - integration: {self.integration_method.name}")
        print(f"  - calib_cam_extrinsics: {self.do_calib_camera_pose}")
        print(f"  - calib_cam_intrinsics: {self.do_calib_camera_intrinsics}")
        print(f"  - calib_cam_timeoffset: {self.do_calib_camera_timeoffset}")
        print(f"  - calib_imu_intrinsics: {self.do_calib_imu_intrinsics}")
        print(f"  - calib_imu_g_sensitivity: {self.do_calib_imu_g_sensitivity}")
        print(f"  - imu_model: {self.imu_model.name}")
        print(f"  - max_clones: {self.max_clone_size}")
        print(f"  - max_slam: {self.max_slam_features}")
        print(f"  - max_slam_in_update: {self.max_slam_in_update}")
        print(f"  - max_msckf_in_update: {self.max_msckf_in_update}")
        print(f"  - max_aruco: {self.max_aruco_features}")
        print(f"  - max_cameras: {self.num_cameras}")
        print(f"  - feat_rep_msckf: {LandmarkRepresentation.as_string(self.feat_rep_msckf)}")
        print(f"  - feat_rep_slam: {LandmarkRepresentation.as_string(self.feat_rep_slam)}")
        print(f"  - feat_rep_aruco: {LandmarkRepresentation.as_string(self.feat_rep_aruco)}")