from typing import Any, Dict, Optional

class FeatureInitializerOptions:
    """
    Struct which stores all our feature initializer options.
    """
    def __init__(self):
        # If we should perform 1d triangulation instead of 3d
        self.triangulate_1d = False

        # If we should perform Levenberg-Marquardt refinement
        self.refine_features = True

        # Max runs for Levenberg-Marquardt
        self.max_runs = 5

        # Init lambda for Levenberg-Marquardt optimization
        self.init_lamda = 1e-3

        # Max lambda for Levenberg-Marquardt optimization
        self.max_lamda = 1e10

        # Cutoff for dx increment to consider as converged
        self.min_dx = 1e-6

        # Cutoff for cost decrement to consider as converged
        self.min_dcost = 1e-6

        # Multiplier to increase/decrease lambda
        self.lam_mult = 10.0

        # Minimum distance to accept triangulated features
        self.min_dist = 0.10

        # Maximum distance to accept triangulated features
        self.max_dist = 60.0

        # Max baseline ratio to accept triangulated features
        self.max_baseline = 40.0

        # Max condition number of linear triangulation matrix to accept triangulated features
        self.max_cond_number = 10000.0

    def print(self, parser: Optional[Dict[str, Any]] = None):
        """
        Nice print function of what parameters we have loaded.
        :param parser: Dictionary containing configuration parameters (simulating YamlParser).
        """
        if parser is not None:
            self.triangulate_1d = parser.get("fi_triangulate_1d", self.triangulate_1d)
            self.refine_features = parser.get("fi_refine_features", self.refine_features)
            self.max_runs = parser.get("fi_max_runs", self.max_runs)
            self.init_lamda = parser.get("fi_init_lamda", self.init_lamda)
            self.max_lamda = parser.get("fi_max_lamda", self.max_lamda)
            self.min_dx = parser.get("fi_min_dx", self.min_dx)
            self.min_dcost = parser.get("fi_min_dcost", self.min_dcost)
            self.lam_mult = parser.get("fi_lam_mult", self.lam_mult)
            self.min_dist = parser.get("fi_min_dist", self.min_dist)
            self.max_dist = parser.get("fi_max_dist", self.max_dist)
            self.max_baseline = parser.get("fi_max_baseline", self.max_baseline)
            self.max_cond_number = parser.get("fi_max_cond_number", self.max_cond_number)

        print(f"\t- triangulate_1d: {int(self.triangulate_1d)}")
        print(f"\t- refine_features: {int(self.refine_features)}")
        print(f"\t- max_runs: {self.max_runs}")
        print(f"\t- init_lamda: {self.init_lamda:.3f}")
        print(f"\t- max_lamda: {self.max_lamda:.3f}")
        print(f"\t- min_dx: {self.min_dx:.7f}")
        print(f"\t- min_dcost: {self.min_dcost:.7f}")
        print(f"\t- lam_mult: {self.lam_mult:.3f}")
        print(f"\t- min_dist: {self.min_dist:.3f}")
        print(f"\t- max_dist: {self.max_dist:.3f}")
        print(f"\t- max_baseline: {self.max_baseline:.3f}")
        print(f"\t- max_cond_number: {self.max_cond_number:.3f}")