class UpdaterOptions:
    """
    Struct which stores general updater options.
    Used for configuring MSCKF, SLAM, and ARUCO update steps.
    """
    def __init__(self):
        # What chi-squared multiplier we should apply
        self.chi2_multipler = 5.0

        # Noise sigma for our raw pixel measurements
        self.sigma_pix = 1.0

        # Covariance for our raw pixel measurements (sigma_pix^2)
        # Note: This is usually calculated after loading sigma_pix
        self.sigma_pix_sq = 1.0

        # Robustness guards to skip numerically weak updates.
        self.min_features_for_update = 3
        self.min_update_rows = 6
        self.max_innovation_condition = 1e12
        self.ekf_innovation_jitter = 1e-9

    def print(self):
        """
        Nice print function of what parameters we have loaded.
        """
        print(f"    - chi2_multipler: {self.chi2_multipler:.1f}")
        print(f"    - sigma_pix: {self.sigma_pix:.2f}")
        print(f"    - min_features_for_update: {self.min_features_for_update}")
        print(f"    - min_update_rows: {self.min_update_rows}")
        print(f"    - max_innovation_condition: {self.max_innovation_condition:.2e}")
        print(f"    - ekf_innovation_jitter: {self.ekf_innovation_jitter:.1e}")