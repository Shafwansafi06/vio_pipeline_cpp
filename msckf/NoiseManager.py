import math

class NoiseManager:
    """
    Struct of our IMU noise parameters.
    """
    def __init__(self):
        # Gyroscope white noise (rad/s/sqrt(hz))
        self.sigma_w = 1.6968e-04
        
        # Gyroscope white noise covariance
        self.sigma_w_2 = self.sigma_w ** 2

        # Gyroscope random walk (rad/s^2/sqrt(hz))
        self.sigma_wb = 1.9393e-05
        
        # Gyroscope random walk covariance
        self.sigma_wb_2 = self.sigma_wb ** 2

        # Accelerometer white noise (m/s^2/sqrt(hz))
        self.sigma_a = 2.0000e-3
        
        # Accelerometer white noise covariance
        self.sigma_a_2 = self.sigma_a ** 2

        # Accelerometer random walk (m/s^3/sqrt(hz))
        self.sigma_ab = 3.0000e-03
        
        # Accelerometer random walk covariance
        self.sigma_ab_2 = self.sigma_ab ** 2

    def print(self):
        """
        Nice print function of what parameters we have loaded.
        """
        print(f"  - gyroscope_noise_density: {self.sigma_w:.6f}")
        print(f"  - accelerometer_noise_density: {self.sigma_a:.5f}")
        print(f"  - gyroscope_random_walk: {self.sigma_wb:.7f}")
        print(f"  - accelerometer_random_walk: {self.sigma_ab:.6f}")