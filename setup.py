from setuptools import setup, find_packages

setup(
    name="vio_pipeline",
    version="0.1.0",
    packages=find_packages(),  # Automatically finds 'vio_pipeline' and subfolders
    install_requires=[
        "numpy",
        "scipy",
        "opencv-python",
        "pyyaml",
        "rospkg" 
    ],
)