from operator import imod
import numpy as np
from dataclasses import dataclass, field
from typing import List
import cv2

@dataclass(order=True)
class ImuData:
    timestamp: float

    wm: np.ndarray = field(compare=False)
    am: np.ndarray = field(compare=False)

    def __post_init__(self):
        if not isinstance(self.wm, np.ndarray):
            self.wm = np.array(self.wm)
        if not isinstance(self.am, np.ndarray):
            self.am = np.array(self.am)

class CameraData:
    def __init__(self):
        self.timestamp = 0.0
        self.sensor_ids: List[int] = []
        self.images: List[np.ndarray] = []
        self.masks: List[np.ndarray] = []
    def __lt__(self, other):
        if not isinstance(other, CameraData):
            return NotImplemented
        if self.timestamp == other.timestamp:
            # If timestamps equal, sort by smallest sensor ID
            id_self = min(self.sensor_ids) if self.sensor_ids else -1
            id_other = min(other.sensor_ids) if other.sensor_ids else -1
            return id_self < id_other
        else:
            return self.timestamp < other.timestamp