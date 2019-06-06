from __future__ import division

import cv2
import numpy as np

def get_connected_component(img, min_area=0, sort=True):
    import process
    if img.dtype == bool:
        img = img.astype(np.uint8)

    _, label, stats, centroids = cv2.connectedComponentsWithStats(img)
    components = [
        ConnectedComponent(
            label == level, level, stat,
            process.get_revese_point(centroid.astype(np.int))
        ) for level, (stat, centroid) in enumerate(zip(stats, centroids))
        if stat[4] > min_area and level != 0
    ]
    if sort:
        components = sorted(components, reverse=True)
    return components, label

class ConnectedComponent(object):
    def __init__(self, img, level, stat, centroid):
        self.img = img
        self.shape = img.shape
        self.level = level
        self.left = stat[0]
        self.right = stat[0] + stat[2] - 1
        self.up = stat[1]
        self.down = stat[1] + stat[3] - 1
        self.area = stat[4]
        self.centroid = centroid
        self.info = None

    def __lt__(self, other):
        return self.area < other.area

    def __le__(self, other):
        return self.area <= other.area

    def __add__(self, other):
        if isinstance(other, ConnectedComponent):
            return self.img_bool + other.img_bool
        elif isinstance(other, np.ndarray):
            return self.img_bool + other
        raise TypeError('ConnectedComponent must add with ConnectedComponent or numpy.ndarray')

    @staticmethod
    def get_connected_component(img, min_area=0, sort=True):
        get_connected_component(img, min_area, sort)

    @property
    def img_uint8(self):
        if self.img.dtype == np.uint8:
            return self.img
        return (self.img * 255 / self.img.max()).astype(np.uint8)

    @property
    def img_bool(self):
        if self.img.dtype == bool:
            return self.img
        return self.img.astype(bool)

    @property
    def canvas(self, dtype=np.uint8):
        return np.zeros(self.shape).astype(dtype)

    def get_bound_point(self, type_='l'):
        import process
        return process.get_bound_point(self.img, type_)

    # def __contains__(self, point):
    #     if process.is_point(point):
    #         return self.up <= point[0] < self.down and self.left <= point[1] < self.right
    #     return False

    def in_range(self, left=None, right=None, up=None, down=None):
        if left is not None and left > self.left:
            return False
        if right is not None and right < self.right:
            return False
        if up is not None and up > self.up:
            return False
        if down is not None and down < self.down:
            return False
        return True
