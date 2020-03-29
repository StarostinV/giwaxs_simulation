# -*- coding: utf-8 -*-
from typing import NamedTuple, Tuple

import numpy as np

from .angle_limits import AngleLimits


class SimProps(NamedTuple):
    w_min: float  # minimal segment width
    w_max: float  # maximal segment width
    min_distance: float  # min distance between segments
    min_a: float  # min angle width of a segment
    max_a: float  # max angle width of a segment


class Labels(NamedTuple):
    rs: np.ndarray
    ws: np.ndarray
    a_max: np.ndarray
    a_min: np.ndarray
    classes: np.ndarray  # 1 for rings, 2 for segments, do not use Enums for simplicity

    @property
    def size(self):
        return self.classes.size

    @property
    def coords(self):
        return np.stack([self.rs, self.ws, (self.a_max + self.a_min) / 2,  self.a_max - self.a_min],
                        axis=1)


class SimulateLabels(object):
    @property
    def props(self) -> SimProps:
        return self._props

    @props.setter
    def props(self, value: SimProps):
        self._props = value

    @property
    def img_size(self) -> Tuple[int, int]:
        return self._al.size

    @property
    def angle_limits(self):
        return self._al

    def __init__(self, props: SimProps):
        self._props = props
        self._al = AngleLimits(0, 0)

    def set_img_size(self, x_size: int, y_size: int):
        self._al.update_size(x_size, y_size)

    def simulate_labels(self, num_of_rings: int, num_of_segments: int):
        rs, ws = self._simulate_radii(num_of_rings + num_of_segments)
        a_min, a_max, classes = self._simulate_angles(rs, num_of_rings / (num_of_rings + num_of_segments))

        idx = np.logical_not(np.isnan(classes))

        return Labels(rs[idx], ws[idx], a_max[idx], a_min[idx], classes[idx])

    def _simulate_radii(self, num: int):
        if not num:
            return np.array([]), np.array([])
        rs = np.sort(np.random.uniform(self._props.w_max, self._al.max_radius - self._props.w_max, num))
        ws = np.random.uniform(self._props.w_min, self._props.w_max, num)
        if not rs.size:
            return self._simulate_radii(num)
        distances = rs[1:] - rs[:-1] - (ws[:-1] + ws[1:]) / 2 - self._props.min_distance
        inds = np.append([True], (distances > 0))
        return rs[inds], ws[inds]

    def _simulate_angles(self, rs: np.ndarray, rings_share: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        # TODO add several segments at same r positions

        num = rs.size
        if not num:
            return np.array([]), np.array([]), np.array([])
        ring_num = int(num * rings_share)
        seg_num = num - ring_num
        classes = np.ones_like(rs)
        classes[:seg_num] = 2
        np.random.shuffle(classes)

        a_min, a_max = self._al.min_func(rs), self._al.max_func(rs)

        for i, (a1, a2, cl) in enumerate(zip(a_min, a_max, classes)):
            if cl == 2:  # segment
                if a2 - a1 < self.props.max_a * 2:
                    classes[i] = np.nan
                    continue
                n_w, n_p = np.random.uniform(0, 1, 2)
                a_w = self.props.min_a + n_w * (self.props.max_a - self.props.min_a)
                a = (a1 + a_w / 2) + (a2 - a1 - a_w) * n_p
                a_min[i] = a - a_w / 2
                a_max[i] = a + a_w / 2

        return a_min, a_max, classes
