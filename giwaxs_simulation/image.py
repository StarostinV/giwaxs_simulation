# -*- coding: utf-8 -*-

from typing import NamedTuple, Tuple

import numpy as np

from gauss_map import gauss_map
from .labels import Labels, SimProps, SimulateLabels
from .utils import time_counter


class GeneralProps(NamedTuple):
    # image size range:
    img_size_min: int
    img_size_max: int

    # total number of segments range:
    seg_num_min: int
    seg_num_max: int

    # segments intensities range:
    intensity_min: int
    intensity_max: int

    # Noise and background props:
    poisson_lambda: float
    background_min: int
    background_max: int
    background_num: int
    b_period_min: float
    b_period_max: float


class SimulateImage(object):
    def __init__(self, sim_props: SimProps, gen_props: GeneralProps):
        self._sim_labels = SimulateLabels(sim_props)
        self._gen_props = gen_props
        self._xx = self._yy = self._rr = self._phi = None
        self._update_map()

    @property
    def sim_props(self) -> SimProps:
        return self._sim_labels.props

    @sim_props.setter
    def sim_props(self, value: SimProps):
        self._sim_labels.props = value

    @property
    def gen_props(self) -> GeneralProps:
        return self._gen_props

    @gen_props.setter
    def gen_props(self, value: GeneralProps):
        self._gen_props = value
        self._update_map()

    @property
    def im_size(self) -> Tuple[int, int]:
        return self._sim_labels.img_size

    @property
    def max_radius(self) -> float:
        return np.sqrt(sum((x ** 2 for x in self.im_size)))

    @time_counter
    def _update_map(self):
        x_size, y_size = self.im_size
        self._xx, self._yy = np.meshgrid(np.arange(x_size), np.arange(y_size))
        self._rr = np.sqrt(self._xx ** 2 + self._yy ** 2)  # beam center is (0, 0) so far
        self._phi = np.arctan2(self._yy, self._xx)

    def gen_img_size(self) -> Tuple[int, int]:
        size = np.random.randint(self.gen_props.img_size_min, self.gen_props.img_size_max, 2)
        return tuple(size)

    def gen_nums(self) -> Tuple[int, int]:
        nums = np.random.randint(self.gen_props.seg_num_min, self.gen_props.seg_num_max, 2)
        return tuple(nums)

    def gen_intensities(self, num: int):
        return np.random.randint(self.gen_props.intensity_min, self.gen_props.intensity_max, num).astype(np.float)

    def simulate_image(self, label: int) -> np.ndarray:
        img_size = self.gen_img_size()
        self._sim_labels.set_img_size(*img_size)
        self._update_map()
        num_of_rings, num_of_segments = self.gen_nums()
        if label == 0:
            num_of_rings, num_of_segments = num_of_rings + num_of_segments, 0
        if label == 1:
            num_of_rings, num_of_segments = 0, num_of_rings + num_of_segments

        labels = self._sim_labels.simulate_labels(num_of_rings, num_of_segments)
        img = self._simulate_img(labels)
        # img = self._apply_noise(img)
        img = self._apply_background(img)
        img = self._apply_noise(img)
        return img

    @time_counter
    def _simulate_img(self, labels: Labels) -> np.ndarray:
        img = np.random.poisson(self.gen_props.poisson_lambda, self.im_size).astype(np.float).T
        intensities = self.gen_intensities(labels.classes.size)

        return gauss_map(img, self._rr, self._phi,
                         labels.rs, labels.ws, (labels.a_max + labels.a_min) / 2,
                         labels.a_max - labels.a_min, intensities)
        # for intensity, (r, w, a, a_std) in zip(intensities, labels.coords):
        #     img += self._gauss_map(r, w, a, a_std) * intensity
        # return img

    @time_counter
    def _apply_noise(self, img: np.ndarray) -> np.ndarray:
        return img * np.random.poisson(self.gen_props.poisson_lambda, img.shape) \
               + np.random.poisson(self.gen_props.poisson_lambda, img.shape).astype(np.float)

    @time_counter
    def _apply_background(self, img: np.ndarray):
        img += _gen_random_background(self._rr, (self.gen_props.background_min, self.gen_props.background_max),
                                      (self.gen_props.b_period_min, self.gen_props.b_period_max),
                                      self.gen_props.background_num)
        # img += _gen_random_background(self._phi, (self.gen_props.background_min, self.gen_props.background_max),
        #                               (self.gen_props.b_period_min, self.gen_props.b_period_max),
        #                               self.gen_props.background_num)
        return img

    # def _gauss_map(self, r, w, a, a_w):
    #     if not w or not a_w:
    #         raise ValueError(f'Non-positive gaussian widths: w={w}, a_w={a_w}')
    #     return np.exp(- 8 * (self._rr - r) ** 2 / w ** 2 - 8 * (self._phi - a) ** 2 / a_w ** 2)


def _gen_random_background(xx: np.ndarray, intensity_range: Tuple[float, float],
                           period_range: Tuple[float, float],
                           num_of_peaks: int = 4):
    xx = xx / np.amax(xx)
    periods = np.random.uniform(*period_range, num_of_peaks)
    phases = np.random.uniform(0, 2 * np.pi, num_of_peaks)
    amps = np.random.uniform(*intensity_range, num_of_peaks)
    img = np.zeros_like(xx)
    for t, p, a in zip(periods, phases, amps):
        img += np.cos(xx * t + p) ** 8 * a  # some very random function
    return img
