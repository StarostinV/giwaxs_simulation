# -*- coding: utf-8 -*-

from typing import Tuple, Union
import numpy as np


class AngleLimits(object):
    def __init__(self, x_size: int = 1000, y_size: int = 1050):
        self._x_size = self._y_size = None
        self.update_size(x_size, y_size)

    @property
    def size(self) -> Tuple[int, int]:
        return self._x_size, self._y_size

    @property
    def max_radius(self) -> float:
        return np.sqrt(self._x_size ** 2 + self._y_size ** 2)

    def update_size(self, x_size: int, y_size: int) -> None:
        self._x_size = x_size
        self._y_size = y_size

    def min_func(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        with np.errstate(all='ignore'):
            return (r > self._x_size) * np.nan_to_num(np.arccos(self._x_size / r))

    def max_func(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        with np.errstate(all='ignore'):
            return (r <= self._y_size) * np.pi + (r > self._y_size) * np.nan_to_num(np.arcsin(self._y_size / r))
