# -*- coding: utf-8 -*-
from typing import Tuple

from h5py import File
from tqdm import tqdm
from numpy import random, ndarray

from .image import SimulateImage

_LABEL_DICT = {0: 'rings', 1: 'segments', 2: 'mixed'}


def save_simulation(sim_image: SimulateImage,
                    num: int, filename: str,
                    rewrite: bool = False) -> None:
    if rewrite:
        mode = 'w'
        first_key = 0
    else:
        mode = 'a'
        with File(filename, 'a') as f:
            if len(f.keys()):
                first_key = max(map(int, list(f.keys()))) + 1
            else:
                first_key = 0
    for i in tqdm(range(first_key, num + first_key)):
        label = random.randint(0, 3)
        img = sim_image.simulate_image(label)
        with File(filename, mode) as f:
            img_dset = f.create_dataset(str(i), data=img)
            img_dset.attrs['class'] = label
            img_dset.attrs['class_name'] = _LABEL_DICT[label]


def read_simulation(filename: str, num: int) -> Tuple[ndarray, int]:
    with File(filename, 'r') as f:
        try:
            img_dset = f[str(num)]
        except KeyError:
            raise IndexError(f'No dataset {num}.')
        img = img_dset[()]
        label = img_dset.attrs['class']
    return img, label
