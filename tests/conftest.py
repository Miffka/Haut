from dataclasses import dataclass

import numpy as np
import pytest
from skimage import draw, transform

gen = np.random.default_rng(24)


@dataclass
class Figure:
    img: np.ndarray
    size: int
    type_: str
    angle: float


@pytest.fixture()
def figure(request):
    type_, size, angle = request.param

    img = np.zeros((size * 2 + 1, size * 2 + 1), dtype=bool)
    if type_ == "square":
        rr, cc = draw.rectangle((size // 2, size // 2), extent=(size, size), shape=img.shape)
        img[rr, cc] = 1
    elif type_ == "circle":
        rr, cc = draw.disk((size, size), size // 2, shape=img.shape)
        img[rr, cc] = 1
    elif type_ == "triangle":
        arr = np.ones((size, size), dtype=bool)
        arr = np.triu(arr)
        img[size // 2 : size + size // 2, size // 2 : size + size // 2] = arr

        rand_angle = gen.uniform(0, 90)
        img = transform.rotate(img, rand_angle)
        rand_scale = gen.uniform(0.3, 3.0)
        img = transform.resize(img, (img.shape[0], img.shape[1] * rand_scale))

    img = transform.rotate(img, angle)

    return Figure(img, size, type_, angle)
