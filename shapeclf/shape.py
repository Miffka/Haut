import logging
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from scipy.spatial.distance import cdist, pdist


@dataclass
class Shape:
    class_name: str
    color: Tuple[int, int, int]


@dataclass
class Shapes:
    CIRCLE = Shape("circle", (255, 0, 0))
    SQUARE = Shape("square", (0, 255, 0))
    TRIANGLE = Shape("triangle", (0, 0, 255))
    UNKNOWN = Shape("unknown", (255, 255, 255))


@dataclass
class Box:
    min_h: int
    min_w: int
    max_h: int
    max_w: int


class FigureProcessor:
    @staticmethod
    def get_bbox(figure: np.ndarray) -> Box:
        assert figure.ndim == 2
        assert np.unique(figure).size == 2

        max_val = figure.max()
        point_coords = np.where(figure == max_val)
        min_h, max_h = point_coords[0].min(), point_coords[0].max()
        min_w, max_w = point_coords[1].min(), point_coords[1].max()

        box = Box(min_h, min_w, max_h + 1, max_w + 1)

        return box

    @staticmethod
    def apply_bbox(figure: np.ndarray, box: Box) -> np.ndarray:
        assert figure.ndim >= 2

        output = figure[box.min_h : box.max_h, box.min_w : box.max_w]

        assert np.prod(figure.shape) >= np.prod(output.shape)
        assert np.count_nonzero(figure) == np.count_nonzero(output)
        return output

    @staticmethod
    def rescale(figure: np.ndarray, target_size: int) -> np.ndarray:
        assert figure.ndim == 2, f"Input ndim: {figure.ndim}"
        assert np.unique(figure).size <= 2, f"Input unique values: {np.unique(figure)}"

        h, w = figure.shape

        if h >= w:
            new_h = target_size
            new_w = int(h / w * target_size)
        else:
            new_w = target_size
            new_h = int(w / h * target_size)

        fig = figure.astype(np.uint8)
        out = cv2.resize(fig, (new_h, new_w), interpolation=cv2.INTER_NEAREST)
        res_h, res_w = out.shape

        assert out.ndim == 2, f"Output ndim: {out.ndim}"
        assert (
            np.unique(out).size == np.unique(figure).size
        ), f"Input unique values: {np.unique(figure)}. output unique values: {np.unique(out)}"
        assert np.isclose(
            h / w, res_h / res_w, rtol=1e-2
        ), f"Size before rescaling: ({h}, {w}), size after rescaling: ({res_h}, {res_w})"

        return out

    @staticmethod
    def process(figure: np.ndarray, target_size: int) -> np.ndarray:
        box = FigureProcessor.get_bbox(figure)
        output = FigureProcessor.apply_bbox(figure, box)
        output = FigureProcessor.rescale(output, target_size)

        return output


class ShapeClassifier:
    def __init__(self, circle_thr: float = 0.08, square_min_cd: float = 5.0, square_rel_d_thr: float = 0.05) -> None:
        self.circle_thr = circle_thr
        self.square_min_cd = square_min_cd
        self.square_rel_d_thr = square_rel_d_thr

    @staticmethod
    def _get_border(figure: np.ndarray) -> np.ndarray:
        fig = figure.astype(np.uint8)
        eroded = cv2.erode(fig, None, iterations=1)
        border = (fig - eroded).astype(figure.dtype)
        return border

    def _predict_dist_based(self, figure: np.ndarray) -> Shape:
        center = (np.asarray(figure.shape) / 2).astype(int)
        border = ShapeClassifier._get_border(figure)
        border_coords = np.asarray(np.where(border > 0)).T

        center2border_dist = cdist(center.reshape(1, -1), border_coords)[0]

        relative_std = center2border_dist.std() / center2border_dist.mean()
        logging.debug(f"Relative std calculated: {relative_std}, CIRCLE threshold: {self.circle_thr}")
        if relative_std <= self.circle_thr:
            return Shapes.CIRCLE

        # if there are at least 4 points in different parts of image with
        # approximately equal distance - they are corners of a square
        farthest4_ids = np.argsort(center2border_dist)[-4:]
        pdist_4_points = pdist(border_coords[farthest4_ids])
        center2farthest4_dist = center2border_dist[farthest4_ids]
        rel_c2f_d = np.abs((center2farthest4_dist - center2farthest4_dist.mean()) / center2farthest4_dist.mean())

        if np.all(pdist_4_points > self.square_min_cd) and np.all(rel_c2f_d < self.square_rel_d_thr):
            return Shapes.SQUARE

        return Shapes.TRIANGLE

    def __call__(self, figure: np.ndarray) -> Shape:
        shape = self._predict_dist_based(figure)

        return shape
