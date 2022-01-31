import logging
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from skimage import draw, measure


@dataclass
class Shape:
    class_name: str
    color: Tuple[int, int, int]
    color_name: str


@dataclass
class Shapes:
    CIRCLE = Shape("circle", (255, 0, 0), "blue")
    SQUARE = Shape("square", (0, 255, 0), "green")
    TRIANGLE = Shape("triangle", (0, 0, 255), "red")


@dataclass
class Box:
    min_h: int
    min_w: int
    max_h: int
    max_w: int

    @property
    def width(self):
        return self.max_w - self.min_w

    @property
    def height(self):
        return self.max_h - self.min_h


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

        # Padding allows to get at least 2 unique values
        hpad = int(box.height * 0.1)
        wpad = int(box.width * 0.1)
        output = np.pad(output, ((hpad, hpad), (wpad, wpad)))

        assert np.prod(figure.shape) >= np.prod(output.shape)
        assert np.count_nonzero(figure) == np.count_nonzero(output)
        return output

    @staticmethod
    def rescale(figure: np.ndarray, target_size: int) -> np.ndarray:
        assert figure.ndim == 2, f"Input ndim: {figure.ndim}"
        assert np.unique(figure).size == 2, f"Input unique values: {np.unique(figure)}"

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
            h / w, res_h / res_w, rtol=1e-1
        ), f"Size before rescaling: ({h}, {w}), size after rescaling: ({res_h}, {res_w})"

        return out

    @staticmethod
    def process(figure: np.ndarray, target_size: int) -> np.ndarray:
        box = FigureProcessor.get_bbox(figure)
        output = FigureProcessor.apply_bbox(figure, box)
        output = FigureProcessor.rescale(output, target_size)

        return output


class FigureClassifier:
    def __init__(
        self, circle_thr: float = 0.05, inscribed_disk_dilation: int = 5, min_area_threshold: int = 10
    ) -> None:
        self.circle_thr = circle_thr
        self.inscribed_disk_dilation = inscribed_disk_dilation
        self.min_area_threshold = min_area_threshold

    @staticmethod
    def _get_border(figure: np.ndarray, pad: int = 2) -> np.ndarray:
        fig = figure.astype(np.uint8)
        eroded = cv2.erode(fig, None, iterations=1)
        border = (fig - eroded).astype(figure.dtype)
        return border

    @staticmethod
    def _get_median(hist: np.ndarray) -> int:
        rel_cumsum = np.cumsum(hist) / hist.sum()
        median_point = np.where(rel_cumsum > 0.5)[0][0]
        return median_point

    @staticmethod
    def _get_center_of_mass(figure: np.ndarray) -> Tuple[int, int]:
        center = np.zeros(2, dtype=int)
        center[0] = FigureClassifier._get_median(figure.sum(axis=1))
        center[1] = FigureClassifier._get_median(figure.sum(axis=0))
        return center

    def _get_inscribed_disk(
        self, center: np.ndarray, center2border_dist: np.ndarray, shape: Tuple[int, int]
    ) -> np.ndarray:
        center2closest_dist = center2border_dist.min()
        inscribed_disk = np.zeros(shape, dtype=bool)
        rr, cc = draw.disk(center, center2closest_dist + self.inscribed_disk_dilation, shape=shape)
        inscribed_disk[rr, cc] = True
        logging.debug(
            f"Inscribed disk with shape {shape}, center {center}, "
            + f"and radius {center2closest_dist + self.inscribed_disk_dilation} is drawn"
        )
        return inscribed_disk

    def _predict(self, figure: np.ndarray) -> Shape:
        center = FigureClassifier._get_center_of_mass(figure)
        border = FigureClassifier._get_border(figure)
        border_coords = np.asarray(np.where(border > 0)).T

        center2border_dist = cdist(center.reshape(1, -1), border_coords)[0]

        # Check for CIRCLE
        relative_std = center2border_dist.std() / center2border_dist.mean()
        logging.debug(f"Relative std calculated: {relative_std}, CIRCLE threshold: {self.circle_thr}")
        if relative_std <= self.circle_thr:
            return Shapes.CIRCLE

        # Split figure by dilated inscribed disk
        inscribed_disk = self._get_inscribed_disk(center, center2border_dist, shape=figure.shape)
        out = figure & ~inscribed_disk

        # Check how many connected components are there
        out = measure.label(out)
        props = measure.regionprops(out)
        props = [prop for prop in props if prop.area > self.min_area_threshold]
        n_comp = len(props)

        # Check for SQUARE
        if n_comp == 4:
            return Shapes.SQUARE

        return Shapes.TRIANGLE

    def __call__(self, figure: np.ndarray) -> Shape:
        shape = self._predict(figure)

        return shape
