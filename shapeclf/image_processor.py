import logging

import numpy as np
from skimage import measure

from shapeclf.figure import FigureClassifier, FigureProcessor, Shape


class ImageProcessor:
    def __init__(self, *args, area_threshold: int = 10, target_size: int = 101, **kwargs) -> None:
        self.area_threshold = area_threshold
        self.target_size = target_size
        self.shape_classifier = FigureClassifier(*args, **kwargs)

    @staticmethod
    def _divide_to_figures(image: np.ndarray) -> np.ndarray:
        img_divided = measure.label(image)
        return img_divided

    def _process_one_figure(self, figure: np.ndarray) -> Shape:
        fig = FigureProcessor.process(figure, target_size=self.target_size)
        shape = self.shape_classifier(fig)

        return shape

    def __call__(self, image: np.ndarray) -> np.ndarray:
        assert image.ndim == 2
        assert np.unique(image).size == 2

        img = ImageProcessor._divide_to_figures(image)
        figure_ids = np.unique(img)[1:]
        out_img = np.zeros((*img.shape, 3), dtype=np.uint8)

        for figure_idx in figure_ids:
            figure_mask = img == figure_idx
            if np.count_nonzero(figure_mask) < self.area_threshold:
                continue

            y_coord = np.argmax(figure_mask.sum(axis=(1)))
            x_coord = np.argmax(figure_mask.sum(axis=(0)))
            figure_shape = self._process_one_figure(figure_mask)
            logging.debug(
                f"Figure {figure_idx} at position (h, w) ({y_coord}, {x_coord}) has shape {figure_shape.class_name}"
            )

            out_img[figure_mask] = figure_shape.color

        return out_img
