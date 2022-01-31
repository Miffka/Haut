from itertools import product

import pytest

from shapeclf.figure import FigureClassifier, FigureProcessor


class TestFigureProcessor:
    @pytest.mark.parametrize(
        "figure", list(product(["square", "triangle"], [71, 101, 131], [0, 30, 45, 60, 90])), indirect=True
    )
    def test_angle_process_figure(self, figure):
        out = FigureProcessor.process(figure.img, target_size=figure.size)
        assert min(out.shape) == figure.size

    @pytest.mark.parametrize("figure", list(product(["circle"], [71, 101, 131], [0])), indirect=True)
    def test_noangle_process_figure(self, figure):
        out = FigureProcessor.process(figure.img, target_size=figure.size)
        assert min(out.shape) == figure.size


class TestFigureClassifier:
    @pytest.mark.parametrize(
        "figure", list(product(["square", "triangle"], [71, 101, 131], [0, 30, 45, 60, 90])), indirect=True
    )
    def test_angle_predict(self, figure):
        out = FigureProcessor.process(figure.img, target_size=figure.size)
        classifier = FigureClassifier()
        shape = classifier._predict(out)
        assert shape.class_name == figure.type_

    @pytest.mark.parametrize("figure", list(product(["circle"], [71, 101, 131], [0])), indirect=True)
    def test_noangle_predict(self, figure):
        out = FigureProcessor.process(figure.img, target_size=figure.size)
        classifier = FigureClassifier()
        shape = classifier._predict(out)
        assert shape.class_name == figure.type_
