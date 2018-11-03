"""Unit tests for architecture_utils.py."""

import unittest
import numpy
from gewittergefahr.deep_learning import architecture_utils

NUM_INPUT_UNITS = 1000

FIRST_NUM_CLASSES = 3
FIRST_NUM_DENSE_LAYERS = 3
FIRST_NUM_INPUTS_BY_LAYER = numpy.array([1000, 144, 21], dtype=int)
FIRST_NUM_OUTPUTS_BY_LAYER = numpy.array([144, 21, 3], dtype=int)

SECOND_NUM_CLASSES = 2
SECOND_NUM_DENSE_LAYERS = 3
SECOND_NUM_INPUTS_BY_LAYER = numpy.array([1000, 100, 10], dtype=int)
SECOND_NUM_OUTPUTS_BY_LAYER = numpy.array([100, 10, 1], dtype=int)

THIRD_NUM_CLASSES = 3
THIRD_NUM_DENSE_LAYERS = 2
THIRD_NUM_INPUTS_BY_LAYER = numpy.array([1000, 55], dtype=int)
THIRD_NUM_OUTPUTS_BY_LAYER = numpy.array([55, 3], dtype=int)

FOURTH_NUM_CLASSES = 2
FOURTH_NUM_DENSE_LAYERS = 2
FOURTH_NUM_INPUTS_BY_LAYER = numpy.array([1000, 32], dtype=int)
FOURTH_NUM_OUTPUTS_BY_LAYER = numpy.array([32, 1], dtype=int)

FIFTH_NUM_CLASSES = 3
FIFTH_NUM_DENSE_LAYERS = 1
FIFTH_NUM_INPUTS_BY_LAYER = numpy.array([NUM_INPUT_UNITS], dtype=int)
FIFTH_NUM_OUTPUTS_BY_LAYER = numpy.array([FIFTH_NUM_CLASSES], dtype=int)

SIXTH_NUM_CLASSES = 2
SIXTH_NUM_DENSE_LAYERS = 1
SIXTH_NUM_INPUTS_BY_LAYER = numpy.array([NUM_INPUT_UNITS], dtype=int)
SIXTH_NUM_OUTPUTS_BY_LAYER = numpy.array([1], dtype=int)


class ArchitectureUtilsTests(unittest.TestCase):
    """Each method is a unit test for architecture_utils.py."""

    def test_get_dense_layer_dimensions_first(self):
        """Ensures correct output from get_dense_layer_dimensions."""

        these_num_input_units, these_num_output_units = (
            architecture_utils.get_dense_layer_dimensions(
                num_input_units=NUM_INPUT_UNITS, num_classes=FIRST_NUM_CLASSES,
                num_dense_layers=FIRST_NUM_DENSE_LAYERS)
        )

        self.assertTrue(numpy.array_equal(
            these_num_input_units, FIRST_NUM_INPUTS_BY_LAYER))
        self.assertTrue(numpy.array_equal(
            these_num_output_units, FIRST_NUM_OUTPUTS_BY_LAYER))

    def test_get_dense_layer_dimensions_second(self):
        """Ensures correct output from get_dense_layer_dimensions."""

        these_num_input_units, these_num_output_units = (
            architecture_utils.get_dense_layer_dimensions(
                num_input_units=NUM_INPUT_UNITS, num_classes=SECOND_NUM_CLASSES,
                num_dense_layers=SECOND_NUM_DENSE_LAYERS)
        )

        self.assertTrue(numpy.array_equal(
            these_num_input_units, SECOND_NUM_INPUTS_BY_LAYER))
        self.assertTrue(numpy.array_equal(
            these_num_output_units, SECOND_NUM_OUTPUTS_BY_LAYER))

    def test_get_dense_layer_dimensions_third(self):
        """Ensures correct output from get_dense_layer_dimensions."""

        these_num_input_units, these_num_output_units = (
            architecture_utils.get_dense_layer_dimensions(
                num_input_units=NUM_INPUT_UNITS, num_classes=THIRD_NUM_CLASSES,
                num_dense_layers=THIRD_NUM_DENSE_LAYERS)
        )

        self.assertTrue(numpy.array_equal(
            these_num_input_units, THIRD_NUM_INPUTS_BY_LAYER))
        self.assertTrue(numpy.array_equal(
            these_num_output_units, THIRD_NUM_OUTPUTS_BY_LAYER))

    def test_get_dense_layer_dimensions_fourth(self):
        """Ensures correct output from get_dense_layer_dimensions."""

        these_num_input_units, these_num_output_units = (
            architecture_utils.get_dense_layer_dimensions(
                num_input_units=NUM_INPUT_UNITS, num_classes=FOURTH_NUM_CLASSES,
                num_dense_layers=FOURTH_NUM_DENSE_LAYERS)
        )

        self.assertTrue(numpy.array_equal(
            these_num_input_units, FOURTH_NUM_INPUTS_BY_LAYER))
        self.assertTrue(numpy.array_equal(
            these_num_output_units, FOURTH_NUM_OUTPUTS_BY_LAYER))

    def test_get_dense_layer_dimensions_fifth(self):
        """Ensures correct output from get_dense_layer_dimensions."""

        these_num_input_units, these_num_output_units = (
            architecture_utils.get_dense_layer_dimensions(
                num_input_units=NUM_INPUT_UNITS, num_classes=FIFTH_NUM_CLASSES,
                num_dense_layers=FIFTH_NUM_DENSE_LAYERS)
        )

        self.assertTrue(numpy.array_equal(
            these_num_input_units, FIFTH_NUM_INPUTS_BY_LAYER))
        self.assertTrue(numpy.array_equal(
            these_num_output_units, FIFTH_NUM_OUTPUTS_BY_LAYER))

    def test_get_dense_layer_dimensions_sixth(self):
        """Ensures correct output from get_dense_layer_dimensions."""

        these_num_input_units, these_num_output_units = (
            architecture_utils.get_dense_layer_dimensions(
                num_input_units=NUM_INPUT_UNITS, num_classes=SIXTH_NUM_CLASSES,
                num_dense_layers=SIXTH_NUM_DENSE_LAYERS)
        )

        self.assertTrue(numpy.array_equal(
            these_num_input_units, SIXTH_NUM_INPUTS_BY_LAYER))
        self.assertTrue(numpy.array_equal(
            these_num_output_units, SIXTH_NUM_OUTPUTS_BY_LAYER))


if __name__ == '__main__':
    unittest.main()
