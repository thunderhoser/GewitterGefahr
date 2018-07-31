"""Unit tests for feature_optimization_plotting.py."""

import unittest
import numpy
from gewittergefahr.deep_learning import feature_optimization
from gewittergefahr.plotting import \
    feature_optimization_plotting as fopt_plotting

TARGET_CLASS = 1
EXAMPLE_INDEX = 2
LAYER_NAME = 'average_pooling2d_3'
CHANNEL_INDEX_BY_EXAMPLE = numpy.array([2, 5, 8], dtype=int)
NEURON_INDEX_MATRIX = numpy.array([[0, 0, 2],
                                   [0, 1, 2],
                                   [1, 1, 2]], dtype=int)

CLASS_OPTIMIZN_VERBOSE_STRING = 'Class 1'
CLASS_OPTIMIZN_ABBREV_STRING = 'class1'
CHANNEL_OPTIMIZN_VERBOSE_STRING = 'Layer "average_pooling2d_3", channel 8'
CHANNEL_OPTIMIZN_ABBREV_STRING = 'layer=average-pooling2d-3_channel8'
NEURON_OPTIMIZN_VERBOSE_STRING = 'Layer "average_pooling2d_3"; neuron (1, 1, 2)'
NEURON_OPTIMIZN_ABBREV_STRING = 'layer=average-pooling2d-3_neuron1,1,2'


class FeatureOptimizationPlottingTests(unittest.TestCase):
    """Each method is a unit test for feature_optimization_plotting.py."""

    def test_optimization_metadata_to_strings_class(self):
        """Ensures correct output from _optimization_metadata_to_strings.

        In this case, optimization is for class probabilities.
        """

        (this_verbose_string, this_abbrev_string
        ) = fopt_plotting._optimization_metadata_to_strings(
            component_type_string=
            feature_optimization.CLASS_COMPONENT_TYPE_STRING,
            example_index=EXAMPLE_INDEX, target_class=TARGET_CLASS,
            layer_name=LAYER_NAME,
            channel_index_by_example=CHANNEL_INDEX_BY_EXAMPLE,
            neuron_index_matrix=NEURON_INDEX_MATRIX)

        self.assertTrue(this_verbose_string == CLASS_OPTIMIZN_VERBOSE_STRING)
        self.assertTrue(this_abbrev_string == CLASS_OPTIMIZN_ABBREV_STRING)

    def test_optimization_metadata_to_strings_channel(self):
        """Ensures correct output from _optimization_metadata_to_strings.

        In this case, optimization is for channel activations.
        """

        (this_verbose_string, this_abbrev_string
        ) = fopt_plotting._optimization_metadata_to_strings(
            component_type_string=
            feature_optimization.CHANNEL_COMPONENT_TYPE_STRING,
            example_index=EXAMPLE_INDEX, target_class=TARGET_CLASS,
            layer_name=LAYER_NAME,
            channel_index_by_example=CHANNEL_INDEX_BY_EXAMPLE,
            neuron_index_matrix=NEURON_INDEX_MATRIX)

        self.assertTrue(this_verbose_string == CHANNEL_OPTIMIZN_VERBOSE_STRING)
        self.assertTrue(this_abbrev_string == CHANNEL_OPTIMIZN_ABBREV_STRING)

    def test_optimization_metadata_to_strings_neuron(self):
        """Ensures correct output from _optimization_metadata_to_strings.

        In this case, optimization is for neuron activations.
        """

        (this_verbose_string, this_abbrev_string
        ) = fopt_plotting._optimization_metadata_to_strings(
            component_type_string=
            feature_optimization.NEURON_COMPONENT_TYPE_STRING,
            example_index=EXAMPLE_INDEX, target_class=TARGET_CLASS,
            layer_name=LAYER_NAME,
            channel_index_by_example=CHANNEL_INDEX_BY_EXAMPLE,
            neuron_index_matrix=NEURON_INDEX_MATRIX)

        self.assertTrue(this_verbose_string == NEURON_OPTIMIZN_VERBOSE_STRING)
        self.assertTrue(this_abbrev_string == NEURON_OPTIMIZN_ABBREV_STRING)


if __name__ == '__main__':
    unittest.main()
