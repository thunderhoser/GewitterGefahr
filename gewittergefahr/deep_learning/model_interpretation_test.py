"""Unit tests for model_interpretation.py."""

import unittest
import numpy
from gewittergefahr.deep_learning import model_interpretation

TARGET_CLASS = 1
LAYER_NAME = 'average_pooling2d_3'
CHANNEL_INDEX = 8
NEURON_INDICES = numpy.array([0, 3, 1], dtype=int)

CLASS_OPTIMIZN_VERBOSE_STRING = 'Class 1'
CLASS_OPTIMIZN_ABBREV_STRING = 'class1'
CHANNEL_OPTIMIZN_VERBOSE_STRING = 'Layer "average_pooling2d_3", channel 8'
CHANNEL_OPTIMIZN_ABBREV_STRING = 'layer=average-pooling2d-3_channel8'
NEURON_OPTIMIZN_VERBOSE_STRING = 'Layer "average_pooling2d_3"; neuron (0, 3, 1)'
NEURON_OPTIMIZN_ABBREV_STRING = 'layer=average-pooling2d-3_neuron0,3,1'


class ModelInterpretationTests(unittest.TestCase):
    """Each method is a unit test for model_interpretation.py."""

    def test_model_component_to_string_class(self):
        """Ensures correct output from model_component_to_string.

        In this case, the component is an output class.
        """

        (this_verbose_string, this_abbrev_string
        ) = model_interpretation.model_component_to_string(
            component_type_string=
            model_interpretation.CLASS_COMPONENT_TYPE_STRING,
            target_class=TARGET_CLASS, layer_name=LAYER_NAME,
            neuron_indices=NEURON_INDICES, channel_index=CHANNEL_INDEX)

        self.assertTrue(this_verbose_string == CLASS_OPTIMIZN_VERBOSE_STRING)
        self.assertTrue(this_abbrev_string == CLASS_OPTIMIZN_ABBREV_STRING)

    def test_model_component_to_string_neuron(self):
        """Ensures correct output from model_component_to_string.

        In this case, the component is a neuron.
        """

        (this_verbose_string, this_abbrev_string
        ) = model_interpretation.model_component_to_string(
            component_type_string=
            model_interpretation.NEURON_COMPONENT_TYPE_STRING,
            target_class=TARGET_CLASS, layer_name=LAYER_NAME,
            neuron_indices=NEURON_INDICES, channel_index=CHANNEL_INDEX)

        self.assertTrue(this_verbose_string == NEURON_OPTIMIZN_VERBOSE_STRING)
        self.assertTrue(this_abbrev_string == NEURON_OPTIMIZN_ABBREV_STRING)

    def test_model_component_to_string_channel(self):
        """Ensures correct output from model_component_to_string.

        In this case, the component is a channel.
        """

        (this_verbose_string, this_abbrev_string
        ) = model_interpretation.model_component_to_string(
            component_type_string=
            model_interpretation.CHANNEL_COMPONENT_TYPE_STRING,
            target_class=TARGET_CLASS, layer_name=LAYER_NAME,
            neuron_indices=NEURON_INDICES, channel_index=CHANNEL_INDEX)

        self.assertTrue(this_verbose_string == CHANNEL_OPTIMIZN_VERBOSE_STRING)
        self.assertTrue(this_abbrev_string == CHANNEL_OPTIMIZN_ABBREV_STRING)


if __name__ == '__main__':
    unittest.main()
