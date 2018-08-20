"""Unit tests for model_activation.py."""

import unittest
import numpy
from gewittergefahr.deep_learning import model_activation

# The following constants are used to test get_hilo_activation_examples.
STORM_ACTIVATIONS = numpy.array(
    [3.0, -0.2, 0.6, -2.3, 4.3, -0.2, -1.3, -2.1, 0.0, 0.3, 1.1, -1.2, 2.5,
     -1.2, -1.5])
NUM_LOW_ACTIVATION_EXAMPLES_FEW = 4
NUM_HIGH_ACTIVATION_EXAMPLES_FEW = 3
NUM_LOW_ACTIVATION_EXAMPLES_MANY = 16
NUM_HIGH_ACTIVATION_EXAMPLES_MANY = 10

LOW_INDICES_FEW = numpy.array([3, 7, 14, 6], dtype=int)
LOW_INDICES_MANY = numpy.array(
    [3, 7, 14, 6, 11, 13, 1, 5, 8, 9, 2, 10, 12, 0, 4], dtype=int)
HIGH_INDICES_FEW = numpy.array([4, 0, 12], dtype=int)
HIGH_INDICES_MANY = numpy.array([4, 0, 12, 10, 2, 9, 8, 5, 1, 13], dtype=int)

# The following constants are used to test get_contingency_table_extremes.
STORM_TARGET_VALUES = numpy.array(
    [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0], dtype=int)
NUM_HITS_FEW = 2
NUM_MISSES_FEW = 3
NUM_FALSE_ALARMS_FEW = 4
NUM_CORRECT_NULLS_FEW = 5

NUM_HITS_MANY = 8
NUM_MISSES_MANY = 9
NUM_FALSE_ALARMS_MANY = 10
NUM_CORRECT_NULLS_MANY = 11

HIT_INDICES_FEW = numpy.array([4, 0], dtype=int)
MISS_INDICES_FEW = numpy.array([6, 11, 13], dtype=int)
FALSE_ALARM_INDICES_FEW = numpy.array([12, 8, 5, 1], dtype=int)
CORRECT_NULL_INDICES_FEW = numpy.array([3, 7, 14, 1, 5], dtype=int)

HIT_INDICES_MANY = numpy.array([4, 0, 10, 2, 9, 13, 11, 6], dtype=int)
MISS_INDICES_MANY = numpy.array([6, 11, 13, 9, 2, 10, 0, 4], dtype=int)
FALSE_ALARM_INDICES_MANY = numpy.array([12, 8, 5, 1, 14, 7, 3], dtype=int)
CORRECT_NULL_INDICES_MANY = numpy.array([3, 7, 14, 1, 5, 8, 12], dtype=int)


class ModelActivationTests(unittest.TestCase):
    """Unit tests for model_activation.py."""

    def test_get_hilo_activation_examples_few(self):
        """Ensures correct output from _get_hilo_activation_examples.

        In this case, only few examples are returned.
        """

        (these_high_indices, these_low_indices
        ) = model_activation.get_hilo_activation_examples(
            storm_activations=STORM_ACTIVATIONS,
            num_low_activation_examples=NUM_LOW_ACTIVATION_EXAMPLES_FEW,
            num_high_activation_examples=NUM_HIGH_ACTIVATION_EXAMPLES_FEW)

        self.assertTrue(numpy.array_equal(these_high_indices, HIGH_INDICES_FEW))
        self.assertTrue(numpy.array_equal(these_low_indices, LOW_INDICES_FEW))

    def test_get_hilo_activation_examples_many(self):
        """Ensures correct output from _get_hilo_activation_examples.

        In this case, many examples are returned.
        """

        (these_high_indices, these_low_indices
        ) = model_activation.get_hilo_activation_examples(
            storm_activations=STORM_ACTIVATIONS,
            num_low_activation_examples=NUM_LOW_ACTIVATION_EXAMPLES_MANY,
            num_high_activation_examples=NUM_HIGH_ACTIVATION_EXAMPLES_MANY)

        self.assertTrue(numpy.array_equal(
            these_high_indices, HIGH_INDICES_MANY))
        self.assertTrue(numpy.array_equal(these_low_indices, LOW_INDICES_MANY))

    def test_get_class_conditional_examples_few(self):
        """Ensures correct output from _get_class_conditional_examples.

        In this case, only few examples are returned.
        """

        this_ct_extreme_dict = model_activation.get_contingency_table_extremes(
            storm_activations=STORM_ACTIVATIONS,
            storm_target_values=STORM_TARGET_VALUES, num_hits=NUM_HITS_FEW,
            num_misses=NUM_MISSES_FEW, num_false_alarms=NUM_FALSE_ALARMS_FEW,
            num_correct_nulls=NUM_CORRECT_NULLS_FEW)

        self.assertTrue(numpy.array_equal(
            this_ct_extreme_dict[model_activation.HIT_INDICES_KEY],
            HIT_INDICES_FEW))
        self.assertTrue(numpy.array_equal(
            this_ct_extreme_dict[model_activation.MISS_INDICES_KEY],
            MISS_INDICES_FEW))
        self.assertTrue(numpy.array_equal(
            this_ct_extreme_dict[model_activation.FALSE_ALARM_INDICES_KEY],
            FALSE_ALARM_INDICES_FEW))
        self.assertTrue(numpy.array_equal(
            this_ct_extreme_dict[model_activation.CORRECT_NULL_INDICES_KEY],
            CORRECT_NULL_INDICES_FEW))

    def test_get_class_conditional_examples_many(self):
        """Ensures correct output from _get_class_conditional_examples.

        In this case, many examples are returned.
        """

        this_ct_extreme_dict = model_activation.get_contingency_table_extremes(
            storm_activations=STORM_ACTIVATIONS,
            storm_target_values=STORM_TARGET_VALUES, num_hits=NUM_HITS_MANY,
            num_misses=NUM_MISSES_MANY, num_false_alarms=NUM_FALSE_ALARMS_MANY,
            num_correct_nulls=NUM_CORRECT_NULLS_MANY)

        self.assertTrue(numpy.array_equal(
            this_ct_extreme_dict[model_activation.HIT_INDICES_KEY],
            HIT_INDICES_MANY))
        self.assertTrue(numpy.array_equal(
            this_ct_extreme_dict[model_activation.MISS_INDICES_KEY],
            MISS_INDICES_MANY))
        self.assertTrue(numpy.array_equal(
            this_ct_extreme_dict[model_activation.FALSE_ALARM_INDICES_KEY],
            FALSE_ALARM_INDICES_MANY))
        self.assertTrue(numpy.array_equal(
            this_ct_extreme_dict[model_activation.CORRECT_NULL_INDICES_KEY],
            CORRECT_NULL_INDICES_MANY))


if __name__ == '__main__':
    unittest.main()
