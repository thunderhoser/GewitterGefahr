"""Unit tests for training_validation_io.py."""

import unittest
import numpy
from gewittergefahr.deep_learning import training_validation_io as trainval_io

TOLERANCE = 1e-6

# The following constants are used to test _get_num_ex_per_batch_by_class.
NUM_EXAMPLES_PER_BATCH = 100
TORNADO_TARGET_NAME = 'tornado_lead-time=0000-3600sec_distance=00001-05000m'

TORNADO_CLASS_TO_FRACTION_DICT = {0: 0.8, 1: 0.2}
TORNADO_CLASS_TO_SMALL_NUM_EX_DICT = {0: 80, 1: 20}
TORNADO_CLASS_TO_LARGE_NUM_EX_DICT = {
    0: NUM_EXAMPLES_PER_BATCH, 1: NUM_EXAMPLES_PER_BATCH
}

WIND_TARGET_NAME = (
    'wind-speed_percentile=100.0_lead-time=1800-3600sec_distance=00001-05000m'
    '_cutoffs=30-50kt')
WIND_CLASS_TO_FRACTION_DICT = {-2: 0.3, 0: 0.4, 1: 0.2, 2: 0.1}
WIND_CLASS_TO_NUM_EXAMPLES_DICT = {-2: 30, 0: 40, 1: 20, 2: 10}

# The following constants are used to test _check_stopping_criterion.
TARGET_VALUES_50ZEROS = numpy.full(50, 0, dtype=int)
TARGET_VALUES_200ZEROS = numpy.full(200, 0, dtype=int)

THESE_INDICES = numpy.linspace(0, 199, num=200, dtype=int)
THESE_INDICES = numpy.random.choice(THESE_INDICES, size=30, replace=False)
TORNADO_TARGET_VALUES_ENOUGH_ONES = TARGET_VALUES_200ZEROS + 0
TORNADO_TARGET_VALUES_ENOUGH_ONES[THESE_INDICES] = 1

THESE_INDICES = numpy.linspace(0, 199, num=200, dtype=int)
THESE_INDICES = numpy.random.choice(THESE_INDICES, size=120, replace=False)
WIND_TARGET_VALUES_ENOUGH = TARGET_VALUES_200ZEROS + 0
WIND_TARGET_VALUES_ENOUGH[THESE_INDICES[:30]] = 2
WIND_TARGET_VALUES_ENOUGH[THESE_INDICES[30:70]] = 1
WIND_TARGET_VALUES_ENOUGH[THESE_INDICES[70:]] = -2


class TrainingValidationIoTests(unittest.TestCase):
    """Each method is a unit test for training_validation_io.py."""

    def test_get_num_ex_per_batch_by_class_tornado(self):
        """Ensures correct output from _get_num_ex_per_batch_by_class.

        In this case, target variable = tornado and downsampling = yes.
        """

        this_dict = trainval_io._get_num_ex_per_batch_by_class(
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
            target_name=TORNADO_TARGET_NAME,
            class_to_sampling_fraction_dict=TORNADO_CLASS_TO_FRACTION_DICT)

        self.assertTrue(this_dict == TORNADO_CLASS_TO_SMALL_NUM_EX_DICT)

    def test_get_num_ex_per_batch_by_class_wind(self):
        """Ensures correct output from _get_num_ex_per_batch_by_class.

        In this case, target variable = wind and downsampling = yes.
        """

        this_dict = trainval_io._get_num_ex_per_batch_by_class(
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
            target_name=WIND_TARGET_NAME,
            class_to_sampling_fraction_dict=WIND_CLASS_TO_FRACTION_DICT)

        self.assertTrue(this_dict == WIND_CLASS_TO_NUM_EXAMPLES_DICT)

    def test_get_num_ex_per_batch_by_class_no_downsampling(self):
        """Ensures correct output from _get_num_ex_per_batch_by_class.

        In this case, target variable = tornado and downsampling = no.
        """

        this_dict = trainval_io._get_num_ex_per_batch_by_class(
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
            target_name=TORNADO_TARGET_NAME,
            class_to_sampling_fraction_dict=None)

        self.assertTrue(this_dict == TORNADO_CLASS_TO_LARGE_NUM_EX_DICT)

    def test_check_stopping_criterion_tor_need_examples(self):
        """Ensures correct output from _check_stopping_criterion.

        In this case, target variable = tornado and more examples are needed.
        """

        this_flag = trainval_io._check_stopping_criterion(
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
            class_to_num_ex_per_batch_dict=TORNADO_CLASS_TO_SMALL_NUM_EX_DICT,
            class_to_sampling_fraction_dict=None,
            target_values_in_memory=TARGET_VALUES_50ZEROS)

        self.assertFalse(this_flag)

    def test_check_stopping_criterion_wind_need_examples(self):
        """Ensures correct output from _check_stopping_criterion.

        In this case, target variable = wind and more examples are needed.
        """

        this_flag = trainval_io._check_stopping_criterion(
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
            class_to_num_ex_per_batch_dict=WIND_CLASS_TO_NUM_EXAMPLES_DICT,
            class_to_sampling_fraction_dict=None,
            target_values_in_memory=TARGET_VALUES_50ZEROS)

        self.assertFalse(this_flag)

    def test_check_stopping_criterion_tor_no_downsampling(self):
        """Ensures correct output from _check_stopping_criterion.

        In this case, target variable = tornado and all examples have
        target = 0.  However, this doesn't matter, because downsampling = no.
        """

        this_flag = trainval_io._check_stopping_criterion(
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
            class_to_num_ex_per_batch_dict=TORNADO_CLASS_TO_SMALL_NUM_EX_DICT,
            class_to_sampling_fraction_dict=None,
            target_values_in_memory=TARGET_VALUES_200ZEROS)

        self.assertTrue(this_flag)

    def test_check_stopping_criterion_wind_no_downsampling(self):
        """Ensures correct output from _check_stopping_criterion.

        In this case, target variable = wind and all examples have
        target = 0.  However, this doesn't matter, because downsampling = no.
        """

        this_flag = trainval_io._check_stopping_criterion(
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
            class_to_num_ex_per_batch_dict=WIND_CLASS_TO_NUM_EXAMPLES_DICT,
            class_to_sampling_fraction_dict=None,
            target_values_in_memory=TARGET_VALUES_200ZEROS)

        self.assertTrue(this_flag)

    def test_check_stopping_criterion_tor_need_ones(self):
        """Ensures correct output from _check_stopping_criterion.

        In this case, target variable = tornado and all examples have
        target = 0.  This will make stopping criterion = False, because
        downsampling is on.
        """

        this_flag = trainval_io._check_stopping_criterion(
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
            class_to_num_ex_per_batch_dict=TORNADO_CLASS_TO_SMALL_NUM_EX_DICT,
            class_to_sampling_fraction_dict=TORNADO_CLASS_TO_FRACTION_DICT,
            target_values_in_memory=TARGET_VALUES_200ZEROS)

        self.assertFalse(this_flag)

    def test_check_stopping_criterion_wind_need_classes(self):
        """Ensures correct output from _check_stopping_criterion.

        In this case, target variable = wind and all examples have
        target = 0.  This will make stopping criterion = False, because
        downsampling is on.
        """

        this_flag = trainval_io._check_stopping_criterion(
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
            class_to_num_ex_per_batch_dict=WIND_CLASS_TO_NUM_EXAMPLES_DICT,
            class_to_sampling_fraction_dict=WIND_CLASS_TO_FRACTION_DICT,
            target_values_in_memory=TARGET_VALUES_200ZEROS)

        self.assertFalse(this_flag)

    def test_check_stopping_criterion_tor_have_ones(self):
        """Ensures correct output from _check_stopping_criterion.

        In this case, target variable = tornado and downsampling is on.  But
        this doesn't matter, because we have enough examples from each class.
        """

        this_flag = trainval_io._check_stopping_criterion(
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
            class_to_num_ex_per_batch_dict=TORNADO_CLASS_TO_SMALL_NUM_EX_DICT,
            class_to_sampling_fraction_dict=TORNADO_CLASS_TO_FRACTION_DICT,
            target_values_in_memory=TORNADO_TARGET_VALUES_ENOUGH_ONES)

        self.assertTrue(this_flag)

    def test_check_stopping_criterion_wind_have_classes(self):
        """Ensures correct output from _check_stopping_criterion.

        In this case, target variable = wind and downsampling is on.  But
        this doesn't matter, because we have enough examples from each class.
        """

        this_flag = trainval_io._check_stopping_criterion(
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
            class_to_num_ex_per_batch_dict=WIND_CLASS_TO_NUM_EXAMPLES_DICT,
            class_to_sampling_fraction_dict=WIND_CLASS_TO_FRACTION_DICT,
            target_values_in_memory=WIND_TARGET_VALUES_ENOUGH)

        self.assertTrue(this_flag)


if __name__ == '__main__':
    unittest.main()
