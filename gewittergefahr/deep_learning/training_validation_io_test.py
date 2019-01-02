"""Unit tests for training_validation_io.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.deep_learning import training_validation_io as trainval_io

TOLERANCE = 1e-6

# The following constants are used to test _get_batch_size_by_class.
NUM_EXAMPLES_PER_BATCH = 100
TORNADO_TARGET_NAME = 'tornado_lead-time=0000-3600sec_distance=00001-05000m'
WIND_TARGET_NAME = (
    'wind-speed_percentile=100.0_lead-time=1800-3600sec_distance=00001-05000m'
    '_cutoffs=30-50kt')

DOWNSAMPLING_DICT_TORNADO = {0: 0.8, 1: 0.2}
CLASS_TO_BATCH_SIZE_DICT_TORNADO_DS = {0: 80, 1: 20}
CLASS_TO_BATCH_SIZE_DICT_TORNADO_NO_DS = {
    0: NUM_EXAMPLES_PER_BATCH, 1: NUM_EXAMPLES_PER_BATCH
}

DOWNSAMPLING_DICT_WIND = {-2: 0.3, 0: 0.4, 1: 0.2, 2: 0.1}
CLASS_TO_BATCH_SIZE_DICT_WIND = {-2: 30, 0: 40, 1: 20, 2: 10}

# The following constants are used to test _check_stopping_criterion.
TARGET_VALUES_50ZEROS = numpy.full(50, 0, dtype=int)
TARGET_VALUES_200ZEROS = numpy.full(200, 0, dtype=int)

THESE_INDICES = numpy.linspace(0, 199, num=200, dtype=int)
THESE_INDICES = numpy.random.choice(THESE_INDICES, size=30, replace=False)
TARGET_VALUES_TORNADO = TARGET_VALUES_200ZEROS + 0
TARGET_VALUES_TORNADO[THESE_INDICES] = 1

THESE_INDICES = numpy.linspace(0, 199, num=200, dtype=int)
THESE_INDICES = numpy.random.choice(THESE_INDICES, size=120, replace=False)
TARGET_VALUES_WIND = TARGET_VALUES_200ZEROS + 0
TARGET_VALUES_WIND[THESE_INDICES[:30]] = 2
TARGET_VALUES_WIND[THESE_INDICES[30:70]] = 1
TARGET_VALUES_WIND[THESE_INDICES[70:]] = -2

# The following constants are used to test layer_ops_to_field_height_pairs.
RADAR_FIELD_NAMES = (
    [radar_utils.REFL_NAME] * 3 +
    [radar_utils.SPECTRUM_WIDTH_NAME] * 3 +
    [radar_utils.VORTICITY_NAME] * 3 +
    [radar_utils.VORTICITY_NAME] * 3
)

MIN_HEIGHTS_M_AGL = numpy.array(
    [1000] * 3 + [1000] * 3 + [2000] * 3 + [5000] * 3,
    dtype=int
)

MAX_HEIGHTS_M_AGL = numpy.array(
    [3000] * 3 + [3000] * 3 + [4000] * 3 + [8000] * 3,
    dtype=int
)

LIST_OF_LAYER_OPERATION_DICTS = [
    {
        input_examples.RADAR_FIELD_KEY: RADAR_FIELD_NAMES[k],
        input_examples.MIN_HEIGHT_KEY: MIN_HEIGHTS_M_AGL[k],
        input_examples.MAX_HEIGHT_KEY: MAX_HEIGHTS_M_AGL[k]
    } for k in range(len(RADAR_FIELD_NAMES))
]

UNIQUE_RADAR_FIELD_NAMES = [
    radar_utils.REFL_NAME, radar_utils.SPECTRUM_WIDTH_NAME,
    radar_utils.VORTICITY_NAME
]

UNIQUE_HEIGHTS_M_AGL = numpy.array(
    [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000], dtype=int)


class TrainingValidationIoTests(unittest.TestCase):
    """Each method is a unit test for training_validation_io.py."""

    def test_get_batch_size_by_class_tornado(self):
        """Ensures correct output from _get_batch_size_by_class.

        In this case, target variable = tornado and downsampling = yes.
        """

        this_dict = trainval_io._get_batch_size_by_class(
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
            target_name=TORNADO_TARGET_NAME,
            class_to_sampling_fraction_dict=DOWNSAMPLING_DICT_TORNADO)

        self.assertTrue(this_dict == CLASS_TO_BATCH_SIZE_DICT_TORNADO_DS)

    def test_get_batch_size_by_class_wind(self):
        """Ensures correct output from _get_batch_size_by_class.

        In this case, target variable = wind and downsampling = yes.
        """

        this_dict = trainval_io._get_batch_size_by_class(
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
            target_name=WIND_TARGET_NAME,
            class_to_sampling_fraction_dict=DOWNSAMPLING_DICT_WIND)

        self.assertTrue(this_dict == CLASS_TO_BATCH_SIZE_DICT_WIND)

    def test_get_batch_size_by_class_no_downsampling(self):
        """Ensures correct output from _get_batch_size_by_class.

        In this case, target variable = tornado and downsampling = no.
        """

        this_dict = trainval_io._get_batch_size_by_class(
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
            target_name=TORNADO_TARGET_NAME,
            class_to_sampling_fraction_dict=None)

        self.assertTrue(this_dict == CLASS_TO_BATCH_SIZE_DICT_TORNADO_NO_DS)

    def test_check_stopping_criterion_tor_need_examples(self):
        """Ensures correct output from _check_stopping_criterion.

        In this case, target variable = tornado and more examples are needed.
        """

        this_flag = trainval_io._check_stopping_criterion(
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
            class_to_batch_size_dict=CLASS_TO_BATCH_SIZE_DICT_TORNADO_DS,
            class_to_sampling_fraction_dict=None,
            target_values_in_memory=TARGET_VALUES_50ZEROS)

        self.assertFalse(this_flag)

    def test_check_stopping_criterion_wind_need_examples(self):
        """Ensures correct output from _check_stopping_criterion.

        In this case, target variable = wind and more examples are needed.
        """

        this_flag = trainval_io._check_stopping_criterion(
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
            class_to_batch_size_dict=CLASS_TO_BATCH_SIZE_DICT_WIND,
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
            class_to_batch_size_dict=CLASS_TO_BATCH_SIZE_DICT_TORNADO_DS,
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
            class_to_batch_size_dict=CLASS_TO_BATCH_SIZE_DICT_WIND,
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
            class_to_batch_size_dict=CLASS_TO_BATCH_SIZE_DICT_TORNADO_DS,
            class_to_sampling_fraction_dict=DOWNSAMPLING_DICT_TORNADO,
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
            class_to_batch_size_dict=CLASS_TO_BATCH_SIZE_DICT_WIND,
            class_to_sampling_fraction_dict=DOWNSAMPLING_DICT_WIND,
            target_values_in_memory=TARGET_VALUES_200ZEROS)

        self.assertFalse(this_flag)

    def test_check_stopping_criterion_tor_have_ones(self):
        """Ensures correct output from _check_stopping_criterion.

        In this case, target variable = tornado and downsampling is on.  But
        this doesn't matter, because we have enough examples from each class.
        """

        this_flag = trainval_io._check_stopping_criterion(
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
            class_to_batch_size_dict=CLASS_TO_BATCH_SIZE_DICT_TORNADO_DS,
            class_to_sampling_fraction_dict=DOWNSAMPLING_DICT_TORNADO,
            target_values_in_memory=TARGET_VALUES_TORNADO)

        self.assertTrue(this_flag)

    def test_check_stopping_criterion_wind_have_classes(self):
        """Ensures correct output from _check_stopping_criterion.

        In this case, target variable = wind and downsampling is on.  But
        this doesn't matter, because we have enough examples from each class.
        """

        this_flag = trainval_io._check_stopping_criterion(
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
            class_to_batch_size_dict=CLASS_TO_BATCH_SIZE_DICT_WIND,
            class_to_sampling_fraction_dict=DOWNSAMPLING_DICT_WIND,
            target_values_in_memory=TARGET_VALUES_WIND)

        self.assertTrue(this_flag)

    def test_get_remaining_batch_size_by_class_tor_no_data(self):
        """Ensures correct output from _get_remaining_batch_size_by_class.

        In this case, target variable is tornado and there are no data in
        memory.
        """

        this_dict = trainval_io._get_remaining_batch_size_by_class(
            class_to_batch_size_dict=CLASS_TO_BATCH_SIZE_DICT_TORNADO_DS,
            target_values_in_memory=None)

        self.assertTrue(this_dict == CLASS_TO_BATCH_SIZE_DICT_TORNADO_DS)

    def test_get_remaining_batch_size_by_class_tor_enough(self):
        """Ensures correct output from _get_remaining_batch_size_by_class.

        In this case, target variable is tornado and there are enough data in
        memory from both classes.
        """

        this_dict = trainval_io._get_remaining_batch_size_by_class(
            class_to_batch_size_dict=CLASS_TO_BATCH_SIZE_DICT_TORNADO_DS,
            target_values_in_memory=TARGET_VALUES_TORNADO)

        self.assertTrue(this_dict == {0: 0, 1: 0})

    def test_get_remaining_batch_size_by_class_wind_no_data(self):
        """Ensures correct output from _get_remaining_batch_size_by_class.

        In this case, target variable is wind and there are no data in
        memory.
        """

        this_dict = trainval_io._get_remaining_batch_size_by_class(
            class_to_batch_size_dict=CLASS_TO_BATCH_SIZE_DICT_WIND,
            target_values_in_memory=None)

        self.assertTrue(this_dict == CLASS_TO_BATCH_SIZE_DICT_WIND)

    def test_get_remaining_batch_size_by_class_wind_enough(self):
        """Ensures correct output from _get_remaining_batch_size_by_class.

        In this case, target variable is wind and there are enough data in
        memory from both classes.
        """

        this_dict = trainval_io._get_remaining_batch_size_by_class(
            class_to_batch_size_dict=CLASS_TO_BATCH_SIZE_DICT_WIND,
            target_values_in_memory=TARGET_VALUES_WIND)

        self.assertTrue(this_dict == {-2: 0, 0: 0, 1: 0, 2: 0})

    def test_layer_ops_to_field_height_pairs(self):
        """Ensures correct output from layer_ops_to_field_height_pairs."""

        these_field_names, these_heights_m_agl = (
            trainval_io.layer_ops_to_field_height_pairs(
                LIST_OF_LAYER_OPERATION_DICTS)
        )

        self.assertTrue(set(these_field_names) == set(UNIQUE_RADAR_FIELD_NAMES))
        self.assertTrue(numpy.array_equal(
            these_heights_m_agl, UNIQUE_HEIGHTS_M_AGL
        ))


if __name__ == '__main__':
    unittest.main()
