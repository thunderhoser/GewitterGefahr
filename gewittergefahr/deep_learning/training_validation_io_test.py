"""Unit tests for training_validation_io.py."""

import copy
import unittest
import numpy
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.deep_learning import storm_images
from gewittergefahr.deep_learning import training_validation_io as trainval_io

TOLERANCE = 1e-6

# The following constants are used to test _remove_storms_with_undef_target.
THESE_STORM_IDS = ['A', 'B', 'C', 'D']
THIS_IMAGE_MATRIX = numpy.reshape(numpy.linspace(1., 24., num=24), (4, 3, 2))
THESE_TARGET_VALUES = numpy.array([-1, 0, -3, 1], dtype=int)

STORM_IMAGE_DICT_WITH_UNDEF_TARGETS = {
    storm_images.STORM_IDS_KEY: THESE_STORM_IDS,
    storm_images.STORM_IMAGE_MATRIX_KEY: THIS_IMAGE_MATRIX,
    storm_images.LABEL_VALUES_KEY: THESE_TARGET_VALUES,
}

THESE_VALID_INDICES = numpy.array([1, 3], dtype=int)
STORM_IMAGE_DICT_NO_UNDEF_TARGETS = {
    storm_images.STORM_IDS_KEY:
        [THESE_STORM_IDS[i] for i in THESE_VALID_INDICES],
    storm_images.STORM_IMAGE_MATRIX_KEY:
        THIS_IMAGE_MATRIX[THESE_VALID_INDICES, :],
    storm_images.LABEL_VALUES_KEY: THESE_TARGET_VALUES[THESE_VALID_INDICES],
}

# The following constants are used to test _get_num_examples_per_batch_by_class.
NUM_EXAMPLES_PER_BATCH = 100
TORNADO_TARGET_NAME = 'tornado_lead-time=0900-3600sec_distance=00001-05000m'
TORNADO_CLASS_FRACTIONS_TO_SAMPLE = numpy.array([0.9, 0.1])
NUM_EXAMPLES_PER_BATCH_BY_TOR_CLASS = numpy.array([90, 10], dtype=int)

WIND_TARGET_NAME = (
    'wind-speed_percentile=097.5_lead-time=0900-3600sec_distance=00001-05000m'
    '_cutoffs=10-20-30-40-50kt')
WIND_CLASS_FRACTIONS_TO_SAMPLE = numpy.array([0.4, 0.25, 0.1, 0.1, 0.1, 0.05])
NUM_EXAMPLES_PER_BATCH_BY_WIND_CLASS = numpy.array(
    [40, 25, 10, 10, 10, 5], dtype=int)

# The following constants are used to test _get_num_examples_remaining_by_class.
NUM_INIT_TIMES_PER_BATCH = 20
CLASS_FRACTIONS_TO_SAMPLE = numpy.array([0.8, 0.2])
NUM_EXAMPLES_PER_BATCH_BY_CLASS = numpy.array([80, 20], dtype=int)
NUM_EXAMPLES_IN_MEMORY_BY_CLASS = numpy.array([1000, 10], dtype=int)
NUM_EXAMPLES_REMAINING_BY_CLASS = numpy.array([0, 10], dtype=int)

# The following constants are used to test _determine_stopping_criterion.
TARGET_VALUES_50ZEROS = numpy.full(50, 0, dtype=int)
TARGET_VALUES_200ZEROS = numpy.full(200, 0, dtype=int)

THESE_INDICES = numpy.linspace(0, 199, num=200, dtype=int)
THESE_INDICES = numpy.random.choice(THESE_INDICES, size=30, replace=False)
TARGET_VALUES_ENOUGH_ONES = copy.deepcopy(TARGET_VALUES_200ZEROS)
TARGET_VALUES_ENOUGH_ONES[THESE_INDICES] = 1

NUM_EXAMPLES_IN_MEMORY_BY_CLASS_50ZEROS = numpy.array([50, 0], dtype=int)
NUM_EXAMPLES_IN_MEMORY_BY_CLASS_200ZEROS = numpy.array([200, 0], dtype=int)
NUM_EXAMPLES_IN_MEMORY_BY_CLASS_ENOUGH_ONES = numpy.array([170, 30], dtype=int)

# The following constants are used to test
# _separate_input_files_for_2d3d_myrorss.
IMAGE_FILE_NAME_MATRIX = numpy.array([['A', 'B', 'C', 'D', 'E'],
                                      ['F', 'G', 'H', 'I', 'J']], dtype=object)
FIELD_NAME_BY_PAIR = [
    radar_utils.MESH_NAME, radar_utils.REFL_NAME, radar_utils.REFL_NAME,
    radar_utils.MID_LEVEL_SHEAR_NAME, radar_utils.LOW_LEVEL_SHEAR_NAME]
HEIGHT_BY_PAIR_M_ASL = numpy.array([250, 5000, 1000, 250, 250], dtype=int)

REFLECTIVITY_INDICES = numpy.array([1, 2], dtype=int)
AZIMUTHAL_SHEAR_INDICES = numpy.array([3, 4], dtype=int)
MESH_INDICES = numpy.array([0], dtype=int)
NON_REFLECTIVITY_INDICES = numpy.array([0, 3, 4], dtype=int)
NON_AZIMUTHAL_SHEAR_INDICES = numpy.array([0, 1, 2], dtype=int)
NON_MESH_INDICES = numpy.array([1, 2, 3, 4], dtype=int)

REFLECTIVITY_FILE_NAME_MATRIX = numpy.array([['C', 'B'],
                                             ['H', 'G']], dtype=object)
AZ_SHEAR_FILE_NAME_MATRIX = numpy.array([['E', 'D'],
                                         ['J', 'I']], dtype=object)


class TrainingValidationIoTests(unittest.TestCase):
    """Each method is a unit test for training_validation_io.py."""

    def test_remove_storms_with_undef_target(self):
        """Ensures correct output from _remove_storms_with_undef_target."""

        this_input_dict = copy.deepcopy(STORM_IMAGE_DICT_WITH_UNDEF_TARGETS)
        this_storm_image_dict, _ = trainval_io._remove_storms_with_undef_target(
            this_input_dict)

        actual_keys = this_storm_image_dict.keys()
        expected_keys = STORM_IMAGE_DICT_NO_UNDEF_TARGETS.keys()
        self.assertTrue(set(actual_keys) == set(expected_keys))

        self.assertTrue(
            this_storm_image_dict[storm_images.STORM_IDS_KEY] ==
            STORM_IMAGE_DICT_NO_UNDEF_TARGETS[storm_images.STORM_IDS_KEY])
        self.assertTrue(numpy.allclose(
            this_storm_image_dict[storm_images.STORM_IMAGE_MATRIX_KEY],
            STORM_IMAGE_DICT_NO_UNDEF_TARGETS[
                storm_images.STORM_IMAGE_MATRIX_KEY],
            atol=TOLERANCE))
        self.assertTrue(numpy.array_equal(
            this_storm_image_dict[storm_images.LABEL_VALUES_KEY],
            STORM_IMAGE_DICT_NO_UNDEF_TARGETS[storm_images.LABEL_VALUES_KEY]))

    def test_get_num_examples_per_batch_by_class_tornado(self):
        """Ensures correct output from _get_num_examples_per_batch_by_class.

        In this case, the target variable is tornado occurrence.
        """

        this_num_examples_per_batch_by_class = (
            trainval_io._get_num_examples_per_batch_by_class(
                num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
                target_name=TORNADO_TARGET_NAME,
                class_fractions_to_sample=TORNADO_CLASS_FRACTIONS_TO_SAMPLE))

        self.assertTrue(numpy.array_equal(
            this_num_examples_per_batch_by_class,
            NUM_EXAMPLES_PER_BATCH_BY_TOR_CLASS))

    def test_get_num_examples_per_batch_by_class_wind(self):
        """Ensures correct output from _get_num_examples_per_batch_by_class.

        In this case, the target variable is wind-speed category.
        """

        this_num_examples_per_batch_by_class = (
            trainval_io._get_num_examples_per_batch_by_class(
                num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
                target_name=WIND_TARGET_NAME,
                class_fractions_to_sample=WIND_CLASS_FRACTIONS_TO_SAMPLE))

        self.assertTrue(numpy.array_equal(
            this_num_examples_per_batch_by_class,
            NUM_EXAMPLES_PER_BATCH_BY_WIND_CLASS))

    def test_get_num_examples_per_batch_by_class_mismatch(self):
        """Ensures correct output from _get_num_examples_per_batch_by_class.

        In this case, the input arguments `target_name` and
        `class_fractions_to_sample` are mismatched.
        """

        with self.assertRaises(TypeError):
            trainval_io._get_num_examples_per_batch_by_class(
                num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
                target_name=TORNADO_TARGET_NAME,
                class_fractions_to_sample=WIND_CLASS_FRACTIONS_TO_SAMPLE)

    def test_get_num_examples_per_batch_by_class_no_fractions(self):
        """Ensures correct output from _get_num_examples_per_batch_by_class.

        In this case, the input argument `class_fractions_to_sample` is empty,
        which means that there will be no downsampling.
        """

        this_num_examples_per_batch_by_class = (
            trainval_io._get_num_examples_per_batch_by_class(
                num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
                target_name=WIND_TARGET_NAME, class_fractions_to_sample=None))

        self.assertTrue(numpy.all(
            this_num_examples_per_batch_by_class == trainval_io.LARGE_INTEGER))

    def test_get_num_examples_remaining_by_class_need_times_and_examples(self):
        """Ensures correct output from _get_num_examples_remaining_by_class.

        In this case there will be no downsampling, because there are not yet
        enough initial times or examples in memory.
        """

        this_num_examples_remaining_by_class = (
            trainval_io._get_num_examples_remaining_by_class(
                num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
                num_init_times_per_batch=NUM_INIT_TIMES_PER_BATCH,
                num_examples_per_batch_by_class=NUM_EXAMPLES_PER_BATCH_BY_CLASS,
                num_examples_in_memory=NUM_EXAMPLES_PER_BATCH - 1,
                num_init_times_in_memory=NUM_INIT_TIMES_PER_BATCH - 1,
                num_examples_in_memory_by_class=NUM_EXAMPLES_IN_MEMORY_BY_CLASS
            ))
        self.assertTrue(this_num_examples_remaining_by_class is None)

    def test_get_num_examples_remaining_by_class_need_times(self):
        """Ensures correct output from _get_num_examples_remaining_by_class.

        In this case there will be no downsampling, because there are not yet
        enough initial times in memory.
        """

        this_num_examples_remaining_by_class = (
            trainval_io._get_num_examples_remaining_by_class(
                num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
                num_init_times_per_batch=NUM_INIT_TIMES_PER_BATCH,
                num_examples_per_batch_by_class=NUM_EXAMPLES_PER_BATCH_BY_CLASS,
                num_examples_in_memory=NUM_EXAMPLES_PER_BATCH + 1,
                num_init_times_in_memory=NUM_INIT_TIMES_PER_BATCH - 1,
                num_examples_in_memory_by_class=NUM_EXAMPLES_IN_MEMORY_BY_CLASS
            ))
        self.assertTrue(this_num_examples_remaining_by_class is None)

    def test_get_num_examples_remaining_by_class_need_examples(self):
        """Ensures correct output from _get_num_examples_remaining_by_class.

        In this case there will be no downsampling, because there are not yet
        enough examples in memory.
        """

        this_num_examples_remaining_by_class = (
            trainval_io._get_num_examples_remaining_by_class(
                num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
                num_init_times_per_batch=NUM_INIT_TIMES_PER_BATCH,
                num_examples_per_batch_by_class=NUM_EXAMPLES_PER_BATCH_BY_CLASS,
                num_examples_in_memory=NUM_EXAMPLES_PER_BATCH - 1,
                num_init_times_in_memory=NUM_INIT_TIMES_PER_BATCH + 1,
                num_examples_in_memory_by_class=NUM_EXAMPLES_IN_MEMORY_BY_CLASS
            ))
        self.assertTrue(this_num_examples_remaining_by_class is None)

    def test_get_num_examples_remaining_by_class_downsampling(self):
        """Ensures correct output from _get_num_examples_remaining_by_class.

        In this case there will be downsampling, because there are already
        enough initial times and examples in memory.
        """

        this_num_examples_remaining_by_class = (
            trainval_io._get_num_examples_remaining_by_class(
                num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
                num_init_times_per_batch=NUM_INIT_TIMES_PER_BATCH,
                num_examples_per_batch_by_class=NUM_EXAMPLES_PER_BATCH_BY_CLASS,
                num_examples_in_memory=NUM_EXAMPLES_PER_BATCH + 1,
                num_init_times_in_memory=NUM_INIT_TIMES_PER_BATCH + 1,
                num_examples_in_memory_by_class=NUM_EXAMPLES_IN_MEMORY_BY_CLASS
            ))
        self.assertTrue(numpy.array_equal(
            this_num_examples_remaining_by_class,
            NUM_EXAMPLES_REMAINING_BY_CLASS))

    def test_determine_stopping_criterion_need_times_and_examples(self):
        """Ensures correct output from _determine_stopping_criterion.

        In this case, there are not yet enough initial times or examples in
        memory.
        """

        this_num_examples_in_memory_by_class, this_stopping_criterion = (
            trainval_io._determine_stopping_criterion(
                num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
                num_init_times_per_batch=NUM_INIT_TIMES_PER_BATCH,
                num_examples_per_batch_by_class=NUM_EXAMPLES_PER_BATCH_BY_CLASS,
                num_init_times_in_memory=NUM_INIT_TIMES_PER_BATCH - 1,
                class_fractions_to_sample=CLASS_FRACTIONS_TO_SAMPLE,
                target_values_in_memory=TARGET_VALUES_50ZEROS))

        self.assertTrue(numpy.array_equal(
            this_num_examples_in_memory_by_class,
            NUM_EXAMPLES_IN_MEMORY_BY_CLASS_50ZEROS))
        self.assertFalse(this_stopping_criterion)

    def test_determine_stopping_criterion_need_times(self):
        """Ensures correct output from _determine_stopping_criterion.

        In this case, there are not yet enough initial times in memory.
        """

        this_num_examples_in_memory_by_class, this_stopping_criterion = (
            trainval_io._determine_stopping_criterion(
                num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
                num_init_times_per_batch=NUM_INIT_TIMES_PER_BATCH,
                num_examples_per_batch_by_class=NUM_EXAMPLES_PER_BATCH_BY_CLASS,
                num_init_times_in_memory=NUM_INIT_TIMES_PER_BATCH - 1,
                class_fractions_to_sample=CLASS_FRACTIONS_TO_SAMPLE,
                target_values_in_memory=TARGET_VALUES_ENOUGH_ONES))

        self.assertTrue(numpy.array_equal(
            this_num_examples_in_memory_by_class,
            NUM_EXAMPLES_IN_MEMORY_BY_CLASS_ENOUGH_ONES))
        self.assertFalse(this_stopping_criterion)

    def test_determine_stopping_criterion_need_examples(self):
        """Ensures correct output from _determine_stopping_criterion.

        In this case, there are not yet enough examples in memory.
        """

        this_num_examples_in_memory_by_class, this_stopping_criterion = (
            trainval_io._determine_stopping_criterion(
                num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
                num_init_times_per_batch=NUM_INIT_TIMES_PER_BATCH,
                num_examples_per_batch_by_class=NUM_EXAMPLES_PER_BATCH_BY_CLASS,
                num_init_times_in_memory=NUM_INIT_TIMES_PER_BATCH + 1,
                class_fractions_to_sample=CLASS_FRACTIONS_TO_SAMPLE,
                target_values_in_memory=TARGET_VALUES_50ZEROS))

        self.assertTrue(numpy.array_equal(
            this_num_examples_in_memory_by_class,
            NUM_EXAMPLES_IN_MEMORY_BY_CLASS_50ZEROS))
        self.assertFalse(this_stopping_criterion)

    def test_determine_stopping_criterion_no_downsampling(self):
        """Ensures correct output from _determine_stopping_criterion.

        In this case, there are enough initial times and examples in memory.
        All examples have target = 0, but this doesn't matter, because we are
        not oversampling.
        """

        this_num_examples_in_memory_by_class, this_stopping_criterion = (
            trainval_io._determine_stopping_criterion(
                num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
                num_init_times_per_batch=NUM_INIT_TIMES_PER_BATCH,
                num_examples_per_batch_by_class=NUM_EXAMPLES_PER_BATCH_BY_CLASS,
                num_init_times_in_memory=NUM_INIT_TIMES_PER_BATCH + 1,
                class_fractions_to_sample=None,
                target_values_in_memory=TARGET_VALUES_200ZEROS))

        self.assertTrue(numpy.array_equal(
            this_num_examples_in_memory_by_class,
            NUM_EXAMPLES_IN_MEMORY_BY_CLASS_200ZEROS))
        self.assertTrue(this_stopping_criterion)

    def test_determine_stopping_criterion_need_ones(self):
        """Ensures correct output from _determine_stopping_criterion.

        In this case, there are enough initial times and examples in memory.
        However, all examples have target = 0.
        """

        this_num_examples_in_memory_by_class, this_stopping_criterion = (
            trainval_io._determine_stopping_criterion(
                num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
                num_init_times_per_batch=NUM_INIT_TIMES_PER_BATCH,
                num_examples_per_batch_by_class=NUM_EXAMPLES_PER_BATCH_BY_CLASS,
                num_init_times_in_memory=NUM_INIT_TIMES_PER_BATCH + 1,
                class_fractions_to_sample=CLASS_FRACTIONS_TO_SAMPLE,
                target_values_in_memory=TARGET_VALUES_200ZEROS))

        self.assertTrue(numpy.array_equal(
            this_num_examples_in_memory_by_class,
            NUM_EXAMPLES_IN_MEMORY_BY_CLASS_200ZEROS))
        self.assertFalse(this_stopping_criterion)

    def test_determine_stopping_criterion_enough_ones(self):
        """Ensures correct output from _determine_stopping_criterion.

        In this case, there are enough initial times and examples in memory.
        Also, there are enough examples with target = 1.
        """

        this_num_examples_in_memory_by_class, this_stopping_criterion = (
            trainval_io._determine_stopping_criterion(
                num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
                num_init_times_per_batch=NUM_INIT_TIMES_PER_BATCH,
                num_examples_per_batch_by_class=NUM_EXAMPLES_PER_BATCH_BY_CLASS,
                num_init_times_in_memory=NUM_INIT_TIMES_PER_BATCH + 1,
                class_fractions_to_sample=CLASS_FRACTIONS_TO_SAMPLE,
                target_values_in_memory=TARGET_VALUES_ENOUGH_ONES))

        self.assertTrue(numpy.array_equal(
            this_num_examples_in_memory_by_class,
            NUM_EXAMPLES_IN_MEMORY_BY_CLASS_ENOUGH_ONES))
        self.assertTrue(this_stopping_criterion)

    def test_separate_input_files_no_refl(self):
        """Ensures correctness of _separate_input_files_for_2d3d_myrorss.

        In this case, there are no reflectivity files.
        """

        with self.assertRaises(ValueError):
            trainval_io._separate_input_files_for_2d3d_myrorss(
                image_file_name_matrix=IMAGE_FILE_NAME_MATRIX[
                    ..., NON_REFLECTIVITY_INDICES],
                test_mode=True,
                field_name_by_pair=
                [FIELD_NAME_BY_PAIR[k] for k in NON_REFLECTIVITY_INDICES],
                height_by_pair_m_asl=HEIGHT_BY_PAIR_M_ASL[
                    NON_REFLECTIVITY_INDICES])

    def test_separate_input_files_no_az_shear(self):
        """Ensures correctness of _separate_input_files_for_2d3d_myrorss.

        In this case, there are no azimuthal-shear files.
        """

        with self.assertRaises(ValueError):
            trainval_io._separate_input_files_for_2d3d_myrorss(
                image_file_name_matrix=IMAGE_FILE_NAME_MATRIX[
                    ..., NON_AZIMUTHAL_SHEAR_INDICES],
                test_mode=True,
                field_name_by_pair=
                [FIELD_NAME_BY_PAIR[k] for k in NON_AZIMUTHAL_SHEAR_INDICES],
                height_by_pair_m_asl=HEIGHT_BY_PAIR_M_ASL[
                    NON_AZIMUTHAL_SHEAR_INDICES])

    def test_separate_input_files_bad_fields(self):
        """Ensures correctness of _separate_input_files_for_2d3d_myrorss.

        In this case, one of the radar fields is neither reflectivity nor
        azimuthal shear.
        """

        with self.assertRaises(ValueError):
            trainval_io._separate_input_files_for_2d3d_myrorss(
                image_file_name_matrix=IMAGE_FILE_NAME_MATRIX[
                    ..., MESH_INDICES],
                test_mode=True,
                field_name_by_pair=
                [FIELD_NAME_BY_PAIR[k] for k in MESH_INDICES],
                height_by_pair_m_asl=HEIGHT_BY_PAIR_M_ASL[MESH_INDICES])

    def test_separate_input_files_all_good(self):
        """Ensures correctness of _separate_input_files_for_2d3d_myrorss.

        In this case, all input files are valid.
        """

        this_refl_file_name_matrix, this_az_shear_file_name_matrix = (
            trainval_io._separate_input_files_for_2d3d_myrorss(
                image_file_name_matrix=IMAGE_FILE_NAME_MATRIX[
                    ..., NON_MESH_INDICES],
                test_mode=True,
                field_name_by_pair=
                [FIELD_NAME_BY_PAIR[k] for k in NON_MESH_INDICES],
                height_by_pair_m_asl=HEIGHT_BY_PAIR_M_ASL[NON_MESH_INDICES]))

        self.assertTrue(numpy.array_equal(
            this_refl_file_name_matrix, REFLECTIVITY_FILE_NAME_MATRIX))
        self.assertTrue(numpy.array_equal(
            this_az_shear_file_name_matrix, AZ_SHEAR_FILE_NAME_MATRIX))


if __name__ == '__main__':
    unittest.main()
