"""Unit tests for input_examples.py."""

import copy
import unittest
import numpy
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.deep_learning import storm_images
from gewittergefahr.deep_learning import input_examples

TOLERANCE = 1e-6

# The following constants are used to test _filter_examples_by_class.
TARGET_VALUES_TORNADO = numpy.array(
    [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0], dtype=int
)

FIRST_TORNADO_CLASS_TO_NUM_EX_DICT = {0: 2, 1: 50}
SECOND_TORNADO_CLASS_TO_NUM_EX_DICT = {0: 0, 1: 50}
FIRST_TORNADO_INDICES_TO_KEEP = numpy.array([0, 1, 2, 4, 6], dtype=int)
SECOND_TORNADO_INDICES_TO_KEEP = numpy.array([2, 4, 6], dtype=int)

TARGET_VALUES_WIND = numpy.array([
    0, -2, 0, 5, 2, 1, 3, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 2, -2, 0, 1,
    3, 2, 4, 1, 0, 0, 1, 0, 0, -2, 4, -2, 0, 0, -2, 0
], dtype=int)

FIRST_WIND_CLASS_TO_NUM_EX_DICT = {-2: 5, 0: 1, 1: 2, 2: 3, 3: 4, 4: 25, 5: 100}
SECOND_WIND_CLASS_TO_NUM_EX_DICT = {-2: 0, 0: 1, 1: 0, 2: 3, 3: 0, 4: 25, 5: 0}

FIRST_WIND_INDICES_TO_KEEP = numpy.array(
    [0, 5, 7, 4, 13, 20, 6, 18, 24, 26, 34, 3, 1, 21, 33, 35, 38], dtype=int
)
SECOND_WIND_INDICES_TO_KEEP = numpy.array([0, 4, 13, 20, 26, 34], dtype=int)

# The following constants are used to test remove_storms_with_undefined_target.
FULL_ID_STRINGS = ['Matthews', 'Tavares', 'Marner', 'Nylander']
STORM_TIMES_UNIX_SEC = numpy.array([1, 2, 3, 4], dtype=int)
TARGET_VALUES = numpy.array([-1, 0, -1, -2], dtype=int)
THIS_RADAR_IMAGE_MATRIX = numpy.random.uniform(
    low=0., high=1., size=(4, 32, 32, 3)
)

RADAR_IMAGE_DICT_UNFILTERED = {
    storm_images.FULL_IDS_KEY: FULL_ID_STRINGS,
    storm_images.VALID_TIMES_KEY: STORM_TIMES_UNIX_SEC,
    storm_images.LABEL_VALUES_KEY: TARGET_VALUES,
    storm_images.STORM_IMAGE_MATRIX_KEY: THIS_RADAR_IMAGE_MATRIX
}

INDICES_TO_KEEP = numpy.array([1, 3], dtype=int)

RADAR_IMAGE_DICT_NO_UNDEF_TARGETS = {
    storm_images.FULL_IDS_KEY: [FULL_ID_STRINGS[i] for i in INDICES_TO_KEEP],
    storm_images.VALID_TIMES_KEY: STORM_TIMES_UNIX_SEC[INDICES_TO_KEEP],
    storm_images.LABEL_VALUES_KEY: TARGET_VALUES[INDICES_TO_KEEP],
    storm_images.STORM_IMAGE_MATRIX_KEY:
        THIS_RADAR_IMAGE_MATRIX[INDICES_TO_KEEP, ...]
}

# The following constants are used to test subset_examples.
EQUALS_SIGN_KEYS = [
    input_examples.ROTATED_GRIDS_KEY, input_examples.TARGET_NAME_KEY,
    input_examples.RADAR_FIELDS_KEY, input_examples.FULL_IDS_KEY
]
ARRAY_EQUAL_KEYS = [
    input_examples.STORM_TIMES_KEY, input_examples.TARGET_VALUES_KEY
]

THESE_FIELD_NAMES = [
    radar_utils.ECHO_TOP_40DBZ_NAME, radar_utils.VIL_NAME,
    radar_utils.REFL_NAME, radar_utils.REFL_NAME
]
THESE_HEIGHTS_M_AGL = numpy.array([
    radar_utils.DEFAULT_HEIGHT_MYRORSS_M_ASL,
    radar_utils.DEFAULT_HEIGHT_MYRORSS_M_ASL, 3000, 5000
])

THIS_RADAR_IMAGE_MATRIX = numpy.random.uniform(
    low=0., high=1., size=(4, 32, 32, 4)
)

EXAMPLE_DICT_2D_ORIG = {
    input_examples.ROTATED_GRIDS_KEY: True,
    input_examples.ROTATED_GRID_SPACING_KEY: 1500.,
    input_examples.TARGET_NAME_KEY: 'foo',
    input_examples.RADAR_FIELDS_KEY: THESE_FIELD_NAMES,
    input_examples.RADAR_HEIGHTS_KEY: THESE_HEIGHTS_M_AGL,
    input_examples.FULL_IDS_KEY: FULL_ID_STRINGS,
    input_examples.STORM_TIMES_KEY: STORM_TIMES_UNIX_SEC,
    input_examples.TARGET_VALUES_KEY: TARGET_VALUES,
    input_examples.RADAR_IMAGE_MATRIX_KEY: THIS_RADAR_IMAGE_MATRIX
}

EXAMPLE_DICT_2D_SUBSET = copy.deepcopy(EXAMPLE_DICT_2D_ORIG)
EXAMPLE_DICT_2D_SUBSET[input_examples.FULL_IDS_KEY] = [
    FULL_ID_STRINGS[k] for k in INDICES_TO_KEEP
]
EXAMPLE_DICT_2D_SUBSET[input_examples.STORM_TIMES_KEY] = (
    STORM_TIMES_UNIX_SEC[INDICES_TO_KEEP]
)
EXAMPLE_DICT_2D_SUBSET[input_examples.TARGET_VALUES_KEY] = (
    TARGET_VALUES[INDICES_TO_KEEP]
)
EXAMPLE_DICT_2D_SUBSET[input_examples.RADAR_IMAGE_MATRIX_KEY] = (
    THIS_RADAR_IMAGE_MATRIX[INDICES_TO_KEEP, ...]
)

THESE_FIELD_NAMES = [
    radar_utils.REFL_NAME, radar_utils.DIFFERENTIAL_REFL_NAME,
    radar_utils.VORTICITY_NAME, radar_utils.DIVERGENCE_NAME
]
THESE_HEIGHTS_M_AGL = numpy.array([1000, 2000, 3000, 4000, 5000, 6000])

THIS_RADAR_IMAGE_MATRIX = numpy.random.uniform(
    low=0., high=1., size=(4, 32, 32, 6, 4)
)

EXAMPLE_DICT_3D_ORIG = {
    input_examples.ROTATED_GRIDS_KEY: True,
    input_examples.ROTATED_GRID_SPACING_KEY: 1500.,
    input_examples.TARGET_NAME_KEY: 'foo',
    input_examples.RADAR_FIELDS_KEY: THESE_FIELD_NAMES,
    input_examples.RADAR_HEIGHTS_KEY: THESE_HEIGHTS_M_AGL,
    input_examples.FULL_IDS_KEY: FULL_ID_STRINGS,
    input_examples.STORM_TIMES_KEY: STORM_TIMES_UNIX_SEC,
    input_examples.TARGET_VALUES_KEY: TARGET_VALUES,
    input_examples.RADAR_IMAGE_MATRIX_KEY: THIS_RADAR_IMAGE_MATRIX
}

EXAMPLE_DICT_3D_SUBSET = copy.deepcopy(EXAMPLE_DICT_3D_ORIG)
EXAMPLE_DICT_3D_SUBSET[input_examples.FULL_IDS_KEY] = [
    FULL_ID_STRINGS[k] for k in INDICES_TO_KEEP
]
EXAMPLE_DICT_3D_SUBSET[input_examples.STORM_TIMES_KEY] = (
    STORM_TIMES_UNIX_SEC[INDICES_TO_KEEP]
)
EXAMPLE_DICT_3D_SUBSET[input_examples.TARGET_VALUES_KEY] = (
    TARGET_VALUES[INDICES_TO_KEEP]
)
EXAMPLE_DICT_3D_SUBSET[input_examples.RADAR_IMAGE_MATRIX_KEY] = (
    THIS_RADAR_IMAGE_MATRIX[INDICES_TO_KEEP, ...]
)

THESE_FIELD_NAMES = [
    radar_utils.LOW_LEVEL_SHEAR_NAME, radar_utils.MID_LEVEL_SHEAR_NAME
]
THESE_HEIGHTS_M_AGL = numpy.array([1000, 2000, 3000, 4000, 5000, 6000, 7000])

THIS_REFL_IMAGE_MATRIX_DBZ = numpy.random.uniform(
    low=0., high=1., size=(4, 32, 32, 7, 1)
)
THIS_AZ_SHEAR_IMAGE_MATRIX_S01 = numpy.random.uniform(
    low=0., high=1., size=(4, 32, 32, 2)
)

EXAMPLE_DICT_2D3D_ORIG = {
    input_examples.ROTATED_GRIDS_KEY: True,
    input_examples.ROTATED_GRID_SPACING_KEY: 1500.,
    input_examples.TARGET_NAME_KEY: 'foo',
    input_examples.RADAR_FIELDS_KEY: THESE_FIELD_NAMES,
    input_examples.RADAR_HEIGHTS_KEY: THESE_HEIGHTS_M_AGL,
    input_examples.FULL_IDS_KEY: FULL_ID_STRINGS,
    input_examples.STORM_TIMES_KEY: STORM_TIMES_UNIX_SEC,
    input_examples.TARGET_VALUES_KEY: TARGET_VALUES,
    input_examples.REFL_IMAGE_MATRIX_KEY: THIS_REFL_IMAGE_MATRIX_DBZ,
    input_examples.AZ_SHEAR_IMAGE_MATRIX_KEY: THIS_AZ_SHEAR_IMAGE_MATRIX_S01
}

EXAMPLE_DICT_2D3D_SUBSET = copy.deepcopy(EXAMPLE_DICT_2D3D_ORIG)
EXAMPLE_DICT_2D3D_SUBSET[input_examples.FULL_IDS_KEY] = [
    FULL_ID_STRINGS[k] for k in INDICES_TO_KEEP
]
EXAMPLE_DICT_2D3D_SUBSET[input_examples.STORM_TIMES_KEY] = (
    STORM_TIMES_UNIX_SEC[INDICES_TO_KEEP]
)
EXAMPLE_DICT_2D3D_SUBSET[input_examples.TARGET_VALUES_KEY] = (
    TARGET_VALUES[INDICES_TO_KEEP]
)
EXAMPLE_DICT_2D3D_SUBSET[input_examples.REFL_IMAGE_MATRIX_KEY] = (
    THIS_REFL_IMAGE_MATRIX_DBZ[INDICES_TO_KEEP, ...]
)
EXAMPLE_DICT_2D3D_SUBSET[input_examples.AZ_SHEAR_IMAGE_MATRIX_KEY] = (
    THIS_AZ_SHEAR_IMAGE_MATRIX_S01[INDICES_TO_KEEP, ...]
)

# The following constants are used to test _check_layer_operation.
OPERATION_DICT_3D_GOOD = {
    input_examples.RADAR_FIELD_KEY: radar_utils.DIFFERENTIAL_REFL_NAME,
    input_examples.OPERATION_NAME_KEY: input_examples.MAX_OPERATION_NAME,
    input_examples.MIN_HEIGHT_KEY: 1000,
    input_examples.MAX_HEIGHT_KEY: 6000
}

OPERATION_DICT_3D_BAD = {
    input_examples.RADAR_FIELD_KEY: radar_utils.DIFFERENTIAL_REFL_NAME,
    input_examples.OPERATION_NAME_KEY: input_examples.MAX_OPERATION_NAME,
    input_examples.MIN_HEIGHT_KEY: 1000,
    input_examples.MAX_HEIGHT_KEY: 7000
}

OPERATION_DICT_2D3D_GOOD = {
    input_examples.RADAR_FIELD_KEY: radar_utils.REFL_NAME,
    input_examples.OPERATION_NAME_KEY: input_examples.MAX_OPERATION_NAME,
    input_examples.MIN_HEIGHT_KEY: 1000,
    input_examples.MAX_HEIGHT_KEY: 6000
}

OPERATION_DICT_2D3D_BAD = {
    input_examples.RADAR_FIELD_KEY: radar_utils.DIFFERENTIAL_REFL_NAME,
    input_examples.OPERATION_NAME_KEY: input_examples.MAX_OPERATION_NAME,
    input_examples.MIN_HEIGHT_KEY: 1000,
    input_examples.MAX_HEIGHT_KEY: 6000
}

# The following constants are used to test _apply_layer_operation.
THESE_FIELD_NAMES = [radar_utils.REFL_NAME, radar_utils.DIFFERENTIAL_REFL_NAME]
THESE_HEIGHTS_M_AGL = numpy.array(
    [1000, 2000, 3000, 4000, 5000, 6000], dtype=int
)

THIS_HEIGHT1_MATRIX = numpy.array([
    [1, 2, 3, 4, 5, 6],
    [7, 8, 9, 10, 11, 12],
    [13, 14, 15, 16, 17, 18],
    [19, 20, 21, 22, 23, 24]
])

THIS_EXAMPLE1_MATRIX = numpy.stack(
    (THIS_HEIGHT1_MATRIX, THIS_HEIGHT1_MATRIX + 6, THIS_HEIGHT1_MATRIX + 12,
     THIS_HEIGHT1_MATRIX + 18, THIS_HEIGHT1_MATRIX + 24,
     THIS_HEIGHT1_MATRIX + 30),
    axis=-1
)

THIS_REFL_MATRIX_DBZ = numpy.stack(
    (THIS_EXAMPLE1_MATRIX, THIS_EXAMPLE1_MATRIX - 10,
     THIS_EXAMPLE1_MATRIX + 10),
    axis=0
)

THIS_RADAR_IMAGE_MATRIX = numpy.stack(
    (THIS_REFL_MATRIX_DBZ, THIS_REFL_MATRIX_DBZ - 1000), axis=-1
).astype(float)

# First operation to verify.
PRE_OPERATION_EXAMPLE_DICT = {
    input_examples.RADAR_FIELDS_KEY: THESE_FIELD_NAMES,
    input_examples.RADAR_HEIGHTS_KEY: THESE_HEIGHTS_M_AGL,
    input_examples.RADAR_IMAGE_MATRIX_KEY: THIS_RADAR_IMAGE_MATRIX
}

MAX_REFL_OPERATION_DICT_ORIG = {
    input_examples.RADAR_FIELD_KEY: radar_utils.REFL_NAME,
    input_examples.OPERATION_NAME_KEY: input_examples.MAX_OPERATION_NAME,
    input_examples.MIN_HEIGHT_KEY: 1750,
    input_examples.MAX_HEIGHT_KEY: 5666
}

MAX_REFL_OPERATION_DICT_NEW = {
    input_examples.RADAR_FIELD_KEY: radar_utils.REFL_NAME,
    input_examples.OPERATION_NAME_KEY: input_examples.MAX_OPERATION_NAME,
    input_examples.MIN_HEIGHT_KEY: 1000,
    input_examples.MAX_HEIGHT_KEY: 6000
}

MAX_REFL_MATRIX_DBZ = numpy.stack(
    (THIS_HEIGHT1_MATRIX + 30, THIS_HEIGHT1_MATRIX + 20,
     THIS_HEIGHT1_MATRIX + 40),
    axis=0
).astype(float)

# Second operation to verify.
MIN_DIFF_REFL_OPERATION_DICT_ORIG = {
    input_examples.RADAR_FIELD_KEY: radar_utils.DIFFERENTIAL_REFL_NAME,
    input_examples.OPERATION_NAME_KEY: input_examples.MIN_OPERATION_NAME,
    input_examples.MIN_HEIGHT_KEY: 1111,
    input_examples.MAX_HEIGHT_KEY: 5555
}

MIN_DIFF_REFL_OPERATION_DICT_NEW = {
    input_examples.RADAR_FIELD_KEY: radar_utils.DIFFERENTIAL_REFL_NAME,
    input_examples.OPERATION_NAME_KEY: input_examples.MIN_OPERATION_NAME,
    input_examples.MIN_HEIGHT_KEY: 1000,
    input_examples.MAX_HEIGHT_KEY: 6000
}

MIN_DIFF_REFL_MATRIX_DB = numpy.stack(
    (THIS_HEIGHT1_MATRIX - 1000, THIS_HEIGHT1_MATRIX - 1010,
     THIS_HEIGHT1_MATRIX - 990),
    axis=0
).astype(float)

# The following constants are used to test find_example_file and
# _file_name_to_batch_number.
TOP_DIRECTORY_NAME = 'foo'
BATCH_NUMBER = 1967
SPC_DATE_STRING = '19670502'

EXAMPLE_FILE_NAME_SHUFFLED = (
    'foo/batches0001000-0001999/input_examples_batch0001967.nc'
)
EXAMPLE_FILE_NAME_UNSHUFFLED = 'foo/1967/input_examples_19670502.nc'


def _compare_radar_image_dicts(first_radar_image_dict, second_radar_image_dict):
    """Compares two dictionaries with storm-centered radar images.

    :param first_radar_image_dict: First dictionary.
    :param second_radar_image_dict: Second dictionary.
    :return: are_dicts_equal: Boolean flag.
    """

    first_keys = first_radar_image_dict.keys()
    second_keys = second_radar_image_dict.keys()
    if set(first_keys) != set(second_keys):
        return False

    for this_key in first_keys:
        if this_key == storm_images.FULL_IDS_KEY:
            if (first_radar_image_dict[this_key] !=
                    second_radar_image_dict[this_key]):
                return False

        elif this_key in [storm_images.VALID_TIMES_KEY,
                          storm_images.LABEL_VALUES_KEY]:
            if not numpy.array_equal(first_radar_image_dict[this_key],
                                     second_radar_image_dict[this_key]):
                return False

        else:
            if not numpy.allclose(first_radar_image_dict[this_key],
                                  second_radar_image_dict[this_key],
                                  atol=TOLERANCE):
                return False

    return True


def _compare_example_dicts(first_example_dict, second_example_dict):
    """Compares two dictionaries with full input examples.

    :param first_example_dict: First dictionary.
    :param second_example_dict: Second dictionary.
    :return: are_dicts_equal: Boolean flag.
    """

    first_keys = first_example_dict.keys()
    second_keys = second_example_dict.keys()
    if set(first_keys) != set(second_keys):
        return False

    for this_key in first_keys:
        if this_key in EQUALS_SIGN_KEYS:
            if first_example_dict[this_key] != second_example_dict[this_key]:
                return False

        elif this_key in ARRAY_EQUAL_KEYS:
            if not numpy.array_equal(first_example_dict[this_key],
                                     second_example_dict[this_key]):
                return False
        else:
            if not numpy.allclose(first_example_dict[this_key],
                                  second_example_dict[this_key],
                                  atol=TOLERANCE):
                return False

    return True


class InputExamplesTests(unittest.TestCase):
    """Each method is a unit test for input_examples.py."""

    def test_filter_examples_by_class_tornado_first(self):
        """Ensures correct output from _filter_examples_by_class.

        In this case, the target phenomenon is tornadogenesis and the number of
        desired examples from all classes is non-zero.
        """

        these_indices_to_keep = input_examples._filter_examples_by_class(
            target_values=TARGET_VALUES_TORNADO,
            class_to_num_examples_dict=FIRST_TORNADO_CLASS_TO_NUM_EX_DICT,
            test_mode=True)

        self.assertTrue(numpy.array_equal(
            these_indices_to_keep, FIRST_TORNADO_INDICES_TO_KEEP
        ))

    def test_filter_examples_by_class_tornado_second(self):
        """Ensures correct output from _filter_examples_by_class.

        In this case, the target phenomenon is tornadogenesis and the number of
        desired examples from some classes is zero.
        """

        these_indices_to_keep = input_examples._filter_examples_by_class(
            target_values=TARGET_VALUES_TORNADO,
            class_to_num_examples_dict=SECOND_TORNADO_CLASS_TO_NUM_EX_DICT,
            test_mode=True)

        self.assertTrue(numpy.array_equal(
            these_indices_to_keep, SECOND_TORNADO_INDICES_TO_KEEP
        ))

    def test_filter_examples_by_class_wind_first(self):
        """Ensures correct output from _filter_examples_by_class.

        In this case, the target phenomenon is wind speed and the number of
        desired examples from all classes is non-zero.
        """

        these_indices_to_keep = input_examples._filter_examples_by_class(
            target_values=TARGET_VALUES_WIND,
            class_to_num_examples_dict=FIRST_WIND_CLASS_TO_NUM_EX_DICT,
            test_mode=True)

        self.assertTrue(numpy.array_equal(
            these_indices_to_keep, FIRST_WIND_INDICES_TO_KEEP
        ))

    def test_filter_examples_by_class_wind_second(self):
        """Ensures correct output from _filter_examples_by_class.

        In this case, the target phenomenon is wind speed and the number of
        desired examples from some classes is zero.
        """

        these_indices_to_keep = input_examples._filter_examples_by_class(
            target_values=TARGET_VALUES_WIND,
            class_to_num_examples_dict=SECOND_WIND_CLASS_TO_NUM_EX_DICT,
            test_mode=True)

        self.assertTrue(numpy.array_equal(
            these_indices_to_keep, SECOND_WIND_INDICES_TO_KEEP
        ))

    def test_remove_storms_with_undefined_target(self):
        """Ensures correct output from remove_storms_with_undefined_target."""

        this_radar_image_dict = (
            input_examples.remove_storms_with_undefined_target(
                copy.deepcopy(RADAR_IMAGE_DICT_UNFILTERED))
        )

        self.assertTrue(_compare_radar_image_dicts(
            this_radar_image_dict, RADAR_IMAGE_DICT_NO_UNDEF_TARGETS
        ))

    def test_subset_examples_2d(self):
        """Ensures correct output from subset_examples.

        In this case examples contain only 2-D radar images.
        """

        this_example_dict = input_examples.subset_examples(
            example_dict=EXAMPLE_DICT_2D_ORIG, indices_to_keep=INDICES_TO_KEEP,
            create_new_dict=True)

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_2D_SUBSET
        ))

    def test_subset_examples_3d(self):
        """Ensures correct output from subset_examples.

        In this case examples contain only 3-D radar images.
        """

        this_example_dict = input_examples.subset_examples(
            example_dict=EXAMPLE_DICT_3D_ORIG, indices_to_keep=INDICES_TO_KEEP,
            create_new_dict=True)

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_3D_SUBSET
        ))

    def test_subset_examples_2d3d(self):
        """Ensures correct output from subset_examples.

        In this case examples contain both 2-D and 3-D radar images.
        """

        this_example_dict = input_examples.subset_examples(
            example_dict=EXAMPLE_DICT_2D3D_ORIG,
            indices_to_keep=INDICES_TO_KEEP, create_new_dict=True)

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_2D3D_SUBSET
        ))

    def test_check_layer_operation_3d_good(self):
        """Ensures correct output from _check_layer_operation.

        In this case, radar images are 3-D only and the dictionary is correctly
        formatted.
        """

        input_examples._check_layer_operation(
            example_dict=EXAMPLE_DICT_3D_ORIG,
            operation_dict=OPERATION_DICT_3D_GOOD)

    def test_check_layer_operation_3d_bad(self):
        """Ensures correct output from _check_layer_operation.

        In this case, radar images are 3-D only and the dictionary is *not*
        correctly formatted.
        """

        with self.assertRaises(ValueError):
            input_examples._check_layer_operation(
                example_dict=EXAMPLE_DICT_3D_ORIG,
                operation_dict=OPERATION_DICT_3D_BAD)

    def test_check_layer_operation_2d3d_good(self):
        """Ensures correct output from _check_layer_operation.

        In this case, radar images are 2D/3D and the dictionary is correctly
        formatted.
        """

        input_examples._check_layer_operation(
            example_dict=EXAMPLE_DICT_2D3D_ORIG,
            operation_dict=OPERATION_DICT_2D3D_GOOD)

    def test_check_layer_operation_2d3d_bad(self):
        """Ensures correct output from _check_layer_operation.

        In this case, radar images are 2D/3D and the dictionary is *not*
        correctly formatted.
        """

        with self.assertRaises(ValueError):
            input_examples._check_layer_operation(
                example_dict=EXAMPLE_DICT_2D3D_ORIG,
                operation_dict=OPERATION_DICT_2D3D_BAD)

    def test_apply_layer_operation_max_refl(self):
        """Ensures correct output from _apply_layer_operation.

        In this case the layer operation is max reflectivity.
        """

        this_radar_matrix, this_operation_dict = (
            input_examples._apply_layer_operation(
                example_dict=PRE_OPERATION_EXAMPLE_DICT,
                operation_dict=copy.deepcopy(MAX_REFL_OPERATION_DICT_ORIG)
            )
        )

        self.assertTrue(numpy.allclose(
            this_radar_matrix, MAX_REFL_MATRIX_DBZ, atol=TOLERANCE
        ))
        self.assertTrue(this_operation_dict == MAX_REFL_OPERATION_DICT_NEW)

    def test_apply_layer_operation_min_diff_refl(self):
        """Ensures correct output from _apply_layer_operation.

        In this case the layer operation is minimum differential reflectivity.
        """

        this_radar_matrix, this_operation_dict = (
            input_examples._apply_layer_operation(
                example_dict=PRE_OPERATION_EXAMPLE_DICT,
                operation_dict=copy.deepcopy(MIN_DIFF_REFL_OPERATION_DICT_ORIG)
            )
        )

        self.assertTrue(numpy.allclose(
            this_radar_matrix, MIN_DIFF_REFL_MATRIX_DB, atol=TOLERANCE
        ))
        self.assertTrue(this_operation_dict == MIN_DIFF_REFL_OPERATION_DICT_NEW)

    def test_find_example_file_shuffled(self):
        """Ensures correct output from find_example_file.

        In this case the hypothetical file is temporally shuffled.
        """

        this_file_name = input_examples.find_example_file(
            top_directory_name=TOP_DIRECTORY_NAME, shuffled=True,
            batch_number=BATCH_NUMBER, raise_error_if_missing=False)

        self.assertTrue(this_file_name == EXAMPLE_FILE_NAME_SHUFFLED)

    def test_find_example_file_unshuffled(self):
        """Ensures correct output from find_example_file.

        In this case the hypothetical file is *not* temporally shuffled.
        """

        this_file_name = input_examples.find_example_file(
            top_directory_name=TOP_DIRECTORY_NAME, shuffled=False,
            spc_date_string=SPC_DATE_STRING, raise_error_if_missing=False)

        self.assertTrue(this_file_name == EXAMPLE_FILE_NAME_UNSHUFFLED)

    def test_file_name_to_batch_number_shuffled(self):
        """Ensures correct output from _file_name_to_batch_number.

        In this case the input file is shuffled, so _file_name_to_batch_number
        should return a batch number.
        """

        this_batch_number = input_examples._file_name_to_batch_number(
            EXAMPLE_FILE_NAME_SHUFFLED)

        self.assertTrue(this_batch_number == BATCH_NUMBER)

    def test_file_name_to_batch_number_unshuffled(self):
        """Ensures correct output from _file_name_to_batch_number.

        In this case the input file is *not* shuffled, so
        _file_name_to_batch_number should be unable to find a batch number, thus
        return an error.
        """

        with self.assertRaises(ValueError):
            input_examples._file_name_to_batch_number(
                EXAMPLE_FILE_NAME_UNSHUFFLED)


if __name__ == '__main__':
    unittest.main()
