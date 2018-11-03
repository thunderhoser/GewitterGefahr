"""Unit tests for input_examples.py."""

import copy
import unittest
import numpy
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.deep_learning import storm_images
from gewittergefahr.deep_learning import input_examples

TOLERANCE = 1e-6

# The following constants are used to test remove_storms_with_undefined_target.
STORM_IDS = ['Matthews', 'Tavares', 'Marner', 'Nylander']
STORM_TIMES_UNIX_SEC = numpy.array([1, 2, 3, 4], dtype=int)
TARGET_VALUES = numpy.array([-1, 0, -1, -2], dtype=int)
THIS_RADAR_IMAGE_MATRIX = numpy.random.uniform(
    low=0., high=1., size=(4, 32, 32, 3))

RADAR_IMAGE_DICT_UNFILTERED = {
    storm_images.STORM_IDS_KEY: STORM_IDS,
    storm_images.VALID_TIMES_KEY: STORM_TIMES_UNIX_SEC,
    storm_images.LABEL_VALUES_KEY: TARGET_VALUES,
    storm_images.STORM_IMAGE_MATRIX_KEY: THIS_RADAR_IMAGE_MATRIX
}

INDICES_TO_KEEP = numpy.array([1, 3], dtype=int)
RADAR_IMAGE_DICT_NO_UNDEF_TARGETS = {
    storm_images.STORM_IDS_KEY: [STORM_IDS[i] for i in INDICES_TO_KEEP],
    storm_images.VALID_TIMES_KEY: STORM_TIMES_UNIX_SEC[INDICES_TO_KEEP],
    storm_images.LABEL_VALUES_KEY: TARGET_VALUES[INDICES_TO_KEEP],
    storm_images.STORM_IMAGE_MATRIX_KEY:
        THIS_RADAR_IMAGE_MATRIX[INDICES_TO_KEEP, ...]
}

# The following constants are used to test subset_examples.
EQUALS_SIGN_KEYS = [
    input_examples.ROTATED_GRIDS_KEY, input_examples.TARGET_NAME_KEY,
    input_examples.RADAR_FIELDS_KEY, input_examples.STORM_IDS_KEY
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
    input_examples.STORM_IDS_KEY: STORM_IDS,
    input_examples.STORM_TIMES_KEY: STORM_TIMES_UNIX_SEC,
    input_examples.TARGET_VALUES_KEY: TARGET_VALUES,
    input_examples.RADAR_IMAGE_MATRIX_KEY: THIS_RADAR_IMAGE_MATRIX
}

EXAMPLE_DICT_2D_SUBSET = copy.deepcopy(EXAMPLE_DICT_2D_ORIG)
EXAMPLE_DICT_2D_SUBSET[
    input_examples.STORM_IDS_KEY
] = [STORM_IDS[k] for k in INDICES_TO_KEEP]
EXAMPLE_DICT_2D_SUBSET[
    input_examples.STORM_TIMES_KEY
] = STORM_TIMES_UNIX_SEC[INDICES_TO_KEEP]
EXAMPLE_DICT_2D_SUBSET[
    input_examples.TARGET_VALUES_KEY
] = TARGET_VALUES[INDICES_TO_KEEP]
EXAMPLE_DICT_2D_SUBSET[
    input_examples.RADAR_IMAGE_MATRIX_KEY
] = THIS_RADAR_IMAGE_MATRIX[INDICES_TO_KEEP, ...]

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
    input_examples.STORM_IDS_KEY: STORM_IDS,
    input_examples.STORM_TIMES_KEY: STORM_TIMES_UNIX_SEC,
    input_examples.TARGET_VALUES_KEY: TARGET_VALUES,
    input_examples.RADAR_IMAGE_MATRIX_KEY: THIS_RADAR_IMAGE_MATRIX
}

EXAMPLE_DICT_3D_SUBSET = copy.deepcopy(EXAMPLE_DICT_3D_ORIG)
EXAMPLE_DICT_3D_SUBSET[
    input_examples.STORM_IDS_KEY
] = [STORM_IDS[k] for k in INDICES_TO_KEEP]
EXAMPLE_DICT_3D_SUBSET[
    input_examples.STORM_TIMES_KEY
] = STORM_TIMES_UNIX_SEC[INDICES_TO_KEEP]
EXAMPLE_DICT_3D_SUBSET[
    input_examples.TARGET_VALUES_KEY
] = TARGET_VALUES[INDICES_TO_KEEP]
EXAMPLE_DICT_3D_SUBSET[
    input_examples.RADAR_IMAGE_MATRIX_KEY
] = THIS_RADAR_IMAGE_MATRIX[INDICES_TO_KEEP, ...]

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
    input_examples.STORM_IDS_KEY: STORM_IDS,
    input_examples.STORM_TIMES_KEY: STORM_TIMES_UNIX_SEC,
    input_examples.TARGET_VALUES_KEY: TARGET_VALUES,
    input_examples.REFL_IMAGE_MATRIX_KEY: THIS_REFL_IMAGE_MATRIX_DBZ,
    input_examples.AZ_SHEAR_IMAGE_MATRIX_KEY: THIS_AZ_SHEAR_IMAGE_MATRIX_S01
}

EXAMPLE_DICT_2D3D_SUBSET = copy.deepcopy(EXAMPLE_DICT_2D3D_ORIG)
EXAMPLE_DICT_2D3D_SUBSET[
    input_examples.STORM_IDS_KEY
] = [STORM_IDS[k] for k in INDICES_TO_KEEP]
EXAMPLE_DICT_2D3D_SUBSET[
    input_examples.STORM_TIMES_KEY
] = STORM_TIMES_UNIX_SEC[INDICES_TO_KEEP]
EXAMPLE_DICT_2D3D_SUBSET[
    input_examples.TARGET_VALUES_KEY
] = TARGET_VALUES[INDICES_TO_KEEP]
EXAMPLE_DICT_2D3D_SUBSET[
    input_examples.REFL_IMAGE_MATRIX_KEY
] = THIS_REFL_IMAGE_MATRIX_DBZ[INDICES_TO_KEEP, ...]
EXAMPLE_DICT_2D3D_SUBSET[
    input_examples.AZ_SHEAR_IMAGE_MATRIX_KEY
] = THIS_AZ_SHEAR_IMAGE_MATRIX_S01[INDICES_TO_KEEP, ...]

# The following constants are used to test find_example_file and
# _file_name_to_batch_number.
TOP_DIRECTORY_NAME = 'foo'
BATCH_NUMBER = 1967
SPC_DATE_STRING = '19670502'

EXAMPLE_FILE_NAME_SHUFFLED = (
    'foo/batches0001000-0001999/input_examples_batch0001967.nc')
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
        if this_key == storm_images.STORM_IDS_KEY:
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

    def test_remove_storms_with_undefined_target(self):
        """Ensures correct output from remove_storms_with_undefined_target."""

        this_radar_image_dict = (
            input_examples.remove_storms_with_undefined_target(
                copy.deepcopy(RADAR_IMAGE_DICT_UNFILTERED))
        )

        self.assertTrue(_compare_radar_image_dicts(
            this_radar_image_dict, RADAR_IMAGE_DICT_NO_UNDEF_TARGETS))

    def test_subset_examples_2d(self):
        """Ensures correct output from subset_examples.

        In this case examples contain only 2-D radar images.
        """

        this_example_dict = input_examples.subset_examples(
            example_dict=EXAMPLE_DICT_2D_ORIG, indices_to_keep=INDICES_TO_KEEP,
            create_new_dict=True)

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_2D_SUBSET))

    def test_subset_examples_3d(self):
        """Ensures correct output from subset_examples.

        In this case examples contain only 3-D radar images.
        """

        this_example_dict = input_examples.subset_examples(
            example_dict=EXAMPLE_DICT_3D_ORIG, indices_to_keep=INDICES_TO_KEEP,
            create_new_dict=True)

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_3D_SUBSET))

    def test_subset_examples_2d3d(self):
        """Ensures correct output from subset_examples.

        In this case examples contain both 2-D and 3-D radar images.
        """

        this_example_dict = input_examples.subset_examples(
            example_dict=EXAMPLE_DICT_2D3D_ORIG,
            indices_to_keep=INDICES_TO_KEEP, create_new_dict=True)

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_2D3D_SUBSET))

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
