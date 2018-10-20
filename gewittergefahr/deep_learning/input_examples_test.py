"""Unit tests for input_examples.py."""

import copy
import unittest
import numpy
from gewittergefahr.deep_learning import storm_images
from gewittergefahr.deep_learning import input_examples

TOLERANCE = 1e-6

# The following constants are used to test remove_storms_with_undefined_target.
THESE_STORM_IDS = ['Matthews', 'Tavares', 'Marner', 'Nylander']
THESE_TIMES_UNIX_SEC = numpy.array([1, 2, 3, 4], dtype=int)
THESE_TARGET_VALUES = numpy.array([-1, 0, -1, -2], dtype=int)
THIS_RADAR_IMAGE_MATRIX = numpy.random.uniform(
    low=0., high=1., size=(4, 32, 32, 3))

RADAR_IMAGE_DICT_UNFILTERED = {
    storm_images.STORM_IDS_KEY: THESE_STORM_IDS,
    storm_images.VALID_TIMES_KEY: THESE_TIMES_UNIX_SEC,
    storm_images.LABEL_VALUES_KEY: THESE_TARGET_VALUES,
    storm_images.STORM_IMAGE_MATRIX_KEY: THIS_RADAR_IMAGE_MATRIX
}

THESE_VALID_INDICES = numpy.array([1, 3], dtype=int)
RADAR_IMAGE_DICT_NO_UNDEF_TARGETS = {
    storm_images.STORM_IDS_KEY:
        [THESE_STORM_IDS[i] for i in THESE_VALID_INDICES],
    storm_images.VALID_TIMES_KEY: THESE_TIMES_UNIX_SEC[THESE_VALID_INDICES],
    storm_images.LABEL_VALUES_KEY: THESE_TARGET_VALUES[THESE_VALID_INDICES],
    storm_images.STORM_IMAGE_MATRIX_KEY:
        THIS_RADAR_IMAGE_MATRIX[THESE_VALID_INDICES, ...]
}

# The following constants are used to test find_example_file.
TOP_DIRECTORY_NAME = 'foo'
BATCH_NUMBER = 1967
EXAMPLE_FILE_NAME = 'foo/batches0001000-0001999/input_examples_batch0001967.nc'


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

    def test_find_example_file(self):
        """Ensures correct output from find_example_file."""

        this_file_name = input_examples.find_example_file(
            top_directory_name=TOP_DIRECTORY_NAME, batch_number=BATCH_NUMBER,
            raise_error_if_missing=False)

        self.assertTrue(this_file_name == EXAMPLE_FILE_NAME)


if __name__ == '__main__':
    unittest.main()
