"""Unit tests for plot_input_examples.py."""

import unittest
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.scripts import plot_input_examples

FIGURE_DIR_NAME = 'figures'
FULL_STORM_ID_STRING = 'foo_bar'
STORM_TIME_UNIX_SEC = 1560297467

FIRST_RADAR_FIELD_NAME = None
FIRST_RADAR_HEIGHT_M_AGL = None
FIRST_LAYER_OPERATION_DICT = None
FIRST_FIGURE_FILE_NAME = 'figures/storm=foo-bar_2019-06-11-235747_radar.jpg'

FIRST_METADATA_DICT = {
    plot_input_examples.FULL_STORM_ID_KEY: FULL_STORM_ID_STRING,
    plot_input_examples.STORM_TIME_KEY: STORM_TIME_UNIX_SEC,
    plot_input_examples.RADAR_FIELD_KEY: FIRST_RADAR_FIELD_NAME,
    plot_input_examples.RADAR_HEIGHT_KEY: FIRST_RADAR_HEIGHT_M_AGL,
    plot_input_examples.LAYER_OPERATION_KEY: FIRST_LAYER_OPERATION_DICT,
    plot_input_examples.PMM_FLAG_KEY: False,
    plot_input_examples.IS_SOUNDING_KEY: False
}

SECOND_RADAR_FIELD_NAME = 'reflectivity_dbz'
SECOND_RADAR_HEIGHT_M_AGL = None
SECOND_LAYER_OPERATION_DICT = None
SECOND_FIGURE_FILE_NAME = (
    'figures/storm=foo-bar_2019-06-11-235747_reflectivity-dbz.jpg')

SECOND_METADATA_DICT = {
    plot_input_examples.FULL_STORM_ID_KEY: FULL_STORM_ID_STRING,
    plot_input_examples.STORM_TIME_KEY: STORM_TIME_UNIX_SEC,
    plot_input_examples.RADAR_FIELD_KEY: SECOND_RADAR_FIELD_NAME,
    plot_input_examples.RADAR_HEIGHT_KEY: SECOND_RADAR_HEIGHT_M_AGL,
    plot_input_examples.LAYER_OPERATION_KEY: SECOND_LAYER_OPERATION_DICT,
    plot_input_examples.PMM_FLAG_KEY: False,
    plot_input_examples.IS_SOUNDING_KEY: False
}

THIRD_RADAR_FIELD_NAME = 'reflectivity_dbz'
THIRD_RADAR_HEIGHT_M_AGL = 3000
THIRD_LAYER_OPERATION_DICT = None
THIRD_FIGURE_FILE_NAME = (
    'figures/storm=foo-bar_2019-06-11-235747_reflectivity-dbz_03000metres.jpg')

THIRD_METADATA_DICT = {
    plot_input_examples.FULL_STORM_ID_KEY: FULL_STORM_ID_STRING,
    plot_input_examples.STORM_TIME_KEY: STORM_TIME_UNIX_SEC,
    plot_input_examples.RADAR_FIELD_KEY: THIRD_RADAR_FIELD_NAME,
    plot_input_examples.RADAR_HEIGHT_KEY: THIRD_RADAR_HEIGHT_M_AGL,
    plot_input_examples.LAYER_OPERATION_KEY: THIRD_LAYER_OPERATION_DICT,
    plot_input_examples.PMM_FLAG_KEY: False,
    plot_input_examples.IS_SOUNDING_KEY: False
}

FOURTH_RADAR_FIELD_NAME = None
FOURTH_RADAR_HEIGHT_M_AGL = None
FOURTH_LAYER_OPERATION_DICT = {
    input_examples.RADAR_FIELD_KEY: 'reflectivity_dbz',
    input_examples.OPERATION_NAME_KEY: 'max',
    input_examples.MIN_HEIGHT_KEY: 1000,
    input_examples.MAX_HEIGHT_KEY: 3000
}

FOURTH_FIGURE_FILE_NAME = (
    'figures/storm=foo-bar_2019-06-11-235747_reflectivity-dbz_'
    'max-01000-03000metres.jpg')

FOURTH_METADATA_DICT = {
    plot_input_examples.FULL_STORM_ID_KEY: FULL_STORM_ID_STRING,
    plot_input_examples.STORM_TIME_KEY: STORM_TIME_UNIX_SEC,
    plot_input_examples.RADAR_FIELD_KEY: FOURTH_RADAR_FIELD_NAME,
    plot_input_examples.RADAR_HEIGHT_KEY: FOURTH_RADAR_HEIGHT_M_AGL,
    plot_input_examples.LAYER_OPERATION_KEY: FOURTH_LAYER_OPERATION_DICT,
    plot_input_examples.PMM_FLAG_KEY: False,
    plot_input_examples.IS_SOUNDING_KEY: False
}


def _compare_metadata_dicts(first_dict, second_dict):
    """Compares dictionaries created by `file_name_to_metadata`.

    :param first_dict: First dictionary.
    :param second_dict: Second dictionary.
    :return: are_dicts_equal: Boolean flag.
    """

    first_keys = list(first_dict.keys())
    second_keys = list(second_dict.keys())
    if set(first_keys) != set(second_keys):
        return False

    for this_key in first_keys:
        if first_dict[this_key] is None and second_dict[this_key] is not None:
            return False

        if second_dict[this_key] is None and first_dict[this_key] is not None:
            return False

        if first_dict[this_key] is None:
            continue

        if first_dict[this_key] != second_dict[this_key]:
            return False

    return True


class PlotInputExamplesTests(unittest.TestCase):
    """Each method is a unit test for plot_input_examples.py."""

    def test_metadata_to_file_name_first(self):
        """Ensures correct output from metadata_to_file_name.

        In this case, using first set of metadata.
        """

        this_file_name = plot_input_examples.metadata_to_file_name(
            output_dir_name=FIGURE_DIR_NAME, is_sounding=False, pmm_flag=False,
            full_storm_id_string=FULL_STORM_ID_STRING,
            storm_time_unix_sec=STORM_TIME_UNIX_SEC,
            radar_field_name=FIRST_RADAR_FIELD_NAME,
            radar_height_m_agl=FIRST_RADAR_HEIGHT_M_AGL,
            layer_operation_dict=FIRST_LAYER_OPERATION_DICT)

        self.assertTrue(this_file_name == FIRST_FIGURE_FILE_NAME)

    def test_file_name_to_metadata_first(self):
        """Ensures correct output from file_name_to_metadata.

        In this case, using first file name.
        """

        this_metadata_dict = (
            plot_input_examples.file_name_to_metadata(FIRST_FIGURE_FILE_NAME)
        )

        self.assertTrue(_compare_metadata_dicts(
            this_metadata_dict, FIRST_METADATA_DICT
        ))

    def test_metadata_to_file_name_second(self):
        """Ensures correct output from metadata_to_file_name.

        In this case, using second set of metadata.
        """

        this_file_name = plot_input_examples.metadata_to_file_name(
            output_dir_name=FIGURE_DIR_NAME, is_sounding=False, pmm_flag=False,
            full_storm_id_string=FULL_STORM_ID_STRING,
            storm_time_unix_sec=STORM_TIME_UNIX_SEC,
            radar_field_name=SECOND_RADAR_FIELD_NAME,
            radar_height_m_agl=SECOND_RADAR_HEIGHT_M_AGL,
            layer_operation_dict=SECOND_LAYER_OPERATION_DICT)

        self.assertTrue(this_file_name == SECOND_FIGURE_FILE_NAME)

    def test_file_name_to_metadata_second(self):
        """Ensures correct output from file_name_to_metadata.

        In this case, using second file name.
        """

        this_metadata_dict = (
            plot_input_examples.file_name_to_metadata(SECOND_FIGURE_FILE_NAME)
        )

        self.assertTrue(_compare_metadata_dicts(
            this_metadata_dict, SECOND_METADATA_DICT
        ))

    def test_metadata_to_file_name_third(self):
        """Ensures correct output from metadata_to_file_name.

        In this case, using third set of metadata.
        """

        this_file_name = plot_input_examples.metadata_to_file_name(
            output_dir_name=FIGURE_DIR_NAME, is_sounding=False, pmm_flag=False,
            full_storm_id_string=FULL_STORM_ID_STRING,
            storm_time_unix_sec=STORM_TIME_UNIX_SEC,
            radar_field_name=THIRD_RADAR_FIELD_NAME,
            radar_height_m_agl=THIRD_RADAR_HEIGHT_M_AGL,
            layer_operation_dict=THIRD_LAYER_OPERATION_DICT)

        self.assertTrue(this_file_name == THIRD_FIGURE_FILE_NAME)

    def test_file_name_to_metadata_third(self):
        """Ensures correct output from file_name_to_metadata.

        In this case, using third file name.
        """

        this_metadata_dict = (
            plot_input_examples.file_name_to_metadata(THIRD_FIGURE_FILE_NAME)
        )

        self.assertTrue(_compare_metadata_dicts(
            this_metadata_dict, THIRD_METADATA_DICT
        ))

    def test_metadata_to_file_name_fourth(self):
        """Ensures correct output from metadata_to_file_name.

        In this case, using fourth set of metadata.
        """

        this_file_name = plot_input_examples.metadata_to_file_name(
            output_dir_name=FIGURE_DIR_NAME, is_sounding=False, pmm_flag=False,
            full_storm_id_string=FULL_STORM_ID_STRING,
            storm_time_unix_sec=STORM_TIME_UNIX_SEC,
            radar_field_name=FOURTH_RADAR_FIELD_NAME,
            radar_height_m_agl=FOURTH_RADAR_HEIGHT_M_AGL,
            layer_operation_dict=FOURTH_LAYER_OPERATION_DICT)

        self.assertTrue(this_file_name == FOURTH_FIGURE_FILE_NAME)

    def test_file_name_to_metadata_fourth(self):
        """Ensures correct output from file_name_to_metadata.

        In this case, using fourth file name.
        """

        this_metadata_dict = (
            plot_input_examples.file_name_to_metadata(FOURTH_FIGURE_FILE_NAME)
        )

        self.assertTrue(_compare_metadata_dicts(
            this_metadata_dict, FOURTH_METADATA_DICT
        ))


if __name__ == '__main__':
    unittest.main()
