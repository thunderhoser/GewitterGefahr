"""Unit tests for myrorss_and_mrms_utils.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import myrorss_and_mrms_utils

TOLERANCE = 1e-6

# The following constants are used to test fields_and_refl_heights_to_dict.
FIELD_NAMES = [
    radar_utils.ECHO_TOP_50DBZ_NAME, radar_utils.LOW_LEVEL_SHEAR_NAME,
    radar_utils.REFL_NAME, radar_utils.REFL_COLUMN_MAX_NAME]
REFLECTIVITY_HEIGHTS_M_ASL = numpy.array(
    [250, 500, 750, 1000, 5000, 10000, 20000])

FIELD_TO_HEIGHTS_DICT_MYRORSS_M_ASL = {
    radar_utils.ECHO_TOP_50DBZ_NAME:
        numpy.array([radar_utils.DEFAULT_HEIGHT_MYRORSS_M_ASL]),
    radar_utils.LOW_LEVEL_SHEAR_NAME:
        numpy.array([radar_utils.SHEAR_HEIGHT_M_ASL]),
    radar_utils.REFL_NAME: REFLECTIVITY_HEIGHTS_M_ASL,
    radar_utils.REFL_COLUMN_MAX_NAME:
        numpy.array([radar_utils.DEFAULT_HEIGHT_MYRORSS_M_ASL])}

FIELD_TO_HEIGHTS_DICT_MRMS_M_ASL = {
    radar_utils.ECHO_TOP_50DBZ_NAME:
        numpy.array([radar_utils.DEFAULT_HEIGHT_MRMS_M_ASL]),
    radar_utils.LOW_LEVEL_SHEAR_NAME:
        numpy.array([radar_utils.SHEAR_HEIGHT_M_ASL]),
    radar_utils.REFL_NAME: REFLECTIVITY_HEIGHTS_M_ASL,
    radar_utils.REFL_COLUMN_MAX_NAME:
        numpy.array([radar_utils.DEFAULT_HEIGHT_MRMS_M_ASL])}

# The following constants are used to test fields_and_refl_heights_to_pairs.
FIELD_NAME_BY_PAIR = [
    radar_utils.ECHO_TOP_50DBZ_NAME, radar_utils.LOW_LEVEL_SHEAR_NAME,
    radar_utils.REFL_NAME, radar_utils.REFL_NAME, radar_utils.REFL_NAME,
    radar_utils.REFL_NAME, radar_utils.REFL_NAME, radar_utils.REFL_NAME,
    radar_utils.REFL_NAME, radar_utils.REFL_COLUMN_MAX_NAME]

THESE_FIRST_HEIGHTS_M_ASL = numpy.array(
    [radar_utils.DEFAULT_HEIGHT_MYRORSS_M_ASL, radar_utils.SHEAR_HEIGHT_M_ASL])
THESE_LAST_HEIGHTS_M_ASL = numpy.array(
    [radar_utils.DEFAULT_HEIGHT_MYRORSS_M_ASL])
HEIGHT_BY_PAIR_MYRORSS_M_ASL = numpy.concatenate((
    THESE_FIRST_HEIGHTS_M_ASL, REFLECTIVITY_HEIGHTS_M_ASL,
    THESE_LAST_HEIGHTS_M_ASL))

THESE_FIRST_HEIGHTS_M_ASL = numpy.array(
    [radar_utils.DEFAULT_HEIGHT_MRMS_M_ASL, radar_utils.SHEAR_HEIGHT_M_ASL])
THESE_LAST_HEIGHTS_M_ASL = numpy.array(
    [radar_utils.DEFAULT_HEIGHT_MRMS_M_ASL])
HEIGHT_BY_PAIR_MRMS_M_ASL = numpy.concatenate((
    THESE_FIRST_HEIGHTS_M_ASL, REFLECTIVITY_HEIGHTS_M_ASL,
    THESE_LAST_HEIGHTS_M_ASL))


class MyrorssAndMrmsUtilsTests(unittest.TestCase):
    """Each method is a unit test for myrorss_and_mrms_utils.py."""

    def test_fields_and_refl_heights_to_dict_myrorss(self):
        """Ensures correct output from fields_and_refl_heights_to_dict.

        In this case, data source is MYRORSS.
        """

        this_dictionary = (
            myrorss_and_mrms_utils.fields_and_refl_heights_to_dict(
                field_names=FIELD_NAMES,
                data_source=radar_utils.MYRORSS_SOURCE_ID,
                refl_heights_m_asl=REFLECTIVITY_HEIGHTS_M_ASL))

        self.assertTrue(this_dictionary == FIELD_TO_HEIGHTS_DICT_MYRORSS_M_ASL)

    def test_fields_and_refl_heights_to_dict_mrms(self):
        """Ensures correct output from fields_and_refl_heights_to_dict.

        In this case, data source is MRMS.
        """

        this_dictionary = (
            myrorss_and_mrms_utils.fields_and_refl_heights_to_dict(
                field_names=FIELD_NAMES,
                data_source=radar_utils.MRMS_SOURCE_ID,
                refl_heights_m_asl=REFLECTIVITY_HEIGHTS_M_ASL))

        self.assertTrue(this_dictionary == FIELD_TO_HEIGHTS_DICT_MRMS_M_ASL)

    def test_fields_and_refl_heights_to_pairs_myrorss(self):
        """Ensures correct output from fields_and_refl_heights_to_pairs.

        In this case, data source is MYRORSS.
        """

        this_field_name_by_pair, this_height_by_pair_m_asl = (
            myrorss_and_mrms_utils.fields_and_refl_heights_to_pairs(
                field_names=FIELD_NAMES,
                data_source=radar_utils.MYRORSS_SOURCE_ID,
                refl_heights_m_asl=REFLECTIVITY_HEIGHTS_M_ASL))

        self.assertTrue(this_field_name_by_pair == FIELD_NAME_BY_PAIR)
        self.assertTrue(numpy.array_equal(
            this_height_by_pair_m_asl, HEIGHT_BY_PAIR_MYRORSS_M_ASL))

    def test_fields_and_refl_heights_to_pairs_mrms(self):
        """Ensures correct output from fields_and_refl_heights_to_pairs.

        In this case, data source is MRMS.
        """

        this_field_name_by_pair, this_height_by_pair_m_asl = (
            myrorss_and_mrms_utils.fields_and_refl_heights_to_pairs(
                field_names=FIELD_NAMES,
                data_source=radar_utils.MRMS_SOURCE_ID,
                refl_heights_m_asl=REFLECTIVITY_HEIGHTS_M_ASL))

        self.assertTrue(this_field_name_by_pair == FIELD_NAME_BY_PAIR)
        self.assertTrue(numpy.array_equal(
            this_height_by_pair_m_asl, HEIGHT_BY_PAIR_MRMS_M_ASL))


if __name__ == '__main__':
    unittest.main()
