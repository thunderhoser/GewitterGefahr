"""Unit tests for grib_io.py."""

import copy
import unittest
import numpy
from gewittergefahr.gg_io import grib_io

GRIB1_FILE_NAME = 'foo' + grib_io.GRIB1_FILE_EXTENSION
GRIB2_FILE_NAME = 'foo' + grib_io.GRIB2_FILE_EXTENSION
NON_GRIB_FILE_NAME = 'foo.txt'

GRIB1_HUMIDITY_NAME = 'SPFH:500 mb'
GRIB2_HUMIDITY_NAME = 'SPFH:500 mb'
GRIB1_HEIGHT_NAME = 'HGT:sfc'
GRIB2_HEIGHT_NAME = 'HGT:surface'
GRIB1_TEMPERATURE_NAME = 'TMP:2 m above gnd'
GRIB2_TEMPERATURE_NAME = 'TMP:2 m above ground'

TOLERANCE = 1e-6
SENTINEL_VALUE = 9.999e20
DATA_MATRIX_WITH_SENTINELS = (
    numpy.array([[999.25, 1000., 1001., 1001.5, SENTINEL_VALUE],
                 [999.5, 1000.75, 1002., 1002.2, SENTINEL_VALUE],
                 [999.9, 1001.1, 1001.7, 1002.3, 1003.],
                 [SENTINEL_VALUE, 1001.5, 1001.6, 1002.1, 1002.5],
                 [SENTINEL_VALUE, SENTINEL_VALUE, 1002., 1002., 1003.3]]))

EXPECTED_DATA_MATRIX_NO_SENTINELS = (
    numpy.array([[999.25, 1000., 1001., 1001.5, numpy.nan],
                 [999.5, 1000.75, 1002., 1002.2, numpy.nan],
                 [999.9, 1001.1, 1001.7, 1002.3, 1003.],
                 [numpy.nan, 1001.5, 1001.6, 1002.1, 1002.5],
                 [numpy.nan, numpy.nan, 1002., 1002., 1003.3]]))

U_WIND_NAME = 'UGRD:500 mb'
V_WIND_NAME = 'VGRD:500 mb'
NON_WIND_NAME = 'TMP:500 mb'

NON_GRIB_FILE_TYPE = 'text'


class GribIoTests(unittest.TestCase):
    """Each method is a unit test for grib_io.py."""

    def test_get_file_type_grib1(self):
        """Ensures correct output from _get_file_type.

        In this case the file type is grib1.
        """

        self.assertEquals(
            grib_io._get_file_type(GRIB1_FILE_NAME), grib_io.GRIB1_FILE_TYPE)

    def test_get_file_type_grib2(self):
        """Ensures correct output from _get_file_type.

        In this case the file type is grib2.
        """

        self.assertEquals(
            grib_io._get_file_type(GRIB2_FILE_NAME), grib_io.GRIB2_FILE_TYPE)

    def test_get_file_type_non_grib(self):
        """Ensures correct output from _get_file_type.

        In this case the file type is non-grib.
        """

        with self.assertRaises(ValueError):
            grib_io._get_file_type(NON_GRIB_FILE_NAME)

    def test_field_name_grib1_to_grib2_humidity(self):
        """Ensures correct output from _field_name_grib1_to_grib2.

        In this case the field is 500-mb specific humidity.
        """

        self.assertEquals(
            grib_io._field_name_grib1_to_grib2(GRIB1_HUMIDITY_NAME),
            GRIB2_HUMIDITY_NAME)

    def test_field_name_grib1_to_grib2_height(self):
        """Ensures correct output from _field_name_grib1_to_grib2.

        In this case the field is surface geopotential height.
        """

        self.assertEquals(
            grib_io._field_name_grib1_to_grib2(GRIB1_HEIGHT_NAME),
            GRIB2_HEIGHT_NAME)

    def test_field_name_grib1_to_grib2_temperature(self):
        """Ensures correct output from _field_name_grib1_to_grib2.

        In this case the field is 2-metre temperature.
        """

        self.assertEquals(
            grib_io._field_name_grib1_to_grib2(GRIB1_TEMPERATURE_NAME),
            GRIB2_TEMPERATURE_NAME)

    def test_replace_sentinels_with_nan_sentinel_value_defined(self):
        """Ensures correct output from _replace_sentinels_with_nan.

        In this case the sentinel value is defined (not None).
        """

        this_data_matrix = grib_io._replace_sentinels_with_nan(
            copy.deepcopy(DATA_MATRIX_WITH_SENTINELS), SENTINEL_VALUE)

        self.assertTrue(
            numpy.allclose(this_data_matrix, EXPECTED_DATA_MATRIX_NO_SENTINELS,
                           atol=TOLERANCE, equal_nan=True))

    def test_replace_sentinels_with_nan_sentinel_value_none(self):
        """Ensures correct output from _replace_sentinels_with_nan.

        In this case the sentinel value is None.
        """

        this_data_matrix = grib_io._replace_sentinels_with_nan(
            copy.deepcopy(DATA_MATRIX_WITH_SENTINELS), None)

        self.assertTrue(
            numpy.allclose(this_data_matrix, DATA_MATRIX_WITH_SENTINELS,
                           atol=TOLERANCE))

    def test_is_u_wind_field_true(self):
        """Ensures correct output from is_u_wind_field.

        In this case the answer is yes.
        """

        self.assertTrue(grib_io.is_u_wind_field(U_WIND_NAME))

    def test_is_u_wind_field_v_wind(self):
        """Ensures correct output from is_u_wind_field.

        In this case the answer is no.
        """

        self.assertFalse(grib_io.is_u_wind_field(V_WIND_NAME))

    def test_is_u_wind_field_non_wind(self):
        """Ensures correct output from is_u_wind_field.

        In this case the answer is no.
        """

        self.assertFalse(grib_io.is_u_wind_field(NON_WIND_NAME))

    def test_is_v_wind_field_true(self):
        """Ensures correct output from is_v_wind_field.

        In this case the answer is yes.
        """

        self.assertTrue(grib_io.is_v_wind_field(V_WIND_NAME))

    def test_is_v_wind_field_u_wind(self):
        """Ensures correct output from is_v_wind_field.

        In this case the answer is no.
        """

        self.assertFalse(grib_io.is_v_wind_field(U_WIND_NAME))

    def test_is_v_wind_field_non_wind(self):
        """Ensures correct output from is_v_wind_field.

        In this case the answer is no.
        """

        self.assertFalse(grib_io.is_v_wind_field(NON_WIND_NAME))

    def test_field_name_switch_u_and_v_input_u(self):
        """Ensures correct output from field_name_switch_u_and_v.

        In this case, input field is u-wind.
        """

        this_v_wind_name = grib_io.field_name_switch_u_and_v(U_WIND_NAME)
        self.assertTrue(this_v_wind_name == V_WIND_NAME)

    def test_field_name_switch_u_and_v_input_v(self):
        """Ensures correct output from field_name_switch_u_and_v.

        In this case, input field is v-wind.
        """

        this_u_wind_name = grib_io.field_name_switch_u_and_v(V_WIND_NAME)
        self.assertTrue(this_u_wind_name == U_WIND_NAME)

    def test_field_name_switch_u_and_v_input_neither(self):
        """Ensures correct output from field_name_switch_u_and_v.

        In this case, input field is non-wind-related.
        """

        this_field_name = grib_io.field_name_switch_u_and_v(NON_WIND_NAME)
        self.assertTrue(this_field_name == NON_WIND_NAME)

    def test_file_type_to_extension_grib1(self):
        """Ensures correct output from file_type_to_extension.

        In this case the file type is grib1.
        """

        self.assertEquals(
            grib_io.file_type_to_extension(grib_io.GRIB1_FILE_TYPE),
            grib_io.GRIB1_FILE_EXTENSION)

    def test_file_type_to_extension_grib2(self):
        """Ensures correct output from file_type_to_extension.

        In this case the file type is grib2.
        """

        self.assertEquals(
            grib_io.file_type_to_extension(grib_io.GRIB2_FILE_TYPE),
            grib_io.GRIB2_FILE_EXTENSION)

    def test_file_type_to_extension_non_grib(self):
        """Ensures correct output from file_type_to_extension.

        In this case the file type is non-grib.
        """

        with self.assertRaises(ValueError):
            grib_io.file_type_to_extension(NON_GRIB_FILE_TYPE)


if __name__ == '__main__':
    unittest.main()
