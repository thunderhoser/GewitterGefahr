"""Unit tests for grib_io.py."""

import copy
import unittest
import numpy
from gewittergefahr.gg_io import grib_io

TOLERANCE = 1e-6

GRIB1_FILE_NAME = 'foo.grb'
GRIB2_FILE_NAME = 'foo.grb2'
NON_GRIB_FILE_NAME = 'foo.txt'

SPECIFIC_HUMIDITY_NAME_GRIB1 = 'SPFH:500 mb'
SPECIFIC_HUMIDITY_NAME_GRIB2 = 'SPFH:500 mb'
HEIGHT_NAME_GRIB1 = 'HGT:sfc'
HEIGHT_NAME_GRIB2 = 'HGT:surface'
TEMPERATURE_NAME_GRIB1 = 'TMP:2 m above gnd'
TEMPERATURE_NAME_GRIB2 = 'TMP:2 m above ground'

SENTINEL_VALUE = 9.999e20
DATA_MATRIX_WITH_SENTINELS = numpy.array(
    [[999.25, 1000., 1001., 1001.5, SENTINEL_VALUE],
     [999.5, 1000.75, 1002., 1002.2, SENTINEL_VALUE],
     [999.9, 1001.1, 1001.7, 1002.3, 1003.],
     [SENTINEL_VALUE, 1001.5, 1001.6, 1002.1, 1002.5],
     [SENTINEL_VALUE, SENTINEL_VALUE, 1002., 1002., 1003.3]])

DATA_MATRIX_NO_SENTINELS = numpy.array(
    [[999.25, 1000., 1001., 1001.5, numpy.nan],
     [999.5, 1000.75, 1002., 1002.2, numpy.nan],
     [999.9, 1001.1, 1001.7, 1002.3, 1003.],
     [numpy.nan, 1001.5, 1001.6, 1002.1, 1002.5],
     [numpy.nan, numpy.nan, 1002., 1002., 1003.3]])

U_WIND_NAME_GRIB1 = 'UGRD:500 mb'
V_WIND_NAME_GRIB1 = 'VGRD:500 mb'
NON_WIND_NAME_GRIB1 = 'TMP:500 mb'

NON_GRIB_FILE_TYPE = 'text'


class GribIoTests(unittest.TestCase):
    """Each method is a unit test for grib_io.py."""

    def test_field_name_grib1_to_grib2_humidity(self):
        """Ensures correct output from _field_name_grib1_to_grib2.

        In this case the field is 500-mb specific humidity.
        """

        self.assertEquals(
            grib_io._field_name_grib1_to_grib2(SPECIFIC_HUMIDITY_NAME_GRIB1),
            SPECIFIC_HUMIDITY_NAME_GRIB2)

    def test_field_name_grib1_to_grib2_height(self):
        """Ensures correct output from _field_name_grib1_to_grib2.

        In this case the field is surface geopotential height.
        """

        self.assertEquals(
            grib_io._field_name_grib1_to_grib2(HEIGHT_NAME_GRIB1),
            HEIGHT_NAME_GRIB2)

    def test_field_name_grib1_to_grib2_temperature(self):
        """Ensures correct output from _field_name_grib1_to_grib2.

        In this case the field is 2-metre temperature.
        """

        self.assertEquals(
            grib_io._field_name_grib1_to_grib2(TEMPERATURE_NAME_GRIB1),
            TEMPERATURE_NAME_GRIB2)

    def test_sentinel_value_to_nan_defined(self):
        """Ensures correct output from _sentinel_value_to_nan.

        In this case the sentinel value is defined.
        """

        this_data_matrix = grib_io._sentinel_value_to_nan(
            data_matrix=copy.deepcopy(DATA_MATRIX_WITH_SENTINELS),
            sentinel_value=SENTINEL_VALUE)

        self.assertTrue(numpy.allclose(
            this_data_matrix, DATA_MATRIX_NO_SENTINELS, atol=TOLERANCE,
            equal_nan=True))

    def test_sentinel_value_to_nan_none(self):
        """Ensures correct output from _sentinel_value_to_nan.

        In this case the sentinel value is None.
        """

        this_data_matrix = grib_io._sentinel_value_to_nan(
            data_matrix=copy.deepcopy(DATA_MATRIX_WITH_SENTINELS),
            sentinel_value=None)

        self.assertTrue(numpy.allclose(
            this_data_matrix, DATA_MATRIX_WITH_SENTINELS, atol=TOLERANCE))

    def test_check_file_type_grib1(self):
        """Ensures correct output from check_file_type.

        Here, file type is grib1.
        """

        grib_io.check_file_type(grib_io.GRIB1_FILE_TYPE)

    def test_check_file_type_grib2(self):
        """Ensures correct output from check_file_type.

        Here, file type is grib2.
        """

        grib_io.check_file_type(grib_io.GRIB2_FILE_TYPE)

    def test_check_file_type_invalid(self):
        """Ensures correct output from check_file_type.

        Here, file type is invalid.
        """

        with self.assertRaises(ValueError):
            grib_io.check_file_type(NON_GRIB_FILE_TYPE)

    def test_file_name_to_type_grib1(self):
        """Ensures correct output from file_name_to_type.

        Here, file type is grib1.
        """

        self.assertEquals(grib_io.file_name_to_type(GRIB1_FILE_NAME),
                          grib_io.GRIB1_FILE_TYPE)

    def test_file_name_to_type_grib2(self):
        """Ensures correct output from file_name_to_type.

        Here, file type is grib2.
        """

        self.assertEquals(grib_io.file_name_to_type(GRIB2_FILE_NAME),
                          grib_io.GRIB2_FILE_TYPE)

    def test_file_name_to_type_invalid(self):
        """Ensures correct output from file_name_to_type.

        Here, file type is invalid.
        """

        with self.assertRaises(ValueError):
            grib_io.file_name_to_type(NON_GRIB_FILE_NAME)

    def test_is_u_wind_field_true(self):
        """Ensures correct output from is_u_wind_field.

        In this case the input field is u-wind, so the answer is True.
        """

        self.assertTrue(grib_io.is_u_wind_field(U_WIND_NAME_GRIB1))

    def test_is_u_wind_field_v_wind(self):
        """Ensures correct output from is_u_wind_field.

        In this case the input field is v-wind, so the answer is False.
        """

        self.assertFalse(grib_io.is_u_wind_field(V_WIND_NAME_GRIB1))

    def test_is_u_wind_field_non_wind(self):
        """Ensures correct output from is_u_wind_field.

        In this case the input field is not wind-related, so the answer is False.
        """

        self.assertFalse(grib_io.is_u_wind_field(NON_WIND_NAME_GRIB1))

    def test_is_v_wind_field_true(self):
        """Ensures correct output from is_v_wind_field.

        In this case the input field is v-wind, so the answer is True.
        """

        self.assertTrue(grib_io.is_v_wind_field(V_WIND_NAME_GRIB1))

    def test_is_v_wind_field_u_wind(self):
        """Ensures correct output from is_v_wind_field.

        In this case the input field is u-wind, so the answer is False.
        """

        self.assertFalse(grib_io.is_v_wind_field(U_WIND_NAME_GRIB1))

    def test_is_v_wind_field_non_wind(self):
        """Ensures correct output from is_v_wind_field.

        In this case the input field is not wind-related, so the answer is
        False.
        """

        self.assertFalse(grib_io.is_v_wind_field(NON_WIND_NAME_GRIB1))

    def test_switch_uv_in_field_name_input_u(self):
        """Ensures correct output from switch_uv_in_field_name.

        In this case the input field is u-wind.
        """

        self.assertTrue(grib_io.switch_uv_in_field_name(U_WIND_NAME_GRIB1) ==
                        V_WIND_NAME_GRIB1)

    def test_switch_uv_in_field_name_input_v(self):
        """Ensures correct output from switch_uv_in_field_name.

        In this case the input field is v-wind.
        """

        self.assertTrue(grib_io.switch_uv_in_field_name(V_WIND_NAME_GRIB1) ==
                        U_WIND_NAME_GRIB1)

    def test_switch_uv_in_field_name_input_non_wind(self):
        """Ensures correct output from switch_uv_in_field_name.

        In this case the input field is not wind-related, so it should be
        unchanged.
        """

        self.assertTrue(grib_io.switch_uv_in_field_name(NON_WIND_NAME_GRIB1) ==
                        NON_WIND_NAME_GRIB1)

    def test_file_type_to_extension_grib1(self):
        """Ensures correct output from file_type_to_extension.

        Here, file type is grib1.
        """

        self.assertEquals(
            grib_io.file_type_to_extension(grib_io.GRIB1_FILE_TYPE),
            grib_io.GRIB1_FILE_EXTENSION)

    def test_file_type_to_extension_grib2(self):
        """Ensures correct output from file_type_to_extension.

        Here, file type is grib2.
        """

        self.assertEquals(
            grib_io.file_type_to_extension(grib_io.GRIB2_FILE_TYPE),
            grib_io.GRIB2_FILE_EXTENSION)

    def test_file_type_to_extension_invalid(self):
        """Ensures correct output from file_type_to_extension.

        Here, file type is invalid.
        """

        with self.assertRaises(ValueError):
            grib_io.file_type_to_extension(NON_GRIB_FILE_TYPE)


if __name__ == '__main__':
    unittest.main()
