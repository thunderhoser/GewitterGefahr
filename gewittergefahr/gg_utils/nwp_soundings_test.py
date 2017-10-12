"""Unit tests for nwp_soundings.py."""

import copy
import unittest
import numpy
import pandas
from gewittergefahr.gg_utils import rap_model_utils
from gewittergefahr.gg_utils import nwp_soundings

FAKE_MODEL_NAME = 'ecmwf'
MINIMUM_PRESSURE_MB = 950.

RAP_SOUNDING_COLUMNS = [
    'temperature_kelvins_950mb', 'temperature_kelvins_975mb',
    'temperature_kelvins_1000mb', rap_model_utils.LOWEST_TEMPERATURE_COLUMN,
    'relative_humidity_950mb', 'relative_humidity_975mb',
    'relative_humidity_1000mb', rap_model_utils.LOWEST_RH_COLUMN,
    'geopotential_height_metres_950mb', 'geopotential_height_metres_975mb',
    'geopotential_height_metres_1000mb', rap_model_utils.LOWEST_GPH_COLUMN,
    'u_wind_m_s01_950mb', 'u_wind_m_s01_975mb', 'u_wind_m_s01_1000mb',
    rap_model_utils.LOWEST_U_WIND_COLUMN,
    'v_wind_m_s01_950mb', 'v_wind_m_s01_975mb', 'v_wind_m_s01_1000mb',
    rap_model_utils.LOWEST_V_WIND_COLUMN]

RAP_SOUNDING_COLUMNS_ORIG = [
    'TMP:950 mb', 'TMP:975 mb', 'TMP:1000 mb',
    rap_model_utils.LOWEST_TEMPERATURE_COLUMN_ORIG,
    'RH:950 mb', 'RH:975 mb', 'RH:1000 mb',
    rap_model_utils.LOWEST_RH_COLUMN_ORIG,
    'HGT:950 mb', 'HGT:975 mb', 'HGT:1000 mb',
    rap_model_utils.LOWEST_GPH_COLUMN_ORIG,
    'UGRD:950 mb', 'UGRD:975 mb', 'UGRD:1000 mb',
    rap_model_utils.LOWEST_U_WIND_COLUMN_ORIG,
    'VGRD:950 mb', 'VGRD:975 mb', 'VGRD:1000 mb',
    rap_model_utils.LOWEST_V_WIND_COLUMN_ORIG]

NARR_SOUNDING_COLUMNS = [
    'temperature_kelvins_950mb', 'temperature_kelvins_975mb',
    'temperature_kelvins_1000mb', 'specific_humidity_950mb',
    'specific_humidity_975mb', 'specific_humidity_1000mb',
    'geopotential_height_metres_950mb', 'geopotential_height_metres_975mb',
    'geopotential_height_metres_1000mb', 'u_wind_m_s01_950mb',
    'u_wind_m_s01_975mb', 'u_wind_m_s01_1000mb', 'v_wind_m_s01_950mb',
    'v_wind_m_s01_975mb', 'v_wind_m_s01_1000mb']

NARR_SOUNDING_COLUMNS_ORIG = [
    'TMP:950 mb', 'TMP:975 mb', 'TMP:1000 mb',
    'SPFH:950 mb', 'SPFH:975 mb', 'SPFH:1000 mb',
    'HGT:950 mb', 'HGT:975 mb', 'HGT:1000 mb',
    'UGRD:950 mb', 'UGRD:975 mb', 'UGRD:1000 mb',
    'VGRD:950 mb', 'VGRD:975 mb', 'VGRD:1000 mb']

SURFACE_ROW = 3
THESE_PRESSURES_MB = numpy.array(
    [1000., 965., 962., 936.87, 936.8669, 925., 904.95, 889., 873.9, 877.])
THESE_HEIGHTS_M_ASL = numpy.array(
    [34., 350., 377.51, 610., 610.0001, 722., 914., 1069.78, 1219., 1188.26])
THESE_TEMPERATURES_DEG_C = numpy.array(
    [30., 27.8, 27.4, 25.51, 25.51, 24.6, 23.05, 21.8, 22.02, 22.2])
THESE_DEWPOINTS_DEG_C = numpy.array(
    [24.5, 23.8, 22.8, 21.72, 21.72, 21.2, 20.43, 19.8, 16.98, 17.3])
THESE_U_WINDS_KT = numpy.array(
    [-8.45, -11.5, -13.2, -17.2, -17.2, -17., -14.36, -11.65, -4.36, -8.34])
THESE_V_WINDS_KT = numpy.array(
    [18.13, 19.92, 22.86, 24.57, 24.57, 29.44, 39.47, 43.47, 49.81, 47.27])

THIS_SOUNDING_DICT = {
    nwp_soundings.PRESSURE_COLUMN_FOR_SOUNDING_INDICES: THESE_PRESSURES_MB,
    nwp_soundings.HEIGHT_COLUMN_FOR_SOUNDING_INDICES: THESE_HEIGHTS_M_ASL,
    nwp_soundings.TEMPERATURE_COLUMN_FOR_SOUNDING_INDICES:
        THESE_TEMPERATURES_DEG_C,
    nwp_soundings.DEWPOINT_COLUMN_FOR_SOUNDING_INDICES: THESE_DEWPOINTS_DEG_C,
    nwp_soundings.U_WIND_COLUMN_FOR_SOUNDING_INDICES: THESE_U_WINDS_KT,
    nwp_soundings.V_WIND_COLUMN_FOR_SOUNDING_INDICES: THESE_V_WINDS_KT
}
SOUNDING_TABLE_ORIG = pandas.DataFrame.from_dict(THIS_SOUNDING_DICT)

SENTINEL_VALUE = nwp_soundings.SENTINEL_VALUE_FOR_SOUNDING_INDICES
THESE_TEMPERATURES_DEG_C = numpy.array(
    [SENTINEL_VALUE, SENTINEL_VALUE, SENTINEL_VALUE,
     25.51, 25.51, 24.6, 23.05, 21.8, 22.02, 22.2])
THESE_DEWPOINTS_DEG_C = numpy.array(
    [SENTINEL_VALUE, SENTINEL_VALUE, SENTINEL_VALUE,
     21.72, 21.72, 21.2, 20.43, 19.8, 16.98, 17.3])
THESE_U_WINDS_KT = numpy.array(
    [SENTINEL_VALUE, SENTINEL_VALUE, SENTINEL_VALUE,
     -17.2, -17.2, -17., -14.36, -11.65, -4.36, -8.34])
THESE_V_WINDS_KT = numpy.array(
    [SENTINEL_VALUE, SENTINEL_VALUE, SENTINEL_VALUE,
     24.57, 24.57, 29.44, 39.47, 43.47, 49.81, 47.27])

THIS_SOUNDING_DICT = {
    nwp_soundings.PRESSURE_COLUMN_FOR_SOUNDING_INDICES: THESE_PRESSURES_MB,
    nwp_soundings.HEIGHT_COLUMN_FOR_SOUNDING_INDICES: THESE_HEIGHTS_M_ASL,
    nwp_soundings.TEMPERATURE_COLUMN_FOR_SOUNDING_INDICES:
        THESE_TEMPERATURES_DEG_C,
    nwp_soundings.DEWPOINT_COLUMN_FOR_SOUNDING_INDICES: THESE_DEWPOINTS_DEG_C,
    nwp_soundings.U_WIND_COLUMN_FOR_SOUNDING_INDICES: THESE_U_WINDS_KT,
    nwp_soundings.V_WIND_COLUMN_FOR_SOUNDING_INDICES: THESE_V_WINDS_KT
}
SOUNDING_TABLE_WITH_SENTINELS = pandas.DataFrame.from_dict(THIS_SOUNDING_DICT)

THESE_PRESSURES_MB = numpy.array(
    [1000., 965., 962., 936.87, 936.8669, 925., 904.95, 889., 877., 873.9])
THESE_HEIGHTS_M_ASL = numpy.array(
    [34., 350., 377.51, 610., 610.0001, 722., 914., 1069.78, 1188.26, 1219.])
THESE_TEMPERATURES_DEG_C = numpy.array(
    [SENTINEL_VALUE, SENTINEL_VALUE, SENTINEL_VALUE,
     25.51, 25.51, 24.6, 23.05, 21.8, 22.2, 22.02])
THESE_DEWPOINTS_DEG_C = numpy.array(
    [SENTINEL_VALUE, SENTINEL_VALUE, SENTINEL_VALUE,
     21.72, 21.72, 21.2, 20.43, 19.8, 17.3, 16.98])
THESE_U_WINDS_KT = numpy.array(
    [SENTINEL_VALUE, SENTINEL_VALUE, SENTINEL_VALUE,
     -17.2, -17.2, -17., -14.36, -11.65, -8.34, -4.36])
THESE_V_WINDS_KT = numpy.array(
    [SENTINEL_VALUE, SENTINEL_VALUE, SENTINEL_VALUE,
     24.57, 24.57, 29.44, 39.47, 43.47, 47.27, 49.81])

THIS_SOUNDING_DICT = {
    nwp_soundings.PRESSURE_COLUMN_FOR_SOUNDING_INDICES: THESE_PRESSURES_MB,
    nwp_soundings.HEIGHT_COLUMN_FOR_SOUNDING_INDICES: THESE_HEIGHTS_M_ASL,
    nwp_soundings.TEMPERATURE_COLUMN_FOR_SOUNDING_INDICES:
        THESE_TEMPERATURES_DEG_C,
    nwp_soundings.DEWPOINT_COLUMN_FOR_SOUNDING_INDICES: THESE_DEWPOINTS_DEG_C,
    nwp_soundings.U_WIND_COLUMN_FOR_SOUNDING_INDICES: THESE_U_WINDS_KT,
    nwp_soundings.V_WIND_COLUMN_FOR_SOUNDING_INDICES: THESE_V_WINDS_KT
}
SOUNDING_TABLE_WITH_SENTINELS_SORTED = pandas.DataFrame.from_dict(
    THIS_SOUNDING_DICT)

THESE_PRESSURES_MB = numpy.array(
    [1000., 965., 962., 936.87, 936.8669, 925., 904.95, 889., 877., 873.9])
THESE_HEIGHTS_M_ASL = numpy.array(
    [34., 350., 377.51, 610., 610.0001, 722., 914., 1069.78, 1188.26, 1219.])
THESE_TEMPERATURES_DEG_C = numpy.array(
    [SENTINEL_VALUE, SENTINEL_VALUE, SENTINEL_VALUE, SENTINEL_VALUE,
     25.51, 24.6, 23.05, 21.8, 22.2, 22.02])
THESE_DEWPOINTS_DEG_C = numpy.array(
    [SENTINEL_VALUE, SENTINEL_VALUE, SENTINEL_VALUE, SENTINEL_VALUE,
     21.72, 21.2, 20.43, 19.8, 17.3, 16.98])
THESE_U_WINDS_KT = numpy.array(
    [SENTINEL_VALUE, SENTINEL_VALUE, SENTINEL_VALUE, SENTINEL_VALUE,
     -17.2, -17., -14.36, -11.65, -8.34, -4.36])
THESE_V_WINDS_KT = numpy.array(
    [SENTINEL_VALUE, SENTINEL_VALUE, SENTINEL_VALUE, SENTINEL_VALUE,
     24.57, 29.44, 39.47, 43.47, 47.27, 49.81])

THIS_SOUNDING_DICT = {
    nwp_soundings.PRESSURE_COLUMN_FOR_SOUNDING_INDICES: THESE_PRESSURES_MB,
    nwp_soundings.HEIGHT_COLUMN_FOR_SOUNDING_INDICES: THESE_HEIGHTS_M_ASL,
    nwp_soundings.TEMPERATURE_COLUMN_FOR_SOUNDING_INDICES:
        THESE_TEMPERATURES_DEG_C,
    nwp_soundings.DEWPOINT_COLUMN_FOR_SOUNDING_INDICES: THESE_DEWPOINTS_DEG_C,
    nwp_soundings.U_WIND_COLUMN_FOR_SOUNDING_INDICES: THESE_U_WINDS_KT,
    nwp_soundings.V_WIND_COLUMN_FOR_SOUNDING_INDICES: THESE_V_WINDS_KT
}
SOUNDING_TABLE_NO_REDUNDANT = pandas.DataFrame.from_dict(THIS_SOUNDING_DICT)

THESE_PRESSURES_MB = numpy.array(
    [1000., 936.87, 936.8669, 925., 904.95, 889., 877., 873.9])
THESE_HEIGHTS_M_ASL = numpy.array(
    [34., 610., 610.0001, 722., 914., 1069.78, 1188.26, 1219.])
THESE_TEMPERATURES_DEG_C = numpy.array(
    [SENTINEL_VALUE, SENTINEL_VALUE, 25.51, 24.6, 23.05, 21.8, 22.2, 22.02])
THESE_DEWPOINTS_DEG_C = numpy.array(
    [SENTINEL_VALUE, SENTINEL_VALUE, 21.72, 21.2, 20.43, 19.8, 17.3, 16.98])
THESE_U_WINDS_KT = numpy.array(
    [SENTINEL_VALUE, SENTINEL_VALUE, -17.2, -17., -14.36, -11.65, -8.34, -4.36])
THESE_V_WINDS_KT = numpy.array(
    [SENTINEL_VALUE, SENTINEL_VALUE, 24.57, 29.44, 39.47, 43.47, 47.27, 49.81])

THIS_SOUNDING_DICT = {
    nwp_soundings.PRESSURE_COLUMN_FOR_SOUNDING_INDICES: THESE_PRESSURES_MB,
    nwp_soundings.HEIGHT_COLUMN_FOR_SOUNDING_INDICES: THESE_HEIGHTS_M_ASL,
    nwp_soundings.TEMPERATURE_COLUMN_FOR_SOUNDING_INDICES:
        THESE_TEMPERATURES_DEG_C,
    nwp_soundings.DEWPOINT_COLUMN_FOR_SOUNDING_INDICES: THESE_DEWPOINTS_DEG_C,
    nwp_soundings.U_WIND_COLUMN_FOR_SOUNDING_INDICES: THESE_U_WINDS_KT,
    nwp_soundings.V_WIND_COLUMN_FOR_SOUNDING_INDICES: THESE_V_WINDS_KT
}
SOUNDING_TABLE_NO_SUBSURFACE = pandas.DataFrame.from_dict(THIS_SOUNDING_DICT)


class NwpSoundingsTests(unittest.TestCase):
    """Each method is a unit test for nwp_soundings.py."""

    def test_check_model_name_rap(self):
        """Ensures correct output from _check_model_name.

        In this case, model is RAP.
        """

        nwp_soundings._check_model_name(nwp_soundings.RAP_MODEL_NAME)

    def test_check_model_name_narr(self):
        """Ensures correct output from _check_model_name.

        In this case, model is NARR.
        """

        nwp_soundings._check_model_name(nwp_soundings.NARR_MODEL_NAME)

    def test_check_model_name_fake(self):
        """Ensures correct output from _check_model_name.

        In this case, model name is not recognized.
        """

        with self.assertRaises(ValueError):
            nwp_soundings._check_model_name(FAKE_MODEL_NAME)

    def test_get_sounding_columns_rap(self):
        """Ensures correct output from _get_sounding_columns.

        In this case, model is RAP.
        """

        (these_sounding_columns,
         these_sounding_columns_orig) = nwp_soundings._get_sounding_columns(
             nwp_soundings.RAP_MODEL_NAME,
             minimum_pressure_mb=MINIMUM_PRESSURE_MB)

        self.assertTrue(these_sounding_columns == RAP_SOUNDING_COLUMNS)
        self.assertTrue(
            these_sounding_columns_orig == RAP_SOUNDING_COLUMNS_ORIG)

    def test_get_sounding_columns_narr(self):
        """Ensures correct output from _get_sounding_columns.

        In this case, model is NARR.
        """

        (these_sounding_columns,
         these_sounding_columns_orig) = nwp_soundings._get_sounding_columns(
             nwp_soundings.NARR_MODEL_NAME,
             minimum_pressure_mb=MINIMUM_PRESSURE_MB)

        self.assertTrue(these_sounding_columns == NARR_SOUNDING_COLUMNS)
        self.assertTrue(
            these_sounding_columns_orig == NARR_SOUNDING_COLUMNS_ORIG)

    def test_remove_subsurface_sounding_data_keep_rows(self):
        """Ensures correct output from _remove_subsurface_sounding_data.

        In this case, remove_subsurface_rows = False.
        """

        this_sounding_table = nwp_soundings._remove_subsurface_sounding_data(
            SOUNDING_TABLE_ORIG, surface_row=SURFACE_ROW,
            remove_subsurface_rows=False)
        self.assertTrue(
            this_sounding_table.equals(SOUNDING_TABLE_WITH_SENTINELS))

    def test_sort_sounding_table_by_height(self):
        """Ensures correct output from _sort_sounding_table_by_height."""

        this_input_table = copy.deepcopy(SOUNDING_TABLE_WITH_SENTINELS_SORTED)
        this_sounding_table = nwp_soundings._sort_sounding_table_by_height(
            this_input_table)

        self.assertTrue(
            this_sounding_table.equals(SOUNDING_TABLE_WITH_SENTINELS_SORTED))

    def test_remove_redundant_sounding_data(self):
        """Ensures correct output from _remove_redundant_sounding_data."""

        this_input_table = copy.deepcopy(SOUNDING_TABLE_WITH_SENTINELS_SORTED)
        this_sounding_table = nwp_soundings._remove_redundant_sounding_data(
            this_input_table, SURFACE_ROW)

        self.assertTrue(this_sounding_table.equals(SOUNDING_TABLE_NO_REDUNDANT))

    def test_remove_subsurface_sounding_data_remove_rows(self):
        """Ensures correct output from _remove_subsurface_sounding_data.

        In this case, remove_subsurface_rows = True.
        """

        this_sounding_table = nwp_soundings._remove_subsurface_sounding_data(
            SOUNDING_TABLE_NO_REDUNDANT, surface_row=SURFACE_ROW,
            remove_subsurface_rows=True)
        self.assertTrue(
            this_sounding_table.equals(SOUNDING_TABLE_NO_SUBSURFACE))


if __name__ == '__main__':
    unittest.main()
