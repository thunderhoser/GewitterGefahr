"""Unit tests for soundings_only.py"""

import unittest
import numpy
import pandas
from gewittergefahr.gg_utils import soundings_only
from gewittergefahr.gg_utils import nwp_model_utils

# The following constants are used to test get_nwp_fields_for_sounding.
MINIMUM_PRESSURE_MB = 950.

(THIS_LOWEST_HEIGHT_NAME, THIS_LOWEST_HEIGHT_NAME_GRIB1
 ) = nwp_model_utils.get_lowest_height_name(nwp_model_utils.RAP_MODEL_NAME)
(THIS_LOWEST_TEMP_NAME, THIS_LOWEST_TEMP_NAME_GRIB1
 ) = nwp_model_utils.get_lowest_temperature_name(nwp_model_utils.RAP_MODEL_NAME)
(THIS_LOWEST_HUMIDITY_NAME, THIS_LOWEST_HUMIDITY_NAME_GRIB1
 ) = nwp_model_utils.get_lowest_humidity_name(nwp_model_utils.RAP_MODEL_NAME)
(THIS_LOWEST_U_WIND_NAME, THIS_LOWEST_U_WIND_NAME_GRIB1
 ) = nwp_model_utils.get_lowest_u_wind_name(nwp_model_utils.RAP_MODEL_NAME)
(THIS_LOWEST_V_WIND_NAME, THIS_LOWEST_V_WIND_NAME_GRIB1
 ) = nwp_model_utils.get_lowest_v_wind_name(nwp_model_utils.RAP_MODEL_NAME)
(THIS_LOWEST_PRESSURE_NAME, THIS_LOWEST_PRESSURE_NAME_GRIB1
 ) = nwp_model_utils.get_lowest_pressure_name(nwp_model_utils.RAP_MODEL_NAME)

SOUNDING_FIELD_NAMES_RAP_WITH_SURFACE = [
    'geopotential_height_metres_950mb', 'geopotential_height_metres_975mb',
    'geopotential_height_metres_1000mb', THIS_LOWEST_HEIGHT_NAME,
    'temperature_kelvins_950mb', 'temperature_kelvins_975mb',
    'temperature_kelvins_1000mb', THIS_LOWEST_TEMP_NAME,
    'relative_humidity_percent_950mb', 'relative_humidity_percent_975mb',
    'relative_humidity_percent_1000mb', THIS_LOWEST_HUMIDITY_NAME,
    'u_wind_m_s01_950mb', 'u_wind_m_s01_975mb', 'u_wind_m_s01_1000mb',
    THIS_LOWEST_U_WIND_NAME,
    'v_wind_m_s01_950mb', 'v_wind_m_s01_975mb', 'v_wind_m_s01_1000mb',
    THIS_LOWEST_V_WIND_NAME,
    THIS_LOWEST_PRESSURE_NAME]

SOUNDING_FIELD_NAMES_RAP_NO_SURFACE = [
    'geopotential_height_metres_950mb', 'geopotential_height_metres_975mb',
    'geopotential_height_metres_1000mb',
    'temperature_kelvins_950mb', 'temperature_kelvins_975mb',
    'temperature_kelvins_1000mb',
    'relative_humidity_percent_950mb', 'relative_humidity_percent_975mb',
    'relative_humidity_percent_1000mb',
    'u_wind_m_s01_950mb', 'u_wind_m_s01_975mb', 'u_wind_m_s01_1000mb',
    'v_wind_m_s01_950mb', 'v_wind_m_s01_975mb', 'v_wind_m_s01_1000mb']

SOUNDING_FIELD_NAMES_RAP_GRIB1_WITH_SURFACE = [
    'HGT:950 mb', 'HGT:975 mb', 'HGT:1000 mb', THIS_LOWEST_HEIGHT_NAME_GRIB1,
    'TMP:950 mb', 'TMP:975 mb', 'TMP:1000 mb', THIS_LOWEST_TEMP_NAME_GRIB1,
    'RH:950 mb', 'RH:975 mb', 'RH:1000 mb', THIS_LOWEST_HUMIDITY_NAME_GRIB1,
    'UGRD:950 mb', 'UGRD:975 mb', 'UGRD:1000 mb', THIS_LOWEST_U_WIND_NAME_GRIB1,
    'VGRD:950 mb', 'VGRD:975 mb', 'VGRD:1000 mb', THIS_LOWEST_V_WIND_NAME_GRIB1,
    THIS_LOWEST_PRESSURE_NAME_GRIB1]

SOUNDING_FIELD_NAMES_RAP_GRIB1_NO_SURFACE = [
    'HGT:950 mb', 'HGT:975 mb', 'HGT:1000 mb',
    'TMP:950 mb', 'TMP:975 mb', 'TMP:1000 mb',
    'RH:950 mb', 'RH:975 mb', 'RH:1000 mb',
    'UGRD:950 mb', 'UGRD:975 mb', 'UGRD:1000 mb',
    'VGRD:950 mb', 'VGRD:975 mb', 'VGRD:1000 mb']

THESE_HEIGHT_NAMES_NO_SURFACE = [
    'geopotential_height_metres_950mb', 'geopotential_height_metres_975mb',
    'geopotential_height_metres_1000mb']
THESE_TEMPERATURE_NAMES_NO_SURFACE = [
    'temperature_kelvins_950mb', 'temperature_kelvins_975mb',
    'temperature_kelvins_1000mb']
THESE_HUMIDITY_NAMES_NO_SURFACE = [
    'relative_humidity_percent_950mb', 'relative_humidity_percent_975mb',
    'relative_humidity_percent_1000mb']
THESE_U_WIND_NAMES_NO_SURFACE = [
    'u_wind_m_s01_950mb', 'u_wind_m_s01_975mb', 'u_wind_m_s01_1000mb']
THESE_V_WIND_NAMES_NO_SURFACE = [
    'v_wind_m_s01_950mb', 'v_wind_m_s01_975mb', 'v_wind_m_s01_1000mb']
THESE_PRESSURE_LEVELS_NO_SURFACE_MB = numpy.array([950., 975., 1000.])

THIS_DICT = {
    soundings_only.PRESSURE_LEVEL_KEY: numpy.concatenate((
        THESE_PRESSURE_LEVELS_NO_SURFACE_MB, numpy.array([numpy.nan])
    )),
    nwp_model_utils.HEIGHT_COLUMN_FOR_SOUNDING_TABLES:
        THESE_HEIGHT_NAMES_NO_SURFACE + [THIS_LOWEST_HEIGHT_NAME],
    nwp_model_utils.TEMPERATURE_COLUMN_FOR_SOUNDING_TABLES:
        THESE_TEMPERATURE_NAMES_NO_SURFACE + [THIS_LOWEST_TEMP_NAME],
    nwp_model_utils.RH_COLUMN_FOR_SOUNDING_TABLES:
        THESE_HUMIDITY_NAMES_NO_SURFACE + [THIS_LOWEST_HUMIDITY_NAME],
    nwp_model_utils.U_WIND_COLUMN_FOR_SOUNDING_TABLES:
        THESE_U_WIND_NAMES_NO_SURFACE + [THIS_LOWEST_U_WIND_NAME],
    nwp_model_utils.V_WIND_COLUMN_FOR_SOUNDING_TABLES:
        THESE_V_WIND_NAMES_NO_SURFACE + [THIS_LOWEST_V_WIND_NAME]
}
SOUNDING_FIELD_NAME_TABLE_RAP_WITH_SURFACE = pandas.DataFrame.from_dict(
    THIS_DICT)

THIS_DICT = {
    soundings_only.PRESSURE_LEVEL_KEY: THESE_PRESSURE_LEVELS_NO_SURFACE_MB,
    nwp_model_utils.HEIGHT_COLUMN_FOR_SOUNDING_TABLES:
        THESE_HEIGHT_NAMES_NO_SURFACE,
    nwp_model_utils.TEMPERATURE_COLUMN_FOR_SOUNDING_TABLES:
        THESE_TEMPERATURE_NAMES_NO_SURFACE,
    nwp_model_utils.RH_COLUMN_FOR_SOUNDING_TABLES:
        THESE_HUMIDITY_NAMES_NO_SURFACE,
    nwp_model_utils.U_WIND_COLUMN_FOR_SOUNDING_TABLES:
        THESE_U_WIND_NAMES_NO_SURFACE,
    nwp_model_utils.V_WIND_COLUMN_FOR_SOUNDING_TABLES:
        THESE_V_WIND_NAMES_NO_SURFACE
}
SOUNDING_FIELD_NAME_TABLE_RAP_NO_SURFACE = pandas.DataFrame.from_dict(THIS_DICT)


class SoundingsOnlyTests(unittest.TestCase):
    """Each method is a unit test for soundings_only.py."""

    def test_get_nwp_fields_for_sounding_rap_no_table_no_surface(self):
        """Ensures correct output from get_nwp_fields_for_sounding.

        In this case, model is RAP; return_table = False; and include_surface
        = False.
        """

        these_field_names, these_field_names_grib1, _ = (
            soundings_only.get_nwp_fields_for_sounding(
                model_name=nwp_model_utils.RAP_MODEL_NAME, return_table=False,
                minimum_pressure_mb=MINIMUM_PRESSURE_MB, include_surface=False))

        self.assertTrue(set(these_field_names) ==
                        set(SOUNDING_FIELD_NAMES_RAP_NO_SURFACE))
        self.assertTrue(set(these_field_names_grib1) ==
                        set(SOUNDING_FIELD_NAMES_RAP_GRIB1_NO_SURFACE))

    def test_get_nwp_fields_for_sounding_rap_no_table_yes_surface(self):
        """Ensures correct output from get_nwp_fields_for_sounding.

        In this case, model is RAP; return_table = False; and include_surface
        = True.
        """

        these_field_names, these_field_names_grib1, _ = (
            soundings_only.get_nwp_fields_for_sounding(
                model_name=nwp_model_utils.RAP_MODEL_NAME, return_table=False,
                minimum_pressure_mb=MINIMUM_PRESSURE_MB, include_surface=True))

        self.assertTrue(set(these_field_names) ==
                        set(SOUNDING_FIELD_NAMES_RAP_WITH_SURFACE))
        self.assertTrue(set(these_field_names_grib1) ==
                        set(SOUNDING_FIELD_NAMES_RAP_GRIB1_WITH_SURFACE))

    def test_get_nwp_fields_for_sounding_rap_yes_table_no_surface(self):
        """Ensures correct output from get_nwp_fields_for_sounding.

        In this case, model is RAP; return_table = True; and include_surface
        = False.
        """

        _, _, this_field_name_table = (
            soundings_only.get_nwp_fields_for_sounding(
                model_name=nwp_model_utils.RAP_MODEL_NAME, return_table=True,
                minimum_pressure_mb=MINIMUM_PRESSURE_MB, include_surface=False))

        self.assertTrue(this_field_name_table.equals(
            SOUNDING_FIELD_NAME_TABLE_RAP_NO_SURFACE))

    def test_get_nwp_fields_for_sounding_rap_yes_table_yes_surface(self):
        """Ensures correct output from get_nwp_fields_for_sounding.

        In this case, model is RAP; return_table = True; and include_surface
        = True.
        """

        _, _, this_field_name_table = (
            soundings_only.get_nwp_fields_for_sounding(
                model_name=nwp_model_utils.RAP_MODEL_NAME, return_table=True,
                minimum_pressure_mb=MINIMUM_PRESSURE_MB, include_surface=True))

        self.assertTrue(this_field_name_table.equals(
            SOUNDING_FIELD_NAME_TABLE_RAP_WITH_SURFACE))


if __name__ == '__main__':
    unittest.main()
