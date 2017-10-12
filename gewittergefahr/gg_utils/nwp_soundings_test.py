"""Unit tests for nwp_soundings.py."""

import unittest
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


if __name__ == '__main__':
    unittest.main()
