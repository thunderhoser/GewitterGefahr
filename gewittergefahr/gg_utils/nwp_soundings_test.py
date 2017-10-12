"""Unit tests for nwp_soundings.py."""

import unittest
from gewittergefahr.gg_utils import nwp_soundings

FAKE_MODEL_NAME = 'ecmwf'

SOME_RAP_SOUNDING_COLUMNS = [
    'temperature_kelvins_225mb', 'temperature_kelvins_725mb',
    'relative_humidity_225mb', 'relative_humidity_725mb',
    'geopotential_height_metres_225mb', 'geopotential_height_metres_725mb',
    'u_wind_m_s01_225mb', 'u_wind_m_s01_725mb',
    'v_wind_m_s01_225mb', 'v_wind_m_s01_725mb']
SOME_RAP_SOUNDING_COLUMNS_ORIG = ['TMP:225 mb', 'TMP:725 mb',
                                  'RH:225 mb', 'RH:725 mb',
                                  'HGT:225 mb', 'HGT:725 mb',
                                  'UGRD:225 mb', 'UGRD:725 mb',
                                  'VGRD:225 mb', 'VGRD:725 mb']

SOME_NARR_SOUNDING_COLUMNS = [
    'temperature_kelvins_550mb', 'temperature_kelvins_925mb',
    'specific_humidity_550mb', 'specific_humidity_925mb',
    'geopotential_height_metres_550mb', 'geopotential_height_metres_925mb',
    'u_wind_m_s01_550mb', 'u_wind_m_s01_925mb',
    'v_wind_m_s01_550mb', 'v_wind_m_s01_925mb']
SOME_NARR_SOUNDING_COLUMNS_ORIG = ['TMP:550 mb', 'TMP:925 mb',
                                   'SPFH:550 mb', 'SPFH:925 mb',
                                   'HGT:550 mb', 'HGT:925 mb',
                                   'UGRD:550 mb', 'UGRD:925 mb',
                                   'VGRD:550 mb', 'VGRD:925 mb']

FAKE_SOUNDING_COLUMNS = [
    'temperature_kelvins_317mb', 'temperature_kelvins_962mb',
    'relative_humidity_317mb', 'relative_humidity_962mb',
    'geopotential_height_metres_317mb', 'geopotential_height_metres_962mb',
    'u_wind_m_s01_317mb', 'u_wind_m_s01_962mb',
    'v_wind_m_s01_317mb', 'v_wind_m_s01_962mb']
FAKE_SOUNDING_COLUMNS_ORIG = ['TMP:317 mb', 'TMP:962 mb',
                              'RH:317 mb', 'RH:962 mb',
                              'HGT:317 mb', 'HGT:962 mb',
                              'UGRD:317 mb', 'UGRD:962 mb',
                              'VGRD:317 mb', 'VGRD:962 mb']


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
             nwp_soundings.RAP_MODEL_NAME)

        for i in range(len(SOME_RAP_SOUNDING_COLUMNS)):
            self.assertTrue(
                SOME_RAP_SOUNDING_COLUMNS[i] in these_sounding_columns)
            self.assertTrue(SOME_RAP_SOUNDING_COLUMNS_ORIG[
                i] in these_sounding_columns_orig)

        for i in range(len(FAKE_SOUNDING_COLUMNS)):
            self.assertFalse(FAKE_SOUNDING_COLUMNS[i] in these_sounding_columns)
            self.assertFalse(
                FAKE_SOUNDING_COLUMNS_ORIG[i] in these_sounding_columns_orig)

    def test_get_sounding_columns_narr(self):
        """Ensures correct output from _get_sounding_columns.

        In this case, model is NARR.
        """

        (these_sounding_columns,
         these_sounding_columns_orig) = nwp_soundings._get_sounding_columns(
             nwp_soundings.NARR_MODEL_NAME)

        for i in range(len(SOME_NARR_SOUNDING_COLUMNS)):
            self.assertTrue(
                SOME_NARR_SOUNDING_COLUMNS[i] in these_sounding_columns)
            self.assertTrue(SOME_NARR_SOUNDING_COLUMNS_ORIG[
                i] in these_sounding_columns_orig)

        for i in range(len(FAKE_SOUNDING_COLUMNS)):
            self.assertFalse(FAKE_SOUNDING_COLUMNS[i] in these_sounding_columns)
            self.assertFalse(
                FAKE_SOUNDING_COLUMNS_ORIG[i] in these_sounding_columns_orig)


if __name__ == '__main__':
    unittest.main()
