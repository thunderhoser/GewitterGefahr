"""Unit tests for sounding_plotting.py."""

import unittest
import numpy
import pandas
from gewittergefahr.plotting import sounding_plotting

TOLERANCE = 1e-6

PRESSURES_MB = numpy.array([978., 973., 947.8, 943., 925., 923., 914.5])
TEMPERATURES_DEG_C = numpy.array([10.4, 10.6, 16., 17., 16.4, 16.2, 16.5])
DEWPOINTS_DEG_C = numpy.array([6.9, 5.6, 2.6, 2., 2.4, 2.2, 1.])
WIND_SPEEDS_KT = numpy.array([5., 11., 39., 39., 37., 37., 36.])
WIND_DIRECTIONS_DEG = numpy.array([160., 166., 195., 197., 205., 205., 205.])
U_WINDS_KT = numpy.array(
    [-1.710101, -2.661141, 10.093943, 11.402496, 15.636876, 15.636876,
     15.214257])
V_WINDS_KT = numpy.array(
    [4.698463, 10.673253, 37.671107, 37.295885, 33.533388, 33.533388,
     32.627080])

SOUNDING_DICT = {
    sounding_plotting.PRESSURE_COLUMN: PRESSURES_MB,
    sounding_plotting.TEMPERATURE_COLUMN: TEMPERATURES_DEG_C,
    sounding_plotting.DEWPOINT_COLUMN: DEWPOINTS_DEG_C,
    sounding_plotting.U_WIND_COLUMN: U_WINDS_KT,
    sounding_plotting.V_WIND_COLUMN: V_WINDS_KT
}
SOUNDING_TABLE = pandas.DataFrame.from_dict(SOUNDING_DICT)

SOUNDING_DICT_FOR_SKEWT = {
    sounding_plotting.PRESSURE_COLUMN_SKEWT: PRESSURES_MB,
    sounding_plotting.TEMPERATURE_COLUMN_SKEWT: TEMPERATURES_DEG_C,
    sounding_plotting.DEWPOINT_COLUMN_SKEWT: DEWPOINTS_DEG_C,
    sounding_plotting.WIND_SPEED_COLUMN_SKEWT: WIND_SPEEDS_KT,
    sounding_plotting.WIND_DIRECTION_COLUMN_SKEWT: WIND_DIRECTIONS_DEG
}


class SoundingPlottingTests(unittest.TestCase):
    """Each method is a unit test for sounding_plotting.py."""

    def test_sounding_table_to_skewt(self):
        """Ensures correct output from sounding_table_to_skewt."""

        this_sounding_dict_for_skewt = (
            sounding_plotting.sounding_table_to_skewt(SOUNDING_TABLE))

        self.assertTrue(set(this_sounding_dict_for_skewt.keys()) ==
                        set(SOUNDING_DICT_FOR_SKEWT.keys()))

        for this_key in SOUNDING_DICT_FOR_SKEWT.keys():
            self.assertTrue(numpy.allclose(
                this_sounding_dict_for_skewt[this_key],
                SOUNDING_DICT_FOR_SKEWT[this_key], atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
