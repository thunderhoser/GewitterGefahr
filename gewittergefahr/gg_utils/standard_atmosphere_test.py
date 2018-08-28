"""Unit tests for standard_atmosphere.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import standard_atmosphere as standard_atmo

TOLERANCE = 1e-5

PRESSURES_PASCALS = numpy.array(
    [105000, 101325, 50000, 22632.1, 20000, 5474.89, 5000, 868.019, 500,
     110.906, 100, 66.9389, 50, 3.95642])
HEIGHTS_M_ASL = numpy.array(
    [-261.446558, 0, 5183.188024, 11000, 11784.057997, 20000, 20591.194585,
     32000, 36021.365489, 47000, 47820.061394, 51000, 53063.025461, 71000],
    dtype=float)


class StandardAtmosphereTests(unittest.TestCase):
    """Each method is a unit test for standard_atmosphere.py."""

    def test_pressure_to_height(self):
        """Ensures correct output from pressure_to_height."""

        these_heights_m_asl = standard_atmo.pressure_to_height(
            PRESSURES_PASCALS)
        self.assertTrue(numpy.allclose(these_heights_m_asl, HEIGHTS_M_ASL))

    def test_height_to_pressure(self):
        """Ensures correct output from height_to_pressure."""

        these_pressures_pascals = standard_atmo.height_to_pressure(
            HEIGHTS_M_ASL)
        self.assertTrue(numpy.allclose(
            these_pressures_pascals, PRESSURES_PASCALS))


if __name__ == '__main__':
    unittest.main()
