"""Unit tests for temperature_conversions.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import temperature_conversions

TOLERANCE = 1e-6

TEMPERATURES_DEG_C = numpy.array(
    [-100., -80., -60., -40., -20., 0., 15.])  # range of comfortable temps
TEMPERATURES_DEG_F = numpy.array([-148., -112., -76., -40., -4., 32., 59.])
TEMPERATURES_KELVINS = numpy.array(
    [173.15, 193.15, 213.15, 233.15, 253.15, 273.15, 288.15])


class TemperatureConversionsTests(unittest.TestCase):
    """Each method is a unit test for temperature_conversions.py."""

    def test_fahrenheit_to_celsius(self):
        """Ensures correct output from fahrenheit_to_celsius."""

        these_temperatures_deg_c = (
            temperature_conversions.fahrenheit_to_celsius(TEMPERATURES_DEG_F))
        self.assertTrue(numpy.allclose(
            these_temperatures_deg_c, TEMPERATURES_DEG_C, atol=TOLERANCE))

    def test_celsius_to_kelvins(self):
        """Ensures correct output from celsius_to_kelvins."""

        these_temperatures_kelvins = (
            temperature_conversions.celsius_to_kelvins(TEMPERATURES_DEG_C))
        self.assertTrue(numpy.allclose(
            these_temperatures_kelvins, TEMPERATURES_KELVINS, atol=TOLERANCE))

    def test_kelvins_to_celsius(self):
        """Ensures correct output from kelvins_to_celsius."""

        these_temperatures_deg_c = temperature_conversions.kelvins_to_celsius(
            TEMPERATURES_KELVINS)
        self.assertTrue(numpy.allclose(
            these_temperatures_deg_c, TEMPERATURES_DEG_C, atol=TOLERANCE))

    def test_celsius_to_fahrenheit(self):
        """Ensures correct output from celsius_to_fahrenheit."""

        these_temperatures_deg_f = (
            temperature_conversions.celsius_to_fahrenheit(TEMPERATURES_DEG_C))
        self.assertTrue(numpy.allclose(
            these_temperatures_deg_f, TEMPERATURES_DEG_F, atol=TOLERANCE))

    def test_fahrenheit_to_kelvins(self):
        """Ensures correct output from fahrenheit_to_kelvins."""

        these_temperatures_kelvins = (
            temperature_conversions.fahrenheit_to_kelvins(TEMPERATURES_DEG_F))
        self.assertTrue(numpy.allclose(
            these_temperatures_kelvins, TEMPERATURES_KELVINS, atol=TOLERANCE))

    def test_kelvins_to_fahrenheit(self):
        """Ensures correct output from kelvins_to_fahrenheit."""

        these_temperatures_deg_f = (
            temperature_conversions.kelvins_to_fahrenheit(TEMPERATURES_KELVINS))
        self.assertTrue(numpy.allclose(
            these_temperatures_deg_f, TEMPERATURES_DEG_F, atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
