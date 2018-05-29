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

TOTAL_PRESSURES_PASCALS = numpy.array([4e4, 5e4, 6e4, 7e4, 8e4, 9e4, 1e5])
MULTIPLIERS = numpy.array(
    [1.299263223, 1.219013654, 1.157138536, 1.107280630, 1.065831558,
     1.030560681, 1.])
POTENTIAL_TEMPERATURES_KELVINS = TEMPERATURES_KELVINS * MULTIPLIERS


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

    def test_temperatures_to_potential_temperatures(self):
        """Ensures correctness of temperatures_to_potential_temperatures."""

        these_potential_temperatures_kelvins = (
            temperature_conversions.temperatures_to_potential_temperatures(
                temperatures_kelvins=TEMPERATURES_KELVINS,
                total_pressures_pascals=TOTAL_PRESSURES_PASCALS))

        self.assertTrue(numpy.allclose(
            these_potential_temperatures_kelvins,
            POTENTIAL_TEMPERATURES_KELVINS, atol=TOLERANCE))

    def test_temperatures_from_potential_temperatures(self):
        """Ensures correctness of temperatures_from_potential_temperatures."""

        these_temperatures_kelvins = (
            temperature_conversions.temperatures_from_potential_temperatures(
                potential_temperatures_kelvins=POTENTIAL_TEMPERATURES_KELVINS,
                total_pressures_pascals=TOTAL_PRESSURES_PASCALS))

        self.assertTrue(numpy.allclose(
            these_temperatures_kelvins, TEMPERATURES_KELVINS, atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
