"""Unit tests for moisture_conversions.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import moisture_conversions

TOLERANCE = 1e-4

SPECIFIC_HUMIDITIES_KG_KG01 = numpy.array([0.1, 1., 5., 10., 20.]) * 1e-3
MIXING_RATIOS_KG_KG01 = numpy.array(
    [0.100010001, 1.001001001, 5.025125628, 10.101010101, 20.408163265]) * 1e-3
TOTAL_PRESSURES_PASCALS = numpy.array([990., 1000., 1010., 1020., 1030.]) * 100
VAPOUR_PRESSURES_PASCALS = numpy.array(
    [15.916152, 160.681325, 809.473927, 1630.038444, 3272.270051])
DEWPOINTS_KELVINS = numpy.array(
    [230.607063, 255.782666, 276.925618, 287.202593, 298.220939])
TEMPERATURES_KELVINS = numpy.array([273.15, 278.15, 283.15, 288.15, 298.15])
RELATIVIE_HUMIDITIES = numpy.array(
    [0.025767, 0.182124, 0.650499, 0.939869, 1.004331])


class MoistureConversionsTests(unittest.TestCase):
    """Each method is a unit test for moisture_conversions.py."""

    def test_specific_humidity_to_mixing_ratio(self):
        """Ensures correct output from specific_humidity_to_mixing_ratio."""

        these_mixing_ratios_kg_kg01 = (
            moisture_conversions.specific_humidity_to_mixing_ratio(
                SPECIFIC_HUMIDITIES_KG_KG01))
        self.assertTrue(numpy.allclose(
            these_mixing_ratios_kg_kg01, MIXING_RATIOS_KG_KG01, atol=TOLERANCE))

    def test_mixing_ratio_to_vapour_pressure(self):
        """Ensures correct output from mixing_ratio_to_vapour_pressure."""

        these_vapour_pressures_pa = (
            moisture_conversions.mixing_ratio_to_vapour_pressure(
                MIXING_RATIOS_KG_KG01, TOTAL_PRESSURES_PASCALS))
        self.assertTrue(numpy.allclose(
            these_vapour_pressures_pa, VAPOUR_PRESSURES_PASCALS,
            atol=TOLERANCE))

    def test_vapour_pressure_to_dewpoint(self):
        """Ensures correct output from vapour_pressure_to_dewpoint."""

        these_dewpoints_k = moisture_conversions.vapour_pressure_to_dewpoint(
            VAPOUR_PRESSURES_PASCALS)
        self.assertTrue(numpy.allclose(
            these_dewpoints_k, DEWPOINTS_KELVINS, atol=TOLERANCE))

    def test_dewpoint_to_vapour_pressure(self):
        """Ensures correct output from dewpoint_to_vapour_pressure."""

        these_vapour_pressures_pa = (
            moisture_conversions.dewpoint_to_vapour_pressure(DEWPOINTS_KELVINS))
        self.assertTrue(numpy.allclose(
            these_vapour_pressures_pa, VAPOUR_PRESSURES_PASCALS,
            atol=TOLERANCE))

    def test_vapour_pressure_to_mixing_ratio(self):
        """Ensures correct output from vapour_pressure_to_mixing_ratio."""

        these_mixing_ratios_kg_kg01 = (
            moisture_conversions.vapour_pressure_to_mixing_ratio(
                VAPOUR_PRESSURES_PASCALS, TOTAL_PRESSURES_PASCALS))
        self.assertTrue(numpy.allclose(
            these_mixing_ratios_kg_kg01, MIXING_RATIOS_KG_KG01, atol=TOLERANCE))

    def test_mixing_ratio_to_specific_humidity(self):
        """Ensures correct output from mixing_ratio_to_specific_humidity."""

        these_specific_humidities_kg_kg01 = (
            moisture_conversions.mixing_ratio_to_specific_humidity(
                MIXING_RATIOS_KG_KG01))
        self.assertTrue(numpy.allclose(
            these_specific_humidities_kg_kg01, SPECIFIC_HUMIDITIES_KG_KG01,
            atol=TOLERANCE))

    def test_specific_humidity_to_dewpoint(self):
        """Ensures correct output from specific_humidity_to_dewpoint."""

        these_dewpoints_k = moisture_conversions.specific_humidity_to_dewpoint(
            SPECIFIC_HUMIDITIES_KG_KG01, TOTAL_PRESSURES_PASCALS)
        self.assertTrue(numpy.allclose(
            these_dewpoints_k, DEWPOINTS_KELVINS, atol=TOLERANCE))

    def test_relative_humidity_to_dewpoint(self):
        """Ensures correct output from relative_humidity_to_dewpoint."""

        these_dewpoints_k = moisture_conversions.relative_humidity_to_dewpoint(
            RELATIVIE_HUMIDITIES, TEMPERATURES_KELVINS)
        self.assertTrue(numpy.allclose(
            these_dewpoints_k, DEWPOINTS_KELVINS, atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
