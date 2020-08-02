"""Unit tests for moisture_conversions.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import moisture_conversions as moisture_conv

TOLERANCE = 1e-4

# Given:
TEMPERATURES_KELVINS = numpy.array([
    223.15, 233.15, 243.15, 253.15, 263.15, 273.15, 283.15, 293.15, 303.15,
    313.15
])
RELATIVE_HUMIDITIES_ORIG = numpy.array([0, 0.5, 1, 0, 0.5, 1, 0, 0.5, 1, 1])
PRESSURES_PASCALS = 100 * numpy.array(
    [0, 100, 1000, 500, 250, 0, 700, 850, 0, 1000], dtype=float
)

# Derived:
RELATIVE_HUMIDITIES_FIXED = numpy.array([0, 0.5, 1, 0, 0.5, 0, 0, 0.5, 0, 1])
DEWPOINTS_KELVINS = numpy.array([
    0, 226.68960048, 243.15, 0, 254.76009248, 0, 0, 282.63260019, 0, 313.15
])
SPECIFIC_HUMIDITIES_KG_KG01 = numpy.array([
    0, 0.00058918, 0.00031664, 0, 0.00358888, 0, 0, 0.00873439, 0, 0.04722717
])
MIXING_RATIOS_KG_KG01 = numpy.array([
    0, 0.00058953, 0.00031674, 0, 0.00360181, 0, 0, 0.00881135, 0, 0.04956813
])
VAPOUR_PRESSURES_PASCALS = numpy.array([
    0, 9.46942803, 50.89921537, 0, 143.94028792, 0, 0, 1187.35688043, 0,
    7381.26290016
])

VIRTUAL_TEMP_DIFFS_KELVINS = numpy.array([
    0, 0.08349084, 0.04679432, 0, 0.57400567, 0, 0, 1.55623736, 0, 8.98871815
])
VIRTUAL_TEMPERATURES_KELVINS = TEMPERATURES_KELVINS + VIRTUAL_TEMP_DIFFS_KELVINS


class MoistureConversionsTests(unittest.TestCase):
    """Each method is a unit test for moisture_conversions.py."""

    def test_relative_humidity_to_dewpoint_first(self):
        """Ensures correct output from relative_humidity_to_dewpoint.

        In this case, using original relative humidities (not always zero when
        air pressure is zero).
        """

        these_dewpoints_kelvins = moisture_conv.relative_humidity_to_dewpoint(
            relative_humidities=RELATIVE_HUMIDITIES_ORIG,
            temperatures_kelvins=TEMPERATURES_KELVINS,
            total_pressures_pascals=PRESSURES_PASCALS
        )

        self.assertTrue(numpy.allclose(
            these_dewpoints_kelvins, DEWPOINTS_KELVINS, atol=TOLERANCE
        ))

    def test_relative_humidity_to_dewpoint_second(self):
        """Ensures correct output from relative_humidity_to_dewpoint.

        In this case, using fixed relative humidities (zero when air pressure is
        zero).
        """

        these_dewpoints_kelvins = moisture_conv.relative_humidity_to_dewpoint(
            relative_humidities=RELATIVE_HUMIDITIES_FIXED,
            temperatures_kelvins=TEMPERATURES_KELVINS,
            total_pressures_pascals=PRESSURES_PASCALS
        )

        self.assertTrue(numpy.allclose(
            these_dewpoints_kelvins, DEWPOINTS_KELVINS, atol=TOLERANCE
        ))

    def test_dewpoint_to_relative_humidity(self):
        """Ensures correct output from dewpoint_to_relative_humidity."""

        these_relative_humidities = moisture_conv.dewpoint_to_relative_humidity(
            dewpoints_kelvins=DEWPOINTS_KELVINS,
            temperatures_kelvins=TEMPERATURES_KELVINS,
            total_pressures_pascals=PRESSURES_PASCALS
        )

        self.assertTrue(numpy.allclose(
            these_relative_humidities, RELATIVE_HUMIDITIES_FIXED, atol=TOLERANCE
        ))

    def test_dewpoint_to_specific_humidity(self):
        """Ensures correct output from dewpoint_to_specific_humidity."""

        these_humidities_kg_kg01 = moisture_conv.dewpoint_to_specific_humidity(
            dewpoints_kelvins=DEWPOINTS_KELVINS,
            temperatures_kelvins=TEMPERATURES_KELVINS,
            total_pressures_pascals=PRESSURES_PASCALS
        )

        self.assertTrue(numpy.allclose(
            these_humidities_kg_kg01, SPECIFIC_HUMIDITIES_KG_KG01,
            atol=TOLERANCE
        ))

    def test_specific_humidity_to_dewpoint(self):
        """Ensures correct output from specific_humidity_to_dewpoint."""

        these_dewpoints_kelvins = moisture_conv.specific_humidity_to_dewpoint(
            specific_humidities_kg_kg01=SPECIFIC_HUMIDITIES_KG_KG01,
            temperatures_kelvins=TEMPERATURES_KELVINS,
            total_pressures_pascals=PRESSURES_PASCALS
        )

        self.assertTrue(numpy.allclose(
            these_dewpoints_kelvins, DEWPOINTS_KELVINS, atol=TOLERANCE
        ))

    def test_specific_humidity_to_mixing_ratio(self):
        """Ensures correct output from specific_humidity_to_mixing_ratio."""

        these_mixing_ratios_kg_kg01 = (
            moisture_conv.specific_humidity_to_mixing_ratio(
                SPECIFIC_HUMIDITIES_KG_KG01
            )
        )

        self.assertTrue(numpy.allclose(
            these_mixing_ratios_kg_kg01, MIXING_RATIOS_KG_KG01, atol=TOLERANCE
        ))

    def test_mixing_ratio_to_specific_humidity(self):
        """Ensures correct output from mixing_ratio_to_specific_humidity."""

        these_humidities_kg_kg01 = (
            moisture_conv.mixing_ratio_to_specific_humidity(
                MIXING_RATIOS_KG_KG01
            )
        )

        self.assertTrue(numpy.allclose(
            these_humidities_kg_kg01, SPECIFIC_HUMIDITIES_KG_KG01,
            atol=TOLERANCE
        ))

    def test_mixing_ratio_to_vapour_pressure(self):
        """Ensures correct output from mixing_ratio_to_vapour_pressure."""

        these_vapour_pressures_pascals = (
            moisture_conv.mixing_ratio_to_vapour_pressure(
                mixing_ratios_kg_kg01=MIXING_RATIOS_KG_KG01,
                total_pressures_pascals=PRESSURES_PASCALS
            )
        )

        self.assertTrue(numpy.allclose(
            these_vapour_pressures_pascals, VAPOUR_PRESSURES_PASCALS,
            atol=TOLERANCE
        ))

    def test_vapour_pressure_to_mixing_ratio(self):
        """Ensures correct output from vapour_pressure_to_mixing_ratio."""

        these_mixing_ratios_kg_kg01 = (
            moisture_conv.vapour_pressure_to_mixing_ratio(
                vapour_pressures_pascals=VAPOUR_PRESSURES_PASCALS,
                total_pressures_pascals=PRESSURES_PASCALS
            )
        )

        self.assertTrue(numpy.allclose(
            these_mixing_ratios_kg_kg01, MIXING_RATIOS_KG_KG01, atol=TOLERANCE
        ))

    def test_vapour_pressure_to_dewpoint(self):
        """Ensures correct output from vapour_pressure_to_dewpoint."""

        these_dewpoints_kelvins = moisture_conv.vapour_pressure_to_dewpoint(
            vapour_pressures_pascals=VAPOUR_PRESSURES_PASCALS,
            temperatures_kelvins=TEMPERATURES_KELVINS
        )

        self.assertTrue(numpy.allclose(
            these_dewpoints_kelvins, DEWPOINTS_KELVINS, atol=TOLERANCE
        ))

    def test_dewpoint_to_vapour_pressure(self):
        """Ensures correct output from dewpoint_to_vapour_pressure."""

        these_vapour_pressures_pascals = (
            moisture_conv.dewpoint_to_vapour_pressure(
                dewpoints_kelvins=DEWPOINTS_KELVINS,
                temperatures_kelvins=TEMPERATURES_KELVINS,
                total_pressures_pascals=PRESSURES_PASCALS
            )
        )

        self.assertTrue(numpy.allclose(
            these_vapour_pressures_pascals, VAPOUR_PRESSURES_PASCALS,
            atol=TOLERANCE
        ))

    def test_temperature_to_virtual_temperature(self):
        """Ensures correct output from temperature_to_virtual_temperature."""

        these_virtual_temps_kelvins = (
            moisture_conv.temperature_to_virtual_temperature(
                temperatures_kelvins=TEMPERATURES_KELVINS,
                total_pressures_pascals=PRESSURES_PASCALS,
                vapour_pressures_pascals=VAPOUR_PRESSURES_PASCALS
            )
        )

        self.assertTrue(numpy.allclose(
            these_virtual_temps_kelvins, VIRTUAL_TEMPERATURES_KELVINS,
            atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
