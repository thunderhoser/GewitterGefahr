"""Conversion methods for atmospheric-moisture variables."""

import numpy
from gewittergefahr.gg_utils import temperature_conversions as temperature_conv
from gewittergefahr.gg_utils import error_checking

DRY_AIR_GAS_CONSTANT_J_KG01_K01 = 287.04
WATER_VAPOUR_GAS_CONSTANT_J_KG01_K01 = 461.5
EPSILON = DRY_AIR_GAS_CONSTANT_J_KG01_K01 / WATER_VAPOUR_GAS_CONSTANT_J_KG01_K01

BASE_VAPOUR_PRESSURE_PASCALS = 610.78
MAGNUS_NUMERATOR_COEFF_WATER = 17.08085
MAGNUS_NUMERATOR_COEFF_ICE = 17.84362
MAGNUS_DENOMINATOR_COEFF_WATER = 234.175
MAGNUS_DENOMINATOR_COEFF_ICE = 245.425


def specific_humidity_to_mixing_ratio(specific_humidities_kg_kg01):
    """Converts one or more specific humidities to mixing ratios.

    :param specific_humidities_kg_kg01: numpy array of specific humidities
        (kg per kg).
    :return: mixing_ratios_kg_kg01: equivalent-size numpy array of mixing ratios
        (kg per kg).
    """

    error_checking.assert_is_real_numpy_array(specific_humidities_kg_kg01)
    return specific_humidities_kg_kg01 / (1 - specific_humidities_kg_kg01)


def mixing_ratio_to_vapour_pressure(mixing_ratios_kg_kg01,
                                    total_pressures_pascals):
    """Converts one or more mixing ratios to vapour pressures.

    :param mixing_ratios_kg_kg01: numpy array of mixing ratios (kg per kg).
    :param total_pressures_pascals: equivalent-size numpy array of total air
        pressures (Pa).
    :return: vapour_pressures_pascals: equivalent-size numpy array of water
        vapour pressures (Pa).
    """

    error_checking.assert_is_real_numpy_array(mixing_ratios_kg_kg01)
    error_checking.assert_is_real_numpy_array(total_pressures_pascals)
    expected_dim = numpy.asarray(mixing_ratios_kg_kg01.shape, dtype=int)
    error_checking.assert_is_numpy_array(
        total_pressures_pascals, exact_dimensions=expected_dim
    )

    return (
        mixing_ratios_kg_kg01 * total_pressures_pascals /
        (EPSILON + mixing_ratios_kg_kg01)
    )


def vapour_pressure_to_dewpoint(vapour_pressures_pascals, temperatures_kelvins):
    """Converts each vapour pressure to dewpoint.

    Source:
    https://content.meteoblue.com/hu/specifications/weather-variables/humidity

    :param vapour_pressures_pascals: numpy array (any shape) of vapour
        pressures.
    :param temperatures_kelvins: numpy array (same shape) of temperatures.
    :return: dewpoints_kelvins: numpy array (same shape) of dewpoints.
    """

    error_checking.assert_is_geq_numpy_array(
        vapour_pressures_pascals, 0., allow_nan=True
    )
    error_checking.assert_is_geq_numpy_array(
        temperatures_kelvins, 0., allow_nan=True
    )
    error_checking.assert_is_numpy_array(
        temperatures_kelvins,
        exact_dimensions=numpy.array(vapour_pressures_pascals.shape, dtype=int)
    )

    logarithms = numpy.log(
        vapour_pressures_pascals / BASE_VAPOUR_PRESSURE_PASCALS
    )

    temperatures_deg_c = temperature_conv.kelvins_to_celsius(
        temperatures_kelvins
    )

    numerator_coeffs = numpy.full(
        temperatures_deg_c.shape, MAGNUS_DENOMINATOR_COEFF_WATER
    )
    numerator_coeffs[temperatures_deg_c < 0] = MAGNUS_DENOMINATOR_COEFF_ICE
    numerators = numerator_coeffs * logarithms

    denominator_coeffs = numpy.full(
        temperatures_deg_c.shape, MAGNUS_NUMERATOR_COEFF_WATER
    )
    denominator_coeffs[temperatures_deg_c < 0] = MAGNUS_NUMERATOR_COEFF_ICE
    denominators = denominator_coeffs - logarithms

    dewpoints_deg_c = numerators / denominators
    return temperature_conv.celsius_to_kelvins(dewpoints_deg_c)


def dewpoint_to_vapour_pressure(dewpoints_kelvins, temperatures_kelvins):
    """Converts each dewpoint to vapour pressure.

    Source:
    https://content.meteoblue.com/hu/specifications/weather-variables/humidity

    :param dewpoints_kelvins: numpy array (any shape) of dewpoints.
    :param temperatures_kelvins: numpy array (same shape) of temperatures.
    :return: vapour_pressures_pascals: numpy array (same shape) of vapour
        pressures.
    """

    error_checking.assert_is_geq_numpy_array(
        dewpoints_kelvins, 0., allow_nan=True
    )
    error_checking.assert_is_geq_numpy_array(
        temperatures_kelvins, 0., allow_nan=True
    )
    error_checking.assert_is_numpy_array(
        temperatures_kelvins,
        exact_dimensions=numpy.array(dewpoints_kelvins.shape, dtype=int)
    )

    dewpoints_deg_c = temperature_conv.kelvins_to_celsius(dewpoints_kelvins)
    temperatures_deg_c = temperature_conv.kelvins_to_celsius(
        temperatures_kelvins
    )

    numerator_coeffs = numpy.full(
        temperatures_deg_c.shape, MAGNUS_NUMERATOR_COEFF_WATER
    )
    numerator_coeffs[temperatures_deg_c < 0] = MAGNUS_NUMERATOR_COEFF_ICE
    numerators = numerator_coeffs * dewpoints_deg_c

    denominator_coeffs = numpy.full(
        temperatures_deg_c.shape, MAGNUS_DENOMINATOR_COEFF_WATER
    )
    denominator_coeffs[temperatures_deg_c < 0] = MAGNUS_DENOMINATOR_COEFF_ICE
    denominators = denominator_coeffs + dewpoints_deg_c

    return BASE_VAPOUR_PRESSURE_PASCALS * numpy.exp(numerators / denominators)


def vapour_pressure_to_mixing_ratio(vapour_pressures_pascals,
                                    total_pressures_pascals):
    """Converts one or more vapour pressures to mixing ratios.

    :param vapour_pressures_pascals: numpy array of vapour pressures (Pa).
    :param total_pressures_pascals: equivalent-size numpy array of total air
        pressures (K).
    :return: mixing_ratios_kg_kg01: equivalent-size numpy array of mixing ratios
        (kg per kg).
    """

    error_checking.assert_is_real_numpy_array(vapour_pressures_pascals)
    error_checking.assert_is_real_numpy_array(total_pressures_pascals)
    expected_dim = numpy.array(vapour_pressures_pascals.shape, dtype=int)
    error_checking.assert_is_numpy_array(
        total_pressures_pascals, exact_dimensions=expected_dim
    )

    return (
        EPSILON * vapour_pressures_pascals /
        (total_pressures_pascals - vapour_pressures_pascals)
    )


def mixing_ratio_to_specific_humidity(mixing_ratios_kg_kg01):
    """Converts one or more mixing ratios to specific humidities.

    :param mixing_ratios_kg_kg01: numpy array of mixing ratios (kg per kg).
    :return: specific_humidities_kg_kg01: equivalent-size numpy array of
        specific humidities (kg per kg).
    """

    error_checking.assert_is_real_numpy_array(mixing_ratios_kg_kg01)
    return mixing_ratios_kg_kg01 / (1 + mixing_ratios_kg_kg01)


def specific_humidity_to_dewpoint(
        specific_humidities_kg_kg01, temperatures_kelvins,
        total_pressures_pascals):
    """Converts each specific humidity to dewpoint.

    :param specific_humidities_kg_kg01: numpy array (any shape) of specific
        humidities (kg/kg).
    :param temperatures_kelvins: numpy array (same shape) of temperatures.
    :param total_pressures_pascals: numpy array (same shape) of total air
        pressures.
    :return: dewpoints_kelvins: numpy array (same shape) of dewpoints.
    """

    mixing_ratios_kg_kg01 = specific_humidity_to_mixing_ratio(
        specific_humidities_kg_kg01
    )
    vapour_pressures_pascals = mixing_ratio_to_vapour_pressure(
        mixing_ratios_kg_kg01, total_pressures_pascals
    )

    return vapour_pressure_to_dewpoint(
        vapour_pressures_pascals=vapour_pressures_pascals,
        temperatures_kelvins=temperatures_kelvins
    )


def dewpoint_to_specific_humidity(dewpoints_kelvins, temperatures_kelvins,
                                  total_pressures_pascals):
    """Converts each dewpoint to specific humidity.

    :param dewpoints_kelvins: numpy array (any shape) of dewpoints.
    :param temperatures_kelvins: numpy array (same shape) of temperatures.
    :param total_pressures_pascals: numpy array (same shape) of total air
        pressures.
    :return: specific_humidities_kg_kg01: numpy array (same shape) of specific
        humidities (kg/kg).
    """

    vapour_pressures_pascals = dewpoint_to_vapour_pressure(
        dewpoints_kelvins=dewpoints_kelvins,
        temperatures_kelvins=temperatures_kelvins
    )
    mixing_ratios_kg_kg01 = vapour_pressure_to_mixing_ratio(
        vapour_pressures_pascals, total_pressures_pascals
    )

    return mixing_ratio_to_specific_humidity(mixing_ratios_kg_kg01)


def relative_humidity_to_dewpoint(relative_humidities, temperatures_kelvins,
                                  total_pressures_pascals):
    """Converts each relative humidity to dewpoint.

    :param relative_humidities: numpy array (any shape) of relative humidities
        (unitless).
    :param temperatures_kelvins: numpy array (same shape) of temperatures.
    :param total_pressures_pascals: numpy array (same shape) of total air
        pressures.
    :return: dewpoints_kelvins: numpy array (same shape) of dewpoints.
    """

    error_checking.assert_is_real_numpy_array(relative_humidities)
    error_checking.assert_is_real_numpy_array(temperatures_kelvins)
    expected_dim = numpy.array(relative_humidities.shape, dtype=int)
    error_checking.assert_is_numpy_array(
        temperatures_kelvins, exact_dimensions=expected_dim
    )

    saturated_vapour_pressures_pascals = dewpoint_to_vapour_pressure(
        dewpoints_kelvins=temperatures_kelvins,
        temperatures_kelvins=temperatures_kelvins
    )
    saturated_mixing_ratios_kg_kg01 = vapour_pressure_to_mixing_ratio(
        saturated_vapour_pressures_pascals, total_pressures_pascals
    )

    vapour_pressures_pascals = mixing_ratio_to_vapour_pressure(
        relative_humidities * saturated_mixing_ratios_kg_kg01,
        total_pressures_pascals
    )
    return vapour_pressure_to_dewpoint(
        vapour_pressures_pascals=vapour_pressures_pascals,
        temperatures_kelvins=temperatures_kelvins
    )


def dewpoint_to_relative_humidity(
        dewpoints_kelvins, temperatures_kelvins, total_pressures_pascals):
    """Converts each dewpoint to specific humidity.

    :param dewpoints_kelvins: numpy array (any shape) of dewpoints.
    :param temperatures_kelvins: numpy array (same shape) of temperatures.
    :param total_pressures_pascals: numpy array (same shape) of total air
        pressures.
    :return: relative_humidities: numpy array (same shape) of relative
        humidities (unitless).
    """

    error_checking.assert_is_real_numpy_array(dewpoints_kelvins)
    error_checking.assert_is_real_numpy_array(temperatures_kelvins)
    expected_dim = numpy.array(dewpoints_kelvins.shape, dtype=int)
    error_checking.assert_is_numpy_array(
        temperatures_kelvins, exact_dimensions=expected_dim
    )

    vapour_pressures_pascals = dewpoint_to_vapour_pressure(
        dewpoints_kelvins=dewpoints_kelvins,
        temperatures_kelvins=temperatures_kelvins
    )
    mixing_ratios_kg_kg01 = vapour_pressure_to_mixing_ratio(
        vapour_pressures_pascals, total_pressures_pascals
    )

    saturated_vapour_pressures_pascals = dewpoint_to_vapour_pressure(
        dewpoints_kelvins=temperatures_kelvins,
        temperatures_kelvins=temperatures_kelvins
    )
    saturated_mixing_ratios_kg_kg01 = vapour_pressure_to_mixing_ratio(
        saturated_vapour_pressures_pascals, total_pressures_pascals
    )

    return mixing_ratios_kg_kg01 / saturated_mixing_ratios_kg_kg01


def temperature_to_virtual_temperature(
        temperatures_kelvins, total_pressures_pascals,
        vapour_pressures_pascals):
    """Converts one or more temperatures to virtual temperatures.

    :param temperatures_kelvins: numpy array of air temperatures (K).
    :param total_pressures_pascals: equivalent-size numpy array of total air
        pressures (Pa).
    :param vapour_pressures_pascals: equivalent-size numpy array of vapour
        pressures (Pa).
    :return: virtual_temperatures_kelvins: equivalent-size numpy array of
        virtual air temperatures (K).
    """

    error_checking.assert_is_real_numpy_array(temperatures_kelvins)
    error_checking.assert_is_real_numpy_array(total_pressures_pascals)
    error_checking.assert_is_real_numpy_array(vapour_pressures_pascals)

    expected_dim = numpy.array(temperatures_kelvins.shape, dtype=int)
    error_checking.assert_is_numpy_array(
        total_pressures_pascals, exact_dimensions=expected_dim
    )
    error_checking.assert_is_numpy_array(
        vapour_pressures_pascals, exact_dimensions=expected_dim
    )

    denominator_values = 1. - (
        (vapour_pressures_pascals / total_pressures_pascals) * (1. - EPSILON)
    )

    return temperatures_kelvins / denominator_values
