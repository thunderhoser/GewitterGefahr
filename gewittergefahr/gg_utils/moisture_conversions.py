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
    """Converts each specific humidity to mixing ratio.

    :param specific_humidities_kg_kg01: numpy array (any shape) of specific
        humidities (kg per kg).
    :return: mixing_ratios_kg_kg01: numpy array (same shape) of mixing ratios
        (kg per kg).
    """

    error_checking.assert_is_geq_numpy_array(
        specific_humidities_kg_kg01, 0., allow_nan=True
    )

    return specific_humidities_kg_kg01 / (1 - specific_humidities_kg_kg01)


def mixing_ratio_to_specific_humidity(mixing_ratios_kg_kg01):
    """Converts each mixing ratio to specific humidity.

    :param mixing_ratios_kg_kg01: numpy array (any shape) of mixing ratios
        (kg per kg).
    :return: specific_humidities_kg_kg01: numpy array (same shape) of specific
        humidities (kg per kg).
    """

    error_checking.assert_is_geq_numpy_array(
        mixing_ratios_kg_kg01, 0., allow_nan=True
    )

    return mixing_ratios_kg_kg01 / (1 + mixing_ratios_kg_kg01)


def mixing_ratio_to_vapour_pressure(
        mixing_ratios_kg_kg01, total_pressures_pascals):
    """Converts each mixing ratio to vapour pressure.

    :param mixing_ratios_kg_kg01: numpy array (any shape) of mixing ratios
        (kg per kg).
    :param total_pressures_pascals: numpy array (same shape) of total air
        pressures.
    :return: vapour_pressures_pascals: numpy array (same shape) of vapour
        pressures.
    """

    error_checking.assert_is_geq_numpy_array(
        mixing_ratios_kg_kg01, 0., allow_nan=True
    )
    error_checking.assert_is_geq_numpy_array(
        total_pressures_pascals, 0., allow_nan=True
    )
    error_checking.assert_is_numpy_array(
        total_pressures_pascals,
        exact_dimensions=numpy.array(mixing_ratios_kg_kg01.shape, dtype=int)
    )

    return (
        mixing_ratios_kg_kg01 * total_pressures_pascals /
        (EPSILON + mixing_ratios_kg_kg01)
    )


def vapour_pressure_to_mixing_ratio(
        vapour_pressures_pascals, total_pressures_pascals):
    """Converts each vapour pressure to mixing ratio.

    :param vapour_pressures_pascals: numpy array (any shape) of vapour
        pressures.
    :param total_pressures_pascals: numpy array (same shape) of total air
        pressures.
    :return: mixing_ratios_kg_kg01: numpy array (same shape) of mixing ratios
        (kg per kg).
    """

    error_checking.assert_is_geq_numpy_array(
        vapour_pressures_pascals, 0., allow_nan=True
    )
    error_checking.assert_is_geq_numpy_array(
        total_pressures_pascals, 0., allow_nan=True
    )
    error_checking.assert_is_numpy_array(
        total_pressures_pascals,
        exact_dimensions=numpy.array(vapour_pressures_pascals.shape, dtype=int)
    )
    error_checking.assert_is_geq_numpy_array(
        total_pressures_pascals - vapour_pressures_pascals, 0., allow_nan=True
    )

    denominators = total_pressures_pascals - vapour_pressures_pascals
    mixing_ratios_kg_kg01 = EPSILON * vapour_pressures_pascals / denominators
    mixing_ratios_kg_kg01[denominators <= 0] = 0

    return mixing_ratios_kg_kg01


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
    dewpoints_deg_c[numpy.invert(numpy.isfinite(dewpoints_deg_c))] = (
        -temperature_conv.CELSIUS_TO_KELVINS_ADDEND
    )

    dewpoints_kelvins = temperature_conv.celsius_to_kelvins(dewpoints_deg_c)
    dewpoints_kelvins[dewpoints_deg_c + numerator_coeffs < 0] = 0.

    return dewpoints_kelvins


def dewpoint_to_vapour_pressure(dewpoints_kelvins, temperatures_kelvins,
                                total_pressures_pascals):
    """Converts each dewpoint to vapour pressure.

    Source:
    https://content.meteoblue.com/hu/specifications/weather-variables/humidity

    :param dewpoints_kelvins: numpy array (any shape) of dewpoints.
    :param temperatures_kelvins: numpy array (same shape) of temperatures.
    :param total_pressures_pascals: numpy array (same shape) of total air
        pressures.
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
    error_checking.assert_is_geq_numpy_array(
        total_pressures_pascals, 0., allow_nan=True
    )
    error_checking.assert_is_numpy_array(
        total_pressures_pascals,
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

    vapour_pressures_pascals = (
        BASE_VAPOUR_PRESSURE_PASCALS * numpy.exp(numerators / denominators)
    )

    vapour_pressures_pascals[
        numpy.invert(numpy.isfinite(vapour_pressures_pascals))
    ] = 0.

    vapour_pressures_pascals[denominators <= 0] = 0.
    return numpy.minimum(vapour_pressures_pascals, total_pressures_pascals)


def specific_humidity_to_dewpoint(
        specific_humidities_kg_kg01, temperatures_kelvins,
        total_pressures_pascals):
    """Converts each specific humidity to dewpoint.

    :param specific_humidities_kg_kg01: numpy array (any shape) of specific
        humidities (kg per kg).
    :param temperatures_kelvins: numpy array (same shape) of temperatures.
    :param total_pressures_pascals: numpy array (same shape) of total air
        pressures.
    :return: dewpoints_kelvins: numpy array (same shape) of dewpoints.
    """

    mixing_ratios_kg_kg01 = specific_humidity_to_mixing_ratio(
        specific_humidities_kg_kg01
    )
    vapour_pressures_pascals = mixing_ratio_to_vapour_pressure(
        mixing_ratios_kg_kg01=mixing_ratios_kg_kg01,
        total_pressures_pascals=total_pressures_pascals
    )

    return vapour_pressure_to_dewpoint(
        vapour_pressures_pascals=vapour_pressures_pascals,
        temperatures_kelvins=temperatures_kelvins
    )


def dewpoint_to_specific_humidity(
        dewpoints_kelvins, temperatures_kelvins, total_pressures_pascals):
    """Converts each dewpoint to specific humidity.

    :param dewpoints_kelvins: numpy array (any shape) of dewpoints.
    :param temperatures_kelvins: numpy array (same shape) of temperatures.
    :param total_pressures_pascals: numpy array (same shape) of total air
        pressures.
    :return: specific_humidities_kg_kg01: numpy array (same shape) of specific
        humidities (kg per kg).
    """

    vapour_pressures_pascals = dewpoint_to_vapour_pressure(
        dewpoints_kelvins=dewpoints_kelvins,
        temperatures_kelvins=temperatures_kelvins,
        total_pressures_pascals=total_pressures_pascals
    )
    mixing_ratios_kg_kg01 = vapour_pressure_to_mixing_ratio(
        vapour_pressures_pascals=vapour_pressures_pascals,
        total_pressures_pascals=total_pressures_pascals
    )

    return mixing_ratio_to_specific_humidity(mixing_ratios_kg_kg01)


def relative_humidity_to_dewpoint(
        relative_humidities, temperatures_kelvins, total_pressures_pascals):
    """Converts each relative humidity to dewpoint.

    :param relative_humidities: numpy array (any shape) of relative humidities
        (unitless).
    :param temperatures_kelvins: numpy array (same shape) of temperatures.
    :param total_pressures_pascals: numpy array (same shape) of total air
        pressures.
    :return: dewpoints_kelvins: numpy array (same shape) of dewpoints.
    """

    error_checking.assert_is_geq_numpy_array(
        relative_humidities, 0., allow_nan=True
    )
    error_checking.assert_is_geq_numpy_array(
        temperatures_kelvins, 0., allow_nan=True
    )
    error_checking.assert_is_numpy_array(
        temperatures_kelvins,
        exact_dimensions=numpy.array(relative_humidities.shape, dtype=int)
    )
    error_checking.assert_is_geq_numpy_array(
        total_pressures_pascals, 0., allow_nan=True
    )
    error_checking.assert_is_numpy_array(
        total_pressures_pascals,
        exact_dimensions=numpy.array(relative_humidities.shape, dtype=int)
    )

    saturated_vapour_pressures_pascals = dewpoint_to_vapour_pressure(
        dewpoints_kelvins=temperatures_kelvins,
        temperatures_kelvins=temperatures_kelvins,
        total_pressures_pascals=total_pressures_pascals
    )
    saturated_mixing_ratios_kg_kg01 = vapour_pressure_to_mixing_ratio(
        vapour_pressures_pascals=saturated_vapour_pressures_pascals,
        total_pressures_pascals=total_pressures_pascals
    )
    vapour_pressures_pascals = mixing_ratio_to_vapour_pressure(
        mixing_ratios_kg_kg01=
        relative_humidities * saturated_mixing_ratios_kg_kg01,
        total_pressures_pascals=total_pressures_pascals
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
    error_checking.assert_is_geq_numpy_array(
        total_pressures_pascals, 0., allow_nan=True
    )
    error_checking.assert_is_numpy_array(
        total_pressures_pascals,
        exact_dimensions=numpy.array(dewpoints_kelvins.shape, dtype=int)
    )

    vapour_pressures_pascals = dewpoint_to_vapour_pressure(
        dewpoints_kelvins=dewpoints_kelvins,
        temperatures_kelvins=temperatures_kelvins,
        total_pressures_pascals=total_pressures_pascals
    )
    mixing_ratios_kg_kg01 = vapour_pressure_to_mixing_ratio(
        vapour_pressures_pascals=vapour_pressures_pascals,
        total_pressures_pascals=total_pressures_pascals
    )
    saturated_vapour_pressures_pascals = dewpoint_to_vapour_pressure(
        dewpoints_kelvins=temperatures_kelvins,
        temperatures_kelvins=temperatures_kelvins,
        total_pressures_pascals=total_pressures_pascals
    )
    saturated_mixing_ratios_kg_kg01 = vapour_pressure_to_mixing_ratio(
        vapour_pressures_pascals=saturated_vapour_pressures_pascals,
        total_pressures_pascals=total_pressures_pascals
    )

    relative_humidities = (
        mixing_ratios_kg_kg01 / saturated_mixing_ratios_kg_kg01
    )
    relative_humidities[numpy.invert(numpy.isfinite(relative_humidities))] = 0.

    return relative_humidities


def temperature_to_virtual_temperature(
        temperatures_kelvins, total_pressures_pascals,
        vapour_pressures_pascals):
    """Converts each temperature to virtual temperature.

    :param temperatures_kelvins: numpy array (any shape) of temperatures.
    :param total_pressures_pascals: numpy array (same shape) of total air
        pressures.
    :param vapour_pressures_pascals: numpy array (same shape) of vapour
        pressures.
    :return: virtual_temperatures_kelvins: numpy array (same shape) of virtual
        temperatures.
    """

    error_checking.assert_is_geq_numpy_array(
        temperatures_kelvins, 0., allow_nan=True
    )
    error_checking.assert_is_geq_numpy_array(
        total_pressures_pascals, 0., allow_nan=True
    )
    error_checking.assert_is_numpy_array(
        total_pressures_pascals,
        exact_dimensions=numpy.array(temperatures_kelvins.shape, dtype=int)
    )
    error_checking.assert_is_geq_numpy_array(
        vapour_pressures_pascals, 0., allow_nan=True
    )
    error_checking.assert_is_numpy_array(
        vapour_pressures_pascals,
        exact_dimensions=numpy.array(temperatures_kelvins.shape, dtype=int)
    )
    error_checking.assert_is_geq_numpy_array(
        total_pressures_pascals - vapour_pressures_pascals, 0., allow_nan=True
    )

    denominator_values = 1. - (
        (vapour_pressures_pascals / total_pressures_pascals) * (1. - EPSILON)
    )

    virtual_temperatures_kelvins = temperatures_kelvins / denominator_values
    virtual_temperatures_kelvins[total_pressures_pascals == 0] = (
        temperatures_kelvins[total_pressures_pascals == 0]
    )

    return virtual_temperatures_kelvins
