"""Conversion methods for atmospheric-moisture variables."""

import numpy
from gewittergefahr.gg_utils import error_checking

DRY_AIR_GAS_CONSTANT_J_KG01_K01 = 287.04
WATER_VAPOUR_GAS_CONSTANT_J_KG01_K01 = 461.5
EPSILON = DRY_AIR_GAS_CONSTANT_J_KG01_K01 / WATER_VAPOUR_GAS_CONSTANT_J_KG01_K01

BASE_TEMPERATURE_KELVINS = 273.
BASE_VAPOUR_PRESSURE_PASCALS = 611.
LATENT_HEAT_OF_CONDENSATION_J_KG01 = 2.5e6


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
    mixing_ratio_dimensions = numpy.asarray(mixing_ratios_kg_kg01.shape)
    error_checking.assert_is_numpy_array(
        total_pressures_pascals, exact_dimensions=mixing_ratio_dimensions)

    return (mixing_ratios_kg_kg01 * total_pressures_pascals) / (
        EPSILON + mixing_ratios_kg_kg01)


def vapour_pressure_to_dewpoint(vapour_pressures_pascals):
    """Converts one or more vapour pressures to dewpoints.

    :param vapour_pressures_pascals: numpy array of vapour pressures (Pa).
    :return: dewpoints_kelvins: equivalent-size numpy array of dewpoints (K).
    """

    error_checking.assert_is_real_numpy_array(vapour_pressures_pascals)

    log_coefficient = (WATER_VAPOUR_GAS_CONSTANT_J_KG01_K01 /
                       LATENT_HEAT_OF_CONDENSATION_J_KG01)
    logarithms = numpy.log(
        vapour_pressures_pascals / BASE_VAPOUR_PRESSURE_PASCALS)

    return (BASE_TEMPERATURE_KELVINS ** -1 - log_coefficient * logarithms) ** -1


def dewpoint_to_vapour_pressure(dewpoints_kelvins):
    """Converts one or more dewpoints to vapour pressures.

    :param dewpoints_kelvins: numpy array of dewpoints (K).
    :return: vapour_pressures_pascals: equivalent-size numpy array of vapour
        pressures (Pa).
    """

    error_checking.assert_is_real_numpy_array(dewpoints_kelvins)

    temperature_coefficient = (LATENT_HEAT_OF_CONDENSATION_J_KG01 /
                               WATER_VAPOUR_GAS_CONSTANT_J_KG01_K01)
    temperature_diffs_kelvins = (
        BASE_TEMPERATURE_KELVINS ** -1 - dewpoints_kelvins ** -1)

    return BASE_VAPOUR_PRESSURE_PASCALS * numpy.exp(
        temperature_coefficient * temperature_diffs_kelvins)


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
    vapour_pressure_dimensions = numpy.asarray(vapour_pressures_pascals.shape)
    error_checking.assert_is_numpy_array(
        total_pressures_pascals, exact_dimensions=vapour_pressure_dimensions)

    return (vapour_pressures_pascals * EPSILON) / (
        total_pressures_pascals - vapour_pressures_pascals)


def mixing_ratio_to_specific_humidity(mixing_ratios_kg_kg01):
    """Converts one or more mixing ratios to specific humidities.

    :param mixing_ratios_kg_kg01: numpy array of mixing ratios (kg per kg).
    :return: specific_humidities_kg_kg01: equivalent-size numpy array of
        specific humidities (kg per kg).
    """

    error_checking.assert_is_real_numpy_array(mixing_ratios_kg_kg01)
    return mixing_ratios_kg_kg01 / (1 + mixing_ratios_kg_kg01)


def specific_humidity_to_dewpoint(specific_humidities_kg_kg01,
                                  total_pressures_pascals):
    """Converts one or more specific humidities to dewpoints.

    :param specific_humidities_kg_kg01: numpy array of specific humidities
        (kg per kg).
    :param total_pressures_pascals: equivalent-size numpy array of total air
        pressures (K).
    :return: dewpoints_kelvins: equivalent-size numpy array of dewpoints (K).
    """

    mixing_ratios_kg_kg01 = specific_humidity_to_mixing_ratio(
        specific_humidities_kg_kg01)
    vapour_pressures_pascals = mixing_ratio_to_vapour_pressure(
        mixing_ratios_kg_kg01, total_pressures_pascals)
    return vapour_pressure_to_dewpoint(vapour_pressures_pascals)


def relative_humidity_to_dewpoint(relative_humidities, temperatures_kelvins):
    """Converts one or more relative humidities to dewpoints.

    :param relative_humidities: numpy array of relative humidities (unitless).
    :param temperatures_kelvins: equivalent-size numpy array of air temperatures
        (K).
    :return: dewpoints_kelvins: equivalent-size numpy array of dewpoints (K).
    """

    error_checking.assert_is_real_numpy_array(relative_humidities)
    error_checking.assert_is_real_numpy_array(temperatures_kelvins)
    rh_dimensions = numpy.asarray(relative_humidities.shape)
    error_checking.assert_is_numpy_array(
        temperatures_kelvins, exact_dimensions=rh_dimensions)

    vapour_pressures_pascals = (
        relative_humidities * dewpoint_to_vapour_pressure(temperatures_kelvins))
    return vapour_pressure_to_dewpoint(vapour_pressures_pascals)
