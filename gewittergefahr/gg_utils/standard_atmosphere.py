"""Calculations based on U.S. Standard Atmosphere."""

import numpy
from gewittergefahr.gg_utils import error_checking

STANDARD_PRESSURES_PASCALS = numpy.array(
    [101325, 22632.1, 5474.89, 868.019, 110.906, 66.9389, 3.95642])
STANDARD_HEIGHTS_M_ASL = numpy.array(
    [0, 11000, 20000, 32000, 47000, 51000, 71000], dtype=float)


def pressure_to_height(pressures_pascals):
    """Converts pressures to heights.

    :param pressures_pascals: numpy array of pressures.
    :return: heights_m_asl: equivalent-size numpy array of heights (metres above
        ground level).
    """

    error_checking.assert_is_greater_numpy_array(pressures_pascals, 0.)

    original_shape = pressures_pascals.shape
    pressures_pascals = numpy.ravel(pressures_pascals)

    num_points = len(pressures_pascals)
    heights_m_asl = numpy.full(num_points, numpy.nan)

    for i in range(len(STANDARD_PRESSURES_PASCALS) + 1):
        if i == 0:
            this_bottom_index = 0
            this_top_index = 1
            this_min_pressure_pascals = STANDARD_PRESSURES_PASCALS[0]
            this_max_pressure_pascals = numpy.inf
        elif i == len(STANDARD_PRESSURES_PASCALS):
            this_bottom_index = -2
            this_top_index = -1
            this_min_pressure_pascals = 0.
            this_max_pressure_pascals = STANDARD_PRESSURES_PASCALS[-1]
        else:
            this_bottom_index = i - 1
            this_top_index = i
            this_min_pressure_pascals = STANDARD_PRESSURES_PASCALS[i]
            this_max_pressure_pascals = STANDARD_PRESSURES_PASCALS[i - 1]

        these_indices = numpy.where(numpy.logical_and(
            pressures_pascals >= this_min_pressure_pascals,
            pressures_pascals < this_max_pressure_pascals
        ))[0]

        if len(these_indices) == 0:
            continue

        this_numerator = (
            STANDARD_HEIGHTS_M_ASL[this_bottom_index] -
            STANDARD_HEIGHTS_M_ASL[this_top_index]
        )
        this_denominator = (
            numpy.log(STANDARD_PRESSURES_PASCALS[this_top_index] /
                      STANDARD_PRESSURES_PASCALS[this_bottom_index])
        )
        this_e_folding_height_metres = this_numerator / this_denominator

        these_logs = numpy.log(
            pressures_pascals[these_indices] /
            STANDARD_PRESSURES_PASCALS[this_bottom_index]
        )
        heights_m_asl[these_indices] = (
            STANDARD_HEIGHTS_M_ASL[this_bottom_index] -
            this_e_folding_height_metres * these_logs
        )

    return numpy.reshape(heights_m_asl, original_shape)


def height_to_pressure(heights_m_agl):
    """Converts heights to pressures.

    :param heights_m_agl: numpy array of heights (metres above ground level).
    :return: pressures_pascals: equivalent-size numpy array of pressures.
    """

    error_checking.assert_is_numpy_array(heights_m_agl)

    original_shape = heights_m_agl.shape
    heights_m_agl = numpy.ravel(heights_m_agl)

    num_points = len(heights_m_agl)
    pressures_pascals = numpy.full(num_points, numpy.nan)

    for i in range(len(STANDARD_HEIGHTS_M_ASL) + 1):
        if i == 0:
            this_bottom_index = 0
            this_top_index = 1
            this_min_height_m_asl = -numpy.inf
            this_max_height_m_asl = STANDARD_HEIGHTS_M_ASL[0]
        elif i == len(STANDARD_PRESSURES_PASCALS):
            this_bottom_index = -2
            this_top_index = -1
            this_min_height_m_asl = STANDARD_HEIGHTS_M_ASL[-1]
            this_max_height_m_asl = numpy.inf
        else:
            this_bottom_index = i - 1
            this_top_index = i
            this_min_height_m_asl = STANDARD_HEIGHTS_M_ASL[i - 1]
            this_max_height_m_asl = STANDARD_HEIGHTS_M_ASL[i]

        these_indices = numpy.where(numpy.logical_and(
            heights_m_agl >= this_min_height_m_asl,
            heights_m_agl < this_max_height_m_asl
        ))[0]

        if len(these_indices) == 0:
            continue

        this_numerator = (
            STANDARD_HEIGHTS_M_ASL[this_bottom_index] -
            STANDARD_HEIGHTS_M_ASL[this_top_index]
        )
        this_denominator = (
            numpy.log(STANDARD_PRESSURES_PASCALS[this_top_index] /
                      STANDARD_PRESSURES_PASCALS[this_bottom_index])
        )
        this_e_folding_height_metres = this_numerator / this_denominator

        these_exponentials = numpy.exp(
            (STANDARD_HEIGHTS_M_ASL[this_bottom_index] -
             heights_m_agl[these_indices]) /
            this_e_folding_height_metres
        )
        pressures_pascals[these_indices] = (
            STANDARD_PRESSURES_PASCALS[this_bottom_index] * these_exponentials)

    return numpy.reshape(pressures_pascals, original_shape)
