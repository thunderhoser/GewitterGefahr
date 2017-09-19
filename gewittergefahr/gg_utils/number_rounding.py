"""Methods for number-rounding."""

import numpy

# TODO(thunderhoser): add error-checking to all methods.


def round_to_nearest(input_value, rounding_base):
    """Rounds numbers to nearest x, where x is any real number.

    :param input_value: Either numpy array of real numbers or scalar real
        number.
    :param rounding_base: Numbers will be rounded to this base.
    :return: output_value: Same as input_value, except rounded.
    """

    return rounding_base * numpy.round(input_value / rounding_base)


def ceiling_to_nearest(input_value, rounding_base):
    """Rounds numbers *up* to nearest x, where x is any real number.

    :param input_value: Either numpy array of real numbers or scalar real
        number.
    :param rounding_base: Numbers will be rounded *up* to this base.
    :return: output_value: Same as input_value, except rounded.
    """

    return rounding_base * numpy.ceil(input_value / rounding_base)


def floor_to_nearest(input_value, rounding_base):
    """Rounds numbers *down* to nearest x, where x is any real number.

    :param input_value: Either numpy array of real numbers or scalar real
        number.
    :param rounding_base: Numbers will be rounded *down* to this base.
    :return: output_value: Same as input_value, except rounded.
    """

    return rounding_base * numpy.floor(input_value / rounding_base)
