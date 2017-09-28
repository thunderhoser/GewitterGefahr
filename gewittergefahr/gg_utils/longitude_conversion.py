"""Methods for longitude conversion.

DEFINITIONS

WH = western hemisphere
"""

import copy
import numpy
from gewittergefahr.gg_utils import error_checking


def convert_lng_negative_in_west(longitudes_deg, allow_nan=True):
    """Converts longitudes so that all WH values are negative.

    In other words, all values in WH are from -180...0 deg E.

    :param longitudes_deg: scalar or numpy array of longitudes (deg E).
    :param: allow_nan: Boolean flag.  If True, NaN longitudes will be allowed.
    :return: longitudes_negative_in_west_deg: Same as input, except that
        longitudes in WH are negative.
    """

    was_input_array = isinstance(longitudes_deg, numpy.ndarray)
    if not was_input_array:
        longitudes_deg = numpy.full(1, longitudes_deg)

    error_checking.assert_is_valid_lng_numpy_array(longitudes_deg,
                                                   allow_nan=allow_nan)

    longitudes_negative_in_west_deg = copy.deepcopy(longitudes_deg)
    longitudes_negative_in_west_deg[
        longitudes_negative_in_west_deg > 180.] -= 360.
    if was_input_array:
        return longitudes_negative_in_west_deg

    return longitudes_negative_in_west_deg[0]


def convert_lng_positive_in_west(longitudes_deg, allow_nan=True):
    """Converts longitudes so that all WH values are positive.

    In other words, all values in WH are from 180...360 deg E.

    :param longitudes_deg: scalar or numpy array of longitudes (deg E).
    :param: allow_nan: Boolean flag.  If True, NaN longitudes will be allowed.
    :return: longitudes_positive_in_west_deg: Same as input, except that
        longitudes in WH are positive.
    """

    was_input_array = isinstance(longitudes_deg, numpy.ndarray)
    if not was_input_array:
        longitudes_deg = numpy.full(1, longitudes_deg)

    error_checking.assert_is_valid_lng_numpy_array(longitudes_deg,
                                                   allow_nan=allow_nan)

    longitudes_positive_in_west_deg = copy.deepcopy(longitudes_deg)
    longitudes_positive_in_west_deg[
        longitudes_positive_in_west_deg < 0.] += 360.
    if was_input_array:
        return longitudes_positive_in_west_deg

    return longitudes_positive_in_west_deg[0]
