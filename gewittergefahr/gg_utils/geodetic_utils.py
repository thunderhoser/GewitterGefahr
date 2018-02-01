"""Methods for geodetic calculations."""

import numpy
import geopy
from geopy.distance import VincentyDistance
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking

RADIANS_TO_DEGREES = 180. / numpy.pi
DEGREES_TO_RADIANS = numpy.pi / 180

MIN_LATITUDE_DEG = -90.
MAX_LATITUDE_DEG = 90.
MIN_LONGITUDE_NEGATIVE_IN_WEST_DEG = -180.
MAX_LONGITUDE_NEGATIVE_IN_WEST_DEG = 180.
MIN_LONGITUDE_POSITIVE_IN_WEST_DEG = 0.
MAX_LONGITUDE_POSITIVE_IN_WEST_DEG = 360.

POSITIVE_LONGITUDE_ARG = 'positive'
NEGATIVE_LONGITUDE_ARG = 'negative'
EITHER_SIGN_LONGITUDE_ARG = 'either'
VALID_LONGITUDE_SIGN_ARGS = [
    POSITIVE_LONGITUDE_ARG, NEGATIVE_LONGITUDE_ARG, EITHER_SIGN_LONGITUDE_ARG]


def find_invalid_latitudes(latitudes_deg):
    """Returns array indices of invalid latitudes.

    :param latitudes_deg: 1-D numpy array of latitudes (deg N).
    :return: invalid_indices: 1-D numpy array with array indices of invalid
        latitudes.
    """

    error_checking.assert_is_real_numpy_array(latitudes_deg)
    error_checking.assert_is_numpy_array(latitudes_deg, num_dimensions=1)

    valid_flags = numpy.logical_and(
        latitudes_deg >= MIN_LATITUDE_DEG, latitudes_deg <= MAX_LATITUDE_DEG)
    return numpy.where(numpy.invert(valid_flags))[0]


def find_invalid_longitudes(
        longitudes_deg, sign_in_western_hemisphere=POSITIVE_LONGITUDE_ARG):
    """Returns array indices of invalid longitudes.

    :param longitudes_deg: 1-D numpy array of longitudes (deg E).
    :param sign_in_western_hemisphere: Required sign in western hemisphere.  May
        be "positive", "negative", or "either".
    :return: invalid_indices: 1-D numpy array with array indices of invalid
        longitudes.
    :raises: ValueError: if `sign_in_western_hemisphere` is not one of the 3
        aforelisted options.
    """

    error_checking.assert_is_real_numpy_array(longitudes_deg)
    error_checking.assert_is_numpy_array(longitudes_deg, num_dimensions=1)
    error_checking.assert_is_string(sign_in_western_hemisphere)

    if sign_in_western_hemisphere == POSITIVE_LONGITUDE_ARG:
        valid_flags = numpy.logical_and(
            longitudes_deg >= MIN_LONGITUDE_POSITIVE_IN_WEST_DEG,
            longitudes_deg <= MAX_LONGITUDE_POSITIVE_IN_WEST_DEG)
    elif sign_in_western_hemisphere == NEGATIVE_LONGITUDE_ARG:
        valid_flags = numpy.logical_and(
            longitudes_deg >= MIN_LONGITUDE_NEGATIVE_IN_WEST_DEG,
            longitudes_deg <= MAX_LONGITUDE_NEGATIVE_IN_WEST_DEG)
    elif sign_in_western_hemisphere == EITHER_SIGN_LONGITUDE_ARG:
        valid_flags = numpy.logical_and(
            longitudes_deg >= MIN_LONGITUDE_NEGATIVE_IN_WEST_DEG,
            longitudes_deg <= MAX_LONGITUDE_POSITIVE_IN_WEST_DEG)
    else:
        error_string = (
            '\n\n{0:s}Valid options for `sign_in_western_hemisphere` are listed'
            ' above and do not include "{1:s}".').format(
                VALID_LONGITUDE_SIGN_ARGS, sign_in_western_hemisphere)
        raise ValueError(error_string)

    return numpy.where(numpy.invert(valid_flags))[0]


def get_latlng_centroid(latitudes_deg, longitudes_deg, allow_nan=True):
    """Finds centroid of lat-long points.

    P = number of points

    :param latitudes_deg: length-P numpy array of latitudes (deg N).
    :param longitudes_deg: length-P numpy array of longitudes (deg E).
    :param allow_nan: Boolean flag.  If True, input arrays may contain NaN's
        (however, NaN's must occur at the exact same positions in the two
        arrays).
    :return: centroid_lat_deg: Centroid latitude (deg N).
    :return: centroid_lng_deg: Centroid longitude (deg E).
    :raises: ValueError: if allow_nan = True but NaN's do not occur at the same
        positions in the two arrays.
    """

    error_checking.assert_is_boolean(allow_nan)
    error_checking.assert_is_valid_lat_numpy_array(latitudes_deg, allow_nan)
    error_checking.assert_is_numpy_array(latitudes_deg, num_dimensions=1)
    num_points = len(latitudes_deg)

    longitudes_deg = lng_conversion.convert_lng_positive_in_west(
        longitudes_deg, allow_nan)
    error_checking.assert_is_numpy_array(
        longitudes_deg, exact_dimensions=numpy.array([num_points]))

    nan_latitude_indices = numpy.where(numpy.isnan(latitudes_deg))[0]
    nan_longitude_indices = numpy.where(numpy.isnan(longitudes_deg))[0]
    if not numpy.array_equal(nan_latitude_indices, nan_longitude_indices):
        error_string = (
            '\nNaN''s occur at the following positions in `latitudes_deg`:\n' +
            str(nan_latitude_indices) +
            '\nand the following positions in `longitudes_deg`:\n' +
            str(nan_longitude_indices) +
            '\nNaN''s should occur at the same positions in the two arrays.')
        raise ValueError(error_string)

    return numpy.nanmean(latitudes_deg), numpy.nanmean(longitudes_deg)


def start_points_and_distances_and_bearings_to_endpoints(
        start_latitudes_deg=None, start_longitudes_deg=None,
        displacements_metres=None, geodetic_bearings_deg=None):
    """Computes endpoint from each start point, displacement, and bearing.

    P = number of start points

    :param start_latitudes_deg: length-P numpy array of beginning latitudes
        (deg N).
    :param start_longitudes_deg: length-P numpy array of beginning longitudes
        (deg E).
    :param displacements_metres: length-P numpy array of displacements.
    :param geodetic_bearings_deg: length-P numpy array of geodetic bearings
        (from start point towards end point, measured clockwise from due north).
    :return: end_latitudes_deg: length-P numpy array of end latitudes (deg N).
    :return: end_longitudes_deg: length-P numpy array of end longitudes (deg E).
    """

    error_checking.assert_is_valid_lat_numpy_array(
        start_latitudes_deg, allow_nan=False)
    error_checking.assert_is_numpy_array(start_latitudes_deg, num_dimensions=1)
    num_points = len(start_latitudes_deg)

    start_longitudes_deg = lng_conversion.convert_lng_positive_in_west(
        start_longitudes_deg, allow_nan=False)
    error_checking.assert_is_numpy_array(
        start_longitudes_deg, exact_dimensions=numpy.array([num_points]))

    error_checking.assert_is_geq_numpy_array(displacements_metres, 0.)
    error_checking.assert_is_numpy_array(
        displacements_metres, exact_dimensions=numpy.array([num_points]))

    error_checking.assert_is_geq_numpy_array(geodetic_bearings_deg, 0.)
    error_checking.assert_is_leq_numpy_array(geodetic_bearings_deg, 360.)
    error_checking.assert_is_numpy_array(
        geodetic_bearings_deg, exact_dimensions=numpy.array([num_points]))

    end_latitudes_deg = numpy.full(num_points, numpy.nan)
    end_longitudes_deg = numpy.full(num_points, numpy.nan)
    for i in range(num_points):
        this_start_point_object = geopy.Point(
            start_latitudes_deg[i], start_longitudes_deg[i])
        this_end_point_object = VincentyDistance(
            meters=displacements_metres[i]).destination(
                this_start_point_object, geodetic_bearings_deg[i])

        end_latitudes_deg[i] = this_end_point_object.latitude
        end_longitudes_deg[i] = this_end_point_object.longitude

    return end_latitudes_deg, lng_conversion.convert_lng_positive_in_west(
        end_longitudes_deg, allow_nan=False)


def xy_components_to_displacements_and_bearings(x_displacements_metres,
                                                y_displacements_metres):
    """For each pair of x- and y-displacement, gets total dsplcmnt and bearing.

    P = number of points

    :param x_displacements_metres: length-P numpy array of eastward
        displacements.
    :param y_displacements_metres: length-P numpy array of northward
        displacements.
    :return: scalar_displacements_metres: length-P numpy array of total
        displacements.
    :return: geodetic_bearings_deg: length-P numpy array of geodetic bearings
        (measured clockwise from due north).
    """

    error_checking.assert_is_numpy_array_without_nan(x_displacements_metres)
    error_checking.assert_is_numpy_array(
        x_displacements_metres, num_dimensions=1)
    num_points = len(x_displacements_metres)

    error_checking.assert_is_numpy_array_without_nan(y_displacements_metres)
    error_checking.assert_is_numpy_array(
        y_displacements_metres, exact_dimensions=numpy.array([num_points]))

    scalar_displacements_metres = numpy.sqrt(
        x_displacements_metres ** 2 + y_displacements_metres ** 2)
    standard_bearings_deg = RADIANS_TO_DEGREES * numpy.arctan2(
        y_displacements_metres, x_displacements_metres)

    return scalar_displacements_metres, standard_to_geodetic_angles(
        standard_bearings_deg)


def displacements_and_bearings_to_xy_components(scalar_displacements_metres,
                                                geodetic_bearings_deg):
    """For each pair of total dsplcmnt and bearing, gets x- and y-displacements.

    P = number of points

    :param scalar_displacements_metres: length-P numpy array of total
        displacements.
    :param geodetic_bearings_deg: length-P numpy array of geodetic bearings
        (measured clockwise from due north).
    :return: x_displacements_metres: length-P numpy array of eastward
        displacements.
    :return: y_displacements_metres: length-P numpy array of northward
        displacements.
    """

    error_checking.assert_is_geq_numpy_array(scalar_displacements_metres, 0.)
    error_checking.assert_is_numpy_array(
        scalar_displacements_metres, num_dimensions=1)
    num_points = len(scalar_displacements_metres)

    error_checking.assert_is_geq_numpy_array(geodetic_bearings_deg, 0.)
    error_checking.assert_is_leq_numpy_array(geodetic_bearings_deg, 360.)
    error_checking.assert_is_numpy_array(
        geodetic_bearings_deg, exact_dimensions=numpy.array([num_points]))

    standard_angles_radians = DEGREES_TO_RADIANS * geodetic_to_standard_angles(
        geodetic_bearings_deg)
    return (scalar_displacements_metres * numpy.cos(standard_angles_radians),
            scalar_displacements_metres * numpy.sin(standard_angles_radians))


def standard_to_geodetic_angles(standard_angles_deg):
    """Converts angles from standard to geodetic format.

    "Standard format" = measured counterclockwise from due east.
    "Geodetic format" = measured clockwise from due north.

    N = number of angles

    :param standard_angles_deg: length-N numpy array of standard angles
        (degrees).
    :return: geodetic_angles_deg: length-N numpy array of geodetic angles
        (degrees).
    """

    error_checking.assert_is_numpy_array_without_nan(standard_angles_deg)
    return numpy.mod((450. - standard_angles_deg), 360.)


def geodetic_to_standard_angles(geodetic_angles_deg):
    """Converts angles from geodetic to standard format.

    "Standard format" = measured counterclockwise from due east.
    "Geodetic format" = measured clockwise from due north.

    N = number of angles

    :param geodetic_angles_deg: length-N numpy array of geodetic angles
        (degrees).
    :return: standard_angles_deg: length-N numpy array of standard angles
        (degrees).
    """

    error_checking.assert_is_numpy_array_without_nan(geodetic_angles_deg)
    return numpy.mod((450. - geodetic_angles_deg), 360.)
