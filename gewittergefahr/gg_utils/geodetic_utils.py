"""Methods for geodetic calculations."""

import numpy
import srtm
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


def _get_elevation(latitude_deg, longitude_deg, srtm_data_object=None):
    """Gets elevation at a single point.

    WARNING: Input longitudes in western hemisphere must be negative.

    :param latitude_deg: Latitude (deg N).
    :param longitude_deg: Longitude (deg E).
    :param srtm_data_object: Instance of `srtm.data.GeoElevationData`.
    :return: elevation_m_asl: Elevation (metres above sea level).
    :return: srtm_data_object: Instance of `srtm.data.GeoElevationData`.
    """

    if srtm_data_object is None:
        srtm_data_object = srtm.get_data()

    elevation_m_asl = srtm_data_object.get_elevation(
        latitude=latitude_deg, longitude=longitude_deg)

    # TODO(thunderhoser): I am concerned about this hack.
    if elevation_m_asl is None:
        elevation_m_asl = 0.
    return elevation_m_asl, srtm_data_object


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


def get_elevations(latitudes_deg, longitudes_deg):
    """Returns elevation of each point.

    N = number of points

    :param latitudes_deg: length-N numpy array of latitudes (deg N).
    :param longitudes_deg: length-N numpy array of longitudes (deg E).
    :return: elevations_m_asl: length-N numpy array of elevations (metres above
        sea level).
    """

    error_checking.assert_is_valid_lat_numpy_array(latitudes_deg)
    error_checking.assert_is_numpy_array(latitudes_deg, num_dimensions=1)
    num_points = len(latitudes_deg)

    longitudes_deg = lng_conversion.convert_lng_negative_in_west(
        longitudes_deg, allow_nan=False)
    error_checking.assert_is_numpy_array(
        longitudes_deg, exact_dimensions=numpy.array([num_points]))

    srtm_data_object = None
    elevations_m_asl = numpy.full(num_points, numpy.nan)

    for i in range(num_points):
        elevations_m_asl[i], srtm_data_object = _get_elevation(
            latitude_deg=latitudes_deg[i], longitude_deg=longitudes_deg[i],
            srtm_data_object=srtm_data_object)

    return elevations_m_asl


def start_points_and_displacements_to_endpoints(
        start_latitudes_deg, start_longitudes_deg, scalar_displacements_metres,
        geodetic_bearings_deg):
    """Computes endpoint from each start point and displacement.

    :param start_latitudes_deg: numpy array with latitudes (deg N) of start
        points.
    :param start_longitudes_deg: equivalent-size numpy array with longitudes
        (deg E) of start points.
    :param scalar_displacements_metres: equivalent-size numpy array of scalar
        displacements.
    :param geodetic_bearings_deg: equivalent-size numpy array of geodetic
        bearings (from start point to end point, measured clockwise from due
        north).
    :return: end_latitudes_deg: equivalent-size numpy array with latitudes
        (deg N) of endpoints.
    :return: end_longitudes_deg: equivalent-size numpy array with longitudes
        (deg E) of endpoints.
    """

    error_checking.assert_is_valid_lat_numpy_array(
        start_latitudes_deg, allow_nan=False)

    start_longitudes_deg = lng_conversion.convert_lng_positive_in_west(
        start_longitudes_deg, allow_nan=False)
    error_checking.assert_is_numpy_array(
        start_longitudes_deg,
        exact_dimensions=numpy.array(start_latitudes_deg.shape))

    error_checking.assert_is_geq_numpy_array(scalar_displacements_metres, 0.)
    error_checking.assert_is_numpy_array(
        scalar_displacements_metres,
        exact_dimensions=numpy.array(start_latitudes_deg.shape))

    error_checking.assert_is_geq_numpy_array(geodetic_bearings_deg, 0.)
    error_checking.assert_is_leq_numpy_array(geodetic_bearings_deg, 360.)
    error_checking.assert_is_numpy_array(
        geodetic_bearings_deg,
        exact_dimensions=numpy.array(start_latitudes_deg.shape))

    end_latitudes_deg = numpy.full(start_latitudes_deg.shape, numpy.nan)
    end_longitudes_deg = numpy.full(start_latitudes_deg.shape, numpy.nan)
    num_points = start_latitudes_deg.size

    for i in range(num_points):
        this_start_point_object = geopy.Point(
            start_latitudes_deg.flat[i], start_longitudes_deg.flat[i])
        this_end_point_object = VincentyDistance(
            meters=scalar_displacements_metres.flat[i]).destination(
                this_start_point_object, geodetic_bearings_deg.flat[i])

        end_latitudes_deg.flat[i] = this_end_point_object.latitude
        end_longitudes_deg.flat[i] = this_end_point_object.longitude

    end_longitudes_deg = lng_conversion.convert_lng_positive_in_west(
        end_longitudes_deg, allow_nan=False)
    return end_latitudes_deg, end_longitudes_deg


def xy_to_scalar_displacements_and_bearings(
        x_displacements_metres, y_displacements_metres):
    """For each displacement vector, converts x-y to magnitude and direction.

    :param x_displacements_metres: numpy array of eastward displacements.
    :param y_displacements_metres: equivalent-size numpy array of northward
        displacements.
    :return: scalar_displacements_metres: equivalent-size numpy array of total
        displacements.
    :return: geodetic_bearings_deg: equivalent-size numpy array of geodetic
        bearings (from start point to end point, measured clockwise from due
        north).
    """

    error_checking.assert_is_numpy_array_without_nan(x_displacements_metres)
    error_checking.assert_is_numpy_array_without_nan(y_displacements_metres)
    error_checking.assert_is_numpy_array(
        y_displacements_metres,
        exact_dimensions=numpy.array(y_displacements_metres.shape))

    scalar_displacements_metres = numpy.sqrt(
        x_displacements_metres ** 2 + y_displacements_metres ** 2)
    standard_bearings_deg = RADIANS_TO_DEGREES * numpy.arctan2(
        y_displacements_metres, x_displacements_metres)

    return scalar_displacements_metres, standard_to_geodetic_angles(
        standard_bearings_deg)


def scalar_displacements_and_bearings_to_xy(
        scalar_displacements_metres, geodetic_bearings_deg):
    """For each displacement vector, converts magnitude and direction to x-y.

    :param scalar_displacements_metres: numpy array of total displacements.
    :param geodetic_bearings_deg: equivalent-size numpy array of geodetic
        bearings (from start point to end point, measured clockwise from due
        north).
    :return: x_displacements_metres: equivalent-size numpy array of eastward
        displacements.
    :return: y_displacements_metres: equivalent-size numpy array of northward
        displacements.
    """

    error_checking.assert_is_geq_numpy_array(scalar_displacements_metres, 0.)
    error_checking.assert_is_geq_numpy_array(geodetic_bearings_deg, 0.)
    error_checking.assert_is_leq_numpy_array(geodetic_bearings_deg, 360.)
    error_checking.assert_is_numpy_array(
        geodetic_bearings_deg,
        exact_dimensions=numpy.array(scalar_displacements_metres.shape))

    standard_angles_radians = DEGREES_TO_RADIANS * geodetic_to_standard_angles(
        geodetic_bearings_deg)
    return (scalar_displacements_metres * numpy.cos(standard_angles_radians),
            scalar_displacements_metres * numpy.sin(standard_angles_radians))


def rotate_displacement_vectors(
        x_displacements_metres, y_displacements_metres, ccw_rotation_angle_deg):
    """Rotates each displacement vector by a certain angle.

    :param x_displacements_metres: numpy array of eastward displacements.
    :param y_displacements_metres: equivalent-size numpy array of northward
        displacements.
    :param ccw_rotation_angle_deg: Rotation angle (degrees).  Each displacement
        vector will be rotated counterclockwise by this amount.
    :return: x_prime_displacements_metres: equivalent-size numpy array of
        "eastward" displacements (in the rotated coordinate system).
    :return: y_prime_displacements_metres: equivalent-size numpy array of
        "northward" displacements (in the rotated coordinate system).
    """

    error_checking.assert_is_numpy_array_without_nan(x_displacements_metres)
    error_checking.assert_is_numpy_array_without_nan(y_displacements_metres)
    error_checking.assert_is_numpy_array(
        y_displacements_metres,
        exact_dimensions=numpy.array(y_displacements_metres.shape))
    error_checking.assert_is_greater(ccw_rotation_angle_deg, -360.)
    error_checking.assert_is_less_than(ccw_rotation_angle_deg, 360.)

    ccw_rotation_angle_rad = DEGREES_TO_RADIANS * ccw_rotation_angle_deg
    rotation_matrix = numpy.array([
        [numpy.cos(ccw_rotation_angle_rad), -numpy.sin(ccw_rotation_angle_rad)],
        [numpy.sin(ccw_rotation_angle_rad), numpy.cos(ccw_rotation_angle_rad)]
    ])

    x_prime_displacements_metres = numpy.full(
        x_displacements_metres.shape, numpy.nan)
    y_prime_displacements_metres = numpy.full(
        x_displacements_metres.shape, numpy.nan)
    num_points = x_prime_displacements_metres.size

    for i in range(num_points):
        this_vector = numpy.transpose(numpy.array(
            [x_displacements_metres.flat[i], y_displacements_metres.flat[i]]))
        this_vector = numpy.matmul(rotation_matrix, this_vector)
        x_prime_displacements_metres.flat[i] = this_vector[0]
        y_prime_displacements_metres.flat[i] = this_vector[1]

    return x_prime_displacements_metres, y_prime_displacements_metres


def standard_to_geodetic_angles(standard_angles_deg):
    """Converts angles from standard to geodetic format.

    "Standard format" = measured counterclockwise from due east
    "Geodetic format" = measured clockwise from due north

    :param standard_angles_deg: numpy array of standard angles (degrees).
    :return: geodetic_angles_deg: equivalent-size numpy array of geodetic
        angles.
    """

    error_checking.assert_is_numpy_array_without_nan(standard_angles_deg)
    return numpy.mod((450. - standard_angles_deg), 360.)


def geodetic_to_standard_angles(geodetic_angles_deg):
    """Converts angles from geodetic to standard format.

    For the definitions of "geodetic format" and "standard format," see doc for
    `standard_to_geodetic_angles`.

    :param geodetic_angles_deg: numpy array of geodetic angles (degrees).
    :return: standard_angles_deg: equivalent-size numpy array of standard
        angles.
    """

    error_checking.assert_is_numpy_array_without_nan(geodetic_angles_deg)
    return numpy.mod((450. - geodetic_angles_deg), 360.)
