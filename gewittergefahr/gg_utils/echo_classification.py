"""Classification of radar echoes (e.g., convective vs. stratiform).

--- NOTATION ---

The following letters will be used throughout this module.

M = number of rows (unique grid-point latitudes)
N = number of columns (unique grid-point longitudes)
H = number of depths (unique grid-point heights)
"""

import numpy
from scipy.ndimage.filters import median_filter, convolve
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking

TOLERANCE = 1e-6
DEGREES_LAT_TO_METRES = 60 * 1852.

MIN_LATITUDE_KEY = 'min_grid_point_latitude_deg'
LATITUDE_SPACING_KEY = 'latitude_spacing_deg'
MIN_LONGITUDE_KEY = 'min_grid_point_longitude_deg'
LONGITUDE_SPACING_KEY = 'longitude_spacing_deg'
MIN_HEIGHT_KEY = 'min_grid_point_height_m_asl'
HEIGHT_SPACING_KEY = 'height_spacing_metres'

LATITUDES_KEY = 'grid_point_latitudes_deg'
LONGITUDES_KEY = 'grid_point_longitudes_deg'
HEIGHTS_KEY = 'grid_point_heights_m_asl'

MELT_LEVEL_INTERCEPT_BY_MONTH_M_ASL = numpy.array(
    [7072, 7896, 8558, 7988, 7464, 6728, 6080, 6270, 6786, 8670, 8892, 7936],
    dtype=float)
MELT_LEVEL_SLOPE_BY_MONTH_M_DEG01 = numpy.array(
    [-124, -152, -160, -128, -100, -65, -39, -44, -67, -137, -160, -147],
    dtype=float)


def _estimate_melting_levels(latitudes_deg, valid_time_unix_sec):
    """Estimates melting level at each point.

    This estimate is based on linear regression with respect to latitude.  There
    is one set of regression coefficients for each month.

    :param latitudes_deg: numpy array of latitudes (deg N).
    :param valid_time_unix_sec: Valid time.
    :return: melting_levels_m_asl: numpy array of melting levels (metres above
        sea level), with same shape as `latitudes_deg`.
    """

    month_index = int(
        time_conversion.unix_sec_to_string(valid_time_unix_sec, '%m'))

    return (
        MELT_LEVEL_INTERCEPT_BY_MONTH_M_ASL[month_index - 1] +
        MELT_LEVEL_SLOPE_BY_MONTH_M_DEG01[month_index - 1] *
        numpy.absolute(latitudes_deg)
    )


def _neigh_metres_to_rowcol(neigh_radius_metres, grid_metadata_dict):
    """Converts neighbourhood radius from metres to num rows/columns.

    :param neigh_radius_metres: Neighbourhood radius.
    :param grid_metadata_dict: See doc for `_apply_convective_criterion1`.
    :return: num_rows: Number of rows in neighbourhood.
    :return: num_columns: Number of columns in neighbourhood.
    """

    y_spacing_metres = (
        grid_metadata_dict[LATITUDE_SPACING_KEY] * DEGREES_LAT_TO_METRES)
    num_rows = 1 + 2 * int(numpy.ceil(neigh_radius_metres / y_spacing_metres))

    mean_latitude_deg = (
        numpy.max(grid_metadata_dict[LATITUDES_KEY]) -
        numpy.min(grid_metadata_dict[LATITUDES_KEY])
    ) / 2
    mean_x_spacing_metres = (
        grid_metadata_dict[LONGITUDE_SPACING_KEY] *
        DEGREES_LAT_TO_METRES * numpy.cos(numpy.deg2rad(mean_latitude_deg))
    )
    num_columns = 1 + 2 * int(
        numpy.ceil(neigh_radius_metres / mean_x_spacing_metres)
    )

    return num_rows, num_columns


def _get_peakedness(
        reflectivity_matrix_dbz, num_rows_in_neigh, num_columns_in_neigh):
    """Computes peakedness at each voxel (3-D grid point).

    :param reflectivity_matrix_dbz: See doc for `find_convective_pixels`.
    :param num_rows_in_neigh: Number of rows in neighbourhood for median filter.
    :param num_columns_in_neigh: Number of columns in neighbourhood for median
        filter.
    :return: peakedness_matrix_dbz: numpy array of peakedness values, with same
        shape as `reflectivity_matrix_dbz`.
    """

    num_heights = reflectivity_matrix_dbz.shape[-1]
    peakedness_matrix_dbz = numpy.full(reflectivity_matrix_dbz.shape, numpy.nan)

    for k in range(num_heights):
        this_filtered_matrix_dbz = median_filter(
            reflectivity_matrix_dbz[..., k],
            size=(num_rows_in_neigh, num_columns_in_neigh),
            mode='constant', cval=0.)

        peakedness_matrix_dbz[..., k] = (
            reflectivity_matrix_dbz[..., k] - this_filtered_matrix_dbz)

    return peakedness_matrix_dbz


def _get_peakedness_thresholds(reflectivity_matrix_dbz):
    """Computes peakedness threshold at each voxel (3-D grid point).

    :param reflectivity_matrix_dbz: See doc for `find_convective_pixels`.
    :return: peakedness_threshold_matrix_dbz: numpy array of thresholds, with
        same shape as `reflectivity_matrix_dbz`.
    """

    this_matrix = 10. - (reflectivity_matrix_dbz ** 2) / 337.5
    this_matrix[this_matrix < 4.] = 4.
    return this_matrix


def _apply_convective_criterion1(
        reflectivity_matrix_dbz, peakedness_neigh_metres, grid_metadata_dict):
    """Applies criterion 1 for convective classification.

    Criterion 1 states: the pixel is convective if >= 50% of values in the
    column exceed the peakedness threshold.

    :param reflectivity_matrix_dbz: See doc for `find_convective_pixels`.
    :param peakedness_neigh_metres: Same.
    :param grid_metadata_dict: Dictionary with keys listed in doc for
        `find_convective_pixels`, plus the following extras.
    grid_metadata_dict['grid_point_latitudes_deg']: length-M numpy array of
        latitudes (deg N) at grid points.
    grid_metadata_dict['grid_point_longitudes_deg']: length-N numpy array of
        longitudes (deg E) at grid points.

    :return: convective_flag_matrix: M-by-N numpy array of Boolean flags (True
        if convective, False if not).
    """

    num_rows_in_neigh, num_columns_in_neigh = _neigh_metres_to_rowcol(
        neigh_radius_metres=peakedness_neigh_metres,
        grid_metadata_dict=grid_metadata_dict)

    peakedness_matrix_dbz = _get_peakedness(
        reflectivity_matrix_dbz=reflectivity_matrix_dbz,
        num_rows_in_neigh=num_rows_in_neigh,
        num_columns_in_neigh=num_columns_in_neigh)

    peakedness_threshold_matrix_dbz = _get_peakedness_thresholds(
        reflectivity_matrix_dbz)

    fractional_exceedance_matrix = numpy.mean(
        peakedness_matrix_dbz > peakedness_threshold_matrix_dbz, axis=-1)

    return fractional_exceedance_matrix >= 0.5


def _apply_convective_criterion2(
        reflectivity_matrix_dbz, convective_flag_matrix, grid_metadata_dict,
        valid_time_unix_sec, min_composite_refl_above_melting_dbz):
    """Applies criterion 2 for convective classification.

    Criterion 2 states: the pixel is convective if already marked convective OR
    composite reflectivity above melting level >= threshold.

    :param reflectivity_matrix_dbz: See doc for `find_convective_pixels`.
    :param convective_flag_matrix: M-by-N numpy array of Boolean flags (True
        if convective, False if not).
    :param grid_metadata_dict: See doc for `_apply_convective_criterion1`.
    :param valid_time_unix_sec: See doc for `find_convective_pixels`.
    :param min_composite_refl_above_melting_dbz: Same.
    :return: convective_flag_matrix: Updated version of input.
    """

    _, latitude_matrix_deg, height_matrix_m_asl = numpy.meshgrid(
        grid_metadata_dict[LONGITUDES_KEY], grid_metadata_dict[LATITUDES_KEY],
        grid_metadata_dict[HEIGHTS_KEY])

    melting_level_matrix_m_asl = _estimate_melting_levels(
        latitudes_deg=latitude_matrix_deg,
        valid_time_unix_sec=valid_time_unix_sec)

    comp_above_melting_refl_matrix_dbz = numpy.max(
        (height_matrix_m_asl >= melting_level_matrix_m_asl + 1000).astype(int) *
        reflectivity_matrix_dbz,
        axis=-1)

    return numpy.logical_or(
        convective_flag_matrix,
        comp_above_melting_refl_matrix_dbz >=
        min_composite_refl_above_melting_dbz
    )


def _apply_convective_criterion3(
        reflectivity_matrix_dbz, convective_flag_matrix, grid_metadata_dict,
        min_echo_top_m_asl, echo_top_level_dbz):
    """Applies criterion 3 for convective classification.

    Criterion 3 states: the pixel is convective if already marked convective OR
    echo top >= threshold.

    :param reflectivity_matrix_dbz: See doc for `find_convective_pixels`.
    :param convective_flag_matrix: M-by-N numpy array of Boolean flags (True
        if convective, False if not).
    :param grid_metadata_dict: See doc for `_apply_convective_criterion1`.
    :param min_echo_top_m_asl:  See doc for `find_convective_pixels`.
    :param echo_top_level_dbz: Same.
    :return: convective_flag_matrix: Updated version of input.
    """

    _, _, height_matrix_m_asl = numpy.meshgrid(
        grid_metadata_dict[LONGITUDES_KEY], grid_metadata_dict[LATITUDES_KEY],
        grid_metadata_dict[HEIGHTS_KEY])

    echo_top_matrix_m_asl = numpy.max(
        (reflectivity_matrix_dbz >= echo_top_level_dbz).astype(int) *
        height_matrix_m_asl,
        axis=-1)

    return numpy.logical_or(
        convective_flag_matrix, echo_top_matrix_m_asl >= min_echo_top_m_asl)


def _apply_convective_criterion4(convective_flag_matrix):
    """Applies criterion 4 for convective classification.

    Criterion 4 states: if pixel (i, j) is marked convective but none of its
    neighbours are marked convective, (i, j) is not actually convective.

    :param convective_flag_matrix: M-by-N numpy array of Boolean flags (True
        if convective, False if not).
    :return: convective_flag_matrix: Updated version of input.
    """

    weight_matrix = numpy.full((3, 3), 1.)
    weight_matrix = weight_matrix / weight_matrix.size

    average_matrix = convolve(
        convective_flag_matrix.astype(float), weights=weight_matrix,
        mode='constant', cval=0.)

    return numpy.logical_and(
        convective_flag_matrix,
        average_matrix > weight_matrix[0, 0] + TOLERANCE)


def _apply_convective_criterion5(
        reflectivity_matrix_dbz, convective_flag_matrix,
        min_composite_refl_dbz):
    """Applies criterion 5 for convective classification.

    Criterion 5 states: if pixel (i, j) neighbours a pixel marked convective and
    has composite reflectivity >= threshold, pixel (i, j) is convective as well.

    :param reflectivity_matrix_dbz: See doc for `find_convective_pixels`.
    :param convective_flag_matrix: M-by-N numpy array of Boolean flags (True
        if convective, False if not).
    :param min_composite_refl_dbz: See doc for `find_convective_pixels`.
    :return: convective_flag_matrix: Updated version of input.
    """

    weight_matrix = numpy.full((3, 3), 1.)
    weight_matrix = weight_matrix / weight_matrix.size

    average_matrix = convolve(
        convective_flag_matrix.astype(float), weights=weight_matrix,
        mode='constant', cval=0.)

    composite_refl_matrix_dbz = numpy.max(reflectivity_matrix_dbz, axis=-1)
    new_convective_flag_matrix = numpy.logical_and(
        average_matrix > 0.,
        composite_refl_matrix_dbz >= min_composite_refl_dbz)

    return numpy.logical_or(convective_flag_matrix, new_convective_flag_matrix)


def find_convective_pixels(
        reflectivity_matrix_dbz, grid_metadata_dict, valid_time_unix_sec,
        peakedness_neigh_metres, min_echo_top_m_asl, echo_top_level_dbz,
        min_composite_refl_dbz, min_composite_refl_above_melting_dbz):
    """Classifies pixels (horiz grid points) as convective or non-convective.

    :param reflectivity_matrix_dbz: M-by-N-by-H numpy array of reflectivity
        values.  Latitude should increase along the first axis; longitude should
        increase along the second axis; height should increase along the third
        axis.  MAKE SURE NOT TO FLIP YOUR LATITUDES.
    :param grid_metadata_dict: Dictionary with the following keys.
    grid_metadata_dict['min_grid_point_latitude_deg']: Minimum latitude (deg N)
        over all grid points.
    grid_metadata_dict['latitude_spacing_deg']: Spacing (deg N) between grid
        points in adjacent rows.
    grid_metadata_dict['min_grid_point_longitude_deg']: Minimum longitude
        (deg E) over all grid points.
    grid_metadata_dict['longitude_spacing_deg']: Spacing (deg E) between grid
        points in adjacent columns.
    grid_metadata_dict['grid_point_heights_m_asl']: length-H numpy array of
        heights (metres above sea level) at grid points.

    :param valid_time_unix_sec: Valid time.
    :param peakedness_neigh_metres: Neighbourhood radius for peakedness
        calculations (metres), used for criterion 1.
    :param min_echo_top_m_asl: Minimum echo top (metres above sea level), used
        for criterion 3.
    :param echo_top_level_dbz: Critical reflectivity (used to compute echo top
        for criterion 3).
    :param min_composite_refl_dbz: Minimum composite (column-max) reflectivity,
        used for criterion 5.
    :param min_composite_refl_above_melting_dbz: Minimum composite reflectivity
        above melting level, used for criterion 2.
    :return: convective_flag_matrix: M-by-N numpy array of Boolean flags (True
        if convective, False if not).
    """

    # TODO(thunderhoser): May have to handle NaN's in some other way than just
    # setting them to zero.

    # Error-checking.
    error_checking.assert_is_numpy_array(
        reflectivity_matrix_dbz, num_dimensions=3)

    peakedness_neigh_metres = float(peakedness_neigh_metres)
    min_echo_top_m_asl = int(numpy.round(min_echo_top_m_asl))
    echo_top_level_dbz = float(echo_top_level_dbz)
    min_composite_refl_dbz = float(min_composite_refl_dbz)
    min_composite_refl_above_melting_dbz = float(
        min_composite_refl_above_melting_dbz)

    error_checking.assert_is_greater(peakedness_neigh_metres, 0.)
    error_checking.assert_is_greater(min_echo_top_m_asl, 0)
    error_checking.assert_is_greater(echo_top_level_dbz, 0.)
    error_checking.assert_is_greater(min_composite_refl_dbz, 0.)
    error_checking.assert_is_greater(min_composite_refl_above_melting_dbz, 0.)

    grid_point_heights_m_asl = numpy.round(
        grid_metadata_dict[HEIGHTS_KEY]).astype(int)

    error_checking.assert_is_numpy_array(
        grid_point_heights_m_asl, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(grid_point_heights_m_asl, 0)
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(grid_point_heights_m_asl), 0)  # Must be in ascending order.

    # Compute grid-point coordinates.
    reflectivity_matrix_dbz[numpy.isnan(reflectivity_matrix_dbz)] = 0.
    num_rows = reflectivity_matrix_dbz.shape[0]
    num_columns = reflectivity_matrix_dbz.shape[1]

    grid_point_latitudes_deg, grid_point_longitudes_deg = (
        grids.get_latlng_grid_points(
            min_latitude_deg=grid_metadata_dict[MIN_LATITUDE_KEY],
            min_longitude_deg=grid_metadata_dict[MIN_LONGITUDE_KEY],
            lat_spacing_deg=grid_metadata_dict[LATITUDE_SPACING_KEY],
            lng_spacing_deg=grid_metadata_dict[LONGITUDE_SPACING_KEY],
            num_rows=num_rows, num_columns=num_columns)
    )

    grid_metadata_dict[LATITUDES_KEY] = grid_point_latitudes_deg
    grid_metadata_dict[LONGITUDES_KEY] = grid_point_longitudes_deg

    convective_flag_matrix = _apply_convective_criterion1(
        reflectivity_matrix_dbz=reflectivity_matrix_dbz,
        peakedness_neigh_metres=peakedness_neigh_metres,
        grid_metadata_dict=grid_metadata_dict)

    convective_flag_matrix = _apply_convective_criterion2(
        reflectivity_matrix_dbz=reflectivity_matrix_dbz,
        convective_flag_matrix=convective_flag_matrix,
        grid_metadata_dict=grid_metadata_dict,
        valid_time_unix_sec=valid_time_unix_sec,
        min_composite_refl_above_melting_dbz=
        min_composite_refl_above_melting_dbz)

    convective_flag_matrix = _apply_convective_criterion3(
        reflectivity_matrix_dbz=reflectivity_matrix_dbz,
        convective_flag_matrix=convective_flag_matrix,
        grid_metadata_dict=grid_metadata_dict,
        min_echo_top_m_asl=min_echo_top_m_asl,
        echo_top_level_dbz=echo_top_level_dbz)

    convective_flag_matrix = _apply_convective_criterion4(
        convective_flag_matrix)

    return _apply_convective_criterion5(
        reflectivity_matrix_dbz=reflectivity_matrix_dbz,
        convective_flag_matrix=convective_flag_matrix,
        min_composite_refl_dbz=min_composite_refl_dbz)
