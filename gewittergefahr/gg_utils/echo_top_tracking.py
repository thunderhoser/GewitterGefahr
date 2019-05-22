"""Implements the echo-top-based storm-tracking algorithm.

This algorithm is discussed in Section 3c of Homeyer et al. (2017).  The main
advantage of this algorithm (in my experience) over segmotion (Lakshmanan and
Smith 2010) is that it provides more intuitive and longer storm tracks.  The
main disadvantage of the echo-top-based algorithm (in my experience) is that it
provides only storm centers, not objects.  In other words, the echo-top-based
algorithm does not provide the bounding polygons.

--- REFERENCES ---

Haberlie, A. and W. Ashley, 2018: "A method for identifying midlatitude
    mesoscale convective systems in radar mosaics, part II: Tracking". Journal
    of Applied Meteorology and Climatology, in press,
    doi:10.1175/JAMC-D-17-0294.1.

Homeyer, C.R., and J.D. McAuliffe, and K.M. Bedka, 2017: "On the development of
    above-anvil cirrus plumes in extratropical convection". Journal of the
    Atmospheric Sciences, 74 (5), 1617-1633.

Lakshmanan, V., and T. Smith, 2010: "Evaluating a storm tracking algorithm".
    26th Conference on Interactive Information Processing Systems, Atlanta, GA,
    American Meteorological Society.
"""

import copy
import os.path
import warnings
from itertools import chain
import numpy
import pandas
from scipy.ndimage.filters import gaussian_filter
from skimage.measure import label as label_image
from gewittergefahr.gg_io import myrorss_and_mrms_io
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import temporal_tracking
from gewittergefahr.gg_utils import track_reanalysis
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import radar_sparse_to_full as radar_s2f
from gewittergefahr.gg_utils import dilation
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import geodetic_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import echo_classification as echo_classifn
from gewittergefahr.gg_utils import error_checking

TOLERANCE = 1e-6

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

RADIANS_TO_DEGREES = 180. / numpy.pi
DEGREES_LAT_TO_METRES = 60 * 1852
CENTRAL_PROJ_LATITUDE_DEG = 35.
CENTRAL_PROJ_LONGITUDE_DEG = 265.

VALID_RADAR_FIELD_NAMES = [
    radar_utils.ECHO_TOP_15DBZ_NAME, radar_utils.ECHO_TOP_18DBZ_NAME,
    radar_utils.ECHO_TOP_20DBZ_NAME, radar_utils.ECHO_TOP_25DBZ_NAME,
    radar_utils.ECHO_TOP_40DBZ_NAME, radar_utils.ECHO_TOP_50DBZ_NAME
]

VALID_RADAR_SOURCE_NAMES = [
    radar_utils.MYRORSS_SOURCE_ID, radar_utils.MRMS_SOURCE_ID
]

DEFAULT_MIN_ECHO_TOP_KM = 4.
DEFAULT_SMOOTHING_RADIUS_DEG_LAT = 0.024
DEFAULT_HALF_WIDTH_FOR_MAX_FILTER_DEG_LAT = 0.06

DEFAULT_MIN_INTERMAX_DISTANCE_METRES = 0.1 * DEGREES_LAT_TO_METRES
DEFAULT_MIN_SIZE_PIXELS = 0
DEFAULT_MAX_LINK_TIME_SECONDS = 360
DEFAULT_MAX_VELOCITY_DIFF_M_S01 = 30.
DEFAULT_MAX_LINK_DISTANCE_M_S01 = (
    0.125 * DEGREES_LAT_TO_METRES / DEFAULT_MAX_LINK_TIME_SECONDS
)

DEFAULT_MAX_JOIN_TIME_SEC = 720
DEFAULT_MAX_JOIN_ERROR_M_S01 = 30.
DEFAULT_MIN_REANALYZED_DURATION_SEC = 1

DUMMY_TRACKING_SCALE_METRES2 = int(numpy.round(numpy.pi * 1e8))  # 10-km radius

MAX_VALUES_KEY = 'max_values'


def _check_radar_field(radar_field_name):
    """Error-checks radar field.

    :param radar_field_name: Field name (string).
    :raises: ValueError: if `radar_field_name not in VALID_RADAR_FIELD_NAMES`.
    """

    error_checking.assert_is_string(radar_field_name)

    if radar_field_name not in VALID_RADAR_FIELD_NAMES:
        error_string = (
            '\n{0:s}\nValid radar fields (listed above) do not include "{1:s}".'
        ).format(str(VALID_RADAR_FIELD_NAMES), radar_field_name)

        raise ValueError(error_string)


def _check_radar_source(radar_source_name):
    """Error-checks source of radar data.

    :param radar_source_name: Data source (string).
    :raises: ValueError: if `radar_source_name not in VALID_RADAR_SOURCE_NAMES`.
    """

    error_checking.assert_is_string(radar_source_name)

    if radar_source_name not in VALID_RADAR_SOURCE_NAMES:
        error_string = (
            '\n{0:s}\nValid radar sources (listed above) do not include '
            '"{1:s}".'
        ).format(str(VALID_RADAR_SOURCE_NAMES), radar_source_name)

        raise ValueError(error_string)


def _gaussian_smooth_radar_field(radar_matrix, e_folding_radius_pixels,
                                 cutoff_radius_pixels=None):
    """Applies Gaussian smoother to radar field.  NaN's are treated as zero.

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)

    :param radar_matrix: M-by-N numpy array of data values.
    :param e_folding_radius_pixels: e-folding radius.
    :param cutoff_radius_pixels: Cutoff radius.  If
        `cutoff_radius_pixels is None`, will default to
        `3 * e_folding_radius_pixels`.
    :return: smoothed_radar_matrix: Smoothed version of input.
    """

    e_folding_radius_pixels = float(e_folding_radius_pixels)
    if cutoff_radius_pixels is None:
        cutoff_radius_pixels = 3 * e_folding_radius_pixels

    radar_matrix[numpy.isnan(radar_matrix)] = 0.

    smoothed_radar_matrix = gaussian_filter(
        input=radar_matrix, sigma=e_folding_radius_pixels, order=0,
        mode='constant', cval=0.,
        truncate=cutoff_radius_pixels / e_folding_radius_pixels)

    smoothed_radar_matrix[
        numpy.absolute(smoothed_radar_matrix) < TOLERANCE
    ] = numpy.nan

    return smoothed_radar_matrix


def _find_local_maxima(radar_matrix, radar_metadata_dict,
                       neigh_half_width_pixels):
    """Finds local maxima in radar field.

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)
    P = number of local maxima

    :param radar_matrix: M-by-N numpy array of data values.
    :param radar_metadata_dict: Dictionary created by
        `myrorss_and_mrms_io.read_metadata_from_raw_file`.
    :param neigh_half_width_pixels: Half-width of neighbourhood for max filter.
    :return: local_max_dict_simple: Dictionary with the following keys.
    local_max_dict_simple['latitudes_deg']: length-P numpy array with latitudes
        (deg N) of local maxima.
    local_max_dict_simple['longitudes_deg']: length-P numpy array with
        longitudes (deg E) of local maxima.
    local_max_dict_simple['max_values']: length-P numpy array with magnitudes of
        local maxima.
    """

    filtered_radar_matrix = dilation.dilate_2d_matrix(
        input_matrix=radar_matrix, percentile_level=100.,
        half_width_in_pixels=neigh_half_width_pixels)

    max_index_arrays = numpy.where(
        numpy.absolute(filtered_radar_matrix - radar_matrix) < TOLERANCE
    )

    max_row_indices = max_index_arrays[0]
    max_column_indices = max_index_arrays[1]

    max_latitudes_deg, max_longitudes_deg = radar_utils.rowcol_to_latlng(
        grid_rows=max_row_indices, grid_columns=max_column_indices,
        nw_grid_point_lat_deg=
        radar_metadata_dict[radar_utils.NW_GRID_POINT_LAT_COLUMN],
        nw_grid_point_lng_deg=
        radar_metadata_dict[radar_utils.NW_GRID_POINT_LNG_COLUMN],
        lat_spacing_deg=radar_metadata_dict[radar_utils.LAT_SPACING_COLUMN],
        lng_spacing_deg=radar_metadata_dict[radar_utils.LNG_SPACING_COLUMN]
    )

    max_values = radar_matrix[max_row_indices, max_column_indices]

    sort_indices = numpy.argsort(-max_values)
    max_values = max_values[sort_indices]
    max_latitudes_deg = max_latitudes_deg[sort_indices]
    max_longitudes_deg = max_longitudes_deg[sort_indices]

    return {
        temporal_tracking.LATITUDES_KEY: max_latitudes_deg,
        temporal_tracking.LONGITUDES_KEY: max_longitudes_deg,
        MAX_VALUES_KEY: max_values
    }


def _remove_redundant_local_maxima(local_max_dict, projection_object,
                                   min_intermax_distance_metres):
    """Removes redundant local maxima at one time.

    P = number of local maxima retained

    :param local_max_dict: Dictionary with at least the following keys.
    local_max_dict['latitudes_deg']: See doc for `_find_local_maxima`.
    local_max_dict['longitudes_deg']: Same.
    local_max_dict['max_values']: Same.

    :param projection_object: Instance of `pyproj.Proj` (used to convert local
        maxima from lat-long to x-y coordinates).
    :param min_intermax_distance_metres: Minimum distance between any pair of
        local maxima.
    :return: local_max_dict: Same as input, except that no pair of maxima is
        within `min_intermax_distance_metres`.  Also contains additional columns
        listed below.
    local_max_dict['x_coords_metres']: length-P numpy array with x-coordinates
        of local maxima.
    local_max_dict['y_coords_metres']: length-P numpy array with y-coordinates
        of local maxima.
    """

    x_coords_metres, y_coords_metres = projections.project_latlng_to_xy(
        latitudes_deg=local_max_dict[temporal_tracking.LATITUDES_KEY],
        longitudes_deg=local_max_dict[temporal_tracking.LONGITUDES_KEY],
        projection_object=projection_object,
        false_easting_metres=0., false_northing_metres=0.)

    local_max_dict.update({
        temporal_tracking.X_COORDS_KEY: x_coords_metres,
        temporal_tracking.Y_COORDS_KEY: y_coords_metres
    })

    num_maxima = len(x_coords_metres)
    keep_max_flags = numpy.full(num_maxima, True, dtype=bool)

    for i in range(num_maxima):
        if not keep_max_flags[i]:
            continue

        these_distances_metres = numpy.sqrt(
            (x_coords_metres - x_coords_metres[i]) ** 2 +
            (y_coords_metres - y_coords_metres[i]) ** 2
        )

        these_redundant_indices = numpy.where(numpy.logical_and(
            these_distances_metres < min_intermax_distance_metres,
            keep_max_flags
        ))[0]

        if len(these_redundant_indices) == 1:
            continue

        keep_max_flags[these_redundant_indices] = False

        this_best_index = numpy.argmax(
            local_max_dict[MAX_VALUES_KEY][these_redundant_indices]
        )
        this_best_index = these_redundant_indices[this_best_index]
        keep_max_flags[this_best_index] = True

    indices_to_keep = numpy.where(keep_max_flags)[0]

    for this_key in local_max_dict:
        if isinstance(local_max_dict[this_key], list):
            local_max_dict[this_key] = [
                local_max_dict[this_key][k] for k in indices_to_keep
            ]
        elif isinstance(local_max_dict[this_key], numpy.ndarray):
            local_max_dict[this_key] = local_max_dict[this_key][
                indices_to_keep]

    # x_coords_metres, y_coords_metres = projections.project_latlng_to_xy(
    #     latitudes_deg=local_max_dict[temporal_tracking.LATITUDES_KEY],
    #     longitudes_deg=local_max_dict[temporal_tracking.LONGITUDES_KEY],
    #     projection_object=projection_object,
    #     false_easting_metres=0., false_northing_metres=0.)
    #
    # coord_matrix_metres = numpy.hstack((
    #     numpy.reshape(x_coords_metres, (x_coords_metres.size, 1)),
    #     numpy.reshape(y_coords_metres, (y_coords_metres.size, 1))
    # ))
    #
    # distance_matrix_metres = euclidean_distances(
    #     X=coord_matrix_metres, Y=coord_matrix_metres)
    #
    # for i in range(len(x_coords_metres)):
    #     distance_matrix_metres[i, i] = numpy.inf
    #
    # these_rows, these_columns = numpy.where(
    #     distance_matrix_metres < min_intermax_distance_metres)
    #
    # for i in range(len(these_rows)):
    #     print (
    #         '{0:d}th max (at {1:.2f} deg N and {2:.2f} deg E) and {3:d}th max '
    #         '(at {4:.2f} deg N and {5:.2f} deg E) are within {6:.1f} metres'
    #     ).format(
    #         these_rows[i],
    #         local_max_dict[temporal_tracking.LATITUDES_KEY][these_rows[i]],
    #         local_max_dict[temporal_tracking.LONGITUDES_KEY][these_rows[i]],
    #         these_columns[i],
    #         local_max_dict[temporal_tracking.LATITUDES_KEY][these_columns[i]],
    #         local_max_dict[temporal_tracking.LONGITUDES_KEY][these_columns[i]],
    #         distance_matrix_metres[these_rows[i], these_columns[i]]
    #     )

    return local_max_dict


def _check_time_period(
        first_spc_date_string, last_spc_date_string, first_time_unix_sec,
        last_time_unix_sec):
    """Error-checks time period.

    :param first_spc_date_string: First SPC date in period (format "yyyymmdd").
    :param last_spc_date_string: Last SPC date in period.
    :param first_time_unix_sec: First time in period.  If
        `first_time_unix_sec is None`, defaults to first time on first SPC date.
    :param last_time_unix_sec: Last time in period.  If
        `last_time_unix_sec is None`, defaults to last time on last SPC date.
    :return: spc_date_strings: 1-D list of SPC dates (format "yyyymmdd").
    :return: first_time_unix_sec: Same as input, but may have been replaced with
        default.
    :return: last_time_unix_sec: Same as input, but may have been replaced with
        default.
    """

    spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string)

    if first_time_unix_sec is None:
        first_time_unix_sec = time_conversion.string_to_unix_sec(
            first_spc_date_string, time_conversion.SPC_DATE_FORMAT
        ) + time_conversion.MIN_SECONDS_INTO_SPC_DATE

    if last_time_unix_sec is None:
        last_time_unix_sec = time_conversion.string_to_unix_sec(
            last_spc_date_string, time_conversion.SPC_DATE_FORMAT
        ) + time_conversion.MAX_SECONDS_INTO_SPC_DATE

    error_checking.assert_is_greater(last_time_unix_sec, first_time_unix_sec)

    assert time_conversion.is_time_in_spc_date(
        first_time_unix_sec, first_spc_date_string)
    assert time_conversion.is_time_in_spc_date(
        last_time_unix_sec, last_spc_date_string)

    return spc_date_strings, first_time_unix_sec, last_time_unix_sec


def _find_input_radar_files(
        top_radar_dir_name, radar_field_name, radar_source_name,
        first_spc_date_string, last_spc_date_string, first_time_unix_sec,
        last_time_unix_sec):
    """Finds radar files (inputs to `run_tracking` -- basically main method).

    T = number of files found

    :param top_radar_dir_name: Name of top-level directory with radar files.
        Files therein will be found by
        `myrorss_and_mrms_io.find_raw_files_one_spc_date`.
    :param radar_field_name: Field name (must be accepted by
        `_check_radar_field`).
    :param radar_source_name: Data source (must be accepted by
        `_check_radar_source`).
    :param first_spc_date_string: See doc for `_check_time_period`.
    :param last_spc_date_string: Same.
    :param first_time_unix_sec: Same.
    :param last_time_unix_sec: Same.
    :return: radar_file_names: length-T list of paths to radar files.
    :return: valid_times_unix_sec: length-T numpy array of valid times.
    """

    _check_radar_field(radar_field_name)
    _check_radar_source(radar_source_name)

    spc_date_strings, first_time_unix_sec, last_time_unix_sec = (
        _check_time_period(
            first_spc_date_string=first_spc_date_string,
            last_spc_date_string=last_spc_date_string,
            first_time_unix_sec=first_time_unix_sec,
            last_time_unix_sec=last_time_unix_sec)
    )

    radar_file_names = []
    valid_times_unix_sec = numpy.array([], dtype=int)
    num_spc_dates = len(spc_date_strings)

    for i in range(num_spc_dates):
        these_file_names = myrorss_and_mrms_io.find_raw_files_one_spc_date(
            spc_date_string=spc_date_strings[i],
            field_name=radar_field_name, data_source=radar_source_name,
            top_directory_name=top_radar_dir_name, raise_error_if_missing=False)

        if len(these_file_names) == 0:
            continue

        if i == 0:
            this_first_time_unix_sec = first_time_unix_sec + 0
        else:
            this_first_time_unix_sec = time_conversion.get_start_of_spc_date(
                spc_date_strings[i])

        if i == num_spc_dates - 1:
            this_last_time_unix_sec = last_time_unix_sec + 0
        else:
            this_last_time_unix_sec = time_conversion.get_end_of_spc_date(
                spc_date_strings[i])

        these_times_unix_sec = numpy.array([
            myrorss_and_mrms_io.raw_file_name_to_time(f)
            for f in these_file_names
        ], dtype=int)

        good_indices = numpy.where(numpy.logical_and(
            these_times_unix_sec >= this_first_time_unix_sec,
            these_times_unix_sec <= this_last_time_unix_sec
        ))[0]

        radar_file_names += [these_file_names[k] for k in good_indices]
        valid_times_unix_sec = numpy.concatenate((
            valid_times_unix_sec, these_times_unix_sec[good_indices]
        ))

    sort_indices = numpy.argsort(valid_times_unix_sec)
    valid_times_unix_sec = valid_times_unix_sec[sort_indices]
    radar_file_names = [radar_file_names[k] for k in sort_indices]

    return radar_file_names, valid_times_unix_sec


def _find_input_tracking_files(
        top_tracking_dir_name, first_spc_date_string, last_spc_date_string,
        first_time_unix_sec, last_time_unix_sec):
    """Finds tracking files (inputs to `reanalyze_across_spc_dates`).

    T = number of SPC dates

    :param top_tracking_dir_name: Name of top-level directory with tracking
        files.  Files therein will be found by
        `storm_tracking_io.find_files_one_spc_date`.
    :param first_spc_date_string: See doc for `_check_time_period`.
    :param last_spc_date_string: Same.
    :param first_time_unix_sec: Same.
    :param last_time_unix_sec: Same.
    :return: spc_date_strings: length-T list of SPC dates (format "yyyymmdd").
    :return: tracking_file_names_by_date: length-T list, where the [i]th element
        is a 1-D list of paths to tracking files for the [i]th date.
    :return: valid_times_by_date_unix_sec: length-T list, where the [i]th
        element is a 1-D numpy array of valid times for the [i]th date.
    """

    spc_date_strings, first_time_unix_sec, last_time_unix_sec = (
        _check_time_period(
            first_spc_date_string=first_spc_date_string,
            last_spc_date_string=last_spc_date_string,
            first_time_unix_sec=first_time_unix_sec,
            last_time_unix_sec=last_time_unix_sec)
    )

    num_spc_dates = len(spc_date_strings)
    tracking_file_names_by_date = [['']] * num_spc_dates
    valid_times_by_date_unix_sec = [numpy.array([], dtype=int)] * num_spc_dates

    keep_date_indices = []

    for i in range(num_spc_dates):
        these_file_names = tracking_io.find_files_one_spc_date(
            spc_date_string=spc_date_strings[i],
            source_name=tracking_utils.SEGMOTION_NAME,
            top_tracking_dir_name=top_tracking_dir_name,
            tracking_scale_metres2=DUMMY_TRACKING_SCALE_METRES2,
            raise_error_if_missing=False
        )[0]

        if len(these_file_names) == 0:
            tracking_file_names_by_date[i] = []
            continue

        keep_date_indices.append(i)

        if i == 0:
            this_first_time_unix_sec = first_time_unix_sec + 0
        else:
            this_first_time_unix_sec = time_conversion.get_start_of_spc_date(
                spc_date_strings[i])

        if i == num_spc_dates - 1:
            this_last_time_unix_sec = last_time_unix_sec + 0
        else:
            this_last_time_unix_sec = time_conversion.get_end_of_spc_date(
                spc_date_strings[i])

        these_times_unix_sec = numpy.array([
            tracking_io.file_name_to_time(f) for f in these_file_names
        ], dtype=int)

        sort_indices = numpy.argsort(these_times_unix_sec)
        these_file_names = [these_file_names[k] for k in sort_indices]
        these_times_unix_sec = these_times_unix_sec[sort_indices]

        good_indices = numpy.where(numpy.logical_and(
            these_times_unix_sec >= this_first_time_unix_sec,
            these_times_unix_sec <= this_last_time_unix_sec
        ))[0]

        tracking_file_names_by_date[i] = [
            these_file_names[k] for k in good_indices
        ]
        valid_times_by_date_unix_sec[i] = these_times_unix_sec[good_indices]

    spc_date_strings = [spc_date_strings[i] for i in keep_date_indices]
    tracking_file_names_by_date = [
        tracking_file_names_by_date[i] for i in keep_date_indices
    ]
    valid_times_by_date_unix_sec = [
        valid_times_by_date_unix_sec[i] for i in keep_date_indices
    ]

    return (spc_date_strings, tracking_file_names_by_date,
            valid_times_by_date_unix_sec)


def _get_grid_points_in_storm(
        centroid_latitude_deg, centroid_longitude_deg, grid_point_latitudes_deg,
        grid_point_longitudes_deg, echo_top_matrix_km, min_echo_top_km,
        min_intermax_distance_metres):
    """Converts local max (one point) to list of grid points in storm.

    M = number of rows in radar grid
    N = number of columns in radar grid
    P = number of grid points in storm

    :param centroid_latitude_deg: Latitude (deg N) of storm centroid.
    :param centroid_longitude_deg: Longitude (deg E) of storm centroid.
    :param grid_point_latitudes_deg: length-M numpy array of grid-point
        latitudes (deg N).
    :param grid_point_longitudes_deg: length-N numpy array of grid-point
        longitudes (deg E).
    :param echo_top_matrix_km: M-by-N numpy array of echo tops.
    :param min_echo_top_km: Minimum echo top used to define storms.
    :param min_intermax_distance_metres: Minimum distance between local maxima
        (storm centroids).
    :return: grid_rows: length-P numpy array of row indices.
    :return: grid_columns: length-P numpy array of column indices.
    """

    echo_top_submatrix_km, row_offsets, column_offsets = (
        grids.extract_latlng_subgrid(
            data_matrix=echo_top_matrix_km,
            grid_point_latitudes_deg=grid_point_latitudes_deg,
            grid_point_longitudes_deg=grid_point_longitudes_deg,
            center_latitude_deg=centroid_latitude_deg,
            center_longitude_deg=centroid_longitude_deg,
            max_distance_from_center_metres=min_intermax_distance_metres)
    )

    echo_top_submatrix_km[numpy.isnan(echo_top_submatrix_km)] = 0.

    region_id_submatrix = label_image(
        echo_top_submatrix_km >= min_echo_top_km, connectivity=2
    )

    centroid_subrow = numpy.argmin(numpy.absolute(
        centroid_latitude_deg - grid_point_latitudes_deg[row_offsets]
    ))

    centroid_subcolumn = numpy.argmin(numpy.absolute(
        centroid_longitude_deg - grid_point_longitudes_deg[column_offsets]
    ))

    centroid_region_id = region_id_submatrix[
        centroid_subrow, centroid_subcolumn
    ]

    if centroid_region_id == 0:
        grid_point_subrows = numpy.array([centroid_subrow], dtype=int)
        grid_point_subcolumns = numpy.array([centroid_subcolumn], dtype=int)
    else:
        grid_point_subrows, grid_point_subcolumns = numpy.where(
            region_id_submatrix == centroid_region_id)

    return (
        grid_point_subrows + row_offsets[0],
        grid_point_subcolumns + column_offsets[0]
    )


def _local_maxima_to_polygons(
        local_max_dict, echo_top_matrix_km, min_echo_top_km,
        radar_metadata_dict, min_intermax_distance_metres):
    """Converts local maxima at one time from points to polygons.

    M = number of rows in grid (unique grid-point latitudes)
    N = number of columns in grid (unique grid-point longitudes)
    P = number of local maxima
    G_i = number of grid points in the [i]th polygon

    :param local_max_dict: Dictionary with the following keys.
    local_max_dict['latitudes_deg']: length-P numpy array of latitudes (deg N).
    local_max_dict['longitudes_deg']: length-P numpy array of longitudes
        (deg E).

    :param echo_top_matrix_km: M-by-N numpy array of echo tops (km above ground
        or sea level).
    :param min_echo_top_km: Minimum echo top (smaller values are not considered
        local maxima).
    :param radar_metadata_dict: Dictionary created by
        `myrorss_and_mrms_io.read_metadata_from_raw_file`.
    :param min_intermax_distance_metres: Minimum distance between any pair of
        local maxima.

    :return: local_max_dict: Same as input but with the following extra columns.
    local_max_dict['grid_point_rows_array_list']: length-P list, where the [i]th
        element is a numpy array (length G_i) with row indices of grid points in
        the [i]th polygon.
    local_max_dict['grid_point_columns_array_list']: Same but for columns.
    local_max_dict['grid_point_lats_array_list_deg']: Same but for latitudes
        (deg N).
    local_max_dict['grid_point_lngs_array_list_deg']: Same but for longitudes
        (deg E).
    local_max_dict['polygon_objects_rowcol']: length-P list of polygons
        (`shapely.geometry.Polygon` objects) with coordinates in row-column
        space.
    local_max_dict['polygon_objects_latlng']: length-P list of polygons
        (`shapely.geometry.Polygon` objects) with coordinates in lat-long space.
    """

    latitude_extent_deg = (
        radar_metadata_dict[radar_utils.LAT_SPACING_COLUMN] *
        (radar_metadata_dict[radar_utils.NUM_LAT_COLUMN] - 1)
    )

    min_grid_point_latitude_deg = (
        radar_metadata_dict[radar_utils.NW_GRID_POINT_LAT_COLUMN] -
        latitude_extent_deg
    )

    grid_point_latitudes_deg, grid_point_longitudes_deg = (
        grids.get_latlng_grid_points(
            min_latitude_deg=min_grid_point_latitude_deg,
            min_longitude_deg=radar_metadata_dict[
                radar_utils.NW_GRID_POINT_LNG_COLUMN],
            lat_spacing_deg=radar_metadata_dict[radar_utils.LAT_SPACING_COLUMN],
            lng_spacing_deg=radar_metadata_dict[radar_utils.LNG_SPACING_COLUMN],
            num_rows=radar_metadata_dict[radar_utils.NUM_LAT_COLUMN],
            num_columns=radar_metadata_dict[radar_utils.NUM_LNG_COLUMN])
    )

    grid_point_latitudes_deg = grid_point_latitudes_deg[::-1]
    num_maxima = len(local_max_dict[temporal_tracking.LATITUDES_KEY])

    local_max_dict[temporal_tracking.GRID_POINT_ROWS_KEY] = [[]] * num_maxima
    local_max_dict[temporal_tracking.GRID_POINT_COLUMNS_KEY] = [[]] * num_maxima
    local_max_dict[
        temporal_tracking.GRID_POINT_LATITUDES_KEY] = [[]] * num_maxima
    local_max_dict[
        temporal_tracking.GRID_POINT_LONGITUDES_KEY] = [[]] * num_maxima

    local_max_dict[temporal_tracking.POLYGON_OBJECTS_ROWCOL_KEY] = numpy.full(
        num_maxima, numpy.nan, dtype=object)
    local_max_dict[temporal_tracking.POLYGON_OBJECTS_LATLNG_KEY] = numpy.full(
        num_maxima, numpy.nan, dtype=object)

    for i in range(num_maxima):
        (local_max_dict[temporal_tracking.GRID_POINT_ROWS_KEY][i],
         local_max_dict[temporal_tracking.GRID_POINT_COLUMNS_KEY][i]
        ) = _get_grid_points_in_storm(
            centroid_latitude_deg=local_max_dict[
                temporal_tracking.LATITUDES_KEY][i],
            centroid_longitude_deg=local_max_dict[
                temporal_tracking.LONGITUDES_KEY][i],
            grid_point_latitudes_deg=grid_point_latitudes_deg,
            grid_point_longitudes_deg=grid_point_longitudes_deg,
            echo_top_matrix_km=echo_top_matrix_km,
            min_echo_top_km=min_echo_top_km,
            min_intermax_distance_metres=min_intermax_distance_metres)

        these_vertex_rows, these_vertex_columns = (
            polygons.grid_points_in_poly_to_vertices(
                grid_point_row_indices=local_max_dict[
                    temporal_tracking.GRID_POINT_ROWS_KEY][i],
                grid_point_column_indices=local_max_dict[
                    temporal_tracking.GRID_POINT_COLUMNS_KEY][i]
            )
        )

        # this_latitude_deg = local_max_dict[temporal_tracking.LATITUDES_KEY][i]
        # this_longitude_deg = local_max_dict[temporal_tracking.LONGITUDES_KEY][i]
        # this_time_unix_sec = local_max_dict[temporal_tracking.VALID_TIME_KEY]
        #
        # if this_time_unix_sec == 1303869300:
        #     if 32.7 <= this_latitude_deg <= 33 and 267 <= this_longitude_deg <= 267.5:
        #         print these_vertex_rows
        #         print '\n'
        #         print these_vertex_columns

        (local_max_dict[temporal_tracking.GRID_POINT_LATITUDES_KEY][i],
         local_max_dict[temporal_tracking.GRID_POINT_LONGITUDES_KEY][i]
        ) = radar_utils.rowcol_to_latlng(
            grid_rows=local_max_dict[temporal_tracking.GRID_POINT_ROWS_KEY][i],
            grid_columns=local_max_dict[
                temporal_tracking.GRID_POINT_COLUMNS_KEY][i],
            nw_grid_point_lat_deg=radar_metadata_dict[
                radar_utils.NW_GRID_POINT_LAT_COLUMN],
            nw_grid_point_lng_deg=radar_metadata_dict[
                radar_utils.NW_GRID_POINT_LNG_COLUMN],
            lat_spacing_deg=radar_metadata_dict[
                radar_utils.LAT_SPACING_COLUMN],
            lng_spacing_deg=radar_metadata_dict[radar_utils.LNG_SPACING_COLUMN]
        )

        these_vertex_latitudes_deg, these_vertex_longitudes_deg = (
            radar_utils.rowcol_to_latlng(
                grid_rows=these_vertex_rows, grid_columns=these_vertex_columns,
                nw_grid_point_lat_deg=radar_metadata_dict[
                    radar_utils.NW_GRID_POINT_LAT_COLUMN],
                nw_grid_point_lng_deg=radar_metadata_dict[
                    radar_utils.NW_GRID_POINT_LNG_COLUMN],
                lat_spacing_deg=radar_metadata_dict[
                    radar_utils.LAT_SPACING_COLUMN],
                lng_spacing_deg=radar_metadata_dict[
                    radar_utils.LNG_SPACING_COLUMN]
            )
        )

        local_max_dict[temporal_tracking.POLYGON_OBJECTS_ROWCOL_KEY][i] = (
            polygons.vertex_arrays_to_polygon_object(
                exterior_x_coords=these_vertex_columns,
                exterior_y_coords=these_vertex_rows)
        )

        local_max_dict[temporal_tracking.POLYGON_OBJECTS_LATLNG_KEY][i] = (
            polygons.vertex_arrays_to_polygon_object(
                exterior_x_coords=these_vertex_longitudes_deg,
                exterior_y_coords=these_vertex_latitudes_deg)
        )

        this_centroid_object_latlng = local_max_dict[
            temporal_tracking.POLYGON_OBJECTS_LATLNG_KEY
        ][i].centroid

        local_max_dict[temporal_tracking.LATITUDES_KEY][i] = (
            this_centroid_object_latlng.y
        )
        local_max_dict[temporal_tracking.LONGITUDES_KEY][i] = (
            this_centroid_object_latlng.x
        )

    return local_max_dict


def _remove_small_polygons(local_max_dict, min_size_pixels):
    """Removes small polygons (storm objects) at one time.

    :param local_max_dict: Dictionary created by `_local_maxima_to_polygons`.
    :param min_size_pixels: Minimum size.
    :return: local_max_dict: Same as input but maybe with fewer storm objects.
    """

    if min_size_pixels == 0:
        return local_max_dict

    num_grid_cells_by_polygon = numpy.array(
        [len(r) for r in local_max_dict[temporal_tracking.GRID_POINT_ROWS_KEY]],
        dtype=int
    )

    indices_to_keep = numpy.where(
        num_grid_cells_by_polygon >= min_size_pixels
    )[0]

    for this_key in local_max_dict:
        if isinstance(local_max_dict[this_key], list):
            local_max_dict[this_key] = [
                local_max_dict[this_key][k] for k in indices_to_keep
            ]
        elif isinstance(local_max_dict[this_key], numpy.ndarray):
            local_max_dict[this_key] = local_max_dict[this_key][indices_to_keep]

    return local_max_dict


def _write_new_tracks(storm_object_table, top_output_dir_name,
                      valid_times_unix_sec):
    """Writes tracking files (one Pickle file per time step).

    These files are the main output of both `run_tracking` and
    `reanalyze_across_spc_dates`.

    :param storm_object_table: See doc for `storm_tracking_io.write_file`.
    :param top_output_dir_name: Name of top-level directory.  File locations
        therein will be determined by `storm_tracking_io.find_file`.
    :param valid_times_unix_sec: 1-D numpy array of valid times.  One file will
        be written for each.
    """

    for this_time_unix_sec in valid_times_unix_sec:
        this_file_name = tracking_io.find_file(
            top_tracking_dir_name=top_output_dir_name,
            valid_time_unix_sec=this_time_unix_sec,
            spc_date_string=time_conversion.time_to_spc_date_string(
                this_time_unix_sec),
            tracking_scale_metres2=DUMMY_TRACKING_SCALE_METRES2,
            source_name=tracking_utils.SEGMOTION_NAME,
            raise_error_if_missing=False)

        print 'Writing new data to: "{0:s}"...'.format(this_file_name)

        tracking_io.write_file(
            storm_object_table=storm_object_table.loc[
                storm_object_table[tracking_utils.VALID_TIME_COLUMN] ==
                this_time_unix_sec
                ],
            pickle_file_name=this_file_name
        )


def _velocities_latlng_to_xy(
        east_velocities_m_s01, north_velocities_m_s01, latitudes_deg,
        longitudes_deg):
    """Converts velocities from lat-long components to x-y components.

    P = number of velocities

    :param east_velocities_m_s01: length-P numpy array of eastward instantaneous
        velocities (metres per second).
    :param north_velocities_m_s01: length-P numpy array of northward
        instantaneous velocities (metres per second).
    :param latitudes_deg: length-P numpy array of current latitudes (deg N).
    :param longitudes_deg: length-P numpy array of current longitudes (deg E).
    :return: x_velocities_m_s01: length-P numpy of x-velocities (metres per
        second in positive x-direction).
    :return: y_velocities_m_s01: Same but for y-direction.
    """

    projection_object = projections.init_azimuthal_equidistant_projection(
        central_latitude_deg=CENTRAL_PROJ_LATITUDE_DEG,
        central_longitude_deg=CENTRAL_PROJ_LONGITUDE_DEG)

    scalar_displacements_metres = numpy.sqrt(
        east_velocities_m_s01 ** 2 + north_velocities_m_s01 ** 2)

    standard_bearings_deg = RADIANS_TO_DEGREES * numpy.arctan2(
        north_velocities_m_s01, east_velocities_m_s01)

    geodetic_bearings_deg = geodetic_utils.standard_to_geodetic_angles(
        standard_bearings_deg)

    new_latitudes_deg, new_longitudes_deg = (
        geodetic_utils.start_points_and_displacements_to_endpoints(
            start_latitudes_deg=latitudes_deg,
            start_longitudes_deg=longitudes_deg,
            scalar_displacements_metres=scalar_displacements_metres,
            geodetic_bearings_deg=geodetic_bearings_deg)
    )

    x_coords_metres, y_coords_metres = projections.project_latlng_to_xy(
        latitudes_deg=latitudes_deg, longitudes_deg=longitudes_deg,
        projection_object=projection_object, false_easting_metres=0.,
        false_northing_metres=0.)

    new_x_coords_metres, new_y_coords_metres = projections.project_latlng_to_xy(
        latitudes_deg=new_latitudes_deg, longitudes_deg=new_longitudes_deg,
        projection_object=projection_object, false_easting_metres=0.,
        false_northing_metres=0.)

    return (new_x_coords_metres - x_coords_metres,
            new_y_coords_metres - y_coords_metres)


def _storm_objects_latlng_to_xy(storm_object_table):
    """Converts centroids and velocities from lat-long to x-y coordinates.

    :param storm_object_table: See doc for `storm_tracking_io.write_file`.
    :return: storm_object_table: Same as input but with the following columns.
    storm_object_table.centroid_x_metres: x-coordinate of storm-object centroid.
    storm_object_table.centroid_y_metres: y-coordinate of storm-object centroid.
    storm_object_table.x_velocity_m_s01: Velocity in +x-direction (metres per
        second).
    storm_object_table.y_velocity_m_s01: Velocity in +y-direction (metres per
        second).
    """

    projection_object = projections.init_azimuthal_equidistant_projection(
        central_latitude_deg=CENTRAL_PROJ_LATITUDE_DEG,
        central_longitude_deg=CENTRAL_PROJ_LONGITUDE_DEG)

    centroid_x_coords_metres, centroid_y_coords_metres = (
        projections.project_latlng_to_xy(
            latitudes_deg=storm_object_table[
                tracking_utils.CENTROID_LATITUDE_COLUMN].values,
            longitudes_deg=storm_object_table[
                tracking_utils.CENTROID_LONGITUDE_COLUMN].values,
            projection_object=projection_object,
            false_easting_metres=0., false_northing_metres=0.)
    )

    x_velocities_m_s01, y_velocities_m_s01 = _velocities_latlng_to_xy(
        east_velocities_m_s01=storm_object_table[
            tracking_utils.EAST_VELOCITY_COLUMN].values,
        north_velocities_m_s01=storm_object_table[
            tracking_utils.NORTH_VELOCITY_COLUMN].values,
        latitudes_deg=storm_object_table[
            tracking_utils.CENTROID_LATITUDE_COLUMN].values,
        longitudes_deg=storm_object_table[
            tracking_utils.CENTROID_LONGITUDE_COLUMN].values
    )

    return storm_object_table.assign(**{
        temporal_tracking.CENTROID_X_COLUMN: centroid_x_coords_metres,
        temporal_tracking.CENTROID_Y_COLUMN: centroid_y_coords_metres,
        temporal_tracking.X_VELOCITY_COLUMN: x_velocities_m_s01,
        temporal_tracking.Y_VELOCITY_COLUMN: y_velocities_m_s01
    })


def _shuffle_tracking_data(
        storm_object_table_by_date, tracking_file_names_by_date,
        valid_times_by_date_unix_sec, current_date_index, top_output_dir_name):
    """Shuffles data into and out of memory.

    T = number of SPC dates

    :param storm_object_table_by_date: length-T list of pandas DataFrames.  If
        data for the [i]th date are currently out of memory,
        storm_object_table_by_date[i] = None.  If data for the [i]th date are
        currently in memory, storm_object_table_by_date[i] has columns listed in
        `storm_tracking_io.write_file`.
    :param tracking_file_names_by_date: See doc for
        `_find_input_tracking_files`.
    :param valid_times_by_date_unix_sec: Same.
    :param current_date_index: Index of date currently being processed.  Must be
        in range 0...(T - 1).
    :param top_output_dir_name: Name of top-level output directory.  See doc for
        `_write_new_tracks`.
    :return: storm_object_table_by_date: Same as input, except that different
        items are in memory.
    """

    num_spc_dates = len(tracking_file_names_by_date)

    # Shuffle data out of memory.
    if current_date_index == num_spc_dates:
        for j in [num_spc_dates - 2, num_spc_dates - 1]:
            if j < 0:
                continue

            _write_new_tracks(
                storm_object_table=storm_object_table_by_date[j],
                top_output_dir_name=top_output_dir_name,
                valid_times_unix_sec=valid_times_by_date_unix_sec[j]
            )

            print '\n'
            storm_object_table_by_date[j] = pandas.DataFrame()

        return storm_object_table_by_date

    if current_date_index >= 2:
        _write_new_tracks(
            storm_object_table=storm_object_table_by_date[
                current_date_index - 2],
            top_output_dir_name=top_output_dir_name,
            valid_times_unix_sec=valid_times_by_date_unix_sec[
                current_date_index - 2]
        )

        print '\n'
        storm_object_table_by_date[current_date_index - 2] = pandas.DataFrame()

    # Shuffle data into memory.
    for j in [current_date_index - 1, current_date_index,
              current_date_index + 1]:

        if j < 0 or j >= num_spc_dates:
            continue
        if not storm_object_table_by_date[j].empty:
            continue

        storm_object_table_by_date[j] = tracking_io.read_many_files(
            tracking_file_names_by_date[j]
        )
        print '\n'

        storm_object_table_by_date[j] = _storm_objects_latlng_to_xy(
            storm_object_table_by_date[j]
        )

    return storm_object_table_by_date


def _radar_times_to_tracking_periods(
        radar_times_unix_sec, max_time_interval_sec):
    """Converts radar times to effective start/end times for tracking.

    When there is a gap of > `max_time_interval_sec` between successive radar
    times t_0 and t_1, tracking effectively ends at t_0 and then restarts at
    t_1.

    T = number of effective tracking periods

    :param radar_times_unix_sec: 1-D numpy array of radar times.
    :param max_time_interval_sec: Max time interval between successive radar
        times.
    :return: tracking_start_times_unix_sec: length-T numpy array of start times.
    :return: tracking_end_times_unix_sec: length-T numpy array of end times.
    """

    radar_time_diffs_sec = numpy.diff(radar_times_unix_sec)
    num_radar_times = len(radar_times_unix_sec)

    gap_indices = numpy.where(radar_time_diffs_sec > max_time_interval_sec)[0]

    tracking_start_indices = numpy.concatenate((
        numpy.array([0], dtype=int),
        gap_indices + 1
    ))

    tracking_end_indices = numpy.concatenate((
        gap_indices,
        numpy.array([num_radar_times - 1], dtype=int)
    ))

    tracking_start_times_unix_sec = radar_times_unix_sec[
        numpy.unique(tracking_start_indices)
    ]
    tracking_end_times_unix_sec = radar_times_unix_sec[
        numpy.unique(tracking_end_indices)
    ]

    tracking_start_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT)
        for t in tracking_start_times_unix_sec
    ]

    tracking_end_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT)
        for t in tracking_end_times_unix_sec
    ]

    print '\n'
    for k in range(len(tracking_start_time_strings)):
        print '{0:d}th tracking period = {1:s} to {2:s}'.format(
            k + 1, tracking_start_time_strings[k],
            tracking_end_time_strings[k]
        )

    print '\n'
    return tracking_start_times_unix_sec, tracking_end_times_unix_sec


def _read_tracking_periods(tracking_file_names):
    """Reads tracking periods from files.

    T = number of tracking periods

    :param tracking_file_names: 1-D list of paths to input files (will be read
        by `storm_tracking_io.read_file`).
    :return: tracking_start_times_unix_sec: length-T numpy array of start times.
    :return: tracking_end_times_unix_sec: length-T numpy array of end times.
    """

    tracking_start_times_unix_sec = numpy.array([], dtype=int)
    tracking_end_times_unix_sec = numpy.array([], dtype=int)

    for this_file_name in tracking_file_names:
        print 'Reading tracking periods from: "{0:s}"...'.format(this_file_name)
        this_storm_object_table = tracking_io.read_file(this_file_name)

        these_start_times_unix_sec = numpy.unique(
            this_storm_object_table[
                tracking_utils.TRACKING_START_TIME_COLUMN].values
        )

        these_end_times_unix_sec = numpy.unique(
            this_storm_object_table[
                tracking_utils.TRACKING_END_TIME_COLUMN].values
        )

        tracking_start_times_unix_sec = numpy.concatenate((
            tracking_start_times_unix_sec, these_start_times_unix_sec
        ))

        tracking_end_times_unix_sec = numpy.concatenate((
            tracking_end_times_unix_sec, these_end_times_unix_sec
        ))

    return (
        numpy.unique(tracking_start_times_unix_sec),
        numpy.unique(tracking_end_times_unix_sec)
    )


def _old_to_new_tracking_periods(
        tracking_start_times_unix_sec, tracking_end_times_unix_sec,
        max_time_interval_sec):
    """Converts old tracking periods to new tracking periods.

    N = number of original tracking periods
    n = number of final tracking periods

    :param tracking_start_times_unix_sec: length-N numpy array with start times
        of original periods.
    :param tracking_end_times_unix_sec: length-N numpy array with end times
        of original periods.
    :param max_time_interval_sec: Max time interval between successive local
        maxima (storm objects).  If successive local maxima are >
        `max_time_interval_sec` apart, they cannot be linked.  This is the max
        time interval for the final tracking periods, and it may be different
        than max interval for the original periods.
    :return: tracking_start_times_unix_sec: length-n numpy array with start
        times of final tracking periods.
    :return: tracking_end_times_unix_sec: length-n numpy array with end times of
        final tracking periods.
    """

    tracking_start_times_unix_sec, tracking_end_times_unix_sec = (
        temporal_tracking.check_tracking_periods(
            tracking_start_times_unix_sec=tracking_start_times_unix_sec,
            tracking_end_times_unix_sec=tracking_end_times_unix_sec)
    )

    interperiod_diffs_sec = (
        tracking_start_times_unix_sec[1:] - tracking_end_times_unix_sec[:-1]
    )

    # TODO(thunderhoser): Remove print statements.
    tracking_start_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT)
        for t in tracking_start_times_unix_sec
    ]

    tracking_end_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT)
        for t in tracking_end_times_unix_sec
    ]

    print '\n'
    for k in range(len(tracking_start_time_strings)):
        this_message_string = (
            '{0:d}th original tracking period = {1:s} to {2:s}'
        ).format(
            k + 1, tracking_start_time_strings[k],
            tracking_end_time_strings[k]
        )

        if k == 0:
            print this_message_string
            continue

        this_message_string += (
            ' ... gap between this and previous period = {0:d} seconds'
        ).format(interperiod_diffs_sec[k - 1])
        print this_message_string

    bad_indices = numpy.where(interperiod_diffs_sec <= max_time_interval_sec)[0]

    tracking_start_times_unix_sec = numpy.delete(
        tracking_start_times_unix_sec, bad_indices + 1
    )
    tracking_end_times_unix_sec = numpy.delete(
        tracking_end_times_unix_sec, bad_indices
    )

    # TODO(thunderhoser): Remove print statements.
    tracking_start_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT)
        for t in tracking_start_times_unix_sec
    ]

    tracking_end_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT)
        for t in tracking_end_times_unix_sec
    ]

    print '\n'
    for k in range(len(tracking_start_time_strings)):
        print '{0:d}th final tracking period = {1:s} to {2:s}'.format(
            k + 1, tracking_start_time_strings[k],
            tracking_end_time_strings[k]
        )

    print '\n'
    return tracking_start_times_unix_sec, tracking_end_times_unix_sec


def run_tracking(
        top_radar_dir_name, top_output_dir_name, first_spc_date_string,
        last_spc_date_string, first_time_unix_sec=None, last_time_unix_sec=None,
        first_numeric_id=None,
        echo_top_field_name=radar_utils.ECHO_TOP_40DBZ_NAME,
        radar_source_name=radar_utils.MYRORSS_SOURCE_ID,
        top_echo_classifn_dir_name=None,
        min_echo_top_km=DEFAULT_MIN_ECHO_TOP_KM,
        smoothing_radius_deg_lat=DEFAULT_SMOOTHING_RADIUS_DEG_LAT,
        half_width_for_max_filter_deg_lat=
        DEFAULT_HALF_WIDTH_FOR_MAX_FILTER_DEG_LAT,
        min_intermax_distance_metres=DEFAULT_MIN_INTERMAX_DISTANCE_METRES,
        min_polygon_size_pixels=DEFAULT_MIN_SIZE_PIXELS,
        max_link_time_seconds=DEFAULT_MAX_LINK_TIME_SECONDS,
        max_velocity_diff_m_s01=DEFAULT_MAX_VELOCITY_DIFF_M_S01,
        max_link_distance_m_s01=DEFAULT_MAX_LINK_DISTANCE_M_S01,
        min_track_duration_seconds=0):
    """Runs echo-top-tracking.  This is effectively the main method.

    :param top_radar_dir_name: See doc for `_find_input_radar_files`.
    :param top_output_dir_name: See doc for `_write_new_tracks`.
    :param first_spc_date_string: See doc for `_check_time_period`.
    :param last_spc_date_string: Same.
    :param first_time_unix_sec: Same.
    :param last_time_unix_sec: Same.
    :param first_numeric_id: First numeric storm ID.  Both primary and secondary
        IDs will start at this number.  Default is 100 * (Unix time at beginning
        of first SPC date).
    :param echo_top_field_name: See doc for `_find_input_radar_files`.
    :param radar_source_name: Same.
    :param top_echo_classifn_dir_name: Name of top-level directory with
        echo-classification files.  Files therein will be found by
        `echo_classification.find_classification_file` and read by
        `echo_classification.read_classifications`.  Tracking will be performed
        only on convective pixels.  If `top_echo_classifn_dir_name is None`,
        tracking will be performed on all pixels.
    :param min_echo_top_km: See doc for `_local_maxima_to_polygons`.
    :param smoothing_radius_deg_lat: See doc for `_gaussian_smooth_radar_field`.
    :param half_width_for_max_filter_deg_lat: See doc for `_find_local_maxima`.
    :param min_intermax_distance_metres: See doc for
        `_remove_redundant_local_maxima`.
    :param min_polygon_size_pixels: See doc for `_remove_small_polygons`.
    :param max_link_time_seconds: See doc for
        `temporal_tracking.link_local_maxima_in_time`.
    :param max_velocity_diff_m_s01: Same.
    :param max_link_distance_m_s01: Same.
    :param min_track_duration_seconds: See doc for
        `temporal_tracking.remove_short_lived_storms`.
    """

    if min_polygon_size_pixels is None:
        min_polygon_size_pixels = 0

    error_checking.assert_is_integer(min_polygon_size_pixels)
    error_checking.assert_is_geq(min_polygon_size_pixels, 0)
    error_checking.assert_is_greater(min_echo_top_km, 0.)

    radar_file_names, radar_times_unix_sec = _find_input_radar_files(
        top_radar_dir_name=top_radar_dir_name,
        radar_field_name=echo_top_field_name,
        radar_source_name=radar_source_name,
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string,
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec)

    radar_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT)
        for t in radar_times_unix_sec
    ]

    if first_numeric_id is None:
        first_numeric_id = 100 * time_conversion.get_start_of_spc_date(
            first_spc_date_string)

    error_checking.assert_is_integer(first_numeric_id)

    projection_object = projections.init_azimuthal_equidistant_projection(
        central_latitude_deg=CENTRAL_PROJ_LATITUDE_DEG,
        central_longitude_deg=CENTRAL_PROJ_LONGITUDE_DEG)

    num_times = len(radar_times_unix_sec)
    local_max_dict_by_time = [{}] * num_times
    keep_time_indices = []

    for i in range(num_times):
        if top_echo_classifn_dir_name is None:
            this_echo_classifn_file_name = None
            keep_time_indices.append(i)
        else:
            this_echo_classifn_file_name = (
                echo_classifn.find_classification_file(
                    top_directory_name=top_echo_classifn_dir_name,
                    valid_time_unix_sec=radar_times_unix_sec[i],
                    desire_zipped=True, allow_zipped_or_unzipped=True,
                    raise_error_if_missing=False)
            )

            if not os.path.isfile(this_echo_classifn_file_name):
                warning_string = (
                    'POTENTIAL PROBLEM.  Cannot find echo-classification file.'
                    '  Expected at: "{0:s}"'
                ).format(this_echo_classifn_file_name)

                warnings.warn(warning_string)
                local_max_dict_by_time[i] = None
                continue

            keep_time_indices.append(i)

        print 'Reading data from: "{0:s}"...'.format(radar_file_names[i])
        this_metadata_dict = myrorss_and_mrms_io.read_metadata_from_raw_file(
            netcdf_file_name=radar_file_names[i], data_source=radar_source_name)

        this_sparse_grid_table = (
            myrorss_and_mrms_io.read_data_from_sparse_grid_file(
                netcdf_file_name=radar_file_names[i],
                field_name_orig=this_metadata_dict[
                    myrorss_and_mrms_io.FIELD_NAME_COLUMN_ORIG],
                data_source=radar_source_name,
                sentinel_values=this_metadata_dict[
                    radar_utils.SENTINEL_VALUE_COLUMN]
            )
        )

        this_echo_top_matrix_km = radar_s2f.sparse_to_full_grid(
            sparse_grid_table=this_sparse_grid_table,
            metadata_dict=this_metadata_dict,
            ignore_if_below=min_echo_top_km
        )[0]

        if this_echo_classifn_file_name is not None:
            print 'Reading data from: "{0:s}"...'.format(
                this_echo_classifn_file_name)

            this_convective_flag_matrix = echo_classifn.read_classifications(
                this_echo_classifn_file_name
            )[0]

            this_convective_flag_matrix = numpy.flip(
                this_convective_flag_matrix, axis=0)
            this_echo_top_matrix_km[this_convective_flag_matrix == False] = 0.

        print 'Finding local maxima in "{0:s}" at {1:s}...'.format(
            echo_top_field_name, radar_time_strings[i]
        )

        this_latitude_spacing_deg = this_metadata_dict[
            radar_utils.LAT_SPACING_COLUMN]

        this_smoothed_et_matrix_km = _gaussian_smooth_radar_field(
            radar_matrix=copy.deepcopy(this_echo_top_matrix_km),
            e_folding_radius_pixels=
            smoothing_radius_deg_lat / this_latitude_spacing_deg
        )

        this_half_width_pixels = int(numpy.round(
            half_width_for_max_filter_deg_lat / this_latitude_spacing_deg
        ))

        local_max_dict_by_time[i] = _find_local_maxima(
            radar_matrix=this_smoothed_et_matrix_km,
            radar_metadata_dict=this_metadata_dict,
            neigh_half_width_pixels=this_half_width_pixels)

        local_max_dict_by_time[i].update(
            {temporal_tracking.VALID_TIME_KEY: radar_times_unix_sec[i]}
        )

        local_max_dict_by_time[i] = _local_maxima_to_polygons(
            local_max_dict=local_max_dict_by_time[i],
            echo_top_matrix_km=this_echo_top_matrix_km,
            min_echo_top_km=min_echo_top_km,
            radar_metadata_dict=this_metadata_dict,
            min_intermax_distance_metres=min_intermax_distance_metres)

        local_max_dict_by_time[i] = _remove_small_polygons(
            local_max_dict=local_max_dict_by_time[i],
            min_size_pixels=min_polygon_size_pixels)

        local_max_dict_by_time[i] = _remove_redundant_local_maxima(
            local_max_dict=local_max_dict_by_time[i],
            projection_object=projection_object,
            min_intermax_distance_metres=min_intermax_distance_metres)

        if i == 0:
            this_current_to_prev_matrix = (
                temporal_tracking.link_local_maxima_in_time(
                    current_local_max_dict=local_max_dict_by_time[i],
                    previous_local_max_dict=None,
                    max_link_time_seconds=max_link_time_seconds,
                    max_velocity_diff_m_s01=max_velocity_diff_m_s01,
                    max_link_distance_m_s01=max_link_distance_m_s01)
            )
        else:
            print (
                'Linking local maxima at {0:s} with those at {1:s}...\n'
            ).format(radar_time_strings[i], radar_time_strings[i - 1])

            this_current_to_prev_matrix = (
                temporal_tracking.link_local_maxima_in_time(
                    current_local_max_dict=local_max_dict_by_time[i],
                    previous_local_max_dict=local_max_dict_by_time[i - 1],
                    max_link_time_seconds=max_link_time_seconds,
                    max_velocity_diff_m_s01=max_velocity_diff_m_s01,
                    max_link_distance_m_s01=max_link_distance_m_s01)
            )

        local_max_dict_by_time[i].update({
            temporal_tracking.CURRENT_TO_PREV_MATRIX_KEY:
                this_current_to_prev_matrix
        })

        if i == 0:
            local_max_dict_by_time[i] = (
                temporal_tracking.get_intermediate_velocities(
                    current_local_max_dict=local_max_dict_by_time[i],
                    previous_local_max_dict=None)
            )
        else:
            local_max_dict_by_time[i] = (
                temporal_tracking.get_intermediate_velocities(
                    current_local_max_dict=local_max_dict_by_time[i],
                    previous_local_max_dict=local_max_dict_by_time[i - 1])
            )

    print SEPARATOR_STRING

    keep_time_indices = numpy.array(keep_time_indices, dtype=int)
    radar_times_unix_sec = radar_times_unix_sec[keep_time_indices]
    del radar_time_strings

    local_max_dict_by_time = [
        local_max_dict_by_time[k] for k in keep_time_indices
    ]

    print 'Converting time series of "{0:s}" maxima to storm tracks...'.format(
        echo_top_field_name)
    storm_object_table = temporal_tracking.local_maxima_to_storm_tracks(
        local_max_dict_by_time=local_max_dict_by_time,
        first_numeric_id=first_numeric_id)

    print 'Removing tracks that last < {0:d} seconds...'.format(
        int(min_track_duration_seconds)
    )
    storm_object_table = temporal_tracking.remove_short_lived_storms(
        storm_object_table=storm_object_table,
        min_duration_seconds=min_track_duration_seconds)

    print 'Computing storm ages...'
    tracking_start_times_unix_sec, tracking_end_times_unix_sec = (
        _radar_times_to_tracking_periods(
            radar_times_unix_sec=radar_times_unix_sec,
            max_time_interval_sec=max_link_time_seconds)
    )

    storm_object_table = temporal_tracking.get_storm_ages(
        storm_object_table=storm_object_table,
        tracking_start_times_unix_sec=tracking_start_times_unix_sec,
        tracking_end_times_unix_sec=tracking_end_times_unix_sec,
        max_link_time_seconds=max_link_time_seconds)

    print 'Computing storm velocities...'
    storm_object_table = temporal_tracking.get_storm_velocities(
        storm_object_table=storm_object_table)

    print SEPARATOR_STRING
    _write_new_tracks(
        storm_object_table=storm_object_table,
        top_output_dir_name=top_output_dir_name,
        valid_times_unix_sec=radar_times_unix_sec)


def reanalyze_across_spc_dates(
        top_input_dir_name, top_output_dir_name, first_spc_date_string,
        last_spc_date_string, first_time_unix_sec=None, last_time_unix_sec=None,
        max_link_time_seconds=DEFAULT_MAX_LINK_TIME_SECONDS,
        max_velocity_diff_m_s01=DEFAULT_MAX_VELOCITY_DIFF_M_S01,
        max_link_distance_m_s01=DEFAULT_MAX_LINK_DISTANCE_M_S01,
        max_join_time_seconds=DEFAULT_MAX_JOIN_TIME_SEC,
        max_join_error_m_s01=DEFAULT_MAX_JOIN_ERROR_M_S01,
        min_track_duration_seconds=DEFAULT_MIN_REANALYZED_DURATION_SEC):
    """Reanalyzes tracks across SPC dates.

    :param top_input_dir_name: Name of top-level directory with original tracks
        (before reanalysis).  For more details, see doc for
        `_find_input_tracking_files`.
    :param top_output_dir_name: Name of top-level directory for new tracks
        (after reanalysis).  For more details, see doc for
        `_write_new_tracks`.
    :param first_spc_date_string: See doc for `_find_input_tracking_files`.
    :param last_spc_date_string: Same.
    :param first_time_unix_sec: Same.
    :param last_time_unix_sec: Same.
    :param max_link_time_seconds: See doc for
        `temporal_tracking.link_local_maxima_in_time`.
    :param max_velocity_diff_m_s01: See doc for
        `temporal_tracking.link_local_maxima_in_time`.  Used only to bridge gap
        between two SPC dates.
    :param max_link_distance_m_s01: Same.
    :param max_join_time_seconds: See doc for
        `track_reanalysis.join_collinear_tracks`.
    :param max_join_error_m_s01: Same.
    :param min_track_duration_seconds: See doc for
        `temporal_tracking.remove_short_lived_storms`.
    """

    (spc_date_strings, tracking_file_names_by_date, valid_times_by_date_unix_sec
    ) = _find_input_tracking_files(
        top_tracking_dir_name=top_input_dir_name,
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string,
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec)

    tracking_file_names = list(chain(*tracking_file_names_by_date))
    tracking_start_times_unix_sec, tracking_end_times_unix_sec = (
        _read_tracking_periods(tracking_file_names)
    )
    print SEPARATOR_STRING

    tracking_start_times_unix_sec, tracking_end_times_unix_sec = (
        _old_to_new_tracking_periods(
            tracking_start_times_unix_sec=tracking_start_times_unix_sec,
            tracking_end_times_unix_sec=tracking_end_times_unix_sec,
            max_time_interval_sec=max([
                max_link_time_seconds, max_join_time_seconds
            ])
        )
    )

    num_spc_dates = len(spc_date_strings)

    if num_spc_dates == 1:
        storm_object_table = tracking_io.read_many_files(
            tracking_file_names_by_date[0]
        )
        print SEPARATOR_STRING

        storm_object_table = _storm_objects_latlng_to_xy(storm_object_table)

        first_late_time_unix_sec = numpy.min(
            storm_object_table[tracking_utils.VALID_TIME_COLUMN].values
        )
        last_late_time_unix_sec = numpy.max(
            storm_object_table[tracking_utils.VALID_TIME_COLUMN].values
        )

        storm_object_table = track_reanalysis.join_collinear_tracks(
            storm_object_table=storm_object_table,
            first_late_time_unix_sec=first_late_time_unix_sec,
            last_late_time_unix_sec=last_late_time_unix_sec,
            max_join_time_seconds=max_join_time_seconds,
            max_join_error_m_s01=max_join_error_m_s01)
        print SEPARATOR_STRING

        print 'Removing tracks that last < {0:d} seconds...'.format(
            int(min_track_duration_seconds)
        )

        storm_object_table = temporal_tracking.remove_short_lived_storms(
            storm_object_table=storm_object_table,
            min_duration_seconds=min_track_duration_seconds)

        print 'Recomputing storm ages...'
        storm_object_table = temporal_tracking.get_storm_ages(
            storm_object_table=storm_object_table,
            tracking_start_times_unix_sec=tracking_start_times_unix_sec,
            tracking_end_times_unix_sec=tracking_end_times_unix_sec,
            max_link_time_seconds=max([
                max_link_time_seconds, max_join_time_seconds
            ])
        )

        print 'Computing storm velocities...'
        storm_object_table = temporal_tracking.get_storm_velocities(
            storm_object_table=storm_object_table)

        print SEPARATOR_STRING
        _write_new_tracks(
            storm_object_table=storm_object_table,
            top_output_dir_name=top_output_dir_name,
            valid_times_unix_sec=valid_times_by_date_unix_sec[0])

        return

    storm_object_table_by_date = [pandas.DataFrame()] * num_spc_dates

    for i in range(num_spc_dates + 1):
        storm_object_table_by_date = _shuffle_tracking_data(
            tracking_file_names_by_date=tracking_file_names_by_date,
            valid_times_by_date_unix_sec=valid_times_by_date_unix_sec,
            storm_object_table_by_date=storm_object_table_by_date,
            current_date_index=i, top_output_dir_name=top_output_dir_name)
        print SEPARATOR_STRING

        if i == num_spc_dates:
            break

        if i != num_spc_dates - 1:
            this_late_time_unix_sec = numpy.min(
                valid_times_by_date_unix_sec[i + 1]
            )

            concat_storm_object_table = pandas.concat(
                [storm_object_table_by_date[k] for k in [i, i + 1]],
                axis=0, ignore_index=True)

            concat_storm_object_table = track_reanalysis.join_collinear_tracks(
                storm_object_table=concat_storm_object_table,
                first_late_time_unix_sec=this_late_time_unix_sec,
                last_late_time_unix_sec=this_late_time_unix_sec,
                max_join_time_seconds=max_link_time_seconds,
                max_join_error_m_s01=max_velocity_diff_m_s01,
                max_join_distance_m_s01=max_link_distance_m_s01)
            print SEPARATOR_STRING

            if i == 0:
                this_first_time_unix_sec = numpy.min(
                    storm_object_table_by_date[i][
                        tracking_utils.VALID_TIME_COLUMN].values
                )
            else:
                this_first_time_unix_sec = numpy.min(
                    storm_object_table_by_date[i + 1][
                        tracking_utils.VALID_TIME_COLUMN].values
                )

            this_last_time_unix_sec = numpy.max(
                storm_object_table_by_date[i + 1][
                    tracking_utils.VALID_TIME_COLUMN].values
            )

            concat_storm_object_table = track_reanalysis.join_collinear_tracks(
                storm_object_table=concat_storm_object_table,
                first_late_time_unix_sec=this_first_time_unix_sec,
                last_late_time_unix_sec=this_last_time_unix_sec,
                max_join_time_seconds=max_join_time_seconds,
                max_join_error_m_s01=max_join_error_m_s01)
            print SEPARATOR_STRING

            storm_object_table_by_date[i] = concat_storm_object_table.loc[
                concat_storm_object_table[tracking_utils.SPC_DATE_COLUMN] ==
                spc_date_strings[i]
            ]

            storm_object_table_by_date[i + 1] = concat_storm_object_table.loc[
                concat_storm_object_table[tracking_utils.SPC_DATE_COLUMN] ==
                spc_date_strings[i + 1]
            ]

        if i == 0:
            indices_to_concat = numpy.array([i, i + 1], dtype=int)
        elif i == num_spc_dates - 1:
            indices_to_concat = numpy.array([i - 1, i], dtype=int)
        else:
            indices_to_concat = numpy.array([i - 1, i, i + 1], dtype=int)

        concat_storm_object_table = pandas.concat(
            [storm_object_table_by_date[k] for k in indices_to_concat],
            axis=0, ignore_index=True)

        print 'Removing tracks that last < {0:d} seconds...'.format(
            int(min_track_duration_seconds)
        )
        concat_storm_object_table = temporal_tracking.remove_short_lived_storms(
            storm_object_table=concat_storm_object_table,
            min_duration_seconds=min_track_duration_seconds)

        print 'Recomputing storm ages...'
        concat_storm_object_table = temporal_tracking.get_storm_ages(
            storm_object_table=concat_storm_object_table,
            tracking_start_times_unix_sec=tracking_start_times_unix_sec,
            tracking_end_times_unix_sec=tracking_end_times_unix_sec,
            max_link_time_seconds=max([
                max_link_time_seconds, max_join_time_seconds
            ])
        )

        print 'Computing storm velocities...'
        concat_storm_object_table = temporal_tracking.get_storm_velocities(
            storm_object_table=concat_storm_object_table)

        storm_object_table_by_date[i] = concat_storm_object_table.loc[
            concat_storm_object_table[tracking_utils.SPC_DATE_COLUMN] ==
            spc_date_strings[i]
        ]
        print SEPARATOR_STRING


def fix_tracking_periods(
        top_radar_dir_name, top_input_tracking_dir_name,
        top_output_tracking_dir_name, first_spc_date_string,
        last_spc_date_string,
        echo_top_field_name=radar_utils.ECHO_TOP_40DBZ_NAME,
        radar_source_name=radar_utils.MYRORSS_SOURCE_ID,
        top_echo_classifn_dir_name=None,
        max_link_time_seconds=DEFAULT_MAX_LINK_TIME_SECONDS):
    """Fixes tracking periods in files written by `run_tracking`.

    :param top_radar_dir_name: See doc for `run_tracking`.
    :param top_input_tracking_dir_name: Name of top-level directory with input
        tracks.  Files therein will be found by `storm_tracking_io.find_file`
        and read by `storm_tracking_io.read_file`.
    :param top_output_tracking_dir_name: Name of top-level directory for output
        tracks (after fixing tracking period).  Files will be written by
        `storm_tracking_io.write_file`, to locations therein determined by
        `storm_tracking_io.find_file`.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param echo_top_field_name: Same.
    :param radar_source_name: Same.
    :param top_echo_classifn_dir_name: Same.
    :param max_link_time_seconds: Same.
    """

    _, radar_times_unix_sec = _find_input_radar_files(
        top_radar_dir_name=top_radar_dir_name,
        radar_field_name=echo_top_field_name,
        radar_source_name=radar_source_name,
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string,
        first_time_unix_sec=None, last_time_unix_sec=None)

    radar_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT)
        for t in radar_times_unix_sec
    ]

    spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string)

    input_tracking_file_names = []

    for this_spc_date_string in spc_date_strings:
        input_tracking_file_names += tracking_io.find_files_one_spc_date(
            top_tracking_dir_name=top_input_tracking_dir_name,
            tracking_scale_metres2=DUMMY_TRACKING_SCALE_METRES2,
            source_name=tracking_utils.SEGMOTION_NAME,
            spc_date_string=this_spc_date_string, raise_error_if_missing=False
        )[0]

    input_tracking_file_names.sort()

    num_times = len(radar_times_unix_sec)
    keep_time_indices = []

    for i in range(num_times):
        print 'Looking for echo-classification file at "{0:s}"...'.format(
            radar_time_strings[i]
        )

        if top_echo_classifn_dir_name is None:
            keep_time_indices.append(i)
            continue

        this_echo_classifn_file_name = echo_classifn.find_classification_file(
            top_directory_name=top_echo_classifn_dir_name,
            valid_time_unix_sec=radar_times_unix_sec[i],
            desire_zipped=True, allow_zipped_or_unzipped=True,
            raise_error_if_missing=False)

        if not os.path.isfile(this_echo_classifn_file_name):
            warning_string = (
                'POTENTIAL PROBLEM.  Cannot find echo-classification file.'
                '  Expected at: "{0:s}"'
            ).format(this_echo_classifn_file_name)

            warnings.warn(warning_string)
            continue

        keep_time_indices.append(i)

    keep_time_indices = numpy.array(keep_time_indices, dtype=int)
    radar_times_unix_sec = radar_times_unix_sec[keep_time_indices]

    print SEPARATOR_STRING
    storm_object_table = tracking_io.read_many_files(input_tracking_file_names)
    print SEPARATOR_STRING

    print 'Computing storm ages...'
    tracking_start_times_unix_sec, tracking_end_times_unix_sec = (
        _radar_times_to_tracking_periods(
            radar_times_unix_sec=radar_times_unix_sec,
            max_time_interval_sec=max_link_time_seconds)
    )

    storm_object_table = temporal_tracking.get_storm_ages(
        storm_object_table=storm_object_table,
        tracking_start_times_unix_sec=tracking_start_times_unix_sec,
        tracking_end_times_unix_sec=tracking_end_times_unix_sec,
        max_link_time_seconds=max_link_time_seconds)

    print SEPARATOR_STRING
    _write_new_tracks(
        storm_object_table=storm_object_table,
        top_output_dir_name=top_output_tracking_dir_name,
        valid_times_unix_sec=radar_times_unix_sec)
