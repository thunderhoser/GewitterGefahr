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
import numpy
import pandas
from scipy.ndimage.filters import gaussian_filter
from geopy.distance import vincenty
from gewittergefahr.gg_io import myrorss_and_mrms_io
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import temporal_tracking
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import radar_sparse_to_full as radar_s2f
from gewittergefahr.gg_utils import dilation
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import geodetic_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import best_tracks
from gewittergefahr.gg_utils import echo_classification as echo_classifn
from gewittergefahr.gg_utils import error_checking

TOLERANCE = 1e-6
DUMMY_TIME_UNIX_SEC = -10000

MAX_STORMS_IN_SPLIT = 2
MAX_STORMS_IN_MERGER = 2

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

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
DEFAULT_MAX_LINK_TIME_SECONDS = 305
DEFAULT_MAX_VELOCITY_DIFF_M_S01 = 10.
DEFAULT_MAX_LINK_DISTANCE_M_S01 = (
    0.125 * DEGREES_LAT_TO_METRES / DEFAULT_MAX_LINK_TIME_SECONDS
)

DEFAULT_MAX_JOIN_TIME_SEC = 610
DEFAULT_MAX_JOIN_ERROR_M_S01 = 20.
DEFAULT_NUM_POINTS_FOR_VELOCITY = 3
DEFAULT_MIN_REANALYZED_DURATION_SEC = 890

DUMMY_TRACKING_SCALE_METRES2 = int(numpy.round(numpy.pi * 1e8))  # 10-km radius

TRACKING_FILE_NAMES_KEY = 'output_tracking_file_names'
VALID_TIMES_KEY = 'unix_times_sec'

LATITUDES_KEY = 'latitudes_deg'
LONGITUDES_KEY = 'longitudes_deg'
MAX_VALUES_KEY = 'max_values'
X_COORDS_KEY = 'x_coords_metres'
Y_COORDS_KEY = 'y_coords_metres'
VALID_TIME_KEY = 'unix_time_sec'
CURRENT_TO_PREV_MATRIX_KEY = 'current_to_previous_matrix'

STORM_IDS_KEY = 'storm_ids'
PRIMARY_IDS_KEY = 'primary_id_strings'
SECONDARY_IDS_KEY = 'secondary_id_strings'

CURRENT_LOCAL_MAXIMA_KEY = 'current_local_max_dict'
PREVIOUS_PRIMARY_ID_KEY = 'prev_primary_id_numeric'
PREVIOUS_SPC_DATE_KEY = 'prev_spc_date_string'
PREVIOUS_SECONDARY_ID_KEY = 'prev_secondary_id_numeric'
OLD_TO_NEW_PRIMARY_IDS_KEY = 'old_to_new_primary_id_dict'

PRIMARY_STORM_ID_COLUMN = 'primary_storm_id'
SECONDARY_STORM_ID_COLUMN = 'secondary_storm_id'

CENTROID_X_COLUMN = 'centroid_x_metres'
CENTROID_Y_COLUMN = 'centroid_y_metres'
X_VELOCITIES_KEY = 'x_velocities_m_s01'
Y_VELOCITIES_KEY = 'y_velocities_m_s01'

GRID_POINT_ROWS_KEY = 'grid_point_rows_array_list'
GRID_POINT_COLUMNS_KEY = 'grid_point_columns_array_list'
GRID_POINT_LATITUDES_KEY = 'grid_point_lats_array_list_deg'
GRID_POINT_LONGITUDES_KEY = 'grid_point_lngs_array_list_deg'
POLYGON_OBJECTS_ROWCOL_KEY = 'polygon_objects_rowcol'
POLYGON_OBJECTS_LATLNG_KEY = 'polygon_objects_latlng'

START_TIME_COLUMN = 'start_time_unix_sec'
END_TIME_COLUMN = 'end_time_unix_sec'
START_LATITUDE_COLUMN = 'start_latitude_deg'
END_LATITUDE_COLUMN = 'end_latitude_deg'
START_LONGITUDE_COLUMN = 'start_longitude_deg'
END_LONGITUDE_COLUMN = 'end_longitude_deg'


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
        LATITUDES_KEY: max_latitudes_deg,
        LONGITUDES_KEY: max_longitudes_deg,
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
        local_max_dict[LATITUDES_KEY], local_max_dict[LONGITUDES_KEY],
        projection_object=projection_object,
        false_easting_metres=0., false_northing_metres=0.)

    local_max_dict.update({
        X_COORDS_KEY: x_coords_metres,
        Y_COORDS_KEY: y_coords_metres
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

        these_distances_metres[i] = numpy.inf
        these_redundant_indices = numpy.where(
            these_distances_metres < min_intermax_distance_metres
        )[0]

        if len(these_redundant_indices) == 0:
            continue

        these_redundant_indices = numpy.concatenate((
            these_redundant_indices, numpy.array([i], dtype=int)
        ))
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
            top_directory_name=top_radar_dir_name, raise_error_if_missing=True)

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
    """Finds tracking files (inputs to `run_tracking` -- basically main method).

    T = number of SPC dates

    :param top_tracking_dir_name: Name of top-level directory with tracking
        files.  Files therein will be found by
        `storm_tracking_io.find_processed_files_one_spc_date`.
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

    for i in range(num_spc_dates):
        these_file_names = tracking_io.find_processed_files_one_spc_date(
            spc_date_string=spc_date_strings[i],
            data_source=tracking_utils.SEGMOTION_SOURCE_ID,
            top_processed_dir_name=top_tracking_dir_name,
            tracking_scale_metres2=DUMMY_TRACKING_SCALE_METRES2
        )[0]

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
            tracking_io.processed_file_name_to_time(f)
            for f in these_file_names
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

    return (spc_date_strings, tracking_file_names_by_date,
            valid_times_by_date_unix_sec)


def _get_final_velocities_one_track(
        centroid_latitudes_deg, centroid_longitudes_deg, valid_times_unix_sec,
        num_points_back):
    """Estimates storm velocity at each time step.

    P = number of points in track

    :param centroid_latitudes_deg: length-P numpy array with latitudes (deg N)
        of storm centroid.
    :param centroid_longitudes_deg: length-P numpy array with longitudes (deg E)
        of storm centroid.
    :param valid_times_unix_sec: length-P numpy array of valid times.
    :param num_points_back: Number of points to use in each estimate (backwards
        differencing in time).
    :return: east_velocities_m_s01: length-P numpy array of eastward velocities
        (metres per second).
    :return: north_velocities_m_s01: length-P numpy array of northward
        velocities (metres per second).
    """

    sort_indices = numpy.argsort(valid_times_unix_sec)

    num_times = len(valid_times_unix_sec)
    east_displacements_metres = numpy.full(num_times, numpy.nan)
    north_displacements_metres = numpy.full(num_times, numpy.nan)
    time_diffs_seconds = numpy.full(num_times, -1, dtype=int)

    for i in range(num_times):
        this_num_points_back = min([i, num_points_back])
        if this_num_points_back == 0:
            continue

        this_end_latitude_deg = centroid_latitudes_deg[sort_indices[i]]
        this_end_longitude_deg = centroid_longitudes_deg[sort_indices[i]]
        this_start_latitude_deg = centroid_latitudes_deg[
            sort_indices[i - this_num_points_back]
        ]
        this_start_longitude_deg = centroid_longitudes_deg[
            sort_indices[i - this_num_points_back]
        ]

        this_end_point = (this_end_latitude_deg, this_end_longitude_deg)
        this_start_point = (this_end_latitude_deg, this_start_longitude_deg)
        east_displacements_metres[i] = vincenty(
            this_start_point, this_end_point
        ).meters

        if this_start_longitude_deg > this_end_longitude_deg:
            east_displacements_metres[i] = -1 * east_displacements_metres[i]

        this_start_point = (this_start_latitude_deg, this_end_longitude_deg)
        north_displacements_metres[i] = vincenty(
            this_start_point, this_end_point
        ).meters

        if this_start_latitude_deg > this_end_latitude_deg:
            north_displacements_metres[i] = -1 * north_displacements_metres[i]

        time_diffs_seconds[i] = (
            valid_times_unix_sec[sort_indices[i]] -
            valid_times_unix_sec[sort_indices[i - this_num_points_back]]
        )

    return (east_displacements_metres / time_diffs_seconds,
            north_displacements_metres / time_diffs_seconds)


def _get_final_velocities(storm_object_table, num_points_back,
                          e_folding_radius_metres):
    """Computes final estimate of storm velocities.

    This method computes one velocity for each storm object (each storm cell at
    each time step).

    :param storm_object_table: pandas DataFrame created by
        `_local_maxima_to_storm_tracks`.
    :param num_points_back: Number of points to use in each estimate (backwards
        differencing in time).
    :param e_folding_radius_metres: See doc for `_estimate_velocity_by_neigh`.
    :return: storm_object_table: Same as input but with the following extra
        columns.
    storm_object_table.east_velocity_m_s01: Eastward velocity (metres per
        second).
    storm_object_table.north_velocity_m_s01: Northward velocity (metres per
        second).
    """

    all_storm_ids = numpy.array(
        storm_object_table[tracking_utils.STORM_ID_COLUMN].values
    )

    unique_storm_ids, storm_ids_object_to_unique = numpy.unique(
        all_storm_ids, return_inverse=True)

    num_storm_objects = len(storm_object_table.index)
    east_velocities_m_s01 = numpy.full(num_storm_objects, numpy.nan)
    north_velocities_m_s01 = numpy.full(num_storm_objects, numpy.nan)

    for i in range(len(unique_storm_ids)):
        these_object_indices = numpy.where(storm_ids_object_to_unique == i)[0]

        (east_velocities_m_s01[these_object_indices],
         north_velocities_m_s01[these_object_indices]
        ) = _get_final_velocities_one_track(
            centroid_latitudes_deg=storm_object_table[
                tracking_utils.CENTROID_LAT_COLUMN
            ].values[these_object_indices],
            centroid_longitudes_deg=storm_object_table[
                tracking_utils.CENTROID_LNG_COLUMN
            ].values[these_object_indices],
            valid_times_unix_sec=storm_object_table[
                tracking_utils.TIME_COLUMN
            ].values[these_object_indices],
            num_points_back=num_points_back)

    east_velocities_m_s01, north_velocities_m_s01 = _estimate_velocity_by_neigh(
        x_coords_metres=storm_object_table[CENTROID_X_COLUMN].values,
        y_coords_metres=storm_object_table[CENTROID_Y_COLUMN].values,
        x_velocities_m_s01=east_velocities_m_s01,
        y_velocities_m_s01=north_velocities_m_s01,
        e_folding_radius_metres=e_folding_radius_metres)

    argument_dict = {
        tracking_utils.EAST_VELOCITY_COLUMN: east_velocities_m_s01,
        tracking_utils.NORTH_VELOCITY_COLUMN: north_velocities_m_s01
    }

    return storm_object_table.assign(**argument_dict)


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
            min_longitude_deg=
            radar_metadata_dict[radar_utils.NW_GRID_POINT_LNG_COLUMN],
            lat_spacing_deg=radar_metadata_dict[radar_utils.LAT_SPACING_COLUMN],
            lng_spacing_deg=radar_metadata_dict[radar_utils.LNG_SPACING_COLUMN],
            num_rows=radar_metadata_dict[radar_utils.NUM_LAT_COLUMN],
            num_columns=radar_metadata_dict[radar_utils.NUM_LNG_COLUMN])
    )

    grid_point_latitudes_deg = grid_point_latitudes_deg[::-1]
    num_maxima = len(local_max_dict[LATITUDES_KEY])

    local_max_dict[GRID_POINT_ROWS_KEY] = [[]] * num_maxima
    local_max_dict[GRID_POINT_COLUMNS_KEY] = [[]] * num_maxima
    local_max_dict[GRID_POINT_LATITUDES_KEY] = [[]] * num_maxima
    local_max_dict[GRID_POINT_LONGITUDES_KEY] = [[]] * num_maxima
    local_max_dict[POLYGON_OBJECTS_ROWCOL_KEY] = numpy.full(
        num_maxima, numpy.nan, dtype=object)
    local_max_dict[POLYGON_OBJECTS_LATLNG_KEY] = numpy.full(
        num_maxima, numpy.nan, dtype=object)

    for i in range(num_maxima):
        this_echo_top_submatrix_km, this_row_offset, this_column_offset = (
            grids.extract_latlng_subgrid(
                data_matrix=echo_top_matrix_km,
                grid_point_latitudes_deg=grid_point_latitudes_deg,
                grid_point_longitudes_deg=grid_point_longitudes_deg,
                center_latitude_deg=local_max_dict[LATITUDES_KEY][i],
                center_longitude_deg=local_max_dict[LONGITUDES_KEY][i],
                max_distance_from_center_metres=min_intermax_distance_metres)
        )

        this_echo_top_submatrix_km[
            numpy.isnan(this_echo_top_submatrix_km)
        ] = 0.

        (local_max_dict[GRID_POINT_ROWS_KEY][i],
         local_max_dict[GRID_POINT_COLUMNS_KEY][i]
        ) = numpy.where(this_echo_top_submatrix_km >= min_echo_top_km)

        if not len(local_max_dict[GRID_POINT_ROWS_KEY][i]):
            this_row = numpy.floor(
                float(this_echo_top_submatrix_km.shape[0]) / 2
            )
            this_column = numpy.floor(
                float(this_echo_top_submatrix_km.shape[1]) / 2
            )

            local_max_dict[GRID_POINT_ROWS_KEY][i] = numpy.array(
                [this_row], dtype=int)
            local_max_dict[GRID_POINT_COLUMNS_KEY][i] = numpy.array(
                [this_column], dtype=int)

        local_max_dict[GRID_POINT_ROWS_KEY][i] = (
            local_max_dict[GRID_POINT_ROWS_KEY][i] + this_row_offset
        )
        local_max_dict[GRID_POINT_COLUMNS_KEY][i] = (
            local_max_dict[GRID_POINT_COLUMNS_KEY][i] + this_column_offset
        )

        these_vertex_rows, these_vertex_columns = (
            polygons.grid_points_in_poly_to_vertices(
                local_max_dict[GRID_POINT_ROWS_KEY][i],
                local_max_dict[GRID_POINT_COLUMNS_KEY][i])
        )

        (local_max_dict[GRID_POINT_LATITUDES_KEY][i],
         local_max_dict[GRID_POINT_LONGITUDES_KEY][i]
        ) = radar_utils.rowcol_to_latlng(
            local_max_dict[GRID_POINT_ROWS_KEY][i],
            local_max_dict[GRID_POINT_COLUMNS_KEY][i],
            nw_grid_point_lat_deg=
            radar_metadata_dict[radar_utils.NW_GRID_POINT_LAT_COLUMN],
            nw_grid_point_lng_deg=
            radar_metadata_dict[radar_utils.NW_GRID_POINT_LNG_COLUMN],
            lat_spacing_deg=
            radar_metadata_dict[radar_utils.LAT_SPACING_COLUMN],
            lng_spacing_deg=
            radar_metadata_dict[radar_utils.LNG_SPACING_COLUMN])

        these_vertex_latitudes_deg, these_vertex_longitudes_deg = (
            radar_utils.rowcol_to_latlng(
                these_vertex_rows, these_vertex_columns,
                nw_grid_point_lat_deg=
                radar_metadata_dict[radar_utils.NW_GRID_POINT_LAT_COLUMN],
                nw_grid_point_lng_deg=
                radar_metadata_dict[radar_utils.NW_GRID_POINT_LNG_COLUMN],
                lat_spacing_deg=
                radar_metadata_dict[radar_utils.LAT_SPACING_COLUMN],
                lng_spacing_deg=
                radar_metadata_dict[radar_utils.LNG_SPACING_COLUMN])
        )

        local_max_dict[POLYGON_OBJECTS_ROWCOL_KEY][i] = (
            polygons.vertex_arrays_to_polygon_object(
                these_vertex_columns, these_vertex_rows)
        )
        local_max_dict[POLYGON_OBJECTS_LATLNG_KEY][i] = (
            polygons.vertex_arrays_to_polygon_object(
                these_vertex_longitudes_deg, these_vertex_latitudes_deg)
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
        [len(r) for r in local_max_dict[GRID_POINT_ROWS_KEY]],
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


def _join_tracks(
        early_storm_object_table, late_storm_object_table, projection_object,
        max_link_time_seconds, max_velocity_diff_m_s01,
        max_link_distance_m_s01):
    """Joins tracks across gap between two periods.

    :param early_storm_object_table: pandas DataFrame for early period.  Each
        row is one storm object.  Must contain the following columns.
    early_storm_object_table.storm_id: See doc for
        `_local_maxima_to_storm_tracks`.
    early_storm_object_table.unix_time_sec: Same.
    early_storm_object_table.centroid_lat_deg: Same.
    early_storm_object_table.centroid_lng_deg: Same.

    :param late_storm_object_table: Same as `early_storm_object_table` but for
        late period.
    :param projection_object: See doc for `_remove_redundant_local_maxima`.
    :param max_link_time_seconds: See doc for `_link_local_maxima_in_time`.
    :param max_velocity_diff_m_s01: Same.
    :param max_link_distance_m_s01: Same.
    :return: late_storm_object_table: Same as input, except that some storm IDs
        may be different.
    """

    num_early_objects = len(early_storm_object_table.index)
    num_late_objects = len(late_storm_object_table.index)

    if num_early_objects == 0 or num_late_objects == 0:
        return late_storm_object_table

    last_early_time_unix_sec = numpy.max(
        early_storm_object_table[tracking_utils.TIME_COLUMN].values
    )

    previous_indices = numpy.where(
        early_storm_object_table[tracking_utils.TIME_COLUMN] ==
        last_early_time_unix_sec
    )[0]

    previous_latitudes_deg = early_storm_object_table[
        tracking_utils.CENTROID_LAT_COLUMN].values[previous_indices]
    previous_longitudes_deg = early_storm_object_table[
        tracking_utils.CENTROID_LNG_COLUMN].values[previous_indices]
    previous_x_coords_metres, previous_y_coords_metres = (
        projections.project_latlng_to_xy(
            previous_latitudes_deg, previous_longitudes_deg,
            projection_object=projection_object,
            false_easting_metres=0., false_northing_metres=0.)
    )

    previous_x_velocities_m_s01, previous_y_velocities_m_s01 = (
        _latlng_velocities_to_xy(
            east_velocities_m_s01=early_storm_object_table[
                tracking_utils.EAST_VELOCITY_COLUMN].values[previous_indices],
            north_velocities_m_s01=early_storm_object_table[
                tracking_utils.NORTH_VELOCITY_COLUMN].values[previous_indices],
            latitudes_deg=previous_latitudes_deg,
            longitudes_deg=previous_longitudes_deg
        )
    )

    previous_local_max_dict = {
        X_COORDS_KEY: previous_x_coords_metres,
        Y_COORDS_KEY: previous_y_coords_metres,
        VALID_TIME_KEY: last_early_time_unix_sec,
        X_VELOCITIES_KEY: previous_x_velocities_m_s01,
        Y_VELOCITIES_KEY: previous_y_velocities_m_s01
    }

    first_late_time_unix_sec = numpy.min(
        late_storm_object_table[tracking_utils.TIME_COLUMN].values
    )
    current_indices = numpy.where(
        late_storm_object_table[tracking_utils.TIME_COLUMN] ==
        first_late_time_unix_sec
    )[0]

    current_latitudes_deg = late_storm_object_table[
        tracking_utils.CENTROID_LAT_COLUMN].values[current_indices]
    current_longitudes_deg = late_storm_object_table[
        tracking_utils.CENTROID_LNG_COLUMN].values[current_indices]
    current_x_coords_metres, current_y_coords_metres = (
        projections.project_latlng_to_xy(
            current_latitudes_deg, current_longitudes_deg,
            projection_object=projection_object,
            false_easting_metres=0., false_northing_metres=0.)
    )

    current_local_max_dict = {
        X_COORDS_KEY: current_x_coords_metres,
        Y_COORDS_KEY: current_y_coords_metres,
        VALID_TIME_KEY: first_late_time_unix_sec
    }

    current_to_previous_indices = temporal_tracking.link_local_maxima_in_time(
        current_local_max_dict=current_local_max_dict,
        previous_local_max_dict=previous_local_max_dict,
        max_link_time_seconds=max_link_time_seconds,
        max_velocity_diff_m_s01=max_velocity_diff_m_s01,
        max_link_distance_m_s01=max_link_distance_m_s01)

    previous_storm_ids = early_storm_object_table[
        tracking_utils.STORM_ID_COLUMN].values[previous_indices]
    orig_current_storm_ids = late_storm_object_table[
        tracking_utils.STORM_ID_COLUMN].values[current_indices]
    num_current_storms = len(orig_current_storm_ids)

    for i in range(num_current_storms):
        if current_to_previous_indices[i] == -1:
            continue

        this_new_storm_id = previous_storm_ids[current_to_previous_indices[i]]
        late_storm_object_table.replace(
            to_replace=orig_current_storm_ids[i], value=this_new_storm_id,
            inplace=True)

    return late_storm_object_table


def _storm_objects_to_tracks(
        storm_object_table, storm_track_table=None, recompute_for_id=None):
    """Converts storm objects to storm tracks.

    Storm object = one storm cell at one time step
    Storm track = one storm cell at all time steps

    If `storm_track_table` is not specified in the inputs, it will be computed
    from scratch.  If `storm_track_table` *is* specified in the inputs, only one
    storm track (based on `recompute_for_id`) will be recomputed.

    :param storm_object_table: pandas DataFrame with the following columns.
        Each row is one storm object.
    storm_object_table.storm_id: Storm ID (string).
    storm_object_table.unix_time_sec: Valid time.
    storm_object_table.centroid_lat_deg: Latitude (deg N) of centroid.
    storm_object_table.centroid_lng_deg: Longitude (deg E) of centroid.

    :param storm_track_table: pandas DataFrame with the following columns.  Each
        row is one storm track.
    storm_track_table.storm_id: Storm ID (string).
    storm_track_table.start_time_unix_sec: First time in track.
    storm_track_table.end_time_unix_sec: Last time in track.
    storm_track_table.start_latitude_deg: First latitude (deg N) in track.
    storm_track_table.end_latitude_deg: Last latitude (deg N) in track.
    storm_track_table.start_longitude_deg: First longitude (deg E) in track.
    storm_track_table.end_longitude_deg: Last longitude (deg E) in track.

    :param recompute_for_id: [used only if `storm_track_table is not None`]
        Storm track will be recomputed only for this ID (string).
    :return: storm_track_table: See input documentation.
    """

    if storm_track_table is None:
        storm_track_ids = numpy.unique(
            storm_object_table[tracking_utils.STORM_ID_COLUMN].values
        ).tolist()

        num_tracks = len(storm_track_ids)
        track_start_times_unix_sec = numpy.full(num_tracks, -1, dtype=int)
        track_end_times_unix_sec = numpy.full(num_tracks, -1, dtype=int)
        track_start_latitudes_deg = numpy.full(num_tracks, numpy.nan)
        track_start_longitudes_deg = numpy.full(num_tracks, numpy.nan)
        track_end_latitudes_deg = numpy.full(num_tracks, numpy.nan)
        track_end_longitudes_deg = numpy.full(num_tracks, numpy.nan)

        storm_track_dict = {
            tracking_utils.STORM_ID_COLUMN: storm_track_ids,
            START_TIME_COLUMN: track_start_times_unix_sec,
            END_TIME_COLUMN: track_end_times_unix_sec,
            START_LATITUDE_COLUMN: track_start_latitudes_deg,
            END_LATITUDE_COLUMN: track_start_longitudes_deg,
            START_LONGITUDE_COLUMN: track_end_latitudes_deg,
            END_LONGITUDE_COLUMN: track_end_longitudes_deg
        }

        storm_track_table = pandas.DataFrame.from_dict(storm_track_dict)
        recompute_for_ids = copy.deepcopy(storm_track_ids)
    else:
        recompute_for_ids = [recompute_for_id]

    for this_id in recompute_for_ids:
        these_object_indices = numpy.where(
            storm_object_table[tracking_utils.STORM_ID_COLUMN].values ==
            this_id
        )[0]

        these_times_unix_sec = storm_object_table[
            tracking_utils.TIME_COLUMN].values[these_object_indices]
        these_latitudes_deg = storm_object_table[
            tracking_utils.CENTROID_LAT_COLUMN].values[these_object_indices]
        these_longitudes_deg = storm_object_table[
            tracking_utils.CENTROID_LNG_COLUMN].values[these_object_indices]

        this_row = numpy.where(
            storm_track_table[tracking_utils.STORM_ID_COLUMN].values == this_id
        )[0][0]

        storm_track_table[START_TIME_COLUMN].values[this_row] = numpy.min(
            these_times_unix_sec)
        storm_track_table[END_TIME_COLUMN].values[this_row] = numpy.max(
            these_times_unix_sec)
        storm_track_table[START_LATITUDE_COLUMN].values[
            this_row] = these_latitudes_deg[numpy.argmin(these_times_unix_sec)]
        storm_track_table[START_LONGITUDE_COLUMN].values[
            this_row] = these_longitudes_deg[numpy.argmin(these_times_unix_sec)]
        storm_track_table[END_LATITUDE_COLUMN].values[
            this_row] = these_latitudes_deg[numpy.argmax(these_times_unix_sec)]
        storm_track_table[END_LONGITUDE_COLUMN].values[
            this_row] = these_longitudes_deg[numpy.argmax(these_times_unix_sec)]

    return storm_track_table


def _get_join_error(storm_track_table, early_track_id, late_track_id):
    """Finds error incurred by extrap early track to first time in late track.

    :param storm_track_table: pandas DataFrame created by
        `_storm_objects_to_tracks`.
    :param early_track_id: Storm ID for early track (string).
    :param late_track_id: Storm ID for late track (string).
    :return: join_error_metres: Extrapolation error.
    """

    # TODO(thunderhoser): Allow num points in velocity estimate to differ?

    early_track_index = numpy.where(
        storm_track_table[tracking_utils.STORM_ID_COLUMN].values ==
        early_track_id
    )[0][0]

    late_track_index = numpy.where(
        storm_track_table[tracking_utils.STORM_ID_COLUMN].values ==
        late_track_id
    )[0][0]

    early_time_diff_seconds = (
        storm_track_table[END_TIME_COLUMN].values[early_track_index] -
        storm_track_table[START_TIME_COLUMN].values[early_track_index]
    )

    if early_time_diff_seconds == 0:
        early_lat_speed_deg_s01 = 0.
        early_lng_speed_deg_s01 = 0.
    else:
        early_lat_speed_deg_s01 = (
            storm_track_table[END_LATITUDE_COLUMN].values[early_track_index] -
            storm_track_table[START_LATITUDE_COLUMN].values[early_track_index]
        ) / early_time_diff_seconds

        early_lng_speed_deg_s01 = (
            storm_track_table[END_LONGITUDE_COLUMN].values[early_track_index] -
            storm_track_table[START_LONGITUDE_COLUMN].values[early_track_index]
        ) / early_time_diff_seconds

    extrap_time_seconds = (
        storm_track_table[START_TIME_COLUMN].values[late_track_index] -
        storm_track_table[END_TIME_COLUMN].values[early_track_index]
    )

    extrap_latitude_deg = (
        storm_track_table[END_LATITUDE_COLUMN].values[early_track_index] +
        early_lat_speed_deg_s01 * extrap_time_seconds
    )

    extrap_longitude_deg = (
        storm_track_table[END_LONGITUDE_COLUMN].values[early_track_index] +
        early_lng_speed_deg_s01 * extrap_time_seconds
    )

    extrap_point = (extrap_latitude_deg, extrap_longitude_deg)
    late_track_start_point = (
        storm_track_table[START_LATITUDE_COLUMN].values[late_track_index],
        storm_track_table[START_LONGITUDE_COLUMN].values[late_track_index]
    )

    return vincenty(extrap_point, late_track_start_point).meters


def _find_nearby_tracks(storm_track_table, late_track_id, max_time_diff_seconds,
                        max_join_error_m_s01):
    """Finds tracks both spatially and temporally near the late track.

    :param storm_track_table: pandas DataFrame created by
        `_storm_objects_to_tracks`.
    :param late_track_id: Storm ID of late track (string).
    :param max_time_diff_seconds: Max time difference (between end of early
        track and start of late track).
    :param max_join_error_m_s01: Max join error (metres per second), created by
        extrapolating early track to first time in late track.
    :return: nearby_track_indices: 1-D numpy array with indices of nearby
        tracks.  These indices are rows of `storm_track_table`, sorted primarily
        by time difference and secondarily by join error.
    """

    late_track_index = numpy.where(
        storm_track_table[tracking_utils.STORM_ID_COLUMN].values ==
        late_track_id
    )[0][0]

    time_diffs_seconds = (
        storm_track_table[START_TIME_COLUMN].values[late_track_index] -
        storm_track_table[END_TIME_COLUMN].values
    )

    time_diffs_seconds[time_diffs_seconds <= 0] = max_time_diff_seconds + 1

    num_tracks = len(storm_track_table.index)
    join_errors_metres = numpy.full(num_tracks, numpy.inf)

    for j in range(num_tracks):
        if time_diffs_seconds[j] > max_time_diff_seconds:
            continue

        this_latitude_diff_deg = numpy.absolute(
            storm_track_table[START_LATITUDE_COLUMN].values[late_track_index] -
            storm_track_table[END_LATITUDE_COLUMN].values[j]
        )

        this_latitude_diff_m_s01 = this_latitude_diff_deg * (
            DEGREES_LAT_TO_METRES / time_diffs_seconds[j]
        )

        if this_latitude_diff_m_s01 > max_join_error_m_s01:
            continue

        join_errors_metres[j] = _get_join_error(
            storm_track_table=storm_track_table, late_track_id=late_track_id,
            early_track_id=
            storm_track_table[tracking_utils.STORM_ID_COLUMN].values[j]
        )

    join_errors_m_s01 = join_errors_metres / time_diffs_seconds

    nearby_track_indices = numpy.where(numpy.logical_and(
        time_diffs_seconds <= max_time_diff_seconds,
        join_errors_m_s01 <= max_join_error_m_s01
    ))[0]

    if len(nearby_track_indices) == 0:
        return None

    sort_indices = numpy.lexsort((
        join_errors_metres[nearby_track_indices],
        time_diffs_seconds[nearby_track_indices]
    ))

    return nearby_track_indices[sort_indices]


def _write_new_tracks(storm_object_table, top_output_dir_name,
                      valid_times_unix_sec):
    """Writes tracking files (one Pickle file per time step).

    These files are the main output of both `run_tracking` and
    `reanalyze_across_spc_dates`.

    :param storm_object_table: See doc for
        `storm_tracking_io.write_processed_file`.
    :param top_output_dir_name: Name of top-level directory.  File locations
        therein will be determined by `storm_tracking_io.find_processed_file`.
    :param valid_times_unix_sec: 1-D numpy array of valid times.  One file will
        be written for each.
    """

    for this_time_unix_sec in valid_times_unix_sec:
        this_file_name = tracking_io.find_processed_file(
            top_processed_dir_name=top_output_dir_name,
            unix_time_sec=this_time_unix_sec,
            spc_date_string=time_conversion.time_to_spc_date_string(
                this_time_unix_sec),
            tracking_scale_metres2=DUMMY_TRACKING_SCALE_METRES2,
            data_source=tracking_utils.SEGMOTION_SOURCE_ID,
            raise_error_if_missing=False)

        print 'Writing new data to: "{0:s}"...'.format(this_file_name)
        tracking_io.write_processed_file(
            storm_object_table=storm_object_table.loc[
                storm_object_table[tracking_utils.TIME_COLUMN] ==
                this_time_unix_sec
            ],
            pickle_file_name=this_file_name
        )


def _shuffle_tracking_data(
        storm_object_table_by_date, tracking_file_names_by_date,
        valid_times_by_date_unix_sec, current_date_index, top_output_dir_name):
    """Shuffles data into and out of memory.

    T = number of SPC dates

    :param storm_object_table_by_date: length-T list of pandas DataFrames.  If
        data for the [i]th date are currently out of memory,
        storm_object_table_by_date[i] = None.  If data for the [i]th date are
        currently in memory, storm_object_table_by_date[i] has columns listed in
        `storm_tracking_io.write_processed_file`.
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

        storm_object_table_by_date[j] = tracking_io.read_many_processed_files(
            tracking_file_names_by_date[j]
        )
        print '\n'

    return storm_object_table_by_date


def _reanalyze_tracks(
        storm_object_table, max_join_time_sec, max_join_error_m_s01):
    """Reanalyzes storm tracks.

    Specifically, joins pairs of tracks that are spatiotemporally nearby.

    :param storm_object_table: pandas DataFrame with the following columns.
    storm_object_table.storm_id: Storm ID (string).
    storm_object_table.unix_time_sec: Valid time.
    storm_object_table.centroid_lat_deg: Latitude (deg N) of centroid.
    storm_object_table.centroid_lng_deg: Longitude (deg E) of centroid.

    :param max_join_time_sec: See doc for `_find_nearby_tracks`.
    :param max_join_error_m_s01: Same.
    :return: storm_object_table: Same as input, except that some IDs may be
        different.
    """

    storm_track_table = _storm_objects_to_tracks(storm_object_table)

    storm_ids_removed = []
    num_storm_tracks = len(storm_track_table.index)

    for i in range(num_storm_tracks):
        if numpy.mod(i, 50) == 0:
            print 'Have reanalyzed {0:d} of {1:d} tracks...'.format(
                i, num_storm_tracks)

        this_storm_id = storm_track_table[
            tracking_utils.STORM_ID_COLUMN].values[i]
        if this_storm_id in storm_ids_removed:
            continue

        while True:
            these_nearby_indices = _find_nearby_tracks(
                storm_track_table=storm_track_table,
                late_track_id=this_storm_id,
                max_time_diff_seconds=max_join_time_sec,
                max_join_error_m_s01=max_join_error_m_s01)

            if these_nearby_indices is None:
                break

            # Change ID of first nearby track.
            this_nearby_index = these_nearby_indices[0]
            this_nearby_storm_id = storm_track_table[
                tracking_utils.STORM_ID_COLUMN].values[this_nearby_index]

            this_replacement_dict = {
                tracking_utils.STORM_ID_COLUMN: {
                    this_nearby_storm_id: this_storm_id
                }
            }

            storm_object_table.replace(
                to_replace=this_replacement_dict, inplace=True)

            # Housekeeping.
            storm_ids_removed.append(this_nearby_storm_id)
            storm_track_table[END_TIME_COLUMN].values[
                this_nearby_index] = DUMMY_TIME_UNIX_SEC

            storm_track_table = _storm_objects_to_tracks(
                storm_object_table=storm_object_table,
                storm_track_table=storm_track_table,
                recompute_for_id=this_storm_id)

    print 'Have reanalyzed all {0:d} tracks!'.format(num_storm_tracks)
    return storm_object_table


def _latlng_velocities_to_xy(
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


def remove_short_lived_tracks(storm_object_table, min_duration_seconds):
    """Removes short-lived storm tracks.

    :param storm_object_table: pandas DataFrame created by
        `_local_maxima_to_storm_tracks`.
    :param min_duration_seconds: Minimum duration.
    :return: storm_object_table: Same as input but maybe with fewer rows.
    """

    error_checking.assert_is_integer(min_duration_seconds)
    error_checking.assert_is_greater(min_duration_seconds, 0)

    all_primary_id_strings = numpy.array(
        storm_object_table[PRIMARY_STORM_ID_COLUMN].values
    )

    unique_primary_id_strings, orig_to_unique_indices = numpy.unique(
        all_primary_id_strings, return_inverse=True)

    num_storm_cells = len(unique_primary_id_strings)
    object_indices_to_remove = numpy.array([], dtype=int)

    for i in range(num_storm_cells):
        these_object_indices = numpy.where(orig_to_unique_indices == i)[0]
        these_times_unix_sec = storm_object_table[
            tracking_utils.TIME_COLUMN
        ].values[these_object_indices]

        this_duration_seconds = (
            numpy.max(these_times_unix_sec) - numpy.min(these_times_unix_sec)
        )
        if this_duration_seconds >= min_duration_seconds:
            continue

        object_indices_to_remove = numpy.concatenate((
            object_indices_to_remove, these_object_indices))

    return storm_object_table.drop(
        storm_object_table.index[object_indices_to_remove], axis=0,
        inplace=False)


def run_tracking(
        top_radar_dir_name, top_output_dir_name, first_spc_date_string,
        last_spc_date_string, first_time_unix_sec=None, last_time_unix_sec=None,
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
        min_track_duration_seconds=0,
        num_points_back_for_velocity=DEFAULT_NUM_POINTS_FOR_VELOCITY):
    """Runs echo-top-tracking.  This is effectively the main method.

    :param top_radar_dir_name: See doc for `_find_input_radar_files`.
    :param top_output_dir_name: See doc for `_write_storm_objects`.
    :param first_spc_date_string: See doc for `_check_time_period`.
    :param last_spc_date_string: Same.
    :param first_time_unix_sec: Same.
    :param last_time_unix_sec: Same.
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
    :param max_link_time_seconds: See doc for `_link_local_maxima_in_time`.
    :param max_velocity_diff_m_s01: Same.
    :param max_link_distance_m_s01: Same.
    :param min_track_duration_seconds: See doc for `remove_short_lived_tracks`.
    :param num_points_back_for_velocity: See doc for `_get_final_velocities`.
    """

    if min_polygon_size_pixels is None:
        min_polygon_size_pixels = 0

    error_checking.assert_is_integer(min_polygon_size_pixels)
    error_checking.assert_is_geq(min_polygon_size_pixels, 0)
    error_checking.assert_is_greater(min_echo_top_km, 0.)

    radar_file_names, valid_times_unix_sec = _find_input_radar_files(
        top_radar_dir_name=top_radar_dir_name,
        radar_field_name=echo_top_field_name,
        radar_source_name=radar_source_name,
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string,
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec)

    num_times = len(valid_times_unix_sec)
    valid_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT)
        for t in valid_times_unix_sec
    ]

    projection_object = projections.init_azimuthal_equidistant_projection(
        central_latitude_deg=CENTRAL_PROJ_LATITUDE_DEG,
        central_longitude_deg=CENTRAL_PROJ_LONGITUDE_DEG)

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
                    valid_time_unix_sec=valid_times_unix_sec[i],
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

        print 'Finding local maxima in "{0:s}" at {1:s}...'.format(
            echo_top_field_name, valid_time_strings[i])

        this_latitude_spacing_deg = this_metadata_dict[
            radar_utils.LAT_SPACING_COLUMN]

        this_echo_top_matrix_km = _gaussian_smooth_radar_field(
            radar_matrix=this_echo_top_matrix_km,
            e_folding_radius_pixels=
            smoothing_radius_deg_lat / this_latitude_spacing_deg
        )

        if this_echo_classifn_file_name is not None:
            print 'Reading data from: "{0:s}"...'.format(
                this_echo_classifn_file_name)

            this_convective_flag_matrix = echo_classifn.read_classifications(
                this_echo_classifn_file_name
            )[0]

            this_convective_flag_matrix = numpy.flip(
                this_convective_flag_matrix, axis=0)
            this_echo_top_matrix_km[this_convective_flag_matrix == False] = 0.

        this_half_width_pixels = int(numpy.round(
            half_width_for_max_filter_deg_lat / this_latitude_spacing_deg
        ))

        local_max_dict_by_time[i] = _find_local_maxima(
            radar_matrix=this_echo_top_matrix_km,
            radar_metadata_dict=this_metadata_dict,
            neigh_half_width_pixels=this_half_width_pixels)

        local_max_dict_by_time[i].update(
            {VALID_TIME_KEY: valid_times_unix_sec[i]}
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
            ).format(valid_time_strings[i], valid_time_strings[i - 1])

            this_current_to_prev_matrix = (
                temporal_tracking.link_local_maxima_in_time(
                    current_local_max_dict=local_max_dict_by_time[i],
                    previous_local_max_dict=local_max_dict_by_time[i - 1],
                    max_link_time_seconds=max_link_time_seconds,
                    max_velocity_diff_m_s01=max_velocity_diff_m_s01,
                    max_link_distance_m_s01=max_link_distance_m_s01)
            )

        local_max_dict_by_time[i].update(
            {CURRENT_TO_PREV_MATRIX_KEY: this_current_to_prev_matrix}
        )

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

    keep_time_indices = numpy.array(keep_time_indices, dtype=int)
    valid_times_unix_sec = valid_times_unix_sec[keep_time_indices]
    local_max_dict_by_time = [
        local_max_dict_by_time[k] for k in keep_time_indices
    ]

    print SEPARATOR_STRING
    print 'Converting time series of "{0:s}" maxima to storm tracks...'.format(
        echo_top_field_name)
    storm_object_table = temporal_tracking.local_maxima_to_storm_tracks(
        local_max_dict_by_time)

    print 'Removing tracks that last < {0:d} seconds...'.format(
        int(min_track_duration_seconds)
    )
    storm_object_table = remove_short_lived_tracks(
        storm_object_table=storm_object_table,
        min_duration_seconds=min_track_duration_seconds)

    print 'Computing storm ages...'
    storm_object_table = best_tracks.get_storm_ages(
        storm_object_table=storm_object_table,
        best_track_start_time_unix_sec=valid_times_unix_sec[0],
        best_track_end_time_unix_sec=valid_times_unix_sec[-1],
        max_extrap_time_for_breakup_sec=max_link_time_seconds,
        max_join_time_sec=max_link_time_seconds)

    print 'Computing storm velocities...'
    storm_object_table = _get_final_velocities(
        storm_object_table=storm_object_table,
        num_points_back=num_points_back_for_velocity)

    print SEPARATOR_STRING
    _write_new_tracks(
        storm_object_table=storm_object_table,
        top_output_dir_name=top_output_dir_name,
        valid_times_unix_sec=valid_times_unix_sec)


def reanalyze_across_spc_dates(
        top_input_dir_name, top_output_dir_name, first_spc_date_string,
        last_spc_date_string, first_time_unix_sec=None, last_time_unix_sec=None,
        tracking_start_time_unix_sec=None, tracking_end_time_unix_sec=None,
        max_link_time_seconds=DEFAULT_MAX_LINK_TIME_SECONDS,
        max_velocity_diff_m_s01=DEFAULT_MAX_VELOCITY_DIFF_M_S01,
        max_link_distance_m_s01=DEFAULT_MAX_LINK_DISTANCE_M_S01,
        max_join_time_sec=DEFAULT_MAX_JOIN_TIME_SEC,
        max_join_error_m_s01=DEFAULT_MAX_JOIN_ERROR_M_S01,
        min_track_duration_seconds=DEFAULT_MIN_REANALYZED_DURATION_SEC,
        num_points_back_for_velocity=DEFAULT_NUM_POINTS_FOR_VELOCITY):
    """Reanalyzes tracks across SPC dates.

    :param top_input_dir_name: Name of top-level directory with original tracks
        (before reanalysis).  For more details, see doc for
        `_find_input_tracking_files`.
    :param top_output_dir_name: Name of top-level directory for new tracks
        (after reanalysis).  For more details, see doc for
        `_write_storm_objects`.
    :param first_spc_date_string: See doc for `_find_input_tracking_files`.
    :param last_spc_date_string: Same.
    :param first_time_unix_sec: Same.
    :param last_time_unix_sec: Same.
    :param tracking_start_time_unix_sec: First time in tracking period.  If
        `tracking_start_time_unix_sec is None`, defaults to
        `first_time_unix_sec`.
    :param tracking_end_time_unix_sec: Last time in tracking period.  If
        `tracking_end_time_unix_sec is None`, defaults to `last_time_unix_sec`.
    :param max_link_time_seconds: See doc for `_link_local_maxima_in_time`.
    :param max_velocity_diff_m_s01: Same.
    :param max_link_distance_m_s01: Same.
    :param max_join_time_sec: See doc for `_reanalyze_tracks`.
    :param max_join_error_m_s01: Same.
    :param min_track_duration_seconds: See doc for `remove_short_lived_tracks`.
    :param num_points_back_for_velocity: See doc for `_get_final_velocities`.
    """

    (spc_date_strings, tracking_file_names_by_date, valid_times_by_date_unix_sec
    ) = _find_input_tracking_files(
        top_tracking_dir_name=top_input_dir_name,
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string,
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec)

    spc_dates_unix_sec = numpy.array([
        time_conversion.spc_date_string_to_unix_sec(d) for d in spc_date_strings
    ], dtype=int)

    if (tracking_start_time_unix_sec is None
            or tracking_end_time_unix_sec is None):

        these_times_unix_sec = numpy.array(
            [numpy.min(t) for t in valid_times_by_date_unix_sec], dtype=int)
        tracking_start_time_unix_sec = numpy.min(these_times_unix_sec)

        these_times_unix_sec = numpy.array(
            [numpy.max(t) for t in valid_times_by_date_unix_sec], dtype=int)
        tracking_end_time_unix_sec = numpy.max(these_times_unix_sec)

    else:
        time_conversion.unix_sec_to_string(
            tracking_start_time_unix_sec, TIME_FORMAT)
        time_conversion.unix_sec_to_string(
            tracking_end_time_unix_sec, TIME_FORMAT)
        error_checking.assert_is_greater(
            tracking_end_time_unix_sec, tracking_start_time_unix_sec)

    projection_object = projections.init_azimuthal_equidistant_projection(
        central_latitude_deg=CENTRAL_PROJ_LATITUDE_DEG,
        central_longitude_deg=CENTRAL_PROJ_LONGITUDE_DEG)

    num_spc_dates = len(spc_date_strings)

    if num_spc_dates == 1:
        storm_object_table = tracking_io.read_many_processed_files(
            tracking_file_names_by_date[0])
        print SEPARATOR_STRING

        print 'Reanalyzing tracks for {0:s}...'.format(spc_date_strings[0])
        storm_object_table = _reanalyze_tracks(
            storm_object_table=storm_object_table,
            max_join_time_sec=max_join_time_sec,
            max_join_error_m_s01=max_join_error_m_s01)
        print SEPARATOR_STRING

        print 'Removing tracks that last < {0:d} seconds...'.format(
            int(min_track_duration_seconds)
        )
        storm_object_table = remove_short_lived_tracks(
            storm_object_table=storm_object_table,
            min_duration_seconds=min_track_duration_seconds)

        print 'Recomputing storm ages...'
        storm_object_table = best_tracks.get_storm_ages(
            storm_object_table=storm_object_table,
            best_track_start_time_unix_sec=tracking_start_time_unix_sec,
            best_track_end_time_unix_sec=tracking_end_time_unix_sec,
            max_extrap_time_for_breakup_sec=max_link_time_seconds,
            max_join_time_sec=max_join_time_sec)

        print 'Recomputing storm velocities...'

        these_x_coords_metres, these_y_coords_metres = (
            projections.project_latlng_to_xy(
                latitudes_deg=storm_object_table[
                    tracking_utils.CENTROID_LAT_COLUMN].values,
                longitudes_deg=storm_object_table[
                    tracking_utils.CENTROID_LNG_COLUMN].values,
                projection_object=projection_object,
                false_easting_metres=0., false_northing_metres=0.)
        )

        argument_dict = {
            CENTROID_X_COLUMN: these_x_coords_metres,
            CENTROID_Y_COLUMN: these_y_coords_metres
        }

        storm_object_table = storm_object_table.assign(**argument_dict)

        storm_object_table = _get_final_velocities(
            storm_object_table=storm_object_table,
            num_points_back=num_points_back_for_velocity)

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
            print 'Joining tracks between {0:s} and {1:s}...'.format(
                spc_date_strings[i], spc_date_strings[i + 1])

            storm_object_table_by_date[i + 1] = _join_tracks(
                early_storm_object_table=storm_object_table_by_date[i],
                late_storm_object_table=storm_object_table_by_date[i + 1],
                projection_object=projection_object,
                max_link_time_seconds=max_link_time_seconds,
                max_velocity_diff_m_s01=max_velocity_diff_m_s01,
                max_link_distance_m_s01=max_link_distance_m_s01)

            print 'Reanalyzing tracks for {0:s} and {1:s}...'.format(
                spc_date_strings[i], spc_date_strings[i + 1]
            )

            indices_to_concat = numpy.array([i, i + 1], dtype=int)
            concat_storm_object_table = pandas.concat(
                [storm_object_table_by_date[k] for k in indices_to_concat],
                axis=0, ignore_index=True)

            concat_storm_object_table = _reanalyze_tracks(
                storm_object_table=concat_storm_object_table,
                max_join_time_sec=max_join_time_sec,
                max_join_error_m_s01=max_join_error_m_s01)
            print MINOR_SEPARATOR_STRING

            storm_object_table_by_date[i] = concat_storm_object_table.loc[
                concat_storm_object_table[tracking_utils.SPC_DATE_COLUMN] ==
                spc_dates_unix_sec[i]
            ]
            storm_object_table_by_date[i + 1] = concat_storm_object_table.loc[
                concat_storm_object_table[tracking_utils.SPC_DATE_COLUMN] ==
                spc_dates_unix_sec[i + 1]
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
        concat_storm_object_table = remove_short_lived_tracks(
            storm_object_table=concat_storm_object_table,
            min_duration_seconds=min_track_duration_seconds)

        print 'Recomputing storm ages...'
        concat_storm_object_table = best_tracks.get_storm_ages(
            storm_object_table=concat_storm_object_table,
            best_track_start_time_unix_sec=tracking_start_time_unix_sec,
            best_track_end_time_unix_sec=tracking_end_time_unix_sec,
            max_extrap_time_for_breakup_sec=max_link_time_seconds,
            max_join_time_sec=max_join_time_sec)

        print 'Recomputing storm velocities...'

        # TODO(thunderhoser): There is probably a more efficient way to add x-y
        # coords.

        these_x_coords_metres, these_y_coords_metres = (
            projections.project_latlng_to_xy(
                latitudes_deg=concat_storm_object_table[
                    tracking_utils.CENTROID_LAT_COLUMN].values,
                longitudes_deg=concat_storm_object_table[
                    tracking_utils.CENTROID_LNG_COLUMN].values,
                projection_object=projection_object,
                false_easting_metres=0., false_northing_metres=0.)
        )

        argument_dict = {
            CENTROID_X_COLUMN: these_x_coords_metres,
            CENTROID_Y_COLUMN: these_y_coords_metres
        }

        concat_storm_object_table = concat_storm_object_table.assign(
            **argument_dict)

        concat_storm_object_table = _get_final_velocities(
            storm_object_table=concat_storm_object_table,
            num_points_back=num_points_back_for_velocity)

        storm_object_table_by_date[i] = concat_storm_object_table.loc[
            concat_storm_object_table[tracking_utils.SPC_DATE_COLUMN] ==
            spc_dates_unix_sec[i]
        ]
        print SEPARATOR_STRING
