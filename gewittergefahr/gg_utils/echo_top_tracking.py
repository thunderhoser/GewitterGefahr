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
import numpy
import pandas
from scipy.ndimage.filters import gaussian_filter
from geopy.distance import vincenty
from gewittergefahr.gg_io import myrorss_and_mrms_io
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import radar_sparse_to_full as radar_s2f
from gewittergefahr.gg_utils import dilation
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import best_tracks
from gewittergefahr.gg_utils import echo_classification as echo_classifn
from gewittergefahr.gg_utils import error_checking

TOLERANCE = 1e-6
DUMMY_TIME_UNIX_SEC = -10000

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

DEGREES_LAT_TO_METRES = 60 * 1852
CENTRAL_PROJ_LATITUDE_DEG = 35.
CENTRAL_PROJ_LONGITUDE_DEG = 265.

VALID_RADAR_FIELDS = [
    radar_utils.ECHO_TOP_18DBZ_NAME, radar_utils.ECHO_TOP_40DBZ_NAME,
    radar_utils.ECHO_TOP_50DBZ_NAME
]
VALID_RADAR_SOURCE_NAMES = [
    radar_utils.MYRORSS_SOURCE_ID, radar_utils.MRMS_SOURCE_ID
]

DEFAULT_MIN_ECHO_TOP_HEIGHT_KM_ASL = 4.
DEFAULT_E_FOLD_RADIUS_FOR_SMOOTHING_DEG_LAT = 0.024
DEFAULT_HALF_WIDTH_FOR_MAX_FILTER_DEG_LAT = 0.06
DEFAULT_MIN_DISTANCE_BETWEEN_MAXIMA_METRES = 0.1 * DEGREES_LAT_TO_METRES
DEFAULT_MAX_LINK_TIME_SECONDS = 305
DEFAULT_MAX_LINK_DISTANCE_M_S01 = (
    0.125 * DEGREES_LAT_TO_METRES / DEFAULT_MAX_LINK_TIME_SECONDS)

DEFAULT_MAX_REANAL_JOIN_TIME_SEC = 600
DEFAULT_MAX_REANAL_EXTRAP_ERROR_M_S01 = 20.

DEFAULT_MIN_TRACK_DURATION_SHORT_SECONDS = 0
DEFAULT_MIN_TRACK_DURATION_LONG_SECONDS = 900
DEFAULT_NUM_POINTS_BACK_FOR_VELOCITY = 3
DUMMY_TRACKING_SCALE_METRES2 = numpy.pi * 1e8  # Radius of 10 km.

TRACKING_FILE_NAMES_KEY = 'output_tracking_file_names'
VALID_TIMES_KEY = 'unix_times_sec'

LATITUDES_KEY = 'latitudes_deg'
LONGITUDES_KEY = 'longitudes_deg'
MAX_VALUES_KEY = 'max_values'
X_COORDS_KEY = 'x_coords_metres'
Y_COORDS_KEY = 'y_coords_metres'
VALID_TIME_KEY = 'unix_time_sec'
CURRENT_TO_PREV_INDICES_KEY = 'current_to_previous_indices'
STORM_IDS_KEY = 'storm_ids'

CENTROID_X_COLUMN = 'centroid_x_metres'
CENTROID_Y_COLUMN = 'centroid_y_metres'

GRID_POINT_ROWS_KEY = 'list_of_grid_point_rows'
GRID_POINT_COLUMNS_KEY = 'list_of_grid_point_columns'
GRID_POINT_LATITUDES_KEY = 'list_of_grid_point_latitudes_deg'
GRID_POINT_LONGITUDES_KEY = 'list_of_grid_point_longitudes_deg'
POLYGON_OBJECTS_ROWCOL_KEY = 'polygon_objects_rowcol'
POLYGON_OBJECTS_LATLNG_KEY = 'polygon_objects_latlng'

START_TIME_COLUMN = 'start_time_unix_sec'
END_TIME_COLUMN = 'end_time_unix_sec'
START_LATITUDE_COLUMN = 'start_latitude_deg'
END_LATITUDE_COLUMN = 'end_latitude_deg'
START_LONGITUDE_COLUMN = 'start_longitude_deg'
END_LONGITUDE_COLUMN = 'end_longitude_deg'


def _check_radar_field(radar_field_name):
    """Ensures that radar field is valid for echo-top-based tracking.

    :param radar_field_name: Field name (string).
    :raises: ValueError: if `radar_field_name not in VALID_RADAR_FIELDS`.
    """

    error_checking.assert_is_string(radar_field_name)

    if radar_field_name not in VALID_RADAR_FIELDS:
        error_string = (
            '\n\n{0:s}\n\nValid radar fields (listed above) do not include '
            '"{1:s}".').format(VALID_RADAR_FIELDS, radar_field_name)
        raise ValueError(error_string)


def _check_radar_source(radar_source_name):
    """Error-checks source of radar data.

    :param radar_source_name: Data source (must be in list
        `VALID_RADAR_SOURCE_NAMES`).
    :raises: ValueError: if `radar_source_name not in VALID_RADAR_SOURCE_NAMES`.
    """

    error_checking.assert_is_string(radar_source_name)

    if radar_source_name not in VALID_RADAR_SOURCE_NAMES:
        error_string = (
            '\n\n{0:s}\n\nValid data sources (listed above) do not include '
            '"{1:s}".').format(VALID_RADAR_SOURCE_NAMES, radar_source_name)
        raise ValueError(error_string)


def _gaussian_smooth_radar_field(
        radar_matrix, e_folding_radius_pixels, cutoff_radius_pixels=None):
    """Applies Gaussian smoother to radar field.  NaN's are treated as zero.

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)

    :param radar_matrix: M-by-N numpy array with values of radar field.
    :param e_folding_radius_pixels: e-folding radius for Gaussian smoother.
    :param cutoff_radius_pixels: Cutoff radius for Gaussian smoother.  Default
        is 3 * e-folding radius.
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
        numpy.absolute(smoothed_radar_matrix) < TOLERANCE] = numpy.nan
    return smoothed_radar_matrix


def _find_local_maxima(
        radar_matrix, radar_metadata_dict, neigh_half_width_in_pixels):
    """Finds local maxima in radar field.

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)
    P = number of local maxima

    :param radar_matrix: M-by-N numpy array with values of radar field.
    :param radar_metadata_dict: Dictionary with metadata for radar grid, created
        by `myrorss_and_mrms_io.read_metadata_from_raw_file`.
    :param neigh_half_width_in_pixels: Neighbourhood half-width for max filter.
    :return: local_max_dict_latlng: Dictionary with the following keys.
    local_max_dict_latlng['latitudes_deg']: length-P numpy array with latitudes
        (deg N) of local maxima.
    local_max_dict_latlng['longitudes_deg']: length-P numpy array with
        longitudes (deg E) of local maxima.
    local_max_dict_latlng['max_values']: length-P numpy array with values of
        local maxima.
    """

    filtered_radar_matrix = dilation.dilate_2d_matrix(
        input_matrix=radar_matrix, percentile_level=100.,
        half_width_in_pixels=neigh_half_width_in_pixels)

    max_index_arrays = numpy.where(
        numpy.absolute(filtered_radar_matrix - radar_matrix) < TOLERANCE)
    max_row_indices = max_index_arrays[0]
    max_column_indices = max_index_arrays[1]

    max_latitudes_deg, max_longitudes_deg = radar_utils.rowcol_to_latlng(
        grid_rows=max_row_indices, grid_columns=max_column_indices,
        nw_grid_point_lat_deg=
        radar_metadata_dict[radar_utils.NW_GRID_POINT_LAT_COLUMN],
        nw_grid_point_lng_deg=
        radar_metadata_dict[radar_utils.NW_GRID_POINT_LNG_COLUMN],
        lat_spacing_deg=radar_metadata_dict[radar_utils.LAT_SPACING_COLUMN],
        lng_spacing_deg=radar_metadata_dict[radar_utils.LNG_SPACING_COLUMN])

    max_values = radar_matrix[max_row_indices, max_column_indices]
    sort_indices = numpy.argsort(-max_values)
    max_values = max_values[sort_indices]
    max_latitudes_deg = max_latitudes_deg[sort_indices]
    max_longitudes_deg = max_longitudes_deg[sort_indices]

    return {
        LATITUDES_KEY: max_latitudes_deg, LONGITUDES_KEY: max_longitudes_deg,
        MAX_VALUES_KEY: max_values}


def _remove_redundant_local_maxima(
        local_max_dict_latlng, projection_object,
        min_distance_between_maxima_metres=
        DEFAULT_MIN_DISTANCE_BETWEEN_MAXIMA_METRES):
    """Removes redundant local maxima in radar field.

    P = number of local maxima

    :param local_max_dict_latlng: Dictionary created by _find_local_maxima.
    :param projection_object: `pyproj.Proj` object, which will be used to
        convert lat-long coordinates to x-y.
    :param min_distance_between_maxima_metres: Minimum distance between any pair
        of local maxima.
    :return: local_max_dictionary: Dictionary with the following keys.
    local_max_dictionary['latitudes_deg']: length-P numpy array with latitudes
        (deg N) of local maxima.
    local_max_dictionary['longitudes_deg']: length-P numpy array with
        longitudes (deg E) of local maxima.
    local_max_dictionary['x_coords_metres']: length-P numpy array with
        x-coordinates of local maxima.
    local_max_dictionary['y_coords_metres']: length-P numpy array with
        y-coordinates of local maxima.
    local_max_dictionary['max_values']: length-P numpy array with values of
        local maxima.
    """

    max_x_coords_metres, max_y_coords_metres = projections.project_latlng_to_xy(
        local_max_dict_latlng[LATITUDES_KEY],
        local_max_dict_latlng[LONGITUDES_KEY],
        projection_object=projection_object, false_easting_metres=0.,
        false_northing_metres=0.)

    num_maxima = len(max_x_coords_metres)
    keep_max_flags = numpy.full(num_maxima, True, dtype=bool)

    for i in range(num_maxima):
        these_distances_metres = numpy.sqrt(
            (max_x_coords_metres - max_x_coords_metres[i]) ** 2 +
            (max_y_coords_metres - max_y_coords_metres[i]) ** 2)
        these_distances_metres[i] = numpy.inf
        keep_max_flags[
            these_distances_metres < min_distance_between_maxima_metres] = False

    keep_max_indices = numpy.where(keep_max_flags)[0]

    return {
        LATITUDES_KEY: local_max_dict_latlng[LATITUDES_KEY][keep_max_indices],
        LONGITUDES_KEY: local_max_dict_latlng[LONGITUDES_KEY][keep_max_indices],
        X_COORDS_KEY: max_x_coords_metres[keep_max_indices],
        Y_COORDS_KEY: max_y_coords_metres[keep_max_indices],
        MAX_VALUES_KEY: local_max_dict_latlng[MAX_VALUES_KEY][keep_max_indices]}


def _link_local_maxima_in_time(
        current_local_max_dict, previous_local_max_dict,
        max_link_time_seconds=DEFAULT_MAX_LINK_TIME_SECONDS,
        max_link_distance_m_s01=DEFAULT_MAX_LINK_DISTANCE_M_S01):
    """Links local maxima between current and previous time steps.

    N_c = number of local maxima at current time
    N_p = number of local maxima at previous time

    :param current_local_max_dict: Dictionary of local maxima for current time
        step.  Contains keys listed in `_remove_redundant_local_maxima`, plus
        those listed below.
    current_local_max_dict['valid_time_unix_sec']: Valid time.

    :param previous_local_max_dict: Same as `current_local_max_dict`, except for
        previous time step.
    :param max_link_time_seconds: Max difference between current and previous
        time steps.  If difference is > `max_link_time_seconds`, local maxima
        will not be linked.
    :param max_link_distance_m_s01: Max distance between current and previous
        time steps.  For two local maxima C and P (at current and previous time
        steps respectively), if distance is >
        `max_link_distance_m_s01 * max_link_time_seconds`, they cannot be
        linked.
    :return: current_to_previous_indices: numpy array (length N_c) with indices
        of previous local maxima to which current local maxima are linked.  In
        other words, if current_to_previous_indices[i] = j, the [i]th current
        local max is linked to the [j]th previous local max.
    """

    num_current_maxima = len(current_local_max_dict[X_COORDS_KEY])
    current_to_previous_indices = numpy.full(num_current_maxima, -1, dtype=int)
    if previous_local_max_dict is None:
        return current_to_previous_indices

    num_previous_maxima = len(previous_local_max_dict[X_COORDS_KEY])
    time_diff_seconds = (
        current_local_max_dict[VALID_TIME_KEY] -
        previous_local_max_dict[VALID_TIME_KEY])

    if (num_current_maxima == 0 or num_previous_maxima == 0 or
            time_diff_seconds > max_link_time_seconds):
        return current_to_previous_indices

    current_to_previous_distances_m_s01 = numpy.full(
        num_current_maxima, numpy.nan)

    for i in range(num_current_maxima):
        these_distances_metres = numpy.sqrt(
            (current_local_max_dict[X_COORDS_KEY][i] -
             previous_local_max_dict[X_COORDS_KEY]) ** 2 +
            (current_local_max_dict[Y_COORDS_KEY][i] -
             previous_local_max_dict[Y_COORDS_KEY]) ** 2)

        these_distances_m_s01 = these_distances_metres / time_diff_seconds
        this_min_distance_m_s01 = numpy.min(these_distances_m_s01)
        if this_min_distance_m_s01 > max_link_distance_m_s01:
            continue

        current_to_previous_distances_m_s01[i] = this_min_distance_m_s01
        this_best_prev_index = numpy.argmin(these_distances_m_s01)
        current_to_previous_indices[i] = this_best_prev_index

    for j in range(num_previous_maxima):
        these_current_indices = numpy.where(current_to_previous_indices == j)[0]
        if len(these_current_indices) < 2:
            continue

        this_best_current_index = numpy.argmin(
            current_to_previous_distances_m_s01[these_current_indices])
        this_best_current_index = these_current_indices[this_best_current_index]

        for i in these_current_indices:
            if i == this_best_current_index:
                continue

            current_to_previous_indices[i] = -1

    return current_to_previous_indices


def _find_input_radar_files(
        top_radar_dir_name, echo_top_field_name, radar_source_name,
        first_spc_date_string, last_spc_date_string, first_time_unix_sec,
        last_time_unix_sec):
    """Finds radar files (inputs to tracking algorithm).

    N = number of files found

    :param top_radar_dir_name: Name of top-level directory with radar files.
        Files therein will be found by
        `myrorss_and_mrms_io.find_raw_files_one_spc_date`.
    :param echo_top_field_name: Name of radar field (must be accepted by
        `radar_utils.check_field_name`).
    :param radar_source_name: Data source (must be accepted by
        `_check_radar_source`).
    :param first_spc_date_string: First SPC date in period (format "yyyymmdd").
    :param last_spc_date_string: Last SPC date in period (format "yyyymmdd").
    :param first_time_unix_sec: First time in period.  Default is 120000 UTC at
        beginning of first SPC date.
    :param last_time_unix_sec: Last time in period.  Default is 115959 UTC at
        end of first SPC date.
    :return: radar_file_names: length-N list of paths to radar files.
    :return: valid_times_unix_sec: length-N numpy array of valid times.
    """

    # Error-checking.
    _check_radar_field(echo_top_field_name)
    _check_radar_source(radar_source_name)

    spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string)

    if first_time_unix_sec is None:
        first_time_unix_sec = (
            time_conversion.MIN_SECONDS_INTO_SPC_DATE +
            time_conversion.string_to_unix_sec(
                first_spc_date_string, time_conversion.SPC_DATE_FORMAT)
        )

    if last_time_unix_sec is None:
        last_time_unix_sec = (
            time_conversion.MAX_SECONDS_INTO_SPC_DATE +
            time_conversion.string_to_unix_sec(
                last_spc_date_string, time_conversion.SPC_DATE_FORMAT)
        )

    error_checking.assert_is_greater(last_time_unix_sec, first_time_unix_sec)
    assert time_conversion.is_time_in_spc_date(
        first_time_unix_sec, first_spc_date_string)
    assert time_conversion.is_time_in_spc_date(
        last_time_unix_sec, last_spc_date_string)

    # Find files.
    radar_file_names = []
    valid_times_unix_sec = numpy.array([], dtype=int)
    num_spc_dates = len(spc_date_strings)

    for i in range(num_spc_dates):
        these_file_names = myrorss_and_mrms_io.find_raw_files_one_spc_date(
            spc_date_string=spc_date_strings[i],
            field_name=echo_top_field_name, data_source=radar_source_name,
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
            valid_times_unix_sec, these_times_unix_sec[good_indices]))

    sort_indices = numpy.argsort(valid_times_unix_sec)
    valid_times_unix_sec = valid_times_unix_sec[sort_indices]
    radar_file_names = [radar_file_names[k] for k in sort_indices]

    return radar_file_names, valid_times_unix_sec


def _create_storm_id(
        storm_start_time_unix_sec, prev_numeric_id_used, prev_spc_date_string):
    """Creates storm ID.

    :param storm_start_time_unix_sec: Start time of storm for which ID is being
        created.
    :param prev_numeric_id_used: Previous numeric ID (integer) used in the
        dataset.
    :param prev_spc_date_string: Previous SPC date (format "yyyymmdd") in the
        dataset.
    :return: string_id: String ID for new storm.
    :return: numeric_id: Numeric ID for new storm.
    :return: spc_date_string: SPC date (format "yyyymmdd") for new storm.
    """

    spc_date_string = time_conversion.time_to_spc_date_string(
        storm_start_time_unix_sec)

    if spc_date_string == prev_spc_date_string:
        numeric_id = prev_numeric_id_used + 1
    else:
        numeric_id = 0

    string_id = '{0:06d}_{1:s}'.format(numeric_id, spc_date_string)
    return string_id, numeric_id, spc_date_string


def _local_maxima_to_storm_tracks(local_max_dict_by_time):
    """Converts time series of local maxima to storm tracks.

    N = number of time steps
    P = number of local maxima at a given time

    :param local_max_dict_by_time: length-N list of dictionaries with the
        following keys.
    local_max_dict_by_time[i]['latitudes_deg']: length-P numpy array with
        latitudes (deg N) of local maxima at the [i]th time.
    local_max_dict_by_time[i]['longitudes_deg']: length-P numpy array with
        longitudes (deg E) of local maxima at the [i]th time.
    local_max_dict_by_time[i]['x_coords_metres']: length-P numpy array with
        x-coordinates of local maxima at the [i]th time.
    local_max_dict_by_time[i]['y_coords_metres']: length-P numpy array with
        y-coordinates of local maxima at the [i]th time.
    local_max_dict_by_time[i]['unix_time_sec']: The [i]th time.
    local_max_dict_by_time[i]['current_to_previous_indices']: length-P numpy
        array with indices of previous local maxima to which current local
        maxima are linked.  In other words, if current_to_previous_indices[j]
        = k, the [j]th current local max is linked to the [k]th previous local
        max.

    :return: storm_object_table: pandas DataFrame with the following columns,
        where each row is one storm object.
    storm_object_table.storm_id: Storm ID (string).  All objects with the same
        ID belong to the same track.
    storm_object_table.unix_time_sec: Valid time.
    storm_object_table.spc_date_unix_sec: SPC date.
    storm_object_table.centroid_lat_deg: Latitude (deg N) at center of storm
        object.
    storm_object_table.centroid_lng_deg: Longitude (deg E) at center of storm
        object.
    storm_object_table.centroid_x_metres: x-coordinate at center of storm
        object.
    storm_object_table.centroid_y_metres: y-coordinate at center of storm
        object.
    """

    num_times = len(local_max_dict_by_time)
    prev_numeric_id_used = -1
    prev_spc_date_string = '00000101'

    all_storm_ids = []
    all_times_unix_sec = numpy.array([], dtype=int)
    all_spc_dates_unix_sec = numpy.array([], dtype=int)
    all_centroid_latitudes_deg = numpy.array([])
    all_centroid_longitudes_deg = numpy.array([])
    all_centroid_x_metres = numpy.array([])
    all_centroid_y_metres = numpy.array([])

    include_polygons = POLYGON_OBJECTS_LATLNG_KEY in local_max_dict_by_time[0]
    if include_polygons:
        all_grid_point_rows_2d_list = []
        all_grid_point_columns_2d_list = []
        all_grid_point_latitudes_deg_2d_list = []
        all_grid_point_longitudes_deg_2d_list = []
        all_polygon_objects_latlng = numpy.array([], dtype=object)
        all_polygon_objects_rowcol = numpy.array([], dtype=object)

    for i in range(num_times):
        this_num_storm_objects = len(local_max_dict_by_time[i][LATITUDES_KEY])
        if this_num_storm_objects == 0:
            continue

        local_max_dict_by_time[i].update(
            {STORM_IDS_KEY: [''] * this_num_storm_objects})

        for j in range(this_num_storm_objects):
            this_previous_index = local_max_dict_by_time[
                i][CURRENT_TO_PREV_INDICES_KEY][j]

            if this_previous_index == -1:
                (local_max_dict_by_time[i][STORM_IDS_KEY][j],
                 prev_numeric_id_used, prev_spc_date_string) = _create_storm_id(
                     storm_start_time_unix_sec=
                     local_max_dict_by_time[i][VALID_TIME_KEY],
                     prev_numeric_id_used=prev_numeric_id_used,
                     prev_spc_date_string=prev_spc_date_string)

            else:
                local_max_dict_by_time[i][STORM_IDS_KEY][j] = (
                    local_max_dict_by_time[i - 1][STORM_IDS_KEY][
                        this_previous_index])

        these_times_unix_sec = numpy.full(
            this_num_storm_objects, local_max_dict_by_time[i][VALID_TIME_KEY],
            dtype=int)
        these_spc_dates_unix_sec = numpy.full(
            this_num_storm_objects,
            time_conversion.time_to_spc_date_unix_sec(these_times_unix_sec[0]),
            dtype=int)

        all_storm_ids += local_max_dict_by_time[i][STORM_IDS_KEY]
        all_times_unix_sec = numpy.concatenate((
            all_times_unix_sec, these_times_unix_sec))
        all_spc_dates_unix_sec = numpy.concatenate((
            all_spc_dates_unix_sec, these_spc_dates_unix_sec))
        all_centroid_latitudes_deg = numpy.concatenate((
            all_centroid_latitudes_deg,
            local_max_dict_by_time[i][LATITUDES_KEY]))
        all_centroid_longitudes_deg = numpy.concatenate((
            all_centroid_longitudes_deg,
            local_max_dict_by_time[i][LONGITUDES_KEY]))
        all_centroid_x_metres = numpy.concatenate((
            all_centroid_x_metres, local_max_dict_by_time[i][X_COORDS_KEY]))
        all_centroid_y_metres = numpy.concatenate((
            all_centroid_y_metres, local_max_dict_by_time[i][Y_COORDS_KEY]))

        if include_polygons:
            all_grid_point_rows_2d_list += local_max_dict_by_time[i][
                GRID_POINT_ROWS_KEY]
            all_grid_point_columns_2d_list += local_max_dict_by_time[i][
                GRID_POINT_COLUMNS_KEY]
            all_grid_point_latitudes_deg_2d_list += local_max_dict_by_time[i][
                GRID_POINT_LATITUDES_KEY]
            all_grid_point_longitudes_deg_2d_list += local_max_dict_by_time[i][
                GRID_POINT_LONGITUDES_KEY]

            all_polygon_objects_latlng = numpy.concatenate((
                all_polygon_objects_latlng,
                local_max_dict_by_time[i][POLYGON_OBJECTS_LATLNG_KEY]))
            all_polygon_objects_rowcol = numpy.concatenate((
                all_polygon_objects_rowcol,
                local_max_dict_by_time[i][POLYGON_OBJECTS_ROWCOL_KEY]))

    storm_object_dict = {
        tracking_utils.STORM_ID_COLUMN: all_storm_ids,
        tracking_utils.TIME_COLUMN: all_times_unix_sec,
        tracking_utils.SPC_DATE_COLUMN: all_spc_dates_unix_sec,
        tracking_utils.CENTROID_LAT_COLUMN: all_centroid_latitudes_deg,
        tracking_utils.CENTROID_LNG_COLUMN: all_centroid_longitudes_deg,
        CENTROID_X_COLUMN: all_centroid_x_metres,
        CENTROID_Y_COLUMN: all_centroid_y_metres
    }

    if include_polygons:
        storm_object_dict.update({
            tracking_utils.GRID_POINT_ROW_COLUMN: all_grid_point_rows_2d_list,
            tracking_utils.GRID_POINT_COLUMN_COLUMN:
                all_grid_point_columns_2d_list,
            tracking_utils.GRID_POINT_LAT_COLUMN:
                all_grid_point_latitudes_deg_2d_list,
            tracking_utils.GRID_POINT_LNG_COLUMN:
                all_grid_point_longitudes_deg_2d_list,
            tracking_utils.POLYGON_OBJECT_LATLNG_COLUMN:
                all_polygon_objects_latlng,
            tracking_utils.POLYGON_OBJECT_ROWCOL_COLUMN:
                all_polygon_objects_rowcol
        })

    return pandas.DataFrame.from_dict(storm_object_dict)


def _remove_short_tracks(storm_object_table, min_duration_seconds):
    """Removes short-lived storm tracks.

    :param storm_object_table: pandas DataFrame created by
        _local_maxima_to_storm_tracks.
    :param min_duration_seconds: Minimum storm duration.  Any track with
        duration < `min_duration_seconds` will be dropped.
    :return storm_object_table: Same as input, except maybe with fewer rows.
    """

    storm_id_by_object = numpy.array(
        storm_object_table[tracking_utils.STORM_ID_COLUMN].values)
    storm_id_by_cell, orig_to_unique_indices = numpy.unique(
        storm_id_by_object, return_inverse=True)

    num_storm_cells = len(storm_id_by_cell)
    object_indices_to_remove = numpy.array([], dtype=int)

    for i in range(num_storm_cells):
        these_object_indices = numpy.where(orig_to_unique_indices == i)[0]
        these_times_unix_sec = storm_object_table[
            tracking_utils.TIME_COLUMN].values[these_object_indices]

        this_duration_seconds = (
            numpy.max(these_times_unix_sec) - numpy.min(these_times_unix_sec))
        if this_duration_seconds >= min_duration_seconds:
            continue

        object_indices_to_remove = numpy.concatenate((
            object_indices_to_remove, these_object_indices))

    return storm_object_table.drop(
        storm_object_table.index[object_indices_to_remove], axis=0,
        inplace=False)


def _get_velocities_one_storm_track(
        centroid_latitudes_deg, centroid_longitudes_deg, unix_times_sec,
        num_points_back):
    """Computes velocity at each point along one storm track.

    Specifically, for each storm object, computes velocity using a backward
    difference (current minus previous position).

    N = number of points (storm objects) in track

    :param centroid_latitudes_deg: length-N numpy array with latitudes (deg N)
        of storm centroid.
    :param centroid_longitudes_deg: length-N numpy array with longitudes (deg E)
        of storm centroid.
    :param unix_times_sec: length-N numpy array of valid times.
    :param num_points_back: Velocity calculation for the [i]th point will
        consider the distance between the [i]th and [i - k]th points, where
        k = `num_points_back`.  Larger values lead to a more smoothly changing
        velocity over the track (less noisy estimates).
    :return: east_velocities_m_s01: length-N numpy array of eastward velocities
        (metres per second).
    :return: north_velocities_m_s01: length-N numpy array of northward
        velocities (metres per second).
    """

    num_storm_objects = len(unix_times_sec)
    east_displacements_metres = numpy.full(num_storm_objects, numpy.nan)
    north_displacements_metres = numpy.full(num_storm_objects, numpy.nan)
    time_diffs_seconds = numpy.full(num_storm_objects, -1, dtype=int)
    sort_indices = numpy.argsort(unix_times_sec)

    for i in range(0, num_storm_objects):
        this_num_points_back = min([i, num_points_back])
        if this_num_points_back == 0:
            continue

        this_end_latitude_deg = centroid_latitudes_deg[sort_indices[i]]
        this_end_longitude_deg = centroid_longitudes_deg[sort_indices[i]]
        this_start_latitude_deg = centroid_latitudes_deg[
            sort_indices[i - this_num_points_back]]
        this_start_longitude_deg = centroid_longitudes_deg[
            sort_indices[i - this_num_points_back]]

        this_end_point = (this_end_latitude_deg, this_end_longitude_deg)
        this_start_point = (this_end_latitude_deg, this_start_longitude_deg)
        east_displacements_metres[i] = vincenty(
            this_start_point, this_end_point).meters
        if this_start_longitude_deg > this_end_longitude_deg:
            east_displacements_metres[i] = -1 * east_displacements_metres[i]

        this_start_point = (this_start_latitude_deg, this_end_longitude_deg)
        north_displacements_metres[i] = vincenty(
            this_start_point, this_end_point).meters
        if this_start_latitude_deg > this_end_latitude_deg:
            north_displacements_metres[i] = -1 * north_displacements_metres[i]

        time_diffs_seconds[i] = (
            unix_times_sec[sort_indices[i]] -
            unix_times_sec[sort_indices[i - this_num_points_back]])

    return (east_displacements_metres / time_diffs_seconds,
            north_displacements_metres / time_diffs_seconds)


def _get_storm_velocities(storm_object_table, num_points_back):
    """Computes storm velocities.

    Specifically, for each storm object, computes velocity using a backward
    difference (current minus previous position).

    :param storm_object_table: pandas DataFrame created by
        `_local_maxima_to_storm_tracks`.
    :param num_points_back: See documentation for
        `_get_velocities_one_storm_track`.
    :return: storm_object_table: Same as input, but with two additional columns.
    storm_object_table.east_velocity_m_s01: Eastward velocity (metres per
        second).
    storm_object_table.north_velocity_m_s01: Northward velocity (metres per
        second).
    """

    all_storm_ids = numpy.array(
        storm_object_table[tracking_utils.STORM_ID_COLUMN].values)
    unique_storm_ids, storm_ids_object_to_unique = numpy.unique(
        all_storm_ids, return_inverse=True)

    num_storm_objects = len(storm_object_table.index)
    east_velocities_m_s01 = numpy.full(num_storm_objects, numpy.nan)
    north_velocities_m_s01 = numpy.full(num_storm_objects, numpy.nan)

    for i in range(len(unique_storm_ids)):
        these_object_indices = numpy.where(storm_ids_object_to_unique == i)[0]

        (east_velocities_m_s01[these_object_indices],
         north_velocities_m_s01[these_object_indices]) = (
             _get_velocities_one_storm_track(
                 centroid_latitudes_deg=
                 storm_object_table[tracking_utils.CENTROID_LAT_COLUMN].values[
                     these_object_indices],
                 centroid_longitudes_deg=
                 storm_object_table[tracking_utils.CENTROID_LNG_COLUMN].values[
                     these_object_indices],
                 unix_times_sec=storm_object_table[
                     tracking_utils.TIME_COLUMN].values[these_object_indices],
                 num_points_back=num_points_back))

    argument_dict = {
        tracking_utils.EAST_VELOCITY_COLUMN: east_velocities_m_s01,
        tracking_utils.NORTH_VELOCITY_COLUMN: north_velocities_m_s01}
    return storm_object_table.assign(**argument_dict)


def _get_grid_points_in_radius(
        x_grid_matrix_metres, y_grid_matrix_metres, x_query_metres,
        y_query_metres, radius_metres):
    """Finds grid points within some radius of query point.
    M = number of rows in grid
    N = number of columns in grid
    P = number of grid points within `radius_metres` of query point
    :param x_grid_matrix_metres: M-by-N numpy array with x-coordinates of grid
        points.
    :param y_grid_matrix_metres: M-by-N numpy array with y-coordinates of grid
        points.
    :param x_query_metres: x-coordinate of query point.
    :param y_query_metres: y-coordinate of query point.
    :param radius_metres: Critical radius from query point.
    :return: row_indices: length-P numpy array with row indices (integers) of
        grid points within `radius_metres` of query point.
    :return: column_indices: length-P numpy array with column indices (integers)
        of grid points within `radius_metres` of query point.
    """

    num_rows = x_grid_matrix_metres.shape[0]
    num_columns = x_grid_matrix_metres.shape[1]
    x_grid_vector_metres = numpy.reshape(
        x_grid_matrix_metres, num_rows * num_columns)
    y_grid_vector_metres = numpy.reshape(
        y_grid_matrix_metres, num_rows * num_columns)

    x_in_range_flags = numpy.logical_and(
        x_grid_vector_metres >= x_query_metres - radius_metres,
        x_grid_vector_metres <= x_query_metres + radius_metres)
    y_in_range_flags = numpy.logical_and(
        y_grid_vector_metres >= y_query_metres - radius_metres,
        y_grid_vector_metres <= y_query_metres + radius_metres)

    try_indices = numpy.where(
        numpy.logical_and(x_in_range_flags, y_in_range_flags))[0]
    distances_metres = numpy.sqrt(
        (x_grid_vector_metres[try_indices] - x_query_metres) ** 2 +
        (y_grid_vector_metres[try_indices] - y_query_metres) ** 2)
    linear_indices = try_indices[
        numpy.where(distances_metres <= radius_metres)[0]]

    return numpy.unravel_index(linear_indices, (num_rows, num_columns))


def _local_maxima_to_polygons(
        local_max_dict, echo_top_matrix_km_asl, min_echo_top_height_km_asl,
        radar_metadata_dict, min_distance_between_maxima_metres):
    """Converts local maxima at one time step from points to polygons.

    P = number of local maxima
    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)
    G_i = number of grid points in the [i]th polygon

    :param local_max_dict: Dictionary with at least the following keys.
    local_max_dict['latitudes_deg']: length-P numpy array with latitudes (deg N)
        of local maxima.
    local_max_dict['longitudes_deg']: length-P numpy array with longitudes
        (deg E) of local maxima.
    :param echo_top_matrix_km_asl: M-by-N numpy array of echo tops (km above sea
        level).
    :param min_echo_top_height_km_asl: Minimum echo-top height (km above sea
        level).  Smaller values are not considered local maxima.
    :param radar_metadata_dict: Dictionary with metadata for radar grid, created
        by `myrorss_and_mrms_io.read_metadata_from_raw_file`.
    :param min_distance_between_maxima_metres: Minimum distance between two
        local maxima.
    :return: local_max_dict: Same as input, but with extra keys listed below.
    local_max_dict['list_of_grid_point_rows']: length-N list, where the [i]th
        element is a numpy array (length G_i) with row indices of grid points in
        polygon.
    local_max_dict['list_of_grid_point_columns']: Same but for columns.
    local_max_dict['list_of_grid_point_latitudes_deg']: Same but for latitudes
        (deg N).
    local_max_dict['list_of_grid_point_longitudes_deg']: Same but for longitudes
        (deg E).
    local_max_dict['polygon_objects_rowcol']: length-N list, where each element
        is an instance of `shapely.geometry.Polygon` with vertices in row-column
        coordinates.
    local_max_dict['polygon_objects_latlng']: Same but for lat-long coordinates.
    """

    # TODO(thunderhoser): I may want to let this influence centroids... ?

    min_grid_point_latitude_deg = (
        radar_metadata_dict[radar_utils.NW_GRID_POINT_LAT_COLUMN] -
        (radar_metadata_dict[radar_utils.LAT_SPACING_COLUMN] *
         (radar_metadata_dict[radar_utils.NUM_LAT_COLUMN] - 1)))

    grid_point_latitudes_deg, grid_point_longitudes_deg = (
        grids.get_latlng_grid_points(
            min_latitude_deg=min_grid_point_latitude_deg,
            min_longitude_deg=
            radar_metadata_dict[radar_utils.NW_GRID_POINT_LNG_COLUMN],
            lat_spacing_deg=radar_metadata_dict[radar_utils.LAT_SPACING_COLUMN],
            lng_spacing_deg=radar_metadata_dict[radar_utils.LNG_SPACING_COLUMN],
            num_rows=radar_metadata_dict[radar_utils.NUM_LAT_COLUMN],
            num_columns=radar_metadata_dict[radar_utils.NUM_LNG_COLUMN]))
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
        this_echo_top_submatrix_km_asl, this_row_offset, this_column_offset = (
            grids.extract_latlng_subgrid(
                data_matrix=echo_top_matrix_km_asl,
                grid_point_latitudes_deg=grid_point_latitudes_deg,
                grid_point_longitudes_deg=grid_point_longitudes_deg,
                center_latitude_deg=local_max_dict[LATITUDES_KEY][i],
                center_longitude_deg=local_max_dict[LONGITUDES_KEY][i],
                max_distance_from_center_metres=
                min_distance_between_maxima_metres))

        this_echo_top_submatrix_km_asl[
            numpy.isnan(this_echo_top_submatrix_km_asl)] = 0.

        (local_max_dict[GRID_POINT_ROWS_KEY][i],
         local_max_dict[GRID_POINT_COLUMNS_KEY][i]) = numpy.where(
             this_echo_top_submatrix_km_asl >= min_echo_top_height_km_asl)

        if not len(local_max_dict[GRID_POINT_ROWS_KEY][i]):
            this_row = numpy.floor(
                float(this_echo_top_submatrix_km_asl.shape[0]) / 2)
            this_column = numpy.floor(
                float(this_echo_top_submatrix_km_asl.shape[1]) / 2)

            local_max_dict[GRID_POINT_ROWS_KEY][i] = numpy.array(
                [this_row], dtype=int)
            local_max_dict[GRID_POINT_COLUMNS_KEY][i] = numpy.array(
                [this_column], dtype=int)

        local_max_dict[GRID_POINT_ROWS_KEY][i] = (
            local_max_dict[GRID_POINT_ROWS_KEY][i] + this_row_offset)
        local_max_dict[GRID_POINT_COLUMNS_KEY][i] = (
            local_max_dict[GRID_POINT_COLUMNS_KEY][i] + this_column_offset)

        these_vertex_rows, these_vertex_columns = (
            polygons.grid_points_in_poly_to_vertices(
                local_max_dict[GRID_POINT_ROWS_KEY][i],
                local_max_dict[GRID_POINT_COLUMNS_KEY][i]))

        (local_max_dict[GRID_POINT_LATITUDES_KEY][i],
         local_max_dict[GRID_POINT_LONGITUDES_KEY][i]) = (
             radar_utils.rowcol_to_latlng(
                 local_max_dict[GRID_POINT_ROWS_KEY][i],
                 local_max_dict[GRID_POINT_COLUMNS_KEY][i],
                 nw_grid_point_lat_deg=
                 radar_metadata_dict[radar_utils.NW_GRID_POINT_LAT_COLUMN],
                 nw_grid_point_lng_deg=
                 radar_metadata_dict[radar_utils.NW_GRID_POINT_LNG_COLUMN],
                 lat_spacing_deg=
                 radar_metadata_dict[radar_utils.LAT_SPACING_COLUMN],
                 lng_spacing_deg=
                 radar_metadata_dict[radar_utils.LNG_SPACING_COLUMN]))

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
                radar_metadata_dict[radar_utils.LNG_SPACING_COLUMN]))

        local_max_dict[POLYGON_OBJECTS_ROWCOL_KEY][i] = (
            polygons.vertex_arrays_to_polygon_object(
                these_vertex_columns, these_vertex_rows))
        local_max_dict[POLYGON_OBJECTS_LATLNG_KEY][i] = (
            polygons.vertex_arrays_to_polygon_object(
                these_vertex_longitudes_deg, these_vertex_latitudes_deg))

    return local_max_dict


def _join_tracks_between_periods(
        early_storm_object_table, late_storm_object_table, projection_object,
        max_link_time_seconds, max_link_distance_m_s01):
    """Joins storm tracks between two time periods.

    :param early_storm_object_table: pandas DataFrame for early period.  Each
        row is one storm object.  Must contain the following columns.
    early_storm_object_table.storm_id: String ID for storm.
    early_storm_object_table.unix_time_sec: Valid time.
    early_storm_object_table.centroid_lat_deg: Latitude (deg N) of storm
        centroid.
    early_storm_object_table.centroid_lng_deg: Longitude (deg E) of storm
        centroid.

    :param late_storm_object_table: Same as above, but for late period.
    :param projection_object: Instance of `pyproj.Proj` (will be used to convert
        lat-long coordinates to x-y).
    :param max_link_time_seconds: See documentation for
        `_link_local_maxima_in_time`.
    :param max_link_distance_m_s01: See doc for `_link_local_maxima_in_time`.
    :return: late_storm_object_table: Same as input, except that some storm IDs
        may be changed.
    """

    if (len(early_storm_object_table.index) == 0 or
            len(late_storm_object_table.index) == 0):
        return late_storm_object_table

    last_early_time_unix_sec = numpy.max(
        early_storm_object_table[tracking_utils.TIME_COLUMN].values)
    previous_indices = numpy.where(
        early_storm_object_table[tracking_utils.TIME_COLUMN] ==
        last_early_time_unix_sec)[0]
    previous_latitudes_deg = early_storm_object_table[
        tracking_utils.CENTROID_LAT_COLUMN].values[previous_indices]
    previous_longitudes_deg = early_storm_object_table[
        tracking_utils.CENTROID_LNG_COLUMN].values[previous_indices]
    previous_x_coords_metres, previous_y_coords_metres = (
        projections.project_latlng_to_xy(
            previous_latitudes_deg, previous_longitudes_deg,
            projection_object=projection_object, false_easting_metres=0.,
            false_northing_metres=0.))

    previous_local_max_dict = {
        X_COORDS_KEY: previous_x_coords_metres,
        Y_COORDS_KEY: previous_y_coords_metres,
        VALID_TIME_KEY: last_early_time_unix_sec}

    first_late_time_unix_sec = numpy.min(
        late_storm_object_table[tracking_utils.TIME_COLUMN].values)
    current_indices = numpy.where(
        late_storm_object_table[tracking_utils.TIME_COLUMN] ==
        first_late_time_unix_sec)[0]
    current_latitudes_deg = late_storm_object_table[
        tracking_utils.CENTROID_LAT_COLUMN].values[current_indices]
    current_longitudes_deg = late_storm_object_table[
        tracking_utils.CENTROID_LNG_COLUMN].values[current_indices]
    current_x_coords_metres, current_y_coords_metres = (
        projections.project_latlng_to_xy(
            current_latitudes_deg, current_longitudes_deg,
            projection_object=projection_object, false_easting_metres=0.,
            false_northing_metres=0.))

    current_local_max_dict = {
        X_COORDS_KEY: current_x_coords_metres,
        Y_COORDS_KEY: current_y_coords_metres,
        VALID_TIME_KEY: first_late_time_unix_sec}

    current_to_previous_indices = _link_local_maxima_in_time(
        current_local_max_dict=current_local_max_dict,
        previous_local_max_dict=previous_local_max_dict,
        max_link_time_seconds=max_link_time_seconds,
        max_link_distance_m_s01=max_link_distance_m_s01)

    previous_storm_ids = early_storm_object_table[
        tracking_utils.STORM_ID_COLUMN].values[previous_indices]
    orig_current_storm_ids = late_storm_object_table[
        tracking_utils.STORM_ID_COLUMN].values[current_indices]
    num_current_storms = len(orig_current_storm_ids)

    for i in range(num_current_storms):
        if current_to_previous_indices[i] == -1:
            continue

        this_new_current_storm_id = previous_storm_ids[
            current_to_previous_indices[i]]
        late_storm_object_table.replace(
            to_replace=orig_current_storm_ids[i],
            value=this_new_current_storm_id, inplace=True)

    return late_storm_object_table


def _storm_objects_to_tracks(
        storm_object_table, storm_track_table=None, recompute_for_id=None):
    """Converts table of storm objects to table of storm tracks.

    If the input arg `storm_track_table` is None, `storm_track_table` will be
    computed from scratch.  Otherwise, only one row of `storm_track_table` will
    be recomputed.

    :param storm_object_table: pandas DataFrame with the following columns.
    storm_object_table.storm_id: String ID for storm cell.
    storm_object_table.unix_time_sec: Valid time.
    storm_object_table.centroid_lat_deg: Latitude (deg N) of storm centroid.
    storm_object_table.centroid_lng_deg: Longitude (deg E) of storm centroid.

    :param storm_track_table: pandas DataFrame with the following columns.
    storm_track_table.storm_id: String ID for storm cell.
    storm_track_table.start_time_unix_sec: Start time.
    storm_track_table.end_time_unix_sec: End time.
    storm_track_table.start_latitude_deg: Start latitude (deg N).
    storm_track_table.end_latitude_deg: End latitude (deg N).
    storm_track_table.start_longitude_deg: Start longitude (deg E).
    storm_track_table.end_longitude_deg: End longitude (deg E).

    :param recompute_for_id: [used only if storm_track_table is not None]
        Track data will be recomputed only for this storm ID.

    :return: storm_track_table: See input documentation.
    """

    if storm_track_table is None:
        storm_id_by_track = numpy.unique(
            storm_object_table[tracking_utils.STORM_ID_COLUMN].values).tolist()

        num_tracks = len(storm_id_by_track)
        track_start_times_unix_sec = numpy.full(num_tracks, -1, dtype=int)
        track_end_times_unix_sec = numpy.full(num_tracks, -1, dtype=int)
        track_start_latitudes_deg = numpy.full(num_tracks, numpy.nan)
        track_start_longitudes_deg = numpy.full(num_tracks, numpy.nan)
        track_end_latitudes_deg = numpy.full(num_tracks, numpy.nan)
        track_end_longitudes_deg = numpy.full(num_tracks, numpy.nan)

        storm_track_dict = {
            tracking_utils.STORM_ID_COLUMN: storm_id_by_track,
            START_TIME_COLUMN: track_start_times_unix_sec,
            END_TIME_COLUMN: track_end_times_unix_sec,
            START_LATITUDE_COLUMN: track_start_latitudes_deg,
            END_LATITUDE_COLUMN: track_start_longitudes_deg,
            START_LONGITUDE_COLUMN: track_end_latitudes_deg,
            END_LONGITUDE_COLUMN: track_end_longitudes_deg
        }

        storm_track_table = pandas.DataFrame.from_dict(storm_track_dict)
        recompute_for_ids = copy.deepcopy(storm_id_by_track)

    else:
        recompute_for_ids = [recompute_for_id]

    for this_id in recompute_for_ids:
        these_object_indices = numpy.where(
            storm_object_table[tracking_utils.STORM_ID_COLUMN].values ==
            this_id)[0]

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


def _get_extrapolation_error(storm_track_table, early_track_id, late_track_id):
    """Finds error incurred by extrapolating early track to late track.

    Specifically, finds error incurred by extrapolating early track from t_1_end
    to t_2_start, where t_1_end = end time of early track and t_2_start =
    start time of late track.

    :param storm_track_table: pandas DataFrame created by
        `_storm_objects_to_tracks`.
    :param early_track_id: Storm ID for early track (string).
    :param late_track_id: Storm ID for late track (string).
    :return: extrap_error_metres: Extrapolation error.
    """

    early_track_index = numpy.where(
        storm_track_table[tracking_utils.STORM_ID_COLUMN].values ==
        early_track_id)[0][0]
    late_track_index = numpy.where(
        storm_track_table[tracking_utils.STORM_ID_COLUMN].values ==
        late_track_id)[0][0]

    early_time_diff_seconds = (
        storm_track_table[END_TIME_COLUMN].values[early_track_index] -
        storm_track_table[START_TIME_COLUMN].values[early_track_index])

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
        storm_track_table[END_TIME_COLUMN].values[early_track_index])
    extrap_latitude_deg = (
        storm_track_table[END_LATITUDE_COLUMN].values[early_track_index] +
        early_lat_speed_deg_s01 * extrap_time_seconds)
    extrap_longitude_deg = (
        storm_track_table[END_LONGITUDE_COLUMN].values[early_track_index] +
        early_lng_speed_deg_s01 * extrap_time_seconds)

    extrap_point = (extrap_latitude_deg, extrap_longitude_deg)
    late_track_start_point = (
        storm_track_table[START_LATITUDE_COLUMN].values[late_track_index],
        storm_track_table[START_LONGITUDE_COLUMN].values[late_track_index])
    return vincenty(extrap_point, late_track_start_point).meters


def _find_nearby_tracks(
        storm_track_table, late_track_id, max_time_diff_seconds,
        max_extrap_error_m_s01):
    """Finds tracks are both spatially and temporally near the late track.

    :param storm_track_table: pandas DataFrame created by
        `_storm_objects_to_tracks`.
    :param late_track_id: Storm ID for late track (string).
    :param max_time_diff_seconds: Max time difference between tracks (end of
        early track and beginning of late track).
    :param max_extrap_error_m_s01: Max error (metres per second) incurred by
        extrapolating early track from t_1_end to t_2_start, where t_1_end = end
        time of early track and t_2_start = start time of late track.
    :return: nearby_track_indices: 1-D numpy array with indices of nearby
        tracks, sorted primarily by time difference (seconds) and secondarily by
        extrapolation error (metres).
    """

    late_track_index = numpy.where(
        storm_track_table[tracking_utils.STORM_ID_COLUMN].values ==
        late_track_id)[0][0]

    time_diffs_seconds = (
        storm_track_table[START_TIME_COLUMN].values[late_track_index] -
        storm_track_table[END_TIME_COLUMN].values)
    time_diffs_seconds[time_diffs_seconds <= 0] = max_time_diff_seconds + 1

    num_tracks = len(storm_track_table.index)
    extrap_errors_metres = numpy.full(num_tracks, numpy.inf)

    for j in range(num_tracks):
        if time_diffs_seconds[j] > max_time_diff_seconds:
            continue

        this_latitude_diff_deg = numpy.absolute(
            storm_track_table[START_LATITUDE_COLUMN].values[late_track_index] -
            storm_track_table[END_LATITUDE_COLUMN].values[j])
        this_latitude_diff_m_s01 = this_latitude_diff_deg * (
            DEGREES_LAT_TO_METRES / time_diffs_seconds[j])
        if this_latitude_diff_m_s01 > max_extrap_error_m_s01:
            continue

        extrap_errors_metres[j] = _get_extrapolation_error(
            storm_track_table=storm_track_table, late_track_id=late_track_id,
            early_track_id=
            storm_track_table[tracking_utils.STORM_ID_COLUMN].values[j])

    extrap_errors_m_s01 = extrap_errors_metres / time_diffs_seconds
    nearby_track_indices = numpy.where(numpy.logical_and(
        time_diffs_seconds <= max_time_diff_seconds,
        extrap_errors_m_s01 <= max_extrap_error_m_s01))[0]
    if not len(nearby_track_indices):
        return None

    sort_indices = numpy.lexsort((
        extrap_errors_metres[nearby_track_indices],
        time_diffs_seconds[nearby_track_indices]))
    return nearby_track_indices[sort_indices]


def _write_storm_objects(
        storm_object_table, top_output_dir_name, output_times_unix_sec):
    """Writes storm objects to files (one Pickle file per time step).

    :param storm_object_table: See doc for
        `storm_tracking_io.write_processed_file`.
    :param top_output_dir_name: Name of top-level output directory.  Files
        will be written by `storm_tracking_io.write_processed_file`, to
        locations therein determined by `storm_tracking_io.find_processed_file`.
    :param output_times_unix_sec: 1-D numpy array of output times.
    """

    for this_time_unix_sec in output_times_unix_sec:
        this_output_file_name = tracking_io.find_processed_file(
            top_processed_dir_name=top_output_dir_name,
            unix_time_sec=this_time_unix_sec,
            spc_date_string=time_conversion.time_to_spc_date_string(
                this_time_unix_sec),
            tracking_scale_metres2=DUMMY_TRACKING_SCALE_METRES2,
            data_source=tracking_utils.SEGMOTION_SOURCE_ID,
            raise_error_if_missing=False)

        print 'Writing data to file: "{0:s}"...'.format(this_output_file_name)
        tracking_io.write_processed_file(
            storm_object_table=storm_object_table.loc[
                storm_object_table[tracking_utils.TIME_COLUMN] ==
                this_time_unix_sec
                ],
            pickle_file_name=this_output_file_name
        )


def reanalyze_tracks(
        storm_object_table, max_join_time_sec=DEFAULT_MAX_REANAL_JOIN_TIME_SEC,
        max_extrap_error_m_s01=DEFAULT_MAX_REANAL_EXTRAP_ERROR_M_S01):
    """Joins pairs of tracks that are spatiotemporally nearby.

    This method is similar to the "reanalysis" discussed in Haberlie and Ashley
    (2018).

    :param storm_object_table: pandas DataFrame with the following columns.
    storm_object_table.storm_id: String ID for storm cell.
    storm_object_table.unix_time_sec: Valid time.
    storm_object_table.centroid_lat_deg: Latitude (deg N) of storm centroid.
    storm_object_table.centroid_lng_deg: Longitude (deg E) of storm centroid.

    :param max_join_time_sec: Max time gap between two tracks (i.e., between the
        end of the early track and beginning of the late track).  If time
        elapsed > `max_join_time_sec`, the tracks cannot be joined.
    :param max_extrap_error_m_s01: Max error (metres per second) incurred by
        extrapolating early track from t_1_end to t_2_start, where t_1_end = end
        time of early track and t_2_start = start time of late track.
    :return: storm_object_table: Same as input, except that some storm IDs may
        have changed.
    """

    # Convert table of storm objects to table of storm tracks.
    storm_track_table = _storm_objects_to_tracks(storm_object_table)

    # Initialize variables.
    track_removed_ids = []
    num_storm_tracks = len(storm_track_table.index)

    for i in range(num_storm_tracks):
        print 'Reanalyzing {0:d}th of {1:d} tracks...'.format(
            i + 1, num_storm_tracks)

        # If this track has been removed (joined with another), skip it.
        this_storm_id = storm_track_table[
            tracking_utils.STORM_ID_COLUMN].values[i]
        if this_storm_id in track_removed_ids:
            continue

        while True:

            # Find other tracks that end shortly before the [i]th track starts.
            these_nearby_indices = _find_nearby_tracks(
                storm_track_table=storm_track_table,
                late_track_id=this_storm_id,
                max_time_diff_seconds=max_join_time_sec,
                max_extrap_error_m_s01=max_extrap_error_m_s01)
            if these_nearby_indices is None:
                break

            # Assign each storm object from nearby track to [i]th track.
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
            track_removed_ids.append(this_nearby_storm_id)
            storm_track_table[END_TIME_COLUMN].values[
                this_nearby_index] = DUMMY_TIME_UNIX_SEC
            storm_track_table = _storm_objects_to_tracks(
                storm_object_table=storm_object_table,
                storm_track_table=storm_track_table,
                recompute_for_id=this_storm_id)

    return storm_object_table


def run_tracking(
        top_radar_dir_name, top_output_dir_name,
        first_spc_date_string, last_spc_date_string,
        first_time_unix_sec=None, last_time_unix_sec=None,
        echo_top_field_name=radar_utils.ECHO_TOP_40DBZ_NAME,
        radar_source_name=radar_utils.MYRORSS_SOURCE_ID,
        top_echo_classifn_dir_name=None,
        min_echo_top_height_km_asl=DEFAULT_MIN_ECHO_TOP_HEIGHT_KM_ASL,
        e_fold_radius_for_smoothing_deg_lat=
        DEFAULT_E_FOLD_RADIUS_FOR_SMOOTHING_DEG_LAT,
        half_width_for_max_filter_deg_lat=
        DEFAULT_HALF_WIDTH_FOR_MAX_FILTER_DEG_LAT,
        min_distance_between_maxima_metres=
        DEFAULT_MIN_DISTANCE_BETWEEN_MAXIMA_METRES,
        max_link_time_seconds=DEFAULT_MAX_LINK_TIME_SECONDS,
        max_link_distance_m_s01=DEFAULT_MAX_LINK_DISTANCE_M_S01,
        min_track_duration_seconds=DEFAULT_MIN_TRACK_DURATION_SHORT_SECONDS,
        num_points_back_for_velocity=DEFAULT_NUM_POINTS_BACK_FOR_VELOCITY):
    """This is effectively the main method for echo-top-tracking.

    :param top_radar_dir_name: See doc for `_find_input_radar_files`.
    :param top_output_dir_name: See doc for `write_storm_objects`.
    :param first_spc_date_string: See doc for `_find_input_radar_files`.
    :param last_spc_date_string: Same.
    :param first_time_unix_sec: Same.
    :param last_time_unix_sec: Same.
    :param echo_top_field_name: Same.
    :param radar_source_name: Same.
    :param top_echo_classifn_dir_name: Name of top-level directory with echo
        classifications.  If None, echo classifications will not be used.  If
        specified, files therein will be found by
        `echo_classification.find_classification_file` and read by
        `echo_classification.read_classifications` and tracking will be run only
        on convective pixels.
    :param min_echo_top_height_km_asl: Minimum echo-top height (km above sea
        level).  Only local maxima >= `min_echo_top_height_km_asl` will be
        tracked.
    :param e_fold_radius_for_smoothing_deg_lat: e-folding radius for
        `_gaussian_smooth_radar_field`.  Units are degrees of latitude.  This
        will be applied separately to the radar field at each time step, before
        finding local maxima.
    :param half_width_for_max_filter_deg_lat: Half-width for max filter used in
        `_find_local_maxima`.  Units are degrees of latitude.
    :param min_distance_between_maxima_metres: See doc for
        `_remove_redundant_local_maxima`.
    :param max_link_time_seconds: See doc for `_link_local_maxima_in_time`.
    :param max_link_distance_m_s01: See doc for `_link_local_maxima_in_time`.
    :param min_track_duration_seconds: Minimum track duration.  Shorter-lived
        storms will be removed.
    :param num_points_back_for_velocity: See doc for
        `_get_velocities_one_storm_track`.
    """

    error_checking.assert_is_greater(min_echo_top_height_km_asl, 0.)

    radar_file_names, valid_times_unix_sec = _find_input_radar_files(
        top_radar_dir_name=top_radar_dir_name,
        echo_top_field_name=echo_top_field_name,
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

    if top_echo_classifn_dir_name is None:
        echo_classifn_file_names = None
    else:
        echo_classifn_file_names = [''] * num_times

        for i in range(num_times):
            echo_classifn_file_names[i] = (
                echo_classifn.find_classification_file(
                    top_directory_name=top_echo_classifn_dir_name,
                    valid_time_unix_sec=valid_times_unix_sec[i])
            )

    projection_object = projections.init_azimuthal_equidistant_projection(
        central_latitude_deg=CENTRAL_PROJ_LATITUDE_DEG,
        central_longitude_deg=CENTRAL_PROJ_LONGITUDE_DEG)

    local_max_dict_by_time = [{}] * num_times

    for i in range(num_times):
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

        this_orig_echo_top_matrix_km_asl = radar_s2f.sparse_to_full_grid(
            sparse_grid_table=this_sparse_grid_table,
            metadata_dict=this_metadata_dict,
            ignore_if_below=min_echo_top_height_km_asl
        )[0]
        this_new_echo_top_matrix_km_asl = this_orig_echo_top_matrix_km_asl + 0.

        if echo_classifn_file_names is not None:
            print 'Reading data from: "{0:s}"...'.format(
                echo_classifn_file_names[i])
            this_convective_flag_matrix = echo_classifn.read_classifications(
                echo_classifn_file_names[i]
            )[0]
            this_convective_flag_matrix = numpy.flip(
                this_convective_flag_matrix, axis=0)

            this_new_echo_top_matrix_km_asl[
                this_convective_flag_matrix == False] = 0.

        print 'Finding local maxima in "{0:s}" at {1:s}...'.format(
            echo_top_field_name, valid_time_strings[i])

        this_latitude_spacing_deg = this_metadata_dict[
            radar_utils.LAT_SPACING_COLUMN]

        this_new_echo_top_matrix_km_asl = _gaussian_smooth_radar_field(
            radar_matrix=this_new_echo_top_matrix_km_asl,
            e_folding_radius_pixels=
            e_fold_radius_for_smoothing_deg_lat / this_latitude_spacing_deg
        )

        this_half_width_in_pixels = int(numpy.round(
            half_width_for_max_filter_deg_lat / this_latitude_spacing_deg))

        local_max_dict_by_time[i] = _find_local_maxima(
            radar_matrix=this_new_echo_top_matrix_km_asl,
            radar_metadata_dict=this_metadata_dict,
            neigh_half_width_in_pixels=this_half_width_in_pixels)

        local_max_dict_by_time[i] = _remove_redundant_local_maxima(
            local_max_dict_latlng=local_max_dict_by_time[i],
            projection_object=projection_object,
            min_distance_between_maxima_metres=
            min_distance_between_maxima_metres
        )

        local_max_dict_by_time[i].update(
            {VALID_TIME_KEY: valid_times_unix_sec[i]})

        local_max_dict_by_time[i] = _local_maxima_to_polygons(
            local_max_dict=local_max_dict_by_time[i],
            echo_top_matrix_km_asl=this_orig_echo_top_matrix_km_asl,
            min_echo_top_height_km_asl=min_echo_top_height_km_asl,
            radar_metadata_dict=this_metadata_dict,
            min_distance_between_maxima_metres=
            min_distance_between_maxima_metres)

        if i == 0:
            these_current_to_prev_indices = _link_local_maxima_in_time(
                current_local_max_dict=local_max_dict_by_time[i],
                previous_local_max_dict=None,
                max_link_time_seconds=max_link_time_seconds,
                max_link_distance_m_s01=max_link_distance_m_s01)
        else:
            print (
                'Linking local maxima at {0:s} with those at {1:s}...\n'
            ).format(valid_time_strings[i], valid_time_strings[i - 1])

            these_current_to_prev_indices = _link_local_maxima_in_time(
                current_local_max_dict=local_max_dict_by_time[i],
                previous_local_max_dict=local_max_dict_by_time[i - 1],
                max_link_time_seconds=max_link_time_seconds,
                max_link_distance_m_s01=max_link_distance_m_s01)

        local_max_dict_by_time[i].update(
            {CURRENT_TO_PREV_INDICES_KEY: these_current_to_prev_indices})

    print SEPARATOR_STRING
    print 'Converting time series of "{0:s}" maxima to storm tracks...'.format(
        echo_top_field_name)
    storm_object_table = _local_maxima_to_storm_tracks(local_max_dict_by_time)

    print 'Removing tracks that last < {0:d} seconds...'.format(
        int(min_track_duration_seconds))
    storm_object_table = _remove_short_tracks(
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
    storm_object_table = _get_storm_velocities(
        storm_object_table=storm_object_table,
        num_points_back=num_points_back_for_velocity)

    print SEPARATOR_STRING
    _write_storm_objects(
        storm_object_table=storm_object_table,
        top_output_dir_name=top_output_dir_name,
        output_times_unix_sec=valid_times_unix_sec)
