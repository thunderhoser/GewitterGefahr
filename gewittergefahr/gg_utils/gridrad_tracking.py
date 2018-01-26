"""Implements the GridRad storm-tracking algorithm.

This algorithm is discussed in Section 3c of Homeyer et al. (2017).  The
algorithm should be applied to echo-top fields only.  The main advantage of this
algorithm (in my experience) over segmotion (Lakshmanan and Smith 2010) is that
it provides more intuitive and longer storm tracks.  The main disadvantage of
the GridRad algorithm (in my experience) is that it provides only storm centers,
not objects.  In other words, the GridRad algorithm does not provide the
bounding polygons.

--- REFERENCES ---

Homeyer, C.R., and J.D. McAuliffe, and K.M. Bedka, 2017: "On the development of
    above-anvil cirrus plumes in extratropical convection". Journal of the
    Atmospheric Sciences, 74 (5), 1617-1633.

Lakshmanan, V., and T. Smith, 2010: "Evaluating a storm tracking algorithm".
    26th Conference on Interactive Information Processing Systems, Atlanta, GA,
    American Meteorological Society.
"""

import numpy
import pandas
from geopy.distance import vincenty
from gewittergefahr.gg_io import myrorss_and_mrms_io
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import radar_sparse_to_full as radar_s2f
from gewittergefahr.gg_utils import dilation
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import grid_smoothing_2d
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import best_tracks
from gewittergefahr.gg_utils import error_checking

TOLERANCE = 1e-6
TIME_FORMAT = '%Y-%m-%d-%H%M%S'

DAYS_TO_SECONDS = 86400
DEGREES_LAT_TO_METRES = 60 * 1852

CENTRAL_PROJ_LATITUDE_DEG = 35.
CENTRAL_PROJ_LONGITUDE_DEG = 265.

VALID_RADAR_FIELDS = [
    radar_utils.ECHO_TOP_18DBZ_NAME, radar_utils.ECHO_TOP_40DBZ_NAME,
    radar_utils.ECHO_TOP_50DBZ_NAME]
VALID_RADAR_DATA_SOURCES = [
    radar_utils.MYRORSS_SOURCE_ID, radar_utils.MRMS_SOURCE_ID]

DEFAULT_MIN_ECHO_TOP_HEIGHT_KM_ASL = 4.
DEFAULT_E_FOLDING_RADIUS_FOR_SMOOTHING_PIXELS = 1.2
DEFAULT_HALF_WIDTH_FOR_MAX_FILTER_PIXELS = 3
DEFAULT_MIN_DISTANCE_BETWEEN_MAXIMA_METRES = 0.1 * DEGREES_LAT_TO_METRES
DEFAULT_MAX_LINK_TIME_SECONDS = 300
DEFAULT_MAX_LINK_DISTANCE_M_S01 = (
    0.125 * DEGREES_LAT_TO_METRES / DEFAULT_MAX_LINK_TIME_SECONDS)

DEFAULT_MIN_TRACK_DURATION_SECONDS = 900
DEFAULT_NUM_POINTS_BACK_FOR_VELOCITY = 3
DEFAULT_STORM_OBJECT_AREA_METRES2 = numpy.pi * 1e8  # Radius of 10 km.

RADAR_FILE_NAMES_KEY = 'input_radar_file_names'
TRACKING_FILE_NAMES_KEY = 'output_tracking_file_names'
RADAR_METADATA_DICTS_KEY = 'list_of_radar_metadata_dicts'
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


def _check_radar_field(radar_field_name):
    """Ensures that radar field is valid for GridRad tracking.

    :param radar_field_name: Field name (string).
    :raises: ValueError: if `radar_field_name not in VALID_RADAR_FIELDS`.
    """

    error_checking.assert_is_string(radar_field_name)
    if radar_field_name not in VALID_RADAR_FIELDS:
        error_string = (
            '\n\n{0:s}\n\nValid radar fields (listed above) do not include '
            '"{1:s}".').format(VALID_RADAR_FIELDS, radar_field_name)
        raise ValueError(error_string)


def _check_radar_data_source(radar_data_source):
    """Ensures that data source is valid for GridRad tracking.

    :param radar_data_source: Data source (string).
    :raises: ValueError: if `radar_data_source not in VALID_RADAR_DATA_SOURCES`.
    """

    error_checking.assert_is_string(radar_data_source)
    if radar_data_source not in VALID_RADAR_DATA_SOURCES:
        error_string = (
            '\n\n{0:s}\n\nValid data sources (listed above) do not include '
            '"{1:s}".').format(VALID_RADAR_DATA_SOURCES, radar_data_source)
        raise ValueError(error_string)


def _gaussian_smooth_radar_field(
        radar_matrix,
        e_folding_radius_pixels=DEFAULT_E_FOLDING_RADIUS_FOR_SMOOTHING_PIXELS,
        cutoff_radius_pixels=None):
    """Applies Gaussian smoother to radar field.  NaN's are treated as zero.

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)

    :param radar_matrix: M-by-N numpy array with values of radar field.
    :param e_folding_radius_pixels: e-folding radius for Gaussian smoother.
    :param cutoff_radius_pixels: Cutoff radius for Gaussian smoother.  Default
        is 3 * e-folding radius.
    :return: radar_matrix: Smoothed version of input.
    """

    e_folding_radius_pixels = float(e_folding_radius_pixels)
    if cutoff_radius_pixels is None:
        cutoff_radius_pixels = 3 * e_folding_radius_pixels

    return grid_smoothing_2d.apply_gaussian(
        input_matrix=radar_matrix, grid_spacing_x=1., grid_spacing_y=1.,
        e_folding_radius=e_folding_radius_pixels,
        cutoff_radius=cutoff_radius_pixels)


def _find_local_maxima(
        radar_matrix, radar_metadata_dict,
        neigh_half_width_in_pixels=DEFAULT_HALF_WIDTH_FOR_MAX_FILTER_PIXELS):
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
        echo_top_field_name, data_source, top_radar_dir_name,
        top_tracking_dir_name, tracking_scale_metres2, start_spc_date_string,
        end_spc_date_string, start_time_unix_sec=None, end_time_unix_sec=None):
    """Finds input radar files for the given time period and radar field.

    Also creates paths for output tracking files.

    N = number of radar files found

    :param echo_top_field_name: Name of radar field to use for tracking.  Must
        be an echo-top field (to my knowledge, the GridRad method hasn't been
        used on other fields, such as reflectivity).
    :param data_source: Data source (must be either "myrorss" or "mrms").
    :param top_radar_dir_name: [input] Name of top-level directory with radar
        data from the given source.
    :param top_tracking_dir_name: [output] Name of top-level directory for
        storm tracks.
    :param tracking_scale_metres2: Tracking scale (storm-object area).  This
        determines names of output tracking files.
    :param start_spc_date_string: First SPC date in period (format "yyyymmdd").
    :param end_spc_date_string: Last SPC date in period (format "yyyymmdd").
    :param start_time_unix_sec: Start of time period.  Default is 1200 UTC at
        beginning of the given SPC date (e.g., if SPC date is "20180124", this
        will be 1200 UTC 24 Jan 2018).
    :param end_time_unix_sec: End of time period.  Default is 1200 UTC at end of
        the given SPC date (e.g., if SPC date is "20180124", this will be 1200
        UTC 25 Jan 2018).
    :return: file_dictionary: Dictionary with the following keys.
    file_dictionary['input_radar_file_names']: length-N list of paths to radar
        files.
    file_dictionary['output_tracking_file_names']: length-N list of paths to
        tracking files.
    file_dictionary['unix_times_sec']: length-N numpy array of time steps.

    :raises: ValueError: if `start_time_unix_sec` is not in
        `start_spc_date_string` or `end_time_unix_sec` is not in
        `end_spc_date_string`.
    """

    # Error-checking.
    _check_radar_field(echo_top_field_name)
    _check_radar_data_source(data_source)
    _ = time_conversion.spc_date_string_to_unix_sec(start_spc_date_string)
    _ = time_conversion.spc_date_string_to_unix_sec(end_spc_date_string)

    if start_time_unix_sec is None:
        start_time_unix_sec = (
            time_conversion.MIN_SECONDS_INTO_SPC_DATE +
            time_conversion.string_to_unix_sec(
                start_spc_date_string, time_conversion.SPC_DATE_FORMAT))

    if end_time_unix_sec is None:
        end_time_unix_sec = (
            time_conversion.MAX_SECONDS_INTO_SPC_DATE +
            time_conversion.string_to_unix_sec(
                end_spc_date_string, time_conversion.SPC_DATE_FORMAT))

    if not time_conversion.is_time_in_spc_date(
            start_time_unix_sec, start_spc_date_string):
        error_string = (
            'Start time ({0:s}) is not in first SPC date ({1:s}).'.format(
                time_conversion.unix_sec_to_string(
                    start_time_unix_sec, TIME_FORMAT), start_spc_date_string))
        raise ValueError(error_string)

    if not time_conversion.is_time_in_spc_date(
            end_time_unix_sec, end_spc_date_string):
        error_string = (
            'End time ({0:s}) is not in last SPC date ({1:s}).'.format(
                time_conversion.unix_sec_to_string(
                    end_time_unix_sec, TIME_FORMAT), end_spc_date_string))
        raise ValueError(error_string)

    error_checking.assert_is_greater(end_time_unix_sec, start_time_unix_sec)

    # Create list of SPC dates in period.
    start_spc_date_unix_sec = time_conversion.spc_date_string_to_unix_sec(
        start_spc_date_string)
    end_spc_date_unix_sec = time_conversion.spc_date_string_to_unix_sec(
        end_spc_date_string)

    num_spc_dates = int(
        1 + (end_spc_date_unix_sec - start_spc_date_unix_sec) / DAYS_TO_SECONDS)
    spc_dates_unix_sec = numpy.linspace(
        start_spc_date_unix_sec, end_spc_date_unix_sec, num=num_spc_dates,
        dtype=int)
    spc_date_strings = [time_conversion.time_to_spc_date_string(t)
                        for t in spc_dates_unix_sec]

    # Find radar files.
    input_radar_file_names = []
    output_tracking_file_names = []
    unix_times_sec = numpy.array([], dtype=int)

    for i in range(num_spc_dates):
        these_radar_file_names = (
            myrorss_and_mrms_io.find_raw_files_one_spc_date(
                spc_date_string=spc_date_strings[i],
                field_name=echo_top_field_name,
                data_source=data_source, top_directory_name=top_radar_dir_name,
                raise_error_if_missing=True))

        # TODO(thunderhoser): stop using protected method.
        these_times_unix_sec = numpy.array(
            [myrorss_and_mrms_io._raw_file_name_to_time(f)
             for f in these_radar_file_names], dtype=int)

        if i == 0:
            keep_time_indices = numpy.where(
                these_times_unix_sec >= start_time_unix_sec)[0]
            these_times_unix_sec = these_times_unix_sec[keep_time_indices]
            these_radar_file_names = [
                these_radar_file_names[k] for k in keep_time_indices]

        if i == num_spc_dates - 1:
            keep_time_indices = numpy.where(
                these_times_unix_sec <= end_time_unix_sec)[0]
            these_times_unix_sec = these_times_unix_sec[keep_time_indices]
            these_radar_file_names = [
                these_radar_file_names[k] for k in keep_time_indices]

        this_num_times = len(these_times_unix_sec)
        these_tracking_file_names = [''] * this_num_times
        for j in range(this_num_times):
            these_tracking_file_names[j] = tracking_io.find_processed_file(
                unix_time_sec=these_times_unix_sec[j],
                data_source=tracking_utils.SEGMOTION_SOURCE_ID,
                top_processed_dir_name=top_tracking_dir_name,
                tracking_scale_metres2=int(numpy.round(tracking_scale_metres2)),
                spc_date_string=spc_date_strings[i],
                raise_error_if_missing=False)

        input_radar_file_names += these_radar_file_names
        output_tracking_file_names += these_tracking_file_names
        unix_times_sec = numpy.concatenate((
            unix_times_sec, these_times_unix_sec))

    sort_indices = numpy.argsort(unix_times_sec)
    unix_times_sec = unix_times_sec[sort_indices]
    input_radar_file_names = [input_radar_file_names[k] for k in sort_indices]
    output_tracking_file_names = [
        output_tracking_file_names[k] for k in sort_indices]

    return {
        RADAR_FILE_NAMES_KEY: input_radar_file_names,
        TRACKING_FILE_NAMES_KEY: output_tracking_file_names,
        VALID_TIMES_KEY: unix_times_sec
    }


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

    for i in range(num_times):
        this_num_storm_objects = len(local_max_dict_by_time[i][LATITUDES_KEY])
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

    storm_object_dict = {
        tracking_utils.STORM_ID_COLUMN: all_storm_ids,
        tracking_utils.TIME_COLUMN: all_times_unix_sec,
        tracking_utils.SPC_DATE_COLUMN: all_spc_dates_unix_sec,
        tracking_utils.CENTROID_LAT_COLUMN: all_centroid_latitudes_deg,
        tracking_utils.CENTROID_LNG_COLUMN: all_centroid_longitudes_deg,
        CENTROID_X_COLUMN: all_centroid_x_metres,
        CENTROID_Y_COLUMN: all_centroid_y_metres
    }

    return pandas.DataFrame.from_dict(storm_object_dict)


def _remove_short_tracks(
        storm_object_table,
        min_duration_seconds=DEFAULT_MIN_TRACK_DURATION_SECONDS):
    """Removes short-lived storm tracks.

    :param storm_object_table: pandas DataFrame created by
        _local_maxima_to_storm_tracks.
    :param min_duration_seconds: Minimum storm duration.  Any track with
        duration < `min_duration_seconds` will be dropped.
    :return: storm_object_table: Same as input, except maybe with fewer rows.
    """

    all_storm_ids = numpy.array(
        storm_object_table[tracking_utils.STORM_ID_COLUMN].values)
    unique_storm_ids, storm_ids_object_to_unique = numpy.unique(
        all_storm_ids, return_inverse=True)
    rows_to_remove = numpy.array([], dtype=int)

    for i in range(len(unique_storm_ids)):
        these_object_indices = numpy.where(storm_ids_object_to_unique == i)[0]
        these_times_unix_sec = storm_object_table[
            tracking_utils.TIME_COLUMN].values[these_object_indices]

        this_duration_seconds = (
            numpy.max(these_times_unix_sec) - numpy.min(these_times_unix_sec))
        if this_duration_seconds >= min_duration_seconds:
            continue

        rows_to_remove = numpy.concatenate((
            rows_to_remove, these_object_indices))

    return storm_object_table.drop(
        storm_object_table.index[rows_to_remove], axis=0, inplace=False)


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


def _storm_objects_to_polygons(
        storm_object_table, file_dictionary, projection_object,
        object_area_metres2):
    """Creates bounding polygon for each storm object.

    N = number of time steps
    P = number of grid points in a given storm object

    :param storm_object_table: pandas DataFrame created by
        `_local_maxima_to_storm_tracks`.
    :param file_dictionary: Dictionary with keys created by
        `_find_input_radar_files`, as well as those listed below.
    file_dictionary['list_of_radar_metadata_dicts']: length-N list of
        dictionaries, where the [i]th dictionary contains metadata for the radar
        grid at the [i]th time step.

    :param projection_object: Instance of `pyproj.Proj`, which will be used to
        convert grid coordinates from lat-long to x-y.
    :param object_area_metres2: Each storm object will have approx this area.
    :return: storm_object_table: Same as input, but with additional columns
        listed below.
    storm_object_table.grid_point_latitudes_deg: length-P numpy array with
        latitudes (deg N) of grid points in storm object.
    storm_object_table.grid_point_longitudes_deg: length-P numpy array with
        longitudes (deg E) of grid points in storm object.
    storm_object_table.grid_point_rows: length-P numpy array with row indices
        (integers) of grid points in storm object.
    storm_object_table.grid_point_columns: length-P numpy array with column
        indices (integers) of grid points in storm object.
    storm_object_table.polygon_object_latlng: Instance of
        `shapely.geometry.Polygon`, with vertices in lat-long coordinates.
    storm_object_table.polygon_object_rowcol: Instance of
        `shapely.geometry.Polygon`, with vertices in row-column coordinates.
    """

    num_storm_objects = len(storm_object_table.index)
    object_array = numpy.full(num_storm_objects, numpy.nan, dtype=object)
    nested_array = storm_object_table[[
        tracking_utils.STORM_ID_COLUMN,
        tracking_utils.STORM_ID_COLUMN]].values.tolist()

    argument_dict = {
        tracking_utils.GRID_POINT_ROW_COLUMN: nested_array,
        tracking_utils.GRID_POINT_COLUMN_COLUMN: nested_array,
        tracking_utils.GRID_POINT_LAT_COLUMN: nested_array,
        tracking_utils.GRID_POINT_LNG_COLUMN: nested_array,
        tracking_utils.POLYGON_OBJECT_LATLNG_COLUMN: object_array,
        tracking_utils.POLYGON_OBJECT_ROWCOL_COLUMN: object_array}
    storm_object_table = storm_object_table.assign(**argument_dict)

    object_radius_metres = numpy.sqrt(float(object_area_metres2) / numpy.pi)
    radar_times_unix_sec = file_dictionary[VALID_TIMES_KEY]
    num_radar_times = len(radar_times_unix_sec)

    for i in range(num_radar_times):
        this_time_string = time_conversion.unix_sec_to_string(
            radar_times_unix_sec[i], TIME_FORMAT)
        print 'Creating polygons for storm objects at {0:s}...'.format(
            this_time_string)

        these_object_indices = numpy.where(
            storm_object_table[tracking_utils.TIME_COLUMN] ==
            radar_times_unix_sec[i])[0]
        this_num_storm_objects = len(these_object_indices)
        if this_num_storm_objects == 0:
            continue

        this_radar_metadata_dict = file_dictionary[RADAR_METADATA_DICTS_KEY][i]
        this_min_latitude_deg = (
            this_radar_metadata_dict[radar_utils.NW_GRID_POINT_LAT_COLUMN] - (
                this_radar_metadata_dict[radar_utils.LAT_SPACING_COLUMN] *
                (this_radar_metadata_dict[radar_utils.NUM_LAT_COLUMN] - 1)))

        these_grid_point_lats_deg, these_grid_point_lngs_deg = (
            grids.get_latlng_grid_points(
                min_latitude_deg=this_min_latitude_deg,
                min_longitude_deg=
                this_radar_metadata_dict[radar_utils.NW_GRID_POINT_LNG_COLUMN],
                lat_spacing_deg=
                this_radar_metadata_dict[radar_utils.LAT_SPACING_COLUMN],
                lng_spacing_deg=
                this_radar_metadata_dict[radar_utils.LNG_SPACING_COLUMN],
                num_rows=this_radar_metadata_dict[radar_utils.NUM_LAT_COLUMN],
                num_columns=
                this_radar_metadata_dict[radar_utils.NUM_LNG_COLUMN]))

        this_latitude_matrix_deg, this_longitude_matrix_deg = (
            grids.latlng_vectors_to_matrices(
                these_grid_point_lats_deg, these_grid_point_lngs_deg))

        this_x_matrix_metres, this_y_matrix_metres = (
            projections.project_latlng_to_xy(
                this_latitude_matrix_deg, this_longitude_matrix_deg,
                projection_object=projection_object, false_easting_metres=0.,
                false_northing_metres=0.))

        for j in these_object_indices:
            these_grid_point_rows, these_grid_point_columns = (
                _get_grid_points_in_radius(
                    x_grid_matrix_metres=this_x_matrix_metres,
                    y_grid_matrix_metres=this_y_matrix_metres,
                    x_query_metres=
                    storm_object_table[CENTROID_X_COLUMN].values[j],
                    y_query_metres=
                    storm_object_table[CENTROID_Y_COLUMN].values[j],
                    radius_metres=object_radius_metres))

            these_vertex_rows, these_vertex_columns = (
                polygons.grid_points_in_poly_to_vertices(
                    these_grid_point_rows, these_grid_point_columns))

            (storm_object_table[tracking_utils.GRID_POINT_ROW_COLUMN].values[j],
             storm_object_table[
                 tracking_utils.GRID_POINT_COLUMN_COLUMN].values[j]) = (
                     polygons.simple_polygon_to_grid_points(
                         these_vertex_rows, these_vertex_columns))

            (storm_object_table[tracking_utils.GRID_POINT_LAT_COLUMN].values[j],
             storm_object_table[
                 tracking_utils.GRID_POINT_LNG_COLUMN].values[j]) = (
                     radar_utils.rowcol_to_latlng(
                         storm_object_table[
                             tracking_utils.GRID_POINT_ROW_COLUMN].values[j],
                         storm_object_table[
                             tracking_utils.GRID_POINT_COLUMN_COLUMN].values[j],
                         nw_grid_point_lat_deg=this_radar_metadata_dict[
                             radar_utils.NW_GRID_POINT_LAT_COLUMN],
                         nw_grid_point_lng_deg=this_radar_metadata_dict[
                             radar_utils.NW_GRID_POINT_LNG_COLUMN],
                         lat_spacing_deg=this_radar_metadata_dict[
                             radar_utils.LAT_SPACING_COLUMN],
                         lng_spacing_deg=this_radar_metadata_dict[
                             radar_utils.LNG_SPACING_COLUMN]))

            these_vertex_lat_deg, these_vertex_lng_deg = (
                radar_utils.rowcol_to_latlng(
                    these_vertex_rows, these_vertex_columns,
                    nw_grid_point_lat_deg=this_radar_metadata_dict[
                        radar_utils.NW_GRID_POINT_LAT_COLUMN],
                    nw_grid_point_lng_deg=this_radar_metadata_dict[
                        radar_utils.NW_GRID_POINT_LNG_COLUMN],
                    lat_spacing_deg=
                    this_radar_metadata_dict[radar_utils.LAT_SPACING_COLUMN],
                    lng_spacing_deg=
                    this_radar_metadata_dict[radar_utils.LNG_SPACING_COLUMN]))

            storm_object_table[
                tracking_utils.POLYGON_OBJECT_ROWCOL_COLUMN].values[j] = (
                    polygons.vertex_arrays_to_polygon_object(
                        these_vertex_columns, these_vertex_rows))

            storm_object_table[
                tracking_utils.POLYGON_OBJECT_LATLNG_COLUMN].values[j] = (
                    polygons.vertex_arrays_to_polygon_object(
                        these_vertex_lng_deg, these_vertex_lat_deg))

    return storm_object_table


def run_tracking(
        top_radar_dir_name, top_tracking_dir_name, start_spc_date_string,
        end_spc_date_string,
        echo_top_field_name=radar_utils.ECHO_TOP_40DBZ_NAME,
        radar_data_source=radar_utils.MYRORSS_SOURCE_ID,
        storm_object_area_metres2=DEFAULT_STORM_OBJECT_AREA_METRES2,
        start_time_unix_sec=None, end_time_unix_sec=None,
        min_echo_top_height_km_asl=DEFAULT_MIN_ECHO_TOP_HEIGHT_KM_ASL,
        e_folding_radius_for_smoothing_pixels=
        DEFAULT_E_FOLDING_RADIUS_FOR_SMOOTHING_PIXELS,
        half_width_for_max_filter_pixels=
        DEFAULT_HALF_WIDTH_FOR_MAX_FILTER_PIXELS,
        min_distance_between_maxima_metres=
        DEFAULT_MIN_DISTANCE_BETWEEN_MAXIMA_METRES,
        max_link_time_seconds=DEFAULT_MAX_LINK_TIME_SECONDS,
        max_link_distance_m_s01=DEFAULT_MAX_LINK_DISTANCE_M_S01,
        min_track_duration_seconds=DEFAULT_MIN_TRACK_DURATION_SECONDS,
        num_points_back_for_velocity=DEFAULT_NUM_POINTS_BACK_FOR_VELOCITY):
    """Runs tracking algorithm for the given time period and radar field.

    :param top_radar_dir_name: See doc for `_find_input_radar_files`.
    :param top_tracking_dir_name: See doc for `_find_input_radar_files`.
    :param start_spc_date_string: See doc for `_find_input_radar_files`.
    :param end_spc_date_string: See doc for `_find_input_radar_files`.
    :param echo_top_field_name: See documentation for `_find_input_radar_files`.
    :param radar_data_source: See doc for `_find_input_radar_files`.
    :param storm_object_area_metres2: Area for bounding polygon around each
        storm object.
    :param start_time_unix_sec: See doc for `_find_input_radar_files`.
    :param end_time_unix_sec: See doc for `_find_input_radar_files`.
    :param min_echo_top_height_km_asl: Minimum echo-top height (km above sea
        level).  Only local maxima >= `min_echo_top_height_km_asl` will be
        tracked.
    :param e_folding_radius_for_smoothing_pixels: See doc for
        `_gaussian_smooth_radar_field`.  This will be applied separately to the
        radar field at each time step, before finding local maxima.
    :param half_width_for_max_filter_pixels: See doc for `_find_local_maxima`.
    :param min_distance_between_maxima_metres: See doc for
        `_remove_redundant_local_maxima`.
    :param max_link_time_seconds: See doc for `_link_local_maxima_in_time`.
    :param max_link_distance_m_s01: See doc for `_link_local_maxima_in_time`.
    :param min_track_duration_seconds: Minimum track duration.  Shorter-lived
        storms will be removed.
    :param num_points_back_for_velocity: See doc for
        `_get_velocities_one_storm_track`.
    :return: storm_object_table: pandas DataFrame with columns listed in
        `storm_tracking_io.write_processed_file`.
    :return: file_dictionary: See documentation for `_find_input_radar_files`.
    """

    error_checking.assert_is_greater(min_echo_top_height_km_asl, 0.)

    file_dictionary = _find_input_radar_files(
        echo_top_field_name=echo_top_field_name, data_source=radar_data_source,
        top_radar_dir_name=top_radar_dir_name,
        top_tracking_dir_name=top_tracking_dir_name,
        tracking_scale_metres2=storm_object_area_metres2,
        start_spc_date_string=start_spc_date_string,
        end_spc_date_string=end_spc_date_string,
        start_time_unix_sec=start_time_unix_sec,
        end_time_unix_sec=end_time_unix_sec)

    input_radar_file_names = file_dictionary[RADAR_FILE_NAMES_KEY]
    unix_times_sec = file_dictionary[VALID_TIMES_KEY]

    projection_object = projections.init_azimuthal_equidistant_projection(
        central_latitude_deg=CENTRAL_PROJ_LATITUDE_DEG,
        central_longitude_deg=CENTRAL_PROJ_LONGITUDE_DEG)

    time_strings = [time_conversion.unix_sec_to_string(t, TIME_FORMAT)
                    for t in unix_times_sec]

    num_times = len(unix_times_sec)
    local_max_dict_by_time = [{}] * num_times
    file_dictionary[RADAR_METADATA_DICTS_KEY] = [{}] * num_times

    for i in range(num_times):
        print 'Finding local maxima in "{0:s}" at {1:s}...'.format(
            echo_top_field_name, time_strings[i])

        file_dictionary[RADAR_METADATA_DICTS_KEY][i] = (
            myrorss_and_mrms_io.read_metadata_from_raw_file(
                input_radar_file_names[i], data_source=radar_data_source))

        this_sparse_grid_table = (
            myrorss_and_mrms_io.read_data_from_sparse_grid_file(
                input_radar_file_names[i],
                field_name_orig=file_dictionary[RADAR_METADATA_DICTS_KEY][i][
                    myrorss_and_mrms_io.FIELD_NAME_COLUMN_ORIG],
                data_source=radar_data_source,
                sentinel_values=file_dictionary[RADAR_METADATA_DICTS_KEY][i][
                    radar_utils.SENTINEL_VALUE_COLUMN]))

        this_echo_top_matrix_km_asl, _, _ = radar_s2f.sparse_to_full_grid(
            this_sparse_grid_table,
            file_dictionary[RADAR_METADATA_DICTS_KEY][i],
            ignore_if_below=min_echo_top_height_km_asl)

        this_echo_top_matrix_km_asl = _gaussian_smooth_radar_field(
            this_echo_top_matrix_km_asl,
            e_folding_radius_pixels=e_folding_radius_for_smoothing_pixels)

        local_max_dict_by_time[i] = _find_local_maxima(
            this_echo_top_matrix_km_asl,
            file_dictionary[RADAR_METADATA_DICTS_KEY][i],
            neigh_half_width_in_pixels=half_width_for_max_filter_pixels)

        local_max_dict_by_time[i] = _remove_redundant_local_maxima(
            local_max_dict_by_time[i], projection_object=projection_object,
            min_distance_between_maxima_metres=
            min_distance_between_maxima_metres)
        local_max_dict_by_time[i].update({VALID_TIME_KEY: unix_times_sec[i]})

        if i == 0:
            these_current_to_prev_indices = _link_local_maxima_in_time(
                current_local_max_dict=local_max_dict_by_time[i],
                previous_local_max_dict=None,
                max_link_time_seconds=max_link_time_seconds,
                max_link_distance_m_s01=max_link_distance_m_s01)
        else:
            print (
                'Linking local maxima at {0:s} with those at {1:s}...\n'.format(
                    time_strings[i], time_strings[i - 1]))

            these_current_to_prev_indices = _link_local_maxima_in_time(
                current_local_max_dict=local_max_dict_by_time[i],
                previous_local_max_dict=local_max_dict_by_time[i - 1],
                max_link_time_seconds=max_link_time_seconds,
                max_link_distance_m_s01=max_link_distance_m_s01)

        local_max_dict_by_time[i].update(
            {CURRENT_TO_PREV_INDICES_KEY: these_current_to_prev_indices})

    print ('Converting time series of local "{0:s}" maxima to storm '
           'tracks...').format(echo_top_field_name)
    storm_object_table = _local_maxima_to_storm_tracks(local_max_dict_by_time)

    print 'Removing tracks with duration < {0:d} seconds...'.format(
        int(min_track_duration_seconds))
    storm_object_table = _remove_short_tracks(
        storm_object_table, min_duration_seconds=min_track_duration_seconds)

    print 'Computing storm age for each storm object...'
    storm_object_table = best_tracks.recompute_attributes(
        storm_object_table, best_track_start_time_unix_sec=unix_times_sec[0],
        best_track_end_time_unix_sec=unix_times_sec[-1])

    print 'Computing velocity for each storm object...\n'
    storm_object_table = _get_storm_velocities(
        storm_object_table, num_points_back=num_points_back_for_velocity)

    storm_object_table = _storm_objects_to_polygons(
        storm_object_table=storm_object_table, file_dictionary=file_dictionary,
        projection_object=projection_object,
        object_area_metres2=storm_object_area_metres2)
    return storm_object_table, file_dictionary


def write_storm_objects(storm_object_table, file_dictionary):
    """Writes storm objects to one Pickle file per time step.

    :param storm_object_table: pandas DataFrame created by `run_tracking`.
    :param file_dictionary: Dictionary created by `_find_input_radar_files`.
    """

    pickle_file_names = file_dictionary[TRACKING_FILE_NAMES_KEY]
    file_times_unix_sec = file_dictionary[VALID_TIMES_KEY]
    num_files = len(pickle_file_names)

    for i in range(num_files):
        print 'Writing storm objects to file: "{0:s}"...'.format(
            pickle_file_names[i])

        this_storm_object_table = storm_object_table.loc[
            storm_object_table[tracking_utils.TIME_COLUMN] ==
            file_times_unix_sec[i]]
        tracking_io.write_processed_file(
            this_storm_object_table, pickle_file_names[i])
