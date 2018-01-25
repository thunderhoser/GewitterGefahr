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
from gewittergefahr.gg_io import myrorss_and_mrms_io
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import radar_sparse_to_full as radar_s2f
from gewittergefahr.gg_utils import dilation
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import grid_smoothing_2d
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking

# TODO(thunderhoser): Add unit tests, method that assigns storm IDs, method that
# assigns storm properties, and IO methods for tracking files.

TOLERANCE = 1e-6
TIME_FORMAT = '%Y-%m-%d-%H%M%S'

DAYS_TO_SECONDS = 86400
DEGREES_LAT_TO_METRES = 60 * 1852

CENTRAL_PROJ_LATITUDE_DEG = 35.
CENTRAL_PROJ_LONGITUDE_DEG = 295.

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
DEFAULT_MAX_LINK_DISTANCE_M_S01 = 50.

RADAR_FILE_NAMES_KEY = 'radar_file_names'
VALID_TIMES_KEY = 'unix_times_sec'

LATITUDES_KEY = 'latitudes_deg'
LONGITUDES_KEY = 'longitudes_deg'
MAX_VALUES_KEY = 'max_values'
X_COORDS_KEY = 'x_coords_metres'
Y_COORDS_KEY = 'y_coords_metres'
VALID_TIME_KEY = 'unix_time_sec'
CURRENT_TO_PREV_INDICES_KEY = 'current_to_previous_indices'


def _check_radar_field(radar_field_name):
    """Ensures that radar field is valid for GridRad tracking.

    :param radar_field_name: Field name (string).
    :raises: ValueError: if `radar_field_name not in VALID_RADAR_FIELDS`.
    """

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
        echo_top_field_name, data_source, top_directory_name,
        start_spc_date_string, end_spc_date_string, start_time_unix_sec=None,
        end_time_unix_sec=None):
    """Finds input radar files for the given time period and radar field.

    N = number of radar files found

    :param echo_top_field_name: Name of radar field to use for tracking.  Must
        be an echo-top field (to my knowledge, the GridRad method hasn't been
        used on other fields, such as reflectivity).
    :param data_source: Data source (must be either "myrorss" or "mrms").
    :param top_directory_name: Name of top-level directory with radar data from
        the given source.
    :param start_spc_date_string: First SPC date in period (format "yyyymmdd").
    :param end_spc_date_string: Last SPC date in period (format "yyyymmdd").
    :param start_time_unix_sec: Start of time period.  Default is 1200 UTC at
        beginning of the given SPC date (e.g., if SPC date is "20180124", this
        will be 1200 UTC 24 Jan 2018).
    :param end_time_unix_sec: End of time period.  Default is 1200 UTC at end of
        the given SPC date (e.g., if SPC date is "20180124", this will be 1200
        UTC 25 Jan 2018).
    :return: file_dictionary: Dictionary with the following keys.
    file_dictionary['radar_file_names']: length-N list of paths to radar files.
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
    unix_times_sec = numpy.array([], dtype=int)

    for i in range(num_spc_dates):
        these_file_names = myrorss_and_mrms_io.find_raw_files_one_spc_date(
            spc_date_string=spc_date_strings[i], field_name=echo_top_field_name,
            data_source=data_source, top_directory_name=top_directory_name,
            raise_error_if_missing=True)

        # TODO(thunderhoser): stop accessing protected method.
        these_times_unix_sec = numpy.array(
            [myrorss_and_mrms_io._raw_file_name_to_time(f)
             for f in these_file_names], dtype=int)

        if i == 0:
            keep_time_indices = numpy.where(
                these_times_unix_sec >= start_time_unix_sec)[0]
            these_times_unix_sec = these_times_unix_sec[keep_time_indices]
            these_file_names = [these_file_names[i] for i in keep_time_indices]

        if i == num_spc_dates - 1:
            keep_time_indices = numpy.where(
                these_times_unix_sec <= end_time_unix_sec)[0]
            these_times_unix_sec = these_times_unix_sec[keep_time_indices]
            these_file_names = [these_file_names[i] for i in keep_time_indices]

        input_radar_file_names += these_file_names
        unix_times_sec = numpy.concatenate((
            unix_times_sec, these_times_unix_sec))

    print input_radar_file_names

    return {
        RADAR_FILE_NAMES_KEY: input_radar_file_names,
        VALID_TIMES_KEY: unix_times_sec
    }


def run_tracking(
        echo_top_field_name, top_radar_dir_name, start_spc_date_string,
        end_spc_date_string, radar_data_source=radar_utils.MYRORSS_SOURCE_ID,
        start_time_unix_sec=None, end_time_unix_sec=None,
        min_echo_top_height_km_asl=DEFAULT_MIN_ECHO_TOP_HEIGHT_KM_ASL,
        e_folding_radius_for_smoothing_pixels=
        DEFAULT_E_FOLDING_RADIUS_FOR_SMOOTHING_PIXELS,
        half_width_for_max_filter_pixels=
        DEFAULT_HALF_WIDTH_FOR_MAX_FILTER_PIXELS,
        min_distance_between_maxima_metres=
        DEFAULT_MIN_DISTANCE_BETWEEN_MAXIMA_METRES,
        max_link_time_seconds=DEFAULT_MAX_LINK_TIME_SECONDS,
        max_link_distance_m_s01=DEFAULT_MAX_LINK_DISTANCE_M_S01):
    """Runs tracking algorithm for the given time period and radar field.

    :param echo_top_field_name: See documentation for `_find_input_radar_files`.
    :param top_radar_dir_name: See doc for `_find_input_radar_files`.
    :param start_spc_date_string: See doc for `_find_input_radar_files`.
    :param end_spc_date_string: See doc for `_find_input_radar_files`.
    :param radar_data_source: See doc for `_find_input_radar_files`.
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
    :return: poop: Haven't decided yet.
    """

    error_checking.assert_is_greater(min_echo_top_height_km_asl, 0.)

    file_dictionary = _find_input_radar_files(
        echo_top_field_name=echo_top_field_name, data_source=radar_data_source,
        top_directory_name=top_radar_dir_name,
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

    for i in range(num_times):
        print 'Finding local maxima in "{0:s}" at {1:s}...'.format(
            echo_top_field_name, time_strings[i])

        this_radar_metadata_dict = (
            myrorss_and_mrms_io.read_metadata_from_raw_file(
                input_radar_file_names[i], data_source=radar_data_source))

        this_sparse_grid_table = (
            myrorss_and_mrms_io.read_data_from_sparse_grid_file(
                input_radar_file_names[i],
                field_name_orig=this_radar_metadata_dict[
                    myrorss_and_mrms_io.FIELD_NAME_COLUMN_ORIG],
                data_source=radar_data_source,
                sentinel_values=
                this_radar_metadata_dict[radar_utils.SENTINEL_VALUE_COLUMN]))

        this_echo_top_matrix_km_asl, _, _ = radar_s2f.sparse_to_full_grid(
            this_sparse_grid_table, this_radar_metadata_dict,
            ignore_if_below=min_echo_top_height_km_asl)

        this_echo_top_matrix_km_asl = _gaussian_smooth_radar_field(
            this_echo_top_matrix_km_asl,
            e_folding_radius_pixels=e_folding_radius_for_smoothing_pixels)

        local_max_dict_by_time[i] = _find_local_maxima(
            this_echo_top_matrix_km_asl, this_radar_metadata_dict,
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
            print 'Linking local maxima at {0:s} with those at {1:s}...'.format(
                time_strings[i], time_strings[i - 1])

            these_current_to_prev_indices = _link_local_maxima_in_time(
                current_local_max_dict=local_max_dict_by_time[i],
                previous_local_max_dict=local_max_dict_by_time[i - 1],
                max_link_time_seconds=max_link_time_seconds,
                max_link_distance_m_s01=max_link_distance_m_s01)

        local_max_dict_by_time[i].update(
            {CURRENT_TO_PREV_INDICES_KEY: these_current_to_prev_indices})
        print local_max_dict_by_time[i]
