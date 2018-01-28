"""IO methods for storm-tracking data (both polygons and tracks)."""

import os
import glob
import pickle
import numpy
import pandas
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

DATE_FORMAT = '%Y%m%d'
TIME_FORMAT = '%Y-%m-%d-%H%M%S'
REGEX_FOR_TIME_FORMAT = (
    '[0-9][0-9][0-9][0-9]-[0-1][0-9]-[0-3][0-9]-[0-2][0-9][0-5][0-9][0-5][0-9]')

PROCESSED_FILE_PREFIX = 'storm-tracking'
PROCESSED_FILE_EXTENSION = '.p'

MANDATORY_COLUMNS = [
    tracking_utils.STORM_ID_COLUMN, tracking_utils.TIME_COLUMN,
    tracking_utils.SPC_DATE_COLUMN, tracking_utils.EAST_VELOCITY_COLUMN,
    tracking_utils.NORTH_VELOCITY_COLUMN, tracking_utils.AGE_COLUMN,
    tracking_utils.CENTROID_LAT_COLUMN, tracking_utils.CENTROID_LNG_COLUMN,
    tracking_utils.GRID_POINT_LAT_COLUMN, tracking_utils.GRID_POINT_LNG_COLUMN,
    tracking_utils.GRID_POINT_ROW_COLUMN,
    tracking_utils.GRID_POINT_COLUMN_COLUMN,
    tracking_utils.POLYGON_OBJECT_LATLNG_COLUMN,
    tracking_utils.POLYGON_OBJECT_ROWCOL_COLUMN,
    tracking_utils.TRACKING_START_TIME_COLUMN,
    tracking_utils.TRACKING_END_TIME_COLUMN]

BEST_TRACK_COLUMNS = [tracking_utils.ORIG_STORM_ID_COLUMN]


def _get_pathless_processed_file_name(unix_time_sec, data_source):
    """Generates pathless name for processed tracking file.

    This file should contain storm objects (bounding polygons) and tracking
    statistics for one time step and one tracking scale.

    :param unix_time_sec: Valid time.
    :param data_source: Data source (string).
    :return: pathless_file_name: Pathless name for processed file.
    """

    return '{0:s}_{1:s}_{2:s}{3:s}'.format(
        PROCESSED_FILE_PREFIX, data_source,
        time_conversion.unix_sec_to_string(unix_time_sec, TIME_FORMAT),
        PROCESSED_FILE_EXTENSION)


def _get_relative_processed_directory(
        data_source, tracking_scale_metres2, unix_time_sec=None,
        spc_date_string=None):
    """Generates relative directory for processed tracking files.

    Each file should contain storm objects (bounding polygons) and tracking
    statistics for one time step and one tracking scale.

    :param data_source: Data source (string).
    :param tracking_scale_metres2: Tracking scale (minimum storm-object area).
    :param unix_time_sec: Valid time (needed only if data_source =
        "probSevere").
    :param spc_date_string: SPC date (format "yyyymmdd") (needed only if
        data_source = "segmotion").
    :return: relative_directory_name: Relative path for processed tracking
        files.
    """

    if data_source == tracking_utils.SEGMOTION_SOURCE_ID:
        date_string = spc_date_string
    else:
        date_string = time_conversion.unix_sec_to_string(
            unix_time_sec, DATE_FORMAT)

    return '{0:s}/scale_{1:d}m2'.format(
        date_string, int(tracking_scale_metres2))


def find_processed_file(
        unix_time_sec, data_source, top_processed_dir_name,
        tracking_scale_metres2, spc_date_string=None,
        raise_error_if_missing=True):
    """Finds processed tracking file.

    This file should contain storm objects (bounding polygons) and tracking
    statistics for one time step and one tracking scale.

    :param unix_time_sec: Valid time.
    :param data_source: Data source (string).
    :param top_processed_dir_name: Name of top-level directory with processed
        tracking files.
    :param tracking_scale_metres2: Tracking scale (minimum storm-object area).
    :param spc_date_string: SPC date (format "yyyymmdd") (needed only if
        data_source = "segmotion").
    :param raise_error_if_missing: Boolean flag.  If True and file is missing,
        this method will raise an error.  If False and file is missing, will
        return the *expected* path.
    :return: processed_file_name: Path to processed tracking file.  If
        raise_error_if_missing = False and file is missing, this is the
        *expected* path.
    :raises: ValueError: if raise_error_if_missing = True and file is missing.
    """

    # Verification.
    tracking_utils.check_data_source(data_source)
    if data_source == tracking_utils.SEGMOTION_SOURCE_ID:
        _ = time_conversion.spc_date_string_to_unix_sec(spc_date_string)

    error_checking.assert_is_string(top_processed_dir_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    pathless_file_name = _get_pathless_processed_file_name(
        unix_time_sec, data_source)
    relative_directory_name = _get_relative_processed_directory(
        data_source=data_source, spc_date_string=spc_date_string,
        unix_time_sec=unix_time_sec,
        tracking_scale_metres2=tracking_scale_metres2)

    processed_file_name = '{0:s}/{1:s}/{2:s}'.format(
        top_processed_dir_name, relative_directory_name, pathless_file_name)

    if raise_error_if_missing and not os.path.isfile(processed_file_name):
        raise ValueError('Cannot find processed file.  Expected at location: ' +
                         processed_file_name)

    return processed_file_name


def find_processed_files_one_spc_date(
        spc_date_string, data_source, top_processed_dir_name,
        tracking_scale_metres2, raise_error_if_missing=True):
    """Finds all processed tracking files for one SPC date.

    Each file should contain storm objects (bounding polygons) and tracking
    statistics for one time step and one tracking scale.

    :param spc_date_string: SPC date (format "yyyymmdd").
    :param data_source: Data source (string).
    :param top_processed_dir_name: Name of top-level directory with processed
        tracking files.
    :param tracking_scale_metres2: Tracking scale (minimum storm-object area).
    :param raise_error_if_missing: Boolean flag.  If True and no files are
        found, this method will raise an error.
    :return: processed_file_names: 1-D list of paths to processed tracking
        files.
    :raises: ValueError: if raise_error_if_missing = True and no files are
        found.
    """

    error_checking.assert_is_boolean(raise_error_if_missing)

    example_time_unix_sec = time_conversion.spc_date_string_to_unix_sec(
        spc_date_string)
    example_file_name = find_processed_file(
        unix_time_sec=example_time_unix_sec, data_source=data_source,
        spc_date_string=spc_date_string,
        top_processed_dir_name=top_processed_dir_name,
        tracking_scale_metres2=tracking_scale_metres2,
        raise_error_if_missing=False)

    example_directory_name, example_pathless_file_name = os.path.split(
        example_file_name)
    example_time_string = time_conversion.unix_sec_to_string(
        example_time_unix_sec, TIME_FORMAT)
    example_pathless_file_name = example_pathless_file_name.replace(
        example_time_string, REGEX_FOR_TIME_FORMAT)

    processed_file_pattern = '{0:s}/{1:s}'.format(
        example_directory_name, example_pathless_file_name)
    processed_file_names = glob.glob(processed_file_pattern)

    if raise_error_if_missing and not processed_file_names:
        error_string = (
            'Could not find any processed files with the following pattern: ' +
            processed_file_pattern)
        raise ValueError(error_string)

    return processed_file_names


def processed_file_name_to_time(processed_file_name):
    """Parses time from name of processed tracking file.

    This file should contain storm objects (bounding polygons) and tracking
    statistics for one time step and one tracking scale.

    :param processed_file_name: Path to processed tracking file.
    :return: unix_time_sec: Valid time.
    """

    error_checking.assert_is_string(processed_file_name)
    _, pathless_file_name = os.path.split(processed_file_name)
    extensionless_file_name, _ = os.path.splitext(pathless_file_name)

    extensionless_file_name_parts = extensionless_file_name.split('_')
    return time_conversion.string_to_unix_sec(
        extensionless_file_name_parts[-1], TIME_FORMAT)


def write_processed_file(storm_object_table, pickle_file_name):
    """Writes tracking data to Pickle file.

    This file should contain storm objects (bounding polygons) and tracking
    statistics for one time step and one tracking scale.

    N = number of storm objects
    P = number of grid points inside a given storm object

    :param storm_object_table: N-row pandas DataFrame with the following
        mandatory columns.  May also contain distance buffers created by
        `storm_tracking_utils.make_buffers_around_storm_objects`.
    storm_object_table.storm_id: String ID for storm cell.
    storm_object_table.unix_time_sec: Valid time.
    storm_object_table.spc_date_unix_sec: SPC date.
    storm_object_table.tracking_start_time_unix_sec: Start time for tracking
        period.
    storm_object_table.tracking_end_time_unix_sec: End time for tracking period.
    storm_object_table.age_sec: Age of storm cell (seconds).
    storm_object_table.east_velocity_m_s01: Eastward velocity of storm cell
        (metres per second).
    storm_object_table.north_velocity_m_s01: Northward velocity of storm cell
        (metres per second).
    storm_object_table.centroid_lat_deg: Latitude at centroid of storm object
        (deg N).
    storm_object_table.centroid_lng_deg: Longitude at centroid of storm object
        (deg E).
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

    :param pickle_file_name: Path to output file.
    """

    distance_buffer_column_names = tracking_utils.get_distance_buffer_columns(
        storm_object_table)
    if distance_buffer_column_names is None:
        distance_buffer_column_names = []

    columns_to_write = MANDATORY_COLUMNS + distance_buffer_column_names
    best_track_column_present_flags = numpy.array(
        [c in list(storm_object_table) for c in BEST_TRACK_COLUMNS])
    if numpy.all(best_track_column_present_flags):
        columns_to_write += BEST_TRACK_COLUMNS

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(storm_object_table[columns_to_write], pickle_file_handle)
    pickle_file_handle.close()


def read_processed_file(pickle_file_name):
    """Reads tracking data from processed file.

    This file should contain storm objects (bounding polygons) and tracking
    statistics for one time step and one tracking scale.

    :param pickle_file_name: Path to input file.
    :return: storm_object_table: See documentation for write_processed_file.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    storm_object_table = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    error_checking.assert_columns_in_dataframe(
        storm_object_table, MANDATORY_COLUMNS)
    return storm_object_table


def read_many_processed_files(pickle_file_names):
    """Reads tracking data from many processed files.

    Data from all files are concatenated into the same pandas DataFrame.

    :param pickle_file_names: 1-D list of paths to input files.
    :return: storm_object_table: See documentation for write processed file.
    """

    error_checking.assert_is_string_list(pickle_file_names)
    error_checking.assert_is_numpy_array(
        numpy.array(pickle_file_names), num_dimensions=1)

    num_files = len(pickle_file_names)
    list_of_storm_object_tables = [None] * num_files

    for i in range(num_files):
        list_of_storm_object_tables[i] = read_processed_file(
            pickle_file_names[i])
        if i == 0:
            continue

        list_of_storm_object_tables[i], _ = (
            list_of_storm_object_tables[i].align(
                list_of_storm_object_tables[0], axis=1))

    return pandas.concat(
        list_of_storm_object_tables, axis=0, ignore_index=True)
