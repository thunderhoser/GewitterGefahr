"""IO methods for storm-tracking data (both polygons and tracks)."""

import os
import copy
import glob
import pickle
import numpy
import pandas
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

DATE_FORMAT = '%Y%m%d'
TIME_FORMAT = '%Y-%m-%d-%H%M%S'

SPC_DATE_REGEX = '[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]'
YEAR_REGEX = '[0-9][0-9][0-9][0-9]'
MONTH_REGEX = '[0-1][0-9]'
DAY_OF_MONTH_REGEX = '[0-3][0-9]'
HOUR_REGEX = '[0-2][0-9]'
MINUTE_REGEX = '[0-5][0-9]'
SECOND_REGEX = '[0-5][0-9]'

TIME_FORMAT_AS_REGEX = '{0:s}-{1:s}-{2:s}-{3:s}{4:s}{5:s}'.format(
    YEAR_REGEX, MONTH_REGEX, DAY_OF_MONTH_REGEX, HOUR_REGEX, MINUTE_REGEX,
    SECOND_REGEX)

PREFIX_FOR_PATHLESS_PROCESSED_FILE_NAME = 'storm-tracking'
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
CELL_TIME_COLUMNS = [
    tracking_utils.CELL_START_TIME_COLUMN, tracking_utils.CELL_END_TIME_COLUMN]

LATITUDE_COLUMN_IN_AMY_FILES = 'Latitude'
LONGITUDE_COLUMN_IN_AMY_FILES = 'Longitude'
AGE_COLUMN_IN_AMY_FILES = 'Age'
EAST_VELOCITY_COLUMN_IN_AMY_FILES = 'MotionEast'
SOUTH_VELOCITY_COLUMN_IN_AMY_FILES = 'MotionSouth'
ORIG_ID_COLUMN_IN_AMY_FILES = 'RowName'
AREA_COLUMN_IN_AMY_FILES = 'Size'
SPEED_COLUMN_IN_AMY_FILES = 'Speed'
VALID_TIME_COLUMN_IN_AMY_FILES = 'ValidTime'
CELL_START_TIME_COLUMN_IN_AMY_FILES = 'StartTime'
CELL_END_TIME_COLUMN_IN_AMY_FILES = 'EndTime'
STORM_ID_COLUMN_IN_AMY_FILES = 'RowName2'

COLUMNS_FOR_AMY = [
    LATITUDE_COLUMN_IN_AMY_FILES, LONGITUDE_COLUMN_IN_AMY_FILES,
    AGE_COLUMN_IN_AMY_FILES, EAST_VELOCITY_COLUMN_IN_AMY_FILES,
    SOUTH_VELOCITY_COLUMN_IN_AMY_FILES, ORIG_ID_COLUMN_IN_AMY_FILES,
    AREA_COLUMN_IN_AMY_FILES, SPEED_COLUMN_IN_AMY_FILES,
    VALID_TIME_COLUMN_IN_AMY_FILES, CELL_START_TIME_COLUMN_IN_AMY_FILES,
    CELL_END_TIME_COLUMN_IN_AMY_FILES, STORM_ID_COLUMN_IN_AMY_FILES
]

TIME_FORMAT_IN_AMY_FILES = '%Y-%m-%d-%H%M%SZ'
METRES2_TO_KM2 = 1e-6


def _get_pathless_processed_file_name(unix_time_sec, data_source):
    """Generates pathless name for processed tracking file.

    This file should contain storm objects (bounding polygons) and tracking
    statistics for one time step and one tracking scale.

    :param unix_time_sec: Valid time.
    :param data_source: Data source (string).
    :return: pathless_file_name: Pathless name for processed file.
    """

    return '{0:s}_{1:s}_{2:s}{3:s}'.format(
        PREFIX_FOR_PATHLESS_PROCESSED_FILE_NAME, data_source,
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
        date_string = time_conversion.time_to_spc_date_string(unix_time_sec)

    return '{0:s}/{1:s}/scale_{2:d}m2'.format(
        date_string[:4], date_string, int(numpy.round(tracking_scale_metres2)))


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
        time_conversion.spc_date_string_to_unix_sec(spc_date_string)

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


def find_processed_files_at_times(
        top_processed_dir_name, tracking_scale_metres2, data_source,
        years=None, months=None, hours=None, raise_error_if_missing=True):
    """Finds processed files with valid time in the specified bins.

    Specifically, this method will find all processed files in the given years,
    months, *and* hours.

    :param top_processed_dir_name: Name of top-level directory with processed
        tracking files.
    :param tracking_scale_metres2: Tracking scale (minimum storm area).
    :param data_source: Data source (must be accepted by
        `storm_tracking_utils.check_data_source`).
    :param years: [may be None]
        1-D numpy array of years.
    :param months: [may be None]
        1-D numpy array of months (must all be in range 1...12).
    :param hours: [may be None]
        1-D numpy array of hours (must all be in range 0...23).
    :param raise_error_if_missing: Boolean flag.  If no files are found and
        `raise_error_if_missing = True`, this method will error out.  If no
        files are found and `raise_error_if_missing = False`, will return empty
        list.
    :return: processed_file_names: 1-D list of paths to processed tracking
        files.
    :return: glob_patterns: 1-D list of glob patterns used to find files.
    :raises: ValueError: if `years`, `months`, and `hours` are all None.
    :raises: ValueError: if no files are found and
        `raise_error_if_missing = True`.
    """

    error_checking.assert_is_string(top_processed_dir_name)
    tracking_scale_metres2 = int(numpy.round(tracking_scale_metres2))
    error_checking.assert_is_greater(tracking_scale_metres2, 0)
    tracking_utils.check_data_source(data_source)
    error_checking.assert_is_boolean(raise_error_if_missing)

    if years is None and months is None and hours is None:
        raise ValueError('`years`, `months`, and `hours` cannot all be None.')

    if years is None:
        years = numpy.array([-1], dtype=int)
    else:
        error_checking.assert_is_integer_numpy_array(years)
        error_checking.assert_is_numpy_array(years, num_dimensions=1)

    if months is None:
        months = numpy.array([-1], dtype=int)
    else:
        error_checking.assert_is_integer_numpy_array(months)
        error_checking.assert_is_numpy_array(months, num_dimensions=1)
        error_checking.assert_is_geq_numpy_array(months, 1)
        error_checking.assert_is_leq_numpy_array(months, 12)

    if hours is None:
        hours = numpy.array([-1], dtype=int)
    else:
        error_checking.assert_is_integer_numpy_array(hours)
        error_checking.assert_is_numpy_array(hours, num_dimensions=1)
        error_checking.assert_is_geq_numpy_array(hours, 0)
        error_checking.assert_is_leq_numpy_array(hours, 23)

    glob_patterns = []
    processed_file_names = []

    for this_year in years:
        for this_month in months:
            for this_hour in hours:
                if this_year == -1:
                    this_year_string = copy.deepcopy(YEAR_REGEX)
                else:
                    this_year_string = '{0:04d}'.format(this_year)

                if this_month == -1:
                    this_month_string = copy.deepcopy(MONTH_REGEX)
                else:
                    this_month_string = '{0:02d}'.format(this_month)

                if this_hour == -1:
                    this_hour_string = copy.deepcopy(HOUR_REGEX)
                else:
                    this_hour_string = '{0:02d}'.format(this_hour)

                this_file_pattern = (
                    '{0:s}/{1:s}/{2:s}/scale_{3:d}m2/'
                    '{4:s}_{5:s}_{6:s}-{7:s}-{8:s}-{9:s}{10:s}{11:s}{12:s}'
                ).format(top_processed_dir_name, YEAR_REGEX, SPC_DATE_REGEX,
                         tracking_scale_metres2,
                         PREFIX_FOR_PATHLESS_PROCESSED_FILE_NAME, data_source,
                         this_year_string, this_month_string,
                         DAY_OF_MONTH_REGEX, this_hour_string, MINUTE_REGEX,
                         SECOND_REGEX, PROCESSED_FILE_EXTENSION)
                glob_patterns.append(this_file_pattern)

                print (
                    'Finding files with pattern "{0:s}" (this may take a few '
                    'minutes)...'
                ).format(this_file_pattern)
                processed_file_names += glob.glob(this_file_pattern)

    if raise_error_if_missing and not len(processed_file_names):
        error_string = (
            '\n\n{0:s}\nCould not find files with any of the patterns listed '
            'above.'
        ).format(str(glob_patterns))
        raise ValueError(error_string)

    return processed_file_names, glob_patterns


def find_processed_files_one_spc_date(
        top_processed_dir_name, tracking_scale_metres2, data_source,
        spc_date_string, raise_error_if_missing=True):
    """Finds processed files with valid time in one SPC date.

    :param top_processed_dir_name: See doc for `find_processed_files_at_times`.
    :param tracking_scale_metres2: Same.
    :param data_source: Same.
    :param spc_date_string: SPC date (format "yyyymmdd").
    :param raise_error_if_missing: See doc for `find_processed_files_at_times`.
    :return: processed_file_names: 1-D list of paths to processed tracking
        files.
    :return: glob_pattern: glob pattern used to find files.
    :raises: ValueError: if no files are found and
        `raise_error_if_missing = True`.
    """

    error_checking.assert_is_string(top_processed_dir_name)
    tracking_scale_metres2 = int(numpy.round(tracking_scale_metres2))
    error_checking.assert_is_greater(tracking_scale_metres2, 0)
    tracking_utils.check_data_source(data_source)
    time_conversion.spc_date_string_to_unix_sec(spc_date_string)
    error_checking.assert_is_boolean(raise_error_if_missing)

    glob_pattern = (
        '{0:s}/{1:s}/{2:s}/scale_{3:d}m2/{4:s}_{5:s}_{6:s}{7:s}'
    ).format(top_processed_dir_name, spc_date_string[:4], spc_date_string,
             tracking_scale_metres2, PREFIX_FOR_PATHLESS_PROCESSED_FILE_NAME,
             data_source, TIME_FORMAT_AS_REGEX, PROCESSED_FILE_EXTENSION)
    processed_file_names = glob.glob(glob_pattern)

    if raise_error_if_missing and not len(processed_file_names):
        error_string = 'Could not find any files with pattern: "{0:s}"'.format(
            glob_pattern)
        raise ValueError(error_string)

    return processed_file_names, glob_pattern


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


def write_csv_file_for_amy(
        storm_object_table, probsevere_storm_object_table,
        csv_file_name):
    """Writes best-track data to CSV file for Amy.

    :param storm_object_table: See doc for `best_track_storm_object_table` in
        `storm_tracking_utils.get_original_probsevere_ids`.
    :param probsevere_storm_object_table: See doc for
        `probsevere_storm_object_table` in
        `storm_tracking_utils.get_original_probsevere_ids`.
    :param csv_file_name: Path to output file.
    """

    print (
        'Matching best-track storm objects with original probSevere objects...')
    orig_probsevere_ids = tracking_utils.get_original_probsevere_ids(
        best_track_storm_object_table=storm_object_table,
        probsevere_storm_object_table=probsevere_storm_object_table)

    argument_dict = {tracking_utils.ORIG_STORM_ID_COLUMN: orig_probsevere_ids}
    storm_object_table = storm_object_table.assign(
        **argument_dict)

    orig_num_storm_objects = len(storm_object_table.index)
    storm_object_table = storm_object_table.loc[
        storm_object_table[tracking_utils.AGE_COLUMN] != -1]
    num_storm_objects = len(storm_object_table.index)

    print (
        'Number of storm objects = {0:d} ... number of objects with valid ages '
        '= {1:d}'
    ).format(orig_num_storm_objects, num_storm_objects)

    valid_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT_IN_AMY_FILES) for
        t in storm_object_table[tracking_utils.TIME_COLUMN].values]
    cell_start_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT_IN_AMY_FILES) for
        t in storm_object_table[tracking_utils.CELL_START_TIME_COLUMN].values]
    cell_end_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT_IN_AMY_FILES) for
        t in storm_object_table[tracking_utils.CELL_END_TIME_COLUMN].values]

    south_velocities_m_s01 = -1 * storm_object_table[
        tracking_utils.NORTH_VELOCITY_COLUMN].values
    speeds_m_s01 = numpy.sqrt(
        storm_object_table[tracking_utils.EAST_VELOCITY_COLUMN].values ** 2 +
        storm_object_table[tracking_utils.NORTH_VELOCITY_COLUMN].values ** 2)

    storm_areas_km2 = numpy.full(num_storm_objects, numpy.nan)
    for i in range(num_storm_objects):
        if numpy.mod(i, 1000) == 0:
            print (
                'Have computed area for {0:d} of {1:d} storm objects...'
            ).format(i, num_storm_objects)

        this_polygon_object_metres, _ = polygons.project_latlng_to_xy(
            storm_object_table[
                tracking_utils.POLYGON_OBJECT_LATLNG_COLUMN].values[i])
        storm_areas_km2[i] = this_polygon_object_metres.area

    storm_areas_km2 = METRES2_TO_KM2 * storm_areas_km2

    columns_to_drop = [
        tracking_utils.TIME_COLUMN, tracking_utils.CELL_START_TIME_COLUMN,
        tracking_utils.CELL_END_TIME_COLUMN,
        tracking_utils.NORTH_VELOCITY_COLUMN
    ]
    storm_object_table.drop(columns_to_drop, axis=1, inplace=True)

    argument_dict = {
        VALID_TIME_COLUMN_IN_AMY_FILES: valid_time_strings,
        CELL_START_TIME_COLUMN_IN_AMY_FILES: cell_start_time_strings,
        CELL_END_TIME_COLUMN_IN_AMY_FILES: cell_end_time_strings,
        SOUTH_VELOCITY_COLUMN_IN_AMY_FILES: south_velocities_m_s01,
        SPEED_COLUMN_IN_AMY_FILES: speeds_m_s01,
        AREA_COLUMN_IN_AMY_FILES: storm_areas_km2,
    }
    storm_object_table = storm_object_table.assign(**argument_dict)

    column_dict_old_to_new = {
        tracking_utils.CENTROID_LAT_COLUMN: LATITUDE_COLUMN_IN_AMY_FILES,
        tracking_utils.CENTROID_LNG_COLUMN: LONGITUDE_COLUMN_IN_AMY_FILES,
        tracking_utils.AGE_COLUMN: AGE_COLUMN_IN_AMY_FILES,
        tracking_utils.EAST_VELOCITY_COLUMN: EAST_VELOCITY_COLUMN_IN_AMY_FILES,
        tracking_utils.ORIG_STORM_ID_COLUMN: ORIG_ID_COLUMN_IN_AMY_FILES,
        tracking_utils.STORM_ID_COLUMN: STORM_ID_COLUMN_IN_AMY_FILES
    }
    storm_object_table.rename(columns=column_dict_old_to_new, inplace=True)

    print 'Writing data to "{0:s}"...'.format(csv_file_name)
    storm_object_table.to_csv(
        csv_file_name, header=True, columns=COLUMNS_FOR_AMY, index=False)


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
    storm_object_table.cell_start_time_unix_sec: [optional] Start time of storm
        cell (first time in track).
    storm_object_table.cell_end_time_unix_sec: [optional] End time of storm cell
        (last time in track).
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

    found_best_track_flags = numpy.array(
        [c in list(storm_object_table) for c in BEST_TRACK_COLUMNS])
    if numpy.all(found_best_track_flags):
        columns_to_write += BEST_TRACK_COLUMNS

    # TODO(thunderhoser): Eventually make these columns mandatory.
    found_cell_time_flags = numpy.array(
        [c in list(storm_object_table) for c in CELL_TIME_COLUMNS])
    if numpy.all(found_cell_time_flags):
        columns_to_write += CELL_TIME_COLUMNS

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
        print 'Reading storm tracks from file: "{0:s}"...'.format(
            pickle_file_names[i])

        list_of_storm_object_tables[i] = read_processed_file(
            pickle_file_names[i])
        if i == 0:
            continue

        list_of_storm_object_tables[i], _ = (
            list_of_storm_object_tables[i].align(
                list_of_storm_object_tables[0], axis=1))

    return pandas.concat(
        list_of_storm_object_tables, axis=0, ignore_index=True)
