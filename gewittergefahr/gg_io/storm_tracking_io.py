"""IO methods for storm-tracking data (both polygons and tracks)."""

import os
import copy
import glob
import pickle
import numpy
import pandas
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

RANDOM_LEAP_YEAR = 4000

FULL_IDS_KEY = 'full_id_strings'
STORM_TIMES_KEY = 'storm_times_unix_sec'

SPC_DATE_REGEX = '[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]'
YEAR_REGEX = '[0-9][0-9][0-9][0-9]'
MONTH_REGEX = '[0-1][0-9]'
DAY_OF_MONTH_REGEX = '[0-3][0-9]'
HOUR_REGEX = '[0-2][0-9]'
MINUTE_REGEX = '[0-5][0-9]'
SECOND_REGEX = '[0-5][0-9]'

FILE_NAME_TIME_FORMAT = '%Y-%m-%d-%H%M%S'
FILE_NAME_TIME_REGEX = '{0:s}-{1:s}-{2:s}-{3:s}{4:s}{5:s}'.format(
    YEAR_REGEX, MONTH_REGEX, DAY_OF_MONTH_REGEX, HOUR_REGEX, MINUTE_REGEX,
    SECOND_REGEX
)

FILE_NAME_PREFIX = 'storm-tracking'
FILE_EXTENSION = '.p'

REQUIRED_COLUMNS = [
    tracking_utils.FULL_ID_COLUMN,
    tracking_utils.PRIMARY_ID_COLUMN, tracking_utils.SECONDARY_ID_COLUMN,
    tracking_utils.FIRST_PREV_SECONDARY_ID_COLUMN,
    tracking_utils.SECOND_PREV_SECONDARY_ID_COLUMN,
    tracking_utils.FIRST_NEXT_SECONDARY_ID_COLUMN,
    tracking_utils.SECOND_NEXT_SECONDARY_ID_COLUMN,
    tracking_utils.VALID_TIME_COLUMN, tracking_utils.SPC_DATE_COLUMN,
    tracking_utils.TRACKING_START_TIME_COLUMN,
    tracking_utils.TRACKING_END_TIME_COLUMN,
    tracking_utils.CELL_START_TIME_COLUMN, tracking_utils.CELL_END_TIME_COLUMN,
    tracking_utils.AGE_COLUMN,
    tracking_utils.CENTROID_LATITUDE_COLUMN,
    tracking_utils.CENTROID_LONGITUDE_COLUMN,
    tracking_utils.EAST_VELOCITY_COLUMN, tracking_utils.NORTH_VELOCITY_COLUMN,
    tracking_utils.LATITUDES_IN_STORM_COLUMN,
    tracking_utils.LONGITUDES_IN_STORM_COLUMN,
    tracking_utils.ROWS_IN_STORM_COLUMN, tracking_utils.COLUMNS_IN_STORM_COLUMN,
    tracking_utils.LATLNG_POLYGON_COLUMN, tracking_utils.ROWCOL_POLYGON_COLUMN
]

MAX_TIME_DIFF_KEY = 'max_time_diff_seconds'
MAX_DISTANCE_KEY = 'max_distance_metres'
SOURCE_DATASET_KEY = 'source_dataset_name'
SOURCE_TRACKING_DIR_KEY = 'source_tracking_dir_name'
TARGET_DATASET_KEY = 'target_dataset_name'
TARGET_TRACKING_DIR_KEY = 'target_tracking_dir_name'

MATCH_METADATA_KEYS = [
    MAX_TIME_DIFF_KEY, MAX_DISTANCE_KEY, SOURCE_DATASET_KEY,
    SOURCE_TRACKING_DIR_KEY, TARGET_DATASET_KEY, TARGET_TRACKING_DIR_KEY
]


def _get_previous_month(month, year):
    """Returns previous month.

    :param month: Month (integer from 1...12).
    :param year: Year (integer).
    :return: previous_month: Previous month (integer from 1...12).
    :return: previous_year: Previous year (integer).
    """

    previous_month = month - 1

    if previous_month == 0:
        previous_month = 12
        previous_year = year - 1
    else:
        previous_year = year - 0

    return previous_month, previous_year


def _get_num_days_in_month(month, year):
    """Returns number of days in month.

    :param month: Month (integer from 1...12).
    :param year: Year (integer).
    :return: num_days_in_month: Number of days in month.
    """

    month_time_string = '{0:04d}-{1:02d}'.format(year, month)
    start_of_month_unix_sec = time_conversion.string_to_unix_sec(
        month_time_string, '%Y-%m')

    _, end_of_month_unix_sec = time_conversion.first_and_last_times_in_month(
        start_of_month_unix_sec)

    last_day_of_month_string = time_conversion.unix_sec_to_string(
        end_of_month_unix_sec, '%d')

    return int(last_day_of_month_string)


def _check_file_finding_args(top_tracking_dir_name, tracking_scale_metres2,
                             source_name, raise_error_if_missing):
    """Error-checks input args for file-finding method.

    :param top_tracking_dir_name: Name of top-level directory with tracking
        data.
    :param tracking_scale_metres2: Tracking scale (minimum storm area).
    :param source_name: Data source (must be accepted by
        `storm_tracking_utils.check_data_source`).
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing = True`, the file-finding method will throw an
        error.
    :return: tracking_scale_metres2: Integer version of input.
    """

    error_checking.assert_is_string(top_tracking_dir_name)
    error_checking.assert_is_greater(tracking_scale_metres2, 0)
    tracking_utils.check_data_source(source_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    return int(numpy.round(tracking_scale_metres2))


def find_file(
        top_tracking_dir_name, tracking_scale_metres2, source_name,
        valid_time_unix_sec, spc_date_string=None, raise_error_if_missing=True):
    """Finds tracking file.

    This file should contain polygons, velocities, and other properties for one
    time step.

    :param top_tracking_dir_name: See doc for `_check_file_finding_args`.
    :param tracking_scale_metres2: Same.
    :param source_name: Same.
    :param valid_time_unix_sec: Valid time.
    :param spc_date_string: [used only if data source is ProbSevere]
        SPC date (format "yyyymmdd").
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing = True`, this method will error out.
    :return: tracking_file_name: Path to tracking file.  If file is missing and
        `raise_error_if_missing = False`, this will be the *expected* path.
    :raises: ValueError: if file is missing and `raise_error_if_missing = True`.
    """

    tracking_scale_metres2 = _check_file_finding_args(
        top_tracking_dir_name=top_tracking_dir_name,
        tracking_scale_metres2=tracking_scale_metres2, source_name=source_name,
        raise_error_if_missing=raise_error_if_missing)

    if source_name == tracking_utils.SEGMOTION_NAME:
        date_string = spc_date_string
    else:
        date_string = time_conversion.time_to_spc_date_string(
            valid_time_unix_sec)

    directory_name = '{0:s}/{1:s}/{2:s}/scale_{3:d}m2'.format(
        top_tracking_dir_name, date_string[:4], date_string,
        tracking_scale_metres2
    )

    tracking_file_name = '{0:s}/{1:s}_{2:s}_{3:s}{4:s}'.format(
        directory_name, FILE_NAME_PREFIX, source_name,
        time_conversion.unix_sec_to_string(
            valid_time_unix_sec, FILE_NAME_TIME_FORMAT),
        FILE_EXTENSION
    )

    if raise_error_if_missing and not os.path.isfile(tracking_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            tracking_file_name)
        raise ValueError(error_string)

    return tracking_file_name


def find_files_at_times(
        top_tracking_dir_name, tracking_scale_metres2, source_name, years=None,
        months=None, hours=None, raise_error_if_missing=True):
    """Finds tracking files with the given valid times.

    :param top_tracking_dir_name: See doc for `_check_file_finding_args`.
    :param tracking_scale_metres2: Same.
    :param source_name: Same.
    :param years: [may be None]
        1-D numpy array of years.
    :param months: [may be None]
        1-D numpy array of months (range 1...12).
    :param hours: [may be None]
        1-D numpy array of hours (range 0...23).
    :param raise_error_if_missing: Boolean flag.  If no files are found and
        `raise_error_if_missing = True`, this method will error out.  If no
        files are found and `raise_error_if_missing = False`, will return empty
        list.
    :return: tracking_file_names: 1-D list of paths to tracking files.
    :return: glob_patterns: 1-D list of glob patterns used to find tracking
        files.
    :raises: ValueError: if
        `years is None and months is None and hours is None`.
    :raises: ValueError: if no files are found are
        `raise_error_if_missing = True`.
    """

    tracking_scale_metres2 = _check_file_finding_args(
        top_tracking_dir_name=top_tracking_dir_name,
        tracking_scale_metres2=tracking_scale_metres2, source_name=source_name,
        raise_error_if_missing=raise_error_if_missing)

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
    tracking_file_names = []

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

                if this_year == -1:
                    if this_month == -1:
                        these_temporal_subdir_names = [
                            '{0:s}/{1:s}'.format(YEAR_REGEX, SPC_DATE_REGEX)
                        ]
                    else:
                        this_month_subdir_name = '{0:s}/{0:s}{1:s}{2:s}'.format(
                            YEAR_REGEX, this_month_string, DAY_OF_MONTH_REGEX
                        )

                        this_prev_month, _ = _get_previous_month(
                            month=this_month, year=RANDOM_LEAP_YEAR)
                        this_num_days_in_prev_month = _get_num_days_in_month(
                            month=this_prev_month, year=RANDOM_LEAP_YEAR)

                        this_prev_month_subdir_name = (
                            '{0:s}/{0:s}{1:02d}{2:02d}'
                        ).format(
                            YEAR_REGEX, this_prev_month,
                            this_num_days_in_prev_month
                        )

                        these_temporal_subdir_names = [
                            this_month_subdir_name, this_prev_month_subdir_name
                        ]

                        if this_prev_month == 2:
                            this_prev_month_subdir_name = (
                                '{0:s}/{0:s}{1:02d}{2:02d}'
                            ).format(
                                YEAR_REGEX, this_prev_month,
                                this_num_days_in_prev_month - 1
                            )

                            these_temporal_subdir_names.append(
                                this_prev_month_subdir_name)

                else:
                    if this_month == -1:
                        this_year_subdir_name = '{0:s}/{0:s}{1:s}{2:s}'.format(
                            this_year_string, MONTH_REGEX, DAY_OF_MONTH_REGEX
                        )
                        this_prev_year_subdir_name = (
                            '{0:04d}/{0:04d}1231'.format(this_year - 1)
                        )

                        these_temporal_subdir_names = [
                            this_year_subdir_name, this_prev_year_subdir_name
                        ]
                    else:
                        this_month_subdir_name = '{0:s}/{0:s}{1:s}{2:s}'.format(
                            this_year_string, this_month_string,
                            DAY_OF_MONTH_REGEX
                        )

                        this_prev_month, this_prev_year = _get_previous_month(
                            month=this_month, year=this_year)
                        this_num_days_in_prev_month = _get_num_days_in_month(
                            month=this_prev_month, year=this_prev_year)

                        this_prev_month_subdir_name = (
                            '{0:04d}/{0:04d}{1:02d}{2:02d}'
                        ).format(
                            this_prev_year, this_prev_month,
                            this_num_days_in_prev_month
                        )

                        these_temporal_subdir_names = [
                            this_month_subdir_name, this_prev_month_subdir_name
                        ]

                for this_temporal_subdir_name in these_temporal_subdir_names:
                    this_file_pattern = (
                        '{0:s}/{1:s}/scale_{2:d}m2/'
                        '{3:s}_{4:s}_{5:s}-{6:s}-{7:s}-{8:s}{9:s}{10:s}{11:s}'
                    ).format(
                        top_tracking_dir_name, this_temporal_subdir_name,
                        tracking_scale_metres2, FILE_NAME_PREFIX, source_name,
                        this_year_string, this_month_string, DAY_OF_MONTH_REGEX,
                        this_hour_string, MINUTE_REGEX, SECOND_REGEX,
                        FILE_EXTENSION
                    )

                    glob_patterns.append(this_file_pattern)

                    print((
                        'Finding files with pattern "{0:s}" (this may take a '
                        'few minutes)...'
                    ).format(this_file_pattern))

                    tracking_file_names += glob.glob(this_file_pattern)

    if raise_error_if_missing and len(tracking_file_names) == 0:
        error_string = (
            '\nCould not find files with any of the patterns listed below.'
            '\n{0:s}'
        ).format(str(glob_patterns))

        raise ValueError(error_string)

    return tracking_file_names, glob_patterns


def find_files_one_spc_date(
        top_tracking_dir_name, tracking_scale_metres2, source_name,
        spc_date_string, raise_error_if_missing=True):
    """Finds tracking files for the given SPC date.

    :param top_tracking_dir_name: See doc for `_check_file_finding_args`.
    :param tracking_scale_metres2: Same.
    :param source_name: Same.
    :param spc_date_string: SPC date (format "yyyymmdd").
    :param raise_error_if_missing: Boolean flag.  If no files are found and
        `raise_error_if_missing = True`, this method will error out.  If no
        files are found and `raise_error_if_missing = False`, will return empty
        list.
    :return: tracking_file_names: 1-D list of paths to tracking files.
    :return: glob_pattern: glob pattern used to find tracking files.
    :raises: ValueError: if no files are found are
        `raise_error_if_missing = True`.
    """

    tracking_scale_metres2 = _check_file_finding_args(
        top_tracking_dir_name=top_tracking_dir_name,
        tracking_scale_metres2=tracking_scale_metres2, source_name=source_name,
        raise_error_if_missing=raise_error_if_missing)

    time_conversion.spc_date_string_to_unix_sec(spc_date_string)

    directory_name = '{0:s}/{1:s}/{2:s}/scale_{3:d}m2'.format(
        top_tracking_dir_name, spc_date_string[:4], spc_date_string,
        tracking_scale_metres2
    )

    glob_pattern = (
        '{0:s}/{1:s}_{2:s}_{3:s}{4:s}'
    ).format(
        directory_name, FILE_NAME_PREFIX, source_name, FILE_NAME_TIME_REGEX,
        FILE_EXTENSION
    )

    tracking_file_names = glob.glob(glob_pattern)

    if raise_error_if_missing and not len(tracking_file_names):
        error_string = 'Could not find any files with pattern: "{0:s}"'.format(
            glob_pattern)
        raise ValueError(error_string)

    return tracking_file_names, glob_pattern


def file_name_to_time(tracking_file_name):
    """Parses valid time from tracking file.

    :param tracking_file_name: Path to tracking file.
    :return: valid_time_unix_sec: Valid time.
    """

    error_checking.assert_is_string(tracking_file_name)
    _, pathless_file_name = os.path.split(tracking_file_name)
    extensionless_file_name, _ = os.path.splitext(pathless_file_name)

    extensionless_file_name_parts = extensionless_file_name.split('_')

    return time_conversion.string_to_unix_sec(
        extensionless_file_name_parts[-1], FILE_NAME_TIME_FORMAT
    )


def write_file(storm_object_table, pickle_file_name):
    """Writes tracking data to Pickle file.

    P = number of vertices in polygon for a given storm object

    Note that, in addition to the required columns listed for
    `storm_object_table`, it may contain one or more distance buffers around
    each storm object.  See `tracking_utils.get_buffer_columns` for legal column
    names.

    :param storm_object_table: pandas DataFrame with the following columns.
        Each row is one storm object.
    storm_object_table.full_id_string: Full storm ID.
    storm_object_table.primary_id_string: Primary storm ID.
    storm_object_table.secondary_id_string: Secondary storm ID.
    storm_object_table.first_prev_secondary_id_string: Secondary ID of first
        predecessor ("" if no predecessors).
    storm_object_table.second_prev_secondary_id_string: Secondary ID of second
        predecessor ("" if only one predecessor).
    storm_object_table.first_next_secondary_id_string: Secondary ID of first
        successor ("" if no successors).
    storm_object_table.second_next_secondary_id_string: Secondary ID of second
        successor ("" if no successors).
    storm_object_table.valid_time_unix_sec: Valid time.
    storm_object_table.spc_date_string: SPC date (format "yyyymmdd").
    storm_object_table.tracking_start_time_unix_sec: Start of tracking period.
    storm_object_table.tracking_end_time_unix_sec: End of tracking period.
    storm_object_table.cell_start_time_unix_sec: Start time of corresponding
        storm cell.
    storm_object_table.cell_end_time_unix_sec: End time of corresponding storm
        cell.
    storm_object_table.age_seconds: Age of corresponding storm cell.
    storm_object_table.centroid_latitude_deg: Latitude (deg N) of storm-object
        centroid.
    storm_object_table.centroid_longitude_deg: Longitude (deg E) of storm-object
        centroid.
    storm_object_table.east_velocity_m_s01: Eastward velocity of corresponding
        storm cell (metres per second).
    storm_object_table.north_velocity_m_s01: Northward velocity of corresponding
        storm cell (metres per second).
    storm_object_table.grid_point_latitudes_deg: length-P numpy array with
        latitudes (deg N) of grid points in storm object.
    storm_object_table.grid_point_longitudes_deg: length-P numpy array with
        longitudes (deg E) of grid points in storm object.
    storm_object_table.grid_point_rows: length-P numpy array with row indices
        (integers) of grid points in storm object.
    storm_object_table.grid_point_columns: length-P numpy array with column
        indices (integers) of grid points in storm object.
    storm_object_table.polygon_object_latlng_deg: Instance of
        `shapely.geometry.Polygon`, with x-coords in longitude (deg E) and
        y-coords in latitude (deg N).
    storm_object_table.polygon_object_rowcol: Instance of
        `shapely.geometry.Polygon`, where x-coords are column indices, and
        y-coords are row indices, in the radar grid.

    :param pickle_file_name: Path to output file.
    """

    buffer_column_names = tracking_utils.get_buffer_columns(storm_object_table)
    if buffer_column_names is None:
        buffer_column_names = []

    columns_to_write = REQUIRED_COLUMNS + buffer_column_names

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(storm_object_table[columns_to_write], pickle_file_handle)
    pickle_file_handle.close()


def read_file(pickle_file_name):
    """Reads tracking data from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: storm_object_table: See documentation for `write_file`.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    storm_object_table = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    storm_object_table[tracking_utils.FULL_ID_COLUMN] = storm_object_table['storm_id']
    storm_object_table[tracking_utils.PRIMARY_ID_COLUMN] = storm_object_table['storm_id']
    storm_object_table[tracking_utils.SECONDARY_ID_COLUMN] = storm_object_table['storm_id']
    storm_object_table[tracking_utils.FIRST_PREV_SECONDARY_ID_COLUMN] = storm_object_table['storm_id']
    storm_object_table[tracking_utils.SECOND_PREV_SECONDARY_ID_COLUMN] = storm_object_table['storm_id']
    storm_object_table[tracking_utils.FIRST_NEXT_SECONDARY_ID_COLUMN] = storm_object_table['storm_id']
    storm_object_table[tracking_utils.SECOND_NEXT_SECONDARY_ID_COLUMN] = storm_object_table['storm_id']
    storm_object_table[tracking_utils.VALID_TIME_COLUMN] = storm_object_table['unix_time_sec']
    storm_object_table[tracking_utils.SPC_DATE_COLUMN] = [time_conversion.time_to_spc_date_string(t) for t in storm_object_table['unix_time_sec'].values]
    storm_object_table[tracking_utils.CELL_START_TIME_COLUMN] = storm_object_table['tracking_start_time_unix_sec']
    storm_object_table[tracking_utils.CELL_END_TIME_COLUMN] = storm_object_table['tracking_end_time_unix_sec']
    storm_object_table[tracking_utils.AGE_COLUMN] = storm_object_table['age_sec']
    storm_object_table[tracking_utils.CENTROID_LATITUDE_COLUMN] = storm_object_table['centroid_lat_deg']
    storm_object_table[tracking_utils.CENTROID_LONGITUDE_COLUMN] = storm_object_table['centroid_lng_deg']
    storm_object_table[tracking_utils.EAST_VELOCITY_COLUMN] = storm_object_table['east_velocity_m_s01']
    storm_object_table[tracking_utils.NORTH_VELOCITY_COLUMN] = storm_object_table['north_velocity_m_s01']
    storm_object_table[tracking_utils.LATITUDES_IN_STORM_COLUMN] = storm_object_table['grid_point_latitudes_deg']
    storm_object_table[tracking_utils.LONGITUDES_IN_STORM_COLUMN] = storm_object_table['grid_point_longitudes_deg']
    storm_object_table[tracking_utils.ROWS_IN_STORM_COLUMN] = storm_object_table['grid_point_rows']
    storm_object_table[tracking_utils.COLUMNS_IN_STORM_COLUMN] = storm_object_table['grid_point_columns']
    storm_object_table[tracking_utils.LATLNG_POLYGON_COLUMN] = storm_object_table['polygon_object_latlng']
    storm_object_table[tracking_utils.ROWCOL_POLYGON_COLUMN] = storm_object_table['polygon_object_rowcol']

    error_checking.assert_columns_in_dataframe(
        storm_object_table, REQUIRED_COLUMNS)

    return storm_object_table


def read_many_files(pickle_file_names):
    """Reads tracking data from many Pickle files.

    This method will concatenate all storm objects into the same table.

    :param pickle_file_names: 1-D list of paths to input files.
    :return: storm_object_table: See documentation for `write_file`.
    """

    error_checking.assert_is_string_list(pickle_file_names)
    error_checking.assert_is_numpy_array(
        numpy.array(pickle_file_names), num_dimensions=1
    )

    num_files = len(pickle_file_names)
    list_of_storm_object_tables = [None] * num_files

    for i in range(num_files):
        print('Reading data from file: "{0:s}"...'.format(pickle_file_names[i]))
        list_of_storm_object_tables[i] = read_file(pickle_file_names[i])

        if i == 0:
            continue

        list_of_storm_object_tables[i] = list_of_storm_object_tables[i].align(
            list_of_storm_object_tables[0], axis=1
        )[0]

    return pandas.concat(list_of_storm_object_tables, axis=0, ignore_index=True)


def write_ids_and_times(full_id_strings, storm_times_unix_sec,
                        pickle_file_name):
    """Writes full storm IDs and valid times (minimal metadata) to Pickle file.

    N = number of storm objects

    :param full_id_strings: length-N list of full IDs.
    :param storm_times_unix_sec: length-N numpy array of valid times.
    :param pickle_file_name: Path to output file.
    """

    error_checking.assert_is_string_list(full_id_strings)
    error_checking.assert_is_numpy_array(
        numpy.array(full_id_strings), num_dimensions=1)
    num_storm_objects = len(full_id_strings)

    error_checking.assert_is_integer_numpy_array(storm_times_unix_sec)
    error_checking.assert_is_numpy_array(
        storm_times_unix_sec,
        exact_dimensions=numpy.array([num_storm_objects], dtype=int)
    )

    metadata_dict = {
        FULL_IDS_KEY: full_id_strings,
        STORM_TIMES_KEY: storm_times_unix_sec,
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(metadata_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_ids_and_times(pickle_file_name):
    """Reads storm IDs and times (minimal metadata) from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: full_id_strings: See doc for `write_ids_and_times`.
    :return: storm_times_unix_sec: Same.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    while True:
        storm_metadata_dict = pickle.load(pickle_file_handle)
        if isinstance(storm_metadata_dict, dict):
            break

    pickle_file_handle.close()
    if not isinstance(storm_metadata_dict, dict):
        raise ValueError('Cannot find dictionary in file.')

    return (storm_metadata_dict[FULL_IDS_KEY],
            storm_metadata_dict[STORM_TIMES_KEY])


def find_match_file(top_directory_name, valid_time_unix_sec,
                    raise_error_if_missing=False):
    """Finds match file.

    A "match file" matches storm objects in one dataset (e.g., MYRORSS or
    GridRad) to those in another dataset, at one time step.

    :param top_directory_name: Name of top-level directory.
    :param valid_time_unix_sec: Valid time.
    :param raise_error_if_missing: See doc for `find_file`.
    :return: match_file_name: Path to match file.  If file is missing and
        `raise_error_if_missing = False`, this will be the *expected* path.
    :raises: ValueError: if file is missing and `raise_error_if_missing = True`.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    spc_date_string = time_conversion.time_to_spc_date_string(
        valid_time_unix_sec)

    match_file_name = '{0:s}/{1:s}/{2:s}/storm-matches_{3:s}.p'.format(
        top_directory_name, spc_date_string[:4], spc_date_string,
        time_conversion.unix_sec_to_string(
            valid_time_unix_sec, FILE_NAME_TIME_FORMAT)
    )

    if raise_error_if_missing and not os.path.isfile(match_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            match_file_name)
        raise ValueError(error_string)

    return match_file_name


def write_matches(
        pickle_file_name, source_to_target_dict, max_time_diff_seconds,
        max_distance_metres, source_dataset_name, source_tracking_dir_name,
        target_dataset_name, target_tracking_dir_name):
    """Writes matches to Pickle file.

    "Matches" are between storm storm objects in one dataset (e.g., MYRORSS or
    GridRad) and those in another dataset, at one time step.

    :param pickle_file_name: Path to output file.
    :param source_to_target_dict: Dictionary, where each key is a tuple with
        (source ID, source time) and each value is a list with [target ID,
        target time].  The IDs are strings, and the times are Unix seconds
        (integers).  For source objects with no match in the target dataset, the
        corresponding value is None (rather than a list).
    :param max_time_diff_seconds: Max time difference between matched storm
        objects.
    :param max_distance_metres: Max distance between matched storm objects.
    :param source_dataset_name: Name of source dataset (must be accepted by
        `radar_utils.check_data_source`).
    :param source_tracking_dir_name: Name of top-level directory with tracks for
        source dataset.
    :param target_dataset_name: Same but for target dataset.
    :param target_tracking_dir_name: Same but for target dataset.
    """

    error_checking.assert_is_string(pickle_file_name)
    error_checking.assert_is_integer(max_time_diff_seconds)
    error_checking.assert_is_geq(max_time_diff_seconds, 0)
    error_checking.assert_is_greater(max_distance_metres, 0.)
    radar_utils.check_data_source(source_dataset_name)
    error_checking.assert_is_string(source_tracking_dir_name)
    radar_utils.check_data_source(target_dataset_name)
    error_checking.assert_is_string(target_tracking_dir_name)

    metadata_dict = {
        MAX_TIME_DIFF_KEY: max_time_diff_seconds,
        MAX_DISTANCE_KEY: max_distance_metres,
        SOURCE_DATASET_KEY: source_dataset_name,
        SOURCE_TRACKING_DIR_KEY: source_tracking_dir_name,
        TARGET_DATASET_KEY: target_dataset_name,
        TARGET_TRACKING_DIR_KEY: target_tracking_dir_name
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(source_to_target_dict, pickle_file_handle)
    pickle.dump(metadata_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_matches(pickle_file_name):
    """Reads matches from Pickle file.

    :param pickle_file_name: Path to input file (created by `write_matches`).
    :return: source_to_target_dict: See doc for `write_matches`.
    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict["max_time_diff_seconds"]: See doc for `write_matches`.
    metadata_dict["max_distance_metres"]: Same.
    metadata_dict["source_dataset_name"]: Same.
    metadata_dict["source_tracking_dir_name"]: Same.
    metadata_dict["target_dataset_name"]: Same.
    metadata_dict["target_tracking_dir_name"]: Same.

    :raises: ValueError: if `metadata_dict` does not contain expected keys.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    source_to_target_dict = pickle.load(pickle_file_handle)
    metadata_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    missing_keys = list(
        set(MATCH_METADATA_KEYS) - set(metadata_dict.keys())
    )

    if len(missing_keys) == 0:
        return source_to_target_dict, metadata_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in metadata '
        'in file "{1:s}".'
    ).format(str(missing_keys), pickle_file_name)

    raise ValueError(error_string)
