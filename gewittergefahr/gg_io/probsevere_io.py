"""IO methods for probSevere storm-tracking."""

import glob
import json
import os.path
import numpy
import pandas
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import geodetic_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import error_checking

# TODO(thunderhoser): Add code to transfer raw files from NSSL to local machine.

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

DAYS_TO_SECONDS = 86400
DUMMY_TRACKING_START_TIME_UNIX_SEC = 0
DUMMY_TRACKING_END_TIME_UNIX_SEC = int(3e9)
DUMMY_TRACKING_SCALE_METRES2 = int(numpy.round(numpy.pi * 1e8))

RAW_FILE_NAME_PREFIX = 'SSEC_AWIPS_PROBSEVERE'
ALT_RAW_FILE_NAME_PREFIX = 'SSEC_AWIPS_CONVECTPROB'
JSON_FILE_EXTENSION = '.json'
ASCII_FILE_EXTENSION = '.ascii'
VALID_RAW_FILE_EXTENSIONS = [JSON_FILE_EXTENSION, ASCII_FILE_EXTENSION]

MONTH_FORMAT = '%Y%m'
DATE_FORMAT = '%Y%m%d'
RAW_FILE_TIME_FORMAT = '%Y%m%d_%H%M%S'
RAW_FILE_TIME_FORMAT_REGEX = (
    '[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]_[0-2][0-9][0-5][0-9][0-5][0-9]'
)

TIME_KEY_IN_JSON_FILES = 'validTime'
FEATURES_KEY_IN_JSON_FILES = 'features'
GEOMETRY_KEY_IN_JSON_FILES = 'geometry'
COORDINATES_KEY_IN_JSON_FILES = 'coordinates'
PROPERTIES_KEY_IN_JSON_FILES = 'properties'

STORM_ID_KEY_IN_JSON_FILES = 'ID'
U_MOTION_KEY_IN_JSON_FILES = 'MOTION_EAST'
V_MOTION_KEY_IN_JSON_FILES = 'MOTION_SOUTH'
LATITUDE_INDEX_IN_JSON_FILES = 1
LONGITUDE_INDEX_IN_JSON_FILES = 0

POLYGON_INDEX_IN_ASCII_FILES = 7
STORM_ID_INDEX_IN_ASCII_FILES = 8
U_MOTION_INDEX_IN_ASCII_FILES = 9
V_MOTION_INDEX_IN_ASCII_FILES = 10
MIN_WORDS_PER_ASCII_LINE = 11
LATITUDE_INDEX_IN_ASCII_FILES = 0
LONGITUDE_INDEX_IN_ASCII_FILES = 1

NW_GRID_POINT_LAT_DEG = 55.
NW_GRID_POINT_LNG_DEG = 230.
GRID_LAT_SPACING_DEG = 0.01
GRID_LNG_SPACING_DEG = 0.01
NUM_GRID_ROWS = 3501
NUM_GRID_COLUMNS = 7001

DATE_INDEX_KEY = 'date_index'


def _check_raw_file_extension(file_extension):
    """Error-checks extension for raw file.

    :param file_extension: File extension (".json" or ".ascii").
    :raises: ValueError: if `file_extension not in VALID_RAW_FILE_EXTENSIONS`.
    """

    error_checking.assert_is_string(file_extension)

    if file_extension not in VALID_RAW_FILE_EXTENSIONS:
        error_string = (
            '\n{0:s}\nValid extensions for raw files (listed above) do not '
            'include "{1:s}".'
        ).format(str(VALID_RAW_FILE_EXTENSIONS), file_extension)

        raise ValueError(error_string)


def _get_pathless_raw_file_name(unix_time_sec, file_extension):
    """Returns pathless name for raw (either ASCII or JSON) file.

    :param unix_time_sec: Valid time.
    :param file_extension: File extension (".json" or ".ascii").
    :return: pathless_file_name: Expected pathless file name.
    """

    return '{0:s}_{1:s}{2:s}'.format(
        RAW_FILE_NAME_PREFIX,
        time_conversion.unix_sec_to_string(
            unix_time_sec, RAW_FILE_TIME_FORMAT),
        file_extension)


def _find_io_files_for_renaming(
        top_input_dir_name, first_date_unix_sec, last_date_unix_sec,
        top_output_dir_name):
    """Finds input and output files for renaming storms.

    N = number of dates

    :param top_input_dir_name: See documentation for `rename_storms.`
    :param first_date_unix_sec: Same.
    :param last_date_unix_sec: Same.
    :param top_output_dir_name: Same.
    :return: input_file_names_by_date: length-N list, where the [i]th item is a
        numpy array of paths to input files for the [i]th date.
    :return: output_file_names_by_date: Same as above, but for output files.
    :return: valid_times_by_date_unix_sec: Same as above, but for valid times.
        All 3 arrays for the [i]th date have the same length.
    """

    dates_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_date_unix_sec,
        end_time_unix_sec=last_date_unix_sec, time_interval_sec=DAYS_TO_SECONDS,
        include_endpoint=True)

    date_strings = [
        time_conversion.unix_sec_to_string(t, DATE_FORMAT)
        for t in dates_unix_sec
    ]

    num_dates = len(date_strings)
    input_file_names_by_date = [numpy.array([], dtype=object)] * num_dates
    output_file_names_by_date = [numpy.array([], dtype=object)] * num_dates
    valid_times_by_date_unix_sec = [numpy.array([], dtype=int)] * num_dates

    for i in range(num_dates):
        print('Finding input files for date {0:s}...'.format(date_strings[i]))

        these_input_file_names = tracking_io.find_files_one_spc_date(
            spc_date_string=date_strings[i],
            source_name=tracking_utils.PROBSEVERE_NAME,
            top_tracking_dir_name=top_input_dir_name,
            tracking_scale_metres2=DUMMY_TRACKING_SCALE_METRES2,
            raise_error_if_missing=True
        )[0]

        these_input_file_names.sort()
        these_valid_times_unix_sec = numpy.array([
            tracking_io.file_name_to_time(f) for f in these_input_file_names
        ], dtype=int)

        these_output_file_names = []
        for t in these_valid_times_unix_sec:
            these_output_file_names.append(tracking_io.find_file(
                valid_time_unix_sec=t,
                source_name=tracking_utils.PROBSEVERE_NAME,
                top_tracking_dir_name=top_output_dir_name,
                tracking_scale_metres2=DUMMY_TRACKING_SCALE_METRES2,
                raise_error_if_missing=False
            ))

        input_file_names_by_date[i] = numpy.array(
            these_input_file_names, dtype=object)
        output_file_names_by_date[i] = numpy.array(
            these_output_file_names, dtype=object)
        valid_times_by_date_unix_sec[i] = these_valid_times_unix_sec

    print(SEPARATOR_STRING)

    return (input_file_names_by_date, output_file_names_by_date,
            valid_times_by_date_unix_sec)


def _get_dates_needed_for_renaming_storms(
        working_date_index, num_dates_in_period):
    """Returns dates needed for renaming storms.

    For more details on renaming storms, see the public method `rename_storms`.

    :param working_date_index: Index of working date.
    :param num_dates_in_period: Number of dates in period.
    :return: date_needed_indices: 1-D numpy array with indices of dates needed.
    """

    date_needed_indices = [working_date_index]
    if working_date_index != 0:
        date_needed_indices.append(working_date_index - 1)
    if working_date_index != num_dates_in_period - 1:
        date_needed_indices.append(working_date_index + 1)

    return numpy.sort(numpy.array(date_needed_indices, dtype=int))


def _shuffle_io_for_renaming(
        input_file_names_by_date, output_file_names_by_date,
        valid_times_by_date_unix_sec, storm_object_table_by_date,
        working_date_index):
    """Shuffles data into and out of memory for renaming storms.

    For more on renaming storms, see doc for `rename_storms`.

    N = number of dates

    :param input_file_names_by_date: length-N list created by
        `_find_io_files_for_renaming`.
    :param output_file_names_by_date: Same.
    :param valid_times_by_date_unix_sec: Same.
    :param storm_object_table_by_date: length-N list, where the [i]th element is
        a pandas DataFrame with tracking data for the [i]th date.  At any given
        time, all but 3 items should be None.  Each table has columns documented
        in `storm_tracking_io.write_file`, plus the following column.
    storm_object_table_by_date.date_index: Array index.  If date_index[i] = j,
        the [i]th row (storm object) comes from the [j]th date.
    :param working_date_index: Index of date currently being processed.  Only
        dates (working_date_index - 1)...(working_date_index + 1) need to be in
        memory.  If None, this method will write/clear all data currently in
        memory, without reading new data.
    :return: storm_object_table_by_date: Same as input, except that different
        items are filled and different items are None.
    """

    num_dates = len(input_file_names_by_date)

    if working_date_index is None:
        date_needed_indices = numpy.array([-1], dtype=int)
    else:
        date_needed_indices = _get_dates_needed_for_renaming_storms(
            working_date_index=working_date_index,
            num_dates_in_period=num_dates)

    for j in range(num_dates):
        if j in date_needed_indices and storm_object_table_by_date[j] is None:
            storm_object_table_by_date[j] = tracking_io.read_many_files(
                input_file_names_by_date[j].tolist()
            )
            print(SEPARATOR_STRING)

            this_num_storm_objects = len(storm_object_table_by_date[j].index)
            these_date_indices = numpy.full(
                this_num_storm_objects, j, dtype=int)

            storm_object_table_by_date[j] = (
                storm_object_table_by_date[j].assign(
                    **{DATE_INDEX_KEY: these_date_indices}
                )
            )

        if j not in date_needed_indices:
            if storm_object_table_by_date[j] is not None:
                these_output_file_names = output_file_names_by_date[j]
                these_valid_times_unix_sec = valid_times_by_date_unix_sec[j]

                for k in range(len(these_valid_times_unix_sec)):
                    print('Writing new data to "{0:s}"...'.format(
                        these_output_file_names[k]
                    ))

                    these_indices = numpy.where(
                        storm_object_table_by_date[j][
                            tracking_utils.VALID_TIME_COLUMN].values ==
                        these_valid_times_unix_sec[k]
                    )[0]

                    tracking_io.write_file(
                        storm_object_table=storm_object_table_by_date[j].iloc[
                            these_indices],
                        pickle_file_name=these_output_file_names[k]
                    )

                print(SEPARATOR_STRING)

            storm_object_table_by_date[j] = None

    if working_date_index is None:
        return storm_object_table_by_date

    for j in date_needed_indices[1:]:
        storm_object_table_by_date[j], _ = storm_object_table_by_date[j].align(
            storm_object_table_by_date[j - 1], axis=1)

    return storm_object_table_by_date


def _rename_storms_one_table(
        storm_object_table, next_id_number, max_dropout_time_seconds,
        working_date_index):
    """Renames storms in one table.

    For more details on renaming storms, see the public method `rename_storms`.

    :param storm_object_table: pandas DataFrame with columns documented in
        `storm_tracking_io.write_file`.
    :param next_id_number: Will start with this ID.
    :param max_dropout_time_seconds: See documentation for `rename_storms`.
    :param working_date_index: Index of working date.
    :return: storm_object_table: Same as input, but maybe with new storm IDs.
    :return: next_id_number: Same as input, but maybe incremented.
    """

    working_object_indices = numpy.where(
        storm_object_table[DATE_INDEX_KEY].values == working_date_index
    )[0]

    object_id_strings = storm_object_table[
        tracking_utils.PRIMARY_ID_COLUMN
    ].values[working_object_indices]

    cell_id_strings = numpy.unique(object_id_strings).tolist()
    num_storm_cells = len(cell_id_strings)

    for j in range(num_storm_cells):
        if numpy.mod(j, 1000) == 0:
            print((
                'Have renamed {0:d} of {1:d} original storm cells (next ID = '
                '{2:d})...'
            ).format(j, num_storm_cells, next_id_number))

        these_object_indices = numpy.where(
            storm_object_table[tracking_utils.PRIMARY_ID_COLUMN].values ==
            cell_id_strings[j]
        )[0]

        these_valid_times_unix_sec = storm_object_table[
            tracking_utils.VALID_TIME_COLUMN
        ].values[these_object_indices]

        these_sort_indices = numpy.argsort(these_valid_times_unix_sec)
        these_object_indices = these_object_indices[these_sort_indices]
        these_valid_times_unix_sec = these_valid_times_unix_sec[
            these_sort_indices]

        these_id_strings, next_id_number = _rename_storms_one_original_id(
            valid_times_unix_sec=these_valid_times_unix_sec,
            next_id_number=next_id_number,
            max_dropout_time_seconds=max_dropout_time_seconds)

        storm_object_table[tracking_utils.PRIMARY_ID_COLUMN].values[
            these_object_indices
        ] = numpy.array(these_id_strings)

    print(SEPARATOR_STRING)
    return storm_object_table, next_id_number


def _rename_storms_one_original_id(
        valid_times_unix_sec, next_id_number, max_dropout_time_seconds):
    """Renames storm objects with the same original ID.

    N = number of storm objects

    :param valid_times_unix_sec: length-N numpy array of valid times.
    :param next_id_number: Will start with this ID.
    :param max_dropout_time_seconds: See documentation for `rename_storms`.
    :return: primary_id_strings: length-N list of new primary IDs.
    :return: next_id_number: Same as input, but maybe incremented.
    """

    time_differences_sec = numpy.diff(valid_times_unix_sec)
    dropout_indices = numpy.where(
        time_differences_sec > max_dropout_time_seconds
    )[0]

    num_storm_objects = len(valid_times_unix_sec)
    storm_cell_start_indices = numpy.concatenate((
        numpy.array([0]), dropout_indices + 1
    )).astype(int)
    storm_cell_end_indices = numpy.concatenate((
        dropout_indices, numpy.array([num_storm_objects - 1])
    )).astype(int)

    num_storm_cells = len(storm_cell_start_indices)
    primary_id_strings = [''] * num_storm_objects

    for j in range(num_storm_cells):
        these_object_indices = numpy.linspace(
            storm_cell_start_indices[j], storm_cell_end_indices[j],
            num=storm_cell_end_indices[j] - storm_cell_start_indices[j] + 1,
            dtype=int)

        for k in these_object_indices:
            primary_id_strings[k] = '{0:d}probSevere'.format(next_id_number)

        next_id_number += 1

    return primary_id_strings, next_id_number


def get_json_file_name_on_ftp(unix_time_sec, ftp_directory_name):
    """Returns name of JSON file on FTP server.

    :param unix_time_sec: Valid time.
    :param ftp_directory_name: Name of FTP directory with JSON files.
    :return: raw_ftp_file_name: Expected path to JSON file on FTP server.
    """

    return '{0:s}/{1:s}'.format(
        ftp_directory_name,
        _get_pathless_raw_file_name(
            unix_time_sec=unix_time_sec, file_extension=JSON_FILE_EXTENSION)
    )


def find_raw_file(top_directory_name, unix_time_sec, file_extension,
                  raise_error_if_missing=True):
    """Finds raw (either ASCII or JSON) file with probSevere data.

    This file should contain all storm-tracking data for one time step.

    :param top_directory_name: Name of top-level directory with raw probSevere
        files.
    :param unix_time_sec: Valid time.
    :param file_extension: File type (either ".json" or ".ascii").
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        raise_error_if_missing = True, this method will error out.  If file is
        missing and raise_error_if_missing = False, will return expected path to
        raw file.
    :return: raw_file_name: Path or expected path to raw file.
    :raises: ValueError: if file is missing and raise_error_if_missing = True.
    """

    error_checking.assert_is_string(top_directory_name)
    _check_raw_file_extension(file_extension)
    error_checking.assert_is_boolean(raise_error_if_missing)

    pathless_file_name = _get_pathless_raw_file_name(
        unix_time_sec=unix_time_sec, file_extension=file_extension)

    raw_file_name = '{0:s}/{1:s}/{2:s}/{3:s}'.format(
        top_directory_name,
        time_conversion.unix_sec_to_string(unix_time_sec, MONTH_FORMAT),
        time_conversion.unix_sec_to_string(unix_time_sec, DATE_FORMAT),
        pathless_file_name)

    if os.path.isfile(raw_file_name) or not raise_error_if_missing:
        return raw_file_name

    pathless_file_name = pathless_file_name.replace(
        RAW_FILE_NAME_PREFIX, ALT_RAW_FILE_NAME_PREFIX)

    alt_raw_file_name = '{0:s}/{1:s}/{2:s}/{3:s}'.format(
        top_directory_name,
        time_conversion.unix_sec_to_string(unix_time_sec, MONTH_FORMAT),
        time_conversion.unix_sec_to_string(unix_time_sec, DATE_FORMAT),
        pathless_file_name)

    if os.path.isfile(alt_raw_file_name):
        return alt_raw_file_name

    error_string = (
        'Cannot find raw probSevere file.  Expected at: {0:s}'
    ).format(raw_file_name)

    raise ValueError(error_string)


def find_raw_files_one_day(top_directory_name, unix_time_sec, file_extension,
                           raise_error_if_all_missing=True):
    """Finds all raw (ASCII or JSON) files for one day.

    :param top_directory_name: Name of top-level directory with raw probSevere
        files.
    :param unix_time_sec: Valid time (any time on the given day).
    :param file_extension: File type (either ".json" or ".ascii").
    :param raise_error_if_all_missing: Boolean flag.  If no files are found and
        raise_error_if_all_missing = True, this method will error out.  If no
        files are found and raise_error_if_all_missing = False, will return
        None.
    :return: raw_file_names: [may be None] 1-D list of paths to raw files.
    :raises: ValueError: if no files are found and raise_error_if_all_missing =
        True.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_boolean(raise_error_if_all_missing)

    dummy_pathless_file_name = _get_pathless_raw_file_name(
        unix_time_sec=unix_time_sec, file_extension=file_extension)
    time_string = time_conversion.unix_sec_to_string(
        unix_time_sec, RAW_FILE_TIME_FORMAT)
    pathless_file_name_pattern = dummy_pathless_file_name.replace(
        time_string, RAW_FILE_TIME_FORMAT_REGEX)

    raw_file_pattern = '{0:s}/{1:s}/{2:s}/{3:s}'.format(
        top_directory_name,
        time_conversion.unix_sec_to_string(unix_time_sec, MONTH_FORMAT),
        time_conversion.unix_sec_to_string(unix_time_sec, DATE_FORMAT),
        pathless_file_name_pattern)

    raw_file_names = glob.glob(raw_file_pattern)
    if len(raw_file_names):
        return raw_file_names

    pathless_file_name_pattern = pathless_file_name_pattern.replace(
        RAW_FILE_NAME_PREFIX, ALT_RAW_FILE_NAME_PREFIX)

    raw_file_pattern = '{0:s}/{1:s}/{2:s}/{3:s}'.format(
        top_directory_name,
        time_conversion.unix_sec_to_string(unix_time_sec, MONTH_FORMAT),
        time_conversion.unix_sec_to_string(unix_time_sec, DATE_FORMAT),
        pathless_file_name_pattern)

    raw_file_names = glob.glob(raw_file_pattern)

    if len(raw_file_names):
        return raw_file_names
    if not raise_error_if_all_missing:
        return None

    error_string = 'Cannot find any files with pattern: "{0:s}"'.format(
        raw_file_pattern)
    raise ValueError(error_string)


def raw_file_name_to_time(raw_file_name):
    """Parses valid time from name of raw (either ASCII or JSON) file.

    :param raw_file_name: Path to raw file.
    :return: unix_time_sec: Valid time.
    """

    error_checking.assert_is_string(raw_file_name)
    _, pathless_file_name = os.path.split(raw_file_name)
    extensionless_file_name, _ = os.path.splitext(pathless_file_name)

    time_string = extensionless_file_name.replace(
        RAW_FILE_NAME_PREFIX + '_', '')
    time_string = time_string.replace(ALT_RAW_FILE_NAME_PREFIX + '_', '')

    return time_conversion.string_to_unix_sec(time_string, RAW_FILE_TIME_FORMAT)


def read_raw_file(raw_file_name):
    """Reads tracking data from raw (either JSON or ASCII) file.

    This file should contain all storm objects at one time step.

    :param raw_file_name: Path to input file.
    :return: storm_object_table: See documentation for
        `storm_tracking_io.write_file`.
    """

    error_checking.assert_file_exists(raw_file_name)
    _, pathless_file_name = os.path.split(raw_file_name)
    _, file_extension = os.path.splitext(pathless_file_name)
    _check_raw_file_extension(file_extension)

    unix_time_sec = raw_file_name_to_time(raw_file_name)

    if file_extension == ASCII_FILE_EXTENSION:
        primary_id_strings = []
        east_velocities_m_s01 = []
        north_velocities_m_s01 = []
        list_of_latitude_vertex_arrays_deg = []
        list_of_longitude_vertex_arrays_deg = []

        for this_line in open(raw_file_name, 'r').readlines():
            these_words = this_line.split(':')
            if len(these_words) < MIN_WORDS_PER_ASCII_LINE:
                continue

            primary_id_strings.append(
                these_words[STORM_ID_INDEX_IN_ASCII_FILES]
            )
            east_velocities_m_s01.append(
                float(these_words[U_MOTION_INDEX_IN_ASCII_FILES])
            )
            north_velocities_m_s01.append(
                -1 * float(these_words[V_MOTION_INDEX_IN_ASCII_FILES])
            )

            these_polygon_words = numpy.array(
                these_words[POLYGON_INDEX_IN_ASCII_FILES].split(',')
            )
            these_latitude_words = these_polygon_words[
                LATITUDE_INDEX_IN_ASCII_FILES::2
            ].tolist()
            these_longitude_words = these_polygon_words[
                LONGITUDE_INDEX_IN_ASCII_FILES::2
            ].tolist()

            these_latitudes_deg = numpy.array([
                float(w) for w in these_latitude_words
            ])
            these_longitudes_deg = numpy.array([
                float(w) for w in these_longitude_words
            ])
            list_of_latitude_vertex_arrays_deg.append(these_latitudes_deg)
            list_of_longitude_vertex_arrays_deg.append(these_longitudes_deg)

        east_velocities_m_s01 = numpy.array(east_velocities_m_s01)
        north_velocities_m_s01 = numpy.array(north_velocities_m_s01)
        num_storms = len(primary_id_strings)

    else:
        with open(raw_file_name) as json_file_handle:
            probsevere_dict = json.load(json_file_handle)

        num_storms = len(probsevere_dict[FEATURES_KEY_IN_JSON_FILES])
        primary_id_strings = [None] * num_storms
        east_velocities_m_s01 = numpy.full(num_storms, numpy.nan)
        north_velocities_m_s01 = numpy.full(num_storms, numpy.nan)
        list_of_latitude_vertex_arrays_deg = [None] * num_storms
        list_of_longitude_vertex_arrays_deg = [None] * num_storms

        for i in range(num_storms):
            primary_id_strings[i] = str(
                probsevere_dict[FEATURES_KEY_IN_JSON_FILES][i][
                    PROPERTIES_KEY_IN_JSON_FILES][STORM_ID_KEY_IN_JSON_FILES]
            )
            east_velocities_m_s01[i] = float(
                probsevere_dict[FEATURES_KEY_IN_JSON_FILES][i][
                    PROPERTIES_KEY_IN_JSON_FILES][U_MOTION_KEY_IN_JSON_FILES]
            )
            north_velocities_m_s01[i] = -1 * float(
                probsevere_dict[FEATURES_KEY_IN_JSON_FILES][i][
                    PROPERTIES_KEY_IN_JSON_FILES][V_MOTION_KEY_IN_JSON_FILES]
            )

            this_vertex_matrix_deg = numpy.array(
                probsevere_dict[FEATURES_KEY_IN_JSON_FILES][i][
                    GEOMETRY_KEY_IN_JSON_FILES
                ][COORDINATES_KEY_IN_JSON_FILES][0]
            )
            list_of_latitude_vertex_arrays_deg[i] = numpy.array(
                this_vertex_matrix_deg[:, LATITUDE_INDEX_IN_JSON_FILES]
            )
            list_of_longitude_vertex_arrays_deg[i] = numpy.array(
                this_vertex_matrix_deg[:, LONGITUDE_INDEX_IN_JSON_FILES]
            )

    unix_times_sec = numpy.full(num_storms, unix_time_sec, dtype=int)
    spc_date_strings = num_storms * [
        time_conversion.time_to_spc_date_string(unix_time_sec)
    ]

    tracking_start_times_unix_sec = numpy.full(
        num_storms, DUMMY_TRACKING_START_TIME_UNIX_SEC, dtype=int)
    tracking_end_times_unix_sec = numpy.full(
        num_storms, DUMMY_TRACKING_END_TIME_UNIX_SEC, dtype=int)

    storm_object_dict = {
        tracking_utils.PRIMARY_ID_COLUMN: primary_id_strings,
        tracking_utils.EAST_VELOCITY_COLUMN: east_velocities_m_s01,
        tracking_utils.NORTH_VELOCITY_COLUMN: north_velocities_m_s01,
        tracking_utils.VALID_TIME_COLUMN: unix_times_sec,
        tracking_utils.SPC_DATE_COLUMN: spc_date_strings,
        tracking_utils.TRACKING_START_TIME_COLUMN:
            tracking_start_times_unix_sec,
        tracking_utils.TRACKING_END_TIME_COLUMN: tracking_end_times_unix_sec
    }

    storm_object_table = pandas.DataFrame.from_dict(storm_object_dict)

    storm_ages_sec = numpy.full(num_storms, numpy.nan)
    simple_array = numpy.full(num_storms, numpy.nan)
    object_array = numpy.full(num_storms, numpy.nan, dtype=object)
    nested_array = storm_object_table[[
        tracking_utils.PRIMARY_ID_COLUMN, tracking_utils.PRIMARY_ID_COLUMN
    ]].values.tolist()

    argument_dict = {
        tracking_utils.AGE_COLUMN: storm_ages_sec,
        tracking_utils.CENTROID_LATITUDE_COLUMN: simple_array,
        tracking_utils.CENTROID_LONGITUDE_COLUMN: simple_array,
        tracking_utils.LATITUDES_IN_STORM_COLUMN: nested_array,
        tracking_utils.LONGITUDES_IN_STORM_COLUMN: nested_array,
        tracking_utils.ROWS_IN_STORM_COLUMN: nested_array,
        tracking_utils.COLUMNS_IN_STORM_COLUMN: nested_array,
        tracking_utils.LATLNG_POLYGON_COLUMN: object_array,
        tracking_utils.ROWCOL_POLYGON_COLUMN: object_array
    }
    storm_object_table = storm_object_table.assign(**argument_dict)

    for i in range(num_storms):
        these_vertex_rows, these_vertex_columns = (
            radar_utils.latlng_to_rowcol(
                latitudes_deg=list_of_latitude_vertex_arrays_deg[i],
                longitudes_deg=list_of_longitude_vertex_arrays_deg[i],
                nw_grid_point_lat_deg=NW_GRID_POINT_LAT_DEG,
                nw_grid_point_lng_deg=NW_GRID_POINT_LNG_DEG,
                lat_spacing_deg=GRID_LAT_SPACING_DEG,
                lng_spacing_deg=GRID_LNG_SPACING_DEG)
        )

        these_vertex_rows, these_vertex_columns = (
            polygons.fix_probsevere_vertices(
                row_indices_orig=these_vertex_rows,
                column_indices_orig=these_vertex_columns)
        )

        these_vertex_latitudes_deg, these_vertex_longitudes_deg = (
            radar_utils.rowcol_to_latlng(
                grid_rows=these_vertex_rows, grid_columns=these_vertex_columns,
                nw_grid_point_lat_deg=NW_GRID_POINT_LAT_DEG,
                nw_grid_point_lng_deg=NW_GRID_POINT_LNG_DEG,
                lat_spacing_deg=GRID_LAT_SPACING_DEG,
                lng_spacing_deg=GRID_LNG_SPACING_DEG)
        )

        (storm_object_table[tracking_utils.ROWCOL_POLYGON_COLUMN].values[i],
         storm_object_table[tracking_utils.COLUMNS_IN_STORM_COLUMN].values[i]
        ) = polygons.simple_polygon_to_grid_points(
            vertex_row_indices=these_vertex_rows,
            vertex_column_indices=these_vertex_columns)

        (storm_object_table[tracking_utils.LATITUDES_IN_STORM_COLUMN].values[i],
         storm_object_table[tracking_utils.LONGITUDES_IN_STORM_COLUMN].values[i]
        ) = radar_utils.rowcol_to_latlng(
            grid_rows=storm_object_table[
                tracking_utils.ROWS_IN_STORM_COLUMN].values[i],
            grid_columns=storm_object_table[
                tracking_utils.COLUMNS_IN_STORM_COLUMN].values[i],
            nw_grid_point_lat_deg=NW_GRID_POINT_LAT_DEG,
            nw_grid_point_lng_deg=NW_GRID_POINT_LNG_DEG,
            lat_spacing_deg=GRID_LAT_SPACING_DEG,
            lng_spacing_deg=GRID_LNG_SPACING_DEG)

        (storm_object_table[tracking_utils.CENTROID_LATITUDE_COLUMN].values[i],
         storm_object_table[tracking_utils.CENTROID_LONGITUDE_COLUMN].values[i]
        ) = geodetic_utils.get_latlng_centroid(
            latitudes_deg=these_vertex_latitudes_deg,
            longitudes_deg=these_vertex_longitudes_deg)

        storm_object_table[tracking_utils.ROWCOL_POLYGON_COLUMN].values[i] = (
            polygons.vertex_arrays_to_polygon_object(
                exterior_x_coords=these_vertex_columns,
                exterior_y_coords=these_vertex_rows)
        )

        storm_object_table[tracking_utils.LATLNG_POLYGON_COLUMN].values[i] = (
            polygons.vertex_arrays_to_polygon_object(
                exterior_x_coords=these_vertex_longitudes_deg,
                exterior_y_coords=these_vertex_latitudes_deg)
        )

    return storm_object_table


def rename_storms(
        top_input_dir_name, first_date_unix_sec, last_date_unix_sec,
        first_id_number, max_dropout_time_seconds, top_output_dir_name):
    """Renames storms.  This ensures that all storm IDs are unique.

    :param top_input_dir_name: Name of top-level directory with input files
        (processed probSevere files, readable by `storm_tracking_io.read_file`).
    :param first_date_unix_sec: First date in time period.  This method will fix
        IDs for all dates from `first_date_unix_sec`...`last_date_unix_sec`.
    :param last_date_unix_sec: See above.
    :param first_id_number: Will start with this ID.
    :param max_dropout_time_seconds: Max dropout time.  For each storm ID "s"
        found in the original data, this method will find all periods where "s"
        appears in consecutive time steps with no dropout longer than
        `max_dropout_time_seconds`.  Each such period will get a new, unique
        storm ID.
    :param top_output_dir_name: Name of top-level directory for output files
        (files with new IDs, to be written by `storm_tracking_io.write_file`).
    """

    error_checking.assert_is_integer(first_id_number)
    error_checking.assert_is_geq(first_id_number, 0)
    error_checking.assert_is_integer(max_dropout_time_seconds)
    error_checking.assert_is_greater(max_dropout_time_seconds, 0)

    (input_file_names_by_date, output_file_names_by_date,
     valid_times_by_date_unix_sec
    ) = _find_io_files_for_renaming(
        top_input_dir_name=top_input_dir_name,
        first_date_unix_sec=first_date_unix_sec,
        last_date_unix_sec=last_date_unix_sec,
        top_output_dir_name=top_output_dir_name)

    num_dates = len(input_file_names_by_date)
    storm_object_table_by_date = [None] * num_dates
    next_id_number = first_id_number + 0

    for i in range(num_dates):
        date_needed_indices = _get_dates_needed_for_renaming_storms(
            working_date_index=i, num_dates_in_period=num_dates)

        storm_object_table_by_date = _shuffle_io_for_renaming(
            input_file_names_by_date=input_file_names_by_date,
            output_file_names_by_date=output_file_names_by_date,
            valid_times_by_date_unix_sec=valid_times_by_date_unix_sec,
            storm_object_table_by_date=storm_object_table_by_date,
            working_date_index=i)

        concat_storm_object_table = pandas.concat(
            [storm_object_table_by_date[j] for j in date_needed_indices],
            axis=0, ignore_index=True
        )

        concat_storm_object_table, next_id_number = _rename_storms_one_table(
            storm_object_table=concat_storm_object_table,
            next_id_number=next_id_number,
            max_dropout_time_seconds=max_dropout_time_seconds,
            working_date_index=i)

        for j in date_needed_indices:
            storm_object_table_by_date[j] = concat_storm_object_table.loc[
                concat_storm_object_table[DATE_INDEX_KEY] == j
            ]

    _shuffle_io_for_renaming(
        input_file_names_by_date=input_file_names_by_date,
        output_file_names_by_date=output_file_names_by_date,
        valid_times_by_date_unix_sec=valid_times_by_date_unix_sec,
        storm_object_table_by_date=storm_object_table_by_date,
        working_date_index=None)
