"""IO methods for probSevere storm-tracking."""

import json
import os.path
import numpy
import pandas
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import geodetic_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking

# TODO(thunderhoser): add file-management code (will include transferring
# raw files from NSSL machine to local machine).

TRACKING_START_TIME_UNIX_SEC = 0
TRACKING_END_TIME_UNIX_SEC = int(3e9)

RAW_FILE_NAME_PREFIX = 'SSEC_AWIPS_PROBSEVERE'
JSON_FILE_EXTENSION = '.json'
ASCII_FILE_EXTENSION = '.ascii'
VALID_RAW_FILE_EXTENSIONS = [JSON_FILE_EXTENSION, ASCII_FILE_EXTENSION]

MONTH_FORMAT = '%Y%m'
DATE_FORMAT = '%Y%m%d'
TIME_FORMAT_IN_JSON_FILES = '%Y%m%d_%H%M%S UTC'
TIME_FORMAT_IN_RAW_FILE_NAMES = '%Y%m%d_%H%M%S'

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

NON_POLYGON_COLUMNS = [
    tracking_utils.STORM_ID_COLUMN, tracking_utils.EAST_VELOCITY_COLUMN,
    tracking_utils.NORTH_VELOCITY_COLUMN, tracking_utils.TIME_COLUMN]

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


def _check_raw_file_extension(file_extension):
    """Error-checks extension for raw file.

    :param file_extension: File extension (".json" or ".ascii").
    :raises: ValueError: if `file_extension not in VALID_RAW_FILE_EXTENSIONS`.
    """

    error_checking.assert_is_string(file_extension)
    if file_extension not in VALID_RAW_FILE_EXTENSIONS:
        error_string = (
            '\n\n{0:s}\nValid extensions for raw files (listed above) do not '
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
            unix_time_sec, TIME_FORMAT_IN_RAW_FILE_NAMES),
        file_extension)


def get_json_file_name_on_ftp(unix_time_sec, ftp_directory_name):
    """Returns name of JSON file on FTP server.

    :param unix_time_sec: Valid time.
    :param ftp_directory_name: Name of FTP directory with JSON files.
    :return: raw_ftp_file_name: Expected path to JSON file on FTP server.
    """

    return '{0:s}/{1:s}'.format(
        ftp_directory_name,
        _get_pathless_raw_file_name(
            unix_time_sec=unix_time_sec, file_extension=JSON_FILE_EXTENSION))


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

    raw_file_name = '{0:s}/{1:s}/{2:s}/{3:s}'.format(
        top_directory_name,
        time_conversion.unix_sec_to_string(unix_time_sec, MONTH_FORMAT),
        time_conversion.unix_sec_to_string(unix_time_sec, DATE_FORMAT),
        _get_pathless_raw_file_name(
            unix_time_sec=unix_time_sec, file_extension=file_extension))

    if raise_error_if_missing and not os.path.isfile(raw_file_name):
        error_string = (
            'Cannot find raw probSevere file.  Expected at: {0:s}'
        ).format(raw_file_name)
        raise ValueError(error_string)

    return raw_file_name


def raw_file_name_to_time(raw_file_name):
    """Parses valid time from name of raw (either ASCII or JSON) file.

    :param raw_file_name: Path to raw file.
    :return: unix_time_sec: Valid time.
    """

    error_checking.assert_is_string(raw_file_name)
    _, pathless_file_name = os.path.split(raw_file_name)
    extensionless_file_name, _ = os.path.splitext(pathless_file_name)

    return time_conversion.string_to_unix_sec(
        extensionless_file_name.replace(RAW_FILE_NAME_PREFIX + '_', ''),
        TIME_FORMAT_IN_RAW_FILE_NAMES)


def read_raw_file(raw_file_name):
    """Reads tracking data from raw (either JSON or ASCII) file.

    This file should contain all storm objects at one time step.

    :param raw_file_name: Path to input file.
    :return: storm_object_table: See documentation for
        `storm_tracking_io.write_processed_file`.
    """

    error_checking.assert_file_exists(raw_file_name)
    _, pathless_file_name = os.path.split(raw_file_name)
    _, file_extension = os.path.splitext(pathless_file_name)
    _check_raw_file_extension(file_extension)

    unix_time_sec = raw_file_name_to_time(raw_file_name)

    if file_extension == ASCII_FILE_EXTENSION:
        storm_ids = []
        east_velocities_m_s01 = []
        north_velocities_m_s01 = []
        list_of_latitude_vertex_arrays_deg = []
        list_of_longitude_vertex_arrays_deg = []

        for this_line in open(raw_file_name, 'r').readlines():
            these_words = this_line.split(':')
            if len(these_words) < MIN_WORDS_PER_ASCII_LINE:
                continue

            storm_ids.append(these_words[STORM_ID_INDEX_IN_ASCII_FILES])
            east_velocities_m_s01.append(
                float(these_words[U_MOTION_INDEX_IN_ASCII_FILES]))
            north_velocities_m_s01.append(
                -1 * float(these_words[V_MOTION_INDEX_IN_ASCII_FILES]))

            these_polygon_words = numpy.array(
                these_words[POLYGON_INDEX_IN_ASCII_FILES].split(','))
            these_latitude_words = these_polygon_words[
                LATITUDE_INDEX_IN_ASCII_FILES::2].tolist()
            these_longitude_words = these_polygon_words[
                LONGITUDE_INDEX_IN_ASCII_FILES::2].tolist()

            these_latitudes_deg = numpy.array(
                [float(w) for w in these_latitude_words])
            these_longitudes_deg = numpy.array(
                [float(w) for w in these_longitude_words])
            list_of_latitude_vertex_arrays_deg.append(these_latitudes_deg)
            list_of_longitude_vertex_arrays_deg.append(these_longitudes_deg)

        east_velocities_m_s01 = numpy.array(east_velocities_m_s01)
        north_velocities_m_s01 = numpy.array(north_velocities_m_s01)
        num_storms = len(storm_ids)

    else:
        with open(raw_file_name) as json_file_handle:
            probsevere_dict = json.load(json_file_handle)

        num_storms = len(probsevere_dict[FEATURES_KEY_IN_JSON_FILES])
        storm_ids = [None] * num_storms
        east_velocities_m_s01 = numpy.full(num_storms, numpy.nan)
        north_velocities_m_s01 = numpy.full(num_storms, numpy.nan)
        list_of_latitude_vertex_arrays_deg = [None] * num_storms
        list_of_longitude_vertex_arrays_deg = [None] * num_storms

        for i in range(num_storms):
            storm_ids[i] = str(
                probsevere_dict[FEATURES_KEY_IN_JSON_FILES][i][
                    PROPERTIES_KEY_IN_JSON_FILES][STORM_ID_KEY_IN_JSON_FILES])
            east_velocities_m_s01[i] = float(
                probsevere_dict[FEATURES_KEY_IN_JSON_FILES][i][
                    PROPERTIES_KEY_IN_JSON_FILES][U_MOTION_KEY_IN_JSON_FILES])
            north_velocities_m_s01[i] = -1 * float(
                probsevere_dict[FEATURES_KEY_IN_JSON_FILES][i][
                    PROPERTIES_KEY_IN_JSON_FILES][V_MOTION_KEY_IN_JSON_FILES])

            this_vertex_matrix_deg = numpy.array(
                probsevere_dict[FEATURES_KEY_IN_JSON_FILES][i][
                    GEOMETRY_KEY_IN_JSON_FILES
                ][COORDINATES_KEY_IN_JSON_FILES][0])
            list_of_latitude_vertex_arrays_deg[i] = numpy.array(
                this_vertex_matrix_deg[:, LATITUDE_INDEX_IN_JSON_FILES])
            list_of_longitude_vertex_arrays_deg[i] = numpy.array(
                this_vertex_matrix_deg[:, LONGITUDE_INDEX_IN_JSON_FILES])

    spc_date_unix_sec = time_conversion.time_to_spc_date_unix_sec(unix_time_sec)
    unix_times_sec = numpy.full(num_storms, unix_time_sec, dtype=int)
    spc_dates_unix_sec = numpy.full(num_storms, spc_date_unix_sec, dtype=int)
    tracking_start_times_unix_sec = numpy.full(
        num_storms, TRACKING_START_TIME_UNIX_SEC, dtype=int)
    tracking_end_times_unix_sec = numpy.full(
        num_storms, TRACKING_END_TIME_UNIX_SEC, dtype=int)

    storm_object_dict = {
        tracking_utils.STORM_ID_COLUMN: storm_ids,
        tracking_utils.EAST_VELOCITY_COLUMN: east_velocities_m_s01,
        tracking_utils.NORTH_VELOCITY_COLUMN: north_velocities_m_s01,
        tracking_utils.TIME_COLUMN: unix_times_sec,
        tracking_utils.SPC_DATE_COLUMN: spc_dates_unix_sec,
        tracking_utils.TRACKING_START_TIME_COLUMN:
            tracking_start_times_unix_sec,
        tracking_utils.TRACKING_END_TIME_COLUMN: tracking_end_times_unix_sec
    }
    storm_object_table = pandas.DataFrame.from_dict(storm_object_dict)

    storm_ages_sec = numpy.full(num_storms, numpy.nan)
    simple_array = numpy.full(num_storms, numpy.nan)
    object_array = numpy.full(num_storms, numpy.nan, dtype=object)
    nested_array = storm_object_table[[
        tracking_utils.STORM_ID_COLUMN,
        tracking_utils.STORM_ID_COLUMN]].values.tolist()

    argument_dict = {
        tracking_utils.AGE_COLUMN: storm_ages_sec,
        tracking_utils.CENTROID_LAT_COLUMN: simple_array,
        tracking_utils.CENTROID_LNG_COLUMN: simple_array,
        tracking_utils.GRID_POINT_LAT_COLUMN: nested_array,
        tracking_utils.GRID_POINT_LNG_COLUMN: nested_array,
        tracking_utils.GRID_POINT_ROW_COLUMN: nested_array,
        tracking_utils.GRID_POINT_COLUMN_COLUMN: nested_array,
        tracking_utils.POLYGON_OBJECT_LATLNG_COLUMN: object_array,
        tracking_utils.POLYGON_OBJECT_ROWCOL_COLUMN: object_array
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
                lng_spacing_deg=GRID_LNG_SPACING_DEG))

        these_vertex_rows, these_vertex_columns = (
            polygons.fix_probsevere_vertices(
                these_vertex_rows, these_vertex_columns))

        these_vertex_latitudes_deg, these_vertex_longitudes_deg = (
            radar_utils.rowcol_to_latlng(
                grid_rows=these_vertex_rows, grid_columns=these_vertex_columns,
                nw_grid_point_lat_deg=NW_GRID_POINT_LAT_DEG,
                nw_grid_point_lng_deg=NW_GRID_POINT_LNG_DEG,
                lat_spacing_deg=GRID_LAT_SPACING_DEG,
                lng_spacing_deg=GRID_LNG_SPACING_DEG))

        (storm_object_table[tracking_utils.GRID_POINT_ROW_COLUMN].values[i],
         storm_object_table[tracking_utils.GRID_POINT_COLUMN_COLUMN].values[i]
        ) = polygons.simple_polygon_to_grid_points(
            these_vertex_rows, these_vertex_columns)

        (storm_object_table[tracking_utils.GRID_POINT_LAT_COLUMN].values[i],
         storm_object_table[tracking_utils.GRID_POINT_LNG_COLUMN].values[i]
        ) = radar_utils.rowcol_to_latlng(
            storm_object_table[tracking_utils.GRID_POINT_ROW_COLUMN].values[i],
            storm_object_table[
                tracking_utils.GRID_POINT_COLUMN_COLUMN].values[i],
            nw_grid_point_lat_deg=NW_GRID_POINT_LAT_DEG,
            nw_grid_point_lng_deg=NW_GRID_POINT_LNG_DEG,
            lat_spacing_deg=GRID_LAT_SPACING_DEG,
            lng_spacing_deg=GRID_LNG_SPACING_DEG)

        (storm_object_table[tracking_utils.CENTROID_LAT_COLUMN].values[i],
         storm_object_table[tracking_utils.CENTROID_LNG_COLUMN].values[i]
        ) = geodetic_utils.get_latlng_centroid(
            these_vertex_latitudes_deg, these_vertex_longitudes_deg)

        storm_object_table[
            tracking_utils.POLYGON_OBJECT_ROWCOL_COLUMN
        ].values[i] = polygons.vertex_arrays_to_polygon_object(
            these_vertex_columns, these_vertex_rows)

        storm_object_table[
            tracking_utils.POLYGON_OBJECT_LATLNG_COLUMN
        ].values[i] = polygons.vertex_arrays_to_polygon_object(
            these_vertex_latitudes_deg, these_vertex_longitudes_deg)

    return storm_object_table
