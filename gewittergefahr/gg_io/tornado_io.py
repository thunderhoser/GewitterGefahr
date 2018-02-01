"""IO methods for tornado reports."""

import re
import os.path
import numpy
import pandas
from gewittergefahr.gg_utils import geodetic_utils
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

TOLERANCE = 1e-6

START_TIME_COLUMN = 'start_time_unix_sec'
END_TIME_COLUMN = 'end_time_unix_sec'
START_LAT_COLUMN = 'start_latitude_deg'
END_LAT_COLUMN = 'end_latitude_deg'
START_LNG_COLUMN = 'start_longitude_deg'
END_LNG_COLUMN = 'end_longitude_deg'
FUJITA_RATING_COLUMN = 'f_or_ef_rating'
WIDTH_COLUMN = 'width_metres'

F_SCALE_RATING_PATTERN = '[fF][0-5]'
EF_SCALE_RATING_PATTERN = '[eE][fF][0-5]'

MANDATORY_COLUMNS = [
    START_TIME_COLUMN, END_TIME_COLUMN, START_LAT_COLUMN, END_LAT_COLUMN,
    START_LNG_COLUMN, END_LNG_COLUMN, FUJITA_RATING_COLUMN, WIDTH_COLUMN]

MANDATORY_COLUMN_TYPE_DICT = {
    START_TIME_COLUMN: int, END_TIME_COLUMN: int, START_LAT_COLUMN: float,
    END_LAT_COLUMN: float, START_LNG_COLUMN: float, END_LNG_COLUMN: float,
    WIDTH_COLUMN: float, FUJITA_RATING_COLUMN: str}


def _is_valid_fujita_rating(fujita_rating_string):
    """Checks validity of F- or EF-scale rating.

    :param fujita_rating_string: F- or EF-scale rating.  Format must be either
        "F[0-5]" or "EF[0-5]".  Examples: "F0", "F1", "F5", "EF2", "EF5".
    :return: valid_flag: Boolean flag.
    """

    if re.match(F_SCALE_RATING_PATTERN, fujita_rating_string) is not None:
        return True
    if re.match(EF_SCALE_RATING_PATTERN, fujita_rating_string) is not None:
        return True

    return False


def remove_invalid_reports(
        tornado_table, check_times_flag=True, check_latitudes_flag=True,
        check_longitudes_flag=True, check_fujita_rating_flag=True,
        check_width_flag=True):
    """Removes invalid tornado reports.

    A report is considered invalid if any of its properties are invalid.

    :param tornado_table: pandas DataFrame created by
        `storm_events_io.read_tornado_reports`.
    :param check_times_flag: Boolean flag.  If True, will check start/end time
        for each report.
    :param check_latitudes_flag: Boolean flag.  If True, will check start/end
        latitude for each report.
    :param check_longitudes_flag: Boolean flag.  If True, will check start/end
        longitude for each report.
    :param check_fujita_rating_flag: Boolean flag.  If True, will check Fujita
        rating for each report.  If False, will ignore Fujita rating.
    :param check_width_flag: Boolean flag.  If True, will check tornado width
        for each report.
    :return: tornado_table: Same as input, but maybe with fewer rows.
    """

    error_checking.assert_is_boolean(check_fujita_rating_flag)
    error_checking.assert_is_boolean(check_width_flag)
    error_checking.assert_is_boolean(check_latitudes_flag)
    error_checking.assert_is_boolean(check_longitudes_flag)
    error_checking.assert_is_boolean(check_times_flag)

    if check_times_flag:
        invalid_flags = numpy.invert(numpy.logical_and(
            tornado_table[START_TIME_COLUMN].values > 0,
            tornado_table[END_TIME_COLUMN].values > 0))
        tornado_table.drop(
            tornado_table.index[numpy.where(invalid_flags)[0]], axis=0,
            inplace=True)

    if check_latitudes_flag:
        invalid_indices = geodetic_utils.find_invalid_latitudes(
            tornado_table[START_LAT_COLUMN].values)
        tornado_table.drop(
            tornado_table.index[invalid_indices], axis=0, inplace=True)

        invalid_indices = geodetic_utils.find_invalid_latitudes(
            tornado_table[END_LAT_COLUMN].values)
        tornado_table.drop(
            tornado_table.index[invalid_indices], axis=0, inplace=True)

    if check_longitudes_flag:
        invalid_indices = geodetic_utils.find_invalid_longitudes(
            tornado_table[START_LNG_COLUMN].values,
            sign_in_western_hemisphere=geodetic_utils.EITHER_SIGN_LONGITUDE_ARG)
        tornado_table.drop(
            tornado_table.index[invalid_indices], axis=0, inplace=True)

        invalid_indices = geodetic_utils.find_invalid_longitudes(
            tornado_table[END_LNG_COLUMN].values,
            sign_in_western_hemisphere=geodetic_utils.EITHER_SIGN_LONGITUDE_ARG)
        tornado_table.drop(
            tornado_table.index[invalid_indices], axis=0, inplace=True)

        tornado_table[START_LNG_COLUMN] = (
            lng_conversion.convert_lng_positive_in_west(
                tornado_table[START_LNG_COLUMN].values))
        tornado_table[END_LNG_COLUMN] = (
            lng_conversion.convert_lng_positive_in_west(
                tornado_table[END_LNG_COLUMN].values))

    if check_fujita_rating_flag:
        invalid_flags = numpy.invert(numpy.array(
            [_is_valid_fujita_rating(s)
             for s in tornado_table[FUJITA_RATING_COLUMN].values]))
        tornado_table.drop(
            tornado_table.index[numpy.where(invalid_flags)[0]], axis=0,
            inplace=True)

    if check_width_flag:
        invalid_flags = numpy.invert(tornado_table[WIDTH_COLUMN].values > 0.)
        tornado_table.drop(
            tornado_table.index[numpy.where(invalid_flags)[0]], axis=0,
            inplace=True)

    return tornado_table


def find_processed_file(directory_name, year, raise_error_if_missing=True):
    """Finds processed file with tornado reports.

    See `write_processed_file` for the definition of a "processed file".

    :param directory_name: Name of directory.
    :param year: Year (integer).
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        raise_error_if_missing = True, this method will error out.
    :return: processed_file_name: Path to file.  If file is missing and
        raise_error_if_missing = True, this will be the *expected* path.
    :raises: ValueError: if file is missing and raise_error_if_missing = True.
    """

    error_checking.assert_is_string(directory_name)
    error_checking.assert_is_integer(year)
    error_checking.assert_is_boolean(raise_error_if_missing)

    processed_file_name = '{0:s}/tornado_reports_{1:04d}.csv'.format(
        directory_name, year)

    if raise_error_if_missing and not os.path.isfile(processed_file_name):
        error_string = (
            'Cannot find processed file with tornado reports.  Expected at: '
            '{0:s}').format(processed_file_name)
        raise ValueError(error_string)

    return processed_file_name


def write_processed_file(tornado_table, csv_file_name):
    """Writes tornado reports to CSV file.

    This is considered a "processed file," as opposed to a "raw file" (one taken
    directly from the Storm Events database).  Raw files with tornado reports
    are handled by storm_events_io.py.

    :param tornado_table: pandas DataFrame with the following columns.
    tornado_table.start_time_unix_sec: Start time.
    tornado_table.end_time_unix_sec: End time.
    tornado_table.start_latitude_deg: Latitude (deg N) of start point.
    tornado_table.start_longitude_deg: Longitude (deg E) of start point.
    tornado_table.end_latitude_deg: Latitude (deg N) of end point.
    tornado_table.end_longitude_deg: Longitude (deg E) of end point.
    tornado_table.fujita_rating: F-scale or EF-scale rating (integer from
        0...5).
    tornado_table.width_metres: Tornado width (metres).

    :param csv_file_name: Path to output file.
    """

    error_checking.assert_columns_in_dataframe(tornado_table, MANDATORY_COLUMNS)
    file_system_utils.mkdir_recursive_if_necessary(file_name=csv_file_name)

    tornado_table.to_csv(
        csv_file_name, header=True, columns=MANDATORY_COLUMNS, index=False)


def read_processed_file(csv_file_name):
    """Reads tornado reports from CSV file.

    See `write_processed_file` for the definition of a "processed file".

    :param csv_file_name: Path to input file.
    :return: tornado_table: See documentation for `write_processed_file`.
    """

    return pandas.read_csv(
        csv_file_name, header=0, usecols=MANDATORY_COLUMNS,
        dtype=MANDATORY_COLUMN_TYPE_DICT)
