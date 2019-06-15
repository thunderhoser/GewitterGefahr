"""IO methods for tornado reports."""

import re
import os.path
import numpy
import pandas
from gewittergefahr.gg_utils import interp
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import geodetic_utils
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

TOLERANCE = 1e-6
TIME_FORMAT = '%Y-%m-%d-%H%M%S'

START_TIME_COLUMN = 'start_time_unix_sec'
END_TIME_COLUMN = 'end_time_unix_sec'
START_LAT_COLUMN = 'start_latitude_deg'
END_LAT_COLUMN = 'end_latitude_deg'
START_LNG_COLUMN = 'start_longitude_deg'
END_LNG_COLUMN = 'end_longitude_deg'
FUJITA_RATING_COLUMN = 'f_or_ef_rating'
WIDTH_COLUMN = 'width_metres'
TORNADO_ID_COLUMN = 'tornado_id_string'

TIME_COLUMN = 'valid_time_unix_sec'
LATITUDE_COLUMN = 'latitude_deg'
LONGITUDE_COLUMN = 'longitude_deg'

F_SCALE_RATING_PATTERN = '[fF][0-5]'
EF_SCALE_RATING_PATTERN = '[eE][fF][0-5]'

MANDATORY_COLUMNS = [
    START_TIME_COLUMN, END_TIME_COLUMN, START_LAT_COLUMN, END_LAT_COLUMN,
    START_LNG_COLUMN, END_LNG_COLUMN, FUJITA_RATING_COLUMN, WIDTH_COLUMN
]

MANDATORY_COLUMN_TYPE_DICT = {
    START_TIME_COLUMN: int,
    END_TIME_COLUMN: int,
    START_LAT_COLUMN: float,
    END_LAT_COLUMN: float,
    START_LNG_COLUMN: float,
    END_LNG_COLUMN: float,
    WIDTH_COLUMN: float,
    FUJITA_RATING_COLUMN: str
}


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


def create_tornado_id(start_time_unix_sec, start_latitude_deg,
                      start_longitude_deg):
    """Creates tornado ID.

    :param start_time_unix_sec: Start time.
    :param start_latitude_deg: Start latitude (deg N).
    :param start_longitude_deg: Start longitude (deg E).
    :return: tornado_id_string: ID.
    """

    return '{0:s}_{1:6.3f}N_{2:7.3f}E'.format(
        time_conversion.unix_sec_to_string(start_time_unix_sec, TIME_FORMAT),
        start_latitude_deg, start_longitude_deg
    )


def add_tornado_ids_to_table(tornado_table):
    """Adds tornado IDs to table.

    :param tornado_table: See doc for `write_processed_file`.
    :return: tornado_table: Same as input but with one extra column.
    tornado_table.tornado_id_string: Tornado ID.
    """

    tornado_id_strings = [
        create_tornado_id(start_time_unix_sec=t, start_latitude_deg=y,
                          start_longitude_deg=x)
        for t, y, x in zip(
            tornado_table[START_TIME_COLUMN].values,
            tornado_table[START_LAT_COLUMN].values,
            tornado_table[START_LNG_COLUMN].values
        )
    ]

    return tornado_table.assign(**{
        TORNADO_ID_COLUMN: tornado_id_strings,
    })


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


def interp_tornadoes_along_tracks(tornado_table, interp_time_interval_sec):
    """Interpolates each tornado to many points along its track.

    :param tornado_table: See doc for `write_processed_file`.
    :param interp_time_interval_sec: Will interpolate at this time interval
        between start and end points.
    :return: tornado_segment_table: pandas DataFrame with the following columns,
        where each row is one tornado-track segment.
    tornado_segment_table.valid_time_unix_sec: Valid time.
    tornado_segment_table.latitude_deg: Latitude (deg N).
    tornado_segment_table.longitude_deg: Longitude (deg E).
    tornado_segment_table.tornado_id_string: Tornado ID.
    tornado_segment_table.fujita_rating: F-scale or EF-scale rating (integer
        from 0...5).
    """

    # TODO(thunderhoser): Return "width" column as well.

    num_tornadoes = len(tornado_table.index)
    tornado_times_unix_sec = numpy.array([], dtype=int)
    tornado_latitudes_deg = numpy.array([])
    tornado_longitudes_deg = numpy.array([])
    tornado_id_strings = []
    fujita_rating_strings = []

    for j in range(num_tornadoes):
        this_start_time_unix_sec = tornado_table[START_TIME_COLUMN].values[j]
        this_end_time_unix_sec = tornado_table[END_TIME_COLUMN].values[j]

        this_num_query_times = 1 + int(numpy.round(
            float(this_end_time_unix_sec - this_start_time_unix_sec) /
            interp_time_interval_sec
        ))

        these_query_times_unix_sec = numpy.linspace(
            this_start_time_unix_sec, this_end_time_unix_sec,
            num=this_num_query_times, dtype=float)

        these_query_times_unix_sec = numpy.round(
            these_query_times_unix_sec
        ).astype(int)

        these_input_latitudes_deg = numpy.array([
            tornado_table[START_LAT_COLUMN].values[j],
            tornado_table[END_LAT_COLUMN].values[j]
        ])

        these_input_longitudes_deg = numpy.array([
            tornado_table[START_LNG_COLUMN].values[j],
            tornado_table[END_LNG_COLUMN].values[j]
        ])

        these_input_times_unix_sec = numpy.array([
            tornado_table[START_TIME_COLUMN].values[j],
            tornado_table[END_TIME_COLUMN].values[j]
        ], dtype=int)

        if this_num_query_times == 1:
            this_query_coord_matrix = numpy.array([
                these_input_longitudes_deg[0], these_input_latitudes_deg[0]
            ])

            this_query_coord_matrix = numpy.reshape(
                this_query_coord_matrix, (2, 1)
            )
        else:
            this_input_coord_matrix = numpy.vstack((
                these_input_longitudes_deg, these_input_latitudes_deg
            ))

            this_query_coord_matrix = interp.interp_in_time(
                input_matrix=this_input_coord_matrix,
                sorted_input_times_unix_sec=these_input_times_unix_sec,
                query_times_unix_sec=these_query_times_unix_sec,
                method_string=interp.LINEAR_METHOD_STRING, extrapolate=False)

        tornado_times_unix_sec = numpy.concatenate((
            tornado_times_unix_sec, these_query_times_unix_sec
        ))

        tornado_latitudes_deg = numpy.concatenate((
            tornado_latitudes_deg, this_query_coord_matrix[1, :]
        ))

        tornado_longitudes_deg = numpy.concatenate((
            tornado_longitudes_deg, this_query_coord_matrix[0, :]
        ))

        this_id_string = create_tornado_id(
            start_time_unix_sec=these_input_times_unix_sec[0],
            start_latitude_deg=these_input_latitudes_deg[0],
            start_longitude_deg=these_input_longitudes_deg[0]
        )

        tornado_id_strings += (
            [this_id_string] * len(these_query_times_unix_sec)
        )
        fujita_rating_strings += (
            [tornado_table[FUJITA_RATING_COLUMN].values[j]]
            * len(these_query_times_unix_sec)
        )

    return pandas.DataFrame.from_dict({
        TIME_COLUMN: tornado_times_unix_sec,
        LATITUDE_COLUMN: tornado_latitudes_deg,
        LONGITUDE_COLUMN: tornado_longitudes_deg,
        TORNADO_ID_COLUMN: tornado_id_strings,
        FUJITA_RATING_COLUMN: fujita_rating_strings
    })


def subset_tornadoes(
        tornado_table, min_time_unix_sec=None, max_time_unix_sec=None,
        min_latitude_deg=None, max_latitude_deg=None, min_longitude_deg=None,
        max_longitude_deg=None):
    """Subsets tornadoes by time, latitude, and/or longitude.

    :param tornado_table: See doc for `write_processed_file`.
    :param min_time_unix_sec: Minimum time.
    :param max_time_unix_sec: Max time.
    :param min_latitude_deg: Minimum latitude (deg N).
    :param max_latitude_deg: Max latitude (deg N).
    :param min_longitude_deg: Minimum longitude (deg E).
    :param max_longitude_deg: Max longitude (deg E).
    :return: tornado_table: Same as input but maybe with fewer rows.
    """

    # TODO(thunderhoser): Needs unit tests.

    if min_time_unix_sec is not None or max_time_unix_sec is not None:
        error_checking.assert_is_integer(min_time_unix_sec)
        error_checking.assert_is_integer(max_time_unix_sec)
        error_checking.assert_is_greater(max_time_unix_sec, min_time_unix_sec)

        good_start_flags = numpy.logical_and(
            tornado_table[START_TIME_COLUMN].values >= min_time_unix_sec,
            tornado_table[START_TIME_COLUMN].values <= max_time_unix_sec
        )

        good_end_flags = numpy.logical_and(
            tornado_table[END_TIME_COLUMN].values >= min_time_unix_sec,
            tornado_table[END_TIME_COLUMN].values <= max_time_unix_sec
        )

        invalid_flags = numpy.invert(numpy.logical_or(
            good_start_flags, good_end_flags
        ))

        invalid_rows = numpy.where(invalid_flags)[0]
        tornado_table.drop(
            tornado_table.index[invalid_rows], axis=0, inplace=True
        )

    if min_latitude_deg is not None or max_latitude_deg is not None:
        error_checking.assert_is_valid_latitude(min_latitude_deg)
        error_checking.assert_is_valid_latitude(max_latitude_deg)
        error_checking.assert_is_greater(max_latitude_deg, min_latitude_deg)

        good_start_flags = numpy.logical_and(
            tornado_table[START_LAT_COLUMN].values >= min_latitude_deg,
            tornado_table[START_LAT_COLUMN].values <= max_latitude_deg
        )

        good_end_flags = numpy.logical_and(
            tornado_table[END_LAT_COLUMN].values >= min_latitude_deg,
            tornado_table[END_LAT_COLUMN].values <= max_latitude_deg
        )

        invalid_flags = numpy.invert(numpy.logical_or(
            good_start_flags, good_end_flags
        ))

        invalid_rows = numpy.where(invalid_flags)[0]
        tornado_table.drop(
            tornado_table.index[invalid_rows], axis=0, inplace=True
        )

    if min_longitude_deg is not None or max_longitude_deg is not None:
        longitude_limits_deg = lng_conversion.convert_lng_positive_in_west(
            longitudes_deg=numpy.array([min_longitude_deg, max_longitude_deg]),
            allow_nan=False
        )

        min_longitude_deg = longitude_limits_deg[0]
        max_longitude_deg = longitude_limits_deg[1]
        error_checking.assert_is_greater(max_longitude_deg, min_longitude_deg)

        good_start_flags = numpy.logical_and(
            tornado_table[START_LNG_COLUMN].values >= min_longitude_deg,
            tornado_table[START_LNG_COLUMN].values <= max_longitude_deg
        )

        good_end_flags = numpy.logical_and(
            tornado_table[END_LNG_COLUMN].values >= min_longitude_deg,
            tornado_table[END_LNG_COLUMN].values <= max_longitude_deg
        )

        invalid_flags = numpy.invert(numpy.logical_or(
            good_start_flags, good_end_flags
        ))

        invalid_rows = numpy.where(invalid_flags)[0]
        tornado_table.drop(
            tornado_table.index[invalid_rows], axis=0, inplace=True
        )

    return tornado_table


def segments_to_tornadoes(tornado_segment_table):
    """Converts list of tornado segments to list of individual tornadoes.

    :param tornado_segment_table: pandas DataFrame created by
        `interp_tornadoes_along_tracks`.
    :return: tornado_table: pandas DataFrame with columns listed in
        `write_processed_file`, plus those listed below.
    tornado_table.tornado_id_string: Tornado ID.
    """

    # TODO(thunderhoser): Add "width" to output columns.

    unique_id_strings, orig_to_unique_indices = numpy.unique(
        tornado_segment_table[TORNADO_ID_COLUMN].values, return_inverse=True
    )

    num_tornadoes = len(unique_id_strings)

    tornado_table = pandas.DataFrame.from_dict({
        TORNADO_ID_COLUMN: unique_id_strings,
        START_TIME_COLUMN: numpy.full(num_tornadoes, -1, dtype=int),
        START_LAT_COLUMN: numpy.full(num_tornadoes, numpy.nan),
        START_LNG_COLUMN: numpy.full(num_tornadoes, numpy.nan),
        END_TIME_COLUMN: numpy.full(num_tornadoes, -1, dtype=int),
        END_LAT_COLUMN: numpy.full(num_tornadoes, numpy.nan),
        END_LNG_COLUMN: numpy.full(num_tornadoes, numpy.nan),
        FUJITA_RATING_COLUMN: [''] * num_tornadoes
    })

    for j in range(num_tornadoes):
        these_indices = numpy.where(orig_to_unique_indices == j)[0]

        this_subindex = numpy.argmin(
            tornado_segment_table[TIME_COLUMN].values[these_indices]
        )
        this_start_index = these_indices[this_subindex]

        tornado_table[START_TIME_COLUMN].values[j] = tornado_segment_table[
            TIME_COLUMN].values[this_start_index]
        tornado_table[START_LAT_COLUMN].values[j] = tornado_segment_table[
            LATITUDE_COLUMN].values[this_start_index]
        tornado_table[START_LNG_COLUMN].values[j] = tornado_segment_table[
            LONGITUDE_COLUMN].values[this_start_index]

        this_subindex = numpy.argmax(
            tornado_segment_table[TIME_COLUMN].values[these_indices]
        )
        this_end_index = these_indices[this_subindex]

        tornado_table[END_TIME_COLUMN].values[j] = tornado_segment_table[
            TIME_COLUMN].values[this_end_index]
        tornado_table[END_LAT_COLUMN].values[j] = tornado_segment_table[
            LATITUDE_COLUMN].values[this_end_index]
        tornado_table[END_LNG_COLUMN].values[j] = tornado_segment_table[
            LONGITUDE_COLUMN].values[this_end_index]

        tornado_table[FUJITA_RATING_COLUMN].values[j] = tornado_segment_table[
            FUJITA_RATING_COLUMN].values[this_end_index]

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
            '"{0:s}"'
        ).format(processed_file_name)

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
