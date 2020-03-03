"""Makes LaTeX table with GridRad days and domains."""

import os.path
import argparse
import numpy
from gewittergefahr.gg_io import gridrad_io
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TIME_INTERVAL_SEC = 300
NICE_TIME_FORMAT = '%d %b %Y'

GRIDRAD_DIR_ARG_NAME = 'input_gridrad_dir_name'
FIRST_DATE_ARG_NAME = 'first_spc_date_string'
LAST_DATE_ARG_NAME = 'last_spc_date_string'

GRIDRAD_DIR_HELP_STRING = (
    'Name of top-level GridRad directory, used to determine which convective '
    'days are covered.  Files therein will be found by `gridrad_io.find_file`.'
)

SPC_DATE_HELP_STRING = (
    'SPC date or convective day (format "yyyymmdd").  This script will look for'
    ' GridRad files in the period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + GRIDRAD_DIR_ARG_NAME, type=str, required=True,
    help=GRIDRAD_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING
)


def _find_domain_for_date(top_gridrad_dir_name, spc_date_string):
    """Finds GridRad domain for the given SPC date.

    If no GridRad files are found the given the SPC date, this method returns
    None for all output variables.

    :param top_gridrad_dir_name: See documentation at top of file.
    :param spc_date_string: SPC date or convective day (format "yyyymmdd").
    :return: domain_limits_deg: length-4 numpy array with
        [min latitude, max latitude, min longitude, max longitude].
        Units are deg N for latitude, deg W for longitude.
    """

    first_time_unix_sec = time_conversion.get_start_of_spc_date(spc_date_string)
    last_time_unix_sec = time_conversion.get_end_of_spc_date(spc_date_string)
    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SEC, include_endpoint=True
    )

    num_times = len(valid_times_unix_sec)
    min_latitudes_deg = numpy.full(num_times, numpy.nan)
    max_latitudes_deg = numpy.full(num_times, numpy.nan)
    min_longitudes_deg = numpy.full(num_times, numpy.nan)
    max_longitudes_deg = numpy.full(num_times, numpy.nan)

    for i in range(num_times):
        this_file_name = gridrad_io.find_file(
            unix_time_sec=valid_times_unix_sec[i],
            top_directory_name=top_gridrad_dir_name,
            raise_error_if_missing=False
        )

        if not os.path.isfile(this_file_name):
            continue

        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_metadata_dict = gridrad_io.read_metadata_from_full_grid_file(
            this_file_name
        )

        max_latitudes_deg[i] = (
            this_metadata_dict[radar_utils.NW_GRID_POINT_LAT_COLUMN]
        )
        min_longitudes_deg[i] = (
            this_metadata_dict[radar_utils.NW_GRID_POINT_LNG_COLUMN]
        )

        max_longitudes_deg[i] = min_longitudes_deg[i] + (
            (this_metadata_dict[radar_utils.NUM_LNG_COLUMN] - 1) *
            this_metadata_dict[radar_utils.LNG_SPACING_COLUMN]
        )

        min_latitudes_deg[i] = max_latitudes_deg[i] - (
            (this_metadata_dict[radar_utils.NUM_LAT_COLUMN] - 1) *
            this_metadata_dict[radar_utils.LAT_SPACING_COLUMN]
        )

    good_indices = numpy.where(numpy.invert(numpy.isnan(min_latitudes_deg)))[0]
    if len(good_indices) == 0:
        return None

    coord_matrix = numpy.vstack((
        min_latitudes_deg[good_indices], max_latitudes_deg[good_indices],
        min_longitudes_deg[good_indices], max_longitudes_deg[good_indices]
    ))
    coord_matrix, num_instances_by_row = numpy.unique(
        numpy.transpose(coord_matrix), axis=0, return_counts=True
    )

    print(coord_matrix)
    print(num_instances_by_row)

    domain_limits_deg = coord_matrix[numpy.argmax(num_instances_by_row), :]
    domain_limits_deg[2:] = -1 * lng_conversion.convert_lng_negative_in_west(
        longitudes_deg=domain_limits_deg[2:], allow_nan=False
    )

    return domain_limits_deg


def _run(top_gridrad_dir_name, first_spc_date_string, last_spc_date_string):
    """Makes LaTeX table with GridRad days and domains.

    This is effectively the main method.

    :param top_gridrad_dir_name: See documentation at top of file.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    """

    all_spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string)

    nice_date_strings = []
    latitude_strings = []
    longitude_strings = []

    for this_spc_date_string in all_spc_date_strings:
        these_limits_deg = _find_domain_for_date(
            top_gridrad_dir_name=top_gridrad_dir_name,
            spc_date_string=this_spc_date_string
        )

        if these_limits_deg is None:
            continue

        print(SEPARATOR_STRING)

        this_time_unix_sec = time_conversion.spc_date_string_to_unix_sec(
            this_spc_date_string
        )
        this_nice_date_string = time_conversion.unix_sec_to_string(
            this_time_unix_sec, NICE_TIME_FORMAT
        )

        nice_date_strings.append(this_nice_date_string)
        latitude_strings.append('{0:.1f}-{1:.1f}'.format(
            these_limits_deg[0], these_limits_deg[1]
        ))
        longitude_strings.append('{0:.1f}-{1:.1f}'.format(
            these_limits_deg[3], these_limits_deg[2]
        ))

    table_string = ''

    for i in range(len(nice_date_strings)):
        if i != 0:
            if numpy.mod(i, 4) == 0:
                table_string += ' \\\\\n'
            else:
                table_string += ' & '

        # table_string += '{0:s}, {1:s} $^{\\circ}$N, {2:s} $^{\\circ}$W'.format(
        #     nice_date_strings[i], latitude_strings[i], longitude_strings[i]
        # )

        table_string += '{0:s}, {1:s}'.format(
            nice_date_strings[i], latitude_strings[i]
        )
        table_string += ' $^{\\circ}$N'
        table_string += ', {0:s}'.format(longitude_strings[i])
        table_string += ' $^{\\circ}$W'

    print(table_string)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_gridrad_dir_name=getattr(INPUT_ARG_OBJECT, GRIDRAD_DIR_ARG_NAME),
        first_spc_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME)
    )
