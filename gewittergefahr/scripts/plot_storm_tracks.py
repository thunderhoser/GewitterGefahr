"""Plots storm tracks for a continuous time period."""

import argparse
import numpy
import pandas
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import storm_plotting

MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

SENTINEL_VALUE = -9999

NUM_PARALLELS = 8
NUM_MERIDIANS = 6
LATLNG_BUFFER_DEG = 0.5
BORDER_COLOUR = numpy.full(3, 0.)
FIGURE_RESOLUTION_DPI = 300

REQUIRED_COLUMNS = [
    tracking_utils.STORM_ID_COLUMN, tracking_utils.TIME_COLUMN,
    tracking_utils.CENTROID_LAT_COLUMN, tracking_utils.CENTROID_LNG_COLUMN
]

TRACKING_DIR_ARG_NAME = 'input_tracking_dir_name'
FIRST_DATE_ARG_NAME = 'first_spc_date_string'
LAST_DATE_ARG_NAME = 'last_spc_date_string'
COLOUR_MAP_ARG_NAME = 'colour_map_name'
MIN_LATITUDE_ARG_NAME = 'min_plot_latitude_deg'
MAX_LATITUDE_ARG_NAME = 'max_plot_latitude_deg'
MIN_LONGITUDE_ARG_NAME = 'min_plot_longitude_deg'
MAX_LONGITUDE_ARG_NAME = 'max_plot_longitude_deg'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

TRACKING_DIR_HELP_STRING = (
    'Name of top-level directory with storm tracks.  Files therein will be '
    'found by `storm_tracking_io.find_processed_files_one_spc_date` and read by'
    ' `storm_tracking_io.read_many_processed_files`.')

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  Storm tracks will be plotted for the period'
    ' `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

COLOUR_MAP_HELP_STRING = (
    'Name of colour scheme (tracks will be coloured by time, according to this '
    'scheme).  This name must be accepted by pyplot, since the command '
    '`pyplot.cm.get_cmap` will be used to convert it from a name to an object.'
    '  If you want random colours instead of colouring by time, make this '
    'empty.')

LATITUDE_HELP_STRING = (
    'Latitude (deg N, in range -90...90).  Plotting area will be '
    '`{0:s}`...`{1:s}`.  To let plotting area be determined by data, make this '
    '{2:d}.'
).format(MIN_LATITUDE_ARG_NAME, MAX_LATITUDE_ARG_NAME, SENTINEL_VALUE)

LONGITUDE_HELP_STRING = (
    'Longitude (deg E, in range 0...360).  Plotting area will be '
    '`{0:s}`...`{1:s}`.  To let plotting area be determined by data, make this '
    '{2:d}.'
).format(MIN_LONGITUDE_ARG_NAME, MAX_LONGITUDE_ARG_NAME, SENTINEL_VALUE)

OUTPUT_FILE_HELP_STRING = 'Path to output file (figure will be saved here).'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_DIR_ARG_NAME, type=str, required=True,
    help=TRACKING_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + COLOUR_MAP_ARG_NAME, type=str, required=False, default='YlOrRd',
    help=COLOUR_MAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_LATITUDE_ARG_NAME, type=float, required=False,
    default=SENTINEL_VALUE, help=LATITUDE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_LATITUDE_ARG_NAME, type=float, required=False,
    default=SENTINEL_VALUE, help=LATITUDE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_LONGITUDE_ARG_NAME, type=float, required=False,
    default=SENTINEL_VALUE, help=LONGITUDE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_LONGITUDE_ARG_NAME, type=float, required=False,
    default=SENTINEL_VALUE, help=LONGITUDE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _run(top_tracking_dir_name, first_spc_date_string, last_spc_date_string,
         colour_map_name, min_plot_latitude_deg, max_plot_latitude_deg,
         min_plot_longitude_deg, max_plot_longitude_deg, output_file_name):
    """Plots storm tracks for a continuous time period.

    This is effectively the main method.

    :param top_tracking_dir_name: See documentation at top of file.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param colour_map_name: Same.
    :param min_plot_latitude_deg: Same.
    :param max_plot_latitude_deg: Same.
    :param min_plot_longitude_deg: Same.
    :param max_plot_longitude_deg: Same.
    :param output_file_name: Same.
    """

    if colour_map_name in ['', 'None']:
        colour_map_object = None
    else:
        colour_map_object = pyplot.cm.get_cmap(colour_map_name)

    if min_plot_latitude_deg <= SENTINEL_VALUE:
        min_plot_latitude_deg = None
    if max_plot_latitude_deg <= SENTINEL_VALUE:
        max_plot_latitude_deg = None
    if min_plot_longitude_deg <= SENTINEL_VALUE:
        min_plot_longitude_deg = None
    if max_plot_longitude_deg <= SENTINEL_VALUE:
        max_plot_longitude_deg = None

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string)

    list_of_storm_object_tables = []

    for this_spc_date_string in spc_date_strings:
        these_file_names = tracking_io.find_processed_files_one_spc_date(
            top_processed_dir_name=top_tracking_dir_name,
            tracking_scale_metres2=
            echo_top_tracking.DUMMY_TRACKING_SCALE_METRES2,
            data_source=tracking_utils.SEGMOTION_SOURCE_ID,
            spc_date_string=this_spc_date_string, raise_error_if_missing=False
        )[0]

        if len(these_file_names) == 0:
            continue

        this_storm_object_table = tracking_io.read_many_processed_files(
            these_file_names
        )[REQUIRED_COLUMNS]

        list_of_storm_object_tables.append(this_storm_object_table)

        if this_spc_date_string != spc_date_strings[-1]:
            print MINOR_SEPARATOR_STRING

        if len(list_of_storm_object_tables) == 1:
            continue

        list_of_storm_object_tables[-1] = list_of_storm_object_tables[-1].align(
            list_of_storm_object_tables[0], axis=1
        )[0]

    print SEPARATOR_STRING
    storm_object_table = pandas.concat(
        list_of_storm_object_tables, axis=0, ignore_index=True)

    if min_plot_latitude_deg is None:
        min_plot_latitude_deg = numpy.min(
            storm_object_table[tracking_utils.CENTROID_LAT_COLUMN].values
        ) - LATLNG_BUFFER_DEG

    if max_plot_latitude_deg is None:
        max_plot_latitude_deg = numpy.max(
            storm_object_table[tracking_utils.CENTROID_LAT_COLUMN].values
        ) + LATLNG_BUFFER_DEG

    if min_plot_longitude_deg is None:
        min_plot_longitude_deg = numpy.min(
            storm_object_table[tracking_utils.CENTROID_LNG_COLUMN].values
        ) - LATLNG_BUFFER_DEG

    if max_plot_longitude_deg is None:
        max_plot_longitude_deg = numpy.max(
            storm_object_table[tracking_utils.CENTROID_LNG_COLUMN].values
        ) + LATLNG_BUFFER_DEG

    _, axes_object, basemap_object = (
        plotting_utils.init_equidistant_cylindrical_map(
            min_latitude_deg=min_plot_latitude_deg,
            max_latitude_deg=max_plot_latitude_deg,
            min_longitude_deg=min_plot_longitude_deg,
            max_longitude_deg=max_plot_longitude_deg, resolution_string='i')
    )

    parallel_spacing_deg = numpy.round(
        (max_plot_latitude_deg - min_plot_latitude_deg) / (NUM_PARALLELS - 1)
    )
    meridian_spacing_deg = numpy.round(
        (max_plot_longitude_deg - min_plot_longitude_deg) / (NUM_MERIDIANS - 1)
    )

    plotting_utils.plot_coastlines(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR)
    plotting_utils.plot_countries(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR)
    plotting_utils.plot_states_and_provinces(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR)
    plotting_utils.plot_parallels(
        basemap_object=basemap_object, axes_object=axes_object,
        bottom_left_lat_deg=-90., upper_right_lat_deg=90.,
        parallel_spacing_deg=parallel_spacing_deg)
    plotting_utils.plot_meridians(
        basemap_object=basemap_object, axes_object=axes_object,
        bottom_left_lng_deg=0., upper_right_lng_deg=360.,
        meridian_spacing_deg=meridian_spacing_deg)

    storm_plotting.plot_storm_tracks(
        storm_object_table=storm_object_table, axes_object=axes_object,
        basemap_object=basemap_object, colour_map_object=colour_map_object)

    print 'Saving figure to: "{0:s}"...'.format(output_file_name)
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_tracking_dir_name=getattr(INPUT_ARG_OBJECT, TRACKING_DIR_ARG_NAME),
        first_spc_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        colour_map_name=getattr(INPUT_ARG_OBJECT, COLOUR_MAP_ARG_NAME),
        min_plot_latitude_deg=getattr(INPUT_ARG_OBJECT, MIN_LATITUDE_ARG_NAME),
        max_plot_latitude_deg=getattr(INPUT_ARG_OBJECT, MAX_LATITUDE_ARG_NAME),
        min_plot_longitude_deg=getattr(
            INPUT_ARG_OBJECT, MIN_LONGITUDE_ARG_NAME),
        max_plot_longitude_deg=getattr(
            INPUT_ARG_OBJECT, MAX_LONGITUDE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
