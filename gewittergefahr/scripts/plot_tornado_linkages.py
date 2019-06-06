"""Plots tornado reports, storm tracks, and linkages."""

import os.path
import argparse
import numpy
import pandas
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_io import tornado_io
from gewittergefahr.gg_utils import linkage
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import storm_plotting

LOG_MESSAGE_TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

SENTINEL_VALUE = -9999
LATLNG_TOLERANCE_DEG = 0.001
MAX_LINK_TIME_SECONDS = 3600

FONT_SIZE = 12
COLOUR_MAP_OBJECT = pyplot.cm.get_cmap('YlOrRd')

TORNADO_MARKER_TYPE = 'o'
TORNADO_MARKER_SIZE = 10
TORNADO_MARKER_EDGE_WIDTH = 1
TORNADO_MARKER_COLOUR = numpy.full(3, 0.)

NUM_PARALLELS = 8
NUM_MERIDIANS = 6
LATLNG_BUFFER_DEG = 0.5
BORDER_COLOUR = numpy.full(3, 0.)
FIGURE_RESOLUTION_DPI = 300

TORNADO_ID_COLUMN = 'tornado_id_string'

LINKAGE_DIR_ARG_NAME = 'input_linkage_dir_name'
TORNADO_DIR_ARG_NAME = 'input_tornado_dir_name'
FIRST_DATE_ARG_NAME = 'first_spc_date_string'
LAST_DATE_ARG_NAME = 'last_spc_date_string'
MIN_LATITUDE_ARG_NAME = 'min_plot_latitude_deg'
MAX_LATITUDE_ARG_NAME = 'max_plot_latitude_deg'
MIN_LONGITUDE_ARG_NAME = 'min_plot_longitude_deg'
MAX_LONGITUDE_ARG_NAME = 'max_plot_longitude_deg'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

LINKAGE_DIR_HELP_STRING = (
    'Name of top-level directory with linkage files.  Files therein will be '
    'found by `linkage.find_linkage_file` and read by '
    '`linkage.read_linkage_file`.')

TORNADO_DIR_HELP_STRING = (
    'Name of directory with tornado observations.  Files therein will be found '
    'by `tornado_io.find_processed_file` and read by '
    '`tornado_io.read_processed_file`.')

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  Linkages will be plotted for the period '
    '`{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

LATITUDE_HELP_STRING = (
    'Latitude (deg N, in range -90...90).  Plotting area will be '
    '`{0:s}`...`{1:s}`.  To let plotting area be determined by storm tracks, '
    'leave this alone.'
).format(MIN_LATITUDE_ARG_NAME, MAX_LATITUDE_ARG_NAME, SENTINEL_VALUE)

LONGITUDE_HELP_STRING = (
    'Longitude (deg E, in range 0...360).  Plotting area will be '
    '`{0:s}`...`{1:s}`.  To let plotting area be determined by storm tracks, '
    'leave this alone.'
).format(MIN_LONGITUDE_ARG_NAME, MAX_LONGITUDE_ARG_NAME, SENTINEL_VALUE)

OUTPUT_FILE_HELP_STRING = 'Path to output file (figure will be saved here).'

DEFAULT_TORNADO_DIR_NAME = (
    '/condo/swatwork/ralager/tornado_observations/processed')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + LINKAGE_DIR_ARG_NAME, type=str, required=True,
    help=LINKAGE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TORNADO_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_TORNADO_DIR_NAME, help=TORNADO_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

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


def _read_tornado_reports(
        tornado_dir_name, min_plot_latitude_deg, max_plot_latitude_deg,
        min_plot_longitude_deg, max_plot_longitude_deg,
        storm_to_tornadoes_table):
    """Reads tornado reports.

    :param tornado_dir_name: See documentation at top of file.
    :param min_plot_latitude_deg: Same.
    :param max_plot_latitude_deg: Same.
    :param min_plot_longitude_deg: Same.
    :param max_plot_longitude_deg: Same.
    :param storm_to_tornadoes_table: pandas DataFrame returned by
        `linkage.read_linkage_file`.
    :return: tornado_table: pandas DataFrame with columns listed in
        `tornado_io.write_processed_file`, plus the following columns.
    tornado_table.tornado_id_string: Tornado ID.
    """

    # TODO(thunderhoser): Fix this "1200" hack.

    first_time_unix_sec = -1200 + numpy.min(
        storm_to_tornadoes_table[tracking_utils.VALID_TIME_COLUMN].values
    )
    last_time_unix_sec = 1200 + numpy.max(
        storm_to_tornadoes_table[tracking_utils.VALID_TIME_COLUMN].values
    )

    first_year = int(
        time_conversion.unix_sec_to_string(first_time_unix_sec, '%Y')
    )
    last_year = int(
        time_conversion.unix_sec_to_string(last_time_unix_sec, '%Y')
    )

    list_of_tornado_tables = []

    for this_year in range(first_year, last_year + 1):
        this_file_name = tornado_io.find_processed_file(
            directory_name=tornado_dir_name, year=this_year)

        print('Reading tornado reports from: "{0:s}"...'.format(
            this_file_name))

        this_tornado_table = tornado_io.read_processed_file(this_file_name)

        this_tornado_table = this_tornado_table.loc[
            (this_tornado_table[tornado_io.START_TIME_COLUMN]
             >= first_time_unix_sec)
            & (this_tornado_table[tornado_io.START_TIME_COLUMN]
               <= last_time_unix_sec)
        ]

        this_tornado_table = this_tornado_table.loc[
            (this_tornado_table[tornado_io.START_LAT_COLUMN]
             >= min_plot_latitude_deg)
            & (this_tornado_table[tornado_io.START_LAT_COLUMN]
               <= max_plot_latitude_deg)
        ]

        this_tornado_table = this_tornado_table.loc[
            (this_tornado_table[tornado_io.START_LNG_COLUMN]
             >= min_plot_longitude_deg)
            & (this_tornado_table[tornado_io.START_LNG_COLUMN]
               <= max_plot_longitude_deg)
        ]

        list_of_tornado_tables.append(this_tornado_table)
        if len(list_of_tornado_tables) == 1:
            continue

        list_of_tornado_tables[-1] = list_of_tornado_tables[-1].align(
            list_of_tornado_tables[0], axis=1
        )[0]

    tornado_table = pandas.concat(
        list_of_tornado_tables, axis=0, ignore_index=True)

    num_tornadoes = len(tornado_table.index)
    tornado_id_strings = [str(j) for j in range(num_tornadoes)]

    tornado_table = tornado_table.assign(**{
        TORNADO_ID_COLUMN: tornado_id_strings
    })

    for j in range(num_tornadoes):
        this_time_string = time_conversion.unix_sec_to_string(
            tornado_table[tornado_io.START_TIME_COLUMN].values[j],
            LOG_MESSAGE_TIME_FORMAT)

        print('Tornado ID = "{0:s}" ... time = {1:s}'.format(
            tornado_table[TORNADO_ID_COLUMN].values[j], this_time_string
        ))

    return tornado_table


def _plot_linkages_one_storm_object(
        storm_to_tornadoes_table, storm_object_index, tornado_table,
        axes_object, basemap_object):
    """Plots linkages for one storm object.

    :param storm_to_tornadoes_table: pandas DataFrame returned by
        `linkage.read_linkage_file`.
    :param storm_object_index: Will plot linkages for the [k]th storm object, or
        [k]th row of `storm_to_tornadoes_table`.
    :param tornado_table: pandas DataFrame created by `_read_tornado_reports`.
    :param axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param basemap_object: Basemap handle (instance of
        `mpl_toolkits.basemap.Basemap`).
    """

    i = storm_object_index

    linked_relative_times_sec = storm_to_tornadoes_table[
        linkage.RELATIVE_EVENT_TIMES_COLUMN].values[i]

    good_indices = numpy.where(
        linked_relative_times_sec <= MAX_LINK_TIME_SECONDS
    )[0]

    if len(good_indices) == 0:
        return

    linked_relative_times_sec = linked_relative_times_sec[good_indices]
    linked_times_unix_sec = (
        storm_to_tornadoes_table[tracking_utils.VALID_TIME_COLUMN].values[i] +
        linked_relative_times_sec
    )

    linked_latitudes_deg = storm_to_tornadoes_table[
        linkage.EVENT_LATITUDES_COLUMN].values[i][good_indices]

    linked_longitudes_deg = storm_to_tornadoes_table[
        linkage.EVENT_LONGITUDES_COLUMN].values[i][good_indices]

    linked_id_strings = []
    num_tornadoes = len(linked_latitudes_deg)

    for j in range(num_tornadoes):
        these_indices = numpy.where(
            tornado_table[tornado_io.START_TIME_COLUMN].values ==
            linked_times_unix_sec[j]
        )[0]

        these_latitude_diffs_deg = numpy.absolute(
            linked_latitudes_deg[j] -
            tornado_table[tornado_io.START_LAT_COLUMN].values[these_indices]
        )
        these_longitude_diffs_deg = numpy.absolute(
            linked_longitudes_deg[j] -
            tornado_table[tornado_io.START_LNG_COLUMN].values[these_indices]
        )

        these_subindices = numpy.where(numpy.logical_and(
            these_latitude_diffs_deg <= LATLNG_TOLERANCE_DEG,
            these_longitude_diffs_deg <= LATLNG_TOLERANCE_DEG
        ))[0]

        these_indices = these_indices[these_subindices]

        # TODO(thunderhoser): This is kind of a HACK.
        if len(these_indices) == 0:
            continue

        linked_id_strings.append(
            tornado_table[TORNADO_ID_COLUMN].values[these_indices[0]]
        )

    x_coord_metres, y_coord_metres = basemap_object(
        storm_to_tornadoes_table[
            tracking_utils.CENTROID_LONGITUDE_COLUMN].values[i],
        storm_to_tornadoes_table[
            tracking_utils.CENTROID_LATITUDE_COLUMN].values[i]
    )

    axes_object.plot(
        x_coord_metres, y_coord_metres, linestyle='None',
        marker=TORNADO_MARKER_TYPE, markersize=TORNADO_MARKER_SIZE / 3,
        markeredgewidth=TORNADO_MARKER_EDGE_WIDTH / 3,
        markerfacecolor='k', markeredgecolor='k')

    axes_object.text(
        x_coord_metres, y_coord_metres, ','.join(linked_id_strings),
        fontsize=FONT_SIZE, color='k',
        horizontalalignment='left', verticalalignment='top')


def _run(top_linkage_dir_name, tornado_dir_name, first_spc_date_string,
         last_spc_date_string, min_plot_latitude_deg, max_plot_latitude_deg,
         min_plot_longitude_deg, max_plot_longitude_deg, output_file_name):
    """Plots tornado reports, storm tracks, and linkages.

    This is effectively the main method.

    :param top_linkage_dir_name: See documentation at top of file.
    :param tornado_dir_name: Same.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param min_plot_latitude_deg: Same.
    :param max_plot_latitude_deg: Same.
    :param min_plot_longitude_deg: Same.
    :param max_plot_longitude_deg: Same.
    :param output_file_name: Same.
    """

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

    list_of_tables = []

    for this_spc_date_string in spc_date_strings:
        this_file_name = linkage.find_linkage_file(
            top_directory_name=top_linkage_dir_name,
            event_type_string=linkage.TORNADO_EVENT_STRING,
            spc_date_string=this_spc_date_string, raise_error_if_missing=False)

        if not os.path.isfile(this_file_name):
            continue

        print('Reading data from: "{0:s}"...'.format(this_file_name))
        list_of_tables.append(
            linkage.read_linkage_file(this_file_name)
        )

        if len(list_of_tables) == 1:
            continue

        list_of_tables[-1] = list_of_tables[-1].align(
            list_of_tables[0], axis=1
        )[0]

    print(SEPARATOR_STRING)
    storm_to_tornadoes_table = pandas.concat(
        list_of_tables, axis=0, ignore_index=True)

    if min_plot_latitude_deg is None:
        min_plot_latitude_deg = numpy.min(
            storm_to_tornadoes_table[
                tracking_utils.CENTROID_LATITUDE_COLUMN].values
        ) - LATLNG_BUFFER_DEG

    if max_plot_latitude_deg is None:
        max_plot_latitude_deg = numpy.max(
            storm_to_tornadoes_table[
                tracking_utils.CENTROID_LATITUDE_COLUMN].values
        ) + LATLNG_BUFFER_DEG

    if min_plot_longitude_deg is None:
        min_plot_longitude_deg = numpy.min(
            storm_to_tornadoes_table[
                tracking_utils.CENTROID_LONGITUDE_COLUMN].values
        ) - LATLNG_BUFFER_DEG

    if max_plot_longitude_deg is None:
        max_plot_longitude_deg = numpy.max(
            storm_to_tornadoes_table[
                tracking_utils.CENTROID_LONGITUDE_COLUMN].values
        ) + LATLNG_BUFFER_DEG

    # TODO(thunderhoser): Should maybe restrict this to an inner domain.
    storm_to_tornadoes_table = storm_to_tornadoes_table.loc[
        (storm_to_tornadoes_table[tracking_utils.CENTROID_LATITUDE_COLUMN]
         >= min_plot_latitude_deg)
        & (storm_to_tornadoes_table[tracking_utils.CENTROID_LATITUDE_COLUMN]
           <= max_plot_latitude_deg)
    ]

    storm_to_tornadoes_table = storm_to_tornadoes_table.loc[
        (storm_to_tornadoes_table[tracking_utils.CENTROID_LONGITUDE_COLUMN]
         >= min_plot_longitude_deg)
        & (storm_to_tornadoes_table[tracking_utils.CENTROID_LONGITUDE_COLUMN]
           <= max_plot_longitude_deg)
    ]

    tornado_table = _read_tornado_reports(
        tornado_dir_name=tornado_dir_name,
        min_plot_latitude_deg=min_plot_latitude_deg,
        max_plot_latitude_deg=max_plot_latitude_deg,
        min_plot_longitude_deg=min_plot_longitude_deg,
        max_plot_longitude_deg=max_plot_longitude_deg,
        storm_to_tornadoes_table=storm_to_tornadoes_table)

    _, axes_object, basemap_object = (
        plotting_utils.create_equidist_cylindrical_map(
            min_latitude_deg=min_plot_latitude_deg,
            max_latitude_deg=max_plot_latitude_deg,
            min_longitude_deg=min_plot_longitude_deg,
            max_longitude_deg=max_plot_longitude_deg, resolution_string='i')
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
        num_parallels=NUM_PARALLELS)

    plotting_utils.plot_meridians(
        basemap_object=basemap_object, axes_object=axes_object,
        num_meridians=NUM_MERIDIANS)

    print('Plotting storm tracks...')
    storm_plotting.plot_storm_tracks(
        storm_object_table=storm_to_tornadoes_table, axes_object=axes_object,
        basemap_object=basemap_object, colour_map_object=COLOUR_MAP_OBJECT,
        start_marker_type=None, end_marker_type=None)

    num_tornadoes = len(tornado_table.index)

    if num_tornadoes == 0:
        print('Saving figure to: "{0:s}"...'.format(output_file_name))
        pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()
        return

    first_storm_time_unix_sec = numpy.min(
        storm_to_tornadoes_table[tracking_utils.VALID_TIME_COLUMN].values
    )
    last_storm_time_unix_sec = numpy.max(
        storm_to_tornadoes_table[tracking_utils.VALID_TIME_COLUMN].values
    )

    colour_norm_object = pyplot.Normalize(
        first_storm_time_unix_sec, last_storm_time_unix_sec)

    tornado_colour_matrix = COLOUR_MAP_OBJECT(colour_norm_object(
        tornado_table[tornado_io.START_TIME_COLUMN].values
    ))

    print('Plotting tornado markers...')
    tornado_x_coords_metres, tornado_y_coords_metres = basemap_object(
        tornado_table[tornado_io.START_LNG_COLUMN].values,
        tornado_table[tornado_io.START_LAT_COLUMN].values
    )

    for j in range(num_tornadoes):
        axes_object.plot(
            tornado_x_coords_metres[j], tornado_y_coords_metres[j],
            linestyle='None',
            marker=TORNADO_MARKER_TYPE, markersize=TORNADO_MARKER_SIZE,
            markeredgewidth=TORNADO_MARKER_EDGE_WIDTH,
            markerfacecolor=plotting_utils.colour_from_numpy_to_tuple(
                tornado_colour_matrix[j, :-1]
            ),
            markeredgecolor='k'
        )

        axes_object.text(
            tornado_x_coords_metres[j], tornado_y_coords_metres[j],
            tornado_table[TORNADO_ID_COLUMN].values[j], fontsize=FONT_SIZE,
            color='k', horizontalalignment='left', verticalalignment='top')

    print('Plotting tornado IDs with storm objects...')
    num_storm_objects = len(storm_to_tornadoes_table.index)

    for i in range(num_storm_objects):
        _plot_linkages_one_storm_object(
            storm_to_tornadoes_table=storm_to_tornadoes_table,
            storm_object_index=i, tornado_table=tornado_table,
            axes_object=axes_object, basemap_object=basemap_object)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_linkage_dir_name=getattr(INPUT_ARG_OBJECT, LINKAGE_DIR_ARG_NAME),
        tornado_dir_name=getattr(INPUT_ARG_OBJECT, TORNADO_DIR_ARG_NAME),
        first_spc_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        min_plot_latitude_deg=getattr(INPUT_ARG_OBJECT, MIN_LATITUDE_ARG_NAME),
        max_plot_latitude_deg=getattr(INPUT_ARG_OBJECT, MAX_LATITUDE_ARG_NAME),
        min_plot_longitude_deg=getattr(
            INPUT_ARG_OBJECT, MIN_LONGITUDE_ARG_NAME),
        max_plot_longitude_deg=getattr(
            INPUT_ARG_OBJECT, MAX_LONGITUDE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
