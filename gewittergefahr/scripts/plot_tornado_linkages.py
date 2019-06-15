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

# TODO(thunderhoser): Put some of this code in linkage_plotting.py.

LOG_MESSAGE_TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

SENTINEL_VALUE = -9999
LATLNG_TOLERANCE_DEG = 0.001
MAX_LINK_TIME_SECONDS = 3600

FONT_SIZE = 12
COLOUR_MAP_OBJECT = pyplot.cm.get_cmap('YlOrRd')

TORNADO_START_MARKER_TYPE = 'o'
TORNADO_END_MARKER_TYPE = 's'
TORNADO_MARKER_SIZE = 16
TORNADO_MARKER_EDGE_WIDTH = 1

NUM_PARALLELS = 8
NUM_MERIDIANS = 6
LATLNG_BUFFER_DEG = 0.5
BORDER_COLOUR = numpy.full(3, 0.)
FIGURE_RESOLUTION_DPI = 300

SHORT_TORNADO_ID_COLUMN = 'short_tornado_id_string'

LINKAGE_DIR_ARG_NAME = 'input_linkage_dir_name'
GENESIS_ONLY_ARG_NAME = 'genesis_only'
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

GENESIS_ONLY_HELP_STRING = (
    'Boolean flag.  If 1, will plot linkages only to tornadogenesis events.  If'
    ' 0, will plot linkages to tornado occurrences.')

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

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + LINKAGE_DIR_ARG_NAME, type=str, required=True,
    help=LINKAGE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + GENESIS_ONLY_ARG_NAME, type=int, required=False, default=1,
    help=GENESIS_ONLY_HELP_STRING)

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


def _long_to_short_tornado_ids(long_id_strings):
    """Converts long tornado IDs to short IDs.

    N = number of IDs

    :param long_id_strings: length-N list of long IDs.
    :return: short_id_strings: length-N list of short IDs.
    """

    unique_long_id_strings, orig_to_unique_indices = numpy.unique(
        numpy.array(long_id_strings), return_inverse=True
    )

    short_id_strings = [''] * len(long_id_strings)

    for i in range(len(unique_long_id_strings)):
        these_indices = numpy.where(orig_to_unique_indices == i)[0]

        for j in these_indices:
            short_id_strings[j] = '{0:d}'.format(i)

    return short_id_strings


def _plot_tornadoes(tornado_table, storm_to_tornadoes_table, axes_object,
                    basemap_object):
    """Plots start/end point of each tornado.

    :param tornado_table: pandas DataFrame returned by
        `linkage.read_linkage_file`.
    :param storm_to_tornadoes_table: Same.
    :param axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param basemap_object: Basemap handle (instance of
        `mpl_toolkits.basemap.Basemap`).
    """

    first_storm_time_unix_sec = numpy.min(
        storm_to_tornadoes_table[tracking_utils.VALID_TIME_COLUMN].values
    )
    last_storm_time_unix_sec = numpy.max(
        storm_to_tornadoes_table[tracking_utils.VALID_TIME_COLUMN].values
    )
    colour_norm_object = pyplot.Normalize(
        first_storm_time_unix_sec, last_storm_time_unix_sec
    )
    
    tornado_id_strings, orig_to_unique_indices = numpy.unique(
        tornado_table[tornado_io.TORNADO_ID_COLUMN].values, return_inverse=True
    )
    
    num_tornadoes = len(tornado_id_strings)

    short_tornado_id_strings = [None] * num_tornadoes
    start_times_unix_sec = numpy.full(num_tornadoes, -1, dtype=int)
    start_latitudes_deg = numpy.full(num_tornadoes, numpy.nan)
    start_longitudes_deg = numpy.full(num_tornadoes, numpy.nan)
    end_times_unix_sec = numpy.full(num_tornadoes, -1, dtype=int)
    end_latitudes_deg = numpy.full(num_tornadoes, numpy.nan)
    end_longitudes_deg = numpy.full(num_tornadoes, numpy.nan)
    
    for j in range(num_tornadoes):
        these_indices = numpy.where(orig_to_unique_indices == j)[0]
        short_tornado_id_strings[j] = tornado_table[
            SHORT_TORNADO_ID_COLUMN
        ].values[these_indices[0]]

        this_subindex = numpy.argmin(
            tornado_table[linkage.EVENT_TIME_COLUMN].values[these_indices]
        )
        this_start_index = these_indices[this_subindex]

        start_times_unix_sec[j] = tornado_table[
            linkage.EVENT_TIME_COLUMN].values[this_start_index]
        start_latitudes_deg[j] = tornado_table[
            linkage.EVENT_LATITUDE_COLUMN].values[this_start_index]
        start_longitudes_deg[j] = tornado_table[
            linkage.EVENT_LONGITUDE_COLUMN].values[this_start_index]

        this_subindex = numpy.argmax(
            tornado_table[linkage.EVENT_TIME_COLUMN].values[these_indices]
        )
        this_end_index = these_indices[this_subindex]

        end_times_unix_sec[j] = tornado_table[
            linkage.EVENT_TIME_COLUMN].values[this_end_index]
        end_latitudes_deg[j] = tornado_table[
            linkage.EVENT_LATITUDE_COLUMN].values[this_end_index]
        end_longitudes_deg[j] = tornado_table[
            linkage.EVENT_LONGITUDE_COLUMN].values[this_end_index]

    start_time_colour_matrix = COLOUR_MAP_OBJECT(colour_norm_object(
        start_times_unix_sec
    ))
    end_time_colour_matrix = COLOUR_MAP_OBJECT(colour_norm_object(
        end_times_unix_sec
    ))

    start_x_coords_metres, start_y_coords_metres = basemap_object(
        start_longitudes_deg, start_latitudes_deg)
    end_x_coords_metres, end_y_coords_metres = basemap_object(
        end_longitudes_deg, end_latitudes_deg)

    for j in range(num_tornadoes):
        axes_object.plot(
            start_x_coords_metres[j], start_y_coords_metres[j],
            linestyle='None', marker=TORNADO_START_MARKER_TYPE,
            markersize=TORNADO_MARKER_SIZE,
            markeredgewidth=TORNADO_MARKER_EDGE_WIDTH,
            markerfacecolor=plotting_utils.colour_from_numpy_to_tuple(
                start_time_colour_matrix[j, :-1]
            ),
            markeredgecolor='k'
        )

        axes_object.text(
            start_x_coords_metres[j], start_y_coords_metres[j],
            short_tornado_id_strings[j], fontsize=FONT_SIZE, color='k',
            horizontalalignment='center', verticalalignment='center')

        # axes_object.plot(
        #     end_x_coords_metres[j], end_y_coords_metres[j], linestyle='None',
        #     marker=TORNADO_END_MARKER_TYPE, markersize=TORNADO_MARKER_SIZE,
        #     markeredgewidth=TORNADO_MARKER_EDGE_WIDTH,
        #     markerfacecolor=plotting_utils.colour_from_numpy_to_tuple(
        #         end_time_colour_matrix[j, :-1]
        #     ),
        #     markeredgecolor='k'
        # )
        #
        # axes_object.text(
        #     end_x_coords_metres[j], end_y_coords_metres[j],
        #     short_tornado_id_strings[j], fontsize=FONT_SIZE, color='k',
        #     horizontalalignment='center', verticalalignment='center')


def _plot_linkages_one_storm_object(
        storm_to_tornadoes_table, storm_object_index, tornado_table,
        axes_object, basemap_object):
    """Plots linkages for one storm object.

    :param storm_to_tornadoes_table: pandas DataFrame returned by
        `linkage.read_linkage_file`.
    :param storm_object_index: Will plot linkages for the [k]th storm object, or
        [k]th row of `storm_to_tornadoes_table`.
    :param tornado_table: pandas DataFrame returned by
        `linkage.read_linkage_file`.
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

    linked_id_strings = [
        storm_to_tornadoes_table[linkage.TORNADO_IDS_COLUMN].values[i][k]
        for k in good_indices
    ]

    linked_short_id_strings = []

    for this_id_string in linked_id_strings:
        these_indices = numpy.where(
            tornado_table[tornado_io.TORNADO_ID_COLUMN].values == this_id_string
        )[0]

        if len(these_indices) == 0:
            continue

        linked_short_id_strings.append(
            tornado_table[SHORT_TORNADO_ID_COLUMN].values[these_indices[0]]
        )

    if len(linked_short_id_strings) == 0:
        return

    x_coord_metres, y_coord_metres = basemap_object(
        storm_to_tornadoes_table[
            tracking_utils.CENTROID_LONGITUDE_COLUMN].values[i],
        storm_to_tornadoes_table[
            tracking_utils.CENTROID_LATITUDE_COLUMN].values[i]
    )

    axes_object.plot(
        x_coord_metres, y_coord_metres, linestyle='None',
        marker=TORNADO_START_MARKER_TYPE, markersize=TORNADO_MARKER_SIZE / 3,
        markeredgewidth=TORNADO_MARKER_EDGE_WIDTH / 3,
        markerfacecolor='k', markeredgecolor='k')

    axes_object.text(
        x_coord_metres, y_coord_metres, ','.join(linked_short_id_strings),
        fontsize=FONT_SIZE, color='k',
        horizontalalignment='left', verticalalignment='top')


def _run(top_linkage_dir_name, genesis_only, first_spc_date_string,
         last_spc_date_string, min_plot_latitude_deg, max_plot_latitude_deg,
         min_plot_longitude_deg, max_plot_longitude_deg, output_file_name):
    """Plots tornado reports, storm tracks, and linkages.

    This is effectively the main method.

    :param top_linkage_dir_name: See documentation at top of file.
    :param genesis_only: Same.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param min_plot_latitude_deg: Same.
    :param max_plot_latitude_deg: Same.
    :param min_plot_longitude_deg: Same.
    :param max_plot_longitude_deg: Same.
    :param output_file_name: Same.
    """

    event_type_string = (
        linkage.TORNADOGENESIS_EVENT_STRING if genesis_only
        else linkage.TORNADO_EVENT_STRING
    )

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

    list_of_linkage_tables = []
    list_of_tornado_tables = []

    for this_spc_date_string in spc_date_strings:
        this_file_name = linkage.find_linkage_file(
            top_directory_name=top_linkage_dir_name,
            event_type_string=event_type_string,
            spc_date_string=this_spc_date_string, raise_error_if_missing=False)

        if not os.path.isfile(this_file_name):
            continue

        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_linkage_table, _, this_tornado_table = linkage.read_linkage_file(
            this_file_name)

        list_of_linkage_tables.append(this_linkage_table)
        list_of_tornado_tables.append(this_tornado_table)

        if len(list_of_linkage_tables) == 1:
            continue

        list_of_linkage_tables[-1] = list_of_linkage_tables[-1].align(
            list_of_linkage_tables[0], axis=1
        )[0]

        list_of_tornado_tables[-1] = list_of_tornado_tables[-1].align(
            list_of_tornado_tables[0], axis=1
        )[0]

    print(SEPARATOR_STRING)

    storm_to_tornadoes_table = pandas.concat(
        list_of_linkage_tables, axis=0, ignore_index=True)
    tornado_table = pandas.concat(
        list_of_tornado_tables, axis=0, ignore_index=True)

    tornado_table = tornado_table.assign(**{
        SHORT_TORNADO_ID_COLUMN: _long_to_short_tornado_ids(
            tornado_table[tornado_io.TORNADO_ID_COLUMN].values
        )
    })

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

    # TODO(thunderhoser): This treats each track segment as an individual thing.
    # Need to consider tornadoes as a whole.  Probably should write a method to
    # convert from "events" to tornadoes.
    tornado_table = tornado_table.loc[
        (tornado_table[linkage.EVENT_LATITUDE_COLUMN] >= min_plot_latitude_deg)
        &
        (tornado_table[linkage.EVENT_LATITUDE_COLUMN] <= max_plot_longitude_deg)
    ]

    tornado_table = tornado_table.loc[
        (tornado_table[linkage.EVENT_LONGITUDE_COLUMN]
         >= min_plot_longitude_deg)
        & (tornado_table[linkage.EVENT_LONGITUDE_COLUMN]
           <= max_plot_longitude_deg)
    ]

    print(SEPARATOR_STRING)

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

    print('Plotting tornado markers...')
    _plot_tornadoes(
        tornado_table=tornado_table,
        storm_to_tornadoes_table=storm_to_tornadoes_table,
        axes_object=axes_object, basemap_object=basemap_object)

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
        genesis_only=bool(getattr(INPUT_ARG_OBJECT, GENESIS_ONLY_ARG_NAME)),
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
