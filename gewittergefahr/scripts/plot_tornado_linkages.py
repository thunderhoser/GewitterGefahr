"""Plots tornado reports, storm tracks, and linkages."""

import os.path
import argparse
import numpy
import pandas
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_io import tornado_io
from gewittergefahr.gg_utils import linkage
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import storm_plotting

# TODO(thunderhoser): Maybe put some of this code in linkage_plotting.py.
#  Either way, methods need unit tests.

LOG_MESSAGE_TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

SENTINEL_VALUE = -9999
LATLNG_TOLERANCE_DEG = 0.001

COLOUR_MAP_NAME = 'YlOrRd'

LINKAGE_FONT_SIZE = 12
LINKAGE_FONT_COLOUR = numpy.full(3, 0.)
LINKAGE_BACKGROUND_OPACITY = 0.5

TORNADO_FONT_SIZE = 18
TORNADO_START_MARKER_TYPE = 'o'
TORNADO_END_MARKER_TYPE = 's'
TORNADO_MARKER_SIZE = 30
TORNADO_MARKER_EDGE_WIDTH = 1.5

NUM_PARALLELS = 8
NUM_MERIDIANS = 6
LATLNG_BUFFER_DEG = 0.5
BORDER_COLOUR = numpy.full(3, 0.)
FIGURE_RESOLUTION_DPI = 300

SHORT_TORNADO_ID_COLUMN = 'short_tornado_id_string'

LINKAGE_DIR_ARG_NAME = 'input_linkage_dir_name'
GENESIS_ONLY_ARG_NAME = 'genesis_only'
MAX_DISTANCE_ARG_NAME = 'max_link_distance_metres'
FIRST_DATE_ARG_NAME = 'first_spc_date_string'
LAST_DATE_ARG_NAME = 'last_spc_date_string'
NUM_COLOURS_ARG_NAME = 'num_colours'
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

MAX_DISTANCE_HELP_STRING = (
    'Max linkage distance.  Will not show linkages with greater distance.  To '
    'plot all linkages, leave this alone.')

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  Linkages will be plotted for the period '
    '`{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

NUM_COLOURS_HELP_STRING = (
    'Number of colours in colour scheme (tornado reports and storm-track '
    'segments are coloured by time).')

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
    '--' + MAX_DISTANCE_ARG_NAME, type=float, required=False, default=0.,
    help=MAX_DISTANCE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_COLOURS_ARG_NAME, type=int, required=False, default=12,
    help=NUM_COLOURS_HELP_STRING)

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


def _truncate_colour_map(
        orig_colour_map_object, num_colours, min_normalized_value=0.25,
        max_normalized_value=1.):
    """Truncates colour map.

    :param orig_colour_map_object: Original colour map (instance of
        `matplotlib.pyplot.cm`).
    :param num_colours: Number of colours in new map.
    :param min_normalized_value: Minimum normalized value for new map (in range
        0...1).
    :param max_normalized_value: Max normalized value for new map (in range
        0...1).
    :return: colour_map_object: New colour map (instance of
        `matplotlib.pyplot.cm`).
    """

    normalized_values = numpy.linspace(
        min_normalized_value, max_normalized_value, num=num_colours)
    colour_matrix = orig_colour_map_object(normalized_values)

    colour_map_name = 'trunc({0:s}, {1:f}, {2:f})'.format(
        orig_colour_map_object.name, min_normalized_value, max_normalized_value
    )

    return matplotlib.colors.LinearSegmentedColormap.from_list(
        name=colour_map_name, colors=colour_matrix, N=num_colours)


def _plot_tornadoes(tornado_table, colour_map_object, colour_norm_object,
                    genesis_only, axes_object, basemap_object):
    """Plots start/end point of each tornado.

    :param tornado_table: pandas DataFrame returned by
        `linkage.read_linkage_file`.
    :param colour_map_object: Colour map (instance of `matplotlib.pyplot.cm`).
        Tornado markers will be coloured by time.
    :param colour_norm_object: Colour-normalizer (instance of
        `matplotlib.colors.Normalize`).  Used to convert from time to colour.
    :param genesis_only: See documentation at top of file.
    :param axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param basemap_object: Basemap handle (instance of
        `mpl_toolkits.basemap.Basemap`).
    """

    start_time_colour_matrix = colour_map_object(colour_norm_object(
        tornado_table[tornado_io.START_TIME_COLUMN].values
    ))

    start_x_coords_metres, start_y_coords_metres = basemap_object(
        tornado_table[tornado_io.START_LNG_COLUMN].values,
        tornado_table[tornado_io.START_LAT_COLUMN].values
    )

    if genesis_only:
        end_time_colour_matrix = None
        end_x_coords_metres = None
        end_y_coords_metres = None
    else:
        end_time_colour_matrix = colour_map_object(colour_norm_object(
            tornado_table[tornado_io.END_TIME_COLUMN].values
        ))

        end_x_coords_metres, end_y_coords_metres = basemap_object(
            tornado_table[tornado_io.END_LNG_COLUMN].values,
            tornado_table[tornado_io.END_LAT_COLUMN].values
        )

    num_tornadoes = len(tornado_table.index)

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
            tornado_table[SHORT_TORNADO_ID_COLUMN].values[j],
            fontsize=TORNADO_FONT_SIZE, color='k', horizontalalignment='center',
            verticalalignment='center')

        if genesis_only:
            continue

        axes_object.plot(
            end_x_coords_metres[j], end_y_coords_metres[j], linestyle='None',
            marker=TORNADO_END_MARKER_TYPE, markersize=TORNADO_MARKER_SIZE,
            markeredgewidth=TORNADO_MARKER_EDGE_WIDTH,
            markerfacecolor=plotting_utils.colour_from_numpy_to_tuple(
                end_time_colour_matrix[j, :-1]
            ),
            markeredgecolor='k'
        )

        axes_object.text(
            end_x_coords_metres[j], end_y_coords_metres[j],
            tornado_table[SHORT_TORNADO_ID_COLUMN].values[j],
            fontsize=TORNADO_FONT_SIZE, fontweight='bold', color='k',
            horizontalalignment='center', verticalalignment='center')


def _plot_linkages_one_storm_object(
        storm_to_tornadoes_table, storm_object_index, tornado_table,
        colour_map_object, colour_norm_object, axes_object, basemap_object,
        max_link_distance_metres):
    """Plots linkages for one storm object.

    :param storm_to_tornadoes_table: pandas DataFrame returned by
        `linkage.read_linkage_file`.
    :param storm_object_index: Will plot linkages for the [k]th storm object, or
        [k]th row of `storm_to_tornadoes_table`.
    :param tornado_table: pandas DataFrame returned by
        `linkage.read_linkage_file`.
    :param colour_map_object: Colour map (instance of `matplotlib.pyplot.cm`).
        Text boxes will be coloured by time.
    :param colour_norm_object: Colour-normalizer (instance of
        `matplotlib.colors.Normalize`).  Used to convert from time to colour.
    :param axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param basemap_object: Basemap handle (instance of
        `mpl_toolkits.basemap.Basemap`).
    :param max_link_distance_metres: See documentation at top of file.
    """

    i = storm_object_index

    linkage_distances_metres = storm_to_tornadoes_table[
        linkage.LINKAGE_DISTANCES_COLUMN].values[i]

    good_indices = numpy.where(
        linkage_distances_metres <= max_link_distance_metres
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

    bg_colour_numpy = colour_map_object(colour_norm_object(
        storm_to_tornadoes_table[
            tracking_utils.VALID_TIME_COLUMN].values[i]
    ))

    bounding_box_dict = {
        'facecolor': plotting_utils.colour_from_numpy_to_tuple(
            bg_colour_numpy[:-1]
        ),
        'alpha': LINKAGE_BACKGROUND_OPACITY,
        'edgecolor': 'k',
        'linewidth': 1
    }

    label_string = ','.join(list(set(linked_short_id_strings)))

    axes_object.text(
        x_coord_metres, y_coord_metres, label_string,
        fontsize=LINKAGE_FONT_SIZE, color=LINKAGE_FONT_COLOUR,
        bbox=bounding_box_dict, horizontalalignment='center',
        verticalalignment='center', zorder=1e10)


def _subset_storms_by_time(storm_to_tornadoes_table, tornado_table,
                           linkage_metadata_dict, genesis_only):
    """Subsets storms by time.

    :param storm_to_tornadoes_table: pandas DataFrame returned by
        `linkage.read_linkage_file`.
    :param tornado_table: Same.
    :param linkage_metadata_dict: Dictionary returned by
        `linkage.read_linkage_file`.
    :param genesis_only: See documentation at top of file.
    :return: storm_to_tornadoes_table: Same as input but maybe with fewer rows.
    """

    storm_to_tornadoes_table = linkage._find_secondary_start_end_times(
        storm_to_tornadoes_table)

    max_start_time_unix_sec = (
        numpy.max(tornado_table[tornado_io.START_TIME_COLUMN].values) +
        linkage_metadata_dict[linkage.MAX_TIME_BEFORE_START_KEY]
    )

    if genesis_only:
        min_end_time_unix_sec = (
            numpy.min(tornado_table[tornado_io.START_TIME_COLUMN].values) -
            linkage_metadata_dict[linkage.MAX_TIME_AFTER_END_KEY]
        )
    else:
        min_end_time_unix_sec = (
            numpy.min(tornado_table[tornado_io.END_TIME_COLUMN].values) -
            linkage_metadata_dict[linkage.MAX_TIME_AFTER_END_KEY]
        )

    # max_start_time_string = time_conversion.unix_sec_to_string(
    #     max_start_time_unix_sec, LOG_MESSAGE_TIME_FORMAT)
    #
    # min_end_time_string = time_conversion.unix_sec_to_string(
    #     min_end_time_unix_sec, LOG_MESSAGE_TIME_FORMAT)
    #
    # print('Max start time = {0:s} ... min end time = {1:s}'.format(
    #     max_start_time_string, min_end_time_string))

    good_indices = numpy.where(numpy.logical_and(
        storm_to_tornadoes_table[linkage.SECONDARY_START_TIME_COLUMN]
        <= max_start_time_unix_sec,
        storm_to_tornadoes_table[linkage.SECONDARY_END_TIME_COLUMN] >=
        min_end_time_unix_sec
    ))[0]

    good_primary_id_strings = numpy.unique(
        storm_to_tornadoes_table[tracking_utils.PRIMARY_ID_COLUMN].values[
            good_indices]
    )

    return storm_to_tornadoes_table.loc[
        storm_to_tornadoes_table[tracking_utils.PRIMARY_ID_COLUMN].isin(
            good_primary_id_strings)
    ]


def _run(top_linkage_dir_name, genesis_only, max_link_distance_metres,
         first_spc_date_string, last_spc_date_string, num_colours,
         min_plot_latitude_deg, max_plot_latitude_deg,
         min_plot_longitude_deg, max_plot_longitude_deg, output_file_name):
    """Plots tornado reports, storm tracks, and linkages.

    This is effectively the main method.

    :param top_linkage_dir_name: See documentation at top of file.
    :param genesis_only: Same.
    :param max_link_distance_metres: Same.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param num_colours: Same.
    :param min_plot_latitude_deg: Same.
    :param max_plot_latitude_deg: Same.
    :param min_plot_longitude_deg: Same.
    :param max_plot_longitude_deg: Same.
    :param output_file_name: Same.
    """

    colour_map_object = _truncate_colour_map(
        orig_colour_map_object=pyplot.cm.get_cmap('YlOrRd'),
        num_colours=num_colours
    )

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
    linkage_metadata_dict = None

    for this_spc_date_string in spc_date_strings:
        this_file_name = linkage.find_linkage_file(
            top_directory_name=top_linkage_dir_name,
            event_type_string=event_type_string,
            spc_date_string=this_spc_date_string, raise_error_if_missing=False)

        if not os.path.isfile(this_file_name):
            continue

        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_linkage_table, linkage_metadata_dict, this_tornado_table = (
            linkage.read_linkage_file(this_file_name)
        )

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

    column_dict_old_to_new = {
        linkage.EVENT_TIME_COLUMN: tornado_io.TIME_COLUMN,
        linkage.EVENT_LATITUDE_COLUMN: tornado_io.LATITUDE_COLUMN,
        linkage.EVENT_LONGITUDE_COLUMN: tornado_io.LONGITUDE_COLUMN
    }

    tornado_table.rename(columns=column_dict_old_to_new, inplace=True)
    tornado_table = tornado_io.segments_to_tornadoes(tornado_table)

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

    tornado_io.subset_tornadoes(
        tornado_table=tornado_table, min_latitude_deg=min_plot_latitude_deg,
        max_latitude_deg=max_plot_latitude_deg,
        min_longitude_deg=min_plot_longitude_deg,
        max_longitude_deg=max_plot_longitude_deg)

    # TODO(thunderhoser): Make this subsetting optional.
    storm_to_tornadoes_table = _subset_storms_by_time(
        storm_to_tornadoes_table=storm_to_tornadoes_table,
        tornado_table=tornado_table,
        linkage_metadata_dict=linkage_metadata_dict, genesis_only=genesis_only)

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
        basemap_object=basemap_object, colour_map_object=colour_map_object,
        start_marker_type=None, end_marker_type=None)

    num_tornadoes = len(tornado_table.index)
    if num_tornadoes == 0:
        print('Saving figure to: "{0:s}"...'.format(output_file_name))
        pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()
        return

    colour_norm_object = pyplot.Normalize(
        numpy.min(
            storm_to_tornadoes_table[tracking_utils.VALID_TIME_COLUMN].values
        ),
        numpy.max(
            storm_to_tornadoes_table[tracking_utils.VALID_TIME_COLUMN].values
        )
    )

    print('Plotting tornado markers...')
    _plot_tornadoes(
        tornado_table=tornado_table, colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object, genesis_only=genesis_only,
        axes_object=axes_object, basemap_object=basemap_object)

    print('Plotting tornado IDs with storm objects...')
    num_storm_objects = len(storm_to_tornadoes_table.index)

    for i in range(0, num_storm_objects, 3):
        _plot_linkages_one_storm_object(
            storm_to_tornadoes_table=storm_to_tornadoes_table,
            storm_object_index=i, tornado_table=tornado_table,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object, axes_object=axes_object,
            basemap_object=basemap_object,
            max_link_distance_metres=max_link_distance_metres)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_linkage_dir_name=getattr(INPUT_ARG_OBJECT, LINKAGE_DIR_ARG_NAME),
        genesis_only=bool(getattr(INPUT_ARG_OBJECT, GENESIS_ONLY_ARG_NAME)),
        max_link_distance_metres=getattr(
            INPUT_ARG_OBJECT, MAX_DISTANCE_ARG_NAME),
        first_spc_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        num_colours=getattr(INPUT_ARG_OBJECT, NUM_COLOURS_ARG_NAME),
        min_plot_latitude_deg=getattr(INPUT_ARG_OBJECT, MIN_LATITUDE_ARG_NAME),
        max_plot_latitude_deg=getattr(INPUT_ARG_OBJECT, MAX_LATITUDE_ARG_NAME),
        min_plot_longitude_deg=getattr(
            INPUT_ARG_OBJECT, MIN_LONGITUDE_ARG_NAME),
        max_plot_longitude_deg=getattr(
            INPUT_ARG_OBJECT, MAX_LONGITUDE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
