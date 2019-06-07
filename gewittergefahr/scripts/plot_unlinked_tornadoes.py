"""Plots tornado reports that could not be linked to a storm."""

import os.path
import argparse
from itertools import chain
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_io import tornado_io
from gewittergefahr.gg_io import myrorss_and_mrms_io
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import linkage
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import radar_sparse_to_full as radar_s2f
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import radar_plotting
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import imagemagick_utils

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

LATLNG_TOLERANCE_DEG = 1e-4
RADAR_TIME_INTERVAL_SEC = 300

TORNADO_MARKER_TYPE = '^'
TORNADO_MARKER_SIZE = 16
TORNADO_MARKER_EDGE_WIDTH = 1
TORNADO_MARKER_COLOUR = numpy.full(3, 0.)

NUM_PARALLELS = 8
NUM_MERIDIANS = 6
BORDER_COLOUR = numpy.full(3, 0.)

TITLE_FONT_SIZE = 20
FIGURE_RESOLUTION_DPI = 300

TORNADO_DIR_ARG_NAME = 'input_tornado_dir_name'
LINKAGE_DIR_ARG_NAME = 'input_linkage_dir_name'
MYRORSS_DIR_ARG_NAME = 'input_myrorss_dir_name'
GENESIS_ONLY_ARG_NAME = 'genesis_only'
SPC_DATE_ARG_NAME = 'spc_date_string'
MAX_DISTANCE_ARG_NAME = 'max_link_distance_metres'
RADAR_FIELD_ARG_NAME = 'radar_field_name'
RADAR_HEIGHT_ARG_NAME = 'radar_height_m_asl'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

TORNADO_DIR_HELP_STRING = (
    'Name of directory with tornado reports.  Files therein will be found by '
    '`tornado_io.find_processed_file` and read by '
    '`tornado_io.read_processed_file`.')

LINKAGE_DIR_HELP_STRING = (
    'Name of top-level directory with tornado linkages.  Files therein will be '
    'found by `linkage.find_linkage_file` and read by '
    '`linkage.read_linkage_file`.')

MYRORSS_DIR_HELP_STRING = (
    'Name of top-level directory with MYRORSS-formatted radar data.  Files '
    'therein will be found by `myrorss_and_mrms_io.find_raw_file` and read by '
    '`myrorss_and_mrms_io.read_data_from_sparse_grid_file`.')

GENESIS_ONLY_HELP_STRING = (
    'Boolean flag.  Will be used only to find linkage files.  If 1, will find '
    'linkage files for tornadogenesis.  If 0, will find linkage files for '
    'tornado occurrence.')

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  Unlinked tornadoes will be found and '
    'plotted for this date only.')

MAX_DISTANCE_HELP_STRING = (
    'Max linkage distance.  Any tornado linked with greater distance will be '
    'considered unlinked.  To use the default max linkage distance (whatever '
    'was used to create the files), leave this argument alone.')

RADAR_FIELD_HELP_STRING = (
    'Radar field to be plotted with unlinked tornadoes.  Must be accepted by '
    '`radar_utils.check_field_name`.')

RADAR_HEIGHT_HELP_STRING = (
    'Height of radar field (required only if field is "{0:s}").'
).format(radar_utils.REFL_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.')

DEFAULT_TORNADO_DIR_NAME = (
    '/condo/swatwork/ralager/tornado_observations/processed')
DEFAULT_MYRORSS_DIR_NAME = (
    '/condo/swatcommon/common/gridrad_final/myrorss_format')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + TORNADO_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_TORNADO_DIR_NAME, help=TORNADO_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LINKAGE_DIR_ARG_NAME, type=str, required=True,
    help=LINKAGE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MYRORSS_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_MYRORSS_DIR_NAME, help=MYRORSS_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + GENESIS_ONLY_ARG_NAME, type=int, required=False, default=1,
    help=GENESIS_ONLY_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_DISTANCE_ARG_NAME, type=float, required=False, default=-1,
    help=MAX_DISTANCE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_FIELD_ARG_NAME, type=str, required=False,
    default=radar_utils.ECHO_TOP_40DBZ_NAME, help=RADAR_FIELD_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_HEIGHT_ARG_NAME, type=str, required=False,
    default=-1, help=RADAR_HEIGHT_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _plot_tornado_and_radar(
        top_myrorss_dir_name, radar_field_name, radar_height_m_asl,
        spc_date_string, tornado_table, tornado_row, output_file_name):
    """Plots one unlinked tornado with radar field.

    :param top_myrorss_dir_name: See documentation at top of file.
    :param radar_field_name: Same.
    :param radar_height_m_asl: Same.
    :param spc_date_string: SPC date for linkage file (format "yyyymmdd").
    :param tornado_table: pandas DataFrame created by
        `linkage._read_input_tornado_reports`.
    :param tornado_row: Will plot only tornado in [j]th row of table, where j =
        `tornado_row`.
    :param output_file_name: Path to output file.  Figure will be saved here.
    """

    tornado_time_unix_sec = tornado_table[
        linkage.EVENT_TIME_COLUMN].values[tornado_row]

    radar_time_unix_sec = number_rounding.round_to_nearest(
        tornado_time_unix_sec, RADAR_TIME_INTERVAL_SEC)

    radar_spc_date_string = time_conversion.time_to_spc_date_string(
        radar_time_unix_sec)

    radar_file_name = myrorss_and_mrms_io.find_raw_file(
        top_directory_name=top_myrorss_dir_name,
        spc_date_string=radar_spc_date_string,
        unix_time_sec=radar_time_unix_sec,
        data_source=radar_utils.MYRORSS_SOURCE_ID,
        field_name=radar_field_name, height_m_asl=radar_height_m_asl,
        raise_error_if_missing=spc_date_string == radar_spc_date_string)

    if not os.path.isfile(radar_file_name):
        first_radar_time_unix_sec = number_rounding.ceiling_to_nearest(
            time_conversion.get_start_of_spc_date(spc_date_string),
            RADAR_TIME_INTERVAL_SEC
        )

        last_radar_time_unix_sec = number_rounding.floor_to_nearest(
            time_conversion.get_end_of_spc_date(spc_date_string),
            RADAR_TIME_INTERVAL_SEC
        )

        radar_time_unix_sec = max([
            radar_time_unix_sec, first_radar_time_unix_sec
        ])

        radar_time_unix_sec = min([
            radar_time_unix_sec, last_radar_time_unix_sec
        ])

        radar_file_name = myrorss_and_mrms_io.find_raw_file(
            top_directory_name=top_myrorss_dir_name,
            spc_date_string=spc_date_string,
            unix_time_sec=radar_time_unix_sec,
            data_source=radar_utils.MYRORSS_SOURCE_ID,
            field_name=radar_field_name, height_m_asl=radar_height_m_asl,
            raise_error_if_missing=True)

    radar_metadata_dict = myrorss_and_mrms_io.read_metadata_from_raw_file(
        netcdf_file_name=radar_file_name,
        data_source=radar_utils.MYRORSS_SOURCE_ID)

    sparse_grid_table = (
        myrorss_and_mrms_io.read_data_from_sparse_grid_file(
            netcdf_file_name=radar_file_name,
            field_name_orig=radar_metadata_dict[
                myrorss_and_mrms_io.FIELD_NAME_COLUMN_ORIG],
            data_source=radar_utils.MYRORSS_SOURCE_ID,
            sentinel_values=radar_metadata_dict[
                radar_utils.SENTINEL_VALUE_COLUMN]
        )
    )

    radar_matrix, grid_point_latitudes_deg, grid_point_longitudes_deg = (
        radar_s2f.sparse_to_full_grid(
            sparse_grid_table=sparse_grid_table,
            metadata_dict=radar_metadata_dict)
    )

    radar_matrix = numpy.flip(radar_matrix, axis=0)
    grid_point_latitudes_deg = grid_point_latitudes_deg[::-1]

    axes_object, basemap_object = (
        plotting_utils.create_equidist_cylindrical_map(
            min_latitude_deg=numpy.min(grid_point_latitudes_deg),
            max_latitude_deg=numpy.max(grid_point_latitudes_deg),
            min_longitude_deg=numpy.min(grid_point_longitudes_deg),
            max_longitude_deg=numpy.max(grid_point_longitudes_deg),
            resolution_string='i'
        )[1:]
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

    radar_plotting.plot_latlng_grid(
        field_matrix=radar_matrix, field_name=radar_field_name,
        axes_object=axes_object,
        min_grid_point_latitude_deg=numpy.min(grid_point_latitudes_deg),
        min_grid_point_longitude_deg=numpy.min(grid_point_longitudes_deg),
        latitude_spacing_deg=numpy.diff(grid_point_latitudes_deg[:2])[0],
        longitude_spacing_deg=numpy.diff(grid_point_longitudes_deg[:2])[0]
    )

    tornado_latitude_deg = tornado_table[
        linkage.EVENT_LATITUDE_COLUMN].values[tornado_row]

    tornado_longitude_deg = tornado_table[
        linkage.EVENT_LONGITUDE_COLUMN].values[tornado_row]

    axes_object.plot(
        tornado_longitude_deg, tornado_latitude_deg, linestyle='None',
        marker=TORNADO_MARKER_TYPE, markersize=TORNADO_MARKER_SIZE,
        markeredgewidth=TORNADO_MARKER_EDGE_WIDTH,
        markerfacecolor=plotting_utils.colour_from_numpy_to_tuple(
            TORNADO_MARKER_COLOUR),
        markeredgecolor=plotting_utils.colour_from_numpy_to_tuple(
            TORNADO_MARKER_COLOUR)
    )

    tornado_time_string = time_conversion.unix_sec_to_string(
        tornado_time_unix_sec, TIME_FORMAT)

    title_string = (
        'Unlinked tornado at {0:s}, {1:.2f} deg N, {2:.2f} deg E'
    ).format(tornado_time_string, tornado_latitude_deg, tornado_longitude_deg)

    pyplot.title(title_string, fontsize=TITLE_FONT_SIZE)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(input_file_name=output_file_name,
                                      output_file_name=output_file_name)


def _run(tornado_dir_name, top_linkage_dir_name, top_myrorss_dir_name,
         genesis_only, spc_date_string, max_link_distance_metres,
         radar_field_name, radar_height_m_asl, output_dir_name):
    """Plots tornado reports that could not be linked to a storm.

    This is effectively the main method.

    :param tornado_dir_name: See documentation at top of file.
    :param top_linkage_dir_name: Same.
    :param top_myrorss_dir_name: Same.
    :param genesis_only: Same.
    :param spc_date_string: Same.
    :param max_link_distance_metres: Same.
    :param radar_field_name: Same.
    :param radar_height_m_asl: Same.
    :param output_dir_name: Same.
    """

    event_type_string = (
        linkage.TORNADOGENESIS_EVENT_STRING if genesis_only
        else linkage.TORNADO_EVENT_STRING
    )

    if max_link_distance_metres <= 0.:
        max_link_distance_metres = numpy.inf

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    linkage_file_name = linkage.find_linkage_file(
        top_directory_name=top_linkage_dir_name,
        event_type_string=event_type_string, spc_date_string=spc_date_string)

    print('Reading data from: "{0:s}"...'.format(linkage_file_name))
    storm_to_tornadoes_table = linkage.read_linkage_file(linkage_file_name)

    num_storm_objects = len(storm_to_tornadoes_table.index)
    if num_storm_objects == 0:
        print('No storms for SPC date "{0:s}".  There is nothing to do!'.format(
            spc_date_string))

        return

    print('Removing linkages with distance > {0:.1f} metres...'.format(
        max_link_distance_metres))

    for i in range(num_storm_objects):
        these_link_distance_metres = storm_to_tornadoes_table[
            linkage.LINKAGE_DISTANCES_COLUMN].values[i]

        if len(these_link_distance_metres) == 0:
            continue

        these_good_indices = numpy.where(
            these_link_distance_metres <= max_link_distance_metres
        )[0]

        storm_to_tornadoes_table[linkage.TORNADO_IDS_COLUMN].values[i] = [
            storm_to_tornadoes_table[linkage.TORNADO_IDS_COLUMN].values[i][k]
            for k in these_good_indices
        ]

    tornado_table = linkage._read_input_tornado_reports(
        input_directory_name=tornado_dir_name,
        storm_times_unix_sec=storm_to_tornadoes_table[
            tracking_utils.VALID_TIME_COLUMN].values,
        max_time_before_storm_start_sec=
        linkage.DEFAULT_MAX_TIME_BEFORE_STORM_SEC,
        max_time_after_storm_end_sec=linkage.DEFAULT_MAX_TIME_AFTER_STORM_SEC)

    num_tornadoes = len(tornado_table.index)
    if num_tornadoes == 0:
        print((
            'No tornadoes for SPC date "{0:s}".  There is nothing to do!'
        ).format(spc_date_string))

        return

    print('\nRemoving tornadoes outside bounding box of storms...')

    min_storm_latitude_deg = numpy.min(
        storm_to_tornadoes_table[tracking_utils.CENTROID_LATITUDE_COLUMN].values
    )
    max_storm_latitude_deg = numpy.max(
        storm_to_tornadoes_table[tracking_utils.CENTROID_LATITUDE_COLUMN].values
    )
    min_storm_longitude_deg = numpy.min(
        storm_to_tornadoes_table[
            tracking_utils.CENTROID_LONGITUDE_COLUMN].values
    )
    max_storm_longitude_deg = numpy.max(
        storm_to_tornadoes_table[
            tracking_utils.CENTROID_LONGITUDE_COLUMN].values
    )

    latitude_flags = numpy.logical_and(
        tornado_table[linkage.EVENT_LATITUDE_COLUMN].values >=
        min_storm_latitude_deg,
        tornado_table[linkage.EVENT_LATITUDE_COLUMN].values <=
        max_storm_latitude_deg
    )

    longitude_flags = numpy.logical_and(
        tornado_table[linkage.EVENT_LONGITUDE_COLUMN].values >=
        min_storm_longitude_deg,
        tornado_table[linkage.EVENT_LONGITUDE_COLUMN].values <=
        max_storm_longitude_deg
    )

    good_indices = numpy.where(
        numpy.logical_and(latitude_flags, longitude_flags)
    )[0]

    tornado_table = tornado_table.iloc[good_indices]
    num_tornadoes = len(tornado_table.index)

    if num_tornadoes == 0:
        print('No tornadoes in bounding box.  There is nothing more to do!')
        return

    print('Finding unlinked tornadoes...')

    tornado_table = tornado_io.add_tornado_ids_to_table(tornado_table)

    linked_tornado_id_strings = list(chain(
        *storm_to_tornadoes_table[linkage.TORNADO_IDS_COLUMN].values
    ))

    unlinked_flags = numpy.array([
        s not in linked_tornado_id_strings
        for s in tornado_table[tornado_io.TORNADO_ID_COLUMN].values
    ], dtype=bool)

    unlinked_indices = numpy.where(unlinked_flags)[0]
    if len(unlinked_indices) == 0:
        print('All tornadoes were linked.  Success!')
        return

    print(SEPARATOR_STRING)

    unlinked_indices = numpy.array(unlinked_indices, dtype=int)
    tornado_table = tornado_table.iloc[unlinked_indices]
    num_tornadoes = len(tornado_table.index)

    for j in range(num_tornadoes):
        this_output_file_name = (
            '{0:s}/unlinked_tornado_{1:s}_{2:03d}.jpg'
        ).format(output_dir_name, spc_date_string, j)

        _plot_tornado_and_radar(
            top_myrorss_dir_name=top_myrorss_dir_name,
            radar_field_name=radar_field_name,
            radar_height_m_asl=radar_height_m_asl,
            spc_date_string=spc_date_string, tornado_table=tornado_table,
            tornado_row=j, output_file_name=this_output_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        tornado_dir_name=getattr(INPUT_ARG_OBJECT, TORNADO_DIR_ARG_NAME),
        top_linkage_dir_name=getattr(INPUT_ARG_OBJECT, LINKAGE_DIR_ARG_NAME),
        top_myrorss_dir_name=getattr(INPUT_ARG_OBJECT, MYRORSS_DIR_ARG_NAME),
        genesis_only=bool(getattr(INPUT_ARG_OBJECT, GENESIS_ONLY_ARG_NAME)),
        spc_date_string=getattr(INPUT_ARG_OBJECT, SPC_DATE_ARG_NAME),
        max_link_distance_metres=getattr(
            INPUT_ARG_OBJECT, MAX_DISTANCE_ARG_NAME),
        radar_field_name=getattr(INPUT_ARG_OBJECT, RADAR_FIELD_ARG_NAME),
        radar_height_m_asl=getattr(INPUT_ARG_OBJECT, RADAR_HEIGHT_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
