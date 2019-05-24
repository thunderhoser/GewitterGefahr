"""Plots tornado reports that could not be linked to a storm."""

import os.path
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
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
SPC_DATE_ARG_NAME = 'spc_date_string'
MAX_DISTANCE_ARG_NAME = 'max_link_distance_metres'
RADAR_FIELD_ARG_NAME = 'radar_field_name'
RADAR_HEIGHT_ARG_NAME = 'radar_height_m_asl'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

TORNADO_DIR_HELP_STRING = (
    'Name of directory with tornado reports.  Files therein will be found by '
    '`tornado_io.find_processed_file` and read by '
    '`tornado_io.read_processed_file`.'
)
LINKAGE_DIR_HELP_STRING = (
    'Name of top-level directory with tornado linkages.  Files therein will be '
    'found by `linkage.find_linkage_file` and read by '
    '`linkage.read_linkage_file`.'
)
MYRORSS_DIR_HELP_STRING = (
    'Name of top-level directory with MYRORSS-formatted radar data.  Files '
    'therein will be found by `myrorss_and_mrms_io.find_raw_file` and read by '
    '`myrorss_and_mrms_io.read_data_from_sparse_grid_file`.'
)
SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  Unlinked tornadoes will be found and '
    'plotted for this date only.'
)
MAX_DISTANCE_HELP_STRING = (
    'Max linkage distance.  Any tornado linked with greater distance will be '
    'considered unlinked.  To use the default max linkage distance (whatever '
    'was used to create the files), leave this argument alone.'
)
RADAR_FIELD_HELP_STRING = (
    'Radar field to be plotted with unlinked tornadoes.  Must be accepted by '
    '`radar_utils.check_field_name`.'
)
RADAR_HEIGHT_HELP_STRING = (
    'Height of radar field (required only if field is "{0:s}").'
).format(radar_utils.REFL_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

DEFAULT_TORNADO_DIR_NAME = (
    '/condo/swatwork/ralager/tornado_observations/processed'
)
DEFAULT_MYRORSS_DIR_NAME = (
    '/condo/swatcommon/common/gridrad_final/myrorss_format'
)

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
        plotting_utils.init_equidistant_cylindrical_map(
            min_latitude_deg=numpy.min(grid_point_latitudes_deg),
            max_latitude_deg=numpy.max(grid_point_latitudes_deg),
            min_longitude_deg=numpy.min(grid_point_longitudes_deg),
            max_longitude_deg=numpy.max(grid_point_longitudes_deg),
            resolution_string='i'
        )[1:]
    )

    parallel_spacing_deg = (
        (basemap_object.urcrnrlat - basemap_object.llcrnrlat) /
        (NUM_PARALLELS - 1)
    )
    meridian_spacing_deg = (
        (basemap_object.urcrnrlon - basemap_object.llcrnrlon) /
        (NUM_MERIDIANS - 1)
    )

    if parallel_spacing_deg < 1.:
        parallel_spacing_deg = number_rounding.round_to_nearest(
            parallel_spacing_deg, 0.1)
    else:
        parallel_spacing_deg = numpy.round(parallel_spacing_deg)

    if meridian_spacing_deg < 1.:
        meridian_spacing_deg = number_rounding.round_to_nearest(
            meridian_spacing_deg, 0.1)
    else:
        meridian_spacing_deg = numpy.round(meridian_spacing_deg)

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
        markerfacecolor=TORNADO_MARKER_COLOUR,
        markeredgecolor=TORNADO_MARKER_COLOUR)

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
         spc_date_string, max_link_distance_metres, radar_field_name,
         radar_height_m_asl, output_dir_name):
    """Plots tornado reports that could not be linked to a storm.

    This is effectively the main method.

    :param tornado_dir_name: See documentation at top of file.
    :param top_linkage_dir_name: Same.
    :param top_myrorss_dir_name: Same.
    :param spc_date_string: Same.
    :param max_link_distance_metres: Same.
    :param radar_field_name: Same.
    :param radar_height_m_asl: Same.
    :param output_dir_name: Same.
    """

    if max_link_distance_metres <= 0.:
        max_link_distance_metres = numpy.inf

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    linkage_file_name = linkage.find_linkage_file(
        top_directory_name=top_linkage_dir_name,
        event_type_string=linkage.TORNADO_EVENT_STRING,
        spc_date_string=spc_date_string)

    print('Reading data from: "{0:s}"...'.format(linkage_file_name))
    storm_to_tornadoes_table = linkage.read_linkage_file(linkage_file_name)

    num_storm_objects = len(storm_to_tornadoes_table.index)
    if num_storm_objects == 0:
        print('No storms for SPC date "{0:s}".  There is nothing to do!'.format(
            spc_date_string))

        return

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

    print('Finding linked tornadoes...')

    linked_tornado_times_unix_sec = numpy.array([], dtype=int)
    linked_tornado_latitudes_deg = numpy.array([])
    linked_tornado_longitudes_deg = numpy.array([])

    for i in range(num_storm_objects):
        this_num_tornadoes = len(
            storm_to_tornadoes_table[linkage.EVENT_LATITUDES_COLUMN].values[i]
        )

        if this_num_tornadoes == 0:
            continue

        these_link_distances_metres = storm_to_tornadoes_table[
            linkage.LINKAGE_DISTANCES_COLUMN].values[i]

        these_good_indices = numpy.where(
            these_link_distances_metres <= max_link_distance_metres
        )[0]

        these_times_unix_sec = (
            storm_to_tornadoes_table[tracking_utils.VALID_TIME_COLUMN].values[i]
            + storm_to_tornadoes_table[
                linkage.RELATIVE_EVENT_TIMES_COLUMN
            ].values[i][these_good_indices]
        )

        these_latitudes_deg = storm_to_tornadoes_table[
            linkage.EVENT_LATITUDES_COLUMN
        ].values[i][these_good_indices]

        these_longitudes_deg = storm_to_tornadoes_table[
            linkage.EVENT_LONGITUDES_COLUMN
        ].values[i][these_good_indices]

        linked_tornado_times_unix_sec = numpy.concatenate((
            linked_tornado_times_unix_sec, these_times_unix_sec
        ))

        linked_tornado_latitudes_deg = numpy.concatenate((
            linked_tornado_latitudes_deg, these_latitudes_deg
        ))

        linked_tornado_longitudes_deg = numpy.concatenate((
            linked_tornado_longitudes_deg, these_longitudes_deg
        ))

    print('Finding unlinked tornadoes...')
    unlinked_indices = []

    for j in range(num_tornadoes):
        if len(linked_tornado_times_unix_sec) == 0:
            this_tornado_time_string = time_conversion.unix_sec_to_string(
                tornado_table[linkage.EVENT_TIME_COLUMN].values[j],
                TIME_FORMAT)

            print((
                'Tornado at {0:s} was not linked.  Nearest linked tornado is '
                'N/A, because no tornadoes were linked.'
            ).format(this_tornado_time_string))

            unlinked_indices.append(j)
            continue

        these_time_diffs_sec = numpy.absolute(
            linked_tornado_times_unix_sec -
            tornado_table[linkage.EVENT_TIME_COLUMN].values[j]
        )

        these_indices = numpy.where(these_time_diffs_sec == 0)[0]

        if len(these_indices) == 0:
            this_tornado_time_string = time_conversion.unix_sec_to_string(
                tornado_table[linkage.EVENT_TIME_COLUMN].values[j],
                TIME_FORMAT)

            this_nearest_index = numpy.argmin(these_time_diffs_sec)

            this_nearest_time_string = time_conversion.unix_sec_to_string(
                linked_tornado_times_unix_sec[this_nearest_index],
                TIME_FORMAT)

            print((
                'Tornado at {0:s} was not linked.  Nearest linked tornado in '
                'time is at {1:s}.'
            ).format(this_tornado_time_string, this_nearest_time_string))

            unlinked_indices.append(j)
            continue

        these_latitude_diffs_deg = numpy.absolute(
            linked_tornado_latitudes_deg[these_indices] -
            tornado_table[linkage.EVENT_LATITUDE_COLUMN].values[j]
        )

        these_longitude_diffs_deg = numpy.absolute(
            linked_tornado_longitudes_deg[these_indices] -
            tornado_table[linkage.EVENT_LONGITUDE_COLUMN].values[j]
        )

        these_subindices = numpy.where(numpy.logical_and(
            these_latitude_diffs_deg <= LATLNG_TOLERANCE_DEG,
            these_longitude_diffs_deg <= LATLNG_TOLERANCE_DEG
        ))[0]

        if len(these_subindices) > 0:
            continue

        this_nearest_subindex = numpy.argmin(numpy.sqrt(
            these_latitude_diffs_deg ** 2 + these_longitude_diffs_deg ** 2
        ))

        this_nearest_index = these_indices[this_nearest_subindex]

        print((
            'Tornado at {0:.4f} deg N and {1:.4f} deg E was not linked.  '
            'Nearest linked tornado in space is at {2:.4f} deg N and '
            '{3:.4f} deg E.'
        ).format(
            tornado_table[linkage.EVENT_LATITUDE_COLUMN].values[j],
            tornado_table[linkage.EVENT_LONGITUDE_COLUMN].values[j],
            linked_tornado_latitudes_deg[this_nearest_index],
            linked_tornado_longitudes_deg[this_nearest_index]
        ))

        unlinked_indices.append(j)

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
        spc_date_string=getattr(INPUT_ARG_OBJECT, SPC_DATE_ARG_NAME),
        max_link_distance_metres=getattr(INPUT_ARG_OBJECT, MAX_DISTANCE_ARG_NAME),
        radar_field_name=getattr(INPUT_ARG_OBJECT, RADAR_FIELD_ARG_NAME),
        radar_height_m_asl=getattr(INPUT_ARG_OBJECT, RADAR_HEIGHT_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
