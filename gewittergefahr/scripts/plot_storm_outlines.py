"""Plots storm outlines (along with IDs) at each time step."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import radar_sparse_to_full as radar_s2f
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_io import myrorss_and_mrms_io
from gewittergefahr.gg_utils import colours
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import storm_plotting
from gewittergefahr.plotting import radar_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

DUMMY_TRACKING_SCALE_METRES2 = int(numpy.round(numpy.pi * 1e8))
DUMMY_SOURCE_NAME = tracking_utils.SEGMOTION_NAME
SENTINEL_VALUE = -9999

FILE_NAME_TIME_FORMAT = '%Y-%m-%d-%H%M%S'
NICE_TIME_FORMAT = '%H%M UTC %-d %b %Y'

NUM_PARALLELS = 8
NUM_MERIDIANS = 6
LATLNG_BUFFER_DEG = 0.5
BORDER_COLOUR = numpy.full(3, 0.)
DEFAULT_TRACK_COLOUR = numpy.full(3, 0.)
FIGURE_RESOLUTION_DPI = 300

TRACKING_DIR_ARG_NAME = 'input_tracking_dir_name'
FIRST_DATE_ARG_NAME = 'first_spc_date_string'
LAST_DATE_ARG_NAME = 'last_spc_date_string'
OUTLINE_COLOUR_ARG_NAME = 'storm_outline_colour'
OUTLINE_OPACITY_ARG_NAME = 'storm_outline_opacity'
INCLUDE_SECONDARY_ARG_NAME = 'include_secondary_ids'
MIN_LATITUDE_ARG_NAME = 'min_plot_latitude_deg'
MAX_LATITUDE_ARG_NAME = 'max_plot_latitude_deg'
MIN_LONGITUDE_ARG_NAME = 'min_plot_longitude_deg'
MAX_LONGITUDE_ARG_NAME = 'max_plot_longitude_deg'
MYRORSS_DIR_ARG_NAME = 'input_myrorss_dir_name'
RADAR_FIELD_ARG_NAME = 'radar_field_name'
RADAR_HEIGHT_ARG_NAME = 'radar_height_m_asl'
RADAR_CMAP_ARG_NAME = 'radar_colour_map_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

TRACKING_DIR_HELP_STRING = (
    'Name of top-level directory with storm tracks.  Files therein will be '
    'found by `storm_tracking_io.find_processed_files_one_spc_date` and read by'
    ' `storm_tracking_io.read_many_processed_files`.')

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  Storm outlines will be plotted for all SPC '
    'dates in the period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

OUTLINE_COLOUR_HELP_STRING = (
    'Colour of storm outlines (length-3 list of elements [R, G, B], each in '
    'range 0...255).')

OUTLINE_OPACITY_HELP_STRING = 'Opacity of storm outlines (in range 0...1).'

INCLUDE_SECONDARY_HELP_STRING = (
    'Boolean flag.  If 1, primary_secondary ID will be plotted next to each '
    'storm object.  If 0, only primary ID will be plotted.')

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

MYRORSS_DIR_HELP_STRING = (
    'Name of top-level directory with MYRORSS data.  If you do not want to '
    'underlay radar data with storm outlines, leave this alone.  Files therein '
    'will be found by `myrorss_and_mrms_io.find_many_raw_files` and read by '
    '`myrorss_and_mrms_io.read_data_from_sparse_grid_file`.')

RADAR_FIELD_HELP_STRING = (
    '[used only if `{0:s}` is not empty] Name of radar field to underlay with '
    'storm outlines.  Must be accepted by `radar_utils.check_field_name`.'
).format(MYRORSS_DIR_ARG_NAME)

RADAR_HEIGHT_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] Height of reflectivity field (metres above '
    'sea level).'
).format(RADAR_FIELD_ARG_NAME, radar_utils.REFL_NAME)

RADAR_CMAP_HELP_STRING = (
    'Name of colour map for radar field.  For example, if name is "Greys", the '
    'colour map used will be `pyplot.cm.Greys`.  This argument supports only '
    'pyplot colour maps.  To use the default colour map, make this argument '
    'empty.')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).')

DEFAULT_OUTLINE_COLOUR = numpy.array([228, 26, 28], dtype=int)
DEFAULT_OUTLINE_OPACITY = 0.5

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
    '--' + OUTLINE_COLOUR_ARG_NAME, type=int, nargs=3, required=False,
    default=DEFAULT_OUTLINE_COLOUR, help=OUTLINE_COLOUR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTLINE_OPACITY_ARG_NAME, type=float, required=False,
    default=DEFAULT_OUTLINE_OPACITY, help=OUTLINE_OPACITY_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + INCLUDE_SECONDARY_ARG_NAME, type=int, required=False,
    default=0, help=INCLUDE_SECONDARY_HELP_STRING)

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
    '--' + MYRORSS_DIR_ARG_NAME, type=str, required=False, default='',
    help=MYRORSS_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_FIELD_ARG_NAME, type=str, required=False,
    default=radar_utils.ECHO_TOP_40DBZ_NAME, help=RADAR_FIELD_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_HEIGHT_ARG_NAME, type=int, required=False, default=-1,
    help=RADAR_HEIGHT_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_CMAP_ARG_NAME, type=str, required=False, default='YlOrRd',
    help=RADAR_CMAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _get_plotting_limits(
        min_plot_latitude_deg, max_plot_latitude_deg, min_plot_longitude_deg,
        max_plot_longitude_deg, storm_object_table):
    """Returns lat-long limits for plotting.

    :param min_plot_latitude_deg: See documentation at top of file.  If
        `min_plot_latitude_deg == SENTINEL_VALUE`, it will be replaced.
        Otherwise, it will be unaltered.
    :param max_plot_latitude_deg: Same.
    :param min_plot_longitude_deg: Same.
    :param max_plot_longitude_deg: Same.
    :param storm_object_table: See doc for `storm_tracking_io.write_file`.
    :return: latitude_limits_deg: length-2 numpy array with [min, max] latitudes
        in deg N.
    :return: longitude_limits_deg: length-2 numpy array with [min, max]
        longitudes in deg E.
    """

    if min_plot_latitude_deg <= SENTINEL_VALUE:
        min_plot_latitude_deg = -LATLNG_BUFFER_DEG + numpy.min(
            storm_object_table[tracking_utils.CENTROID_LATITUDE_COLUMN].values
        )

    if max_plot_latitude_deg <= SENTINEL_VALUE:
        max_plot_latitude_deg = LATLNG_BUFFER_DEG + numpy.max(
            storm_object_table[tracking_utils.CENTROID_LATITUDE_COLUMN].values
        )

    if min_plot_longitude_deg <= SENTINEL_VALUE:
        min_plot_longitude_deg = -LATLNG_BUFFER_DEG + numpy.min(
            storm_object_table[tracking_utils.CENTROID_LONGITUDE_COLUMN].values
        )

    if max_plot_longitude_deg <= SENTINEL_VALUE:
        max_plot_longitude_deg = LATLNG_BUFFER_DEG + numpy.max(
            storm_object_table[tracking_utils.CENTROID_LONGITUDE_COLUMN].values
        )

    latitude_limits_deg = numpy.array([
        min_plot_latitude_deg, max_plot_latitude_deg
    ])
    longitude_limits_deg = numpy.array([
        min_plot_longitude_deg, max_plot_longitude_deg
    ])

    return latitude_limits_deg, longitude_limits_deg


def _find_relevant_storm_objects(storm_object_table, current_rows):
    """Finds relevant storm objects.

    "Relevant" storm objects include:

    - Current objects (those at `current_rows` in `storm_object_table`)
    - Those sharing an ID with a current object and occurring at an earlier time

    :param storm_object_table: See doc for `storm_tracking_io.write_file`.
    :param current_rows: 1-D numpy array with rows of current storm objects.
    :return: relevant_storm_object_table: Same as input but with fewer rows.
    """

    current_time_unix_sec = storm_object_table[
        tracking_utils.VALID_TIME_COLUMN
    ].values[current_rows[0]]

    current_primary_id_strings = storm_object_table[
        tracking_utils.PRIMARY_ID_COLUMN
    ].values[current_rows]

    relevant_id_flags = numpy.array([
        p in current_primary_id_strings for p in
        storm_object_table[tracking_utils.PRIMARY_ID_COLUMN].values
    ], dtype=bool)

    relevant_rows = numpy.where(numpy.logical_and(
        relevant_id_flags,
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values <=
        current_time_unix_sec
    ))[0]

    return storm_object_table.iloc[relevant_rows]


def _filter_storm_objects_latlng(
        storm_object_table, min_latitude_deg, max_latitude_deg,
        min_longitude_deg, max_longitude_deg):
    """Filters storm objects by lat-long rectangle.

    :param storm_object_table: See doc for `storm_tracking_io.write_file`.
    :param min_latitude_deg: Minimum latitude (deg N).
    :param max_latitude_deg: Max latitude (deg N).
    :param min_longitude_deg: Minimum longitude (deg E).
    :param max_longitude_deg: Max longitude (deg E).
    :return: relevant_rows: 1-D numpy array with rows of storm objects in the
        lat-long box.  These are rows in `storm_object_table`.
    """

    latitude_flags = numpy.logical_and(
        storm_object_table[tracking_utils.CENTROID_LATITUDE_COLUMN].values >=
        min_latitude_deg,
        storm_object_table[tracking_utils.CENTROID_LATITUDE_COLUMN].values <=
        max_latitude_deg
    )

    longitude_flags = numpy.logical_and(
        storm_object_table[tracking_utils.CENTROID_LONGITUDE_COLUMN].values >=
        min_longitude_deg,
        storm_object_table[tracking_utils.CENTROID_LONGITUDE_COLUMN].values <=
        max_longitude_deg
    )

    return numpy.where(numpy.logical_and(
        latitude_flags, longitude_flags
    ))[0]


def _assign_colours_to_storms(storm_object_table, radar_colour_map_object):
    """Assigns colour to each primary storm ID.  Will be used to colour tracks.

    :param storm_object_table: See doc for `storm_tracking_io.write_file`.
    :param radar_colour_map_object: See doc for
        `radar_plotting.plot_latlng_grid`.
    :return: primary_id_to_track_colour: Dictionary, where each key is a primary
        storm ID (string) and each value is an RGB colour (length-3 numpy
        array).
    """

    unique_primary_id_strings = numpy.unique(
        storm_object_table[tracking_utils.PRIMARY_ID_COLUMN].values
    )

    colour_to_exclude = numpy.array(
        radar_colour_map_object(0.5)[:-1]
    )

    rgb_matrix = colours.get_random_colours(
        num_colours=len(unique_primary_id_strings),
        colour_to_exclude_rgb=colour_to_exclude
    )

    num_colours = rgb_matrix.shape[0]
    primary_id_to_track_colour = {}

    for i in range(len(unique_primary_id_strings)):
        primary_id_to_track_colour[unique_primary_id_strings[i]] = rgb_matrix[
            numpy.mod(i, num_colours), ...
        ]

    return primary_id_to_track_colour


def _plot_storm_outlines_one_time(
        storm_object_table, valid_time_unix_sec, axes_object, basemap_object,
        storm_outline_colour, storm_outline_opacity, include_secondary_ids,
        output_dir_name, primary_id_to_track_colour=None, radar_matrix=None,
        radar_field_name=None, radar_latitudes_deg=None,
        radar_longitudes_deg=None, radar_colour_map_object=None):
    """Plots storm outlines (and may underlay radar data) at one time step.

    M = number of rows in radar grid
    N = number of columns in radar grid
    K = number of storm objects

    If `primary_id_to_track_colour is None`, all storm tracks will be the same
    colour.

    :param storm_object_table: See doc for `storm_plotting.plot_storm_outlines`.
    :param valid_time_unix_sec: Will plot storm outlines only at this time.
        Will plot tracks up to and including this time.
    :param axes_object: See doc for `storm_plotting.plot_storm_outlines`.
    :param basemap_object: Same.
    :param storm_outline_colour: Same.
    :param storm_outline_opacity: Same.
    :param include_secondary_ids: Same.
    :param output_dir_name: See documentation at top of file.
    :param primary_id_to_track_colour: Dictionary created by
        `_assign_colours_to_storms`.  If this is None, all storm tracks will be
        the same colour.
    :param radar_matrix: M-by-N numpy array of radar values.  If
        `radar_matrix is None`, radar data will simply not be plotted.
    :param radar_field_name: [used only if `radar_matrix is not None`]
        See documentation at top of file.
    :param radar_latitudes_deg: [used only if `radar_matrix is not None`]
        length-M numpy array of grid-point latitudes (deg N).
    :param radar_longitudes_deg: [used only if `radar_matrix is not None`]
        length-N numpy array of grid-point longitudes (deg E).
    :param radar_colour_map_object: [used only if `radar_matrix is not None`]
        Colour map (instance of `matplotlib.pyplot.cm`).  If None, will use
        default for the given field.
    """

    plot_storm_ids = radar_matrix is None or radar_colour_map_object is None

    min_plot_latitude_deg = basemap_object.llcrnrlat
    max_plot_latitude_deg = basemap_object.urcrnrlat
    min_plot_longitude_deg = basemap_object.llcrnrlon
    max_plot_longitude_deg = basemap_object.urcrnrlon

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

    if radar_matrix is not None:
        custom_colour_map = radar_colour_map_object is not None

        good_indices = numpy.where(numpy.logical_and(
            radar_latitudes_deg >= min_plot_latitude_deg,
            radar_latitudes_deg <= max_plot_latitude_deg
        ))[0]

        radar_latitudes_deg = radar_latitudes_deg[good_indices]
        radar_matrix = radar_matrix[good_indices, :]

        good_indices = numpy.where(numpy.logical_and(
            radar_longitudes_deg >= min_plot_longitude_deg,
            radar_longitudes_deg <= max_plot_longitude_deg
        ))[0]

        radar_longitudes_deg = radar_longitudes_deg[good_indices]
        radar_matrix = radar_matrix[:, good_indices]

        latitude_spacing_deg = radar_latitudes_deg[1] - radar_latitudes_deg[0]
        longitude_spacing_deg = (
            radar_longitudes_deg[1] - radar_longitudes_deg[0]
        )

        if radar_colour_map_object is None:
            colour_map_object, colour_norm_object = (
                radar_plotting.get_default_colour_scheme(radar_field_name)
            )
        else:
            colour_norm_object = radar_plotting.get_default_colour_scheme(
                radar_field_name
            )[-1]

            this_ratio = radar_plotting._field_to_plotting_units(
                field_matrix=1., field_name=radar_field_name)

            colour_norm_object = pyplot.Normalize(
                colour_norm_object.vmin / this_ratio,
                colour_norm_object.vmax / this_ratio)

        radar_plotting.plot_latlng_grid(
            field_matrix=radar_matrix, field_name=radar_field_name,
            axes_object=axes_object,
            min_grid_point_latitude_deg=numpy.min(radar_latitudes_deg),
            min_grid_point_longitude_deg=numpy.min(radar_longitudes_deg),
            latitude_spacing_deg=latitude_spacing_deg,
            longitude_spacing_deg=longitude_spacing_deg,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object)

        latitude_range_deg = max_plot_latitude_deg - min_plot_latitude_deg
        longitude_range_deg = max_plot_longitude_deg - min_plot_longitude_deg

        if latitude_range_deg > longitude_range_deg:
            orientation_string = 'vertical'
        else:
            orientation_string = 'horizontal'

        colour_bar_object = plotting_utils.plot_colour_bar(
            axes_object_or_matrix=axes_object, data_matrix=radar_matrix,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            orientation_string=orientation_string,
            extend_min=radar_field_name in radar_plotting.SHEAR_VORT_DIV_NAMES,
            extend_max=True, fraction_of_axis_length=0.9)

        colour_bar_object.set_label(
            radar_plotting.FIELD_NAME_TO_VERBOSE_DICT[radar_field_name]
        )

        if custom_colour_map:
            if orientation_string == 'horizontal':
                tick_values = colour_bar_object.ax.get_xticks()
            else:
                tick_values = colour_bar_object.ax.get_yticks()

            tick_label_strings = ['{0:.1f}'.format(x) for x in tick_values]
            colour_bar_object.set_ticks(tick_values)
            colour_bar_object.set_ticklabels(tick_label_strings)

    valid_time_rows = numpy.where(
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values ==
        valid_time_unix_sec
    )[0]

    this_colour = matplotlib.colors.to_rgba(
        storm_outline_colour, storm_outline_opacity)

    storm_plotting.plot_storm_outlines(
        storm_object_table=storm_object_table.iloc[valid_time_rows],
        axes_object=axes_object, basemap_object=basemap_object,
        line_colour=this_colour)

    if plot_storm_ids:
        storm_plotting.plot_storm_ids(
            storm_object_table=storm_object_table.iloc[valid_time_rows],
            axes_object=axes_object, basemap_object=basemap_object,
            plot_near_centroids=False,
            include_secondary_ids=include_secondary_ids,
            font_colour=storm_plotting.DEFAULT_FONT_COLOUR)

    if primary_id_to_track_colour is None:
        storm_plotting.plot_storm_tracks(
            storm_object_table=storm_object_table, axes_object=axes_object,
            basemap_object=basemap_object, colour_map_object=None,
            line_colour=DEFAULT_TRACK_COLOUR)
    else:
        for this_primary_id_string in primary_id_to_track_colour:
            this_storm_object_table = storm_object_table.loc[
                storm_object_table[tracking_utils.PRIMARY_ID_COLUMN] ==
                this_primary_id_string
            ]

            if len(this_storm_object_table.index) == 0:
                continue

            storm_plotting.plot_storm_tracks(
                storm_object_table=this_storm_object_table,
                axes_object=axes_object, basemap_object=basemap_object,
                colour_map_object=None,
                line_colour=primary_id_to_track_colour[this_primary_id_string]
            )

    nice_time_string = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, NICE_TIME_FORMAT)

    abbrev_time_string = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, FILE_NAME_TIME_FORMAT)

    pyplot.title('Storm objects at {0:s}'.format(nice_time_string))
    output_file_name = '{0:s}/storm_outlines_{1:s}.jpg'.format(
        output_dir_name, abbrev_time_string)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
                   bbox_inches='tight')
    pyplot.close()


def _run(top_tracking_dir_name, first_spc_date_string, last_spc_date_string,
         storm_outline_colour, storm_outline_opacity, include_secondary_ids,
         min_plot_latitude_deg, max_plot_latitude_deg, min_plot_longitude_deg,
         max_plot_longitude_deg, top_myrorss_dir_name, radar_field_name,
         radar_height_m_asl, radar_colour_map_name, output_dir_name):
    """Plots storm outlines (along with IDs) at each time step.

    This is effectively the main method.

    :param top_tracking_dir_name: See documentation at top of file.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param storm_outline_colour: Same.
    :param storm_outline_opacity: Same.
    :param include_secondary_ids: Same.
    :param min_plot_latitude_deg: Same.
    :param max_plot_latitude_deg: Same.
    :param min_plot_longitude_deg: Same.
    :param max_plot_longitude_deg: Same.
    :param top_myrorss_dir_name: Same.
    :param radar_field_name: Same.
    :param radar_height_m_asl: Same.
    :param radar_colour_map_name: Same.
    :param output_dir_name: Same.
    """

    if top_myrorss_dir_name in ['', 'None']:
        top_myrorss_dir_name = None

    if radar_field_name != radar_utils.REFL_NAME:
        radar_height_m_asl = None

    if radar_colour_map_name in ['', 'None']:
        radar_colour_map_object = None
    else:
        radar_colour_map_object = pyplot.get_cmap(radar_colour_map_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string)

    tracking_file_names = []

    for this_spc_date_string in spc_date_strings:
        tracking_file_names += (
            tracking_io.find_files_one_spc_date(
                top_tracking_dir_name=top_tracking_dir_name,
                tracking_scale_metres2=DUMMY_TRACKING_SCALE_METRES2,
                source_name=DUMMY_SOURCE_NAME,
                spc_date_string=this_spc_date_string,
                raise_error_if_missing=False
            )[0]
        )

    storm_object_table = tracking_io.read_many_files(tracking_file_names)
    print(SEPARATOR_STRING)

    latitude_limits_deg, longitude_limits_deg = _get_plotting_limits(
        min_plot_latitude_deg=min_plot_latitude_deg,
        max_plot_latitude_deg=max_plot_latitude_deg,
        min_plot_longitude_deg=min_plot_longitude_deg,
        max_plot_longitude_deg=max_plot_longitude_deg,
        storm_object_table=storm_object_table)

    min_plot_latitude_deg = latitude_limits_deg[0]
    max_plot_latitude_deg = latitude_limits_deg[1]
    min_plot_longitude_deg = longitude_limits_deg[0]
    max_plot_longitude_deg = longitude_limits_deg[1]

    if radar_colour_map_object is None:
        primary_id_to_track_colour = None
    else:
        primary_id_to_track_colour = _assign_colours_to_storms(
            storm_object_table=storm_object_table,
            radar_colour_map_object=radar_colour_map_object)

    valid_times_unix_sec = numpy.unique(
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values
    )
    num_times = len(valid_times_unix_sec)

    for i in range(num_times):
        these_current_rows = numpy.where(
            storm_object_table[tracking_utils.VALID_TIME_COLUMN].values ==
            valid_times_unix_sec[i]
        )[0]

        these_current_subrows = _filter_storm_objects_latlng(
            storm_object_table=storm_object_table.iloc[these_current_rows],
            min_latitude_deg=min_plot_latitude_deg,
            max_latitude_deg=max_plot_latitude_deg,
            min_longitude_deg=min_plot_longitude_deg,
            max_longitude_deg=max_plot_longitude_deg)

        if len(these_current_subrows) == 0:
            continue

        these_current_rows = these_current_rows[these_current_subrows]

        this_storm_object_table = _find_relevant_storm_objects(
            storm_object_table=storm_object_table,
            current_rows=these_current_rows)

        these_latlng_rows = _filter_storm_objects_latlng(
            storm_object_table=this_storm_object_table,
            min_latitude_deg=min_plot_latitude_deg,
            max_latitude_deg=max_plot_latitude_deg,
            min_longitude_deg=min_plot_longitude_deg,
            max_longitude_deg=max_plot_longitude_deg)

        if top_myrorss_dir_name is None:
            this_radar_matrix = None
            these_radar_latitudes_deg = None
            these_radar_longitudes_deg = None
        else:
            this_myrorss_file_name = myrorss_and_mrms_io.find_raw_file(
                top_directory_name=top_myrorss_dir_name,
                unix_time_sec=valid_times_unix_sec[i],
                spc_date_string=time_conversion.time_to_spc_date_string(
                    valid_times_unix_sec[i]),
                field_name=radar_field_name,
                data_source=radar_utils.MYRORSS_SOURCE_ID,
                height_m_asl=radar_height_m_asl,
                raise_error_if_missing=True)

            print('Reading data from: "{0:s}"...'.format(
                this_myrorss_file_name))

            this_metadata_dict = (
                myrorss_and_mrms_io.read_metadata_from_raw_file(
                    netcdf_file_name=this_myrorss_file_name,
                    data_source=radar_utils.MYRORSS_SOURCE_ID)
            )

            this_sparse_grid_table = (
                myrorss_and_mrms_io.read_data_from_sparse_grid_file(
                    netcdf_file_name=this_myrorss_file_name,
                    field_name_orig=this_metadata_dict[
                        myrorss_and_mrms_io.FIELD_NAME_COLUMN_ORIG],
                    data_source=radar_utils.MYRORSS_SOURCE_ID,
                    sentinel_values=this_metadata_dict[
                        radar_utils.SENTINEL_VALUE_COLUMN]
                )
            )

            (this_radar_matrix, these_radar_latitudes_deg,
             these_radar_longitudes_deg
            ) = radar_s2f.sparse_to_full_grid(
                sparse_grid_table=this_sparse_grid_table,
                metadata_dict=this_metadata_dict)

            this_radar_matrix = numpy.flipud(this_radar_matrix)
            these_radar_latitudes_deg = these_radar_latitudes_deg[::-1]

        _, this_axes_object, this_basemap_object = (
            plotting_utils.create_equidist_cylindrical_map(
                min_latitude_deg=min_plot_latitude_deg,
                max_latitude_deg=max_plot_latitude_deg,
                min_longitude_deg=min_plot_longitude_deg,
                max_longitude_deg=max_plot_longitude_deg, resolution_string='i')
        )

        _plot_storm_outlines_one_time(
            storm_object_table=this_storm_object_table.iloc[these_latlng_rows],
            valid_time_unix_sec=valid_times_unix_sec[i],
            axes_object=this_axes_object, basemap_object=this_basemap_object,
            storm_outline_colour=storm_outline_colour,
            storm_outline_opacity=storm_outline_opacity,
            include_secondary_ids=include_secondary_ids,
            output_dir_name=output_dir_name,
            primary_id_to_track_colour=primary_id_to_track_colour,
            radar_matrix=this_radar_matrix, radar_field_name=radar_field_name,
            radar_latitudes_deg=these_radar_latitudes_deg,
            radar_longitudes_deg=these_radar_longitudes_deg,
            radar_colour_map_object=radar_colour_map_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_tracking_dir_name=getattr(INPUT_ARG_OBJECT, TRACKING_DIR_ARG_NAME),
        first_spc_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        storm_outline_colour=numpy.array(
            getattr(INPUT_ARG_OBJECT, OUTLINE_COLOUR_ARG_NAME), dtype=float
        ) / 255,
        storm_outline_opacity=getattr(
            INPUT_ARG_OBJECT, OUTLINE_OPACITY_ARG_NAME),
        include_secondary_ids=bool(getattr(
            INPUT_ARG_OBJECT, INCLUDE_SECONDARY_ARG_NAME)),
        min_plot_latitude_deg=getattr(INPUT_ARG_OBJECT, MIN_LATITUDE_ARG_NAME),
        max_plot_latitude_deg=getattr(INPUT_ARG_OBJECT, MAX_LATITUDE_ARG_NAME),
        min_plot_longitude_deg=getattr(
            INPUT_ARG_OBJECT, MIN_LONGITUDE_ARG_NAME),
        max_plot_longitude_deg=getattr(
            INPUT_ARG_OBJECT, MAX_LONGITUDE_ARG_NAME),
        top_myrorss_dir_name=getattr(INPUT_ARG_OBJECT, MYRORSS_DIR_ARG_NAME),
        radar_field_name=getattr(INPUT_ARG_OBJECT, RADAR_FIELD_ARG_NAME),
        radar_height_m_asl=getattr(INPUT_ARG_OBJECT, RADAR_HEIGHT_ARG_NAME),
        radar_colour_map_name=getattr(INPUT_ARG_OBJECT, RADAR_CMAP_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
