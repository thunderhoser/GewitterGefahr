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
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import storm_plotting
from gewittergefahr.plotting import radar_plotting
from gewittergefahr.plotting import imagemagick_utils

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
TRACK_COLOUR = numpy.full(3, 0.)
FIGURE_RESOLUTION_DPI = 300

TRACKING_DIR_ARG_NAME = 'input_tracking_dir_name'
FIRST_DATE_ARG_NAME = 'first_spc_date_string'
LAST_DATE_ARG_NAME = 'last_spc_date_string'
STORM_COLOUR_ARG_NAME = 'storm_colour'
STORM_OPACITY_ARG_NAME = 'storm_opacity'
INCLUDE_SECONDARY_ARG_NAME = 'include_secondary_ids'
MIN_LATITUDE_ARG_NAME = 'min_plot_latitude_deg'
MAX_LATITUDE_ARG_NAME = 'max_plot_latitude_deg'
MIN_LONGITUDE_ARG_NAME = 'min_plot_longitude_deg'
MAX_LONGITUDE_ARG_NAME = 'max_plot_longitude_deg'
MYRORSS_DIR_ARG_NAME = 'input_myrorss_dir_name'
RADAR_FIELD_ARG_NAME = 'radar_field_name'
RADAR_HEIGHT_ARG_NAME = 'radar_height_m_asl'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

TRACKING_DIR_HELP_STRING = (
    'Name of top-level directory with storm tracks.  Files therein will be '
    'found by `storm_tracking_io.find_processed_files_one_spc_date` and read by'
    ' `storm_tracking_io.read_many_processed_files`.')

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  Storm outlines will be plotted for all SPC '
    'dates in the period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

STORM_COLOUR_HELP_STRING = (
    'Colour of storm outlines (length-3 list of elements [R, G, B], each in '
    'range 0...255).')

STORM_OPACITY_HELP_STRING = 'Opacity of storm outlines (in range 0...1).'

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

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).')

DEFAULT_STORM_COLOUR = numpy.array([228, 26, 28], dtype=int)
DEFAULT_STORM_OPACITY = 0.5

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
    '--' + STORM_COLOUR_ARG_NAME, type=int, nargs=3, required=False,
    default=DEFAULT_STORM_COLOUR, help=STORM_COLOUR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + STORM_OPACITY_ARG_NAME, type=float, required=False,
    default=DEFAULT_STORM_OPACITY, help=STORM_OPACITY_HELP_STRING)

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
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)

# TODO(thunderhoser): Allow tracks to be included or not.


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


def _find_relevant_storm_objects(storm_object_table, current_time_unix_sec):
    """Finds relevant storm objects at valid time t_0.

    "Relevant" storm objects include:

    - all those at time t_0
    - all those at time < t_0 with the same primary ID as one at t_0

    :param storm_object_table: See doc for `storm_tracking_io.write_file`.
    :param current_time_unix_sec: t_0 in the above discussion.
    :return: relevant_storm_object_table: Same as input but with fewer rows.
    """

    current_rows = numpy.where(
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values ==
        current_time_unix_sec
    )[0]

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


def _plot_storm_outlines_one_time(
        storm_object_table, valid_time_unix_sec, axes_object, basemap_object,
        storm_colour, storm_opacity, include_secondary_ids,
        output_dir_name, radar_matrix=None, radar_field_name=None,
        radar_latitudes_deg=None, radar_longitudes_deg=None):
    """Plots storm outlines (and may underlay radar data) at one time step.

    M = number of rows in radar grid
    N = number of columns in radar grid
    K = number of storm objects

    :param storm_object_table: See doc for `storm_plotting.plot_storm_outlines`.
    :param valid_time_unix_sec: Will plot storm outlines only at this time.
        Will plot tracks up to and including this time.
    :param axes_object: Same.
    :param basemap_object: Same.
    :param storm_colour: Same.
    :param storm_opacity: Same.
    :param include_secondary_ids: Same.
    :param output_dir_name: See documentation at top of file.
    :param radar_matrix: M-by-N numpy array of radar values.  If
        `radar_matrix is None`, radar data will simply not be plotted.
    :param radar_field_name: [used only if `radar_matrix is not None`]
        See documentation at top of file.
    :param radar_latitudes_deg: [used only if `radar_matrix is not None`]
        length-M numpy array of grid-point latitudes (deg N).
    :param radar_longitudes_deg: [used only if `radar_matrix is not None`]
        length-N numpy array of grid-point longitudes (deg E).
    """

    min_plot_latitude_deg = basemap_object.llcrnrlat
    max_plot_latitude_deg = basemap_object.urcrnrlat
    min_plot_longitude_deg = basemap_object.llcrnrlon
    max_plot_longitude_deg = basemap_object.urcrnrlon

    parallel_spacing_deg = (
        (max_plot_latitude_deg - min_plot_latitude_deg) / (NUM_PARALLELS - 1)
    )
    meridian_spacing_deg = (
        (max_plot_longitude_deg - min_plot_longitude_deg) / (NUM_MERIDIANS - 1)
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

    if radar_matrix is not None:
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

        radar_plotting.plot_latlng_grid(
            field_matrix=radar_matrix, field_name=radar_field_name,
            axes_object=axes_object,
            min_grid_point_latitude_deg=numpy.min(radar_latitudes_deg),
            min_grid_point_longitude_deg=numpy.min(radar_longitudes_deg),
            latitude_spacing_deg=latitude_spacing_deg,
            longitude_spacing_deg=longitude_spacing_deg)

        colour_map_object, colour_norm_object = (
            radar_plotting.get_default_colour_scheme(radar_field_name)
        )

        latitude_range_deg = max_plot_latitude_deg - min_plot_latitude_deg
        longitude_range_deg = max_plot_longitude_deg - min_plot_longitude_deg

        if latitude_range_deg > longitude_range_deg:
            orientation_string = 'vertical'
        else:
            orientation_string = 'horizontal'

        colour_bar_object = plotting_utils.add_colour_bar(
            axes_object_or_list=axes_object, values_to_colour=radar_matrix,
            colour_map=colour_map_object, colour_norm_object=colour_norm_object,
            orientation=orientation_string,
            extend_min=radar_field_name in radar_plotting.SHEAR_VORT_DIV_NAMES,
            extend_max=True, fraction_of_axis_length=0.9)

        colour_bar_object.set_label(
            radar_plotting.FIELD_NAME_TO_VERBOSE_DICT[radar_field_name]
        )

    valid_time_rows = numpy.where(
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values ==
        valid_time_unix_sec
    )[0]

    line_colour = matplotlib.colors.to_rgba(storm_colour, storm_opacity)

    storm_plotting.plot_storm_outlines(
        storm_object_table=storm_object_table.iloc[valid_time_rows],
        axes_object=axes_object, basemap_object=basemap_object,
        line_colour=line_colour)

    storm_plotting.plot_storm_ids(
        storm_object_table=storm_object_table.iloc[valid_time_rows],
        axes_object=axes_object, basemap_object=basemap_object,
        plot_near_centroids=False, include_secondary_ids=include_secondary_ids,
        font_colour=storm_plotting.DEFAULT_FONT_COLOUR)

    storm_plotting.plot_storm_tracks(
        storm_object_table=storm_object_table, axes_object=axes_object,
        basemap_object=basemap_object, colour_map_object=None,
        line_colour=TRACK_COLOUR,
        start_marker_type=',', end_marker_type=',',
        start_marker_size=1, end_marker_size=1)

    nice_time_string = time_conversion.unix_sec_to_string(
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values[0],
        NICE_TIME_FORMAT
    )

    abbrev_time_string = time_conversion.unix_sec_to_string(
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values[0],
        FILE_NAME_TIME_FORMAT
    )

    pyplot.title('Storm objects at {0:s}'.format(nice_time_string))
    output_file_name = '{0:s}/storm_outlines_{1:s}.jpg'.format(
        output_dir_name, abbrev_time_string)

    print 'Saving figure to: "{0:s}"...'.format(output_file_name)
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(input_file_name=output_file_name,
                                      output_file_name=output_file_name)


def _run(top_tracking_dir_name, first_spc_date_string, last_spc_date_string,
         storm_colour, storm_opacity, include_secondary_ids,
         min_plot_latitude_deg, max_plot_latitude_deg, min_plot_longitude_deg,
         max_plot_longitude_deg, top_myrorss_dir_name, radar_field_name,
         radar_height_m_asl, output_dir_name):
    """Plots storm outlines (along with IDs) at each time step.

    This is effectively the main method.

    :param top_tracking_dir_name: See documentation at top of file.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param storm_colour: Same.
    :param storm_opacity: Same.
    :param include_secondary_ids: Same.
    :param min_plot_latitude_deg: Same.
    :param max_plot_latitude_deg: Same.
    :param min_plot_longitude_deg: Same.
    :param max_plot_longitude_deg: Same.
    :param top_myrorss_dir_name: Same.
    :param radar_field_name: Same.
    :param radar_height_m_asl: Same.
    :param output_dir_name: Same.
    """

    if top_myrorss_dir_name in ['', 'None']:
        top_myrorss_dir_name = None

    if radar_field_name != radar_utils.REFL_NAME:
        radar_height_m_asl = None

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
    print SEPARATOR_STRING

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

    valid_times_unix_sec = numpy.unique(
        storm_object_table[tracking_utils.VALID_TIME_COLUMN].values
    )
    num_times = len(valid_times_unix_sec)

    for i in range(num_times):
        this_storm_object_table = _find_relevant_storm_objects(
            storm_object_table=storm_object_table,
            current_time_unix_sec=valid_times_unix_sec[i]
        )

        these_latlng_rows = _filter_storm_objects_latlng(
            storm_object_table=this_storm_object_table,
            min_latitude_deg=min_plot_latitude_deg,
            max_latitude_deg=max_plot_latitude_deg,
            min_longitude_deg=min_plot_longitude_deg,
            max_longitude_deg=max_plot_longitude_deg)

        if len(these_latlng_rows) == 0:
            continue

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

            print 'Reading data from: "{0:s}"...'.format(
                this_myrorss_file_name)

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
            plotting_utils.init_equidistant_cylindrical_map(
                min_latitude_deg=min_plot_latitude_deg,
                max_latitude_deg=max_plot_latitude_deg,
                min_longitude_deg=min_plot_longitude_deg,
                max_longitude_deg=max_plot_longitude_deg, resolution_string='i')
        )

        _plot_storm_outlines_one_time(
            storm_object_table=this_storm_object_table.iloc[these_latlng_rows],
            valid_time_unix_sec=valid_times_unix_sec[i],
            axes_object=this_axes_object, basemap_object=this_basemap_object,
            storm_colour=storm_colour, storm_opacity=storm_opacity,
            include_secondary_ids=include_secondary_ids,
            output_dir_name=output_dir_name, radar_matrix=this_radar_matrix,
            radar_field_name=radar_field_name,
            radar_latitudes_deg=these_radar_latitudes_deg,
            radar_longitudes_deg=these_radar_longitudes_deg)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_tracking_dir_name=getattr(INPUT_ARG_OBJECT, TRACKING_DIR_ARG_NAME),
        first_spc_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        storm_colour=numpy.array(
            getattr(INPUT_ARG_OBJECT, STORM_COLOUR_ARG_NAME), dtype=float
        ) / 255,
        storm_opacity=getattr(INPUT_ARG_OBJECT, STORM_OPACITY_ARG_NAME),
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
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
