"""Makes figure to explain storm detection."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_io import myrorss_and_mrms_io
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import radar_sparse_to_full as radar_s2f
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import echo_classification as echo_classifn
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import radar_plotting
from gewittergefahr.plotting import storm_plotting
from gewittergefahr.plotting import imagemagick_utils

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MIN_POLYGON_SIZES_PX = numpy.array([0, 5], dtype=int)

STORM_WIDTH = 2
STORM_COLOUR = numpy.full(3, 0.)

MARKER_TYPE = 'o'
MARKER_SIZE = 12
MARKER_EDGE_WIDTH = 1

NUM_PARALLELS = 8
NUM_MERIDIANS = 6
FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

RADAR_DIR_ARG_NAME = 'input_radar_dir_name'
ECHO_CLASSIFN_DIR_ARG_NAME = 'input_echo_classifn_dir_name'
VALID_TIME_ARG_NAME = 'valid_time_string'
MIN_LATITUDE_ARG_NAME = 'min_latitude_deg'
MAX_LATITUDE_ARG_NAME = 'max_latitude_deg'
MIN_LONGITUDE_ARG_NAME = 'min_longitude_deg'
MAX_LONGITUDE_ARG_NAME = 'max_longitude_deg'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

RADAR_DIR_HELP_STRING = (
    'Name of top-level radar directory.  Files therein will be found by '
    '`echo_top_tracking._find_input_radar_files`.')

ECHO_CLASSIFN_DIR_HELP_STRING = (
    'Name of top-level directory with echo-classification files.  Files therein'
    ' will be found by `echo_classification.find_classification_file` and read '
    'by `echo_classification.read_classifications`.')

VALID_TIME_HELP_STRING = 'Valid time (format "yyyy-mm-dd-HHMMSS").'
MIN_LATITUDE_HELP_STRING = 'Minimum latitude (deg N) for plotting.'
MAX_LATITUDE_HELP_STRING = 'Max latitude (deg N) for plotting.'
MIN_LONGITUDE_HELP_STRING = 'Minimum longitude (deg E) for plotting.'
MAX_LONGITUDE_HELP_STRING = 'Max longitude (deg E) for plotting.'

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_DIR_ARG_NAME, type=str, required=True,
    help=RADAR_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + ECHO_CLASSIFN_DIR_ARG_NAME, type=str, required=True,
    help=ECHO_CLASSIFN_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + VALID_TIME_ARG_NAME, type=str, required=False,
    default='2011-04-27-043010', help=VALID_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_LATITUDE_ARG_NAME, type=float, required=False, default=32.,
    help=MIN_LATITUDE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_LATITUDE_ARG_NAME, type=float, required=False, default=34.,
    help=MAX_LATITUDE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_LONGITUDE_ARG_NAME, type=float, required=False, default=266.,
    help=MIN_LONGITUDE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_LONGITUDE_ARG_NAME, type=float, required=False, default=269.,
    help=MAX_LONGITUDE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _plot_echo_tops(echo_top_matrix_km_asl, latitudes_deg, longitudes_deg,
                    plot_colour_bar, convective_flag_matrix=None):
    """Plots grid of 40-dBZ echo tops.

    M = number of rows in grid
    N = number of columns in grid

    :param echo_top_matrix_km_asl: M-by-N numpy array of echo tops (km above sea
        level).
    :param latitudes_deg: length-M numpy array of latitudes (deg N).
    :param longitudes_deg: length-N numpy array of longitudes (deg E).
    :param plot_colour_bar: Boolean flag.
    :param convective_flag_matrix: M-by-N numpy array of Boolean flags,
        indicating which grid cells are convective.  If
        `convective_flag_matrix is None`, all grid cells will be plotted.  If
        `convective_flag_matrix is not None`, only convective grid cells will be
        plotted.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :return: basemap_object: Basemap handle (instance of
        `mpl_toolkits.basemap.Basemap`).
    """

    figure_object, axes_object, basemap_object = (
        plotting_utils.create_equidist_cylindrical_map(
            min_latitude_deg=numpy.min(latitudes_deg),
            max_latitude_deg=numpy.max(latitudes_deg),
            min_longitude_deg=numpy.min(longitudes_deg),
            max_longitude_deg=numpy.max(longitudes_deg), resolution_string='h'
        )
    )

    plotting_utils.plot_coastlines(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=plotting_utils.DEFAULT_COUNTRY_COLOUR)

    plotting_utils.plot_countries(
        basemap_object=basemap_object, axes_object=axes_object)

    plotting_utils.plot_states_and_provinces(
        basemap_object=basemap_object, axes_object=axes_object)

    plotting_utils.plot_parallels(
        basemap_object=basemap_object, axes_object=axes_object,
        num_parallels=NUM_PARALLELS, line_width=0)

    plotting_utils.plot_meridians(
        basemap_object=basemap_object, axes_object=axes_object,
        num_meridians=NUM_MERIDIANS, line_width=0)

    matrix_to_plot = echo_top_matrix_km_asl + 0.
    if convective_flag_matrix is not None:
        matrix_to_plot[convective_flag_matrix == False] = numpy.nan

    radar_plotting.plot_latlng_grid(
        field_matrix=matrix_to_plot, field_name=radar_utils.ECHO_TOP_40DBZ_NAME,
        axes_object=axes_object,
        min_grid_point_latitude_deg=numpy.min(latitudes_deg),
        min_grid_point_longitude_deg=numpy.min(longitudes_deg),
        latitude_spacing_deg=numpy.diff(latitudes_deg[:2])[0],
        longitude_spacing_deg=numpy.diff(longitudes_deg[:2])[0]
    )

    if not plot_colour_bar:
        return figure_object, axes_object, basemap_object

    colour_map_object, colour_norm_object = (
        radar_plotting.get_default_colour_scheme(
            radar_utils.ECHO_TOP_40DBZ_NAME)
    )

    colour_bar_object = plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=matrix_to_plot,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object, orientation_string='horizontal',
        extend_min=False, extend_max=True, fraction_of_axis_length=1.)

    colour_bar_object.set_label('40-dBZ echo top (kft ASL)')

    return figure_object, axes_object, basemap_object


def _run(top_radar_dir_name, top_echo_classifn_dir_name, valid_time_string,
         min_latitude_deg, max_latitude_deg, min_longitude_deg,
         max_longitude_deg, output_dir_name):
    """Makes figure to explain storm detection.

    This is effectively the main method.

    :param top_radar_dir_name: See documentation at top of file.
    :param top_echo_classifn_dir_name: Same.
    :param valid_time_string: Same.
    :param min_latitude_deg: Same.
    :param max_latitude_deg: Same.
    :param min_longitude_deg: Same.
    :param max_longitude_deg: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    valid_time_unix_sec = time_conversion.string_to_unix_sec(
        valid_time_string, TIME_FORMAT)
    spc_date_string = time_conversion.time_to_spc_date_string(
        valid_time_unix_sec)

    num_polygon_sizes = len(MIN_POLYGON_SIZES_PX)
    tracking_dir_names = [None] * num_polygon_sizes

    for k in range(num_polygon_sizes):
        tracking_dir_names[k] = (
            '{0:s}/tracking/min_polygon_size_px={1:d}'
        ).format(output_dir_name, MIN_POLYGON_SIZES_PX[k])

        echo_top_tracking.run_tracking(
            top_radar_dir_name=top_radar_dir_name,
            top_output_dir_name=tracking_dir_names[k],
            first_spc_date_string=spc_date_string,
            last_spc_date_string=spc_date_string,
            first_time_unix_sec=valid_time_unix_sec,
            last_time_unix_sec=valid_time_unix_sec + 1,
            top_echo_classifn_dir_name=top_echo_classifn_dir_name,
            min_polygon_size_pixels=MIN_POLYGON_SIZES_PX[k],
            min_track_duration_seconds=0)
        print(SEPARATOR_STRING)

    echo_top_file_name = myrorss_and_mrms_io.find_raw_file(
        top_directory_name=top_radar_dir_name,
        unix_time_sec=valid_time_unix_sec, spc_date_string=spc_date_string,
        field_name=radar_utils.ECHO_TOP_40DBZ_NAME,
        data_source=radar_utils.MYRORSS_SOURCE_ID, raise_error_if_missing=True)

    print('Reading data from: "{0:s}"...'.format(echo_top_file_name))

    metadata_dict = myrorss_and_mrms_io.read_metadata_from_raw_file(
        netcdf_file_name=echo_top_file_name,
        data_source=radar_utils.MYRORSS_SOURCE_ID)

    sparse_grid_table = myrorss_and_mrms_io.read_data_from_sparse_grid_file(
        netcdf_file_name=echo_top_file_name,
        field_name_orig=metadata_dict[
            myrorss_and_mrms_io.FIELD_NAME_COLUMN_ORIG],
        data_source=radar_utils.MYRORSS_SOURCE_ID,
        sentinel_values=metadata_dict[radar_utils.SENTINEL_VALUE_COLUMN]
    )

    echo_top_matrix_km_asl, radar_latitudes_deg, radar_longitudes_deg = (
        radar_s2f.sparse_to_full_grid(
            sparse_grid_table=sparse_grid_table, metadata_dict=metadata_dict)
    )

    echo_top_matrix_km_asl = numpy.flip(echo_top_matrix_km_asl, axis=0)
    radar_latitudes_deg = radar_latitudes_deg[::-1]

    echo_classifn_file_name = echo_classifn.find_classification_file(
        top_directory_name=top_echo_classifn_dir_name,
        valid_time_unix_sec=valid_time_unix_sec,
        desire_zipped=True, allow_zipped_or_unzipped=True,
        raise_error_if_missing=True)

    print('Reading data from: "{0:s}"...'.format(echo_classifn_file_name))
    convective_flag_matrix = echo_classifn.read_classifications(
        echo_classifn_file_name
    )[0]

    good_indices = numpy.where(numpy.logical_and(
        radar_latitudes_deg >= min_latitude_deg,
        radar_latitudes_deg <= max_latitude_deg
    ))[0]

    echo_top_matrix_km_asl = echo_top_matrix_km_asl[good_indices, ...]
    convective_flag_matrix = convective_flag_matrix[good_indices, ...]
    radar_latitudes_deg = radar_latitudes_deg[good_indices]

    good_indices = numpy.where(numpy.logical_and(
        radar_longitudes_deg >= min_longitude_deg,
        radar_longitudes_deg <= max_longitude_deg
    ))[0]

    echo_top_matrix_km_asl = echo_top_matrix_km_asl[..., good_indices]
    convective_flag_matrix = convective_flag_matrix[..., good_indices]
    radar_longitudes_deg = radar_longitudes_deg[good_indices]

    this_figure_object, this_axes_object = _plot_echo_tops(
        echo_top_matrix_km_asl=echo_top_matrix_km_asl,
        latitudes_deg=radar_latitudes_deg, longitudes_deg=radar_longitudes_deg,
        plot_colour_bar=False, convective_flag_matrix=None
    )[:2]

    this_axes_object.set_title('All grid cells')
    plotting_utils.label_axes(axes_object=this_axes_object, label_string='(a)')

    panel_file_names = [
        '{0:s}/before_echo_classification.jpg'.format(output_dir_name)
    ]

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    this_figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(this_figure_object)

    this_figure_object, this_axes_object = _plot_echo_tops(
        echo_top_matrix_km_asl=echo_top_matrix_km_asl,
        latitudes_deg=radar_latitudes_deg, longitudes_deg=radar_longitudes_deg,
        plot_colour_bar=False, convective_flag_matrix=convective_flag_matrix
    )[:2]

    this_axes_object.set_title('Convective grid cells only')
    plotting_utils.label_axes(axes_object=this_axes_object, label_string='(b)')

    panel_file_names.append(
        '{0:s}/after_echo_classification.jpg'.format(output_dir_name)
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    this_figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(this_figure_object)

    letter_label = 'b'

    for k in range(num_polygon_sizes):
        this_tracking_file_name = tracking_io.find_file(
            top_tracking_dir_name=tracking_dir_names[k],
            tracking_scale_metres2=
            echo_top_tracking.DUMMY_TRACKING_SCALE_METRES2,
            source_name=tracking_utils.SEGMOTION_NAME,
            valid_time_unix_sec=valid_time_unix_sec,
            spc_date_string=spc_date_string,
            raise_error_if_missing=True)

        print('Reading data from: "{0:s}"...'.format(this_tracking_file_name))
        this_storm_object_table = tracking_io.read_file(this_tracking_file_name)

        this_figure_object, this_axes_object, this_basemap_object = (
            _plot_echo_tops(
                echo_top_matrix_km_asl=echo_top_matrix_km_asl,
                latitudes_deg=radar_latitudes_deg,
                longitudes_deg=radar_longitudes_deg, plot_colour_bar=True,
                convective_flag_matrix=convective_flag_matrix)
        )

        storm_plotting.plot_storm_outlines(
            storm_object_table=this_storm_object_table,
            axes_object=this_axes_object, basemap_object=this_basemap_object,
            line_width=STORM_WIDTH, line_colour=STORM_COLOUR)

        this_title_string = (
            'Detection with min polygon size = {0:d} grid cells'
        ).format(MIN_POLYGON_SIZES_PX[k])
        this_axes_object.set_title(this_title_string)

        letter_label = chr(ord(letter_label) + 1)
        plotting_utils.label_axes(
            axes_object=this_axes_object,
            label_string='({0:s})'.format(letter_label)
        )

        panel_file_names.append(
            '{0:s}/detection{1:d}.jpg'.format(output_dir_name, k)
        )

        print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
        this_figure_object.savefig(
            panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(this_figure_object)

    concat_file_name = '{0:s}/storm_detection.jpg'.format(output_dir_name)
    print('Concatenating panels to: "{0:s}"...'.format(concat_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names, output_file_name=concat_file_name,
        num_panel_rows=2, num_panel_columns=2)

    imagemagick_utils.resize_image(
        input_file_name=concat_file_name, output_file_name=concat_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_radar_dir_name=getattr(INPUT_ARG_OBJECT, RADAR_DIR_ARG_NAME),
        top_echo_classifn_dir_name=getattr(
            INPUT_ARG_OBJECT, ECHO_CLASSIFN_DIR_ARG_NAME),
        valid_time_string=getattr(INPUT_ARG_OBJECT, VALID_TIME_ARG_NAME),
        min_latitude_deg=getattr(INPUT_ARG_OBJECT, MIN_LATITUDE_ARG_NAME),
        max_latitude_deg=getattr(INPUT_ARG_OBJECT, MAX_LATITUDE_ARG_NAME),
        min_longitude_deg=getattr(INPUT_ARG_OBJECT, MIN_LONGITUDE_ARG_NAME),
        max_longitude_deg=getattr(INPUT_ARG_OBJECT, MAX_LONGITUDE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
