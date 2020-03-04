"""Makes figure with CONUS-wide MYRORSS and GridRad data."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_io import myrorss_and_mrms_io
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import radar_sparse_to_full as radar_s2f
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import radar_plotting
from gewittergefahr.plotting import imagemagick_utils

INPUT_TIME_FORMAT = '%Y-%m-%d-%H%M%S'
RADAR_FIELD_NAME = radar_utils.REFL_COLUMN_MAX_NAME

NUM_PARALLELS = 8
NUM_MERIDIANS = 6
BORDER_COLOUR = numpy.full(3, 0.)
FIGURE_RESOLUTION_DPI = 300

MYRORSS_DIR_ARG_NAME = 'input_myrorss_dir_name'
GRIDRAD_DIR_ARG_NAME = 'input_gridrad_dir_name'
VALID_TIME_ARG_NAME = 'valid_time_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MYRORSS_DIR_HELP_STRING = (
    'Name of top-level directory with MYRORSS data.  The file will be found by '
    '`myrorss_and_mrms_io.find_raw_file_inexact_time` and read by '
    '`myrorss_and_mrms_io.read_data_from_sparse_grid_file`.'
)
GRIDRAD_DIR_HELP_STRING = (
    'Name of top-level directory with GridRad data (but formatted as MYRORSS '
    'data).  The file will be found by `myrorss_and_mrms_io.find_raw_file` and '
    'read by `myrorss_and_mrms_io.read_data_from_sparse_grid_file`.'
)
VALID_TIME_HELP_STRING = 'Valid time (format "yyyy-mm-dd-HHMMSS").'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MYRORSS_DIR_ARG_NAME, type=str, required=True,
    help=MYRORSS_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + GRIDRAD_DIR_ARG_NAME, type=str, required=True,
    help=GRIDRAD_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + VALID_TIME_ARG_NAME, type=str, required=True,
    help=VALID_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _read_file(radar_file_name):
    """Reads radar data from file.

    M = number of rows in grid
    N = number of columns in grid

    :param radar_file_name: Path to input file.
    :return: reflectivity_matrix_dbz: M-by-N numpy array of reflectivity values.
    :return: latitudes_deg: length-M numpy array of latitudes (deg N).
    :return: longitudes_deg: length-N numpy array of longitudes (deg E).
    """

    metadata_dict = myrorss_and_mrms_io.read_metadata_from_raw_file(
        netcdf_file_name=radar_file_name,
        data_source=radar_utils.MYRORSS_SOURCE_ID
    )
    sparse_grid_table = myrorss_and_mrms_io.read_data_from_sparse_grid_file(
        netcdf_file_name=radar_file_name,
        field_name_orig=
        metadata_dict[myrorss_and_mrms_io.FIELD_NAME_COLUMN_ORIG],
        data_source=radar_utils.MYRORSS_SOURCE_ID,
        sentinel_values=metadata_dict[radar_utils.SENTINEL_VALUE_COLUMN]
    )
    reflectivity_matrix_dbz, latitudes_deg, longitudes_deg = (
        radar_s2f.sparse_to_full_grid(
            sparse_grid_table=sparse_grid_table,
            metadata_dict=metadata_dict)
    )

    reflectivity_matrix_dbz = numpy.flipud(reflectivity_matrix_dbz)
    latitudes_deg = latitudes_deg[::-1]

    return reflectivity_matrix_dbz, latitudes_deg, longitudes_deg


def _plot_one_field(
        reflectivity_matrix_dbz, latitudes_deg, longitudes_deg, add_colour_bar,
        panel_letter, output_file_name):
    """Plots reflectivity field from one dataset.

    :param reflectivity_matrix_dbz: See doc for `_read_file`.
    :param latitudes_deg: Same.
    :param longitudes_deg: Same.
    :param add_colour_bar: Boolean flag.
    :param panel_letter: Panel letter (will be printed at top left of figure).
    :param output_file_name: Path to output file (figure will be saved here).
    """

    (
        figure_object, axes_object, basemap_object
    ) = plotting_utils.create_equidist_cylindrical_map(
        min_latitude_deg=numpy.min(latitudes_deg),
        max_latitude_deg=numpy.max(latitudes_deg),
        min_longitude_deg=numpy.min(longitudes_deg),
        max_longitude_deg=numpy.max(longitudes_deg), resolution_string='i'
    )

    plotting_utils.plot_coastlines(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR
    )
    plotting_utils.plot_countries(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR
    )
    plotting_utils.plot_states_and_provinces(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR
    )
    plotting_utils.plot_parallels(
        basemap_object=basemap_object, axes_object=axes_object,
        num_parallels=NUM_PARALLELS
    )
    plotting_utils.plot_meridians(
        basemap_object=basemap_object, axes_object=axes_object,
        num_meridians=NUM_MERIDIANS
    )

    radar_plotting.plot_latlng_grid(
        field_matrix=reflectivity_matrix_dbz,
        field_name=RADAR_FIELD_NAME, axes_object=axes_object,
        min_grid_point_latitude_deg=numpy.min(latitudes_deg),
        min_grid_point_longitude_deg=numpy.min(longitudes_deg),
        latitude_spacing_deg=latitudes_deg[1] - latitudes_deg[0],
        longitude_spacing_deg=longitudes_deg[1] - longitudes_deg[0]
    )

    if add_colour_bar:
        colour_map_object, colour_norm_object = (
            radar_plotting.get_default_colour_scheme(RADAR_FIELD_NAME)
        )

        plotting_utils.plot_colour_bar(
            axes_object_or_matrix=axes_object,
            data_matrix=reflectivity_matrix_dbz,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            orientation_string='horizontal', padding=0.05,
            extend_min=False, extend_max=True, fraction_of_axis_length=0.75
        )

    plotting_utils.label_axes(
        axes_object=axes_object, label_string='({0:s})'.format(panel_letter),
        y_coord_normalized=1.03
    )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(top_myrorss_dir_name, top_gridrad_dir_name, valid_time_string,
         output_dir_name):
    """Makes figure with CONUS-wide MYRORSS and GridRad data.

    This is effectively the main method.

    :param top_myrorss_dir_name: See documentation at top of file.
    :param top_gridrad_dir_name: Same.
    :param valid_time_string: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    valid_time_unix_sec = time_conversion.string_to_unix_sec(
        valid_time_string, INPUT_TIME_FORMAT
    )
    spc_date_string = time_conversion.time_to_spc_date_string(
        valid_time_unix_sec
    )

    myrorss_file_name = myrorss_and_mrms_io.find_raw_file_inexact_time(
        desired_time_unix_sec=valid_time_unix_sec,
        spc_date_string=spc_date_string,
        field_name=RADAR_FIELD_NAME, data_source=radar_utils.MYRORSS_SOURCE_ID,
        top_directory_name=top_myrorss_dir_name,
        raise_error_if_missing=True
    )

    gridrad_file_name = myrorss_and_mrms_io.find_raw_file(
        unix_time_sec=valid_time_unix_sec, spc_date_string=spc_date_string,
        field_name=RADAR_FIELD_NAME, data_source=radar_utils.MYRORSS_SOURCE_ID,
        top_directory_name=top_gridrad_dir_name,
        raise_error_if_missing=True
    )

    print('Reading data from: "{0:s}"...'.format(myrorss_file_name))
    reflectivity_matrix_dbz, latitudes_deg, longitudes_deg = _read_file(
        myrorss_file_name
    )

    panel_file_names = [
        '{0:s}/myrorss_panel.jpg'.format(output_dir_name),
        '{0:s}/gridrad_panel.jpg'.format(output_dir_name)
    ]

    _plot_one_field(
        reflectivity_matrix_dbz=reflectivity_matrix_dbz,
        latitudes_deg=latitudes_deg, longitudes_deg=longitudes_deg,
        panel_letter='a', add_colour_bar=False,
        output_file_name=panel_file_names[0]
    )

    print('Reading data from: "{0:s}"...'.format(gridrad_file_name))
    reflectivity_matrix_dbz, latitudes_deg, longitudes_deg = _read_file(
        gridrad_file_name
    )

    _plot_one_field(
        reflectivity_matrix_dbz=reflectivity_matrix_dbz,
        latitudes_deg=latitudes_deg, longitudes_deg=longitudes_deg,
        panel_letter='b', add_colour_bar=True,
        output_file_name=panel_file_names[1]
    )

    figure_file_name = '{0:s}/conus_radar_comparison.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=figure_file_name,
        num_panel_rows=2, num_panel_columns=1
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=figure_file_name, output_file_name=figure_file_name,
        border_width_pixels=10
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_myrorss_dir_name=getattr(INPUT_ARG_OBJECT, MYRORSS_DIR_ARG_NAME),
        top_gridrad_dir_name=getattr(INPUT_ARG_OBJECT, GRIDRAD_DIR_ARG_NAME),
        valid_time_string=getattr(INPUT_ARG_OBJECT, VALID_TIME_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
