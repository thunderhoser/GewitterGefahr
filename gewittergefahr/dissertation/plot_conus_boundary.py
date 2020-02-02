"""Plots boundary of continental United States (CONUS)."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import conus_boundary
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils

NUM_PARALLELS = 8
NUM_MERIDIANS = 6
MIN_PLOT_LATITUDE_DEG = 20.
MAX_PLOT_LATITUDE_DEG = 50.
MIN_PLOT_LONGITUDE_DEG = 235.
MAX_PLOT_LONGITUDE_DEG = 300.

BORDER_WIDTH = 1.
BORDER_COLOUR = numpy.full(3, 0.)
CONUS_LINE_WIDTH = 4.
CONUS_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255

FIGURE_RESOLUTION_DPI = 600

EROSION_DISTANCE_ARG_NAME = 'erosion_distance_metres'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

EROSION_DISTANCE_HELP_STRING = (
    'Erosion distance for CONUS boundary.  If you do not want to erode the '
    'polygon, leave this alone.'
)
OUTPUT_FILE_HELP_STRING = 'Path to output file (figure will be saved here).'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EROSION_DISTANCE_ARG_NAME, type=float, required=False, default=0.,
    help=EROSION_DISTANCE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(erosion_distance_metres, output_file_name):
    """Plots boundary of continental United States (CONUS).

    This is effectively the main method.

    :param erosion_distance_metres: See documentation at top of file.
    :param output_file_name: Same.
    """

    error_checking.assert_is_geq(erosion_distance_metres, 0.)
    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    latitudes_deg, longitudes_deg = conus_boundary.read_from_netcdf()

    if erosion_distance_metres > 0:
        latitudes_deg, longitudes_deg = conus_boundary.erode_boundary(
            latitudes_deg=latitudes_deg, longitudes_deg=longitudes_deg,
            erosion_distance_metres=erosion_distance_metres
        )

    figure_object, axes_object, basemap_object = (
        plotting_utils.create_lambert_conformal_map(
            min_latitude_deg=MIN_PLOT_LATITUDE_DEG,
            max_latitude_deg=MAX_PLOT_LATITUDE_DEG,
            min_longitude_deg=MIN_PLOT_LONGITUDE_DEG,
            max_longitude_deg=MAX_PLOT_LONGITUDE_DEG,
            resolution_string='i'
        )
    )

    plotting_utils.plot_coastlines(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR, line_width=BORDER_WIDTH
    )
    plotting_utils.plot_countries(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR, line_width=BORDER_WIDTH
    )
    plotting_utils.plot_states_and_provinces(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR, line_width=BORDER_WIDTH
    )
    plotting_utils.plot_parallels(
        basemap_object=basemap_object, axes_object=axes_object,
        num_parallels=NUM_PARALLELS, line_width=BORDER_WIDTH
    )
    plotting_utils.plot_meridians(
        basemap_object=basemap_object, axes_object=axes_object,
        num_meridians=NUM_MERIDIANS, line_width=BORDER_WIDTH
    )

    x_coords_metres, y_coords_metres = basemap_object(
        longitudes_deg, latitudes_deg
    )
    axes_object.plot(
        x_coords_metres, y_coords_metres,
        color=CONUS_COLOUR, linestyle='solid', linewidth=CONUS_LINE_WIDTH
    )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        erosion_distance_metres=getattr(
            INPUT_ARG_OBJECT, EROSION_DISTANCE_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
