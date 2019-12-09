"""Plots GridRad domains.

Specifically, plots number of convective days with GridRad data at each grid
point.
"""

import os.path
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from mpl_toolkits.basemap import Basemap
from gewittergefahr.gg_io import gridrad_io
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils

TOLERANCE = 1e-6
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TIME_INTERVAL_SEC = 300

OVERALL_MIN_LATITUDE_DEG = 20.
OVERALL_MAX_LATITUDE_DEG = 55.
OVERALL_MIN_LONGITUDE_DEG = 230.
OVERALL_MAX_LONGITUDE_DEG = 300.

LAMBERT_CONFORMAL_STRING = 'lcc'

NUM_PARALLELS = 8
NUM_MERIDIANS = 6
RESOLUTION_STRING = 'l'
BORDER_COLOUR = numpy.full(3, 0.)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_DIR_ARG_NAME = 'input_gridrad_dir_name'
FIRST_DATE_ARG_NAME = 'first_spc_date_string'
LAST_DATE_ARG_NAME = 'last_spc_date_string'
COLOUR_MAP_ARG_NAME = 'colour_map_name'
GRID_SPACING_ARG_NAME = 'grid_spacing_metres'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_DIR_HELP_STRING = (
    'Name of top-level input directory.  GridRad files therein will be found by'
    ' `gridrad_io.find_file` and read by '
    '`gridrad_io.read_field_from_full_grid_file`.')

SPC_DATE_HELP_STRING = (
    'SPC date or convective day (format "yyyymmdd").  This script will look for'
    ' GridRad files in the period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

COLOUR_MAP_HELP_STRING = (
    'Name of colour scheme for gridded plot (must be accepted by '
    '`pyplot.get_cmap`).')

GRID_SPACING_HELP_STRING = 'Spacing (metres) of Lambert conformal grid.'

OUTPUT_FILE_HELP_STRING = 'Path to output file.  Figure will be saved here.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + COLOUR_MAP_ARG_NAME, type=str, required=False, default='YlOrRd',
    help=COLOUR_MAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + GRID_SPACING_ARG_NAME, type=float, required=False, default=1e5,
    help=GRID_SPACING_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _get_domain_one_file(gridrad_file_name):
    """Returns spatial domain for one file.

    :param gridrad_file_name: Path to input file.
    :return: domain_limits_deg: length-4 numpy array with
        [min latitude, max latitude, min longitude, max longitude].
        Latitudes are in deg N, and longitudes are in deg E.
    """

    print('Reading metadata from: "{0:s}"...'.format(gridrad_file_name))
    metadata_dict = gridrad_io.read_metadata_from_full_grid_file(
        gridrad_file_name)

    max_latitude_deg = metadata_dict[radar_utils.NW_GRID_POINT_LAT_COLUMN]
    min_longitude_deg = metadata_dict[radar_utils.NW_GRID_POINT_LNG_COLUMN]
    latitude_spacing_deg = metadata_dict[radar_utils.LAT_SPACING_COLUMN]
    longitude_spacing_deg = metadata_dict[radar_utils.LNG_SPACING_COLUMN]
    num_rows = metadata_dict[radar_utils.NUM_LAT_COLUMN]
    num_columns = metadata_dict[radar_utils.NUM_LNG_COLUMN]

    min_latitude_deg = max_latitude_deg - (num_rows - 1) * latitude_spacing_deg
    max_longitude_deg = min_longitude_deg + (
        (num_columns - 1) * longitude_spacing_deg
    )

    return numpy.array([
        min_latitude_deg, max_latitude_deg, min_longitude_deg, max_longitude_deg
    ])


def _get_lcc_params(projection_object):
    """Finds parameters for LCC (Lambert conformal conic) projection.

    :param projection_object: Instance of `pyproj.Proj`.
    :return: standard_latitudes_deg: length-2 numpy array of standard latitudes
        (deg N).
    :return: central_longitude_deg: Central longitude (deg E).
    :raises: ValueError: if projection is not LCC.
    """

    projection_string = projection_object.srs
    words = projection_string.split()

    property_names = [w.split('=')[0][1:] for w in words]
    property_values = [w.split('=')[1] for w in words]
    projection_dict = dict(list(
        zip(property_names, property_values)
    ))

    if projection_dict['proj'] != LAMBERT_CONFORMAL_STRING:
        error_string = 'Grid projection should be "{0:s}", not "{1:s}".'.format(
            LAMBERT_CONFORMAL_STRING, projection_dict['proj']
        )

        raise ValueError(error_string)

    central_longitude_deg = float(projection_dict['lon_0'])
    standard_latitudes_deg = numpy.array([
        float(projection_dict['lat_1']), float(projection_dict['lat_2'])
    ])

    return standard_latitudes_deg, central_longitude_deg


def _get_basemap(grid_metadata_dict):
    """Creates basemap.

    M = number of rows in grid
    M = number of columns in grid

    :param grid_metadata_dict: Dictionary created by
        `grids.create_equidistant_grid`.
    :return: basemap_object: Basemap handle (instance of
        `mpl_toolkits.basemap.Basemap`).
    :return: basemap_x_matrix_metres: M-by-N numpy array of x-coordinates under
        Basemap projection (different than pyproj projection).
    :return: basemap_y_matrix_metres: Same but for y-coordinates.
    """

    x_matrix_metres, y_matrix_metres = grids.xy_vectors_to_matrices(
        x_unique_metres=grid_metadata_dict[grids.X_COORDS_KEY],
        y_unique_metres=grid_metadata_dict[grids.Y_COORDS_KEY]
    )

    projection_object = grid_metadata_dict[grids.PROJECTION_KEY]

    latitude_matrix_deg, longitude_matrix_deg = (
        projections.project_xy_to_latlng(
            x_coords_metres=x_matrix_metres, y_coords_metres=y_matrix_metres,
            projection_object=projection_object)
    )

    standard_latitudes_deg, central_longitude_deg = _get_lcc_params(
        projection_object)

    basemap_object = Basemap(
        projection='lcc', lat_1=standard_latitudes_deg[0],
        lat_2=standard_latitudes_deg[1], lon_0=central_longitude_deg,
        rsphere=projections.DEFAULT_EARTH_RADIUS_METRES,
        ellps=projections.SPHERE_NAME, resolution=RESOLUTION_STRING,
        llcrnrx=x_matrix_metres[0, 0], llcrnry=y_matrix_metres[0, 0],
        urcrnrx=x_matrix_metres[-1, -1], urcrnry=y_matrix_metres[-1, -1]
    )

    basemap_x_matrix_metres, basemap_y_matrix_metres = basemap_object(
        longitude_matrix_deg, latitude_matrix_deg)

    return basemap_object, basemap_x_matrix_metres, basemap_y_matrix_metres


def _plot_data(num_days_matrix, grid_metadata_dict, colour_map_object):
    """Plots data.

    M = number of rows in grid
    N = number of columns in grid

    :param num_days_matrix: M-by-N numpy array with number of convective days
        for which grid cell is in domain.
    :param grid_metadata_dict: Dictionary created by
        `grids.create_equidistant_grid`.
    :param colour_map_object: See documentation at top of file.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    basemap_object, basemap_x_matrix_metres, basemap_y_matrix_metres = (
        _get_basemap(grid_metadata_dict)
    )

    num_grid_rows = num_days_matrix.shape[0]
    num_grid_columns = num_days_matrix.shape[1]
    x_spacing_metres = (
        (basemap_x_matrix_metres[0, -1] - basemap_x_matrix_metres[0, 0]) /
        (num_grid_columns - 1)
    )
    y_spacing_metres = (
        (basemap_y_matrix_metres[-1, 0] - basemap_y_matrix_metres[0, 0]) /
        (num_grid_rows - 1)
    )

    matrix_to_plot, edge_x_coords_metres, edge_y_coords_metres = (
        grids.xy_field_grid_points_to_edges(
            field_matrix=num_days_matrix,
            x_min_metres=basemap_x_matrix_metres[0, 0],
            y_min_metres=basemap_y_matrix_metres[0, 0],
            x_spacing_metres=x_spacing_metres,
            y_spacing_metres=y_spacing_metres)
    )

    matrix_to_plot = numpy.ma.masked_where(matrix_to_plot == 0, matrix_to_plot)

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

    basemap_object.pcolormesh(
        edge_x_coords_metres, edge_y_coords_metres, matrix_to_plot,
        cmap=colour_map_object, vmin=0, vmax=numpy.max(num_days_matrix),
        shading='flat', edgecolors='None', axes=axes_object, zorder=-1e12)

    colour_bar_object = plotting_utils.plot_linear_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=num_days_matrix,
        colour_map_object=colour_map_object, min_value=0,
        max_value=numpy.max(num_days_matrix), orientation_string='horizontal',
        extend_min=False, extend_max=False, padding=0.05)

    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:.1f}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    return figure_object, axes_object


def _run(top_gridrad_dir_name, first_spc_date_string, last_spc_date_string,
         colour_map_name, grid_spacing_metres, output_file_name):
    """Plots GridRad domains.

    This is effectively the main method.

    :param top_gridrad_dir_name: See documentation at top of file.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param colour_map_name: Same.
    :param grid_spacing_metres: Same.
    :param output_file_name: Same.
    """

    colour_map_object = pyplot.get_cmap(colour_map_name)
    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    first_time_unix_sec = time_conversion.get_start_of_spc_date(
        first_spc_date_string)
    last_time_unix_sec = time_conversion.get_end_of_spc_date(
        last_spc_date_string)

    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SEC, include_endpoint=True)

    min_latitudes_deg = []
    max_latitudes_deg = []
    min_longitudes_deg = []
    max_longitudes_deg = []
    last_limits_deg = numpy.full(4, numpy.nan)

    for this_time_unix_sec in valid_times_unix_sec:
        this_gridrad_file_name = gridrad_io.find_file(
            unix_time_sec=this_time_unix_sec,
            top_directory_name=top_gridrad_dir_name,
            raise_error_if_missing=False)

        if not os.path.isfile(this_gridrad_file_name):
            continue

        these_limits_deg = _get_domain_one_file(this_gridrad_file_name)
        if numpy.allclose(these_limits_deg, last_limits_deg, TOLERANCE):
            continue

        min_latitudes_deg.append(these_limits_deg[0])
        max_latitudes_deg.append(these_limits_deg[1])
        min_longitudes_deg.append(these_limits_deg[2])
        max_longitudes_deg.append(these_limits_deg[3])

    print(SEPARATOR_STRING)

    min_latitudes_deg = numpy.array(min_latitudes_deg)
    max_latitudes_deg = numpy.array(max_latitudes_deg)
    min_longitudes_deg = numpy.array(min_longitudes_deg)
    max_longitudes_deg = numpy.array(max_longitudes_deg)
    num_domains = len(min_latitudes_deg)

    grid_metadata_dict = grids.create_equidistant_grid(
        min_latitude_deg=OVERALL_MIN_LATITUDE_DEG,
        max_latitude_deg=OVERALL_MAX_LATITUDE_DEG,
        min_longitude_deg=OVERALL_MIN_LONGITUDE_DEG,
        max_longitude_deg=OVERALL_MAX_LONGITUDE_DEG,
        x_spacing_metres=grid_spacing_metres,
        y_spacing_metres=grid_spacing_metres, azimuthal=False)

    unique_x_coords_metres = grid_metadata_dict[grids.X_COORDS_KEY]
    unique_y_coords_metres = grid_metadata_dict[grids.Y_COORDS_KEY]
    projection_object = grid_metadata_dict[grids.PROJECTION_KEY]

    x_coord_matrix_metres, y_coord_matrix_metres = grids.xy_vectors_to_matrices(
        x_unique_metres=unique_x_coords_metres,
        y_unique_metres=unique_y_coords_metres)

    latitude_matrix_deg, longitude_matrix_deg = (
        projections.project_xy_to_latlng(
            x_coords_metres=x_coord_matrix_metres,
            y_coords_metres=y_coord_matrix_metres,
            projection_object=projection_object)
    )

    num_grid_rows = latitude_matrix_deg.shape[0]
    num_grid_columns = latitude_matrix_deg.shape[1]
    num_days_matrix = numpy.full((num_grid_rows, num_grid_columns), 0)

    for i in range(num_domains):
        if numpy.mod(i, 10) == 0:
            print('Have found grid points in {0:d} of {1:d} domains...'.format(
                i, num_domains
            ))

        this_lat_flag_matrix = numpy.logical_and(
            latitude_matrix_deg >= min_latitudes_deg[i],
            latitude_matrix_deg <= max_latitudes_deg[i]
        )
        this_lng_flag_matrix = numpy.logical_and(
            longitude_matrix_deg >= min_longitudes_deg[i],
            longitude_matrix_deg <= max_longitudes_deg[i]
        )

        num_days_matrix += numpy.logical_and(
            this_lat_flag_matrix, this_lng_flag_matrix
        ).astype(int)

    print(SEPARATOR_STRING)

    figure_object = _plot_data(
        num_days_matrix=num_days_matrix, grid_metadata_dict=grid_metadata_dict,
        colour_map_object=colour_map_object
    )[0]

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight')
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_gridrad_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_spc_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        colour_map_name=getattr(INPUT_ARG_OBJECT, COLOUR_MAP_ARG_NAME),
        grid_spacing_metres=getattr(INPUT_ARG_OBJECT, GRID_SPACING_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
