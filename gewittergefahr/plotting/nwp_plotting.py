"""Plots gridded NWP (numerical weather prediction) output.

--- DEFINITIONS ---

"Subgrid" = contiguous rectangular subset of the full model grid.  This need not
be a *strict* subset (in other words, the subgrid could be the full grid).

M = number of rows in subgrid (unique y-coordinates at grid points)
N = number of columns in subgrid (unique x-coordinates at grid points)
"""

import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from mpl_toolkits.basemap import Basemap
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import error_checking

DEFAULT_FIGURE_WIDTH_INCHES = 15.
DEFAULT_FIGURE_HEIGHT_INCHES = 15.
DEFAULT_BOUNDARY_RESOLUTION_STRING = 'l'


def _get_xy_grid_point_matrices(
        model_name, first_row_in_full_grid, last_row_in_full_grid,
        first_column_in_full_grid, last_column_in_full_grid, grid_id=None,
        basemap_object=None):
    """Returns x-y coordinates for a subgrid of the full model grid.

    This method generates different x-y coordinates than
    `nwp_model_utils.get_xy_grid_point_matrices`, because (like
    `mpl_toolkits.basemap.Basemap`) this method sets false easting = false
    northing = 0 metres.

    :param model_name: Name of NWP model (must be accepted by
        `nwp_model_utils.check_grid_id`).
    :param first_row_in_full_grid: Row 0 in the subgrid is row
        `first_row_in_full_grid` in the full grid.
    :param last_row_in_full_grid: Last row in the subgrid is row
        `last_row_in_full_grid` in the full grid.  If you want last row in the
        subgrid to equal last row in the full grid, make this -1.
    :param first_column_in_full_grid: Column 0 in the subgrid is column
        `first_column_in_full_grid` in the full grid.
    :param last_column_in_full_grid: Last column in the subgrid is column
        `last_column_in_full_grid` in the full grid.  If you want last column in
        the subgrid to equal last column in the full grid, make this -1.
    :param grid_id: Grid for NWP model (must be accepted by
        `nwp_model_utils.check_grid_id`).
    :param basemap_object: Instance of `mpl_toolkits.basemap.Basemap` for the
        given NWP model.  If you don't have one, no big deal -- leave this
        argument empty.
    :return: grid_point_x_matrix_metres: M-by-N numpy array of x-coordinates.
    :return: grid_point_y_matrix_metres: M-by-N numpy array of y-coordinates.
    """

    (num_rows_in_full_grid, num_columns_in_full_grid
    ) = nwp_model_utils.get_grid_dimensions(
        model_name=model_name, grid_id=grid_id)

    error_checking.assert_is_integer(first_row_in_full_grid)
    error_checking.assert_is_geq(first_row_in_full_grid, 0)
    error_checking.assert_is_integer(last_row_in_full_grid)
    if last_row_in_full_grid < 0:
        last_row_in_full_grid += num_rows_in_full_grid

    error_checking.assert_is_greater(
        last_row_in_full_grid, first_row_in_full_grid)
    error_checking.assert_is_less_than(
        last_row_in_full_grid, num_rows_in_full_grid)

    error_checking.assert_is_integer(first_column_in_full_grid)
    error_checking.assert_is_geq(first_column_in_full_grid, 0)
    error_checking.assert_is_integer(last_column_in_full_grid)
    if last_column_in_full_grid < 0:
        last_column_in_full_grid += num_columns_in_full_grid

    error_checking.assert_is_greater(
        last_column_in_full_grid, first_column_in_full_grid)
    error_checking.assert_is_less_than(
        last_column_in_full_grid, num_columns_in_full_grid)

    (latitude_matrix_deg, longitude_matrix_deg
    ) = nwp_model_utils.get_latlng_grid_point_matrices(
        model_name=model_name, grid_id=grid_id)

    latitude_matrix_deg = latitude_matrix_deg[
        first_row_in_full_grid:(last_row_in_full_grid + 1),
        first_column_in_full_grid:(last_column_in_full_grid + 1)]
    longitude_matrix_deg = longitude_matrix_deg[
        first_row_in_full_grid:(last_row_in_full_grid + 1),
        first_column_in_full_grid:(last_column_in_full_grid + 1)]

    if basemap_object is None:
        (standard_latitudes_deg, central_longitude_deg
        ) = nwp_model_utils.get_projection_params(model_name)

        projection_object = projections.init_lambert_conformal_projection(
            standard_latitudes_deg=standard_latitudes_deg,
            central_longitude_deg=central_longitude_deg)

        (grid_point_x_matrix_metres, grid_point_y_matrix_metres
        ) = projections.project_latlng_to_xy(
            latitudes_deg=latitude_matrix_deg,
            longitudes_deg=longitude_matrix_deg,
            projection_object=projection_object, false_northing_metres=0.,
            false_easting_metres=0.)
    else:
        (grid_point_x_matrix_metres, grid_point_y_matrix_metres
        ) = basemap_object(longitude_matrix_deg, latitude_matrix_deg)

    return grid_point_x_matrix_metres, grid_point_y_matrix_metres


def init_basemap(
        model_name, grid_id=None,
        figure_width_inches=DEFAULT_FIGURE_WIDTH_INCHES,
        figure_height_inches=DEFAULT_FIGURE_HEIGHT_INCHES,
        resolution_string=DEFAULT_BOUNDARY_RESOLUTION_STRING,
        first_row_in_full_grid=0, last_row_in_full_grid=-1,
        first_column_in_full_grid=0, last_column_in_full_grid=-1):
    """Initializes basemap with the given model's projection.

    :param model_name: Name of NWP model (must be accepted by
        `nwp_model_utils.check_grid_id`).
    :param grid_id: Grid for NWP model (must be accepted by
        `nwp_model_utils.check_grid_id`).
    :param figure_width_inches: Figure width.
    :param figure_height_inches: Figure height.
    :param resolution_string: Resolution for boundaries (e.g., coastlines and
        political borders).  Options are "c" for crude, "l" for low, "i" for
        intermediate, "h" for high, and "f" for full.  Keep in mind that higher-
        resolution boundaries take much longer to draw.
    :param first_row_in_full_grid: See doc for `_get_xy_grid_point_matrices`.
    :param last_row_in_full_grid: Same.
    :param first_column_in_full_grid: Same.
    :param last_column_in_full_grid: Same.
    :return: figure_object: Instance of `matplotlib.figure.Figure`.
    :return: axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :return: basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
    """

    error_checking.assert_is_greater(figure_width_inches, 0.)
    error_checking.assert_is_greater(figure_height_inches, 0.)
    error_checking.assert_is_string(resolution_string)

    (grid_point_x_matrix_metres, grid_point_y_matrix_metres
    ) = _get_xy_grid_point_matrices(
        model_name=model_name, grid_id=grid_id,
        first_row_in_full_grid=first_row_in_full_grid,
        last_row_in_full_grid=last_row_in_full_grid,
        first_column_in_full_grid=first_column_in_full_grid,
        last_column_in_full_grid=last_column_in_full_grid)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(figure_width_inches, figure_height_inches))

    (standard_latitudes_deg, central_longitude_deg
    ) = nwp_model_utils.get_projection_params(model_name)

    basemap_object = Basemap(
        projection='lcc', lat_1=standard_latitudes_deg[0],
        lat_2=standard_latitudes_deg[1], lon_0=central_longitude_deg,
        rsphere=projections.EARTH_RADIUS_METRES, ellps='sphere',
        resolution=resolution_string,
        llcrnrx=grid_point_x_matrix_metres[0, 0],
        llcrnry=grid_point_y_matrix_metres[0, 0],
        urcrnrx=grid_point_x_matrix_metres[-1, -1],
        urcrnry=grid_point_y_matrix_metres[-1, -1])

    return figure_object, axes_object, basemap_object


def plot_subgrid(
        field_matrix, model_name, axes_object, basemap_object, colour_map,
        min_value_in_colour_map, max_value_in_colour_map, grid_id=None,
        first_row_in_full_grid=0, first_column_in_full_grid=0, opacity=1.):
    """Plots colour map over subgrid.

    :param field_matrix: M-by-N numpy array with field to plot.
    :param model_name: Name of NWP model (must be accepted by
        `nwp_model_utils.check_grid_id`).
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
    :param colour_map: Instance of `matplotlib.pyplot.cm`.
    :param min_value_in_colour_map: Minimum value in colour map.
    :param max_value_in_colour_map: Max value in colour map.
    :param grid_id: Grid for NWP model (must be accepted by
        `nwp_model_utils.check_grid_id`).
    :param first_row_in_full_grid: Row 0 in the subgrid (i.e., row 0 in
        field_matrix) is row `first_row_in_full_grid` in the full grid.
    :param first_column_in_full_grid: Column 0 in the subgrid (i.e., column 0 in
        field_matrix) is column `first_column_in_full_grid` in the full grid.
    :param opacity: Opacity of colour map (from 0...1).
    """

    error_checking.assert_is_real_numpy_array(field_matrix)
    error_checking.assert_is_numpy_array(field_matrix, num_dimensions=2)
    error_checking.assert_is_greater(
        max_value_in_colour_map, min_value_in_colour_map)

    num_rows_in_subgrid = field_matrix.shape[0]
    num_columns_in_subgrid = field_matrix.shape[1]

    (grid_point_x_matrix_metres, grid_point_y_matrix_metres
    ) = _get_xy_grid_point_matrices(
        model_name=model_name, grid_id=grid_id,
        first_row_in_full_grid=first_row_in_full_grid,
        last_row_in_full_grid=first_row_in_full_grid + num_rows_in_subgrid - 1,
        first_column_in_full_grid=first_column_in_full_grid,
        last_column_in_full_grid=
        first_column_in_full_grid + num_columns_in_subgrid - 1,
        basemap_object=basemap_object)

    x_spacing_metres = (
        (grid_point_x_matrix_metres[0, -1] - grid_point_x_matrix_metres[0, 0]) /
        (num_columns_in_subgrid - 1))
    y_spacing_metres = (
        (grid_point_y_matrix_metres[-1, 0] - grid_point_y_matrix_metres[0, 0]) /
        (num_rows_in_subgrid - 1))

    (field_matrix_at_edges, grid_cell_edges_x_metres, grid_cell_edges_y_metres
    ) = grids.xy_field_grid_points_to_edges(
        field_matrix=field_matrix,
        x_min_metres=grid_point_x_matrix_metres[0, 0],
        y_min_metres=grid_point_y_matrix_metres[0, 0],
        x_spacing_metres=x_spacing_metres, y_spacing_metres=y_spacing_metres)

    field_matrix_at_edges = numpy.ma.masked_where(
        numpy.isnan(field_matrix_at_edges), field_matrix_at_edges)

    basemap_object.pcolormesh(
        grid_cell_edges_x_metres, grid_cell_edges_y_metres,
        field_matrix_at_edges, cmap=colour_map, vmin=min_value_in_colour_map,
        vmax=max_value_in_colour_map, shading='flat', edgecolors='None',
        axes=axes_object, zorder=-1e9, alpha=opacity)
