"""Plotting methods for NWP (numerical weather prediction) data."""

import numpy
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import error_checking


def plot_xy_grid(axes_object=None, basemap_object=None, field_matrix=None,
                 model_name=None, grid_id=None, colour_map=None,
                 colour_minimum=None, colour_maximum=None):
    """Plots x-y grid of a single NWP field.

    M = number of rows (unique grid-point x-coordinates)
    N = number of columns (unique grid-point y-coordinates)

    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
    :param field_matrix: M-by-N numpy array with values of NWP field.
    :param model_name: Name of model.
    :param grid_id: String ID for model grid.
    :param colour_map: Instance of `matplotlib.pyplot.cm`.
    :param colour_minimum: Minimum value for colour map.
    :param colour_maximum: Maximum value for colour map.
    """

    error_checking.assert_is_greater(colour_maximum, colour_minimum)

    nw_latitude_deg_as_array, nw_longitude_deg_as_array = (
        nwp_model_utils.project_xy_to_latlng(
            numpy.array([nwp_model_utils.MIN_GRID_POINT_X_METRES]),
            numpy.array([nwp_model_utils.MIN_GRID_POINT_Y_METRES]),
            projection_object=None, model_name=model_name, grid_id=grid_id))

    x_min_in_basemap_proj_metres, y_min_in_basemap_proj_metres = basemap_object(
        nw_longitude_deg_as_array[0], nw_latitude_deg_as_array[0])

    x_spacing_metres, y_spacing_metres = nwp_model_utils.get_xy_grid_spacing(
        model_name, grid_id)

    field_matrix_at_edges, grid_cell_edge_x_metres, grid_cell_edge_y_metres = (
        grids.xy_field_grid_points_to_edges(
            field_matrix=field_matrix,
            x_min_metres=x_min_in_basemap_proj_metres,
            y_min_metres=y_min_in_basemap_proj_metres,
            x_spacing_metres=x_spacing_metres,
            y_spacing_metres=y_spacing_metres))

    field_matrix_at_edges = numpy.ma.masked_where(
        numpy.isnan(field_matrix_at_edges), field_matrix_at_edges)

    basemap_object.pcolormesh(
        grid_cell_edge_x_metres, grid_cell_edge_y_metres,
        field_matrix_at_edges, cmap=colour_map, vmin=colour_minimum,
        vmax=colour_maximum, shading='flat', edgecolors='None',
        axes=axes_object, zorder=-1e9)
