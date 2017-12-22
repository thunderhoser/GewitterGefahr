"""Plotting methods for probability."""

import numpy
import matplotlib.pyplot as pyplot
import matplotlib.colors
from gewittergefahr.gg_io import radar_io
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import error_checking


def _get_default_colour_map():
    """Returns default colour map for probability.

    N = number of colours

    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    :return: colour_bounds: length-(N + 1) numpy array of colour boundaries.
        colour_bounds[0] and colour_bounds[1] are the boundaries for the 1st
        colour; colour_bounds[1] and colour_bounds[2] are the boundaries for the
        2nd colour; ...; colour_bounds[i] and colour_bounds[i + 1] are the
        boundaries for the (i + 1)th colour.
    """

    main_colour_list = [
        numpy.array([0., 90., 50.]), numpy.array([35., 139., 69.]),
        numpy.array([65., 171., 93.]), numpy.array([116., 196., 118.]),
        numpy.array([161., 217., 155.]), numpy.array([8., 69., 148.]),
        numpy.array([33., 113., 181.]), numpy.array([66., 146., 198.]),
        numpy.array([107., 174., 214.]), numpy.array([158., 202., 225.]),
        numpy.array([74., 20., 134.]), numpy.array([106., 81., 163.]),
        numpy.array([128., 125., 186.]), numpy.array([158., 154., 200.]),
        numpy.array([188., 189., 220.]), numpy.array([153., 0., 13.]),
        numpy.array([203., 24., 29.]), numpy.array([239., 59., 44.]),
        numpy.array([251., 106., 74.]), numpy.array([252., 146., 114.])]

    for i in range(len(main_colour_list)):
        main_colour_list[i] /= 255

    colour_map_object = matplotlib.colors.ListedColormap(main_colour_list)
    colour_map_object.set_under(numpy.array([1., 1., 1.]))
    colour_map_object.set_over(numpy.array([1., 1., 1.]))

    main_colour_bounds = numpy.linspace(0.05, 0.95, num=19)
    main_colour_bounds = numpy.concatenate((
        numpy.array([0.01]), main_colour_bounds, numpy.array([1.])))
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        main_colour_bounds, colour_map_object.N)

    colour_bounds = numpy.concatenate((
        numpy.array([0.]), main_colour_bounds, numpy.array([2.])))
    return colour_map_object, colour_norm_object, colour_bounds


def plot_latlng_grid(
        axes_object, probability_matrix, min_latitude_deg, min_longitude_deg,
        latitude_spacing_deg, longitude_spacing_deg, colour_map=None,
        colour_minimum=None, colour_maximum=None):
    """Plots lat-long grid of probabilities.

    Because this method plots a lat-long grid, rather than an x-y grid, the
    projection used for the basemap must be cylindrical equidistant (which is
    the same as a lat-long projection).

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)

    All probabilities (in `probability_matrix`, `colour_minimum`, and
    `colour_maximum`) should be dimensionless, ranging from 0...1.

    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param probability_matrix: M-by-N numpy array of probabilities.  Latitude
        should increase while traveling down each column, and longitude should
        increase while traveling to the right along each row.
    :param min_latitude_deg: Minimum latitude over all grid points (deg N).
    :param min_longitude_deg: Minimum longitude over all grid points (deg E).
    :param latitude_spacing_deg: Spacing between meridionally adjacent grid
        points (i.e., between adjacent rows).
    :param longitude_spacing_deg: Spacing between zonally adjacent grid points
        (i.e., between adjacent columns).
    :param colour_map: Instance of `matplotlib.pyplot.cm`.  If None, this method
        will use _get_default_colour_map.
    :param colour_minimum: Minimum value for colour map.
    :param colour_maximum: Maximum value for colour map.
    """

    (probability_matrix_at_edges,
     grid_cell_edge_latitudes_deg,
     grid_cell_edge_longitudes_deg) = grids.latlng_field_grid_points_to_edges(
        field_matrix=probability_matrix, min_latitude_deg=min_latitude_deg,
        min_longitude_deg=min_longitude_deg,
        lat_spacing_deg=latitude_spacing_deg,
        lng_spacing_deg=longitude_spacing_deg)

    probability_matrix_at_edges = numpy.ma.masked_where(
        numpy.isnan(probability_matrix_at_edges), probability_matrix_at_edges)

    if colour_map is None:
        colour_map, colour_norm_object, _ = _get_default_colour_map()
        colour_minimum = colour_norm_object.boundaries[0]
        colour_maximum = colour_norm_object.boundaries[-1]
    else:
        error_checking.assert_is_greater(colour_maximum, colour_minimum)
        colour_norm_object = None

    pyplot.pcolormesh(
        grid_cell_edge_longitudes_deg, grid_cell_edge_latitudes_deg,
        probability_matrix_at_edges, cmap=colour_map, norm=colour_norm_object,
        vmin=colour_minimum, vmax=colour_maximum, shading='flat',
        edgecolors='None', axes=axes_object)
