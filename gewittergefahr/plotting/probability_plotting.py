"""Plotting methods for probability."""

import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import error_checking


def get_default_colour_map():
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
        numpy.array([0, 90, 50]), numpy.array([35, 139, 69]),
        numpy.array([65, 171, 93]), numpy.array([116, 196, 118]),
        numpy.array([161, 217, 155]), numpy.array([8, 69, 148]),
        numpy.array([33, 113, 181]), numpy.array([66, 146, 198]),
        numpy.array([107, 174, 214]), numpy.array([158, 202, 225]),
        numpy.array([74, 20, 134]), numpy.array([106, 81, 163]),
        numpy.array([128, 125, 186]), numpy.array([158, 154, 200]),
        numpy.array([188, 189, 220]), numpy.array([153, 0, 13]),
        numpy.array([203, 24, 29]), numpy.array([239, 59, 44]),
        numpy.array([251, 106, 74]), numpy.array([252, 146, 114])
    ]

    for i in range(len(main_colour_list)):
        main_colour_list[i] = main_colour_list[i].astype(float) / 255

    colour_map_object = matplotlib.colors.ListedColormap(main_colour_list)
    colour_map_object.set_under(numpy.full(3, 1.))
    colour_map_object.set_over(numpy.full(3, 1.))

    colour_boundaries = numpy.linspace(0.05, 0.95, num=19)
    colour_boundaries = numpy.concatenate((
        numpy.array([0.01]), colour_boundaries, numpy.array([1.])
    ))

    colour_norm_object = matplotlib.colors.BoundaryNorm(
        colour_boundaries, colour_map_object.N)

    return colour_map_object, colour_norm_object


def plot_xy_grid(
        probability_matrix, x_min_metres, y_min_metres, x_spacing_metres,
        y_spacing_metres, axes_object, basemap_object,
        colour_map_object=None, min_colour_value=None, max_colour_value=None):
    """Plots x-y grid of probabilities.

    M = number of rows in grid
    N = number of columns in grid

    :param probability_matrix: M-by-N numpy array of probabilities.
    :param x_min_metres: x-coord for probability_matrix[0, 0].
    :param y_min_metres: y-coord for probability_matrix[0, 0].
    :param x_spacing_metres: Spacing between adjacent columns.
    :param y_spacing_metres: Spacing between adjacent rows.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param basemap_object: Will use this object (instance of
        `mpl_toolkits.basemap.Basemap`) to convert between lat-long and x-y
        coords.
    :param colour_map_object: See doc for `plot_latlng_grid`.
    :param min_colour_value: Same.
    :param max_colour_value: Same.
    """

    probability_matrix, x_coords_metres, y_coords_metres = (
        grids.xy_field_grid_points_to_edges(
            field_matrix=probability_matrix, x_min_metres=x_min_metres,
            y_min_metres=y_min_metres, x_spacing_metres=x_spacing_metres,
            y_spacing_metres=y_spacing_metres)
    )

    if colour_map_object is None:
        colour_map_object, colour_norm_object = get_default_colour_map()
        min_colour_value = colour_norm_object.boundaries[0]
        max_colour_value = colour_norm_object.boundaries[-1]

        probability_matrix = numpy.ma.masked_where(
            probability_matrix < min_colour_value, probability_matrix
        )
    else:
        error_checking.assert_is_greater(min_colour_value, max_colour_value)
        colour_norm_object = None

        probability_matrix = numpy.ma.masked_where(
            numpy.isnan(probability_matrix), probability_matrix
        )

    basemap_object.pcolormesh(
        x_coords_metres, y_coords_metres, probability_matrix,
        cmap=colour_map_object, norm=colour_norm_object,
        vmin=min_colour_value, vmax=max_colour_value, shading='flat',
        edgecolors='None', axes=axes_object, zorder=-1e9)
