"""Plotting methods for visualizing a CNN (convolutional neural net).

--- NOTATION ---

The following letters will be used throughout this module.

M = number of rows (length of first spatial dimension)
N = number of columns (length of second spatial dimension)
C = number of channels (transformed input variables)
"""

import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import error_checking

matplotlib.rcParams['axes.linewidth'] = 2

FONT_SIZE = 30
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)


def plot_2d_feature_map(
        feature_matrix_2d, axes_object, colour_map_object,
        colour_norm_object=None, min_value_in_colour_map=None,
        max_value_in_colour_map=None):
    """Plots 2-D feature map (spatial map of intermediate-layer activations).

    If `colour_norm_object is None`, the arguments `min_value_in_colour_map` and
    `max_value_in_colour_map` are required.

    :param feature_matrix_2d: M-by-N numpy array with activations of a single
        intermediate layer for a single example and channel.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param colour_map_object: Instance of `matplotlib.pyplot.cm`.
    :param colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    :param min_value_in_colour_map: Minimum value in colour map.
    :param max_value_in_colour_map: Max value in colour map.
    """

    error_checking.assert_is_numpy_array_without_nan(feature_matrix_2d)
    error_checking.assert_is_numpy_array(feature_matrix_2d, num_dimensions=2)

    if colour_norm_object is None:
        error_checking.assert_is_greater(
            max_value_in_colour_map, min_value_in_colour_map)
        colour_norm_object = None
    else:
        min_value_in_colour_map = colour_norm_object.boundaries[0]
        max_value_in_colour_map = colour_norm_object.boundaries[-1]

    axes_object.pcolormesh(
        feature_matrix_2d, cmap=colour_map_object, norm=colour_norm_object,
        vmin=min_value_in_colour_map, vmax=max_value_in_colour_map,
        shading='flat', edgecolors='None')

    axes_object.set_xticks([])
    axes_object.set_yticks([])


def plot_many_2d_feature_maps(
        feature_matrix_3d, num_panel_rows, colour_map_object,
        figure_width_inches=15., figure_height_inches=15.,
        colour_norm_object=None, min_value_in_colour_map=None,
        max_value_in_colour_map=None):
    """Plots many 2-D feature maps in the same figure (one panel per channel).

    :param feature_matrix_3d: M-by-N-by-C numpy array with activations of a
        single intermediate layer for a single example.
    :param num_panel_rows: Number of rows in paneled figure (different than M,
        the number of spatial rows per feature map).
    :param colour_map_object: See doc for `plot_2d_feature_map`.
    :param figure_width_inches: Figure width.
    :param figure_height_inches: Figure height.
    :param colour_norm_object: See doc for `plot_2d_feature_map`.
    :param min_value_in_colour_map: Same.
    :param max_value_in_colour_map: Same.
    :return: figure_object: Instance of `matplotlib.figure.Figure`.
    """

    error_checking.assert_is_numpy_array_without_nan(feature_matrix_3d)
    error_checking.assert_is_numpy_array(feature_matrix_3d, num_dimensions=3)
    num_channels = feature_matrix_3d.shape[-1]

    error_checking.assert_is_integer(num_panel_rows)
    error_checking.assert_is_geq(num_panel_rows, 1)
    error_checking.assert_is_leq(num_panel_rows, num_channels)

    num_panel_columns = int(numpy.ceil(float(num_channels) / num_panel_rows))
    figure_object, axes_objects_2d_list = pyplot.subplots(
        num_panel_rows, num_panel_columns,
        figsize=(figure_width_inches, figure_height_inches),
        sharex=True, sharey=True)
    pyplot.subplots_adjust(
        left=0.01, bottom=0.01, right=0.95, top=0.95, hspace=0, wspace=0)

    for i in range(num_panel_rows):
        for j in range(num_panel_columns):
            this_channel_index = i * num_panel_columns + j
            plot_2d_feature_map(
                feature_matrix_2d=feature_matrix_3d[..., this_channel_index],
                axes_object=axes_objects_2d_list[i][j],
                colour_map_object=colour_map_object,
                colour_norm_object=colour_norm_object,
                min_value_in_colour_map=min_value_in_colour_map,
                max_value_in_colour_map=max_value_in_colour_map)

    return figure_object
