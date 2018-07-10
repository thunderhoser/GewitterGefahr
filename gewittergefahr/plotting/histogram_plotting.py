"""Plotting of histograms."""

import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import histograms
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import error_checking

DEFAULT_HISTOGRAM_FACE_COLOUR = numpy.array([55., 126., 184.]) / 255
DEFAULT_HISTOGRAM_EDGE_COLOUR = numpy.array([0., 0., 0.]) / 255
DEFAULT_HISTOGRAM_EDGE_WIDTH = 2.

FONT_SIZE = 30
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)


def plot_histogram(
        input_values, num_bins, min_value, max_value, axes_object,
        x_tick_spacing_num_bins, y_tick_spacing=None,
        bar_face_colour=DEFAULT_HISTOGRAM_FACE_COLOUR,
        bar_edge_colour=DEFAULT_HISTOGRAM_EDGE_COLOUR,
        bar_edge_width=DEFAULT_HISTOGRAM_EDGE_WIDTH):
    """Plots histogram.

    :param input_values: See documentation for `histograms.create_histogram`.
    :param num_bins: See documentation for `histograms.create_histogram`.
    :param min_value: See documentation for `histograms.create_histogram`.
    :param max_value: See documentation for `histograms.create_histogram`.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param x_tick_spacing_num_bins: Spacing between adjacent tick marks on
        x-axis, in terms of # bins.
    :param y_tick_spacing: Spacing between adjacent tick marks on
        y-axis, in terms of frequency.
    :param bar_face_colour: Colour (in any format accepted by
        `matplotlib.colors`) for interior of each bar.
    :param bar_edge_colour: Colour for edge of each bar.
    :param bar_edge_width: Width for edge of each bar.
    """

    # TODO(thunderhoser): Make input args nicer, especially `y_tick_spacing`.

    error_checking.assert_is_integer(x_tick_spacing_num_bins)
    error_checking.assert_is_greater(x_tick_spacing_num_bins, 0)

    _, num_examples_by_bin = histograms.create_histogram(
        input_values=input_values, num_bins=num_bins, min_value=min_value,
        max_value=max_value)
    print num_examples_by_bin

    fraction_of_examples_by_bin = (
            num_examples_by_bin.astype(float) / numpy.sum(num_examples_by_bin))

    bin_edges = numpy.linspace(min_value, max_value, num=num_bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[:-1] + bin_width / 2

    axes_object.bar(
        bin_centers, fraction_of_examples_by_bin, bin_width,
        color=bar_face_colour, edgecolor=bar_edge_colour,
        linewidth=bar_edge_width)

    x_tick_indices = numpy.arange(
        0, num_bins - 1, step=x_tick_spacing_num_bins, dtype=int)
    x_tick_indices = x_tick_indices[x_tick_indices < num_bins]
    x_tick_values = bin_centers[x_tick_indices]
    pyplot.xticks(x_tick_values, axes=axes_object)

    if y_tick_spacing is not None:
        error_checking.assert_is_greater(y_tick_spacing, 0.)

        max_y_tick_value = rounder.ceiling_to_nearest(
            numpy.max(fraction_of_examples_by_bin), y_tick_spacing)
        num_y_ticks = 1 + int(numpy.round(max_y_tick_value / y_tick_spacing))
        y_tick_values = numpy.linspace(0., max_y_tick_value, num=num_y_ticks)
        pyplot.yticks(y_tick_values, axes=axes_object)

    axes_object.set_xlim(bin_edges[0], bin_edges[-1])
    axes_object.set_ylim(0., max_y_tick_value)
