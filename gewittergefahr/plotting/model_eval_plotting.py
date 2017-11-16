"""Plotting methods for model evaluation.

This module can be used to evaluate any kind of weather model (machine learning,
NWP, heuristics, human forecasting, etc.).  This module is completely agnostic
of where the forecasts come from.

--- REFERENCES ---

Hsu, W., and A. Murphy, 1986: "The attributes diagram: A geometrical framework
    for assessing the quality of probability forecasts". International Journal
    of Forecasting, 2 (3), 285-293.
"""

import numpy
from descartes import PolygonPatch
import matplotlib.pyplot as pyplot
import matplotlib.colors
from gewittergefahr.gg_utils import model_evaluation as model_eval
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils

# TODO(thunderhoser): Make this module deal with NaN's.
# TODO(thunderhoser): Add confidence intervals (from bootstrapping or
# otherwise).

DEFAULT_ROC_LINE_COLOUR = numpy.array([228., 26., 28.]) / 255
DEFAULT_ROC_LINE_WIDTH = 3.
DEFAULT_ROC_RANDOM_LINE_COLOUR = numpy.array([152., 152., 152.]) / 255
DEFAULT_ROC_RANDOM_LINE_WIDTH = 2.

DEFAULT_PERF_DIAG_LINE_COLOUR = numpy.array([228., 26., 28.]) / 255
DEFAULT_PERF_DIAG_LINE_WIDTH = 3.

DEFAULT_FREQ_BIAS_LINE_COLOUR = numpy.array([152., 152., 152.]) / 255
DEFAULT_FREQ_BIAS_LINE_WIDTH = 2.
STRING_FORMAT_FOR_FREQ_BIAS_LABELS = '%.2f'
PIXEL_PADDING_FOR_FREQ_BIAS_LABELS = 10
LEVELS_FOR_FREQ_BIAS_CONTOURS = numpy.array(
    [0.25, 0.5, 0.75, 1., 1.5, 2., 3., 5.])

DEFAULT_CSI_COLOUR_MAP = pyplot.cm.Blues
LEVELS_FOR_CSI_CONTOURS = numpy.array(
    [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])

DEFAULT_RELIA_LINE_COLOUR = numpy.array([228., 26., 28.]) / 255
DEFAULT_RELIA_LINE_WIDTH = 3.
DEFAULT_PERFECT_RELIA_LINE_COLOUR = numpy.array([152., 152., 152.]) / 255
DEFAULT_PERFECT_RELIA_LINE_WIDTH = 2.
DEFAULT_NO_SKILL_RELIA_LINE_COLOUR = numpy.array([31., 120., 180.]) / 255
DEFAULT_NO_SKILL_RELIA_LINE_WIDTH = 2.
TRANSPARENCY_FOR_SKILL_AREA = 0.2
DEFAULT_CLIMATOLOGY_LINE_COLOUR = numpy.array([152., 152., 152.]) / 255
DEFAULT_CLIMATOLOGY_LINE_WIDTH = 2.

DEFAULT_HISTOGRAM_FACE_COLOUR = numpy.array([228., 26., 28.]) / 255
DEFAULT_HISTOGRAM_EDGE_COLOUR = numpy.array([0., 0., 0.]) / 255
DEFAULT_HISTOGRAM_EDGE_WIDTH = 2.

INSET_HISTOGRAM_LEFT_EDGE = 0.575
INSET_HISTOGRAM_BOTTOM_EDGE = 0.175
INSET_HISTOGRAM_WIDTH = 0.3
INSET_HISTOGRAM_HEIGHT = 0.3
INSET_HISTOGRAM_X_TICKS = numpy.linspace(0., 1., num=6)
INSET_HISTOGRAM_Y_TICK_SPACING = 0.1

FONT_SIZE = 30
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)


def _plot_background_of_attributes_diagram(
        axes_object=None, mean_observed_label=None,
        no_skill_line_colour=DEFAULT_NO_SKILL_RELIA_LINE_COLOUR,
        no_skill_line_width=DEFAULT_NO_SKILL_RELIA_LINE_WIDTH,
        other_line_colour=DEFAULT_CLIMATOLOGY_LINE_COLOUR,
        other_line_width=DEFAULT_CLIMATOLOGY_LINE_WIDTH):
    """Plots background (references lines and polygons) of attributes diagram.

    For more on the attributes diagram, see Hsu and Murphy (1986).

    BSS = Brier skill score.  For more on the BSS, see
    `model_evaluation.get_brier_skill_score`.

    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param mean_observed_label: Mean observed label (event frequency) for the
        whole dataset.
    :param no_skill_line_colour: Colour (in any format accepted by
        `matplotlib.colors`) of no-skill line, where BSS = 0.
    :param no_skill_line_width: Width (real positive number) of no-skill line.
    :param other_line_colour: Colour of climatology and no-resolution lines.
    :param other_line_width: Width of climatology and no-resolution lines.
    """

    error_checking.assert_is_geq(mean_observed_label, 0.)
    error_checking.assert_is_leq(mean_observed_label, 1.)

    (x_vertices_for_left_skill_area,
     y_vertices_for_left_skill_area,
     x_vertices_for_right_skill_area,
     y_vertices_for_right_skill_area) = (
         model_eval.get_skill_areas_in_reliability_curve(mean_observed_label))

    skill_area_colour = matplotlib.colors.to_rgba(
        no_skill_line_colour, TRANSPARENCY_FOR_SKILL_AREA)

    left_polygon_object = polygons.vertex_arrays_to_polygon_object(
        x_vertices_for_left_skill_area, y_vertices_for_left_skill_area)
    left_polygon_patch = PolygonPatch(
        left_polygon_object, lw=0, ec=skill_area_colour, fc=skill_area_colour)
    axes_object.add_patch(left_polygon_patch)

    right_polygon_object = polygons.vertex_arrays_to_polygon_object(
        x_vertices_for_right_skill_area, y_vertices_for_right_skill_area)
    right_polygon_patch = PolygonPatch(
        right_polygon_object, lw=0, ec=skill_area_colour, fc=skill_area_colour)
    axes_object.add_patch(right_polygon_patch)

    no_skill_x_coords, no_skill_y_coords = (
        model_eval.get_no_skill_reliability_curve(mean_observed_label))
    axes_object.plot(
        no_skill_x_coords, no_skill_y_coords, color=no_skill_line_colour,
        linestyle='solid', linewidth=no_skill_line_width)

    climo_x_coords, climo_y_coords = (
        model_eval.get_climatology_line_for_reliability_curve(
            mean_observed_label))
    axes_object.plot(
        climo_x_coords, climo_y_coords, color=other_line_colour,
        linestyle='dashed', linewidth=other_line_width)

    no_resolution_x_coords, no_resolution_y_coords = (
        model_eval.get_no_resolution_line_for_reliability_curve(
            mean_observed_label))
    axes_object.plot(
        no_resolution_x_coords, no_resolution_y_coords, color=other_line_colour,
        linestyle='dashed', linewidth=other_line_width)


def _plot_inset_histogram_for_attributes_diagram(
        figure_object=None, num_examples_by_bin=None,
        bar_face_colour=DEFAULT_HISTOGRAM_FACE_COLOUR,
        bar_edge_colour=DEFAULT_HISTOGRAM_EDGE_COLOUR,
        bar_edge_width=DEFAULT_HISTOGRAM_EDGE_WIDTH):
    """Plots forecast histogram inset in attributes diagram.

    For more on the attributes diagram, see Hsu and Murphy (1986).

    B = number of forecast bins

    :param figure_object: Instance of `matplotlib.figure.Figure`.
    :param num_examples_by_bin: length-B numpy array with number of examples in
        each forecast bin.
    :param bar_face_colour: Colour (in any format accepted by
        `matplotlib.colors`) for interior of histogram bars.
    :param bar_edge_colour: Colour for edge of histogram bars.
    :param bar_edge_width: Width for edge of histogram bars.
    """

    error_checking.assert_is_integer_numpy_array(num_examples_by_bin)
    error_checking.assert_is_numpy_array(num_examples_by_bin, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(num_examples_by_bin, 0)
    num_forecast_bins = len(num_examples_by_bin)
    error_checking.assert_is_geq(num_forecast_bins, 2)

    example_frequency_by_bin = (
        num_examples_by_bin.astype(float) / numpy.sum(num_examples_by_bin))

    forecast_bin_edges = numpy.linspace(0., 1., num=num_forecast_bins + 1)
    forecast_bin_width = forecast_bin_edges[1] - forecast_bin_edges[0]
    forecast_bin_centers = forecast_bin_edges[:-1] + forecast_bin_width / 2

    inset_axes_object = figure_object.add_axes(
        [INSET_HISTOGRAM_LEFT_EDGE, INSET_HISTOGRAM_BOTTOM_EDGE,
         INSET_HISTOGRAM_WIDTH, INSET_HISTOGRAM_HEIGHT])
    inset_axes_object.bar(
        forecast_bin_centers, example_frequency_by_bin, forecast_bin_width,
        color=bar_face_colour, edgecolor=bar_edge_colour,
        linewidth=bar_edge_width)

    max_y_tick_value = rounder.floor_to_nearest(
        1.05 * numpy.max(example_frequency_by_bin),
        INSET_HISTOGRAM_Y_TICK_SPACING)
    num_y_ticks = 1 + int(numpy.round(
        max_y_tick_value / INSET_HISTOGRAM_Y_TICK_SPACING))
    y_tick_values = numpy.linspace(0., max_y_tick_value, num=num_y_ticks)

    pyplot.xticks(INSET_HISTOGRAM_X_TICKS, axes=inset_axes_object)
    pyplot.yticks(y_tick_values, axes=inset_axes_object)
    inset_axes_object.set_xlim(0., 1.)
    inset_axes_object.set_ylim(0., 1.05 * numpy.max(example_frequency_by_bin))


def plot_roc_curve(
        axes_object=None, pod_by_threshold=None, pofd_by_threshold=None,
        line_colour=DEFAULT_ROC_LINE_COLOUR, line_width=DEFAULT_ROC_LINE_WIDTH,
        random_line_colour=DEFAULT_ROC_RANDOM_LINE_COLOUR,
        random_line_width=DEFAULT_ROC_RANDOM_LINE_WIDTH):
    """Plots ROC (receiver operating characteristic) curve.

    T = number of binarization thresholds

    For the definition of a "binarization threshold" and the role they play in
    ROC curves, see `model_evaluation.get_points_in_roc_curve`.

    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param pod_by_threshold: length-T numpy array of POD (probability of
        detection) values.
    :param pofd_by_threshold: length-T numpy array of POFD (probability of false
        detection) values.
    :param line_colour: Colour (in any format accepted by `matplotlib.colors`).
    :param line_width: Line width (real positive number).
    :param random_line_colour: Colour of reference line (ROC curve for a random
        predictor).
    :param random_line_width: Width of reference line.
    """

    error_checking.assert_is_numpy_array(pod_by_threshold, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(pod_by_threshold, 0.)
    error_checking.assert_is_leq_numpy_array(pod_by_threshold, 1.)
    num_thresholds = len(pod_by_threshold)

    error_checking.assert_is_numpy_array(
        pofd_by_threshold, exact_dimensions=numpy.array([num_thresholds]))
    error_checking.assert_is_geq_numpy_array(pofd_by_threshold, 0.)
    error_checking.assert_is_leq_numpy_array(pofd_by_threshold, 1.)

    random_x_coords, random_y_coords = model_eval.get_random_roc_curve()
    axes_object.plot(
        random_x_coords, random_y_coords, color=random_line_colour,
        linestyle='dashed', linewidth=random_line_width)

    axes_object.plot(
        pofd_by_threshold, pod_by_threshold, color=line_colour,
        linestyle='solid', linewidth=line_width)

    axes_object.set_xlabel('POFD (probability of false detection)')
    axes_object.set_ylabel('POD (probability of detection)')
    axes_object.set_xlim(0., 1.)
    axes_object.set_ylim(0., 1.)


def plot_performance_diagram(
        axes_object=None, pod_by_threshold=None,
        success_ratio_by_threshold=None,
        line_colour=DEFAULT_PERF_DIAG_LINE_COLOUR,
        line_width=DEFAULT_PERF_DIAG_LINE_WIDTH,
        bias_line_colour=DEFAULT_FREQ_BIAS_LINE_COLOUR,
        bias_line_width=DEFAULT_FREQ_BIAS_LINE_WIDTH,
        csi_colour_map=DEFAULT_CSI_COLOUR_MAP):
    """Plots performance diagram.

    T = number of binarization thresholds

    For the definition of a "binarization threshold" and the role they play in
    performance diagrams, see
    `model_evaluation.get_points_in_performance_diagram`.

    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param pod_by_threshold: length-T numpy array of POD (probability of
        detection) values.
    :param success_ratio_by_threshold: length-T numpy array of success ratios.
    :param line_colour: Colour (in any format accepted by `matplotlib.colors`).
    :param line_width: Line width (real positive number).
    :param bias_line_colour: Colour of contour lines for frequency bias.
    :param bias_line_width: Width of contour lines for frequency bias.
    :param csi_colour_map: Colour map (instance of `matplotlib.pyplot.cm`) for
        CSI (critical success index) contours.
    """

    error_checking.assert_is_numpy_array(pod_by_threshold, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(pod_by_threshold, 0.)
    error_checking.assert_is_leq_numpy_array(pod_by_threshold, 1.)
    num_thresholds = len(pod_by_threshold)

    error_checking.assert_is_numpy_array(
        success_ratio_by_threshold,
        exact_dimensions=numpy.array([num_thresholds]))
    error_checking.assert_is_geq_numpy_array(success_ratio_by_threshold, 0.)
    error_checking.assert_is_leq_numpy_array(success_ratio_by_threshold, 1.)

    success_ratio_matrix, pod_matrix = model_eval._get_sr_pod_grid()
    csi_matrix = model_eval.csi_from_sr_and_pod(
        success_ratio_matrix, pod_matrix)
    frequency_bias_matrix = model_eval.frequency_bias_from_sr_and_pod(
        success_ratio_matrix, pod_matrix)

    pyplot.contourf(
        success_ratio_matrix, pod_matrix, csi_matrix, LEVELS_FOR_CSI_CONTOURS,
        cmap=csi_colour_map, vmin=0., vmax=1., axes=axes_object)
    colour_bar_object = plotting_utils.add_linear_colour_bar(
        axes_object, values_to_colour=csi_matrix, colour_map=csi_colour_map,
        colour_min=0., colour_max=1., orientation='vertical', extend_min=False,
        extend_max=False)
    colour_bar_object.set_label('CSI (critical success index)')

    bias_colour_tuple = ()
    for _ in range(len(LEVELS_FOR_FREQ_BIAS_CONTOURS)):
        bias_colour_tuple += (bias_line_colour,)

    bias_contour_object = pyplot.contour(
        success_ratio_matrix, pod_matrix, frequency_bias_matrix,
        LEVELS_FOR_FREQ_BIAS_CONTOURS, colors=bias_colour_tuple,
        linewidths=bias_line_width, linestyles='dashed', axes=axes_object)
    pyplot.clabel(
        bias_contour_object, inline=True,
        inline_spacing=PIXEL_PADDING_FOR_FREQ_BIAS_LABELS,
        fmt=STRING_FORMAT_FOR_FREQ_BIAS_LABELS, fontsize=FONT_SIZE)

    axes_object.plot(
        success_ratio_by_threshold, pod_by_threshold, color=line_colour,
        linestyle='solid', linewidth=line_width)

    axes_object.set_xlabel('Success ratio (1 - FAR)')
    axes_object.set_ylabel('POD (probability of detection)')
    axes_object.set_xlim(0., 1.)
    axes_object.set_ylim(0., 1.)


def plot_reliability_curve(
        axes_object=None, mean_forecast_prob_by_bin=None,
        mean_observed_label_by_bin=None,
        line_colour=DEFAULT_RELIA_LINE_COLOUR,
        line_width=DEFAULT_RELIA_LINE_WIDTH,
        perfect_line_colour=DEFAULT_PERFECT_RELIA_LINE_COLOUR,
        perfect_line_width=DEFAULT_PERFECT_RELIA_LINE_WIDTH):
    """Plots reliability curve.

    B = number of forecast bins

    To learn more about `mean_forecast_prob_by_bin` and
    `mean_observed_label_by_bin`, see
    `model_evaluation.get_points_in_reliability_curve`.

    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param mean_forecast_prob_by_bin: length-B numpy array of mean forecast
        probabilities.
    :param mean_observed_label_by_bin: length-B numpy array of mean observed
        labels (conditional event frequencies).
    :param line_colour: Colour (in any format accepted by `matplotlib.colors`).
    :param line_width: Line width (real positive number).
    :param perfect_line_colour: Colour of reference line (reliability curve with
        reliability = 0).
    :param perfect_line_width: Width of reference line.
    """

    error_checking.assert_is_numpy_array(
        mean_forecast_prob_by_bin, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(mean_forecast_prob_by_bin, 0.)
    error_checking.assert_is_leq_numpy_array(mean_forecast_prob_by_bin, 1.)
    num_bins = len(mean_forecast_prob_by_bin)

    error_checking.assert_is_numpy_array(
        mean_observed_label_by_bin, exact_dimensions=numpy.array([num_bins]))
    error_checking.assert_is_geq_numpy_array(mean_observed_label_by_bin, 0.)
    error_checking.assert_is_leq_numpy_array(mean_observed_label_by_bin, 1.)

    perfect_x_coords, perfect_y_coords = (
        model_eval.get_perfect_reliability_curve())
    axes_object.plot(
        perfect_x_coords, perfect_y_coords, color=perfect_line_colour,
        linestyle='dashed', linewidth=perfect_line_width)

    axes_object.plot(
        mean_forecast_prob_by_bin, mean_observed_label_by_bin,
        color=line_colour, linestyle='solid', linewidth=line_width)

    axes_object.set_xlabel('Forecast probability')
    axes_object.set_ylabel('Conditional event frequency')
    axes_object.set_xlim(0., 1.)
    axes_object.set_ylim(0., 1.)


def plot_attributes_diagram(
        figure_object=None, axes_object=None, mean_forecast_prob_by_bin=None,
        mean_observed_label_by_bin=None, num_examples_by_bin=None,
        reliability_line_colour=DEFAULT_RELIA_LINE_COLOUR,
        reliability_line_width=DEFAULT_RELIA_LINE_WIDTH,
        perfect_relia_line_colour=DEFAULT_PERFECT_RELIA_LINE_COLOUR,
        perfect_relia_line_width=DEFAULT_PERFECT_RELIA_LINE_WIDTH,
        no_skill_line_colour=DEFAULT_NO_SKILL_RELIA_LINE_COLOUR,
        no_skill_line_width=DEFAULT_NO_SKILL_RELIA_LINE_WIDTH,
        other_line_colour=DEFAULT_CLIMATOLOGY_LINE_COLOUR,
        other_line_width=DEFAULT_CLIMATOLOGY_LINE_WIDTH,
        histogram_bar_face_colour=DEFAULT_HISTOGRAM_FACE_COLOUR,
        histogram_bar_edge_colour=DEFAULT_HISTOGRAM_EDGE_COLOUR,
        histogram_bar_edge_width=DEFAULT_HISTOGRAM_EDGE_WIDTH):
    """Plots attributes diagram (Hsu and Murphy 1986).

    :param figure_object: Instance of `matplotlib.figure.Figure`.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param mean_forecast_prob_by_bin: See documentation for
        plot_reliability_curve.
    :param mean_observed_label_by_bin: See doc for plot_reliability_curve.
    :param num_examples_by_bin: See doc for
        _plot_inset_histogram_for_attributes_diagram.
    :param reliability_line_colour: See doc for plot_reliability_curve.
    :param reliability_line_width: See doc for plot_reliability_curve.
    :param perfect_relia_line_colour: See doc for plot_reliability_curve.
    :param perfect_relia_line_width: See doc for plot_reliability_curve.
    :param no_skill_line_colour: See doc for
        _plot_background_of_attributes_diagram.
    :param no_skill_line_width: See doc for
        _plot_background_of_attributes_diagram.
    :param other_line_colour: See doc for
        _plot_background_of_attributes_diagram.
    :param other_line_width: See doc for _plot_background_of_attributes_diagram.
    :param histogram_bar_face_colour: See doc for
        _plot_inset_histogram_for_attributes_diagram.
    :param histogram_bar_edge_colour: See doc for
        _plot_inset_histogram_for_attributes_diagram.
    :param histogram_bar_edge_width: See doc for
        _plot_inset_histogram_for_attributes_diagram.
    """

    error_checking.assert_is_numpy_array(
        mean_observed_label_by_bin, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(mean_observed_label_by_bin, 0.)
    error_checking.assert_is_leq_numpy_array(mean_observed_label_by_bin, 1.)
    num_bins = len(mean_observed_label_by_bin)

    error_checking.assert_is_integer_numpy_array(num_examples_by_bin)
    error_checking.assert_is_numpy_array(
        num_examples_by_bin, exact_dimensions=numpy.array([num_bins]))
    error_checking.assert_is_geq_numpy_array(num_examples_by_bin, 0)

    mean_observed_label = numpy.average(
        mean_observed_label_by_bin, weights=num_examples_by_bin)

    _plot_background_of_attributes_diagram(
        axes_object=axes_object, mean_observed_label=mean_observed_label,
        no_skill_line_colour=no_skill_line_colour,
        no_skill_line_width=no_skill_line_width,
        other_line_colour=other_line_colour, other_line_width=other_line_width)

    _plot_inset_histogram_for_attributes_diagram(
        figure_object=figure_object, num_examples_by_bin=num_examples_by_bin,
        bar_face_colour=histogram_bar_face_colour,
        bar_edge_colour=histogram_bar_edge_colour,
        bar_edge_width=histogram_bar_edge_width)

    plot_reliability_curve(
        axes_object=axes_object,
        mean_forecast_prob_by_bin=mean_forecast_prob_by_bin,
        mean_observed_label_by_bin=mean_observed_label_by_bin,
        line_colour=reliability_line_colour, line_width=reliability_line_width,
        perfect_line_colour=perfect_relia_line_colour,
        perfect_line_width=perfect_relia_line_width)
