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
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
import matplotlib.colors
from gewittergefahr.gg_utils import model_evaluation as model_eval
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils

# TODO(thunderhoser): Some of the terminology in this file is too verbose, and
# some of the methods have way too many input args.

DEFAULT_ROC_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255
DEFAULT_ROC_WIDTH = 3.
DEFAULT_RANDOM_ROC_COLOUR = numpy.full(3, 152. / 255)
DEFAULT_RANDOM_ROC_WIDTH = 2.

DEFAULT_PERFORMANCE_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255
DEFAULT_PERFORMANCE_WIDTH = 3.

DEFAULT_FREQ_BIAS_COLOUR = numpy.full(3, 152. / 255)
DEFAULT_FREQ_BIAS_WIDTH = 2.
STRING_FORMAT_FOR_FREQ_BIAS_LABELS = '%.2f'
PIXEL_PADDING_FOR_FREQ_BIAS_LABELS = 10
LEVELS_FOR_FREQ_BIAS_CONTOURS = numpy.array([
    0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5
])

LEVELS_FOR_CSI_CONTOURS = numpy.linspace(0, 1, num=11, dtype=float)
LEVELS_FOR_PEIRCE_CONTOURS = numpy.linspace(0, 1, num=11, dtype=float)

DEFAULT_RELIABILITY_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255
DEFAULT_RELIABILITY_WIDTH = 3.
DEFAULT_PERFECT_RELIABILITY_COLOUR = numpy.full(3, 152. / 255)
DEFAULT_PERFECT_RELIABILITY_WIDTH = 2.
DEFAULT_ZERO_BSS_COLOUR = numpy.array([31, 120, 180], dtype=float) / 255
DEFAULT_ZERO_BSS_WIDTH = 2.
DEFAULT_CLIMATOLOGY_COLOUR = numpy.full(3, 152. / 255)
DEFAULT_CLIMATOLOGY_WIDTH = 2.

DEFAULT_HISTOGRAM_FACE_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255
DEFAULT_HISTOGRAM_EDGE_COLOUR = numpy.full(3, 0.)
DEFAULT_HISTOGRAM_EDGE_WIDTH = 2.

INSET_HISTOGRAM_LEFT_EDGE = 0.575
INSET_HISTOGRAM_BOTTOM_EDGE = 0.175
INSET_HISTOGRAM_WIDTH = 0.3
INSET_HISTOGRAM_HEIGHT = 0.3
INSET_HISTOGRAM_X_TICKS = numpy.linspace(0., 1., num=6)
INSET_HISTOGRAM_Y_TICK_SPACING = 0.1

TRANSPARENCY_FOR_CONFIDENCE_INTERVAL = 0.5
TRANSPARENCY_FOR_POSITIVE_BSS_AREA = 0.2

FONT_SIZE = 30
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)


def _get_csi_colour_scheme():
    """Returns colour scheme for CSI (critical success index).

    :return: colour_map_object: Colour scheme (instance of
        `matplotlib.colors.ListedColormap`).
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`,
        defining the scale of the colour map.
    """

    this_colour_map_object = pyplot.cm.Blues
    this_colour_norm_object = matplotlib.colors.BoundaryNorm(
        LEVELS_FOR_CSI_CONTOURS, this_colour_map_object.N)

    rgba_matrix = this_colour_map_object(this_colour_norm_object(
        LEVELS_FOR_CSI_CONTOURS
    ))

    colour_list = [
        rgba_matrix[i, ..., :-1] for i in range(rgba_matrix.shape[0])
    ]

    colour_map_object = matplotlib.colors.ListedColormap(colour_list)
    colour_map_object.set_under(numpy.full(3, 1.))
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        LEVELS_FOR_CSI_CONTOURS, colour_map_object.N)

    return colour_map_object, colour_norm_object


def _get_peirce_colour_scheme():
    """Returns colour scheme for Peirce score.

    :return: colour_map_object: Colour scheme (instance of
        `matplotlib.colors.ListedColormap`).
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`,
        defining the scale of the colour map.
    """

    this_colour_map_object = pyplot.cm.Blues
    this_colour_norm_object = matplotlib.colors.BoundaryNorm(
        LEVELS_FOR_PEIRCE_CONTOURS, this_colour_map_object.N)

    rgba_matrix = this_colour_map_object(this_colour_norm_object(
        LEVELS_FOR_PEIRCE_CONTOURS
    ))

    colour_list = [
        rgba_matrix[i, ..., :-1] for i in range(rgba_matrix.shape[0])
    ]

    colour_map_object = matplotlib.colors.ListedColormap(colour_list)
    colour_map_object.set_under(numpy.full(3, 1.))
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        LEVELS_FOR_PEIRCE_CONTOURS, colour_map_object.N)

    return colour_map_object, colour_norm_object


def _confidence_interval_to_polygon(
        x_coords_bottom, y_coords_bottom, x_coords_top, y_coords_top,
        for_performance_diagram=False):
    """Generates polygon for confidence interval.

    P = number of points in bottom curve = number of points in top curve

    :param x_coords_bottom: length-P numpy with x-coordinates of bottom curve
        (lower end of confidence interval).
    :param y_coords_bottom: Same but for y-coordinates.
    :param x_coords_top: length-P numpy with x-coordinates of top curve (upper
        end of confidence interval).
    :param y_coords_top: Same but for y-coordinates.
    :param for_performance_diagram: Boolean flag.  If True, confidence interval
        is for a performance diagram, which means that coordinates will be
        sorted in a slightly different way.
    :return: polygon_object: Instance of `shapely.geometry.Polygon`.
    """

    nan_flags_top = numpy.logical_or(
        numpy.isnan(x_coords_top), numpy.isnan(y_coords_top)
    )
    if numpy.all(nan_flags_top):
        return None

    nan_flags_bottom = numpy.logical_or(
        numpy.isnan(x_coords_bottom), numpy.isnan(y_coords_bottom)
    )
    if numpy.all(nan_flags_bottom):
        return None

    real_indices_top = numpy.where(
        numpy.invert(nan_flags_top)
    )[0]

    real_indices_bottom = numpy.where(
        numpy.invert(nan_flags_bottom)
    )[0]

    if for_performance_diagram:
        y_coords_top = y_coords_top[real_indices_top]
        sort_indices_top = numpy.argsort(y_coords_top)
        y_coords_top = y_coords_top[sort_indices_top]
        x_coords_top = x_coords_top[real_indices_top][sort_indices_top]

        y_coords_bottom = y_coords_bottom[real_indices_bottom]
        sort_indices_bottom = numpy.argsort(-y_coords_bottom)
        y_coords_bottom = y_coords_bottom[sort_indices_bottom]
        x_coords_bottom = x_coords_bottom[real_indices_bottom][
            sort_indices_bottom
        ]
    else:
        x_coords_top = x_coords_top[real_indices_top]
        sort_indices_top = numpy.argsort(-x_coords_top)
        x_coords_top = x_coords_top[sort_indices_top]
        y_coords_top = y_coords_top[real_indices_top][sort_indices_top]

        x_coords_bottom = x_coords_bottom[real_indices_bottom]
        sort_indices_bottom = numpy.argsort(x_coords_bottom)
        x_coords_bottom = x_coords_bottom[sort_indices_bottom]
        y_coords_bottom = y_coords_bottom[real_indices_bottom][
            sort_indices_bottom
        ]

    polygon_x_coords = numpy.concatenate((
        x_coords_top, x_coords_bottom, numpy.array([x_coords_top[0]])
    ))
    polygon_y_coords = numpy.concatenate((
        y_coords_top, y_coords_bottom, numpy.array([y_coords_top[0]])
    ))

    return polygons.vertex_arrays_to_polygon_object(
        polygon_x_coords, polygon_y_coords)


def _plot_background_of_attributes_diagram(
        axes_object, climatology,
        no_skill_line_colour=DEFAULT_ZERO_BSS_COLOUR,
        no_skill_line_width=DEFAULT_ZERO_BSS_WIDTH,
        other_line_colour=DEFAULT_CLIMATOLOGY_COLOUR,
        other_line_width=DEFAULT_CLIMATOLOGY_WIDTH):
    """Plots background (references lines and polygons) of attributes diagram.

    For more on the attributes diagram, see Hsu and Murphy (1986).

    BSS = Brier skill score.  For more on the BSS, see
    `model_evaluation.get_brier_skill_score`.

    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param climatology: Event frequency for the entire dataset.
    :param no_skill_line_colour: Colour (in any format accepted by
        `matplotlib.colors`) of no-skill line, where BSS = 0.
    :param no_skill_line_width: Width (real positive number) of no-skill line.
    :param other_line_colour: Colour of climatology and no-resolution lines.
    :param other_line_width: Width of climatology and no-resolution lines.
    """

    error_checking.assert_is_geq(climatology, 0.)
    error_checking.assert_is_leq(climatology, 1.)

    (x_vertices_for_left_skill_area,
     y_vertices_for_left_skill_area,
     x_vertices_for_right_skill_area,
     y_vertices_for_right_skill_area
    ) = model_eval.get_skill_areas_in_reliability_curve(climatology)

    skill_area_colour = matplotlib.colors.to_rgba(
        plotting_utils.colour_from_numpy_to_tuple(no_skill_line_colour),
        TRANSPARENCY_FOR_POSITIVE_BSS_AREA
    )

    left_polygon_object = polygons.vertex_arrays_to_polygon_object(
        x_vertices_for_left_skill_area, y_vertices_for_left_skill_area
    )
    left_polygon_patch = PolygonPatch(
        left_polygon_object, lw=0, ec=skill_area_colour, fc=skill_area_colour
    )

    axes_object.add_patch(left_polygon_patch)

    right_polygon_object = polygons.vertex_arrays_to_polygon_object(
        x_vertices_for_right_skill_area, y_vertices_for_right_skill_area
    )
    right_polygon_patch = PolygonPatch(
        right_polygon_object, lw=0, ec=skill_area_colour, fc=skill_area_colour
    )

    axes_object.add_patch(right_polygon_patch)

    no_skill_x_coords, no_skill_y_coords = (
        model_eval.get_no_skill_reliability_curve(climatology)
    )

    axes_object.plot(
        no_skill_x_coords, no_skill_y_coords,
        color=plotting_utils.colour_from_numpy_to_tuple(no_skill_line_colour),
        linestyle='solid', linewidth=no_skill_line_width
    )

    climo_x_coords, climo_y_coords = (
        model_eval.get_climatology_line_for_reliability_curve(
            climatology)
    )

    axes_object.plot(
        climo_x_coords, climo_y_coords,
        color=plotting_utils.colour_from_numpy_to_tuple(other_line_colour),
        linestyle='dashed', linewidth=other_line_width
    )

    no_resolution_x_coords, no_resolution_y_coords = (
        model_eval.get_no_resolution_line_for_reliability_curve(
            climatology)
    )

    axes_object.plot(
        no_resolution_x_coords, no_resolution_y_coords,
        color=plotting_utils.colour_from_numpy_to_tuple(other_line_colour),
        linestyle='dashed', linewidth=other_line_width
    )


def _plot_inset_histogram_for_attributes_diagram(
        figure_object, num_examples_by_bin,
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
        num_examples_by_bin.astype(float) / numpy.sum(num_examples_by_bin)
    )

    forecast_bin_edges = numpy.linspace(0., 1., num=num_forecast_bins + 1)
    forecast_bin_width = forecast_bin_edges[1] - forecast_bin_edges[0]
    forecast_bin_centers = forecast_bin_edges[:-1] + forecast_bin_width / 2

    inset_axes_object = figure_object.add_axes([
        INSET_HISTOGRAM_LEFT_EDGE, INSET_HISTOGRAM_BOTTOM_EDGE,
        INSET_HISTOGRAM_WIDTH, INSET_HISTOGRAM_HEIGHT
    ])

    inset_axes_object.bar(
        forecast_bin_centers, example_frequency_by_bin, forecast_bin_width,
        color=plotting_utils.colour_from_numpy_to_tuple(bar_face_colour),
        edgecolor=plotting_utils.colour_from_numpy_to_tuple(bar_edge_colour),
        linewidth=bar_edge_width
    )

    max_y_tick_value = rounder.floor_to_nearest(
        1.05 * numpy.max(example_frequency_by_bin),
        INSET_HISTOGRAM_Y_TICK_SPACING
    )
    num_y_ticks = 1 + int(numpy.round(
        max_y_tick_value / INSET_HISTOGRAM_Y_TICK_SPACING
    ))
    y_tick_values = numpy.linspace(0., max_y_tick_value, num=num_y_ticks)

    pyplot.xticks(INSET_HISTOGRAM_X_TICKS, axes=inset_axes_object)
    pyplot.yticks(y_tick_values, axes=inset_axes_object)
    inset_axes_object.set_xlim(0., 1.)
    inset_axes_object.set_ylim(0., 1.05 * numpy.max(example_frequency_by_bin))


def plot_roc_curve(
        axes_object, pod_by_threshold, pofd_by_threshold,
        line_colour=DEFAULT_ROC_COLOUR, line_width=DEFAULT_ROC_WIDTH,
        random_line_colour=DEFAULT_RANDOM_ROC_COLOUR,
        random_line_width=DEFAULT_RANDOM_ROC_WIDTH):
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
    error_checking.assert_is_geq_numpy_array(
        pod_by_threshold, 0., allow_nan=True)
    error_checking.assert_is_leq_numpy_array(
        pod_by_threshold, 1., allow_nan=True)

    num_thresholds = len(pod_by_threshold)
    expected_dim = numpy.array([num_thresholds], dtype=int)

    error_checking.assert_is_numpy_array(
        pofd_by_threshold, exact_dimensions=expected_dim)
    error_checking.assert_is_geq_numpy_array(
        pofd_by_threshold, 0., allow_nan=True)
    error_checking.assert_is_leq_numpy_array(
        pofd_by_threshold, 1., allow_nan=True)

    pofd_matrix, pod_matrix = model_eval.get_pofd_pod_grid()
    peirce_score_matrix = pod_matrix - pofd_matrix

    this_colour_map_object, this_colour_norm_object = (
        _get_peirce_colour_scheme()
    )

    pyplot.contourf(
        pofd_matrix, pod_matrix, peirce_score_matrix, LEVELS_FOR_CSI_CONTOURS,
        cmap=this_colour_map_object, norm=this_colour_norm_object, vmin=0.,
        vmax=1., axes=axes_object)

    colour_bar_object = plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=peirce_score_matrix,
        colour_map_object=this_colour_map_object,
        colour_norm_object=this_colour_norm_object,
        orientation_string='vertical', extend_min=False, extend_max=False)

    colour_bar_object.set_label('Peirce score (POD minus POFD)')

    random_x_coords, random_y_coords = model_eval.get_random_roc_curve()
    axes_object.plot(
        random_x_coords, random_y_coords,
        color=plotting_utils.colour_from_numpy_to_tuple(random_line_colour),
        linestyle='dashed', linewidth=random_line_width
    )

    nan_flags = numpy.logical_or(
        numpy.isnan(pofd_by_threshold), numpy.isnan(pod_by_threshold)
    )

    if not numpy.all(nan_flags):
        real_indices = numpy.where(numpy.invert(nan_flags))[0]

        axes_object.plot(
            pofd_by_threshold[real_indices], pod_by_threshold[real_indices],
            color=plotting_utils.colour_from_numpy_to_tuple(line_colour),
            linestyle='solid', linewidth=line_width
        )

    axes_object.set_xlabel('POFD (probability of false detection)')
    axes_object.set_ylabel('POD (probability of detection)')
    axes_object.set_xlim(0., 1.)
    axes_object.set_ylim(0., 1.)


def plot_bootstrapped_roc_curve(
        axes_object, ci_bottom_dict, ci_mean_dict, ci_top_dict,
        line_colour=DEFAULT_ROC_COLOUR,
        line_width=DEFAULT_ROC_WIDTH,
        random_line_colour=DEFAULT_RANDOM_ROC_COLOUR,
        random_line_width=DEFAULT_RANDOM_ROC_WIDTH):
    """Bootstrapped version of plot_roc_curve.

    T = number of probability thresholds in curve

    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param ci_bottom_dict: Dictionary with the following keys, representing the
        bottom of the confidence interval.
    ci_bottom_dict['pod_by_threshold']: length-T numpy array of POD values
        (probability of detection).
    ci_bottom_dict['pofd_by_threshold']: length-T numpy array of POFD values
        (probability of false detection).

    :param ci_mean_dict: Same but for mean of confidence interval.
    :param ci_top_dict: Same but for top of confidence interval.
    :param line_colour: Colour for ROC curve (mean of confidence interval; in
        any format accepted by `matplotlib.colors`).  The rest of the confidence
        interval will be shaded with the same colour but 50% opacity.
    :param line_width: Line width for ROC curve (mean of confidence interval).
    :param random_line_colour: Colour of reference line (ROC curve for a random
        predictor).
    :param random_line_width: Width of reference line.
    """

    plot_roc_curve(
        axes_object=axes_object,
        pod_by_threshold=ci_mean_dict[model_eval.POD_BY_THRESHOLD_KEY],
        pofd_by_threshold=ci_mean_dict[model_eval.POFD_BY_THRESHOLD_KEY],
        line_colour=line_colour, line_width=line_width,
        random_line_colour=random_line_colour,
        random_line_width=random_line_width)

    polygon_object = _confidence_interval_to_polygon(
        x_coords_bottom=ci_bottom_dict[model_eval.POFD_BY_THRESHOLD_KEY],
        y_coords_bottom=ci_bottom_dict[model_eval.POD_BY_THRESHOLD_KEY],
        x_coords_top=ci_top_dict[model_eval.POFD_BY_THRESHOLD_KEY],
        y_coords_top=ci_top_dict[model_eval.POD_BY_THRESHOLD_KEY]
    )

    if polygon_object is None:
        return

    polygon_colour = matplotlib.colors.to_rgba(
        plotting_utils.colour_from_numpy_to_tuple(line_colour),
        TRANSPARENCY_FOR_CONFIDENCE_INTERVAL
    )

    polygon_patch = PolygonPatch(
        polygon_object, lw=0, ec=polygon_colour, fc=polygon_colour)

    axes_object.add_patch(polygon_patch)


def plot_performance_diagram(
        axes_object, pod_by_threshold, success_ratio_by_threshold,
        line_colour=DEFAULT_PERFORMANCE_COLOUR,
        line_width=DEFAULT_PERFORMANCE_WIDTH,
        bias_line_colour=DEFAULT_FREQ_BIAS_COLOUR,
        bias_line_width=DEFAULT_FREQ_BIAS_WIDTH):
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
    """

    error_checking.assert_is_numpy_array(pod_by_threshold, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(
        pod_by_threshold, 0., allow_nan=True)
    error_checking.assert_is_leq_numpy_array(
        pod_by_threshold, 1., allow_nan=True)

    num_thresholds = len(pod_by_threshold)
    expected_dim = numpy.array([num_thresholds], dtype=int)

    error_checking.assert_is_numpy_array(
        success_ratio_by_threshold, exact_dimensions=expected_dim)
    error_checking.assert_is_geq_numpy_array(
        success_ratio_by_threshold, 0., allow_nan=True)
    error_checking.assert_is_leq_numpy_array(
        success_ratio_by_threshold, 1., allow_nan=True)

    success_ratio_matrix, pod_matrix = model_eval.get_sr_pod_grid()
    csi_matrix = model_eval.csi_from_sr_and_pod(
        success_ratio_matrix, pod_matrix)
    frequency_bias_matrix = model_eval.frequency_bias_from_sr_and_pod(
        success_ratio_matrix, pod_matrix)

    this_colour_map_object, this_colour_norm_object = _get_csi_colour_scheme()

    pyplot.contourf(
        success_ratio_matrix, pod_matrix, csi_matrix, LEVELS_FOR_CSI_CONTOURS,
        cmap=this_colour_map_object, norm=this_colour_norm_object, vmin=0.,
        vmax=1., axes=axes_object)

    colour_bar_object = plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=csi_matrix,
        colour_map_object=this_colour_map_object,
        colour_norm_object=this_colour_norm_object,
        orientation_string='vertical', extend_min=False, extend_max=False)

    colour_bar_object.set_label('CSI (critical success index)')

    bias_colour_tuple = plotting_utils.colour_from_numpy_to_tuple(
        bias_line_colour)

    bias_colours_2d_tuple = ()
    for _ in range(len(LEVELS_FOR_FREQ_BIAS_CONTOURS)):
        bias_colours_2d_tuple += (bias_colour_tuple,)

    bias_contour_object = pyplot.contour(
        success_ratio_matrix, pod_matrix, frequency_bias_matrix,
        LEVELS_FOR_FREQ_BIAS_CONTOURS, colors=bias_colours_2d_tuple,
        linewidths=bias_line_width, linestyles='dashed', axes=axes_object)

    pyplot.clabel(
        bias_contour_object, inline=True,
        inline_spacing=PIXEL_PADDING_FOR_FREQ_BIAS_LABELS,
        fmt=STRING_FORMAT_FOR_FREQ_BIAS_LABELS, fontsize=FONT_SIZE)

    nan_flags = numpy.logical_or(
        numpy.isnan(success_ratio_by_threshold), numpy.isnan(pod_by_threshold)
    )

    if not numpy.all(nan_flags):
        real_indices = numpy.where(numpy.invert(nan_flags))[0]

        axes_object.plot(
            success_ratio_by_threshold[real_indices],
            pod_by_threshold[real_indices],
            color=plotting_utils.colour_from_numpy_to_tuple(line_colour),
            linestyle='solid', linewidth=line_width
        )

    axes_object.set_xlabel('Success ratio (1 - FAR)')
    axes_object.set_ylabel('POD (probability of detection)')
    axes_object.set_xlim(0., 1.)
    axes_object.set_ylim(0., 1.)


def plot_bootstrapped_performance_diagram(
        axes_object, ci_bottom_dict, ci_mean_dict, ci_top_dict,
        line_colour=DEFAULT_PERFORMANCE_COLOUR,
        line_width=DEFAULT_PERFORMANCE_WIDTH,
        bias_line_colour=DEFAULT_FREQ_BIAS_COLOUR,
        bias_line_width=DEFAULT_FREQ_BIAS_WIDTH):
    """Bootstrapped version of plot_performance_diagram.

    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param ci_bottom_dict: Dictionary with the following keys,
        representing the bottom of the confidence interval.
    ci_bottom_dict['pod_by_threshold']: length-T numpy array of POD
        values (probability of detection).
    ci_bottom_dict['success_ratio_by_threshold']: length-T numpy array of
        success ratios.

    :param ci_mean_dict: Same but for mean of confidence interval.
    :param ci_top_dict: Same but for top of confidence interval.
    :param line_colour: Colour for performance curve (mean of confidence
        interval; in any format accepted by `matplotlib.colors`).  The rest of
        the confidence interval will be shaded with the same colour but 50%
        opacity.
    :param line_width: Line width for performance curve (mean of confidence
        interval).
    :param bias_line_colour: Colour of contour lines for frequency bias.
    :param bias_line_width: Width of contour lines for frequency bias.
    """

    plot_performance_diagram(
        axes_object=axes_object,
        pod_by_threshold=ci_mean_dict[model_eval.POD_BY_THRESHOLD_KEY],
        success_ratio_by_threshold=ci_mean_dict[model_eval.SR_BY_THRESHOLD_KEY],
        line_colour=line_colour, line_width=line_width,
        bias_line_colour=bias_line_colour, bias_line_width=bias_line_width)

    polygon_object = _confidence_interval_to_polygon(
        x_coords_bottom=ci_bottom_dict[model_eval.SR_BY_THRESHOLD_KEY],
        y_coords_bottom=ci_bottom_dict[model_eval.POD_BY_THRESHOLD_KEY],
        x_coords_top=ci_top_dict[model_eval.SR_BY_THRESHOLD_KEY],
        y_coords_top=ci_top_dict[model_eval.POD_BY_THRESHOLD_KEY],
        for_performance_diagram=True)

    if polygon_object is None:
        return

    polygon_colour = matplotlib.colors.to_rgba(
        plotting_utils.colour_from_numpy_to_tuple(line_colour),
        TRANSPARENCY_FOR_CONFIDENCE_INTERVAL
    )

    polygon_patch = PolygonPatch(
        polygon_object, lw=0, ec=polygon_colour, fc=polygon_colour)

    axes_object.add_patch(polygon_patch)


def plot_reliability_curve(
        axes_object, mean_forecast_by_bin, event_frequency_by_bin,
        line_colour=DEFAULT_RELIABILITY_COLOUR,
        line_width=DEFAULT_RELIABILITY_WIDTH,
        perfect_line_colour=DEFAULT_PERFECT_RELIABILITY_COLOUR,
        perfect_line_width=DEFAULT_PERFECT_RELIABILITY_WIDTH):
    """Plots reliability curve.

    B = number of bins (separated by forecast probability)

    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param mean_forecast_by_bin: length-B numpy array of mean forecast
        probabilities.
    :param event_frequency_by_bin: length-B numpy array of mean observed
        labels (conditional event frequencies).
    :param line_colour: Colour (in any format accepted by `matplotlib.colors`).
    :param line_width: Line width (real positive number).
    :param perfect_line_colour: Colour of reference line (reliability curve with
        reliability = 0).
    :param perfect_line_width: Width of reference line.
    """

    error_checking.assert_is_numpy_array(
        mean_forecast_by_bin, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(
        mean_forecast_by_bin, 0., allow_nan=True)
    error_checking.assert_is_leq_numpy_array(
        mean_forecast_by_bin, 1., allow_nan=True)

    num_bins = len(mean_forecast_by_bin)
    expected_dim = numpy.array([num_bins], dtype=int)

    error_checking.assert_is_numpy_array(
        event_frequency_by_bin, exact_dimensions=expected_dim)
    error_checking.assert_is_geq_numpy_array(
        event_frequency_by_bin, 0., allow_nan=True)
    error_checking.assert_is_leq_numpy_array(
        event_frequency_by_bin, 1., allow_nan=True)

    perfect_x_coords, perfect_y_coords = (
        model_eval.get_perfect_reliability_curve()
    )

    axes_object.plot(
        perfect_x_coords, perfect_y_coords,
        color=plotting_utils.colour_from_numpy_to_tuple(perfect_line_colour),
        linestyle='dashed', linewidth=perfect_line_width
    )

    nan_flags = numpy.logical_or(
        numpy.isnan(mean_forecast_by_bin), numpy.isnan(event_frequency_by_bin)
    )

    if not numpy.all(nan_flags):
        real_indices = numpy.where(numpy.invert(nan_flags))[0]

        print(mean_forecast_by_bin[real_indices])
        print(event_frequency_by_bin[real_indices])

        axes_object.plot(
            mean_forecast_by_bin[real_indices],
            event_frequency_by_bin[real_indices],
            color=plotting_utils.colour_from_numpy_to_tuple(line_colour),
            linestyle='solid', linewidth=line_width
        )

    axes_object.set_xlabel('Forecast probability')
    axes_object.set_ylabel('Conditional event frequency')
    axes_object.set_xlim(0., 1.)
    axes_object.set_ylim(0., 1.)


def plot_bootstrapped_reliability_curve(
        axes_object, ci_bottom_dict, ci_mean_dict, ci_top_dict,
        line_colour=DEFAULT_RELIABILITY_COLOUR,
        line_width=DEFAULT_RELIABILITY_WIDTH,
        perfect_line_colour=DEFAULT_PERFECT_RELIABILITY_COLOUR,
        perfect_line_width=DEFAULT_PERFECT_RELIABILITY_WIDTH):
    """Bootstrapped version of plot_reliability_curve.

    B = number of bins (separated by forecast probability)

    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param ci_bottom_dict: Dictionary with the following keys,
        representing the bottom of the confidence interval.
    ci_bottom_dict['mean_forecast_by_bin']: length-B numpy array of mean
        forecast probabilities.
    ci_bottom_dict['event_frequency_by_bin']: length-B numpy array of
        conditional event frequencies.

    :param ci_mean_dict: Same but for mean of confidence interval.
    :param ci_top_dict: Same but for top of confidence interval.
    :param line_colour: Colour for reliability curve (mean of confidence
        interval; in any format accepted by `matplotlib.colors`).  The rest of
        the confidence interval will be shaded with the same colour but 50%
        opacity.
    :param line_width: Line width for reliability curve (mean of confidence
        interval).
    :param perfect_line_colour: Colour of reference line (perfect reliability
        curve).
    :param perfect_line_width: Width of reference line.
    """

    plot_reliability_curve(
        axes_object=axes_object,
        mean_forecast_by_bin=ci_mean_dict[
            model_eval.MEAN_FORECAST_BY_BIN_KEY],
        event_frequency_by_bin=ci_mean_dict[model_eval.EVENT_FREQ_BY_BIN_KEY],
        line_colour=line_colour, line_width=line_width,
        perfect_line_colour=perfect_line_colour,
        perfect_line_width=perfect_line_width)

    polygon_object = _confidence_interval_to_polygon(
        x_coords_bottom=ci_bottom_dict[model_eval.MEAN_FORECAST_BY_BIN_KEY],
        y_coords_bottom=ci_bottom_dict[model_eval.EVENT_FREQ_BY_BIN_KEY],
        x_coords_top=ci_top_dict[model_eval.MEAN_FORECAST_BY_BIN_KEY],
        y_coords_top=ci_top_dict[model_eval.EVENT_FREQ_BY_BIN_KEY]
    )

    if polygon_object is None:
        return

    polygon_colour = matplotlib.colors.to_rgba(
        plotting_utils.colour_from_numpy_to_tuple(line_colour),
        TRANSPARENCY_FOR_CONFIDENCE_INTERVAL
    )

    polygon_patch = PolygonPatch(
        polygon_object, lw=0, ec=polygon_colour, fc=polygon_colour)

    axes_object.add_patch(polygon_patch)


def plot_attributes_diagram(
        figure_object, axes_object, mean_forecast_by_bin,
        event_frequency_by_bin, num_examples_by_bin,
        reliability_line_colour=DEFAULT_RELIABILITY_COLOUR,
        reliability_line_width=DEFAULT_RELIABILITY_WIDTH,
        perfect_relia_line_colour=DEFAULT_PERFECT_RELIABILITY_COLOUR,
        perfect_relia_line_width=DEFAULT_PERFECT_RELIABILITY_WIDTH,
        no_skill_line_colour=DEFAULT_ZERO_BSS_COLOUR,
        no_skill_line_width=DEFAULT_ZERO_BSS_WIDTH,
        other_line_colour=DEFAULT_CLIMATOLOGY_COLOUR,
        other_line_width=DEFAULT_CLIMATOLOGY_WIDTH,
        histogram_bar_face_colour=DEFAULT_HISTOGRAM_FACE_COLOUR,
        histogram_bar_edge_colour=DEFAULT_HISTOGRAM_EDGE_COLOUR,
        histogram_bar_edge_width=DEFAULT_HISTOGRAM_EDGE_WIDTH):
    """Plots attributes diagram (Hsu and Murphy 1986).

    :param figure_object: Instance of `matplotlib.figure.Figure`.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param mean_forecast_by_bin: See doc for `plot_reliability_curve`.
    :param event_frequency_by_bin: Same.
    :param num_examples_by_bin: See doc for
        `_plot_inset_histogram_for_attributes_diagram`.
    :param reliability_line_colour: See doc for `plot_reliability_curve`.
    :param reliability_line_width: Same.
    :param perfect_relia_line_colour: Same.
    :param perfect_relia_line_width: Same.
    :param no_skill_line_colour: See doc for
        `_plot_background_of_attributes_diagram`.
    :param no_skill_line_width: Same.
    :param other_line_colour: Same.
    :param other_line_width: Same.
    :param histogram_bar_face_colour: See doc for
        `_plot_inset_histogram_for_attributes_diagram`.
    :param histogram_bar_edge_colour: Same.
    :param histogram_bar_edge_width: Same.
    """

    error_checking.assert_is_numpy_array(
        event_frequency_by_bin, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(
        event_frequency_by_bin, 0., allow_nan=True)
    error_checking.assert_is_leq_numpy_array(
        event_frequency_by_bin, 1., allow_nan=True)

    num_bins = len(mean_forecast_by_bin)
    expected_dim = numpy.array([num_bins], dtype=int)

    error_checking.assert_is_integer_numpy_array(num_examples_by_bin)
    error_checking.assert_is_numpy_array(
        num_examples_by_bin, exact_dimensions=expected_dim)
    error_checking.assert_is_geq_numpy_array(num_examples_by_bin, 0)

    non_empty_bin_indices = numpy.where(num_examples_by_bin > 0)[0]
    error_checking.assert_is_numpy_array_without_nan(
        event_frequency_by_bin[non_empty_bin_indices]
    )

    climatology = numpy.average(
        event_frequency_by_bin[non_empty_bin_indices],
        weights=num_examples_by_bin[non_empty_bin_indices]
    )

    _plot_background_of_attributes_diagram(
        axes_object=axes_object, climatology=climatology,
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
        mean_forecast_by_bin=mean_forecast_by_bin,
        event_frequency_by_bin=event_frequency_by_bin,
        line_colour=reliability_line_colour, line_width=reliability_line_width,
        perfect_line_colour=perfect_relia_line_colour,
        perfect_line_width=perfect_relia_line_width)


def plot_bootstrapped_attributes_diagram(
        figure_object, axes_object, ci_bottom_dict, ci_mean_dict, ci_top_dict,
        num_examples_by_bin,
        reliability_line_colour=DEFAULT_RELIABILITY_COLOUR,
        reliability_line_width=DEFAULT_RELIABILITY_WIDTH,
        perfect_relia_line_colour=DEFAULT_PERFECT_RELIABILITY_COLOUR,
        perfect_relia_line_width=DEFAULT_PERFECT_RELIABILITY_WIDTH,
        no_skill_line_colour=DEFAULT_ZERO_BSS_COLOUR,
        no_skill_line_width=DEFAULT_ZERO_BSS_WIDTH,
        other_line_colour=DEFAULT_CLIMATOLOGY_COLOUR,
        other_line_width=DEFAULT_CLIMATOLOGY_WIDTH,
        histogram_bar_face_colour=DEFAULT_HISTOGRAM_FACE_COLOUR,
        histogram_bar_edge_colour=DEFAULT_HISTOGRAM_EDGE_COLOUR,
        histogram_bar_edge_width=DEFAULT_HISTOGRAM_EDGE_WIDTH):
    """Bootstrapped version of plot_attributes_diagram.

    :param figure_object: Instance of `matplotlib.figure.Figure`.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param ci_bottom_dict: See doc for `plot_bootstrapped_reliability_curve`.
    :param ci_mean_dict: Same.
    :param ci_top_dict: Same.
    :param num_examples_by_bin: See doc for `plot_attributes_diagram`.
    :param reliability_line_colour: Same.
    :param reliability_line_width: Same.
    :param perfect_relia_line_colour: Same.
    :param perfect_relia_line_width: Same.
    :param no_skill_line_colour: Same.
    :param no_skill_line_width: Same.
    :param other_line_colour: Same.
    :param other_line_width: Same.
    :param histogram_bar_face_colour: Same.
    :param histogram_bar_edge_colour: Same.
    :param histogram_bar_edge_width: Same.
    """

    plot_attributes_diagram(
        figure_object=figure_object, axes_object=axes_object,
        mean_forecast_by_bin=ci_mean_dict[model_eval.MEAN_FORECAST_BY_BIN_KEY],
        event_frequency_by_bin=ci_mean_dict[model_eval.EVENT_FREQ_BY_BIN_KEY],
        num_examples_by_bin=num_examples_by_bin,
        reliability_line_colour=reliability_line_colour,
        reliability_line_width=reliability_line_width,
        perfect_relia_line_colour=perfect_relia_line_colour,
        perfect_relia_line_width=perfect_relia_line_width,
        no_skill_line_colour=no_skill_line_colour,
        no_skill_line_width=no_skill_line_width,
        other_line_colour=other_line_colour, other_line_width=other_line_width,
        histogram_bar_face_colour=histogram_bar_face_colour,
        histogram_bar_edge_colour=histogram_bar_edge_colour,
        histogram_bar_edge_width=histogram_bar_edge_width)

    polygon_object = _confidence_interval_to_polygon(
        x_coords_bottom=ci_bottom_dict[model_eval.MEAN_FORECAST_BY_BIN_KEY],
        y_coords_bottom=ci_bottom_dict[model_eval.EVENT_FREQ_BY_BIN_KEY],
        x_coords_top=ci_top_dict[model_eval.MEAN_FORECAST_BY_BIN_KEY],
        y_coords_top=ci_top_dict[model_eval.EVENT_FREQ_BY_BIN_KEY]
    )

    if polygon_object is None:
        return

    polygon_colour = matplotlib.colors.to_rgba(
        plotting_utils.colour_from_numpy_to_tuple(reliability_line_colour),
        TRANSPARENCY_FOR_CONFIDENCE_INTERVAL
    )

    polygon_patch = PolygonPatch(
        polygon_object, lw=0, ec=polygon_colour, fc=polygon_colour)

    axes_object.add_patch(polygon_patch)
