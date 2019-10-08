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

# TODO(thunderhoser): Variable and method names are way too verbose.

ROC_CURVE_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255
ROC_CURVE_WIDTH = 3.
RANDOM_ROC_COLOUR = numpy.full(3, 152. / 255)
RANDOM_ROC_WIDTH = 2.

PERF_DIAGRAM_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255
PERF_DIAGRAM_WIDTH = 3.

FREQ_BIAS_COLOUR = numpy.full(3, 152. / 255)
FREQ_BIAS_WIDTH = 2.
FREQ_BIAS_STRING_FORMAT = '%.2f'
FREQ_BIAS_PADDING = 10
FREQ_BIAS_LEVELS = numpy.array([0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5])

CSI_LEVELS = numpy.linspace(0, 1, num=11, dtype=float)
PEIRCE_SCORE_LEVELS = numpy.linspace(0, 1, num=11, dtype=float)

RELIABILITY_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255
RELIABILITY_WIDTH = 3.
PERFECT_RELIA_COLOUR = numpy.full(3, 152. / 255)
PERFECT_RELIA_WIDTH = 2.
ZERO_BSS_COLOUR = numpy.array([31, 120, 180], dtype=float) / 255
ZERO_BSS_LINE_WIDTH = 2.
CLIMO_COLOUR = numpy.full(3, 152. / 255)
CLIMO_LINE_WIDTH = 2.

BAR_FACE_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255
BAR_EDGE_COLOUR = numpy.full(3, 0.)
BAR_EDGE_WIDTH = 2.

HISTOGRAM_LEFT_EDGE = 0.2
HISTOGRAM_BOTTOM_EDGE = 0.575
HISTOGRAM_AXES_WIDTH = 0.25
HISTOGRAM_AXES_HEIGHT = 0.25
HISTOGRAM_X_VALUES = numpy.linspace(0., 1., num=6)
HISTOGRAM_Y_SPACING = 0.1

POLYGON_OPACITY = 0.7
POSITIVE_BSS_OPACITY = 0.2

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

    this_colour_map_object = pyplot.get_cmap('Blues')
    this_colour_norm_object = matplotlib.colors.BoundaryNorm(
        CSI_LEVELS, this_colour_map_object.N)

    rgba_matrix = this_colour_map_object(this_colour_norm_object(CSI_LEVELS))
    colour_list = [
        rgba_matrix[i, ..., :-1] for i in range(rgba_matrix.shape[0])
    ]

    colour_map_object = matplotlib.colors.ListedColormap(colour_list)
    colour_map_object.set_under(numpy.full(3, 1.))
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        CSI_LEVELS, colour_map_object.N)

    return colour_map_object, colour_norm_object


def _get_peirce_colour_scheme():
    """Returns colour scheme for Peirce score.

    :return: colour_map_object: Colour scheme (instance of
        `matplotlib.colors.ListedColormap`).
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`,
        defining the scale of the colour map.
    """

    this_colour_map_object = pyplot.get_cmap('Blues')
    this_colour_norm_object = matplotlib.colors.BoundaryNorm(
        PEIRCE_SCORE_LEVELS, this_colour_map_object.N)

    rgba_matrix = this_colour_map_object(
        this_colour_norm_object(PEIRCE_SCORE_LEVELS)
    )
    colour_list = [
        rgba_matrix[i, ..., :-1] for i in range(rgba_matrix.shape[0])
    ]

    colour_map_object = matplotlib.colors.ListedColormap(colour_list)
    colour_map_object.set_under(numpy.full(3, 1.))
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        PEIRCE_SCORE_LEVELS, colour_map_object.N)

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

    real_indices_top = numpy.where(numpy.invert(nan_flags_top))[0]
    real_indices_bottom = numpy.where(numpy.invert(nan_flags_bottom))[0]

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


def _plot_background_of_attributes_diagram(axes_object, climatology):
    """Plots background (references lines and polygons) of attributes diagram.

    For more on the attributes diagram, see Hsu and Murphy (1986).

    BSS = Brier skill score.  For more on the BSS, see
    `model_evaluation.get_brier_skill_score`.

    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param climatology: Event frequency for the entire dataset.
    """

    error_checking.assert_is_geq(climatology, 0.)
    error_checking.assert_is_leq(climatology, 1.)

    (x_coords_left_skill_area, y_coords_left_skill_area,
     x_coords_right_skill_area, y_coords_right_skill_area
    ) = model_eval.get_skill_areas_in_reliability_curve(climatology)

    skill_area_colour = matplotlib.colors.to_rgba(
        plotting_utils.colour_from_numpy_to_tuple(ZERO_BSS_COLOUR),
        POSITIVE_BSS_OPACITY
    )

    left_polygon_object = polygons.vertex_arrays_to_polygon_object(
        x_coords_left_skill_area, y_coords_left_skill_area
    )
    left_polygon_patch = PolygonPatch(
        left_polygon_object, lw=0, ec=skill_area_colour, fc=skill_area_colour
    )

    axes_object.add_patch(left_polygon_patch)

    right_polygon_object = polygons.vertex_arrays_to_polygon_object(
        x_coords_right_skill_area, y_coords_right_skill_area
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
        color=plotting_utils.colour_from_numpy_to_tuple(ZERO_BSS_COLOUR),
        linestyle='solid', linewidth=ZERO_BSS_LINE_WIDTH
    )

    climo_x_coords, climo_y_coords = (
        model_eval.get_climatology_line_for_reliability_curve(climatology)
    )

    axes_object.plot(
        climo_x_coords, climo_y_coords,
        color=plotting_utils.colour_from_numpy_to_tuple(CLIMO_COLOUR),
        linestyle='dashed', linewidth=CLIMO_LINE_WIDTH
    )

    no_resolution_x_coords, no_resolution_y_coords = (
        model_eval.get_no_resolution_line_for_reliability_curve(climatology)
    )

    axes_object.plot(
        no_resolution_x_coords, no_resolution_y_coords,
        color=plotting_utils.colour_from_numpy_to_tuple(CLIMO_COLOUR),
        linestyle='dashed', linewidth=CLIMO_LINE_WIDTH
    )


def _plot_inset_histogram_for_attributes_diagram(
        figure_object, num_examples_by_bin):
    """Plots forecast histogram inset in attributes diagram.

    For more on the attributes diagram, see Hsu and Murphy (1986).

    B = number of forecast bins

    :param figure_object: Instance of `matplotlib.figure.Figure`.
    :param num_examples_by_bin: length-B numpy array with number of examples in
        each forecast bin.
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
        HISTOGRAM_LEFT_EDGE, HISTOGRAM_BOTTOM_EDGE,
        HISTOGRAM_AXES_WIDTH, HISTOGRAM_AXES_HEIGHT
    ])

    inset_axes_object.bar(
        forecast_bin_centers, example_frequency_by_bin, forecast_bin_width,
        color=plotting_utils.colour_from_numpy_to_tuple(BAR_FACE_COLOUR),
        edgecolor=plotting_utils.colour_from_numpy_to_tuple(BAR_EDGE_COLOUR),
        linewidth=BAR_EDGE_WIDTH
    )

    max_y_tick_value = rounder.floor_to_nearest(
        1.05 * numpy.max(example_frequency_by_bin), HISTOGRAM_Y_SPACING
    )
    num_y_ticks = 1 + int(numpy.round(
        max_y_tick_value / HISTOGRAM_Y_SPACING
    ))
    y_tick_values = numpy.linspace(0., max_y_tick_value, num=num_y_ticks)

    pyplot.xticks(HISTOGRAM_X_VALUES, axes=inset_axes_object)
    pyplot.yticks(y_tick_values, axes=inset_axes_object)
    inset_axes_object.set_xlim(0., 1.)
    inset_axes_object.set_ylim(0., 1.05 * numpy.max(example_frequency_by_bin))

    inset_axes_object.set_title('Forecast histogram', fontsize=20)


def plot_roc_curve(axes_object, pod_by_threshold, pofd_by_threshold,
                   line_colour=ROC_CURVE_COLOUR, plot_background=True):
    """Plots ROC (receiver operating characteristic) curve.

    T = number of binarization thresholds

    For the definition of a "binarization threshold" and the role they play in
    ROC curves, see `model_evaluation.get_points_in_roc_curve`.

    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param pod_by_threshold: length-T numpy array of POD (probability of
        detection) values.
    :param pofd_by_threshold: length-T numpy array of POFD (probability of false
        detection) values.
    :param line_colour: Line colour.
    :param plot_background: Boolean flag.  If True, will plot background
        (reference line and Peirce-score contours).
    :return: line_handle: Line handle for ROC curve.
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

    error_checking.assert_is_boolean(plot_background)

    if plot_background:
        pofd_matrix, pod_matrix = model_eval.get_pofd_pod_grid()
        peirce_score_matrix = pod_matrix - pofd_matrix

        this_colour_map_object, this_colour_norm_object = (
            _get_peirce_colour_scheme()
        )

        pyplot.contourf(
            pofd_matrix, pod_matrix, peirce_score_matrix, CSI_LEVELS,
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
            color=plotting_utils.colour_from_numpy_to_tuple(RANDOM_ROC_COLOUR),
            linestyle='dashed', linewidth=RANDOM_ROC_WIDTH
        )

    nan_flags = numpy.logical_or(
        numpy.isnan(pofd_by_threshold), numpy.isnan(pod_by_threshold)
    )

    if numpy.all(nan_flags):
        line_handle = None
    else:
        real_indices = numpy.where(numpy.invert(nan_flags))[0]

        line_handle = axes_object.plot(
            pofd_by_threshold[real_indices], pod_by_threshold[real_indices],
            color=plotting_utils.colour_from_numpy_to_tuple(line_colour),
            linestyle='solid', linewidth=ROC_CURVE_WIDTH
        )[0]

    axes_object.set_xlabel('POFD (probability of false detection)')
    axes_object.set_ylabel('POD (probability of detection)')
    axes_object.set_xlim(0., 1.)
    axes_object.set_ylim(0., 1.)

    return line_handle


def plot_bootstrapped_roc_curve(
        axes_object, ci_bottom_dict, ci_mean_dict, ci_top_dict,
        line_colour=ROC_CURVE_COLOUR, plot_background=True):
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
    :param line_colour: See doc for `plot_roc_curve`.
    :param plot_background: Same.
    :return: line_handle: Same.
    """

    line_handle = plot_roc_curve(
        axes_object=axes_object,
        pod_by_threshold=ci_mean_dict[model_eval.POD_BY_THRESHOLD_KEY],
        pofd_by_threshold=ci_mean_dict[model_eval.POFD_BY_THRESHOLD_KEY],
        line_colour=line_colour, plot_background=plot_background
    )

    polygon_object = _confidence_interval_to_polygon(
        x_coords_bottom=ci_bottom_dict[model_eval.POFD_BY_THRESHOLD_KEY],
        y_coords_bottom=ci_bottom_dict[model_eval.POD_BY_THRESHOLD_KEY],
        x_coords_top=ci_top_dict[model_eval.POFD_BY_THRESHOLD_KEY],
        y_coords_top=ci_top_dict[model_eval.POD_BY_THRESHOLD_KEY]
    )

    if polygon_object is None:
        return line_handle

    polygon_colour = matplotlib.colors.to_rgba(
        plotting_utils.colour_from_numpy_to_tuple(line_colour),
        POLYGON_OPACITY
    )

    polygon_patch = PolygonPatch(
        polygon_object, lw=0, ec=polygon_colour, fc=polygon_colour)

    axes_object.add_patch(polygon_patch)

    return line_handle


def plot_performance_diagram(
        axes_object, pod_by_threshold, success_ratio_by_threshold,
        line_colour=PERF_DIAGRAM_COLOUR, plot_background=True):
    """Plots performance diagram.

    T = number of binarization thresholds

    For the definition of a "binarization threshold" and the role they play in
    performance diagrams, see
    `model_evaluation.get_points_in_performance_diagram`.

    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param pod_by_threshold: length-T numpy array of POD (probability of
        detection) values.
    :param success_ratio_by_threshold: length-T numpy array of success ratios.
    :param line_colour: Line colour.
    :param plot_background: Boolean flag.  If True, will plot background
        (frequency-bias and CSI contours).
    :return: line_handle: Line handle for ROC curve.
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

    error_checking.assert_is_boolean(plot_background)

    if plot_background:
        success_ratio_matrix, pod_matrix = model_eval.get_sr_pod_grid()
        csi_matrix = model_eval.csi_from_sr_and_pod(
            success_ratio_array=success_ratio_matrix, pod_array=pod_matrix
        )
        frequency_bias_matrix = model_eval.frequency_bias_from_sr_and_pod(
            success_ratio_array=success_ratio_matrix, pod_array=pod_matrix
        )

        this_colour_map_object, this_colour_norm_object = (
            _get_csi_colour_scheme()
        )

        pyplot.contourf(
            success_ratio_matrix, pod_matrix, csi_matrix, CSI_LEVELS,
            cmap=this_colour_map_object, norm=this_colour_norm_object, vmin=0.,
            vmax=1., axes=axes_object)

        colour_bar_object = plotting_utils.plot_colour_bar(
            axes_object_or_matrix=axes_object, data_matrix=csi_matrix,
            colour_map_object=this_colour_map_object,
            colour_norm_object=this_colour_norm_object,
            orientation_string='vertical', extend_min=False, extend_max=False)

        colour_bar_object.set_label('CSI (critical success index)')

        bias_colour_tuple = plotting_utils.colour_from_numpy_to_tuple(
            FREQ_BIAS_COLOUR)

        bias_colours_2d_tuple = ()
        for _ in range(len(FREQ_BIAS_LEVELS)):
            bias_colours_2d_tuple += (bias_colour_tuple,)

        bias_contour_object = pyplot.contour(
            success_ratio_matrix, pod_matrix, frequency_bias_matrix,
            FREQ_BIAS_LEVELS, colors=bias_colours_2d_tuple,
            linewidths=FREQ_BIAS_WIDTH, linestyles='dashed', axes=axes_object)

        pyplot.clabel(
            bias_contour_object, inline=True, inline_spacing=FREQ_BIAS_PADDING,
            fmt=FREQ_BIAS_STRING_FORMAT, fontsize=FONT_SIZE)

    nan_flags = numpy.logical_or(
        numpy.isnan(success_ratio_by_threshold), numpy.isnan(pod_by_threshold)
    )

    if numpy.all(nan_flags):
        line_handle = None
    else:
        real_indices = numpy.where(numpy.invert(nan_flags))[0]

        line_handle = axes_object.plot(
            success_ratio_by_threshold[real_indices],
            pod_by_threshold[real_indices],
            color=plotting_utils.colour_from_numpy_to_tuple(line_colour),
            linestyle='solid', linewidth=PERF_DIAGRAM_WIDTH
        )[0]

    axes_object.set_xlabel('Success ratio (1 - FAR)')
    axes_object.set_ylabel('POD (probability of detection)')
    axes_object.set_xlim(0., 1.)
    axes_object.set_ylim(0., 1.)

    return line_handle


def plot_bootstrapped_performance_diagram(
        axes_object, ci_bottom_dict, ci_mean_dict, ci_top_dict,
        line_colour=PERF_DIAGRAM_COLOUR, plot_background=True):
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
    :param line_colour: See doc for `plot_performance_diagram`.
    :param plot_background: Same.
    :return: line_colour: Same.
    """

    line_handle = plot_performance_diagram(
        axes_object=axes_object,
        pod_by_threshold=ci_mean_dict[model_eval.POD_BY_THRESHOLD_KEY],
        success_ratio_by_threshold=ci_mean_dict[model_eval.SR_BY_THRESHOLD_KEY],
        line_colour=line_colour, plot_background=plot_background
    )

    polygon_object = _confidence_interval_to_polygon(
        x_coords_bottom=ci_bottom_dict[model_eval.SR_BY_THRESHOLD_KEY],
        y_coords_bottom=ci_bottom_dict[model_eval.POD_BY_THRESHOLD_KEY],
        x_coords_top=ci_top_dict[model_eval.SR_BY_THRESHOLD_KEY],
        y_coords_top=ci_top_dict[model_eval.POD_BY_THRESHOLD_KEY],
        for_performance_diagram=True)

    if polygon_object is None:
        return line_handle

    polygon_colour = matplotlib.colors.to_rgba(
        plotting_utils.colour_from_numpy_to_tuple(line_colour),
        POLYGON_OPACITY
    )

    polygon_patch = PolygonPatch(
        polygon_object, lw=0, ec=polygon_colour, fc=polygon_colour)

    axes_object.add_patch(polygon_patch)
    return line_handle


def plot_reliability_curve(
        axes_object, mean_forecast_by_bin, event_frequency_by_bin):
    """Plots reliability curve.

    B = number of bins (separated by forecast probability)

    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param mean_forecast_by_bin: length-B numpy array of mean forecast
        probabilities.
    :param event_frequency_by_bin: length-B numpy array of mean observed
        labels (conditional event frequencies).
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
        color=plotting_utils.colour_from_numpy_to_tuple(PERFECT_RELIA_COLOUR),
        linestyle='dashed', linewidth=PERFECT_RELIA_WIDTH
    )

    nan_flags = numpy.logical_or(
        numpy.isnan(mean_forecast_by_bin), numpy.isnan(event_frequency_by_bin)
    )

    if not numpy.all(nan_flags):
        real_indices = numpy.where(numpy.invert(nan_flags))[0]

        axes_object.plot(
            mean_forecast_by_bin[real_indices],
            event_frequency_by_bin[real_indices],
            color=plotting_utils.colour_from_numpy_to_tuple(RELIABILITY_COLOUR),
            linestyle='solid', linewidth=RELIABILITY_WIDTH
        )

    axes_object.set_xlabel('Forecast probability')
    axes_object.set_ylabel('Conditional event frequency')
    axes_object.set_xlim(0., 1.)
    axes_object.set_ylim(0., 1.)


def plot_bootstrapped_reliability_curve(
        axes_object, ci_bottom_dict, ci_mean_dict, ci_top_dict):
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
    """

    plot_reliability_curve(
        axes_object=axes_object,
        mean_forecast_by_bin=ci_mean_dict[model_eval.MEAN_FORECAST_BY_BIN_KEY],
        event_frequency_by_bin=ci_mean_dict[model_eval.EVENT_FREQ_BY_BIN_KEY]
    )

    polygon_object = _confidence_interval_to_polygon(
        x_coords_bottom=ci_bottom_dict[model_eval.MEAN_FORECAST_BY_BIN_KEY],
        y_coords_bottom=ci_bottom_dict[model_eval.EVENT_FREQ_BY_BIN_KEY],
        x_coords_top=ci_top_dict[model_eval.MEAN_FORECAST_BY_BIN_KEY],
        y_coords_top=ci_top_dict[model_eval.EVENT_FREQ_BY_BIN_KEY]
    )

    if polygon_object is None:
        return

    polygon_colour = matplotlib.colors.to_rgba(
        plotting_utils.colour_from_numpy_to_tuple(RELIABILITY_COLOUR),
        POLYGON_OPACITY
    )

    polygon_patch = PolygonPatch(
        polygon_object, lw=0, ec=polygon_colour, fc=polygon_colour)

    axes_object.add_patch(polygon_patch)


def plot_attributes_diagram(
        figure_object, axes_object, mean_forecast_by_bin,
        event_frequency_by_bin, num_examples_by_bin):
    """Plots attributes diagram (Hsu and Murphy 1986).

    :param figure_object: Instance of `matplotlib.figure.Figure`.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param mean_forecast_by_bin: See doc for `plot_reliability_curve`.
    :param event_frequency_by_bin: Same.
    :param num_examples_by_bin: See doc for
        `_plot_inset_histogram_for_attributes_diagram`.
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
        axes_object=axes_object, climatology=climatology)

    _plot_inset_histogram_for_attributes_diagram(
        figure_object=figure_object, num_examples_by_bin=num_examples_by_bin)

    plot_reliability_curve(
        axes_object=axes_object, mean_forecast_by_bin=mean_forecast_by_bin,
        event_frequency_by_bin=event_frequency_by_bin)


def plot_bootstrapped_attributes_diagram(
        figure_object, axes_object, ci_bottom_dict, ci_mean_dict, ci_top_dict,
        num_examples_by_bin):
    """Bootstrapped version of plot_attributes_diagram.

    :param figure_object: Instance of `matplotlib.figure.Figure`.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param ci_bottom_dict: See doc for `plot_bootstrapped_reliability_curve`.
    :param ci_mean_dict: Same.
    :param ci_top_dict: Same.
    :param num_examples_by_bin: See doc for `plot_attributes_diagram`.
    """

    plot_attributes_diagram(
        figure_object=figure_object, axes_object=axes_object,
        mean_forecast_by_bin=ci_mean_dict[model_eval.MEAN_FORECAST_BY_BIN_KEY],
        event_frequency_by_bin=ci_mean_dict[model_eval.EVENT_FREQ_BY_BIN_KEY],
        num_examples_by_bin=num_examples_by_bin
    )

    polygon_object = _confidence_interval_to_polygon(
        x_coords_bottom=ci_bottom_dict[model_eval.MEAN_FORECAST_BY_BIN_KEY],
        y_coords_bottom=ci_bottom_dict[model_eval.EVENT_FREQ_BY_BIN_KEY],
        x_coords_top=ci_top_dict[model_eval.MEAN_FORECAST_BY_BIN_KEY],
        y_coords_top=ci_top_dict[model_eval.EVENT_FREQ_BY_BIN_KEY]
    )

    if polygon_object is None:
        return

    polygon_colour = matplotlib.colors.to_rgba(
        plotting_utils.colour_from_numpy_to_tuple(RELIABILITY_COLOUR),
        POLYGON_OPACITY
    )

    polygon_patch = PolygonPatch(
        polygon_object, lw=0, ec=polygon_colour, fc=polygon_colour)

    axes_object.add_patch(polygon_patch)
