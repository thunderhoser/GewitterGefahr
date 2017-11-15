"""Model evaluation.

This module can be used to evaluate any kind of weather model (machine learning,
NWP, heuristics, human forecasting, etc.).  This module is completely agnostic
of where the forecasts come from.

--- REFERENCES ---

Roebber, P., 2009: Visualizing multiple measures of forecast quality. Weather
    and Forecasting, 24 (2), 601-608.
"""

import copy
import numpy
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import error_checking

# TODO(thunderhoser): This will probably be split into different modules.  I'm
# thinking of putting all ROC-curve things in one module, all performance-
# diagram things in one, and all reliability-curve things in one.  May also
# create different modules for binary classification, multi-class
# classification, and regression.

# TODO(thunderhoser): When creating thresholds, need to ensure that ROC curve
# and performance diagram will have corner points.

# TODO(thunderhoser): Add plotting methods.  This will be done in the
# `gewittergefahr.plotting` package.

TOLERANCE = 1e-6

NUM_TRUE_POSITIVES_KEY = 'num_true_positives'
NUM_FALSE_POSITIVES_KEY = 'num_false_positives'
NUM_FALSE_NEGATIVES_KEY = 'num_false_negatives'
NUM_TRUE_NEGATIVES_KEY = 'num_true_negatives'

DEFAULT_NUM_BINS_FOR_RELIABILITY_CURVE = 10
DEFAULT_PRECISION_FOR_THRESHOLDS = 1e-4
THRESHOLD_ARG_FOR_UNIQUE_FORECASTS = 'unique_forecasts'

DEFAULT_SUCCESS_RATIO_SPACING = 0.01
DEFAULT_POD_SPACING = 0.01


def _check_forecast_probs_and_observed_labels(
        forecast_probabilities, observed_labels):
    """Error-checks forecast probabilities and observed labels.

    N = number of forecasts

    :param forecast_probabilities: length-N numpy array with forecast
        probabilities of some event (e.g., tornado).
    :param observed_labels: length-N integer numpy array of observed labels
        (1 for "yes", 0 for "no").
    """

    error_checking.assert_is_numpy_array(
        forecast_probabilities, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(forecast_probabilities, 0.)
    error_checking.assert_is_leq_numpy_array(forecast_probabilities, 1.)
    num_forecasts = len(forecast_probabilities)

    error_checking.assert_is_integer_numpy_array(observed_labels)
    error_checking.assert_is_numpy_array(
        observed_labels, exact_dimensions=numpy.array([num_forecasts]))
    error_checking.assert_is_geq_numpy_array(observed_labels, 0)
    error_checking.assert_is_leq_numpy_array(observed_labels, 1)


def _check_forecast_and_observed_labels(forecast_labels, observed_labels):
    """Error-checks forecast and observed labels.

    N = number of forecasts

    :param forecast_labels: length-N integer numpy array of forecast labels
        (1 for "yes", 0 for "no").
    :param observed_labels: Same but for observed labels.
    """

    error_checking.assert_is_integer_numpy_array(forecast_labels)
    error_checking.assert_is_numpy_array(forecast_labels, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(forecast_labels, 0)
    error_checking.assert_is_leq_numpy_array(forecast_labels, 1)
    num_forecasts = len(forecast_labels)

    error_checking.assert_is_integer_numpy_array(observed_labels)
    error_checking.assert_is_numpy_array(
        observed_labels, exact_dimensions=numpy.array([num_forecasts]))
    error_checking.assert_is_geq_numpy_array(observed_labels, 0)
    error_checking.assert_is_leq_numpy_array(observed_labels, 1)


def _get_binarization_thresholds(
        threshold_arg=None, forecast_probabilities=None,
        unique_forecast_precision=DEFAULT_PRECISION_FOR_THRESHOLDS):
    """Determines binarization thresholds.

    To understand the role of binarization thresholds, see
    _binarize_forecast_probs.

    :param threshold_arg: Main threshold argument.  May be in one of 3 formats.
    [1] threshold_arg = "unique_forecasts".  In this case unique forecast
        probabilities will be used as thresholds.
    [2] 1-D numpy array.  In this case threshold_arg will be interpreted as an
        array of exact binarization thresholds.
    [3] Positive integer.  In this case threshold_arg will be interpreted as the
        number of binarization thresholds, equally spaced from 0...1.

    :param forecast_probabilities: 1-D numpy array of forecast probabilities to
        binarize.  If threshold_arg != "unique_forecasts", this will not be used
        (so leave it as None).
    :param unique_forecast_precision: Before taking unique forecast probs, all
        probs will be rounded to the nearest `unique_forecast_precision`.  If
        threshold_arg != "unique_forecasts", this will not be used (so leave it
        as None).
    :return: binarization_thresholds: 1-D numpy array of binarization
        thresholds.
    :raises: ValueError: if threshold_arg cannot be interpreted.
    """

    if isinstance(threshold_arg, str):
        if threshold_arg != THRESHOLD_ARG_FOR_UNIQUE_FORECASTS:
            raise ValueError(
                'If string, threshold_arg must be "' +
                THRESHOLD_ARG_FOR_UNIQUE_FORECASTS + '".  Instead, got "' +
                threshold_arg + '".')

        error_checking.assert_is_geq(unique_forecast_precision, 0.)
        error_checking.assert_is_leq(unique_forecast_precision, 1.)

        return numpy.unique(rounder.round_to_nearest(
            copy.deepcopy(forecast_probabilities), unique_forecast_precision))

    if isinstance(threshold_arg, numpy.ndarray):
        binarization_thresholds = copy.deepcopy(threshold_arg)

        error_checking.assert_is_numpy_array(
            binarization_thresholds, num_dimensions=1)
        error_checking.assert_is_geq_numpy_array(binarization_thresholds, 0.)
        error_checking.assert_is_leq_numpy_array(binarization_thresholds, 1.)

        return binarization_thresholds

    num_thresholds = copy.deepcopy(threshold_arg)
    error_checking.assert_is_integer(num_thresholds)
    error_checking.assert_is_geq(num_thresholds, 2)

    return numpy.linspace(0., 1., num=num_thresholds)


def _binarize_forecast_probs(forecast_probabilities, binarization_threshold):
    """Binarizes probabilistic forecasts, turning them into deterministic ones.

    N = number of forecasts

    :param forecast_probabilities: length-N numpy array with forecast
        probabilities of some event (e.g., tornado).
    :param binarization_threshold: Binarization threshold (f*).  All forecasts
        >= f* will be turned into "yes" forecasts; all forecasts < f* will be
        turned into "no".
    :return: forecast_labels: length-N integer numpy array of deterministic
        forecasts (1 for "yes", 0 for "no").
    """

    error_checking.assert_is_numpy_array(
        forecast_probabilities, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(forecast_probabilities, 0.)
    error_checking.assert_is_leq_numpy_array(forecast_probabilities, 1.)

    error_checking.assert_is_geq(binarization_threshold, 0.)
    error_checking.assert_is_leq(binarization_threshold, 1.)

    forecast_labels = numpy.full(len(forecast_probabilities), False, dtype=bool)
    positive_label_indices = numpy.where(
        forecast_probabilities >= binarization_threshold)[0]
    forecast_labels[positive_label_indices] = True

    return forecast_labels.astype(int)


def _get_sr_pod_grid(
        success_ratio_spacing=DEFAULT_SUCCESS_RATIO_SPACING,
        pod_spacing=DEFAULT_POD_SPACING):
    """Creates grid in SR-POD space

    SR = success ratio
    POD = probability of detection
    SR-POD space is the same as performance-diagram.

    M = number of rows (unique POD values) in grid
    N = number of columns (unique success ratios) in grid

    :param success_ratio_spacing: Spacing between adjacent success ratios
        (x-values) in grid.
    :param pod_spacing: Spacing between adjacent POD values (y-values) in grid.
    :return: success_ratio_matrix: M-by-N numpy array of success ratios.
        Success ratio increases while traveling right along a row.
    :return: pod_matrix: M-by-N numpy array of POD values.  POD increases while
        traveling up a column.
    """

    error_checking.assert_is_greater(success_ratio_spacing, 0.)
    error_checking.assert_is_less_than(success_ratio_spacing, 1.)
    error_checking.assert_is_greater(pod_spacing, 0.)
    error_checking.assert_is_less_than(pod_spacing, 1.)

    num_success_ratios = int(numpy.ceil(1. / success_ratio_spacing))
    num_pod_values = int(numpy.ceil(1. / pod_spacing))
    success_ratio_spacing = 1. / num_success_ratios
    pod_spacing = 1. / num_pod_values

    unique_success_ratios, unique_pod_values = grids.get_xy_grid_points(
        x_min_metres=success_ratio_spacing / 2, y_min_metres=pod_spacing / 2,
        x_spacing_metres=success_ratio_spacing, y_spacing_metres=pod_spacing,
        num_rows=num_pod_values, num_columns=num_success_ratios)
    return grids.xy_vectors_to_matrices(
        unique_success_ratios, unique_pod_values[::-1])


def get_contingency_table(forecast_labels, observed_labels):
    """Computes contingency table.

    N = number of forecasts

    :param forecast_labels: See documentation for
        _check_forecast_and_observed_labels.
    :param observed_labels: See doc for _check_forecast_and_observed_labels.
    :return: contingency_table_as_dict: Dictionary with the following keys.
    contingency_table_as_dict['num_true_positives']: Number of true positives.
    contingency_table_as_dict['num_false_positives']: Number of false positives.
    contingency_table_as_dict['num_false_negatives']: Number of false negatives.
    contingency_table_as_dict['num_true_negatives']: Number of true negatives.
    """

    _check_forecast_and_observed_labels(forecast_labels, observed_labels)

    true_positive_indices = numpy.where(numpy.logical_and(
        forecast_labels == 1, observed_labels == 1))[0]
    false_positive_indices = numpy.where(numpy.logical_and(
        forecast_labels == 1, observed_labels == 0))[0]
    false_negative_indices = numpy.where(numpy.logical_and(
        forecast_labels == 0, observed_labels == 1))[0]
    true_negative_indices = numpy.where(numpy.logical_and(
        forecast_labels == 0, observed_labels == 0))[0]

    return {
        NUM_TRUE_POSITIVES_KEY: len(true_positive_indices),
        NUM_FALSE_POSITIVES_KEY: len(false_positive_indices),
        NUM_FALSE_NEGATIVES_KEY: len(false_negative_indices),
        NUM_TRUE_NEGATIVES_KEY: len(true_negative_indices)
    }


def get_pod(contingency_table_as_dict):
    """Computes POD (probability of detection).

    :param contingency_table_as_dict: Dictionary created by
        get_contingency_table.
    :return: probability_of_detection: POD.
    """

    return float(contingency_table_as_dict[NUM_TRUE_POSITIVES_KEY]) / (
        contingency_table_as_dict[NUM_TRUE_POSITIVES_KEY] +
        contingency_table_as_dict[NUM_FALSE_NEGATIVES_KEY])


def get_fom(contingency_table_as_dict):
    """Computes FOM (frequency of misses).

    :param contingency_table_as_dict: Dictionary created by
        get_contingency_table.
    :return: frequency_of_misses: FOM.
    """

    return 1. - get_pod(contingency_table_as_dict)


def get_pofd(contingency_table_as_dict):
    """Computes POFD (probability of false detection).

    :param contingency_table_as_dict: Dictionary created by
        get_contingency_table.
    :return: probability_of_false_detection: POFD.
    """

    return float(contingency_table_as_dict[NUM_FALSE_POSITIVES_KEY]) / (
        contingency_table_as_dict[NUM_FALSE_POSITIVES_KEY] +
        contingency_table_as_dict[NUM_TRUE_NEGATIVES_KEY])


def get_npv(contingency_table_as_dict):
    """Computes NPV (negative predictive value).

    :param contingency_table_as_dict: Dictionary created by
        get_contingency_table.
    :return: negative_predictive_value: NPV.
    """

    return 1. - get_pofd(contingency_table_as_dict)


def get_success_ratio(contingency_table_as_dict):
    """Computes success ratio.

    :param contingency_table_as_dict: Dictionary created by
        get_contingency_table.
    :return: success_ratio: Success ratio.
    """

    return float(contingency_table_as_dict[NUM_TRUE_POSITIVES_KEY]) / (
        contingency_table_as_dict[NUM_TRUE_POSITIVES_KEY] +
        contingency_table_as_dict[NUM_FALSE_POSITIVES_KEY])


def get_far(contingency_table_as_dict):
    """Computes FAR (false-alarm rate).

    :param contingency_table_as_dict: Dictionary created by
        get_contingency_table.
    :return: false_alarm_rate: FAR.
    """

    return 1. - get_success_ratio(contingency_table_as_dict)


def get_dfr(contingency_table_as_dict):
    """Computes DFR (detection-failure ratio).

    :param contingency_table_as_dict: Dictionary created by
        get_contingency_table.
    :return: detection_failure_ratio: DFR.
    """

    return float(contingency_table_as_dict[NUM_FALSE_NEGATIVES_KEY]) / (
        contingency_table_as_dict[NUM_FALSE_NEGATIVES_KEY] +
        contingency_table_as_dict[NUM_TRUE_NEGATIVES_KEY])


def get_focn(contingency_table_as_dict):
    """Computes FOCN (frequency of correct nulls).

    :param contingency_table_as_dict: Dictionary created by
        get_contingency_table.
    :return: frequency_of_correct_nulls: FOCN.
    """

    return 1. - get_dfr(contingency_table_as_dict)


def get_accuracy(contingency_table_as_dict):
    """Computes accuracy.

    :param contingency_table_as_dict: Dictionary created by
        get_contingency_table.
    :return: accuracy: Accuracy.
    """

    return float(contingency_table_as_dict[NUM_TRUE_POSITIVES_KEY] +
                 contingency_table_as_dict[NUM_TRUE_NEGATIVES_KEY]) / (
                     contingency_table_as_dict[NUM_TRUE_POSITIVES_KEY] +
                     contingency_table_as_dict[NUM_FALSE_POSITIVES_KEY] +
                     contingency_table_as_dict[NUM_FALSE_NEGATIVES_KEY] +
                     contingency_table_as_dict[NUM_TRUE_NEGATIVES_KEY])


def get_csi(contingency_table_as_dict):
    """Computes CSI (critical success index).

    :param contingency_table_as_dict: Dictionary created by
        get_contingency_table.
    :return: critical_success_index: CSI.
    """

    return float(contingency_table_as_dict[NUM_TRUE_POSITIVES_KEY]) / (
        contingency_table_as_dict[NUM_TRUE_POSITIVES_KEY] +
        contingency_table_as_dict[NUM_FALSE_POSITIVES_KEY] +
        contingency_table_as_dict[NUM_FALSE_NEGATIVES_KEY])


def get_frequency_bias(contingency_table_as_dict):
    """Computes frequency bias.

    :param contingency_table_as_dict: Dictionary created by
        get_contingency_table.
    :return: frequency_bias: Frequency bias.
    """

    return float(contingency_table_as_dict[NUM_TRUE_POSITIVES_KEY] +
                 contingency_table_as_dict[NUM_FALSE_POSITIVES_KEY]) / (
                     contingency_table_as_dict[NUM_TRUE_POSITIVES_KEY] +
                     contingency_table_as_dict[NUM_FALSE_NEGATIVES_KEY])


def get_points_in_roc_curve(
        forecast_probabilities=None, observed_labels=None, threshold_arg=None,
        unique_forecast_precision=DEFAULT_PRECISION_FOR_THRESHOLDS):
    """Determines points in ROC (receiver operating characteristic) curve.

    N = number of forecasts
    T = number of binarization thresholds

    :param forecast_probabilities: See documentation for
        _check_forecast_probs_and_observed_labels.
    :param observed_labels: See doc for
        _check_forecast_probs_and_observed_labels.
    :param threshold_arg: See documentation for _get_binarization_thresholds.
    :param unique_forecast_precision: See doc for _get_binarization_thresholds.
    :return: pofd_by_threshold: length-T numpy array of POFD values, to be
        plotted on the x-axis.
    :return: pod_by_threshold: length-T numpy array of POD values, to be plotted
        on the y-axis.
    """

    _check_forecast_probs_and_observed_labels(
        forecast_probabilities, observed_labels)

    binarization_thresholds = _get_binarization_thresholds(
        threshold_arg=threshold_arg,
        forecast_probabilities=forecast_probabilities,
        unique_forecast_precision=unique_forecast_precision)

    num_thresholds = len(binarization_thresholds)
    pofd_by_threshold = numpy.full(num_thresholds, numpy.nan)
    pod_by_threshold = numpy.full(num_thresholds, numpy.nan)

    for i in range(num_thresholds):
        these_forecast_labels = _binarize_forecast_probs(
            forecast_probabilities, binarization_thresholds[i])
        this_contingency_table_as_dict = get_contingency_table(
            these_forecast_labels, observed_labels)

        pofd_by_threshold[i] = get_pofd(this_contingency_table_as_dict)
        pod_by_threshold[i] = get_pod(this_contingency_table_as_dict)

    return pofd_by_threshold, pod_by_threshold


def get_random_roc_curve():
    """Returns points in random ROC (receiver operating characteristic) curve.

    :return: pofd_by_threshold: length-2 numpy array of POFD values.
    :return: pod_by_threshold: length-2 numpy array of POD values.
    """

    return numpy.array([0., 1.]), numpy.array([0., 1.])


def get_points_in_performance_diagram(
        forecast_probabilities=None, observed_labels=None, threshold_arg=None,
        unique_forecast_precision=DEFAULT_PRECISION_FOR_THRESHOLDS):
    """Determines points in performance diagram (Roebber 2009).

    T = number of binarization thresholds

    :param forecast_probabilities: See documentation for
        _check_forecast_probs_and_observed_labels.
    :param observed_labels: See doc for
        _check_forecast_probs_and_observed_labels.
    :param threshold_arg: See doc for _get_binarization_thresholds.
    :param unique_forecast_precision: See doc for _get_binarization_thresholds.
    :return: success_ratio_by_threshold: length-T numpy array of success ratios,
        to be plotted on the x-axis.
    :return: pod_by_threshold: length-T numpy array of POD values, to be plotted
        on the y-axis.
    """

    _check_forecast_probs_and_observed_labels(
        forecast_probabilities, observed_labels)

    binarization_thresholds = _get_binarization_thresholds(
        threshold_arg=threshold_arg,
        forecast_probabilities=forecast_probabilities,
        unique_forecast_precision=unique_forecast_precision)

    num_thresholds = len(binarization_thresholds)
    success_ratio_by_threshold = numpy.full(num_thresholds, numpy.nan)
    pod_by_threshold = numpy.full(num_thresholds, numpy.nan)

    for i in range(num_thresholds):
        these_forecast_labels = _binarize_forecast_probs(
            forecast_probabilities, binarization_thresholds[i])
        this_contingency_table_as_dict = get_contingency_table(
            these_forecast_labels, observed_labels)

        success_ratio_by_threshold[i] = get_success_ratio(
            this_contingency_table_as_dict)
        pod_by_threshold[i] = get_pod(this_contingency_table_as_dict)

    return success_ratio_by_threshold, pod_by_threshold


def frequency_bias_from_sr_and_pod(success_ratio_array, pod_array):
    """Computes frequency bias from success ratio and POD.

    POD = probability of detection

    :param success_ratio_array: numpy array (any shape) of success ratios.
    :param pod_array: numpy array (same shape) of POD values.
    :return: frequency_bias_array: numpy array (same shape) of frequency biases.
    """

    error_checking.assert_is_numpy_array(success_ratio_array)
    error_checking.assert_is_geq_numpy_array(success_ratio_array, 0.)
    error_checking.assert_is_leq_numpy_array(success_ratio_array, 1.)

    success_ratio_dimensions = numpy.asarray(success_ratio_array.shape)
    error_checking.assert_is_numpy_array(
        pod_array, exact_dimensions=success_ratio_dimensions)
    error_checking.assert_is_geq_numpy_array(pod_array, 0.)
    error_checking.assert_is_leq_numpy_array(pod_array, 1.)

    return pod_array / success_ratio_array


def csi_from_sr_and_pod(success_ratio_array, pod_array):
    """Computes CSI (critical success index) from success ratio and POD.

    POD = probability of detection

    :param success_ratio_array: numpy array (any shape) of success ratios.
    :param pod_array: numpy array (same shape) of POD values.
    :return: csi_array: numpy array (same shape) of CSI values.
    """

    error_checking.assert_is_numpy_array(success_ratio_array)
    error_checking.assert_is_geq_numpy_array(success_ratio_array, 0.)
    error_checking.assert_is_leq_numpy_array(success_ratio_array, 1.)

    success_ratio_dimensions = numpy.asarray(success_ratio_array.shape)
    error_checking.assert_is_numpy_array(
        pod_array, exact_dimensions=success_ratio_dimensions)
    error_checking.assert_is_geq_numpy_array(pod_array, 0.)
    error_checking.assert_is_leq_numpy_array(pod_array, 1.)

    return (success_ratio_array ** -1 + pod_array ** -1 - 1.) ** -1


def get_points_in_reliability_curve(
        forecast_probabilities=None, observed_labels=None,
        num_forecast_bins=DEFAULT_NUM_BINS_FOR_RELIABILITY_CURVE):
    """Determines points in reliability curve.

    B = number of forecast bins

    :param forecast_probabilities: See documentation for
        _check_forecast_probs_and_observed_labels.
    :param observed_labels: See doc for
        _check_forecast_probs_and_observed_labels.
    :param num_forecast_bins: Number of bins in which to discretize forecast
        probabilities.
    :return: mean_forecast_prob_by_bin: length-B numpy array of mean forecast
        probabilities.
    :return: mean_observed_label_by_bin: length-B numpy array of mean observed
        labels (conditional event frequencies).
    """

    # TODO(thunderhoser): split into 2 methods and add unit tests.

    _check_forecast_probs_and_observed_labels(
        forecast_probabilities, observed_labels)

    error_checking.assert_is_integer(num_forecast_bins)
    error_checking.assert_is_geq(num_forecast_bins, 2)

    forecast_bin_cutoffs = numpy.linspace(0., 1., num=num_forecast_bins + 1)
    forecast_bin_cutoffs[-1] = forecast_bin_cutoffs[-1] + TOLERANCE
    bin_index_by_example = numpy.digitize(
        forecast_probabilities, forecast_bin_cutoffs, right=False)

    mean_forecast_prob_by_bin = numpy.full(num_forecast_bins, numpy.nan)
    mean_observed_label_by_bin = numpy.full(num_forecast_bins, numpy.nan)

    for i in range(num_forecast_bins):
        these_example_indices = numpy.where(bin_index_by_example == i)[0]
        mean_forecast_prob_by_bin[i] = numpy.mean(
            forecast_probabilities[these_example_indices])
        mean_observed_label_by_bin = numpy.mean(
            observed_labels[these_example_indices].astype(float))

    return mean_forecast_prob_by_bin, mean_observed_label_by_bin


def get_perfect_reliability_curve():
    """Returns points in perfect reliability curve.

    :return: mean_forecast_prob_by_bin: length-2 numpy array of mean forecast
        probabilities.
    :return: mean_observed_label_by_bin: length-2 numpy array of mean observed
        labels (conditional event frequencies).
    """

    return numpy.array([0., 1.]), numpy.array([0., 1.])


def get_no_skill_reliability_curve(mean_observed_label):
    """Returns points in no-skill reliability curve.

    This is a reliability curve with Brier skill score (BSS) = 0.

    :param mean_observed_label: Mean observed label (event frequency) for the
        full dataset (not just for one forecast bin).
    :return: x_values: length-2 numpy array of x-values in no-skill line.
    :return: y_values: length-2 numpy array of y-values in no-skill line.
    """

    error_checking.assert_is_geq(mean_observed_label, 0.)
    error_checking.assert_is_leq(mean_observed_label, 1.)

    y_values = numpy.array(
        [mean_observed_label, 1 + mean_observed_label]) / 2
    return numpy.array([0., 1.]), y_values


def get_skill_areas_in_reliability_curve(mean_observed_label):
    """Returns positive-skill areas (polygons) for reliability curve.

    BSS (Brier skill score) > 0 inside these polygons.

    :param mean_observed_label: Mean observed label (event frequency) for the
        full dataset (not just for one forecast bin).
    :return: list_of_x_vertex_arrays: length-2 list, where each element is a
        length-5 numpy array with x-coordinates of polygon vertices.
    :return: list_of_y_vertex_arrays: Same but for y-coordinates.
    """

    list_of_x_vertex_arrays = [[]] * 2
    list_of_y_vertex_arrays = [[]] * 2

    list_of_x_vertex_arrays[0] = numpy.array(
        [0., mean_observed_label, mean_observed_label, 0., 0.])
    list_of_y_vertex_arrays[0] = numpy.array(
        [0., 0., mean_observed_label, mean_observed_label / 2, 0.])

    list_of_x_vertex_arrays[1] = numpy.array(
        [mean_observed_label, 1., 1., mean_observed_label, mean_observed_label])
    list_of_y_vertex_arrays[1] = numpy.array(
        [mean_observed_label, (1 + mean_observed_label) / 2,
         1., 1., mean_observed_label])

    return list_of_x_vertex_arrays, list_of_y_vertex_arrays


def get_climatology_line_for_reliability_curve(mean_observed_label):
    """Returns climatology line for reliability curve.

    "Climatology" is another term for mean observed label.  The "climatology
    line" is a vertical line at x = climatology.

    :param mean_observed_label: Mean observed label (event frequency) for the
        full dataset (not just for one forecast bin).
    :return: x_values: length-2 numpy array of x-values in climatology line.
    :return: y_values: length-2 numpy array of y-values in climatology line.
    """

    return numpy.full(2, mean_observed_label), numpy.array([0., 1.])


def get_no_resolution_line_for_reliability_curve(mean_observed_label):
    """Returns no-resolution line for reliability curve.

    This is a horizontal line at y = mean observed label.

    :param mean_observed_label: Mean observed label (event frequency) for the
        full dataset (not just for one forecast bin).
    :return: x_values: length-2 numpy array of x-values in no-resolution line.
    :return: y_values: length-2 numpy array of y-values in no-resolution line.
    """

    return numpy.array([0., 1.]), numpy.full(2, mean_observed_label)
