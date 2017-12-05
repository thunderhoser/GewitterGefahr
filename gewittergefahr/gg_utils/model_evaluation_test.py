"""Unit tests for model_evaluation.py."""

import copy
import unittest
import numpy
from gewittergefahr.gg_utils import model_evaluation as model_eval

TOLERANCE = 1e-6
FAKE_THRESHOLD_ARG = 'foo'

UNIQUE_FORECAST_PRECISION_FOR_THRESHOLDS = 0.01
FORECAST_PROBABILITIES = numpy.array(
    [0.0801, 0.0503, 0.1805, 0.111, 0.042, 0.803, 0.294, 0.273, 0.952, 0.951])
OBSERVED_LABELS = numpy.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=int)

# The following constants are used to test _get_binarization_thresholds.
THRESHOLDS_FOR_DIRECT_INPUT = numpy.array(
    [0., 0.001, 0.005, 0.01, 0.1, 0.2, 0.3, 0.5, 0.75, 0.99, 0.999, 1.])
THRESHOLDS_FROM_DIRECT_INPUT = numpy.array(
    [0., 0.001, 0.005, 0.01, 0.1, 0.2, 0.3, 0.5, 0.75, 0.99, 0.999, 1.,
     model_eval.MAX_BINARIZATION_THRESHOLD])

NUM_THRESHOLDS_FOR_INPUT = 11
THRESHOLDS_FROM_NUMBER = numpy.array(
    [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.,
     model_eval.MAX_BINARIZATION_THRESHOLD])

FORECAST_PROBS_FOR_THRESHOLDS = numpy.array(
    [0.22, 0.39, 0.86, 1., 0., 0.221, 0.10, 0.393, 0.02, 0.018])
THRESHOLDS_FROM_UNIQUE_FORECASTS = numpy.array(
    [0., 0.02, 0.10, 0.22, 0.39, 0.86, 1.,
     model_eval.MAX_BINARIZATION_THRESHOLD])

# The following constants are used to test _pad_binarization_thresholds.
THRESHOLDS_WITH_NO_PADDING = numpy.array(
    [0.1, 0.25, 0.3, 0.4, 0.55, 0.61, 0.777, 0.8, 0.9, 0.95, 0.99])
THRESHOLDS_WITH_MIN_PADDING = numpy.array(
    [model_eval.MIN_BINARIZATION_THRESHOLD,
     0.1, 0.25, 0.3, 0.4, 0.55, 0.61, 0.777, 0.8, 0.9, 0.95, 0.99])
THRESHOLDS_WITH_MAX_PADDING = numpy.array(
    [0.1, 0.25, 0.3, 0.4, 0.55, 0.61, 0.777, 0.8, 0.9, 0.95, 0.99,
     model_eval.MAX_BINARIZATION_THRESHOLD])
THRESHOLDS_WITH_MINMAX_PADDING = numpy.array(
    [model_eval.MIN_BINARIZATION_THRESHOLD,
     0.1, 0.25, 0.3, 0.4, 0.55, 0.61, 0.777, 0.8, 0.9, 0.95, 0.99,
     model_eval.MAX_BINARIZATION_THRESHOLD])

# The following constants are used to test _binarize_forecast_probs.
BINARIZATION_THRESHOLD_HALF = 0.5
FORECAST_LABELS_THRESHOLD_HALF = numpy.array(
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 1], dtype=int)

# The following constants are used to test get_contingency_table.
CONTINGENCY_TABLE_THRESHOLD_HALF = {model_eval.NUM_TRUE_POSITIVES_KEY: 3,
                                    model_eval.NUM_FALSE_POSITIVES_KEY: 0,
                                    model_eval.NUM_FALSE_NEGATIVES_KEY: 2,
                                    model_eval.NUM_TRUE_NEGATIVES_KEY: 5}

# The following constants are used to test get_pod, get_fom, get_pofd, get_npv,
# get_success_ratio, get_far, get_dfr, get_focn, get_accuracy, get_csi,
# get_frequency_bias, get_peirce_score, and get_heidke_score.
POD_THRESHOLD_HALF = 0.6
FOM_THRESHOLD_HALF = 0.4
POFD_THRESHOLD_HALF = 0.
NPV_THRESHOLD_HALF = 1.
SUCCESS_RATIO_THRESHOLD_HALF = 1.
FAR_THRESHOLD_HALF = 0.
DFR_THRESHOLD_HALF = 2. / 7
FOCN_THRESHOLD_HALF = 5. / 7
ACCURACY_THRESHOLD_HALF = 0.8
CSI_THRESHOLD_HALF = 0.6
FREQUENCY_BIAS_THRESHOLD_HALF = 0.6
PEIRCE_SCORE_THRESHOLD_HALF = 0.6
HEIDKE_SCORE_THRESHOLD_HALF = 0.6

CONTINGENCY_TABLE_ALL_ZEROS = {model_eval.NUM_TRUE_POSITIVES_KEY: 0,
                               model_eval.NUM_FALSE_POSITIVES_KEY: 0,
                               model_eval.NUM_FALSE_NEGATIVES_KEY: 0,
                               model_eval.NUM_TRUE_NEGATIVES_KEY: 0}

# The following constants are used to test get_brier_score.
FORECAST_PROBS_FOR_BS_AND_XENTROPY = numpy.array(
    [0.2, 0.8, 0.5, 0., 0.3, 0.7, 0.25, 1., 0.9, 0.8])
BRIER_SCORE = 0.17225

# The following constants are used to test get_cross_entropy.
MODIFIED_FORECAST_PROBS_FOR_XENTROPY = numpy.array(
    [0.2, 0.8, 0.5, model_eval.MIN_FORECAST_PROB_FOR_XENTROPY, 0.3, 0.7, 0.25,
     model_eval.MAX_FORECAST_PROB_FOR_XENTROPY, 0.9, 0.8])
CROSS_ENTROPY = -0.1 * numpy.sum(
    numpy.log2(MODIFIED_FORECAST_PROBS_FOR_XENTROPY[-5:]) +
    numpy.log2(1 - MODIFIED_FORECAST_PROBS_FOR_XENTROPY[:5]))

# The following constants are used to test get_points_in_roc_curve.
ROC_AND_PERFORMANCE_THRESHOLDS = numpy.array(
    [0., 0.04, 0.05, 0.08, 0.11, 0.18, 0.27, 0.29, 0.8, 0.95,
     model_eval.MAX_BINARIZATION_THRESHOLD])
POD_BY_THRESHOLD = numpy.array([1., 1., 1., 1., 1., 1., 1., 0.8, 0.6, 0.4, 0.])
POFD_BY_THRESHOLD = numpy.array(
    [1., 1., 0.8, 0.6, 0.4, 0.2, 0., 0., 0., 0., 0.])

# The following constants are used to test get_points_in_performance_diagram.
SUCCESS_RATIO_BY_THRESHOLD = numpy.array(
    [0.5, 0.5, 5. / 9, 0.625, 5. / 7, 0.833333, 1., 1., 1., 1., numpy.nan])

# The following constants are used to test get_sr_pod_grid.
SUCCESS_RATIO_SPACING_FOR_GRID = 0.5
POD_SPACING_FOR_GRID = 0.5
SUCCESS_RATIO_MATRIX = numpy.array([[0.25, 0.75], [0.25, 0.75]])
POD_MATRIX = numpy.array([[0.75, 0.75], [0.25, 0.25]])

# The following constants are used to test frequency_bias_from_sr_and_pod and
# csi_from_sr_and_pod.
FREQUENCY_BIAS_MATRIX = numpy.array([[3., 1.], [1., 0.333333]])
CSI_MATRIX = 3. / numpy.array([[13., 5.], [21., 13.]])

# The following constants are used to test _split_forecasts_into_bins and
# get_points_in_reliability_curve.
NUM_FORECAST_BINS = 10
BIN_INDEX_BY_FORECAST = numpy.array([0, 0, 1, 1, 0, 8, 2, 2, 9, 9], dtype=int)

MEAN_FORECAST_PROB_BIN0 = numpy.mean(
    numpy.array([0.0801, 0.0503, 0.042]))
MEAN_FORECAST_PROB_BIN1 = numpy.mean(numpy.array([0.1805, 0.111]))
MEAN_FORECAST_PROB_BIN2 = numpy.mean(numpy.array([0.294, 0.273]))
MEAN_FORECAST_PROB_BIN9 = numpy.mean(numpy.array([0.952, 0.951]))

MEAN_FORECAST_PROB_BY_BIN = numpy.array(
    [MEAN_FORECAST_PROB_BIN0, MEAN_FORECAST_PROB_BIN1, MEAN_FORECAST_PROB_BIN2,
     numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, 0.803,
     MEAN_FORECAST_PROB_BIN9])

MEAN_OBSERVED_LABEL_BY_BIN = numpy.array(
    [0., 0., 1., numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, 1., 1.])
NUM_EXAMPLES_BY_BIN = numpy.array([3, 2, 2, 0, 0, 0, 0, 0, 1, 2])

# The following constants are used to test get_brier_skill_score.
CLIMATOLOGY_FOR_BSS = 0.2
THIS_UNCERTAINTY = 0.16
THIS_RESOLUTION = 0.34

THESE_INDICES = numpy.array([0, 1], dtype=int)
THIS_RELIABILITY_LABEL0 = 0.1 * numpy.sum(
    NUM_EXAMPLES_BY_BIN[THESE_INDICES] *
    MEAN_FORECAST_PROB_BY_BIN[THESE_INDICES] ** 2)

THESE_INDICES = numpy.array([2, 8, 9], dtype=int)
THIS_RELIABILITY_LABEL1 = 0.1 * numpy.sum(
    NUM_EXAMPLES_BY_BIN[THESE_INDICES] *
    (1. - MEAN_FORECAST_PROB_BY_BIN[THESE_INDICES]) ** 2)

THIS_RELIABILITY = THIS_RELIABILITY_LABEL0 + THIS_RELIABILITY_LABEL1
THIS_BRIER_SCORE = THIS_UNCERTAINTY + THIS_RELIABILITY - THIS_RESOLUTION
THIS_BSS = (THIS_RESOLUTION - THIS_RELIABILITY) / THIS_UNCERTAINTY

BSS_DICTIONARY = {model_eval.BRIER_SKILL_SCORE_KEY: THIS_BSS,
                  model_eval.BRIER_SCORE_KEY: THIS_BRIER_SCORE,
                  model_eval.RELIABILITY_KEY: THIS_RELIABILITY,
                  model_eval.RESOLUTION_KEY: THIS_RESOLUTION,
                  model_eval.UNCERTAINTY_KEY: THIS_UNCERTAINTY}

# The following constants are used to test get_no_skill_reliability_curve,
# get_skill_areas_in_reliability_curve,
# get_climatology_line_for_reliability_curve, and
# get_no_resolution_line_for_reliability_curve.
MEAN_OBSERVED_LABEL = 0.2
X_VALUES_FOR_NO_SKILL_RELIABILITY = numpy.array([0., 1.])
Y_VALUES_FOR_NO_SKILL_RELIABILITY = numpy.array([0.1, 0.6])

X_VERTICES_FOR_LEFT_SKILL_AREA = numpy.array([0., 0.2, 0.2, 0., 0.])
Y_VERTICES_FOR_LEFT_SKILL_AREA = numpy.array([0., 0., 0.2, 0.1, 0.])
X_VERTICES_FOR_RIGHT_SKILL_AREA = numpy.array([0.2, 1., 1., 0.2, 0.2])
Y_VERTICES_FOR_RIGHT_SKILL_AREA = numpy.array([0.2, 0.6, 1., 1., 0.2])

X_VALUES_FOR_CLIMATOLOGY_LINE = numpy.array([0.2, 0.2])
Y_VALUES_FOR_CLIMATOLOGY_LINE = numpy.array([0., 1.])

X_VALUES_FOR_NO_RESOLUTION_LINE = numpy.array([0., 1.])
Y_VALUES_FOR_NO_RESOLUTION_LINE = numpy.array([0.2, 0.2])


class ModelEvaluationTests(unittest.TestCase):
    """Each method is a unit test for model_evaluation.py."""

    def test_get_binarization_thresholds_direct_input(self):
        """Ensures correct output from _get_binarization_thresholds.

        In this case, desired thresholds are input directly.
        """

        these_thresholds = model_eval._get_binarization_thresholds(
            threshold_arg=THRESHOLDS_FROM_DIRECT_INPUT)
        self.assertTrue(numpy.allclose(
            these_thresholds, THRESHOLDS_FROM_DIRECT_INPUT, atol=TOLERANCE))

    def test_get_binarization_thresholds_from_number(self):
        """Ensures correct output from _get_binarization_thresholds.

        In this case, only number of thresholds is input directly.
        """

        these_thresholds = model_eval._get_binarization_thresholds(
            threshold_arg=NUM_THRESHOLDS_FOR_INPUT)
        self.assertTrue(numpy.allclose(
            these_thresholds, THRESHOLDS_FROM_NUMBER, atol=TOLERANCE))

    def test_get_binarization_thresholds_from_unique_forecasts(self):
        """Ensures correct output from _get_binarization_thresholds.

        In this case, binarization thresholds are determined from unique
        forecasts.
        """

        these_thresholds = model_eval._get_binarization_thresholds(
            threshold_arg=model_eval.THRESHOLD_ARG_FOR_UNIQUE_FORECASTS,
            forecast_probabilities=FORECAST_PROBS_FOR_THRESHOLDS,
            unique_forecast_precision=UNIQUE_FORECAST_PRECISION_FOR_THRESHOLDS)

        self.assertTrue(numpy.allclose(
            these_thresholds, THRESHOLDS_FROM_UNIQUE_FORECASTS, atol=TOLERANCE))

    def test_get_binarization_thresholds_bad_input(self):
        """Ensures correct output from _get_binarization_thresholds.

        In this case, input `threshold_arg` is invalid.
        """

        with self.assertRaises(ValueError):
            these_thresholds = model_eval._get_binarization_thresholds(
                threshold_arg=FAKE_THRESHOLD_ARG)

    def test_pad_binarization_thresholds_input_no_padding(self):
        """Ensures correct output from _pad_binarization_thresholds.

        In this case, input array needs padding at both ends.
        """

        these_thresholds = model_eval._pad_binarization_thresholds(
            THRESHOLDS_WITH_NO_PADDING)
        self.assertTrue(numpy.allclose(
            these_thresholds, THRESHOLDS_WITH_MINMAX_PADDING, atol=TOLERANCE))

    def test_pad_binarization_thresholds_input_min_padding(self):
        """Ensures correct output from _pad_binarization_thresholds.

        In this case, input array needs padding at upper end only.
        """

        these_thresholds = model_eval._pad_binarization_thresholds(
            THRESHOLDS_WITH_MIN_PADDING)
        self.assertTrue(numpy.allclose(
            these_thresholds, THRESHOLDS_WITH_MINMAX_PADDING, atol=TOLERANCE))

    def test_pad_binarization_thresholds_input_max_padding(self):
        """Ensures correct output from _pad_binarization_thresholds.

        In this case, input array needs padding at lower end only.
        """

        these_thresholds = model_eval._pad_binarization_thresholds(
            THRESHOLDS_WITH_MAX_PADDING)
        self.assertTrue(numpy.allclose(
            these_thresholds, THRESHOLDS_WITH_MINMAX_PADDING, atol=TOLERANCE))

    def test_pad_binarization_thresholds_input_minmax_padding(self):
        """Ensures correct output from _pad_binarization_thresholds.

        In this case, input array does not need padding.
        """

        these_thresholds = model_eval._pad_binarization_thresholds(
            THRESHOLDS_WITH_MINMAX_PADDING)
        self.assertTrue(numpy.allclose(
            these_thresholds, THRESHOLDS_WITH_MINMAX_PADDING, atol=TOLERANCE))

    def test_binarize_forecast_probs(self):
        """Ensures correct output from _binarize_forecast_probs."""

        these_forecast_labels = model_eval._binarize_forecast_probs(
            FORECAST_PROBABILITIES, BINARIZATION_THRESHOLD_HALF)
        self.assertTrue(numpy.array_equal(
            these_forecast_labels, FORECAST_LABELS_THRESHOLD_HALF))

    def test_get_contingency_table(self):
        """Ensures correct output from get_contingency_table."""

        this_contingency_table = model_eval.get_contingency_table(
            FORECAST_LABELS_THRESHOLD_HALF, OBSERVED_LABELS)
        self.assertTrue(
            this_contingency_table == CONTINGENCY_TABLE_THRESHOLD_HALF)

    def test_get_pod(self):
        """Ensures correct output from get_pod; input values are non-zero."""

        this_probability_of_detection = model_eval.get_pod(
            CONTINGENCY_TABLE_THRESHOLD_HALF)
        self.assertTrue(numpy.isclose(
            this_probability_of_detection, POD_THRESHOLD_HALF, atol=TOLERANCE))

    def test_get_pod_all_zeros(self):
        """Ensures correct output from get_pod; input values are all zero."""

        this_probability_of_detection = model_eval.get_pod(
            CONTINGENCY_TABLE_ALL_ZEROS)
        self.assertTrue(numpy.isnan(this_probability_of_detection))

    def test_get_fom(self):
        """Ensures correct output from get_fom; input values are non-zero."""

        this_frequency_of_misses = model_eval.get_fom(
            CONTINGENCY_TABLE_THRESHOLD_HALF)
        self.assertTrue(numpy.isclose(
            this_frequency_of_misses, FOM_THRESHOLD_HALF, atol=TOLERANCE))

    def test_get_fom_all_zeros(self):
        """Ensures correct output from get_fom; input values are all zero."""

        this_frequency_of_misses = model_eval.get_fom(
            CONTINGENCY_TABLE_ALL_ZEROS)
        self.assertTrue(numpy.isnan(this_frequency_of_misses))

    def test_get_pofd(self):
        """Ensures correct output from get_pofd; input values are non-zero."""

        this_probability_of_false_detection = model_eval.get_pofd(
            CONTINGENCY_TABLE_THRESHOLD_HALF)
        self.assertTrue(numpy.isclose(
            this_probability_of_false_detection, POFD_THRESHOLD_HALF,
            atol=TOLERANCE))

    def test_get_pofd_all_zeros(self):
        """Ensures correct output from get_pofd; input values are all zero."""

        this_probability_of_false_detection = model_eval.get_pofd(
            CONTINGENCY_TABLE_ALL_ZEROS)
        self.assertTrue(numpy.isnan(this_probability_of_false_detection))

    def test_get_npv(self):
        """Ensures correct output from get_npv; input values are non-zero."""

        this_negative_predictive_value = model_eval.get_npv(
            CONTINGENCY_TABLE_THRESHOLD_HALF)
        self.assertTrue(numpy.isclose(
            this_negative_predictive_value, NPV_THRESHOLD_HALF, atol=TOLERANCE))

    def test_get_npv_all_zeros(self):
        """Ensures correct output from get_npv; input values are all zero."""

        this_negative_predictive_value = model_eval.get_npv(
            CONTINGENCY_TABLE_ALL_ZEROS)
        self.assertTrue(numpy.isnan(this_negative_predictive_value))

    def test_get_success_ratio(self):
        """Ensures correct output from get_success_ratio.

        In this case, input values are non-zero.
        """

        this_success_ratio = model_eval.get_success_ratio(
            CONTINGENCY_TABLE_THRESHOLD_HALF)
        self.assertTrue(numpy.isclose(
            this_success_ratio, SUCCESS_RATIO_THRESHOLD_HALF, atol=TOLERANCE))

    def test_get_success_ratio_all_zeros(self):
        """Ensures correct output from get_success_ratio.

        In this case, input values are all zero.
        """

        this_success_ratio = model_eval.get_success_ratio(
            CONTINGENCY_TABLE_ALL_ZEROS)
        self.assertTrue(numpy.isnan(this_success_ratio))

    def test_get_far(self):
        """Ensures correct output from get_far; input values are non-zero."""

        this_false_alarm_rate = model_eval.get_far(
            CONTINGENCY_TABLE_THRESHOLD_HALF)
        self.assertTrue(numpy.isclose(
            this_false_alarm_rate, FAR_THRESHOLD_HALF, atol=TOLERANCE))

    def test_get_far_all_zeros(self):
        """Ensures correct output from get_far; input values are all zero."""

        this_false_alarm_rate = model_eval.get_far(
            CONTINGENCY_TABLE_ALL_ZEROS)
        self.assertTrue(numpy.isnan(this_false_alarm_rate))

    def test_get_dfr(self):
        """Ensures correct output from get_dfr; input values are non-zero."""

        this_detection_failure_ratio = model_eval.get_dfr(
            CONTINGENCY_TABLE_THRESHOLD_HALF)
        self.assertTrue(numpy.isclose(
            this_detection_failure_ratio, DFR_THRESHOLD_HALF, atol=TOLERANCE))

    def test_get_dfr_all_zeros(self):
        """Ensures correct output from get_dfr; input values are all zero."""

        this_detection_failure_ratio = model_eval.get_dfr(
            CONTINGENCY_TABLE_ALL_ZEROS)
        self.assertTrue(numpy.isnan(this_detection_failure_ratio))

    def test_get_focn(self):
        """Ensures correct output from get_focn; input values are non-zero."""

        this_frequency_of_correct_nulls = model_eval.get_focn(
            CONTINGENCY_TABLE_THRESHOLD_HALF)
        self.assertTrue(numpy.isclose(
            this_frequency_of_correct_nulls, FOCN_THRESHOLD_HALF,
            atol=TOLERANCE))

    def test_get_focn_all_zeros(self):
        """Ensures correct output from get_focn; input values are all zero."""

        this_frequency_of_correct_nulls = model_eval.get_focn(
            CONTINGENCY_TABLE_ALL_ZEROS)
        self.assertTrue(numpy.isnan(this_frequency_of_correct_nulls))

    def test_get_accuracy(self):
        """Ensures correctness of get_accuracy; input values are non-zero."""

        this_accuracy = model_eval.get_accuracy(
            CONTINGENCY_TABLE_THRESHOLD_HALF)
        self.assertTrue(numpy.isclose(
            this_accuracy, ACCURACY_THRESHOLD_HALF, atol=TOLERANCE))

    def test_get_accuracy_all_zeros(self):
        """Ensures correctness of get_accuracy; input values are all zero."""

        this_accuracy = model_eval.get_accuracy(CONTINGENCY_TABLE_ALL_ZEROS)
        self.assertTrue(numpy.isnan(this_accuracy))

    def test_get_csi(self):
        """Ensures correct output from get_csi; input values are non-zero."""

        this_critical_success_index = model_eval.get_csi(
            CONTINGENCY_TABLE_THRESHOLD_HALF)
        self.assertTrue(numpy.isclose(
            this_critical_success_index, CSI_THRESHOLD_HALF, atol=TOLERANCE))

    def test_get_csi_all_zeros(self):
        """Ensures correct output from get_csi; input values are all zero."""

        this_critical_success_index = model_eval.get_csi(
            CONTINGENCY_TABLE_ALL_ZEROS)
        self.assertTrue(numpy.isnan(this_critical_success_index))

    def test_get_frequency_bias(self):
        """Ensures crctness of get_frequency_bias; input values are non-zero."""

        this_frequency_bias = model_eval.get_frequency_bias(
            CONTINGENCY_TABLE_THRESHOLD_HALF)
        self.assertTrue(numpy.isclose(
            this_frequency_bias, FREQUENCY_BIAS_THRESHOLD_HALF, atol=TOLERANCE))

    def test_get_frequency_bias_all_zeros(self):
        """Ensures crctness of get_frequency_bias; input values are all zero."""

        this_frequency_bias = model_eval.get_frequency_bias(
            CONTINGENCY_TABLE_ALL_ZEROS)
        self.assertTrue(numpy.isnan(this_frequency_bias))

    def test_get_peirce_score(self):
        """Ensures crctness of get_peirce_score; input values are non-zero."""

        this_peirce_score = model_eval.get_peirce_score(
            CONTINGENCY_TABLE_THRESHOLD_HALF)
        self.assertTrue(numpy.isclose(
            this_peirce_score, PEIRCE_SCORE_THRESHOLD_HALF, atol=TOLERANCE))

    def test_get_peirce_score_all_zeros(self):
        """Ensures crctness of get_peirce_score; input values are all zero."""

        this_peirce_score = model_eval.get_peirce_score(
            CONTINGENCY_TABLE_ALL_ZEROS)
        self.assertTrue(numpy.isnan(this_peirce_score))

    def test_get_heidke_score(self):
        """Ensures crctness of get_heidke_score; input values are non-zero."""

        this_heidke_score = model_eval.get_heidke_score(
            CONTINGENCY_TABLE_THRESHOLD_HALF)
        self.assertTrue(numpy.isclose(
            this_heidke_score, HEIDKE_SCORE_THRESHOLD_HALF, atol=TOLERANCE))

    def test_get_heidke_score_all_zeros(self):
        """Ensures crctness of get_heidke_score; input values are all zero."""

        this_heidke_score = model_eval.get_heidke_score(
            CONTINGENCY_TABLE_ALL_ZEROS)
        self.assertTrue(numpy.isnan(this_heidke_score))

    def test_get_brier_score(self):
        """Ensures correct output from get_brier_score."""

        this_brier_score = model_eval.get_brier_score(
            forecast_probabilities=FORECAST_PROBS_FOR_BS_AND_XENTROPY,
            observed_labels=OBSERVED_LABELS)
        self.assertTrue(numpy.isclose(
            this_brier_score, BRIER_SCORE, atol=TOLERANCE))

    def test_get_cross_entropy(self):
        """Ensures correct output from get_cross_entropy."""

        this_cross_entropy = model_eval.get_cross_entropy(
            forecast_probabilities=FORECAST_PROBS_FOR_BS_AND_XENTROPY,
            observed_labels=OBSERVED_LABELS)
        self.assertTrue(numpy.isclose(
            this_cross_entropy, CROSS_ENTROPY, atol=TOLERANCE))

    def test_get_area_under_roc_curve_no_nan(self):
        """Ensures correct output from get_area_under_roc_curve.

        In this case, input arrays do not contain NaN.
        """

        this_auc = model_eval.get_area_under_roc_curve(
            POFD_BY_THRESHOLD, POD_BY_THRESHOLD)
        self.assertFalse(numpy.isnan(this_auc))

    def test_get_area_under_roc_curve_some_nan(self):
        """Ensures correct output from get_area_under_roc_curve.

        In this case, input arrays contain some NaN's.
        """

        these_pofd_by_threshold = copy.deepcopy(POFD_BY_THRESHOLD)
        these_pod_by_threshold = copy.deepcopy(POD_BY_THRESHOLD)

        nan_indices = numpy.array(
            [0, len(these_pod_by_threshold) - 1], dtype=int)
        these_pofd_by_threshold[nan_indices] = numpy.nan
        these_pod_by_threshold[nan_indices] = numpy.nan

        this_auc = model_eval.get_area_under_roc_curve(
            these_pofd_by_threshold, these_pod_by_threshold)
        self.assertFalse(numpy.isnan(this_auc))

    def test_get_area_under_roc_curve_all_nan(self):
        """Ensures correct output from get_area_under_roc_curve.

        In this case, input arrays are all NaN's.
        """

        these_pofd_by_threshold = numpy.full(len(POFD_BY_THRESHOLD), numpy.nan)
        these_pod_by_threshold = copy.deepcopy(these_pofd_by_threshold)

        this_auc = model_eval.get_area_under_roc_curve(
            these_pofd_by_threshold, these_pod_by_threshold)
        self.assertTrue(numpy.isnan(this_auc))

    def test_get_points_in_roc_curve(self):
        """Ensures correct output from get_points_in_roc_curve."""

        these_pofd_by_threshold, these_pod_by_threshold = (
            model_eval.get_points_in_roc_curve(
                forecast_probabilities=FORECAST_PROBABILITIES,
                observed_labels=OBSERVED_LABELS,
                threshold_arg=model_eval.THRESHOLD_ARG_FOR_UNIQUE_FORECASTS,
                unique_forecast_precision=
                UNIQUE_FORECAST_PRECISION_FOR_THRESHOLDS))

        self.assertTrue(numpy.allclose(
            these_pofd_by_threshold, POFD_BY_THRESHOLD, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_pod_by_threshold, POD_BY_THRESHOLD, atol=TOLERANCE))

    def test_get_points_in_performance_diagram(self):
        """Ensures correct output from get_points_in_performance_diagram."""

        these_success_ratio_by_threshold, these_pod_by_threshold = (
            model_eval.get_points_in_performance_diagram(
                forecast_probabilities=FORECAST_PROBABILITIES,
                observed_labels=OBSERVED_LABELS,
                threshold_arg=model_eval.THRESHOLD_ARG_FOR_UNIQUE_FORECASTS,
                unique_forecast_precision=
                UNIQUE_FORECAST_PRECISION_FOR_THRESHOLDS))

        self.assertTrue(numpy.allclose(
            these_success_ratio_by_threshold, SUCCESS_RATIO_BY_THRESHOLD,
            atol=TOLERANCE, equal_nan=True))
        self.assertTrue(numpy.allclose(
            these_pod_by_threshold, POD_BY_THRESHOLD, atol=TOLERANCE))

    def test_get_sr_pod_grid(self):
        """Ensures correct output from get_sr_pod_grid."""

        this_success_ratio_matrix, this_pod_matrix = (
            model_eval.get_sr_pod_grid(
                SUCCESS_RATIO_SPACING_FOR_GRID, POD_SPACING_FOR_GRID))

        self.assertTrue(numpy.allclose(
            this_success_ratio_matrix, SUCCESS_RATIO_MATRIX, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            this_pod_matrix, POD_MATRIX, atol=TOLERANCE))

    def test_frequency_bias_from_sr_and_pod(self):
        """Ensures correct output from frequency_bias_from_sr_and_pod."""

        this_frequency_bias_matrix = model_eval.frequency_bias_from_sr_and_pod(
            SUCCESS_RATIO_MATRIX, POD_MATRIX)
        self.assertTrue(numpy.allclose(
            this_frequency_bias_matrix, FREQUENCY_BIAS_MATRIX, atol=TOLERANCE))

    def test_csi_from_sr_and_pod(self):
        """Ensures correct output from csi_from_sr_and_pod."""

        this_csi_matrix = model_eval.csi_from_sr_and_pod(
            SUCCESS_RATIO_MATRIX, POD_MATRIX)
        self.assertTrue(numpy.allclose(
            this_csi_matrix, CSI_MATRIX, atol=TOLERANCE))

    def test_split_forecast_probs_into_bins(self):
        """Ensures correct output from _split_forecast_probs_into_bins."""

        these_bin_indices = model_eval._split_forecast_probs_into_bins(
            FORECAST_PROBABILITIES, NUM_FORECAST_BINS)
        self.assertTrue(numpy.array_equal(
            these_bin_indices, BIN_INDEX_BY_FORECAST))

    def test_get_points_in_reliability_curve(self):
        """Ensures correct output from get_points_in_reliability_curve."""

        (these_mean_forecast_probs,
         these_mean_observed_labels,
         these_num_examples_by_bin) = (
             model_eval.get_points_in_reliability_curve(
                 FORECAST_PROBABILITIES, OBSERVED_LABELS,
                 num_forecast_bins=NUM_FORECAST_BINS))

        self.assertTrue(numpy.allclose(
            these_mean_forecast_probs, MEAN_FORECAST_PROB_BY_BIN,
            atol=TOLERANCE, equal_nan=True))
        self.assertTrue(numpy.allclose(
            these_mean_observed_labels, MEAN_OBSERVED_LABEL_BY_BIN,
            atol=TOLERANCE, equal_nan=True))
        self.assertTrue(numpy.array_equal(
            these_num_examples_by_bin, NUM_EXAMPLES_BY_BIN))

    def test_get_brier_skill_score(self):
        """Ensures correct output from get_brier_skill_score."""

        this_bss_dict = model_eval.get_brier_skill_score(
            mean_forecast_prob_by_bin=MEAN_FORECAST_PROB_BY_BIN,
            mean_observed_label_by_bin=MEAN_OBSERVED_LABEL_BY_BIN,
            num_examples_by_bin=NUM_EXAMPLES_BY_BIN,
            climatology=CLIMATOLOGY_FOR_BSS)

        self.assertTrue(set(this_bss_dict.keys()) == set(BSS_DICTIONARY.keys()))

        for this_key in BSS_DICTIONARY.keys():
            self.assertTrue(numpy.isclose(
                this_bss_dict[this_key], BSS_DICTIONARY[this_key],
                atol=TOLERANCE))

    def test_get_no_skill_reliability_curve(self):
        """Ensures correct output from get_no_skill_reliability_curve."""

        these_x_values, these_y_values = (
            model_eval.get_no_skill_reliability_curve(MEAN_OBSERVED_LABEL))
        self.assertTrue(numpy.allclose(
            these_x_values, X_VALUES_FOR_NO_SKILL_RELIABILITY, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_y_values, Y_VALUES_FOR_NO_SKILL_RELIABILITY, atol=TOLERANCE))

    def test_get_skill_areas_in_reliability_curve(self):
        """Ensures correct output from get_skill_areas_in_reliability_curve."""

        (these_x_vertices_left,
         these_y_vertices_left,
         these_x_vertices_right,
         these_y_vertices_right) = (
             model_eval.get_skill_areas_in_reliability_curve(
                 MEAN_OBSERVED_LABEL))

        self.assertTrue(numpy.allclose(
            these_x_vertices_left, X_VERTICES_FOR_LEFT_SKILL_AREA,
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_y_vertices_left, Y_VERTICES_FOR_LEFT_SKILL_AREA,
            atol=TOLERANCE))

        self.assertTrue(numpy.allclose(
            these_x_vertices_right, X_VERTICES_FOR_RIGHT_SKILL_AREA,
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_y_vertices_right, Y_VERTICES_FOR_RIGHT_SKILL_AREA,
            atol=TOLERANCE))

    def test_get_climatology_line_for_reliability_curve(self):
        """Ensures correctness of get_climatology_line_for_reliability_curve."""

        these_x_values, these_y_values = (
            model_eval.get_climatology_line_for_reliability_curve(
                MEAN_OBSERVED_LABEL))
        self.assertTrue(numpy.allclose(
            these_x_values, X_VALUES_FOR_CLIMATOLOGY_LINE, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_y_values, Y_VALUES_FOR_CLIMATOLOGY_LINE, atol=TOLERANCE))

    def test_get_no_resolution_line_for_reliability_curve(self):
        """Ensures crrctness of get_no_resolution_line_for_reliability_curve."""

        these_x_values, these_y_values = (
            model_eval.get_no_resolution_line_for_reliability_curve(
                MEAN_OBSERVED_LABEL))
        self.assertTrue(numpy.allclose(
            these_x_values, X_VALUES_FOR_NO_RESOLUTION_LINE, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_y_values, Y_VALUES_FOR_NO_RESOLUTION_LINE, atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
