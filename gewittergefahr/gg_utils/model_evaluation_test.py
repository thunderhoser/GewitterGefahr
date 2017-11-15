"""Unit tests for model_evaluation.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import model_evaluation as model_eval

TOLERANCE = 1e-6

UNIQUE_FORECAST_PRECISION_FOR_THRESHOLDS = 0.01
FORECAST_PROBABILITIES = numpy.array(
    [0.0801, 0.0503, 0.1805, 0.111, 0.042, 0.803, 0.294, 0.273, 0.952, 0.951])
OBSERVED_LABELS = numpy.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=int)

# The following constants are used to test _get_binarization_thresholds.
FAKE_THRESHOLD_ARG = 'foo'
THRESHOLDS_FROM_DIRECT_INPUT = numpy.array(
    [0., 0.001, 0.005, 0.01, 0.1, 0.2, 0.3, 0.5, 0.75, 0.99, 0.999, 1.])

NUM_THRESHOLDS_FOR_INPUT = 11
THRESHOLDS_FROM_NUMBER = numpy.array(
    [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])

FORECAST_PROBS_FOR_THRESHOLDS = numpy.array(
    [0.22, 0.39, 0.86, 1., 0., 0.221, 0.10, 0.393, 0.02, 0.018])
THRESHOLDS_FROM_UNIQUE_FORECASTS = numpy.array(
    [0., 0.02, 0.10, 0.22, 0.39, 0.86, 1.])

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
# get_success_ratio, get_far, get_dfr, get_focn, get_accuracy, get_csi, and
# get_frequency_bias.
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

# The following constants are used to test get_points_in_roc_curve.
UNIQUE_FORECAST_PROBS = numpy.array(
    [0.04, 0.05, 0.08, 0.11, 0.18, 0.27, 0.29, 0.8, 0.95])
POD_BY_UNIQUE_THRESHOLD = numpy.array([1., 1., 1., 1., 1., 1., 0.8, 0.6, 0.4])
POFD_BY_UNIQUE_THRESHOLD = numpy.array([1., 0.8, 0.6, 0.4, 0.2, 0., 0., 0., 0.])

# The following constants are used to test get_points_in_performance_diagram.
SUCCESS_RATIO_BY_UNIQUE_THRESHOLD = numpy.array(
    [0.5, 5. / 9, 0.625, 5. / 7, 0.833333, 1., 1., 1., 1.])

# The following constants are used to test _get_sr_pod_grid.
SUCCESS_RATIO_SPACING_FOR_GRID = 0.5
POD_SPACING_FOR_GRID = 0.5
SUCCESS_RATIO_MATRIX = numpy.array([[0.25, 0.75], [0.25, 0.75]])
POD_MATRIX = numpy.array([[0.75, 0.75], [0.25, 0.25]])

# The following constants are used to test frequency_bias_from_sr_and_pod and
# csi_from_sr_and_pod.
FREQUENCY_BIAS_MATRIX = numpy.array([[3., 1.], [1., 0.333333]])
CSI_MATRIX = 3. / numpy.array([[13., 5.], [21., 13.]])

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
        """Ensures correct output from get_pod."""

        this_probability_of_detection = model_eval.get_pod(
            CONTINGENCY_TABLE_THRESHOLD_HALF)
        self.assertTrue(numpy.isclose(
            this_probability_of_detection, POD_THRESHOLD_HALF, atol=TOLERANCE))

    def test_get_fom(self):
        """Ensures correct output from get_fom."""

        this_frequency_of_misses = model_eval.get_fom(
            CONTINGENCY_TABLE_THRESHOLD_HALF)
        self.assertTrue(numpy.isclose(
            this_frequency_of_misses, FOM_THRESHOLD_HALF, atol=TOLERANCE))

    def test_get_pofd(self):
        """Ensures correct output from get_pofd."""

        this_probability_of_false_detection = model_eval.get_pofd(
            CONTINGENCY_TABLE_THRESHOLD_HALF)
        self.assertTrue(numpy.isclose(
            this_probability_of_false_detection, POFD_THRESHOLD_HALF,
            atol=TOLERANCE))

    def test_get_npv(self):
        """Ensures correct output from get_npv."""

        this_negative_predictive_value = model_eval.get_npv(
            CONTINGENCY_TABLE_THRESHOLD_HALF)
        self.assertTrue(numpy.isclose(
            this_negative_predictive_value, NPV_THRESHOLD_HALF, atol=TOLERANCE))

    def test_get_success_ratio(self):
        """Ensures correct output from get_success_ratio."""

        this_success_ratio = model_eval.get_success_ratio(
            CONTINGENCY_TABLE_THRESHOLD_HALF)
        self.assertTrue(numpy.isclose(
            this_success_ratio, SUCCESS_RATIO_THRESHOLD_HALF, atol=TOLERANCE))

    def test_get_far(self):
        """Ensures correct output from get_far."""

        this_false_alarm_rate = model_eval.get_far(
            CONTINGENCY_TABLE_THRESHOLD_HALF)
        self.assertTrue(numpy.isclose(
            this_false_alarm_rate, FAR_THRESHOLD_HALF, atol=TOLERANCE))

    def test_get_dfr(self):
        """Ensures correct output from get_dfr."""

        this_detection_failure_ratio = model_eval.get_dfr(
            CONTINGENCY_TABLE_THRESHOLD_HALF)
        self.assertTrue(numpy.isclose(
            this_detection_failure_ratio, DFR_THRESHOLD_HALF, atol=TOLERANCE))

    def test_get_focn(self):
        """Ensures correct output from get_focn."""

        this_frequency_of_correct_nulls = model_eval.get_focn(
            CONTINGENCY_TABLE_THRESHOLD_HALF)
        self.assertTrue(numpy.isclose(
            this_frequency_of_correct_nulls, FOCN_THRESHOLD_HALF,
            atol=TOLERANCE))

    def test_get_accuracy(self):
        """Ensures correct output from get_accuracy."""

        this_accuracy = model_eval.get_accuracy(
            CONTINGENCY_TABLE_THRESHOLD_HALF)
        self.assertTrue(numpy.isclose(
            this_accuracy, ACCURACY_THRESHOLD_HALF, atol=TOLERANCE))

    def test_get_csi(self):
        """Ensures correct output from get_csi."""

        this_critical_success_index = model_eval.get_csi(
            CONTINGENCY_TABLE_THRESHOLD_HALF)
        self.assertTrue(numpy.isclose(
            this_critical_success_index, CSI_THRESHOLD_HALF, atol=TOLERANCE))

    def test_get_frequency_bias(self):
        """Ensures correct output from get_frequency_bias."""

        this_frequency_bias = model_eval.get_frequency_bias(
            CONTINGENCY_TABLE_THRESHOLD_HALF)
        self.assertTrue(numpy.isclose(
            this_frequency_bias, FREQUENCY_BIAS_THRESHOLD_HALF, atol=TOLERANCE))

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
            these_pofd_by_threshold, POFD_BY_UNIQUE_THRESHOLD, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_pod_by_threshold, POD_BY_UNIQUE_THRESHOLD, atol=TOLERANCE))

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
            these_success_ratio_by_threshold, SUCCESS_RATIO_BY_UNIQUE_THRESHOLD,
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_pod_by_threshold, POD_BY_UNIQUE_THRESHOLD, atol=TOLERANCE))

    def test_get_sr_pod_grid(self):
        """Ensures correct output from _get_sr_pod_grid."""

        this_success_ratio_matrix, this_pod_matrix = (
            model_eval._get_sr_pod_grid(
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

        list_of_x_vertex_arrays, list_of_y_vertex_arrays = (
            model_eval.get_skill_areas_in_reliability_curve(
                MEAN_OBSERVED_LABEL))

        self.assertTrue(numpy.allclose(
            list_of_x_vertex_arrays[0], X_VERTICES_FOR_LEFT_SKILL_AREA,
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            list_of_y_vertex_arrays[0], Y_VERTICES_FOR_LEFT_SKILL_AREA,
            atol=TOLERANCE))

        self.assertTrue(numpy.allclose(
            list_of_x_vertex_arrays[1], X_VERTICES_FOR_RIGHT_SKILL_AREA,
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            list_of_y_vertex_arrays[1], Y_VERTICES_FOR_RIGHT_SKILL_AREA,
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
