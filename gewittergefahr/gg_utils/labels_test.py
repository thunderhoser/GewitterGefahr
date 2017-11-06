"""Unit tests for labels.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import labels

TOLERANCE = 1e-6

# The following constants are used to test _check_regression_params.
MIN_LEAD_TIME_SEC = 0
MAX_LEAD_TIME_SEC = 900
MIN_DISTANCE_ORIG_METRES = 0.7
MAX_DISTANCE_ORIG_METRES = 4999.7
PERCENTILE_LEVEL_ORIG = 99.96

MIN_DISTANCE_METRES = 1.
MAX_DISTANCE_METRES = 5000.
PERCENTILE_LEVEL = 100.
REGRESSION_PARAM_DICT = {
    labels.MIN_LEAD_TIME_NAME: MIN_LEAD_TIME_SEC,
    labels.MAX_LEAD_TIME_NAME: MAX_LEAD_TIME_SEC,
    labels.MIN_DISTANCE_NAME: MIN_DISTANCE_METRES,
    labels.MAX_DISTANCE_NAME: MAX_DISTANCE_METRES,
    labels.PERCENTILE_LEVEL_NAME: PERCENTILE_LEVEL
}

# The following constants are used to test _check_class_cutoffs.
CLASS_CUTOFFS_ORIG_KT = numpy.array([10.1, 20.2, 30.3, 40.4, 50.5])
CLASS_CUTOFFS_KT = numpy.array([10., 20., 30., 40., 50.])

# The following constants are used to test get_column_name_for_regression_label
# and get_column_name_for_classification_label.
COLUMN_NAME_FOR_REGRESSION = (
    'wind_speed_m_s01_percentile=100.0_lead-time=0000-0900sec_' +
    'distance=00001-05000m')
COLUMN_NAME_FOR_CLASSIFICATION = (
    'wind_speed_percentile=100.0_lead-time=0000-0900sec_' +
    'distance=00001-05000m_cutoffs=10-20-30-40-50kt')
FAKE_LABEL_COLUMN_NAME = 'poop'


class LabelsTests(unittest.TestCase):
    """Each method is a unit test for labels.py."""

    def test_check_regression_params(self):
        """Ensures correct output from _check_regression_params."""

        this_parameter_dict = labels._check_regression_params(
            min_lead_time_sec=MIN_LEAD_TIME_SEC,
            max_lead_time_sec=MAX_LEAD_TIME_SEC,
            min_distance_metres=MIN_DISTANCE_ORIG_METRES,
            max_distance_metres=MAX_DISTANCE_ORIG_METRES,
            percentile_level=PERCENTILE_LEVEL_ORIG)

        self.assertTrue(this_parameter_dict == REGRESSION_PARAM_DICT)

    def test_check_class_cutoffs(self):
        """Ensures correct output from _check_class_cutoffs."""

        these_class_cutoffs_kt = labels._check_class_cutoffs(
            CLASS_CUTOFFS_ORIG_KT)
        self.assertTrue(numpy.allclose(
            these_class_cutoffs_kt, CLASS_CUTOFFS_KT, atol=TOLERANCE))

    def test_column_name_to_label_params_regression(self):
        """Ensures correct output from column_name_to_label_params.

        In this case, column is a regression label.
        """

        this_parameter_dict = labels.column_name_to_label_params(
            COLUMN_NAME_FOR_REGRESSION)

        self.assertTrue(
            this_parameter_dict[labels.MIN_LEAD_TIME_NAME] == MIN_LEAD_TIME_SEC)
        self.assertTrue(
            this_parameter_dict[labels.MAX_LEAD_TIME_NAME] == MAX_LEAD_TIME_SEC)
        self.assertTrue(this_parameter_dict[labels.MIN_DISTANCE_NAME] ==
                        MIN_DISTANCE_METRES)
        self.assertTrue(this_parameter_dict[labels.MAX_DISTANCE_NAME] ==
                        MAX_DISTANCE_METRES)
        self.assertTrue(this_parameter_dict[labels.PERCENTILE_LEVEL_NAME] ==
                        PERCENTILE_LEVEL)
        self.assertTrue(this_parameter_dict[labels.CLASS_CUTOFFS_NAME] is None)

    def test_column_name_to_label_params_classification(self):
        """Ensures correct output from column_name_to_label_params.

        In this case, column is a classification label.
        """

        this_parameter_dict = labels.column_name_to_label_params(
            COLUMN_NAME_FOR_CLASSIFICATION)

        self.assertTrue(
            this_parameter_dict[labels.MIN_LEAD_TIME_NAME] == MIN_LEAD_TIME_SEC)
        self.assertTrue(
            this_parameter_dict[labels.MAX_LEAD_TIME_NAME] == MAX_LEAD_TIME_SEC)
        self.assertTrue(this_parameter_dict[labels.MIN_DISTANCE_NAME] ==
                        MIN_DISTANCE_METRES)
        self.assertTrue(this_parameter_dict[labels.MAX_DISTANCE_NAME] ==
                        MAX_DISTANCE_METRES)
        self.assertTrue(this_parameter_dict[labels.PERCENTILE_LEVEL_NAME] ==
                        PERCENTILE_LEVEL)
        self.assertTrue(numpy.array_equal(
            this_parameter_dict[labels.CLASS_CUTOFFS_NAME], CLASS_CUTOFFS_KT))

    def test_column_name_to_label_params_fake(self):
        """Ensures correct output from column_name_to_label_params.

        In this case, column is a fake label.
        """

        this_parameter_dict = labels.column_name_to_label_params(
            FAKE_LABEL_COLUMN_NAME)
        self.assertTrue(this_parameter_dict is None)

    def test_get_column_name_for_regression_label(self):
        """Ensures correct output from get_column_name_for_regression_label."""

        this_column_name = labels.get_column_name_for_regression_label(
            min_lead_time_sec=MIN_LEAD_TIME_SEC,
            max_lead_time_sec=MAX_LEAD_TIME_SEC,
            min_distance_metres=MIN_DISTANCE_METRES,
            max_distance_metres=MAX_DISTANCE_METRES,
            percentile_level=PERCENTILE_LEVEL)

        self.assertTrue(this_column_name == COLUMN_NAME_FOR_REGRESSION)

    def test_get_column_name_for_classification_label(self):
        """Ensures correctness of get_column_name_for_classification_label."""

        this_column_name = labels.get_column_name_for_classification_label(
            min_lead_time_sec=MIN_LEAD_TIME_SEC,
            max_lead_time_sec=MAX_LEAD_TIME_SEC,
            min_distance_metres=MIN_DISTANCE_METRES,
            max_distance_metres=MAX_DISTANCE_METRES,
            percentile_level=PERCENTILE_LEVEL,
            class_cutoffs_kt=CLASS_CUTOFFS_KT)

        self.assertTrue(this_column_name == COLUMN_NAME_FOR_CLASSIFICATION)


if __name__ == '__main__':
    unittest.main()
