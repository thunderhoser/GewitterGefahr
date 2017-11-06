"""Unit tests for classification_utils.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import classification_utils as classifn_utils

TOLERANCE = 1e-6

# The following constants are used to test classification_cutoffs_to_ranges.
CLASS_CUTOFFS = numpy.array([10., 20., 30., 40., 50.])
CLASS_MINIMA_NEGATIVE_ALLOWED = numpy.array(
    [-numpy.inf, 10., 20., 30., 40., 50.])
CLASS_MINIMA_NEGATIVE_FORBIDDEN = numpy.array([0., 10., 20., 30., 40., 50.])
CLASS_MAXIMA = numpy.array([10., 20., 30., 40., 50., numpy.inf])

# The following constants are used to test classify_values.
INPUT_VALUES_NEGATIVE_ALLOWED = numpy.array(
    [-20., 0., 5., 10., 15., 20., 25., 30., 35., 40., 45., 50., 60.])
INPUT_VALUES_NEGATIVE_FORBIDDEN = INPUT_VALUES_NEGATIVE_ALLOWED[1:]
CLASS_LABELS_NEGATIVE_ALLOWED = numpy.array(
    [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=int)
CLASS_LABELS_NEGATIVE_FORBIDDEN = CLASS_LABELS_NEGATIVE_ALLOWED[1:]


class ClassificationUtilsTests(unittest.TestCase):
    """Each method is a unit test for classification_utils.py."""

    def test_classification_cutoffs_to_ranges_neg_allowed(self):
        """Ensures correct output from classification_cutoffs_to_ranges.

        In this case, negative values are allowed.
        """

        these_class_minima, these_class_maxima = (
            classifn_utils.classification_cutoffs_to_ranges(
                CLASS_CUTOFFS, non_negative_only=False))
        self.assertTrue(numpy.allclose(
            these_class_minima, CLASS_MINIMA_NEGATIVE_ALLOWED, atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_class_maxima, CLASS_MAXIMA, atol=TOLERANCE))

    def test_classification_cutoffs_to_ranges_neg_forbidden(self):
        """Ensures correct output from classification_cutoffs_to_ranges.

        In this case, negative values are forbidden.
        """

        these_class_minima, these_class_maxima = (
            classifn_utils.classification_cutoffs_to_ranges(
                CLASS_CUTOFFS, non_negative_only=True))
        self.assertTrue(numpy.allclose(
            these_class_minima, CLASS_MINIMA_NEGATIVE_FORBIDDEN,
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            these_class_maxima, CLASS_MAXIMA, atol=TOLERANCE))

    def test_classify_values_neg_allowed(self):
        """Ensures correct output from classify_values.

        In this case, negative values are allowed.
        """

        these_class_labels = classifn_utils.classify_values(
            INPUT_VALUES_NEGATIVE_ALLOWED, CLASS_CUTOFFS,
            non_negative_only=False)
        self.assertTrue(numpy.array_equal(
            these_class_labels, CLASS_LABELS_NEGATIVE_ALLOWED))

    def test_classify_values_neg_forbidden(self):
        """Ensures correct output from classify_values.

        In this case, negative values are forbidden.
        """

        these_class_labels = classifn_utils.classify_values(
            INPUT_VALUES_NEGATIVE_FORBIDDEN, CLASS_CUTOFFS,
            non_negative_only=False)
        self.assertTrue(numpy.array_equal(
            these_class_labels, CLASS_LABELS_NEGATIVE_FORBIDDEN))


if __name__ == '__main__':
    unittest.main()
