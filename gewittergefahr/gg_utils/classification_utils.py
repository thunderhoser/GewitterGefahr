"""Helper methods for classification."""

import numpy
from gewittergefahr.gg_utils import error_checking


def classification_cutoffs_to_ranges(class_cutoffs, non_negative_only=True):
    """Converts classification cutoffs to min/max for each class.

    C = number of classes
    c = C - 1 = number of cutoffs

    :param class_cutoffs: length-c numpy array of class cutoffs.
    :param non_negative_only: Boolean flag.  If True, class cutoffs/minima/
        maxima must be non-negative.
    :return: class_cutoffs: Same as input, but containing only unique values and
        sorted in ascending order.
    :return: class_minima: length-C numpy array of class minima, sorted in
        ascending order.
    :return: class_maxima: length-C numpy array of class maxima, sorted in
        ascending order.
    """

    error_checking.assert_is_boolean(non_negative_only)
    error_checking.assert_is_numpy_array(class_cutoffs, num_dimensions=1)
    if non_negative_only:
        error_checking.assert_is_greater_numpy_array(class_cutoffs, 0.)
    else:
        error_checking.assert_is_numpy_array_without_nan(class_cutoffs)

    class_cutoffs = numpy.sort(numpy.unique(class_cutoffs))
    num_classes = len(class_cutoffs) + 1
    class_minima = numpy.full(num_classes, numpy.nan)
    class_maxima = numpy.full(num_classes, numpy.nan)

    for k in range(num_classes):
        if k == 0:
            class_maxima[k] = class_cutoffs[k]
            if non_negative_only:
                class_minima[k] = 0.
            else:
                class_minima[k] = -numpy.inf

        elif k == num_classes - 1:
            class_minima[k] = class_cutoffs[k - 1]
            class_maxima[k] = numpy.inf
        else:
            class_minima[k] = class_cutoffs[k - 1]
            class_maxima[k] = class_cutoffs[k]

    return class_cutoffs, class_minima, class_maxima


def classify_values(input_values, class_cutoffs, non_negative_only=True):
    """Assigns each element of input array to one class.

    N = number of values to classify
    C = number of classes
    c = C - 1 = number of cutoffs

    :param input_values: length-N numpy array of values to classify.
    :param class_cutoffs: length-c numpy array of class cutoffs.
    :param non_negative_only: Boolean flag.  If True, all values (class ranges
        and values to classify) must be non-negative.
    :return: class_labels: length-N numpy array of integer class labels.
    """

    _, class_minima, class_maxima = classification_cutoffs_to_ranges(
        class_cutoffs, non_negative_only=non_negative_only)

    error_checking.assert_is_numpy_array_without_nan(input_values)
    error_checking.assert_is_numpy_array(input_values, num_dimensions=1)
    if non_negative_only:
        error_checking.assert_is_geq_numpy_array(input_values, 0.)

    num_inputs = len(input_values)
    class_labels = numpy.full(num_inputs, -1, dtype=int)
    num_classes = len(class_minima)

    for k in range(num_classes):
        these_flags = numpy.logical_and(
            input_values >= class_minima[k], input_values < class_maxima[k])
        these_indices = numpy.where(these_flags)[0]
        class_labels[these_indices] = k

    return class_labels
