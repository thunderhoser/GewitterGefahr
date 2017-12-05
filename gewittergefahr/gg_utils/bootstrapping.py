"""Generates samples and confidence intervals for bootstrapping.

For more on bootstrapping, see Efron (1979).

--- REFERENCES ---

Efron, B., 1979: Bootstrap methods: another look at the jackknife. Annals of
    Statistics, 7 (1), 1-26.
"""

import numpy
from gewittergefahr.gg_utils import error_checking

DEFAULT_CONFIDENCE_LEVEL = 0.95


def draw_sample(input_vector, num_examples_to_draw=None):
    """Draws sample for bootstrapping.

    N = number of original examples
    M = number of examples drawn

    This is done by resampling with replacement from input_vector.  The number
    of examples drawn may range from 1...N.

    :param input_vector: length-N numpy array.
    :param num_examples_to_draw: Number of examples to draw.  Default is N.
    :return: sample_vector: length-M numpy array, where each element is drawn
        from input_vector.
    :return: sample_indices: length-M numpy array of indices taken from
        input_vector.  If sample_indices[k] = j, this means that
        sample_vector[k] = input_vector[j].
    """

    error_checking.assert_is_numpy_array(input_vector, num_dimensions=1)
    num_examples = len(input_vector)

    if num_examples_to_draw is None:
        num_examples_to_draw = num_examples

    error_checking.assert_is_integer(num_examples_to_draw)
    error_checking.assert_is_greater(num_examples_to_draw, 0)
    error_checking.assert_is_leq(num_examples_to_draw, num_examples)

    input_indices = numpy.linspace(
        0, num_examples - 1, num=num_examples, dtype=int)
    sample_indices = numpy.random.choice(
        input_indices, size=num_examples_to_draw, replace=True)
    return input_vector[sample_indices], sample_indices


def get_confidence_interval(stat_values,
                            confidence_level=DEFAULT_CONFIDENCE_LEVEL):
    """Computes confidence interval for bootstrapped statistic.

    K = number of bootstrapping iterations (number of samples drawn)

    :param stat_values: length-K numpy array with values of bootstrapped
        statistic.  Each value comes from a different sample ("bootstrap
        replicate").
    :param confidence_level: Level for confidence interval (may range from
        0...1).  For example, if confidence_level = 0.95, this method will
        create a 95% confidence interval.
    :return: confidence_interval_min: Minimum value in confidence interval.
    :return: confidence_interval_max: Maximum value in confidence interval.
    """

    error_checking.assert_is_numpy_array(stat_values, num_dimensions=1)
    error_checking.assert_is_real_numpy_array(stat_values)
    error_checking.assert_is_geq(confidence_level, 0.)
    error_checking.assert_is_leq(confidence_level, 1.)

    min_percentile = 50. * (1. - confidence_level)
    max_percentile = 50. * (1. + confidence_level)

    confidence_interval_min = numpy.nanpercentile(
        stat_values, min_percentile, interpolation='linear')
    confidence_interval_max = numpy.nanpercentile(
        stat_values, max_percentile, interpolation='linear')
    return confidence_interval_min, confidence_interval_max
