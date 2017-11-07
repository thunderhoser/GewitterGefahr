"""Splits data into TV&T (training, validation, and testing) sets."""

import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking

TIME_STRING_FORMAT = '%Y-%m-%d-%H%M%S'

DEFAULT_VALIDATION_FRACTION = 0.2
DEFAULT_TESTING_FRACTION = 0.2
DEFAULT_TIME_SEPARATION_SEC = 86400


def _apply_time_separation(unix_times_sec, early_indices=None,
                           late_indices=None,
                           time_separation_sec=DEFAULT_TIME_SEPARATION_SEC):
    """Applies separation (buffer) between two sets of times.

    :param unix_times_sec: 1-D numpy array of valid times.
    :param early_indices: 1-D numpy array with indices of times in the first
        (earlier) set.  These are indices into unix_times_sec.
    :param late_indices: Same as `early_indices`, but for the second (later)
        set.
    :param time_separation_sec: Time separation between sets.  No example in the
        first set may occur within `time_separation_sec` of an example in the
        second set, and vice-versa.
    :return: early_indices: Subset of the input.  Some indices may have been
        removed to ensure the temporal separation.
    :return: late_indices: See above.
    """

    last_early_time_unix_sec = numpy.max(unix_times_sec[early_indices])
    first_late_time_unix_sec = numpy.min(unix_times_sec[late_indices])
    min_diff_between_sets_sec = (
        first_late_time_unix_sec - last_early_time_unix_sec)
    if min_diff_between_sets_sec >= time_separation_sec:
        return early_indices, late_indices

    erosion_time_sec = int(numpy.ceil(
        float(time_separation_sec - min_diff_between_sets_sec) / 2))

    last_early_time_to_keep_unix_sec = (
        last_early_time_unix_sec - erosion_time_sec)
    keep_early_time_flags = (
        unix_times_sec[early_indices] <= last_early_time_to_keep_unix_sec)
    keep_early_time_indices = numpy.where(keep_early_time_flags)[0]
    early_indices = early_indices[keep_early_time_indices]

    first_late_time_to_keep_unix_sec = (
        first_late_time_unix_sec + erosion_time_sec)
    keep_late_time_flags = (
        unix_times_sec[late_indices] >= first_late_time_to_keep_unix_sec)
    keep_late_time_indices = numpy.where(keep_late_time_flags)[0]
    late_indices = late_indices[keep_late_time_indices]

    check_time_separation(
        unix_times_sec, early_indices=early_indices, late_indices=late_indices,
        time_separation_sec=time_separation_sec)

    return early_indices, late_indices


def check_time_separation(unix_times_sec, early_indices=None, late_indices=None,
                          time_separation_sec=DEFAULT_TIME_SEPARATION_SEC):
    """Ensures that there is a separation (buffer) between two sets of times.

    :param unix_times_sec: See documentation for _apply_time_separation.
    :param early_indices: See documentation for _apply_time_separation.
    :param late_indices: See documentation for _apply_time_separation.
    :param time_separation_sec: See documentation for _apply_time_separation.
    :raises: ValueError: if separation between sets is < `time_separation_sec`.
    """

    error_checking.assert_is_integer_numpy_array(unix_times_sec)
    error_checking.assert_is_numpy_array_without_nan(unix_times_sec)
    error_checking.assert_is_numpy_array(unix_times_sec, num_dimensions=1)

    num_times = len(unix_times_sec)

    error_checking.assert_is_integer_numpy_array(early_indices)
    error_checking.assert_is_numpy_array(early_indices, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(early_indices, 0)
    error_checking.assert_is_leq_numpy_array(early_indices, num_times - 1)

    error_checking.assert_is_integer_numpy_array(late_indices)
    error_checking.assert_is_numpy_array(late_indices, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(late_indices, 0)
    error_checking.assert_is_leq_numpy_array(late_indices, num_times - 1)
    error_checking.assert_is_greater_numpy_array(
        unix_times_sec[late_indices], numpy.max(unix_times_sec[early_indices]))

    error_checking.assert_is_integer(time_separation_sec)
    error_checking.assert_is_greater(time_separation_sec, 0)

    last_early_time_unix_sec = numpy.max(unix_times_sec[early_indices])
    first_late_time_unix_sec = numpy.min(unix_times_sec[late_indices])
    min_diff_between_sets_sec = (
        first_late_time_unix_sec - last_early_time_unix_sec)
    if min_diff_between_sets_sec < time_separation_sec:
        last_early_time_string = time_conversion.unix_sec_to_string(
            last_early_time_unix_sec, TIME_STRING_FORMAT)
        first_late_time_string = time_conversion.unix_sec_to_string(
            first_late_time_unix_sec, TIME_STRING_FORMAT)

        error_string = (
            'Last time in early set is ' + last_early_time_string +
            '.  First time in late set is ' + first_late_time_string +
            '.  This is a time separation of ' + str(min_diff_between_sets_sec)
            + ' seconds between sets.  Required separation is >= ' +
            str(time_separation_sec) + ' s.')
        raise ValueError(error_string)


def split_training_validation_testing(
        unix_times_sec, validation_fraction=DEFAULT_VALIDATION_FRACTION,
        testing_fraction=DEFAULT_TESTING_FRACTION,
        time_separation_sec=DEFAULT_TIME_SEPARATION_SEC):
    """Splits data into training, validation, and testing sets.

    N = number of examples

    :param unix_times_sec: length-N numpy array of valid times.
    :param validation_fraction: Fraction of examples to be used for validation.
    :param testing_fraction: Fraction of examples to be used for testing.
    :param time_separation_sec: Time separation between sets.  No example in one
        set may occur within `time_separation_sec` of any example in any other
        set.  In other words, no training example may occur within
        `time_separation_sec` of a validation or testing example, and no
        validation example may occur within `time_separation_sec` of a testing
        example.
    :return: training_indices: 1-D numpy array with indices of training
        examples.
    :return: validation_indices: 1-D numpy array with indices of validation
        examples.
    :return: testing_indices: 1-D numpy array with indices of testing examples.
    """

    error_checking.assert_is_integer_numpy_array(unix_times_sec)
    error_checking.assert_is_numpy_array_without_nan(unix_times_sec)
    error_checking.assert_is_numpy_array(unix_times_sec, num_dimensions=1)

    error_checking.assert_is_greater(validation_fraction, 0.)
    error_checking.assert_is_less_than(validation_fraction, 1.)
    error_checking.assert_is_greater(testing_fraction, 0.)
    error_checking.assert_is_less_than(testing_fraction, 1.)
    error_checking.assert_is_integer(time_separation_sec)
    error_checking.assert_is_greater(time_separation_sec, 0)

    training_fraction = 1. - validation_fraction - testing_fraction
    error_checking.assert_is_greater(training_fraction, 0.)
    error_checking.assert_is_less_than(training_fraction, 1.)

    num_examples = len(unix_times_sec)
    num_training_examples = int(numpy.round(training_fraction * num_examples))
    num_validation_examples = int(numpy.round(
        validation_fraction * num_examples))
    num_testing_examples = (
        num_examples - num_training_examples - num_validation_examples)

    training_indices = numpy.linspace(
        0, num_training_examples - 1, num=num_training_examples, dtype=int)
    validation_indices = numpy.linspace(
        num_training_examples,
        num_training_examples + num_validation_examples - 1,
        num=num_validation_examples, dtype=int)
    testing_indices = numpy.linspace(
        num_training_examples + num_validation_examples, num_examples - 1,
        num=num_testing_examples, dtype=int)

    sort_indices = numpy.argsort(unix_times_sec)
    training_indices = sort_indices[training_indices]
    validation_indices = sort_indices[validation_indices]
    testing_indices = sort_indices[testing_indices]

    training_indices, validation_indices = _apply_time_separation(
        unix_times_sec, early_indices=training_indices,
        late_indices=validation_indices,
        time_separation_sec=time_separation_sec)
    validation_indices, testing_indices = _apply_time_separation(
        unix_times_sec, early_indices=validation_indices,
        late_indices=testing_indices, time_separation_sec=time_separation_sec)

    return training_indices, validation_indices, testing_indices


def split_training_for_bias_correction(
        all_base_model_times_unix_sec=None,
        all_bias_correction_times_unix_sec=None, base_model_fraction=None,
        first_non_training_time_unix_sec=None,
        time_separation_sec=DEFAULT_TIME_SEPARATION_SEC):
    """Splits training examples into two sets.

    One set is for training the base model, and one is for training the bias-
    correction model (which alters forecasts from the base model).  In
    general, the datasets used for the base model and bias correction are
    completely different, which is why this method requires two sets of times.

    All training times will precede non-training (validation and testing) times.

    P = number of base-model examples
    Q = number of bias-correction examples

    :param all_base_model_times_unix_sec: length-P numpy array of valid times.
    :param all_bias_correction_times_unix_sec: length-Q numpy array of valid
        times.
    :param base_model_fraction: Fraction of training time period to use for
        base model.
    :param first_non_training_time_unix_sec: First time in validation or testing
        set.
    :param time_separation_sec: Time separation between sets.  No example in one
        set may occur within `time_separation_sec` of any example in any other
        set.
    :return: base_model_training_indices: 1-D numpy array with indices of base-
        model-training examples.  These are indices into
        all_base_model_times_unix_sec.
    :return: bias_correction_training_indices: 1-D numpy array with indices of
        bias-correction-training examples.  These are indices into
        all_bias_correction_times_unix_sec.
    """

    error_checking.assert_is_integer_numpy_array(all_base_model_times_unix_sec)
    error_checking.assert_is_numpy_array_without_nan(
        all_base_model_times_unix_sec)
    error_checking.assert_is_numpy_array(
        all_base_model_times_unix_sec, num_dimensions=1)

    error_checking.assert_is_integer_numpy_array(
        all_bias_correction_times_unix_sec)
    error_checking.assert_is_numpy_array_without_nan(
        all_bias_correction_times_unix_sec)
    error_checking.assert_is_numpy_array(
        all_bias_correction_times_unix_sec, num_dimensions=1)

    error_checking.assert_is_greater(base_model_fraction, 0.)
    error_checking.assert_is_less_than(base_model_fraction, 1.)
    error_checking.assert_is_integer(first_non_training_time_unix_sec)
    error_checking.assert_is_integer(time_separation_sec)
    error_checking.assert_is_greater(time_separation_sec, 0)

    last_bias_training_time_unix_sec = (
        first_non_training_time_unix_sec - time_separation_sec)
    first_base_training_time_unix_sec = numpy.min(numpy.concatenate((
        all_base_model_times_unix_sec, all_bias_correction_times_unix_sec)))

    training_period_length_sec = (
        last_bias_training_time_unix_sec - first_base_training_time_unix_sec)
    last_base_training_time_unix_sec = first_base_training_time_unix_sec + int(
        numpy.round(base_model_fraction * float(training_period_length_sec)))
    first_bias_training_time_unix_sec = (
        last_base_training_time_unix_sec + time_separation_sec)

    base_model_training_flags = numpy.logical_and(
        all_base_model_times_unix_sec >= first_base_training_time_unix_sec,
        all_base_model_times_unix_sec <= last_base_training_time_unix_sec)
    base_model_training_indices = numpy.where(base_model_training_flags)[0]

    bias_correction_training_flags = numpy.logical_and(
        all_bias_correction_times_unix_sec >= first_bias_training_time_unix_sec,
        all_bias_correction_times_unix_sec <= last_bias_training_time_unix_sec)
    bias_correction_training_indices = numpy.where(
        bias_correction_training_flags)[0]

    return base_model_training_indices, bias_correction_training_indices


def split_tvt_for_bias_correction(
        base_model_times_unix_sec=None, bias_correction_times_unix_sec=None,
        validation_fraction=DEFAULT_VALIDATION_FRACTION,
        testing_fraction=DEFAULT_TESTING_FRACTION,
        base_model_training_fraction=None,
        time_separation_sec=DEFAULT_TIME_SEPARATION_SEC):
    """Splits data into two training sets, one validation set, one testing set.

    One training set is for the base model, and one is for the bias-correction
    model (which alters forecasts generated by the base model).  In general, the
    datasets used for the base model and bias correction are completely
    different, which is why this method requires two sets of times.

    Validation and testing are done only with examples in the bias-correction
    set.  Thus, the base-model set is used only for training.

    P = number of base-model examples
    Q = number of bias-correction examples

    :param base_model_times_unix_sec: length-P numpy array of valid times.
    :param bias_correction_times_unix_sec: length-Q numpy array of valid times.
    :param validation_fraction: Fraction of bias-correction examples to be used
        for validation.
    :param testing_fraction: Fraction of bias-correction examples to be used for
        testing.
    :param base_model_training_fraction: Fraction of training period to be used
        for base model.
    :param time_separation_sec: Time separation between sets.  No example in one
        set may occur within `time_separation_sec` of any example in any other
        set.
    :return: base_model_training_indices: 1-D numpy array with indices of base-
        model-training examples.  These are indices into
        base_model_times_unix_sec.
    :return: bias_correction_training_indices: 1-D numpy array with indices of
        bias-correction-training examples.  These are indices into
        bias_correction_times_unix_sec.
    :return: validation_indices: 1-D numpy array with indices of validation
        examples.  These are indices into bias_correction_times_unix_sec.
    :return: testing_indices: 1-D numpy array with indices of testing examples.
        These are indices into bias_correction_times_unix_sec.
    """

    _, validation_indices, testing_indices = split_training_validation_testing(
        bias_correction_times_unix_sec, validation_fraction=validation_fraction,
        testing_fraction=testing_fraction,
        time_separation_sec=time_separation_sec)

    base_model_training_indices, bias_correction_training_indices = (
        split_training_for_bias_correction(
            all_base_model_times_unix_sec=base_model_times_unix_sec,
            all_bias_correction_times_unix_sec=bias_correction_times_unix_sec,
            base_model_fraction=base_model_training_fraction,
            first_non_training_time_unix_sec=
            numpy.min(bias_correction_times_unix_sec[validation_indices]),
            time_separation_sec=time_separation_sec))

    return (base_model_training_indices, bias_correction_training_indices,
            validation_indices, testing_indices)
