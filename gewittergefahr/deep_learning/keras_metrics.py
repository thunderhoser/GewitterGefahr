"""Performance metrics used to monitor Keras model while training.

WARNING: these metrics have the following properties, which some users may find
undesirable.

[1] Used only for monitoring, not to serve as loss functions.
[2] Binary metrics treat the highest class as the positive class, all others as
    the negative class.  In other words, binary metrics are for "highest class
    vs. all".
[3] Metrics are usually based on a contingency table, which contains
    deterministic forecasts.  However, metrics in this module are based only on
    probabilistic forecasts (it would take too long to compute metrics at
    various probability thresholds during training).

--- NOTATION ---

Throughout this module, I will use the following letters to denote elements of
the contingency table (even though, as mentioned above, there are no actual
contingency tables).

a = number of true positives ("hits")
b = number of false positives ("false alarms")
c = number of false negatives ("misses")
d = number of true negatives ("correct nulls")

I will also use the following letters to denote matrix dimensions.

E = number of examples
K = number of classes (possible values of target variable)
"""

import keras.backend as K


def _get_num_true_positives(target_tensor, forecast_probability_tensor):
    """Returns number of true positives (a in docstring).

    :param target_tensor: E-by-K tensor of target values (observed classes).  If
        target_tensor[i, k] = 1, the [i]th example belongs to the [k]th class.
    :param forecast_probability_tensor: E-by-K tensor of forecast probabilities.
        forecast_probability_tensor[i, k] = forecast probability that the [i]th
        example belongs to the [k]th class.
    :return: num_true_positives: Number of true positives.
    """

    return K.sum(K.clip(
        target_tensor[..., -1] * forecast_probability_tensor[..., -1],
        0., 1.))


def _get_num_false_positives(target_tensor, forecast_probability_tensor):
    """Returns number of false positives (b in docstring).

    :param target_tensor: See documentation for `_get_num_true_positives`.
    :param forecast_probability_tensor: See documentation for
        `_get_num_true_positives`.
    :return: num_false_positives: Number of false positives.
    """

    return K.sum(K.clip(
        (1. - target_tensor[..., -1]) * forecast_probability_tensor[..., -1],
        0., 1.))


def _get_num_false_negatives(target_tensor, forecast_probability_tensor):
    """Returns number of false negatives (c in docstring).

    :param target_tensor: See documentation for `_get_num_true_positives`.
    :param forecast_probability_tensor: See documentation for
        `_get_num_true_positives`.
    :return: num_false_negatives: Number of false negatives.
    """

    return K.sum(K.clip(
        target_tensor[..., -1] * (1. - forecast_probability_tensor[..., -1]),
        0., 1.))


def _get_num_true_negatives(target_tensor, forecast_probability_tensor):
    """Returns number of false negatives (d in docstring).

    :param target_tensor: See documentation for `_get_num_true_positives`.
    :param forecast_probability_tensor: See documentation for
        `_get_num_true_positives`.
    :return: num_true_negatives: Number of true negatives.
    """

    return K.sum(K.clip(
        (1. - target_tensor[..., -1]) *
        (1. - forecast_probability_tensor[..., -1]),
        0., 1.))


def accuracy(target_tensor, forecast_probability_tensor):
    """Returns accuracy.

    :param target_tensor: See documentation for `_get_num_true_positives`.
    :param forecast_probability_tensor: See documentation for
        `_get_num_true_positives`.
    :return: accuracy: Accuracy.
    """

    return K.mean(K.clip(target_tensor * forecast_probability_tensor, 0., 1.))


def binary_accuracy(target_tensor, forecast_probability_tensor):
    """Returns binary accuracy ([a + d] / [a + b + c + d]).

    :param target_tensor: See documentation for `_get_num_true_positives`.
    :param forecast_probability_tensor: See documentation for
        `_get_num_true_positives`.
    :return: binary_accuracy: Binary accuracy.
    """

    a = _get_num_true_positives(target_tensor, forecast_probability_tensor)
    b = _get_num_false_positives(target_tensor, forecast_probability_tensor)
    c = _get_num_false_negatives(target_tensor, forecast_probability_tensor)
    d = _get_num_true_negatives(target_tensor, forecast_probability_tensor)

    return (a + d) / (a + b + c + d + K.epsilon())


def binary_csi(target_tensor, forecast_probability_tensor):
    """Returns binary critical success index (a / [a + b + c]).

    :param target_tensor: See documentation for `_get_num_true_positives`.
    :param forecast_probability_tensor: See documentation for
        `_get_num_true_positives`.
    :return: binary_csi: Binary CSI.
    """

    a = _get_num_true_positives(target_tensor, forecast_probability_tensor)
    b = _get_num_false_positives(target_tensor, forecast_probability_tensor)
    c = _get_num_false_negatives(target_tensor, forecast_probability_tensor)

    return a / (a + b + c + K.epsilon())


def binary_frequency_bias(target_tensor, forecast_probability_tensor):
    """Returns binary frequency bias ([a + b] / [a + c]).

    :param target_tensor: See documentation for `_get_num_true_positives`.
    :param forecast_probability_tensor: See documentation for
        `_get_num_true_positives`.
    :return: binary_frequency_bias: Binary frequency bias.
    """

    a = _get_num_true_positives(target_tensor, forecast_probability_tensor)
    b = _get_num_false_positives(target_tensor, forecast_probability_tensor)
    c = _get_num_false_negatives(target_tensor, forecast_probability_tensor)

    return (a + b) / (a + c + K.epsilon())


def binary_pod(target_tensor, forecast_probability_tensor):
    """Returns binary probability of detection (a / [a + c]).

    :param target_tensor: See documentation for `_get_num_true_positives`.
    :param forecast_probability_tensor: See documentation for
        `_get_num_true_positives`.
    :return: binary_pod: Binary POD.
    """

    a = _get_num_true_positives(target_tensor, forecast_probability_tensor)
    c = _get_num_false_negatives(target_tensor, forecast_probability_tensor)

    return a / (a + c + K.epsilon())


def binary_fom(target_tensor, forecast_probability_tensor):
    """Returns binary frequency of misses (c / [a + c]).

    :param target_tensor: See documentation for `_get_num_true_positives`.
    :param forecast_probability_tensor: See documentation for
        `_get_num_true_positives`.
    :return: binary_fom: Binary FOM.
    """

    return 1. - binary_pod(target_tensor, forecast_probability_tensor)


def binary_pofd(target_tensor, forecast_probability_tensor):
    """Returns binary probability of false detection (b / [b + d]).

    :param target_tensor: See documentation for `_get_num_true_positives`.
    :param forecast_probability_tensor: See documentation for
        `_get_num_true_positives`.
    :return: binary_pofd: Binary POFD.
    """

    b = _get_num_false_positives(target_tensor, forecast_probability_tensor)
    d = _get_num_true_negatives(target_tensor, forecast_probability_tensor)

    return b / (b + d + K.epsilon())


def binary_npv(target_tensor, forecast_probability_tensor):
    """Returns binary negative predictive value (d / [b + d]).

    :param target_tensor: See documentation for `_get_num_true_positives`.
    :param forecast_probability_tensor: See documentation for
        `_get_num_true_positives`.
    :return: binary_npv: Binary NPV.
    """

    return 1. - binary_pofd(target_tensor, forecast_probability_tensor)


def binary_success_ratio(target_tensor, forecast_probability_tensor):
    """Returns binary success ratio (a / [a + b]).

    :param target_tensor: See documentation for `_get_num_true_positives`.
    :param forecast_probability_tensor: See documentation for
        `_get_num_true_positives`.
    :return: binary_success_ratio: Binary success ratio.
    """

    a = _get_num_true_positives(target_tensor, forecast_probability_tensor)
    b = _get_num_false_positives(target_tensor, forecast_probability_tensor)

    return a / (a + b + K.epsilon())


def binary_far(target_tensor, forecast_probability_tensor):
    """Returns binary false-alarm rate (b / [a + b]).

    :param target_tensor: See documentation for `_get_num_true_positives`.
    :param forecast_probability_tensor: See documentation for
        `_get_num_true_positives`.
    :return: binary_far: Binary false-alarm rate.
    """

    return 1. - binary_success_ratio(target_tensor, forecast_probability_tensor)


def binary_dfr(target_tensor, forecast_probability_tensor):
    """Returns binary detection-failure ratio (c / [c + d]).

    :param target_tensor: See documentation for `_get_num_true_positives`.
    :param forecast_probability_tensor: See documentation for
        `_get_num_true_positives`.
    :return: binary_dfr: Binary DFR.
    """

    c = _get_num_false_negatives(target_tensor, forecast_probability_tensor)
    d = _get_num_true_negatives(target_tensor, forecast_probability_tensor)

    return c / (c + d + K.epsilon())


def binary_focn(target_tensor, forecast_probability_tensor):
    """Returns binary frequency of correct nulls (d / [c + d]).

    :param target_tensor: See documentation for `_get_num_true_positives`.
    :param forecast_probability_tensor: See documentation for
        `_get_num_true_positives`.
    :return: binary_focn: Binary FOCN.
    """

    return 1. - binary_dfr(target_tensor, forecast_probability_tensor)
