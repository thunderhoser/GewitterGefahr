"""High-level methods for model evaluation.

To be used by scripts (i.e., files in the "scripts" package).

WARNING: This file works for only binary classification (not regression or
multiclass classification).
"""

import numpy
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import model_evaluation as model_eval
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import model_eval_plotting

# TODO(thunderhoser): Generalize for multiclass classification.

FORECAST_PRECISION_FOR_THRESHOLDS = 1e-4

DOTS_PER_INCH = 300
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15


def _create_roc_curve(forecast_probabilities, observed_labels, output_dir_name):
    """Creates ROC (receiver operating characteristic) curve.

    N = number of forecast-observation pairs

    :param forecast_probabilities: See doc for `run_evaluation`.
    :param observed_labels: Same.
    :param output_dir_name: Same.
    :return: auc: Area under ROC curve, calculated by GewitterGefahr.
    :return: scikit_learn_auc: Area under ROC curve, calculated by scikit-learn.
    """

    pofd_by_threshold, pod_by_threshold = model_eval.get_points_in_roc_curve(
        forecast_probabilities=forecast_probabilities,
        observed_labels=observed_labels,
        threshold_arg=model_eval.THRESHOLD_ARG_FOR_UNIQUE_FORECASTS,
        unique_forecast_precision=FORECAST_PRECISION_FOR_THRESHOLDS)

    auc = model_eval.get_area_under_roc_curve(
        pofd_by_threshold=pofd_by_threshold,
        pod_by_threshold=pod_by_threshold)
    scikit_learn_auc = roc_auc_score(
        y_true=observed_labels, y_score=forecast_probabilities)

    title_string = 'AUC = {0:.4f} ... scikit-learn AUC = {1:.4f}'.format(
        auc, scikit_learn_auc)
    print title_string

    figure_file_name = '{0:s}/roc_curve.jpg'.format(output_dir_name)
    print 'Saving ROC curve to: "{0:s}"...\n'.format(figure_file_name)

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))
    model_eval_plotting.plot_roc_curve(
        axes_object=axes_object, pod_by_threshold=pod_by_threshold,
        pofd_by_threshold=pofd_by_threshold)

    pyplot.title(title_string)
    pyplot.savefig(figure_file_name, dpi=DOTS_PER_INCH)
    pyplot.close()

    return auc, scikit_learn_auc


def _create_performance_diagram(
        forecast_probabilities, observed_labels, output_dir_name):
    """Creates performance diagram.

    :param forecast_probabilities: See doc for `run_evaluation`.
    :param observed_labels: Same.
    :param output_dir_name: Same.
    :return: aupd: Area under performance diagram.
    """

    success_ratio_by_threshold, pod_by_threshold = (
        model_eval.get_points_in_performance_diagram(
            forecast_probabilities=forecast_probabilities,
            observed_labels=observed_labels,
            threshold_arg=model_eval.THRESHOLD_ARG_FOR_UNIQUE_FORECASTS,
            unique_forecast_precision=FORECAST_PRECISION_FOR_THRESHOLDS))

    aupd = model_eval.get_area_under_perf_diagram(
        success_ratio_by_threshold=success_ratio_by_threshold,
        pod_by_threshold=pod_by_threshold)

    title_string = 'AUPD = {0:.4f}'.format(aupd)
    print title_string

    figure_file_name = '{0:s}/performance_diagram.jpg'.format(output_dir_name)
    print 'Saving performance diagram to: "{0:s}"...\n'.format(figure_file_name)

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))
    model_eval_plotting.plot_performance_diagram(
        axes_object=axes_object, pod_by_threshold=pod_by_threshold,
        success_ratio_by_threshold=success_ratio_by_threshold)

    pyplot.title(title_string)
    pyplot.savefig(figure_file_name, dpi=DOTS_PER_INCH)
    pyplot.close()

    return aupd


def _create_attributes_diagram(
        forecast_probabilities, observed_labels, output_dir_name):
    """Creates attributes diagram.

    :param forecast_probabilities: See doc for `run_evaluation`.
    :param observed_labels: Same.
    :param output_dir_name: Same.
    :return: bss_dict: Dictionary created by
        `model_evaluation.get_brier_skill_score`.
    """

    mean_forecast_by_bin, class_frequency_by_bin, num_examples_by_bin = (
        model_eval.get_points_in_reliability_curve(
            forecast_probabilities=forecast_probabilities,
            observed_labels=observed_labels))

    climatology = numpy.mean(observed_labels)
    bss_dict = model_eval.get_brier_skill_score(
        mean_forecast_prob_by_bin=mean_forecast_by_bin,
        mean_observed_label_by_bin=class_frequency_by_bin,
        num_examples_by_bin=num_examples_by_bin, climatology=climatology)

    print (
        'Climatology = {0:.4f} ... reliability = {1:.4f} ... resolution = '
        '{2:.4f} ... BSS = {3:.4f}'
    ).format(climatology, bss_dict[model_eval.RELIABILITY_KEY],
             bss_dict[model_eval.RESOLUTION_KEY],
             bss_dict[model_eval.BRIER_SKILL_SCORE_KEY])

    figure_file_name = '{0:s}/reliability_curve.jpg'.format(output_dir_name)
    print 'Saving reliability curve to: "{0:s}"...\n'.format(figure_file_name)

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))
    model_eval_plotting.plot_reliability_curve(
        axes_object=axes_object, mean_forecast_prob_by_bin=mean_forecast_by_bin,
        mean_observed_label_by_bin=class_frequency_by_bin)

    title_string = 'REL = {0:.4f} ... RES = {1:.4f} ... BSS = {2:.4f}'.format(
        bss_dict[model_eval.RELIABILITY_KEY],
        bss_dict[model_eval.RESOLUTION_KEY],
        bss_dict[model_eval.BRIER_SKILL_SCORE_KEY])
    pyplot.title(title_string)
    pyplot.savefig(figure_file_name, dpi=DOTS_PER_INCH)
    pyplot.close()

    figure_file_name = '{0:s}/attributes_diagram.jpg'.format(output_dir_name)
    print 'Saving attributes diagram to: "{0:s}"...\n'.format(figure_file_name)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))
    model_eval_plotting.plot_attributes_diagram(
        figure_object=figure_object, axes_object=axes_object,
        mean_forecast_prob_by_bin=mean_forecast_by_bin,
        mean_observed_label_by_bin=class_frequency_by_bin,
        num_examples_by_bin=num_examples_by_bin)

    pyplot.savefig(figure_file_name, dpi=DOTS_PER_INCH)
    pyplot.close()

    return bss_dict


def run_evaluation(forecast_probabilities, observed_labels, output_dir_name):
    """Evaluates forecast-observation pairs from any forecasting method.

    Specifically, this method does the following:

    - creates ROC (receiver operating characteristic) curve
    - creates performance diagram
    - creates attributes diagram
    - saves each of the aforelisted figures to a .jpg file
    - computes many performance metrics and saves them to a Pickle file

    :param forecast_probabilities: length-N numpy array of forecast event
        probabilities.
    :param observed_labels: length-N numpy array of observed labels (1 for
        "yes", 0 for "no").
    :param output_dir_name: Name of output directory.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    # TODO(thunderhoser): Make binarization threshold an input argument to this
    # method.
    (binarization_threshold, best_csi
    ) = model_eval.find_best_binarization_threshold(
        forecast_probabilities=forecast_probabilities,
        observed_labels=observed_labels,
        threshold_arg=model_eval.THRESHOLD_ARG_FOR_UNIQUE_FORECASTS,
        criterion_function=model_eval.get_csi,
        optimization_direction=model_eval.MAX_OPTIMIZATION_DIRECTION,
        unique_forecast_precision=FORECAST_PRECISION_FOR_THRESHOLDS)

    print (
        'Best binarization threshold = {0:.4f} ... corresponding CSI = {1:.4f}'
    ).format(binarization_threshold, best_csi)

    print 'Binarizing forecast probabilities...'
    forecast_labels = model_eval.binarize_forecast_probs(
        forecast_probabilities=forecast_probabilities,
        binarization_threshold=binarization_threshold)

    print 'Creating contingency table...'
    contingency_table_as_dict = model_eval.get_contingency_table(
        forecast_labels=forecast_labels, observed_labels=observed_labels)
    print '{0:s}\n'.format(str(contingency_table_as_dict))

    print 'Computing performance metrics...'
    pod = model_eval.get_pod(contingency_table_as_dict)
    pofd = model_eval.get_pofd(contingency_table_as_dict)
    success_ratio = model_eval.get_success_ratio(contingency_table_as_dict)
    focn = model_eval.get_focn(contingency_table_as_dict)
    accuracy = model_eval.get_accuracy(contingency_table_as_dict)
    csi = model_eval.get_csi(contingency_table_as_dict)
    frequency_bias = model_eval.get_frequency_bias(contingency_table_as_dict)
    peirce_score = model_eval.get_peirce_score(contingency_table_as_dict)
    heidke_score = model_eval.get_heidke_score(contingency_table_as_dict)

    print (
        'POD = {0:.4f} ... POFD = {1:.4f} ... success ratio = {2:.4f} ... '
        'FOCN = {3:.4f} ... accuracy = {4:.4f} ... CSI = {5:.4f} ... frequency '
        'bias = {6:.4f} ... Peirce score = {7:.4f} ... Heidke score = {8:.4f}\n'
    ).format(pod, pofd, success_ratio, focn, accuracy, csi, frequency_bias,
             peirce_score, heidke_score)

    auc, scikit_learn_auc = _create_roc_curve(
        forecast_probabilities=forecast_probabilities,
        observed_labels=observed_labels, output_dir_name=output_dir_name)
    print '\n'

    bss_dict = _create_attributes_diagram(
        forecast_probabilities=forecast_probabilities,
        observed_labels=observed_labels, output_dir_name=output_dir_name)
    print '\n'

    aupd = _create_performance_diagram(
        forecast_probabilities=forecast_probabilities,
        observed_labels=observed_labels, output_dir_name=output_dir_name)
    print '\n'

    evaluation_file_name = '{0:s}/model_evaluation.p'.format(output_dir_name)
    print 'Writing results to: "{0:s}"...'.format(evaluation_file_name)
    model_eval.write_results(
        forecast_probabilities=forecast_probabilities,
        observed_labels=observed_labels,
        binarization_threshold=binarization_threshold, pod=pod, pofd=pofd,
        success_ratio=success_ratio, focn=focn, accuracy=accuracy, csi=csi,
        frequency_bias=frequency_bias, peirce_score=peirce_score,
        heidke_score=heidke_score, auc=auc, scikit_learn_auc=scikit_learn_auc,
        aupd=aupd, bss_dict=bss_dict, pickle_file_name=evaluation_file_name)
