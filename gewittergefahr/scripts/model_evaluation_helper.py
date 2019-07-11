"""High-level methods for model evaluation.

To be used by scripts (i.e., files in the "scripts" package).

WARNING: This file works for only binary classification (not regression or
multiclass classification).
"""

import copy
import numpy
import pandas
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import bootstrapping
from gewittergefahr.gg_utils import model_evaluation as model_eval
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import model_eval_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

NUM_RELIABILITY_BINS = 20
FORECAST_PRECISION_FOR_THRESHOLDS = 1e-4

NUM_TRUE_POSITIVES_KEY = 'num_true_positives'
NUM_FALSE_POSITIVES_KEY = 'num_false_positives'
NUM_FALSE_NEGATIVES_KEY = 'num_false_negatives'
NUM_TRUE_NEGATIVES_KEY = 'num_true_negatives'
POD_KEY = 'probability_of_detection'
POFD_KEY = 'probability_of_false_detection'
SUCCESS_RATIO_KEY = 'success_ratio'
FOCN_KEY = 'frequency_of_correct_nulls'
ACCURACY_KEY = 'accuracy'
CSI_KEY = 'critical_success_index'
FREQUENCY_BIAS_KEY = 'frequency_bias'
PEIRCE_SCORE_KEY = 'peirce_score'
HEIDKE_SCORE_KEY = 'heidke_score'
POD_BY_THRESHOLD_KEY = 'pod_by_threshold'
POFD_BY_THRESHOLD_KEY = 'pofd_by_threshold'
SR_BY_THRESHOLD_KEY = 'success_ratio_by_threshold'
MEAN_FORECAST_BY_BIN_KEY = 'mean_forecast_by_bin'
EVENT_FREQ_BY_BIN_KEY = 'event_frequency_by_bin'
AUC_KEY = 'area_under_roc_curve'
AUPD_KEY = 'area_under_perf_diagram'
RELIABILITY_KEY = 'reliability'
RESOLUTION_KEY = 'resolution'
BSS_KEY = 'brier_skill_score'

FIGURE_RESOLUTION_DPI = 300
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15


def _plot_roc_curve(evaluation_table, output_file_name, confidence_level=None):
    """Plots ROC curve.

    :param evaluation_table: See documentation for `_compute_scores`.  The only
        difference is that this table may have multiple rows (one per bootstrap
        replicate).
    :param output_file_name: Path to output file (figure will be saved here).
    :param confidence_level: Confidence level for bootstrapping.
    """

    pod_matrix = numpy.vstack(tuple(
        evaluation_table[POD_BY_THRESHOLD_KEY].values.tolist()
    ))
    pofd_matrix = numpy.vstack(tuple(
        evaluation_table[POD_BY_THRESHOLD_KEY].values.tolist()
    ))

    mean_auc = numpy.mean(evaluation_table[AUC_KEY].values)
    title_string = 'AUC = {0:.3f}'.format(mean_auc)

    num_bootstrap_reps = pod_matrix.shape[0]
    num_prob_thresholds = pod_matrix.shape[1]

    if num_bootstrap_reps > 1:
        min_auc, max_auc = bootstrapping.get_confidence_interval(
            stat_values=evaluation_table[AUC_KEY].values,
            confidence_level=confidence_level)

        title_string += ' [{0:.3f}, {1:.3f}]'.format(min_auc, max_auc)

    print(title_string)

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    if num_bootstrap_reps > 1:
        bottom_dict = {
            model_eval.POD_BY_THRESHOLD_KEY:
                numpy.full(num_prob_thresholds, numpy.nan),
            model_eval.POFD_BY_THRESHOLD_KEY:
                numpy.full(num_prob_thresholds, numpy.nan)
        }

        top_dict = copy.deepcopy(bottom_dict)
        mean_dict = copy.deepcopy(bottom_dict)

        for j in range(num_prob_thresholds):
            (bottom_dict[model_eval.POD_BY_THRESHOLD_KEY][j],
             top_dict[model_eval.POD_BY_THRESHOLD_KEY][j]
            ) = bootstrapping.get_confidence_interval(
                stat_values=pod_matrix[:, j], confidence_level=confidence_level
            )

            (top_dict[model_eval.POFD_BY_THRESHOLD_KEY][j],
             bottom_dict[model_eval.POFD_BY_THRESHOLD_KEY][j]
            ) = bootstrapping.get_confidence_interval(
                stat_values=pofd_matrix[:, j], confidence_level=confidence_level
            )

            mean_dict[model_eval.POD_BY_THRESHOLD_KEY][j] = (
                numpy.mean(pod_matrix[:, j])
            )

            mean_dict[model_eval.POFD_BY_THRESHOLD_KEY][j] = (
                numpy.mean(pofd_matrix[:, j])
            )

        model_eval_plotting.plot_bootstrapped_roc_curve(
            axes_object=axes_object, roc_dictionary_bottom=bottom_dict,
            roc_dictionary_mean=mean_dict, roc_dictionary_top=top_dict)
    else:
        model_eval_plotting.plot_roc_curve(
            axes_object=axes_object, pod_by_threshold=pod_matrix[0, :],
            pofd_by_threshold=pofd_matrix[0, :]
        )

    pyplot.title(title_string)

    print('Saving ROC curve to: "{0:s}"...'.format(output_file_name))
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()


def _plot_performance_diagram(evaluation_table, output_file_name,
                              confidence_level=None):
    """Plots performance diagram.

    :param evaluation_table: See documentation for `_compute_scores`.  The only
        difference is that this table may have multiple rows (one per bootstrap
        replicate).
    :param output_file_name: Path to output file (figure will be saved here).
    :param confidence_level: Confidence level for bootstrapping.
    """

    pod_matrix = numpy.vstack(tuple(
        evaluation_table[POD_BY_THRESHOLD_KEY].values.tolist()
    ))
    success_ratio_matrix = numpy.vstack(tuple(
        evaluation_table[SR_BY_THRESHOLD_KEY].values.tolist()
    ))

    mean_aupd = numpy.mean(evaluation_table[AUPD_KEY].values)
    title_string = 'AUPD = {0:.3f}'.format(mean_aupd)

    num_bootstrap_reps = pod_matrix.shape[0]
    num_prob_thresholds = pod_matrix.shape[1]

    if num_bootstrap_reps > 1:
        min_aupd, max_aupd = bootstrapping.get_confidence_interval(
            stat_values=evaluation_table[AUPD_KEY].values,
            confidence_level=confidence_level)

        title_string += ' [{0:.3f}, {1:.3f}]'.format(min_aupd, max_aupd)

    print(title_string)

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    if num_bootstrap_reps > 1:
        bottom_dict = {
            model_eval.POD_BY_THRESHOLD_KEY:
                numpy.full(num_prob_thresholds, numpy.nan),
            model_eval.SUCCESS_RATIO_BY_THRESHOLD_KEY:
                numpy.full(num_prob_thresholds, numpy.nan)
        }

        top_dict = copy.deepcopy(bottom_dict)
        mean_dict = copy.deepcopy(bottom_dict)

        for j in range(num_prob_thresholds):
            (bottom_dict[model_eval.POD_BY_THRESHOLD_KEY][j],
             top_dict[model_eval.POD_BY_THRESHOLD_KEY][j]
            ) = bootstrapping.get_confidence_interval(
                stat_values=pod_matrix[:, j], confidence_level=confidence_level
            )

            (bottom_dict[model_eval.SUCCESS_RATIO_BY_THRESHOLD_KEY][j],
             top_dict[model_eval.SUCCESS_RATIO_BY_THRESHOLD_KEY][j]
            ) = bootstrapping.get_confidence_interval(
                stat_values=success_ratio_matrix[:, j],
                confidence_level=confidence_level
            )

            mean_dict[model_eval.POD_BY_THRESHOLD_KEY][j] = (
                numpy.mean(pod_matrix[:, j])
            )

            mean_dict[model_eval.SUCCESS_RATIO_BY_THRESHOLD_KEY][j] = (
                numpy.mean(success_ratio_matrix[:, j])
            )

        model_eval_plotting.plot_bootstrapped_performance_diagram(
            axes_object=axes_object,
            performance_diagram_dict_bottom=bottom_dict,
            performance_diagram_dict_mean=mean_dict,
            performance_diagram_dict_top=top_dict)
    else:
        model_eval_plotting.plot_performance_diagram(
            axes_object=axes_object, pod_by_threshold=pod_matrix[0, :],
            success_ratio_by_threshold=success_ratio_matrix[0, :]
        )

    pyplot.title(title_string)

    print('Saving performance diagram to: "{0:s}"...'.format(output_file_name))
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()


def _plot_attributes_diagram(
        evaluation_table, num_examples_by_bin, output_file_name,
        confidence_level=None):
    """Plots attributes diagram.

    B = number of bins

    :param evaluation_table: See documentation for `_compute_scores`.  The only
        difference is that this table may have multiple rows (one per bootstrap
        replicate).
    :param num_examples_by_bin: length-B numpy array with number of examples per
        bin for the entire dataset.
    :param output_file_name: Path to output file (figure will be saved here).
    :param confidence_level: Confidence level for bootstrapping.
    """

    mean_forecast_prob_matrix = numpy.vstack(tuple(
        evaluation_table[MEAN_FORECAST_BY_BIN_KEY].values.tolist()
    ))
    event_frequency_matrix = numpy.vstack(tuple(
        evaluation_table[EVENT_FREQ_BY_BIN_KEY].values.tolist()
    ))

    mean_bss = numpy.mean(evaluation_table[BSS_KEY].values)
    title_string = 'BSS = {0:.3f}'.format(mean_bss)

    num_bootstrap_reps = mean_forecast_prob_matrix.shape[0]
    num_bins = mean_forecast_prob_matrix.shape[1]

    if num_bootstrap_reps > 1:
        min_bss, max_bss = bootstrapping.get_confidence_interval(
            stat_values=evaluation_table[BSS_KEY].values,
            confidence_level=confidence_level)

        title_string += ' [{0:.3f}, {1:.3f}]'.format(min_bss, max_bss)

    print(title_string)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    if num_bootstrap_reps > 1:
        bottom_dict = {
            model_eval.MEAN_FORECAST_PROB_BY_BIN_KEY:
                numpy.full(num_bins, numpy.nan),
            model_eval.MEAN_OBSERVED_LABEL_BY_BIN_KEY:
                numpy.full(num_bins, numpy.nan)
        }

        top_dict = copy.deepcopy(bottom_dict)
        mean_dict = copy.deepcopy(bottom_dict)

        for j in range(num_bins):
            (top_dict[model_eval.MEAN_FORECAST_PROB_BY_BIN_KEY][j],
             bottom_dict[model_eval.MEAN_FORECAST_PROB_BY_BIN_KEY][j]
            ) = bootstrapping.get_confidence_interval(
                stat_values=mean_forecast_prob_matrix[:, j],
                confidence_level=confidence_level
            )

            (bottom_dict[model_eval.MEAN_OBSERVED_LABEL_BY_BIN_KEY][j],
             top_dict[model_eval.MEAN_OBSERVED_LABEL_BY_BIN_KEY][j]
            ) = bootstrapping.get_confidence_interval(
                stat_values=event_frequency_matrix[:, j],
                confidence_level=confidence_level
            )

            mean_dict[model_eval.MEAN_FORECAST_PROB_BY_BIN_KEY][j] = (
                numpy.mean(mean_forecast_prob_matrix[:, j])
            )

            mean_dict[model_eval.MEAN_OBSERVED_LABEL_BY_BIN_KEY][j] = (
                numpy.mean(event_frequency_matrix[:, j])
            )

        model_eval_plotting.plot_bootstrapped_attributes_diagram(
            figure_object=figure_object, axes_object=axes_object,
            reliability_dict_bottom=bottom_dict,
            reliability_dict_mean=mean_dict, reliability_dict_top=top_dict,
            num_examples_by_bin=num_examples_by_bin)
    else:
        model_eval_plotting.plot_attributes_diagram(
            figure_object=figure_object, axes_object=axes_object,
            mean_forecast_prob_by_bin=mean_forecast_prob_matrix[0, :],
            mean_observed_label_by_bin=event_frequency_matrix[0, :],
            num_examples_by_bin=num_examples_by_bin)

    axes_object.set_title(title_string)

    print('Saving attributes diagram to: "{0:s}"...'.format(output_file_name))
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()


def _compute_scores(forecast_probabilities, observed_labels,
                    best_prob_threshold, all_prob_thresholds, climatology):
    """Computes evaluation scores.

    The input args `forecast_probabilities` and `observed_labels` may contain
    all data or just one bootstrap replicate.

    E = number of examples
    T = number of probability thresholds
    B = number of bins for reliability curve

    :param forecast_probabilities: length-E numpy array of forecast probs.
    :param observed_labels: length-E numpy array of observed labels (all 0 or
        1).
    :param best_prob_threshold: Best probability threshold (will be used to
        binarize forecasts).
    :param all_prob_thresholds: length-T numpy array of probability thresholds
        to use for ROC curve and performance diagram.
    :param climatology: Climatology (frequency of positive class in the entire
        dataset).
    :return: evaluation_table: pandas DataFrame with one row and the following
        columns.  All values are scalar unless noted otherwise.
    evaluation_table['num_true_positives']
    evaluation_table['num_false_positives']
    evaluation_table['num_false_negatives']
    evaluation_table['num_true_negatives']
    evaluation_table['probability_of_detection']
    evaluation_table['probability_of_false_detection']
    evaluation_table['success_ratio']
    evaluation_table['frequency_of_correct_nulls']
    evaluation_table['accuracy']
    evaluation_table['critical_success_index']
    evaluation_table['frequency_bias']
    evaluation_table['peirce_score']
    evaluation_table['heidke_score']
    evaluation_table['pod_by_threshold']: length-T numpy array of POD values
        (probability of detection).
    evaluation_table['pofd_by_threshold']: length-T numpy array of POFD values
        (probability of false detection).
    evaluation_table['area_under_roc_curve']
    evaluation_table['success_ratio_by_threshold']: length-T numpy array of
        success ratios.
    evaluation_table['area_under_perf_diagram']
    evaluation_table['mean_forecast_by_bin']: length-B numpy array of mean
        forecast probabilities.
    evaluation_table['event_frequency_by_bin']: length-B numpy array of event
        frequencies.
    evaluation_table['reliability']
    evaluation_table['resolution']
    evaluation_table['brier_skill_score']
    """

    # TODO(thunderhoser): Put this in model_evaluation.py.

    forecast_labels = model_eval.binarize_forecast_probs(
        forecast_probabilities=forecast_probabilities,
        binarization_threshold=best_prob_threshold)

    contingency_table_as_dict = model_eval.get_contingency_table(
        forecast_labels=forecast_labels,
        observed_labels=observed_labels)

    evaluation_dict = {
        NUM_TRUE_POSITIVES_KEY:
            contingency_table_as_dict[model_eval.NUM_TRUE_POSITIVES_KEY],
        NUM_FALSE_POSITIVES_KEY:
            contingency_table_as_dict[model_eval.NUM_FALSE_POSITIVES_KEY],
        NUM_FALSE_NEGATIVES_KEY:
            contingency_table_as_dict[model_eval.NUM_FALSE_NEGATIVES_KEY],
        NUM_TRUE_NEGATIVES_KEY:
            contingency_table_as_dict[model_eval.NUM_TRUE_NEGATIVES_KEY],
        POD_KEY: model_eval.get_pod(contingency_table_as_dict),
        POFD_KEY: model_eval.get_pofd(contingency_table_as_dict),
        SUCCESS_RATIO_KEY:
            model_eval.get_success_ratio(contingency_table_as_dict),
        FOCN_KEY: model_eval.get_focn(contingency_table_as_dict),
        ACCURACY_KEY: model_eval.get_accuracy(contingency_table_as_dict),
        CSI_KEY: model_eval.get_csi(contingency_table_as_dict),
        FREQUENCY_BIAS_KEY:
            model_eval.get_frequency_bias(contingency_table_as_dict),
        PEIRCE_SCORE_KEY:
            model_eval.get_peirce_score(contingency_table_as_dict),
        HEIDKE_SCORE_KEY: model_eval.get_heidke_score(contingency_table_as_dict)
    }

    print('\n{0:s}\n'.format(str(evaluation_dict)))

    for this_key in evaluation_dict:
        if this_key in contingency_table_as_dict:
            evaluation_dict[this_key] = numpy.array(
                [evaluation_dict[this_key]], dtype=int
            )
        else:
            evaluation_dict[this_key] = numpy.array(
                [evaluation_dict[this_key]], dtype=float
            )

    evaluation_table = pandas.DataFrame.from_dict(evaluation_dict)
    nested_array = evaluation_table[[CSI_KEY, CSI_KEY]].values.tolist()

    evaluation_table = evaluation_table.assign(**{
        POD_BY_THRESHOLD_KEY: nested_array,
        POFD_BY_THRESHOLD_KEY: nested_array,
        SR_BY_THRESHOLD_KEY: nested_array,
        MEAN_FORECAST_BY_BIN_KEY: nested_array,
        EVENT_FREQ_BY_BIN_KEY: nested_array
    })

    (evaluation_table[POFD_BY_THRESHOLD_KEY].values[0],
     evaluation_table[POD_BY_THRESHOLD_KEY].values[0]
    ) = model_eval.get_points_in_roc_curve(
        forecast_probabilities=forecast_probabilities,
        observed_labels=observed_labels,
        threshold_arg=all_prob_thresholds)

    auc = model_eval.get_area_under_roc_curve(
        pofd_by_threshold=evaluation_table[POFD_BY_THRESHOLD_KEY].values[0],
        pod_by_threshold=evaluation_table[POD_BY_THRESHOLD_KEY].values[0]
    )

    evaluation_table[SR_BY_THRESHOLD_KEY].values[0], _ = (
        model_eval.get_points_in_performance_diagram(
            forecast_probabilities=forecast_probabilities,
            observed_labels=observed_labels,
            threshold_arg=all_prob_thresholds)
    )

    aupd = model_eval.get_area_under_perf_diagram(
        success_ratio_by_threshold=
        evaluation_table[SR_BY_THRESHOLD_KEY].values[0],
        pod_by_threshold=evaluation_table[POD_BY_THRESHOLD_KEY].values[0]
    )

    (evaluation_table[MEAN_FORECAST_BY_BIN_KEY].values[0],
     evaluation_table[EVENT_FREQ_BY_BIN_KEY].values[0],
     num_examples_by_bin
    ) = model_eval.get_points_in_reliability_curve(
        forecast_probabilities=forecast_probabilities,
        observed_labels=observed_labels,
        num_forecast_bins=NUM_RELIABILITY_BINS)

    this_bss_dict = model_eval.get_brier_skill_score(
        mean_forecast_prob_by_bin=evaluation_table[
            MEAN_FORECAST_BY_BIN_KEY].values[0],
        mean_observed_label_by_bin=evaluation_table[
            EVENT_FREQ_BY_BIN_KEY].values[0],
        num_examples_by_bin=num_examples_by_bin, climatology=climatology)

    reliability = this_bss_dict[model_eval.RELIABILITY_KEY]
    resolution = this_bss_dict[model_eval.RESOLUTION_KEY]
    brier_skill_score = this_bss_dict[model_eval.BRIER_SKILL_SCORE_KEY]

    return evaluation_table.assign(**{
        AUC_KEY: auc,
        AUPD_KEY: aupd,
        RELIABILITY_KEY: reliability,
        RESOLUTION_KEY: resolution,
        BSS_KEY: brier_skill_score
    })


def run_evaluation(forecast_probabilities, observed_labels, output_dir_name,
                   num_bootstrap_reps, confidence_level):
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
    :param num_bootstrap_reps: Number of bootstrap replicates.  This may be 1,
        in which case no bootstrapping will be done.
    :param confidence_level: Confidence level for bootstrapping.
    """

    # TODO(thunderhoser): Make binarization threshold an input argument to this
    # method.

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    all_prob_thresholds = model_eval.get_binarization_thresholds(
        threshold_arg=model_eval.THRESHOLD_ARG_FOR_UNIQUE_FORECASTS,
        forecast_probabilities=forecast_probabilities,
        unique_forecast_precision=FORECAST_PRECISION_FOR_THRESHOLDS)

    best_prob_threshold, best_csi = model_eval.find_best_binarization_threshold(
        forecast_probabilities=forecast_probabilities,
        observed_labels=observed_labels, threshold_arg=all_prob_thresholds,
        criterion_function=model_eval.get_csi,
        optimization_direction=model_eval.MAX_OPTIMIZATION_DIRECTION)

    print((
        'Best probability threshold = {0:.4f} ... corresponding CSI = {1:.4f}'
    ).format(
        best_prob_threshold, best_csi
    ))

    num_examples_by_bin = model_eval.get_points_in_reliability_curve(
        forecast_probabilities=forecast_probabilities,
        observed_labels=observed_labels, num_forecast_bins=NUM_RELIABILITY_BINS
    )[-1]

    list_of_evaluation_tables = []

    for i in range(num_bootstrap_reps):
        print((
            'Computing scores for {0:d}th of {1:d} bootstrap replicates...'
        ).format(
            i + 1, num_bootstrap_reps
        ))

        _, these_indices = bootstrapping.draw_sample(forecast_probabilities)

        list_of_evaluation_tables.append(_compute_scores(
            forecast_probabilities=forecast_probabilities[these_indices],
            observed_labels=observed_labels[these_indices],
            best_prob_threshold=best_prob_threshold,
            all_prob_thresholds=all_prob_thresholds,
            climatology=numpy.mean(observed_labels)
        ))

        if i == num_bootstrap_reps - 1:
            print(SEPARATOR_STRING)
        else:
            print(MINOR_SEPARATOR_STRING)

        if i == 0:
            continue

        list_of_evaluation_tables[-1] = list_of_evaluation_tables[-1].align(
            list_of_evaluation_tables[0], axis=1
        )[0]

    evaluation_table = pandas.concat(
        list_of_evaluation_tables, axis=0, ignore_index=True)

    _plot_roc_curve(
        evaluation_table=evaluation_table,
        output_file_name='{0:s}/roc_curve.jpg'.format(output_dir_name),
        confidence_level=confidence_level)

    _plot_performance_diagram(
        evaluation_table=evaluation_table,
        output_file_name='{0:s}/performance_diagram.jpg'.format(
            output_dir_name),
        confidence_level=confidence_level)

    _plot_attributes_diagram(
        evaluation_table=evaluation_table,
        num_examples_by_bin=num_examples_by_bin,
        output_file_name='{0:s}/attributes_diagram.jpg'.format(
            output_dir_name),
        confidence_level=confidence_level)

    # TODO(thunderhoser): Fix file format.
    # TODO(thunderhoser): Fix calls from scripts to this method.

    # model_eval.write_results(
    #     forecast_probabilities=forecast_probabilities,
    #     observed_labels=observed_labels,
    #     binarization_threshold=best_prob_threshold, pod=pod, pofd=pofd,
    #     success_ratio=success_ratio, focn=focn, accuracy=accuracy, csi=csi,
    #     frequency_bias=frequency_bias, peirce_score=peirce_score,
    #     heidke_score=heidke_score, auc=auc, scikit_learn_auc=scikit_learn_auc,
    #     aupd=aupd, bss_dict=bss_dict, pickle_file_name=evaluation_file_name)
