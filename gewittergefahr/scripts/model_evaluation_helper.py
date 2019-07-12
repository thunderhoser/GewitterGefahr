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
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import model_eval_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

FORECAST_PRECISION_FOR_THRESHOLDS = 1e-4

FIGURE_RESOLUTION_DPI = 300
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15


def _plot_roc_curve(evaluation_table, output_file_name, confidence_level=None):
    """Plots ROC curve.

    :param evaluation_table: See doc for
        `model_evaluation.eval_binary_classifn`.  The only difference is that
        this table may have multiple rows (one per bootstrap replicate).
    :param output_file_name: Path to output file (figure will be saved here).
    :param confidence_level: Confidence level for bootstrapping.
    """

    pod_matrix = numpy.vstack(tuple(
        evaluation_table[model_eval.POD_BY_THRESHOLD_KEY].values.tolist()
    ))
    pofd_matrix = numpy.vstack(tuple(
        evaluation_table[model_eval.POD_BY_THRESHOLD_KEY].values.tolist()
    ))

    mean_auc = numpy.mean(evaluation_table[model_eval.AUC_KEY].values)
    title_string = 'AUC = {0:.3f}'.format(mean_auc)

    num_bootstrap_reps = pod_matrix.shape[0]
    num_prob_thresholds = pod_matrix.shape[1]

    if num_bootstrap_reps > 1:
        min_auc, max_auc = bootstrapping.get_confidence_interval(
            stat_values=evaluation_table[model_eval.AUC_KEY].values,
            confidence_level=confidence_level)

        title_string += ' [{0:.3f}, {1:.3f}]'.format(min_auc, max_auc)

    print(title_string)

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    if num_bootstrap_reps > 1:
        ci_bottom_dict = {
            model_eval.POD_BY_THRESHOLD_KEY:
                numpy.full(num_prob_thresholds, numpy.nan),
            model_eval.POFD_BY_THRESHOLD_KEY:
                numpy.full(num_prob_thresholds, numpy.nan)
        }

        ci_top_dict = copy.deepcopy(ci_bottom_dict)
        ci_mean_dict = copy.deepcopy(ci_bottom_dict)

        for j in range(num_prob_thresholds):
            (ci_bottom_dict[model_eval.POD_BY_THRESHOLD_KEY][j],
             ci_top_dict[model_eval.POD_BY_THRESHOLD_KEY][j]
            ) = bootstrapping.get_confidence_interval(
                stat_values=pod_matrix[:, j], confidence_level=confidence_level
            )

            (ci_top_dict[model_eval.POFD_BY_THRESHOLD_KEY][j],
             ci_bottom_dict[model_eval.POFD_BY_THRESHOLD_KEY][j]
            ) = bootstrapping.get_confidence_interval(
                stat_values=pofd_matrix[:, j], confidence_level=confidence_level
            )

            ci_mean_dict[model_eval.POD_BY_THRESHOLD_KEY][j] = numpy.mean(
                pod_matrix[:, j]
            )

            ci_mean_dict[model_eval.POFD_BY_THRESHOLD_KEY][j] = numpy.mean(
                pofd_matrix[:, j]
            )

        model_eval_plotting.plot_bootstrapped_roc_curve(
            axes_object=axes_object, ci_bottom_dict=ci_bottom_dict,
            ci_mean_dict=ci_mean_dict, ci_top_dict=ci_top_dict)
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

    :param evaluation_table: See doc for
        `model_evaluation.eval_binary_classifn`.  The only difference is that
        this table may have multiple rows (one per bootstrap replicate).
    :param output_file_name: Path to output file (figure will be saved here).
    :param confidence_level: Confidence level for bootstrapping.
    """

    pod_matrix = numpy.vstack(tuple(
        evaluation_table[model_eval.POD_BY_THRESHOLD_KEY].values.tolist()
    ))
    success_ratio_matrix = numpy.vstack(tuple(
        evaluation_table[model_eval.SR_BY_THRESHOLD_KEY].values.tolist()
    ))

    mean_aupd = numpy.mean(evaluation_table[model_eval.AUPD_KEY].values)
    title_string = 'AUPD = {0:.3f}'.format(mean_aupd)

    num_bootstrap_reps = pod_matrix.shape[0]
    num_prob_thresholds = pod_matrix.shape[1]

    if num_bootstrap_reps > 1:
        min_aupd, max_aupd = bootstrapping.get_confidence_interval(
            stat_values=evaluation_table[model_eval.AUPD_KEY].values,
            confidence_level=confidence_level)

        title_string += ' [{0:.3f}, {1:.3f}]'.format(min_aupd, max_aupd)

    print(title_string)

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    if num_bootstrap_reps > 1:
        ci_bottom_dict = {
            model_eval.POD_BY_THRESHOLD_KEY:
                numpy.full(num_prob_thresholds, numpy.nan),
            model_eval.SR_BY_THRESHOLD_KEY:
                numpy.full(num_prob_thresholds, numpy.nan)
        }

        ci_top_dict = copy.deepcopy(ci_bottom_dict)
        ci_mean_dict = copy.deepcopy(ci_bottom_dict)

        for j in range(num_prob_thresholds):
            (ci_bottom_dict[model_eval.POD_BY_THRESHOLD_KEY][j],
             ci_top_dict[model_eval.POD_BY_THRESHOLD_KEY][j]
            ) = bootstrapping.get_confidence_interval(
                stat_values=pod_matrix[:, j], confidence_level=confidence_level
            )

            (ci_bottom_dict[model_eval.SR_BY_THRESHOLD_KEY][j],
             ci_top_dict[model_eval.SR_BY_THRESHOLD_KEY][j]
            ) = bootstrapping.get_confidence_interval(
                stat_values=success_ratio_matrix[:, j],
                confidence_level=confidence_level
            )

            ci_mean_dict[model_eval.POD_BY_THRESHOLD_KEY][j] = numpy.mean(
                pod_matrix[:, j]
            )

            ci_mean_dict[model_eval.SR_BY_THRESHOLD_KEY][j] = numpy.mean(
                success_ratio_matrix[:, j]
            )

        model_eval_plotting.plot_bootstrapped_performance_diagram(
            axes_object=axes_object, ci_bottom_dict=ci_bottom_dict,
            ci_mean_dict=ci_mean_dict, ci_top_dict=ci_top_dict)
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
        evaluation_table[model_eval.MEAN_FORECAST_BY_BIN_KEY].values.tolist()
    ))
    event_frequency_matrix = numpy.vstack(tuple(
        evaluation_table[model_eval.EVENT_FREQ_BY_BIN_KEY].values.tolist()
    ))

    mean_bss = numpy.mean(evaluation_table[model_eval.BSS_KEY].values)
    title_string = 'BSS = {0:.3f}'.format(mean_bss)

    num_bootstrap_reps = mean_forecast_prob_matrix.shape[0]
    num_bins = mean_forecast_prob_matrix.shape[1]

    if num_bootstrap_reps > 1:
        min_bss, max_bss = bootstrapping.get_confidence_interval(
            stat_values=evaluation_table[model_eval.BSS_KEY].values,
            confidence_level=confidence_level)

        title_string += ' [{0:.3f}, {1:.3f}]'.format(min_bss, max_bss)

    print(title_string)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    if num_bootstrap_reps > 1:
        ci_bottom_dict = {
            model_eval.MEAN_FORECAST_BY_BIN_KEY:
                numpy.full(num_bins, numpy.nan),
            model_eval.EVENT_FREQ_BY_BIN_KEY: numpy.full(num_bins, numpy.nan)
        }

        ci_top_dict = copy.deepcopy(ci_bottom_dict)
        ci_mean_dict = copy.deepcopy(ci_bottom_dict)

        for j in range(num_bins):
            (ci_top_dict[model_eval.MEAN_FORECAST_BY_BIN_KEY][j],
             ci_bottom_dict[model_eval.MEAN_FORECAST_BY_BIN_KEY][j]
            ) = bootstrapping.get_confidence_interval(
                stat_values=mean_forecast_prob_matrix[:, j],
                confidence_level=confidence_level
            )

            (ci_bottom_dict[model_eval.EVENT_FREQ_BY_BIN_KEY][j],
             ci_top_dict[model_eval.EVENT_FREQ_BY_BIN_KEY][j]
            ) = bootstrapping.get_confidence_interval(
                stat_values=event_frequency_matrix[:, j],
                confidence_level=confidence_level
            )

            ci_mean_dict[model_eval.EVENT_FREQ_BY_BIN_KEY][j] = numpy.mean(
                mean_forecast_prob_matrix[:, j]
            )

            ci_mean_dict[model_eval.EVENT_FREQ_BY_BIN_KEY][j] = numpy.mean(
                event_frequency_matrix[:, j]
            )

        model_eval_plotting.plot_bootstrapped_attributes_diagram(
            figure_object=figure_object, axes_object=axes_object,
            ci_bottom_dict=ci_bottom_dict, ci_mean_dict=ci_mean_dict,
            ci_top_dict=ci_top_dict, num_examples_by_bin=num_examples_by_bin)
    else:
        model_eval_plotting.plot_attributes_diagram(
            figure_object=figure_object, axes_object=axes_object,
            mean_forecast_by_bin=mean_forecast_prob_matrix[0, :],
            event_frequency_by_bin=event_frequency_matrix[0, :],
            num_examples_by_bin=num_examples_by_bin)

    axes_object.set_title(title_string)

    print('Saving attributes diagram to: "{0:s}"...'.format(output_file_name))
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()


def run_evaluation(forecast_probabilities, observed_labels, num_bootstrap_reps,
                   output_dir_name, confidence_level=None):
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
    :param num_bootstrap_reps: Number of bootstrap replicates.  This may be 1,
        in which case no bootstrapping will be done.
    :param output_dir_name: Name of output directory.
    :param confidence_level: [used only if `num_bootstrap_reps > 1`]
        Confidence level for bootstrapping.
    """

    # TODO(thunderhoser): Make binarization threshold an input argument to this
    # method.

    error_checking.assert_is_integer(num_bootstrap_reps)
    num_bootstrap_reps = max([num_bootstrap_reps, 1])

    if num_bootstrap_reps > 1:
        error_checking.assert_is_geq(confidence_level, 0.5)
        error_checking.assert_is_less_than(confidence_level, 1.)

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
        optimization_direction=model_eval.MAX_OPTIMIZATION_STRING)

    print((
        'Best probability threshold = {0:.4f} ... corresponding CSI = {1:.4f}'
    ).format(
        best_prob_threshold, best_csi
    ))

    num_examples_by_bin = model_eval.get_points_in_reliability_curve(
        forecast_probabilities=forecast_probabilities,
        observed_labels=observed_labels,
        num_forecast_bins=model_eval.DEFAULT_NUM_RELIABILITY_BINS
    )[-1]

    list_of_evaluation_tables = []

    for i in range(num_bootstrap_reps):
        print((
            'Computing scores for {0:d}th of {1:d} bootstrap replicates...'
        ).format(
            i + 1, num_bootstrap_reps
        ))

        _, these_indices = bootstrapping.draw_sample(forecast_probabilities)

        this_evaluation_table = model_eval.eval_binary_classifn(
            forecast_probabilities=forecast_probabilities[these_indices],
            observed_labels=observed_labels[these_indices],
            best_prob_threshold=best_prob_threshold,
            all_prob_thresholds=all_prob_thresholds,
            climatology=numpy.mean(observed_labels)
        )

        list_of_evaluation_tables.append(this_evaluation_table)

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

    output_file_name = '{0:s}/evaluation_results.p'.format(output_dir_name)
    print('Writing results to: "{0:s}"...'.format(output_file_name))

    model_eval.write_binary_classifn_results(
        pickle_file_name=output_file_name,
        forecast_probabilities=forecast_probabilities,
        observed_labels=observed_labels,
        best_prob_threshold=best_prob_threshold,
        all_prob_thresholds=all_prob_thresholds,
        evaluation_table=evaluation_table)
