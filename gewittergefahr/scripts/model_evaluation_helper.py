"""High-level methods for model evaluation.

To be used by scripts (i.e., files in the "scripts" package).

WARNING: This file works for only binary classification (not regression or
multiclass classification).
"""

import copy
import os.path
import numpy
import pandas
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import bootstrapping
from gewittergefahr.gg_utils import model_evaluation as model_eval
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.plotting import model_eval_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

FORECAST_PRECISION = 1e-4
DUMMY_TARGET_NAME = 'tornado_lead-time=0000-3600sec_distance=00000-10000m'

FIGURE_RESOLUTION_DPI = 300
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15


def _plot_roc_curve(evaluation_table, output_file_name, confidence_level=None):
    """Plots ROC curve.

    :param evaluation_table: See doc for
        `model_evaluation.run_evaluation`.  The only difference is that
        this table may have multiple rows (one per bootstrap replicate).
    :param output_file_name: Path to output file (figure will be saved here).
    :param confidence_level: Confidence level for bootstrapping.
    """

    pod_matrix = numpy.vstack(tuple(
        evaluation_table[model_eval.POD_BY_THRESHOLD_KEY].values.tolist()
    ))
    pofd_matrix = numpy.vstack(tuple(
        evaluation_table[model_eval.POFD_BY_THRESHOLD_KEY].values.tolist()
    ))

    mean_auc = numpy.nanmean(evaluation_table[model_eval.AUC_KEY].values)
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

            ci_mean_dict[model_eval.POD_BY_THRESHOLD_KEY][j] = numpy.nanmean(
                pod_matrix[:, j]
            )

            ci_mean_dict[model_eval.POFD_BY_THRESHOLD_KEY][j] = numpy.nanmean(
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
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
                   bbox_inches='tight')
    pyplot.close()


def _plot_performance_diagram(evaluation_table, output_file_name,
                              confidence_level=None):
    """Plots performance diagram.

    :param evaluation_table: See doc for
        `model_evaluation.run_evaluation`.  The only difference is that
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

    mean_aupd = numpy.nanmean(evaluation_table[model_eval.AUPD_KEY].values)
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

            ci_mean_dict[model_eval.POD_BY_THRESHOLD_KEY][j] = numpy.nanmean(
                pod_matrix[:, j]
            )

            ci_mean_dict[model_eval.SR_BY_THRESHOLD_KEY][j] = numpy.nanmean(
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
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
                   bbox_inches='tight')
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

    mean_bss = numpy.nanmean(evaluation_table[model_eval.BSS_KEY].values)
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

            ci_mean_dict[model_eval.MEAN_FORECAST_BY_BIN_KEY][j] = (
                numpy.nanmean(mean_forecast_prob_matrix[:, j])
            )

            ci_mean_dict[model_eval.EVENT_FREQ_BY_BIN_KEY][j] = numpy.nanmean(
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
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
                   bbox_inches='tight')
    pyplot.close()


def run_evaluation(
        forecast_probabilities, observed_labels, num_bootstrap_reps,
        main_output_file_name, best_prob_threshold=None, downsampling_dict=None,
        confidence_level=None):
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
    :param main_output_file_name: Path to main output file (will be written by
        `model_evaluation.write_evaluation`).
    :param best_prob_threshold: Best probability threshold (used to turn
        probabilities into deterministic predictions).  If None, will use
        threshold that yields the best CSI.
    :param downsampling_dict: Dictionary used to downsample classes.  See doc
        for `deep_learning_utils.sample_by_class`.  If this is None, there will
        be no downsampling.
    :param confidence_level: [used only if `num_bootstrap_reps > 1`]
        Confidence level for bootstrapping.
    """

    error_checking.assert_is_integer(num_bootstrap_reps)
    num_bootstrap_reps = max([num_bootstrap_reps, 1])

    if num_bootstrap_reps > 1:
        error_checking.assert_is_geq(confidence_level, 0.5)
        error_checking.assert_is_less_than(confidence_level, 1.)

    file_system_utils.mkdir_recursive_if_necessary(
        file_name=main_output_file_name)

    figure_dir_name = os.path.splitext(main_output_file_name)[0]
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=figure_dir_name)

    num_examples_by_class = numpy.unique(
        observed_labels, return_counts=True
    )[-1]

    print('Number of examples by class (sans downsampling): {0:s}'.format(
        str(num_examples_by_class)
    ))

    num_examples = len(observed_labels)
    positive_example_indices = numpy.where(observed_labels == 1)[0]
    negative_example_indices = numpy.where(observed_labels == 0)[0]

    if downsampling_dict is None:
        these_indices = numpy.linspace(
            0, num_examples - 1, num=num_examples, dtype=int)
    else:
        these_indices = dl_utils.sample_by_class(
            sampling_fraction_by_class_dict=downsampling_dict,
            target_name=DUMMY_TARGET_NAME, target_values=observed_labels,
            num_examples_total=num_examples)

        this_num_ex_by_class = numpy.unique(
            observed_labels[these_indices], return_counts=True
        )[-1]

        print('Number of examples by class: {0:s}'.format(
            str(this_num_ex_by_class)
        ))

    all_prob_thresholds = model_eval.get_binarization_thresholds(
        threshold_arg=model_eval.THRESHOLD_ARG_FOR_UNIQUE_FORECASTS,
        forecast_probabilities=forecast_probabilities[these_indices],
        forecast_precision=FORECAST_PRECISION)

    if best_prob_threshold is None:
        best_prob_threshold, best_csi = (
            model_eval.find_best_binarization_threshold(
                forecast_probabilities=forecast_probabilities[these_indices],
                observed_labels=observed_labels[these_indices],
                threshold_arg=all_prob_thresholds,
                criterion_function=model_eval.get_csi,
                optimization_direction=model_eval.MAX_OPTIMIZATION_STRING)
        )
    else:
        these_forecast_labels = model_eval.binarize_forecast_probs(
            forecast_probabilities=forecast_probabilities[these_indices],
            binarization_threshold=best_prob_threshold)

        this_contingency_dict = model_eval.get_contingency_table(
            forecast_labels=these_forecast_labels,
            observed_labels=observed_labels[these_indices]
        )

        best_csi = model_eval.get_csi(this_contingency_dict)

    print((
        'Best probability threshold = {0:.4f} ... corresponding CSI = '
        '{1:.4f}'
    ).format(
        best_prob_threshold, best_csi
    ))

    num_examples_by_bin = model_eval.get_points_in_reliability_curve(
        forecast_probabilities=forecast_probabilities[these_indices],
        observed_labels=observed_labels[these_indices],
        num_forecast_bins=model_eval.DEFAULT_NUM_RELIABILITY_BINS
    )[-1]

    list_of_evaluation_tables = []

    for i in range(num_bootstrap_reps):
        print((
            'Computing scores for {0:d}th of {1:d} bootstrap replicates...'
        ).format(
            i + 1, num_bootstrap_reps
        ))

        if num_bootstrap_reps == 1:
            if downsampling_dict is None:
                these_indices = numpy.linspace(
                    0, num_examples - 1, num=num_examples, dtype=int)
            else:
                these_indices = dl_utils.sample_by_class(
                    sampling_fraction_by_class_dict=downsampling_dict,
                    target_name=DUMMY_TARGET_NAME,
                    target_values=observed_labels,
                    num_examples_total=num_examples)
        else:
            if len(positive_example_indices) > 0:
                these_positive_indices = bootstrapping.draw_sample(
                    positive_example_indices
                )[0]
            else:
                these_positive_indices = numpy.array([], dtype=int)

            these_negative_indices = bootstrapping.draw_sample(
                negative_example_indices
            )[0]

            these_indices = numpy.concatenate((
                these_positive_indices, these_negative_indices))

            if downsampling_dict is not None:
                these_subindices = dl_utils.sample_by_class(
                    sampling_fraction_by_class_dict=downsampling_dict,
                    target_name=DUMMY_TARGET_NAME,
                    target_values=observed_labels[these_indices],
                    num_examples_total=num_examples)

                these_indices = these_indices[these_subindices]

        if downsampling_dict is not None:
            this_num_ex_by_class = numpy.unique(
                observed_labels[these_indices], return_counts=True
            )[-1]

            print('Number of examples by class: {0:s}'.format(
                str(this_num_ex_by_class)
            ))

        this_evaluation_table = model_eval.run_evaluation(
            forecast_probabilities=forecast_probabilities[these_indices],
            observed_labels=observed_labels[these_indices],
            best_prob_threshold=best_prob_threshold,
            all_prob_thresholds=all_prob_thresholds,
            climatology=numpy.mean(observed_labels[these_indices])
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
        output_file_name='{0:s}/roc_curve.jpg'.format(figure_dir_name),
        confidence_level=confidence_level)

    _plot_performance_diagram(
        evaluation_table=evaluation_table,
        output_file_name='{0:s}/performance_diagram.jpg'.format(
            figure_dir_name),
        confidence_level=confidence_level)

    _plot_attributes_diagram(
        evaluation_table=evaluation_table,
        num_examples_by_bin=num_examples_by_bin,
        output_file_name='{0:s}/attributes_diagram.jpg'.format(
            figure_dir_name),
        confidence_level=confidence_level)

    print('Writing results to: "{0:s}"...'.format(main_output_file_name))

    model_eval.write_evaluation(
        pickle_file_name=main_output_file_name,
        forecast_probabilities=forecast_probabilities,
        observed_labels=observed_labels,
        best_prob_threshold=best_prob_threshold,
        all_prob_thresholds=all_prob_thresholds,
        evaluation_table=evaluation_table)
