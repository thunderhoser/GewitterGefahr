"""Plots model evaluation.  Specifically, plots the following figures.

- ROC curve
- performance diagram
- attributes diagram
"""

import copy
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import model_evaluation as model_eval
from gewittergefahr.gg_utils import bootstrapping
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import model_eval_plotting

BOUNDING_BOX_DICT = {
    'facecolor': 'white',
    'alpha': 0.5,
    'edgecolor': 'black',
    'linewidth': 2,
    'boxstyle': 'round'
}

MARKER_TYPE = '*'
MARKER_SIZE = 32
MARKER_EDGE_WIDTH = 0

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_eval_file_name'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file (will be read by `model_evaluation.read_evaluation`).')

CONFIDENCE_LEVEL_HELP_STRING = (
    'Level for confidence interval.  If input does not contain bootstrapped '
    'scores, no confidence interval will be plotted, so this will be '
    'irrelevant.')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=0.95,
    help=CONFIDENCE_LEVEL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _plot_roc_curve(evaluation_table, best_threshold_index, output_file_name,
                    confidence_level=None):
    """Plots ROC curve.

    :param evaluation_table: See doc for
        `model_evaluation.run_evaluation`.  The only difference is that
        this table may have multiple rows (one per bootstrap replicate).
    :param best_threshold_index: Array index of best probability threshold.
    :param output_file_name: Path to output file (figure will be saved here).
    :param confidence_level: Confidence level for bootstrapping.
    """

    pod_matrix = numpy.vstack(tuple(
        evaluation_table[model_eval.POD_BY_THRESHOLD_KEY].values.tolist()
    ))
    pofd_matrix = numpy.vstack(tuple(
        evaluation_table[model_eval.POFD_BY_THRESHOLD_KEY].values.tolist()
    ))

    num_bootstrap_reps = pod_matrix.shape[0]
    num_prob_thresholds = pod_matrix.shape[1]

    if num_bootstrap_reps > 1:
        min_auc, max_auc = bootstrapping.get_confidence_interval(
            stat_values=evaluation_table[model_eval.AUC_KEY].values,
            confidence_level=confidence_level)

        annotation_string = 'Area under curve = [{0:.3f}, {1:.3f}]'.format(
            min_auc, max_auc)
    else:
        mean_auc = numpy.nanmean(evaluation_table[model_eval.AUC_KEY].values)
        annotation_string = 'Area under curve = {0:.3f}'.format(mean_auc)

    print(annotation_string)

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

        best_x = ci_mean_dict[model_eval.POFD_BY_THRESHOLD_KEY][
            best_threshold_index]
        best_y = ci_mean_dict[model_eval.POD_BY_THRESHOLD_KEY][
            best_threshold_index]
    else:
        model_eval_plotting.plot_roc_curve(
            axes_object=axes_object, pod_by_threshold=pod_matrix[0, :],
            pofd_by_threshold=pofd_matrix[0, :]
        )

        best_x = pofd_matrix[0, best_threshold_index]
        best_y = pod_matrix[0, best_threshold_index]

    print((
        'POD and POFD at best probability threshold = {0:.3f}, {1:.3f}'
    ).format(
        best_y, best_x
    ))

    marker_colour = model_eval_plotting.ROC_CURVE_COLOUR
    # axes_object.plot(
    #     best_x, best_y, linestyle='None', marker=MARKER_TYPE,
    #     markersize=MARKER_SIZE, markeredgewidth=MARKER_EDGE_WIDTH,
    #     markerfacecolor=marker_colour, markeredgecolor=marker_colour)

    axes_object.text(
        0.98, 0.02, annotation_string, bbox=BOUNDING_BOX_DICT, color='k',
        horizontalalignment='right', verticalalignment='bottom',
        transform=axes_object.transAxes)

    axes_object.set_title('ROC curve')
    plotting_utils.label_axes(
        axes_object=axes_object, label_string='(a)', y_coord_normalized=1.025
    )

    axes_object.set_aspect('equal')

    print('Saving ROC curve to: "{0:s}"...'.format(output_file_name))
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
                   bbox_inches='tight')
    pyplot.close()


def _plot_performance_diagram(
        evaluation_table, best_threshold_index, output_file_name,
        confidence_level=None):
    """Plots performance diagram.

    :param evaluation_table: See doc for `_plot_roc_curve`.
    :param best_threshold_index: Array index of best probability threshold.
    :param output_file_name: Same.
    :param confidence_level: Same.
    """

    pod_matrix = numpy.vstack(tuple(
        evaluation_table[model_eval.POD_BY_THRESHOLD_KEY].values.tolist()
    ))
    success_ratio_matrix = numpy.vstack(tuple(
        evaluation_table[model_eval.SR_BY_THRESHOLD_KEY].values.tolist()
    ))

    mean_aupd = numpy.nanmean(evaluation_table[model_eval.AUPD_KEY].values)
    annotation_string = 'Area under curve = {0:.3f}'.format(mean_aupd)

    num_bootstrap_reps = pod_matrix.shape[0]
    num_prob_thresholds = pod_matrix.shape[1]

    if num_bootstrap_reps > 1:
        min_aupd, max_aupd = bootstrapping.get_confidence_interval(
            stat_values=evaluation_table[model_eval.AUPD_KEY].values,
            confidence_level=confidence_level)

        annotation_string = 'Area under curve = [{0:.3f}, {1:.3f}]'.format(
            min_aupd, max_aupd)

    print(annotation_string)

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

        best_x = ci_mean_dict[model_eval.SR_BY_THRESHOLD_KEY][
            best_threshold_index]
        best_y = ci_mean_dict[model_eval.POD_BY_THRESHOLD_KEY][
            best_threshold_index]
    else:
        model_eval_plotting.plot_performance_diagram(
            axes_object=axes_object, pod_by_threshold=pod_matrix[0, :],
            success_ratio_by_threshold=success_ratio_matrix[0, :]
        )

        best_x = success_ratio_matrix[0, best_threshold_index]
        best_y = pod_matrix[0, best_threshold_index]

    print((
        'POD and success ratio at best probability threshold = {0:.3f}, {1:.3f}'
    ).format(
        best_y, best_x
    ))

    marker_colour = model_eval_plotting.PERF_DIAGRAM_COLOUR
    # axes_object.plot(
    #     best_x, best_y, linestyle='None', marker=MARKER_TYPE,
    #     markersize=MARKER_SIZE, markeredgewidth=MARKER_EDGE_WIDTH,
    #     markerfacecolor=marker_colour, markeredgecolor=marker_colour)

    # axes_object.text(
    #     0.98, 0.98, annotation_string, bbox=BOUNDING_BOX_DICT, color='k',
    #     horizontalalignment='right', verticalalignment='top',
    #     transform=axes_object.transAxes)

    axes_object.set_title('Performance diagram')
    plotting_utils.label_axes(
        axes_object=axes_object, label_string='(b)', y_coord_normalized=1.025
    )

    axes_object.set_aspect('equal')

    print('Saving performance diagram to: "{0:s}"...'.format(output_file_name))
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
                   bbox_inches='tight')
    pyplot.close()


def _plot_attributes_diagram(
        evaluation_table, num_examples_by_bin, output_file_name,
        confidence_level=None):
    """Plots attributes diagram.

    K = number of bins for forecast probability

    :param evaluation_table: See doc for `_plot_roc_curve`.
    :param num_examples_by_bin: length-K numpy array with number of examples in
        each bin.
    :param output_file_name: See doc for `_plot_roc_curve`.
    :param confidence_level: Same.
    """

    mean_forecast_prob_matrix = numpy.vstack(tuple(
        evaluation_table[model_eval.MEAN_FORECAST_BY_BIN_KEY].values.tolist()
    ))
    event_frequency_matrix = numpy.vstack(tuple(
        evaluation_table[model_eval.EVENT_FREQ_BY_BIN_KEY].values.tolist()
    ))

    mean_bss = numpy.nanmean(evaluation_table[model_eval.BSS_KEY].values)
    annotation_string = 'Brier skill score = {0:.3f}'.format(mean_bss)

    num_bootstrap_reps = mean_forecast_prob_matrix.shape[0]
    num_bins = mean_forecast_prob_matrix.shape[1]

    if num_bootstrap_reps > 1:
        min_bss, max_bss = bootstrapping.get_confidence_interval(
            stat_values=evaluation_table[model_eval.BSS_KEY].values,
            confidence_level=confidence_level)

        annotation_string = 'Brier skill score = [{0:.3f}, {1:.3f}]'.format(
            min_bss, max_bss)

    print(annotation_string)

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

    axes_object.text(
        0.02, 0.98, annotation_string, bbox=BOUNDING_BOX_DICT, color='k',
        horizontalalignment='left', verticalalignment='top',
        transform=axes_object.transAxes)

    axes_object.set_title('Attributes diagram')
    plotting_utils.label_axes(
        axes_object=axes_object, label_string='(c)', y_coord_normalized=1.025
    )

    axes_object.set_aspect('equal')

    print('Saving attributes diagram to: "{0:s}"...'.format(output_file_name))
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
                   bbox_inches='tight')
    pyplot.close()


def _run(evaluation_file_name, confidence_level, output_dir_name):
    """Plots model evaluation.  Specifically, plots the following figures.

    This is effectively the main method.

    :param evaluation_file_name: See documentation at top of file.
    :param confidence_level: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    print('Reading data from: "{0:s}"...'.format(evaluation_file_name))
    evaluation_dict = model_eval.read_evaluation(evaluation_file_name)

    evaluation_table = evaluation_dict[model_eval.EVALUATION_TABLE_KEY]
    num_examples_by_forecast_bin = evaluation_dict[
        model_eval.NUM_EXAMPLES_BY_BIN_KEY]

    best_threshold_index = numpy.argmin(numpy.absolute(
        evaluation_dict[model_eval.BEST_THRESHOLD_KEY] -
        evaluation_dict[model_eval.ALL_THRESHOLDS_KEY]
    ))

    _plot_roc_curve(
        evaluation_table=evaluation_table,
        best_threshold_index=best_threshold_index,
        output_file_name='{0:s}/roc_curve.jpg'.format(output_dir_name),
        confidence_level=confidence_level)

    _plot_performance_diagram(
        evaluation_table=evaluation_table,
        best_threshold_index=best_threshold_index,
        output_file_name='{0:s}/performance_diagram.jpg'.format(
            output_dir_name),
        confidence_level=confidence_level)

    _plot_attributes_diagram(
        evaluation_table=evaluation_table,
        num_examples_by_bin=num_examples_by_forecast_bin,
        output_file_name='{0:s}/attributes_diagram.jpg'.format(output_dir_name),
        confidence_level=confidence_level)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        evaluation_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
