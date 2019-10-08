"""Plots evaluation for many models on the same axes.

Specifically, plots the following figures:

- ROC curve
- performance diagram
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
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import model_eval_plotting

MARKER_TYPE = '*'
MARKER_SIZE = 32
MARKER_EDGE_WIDTH = 0

COLOUR_MATRIX = numpy.array([
    [27, 158, 119],
    [217, 95, 2]
], dtype=float) / 255

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_FILES_ARG_NAME = 'input_eval_file_names'
MODEL_NAMES_ARG_NAME = 'model_names'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILES_HELP_STRING = (
    'List of paths to input files (one for each model).  Will be read by '
    '`model_evaluation.read_evaluation`.')

MODEL_NAMES_HELP_STRING = (
    'List of model names (will be used in legend for each figure).  List should'
    ' be space-separated.  Underscores within each item will be turned into '
    'spaces.')

CONFIDENCE_LEVEL_HELP_STRING = (
    'Level for confidence interval.  If input does not contain bootstrapped '
    'scores, no confidence interval will be plotted, so this will be '
    'irrelevant.')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_FILES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_NAMES_ARG_NAME, type=str, nargs='+', required=True,
    help=MODEL_NAMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=0.95,
    help=CONFIDENCE_LEVEL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _get_ci_one_model(evaluation_table, for_roc_curve, confidence_level):
    """Returns confidence interval for one model.

    T = number of probability thresholds

    :param evaluation_table: Similar to pandas DataFrame created by
        `model_evaluation.run_evaluation`, except that this table has multiple
        rows (one per bootstrap replicate).
    :param for_roc_curve: Boolean flag.  If True, will return confidence
        interval for ROC curve.  If False, for performance diagram.
    :param confidence_level: Confidence level (in range 0...1).
    :return: ci_bottom_dict: Dictionary with the following keys (for bottom of
        confidence interval).
    ci_bottom_dict["pod_by_threshold"]: length-T numpy array of POD values.
    ci_bottom_dict["pofd_by_threshold"]: length-T numpy array of POFD values.
        If `for_roc_curve == False`, this key is missing.
    ci_bottom_dict["success_ratio_by_threshold"]: length-T numpy array of success
        ratios.  If `for_roc_curve == True`, this key is missing.

    :return: ci_mean_dict: Same but for mean of confidence interval.
    :return: ci_top_dict: Same but for top of confidence interval.
    """

    pod_matrix = numpy.vstack(tuple(
        evaluation_table[model_eval.POD_BY_THRESHOLD_KEY].values.tolist()
    ))

    num_thresholds = pod_matrix.shape[1]

    if for_roc_curve:
        pofd_matrix = numpy.vstack(tuple(
            evaluation_table[model_eval.POFD_BY_THRESHOLD_KEY].values.tolist()
        ))

        ci_bottom_dict = {
            model_eval.POD_BY_THRESHOLD_KEY:
                numpy.full(num_thresholds, numpy.nan),
            model_eval.POFD_BY_THRESHOLD_KEY:
                numpy.full(num_thresholds, numpy.nan)
        }
    else:
        success_ratio_matrix = numpy.vstack(tuple(
            evaluation_table[model_eval.SR_BY_THRESHOLD_KEY].values.tolist()
        ))

        ci_bottom_dict = {
            model_eval.POD_BY_THRESHOLD_KEY:
                numpy.full(num_thresholds, numpy.nan),
            model_eval.SR_BY_THRESHOLD_KEY:
                numpy.full(num_thresholds, numpy.nan)
        }

    ci_mean_dict = copy.deepcopy(ci_bottom_dict)
    ci_top_dict = copy.deepcopy(ci_bottom_dict)

    for j in range(num_thresholds):
        this_min_pod, this_max_pod = bootstrapping.get_confidence_interval(
            stat_values=pod_matrix[:, j], confidence_level=confidence_level
        )

        ci_bottom_dict[model_eval.POD_BY_THRESHOLD_KEY][j] = this_min_pod
        ci_top_dict[model_eval.POD_BY_THRESHOLD_KEY][j] = this_max_pod
        ci_mean_dict[model_eval.POD_BY_THRESHOLD_KEY][j] = numpy.nanmean(
            pod_matrix[:, j]
        )

        if for_roc_curve:
            this_min_pofd, this_max_pofd = (
                bootstrapping.get_confidence_interval(
                    stat_values=pofd_matrix[:, j],
                    confidence_level=confidence_level
                )
            )

            ci_bottom_dict[model_eval.POFD_BY_THRESHOLD_KEY][j] = this_min_pofd
            ci_top_dict[model_eval.POFD_BY_THRESHOLD_KEY][j] = this_max_pofd
            ci_mean_dict[model_eval.POFD_BY_THRESHOLD_KEY][j] = numpy.nanmean(
                pofd_matrix[:, j]
            )

            continue

        this_min_success_ratio, this_max_success_ratio = (
            bootstrapping.get_confidence_interval(
                stat_values=success_ratio_matrix[:, j],
                confidence_level=confidence_level
            )
        )

        ci_bottom_dict[model_eval.SR_BY_THRESHOLD_KEY][
            j] = this_min_success_ratio
        ci_top_dict[model_eval.SR_BY_THRESHOLD_KEY][j] = this_max_success_ratio
        ci_mean_dict[model_eval.SR_BY_THRESHOLD_KEY][j] = numpy.nanmean(
            success_ratio_matrix[:, j]
        )

    return ci_bottom_dict, ci_mean_dict, ci_top_dict


def _plot_roc_curves(evaluation_tables, model_names, best_threshold_indices,
                     output_file_name, confidence_level=None):
    """Plots ROC curves (one for each model).

    M = number of models

    :param evaluation_tables: length-M list of pandas DataFrames.  See
        `model_evaluation.run_evaluation` for columns in each DataFrame.  The
        only difference is that each table here may have multiple rows (one per
        bootstrap replicate).
    :param model_names: length-M list of model names (will be used in legend).
    :param best_threshold_indices: length-M numpy array with index of best
        probability threshold for each model.
    :param output_file_name: Path to output file (figure will be saved here).
    :param confidence_level: Confidence level for bootstrapping.
    """

    num_models = len(evaluation_tables)
    pod_matrices = [None] * num_models
    pofd_matrices = [None] * num_models
    legend_strings = [None] * num_models

    num_bootstrap_reps = None

    for i in range(num_models):
        pod_matrices[i] = numpy.vstack(tuple(
            evaluation_tables[i][
                model_eval.POD_BY_THRESHOLD_KEY
            ].values.tolist()
        ))

        pofd_matrices[i] = numpy.vstack(tuple(
            evaluation_tables[i][
                model_eval.POFD_BY_THRESHOLD_KEY
            ].values.tolist()
        ))

        if num_bootstrap_reps is None:
            num_bootstrap_reps = pod_matrices[i].shape[0]

        this_num_bootstrap_reps = pod_matrices[i].shape[0]
        # assert num_bootstrap_reps == this_num_bootstrap_reps

        if num_bootstrap_reps > 1:
            this_min_auc, this_max_auc = bootstrapping.get_confidence_interval(
                stat_values=evaluation_tables[i][model_eval.AUC_KEY].values,
                confidence_level=confidence_level)

            legend_strings[i] = '{0:s} ... AUC = [{1:.3f}, {2:.3f}]'.format(
                model_names[i], this_min_auc, this_max_auc
            )
        else:
            this_auc = evaluation_tables[i][model_eval.AUC_KEY].values[0]
            legend_strings[i] = '{0:s} ... AUC = {1:.3f}'.format(
                model_names[i], this_auc
            )

        print(legend_strings[i])

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    legend_handles = [None] * num_models
    num_colours = COLOUR_MATRIX.shape[0]

    for i in range(num_models):
        this_colour = COLOUR_MATRIX[numpy.mod(i, num_colours), ...]

        if num_bootstrap_reps == 1:
            legend_handles[i] = model_eval_plotting.plot_roc_curve(
                axes_object=axes_object,
                pod_by_threshold=pod_matrices[i][0, :],
                pofd_by_threshold=pofd_matrices[i][0, :],
                line_colour=this_colour, plot_background=i == 0
            )

            this_x = pofd_matrices[i][0, best_threshold_indices[i]]
            this_y = pod_matrices[i][0, best_threshold_indices[i]]
        else:
            this_ci_bottom_dict, this_ci_mean_dict, this_ci_top_dict = (
                _get_ci_one_model(
                    evaluation_table=evaluation_tables[i], for_roc_curve=True,
                    confidence_level=confidence_level)
            )

            legend_handles[i] = model_eval_plotting.plot_bootstrapped_roc_curve(
                axes_object=axes_object, ci_bottom_dict=this_ci_bottom_dict,
                ci_mean_dict=this_ci_mean_dict, ci_top_dict=this_ci_top_dict,
                line_colour=this_colour, plot_background=i == 0)

            this_x = this_ci_mean_dict[model_eval.POFD_BY_THRESHOLD_KEY][
                best_threshold_indices[i]
            ]
            this_y = this_ci_mean_dict[model_eval.POD_BY_THRESHOLD_KEY][
                best_threshold_indices[i]
            ]

        print((
            'POD and POFD at best probability threshold = {0:.3f}, {1:.3f}'
        ).format(
            this_y, this_x
        ))

        axes_object.plot(
            this_x, this_y, linestyle='None', marker=MARKER_TYPE,
            markersize=MARKER_SIZE, markeredgewidth=MARKER_EDGE_WIDTH,
            markerfacecolor=this_colour, markeredgecolor=this_colour)

    axes_object.legend(
        legend_handles, legend_strings, loc='lower center',
        bbox_to_anchor=(0.5, 0.025), fancybox=True, shadow=True,
        ncol=len(legend_handles)
    )

    axes_object.set_title('ROC curve')
    plotting_utils.label_axes(
        axes_object=axes_object, label_string='(a)', y_coord_normalized=1.025
    )

    print('Saving ROC curve to: "{0:s}"...'.format(output_file_name))
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
                   bbox_inches='tight')
    pyplot.close()


def _run(evaluation_file_names, model_names, confidence_level, output_dir_name):
    """Plots evaluation for many models on the same axes.

    This is effectively the main method.

    :param evaluation_file_names: See documentation at top of file.
    :param model_names: Same.
    :param confidence_level: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    model_names = [n.replace('_', ' ') for n in model_names]

    num_models = len(evaluation_file_names)
    expected_dim = numpy.array([num_models], dtype=int)
    error_checking.assert_is_numpy_array(
        numpy.array(model_names), exact_dimensions=expected_dim
    )

    evaluation_tables = [None] * num_models
    best_threshold_indices = numpy.full(num_models, -1, dtype=int)

    for i in range(num_models):
        print('Reading data from: "{0:s}"...'.format(evaluation_file_names[i]))
        this_evaluation_dict = model_eval.read_evaluation(
            evaluation_file_names[i]
        )

        evaluation_tables[i] = this_evaluation_dict[
            model_eval.EVALUATION_TABLE_KEY]

        best_threshold_indices[i] = numpy.argmin(numpy.absolute(
            this_evaluation_dict[model_eval.BEST_THRESHOLD_KEY] -
            this_evaluation_dict[model_eval.ALL_THRESHOLDS_KEY]
        ))

    print('\n')

    _plot_roc_curves(
        evaluation_tables=evaluation_tables, model_names=model_names,
        best_threshold_indices=best_threshold_indices,
        output_file_name='{0:s}/roc_curves.jpg'.format(output_dir_name),
        confidence_level=confidence_level
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        evaluation_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        model_names=getattr(INPUT_ARG_OBJECT, MODEL_NAMES_ARG_NAME),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
