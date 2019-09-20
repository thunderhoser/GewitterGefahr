"""Plots results of hyperparameter experiment with 3-D GridRad data."""

import os.path
import warnings
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from generalexam.machine_learning import evaluation_utils
from gewittergefahr.gg_utils import model_evaluation as model_eval
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils

FIGURE_RESOLUTION_DPI = 300

DROPOUT_RATES = numpy.linspace(0.25, 0.75, num=5)
L2_WEIGHTS = numpy.logspace(-3, -1, num=5)
DENSE_LAYER_COUNTS = numpy.array([1, 2], dtype=int)
DATA_AUGMENTATION_FLAGS = numpy.array([0, 1], dtype=bool)

INPUT_DIR_ARG_NAME = 'input_dir_name'
MAIN_CMAP_ARG_NAME = 'main_colour_map_name'
MAX_PERCENTILE_ARG_NAME = 'max_colour_percentile'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MAIN_CMAP_HELP_STRING = (
    'Name of main colour map (for all scores except frequency bias).  Must be '
    'accepted by `pyplot.get_cmap`.')

MAX_PERCENTILE_HELP_STRING = (
    'Used to determine min and max values in each colour map.  Max value will '
    'be [q]th percentile over all grid cells, and min value will be [100 - q]th'
    ' percentile, where q = `{0:s}`.'
).format(MAX_PERCENTILE_ARG_NAME)

INPUT_DIR_HELP_STRING = (
    'Name of top-level input directory (with one subdirectory per model).')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAIN_CMAP_ARG_NAME, type=str, required=False, default='plasma',
    help=MAIN_CMAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False, default=99.,
    help=MAX_PERCENTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(top_input_dir_name, main_colour_map_name, max_colour_percentile,
         output_dir_name):
    """Plots results of hyperparameter experiment with 3-D GridRad data.

    This is effectively the main method.

    :param top_input_dir_name: See documentation at top of file.
    :param main_colour_map_name: Same.
    :param max_colour_percentile: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    main_colour_map_object = pyplot.get_cmap(main_colour_map_name)
    error_checking.assert_is_geq(max_colour_percentile, 90.)
    error_checking.assert_is_leq(max_colour_percentile, 100.)

    num_dropout_rates = len(DROPOUT_RATES)
    num_l2_weights = len(L2_WEIGHTS)
    num_dense_layer_counts = len(DENSE_LAYER_COUNTS)
    num_data_aug_flags = len(DATA_AUGMENTATION_FLAGS)

    dimensions = (
        num_dropout_rates, num_l2_weights, num_dense_layer_counts,
        num_data_aug_flags
    )

    auc_matrix = numpy.full(dimensions, numpy.nan)
    csi_matrix = numpy.full(dimensions, numpy.nan)
    pod_matrix = numpy.full(dimensions, numpy.nan)
    far_matrix = numpy.full(dimensions, numpy.nan)
    frequency_bias_matrix = numpy.full(dimensions, numpy.nan)

    for i in range(num_dropout_rates):
        for j in range(num_l2_weights):
            for k in range(num_dense_layer_counts):
                for m in range(num_data_aug_flags):
                    this_eval_file_name = (
                        '{0:s}/dropout={1:.3f}_l2={2:.6f}_'
                        'num-dense-layers={3:d}_data-aug={4:d}/validation/'
                        'evaluation_results.p'
                    ).format(
                        top_input_dir_name, DROPOUT_RATES[i], L2_WEIGHTS[j],
                        DENSE_LAYER_COUNTS[k], int(DATA_AUGMENTATION_FLAGS[m])
                    )

                    if not os.path.isfile(this_eval_file_name):
                        warning_string = (
                            'Cannot find file (this may or may not be a '
                            'PROBLEM).  Expected at: "{0:s}"'
                        ).format(this_eval_file_name)

                        warnings.warn(warning_string)
                        continue

                    print('Reading data from: "{0:s}"...'.format(
                        this_eval_file_name
                    ))

                    this_evaluation_table = model_eval.read_evaluation(
                        this_eval_file_name
                    )[model_eval.EVALUATION_TABLE_KEY]

                    auc_matrix[i, j, k, m] = numpy.nanmean(
                        this_evaluation_table[model_eval.AUC_KEY].values
                    )
                    csi_matrix[i, j, k, m] = numpy.nanmean(
                        this_evaluation_table[model_eval.CSI_KEY].values
                    )
                    pod_matrix[i, j, k, m] = numpy.nanmean(
                        this_evaluation_table[model_eval.POD_KEY].values
                    )
                    far_matrix[i, j, k, m] = 1. - numpy.nanmean(
                        this_evaluation_table[
                            model_eval.SUCCESS_RATIO_KEY].values
                    )
                    frequency_bias_matrix[i, j, k, m] = numpy.nanmean(
                        this_evaluation_table[
                            model_eval.FREQUENCY_BIAS_KEY].values
                    )

    figure_object, axes_object_matrix = plotting_utils.create_paneled_figure(
        num_rows=num_dense_layer_counts, num_columns=num_data_aug_flags,
        shared_x_axis=False, shared_y_axis=False, keep_aspect_ratio=True)

    max_colour_value = numpy.nanpercentile(auc_matrix, max_colour_percentile)
    min_colour_value = numpy.nanpercentile(
        auc_matrix, 100. - max_colour_percentile)

    x_tick_labels = ['{0:.1f}'.format(w) for w in numpy.log10(L2_WEIGHTS)]
    dummy_x_tick_labels = ['' for l in x_tick_labels]

    y_tick_labels = ['{0:.3f}'.format(d) for d in DROPOUT_RATES]
    dummy_y_tick_labels = ['' for l in y_tick_labels]

    x_axis_label = r'L$_2$ weight (log$_{10}$)'
    y_axis_label = 'Dropout rate'

    for k in range(num_dense_layer_counts):
        for m in range(num_data_aug_flags):
            these_x_tick_labels = (
                x_tick_labels if k == num_dense_layer_counts - 1
                else dummy_x_tick_labels
            )
            these_y_tick_labels = (
                y_tick_labels if m == 0 else dummy_y_tick_labels
            )

            evaluation_utils.plot_scores_2d(
                score_matrix=auc_matrix[..., k, m],
                min_colour_value=min_colour_value,
                max_colour_value=max_colour_value,
                x_tick_label_strings=these_x_tick_labels,
                y_tick_label_strings=these_y_tick_labels,
                colour_map_object=main_colour_map_object,
                axes_object=axes_object_matrix[k, m]
            )

            if these_x_tick_labels[0] != '':
                axes_object_matrix[k, m].set_xlabel(x_axis_label)
            if these_y_tick_labels[0] != '':
                axes_object_matrix[k, m].set_ylabel(y_axis_label)

            this_title_string = '{0:d} dense layers, DA {1:s}'.format(
                DENSE_LAYER_COUNTS[k],
                'on' if DATA_AUGMENTATION_FLAGS[m] else 'off'
            )

            axes_object_matrix[k, m].set_title(this_title_string)

            if m != num_data_aug_flags - 1:
                continue

            plotting_utils.plot_linear_colour_bar(
                axes_object_or_matrix=axes_object_matrix[k, m],
                data_matrix=auc_matrix[..., k, m],
                colour_map_object=main_colour_map_object,
                min_value=min_colour_value, max_value=max_colour_value,
                orientation_string='vertical', fraction_of_axis_length=0.85,
                extend_min=True, extend_max=True)

    output_file_name = '{0:s}/auc.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))

    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight')
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        main_colour_map_name=getattr(INPUT_ARG_OBJECT, MAIN_CMAP_ARG_NAME),
        max_colour_percentile=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
