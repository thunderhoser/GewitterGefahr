"""Plots temporally subset model evaluation.

Scores may be plotted by month, by hour, or both.
"""

import os.path
import warnings
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from matplotlib import pyplot
from gewittergefahr.gg_utils import bootstrapping
from gewittergefahr.gg_utils import temporal_subsetting
from gewittergefahr.gg_utils import model_evaluation as model_eval
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import imagemagick_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

FONT_SIZE = 30
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

LINE_WIDTH = 4
HISTOGRAM_EDGE_WIDTH = 1.5
ERROR_BAR_WIDTH = 3
ERROR_CAP_LENGTH = 6

MARKER_TYPE = 'o'
MARKER_SIZE_SANS_BOOTSTRAP = 16
MARKER_SIZE_WITH_BOOTSTRAP = 10

AUC_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
CSI_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
POD_COLOUR = AUC_COLOUR
FAR_COLOUR = CSI_COLOUR

HISTOGRAM_FACE_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
HISTOGRAM_FACE_COLOUR = matplotlib.colors.to_rgba(HISTOGRAM_FACE_COLOUR, 0.5)
HISTOGRAM_EDGE_COLOUR = numpy.full(3, 0.)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

INPUT_DIR_ARG_NAME = 'input_evaluation_dir_name'
NUM_MONTHS_ARG_NAME = 'num_months_per_chunk'
NUM_HOURS_ARG_NAME = 'num_hours_per_chunk'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Evaluation files therein will be found by '
    '`model_evaluation.find_file` and read by '
    '`model_evaluation.read_evaluation`.')

NUM_MONTHS_HELP_STRING = (
    'Number of months per chunk.  Must be in the list below.\n{0:s}'
).format(str(temporal_subsetting.VALID_MONTH_COUNTS))

NUM_HOURS_HELP_STRING = (
    'Number of hours per chunk.  Must be in the list below.\n{0:s}'
).format(str(temporal_subsetting.VALID_HOUR_COUNTS))

CONFIDENCE_LEVEL_HELP_STRING = 'Confidence level for error bars.'

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_MONTHS_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_MONTHS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_HOURS_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_HOURS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=0.95,
    help=CONFIDENCE_LEVEL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _plot_auc_and_csi(auc_matrix, csi_matrix, num_examples_by_chunk,
                      num_bootstrap_reps, plot_legend):
    """Plots AUC and CSI either by hour or by month.

    N = number of monthly or hourly chunks

    :param auc_matrix: N-by-3 numpy array of AUC values.  The [i]th row contains
        [min, mean, max] for the [i]th chunk.
    :param csi_matrix: Same but for CSI.
    :param num_examples_by_chunk: length-N numpy array with number of examples
        for each chunk.
    :param num_bootstrap_reps: Number of bootstrap replicates.
    :param plot_legend: Boolean flag.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    # Housekeeping.
    figure_object, main_axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    histogram_axes_object = main_axes_object.twinx()
    main_axes_object.set_zorder(histogram_axes_object.get_zorder() + 1)
    main_axes_object.patch.set_visible(False)

    num_chunks = auc_matrix.shape[0]
    x_values = numpy.linspace(0, num_chunks - 1, num=num_chunks, dtype=float)

    if num_bootstrap_reps == 1:
        marker_size = MARKER_SIZE_SANS_BOOTSTRAP
    else:
        marker_size = MARKER_SIZE_WITH_BOOTSTRAP

    # Plot mean AUC.
    main_axes_object.plot(
        x_values, auc_matrix[:, 1], linestyle='None', marker=MARKER_TYPE,
        markersize=marker_size, markerfacecolor=AUC_COLOUR,
        markeredgecolor=AUC_COLOUR, markeredgewidth=0)

    # Plot confidence interval for AUC.
    if num_bootstrap_reps == 1:
        this_handle = main_axes_object.plot(
            x_values, auc_matrix[:, 1], color=AUC_COLOUR, linewidth=LINE_WIDTH
        )[0]
    else:
        negative_errors = auc_matrix[:, 1] - auc_matrix[:, 0]
        positive_errors = auc_matrix[:, 2] - auc_matrix[:, 1]
        error_matrix = numpy.vstack((negative_errors, positive_errors))

        this_handle = main_axes_object.errorbar(
            x_values, auc_matrix[:, 1], yerr=error_matrix, color=AUC_COLOUR,
            linewidth=LINE_WIDTH, elinewidth=ERROR_BAR_WIDTH,
            capsize=ERROR_CAP_LENGTH, capthick=ERROR_BAR_WIDTH
        )[0]

    legend_handles = [this_handle]
    legend_strings = ['AUC']

    # Plot mean CSI.
    main_axes_object.plot(
        x_values, csi_matrix[:, 1], linestyle='None', marker=MARKER_TYPE,
        markersize=marker_size, markerfacecolor=CSI_COLOUR,
        markeredgecolor=CSI_COLOUR, markeredgewidth=0)

    # Plot confidence interval for CSI.
    if num_bootstrap_reps == 1:
        this_handle = main_axes_object.plot(
            x_values, csi_matrix[:, 1], color=CSI_COLOUR, linewidth=LINE_WIDTH
        )[0]
    else:
        negative_errors = csi_matrix[:, 1] - csi_matrix[:, 0]
        positive_errors = csi_matrix[:, 2] - csi_matrix[:, 1]
        error_matrix = numpy.vstack((negative_errors, positive_errors))

        this_handle = main_axes_object.errorbar(
            x_values, csi_matrix[:, 1], yerr=error_matrix, color=CSI_COLOUR,
            linewidth=LINE_WIDTH, elinewidth=ERROR_BAR_WIDTH,
            capsize=ERROR_CAP_LENGTH, capthick=ERROR_BAR_WIDTH
        )[0]

    legend_handles.append(this_handle)
    legend_strings.append('CSI')

    main_axes_object.set_ylabel('AUC or CSI')
    main_axes_object.set_xlim([
        numpy.min(x_values) - 0.5, numpy.max(x_values) + 0.5
    ])

    # Plot histogram of example counts.
    y_values = numpy.maximum(numpy.log10(num_examples_by_chunk), 0.)

    this_handle = histogram_axes_object.bar(
        x=x_values, height=y_values, width=1., color=HISTOGRAM_FACE_COLOUR,
        edgecolor=HISTOGRAM_EDGE_COLOUR, linewidth=HISTOGRAM_EDGE_WIDTH
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append(r'Number of examples (log$_{10}$)')
    histogram_axes_object.set_ylabel(r'Number of examples (log$_{10}$)')

    if plot_legend:
        main_axes_object.legend(
            legend_handles, legend_strings, loc='lower center',
            bbox_to_anchor=(0.5, 1), fancybox=True, shadow=True,
            ncol=len(legend_handles)
        )

    return figure_object, main_axes_object


def _plot_pod_and_far(pod_matrix, far_matrix, num_positive_ex_by_chunk,
                      num_bootstrap_reps, plot_legend):
    """Plots POD and FAR either by hour or by month.

    N = number of monthly or hourly chunks

    :param pod_matrix: N-by-3 numpy array of POD values.  The [i]th row contains
        [min, mean, max] for the [i]th chunk.
    :param far_matrix: Same but for FAR.
    :param num_positive_ex_by_chunk: length-N numpy array with number of
        positive examples for each chunk.
    :param num_bootstrap_reps: Number of bootstrap replicates.
    :param plot_legend: Boolean flag.
    :return: figure_object: See doc for `_plot_auc_and_csi`.
    :return: axes_object: Same.
    """

    # Housekeeping.
    figure_object, main_axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    histogram_axes_object = main_axes_object.twinx()
    main_axes_object.set_zorder(histogram_axes_object.get_zorder() + 1)
    main_axes_object.patch.set_visible(False)

    num_chunks = pod_matrix.shape[0]
    x_values = numpy.linspace(0, num_chunks - 1, num=num_chunks, dtype=float)

    if num_bootstrap_reps == 1:
        marker_size = MARKER_SIZE_SANS_BOOTSTRAP
    else:
        marker_size = MARKER_SIZE_WITH_BOOTSTRAP

    # Plot mean POD.
    main_axes_object.plot(
        x_values, pod_matrix[:, 1], linestyle='None', marker=MARKER_TYPE,
        markersize=marker_size, markerfacecolor=POD_COLOUR,
        markeredgecolor=POD_COLOUR, markeredgewidth=0)

    # Plot confidence interval for POD.
    if num_bootstrap_reps == 1:
        this_handle = main_axes_object.plot(
            x_values, pod_matrix[:, 1], color=POD_COLOUR, linewidth=LINE_WIDTH
        )[0]
    else:
        negative_errors = pod_matrix[:, 1] - pod_matrix[:, 0]
        positive_errors = pod_matrix[:, 2] - pod_matrix[:, 1]
        error_matrix = numpy.vstack((negative_errors, positive_errors))

        this_handle = main_axes_object.errorbar(
            x_values, pod_matrix[:, 1], yerr=error_matrix, color=POD_COLOUR,
            linewidth=LINE_WIDTH, elinewidth=ERROR_BAR_WIDTH,
            capsize=ERROR_CAP_LENGTH, capthick=ERROR_BAR_WIDTH
        )[0]

    legend_handles = [this_handle]
    legend_strings = ['POD']

    # Plot mean FAR.
    main_axes_object.plot(
        x_values, far_matrix[:, 1], linestyle='None', marker=MARKER_TYPE,
        markersize=marker_size, markerfacecolor=FAR_COLOUR,
        markeredgecolor=FAR_COLOUR, markeredgewidth=0)

    # Plot confidence interval for FAR.
    if num_bootstrap_reps == 1:
        this_handle = main_axes_object.plot(
            x_values, far_matrix[:, 1], color=FAR_COLOUR, linewidth=LINE_WIDTH
        )[0]
    else:
        negative_errors = far_matrix[:, 1] - far_matrix[:, 0]
        positive_errors = far_matrix[:, 2] - far_matrix[:, 1]
        error_matrix = numpy.vstack((negative_errors, positive_errors))

        this_handle = main_axes_object.errorbar(
            x_values, far_matrix[:, 1], yerr=error_matrix, color=FAR_COLOUR,
            linewidth=LINE_WIDTH, elinewidth=ERROR_BAR_WIDTH,
            capsize=ERROR_CAP_LENGTH, capthick=ERROR_BAR_WIDTH
        )[0]

    legend_handles.append(this_handle)
    legend_strings.append('FAR')

    main_axes_object.set_ylabel('POD or FAR')
    main_axes_object.set_xlim([
        numpy.min(x_values) - 0.5, numpy.max(x_values) + 0.5
    ])

    # Plot histogram of positive-example counts.
    this_handle = histogram_axes_object.bar(
        x=x_values, height=num_positive_ex_by_chunk, width=1.,
        color=HISTOGRAM_FACE_COLOUR, edgecolor=HISTOGRAM_EDGE_COLOUR,
        linewidth=HISTOGRAM_EDGE_WIDTH
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append('Number of positive examples')
    histogram_axes_object.set_ylabel('Number of positive examples')

    if plot_legend:
        main_axes_object.legend(
            legend_handles, legend_strings, loc='lower center',
            bbox_to_anchor=(0.5, 1), fancybox=True, shadow=True,
            ncol=len(legend_handles)
        )

    return figure_object, main_axes_object


def _plot_by_month(evaluation_dir_name, num_months_per_chunk,
                   confidence_level, output_dir_name):
    """Plots model evaluation by month.

    :param evaluation_dir_name: See documentation at top of file.
    :param num_months_per_chunk: Same.
    :param confidence_level: Same.
    :param output_dir_name: Same.
    :return: output_file_names: Paths to figures saved by this method.
    """

    chunk_to_months_dict = temporal_subsetting.get_monthly_chunks(
        num_months_per_chunk=num_months_per_chunk, verbose=False)

    num_bootstrap_reps = None
    num_chunks = len(chunk_to_months_dict.keys())

    auc_matrix = numpy.full((num_chunks, 3), numpy.nan)
    pod_matrix = numpy.full((num_chunks, 3), numpy.nan)
    far_matrix = numpy.full((num_chunks, 3), numpy.nan)
    csi_matrix = numpy.full((num_chunks, 3), numpy.nan)
    num_examples_by_chunk = numpy.full(num_chunks, 0, dtype=int)
    num_positive_ex_by_chunk = numpy.full(num_chunks, 0, dtype=int)

    for i in range(num_chunks):
        this_eval_file_name = model_eval.find_file(
            directory_name=evaluation_dir_name,
            months_in_subset=chunk_to_months_dict[i],
            raise_error_if_missing=False)

        if not os.path.isfile(this_eval_file_name):
            warning_string = (
                'Cannot find file (this may or may not be a problem).  Expected'
                ' at: "{0:s}"'
            ).format(this_eval_file_name)

            warnings.warn(warning_string)
            continue

        print('Reading data from: "{0:s}"...'.format(this_eval_file_name))
        this_evaluation_dict = model_eval.read_evaluation(this_eval_file_name)

        num_examples_by_chunk[i] = len(
            this_evaluation_dict[model_eval.OBSERVED_LABELS_KEY]
        )
        num_positive_ex_by_chunk[i] = numpy.sum(
            this_evaluation_dict[model_eval.OBSERVED_LABELS_KEY]
        )

        this_evaluation_table = this_evaluation_dict[
            model_eval.EVALUATION_TABLE_KEY]
        this_num_bootstrap_reps = len(this_evaluation_table.index)

        if num_bootstrap_reps is None:
            num_bootstrap_reps = this_num_bootstrap_reps
        assert num_bootstrap_reps == this_num_bootstrap_reps

        these_auc = this_evaluation_table[model_eval.AUC_KEY].values
        these_pod = this_evaluation_table[model_eval.POD_KEY].values
        these_far = (
            1. - this_evaluation_table[model_eval.SUCCESS_RATIO_KEY].values
        )
        these_csi = this_evaluation_table[model_eval.CSI_KEY].values

        auc_matrix[i, 1] = numpy.nanmean(these_auc)
        pod_matrix[i, 1] = numpy.nanmean(these_pod)
        far_matrix[i, 1] = numpy.nanmean(these_far)
        csi_matrix[i, 1] = numpy.nanmean(these_csi)

        auc_matrix[i, 0], auc_matrix[i, 2] = (
            bootstrapping.get_confidence_interval(
                stat_values=these_auc, confidence_level=confidence_level)
        )
        pod_matrix[i, 0], pod_matrix[i, 2] = (
            bootstrapping.get_confidence_interval(
                stat_values=these_pod, confidence_level=confidence_level)
        )
        far_matrix[i, 0], far_matrix[i, 2] = (
            bootstrapping.get_confidence_interval(
                stat_values=these_far, confidence_level=confidence_level)
        )
        csi_matrix[i, 0], csi_matrix[i, 2] = (
            bootstrapping.get_confidence_interval(
                stat_values=these_csi, confidence_level=confidence_level)
        )

    x_tick_values = numpy.linspace(
        0, num_chunks - 1, num=num_chunks, dtype=float)

    # TODO(thunderhoser): This code is hacky.
    if num_months_per_chunk == 1:
        x_tick_labels = [
            'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
            'Oct', 'Nov', 'Dec'
        ]
    elif num_months_per_chunk == 3:
        x_tick_labels = ['Winter', 'Spring', 'Summer', 'Fall']
    else:
        x_tick_labels = None

    figure_object, axes_object = _plot_auc_and_csi(
        auc_matrix=auc_matrix, csi_matrix=csi_matrix,
        num_examples_by_chunk=num_examples_by_chunk,
        num_bootstrap_reps=num_bootstrap_reps, plot_legend=True)

    axes_object.set_xticks(x_tick_values)
    axes_object.set_xticklabels(x_tick_labels, rotation=90.)

    auc_csi_file_name = '{0:s}/monthly_auc_and_csi.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(auc_csi_file_name))

    figure_object.savefig(
        auc_csi_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object, axes_object = _plot_pod_and_far(
        pod_matrix=pod_matrix, far_matrix=far_matrix,
        num_positive_ex_by_chunk=num_positive_ex_by_chunk,
        num_bootstrap_reps=num_bootstrap_reps, plot_legend=True)

    axes_object.set_xticks(x_tick_values)
    axes_object.set_xticklabels(x_tick_labels, rotation=90.)

    pod_far_file_name = '{0:s}/monthly_pod_and_far.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(pod_far_file_name))

    figure_object.savefig(
        pod_far_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    return [auc_csi_file_name, pod_far_file_name]


def _plot_by_hour(evaluation_dir_name, num_hours_per_chunk,
                  confidence_level, output_dir_name):
    """Plots model evaluation by hour.

    :param evaluation_dir_name: See documentation at top of file.
    :param num_hours_per_chunk: Same.
    :param confidence_level: Same.
    :param output_dir_name: Same.
    :return: output_file_names: Paths to figures saved by this method.
    """

    chunk_to_hours_dict = temporal_subsetting.get_hourly_chunks(
        num_hours_per_chunk=num_hours_per_chunk, verbose=False)

    num_bootstrap_reps = None
    num_chunks = len(chunk_to_hours_dict.keys())

    auc_matrix = numpy.full((num_chunks, 3), numpy.nan)
    pod_matrix = numpy.full((num_chunks, 3), numpy.nan)
    far_matrix = numpy.full((num_chunks, 3), numpy.nan)
    csi_matrix = numpy.full((num_chunks, 3), numpy.nan)
    num_examples_by_chunk = numpy.full(num_chunks, 0, dtype=int)
    num_positive_ex_by_chunk = numpy.full(num_chunks, 0, dtype=int)

    for i in range(num_chunks):
        this_eval_file_name = model_eval.find_file(
            directory_name=evaluation_dir_name,
            hours_in_subset=chunk_to_hours_dict[i],
            raise_error_if_missing=False)

        if not os.path.isfile(this_eval_file_name):
            warning_string = (
                'Cannot find file (this may or may not be a problem).  Expected'
                ' at: "{0:s}"'
            ).format(this_eval_file_name)

            warnings.warn(warning_string)
            continue

        print('Reading data from: "{0:s}"...'.format(this_eval_file_name))
        this_evaluation_dict = model_eval.read_evaluation(this_eval_file_name)

        num_examples_by_chunk[i] = len(
            this_evaluation_dict[model_eval.OBSERVED_LABELS_KEY]
        )
        num_positive_ex_by_chunk[i] = numpy.sum(
            this_evaluation_dict[model_eval.OBSERVED_LABELS_KEY]
        )

        this_evaluation_table = this_evaluation_dict[
            model_eval.EVALUATION_TABLE_KEY]
        this_num_bootstrap_reps = len(this_evaluation_table.index)

        if num_bootstrap_reps is None:
            num_bootstrap_reps = this_num_bootstrap_reps
        assert num_bootstrap_reps == this_num_bootstrap_reps

        these_auc = this_evaluation_table[model_eval.AUC_KEY].values
        these_pod = this_evaluation_table[model_eval.POD_KEY].values
        these_far = (
            1. - this_evaluation_table[model_eval.SUCCESS_RATIO_KEY].values
        )
        these_csi = this_evaluation_table[model_eval.CSI_KEY].values

        auc_matrix[i, 1] = numpy.nanmean(these_auc)
        pod_matrix[i, 1] = numpy.nanmean(these_pod)
        far_matrix[i, 1] = numpy.nanmean(these_far)
        csi_matrix[i, 1] = numpy.nanmean(these_csi)

        auc_matrix[i, 0], auc_matrix[i, 2] = (
            bootstrapping.get_confidence_interval(
                stat_values=these_auc, confidence_level=confidence_level)
        )
        pod_matrix[i, 0], pod_matrix[i, 2] = (
            bootstrapping.get_confidence_interval(
                stat_values=these_pod, confidence_level=confidence_level)
        )
        far_matrix[i, 0], far_matrix[i, 2] = (
            bootstrapping.get_confidence_interval(
                stat_values=these_far, confidence_level=confidence_level)
        )
        csi_matrix[i, 0], csi_matrix[i, 2] = (
            bootstrapping.get_confidence_interval(
                stat_values=these_csi, confidence_level=confidence_level)
        )

    x_tick_labels = [None] * num_chunks
    x_tick_values = numpy.linspace(
        0, num_chunks - 1, num=num_chunks, dtype=float)

    for i in range(num_chunks):
        these_hours = chunk_to_hours_dict[i]

        if len(these_hours) == 1:
            x_tick_labels[i] = '{0:02d}'.format(these_hours[0])
        else:
            x_tick_labels[i] = '{0:02d}-{1:02d}'.format(
                numpy.min(these_hours), numpy.max(these_hours)
            )

    figure_object, axes_object = _plot_auc_and_csi(
        auc_matrix=auc_matrix, csi_matrix=csi_matrix,
        num_examples_by_chunk=num_examples_by_chunk,
        num_bootstrap_reps=num_bootstrap_reps, plot_legend=False)

    axes_object.set_xticks(x_tick_values)
    axes_object.set_xticklabels(x_tick_labels, rotation=90.)
    axes_object.set_xlabel('Hour')

    auc_csi_file_name = '{0:s}/hourly_auc_and_csi.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(auc_csi_file_name))

    figure_object.savefig(
        auc_csi_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object, axes_object = _plot_pod_and_far(
        pod_matrix=pod_matrix, far_matrix=far_matrix,
        num_positive_ex_by_chunk=num_positive_ex_by_chunk,
        num_bootstrap_reps=num_bootstrap_reps, plot_legend=False)

    axes_object.set_xticks(x_tick_values)
    axes_object.set_xticklabels(x_tick_labels, rotation=90.)
    axes_object.set_xlabel('Hour')

    pod_far_file_name = '{0:s}/hourly_pod_and_far.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(pod_far_file_name))

    figure_object.savefig(
        pod_far_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    return [auc_csi_file_name, pod_far_file_name]


def _run(evaluation_dir_name, num_months_per_chunk, num_hours_per_chunk,
         confidence_level, output_dir_name):
    """Plots temporally subset model evaluation.

    This is effectively the main method.

    :param evaluation_dir_name: See documentation at top of file.
    :param num_months_per_chunk: Same.
    :param num_hours_per_chunk: Same.
    :param confidence_level: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    panel_file_names = _plot_by_month(
        evaluation_dir_name=evaluation_dir_name,
        num_months_per_chunk=num_months_per_chunk,
        confidence_level=confidence_level, output_dir_name=output_dir_name)

    panel_file_names += _plot_by_hour(
        evaluation_dir_name=evaluation_dir_name,
        num_hours_per_chunk=num_hours_per_chunk,
        confidence_level=confidence_level, output_dir_name=output_dir_name)

    concat_file_name = '{0:s}/temporally_subset_eval.jpg'.format(
        output_dir_name)

    print('Concatenating panels to: "{0:s}"...'.format(concat_file_name))
    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names, output_file_name=concat_file_name,
        num_panel_rows=2, num_panel_columns=2)

    imagemagick_utils.resize_image(
        input_file_name=concat_file_name, output_file_name=concat_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        evaluation_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        num_months_per_chunk=getattr(INPUT_ARG_OBJECT, NUM_MONTHS_ARG_NAME),
        num_hours_per_chunk=getattr(INPUT_ARG_OBJECT, NUM_HOURS_ARG_NAME),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
