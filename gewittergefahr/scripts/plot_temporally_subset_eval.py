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
from gewittergefahr.gg_utils import model_evaluation as model_eval
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.scripts import subset_predictions_by_time as subsetting

FONT_SIZE = 30
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

LINE_WIDTH = 3
HISTOGRAM_EDGE_WIDTH = 1.5
MARKER_TYPE = 'o'
MARKER_SIZE = 14

AUC_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
POD_COLOUR = AUC_COLOUR
FAR_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
CSI_COLOUR = FAR_COLOUR

HISTOGRAM_FACE_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
HISTOGRAM_FACE_COLOUR = matplotlib.colors.to_rgba(HISTOGRAM_FACE_COLOUR, 0.5)
HISTOGRAM_EDGE_COLOUR = numpy.full(3, 0.)

AUC_LINE_STYLE = 'solid'
POD_LINE_STYLE = 'dashed'
FAR_LINE_STYLE = 'dashed'
CSI_LINE_STYLE = 'solid'

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_DIR_ARG_NAME = 'input_evaluation_dir_name'
NUM_MONTHS_ARG_NAME = 'num_months_per_chunk'
NUM_HOURS_ARG_NAME = 'num_hours_per_chunk'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

# TODO(thunderhoser): Fix this.
INPUT_DIR_HELP_STRING = (
    'Name of top-level input directory.  Evaluation files will be found therein'
    ' by FOO and read by `model_evaluation.read_binary_classifn_results`.')

# TODO(thunderhoser): Allow negative option in subset_predictions_by_time.py.
NUM_MONTHS_HELP_STRING = (
    'Number of months in each chunk.  Must be in the list below.  Or, if you do'
    ' not want to plot by month, make this negative.\n{0:s}'
).format(str(subsetting.VALID_MONTH_COUNTS))

NUM_HOURS_HELP_STRING = (
    'Number of hours in each chunk.  Must be in the list below.  Or, if you do'
    ' not want to plot by hour, make this negative.\n{0:s}'
).format(str(subsetting.VALID_HOUR_COUNTS))

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
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _plot_scores(auc_by_chunk, pod_by_chunk, far_by_chunk, csi_by_chunk,
                 num_examples_by_chunk):
    """Plots scores for either monthly or hourly chunks.

    N = number of chunks

    :param auc_by_chunk: length-N numpy array of AUC values.
    :param pod_by_chunk: length-N numpy array of POD values.
    :param far_by_chunk: length-N numpy array of FAR values.
    :param csi_by_chunk: length-N numpy array of CSI values.
    :param num_examples_by_chunk: length-N numpy array of example counts.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: main_axes_object: Handle for main axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :return: histogram_axes_object: Handle for histogram axes.
    """

    legend_handles = []
    legend_strings = []

    figure_object, main_axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    histogram_axes_object = main_axes_object.twinx()
    main_axes_object.set_zorder(histogram_axes_object.get_zorder() + 1)
    main_axes_object.patch.set_visible(False)

    num_chunks = len(auc_by_chunk)
    x_values = numpy.linspace(0, num_chunks - 1, num=num_chunks, dtype=float)
    real_indices = numpy.where(
        numpy.invert(numpy.isnan(auc_by_chunk))
    )[0]

    main_axes_object.plot(
        x_values[real_indices], auc_by_chunk[real_indices], linestyle='None',
        marker=MARKER_TYPE, markersize=MARKER_SIZE,
        markerfacecolor=AUC_COLOUR, markeredgecolor=AUC_COLOUR,
        markeredgewidth=0)

    this_handle = main_axes_object.plot(
        x_values[real_indices], auc_by_chunk[real_indices], color=AUC_COLOUR,
        linestyle=AUC_LINE_STYLE, linewidth=LINE_WIDTH
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append('AUC')

    main_axes_object.plot(
        x_values[real_indices], pod_by_chunk[real_indices], linestyle='None',
        marker=MARKER_TYPE, markersize=MARKER_SIZE,
        markerfacecolor=POD_COLOUR, markeredgecolor=POD_COLOUR,
        markeredgewidth=0)

    this_handle = main_axes_object.plot(
        x_values[real_indices], pod_by_chunk[real_indices], color=POD_COLOUR,
        linestyle=POD_LINE_STYLE, linewidth=LINE_WIDTH
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append('POD')

    main_axes_object.plot(
        x_values[real_indices], far_by_chunk[real_indices], linestyle='None',
        marker=MARKER_TYPE, markersize=MARKER_SIZE,
        markerfacecolor=FAR_COLOUR, markeredgecolor=FAR_COLOUR,
        markeredgewidth=0)

    this_handle = main_axes_object.plot(
        x_values[real_indices], far_by_chunk[real_indices], color=FAR_COLOUR,
        linestyle=FAR_LINE_STYLE, linewidth=LINE_WIDTH
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append('FAR')

    main_axes_object.plot(
        x_values[real_indices], csi_by_chunk[real_indices], linestyle='None',
        marker=MARKER_TYPE, markersize=MARKER_SIZE,
        markerfacecolor=CSI_COLOUR, markeredgecolor=CSI_COLOUR,
        markeredgewidth=0)

    this_handle = main_axes_object.plot(
        x_values[real_indices], csi_by_chunk[real_indices], color=CSI_COLOUR,
        linestyle=CSI_LINE_STYLE, linewidth=LINE_WIDTH
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append('CSI')

    main_axes_object.legend(
        legend_handles, legend_strings, loc='lower center',
        bbox_to_anchor=(0.5, 1), fancybox=True, shadow=True,
        ncol=len(legend_handles)
    )

    main_axes_object.set_ylabel('Score')
    main_axes_object.set_xlim([
        numpy.min(x_values) - 0.5, numpy.max(x_values) + 0.5
    ])

    y_values = numpy.maximum(numpy.log10(num_examples_by_chunk), 0.)
    histogram_axes_object.bar(
        x=x_values, height=y_values, width=1.,
        color=HISTOGRAM_FACE_COLOUR, edgecolor=HISTOGRAM_EDGE_COLOUR,
        linewidth=HISTOGRAM_EDGE_WIDTH)

    histogram_axes_object.set_ylabel(r'Number of examples (log$_{10}$)')

    return figure_object, main_axes_object, histogram_axes_object


def _plot_by_month(top_evaluation_dir_name, num_months_per_chunk,
                   output_dir_name):
    """Plots model evaluation by month.

    :param top_evaluation_dir_name: See documentation at top of file.
    :param num_months_per_chunk: Same.
    :param output_dir_name: Same.
    """

    chunk_to_months_dict = subsetting._get_months_in_each_chunk(
        num_months_per_chunk)
    num_chunks = len(chunk_to_months_dict.keys())

    auc_by_chunk = numpy.full(num_chunks, numpy.nan)
    pod_by_chunk = numpy.full(num_chunks, numpy.nan)
    far_by_chunk = numpy.full(num_chunks, numpy.nan)
    csi_by_chunk = numpy.full(num_chunks, numpy.nan)
    num_examples_by_chunk = numpy.full(num_chunks, 0, dtype=int)

    for i in range(num_chunks):
        this_subdir_name = '-'.join([
            '{0:02d}'.format(m) for m in chunk_to_months_dict[i]
        ])

        this_eval_file_name = '{0:s}/months={1:s}/evaluation_results.p'.format(
            top_evaluation_dir_name, this_subdir_name)

        if not os.path.isfile(this_eval_file_name):
            warning_string = (
                'Cannot find file (this may or may not be a problem).  Expected'
                ' at: "{0:s}"'
            ).format(this_eval_file_name)

            warnings.warn(warning_string)
            continue

        print('Reading data from: "{0:s}"...'.format(this_eval_file_name))
        this_evaluation_dict = model_eval.read_binary_classifn_results(
            this_eval_file_name)

        num_examples_by_chunk[i] = len(
            this_evaluation_dict[model_eval.FORECAST_PROBABILITIES_KEY]
        )

        this_evaluation_table = this_evaluation_dict[
            model_eval.EVALUATION_TABLE_KEY]

        auc_by_chunk[i] = numpy.mean(
            this_evaluation_table[model_eval.AUC_KEY].values
        )
        pod_by_chunk[i] = numpy.mean(
            this_evaluation_table[model_eval.POD_KEY].values
        )
        far_by_chunk[i] = 1. - numpy.mean(
            this_evaluation_table[model_eval.SUCCESS_RATIO_KEY].values
        )
        csi_by_chunk[i] = numpy.mean(
            this_evaluation_table[model_eval.CSI_KEY]
        )

    figure_object, axes_object = _plot_scores(
        auc_by_chunk=auc_by_chunk, pod_by_chunk=pod_by_chunk,
        far_by_chunk=far_by_chunk, csi_by_chunk=csi_by_chunk,
        num_examples_by_chunk=num_examples_by_chunk
    )[:-1]

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

    x_tick_values = numpy.linspace(
        0, num_chunks - 1, num=num_chunks, dtype=float)

    axes_object.set_xticks(x_tick_values)
    axes_object.set_xticklabels(x_tick_labels, rotation=90.)

    output_file_name = '{0:s}/scores_by_month.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))

    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _plot_by_hour(top_evaluation_dir_name, num_hours_per_chunk,
                  output_dir_name):
    """Plots model evaluation by hour.

    :param top_evaluation_dir_name: See documentation at top of file.
    :param num_hours_per_chunk: Same.
    :param output_dir_name: Same.
    """

    chunk_to_hours_dict = subsetting._get_hours_in_each_chunk(
        num_hours_per_chunk)
    num_chunks = len(chunk_to_hours_dict.keys())

    auc_by_chunk = numpy.full(num_chunks, numpy.nan)
    pod_by_chunk = numpy.full(num_chunks, numpy.nan)
    far_by_chunk = numpy.full(num_chunks, numpy.nan)
    csi_by_chunk = numpy.full(num_chunks, numpy.nan)
    num_examples_by_chunk = numpy.full(num_chunks, 0, dtype=int)

    for i in range(num_chunks):
        this_subdir_name = '-'.join([
            '{0:02d}'.format(h) for h in chunk_to_hours_dict[i]
        ])

        this_eval_file_name = '{0:s}/hours={1:s}/evaluation_results.p'.format(
            top_evaluation_dir_name, this_subdir_name)

        if not os.path.isfile(this_eval_file_name):
            warning_string = (
                'Cannot find file (this may or may not be a problem).  Expected'
                ' at: "{0:s}"'
            ).format(this_eval_file_name)

            warnings.warn(warning_string)
            continue

        print('Reading data from: "{0:s}"...'.format(this_eval_file_name))
        this_evaluation_dict = model_eval.read_binary_classifn_results(
            this_eval_file_name)

        num_examples_by_chunk[i] = len(
            this_evaluation_dict[model_eval.FORECAST_PROBABILITIES_KEY]
        )

        this_evaluation_table = this_evaluation_dict[
            model_eval.EVALUATION_TABLE_KEY]

        auc_by_chunk[i] = numpy.mean(
            this_evaluation_table[model_eval.AUC_KEY].values
        )
        pod_by_chunk[i] = numpy.mean(
            this_evaluation_table[model_eval.POD_KEY].values
        )
        far_by_chunk[i] = 1. - numpy.mean(
            this_evaluation_table[model_eval.SUCCESS_RATIO_KEY].values
        )
        csi_by_chunk[i] = numpy.mean(
            this_evaluation_table[model_eval.CSI_KEY]
        )

    figure_object, axes_object = _plot_scores(
        auc_by_chunk=auc_by_chunk, pod_by_chunk=pod_by_chunk,
        far_by_chunk=far_by_chunk, csi_by_chunk=csi_by_chunk,
        num_examples_by_chunk=num_examples_by_chunk
    )[:-1]

    x_tick_labels = [None] * num_chunks

    for i in range(num_chunks):
        these_hours = chunk_to_hours_dict[i]

        if len(these_hours) == 1:
            x_tick_labels[i] = '{0:02d}'.format(these_hours[0])
        else:
            x_tick_labels[i] = '{0:02d}-{1:02d}'.format(
                numpy.min(these_hours), numpy.max(these_hours)
            )

    x_tick_values = numpy.linspace(
        0, num_chunks - 1, num=num_chunks, dtype=float)

    axes_object.set_xticks(x_tick_values)
    axes_object.set_xticklabels(x_tick_labels, rotation=90.)
    axes_object.set_xlabel('Hour')

    output_file_name = '{0:s}/scores_by_hour.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))

    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(top_evaluation_dir_name, num_months_per_chunk, num_hours_per_chunk,
         output_dir_name):
    """Plots temporally subset model evaluation.

    This is effectively the main method.

    :param top_evaluation_dir_name: See documentation at top of file.
    :param num_months_per_chunk: Same.
    :param num_hours_per_chunk: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    if num_months_per_chunk > 0:
        _plot_by_month(
            top_evaluation_dir_name=top_evaluation_dir_name,
            num_months_per_chunk=num_months_per_chunk,
            output_dir_name=output_dir_name)

    if num_hours_per_chunk > 0:
        _plot_by_hour(
            top_evaluation_dir_name=top_evaluation_dir_name,
            num_hours_per_chunk=num_hours_per_chunk,
            output_dir_name=output_dir_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_evaluation_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        num_months_per_chunk=getattr(INPUT_ARG_OBJECT, NUM_MONTHS_ARG_NAME),
        num_hours_per_chunk=getattr(INPUT_ARG_OBJECT, NUM_HOURS_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
