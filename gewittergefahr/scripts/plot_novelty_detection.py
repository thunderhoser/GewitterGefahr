"""Plots results of novelty detection."""

import os.path
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import novelty_detection
from gewittergefahr.scripts import plot_input_examples as plot_examples

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_FILE_ARG_NAME = 'input_file_name'
COLOUR_MAP_ARG_NAME = 'diff_colour_map_name'
MAX_PERCENTILE_ARG_NAME = 'max_diff_percentile'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'
ALLOW_WHITESPACE_ARG_NAME = plot_examples.ALLOW_WHITESPACE_ARG_NAME
PLOT_PANEL_NAMES_ARG_NAME = plot_examples.PLOT_PANEL_NAMES_ARG_NAME
ADD_TITLES_ARG_NAME = plot_examples.ADD_TITLES_ARG_NAME
LABEL_CBARS_ARG_NAME = plot_examples.LABEL_CBARS_ARG_NAME
CBAR_LENGTH_ARG_NAME = plot_examples.CBAR_LENGTH_ARG_NAME

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `novelty_detection.read_file`.')

COLOUR_MAP_HELP_STRING = (
    'Name of colour map (must be accepted by `pyplot.get_cmap`).  Will be used '
    'to plot novelty.')

MAX_PERCENTILE_HELP_STRING = (
    'Determines max value in colour scheme for novelty.  For each example and '
    'each radar field, max value will be [q]th percentile of absolute '
    'differences, where q = `{0:s}`.'
).format(MAX_PERCENTILE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.')

ALLOW_WHITESPACE_HELP_STRING = plot_examples.ALLOW_WHITESPACE_HELP_STRING
PLOT_PANEL_NAMES_HELP_STRING = plot_examples.PLOT_PANEL_NAMES_HELP_STRING
ADD_TITLES_HELP_STRING = plot_examples.ADD_TITLES_HELP_STRING
LABEL_CBARS_HELP_STRING = plot_examples.LABEL_CBARS_HELP_STRING
CBAR_LENGTH_HELP_STRING = plot_examples.CBAR_LENGTH_HELP_STRING

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + COLOUR_MAP_ARG_NAME, type=str, required=False, default='seismic',
    help=COLOUR_MAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False,
    default=99., help=MAX_PERCENTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + ALLOW_WHITESPACE_ARG_NAME, type=int, required=False, default=1,
    help=ALLOW_WHITESPACE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_PANEL_NAMES_ARG_NAME, type=int, required=False, default=1,
    help=PLOT_PANEL_NAMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + ADD_TITLES_ARG_NAME, type=int, required=False, default=1,
    help=ADD_TITLES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LABEL_CBARS_ARG_NAME, type=int, required=False, default=0,
    help=LABEL_CBARS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CBAR_LENGTH_ARG_NAME, type=float, required=False, default=0.8,
    help=CBAR_LENGTH_HELP_STRING)


def _run(input_file_name, diff_colour_map_name, max_diff_percentile,
         allow_whitespace, plot_panel_names, add_titles, label_colour_bars,
         colour_bar_length, top_output_dir_name):
    """Plots results of novelty detection.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param diff_colour_map_name: Same.
    :param max_diff_percentile: Same.
    :param allow_whitespace: Same.
    :param plot_panel_names: Same.
    :param add_titles: Same.
    :param label_colour_bars: Same.
    :param colour_bar_length: Same.
    :param top_output_dir_name: Same.
    """

    diff_colour_map_object = pyplot.cm.get_cmap(diff_colour_map_name)

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    novelty_dict, pmm_flag = novelty_detection.read_file(input_file_name)

    if pmm_flag:
        trial_radar_matrix = numpy.expand_dims(
            novelty_dict.pop(novelty_detection.MEAN_NOVEL_MATRIX_KEY), axis=0
        )
        upconv_radar_matrix = numpy.expand_dims(
            novelty_dict.pop(novelty_detection.MEAN_UPCONV_MATRIX_KEY), axis=0
        )
        upconv_svd_radar_matrix = numpy.expand_dims(
            novelty_dict.pop(novelty_detection.MEAN_UPCONV_SVD_MATRIX_KEY),
            axis=0
        )

        full_storm_id_strings = [None]
        storm_times_unix_sec = [None]
    else:
        novel_indices = novelty_dict[novelty_detection.NOVEL_INDICES_KEY]
        trial_radar_matrix = novelty_dict.pop(
            novelty_detection.TRIAL_MATRIX_KEY
        )[novel_indices, ...]

        upconv_radar_matrix = novelty_dict.pop(
            novelty_detection.UPCONV_MATRIX_KEY)
        upconv_svd_radar_matrix = novelty_dict.pop(
            novelty_detection.UPCONV_SVD_MATRIX_KEY)

        full_storm_id_strings = novelty_dict[
            novelty_detection.TRIAL_STORM_IDS_KEY]
        storm_times_unix_sec = novelty_dict[
            novelty_detection.TRIAL_STORM_TIMES_KEY]

    cnn_file_name = novelty_dict[novelty_detection.CNN_FILE_KEY]
    cnn_metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(cnn_file_name)[0]
    )

    print('Reading metadata from: "{0:s}"...'.format(cnn_metafile_name))
    cnn_metadata_dict = cnn.read_model_metadata(cnn_metafile_name)
    print(SEPARATOR_STRING)

    plot_examples.plot_examples(
        list_of_predictor_matrices=[trial_radar_matrix],
        model_metadata_dict=cnn_metadata_dict, pmm_flag=pmm_flag,
        output_dir_name='{0:s}/novel_examples'.format(top_output_dir_name),
        plot_soundings=False, allow_whitespace=allow_whitespace,
        plot_panel_names=plot_panel_names, add_titles=add_titles,
        label_colour_bars=label_colour_bars,
        colour_bar_length=colour_bar_length, plot_radar_diffs=False,
        full_storm_id_strings=full_storm_id_strings,
        storm_times_unix_sec=storm_times_unix_sec)
    print(SEPARATOR_STRING)

    plot_examples.plot_examples(
        list_of_predictor_matrices=[upconv_radar_matrix],
        model_metadata_dict=cnn_metadata_dict, pmm_flag=pmm_flag,
        output_dir_name='{0:s}/upconvnet'.format(top_output_dir_name),
        plot_soundings=False, allow_whitespace=allow_whitespace,
        plot_panel_names=plot_panel_names, add_titles=add_titles,
        label_colour_bars=label_colour_bars,
        colour_bar_length=colour_bar_length, plot_radar_diffs=False,
        full_storm_id_strings=full_storm_id_strings,
        storm_times_unix_sec=storm_times_unix_sec)
    print(SEPARATOR_STRING)

    plot_examples.plot_examples(
        list_of_predictor_matrices=[upconv_svd_radar_matrix],
        model_metadata_dict=cnn_metadata_dict, pmm_flag=pmm_flag,
        output_dir_name='{0:s}/upconvnet_svd'.format(top_output_dir_name),
        plot_soundings=False, allow_whitespace=allow_whitespace,
        plot_panel_names=plot_panel_names, add_titles=add_titles,
        label_colour_bars=label_colour_bars,
        colour_bar_length=colour_bar_length, plot_radar_diffs=False,
        full_storm_id_strings=full_storm_id_strings,
        storm_times_unix_sec=storm_times_unix_sec)
    print(SEPARATOR_STRING)

    plot_examples.plot_examples(
        list_of_predictor_matrices=[
            upconv_radar_matrix - upconv_svd_radar_matrix
        ],
        model_metadata_dict=cnn_metadata_dict, pmm_flag=pmm_flag,
        output_dir_name='{0:s}/novelty'.format(top_output_dir_name),
        plot_soundings=False, allow_whitespace=allow_whitespace,
        plot_panel_names=plot_panel_names, add_titles=add_titles,
        label_colour_bars=label_colour_bars,
        colour_bar_length=colour_bar_length,
        plot_radar_diffs=True, diff_colour_map_object=diff_colour_map_object,
        max_diff_percentile=max_diff_percentile,
        full_storm_id_strings=full_storm_id_strings,
        storm_times_unix_sec=storm_times_unix_sec)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        diff_colour_map_name=getattr(INPUT_ARG_OBJECT, COLOUR_MAP_ARG_NAME),
        max_diff_percentile=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME),
        allow_whitespace=bool(getattr(
            INPUT_ARG_OBJECT, ALLOW_WHITESPACE_ARG_NAME
        )),
        plot_panel_names=bool(getattr(
            INPUT_ARG_OBJECT, PLOT_PANEL_NAMES_ARG_NAME
        )),
        add_titles=bool(getattr(INPUT_ARG_OBJECT, ADD_TITLES_ARG_NAME)),
        label_colour_bars=bool(getattr(
            INPUT_ARG_OBJECT, LABEL_CBARS_ARG_NAME
        )),
        colour_bar_length=getattr(INPUT_ARG_OBJECT, CBAR_LENGTH_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
