"""Plots results of backwards optimization.

Specifically, this script plots inputs (non-optimized examples), outputs
(optimized examples), and differences.
"""

import os.path
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import backwards_optimization as backwards_opt
from gewittergefahr.scripts import plot_input_examples as plot_examples

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_FILE_ARG_NAME = 'input_file_name'
COLOUR_MAP_ARG_NAME = 'diff_colour_map_name'
MAX_PERCENTILE_ARG_NAME = 'max_diff_percentile'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'
PLOT_SOUNDINGS_ARG_NAME = plot_examples.PLOT_SOUNDINGS_ARG_NAME
ALLOW_WHITESPACE_ARG_NAME = plot_examples.ALLOW_WHITESPACE_ARG_NAME
PLOT_PANEL_NAMES_ARG_NAME = plot_examples.PLOT_PANEL_NAMES_ARG_NAME
ADD_TITLES_ARG_NAME = plot_examples.ADD_TITLES_ARG_NAME
LABEL_CBARS_ARG_NAME = plot_examples.LABEL_CBARS_ARG_NAME
CBAR_LENGTH_ARG_NAME = plot_examples.CBAR_LENGTH_ARG_NAME

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `backwards_optimization.read_file`.')

COLOUR_MAP_HELP_STRING = (
    'Name of colour map (must be accepted by `pyplot.get_cmap`).  Will be used '
    'to plot radar differences (after minus before).')

MAX_PERCENTILE_HELP_STRING = (
    'Determines max value in colour scheme for radar differences.  For each '
    'example and each radar field, max value will be [q]th percentile of '
    'absolute differences, where q = `{0:s}`.'
).format(MAX_PERCENTILE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.')

PLOT_SOUNDINGS_HELP_STRING = plot_examples.PLOT_SOUNDINGS_HELP_STRING
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
    '--' + PLOT_SOUNDINGS_ARG_NAME, type=int, required=False, default=1,
    help=PLOT_SOUNDINGS_HELP_STRING)

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
         plot_soundings, allow_whitespace, plot_panel_names, add_titles,
         label_colour_bars, colour_bar_length, top_output_dir_name):
    """Plots results of backwards optimization.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param diff_colour_map_name: Same.
    :param max_diff_percentile: Same.
    :param plot_soundings: Same.
    :param allow_whitespace: Same.
    :param plot_panel_names: Same.
    :param add_titles: Same.
    :param label_colour_bars: Same.
    :param colour_bar_length: Same.
    :param top_output_dir_name: Same.
    """

    diff_colour_map_object = pyplot.cm.get_cmap(diff_colour_map_name)

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    bwo_dictionary, pmm_flag = backwards_opt.read_file(input_file_name)

    if pmm_flag:
        input_matrices = bwo_dictionary.pop(
            backwards_opt.MEAN_INPUT_MATRICES_KEY)
        output_matrices = bwo_dictionary.pop(
            backwards_opt.MEAN_OUTPUT_MATRICES_KEY)

        full_storm_id_strings = [None]
        storm_times_unix_sec = [None]

        mean_sounding_pressures_pa = bwo_dictionary[
            backwards_opt.MEAN_SOUNDING_PRESSURES_KEY]

        if mean_sounding_pressures_pa is None:
            sounding_pressure_matrix_pa = None
        else:
            sounding_pressure_matrix_pa = numpy.reshape(
                mean_sounding_pressures_pa, (1, len(mean_sounding_pressures_pa))
            )

        for i in range(len(input_matrices)):
            input_matrices[i] = numpy.expand_dims(input_matrices[i], axis=0)
            output_matrices[i] = numpy.expand_dims(output_matrices[i], axis=0)
    else:
        input_matrices = bwo_dictionary.pop(backwards_opt.INPUT_MATRICES_KEY)
        output_matrices = bwo_dictionary.pop(backwards_opt.OUTPUT_MATRICES_KEY)

        full_storm_id_strings = bwo_dictionary[backwards_opt.FULL_STORM_IDS_KEY]
        storm_times_unix_sec = bwo_dictionary[backwards_opt.STORM_TIMES_KEY]
        sounding_pressure_matrix_pa = bwo_dictionary[
            backwards_opt.SOUNDING_PRESSURES_KEY]

    model_file_name = bwo_dictionary[backwards_opt.MODEL_FILE_KEY]
    model_metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0]
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_model_metadata(model_metafile_name)
    print(SEPARATOR_STRING)

    plot_examples.plot_examples(
        list_of_predictor_matrices=input_matrices,
        model_metadata_dict=model_metadata_dict, pmm_flag=pmm_flag,
        output_dir_name='{0:s}/before_optimization'.format(top_output_dir_name),
        plot_soundings=plot_soundings,
        sounding_pressure_matrix_pascals=sounding_pressure_matrix_pa,
        allow_whitespace=allow_whitespace, plot_panel_names=plot_panel_names,
        add_titles=add_titles, label_colour_bars=label_colour_bars,
        colour_bar_length=colour_bar_length, plot_radar_diffs=False,
        full_storm_id_strings=full_storm_id_strings,
        storm_times_unix_sec=storm_times_unix_sec)
    print(SEPARATOR_STRING)

    plot_examples.plot_examples(
        list_of_predictor_matrices=output_matrices,
        model_metadata_dict=model_metadata_dict, pmm_flag=pmm_flag,
        output_dir_name='{0:s}/after_optimization'.format(top_output_dir_name),
        plot_soundings=plot_soundings,
        sounding_pressure_matrix_pascals=sounding_pressure_matrix_pa,
        allow_whitespace=allow_whitespace, plot_panel_names=plot_panel_names,
        add_titles=add_titles, label_colour_bars=label_colour_bars,
        colour_bar_length=colour_bar_length, plot_radar_diffs=False,
        full_storm_id_strings=full_storm_id_strings,
        storm_times_unix_sec=storm_times_unix_sec)
    print(SEPARATOR_STRING)

    difference_matrices = [
        b - a for b, a in zip(output_matrices, input_matrices)
    ]

    plot_examples.plot_examples(
        list_of_predictor_matrices=difference_matrices,
        model_metadata_dict=model_metadata_dict, pmm_flag=pmm_flag,
        output_dir_name='{0:s}/difference'.format(top_output_dir_name),
        plot_soundings=False, allow_whitespace=allow_whitespace,
        plot_panel_names=plot_panel_names, add_titles=add_titles,
        label_colour_bars=label_colour_bars,
        colour_bar_length=colour_bar_length, plot_radar_diffs=True,
        diff_colour_map_object=diff_colour_map_object,
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
        plot_soundings=bool(getattr(INPUT_ARG_OBJECT, PLOT_SOUNDINGS_ARG_NAME)),
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
