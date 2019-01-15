"""Plots many input examples (one per storm object)."""

import os.path
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import testing_io
from gewittergefahr.deep_learning import model_activation
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import radar_plotting
from gewittergefahr.plotting import sounding_plotting

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

SOUNDING_FIELD_NAMES = [
    soundings.U_WIND_NAME, soundings.V_WIND_NAME,
    soundings.TEMPERATURE_NAME, soundings.SPECIFIC_HUMIDITY_NAME,
    soundings.PRESSURE_NAME
]
SOUNDING_HEIGHTS_M_AGL = soundings.DEFAULT_HEIGHT_LEVELS_M_AGL

ACTIVATIONS_KEY = 'storm_activations'

FONT_SIZE = 20
NUM_PANEL_ROWS = 3
FIGURE_RESOLUTION_DPI = 300

ACTIVATION_FILE_ARG_NAME = 'input_activation_file_name'
STORM_METAFILE_ARG_NAME = 'input_storm_metafile_name'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
RADAR_FIELDS_ARG_NAME = 'radar_field_names'
RADAR_HEIGHTS_ARG_NAME = 'radar_heights_m_agl'
PLOT_SOUNDINGS_ARG_NAME = 'plot_soundings'
NUM_ROWS_ARG_NAME = 'num_radar_rows'
NUM_COLUMNS_ARG_NAME = 'num_radar_columns'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

ACTIVATION_FILE_HELP_STRING = (
    'Path to activation file (will be read by `model_activation.read_file`).  '
    'If this argument is empty, will use `{0:s}`.'
).format(STORM_METAFILE_ARG_NAME)

STORM_METAFILE_HELP_STRING = (
    'Path to Pickle file with storm IDs and times (will be read by '
    '`storm_tracking_io.read_ids_and_times`).  If this argument is empty, will '
    'use `{0:s}`.'
).format(ACTIVATION_FILE_ARG_NAME)

NUM_EXAMPLES_HELP_STRING = (
    'Number of examples (storm objects) to read from `{0:s}` or `{1:s}`.  If '
    'you want to read all examples, make this non-positive.'
).format(ACTIVATION_FILE_ARG_NAME, STORM_METAFILE_ARG_NAME)

EXAMPLE_DIR_HELP_STRING = (
    'Name of top-level directory with input examples.  Files therein will be '
    'found by `input_examples.find_example_file` and read by '
    '`input_examples.read_example_file`.')

RADAR_FIELDS_HELP_STRING = (
    '[used only if `{0:s}` is empty] List of radar fields (used as input to '
    '`input_examples.read_example_file`).  If you want to plot all radar '
    'fields, leave this argument empty.'
).format(ACTIVATION_FILE_ARG_NAME)

RADAR_HEIGHTS_HELP_STRING = (
    '[used only if `{0:s}` is empty] List of radar heights (used as input to '
    '`input_examples.read_example_file`).  If you want to plot all radar '
    'heights, leave this argument empty.'
).format(ACTIVATION_FILE_ARG_NAME)

PLOT_SOUNDINGS_HELP_STRING = (
    'Boolean flag.  If 1, will plot sounding for each example.  If 0, will not '
    'plot soundings.')

NUM_ROWS_HELP_STRING = (
    '[used only if `{0:s}` is empty] Number of rows in each storm-centered '
    'radar grid.  If you want to plot the largest grids available, leave this '
    'argument empty.'
).format(ACTIVATION_FILE_ARG_NAME)

NUM_COLUMNS_HELP_STRING = (
    '[used only if `{0:s}` is empty] Number of columns in each storm-centered '
    'radar grid.  If you want to plot the largest grids available, leave this '
    'argument empty.'
).format(ACTIVATION_FILE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + ACTIVATION_FILE_ARG_NAME, type=str, required=False, default='',
    help=ACTIVATION_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + STORM_METAFILE_ARG_NAME, type=str, required=False, default='',
    help=STORM_METAFILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_EXAMPLES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_FIELDS_ARG_NAME, type=str, nargs='+', required=False,
    default=[''], help=RADAR_FIELDS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_HEIGHTS_ARG_NAME, type=int, nargs='+', required=False,
    default=[-1], help=RADAR_HEIGHTS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_SOUNDINGS_ARG_NAME, type=int, required=False, default=1,
    help=PLOT_SOUNDINGS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_ROWS_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_ROWS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_COLUMNS_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_COLUMNS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _plot_examples(
        list_of_predictor_matrices, storm_ids, storm_times_unix_sec,
        training_option_dict, output_dir_name, storm_activations=None):
    """Plots one or more learning examples.

    E = number of examples (storm objects)

    :param list_of_predictor_matrices: List created by
        `testing_io.read_specific_examples`.  Contains data to be plotted.
    :param storm_ids: length-E list of storm IDs.
    :param storm_times_unix_sec: length-E numpy array of storm times.
    :param training_option_dict: Dictionary returned by
        `cnn.read_model_metadata`.  Contains metadata for
        `list_of_predictor_matrices`.
    :param output_dir_name: Name of output directory (figures will be saved
        here).
    :param storm_activations: length-E numpy array of storm activations (may be
        None).  Will be included in title of each figure.
    """

    sounding_field_names = training_option_dict[trainval_io.SOUNDING_FIELDS_KEY]
    plot_soundings = sounding_field_names is not None

    if plot_soundings:
        list_of_metpy_dictionaries = dl_utils.soundings_to_metpy_dictionaries(
            sounding_matrix=list_of_predictor_matrices[-1],
            field_names=sounding_field_names)

    num_storms = len(storm_ids)
    myrorss_2d3d = len(list_of_predictor_matrices) == 3

    for i in range(num_storms):
        this_time_string = time_conversion.unix_sec_to_string(
            storm_times_unix_sec[i], TIME_FORMAT)
        this_base_title_string = 'Storm "{0:s}" at {1:s}'.format(
            storm_ids[i], this_time_string)

        if storm_activations is not None:
            this_base_title_string += ' (activation = {0:.3f})'.format(
                storm_activations[i])

        this_base_file_name = '{0:s}/storm={1:s}_{2:s}'.format(
            output_dir_name, storm_ids[i].replace('_', '-'), this_time_string)

        if plot_soundings:
            sounding_plotting.plot_sounding(
                sounding_dict_for_metpy=list_of_metpy_dictionaries[i],
                title_string=this_base_title_string)

            this_file_name = '{0:s}_sounding.jpg'.format(this_base_file_name)
            print 'Saving figure to: "{0:s}"...'.format(this_file_name)
            pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
            pyplot.close()

        if myrorss_2d3d:
            this_radar_matrix = numpy.flip(
                list_of_predictor_matrices[0][i, ..., 0], axis=0)

            _, these_axes_objects = radar_plotting.plot_3d_grid_without_coords(
                field_matrix=this_radar_matrix,
                field_name=radar_utils.REFL_NAME,
                grid_point_heights_metres=training_option_dict[
                    trainval_io.RADAR_HEIGHTS_KEY],
                ground_relative=True, num_panel_rows=NUM_PANEL_ROWS,
                font_size=FONT_SIZE)

            this_colour_map_object, this_colour_norm_object = (
                radar_plotting.get_default_colour_scheme(
                    radar_utils.REFL_NAME
                )[:2]
            )

            plotting_utils.add_colour_bar(
                axes_object_or_list=these_axes_objects,
                values_to_colour=this_radar_matrix,
                colour_map=this_colour_map_object,
                colour_norm_object=this_colour_norm_object,
                orientation='horizontal', extend_min=True, extend_max=True)

            this_title_string = '{0:s}; {1:s}'.format(
                this_base_title_string, radar_utils.REFL_NAME)
            this_file_name = '{0:s}_reflectivity.jpg'.format(
                this_base_file_name)

            pyplot.suptitle(this_title_string, fontsize=FONT_SIZE)
            print 'Saving figure to: "{0:s}"...'.format(this_file_name)
            pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
            pyplot.close()

            these_heights_m_agl = numpy.full(
                len(training_option_dict[trainval_io.RADAR_FIELDS_KEY]),
                radar_utils.SHEAR_HEIGHT_M_ASL)

            this_radar_matrix = numpy.flip(
                list_of_predictor_matrices[1][i, ..., 0], axis=0)

            _, these_axes_objects = (
                radar_plotting.plot_many_2d_grids_without_coords(
                    field_matrix=this_radar_matrix,
                    field_name_by_pair=training_option_dict[
                        trainval_io.RADAR_FIELDS_KEY],
                    height_by_pair_metres=these_heights_m_agl,
                    ground_relative=True, num_panel_rows=1,
                    plot_colour_bars=False, font_size=FONT_SIZE)
            )

            this_colour_map_object, this_colour_norm_object = (
                radar_plotting.get_default_colour_scheme(
                    radar_utils.LOW_LEVEL_SHEAR_NAME
                )[:2]
            )

            plotting_utils.add_colour_bar(
                axes_object_or_list=these_axes_objects,
                values_to_colour=this_radar_matrix,
                colour_map=this_colour_map_object,
                colour_norm_object=this_colour_norm_object,
                orientation='horizontal', extend_min=True, extend_max=True)

            this_file_name = '{0:s}_shear.jpg'.format(this_base_file_name)
            pyplot.suptitle(this_base_title_string, fontsize=FONT_SIZE)
            print 'Saving figure to: "{0:s}"...'.format(this_file_name)
            pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
            pyplot.close()

            continue

        radar_field_names = training_option_dict[trainval_io.RADAR_FIELDS_KEY]
        radar_heights_m_agl = training_option_dict[
            trainval_io.RADAR_HEIGHTS_KEY]

        this_radar_matrix = list_of_predictor_matrices[0]
        num_radar_dimensions = len(list_of_predictor_matrices[0].shape) - 2

        if num_radar_dimensions == 2:
            j_max = 1
        else:
            j_max = len(radar_field_names)

        for j in range(j_max):
            if num_radar_dimensions == 2:
                radar_plotting.plot_many_2d_grids_without_coords(
                    field_matrix=numpy.flip(this_radar_matrix[i, ...], axis=0),
                    field_name_by_pair=radar_field_names,
                    height_by_pair_metres=radar_heights_m_agl,
                    ground_relative=True, num_panel_rows=NUM_PANEL_ROWS,
                    plot_colour_bars=True, font_size=FONT_SIZE)

                this_title_string = this_base_title_string + ''
                this_file_name = '{0:s}.jpg'.format(this_base_file_name)
            else:
                _, these_axes_objects = (
                    radar_plotting.plot_3d_grid_without_coords(
                        field_matrix=numpy.flip(
                            this_radar_matrix[i, ..., j], axis=0),
                        field_name=radar_field_names[j],
                        grid_point_heights_metres=radar_heights_m_agl,
                        ground_relative=True, num_panel_rows=NUM_PANEL_ROWS,
                        font_size=FONT_SIZE)
                )

                this_colour_map_object, this_colour_norm_object = (
                    radar_plotting.get_default_colour_scheme(
                        radar_field_names[j]
                    )[:2]
                )

                plotting_utils.add_colour_bar(
                    axes_object_or_list=these_axes_objects,
                    values_to_colour=this_radar_matrix[i, ..., j],
                    colour_map=this_colour_map_object,
                    colour_norm_object=this_colour_norm_object,
                    orientation='horizontal', extend_min=True, extend_max=True)

                this_title_string = '{0:s}; {1:s}'.format(
                    this_base_title_string, radar_field_names[j])
                this_file_name = '{0:s}_{1:s}.jpg'.format(
                    this_base_file_name, radar_field_names[j].replace('_', '-')
                )

            pyplot.suptitle(this_title_string, fontsize=FONT_SIZE)
            print 'Saving figure to: "{0:s}"...'.format(this_file_name)
            pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
            pyplot.close()


def _run(activation_file_name, storm_metafile_name, num_examples,
         top_example_dir_name, radar_field_names, radar_heights_m_agl,
         plot_soundings, num_radar_rows, num_radar_columns, output_dir_name):
    """Plots many input examples (one per storm object).

    This is effectively the main method.

    :param activation_file_name: See documentation at top of file.
    :param storm_metafile_name: Same.
    :param num_examples: Same.
    :param top_example_dir_name: Same.
    :param radar_field_names: Same.
    :param radar_heights_m_agl: Same.
    :param plot_soundings: Same.
    :param num_radar_rows: Same.
    :param num_radar_columns: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if activation file contains activations for more than
        one model component.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    storm_activations = None
    if activation_file_name in ['', 'None']:
        activation_file_name = None

    if activation_file_name is None:
        print 'Reading data from: "{0:s}"...'.format(storm_metafile_name)
        storm_ids, storm_times_unix_sec = tracking_io.read_ids_and_times(
            storm_metafile_name)

        training_option_dict = dict()
        training_option_dict[trainval_io.RADAR_FIELDS_KEY] = radar_field_names
        training_option_dict[
            trainval_io.RADAR_HEIGHTS_KEY] = radar_heights_m_agl

        if plot_soundings:
            training_option_dict[
                trainval_io.SOUNDING_FIELDS_KEY] = SOUNDING_FIELD_NAMES
            training_option_dict[
                trainval_io.SOUNDING_HEIGHTS_KEY] = SOUNDING_HEIGHTS_M_AGL
        else:
            training_option_dict[trainval_io.SOUNDING_FIELDS_KEY] = None
            training_option_dict[trainval_io.SOUNDING_HEIGHTS_KEY] = None

        training_option_dict[trainval_io.NUM_ROWS_KEY] = num_radar_rows
        training_option_dict[trainval_io.NUM_COLUMNS_KEY] = num_radar_columns
        training_option_dict[trainval_io.NORMALIZATION_TYPE_KEY] = None
        training_option_dict[trainval_io.BINARIZE_TARGET_KEY] = False
        training_option_dict[trainval_io.SAMPLING_FRACTIONS_KEY] = None
        training_option_dict[trainval_io.REFLECTIVITY_MASK_KEY] = None
    else:
        print 'Reading data from: "{0:s}"...'.format(activation_file_name)
        activation_matrix, activation_metadata_dict = (
            model_activation.read_file(activation_file_name))

        num_model_components = activation_matrix.shape[1]
        if num_model_components > 1:
            error_string = (
                'The file should contain activations for only one model '
                'component, not {0:d}.'
            ).format(num_model_components)

            raise ValueError(error_string)

        storm_ids = activation_metadata_dict[model_activation.STORM_IDS_KEY]
        storm_times_unix_sec = activation_metadata_dict[
            model_activation.STORM_TIMES_KEY]
        storm_activations = activation_matrix[:, 0]

        model_file_name = activation_metadata_dict[
            model_activation.MODEL_FILE_NAME_KEY]
        model_metafile_name = '{0:s}/model_metadata.p'.format(
            os.path.split(model_file_name)[0])

        print 'Reading metadata from: "{0:s}"...'.format(model_metafile_name)
        model_metadata_dict = cnn.read_model_metadata(model_metafile_name)

        training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
        training_option_dict[trainval_io.NORMALIZATION_TYPE_KEY] = None
        training_option_dict[trainval_io.SAMPLING_FRACTIONS_KEY] = None
        training_option_dict[trainval_io.REFLECTIVITY_MASK_KEY] = None

        if plot_soundings:
            training_option_dict[
                trainval_io.SOUNDING_FIELDS_KEY] = SOUNDING_FIELD_NAMES
        else:
            training_option_dict[trainval_io.SOUNDING_FIELDS_KEY] = None

    if 0 < num_examples < len(storm_ids):
        storm_ids = storm_ids[:num_examples]
        storm_times_unix_sec = storm_times_unix_sec[:num_examples]
        if storm_activations is not None:
            storm_activations = storm_activations[:num_examples]

    print SEPARATOR_STRING
    list_of_predictor_matrices = testing_io.read_specific_examples(
        desired_storm_ids=storm_ids,
        desired_times_unix_sec=storm_times_unix_sec,
        option_dict=training_option_dict,
        top_example_dir_name=top_example_dir_name
    )[0]
    print SEPARATOR_STRING

    _plot_examples(
        list_of_predictor_matrices=list_of_predictor_matrices,
        storm_ids=storm_ids, storm_times_unix_sec=storm_times_unix_sec,
        storm_activations=storm_activations,
        training_option_dict=training_option_dict,
        output_dir_name=output_dir_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        activation_file_name=getattr(
            INPUT_ARG_OBJECT, ACTIVATION_FILE_ARG_NAME),
        storm_metafile_name=getattr(INPUT_ARG_OBJECT, STORM_METAFILE_ARG_NAME),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        top_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        radar_field_names=getattr(INPUT_ARG_OBJECT, RADAR_FIELDS_ARG_NAME),
        radar_heights_m_agl=numpy.array(
            getattr(INPUT_ARG_OBJECT, RADAR_HEIGHTS_ARG_NAME), dtype=int),
        plot_soundings=bool(getattr(INPUT_ARG_OBJECT, PLOT_SOUNDINGS_ARG_NAME)),
        num_radar_rows=getattr(INPUT_ARG_OBJECT, NUM_ROWS_ARG_NAME),
        num_radar_columns=getattr(INPUT_ARG_OBJECT, NUM_COLUMNS_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
