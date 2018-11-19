"""Plots many input examples (one per storm object)."""

import os.path
import argparse
import numpy
import netCDF4
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.deep_learning import testing_io
from gewittergefahr.deep_learning import model_activation
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import radar_plotting
from gewittergefahr.plotting import sounding_plotting

LARGE_INTEGER = int(1e10)
TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

SOUNDING_FIELD_NAMES = [
    soundings.U_WIND_NAME, soundings.V_WIND_NAME,
    soundings.TEMPERATURE_NAME, soundings.SPECIFIC_HUMIDITY_NAME,
    soundings.PRESSURE_NAME
]
SOUNDING_HEIGHTS_M_AGL = soundings.DEFAULT_HEIGHT_LEVELS_M_AGL

ACTIVATIONS_KEY = 'storm_activations'

NUM_PANEL_ROWS = 3
TITLE_FONT_SIZE = 20
FIGURE_RESOLUTION_DPI = 300

ACTIVATION_FILE_ARG_NAME = 'input_activation_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
STORM_IDS_ARG_NAME = 'storm_ids'
STORM_TIMES_ARG_NAME = 'storm_time_strings'
RADAR_FIELDS_ARG_NAME = 'radar_field_names'
RADAR_HEIGHTS_ARG_NAME = 'radar_heights_m_agl'
PLOT_SOUNDINGS_ARG_NAME = 'plot_soundings'
NUM_ROWS_ARG_NAME = 'num_radar_rows'
NUM_COLUMNS_ARG_NAME = 'num_radar_columns'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

ACTIVATION_FILE_HELP_STRING = (
    'Path to activation file.  If this argument is non-empty, the file will be '
    'read by `model_activation.read_file` and all input arguments other than '
    'this and `{0:s}` will be ignored.'
).format(EXAMPLE_DIR_ARG_NAME)

EXAMPLE_DIR_HELP_STRING = (
    'Name of top-level directory with input examples.  Files therein will be '
    'found by `input_examples.find_example_file` and read by '
    '`input_examples.read_example_file`.')

STORM_IDS_HELP_STRING = 'List of storm IDs (one per example).'

STORM_TIMES_HELP_STRING = (
    'List of storm times (format "yyyy-mm-dd-HHMMSS"; one per example).')

RADAR_FIELDS_HELP_STRING = (
    'List of radar fields (used as input to `input_examples.read_example_file`)'
    '.  If you want to plot all radar fields, leave this argument empty.')

RADAR_HEIGHTS_HELP_STRING = (
    'List of radar heights (used as input to `input_examples.read_example_file`'
    ').  If you want to plot all radar heights, leave this argument empty.')

PLOT_SOUNDINGS_HELP_STRING = (
    'Boolean flag.  If 1, will plot sounding for each example.  If 0, will not '
    'plot soundings.')

NUM_ROWS_HELP_STRING = (
    'Number of rows in each storm-centered radar grid.  If you want to plot the'
    ' largest grids available, leave this argument empty.')

NUM_COLUMNS_HELP_STRING = (
    'Number of columns in each storm-centered radar grid.  If you want to plot '
    'the largest grids available, leave this argument empty.')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + ACTIVATION_FILE_ARG_NAME, type=str, required=False, default='',
    help=ACTIVATION_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + STORM_IDS_ARG_NAME, type=str, nargs='+', required=False,
    default=[''], help=STORM_IDS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + STORM_TIMES_ARG_NAME, type=str, nargs='+', required=False,
    default=[''], help=STORM_TIMES_HELP_STRING)

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


def _plot_examples(storm_object_dict, training_option_dict, output_dir_name):
    """Plots one or more input examples.

    :param storm_object_dict: Dictionary created by
        `testing_io.example_generator_2d_or_3d` or
        `testing_io.example_generator_2d3d_myrorss`.
    :param training_option_dict: Input to
        `testing_io.example_generator_2d_or_3d` or
        `testing_io.example_generator_2d3d_myrorss`, used to generate
        `storm_object_dict`.
    :param output_dir_name: Same.
    """

    sounding_field_names = training_option_dict[trainval_io.SOUNDING_FIELDS_KEY]
    plot_soundings = sounding_field_names is not None

    if plot_soundings:
        list_of_metpy_dictionaries = dl_utils.soundings_to_metpy_dictionaries(
            sounding_matrix=storm_object_dict[
                testing_io.INPUT_MATRICES_KEY][-1],
            field_names=sounding_field_names)

    storm_ids = storm_object_dict[testing_io.STORM_IDS_KEY]
    storm_times_unix_sec = storm_object_dict[testing_io.STORM_TIMES_KEY]
    num_storms = len(storm_ids)
    myrorss_2d3d = len(storm_object_dict[testing_io.INPUT_MATRICES_KEY]) == 3

    for i in range(num_storms):
        this_time_string = time_conversion.unix_sec_to_string(
            storm_times_unix_sec[i], TIME_FORMAT)
        this_base_title_string = 'Storm "{0:s}" at {1:s}'.format(
            storm_ids[i], this_time_string)

        if ACTIVATIONS_KEY in storm_object_dict:
            this_base_title_string += ' (activation = {0:.3f})'.format(
                storm_object_dict[ACTIVATIONS_KEY][i])

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
                storm_object_dict[testing_io.INPUT_MATRICES_KEY][0][i, ..., 0],
                axis=0)

            _, these_axes_objects = radar_plotting.plot_3d_grid_without_coords(
                field_matrix=this_radar_matrix,
                field_name=radar_utils.REFL_NAME,
                grid_point_heights_metres=training_option_dict[
                    trainval_io.RADAR_HEIGHTS_KEY],
                ground_relative=True, num_panel_rows=NUM_PANEL_ROWS)

            this_colour_map_object, this_colour_norm_object = (
                radar_plotting.get_default_colour_scheme(
                    radar_utils.REFL_NAME)[:2]
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

            pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)
            print 'Saving figure to: "{0:s}"...'.format(this_file_name)
            pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
            pyplot.close()

            these_heights_m_agl = numpy.full(
                len(training_option_dict[trainval_io.RADAR_FIELDS_KEY]),
                radar_utils.SHEAR_HEIGHT_M_ASL)

            this_radar_matrix = numpy.flip(
                storm_object_dict[testing_io.INPUT_MATRICES_KEY][1][i, ..., 0],
                axis=0)

            _, these_axes_objects = (
                radar_plotting.plot_many_2d_grids_without_coords(
                    field_matrix=this_radar_matrix,
                    field_name_by_pair=training_option_dict[
                        trainval_io.RADAR_FIELDS_KEY],
                    height_by_pair_metres=these_heights_m_agl,
                    ground_relative=True, num_panel_rows=1)
            )

            this_colour_map_object, this_colour_norm_object = (
                radar_plotting.get_default_colour_scheme(
                    radar_utils.LOW_LEVEL_SHEAR_NAME)[:2]
            )

            plotting_utils.add_colour_bar(
                axes_object_or_list=these_axes_objects,
                values_to_colour=this_radar_matrix,
                colour_map=this_colour_map_object,
                colour_norm_object=this_colour_norm_object,
                orientation='horizontal', extend_min=True, extend_max=True)

            this_file_name = '{0:s}_shear.jpg'.format(this_base_file_name)
            pyplot.suptitle(this_base_title_string, fontsize=TITLE_FONT_SIZE)
            print 'Saving figure to: "{0:s}"...'.format(this_file_name)
            pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
            pyplot.close()

            continue

        radar_field_names = training_option_dict[trainval_io.RADAR_FIELDS_KEY]
        radar_heights_m_agl = training_option_dict[
            trainval_io.RADAR_HEIGHTS_KEY]
        this_radar_matrix = storm_object_dict[testing_io.INPUT_MATRICES_KEY][0]

        num_radar_dimensions = (
            len(storm_object_dict[testing_io.INPUT_MATRICES_KEY][0].shape) - 2
        )

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
                    ground_relative=True, num_panel_rows=NUM_PANEL_ROWS)

                this_title_string = this_base_title_string + ''
                this_file_name = '{0:s}.jpg'.format(this_base_file_name)
            else:
                _, these_axes_objects = (
                    radar_plotting.plot_3d_grid_without_coords(
                        field_matrix=numpy.flip(
                            this_radar_matrix[i, ..., j], axis=0),
                        field_name=radar_field_names[j],
                        grid_point_heights_metres=radar_heights_m_agl,
                        ground_relative=True, num_panel_rows=NUM_PANEL_ROWS)
                )

                this_colour_map_object, this_colour_norm_object = (
                    radar_plotting.get_default_colour_scheme(
                        radar_field_names[j])[:2]
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

            pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)
            print 'Saving figure to: "{0:s}"...'.format(this_file_name)
            pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
            pyplot.close()


def _run(activation_file_name, top_example_dir_name, storm_ids,
         storm_time_strings, radar_field_names, radar_heights_m_agl,
         plot_soundings, num_radar_rows, num_radar_columns, output_dir_name):
    """Plots many input examples (one per storm object).

    This is effectively the main method.

    :param activation_file_name: See documentation at top of file.
    :param top_example_dir_name: Same.
    :param storm_ids: Same.
    :param storm_time_strings: Same.
    :param radar_field_names: Same.
    :param radar_heights_m_agl: Same.
    :param plot_soundings: Same.
    :param num_radar_rows: Same.
    :param num_radar_columns: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if activation file contains activations for more than
        one model component.
    """

    myrorss_2d3d = None
    storm_activations = None

    if activation_file_name in ['', 'None']:
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
        myrorss_2d3d = model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]

        training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
        training_option_dict[trainval_io.NORMALIZATION_TYPE_KEY] = None
        training_option_dict[trainval_io.SAMPLING_FRACTIONS_KEY] = None
        training_option_dict[trainval_io.REFLECTIVITY_MASK_KEY] = None

        if plot_soundings:
            training_option_dict[trainval_io.SOUNDING_FIELDS_KEY] = None
        else:
            training_option_dict[
                trainval_io.SOUNDING_FIELDS_KEY] = SOUNDING_FIELD_NAMES

        print SEPARATOR_STRING
    else:
        if len(storm_ids) != len(storm_time_strings):
            error_string = (
                'Number of storm IDs ({0:d}) must equal number of storm times '
                '({1:d}).'
            ).format(len(storm_ids), len(storm_time_strings))

            raise ValueError(error_string)

        storm_times_unix_sec = numpy.array([
            time_conversion.string_to_unix_sec(s, TIME_FORMAT)
            for s in storm_time_strings
        ], dtype=int)

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

    storm_spc_dates_unix_sec = numpy.array([
        time_conversion.time_to_spc_date_unix_sec(t)
        for t in storm_times_unix_sec
    ], dtype=int)

    unique_spc_dates_unix_sec = numpy.unique(storm_spc_dates_unix_sec)

    for this_spc_date_unix_sec in unique_spc_dates_unix_sec:
        this_spc_date_string = time_conversion.time_to_spc_date_string(
            this_spc_date_unix_sec)
        this_start_time_unix_sec = time_conversion.get_start_of_spc_date(
            this_spc_date_string)
        this_end_time_unix_sec = time_conversion.get_end_of_spc_date(
            this_spc_date_string)

        this_example_file_name = input_examples.find_example_file(
            top_directory_name=top_example_dir_name, shuffled=False,
            spc_date_string=this_spc_date_string)

        training_option_dict[
            trainval_io.EXAMPLE_FILES_KEY] = [this_example_file_name]
        training_option_dict[
            trainval_io.FIRST_STORM_TIME_KEY] = this_start_time_unix_sec
        training_option_dict[
            trainval_io.LAST_STORM_TIME_KEY] = this_end_time_unix_sec

        if myrorss_2d3d is None:
            netcdf_dataset = netCDF4.Dataset(this_example_file_name)
            myrorss_2d3d = (
                input_examples.REFL_IMAGE_MATRIX_KEY in netcdf_dataset.variables
            )
            netcdf_dataset.close()

        if myrorss_2d3d:
            this_generator = testing_io.example_generator_2d3d_myrorss(
                option_dict=training_option_dict,
                num_examples_total=LARGE_INTEGER)
        else:
            this_generator = testing_io.example_generator_2d_or_3d(
                option_dict=training_option_dict,
                num_examples_total=LARGE_INTEGER)

        this_storm_object_dict = next(this_generator)

        these_indices = numpy.where(numpy.logical_and(
            storm_times_unix_sec >= this_start_time_unix_sec,
            storm_times_unix_sec <= this_end_time_unix_sec
        ))[0]

        if storm_activations is not None:
            these_activations = storm_activations[these_indices]
            this_storm_object_dict[ACTIVATIONS_KEY] = these_activations

        these_indices = tracking_utils.find_storm_objects(
            all_storm_ids=this_storm_object_dict[testing_io.STORM_IDS_KEY],
            all_times_unix_sec=this_storm_object_dict[
                testing_io.STORM_TIMES_KEY],
            storm_ids_to_keep=[storm_ids[k] for k in these_indices],
            times_to_keep_unix_sec=storm_times_unix_sec[these_indices],
            allow_missing=False)

        this_storm_object_dict[testing_io.STORM_IDS_KEY] = [
            this_storm_object_dict[testing_io.STORM_IDS_KEY][k]
            for k in these_indices
        ]
        this_storm_object_dict[testing_io.STORM_TIMES_KEY] = (
            this_storm_object_dict[testing_io.STORM_TIMES_KEY][
                these_indices]
        )

        this_num_matrices = len(
            this_storm_object_dict[testing_io.INPUT_MATRICES_KEY])
        for k in range(this_num_matrices):
            this_storm_object_dict[testing_io.INPUT_MATRICES_KEY][k] = (
                this_storm_object_dict[testing_io.INPUT_MATRICES_KEY][k][
                    these_indices, ...]
            )

        _plot_examples(
            storm_object_dict=this_storm_object_dict,
            training_option_dict=training_option_dict,
            output_dir_name=output_dir_name)
        print SEPARATOR_STRING


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        activation_file_name=getattr(
            INPUT_ARG_OBJECT, ACTIVATION_FILE_ARG_NAME),
        top_example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        storm_ids=getattr(INPUT_ARG_OBJECT, STORM_IDS_ARG_NAME),
        storm_time_strings=getattr(INPUT_ARG_OBJECT, STORM_TIMES_ARG_NAME),
        radar_field_names=getattr(INPUT_ARG_OBJECT, RADAR_FIELDS_ARG_NAME),
        radar_heights_m_agl=numpy.array(
            getattr(INPUT_ARG_OBJECT, RADAR_HEIGHTS_ARG_NAME), dtype=int),
        plot_soundings=bool(getattr(INPUT_ARG_OBJECT, PLOT_SOUNDINGS_ARG_NAME)),
        num_radar_rows=getattr(INPUT_ARG_OBJECT, NUM_ROWS_ARG_NAME),
        num_radar_columns=getattr(INPUT_ARG_OBJECT, NUM_COLUMNS_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
