"""Plots results of backwards optimization."""

import os.path
import argparse
import numpy
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.deep_learning import model_interpretation
from gewittergefahr.deep_learning import backwards_optimization as backwards_opt
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import radar_plotting
from gewittergefahr.plotting import sounding_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TITLE_FONT_SIZE = 20
FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_file_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `backwards_opt.read_file`.')

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory.  Figures will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _plot_examples(list_of_predictor_matrices, training_option_dict, optimized,
                   output_dir_name):
    """Plots one or more (either original or optimized) examples.

    :param list_of_predictor_matrices: List created by
        `testing_io.read_specific_examples`.  Contains data to be plotted.
    :param training_option_dict: Dictionary returned by
        `cnn.read_model_metadata`.  Contains metadata for
        `list_of_predictor_matrices`.
    :param optimized: Boolean flag.  If True, `list_of_predictor_matrices`
        contains optimized input examples.  If False, contains original examples
        (pre-optimization).  This piece of metadata will be reflected in file
        names and titles.
    :param output_dir_name: Name of output directory (figures will be saved
        here).
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    num_storms = list_of_predictor_matrices[0].shape[0]
    sounding_field_names = training_option_dict[trainval_io.SOUNDING_FIELDS_KEY]
    plot_soundings = sounding_field_names is not None

    if plot_soundings:
        list_of_metpy_dictionaries = dl_utils.soundings_to_metpy_dictionaries(
            sounding_matrix=list_of_predictor_matrices[-1],
            field_names=sounding_field_names,
            height_levels_m_agl=training_option_dict[
                trainval_io.SOUNDING_HEIGHTS_KEY],
            storm_elevations_m_asl=numpy.zeros(num_storms))

    myrorss_2d3d = len(list_of_predictor_matrices) == 3

    if optimized:
        optimized_flag_as_string = 'optimized'
    else:
        optimized_flag_as_string = 'original (pre-optimization)'

    for i in range(num_storms):
        this_title_string = 'Example {0:d}, {1:s}'.format(
            i, optimized_flag_as_string)

        if plot_soundings:
            this_file_name = (
                '{0:s}/example{1:06d}_optimized={2:d}_sounding.jpg'
            ).format(output_dir_name, i, int(optimized))

            sounding_plotting.plot_sounding(
                sounding_dict_for_metpy=list_of_metpy_dictionaries[i],
                title_string=this_title_string)

            print 'Saving figure to: "{0:s}"...'.format(this_file_name)
            pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
            pyplot.close()

        if myrorss_2d3d:
            this_radar_matrix = numpy.flip(
                list_of_predictor_matrices[0][i, ..., 0], axis=0)

            this_num_heights = this_radar_matrix.shape[-1]
            this_num_panel_rows = int(numpy.floor(numpy.sqrt(this_num_heights)))

            _, these_axes_objects = radar_plotting.plot_3d_grid_without_coords(
                field_matrix=this_radar_matrix,
                field_name=radar_utils.REFL_NAME,
                grid_point_heights_metres=training_option_dict[
                    trainval_io.RADAR_HEIGHTS_KEY],
                ground_relative=True, num_panel_rows=this_num_panel_rows)

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

            this_file_name = (
                '{0:s}/example{1:06d}_optimized={2:d}_reflectivity.jpg'
            ).format(output_dir_name, i, int(optimized))

            pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)
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

            this_file_name = (
                '{0:s}/example{1:06d}_optimized={2:d}_azimuthal-shear.jpg'
            ).format(output_dir_name, i, int(optimized))

            pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)
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
                this_num_predictors = this_radar_matrix.shape[-1]
                this_num_panel_rows = int(
                    numpy.floor(numpy.sqrt(this_num_predictors))
                )

                radar_plotting.plot_many_2d_grids_without_coords(
                    field_matrix=numpy.flip(this_radar_matrix[i, ...], axis=0),
                    field_name_by_pair=radar_field_names,
                    height_by_pair_metres=radar_heights_m_agl,
                    ground_relative=True, num_panel_rows=this_num_panel_rows)

                this_file_name = (
                    '{0:s}/example{1:06d}_optimized={2:d}_radar.jpg'
                ).format(output_dir_name, i, int(optimized))

            else:
                this_num_heights = this_radar_matrix.shape[-2]
                this_num_panel_rows = int(
                    numpy.floor(numpy.sqrt(this_num_heights))
                )

                _, these_axes_objects = (
                    radar_plotting.plot_3d_grid_without_coords(
                        field_matrix=numpy.flip(
                            this_radar_matrix[i, ..., j], axis=0),
                        field_name=radar_field_names[j],
                        grid_point_heights_metres=radar_heights_m_agl,
                        ground_relative=True,
                        num_panel_rows=this_num_panel_rows)
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

                this_file_name = (
                    '{0:s}/example{1:06d}_optimized={2:d}_{3:s}.jpg'
                ).format(
                    output_dir_name, i, int(optimized),
                    radar_field_names[j].replace('_', '-')
                )

            pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)
            print 'Saving figure to: "{0:s}"...'.format(this_file_name)
            pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
            pyplot.close()


def _run(input_file_name, top_output_dir_name):
    """Plots results of backwards optimization.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param top_output_dir_name: Same.
    """

    print 'Reading data from: "{0:s}"...'.format(input_file_name)
    list_of_optimized_input_matrices, backwards_opt_metadata_dict = (
        backwards_opt.read_results(input_file_name)
    )

    model_file_name = backwards_opt_metadata_dict[
        backwards_opt.MODEL_FILE_NAME_KEY]
    model_metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0]
    )

    print 'Reading metadata from: "{0:s}"...'.format(model_metafile_name)
    model_metadata_dict = cnn.read_model_metadata(model_metafile_name)
    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]

    print 'Denormalizing optimized input data...'
    list_of_optimized_input_matrices = model_interpretation.denormalize_data(
        list_of_input_matrices=list_of_optimized_input_matrices,
        model_metadata_dict=model_metadata_dict)

    init_function_name_or_matrices = backwards_opt_metadata_dict[
        backwards_opt.INIT_FUNCTION_KEY]

    if isinstance(init_function_name_or_matrices, str):
        print SEPARATOR_STRING
        _plot_examples(
            list_of_predictor_matrices=list_of_optimized_input_matrices,
            training_option_dict=training_option_dict, optimized=True,
            output_dir_name=top_output_dir_name)

        return

    print 'Denormalizing original input data...'
    init_function_name_or_matrices = model_interpretation.denormalize_data(
        list_of_input_matrices=init_function_name_or_matrices,
        model_metadata_dict=model_metadata_dict)
    print SEPARATOR_STRING

    original_output_dir_name = '{0:s}/original'.format(top_output_dir_name)
    _plot_examples(
        list_of_predictor_matrices=init_function_name_or_matrices,
        training_option_dict=training_option_dict, optimized=False,
        output_dir_name=original_output_dir_name)
    print SEPARATOR_STRING

    optimized_output_dir_name = '{0:s}/optimized'.format(top_output_dir_name)
    _plot_examples(
        list_of_predictor_matrices=list_of_optimized_input_matrices,
        training_option_dict=training_option_dict, optimized=True,
        output_dir_name=optimized_output_dir_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
