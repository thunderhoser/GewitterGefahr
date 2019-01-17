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
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.deep_learning import model_interpretation
from gewittergefahr.deep_learning import backwards_optimization as backwards_opt
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import radar_plotting
from gewittergefahr.plotting import sounding_plotting

METRES_TO_KM = 0.001
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

FONT_SIZE = 16
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


def _layer_ops_to_field_names_and_annotations(list_of_layer_operation_dicts):
    """Converts list of layer operations to list of field names and annotations.

    C = number of layer operations = number of radar channels

    :param list_of_layer_operation_dicts: See doc for
        `input_examples.reduce_examples_3d_to_2d`.
    :return: field_name_by_channel: length-C list with names of radar fields.
    :return: annotation_string_by_channel: length-C list of annotations (to be
        printed at bottoms of figure panels).
    """

    # TODO(thunderhoser): Put this method somewhere else.

    num_channels = len(list_of_layer_operation_dicts)
    field_name_by_channel = [''] * num_channels
    annotation_string_by_channel = [''] * num_channels

    for m in range(num_channels):
        this_operation_dict = list_of_layer_operation_dicts[m]
        field_name_by_channel[m] = this_operation_dict[
            input_examples.RADAR_FIELD_KEY]

        this_min_height_m_agl = int(numpy.round(
            this_operation_dict[input_examples.MIN_HEIGHT_KEY] * METRES_TO_KM
        ))
        this_max_height_m_agl = int(numpy.round(
            this_operation_dict[input_examples.MAX_HEIGHT_KEY] * METRES_TO_KM
        ))

        annotation_string_by_channel[m] = (
            '{0:s}\n{1:s} from {2:d}-{3:d} km AGL'
        ).format(
            field_name_by_channel[m],
            this_operation_dict[input_examples.OPERATION_NAME_KEY].upper(),
            this_min_height_m_agl, this_max_height_m_agl
        )

    return field_name_by_channel, annotation_string_by_channel


def _radar_fields_and_heights_to_annotations(field_name_by_channel,
                                             height_by_channel_m_agl):
    """Converts list of radar fields and heights to annotations.

    C = number of channels (field/height pairs)

    :param field_name_by_channel: length-C list with names of radar fields.
    :param height_by_channel_m_agl: length-C numpy array of radar heights
        (metres above ground level).
    :return: annotation_string_by_channel: length-C list of annotations (to be
        printed at bottoms of figure panels).
    """

    # TODO(thunderhoser): Put this method somewhere else.

    num_channels = len(field_name_by_channel)
    annotation_string_by_channel = [''] * num_channels

    for m in range(num_channels):
        annotation_string_by_channel[m] = '{0:s}\nat {1:.2f} km AGL'.format(
            field_name_by_channel[m], height_by_channel_m_agl[m] * METRES_TO_KM
        )

    return annotation_string_by_channel


def _plot_examples(list_of_predictor_matrices, model_metadata_dict, optimized,
                   output_dir_name):
    """Plots one or more (either original or optimized) examples.

    :param list_of_predictor_matrices: List created by
        `testing_io.read_specific_examples`.  Contains data to be plotted.
    :param model_metadata_dict: See doc for `cnn.read_model_metadata`.
    :param optimized: Boolean flag.  If True, `list_of_predictor_matrices`
        contains optimized input examples.  If False, contains original examples
        (pre-optimization).  This piece of metadata will be reflected in file
        names and titles.
    :param output_dir_name: Name of output directory (figures will be saved
        here).
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    myrorss_2d3d = model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]
    list_of_layer_operation_dicts = model_metadata_dict[
        cnn.LAYER_OPERATIONS_KEY]

    sounding_field_names = training_option_dict[trainval_io.SOUNDING_FIELDS_KEY]
    plot_soundings = sounding_field_names is not None
    num_storms = list_of_predictor_matrices[0].shape[0]

    if plot_soundings:
        list_of_metpy_dictionaries = dl_utils.soundings_to_metpy_dictionaries(
            sounding_matrix=list_of_predictor_matrices[-1],
            field_names=sounding_field_names,
            height_levels_m_agl=training_option_dict[
                trainval_io.SOUNDING_HEIGHTS_KEY],
            storm_elevations_m_asl=numpy.zeros(num_storms))
    else:
        list_of_metpy_dictionaries = None

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
            this_reflectivity_matrix_dbz = numpy.flip(
                list_of_predictor_matrices[0][i, ..., 0], axis=0)

            this_num_heights = this_reflectivity_matrix_dbz.shape[-1]
            this_num_panel_rows = int(numpy.floor(numpy.sqrt(this_num_heights)))

            _, these_axes_objects = radar_plotting.plot_3d_grid_without_coords(
                field_matrix=this_reflectivity_matrix_dbz,
                field_name=radar_utils.REFL_NAME,
                grid_point_heights_metres=training_option_dict[
                    trainval_io.RADAR_HEIGHTS_KEY],
                ground_relative=True, num_panel_rows=this_num_panel_rows,
                font_size=FONT_SIZE)

            this_colour_map_object, this_colour_norm_object = (
                radar_plotting.get_default_colour_scheme(radar_utils.REFL_NAME)
            )

            plotting_utils.add_colour_bar(
                axes_object_or_list=these_axes_objects,
                values_to_colour=this_reflectivity_matrix_dbz,
                colour_map=this_colour_map_object,
                colour_norm_object=this_colour_norm_object,
                orientation='horizontal', extend_min=True, extend_max=True)

            this_file_name = (
                '{0:s}/example{1:06d}_optimized={2:d}_reflectivity.jpg'
            ).format(output_dir_name, i, int(optimized))

            pyplot.suptitle(this_title_string, fontsize=FONT_SIZE)
            print 'Saving figure to: "{0:s}"...'.format(this_file_name)
            pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
            pyplot.close()

            this_az_shear_matrix_s01 = numpy.flip(
                list_of_predictor_matrices[1][i, ..., 0], axis=0)

            _, these_axes_objects = (
                radar_plotting.plot_many_2d_grids_without_coords(
                    field_matrix=this_az_shear_matrix_s01,
                    field_name_by_channel=training_option_dict[
                        trainval_io.RADAR_FIELDS_KEY],
                    annotation_string_by_channel=training_option_dict[
                        trainval_io.RADAR_FIELDS_KEY],
                    num_panel_rows=1, plot_colour_bars=False,
                    font_size=FONT_SIZE)
            )

            this_colour_map_object, this_colour_norm_object = (
                radar_plotting.get_default_colour_scheme(
                    radar_utils.LOW_LEVEL_SHEAR_NAME)
            )

            plotting_utils.add_colour_bar(
                axes_object_or_list=these_axes_objects,
                values_to_colour=this_az_shear_matrix_s01,
                colour_map=this_colour_map_object,
                colour_norm_object=this_colour_norm_object,
                orientation='horizontal', extend_min=True, extend_max=True)

            this_file_name = (
                '{0:s}/example{1:06d}_optimized={2:d}_azimuthal-shear.jpg'
            ).format(output_dir_name, i, int(optimized))

            pyplot.suptitle(this_title_string, fontsize=FONT_SIZE)
            print 'Saving figure to: "{0:s}"...'.format(this_file_name)
            pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
            pyplot.close()

            continue

        this_radar_matrix = list_of_predictor_matrices[0]
        num_radar_dimensions = len(list_of_predictor_matrices[0].shape) - 2

        if num_radar_dimensions == 2:
            j_max = 1
        else:
            j_max = len(training_option_dict[trainval_io.RADAR_FIELDS_KEY])

        for j in range(j_max):
            if num_radar_dimensions == 2:
                if list_of_layer_operation_dicts:
                    field_name_by_channel, annotation_string_by_channel = (
                        _layer_ops_to_field_names_and_annotations(
                            list_of_layer_operation_dicts)
                    )
                else:
                    field_name_by_channel = training_option_dict[
                        trainval_io.RADAR_FIELDS_KEY]

                    annotation_string_by_channel = (
                        _radar_fields_and_heights_to_annotations(
                            field_name_by_channel=field_name_by_channel,
                            height_by_channel_m_agl=training_option_dict[
                                trainval_io.RADAR_HEIGHTS_KEY]
                        )
                    )

                this_num_predictors = this_radar_matrix.shape[-1]
                this_num_panel_rows = int(
                    numpy.floor(numpy.sqrt(this_num_predictors))
                )

                radar_plotting.plot_many_2d_grids_without_coords(
                    field_matrix=numpy.flip(this_radar_matrix[i, ...], axis=0),
                    field_name_by_channel=field_name_by_channel,
                    annotation_string_by_channel=annotation_string_by_channel,
                    num_panel_rows=this_num_panel_rows, plot_colour_bars=True,
                    font_size=FONT_SIZE)

                this_file_name = (
                    '{0:s}/example{1:06d}_optimized={2:d}_radar.jpg'
                ).format(output_dir_name, i, int(optimized))

            else:
                radar_field_names = training_option_dict[
                    trainval_io.RADAR_FIELDS_KEY]
                radar_heights_m_agl = training_option_dict[
                    trainval_io.RADAR_HEIGHTS_KEY]

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
                        num_panel_rows=this_num_panel_rows, font_size=FONT_SIZE)
                )

                this_colour_map_object, this_colour_norm_object = (
                    radar_plotting.get_default_colour_scheme(
                        radar_field_names[j])
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

            pyplot.suptitle(this_title_string, fontsize=FONT_SIZE)
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
            model_metadata_dict=model_metadata_dict, optimized=True,
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
        model_metadata_dict=model_metadata_dict, optimized=False,
        output_dir_name=original_output_dir_name)
    print SEPARATOR_STRING

    optimized_output_dir_name = '{0:s}/optimized'.format(top_output_dir_name)
    _plot_examples(
        list_of_predictor_matrices=list_of_optimized_input_matrices,
        model_metadata_dict=model_metadata_dict, optimized=True,
        output_dir_name=optimized_output_dir_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
