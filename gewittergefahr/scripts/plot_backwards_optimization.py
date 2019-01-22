"""Plots results of backwards optimization."""

import os.path
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.deep_learning import backwards_optimization as backwards_opt
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import radar_plotting
from gewittergefahr.plotting import sounding_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

METRES_TO_KM = 0.001
TIME_FORMAT = '%Y-%m-%d-%H%M%S'

TITLE_FONT_SIZE = 20
FONT_SIZE_WITH_COLOUR_BARS = 16
FONT_SIZE_SANS_COLOUR_BARS = 20
FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_file_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `backwards_opt.read_standard_file` or'
    ' `backwards_opt.read_pmm_file`.')

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory.  Figures will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _plot_inputs_or_outputs(
        list_of_predictor_matrices, model_metadata_dict, output_dir_name,
        optimized_flag, pmm_flag, storm_ids=None, storm_times_unix_sec=None):
    """Plots either inputs or outputs (before or after backwards optimization).

    T = number of input tensors to the model
    E = number of examples (storm objects)

    :param list_of_predictor_matrices: length-T list of numpy arrays, where the
        [i]th array is either the optimized or non-optimized version of the
        [i]th input matrix to the model.
    :param model_metadata_dict: Dictionary returned by `cnn.read_model_metadata`.
    :param output_dir_name: Path to output directory (figures will be saved
        here).
    :param optimized_flag: Boolean flag.  If True, `list_of_predictor_matrices`
        contains optimized examples.  If False, `list_of_predictor_matrices`
        contains input examples (before optimization).
    :param pmm_flag: Boolean flag.  If True, `list_of_predictor_matrices`
        contains probability-matched means.
    :param storm_ids: [optional and used only if `pmm_flag = False`]
        length-E list of storm IDs (strings).
    :param storm_times_unix_sec: [optional and used only if `pmm_flag = False`]
        length-E numpy array of storm times.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    if pmm_flag:
        have_storm_ids = False
    else:
        have_storm_ids = not (storm_ids is None or storm_times_unix_sec is None)

    if optimized_flag:
        optimized_flag_as_string = 'after optimization'
    else:
        optimized_flag_as_string = 'before optimization'

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

    for i in range(num_storms):
        if pmm_flag:
            this_title_string = 'Probability-matched mean'
            this_base_file_name = '{0:s}/pmm_optimized={1:d}'.format(
                output_dir_name, int(optimized_flag)
            )

        else:
            if have_storm_ids:
                this_storm_time_string = time_conversion.unix_sec_to_string(
                    storm_times_unix_sec[i], TIME_FORMAT)

                this_title_string = 'Storm "{0:s}" at {1:s}'.format(
                    storm_ids[i], this_storm_time_string)

                this_base_file_name = (
                    '{0:s}/{1:s}_{2:s}_optimized={3:d}'
                ).format(
                    output_dir_name, storm_ids[i].replace('_', '-'),
                    this_storm_time_string, int(optimized_flag)
                )

            else:
                this_title_string = 'Example {0:d}'.format(i + 1)

                this_base_file_name = (
                    '{0:s}/example{1:06d}_optimized={2:d}'
                ).format(
                    output_dir_name, i, int(optimized_flag)
                )

        this_title_string += ' ({0:s})'.format(optimized_flag_as_string)

        if plot_soundings:
            this_file_name = '{0:s}_sounding.jpg'.format(this_base_file_name)
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
            this_num_panel_rows = int(numpy.floor(
                numpy.sqrt(this_num_heights)
            ))

            _, these_axes_objects = radar_plotting.plot_3d_grid_without_coords(
                field_matrix=this_reflectivity_matrix_dbz,
                field_name=radar_utils.REFL_NAME,
                grid_point_heights_metres=training_option_dict[
                    trainval_io.RADAR_HEIGHTS_KEY],
                ground_relative=True, num_panel_rows=this_num_panel_rows,
                font_size=FONT_SIZE_SANS_COLOUR_BARS)

            this_colour_map_object, this_colour_norm_object = (
                radar_plotting.get_default_colour_scheme(radar_utils.REFL_NAME)
            )

            plotting_utils.add_colour_bar(
                axes_object_or_list=these_axes_objects,
                values_to_colour=this_reflectivity_matrix_dbz,
                colour_map=this_colour_map_object,
                colour_norm_object=this_colour_norm_object,
                orientation='horizontal', extend_min=True, extend_max=True)

            pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)
            this_file_name = '{0:s}_reflectivity.jpg'.format(
                this_base_file_name)

            print 'Saving figure to: "{0:s}"...'.format(this_file_name)
            pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
            pyplot.close()

            this_az_shear_matrix_s01 = numpy.flip(
                list_of_predictor_matrices[1][i, ..., 0], axis=0)

            _, these_axes_objects = (
                radar_plotting.plot_many_2d_grids_without_coords(
                    field_matrix=this_az_shear_matrix_s01,
                    field_name_by_panel=training_option_dict[
                        trainval_io.RADAR_FIELDS_KEY],
                    num_panel_rows=1,
                    panel_names=training_option_dict[
                        trainval_io.RADAR_FIELDS_KEY],
                    font_size=FONT_SIZE_SANS_COLOUR_BARS,
                    plot_colour_bars=False)
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

            pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)
            this_file_name = '{0:s}_azimuthal-shear.jpg'.format(
                this_base_file_name)

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
                if list_of_layer_operation_dicts is None:
                    field_name_by_panel = training_option_dict[
                        trainval_io.RADAR_FIELDS_KEY]

                    panel_names = (
                        radar_plotting.radar_fields_and_heights_to_panel_names(
                            field_names=field_name_by_panel,
                            heights_m_agl=training_option_dict[
                                trainval_io.RADAR_HEIGHTS_KEY]
                        )
                    )
                else:
                    field_name_by_panel, panel_names = (
                        radar_plotting.layer_ops_to_field_and_panel_names(
                            list_of_layer_operation_dicts)
                    )

                this_num_predictors = this_radar_matrix.shape[-1]
                this_num_panel_rows = int(numpy.floor(
                    numpy.sqrt(this_num_predictors)
                ))

                radar_plotting.plot_many_2d_grids_without_coords(
                    field_matrix=numpy.flip(this_radar_matrix[i, ...], axis=0),
                    field_name_by_panel=field_name_by_panel,
                    num_panel_rows=this_num_panel_rows, panel_names=panel_names,
                    font_size=FONT_SIZE_WITH_COLOUR_BARS, plot_colour_bars=True)

                this_file_name = '{0:s}_radar.jpg'.format(this_base_file_name)

            else:
                radar_field_names = training_option_dict[
                    trainval_io.RADAR_FIELDS_KEY]
                radar_heights_m_agl = training_option_dict[
                    trainval_io.RADAR_HEIGHTS_KEY]

                this_num_heights = this_radar_matrix.shape[-2]
                this_num_panel_rows = int(numpy.floor(
                    numpy.sqrt(this_num_heights)
                ))

                _, these_axes_objects = (
                    radar_plotting.plot_3d_grid_without_coords(
                        field_matrix=numpy.flip(
                            this_radar_matrix[i, ..., j], axis=0),
                        field_name=radar_field_names[j],
                        grid_point_heights_metres=radar_heights_m_agl,
                        ground_relative=True,
                        num_panel_rows=this_num_panel_rows,
                        font_size=FONT_SIZE_SANS_COLOUR_BARS)
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

                this_file_name = '{0:s}_{1:s}.jpg'.format(
                    this_base_file_name, radar_field_names[j].replace('_', '-')
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
    pmm_flag = False

    try:
        backwards_opt_dict = backwards_opt.read_standard_file(input_file_name)
        list_of_optimized_matrices = backwards_opt_dict.pop(
            backwards_opt.OPTIMIZED_MATRICES_KEY)
        list_of_input_matrices = backwards_opt_dict.pop(
            backwards_opt.INIT_FUNCTION_KEY)

        if not isinstance(list_of_input_matrices, list):
            list_of_input_matrices = None

        bwo_metadata_dict = backwards_opt_dict
        storm_ids = bwo_metadata_dict[backwards_opt.STORM_IDS_KEY]
        storm_times_unix_sec = bwo_metadata_dict[backwards_opt.STORM_TIMES_KEY]

    except ValueError:
        pmm_flag = True
        backwards_opt_dict = backwards_opt.read_pmm_file(input_file_name)

        list_of_input_matrices = backwards_opt_dict.pop(
            backwards_opt.MEAN_INPUT_MATRICES_KEY)
        list_of_optimized_matrices = backwards_opt_dict.pop(
            backwards_opt.MEAN_OPTIMIZED_MATRICES_KEY)

        for i in range(len(list_of_input_matrices)):
            list_of_input_matrices[i] = numpy.expand_dims(
                list_of_input_matrices[i], axis=0)
            list_of_optimized_matrices[i] = numpy.expand_dims(
                list_of_optimized_matrices[i], axis=0)

        original_bwo_file_name = backwards_opt_dict[
            backwards_opt.STANDARD_FILE_NAME_KEY]

        print 'Reading metadata from: "{0:s}"...'.format(
            original_bwo_file_name)
        original_bwo_dict = backwards_opt.read_standard_file(
            original_bwo_file_name)

        original_bwo_dict.pop(backwards_opt.OPTIMIZED_MATRICES_KEY)
        original_bwo_dict.pop(backwards_opt.INIT_FUNCTION_KEY)
        bwo_metadata_dict = original_bwo_dict

        storm_ids = None
        storm_times_unix_sec = None

    model_file_name = bwo_metadata_dict[backwards_opt.MODEL_FILE_NAME_KEY]
    model_metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0]
    )

    print 'Reading metadata from: "{0:s}"...'.format(model_metafile_name)
    model_metadata_dict = cnn.read_model_metadata(model_metafile_name)

    print SEPARATOR_STRING

    if list_of_input_matrices is None:
        _plot_inputs_or_outputs(
            list_of_predictor_matrices=list_of_optimized_matrices,
            model_metadata_dict=model_metadata_dict,
            output_dir_name=top_output_dir_name,
            optimized_flag=True, pmm_flag=pmm_flag,
            storm_ids=storm_ids, storm_times_unix_sec=storm_times_unix_sec)
        return

    before_dir_name = '{0:s}/before_optimization'.format(top_output_dir_name)
    _plot_inputs_or_outputs(
        list_of_predictor_matrices=list_of_input_matrices,
        model_metadata_dict=model_metadata_dict,
        output_dir_name=before_dir_name,
        optimized_flag=False, pmm_flag=pmm_flag,
        storm_ids=storm_ids, storm_times_unix_sec=storm_times_unix_sec)
    print SEPARATOR_STRING

    after_dir_name = '{0:s}/after_optimization'.format(top_output_dir_name)
    _plot_inputs_or_outputs(
        list_of_predictor_matrices=list_of_input_matrices,
        model_metadata_dict=model_metadata_dict,
        output_dir_name=after_dir_name,
        optimized_flag=True, pmm_flag=pmm_flag,
        storm_ids=storm_ids, storm_times_unix_sec=storm_times_unix_sec)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
