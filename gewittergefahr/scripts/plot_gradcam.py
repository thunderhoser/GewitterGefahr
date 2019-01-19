"""Plots Grad-CAM output (class-activation maps)."""

import os
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import gradcam
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import radar_plotting
from gewittergefahr.plotting import cam_plotting

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_CONTOURS = 12

TITLE_FONT_SIZE = 20
FONT_SIZE_WITH_COLOUR_BARS = 16
FONT_SIZE_SANS_COLOUR_BARS = 20
FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_file_name'
COLOUR_MAP_ARG_NAME = 'cam_colour_map_name'
MAX_PERCENTILE_ARG_NAME = 'max_colour_prctile_for_cam'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `gradcam.read_standard_file`.')

COLOUR_MAP_HELP_STRING = (
    'Name of colour map for class activations.  The same colour map will be '
    'used for all predictors and examples.  This argument supports only pyplot '
    'colour maps (those accepted by `pyplot.cm.get_cmap`).')

MAX_PERCENTILE_HELP_STRING = (
    'Determines max value in colour scheme for each class-activation map (CAM).'
    '  The max value for the [i]th example will be the [q]th percentile of all '
    'class activations for the [i]th example, where q = `{0:s}`.'
).format(MAX_PERCENTILE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Figures will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + COLOUR_MAP_ARG_NAME, type=str, required=False, default='gist_yarg',
    help=COLOUR_MAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False,
    default=99, help=MAX_PERCENTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _plot_3d_radar_cams(
        radar_matrix, cam_metadata_dict, model_metadata_dict,
        cam_colour_map_object, max_colour_prctile_for_cam, output_dir_name,
        class_activation_matrix=None, ggradcam_output_matrix=None):
    """Plots class-activation maps for 3-D radar data.

    E = number of examples
    M = number of rows in spatial grid
    N = number of columns in spatial grid
    H = number of heights in spatial grid
    F = number of radar fields

    This method will plot either `class_activation_matrix` or
    `ggradcam_output_matrix`, not both.

    :param radar_matrix: E-by-M-by-N-by-H-by-F numpy array of radar values.
    :param cam_metadata_dict: Dictionary with CAM metadata (see doc for
        `gradcam.read_standard_file`).
    :param model_metadata_dict: Dictionary with CNN metadata (see doc for
        `cnn.read_model_metadata`).
    :param cam_colour_map_object: See documentation at top of file.
    :param max_colour_prctile_for_cam: Same.
    :param output_dir_name: Same.
    :param class_activation_matrix: E-by-M-by-N-by-H numpy array of class
        activations.
    :param ggradcam_output_matrix: E-by-M-by-N-by-H-by-F numpy array of output
        values from guided Grad-CAM.
    """

    num_examples = radar_matrix.shape[0]
    num_heights = radar_matrix.shape[-2]
    num_fields = radar_matrix.shape[-1]
    num_panel_rows = int(numpy.floor(numpy.sqrt(num_heights)))

    if class_activation_matrix is None:
        quantity_string = 'max guided Grad-CAM output'
        pathless_file_name_prefix = 'guided-gradcam'
    else:
        quantity_string = 'max class activation'
        pathless_file_name_prefix = 'gradcam'

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]

    for i in range(num_examples):
        this_storm_id = cam_metadata_dict[gradcam.STORM_IDS_KEY][i]
        this_storm_time_string = time_conversion.unix_sec_to_string(
            cam_metadata_dict[gradcam.STORM_TIMES_KEY][i], TIME_FORMAT)

        for k in range(num_fields):
            this_field_name = training_option_dict[
                trainval_io.RADAR_FIELDS_KEY][k]

            _, these_axes_objects = radar_plotting.plot_3d_grid_without_coords(
                field_matrix=numpy.flip(radar_matrix[i, ..., k], axis=0),
                field_name=this_field_name,
                grid_point_heights_metres=training_option_dict[
                    trainval_io.RADAR_HEIGHTS_KEY],
                ground_relative=True, num_panel_rows=num_panel_rows,
                font_size=FONT_SIZE_SANS_COLOUR_BARS)

            if class_activation_matrix is None:
                this_matrix = ggradcam_output_matrix[i, ..., k]
            else:
                this_matrix = class_activation_matrix[i, ...]

            this_max_contour_level = numpy.percentile(
                this_matrix, max_colour_prctile_for_cam)

            cam_plotting.plot_many_2d_grids(
                class_activation_matrix_3d=numpy.flip(this_matrix, axis=0),
                axes_objects_2d_list=these_axes_objects,
                colour_map_object=cam_colour_map_object,
                max_contour_level=this_max_contour_level,
                contour_interval=this_max_contour_level / NUM_CONTOURS)

            this_colour_map_object, this_colour_norm_object = (
                radar_plotting.get_default_colour_scheme(this_field_name)
            )

            plotting_utils.add_colour_bar(
                axes_object_or_list=these_axes_objects,
                values_to_colour=radar_matrix[i, ..., k],
                colour_map=this_colour_map_object,
                colour_norm_object=this_colour_norm_object,
                orientation='horizontal', extend_min=True, extend_max=True)

            this_title_string = (
                'Storm "{0:s}" at {1:s} ({2:s} = {3:.3f})'
            ).format(
                this_storm_id, this_storm_time_string, quantity_string,
                this_max_contour_level
            )

            pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)

            this_figure_file_name = '{0:s}/{1:s}_{2:s}_{3:s}_{4:s}.jpg'.format(
                output_dir_name, pathless_file_name_prefix,
                this_storm_id.replace('_', '-'), this_storm_time_string,
                this_field_name.replace('_', '-')
            )

            print 'Saving figure to file: "{0:s}"...'.format(
                this_figure_file_name)
            pyplot.savefig(this_figure_file_name, dpi=FIGURE_RESOLUTION_DPI)
            pyplot.close()


def _plot_2d_radar_cams(
        radar_matrix, cam_metadata_dict, model_metadata_dict,
        cam_colour_map_object, max_colour_prctile_for_cam, output_dir_name,
        class_activation_matrix=None, ggradcam_output_matrix=None):
    """Plots class-activation maps for 2-D radar data.

    E = number of examples
    M = number of rows in spatial grid
    N = number of columns in spatial grid
    C = number of channels (field/height pairs)

    This method will plot either `class_activation_matrix` or
    `ggradcam_output_matrix`, not both.

    :param radar_matrix: E-by-M-by-N-by-C numpy array of radar values.
    :param cam_metadata_dict: See doc for `_plot_3d_radar_cams`.
    :param model_metadata_dict: Same.
    :param cam_colour_map_object: Same.
    :param max_colour_prctile_for_cam: Same.
    :param output_dir_name: Same.
    :param class_activation_matrix: E-by-M-by-N numpy array of class
        activations.
    :param ggradcam_output_matrix: E-by-M-by-N-by-C numpy array of output values
        from guided Grad-CAM.
    """

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    list_of_layer_operation_dicts = model_metadata_dict[
        cnn.LAYER_OPERATIONS_KEY]

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

    num_examples = radar_matrix.shape[0]
    num_channels = radar_matrix.shape[-1]
    num_panel_rows = int(numpy.floor(numpy.sqrt(num_channels)))

    if class_activation_matrix is None:
        quantity_string = 'max guided Grad-CAM output'
        pathless_file_name_prefix = 'guided-gradcam'
    else:
        quantity_string = 'max class activation'
        pathless_file_name_prefix = 'gradcam'

    for i in range(num_examples):
        this_storm_id = cam_metadata_dict[gradcam.STORM_IDS_KEY][i]
        this_storm_time_string = time_conversion.unix_sec_to_string(
            cam_metadata_dict[gradcam.STORM_TIMES_KEY][i], TIME_FORMAT)

        _, these_axes_objects = (
            radar_plotting.plot_many_2d_grids_without_coords(
                field_matrix=numpy.flip(radar_matrix[i, ...], axis=0),
                field_name_by_panel=field_name_by_panel,
                num_panel_rows=num_panel_rows, panel_names=panel_names,
                font_size=FONT_SIZE_WITH_COLOUR_BARS, plot_colour_bars=False)
        )

        if class_activation_matrix is None:
            this_matrix = ggradcam_output_matrix[i, ...]
        else:
            this_matrix = numpy.expand_dims(
                class_activation_matrix[i, ...], axis=-1)
            this_matrix = numpy.repeat(
                this_matrix, repeats=num_channels, axis=-1)

        this_max_contour_level = numpy.percentile(
            this_matrix, max_colour_prctile_for_cam)

        cam_plotting.plot_many_2d_grids(
            class_activation_matrix_3d=numpy.flip(this_matrix, axis=0),
            axes_objects_2d_list=these_axes_objects,
            colour_map_object=cam_colour_map_object,
            max_contour_level=this_max_contour_level,
            contour_interval=this_max_contour_level / NUM_CONTOURS)

        this_title_string = 'Storm "{0:s}" at {1:s} ({2:s} = {3:.3f})'.format(
            this_storm_id, this_storm_time_string, quantity_string,
            this_max_contour_level)

        pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)

        this_figure_file_name = '{0:s}/{1:s}_{2:s}_{3:s}_radar.jpg'.format(
            output_dir_name, pathless_file_name_prefix,
            this_storm_id.replace('_', '-'), this_storm_time_string)

        print 'Saving figure to file: "{0:s}"...'.format(this_figure_file_name)
        pyplot.savefig(this_figure_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()


def _run(input_file_name, cam_colour_map_name, max_colour_prctile_for_cam,
         top_output_dir_name):
    """Plots Grad-CAM output (class-activation maps).

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param cam_colour_map_name: Same.
    :param max_colour_prctile_for_cam: Same.
    :param top_output_dir_name: Same.
    :raises: TypeError: if input file contains class-activation maps for
        soundings.
    """

    main_gradcam_dir_name = '{0:s}/main_gradcam'.format(top_output_dir_name)
    guided_gradcam_dir_name = '{0:s}/guided_gradcam'.format(top_output_dir_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=main_gradcam_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=guided_gradcam_dir_name)

    # Check input args.
    error_checking.assert_is_geq(max_colour_prctile_for_cam, 0.)
    error_checking.assert_is_leq(max_colour_prctile_for_cam, 100.)
    cam_colour_map_object = pyplot.cm.get_cmap(cam_colour_map_name)

    print 'Reading data from: "{0:s}"...'.format(input_file_name)
    gradcam_dict = gradcam.read_standard_file(input_file_name)

    list_of_input_matrices = gradcam_dict.pop(gradcam.INPUT_MATRICES_KEY)
    class_activation_matrix = gradcam_dict.pop(gradcam.CLASS_ACTIVATIONS_KEY)
    ggradcam_output_matrix = gradcam_dict.pop(gradcam.GUIDED_GRADCAM_KEY)
    cam_metadata_dict = gradcam_dict

    num_spatial_dimensions = len(class_activation_matrix.shape) - 1
    if num_spatial_dimensions == 1:
        raise TypeError('This script is not yet equipped to plot '
                        'class-activation maps for soundings.')

    # Read metadata for CNN.
    model_file_name = cam_metadata_dict[gradcam.MODEL_FILE_NAME_KEY]
    model_metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0]
    )

    print 'Reading model metadata from: "{0:s}"...'.format(model_metafile_name)
    model_metadata_dict = cnn.read_model_metadata(model_metafile_name)
    print SEPARATOR_STRING

    # Do plotting.
    if num_spatial_dimensions == 3:
        _plot_3d_radar_cams(
            radar_matrix=list_of_input_matrices[0],
            class_activation_matrix=class_activation_matrix,
            cam_metadata_dict=cam_metadata_dict,
            model_metadata_dict=model_metadata_dict,
            cam_colour_map_object=cam_colour_map_object,
            max_colour_prctile_for_cam=max_colour_prctile_for_cam,
            output_dir_name=main_gradcam_dir_name)
        print SEPARATOR_STRING

        _plot_3d_radar_cams(
            radar_matrix=list_of_input_matrices[0],
            ggradcam_output_matrix=ggradcam_output_matrix,
            cam_metadata_dict=cam_metadata_dict,
            model_metadata_dict=model_metadata_dict,
            cam_colour_map_object=cam_colour_map_object,
            max_colour_prctile_for_cam=max_colour_prctile_for_cam,
            output_dir_name=guided_gradcam_dir_name)

    else:
        if len(list_of_input_matrices) == 3:
            radar_matrix = list_of_input_matrices[1]
        else:
            radar_matrix = list_of_input_matrices[0]

        _plot_2d_radar_cams(
            radar_matrix=radar_matrix,
            class_activation_matrix=class_activation_matrix,
            cam_metadata_dict=cam_metadata_dict,
            model_metadata_dict=model_metadata_dict,
            cam_colour_map_object=cam_colour_map_object,
            max_colour_prctile_for_cam=max_colour_prctile_for_cam,
            output_dir_name=main_gradcam_dir_name)
        print SEPARATOR_STRING

        _plot_2d_radar_cams(
            radar_matrix=radar_matrix,
            ggradcam_output_matrix=ggradcam_output_matrix,
            cam_metadata_dict=cam_metadata_dict,
            model_metadata_dict=model_metadata_dict,
            cam_colour_map_object=cam_colour_map_object,
            max_colour_prctile_for_cam=max_colour_prctile_for_cam,
            output_dir_name=guided_gradcam_dir_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        cam_colour_map_name=getattr(INPUT_ARG_OBJECT, COLOUR_MAP_ARG_NAME),
        max_colour_prctile_for_cam=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
