"""Plots Grad-CAM output (guided and unguided class-activation maps)."""

import os
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import general_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import gradcam
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import cam_plotting
from gewittergefahr.plotting import saliency_plotting
from gewittergefahr.scripts import plot_input_examples as plot_examples

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MIN_UNGUIDED_VALUE_LOG10 = -2.

COLOUR_BAR_FONT_SIZE = plot_examples.DEFAULT_CBAR_FONT_SIZE
FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_file_name'
COLOUR_MAP_ARG_NAME = 'colour_map_name'
MAX_UNGUIDED_VALUE_ARG_NAME = 'max_unguided_value'
NUM_UNGUIDED_CONTOURS_ARG_NAME = 'num_unguided_contours'
MAX_GUIDED_VALUE_ARG_NAME = 'max_guided_value'
HALF_NUM_GUIDED_CONTOURS_ARG_NAME = 'half_num_guided_contours'
SMOOTHING_RADIUS_ARG_NAME = 'smoothing_radius_grid_cells'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'
ALLOW_WHITESPACE_ARG_NAME = plot_examples.ALLOW_WHITESPACE_ARG_NAME
PLOT_PANEL_NAMES_ARG_NAME = plot_examples.PLOT_PANEL_NAMES_ARG_NAME
ADD_TITLES_ARG_NAME = plot_examples.ADD_TITLES_ARG_NAME
LABEL_CBARS_ARG_NAME = plot_examples.LABEL_CBARS_ARG_NAME
CBAR_LENGTH_ARG_NAME = plot_examples.CBAR_LENGTH_ARG_NAME

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `gradcam.read_file`.')

COLOUR_MAP_HELP_STRING = (
    'Name of colour map for class activations.  The same colour map will be '
    'used for all predictors and examples.  This argument supports only pyplot '
    'colour maps (those accepted by `pyplot.get_cmap`).')

MAX_UNGUIDED_VALUE_HELP_STRING = (
    'Max value in colour scheme for unguided CAMs.  Keep in mind that unguided '
    'class activation >= 0 always.')

NUM_UNGUIDED_CONTOURS_HELP_STRING = 'Number of contours for unguided CAMs.'

MAX_GUIDED_VALUE_HELP_STRING = (
    'Max value in colour scheme for guided CAMs.  Keep in mind that the colour '
    'scheme encodes *absolute* value, with positive values in solid contours '
    'and negative values in dashed contours.')

HALF_NUM_GUIDED_CONTOURS_HELP_STRING = (
    'Number of contours on each side of zero for guided CAMs.')

SMOOTHING_RADIUS_HELP_STRING = (
    'e-folding radius for Gaussian smoother (num grid cells).  If you do not '
    'want to smooth class-activation maps, make this non-positive.')

OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Figures will be saved here.')

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
    '--' + COLOUR_MAP_ARG_NAME, type=str, required=False, default='gist_yarg',
    help=COLOUR_MAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_UNGUIDED_VALUE_ARG_NAME, type=float, required=False,
    default=10 ** 1.5, help=MAX_UNGUIDED_VALUE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_UNGUIDED_CONTOURS_ARG_NAME, type=int, required=False,
    default=15, help=NUM_UNGUIDED_CONTOURS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_GUIDED_VALUE_ARG_NAME, type=float, required=False,
    default=1.25, help=MAX_GUIDED_VALUE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + HALF_NUM_GUIDED_CONTOURS_ARG_NAME, type=int, required=False,
    default=10, help=HALF_NUM_GUIDED_CONTOURS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SMOOTHING_RADIUS_ARG_NAME, type=float, required=False,
    default=2., help=SMOOTHING_RADIUS_HELP_STRING)

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


def _plot_3d_radar_cam(
        colour_map_object, max_unguided_value, num_unguided_contours,
        max_guided_value, half_num_guided_contours, label_colour_bars,
        colour_bar_length, figure_objects, axes_object_matrices,
        model_metadata_dict, output_dir_name, cam_matrix=None,
        guided_cam_matrix=None, full_storm_id_string=None,
        storm_time_unix_sec=None):
    """Plots class-activation map for 3-D radar data.

    M = number of rows in spatial grid
    N = number of columns in spatial grid
    H = number of heights in spatial grid
    F = number of radar fields

    If this method is plotting a composite rather than single example (storm
    object), `full_storm_id_string` and `storm_time_unix_sec` can be None.

    :param colour_map_object: See documentation at top of file.
    :param max_unguided_value: Same.
    :param num_unguided_contours: Same.
    :param max_guided_value: Same.
    :param half_num_guided_contours: Same.
    :param label_colour_bars: Same.
    :param colour_bar_length: Same.
    :param figure_objects: See doc for
        `plot_input_examples._plot_3d_radar_scan`.
    :param axes_object_matrices: Same.
    :param model_metadata_dict: Dictionary returned by
        `cnn.read_model_metadata`.
    :param output_dir_name: Path to output directory.  Figure(s) will be saved
        here.
    :param cam_matrix: M-by-N-by-H numpy array of unguided class activations.
    :param guided_cam_matrix: [used only if `cam_matrix is None`]
        M-by-N-by-H-by-F numpy array of guided class activations.
    :param full_storm_id_string: Full storm ID.
    :param storm_time_unix_sec: Storm time.
    """

    pmm_flag = full_storm_id_string is None and storm_time_unix_sec is None
    conv_2d3d = model_metadata_dict[cnn.CONV_2D3D_KEY]

    if conv_2d3d:
        loop_max = 1
        radar_field_names = ['reflectivity']
    else:
        loop_max = len(figure_objects)
        training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
        radar_field_names = training_option_dict[trainval_io.RADAR_FIELDS_KEY]

    max_unguided_value_log10 = numpy.log10(max_unguided_value)
    contour_interval_log10 = (
        (max_unguided_value_log10 - MIN_UNGUIDED_VALUE_LOG10) /
        (num_unguided_contours - 1)
    )

    for j in range(loop_max):
        if cam_matrix is None:
            saliency_plotting.plot_many_2d_grids_with_contours(
                saliency_matrix_3d=numpy.flip(
                    guided_cam_matrix[..., j], axis=0
                ),
                axes_object_matrix=axes_object_matrices[j],
                colour_map_object=colour_map_object,
                max_absolute_contour_level=max_guided_value,
                contour_interval=max_guided_value / half_num_guided_contours)

            this_colour_bar_object = plotting_utils.plot_linear_colour_bar(
                axes_object_or_matrix=axes_object_matrices[j],
                data_matrix=guided_cam_matrix[..., j],
                colour_map_object=colour_map_object, min_value=0.,
                max_value=max_guided_value, orientation_string='horizontal',
                fraction_of_axis_length=colour_bar_length,
                extend_min=False, extend_max=True,
                font_size=COLOUR_BAR_FONT_SIZE)

            if label_colour_bars:
                this_colour_bar_object.set_label(
                    'Absolute guided class activation',
                    fontsize=COLOUR_BAR_FONT_SIZE)
        else:
            cam_matrix_log10 = numpy.log10(cam_matrix)

            cam_plotting.plot_many_2d_grids(
                class_activation_matrix_3d=numpy.flip(cam_matrix_log10, axis=0),
                axes_object_matrix=axes_object_matrices[j],
                colour_map_object=colour_map_object,
                min_contour_level=MIN_UNGUIDED_VALUE_LOG10,
                max_contour_level=max_unguided_value_log10,
                contour_interval=contour_interval_log10)

            this_colour_bar_object = plotting_utils.plot_linear_colour_bar(
                axes_object_or_matrix=axes_object_matrices[j],
                data_matrix=cam_matrix_log10,
                colour_map_object=colour_map_object,
                min_value=MIN_UNGUIDED_VALUE_LOG10,
                max_value=max_unguided_value_log10,
                orientation_string='horizontal',
                fraction_of_axis_length=colour_bar_length,
                extend_min=True, extend_max=True,
                font_size=COLOUR_BAR_FONT_SIZE)

            if label_colour_bars:
                this_colour_bar_object.set_label(
                    r'Class activation (log$_{10}$)',
                    fontsize=COLOUR_BAR_FONT_SIZE
                )

        this_file_name = plot_examples.metadata_to_file_name(
            output_dir_name=output_dir_name, is_sounding=False,
            pmm_flag=pmm_flag, full_storm_id_string=full_storm_id_string,
            storm_time_unix_sec=storm_time_unix_sec,
            radar_field_name=radar_field_names[j]
        )

        print('Saving figure to: "{0:s}"...'.format(this_file_name))
        figure_objects[j].savefig(
            this_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
            bbox_inches='tight'
        )
        pyplot.close(figure_objects[j])


def _plot_2d_radar_cam(
        colour_map_object, max_unguided_value, num_unguided_contours,
        max_guided_value, half_num_guided_contours, label_colour_bars,
        colour_bar_length, figure_objects, axes_object_matrices,
        model_metadata_dict, output_dir_name, cam_matrix=None,
        guided_cam_matrix=None, full_storm_id_string=None,
        storm_time_unix_sec=None):
    """Plots class-activation map for 2-D radar data.

    M = number of rows in spatial grid
    N = number of columns in spatial grid
    F = number of radar fields

    If this method is plotting a composite rather than single example (storm
    object), `full_storm_id_string` and `storm_time_unix_sec` can be None.

    :param colour_map_object: See doc for `_plot_3d_radar_cam`.
    :param max_unguided_value: Same.
    :param num_unguided_contours: Same.
    :param max_guided_value: Same.
    :param half_num_guided_contours: Same.
    :param label_colour_bars: Same.
    :param colour_bar_length: Same.
    :param figure_objects: See doc for
        `plot_input_examples._plot_2d_radar_scan`.
    :param axes_object_matrices: Same.
    :param model_metadata_dict: See doc for `_plot_3d_radar_cam`.
    :param output_dir_name: Same.
    :param cam_matrix: M-by-N numpy array of unguided class activations.
    :param guided_cam_matrix: [used only if `cam_matrix is None`]
        M-by-N-by-F numpy array of guided class activations.
    :param full_storm_id_string: Full storm ID.
    :param storm_time_unix_sec: Storm time.
    """

    pmm_flag = full_storm_id_string is None and storm_time_unix_sec is None
    conv_2d3d = model_metadata_dict[cnn.CONV_2D3D_KEY]

    if conv_2d3d:
        figure_index = 1
        radar_field_name = 'shear'
    else:
        figure_index = 0
        radar_field_name = None

    list_of_layer_operation_dicts = model_metadata_dict[
        cnn.LAYER_OPERATIONS_KEY]

    if list_of_layer_operation_dicts is None:
        training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
        radar_field_names = training_option_dict[trainval_io.RADAR_FIELDS_KEY]
        num_channels = len(radar_field_names)
    else:
        num_channels = len(list_of_layer_operation_dicts)

    max_unguided_value_log10 = numpy.log10(max_unguided_value)
    contour_interval_log10 = (
        (max_unguided_value_log10 - MIN_UNGUIDED_VALUE_LOG10) /
        (num_unguided_contours - 1)
    )

    if cam_matrix is None:
        saliency_plotting.plot_many_2d_grids_with_contours(
            saliency_matrix_3d=numpy.flip(guided_cam_matrix, axis=0),
            axes_object_matrix=axes_object_matrices[figure_index],
            colour_map_object=colour_map_object,
            max_absolute_contour_level=max_guided_value,
            contour_interval=max_guided_value / half_num_guided_contours,
            row_major=False)

        this_colour_bar_object = plotting_utils.plot_linear_colour_bar(
            axes_object_or_matrix=axes_object_matrices[figure_index],
            data_matrix=guided_cam_matrix,
            colour_map_object=colour_map_object, min_value=0.,
            max_value=max_guided_value, orientation_string='horizontal',
            fraction_of_axis_length=colour_bar_length,
            extend_min=False, extend_max=True,
            font_size=COLOUR_BAR_FONT_SIZE)

        if label_colour_bars:
            this_colour_bar_object.set_label(
                'Absolute guided class activation',
                fontsize=COLOUR_BAR_FONT_SIZE)
    else:
        this_cam_matrix_log10 = numpy.log10(
            numpy.expand_dims(cam_matrix, axis=-1)
        )
        this_cam_matrix_log10 = numpy.repeat(
            this_cam_matrix_log10, repeats=num_channels, axis=-1)

        cam_plotting.plot_many_2d_grids(
            class_activation_matrix_3d=numpy.flip(
                this_cam_matrix_log10, axis=0
            ),
            axes_object_matrix=axes_object_matrices[figure_index],
            colour_map_object=colour_map_object,
            min_contour_level=MIN_UNGUIDED_VALUE_LOG10,
            max_contour_level=max_unguided_value_log10,
            contour_interval=contour_interval_log10, row_major=False)

        this_colour_bar_object = plotting_utils.plot_linear_colour_bar(
            axes_object_or_matrix=axes_object_matrices[figure_index],
            data_matrix=this_cam_matrix_log10,
            colour_map_object=colour_map_object,
            min_value=MIN_UNGUIDED_VALUE_LOG10,
            max_value=max_unguided_value_log10,
            orientation_string='horizontal',
            fraction_of_axis_length=colour_bar_length,
            extend_min=True, extend_max=True,
            font_size=COLOUR_BAR_FONT_SIZE)

        if label_colour_bars:
            this_colour_bar_object.set_label(
                r'Class activation (log$_{10}$)', fontsize=COLOUR_BAR_FONT_SIZE
            )

    output_file_name = plot_examples.metadata_to_file_name(
        output_dir_name=output_dir_name, is_sounding=False, pmm_flag=pmm_flag,
        full_storm_id_string=full_storm_id_string,
        storm_time_unix_sec=storm_time_unix_sec,
        radar_field_name=radar_field_name)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_objects[figure_index].savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_objects[figure_index])


def _smooth_maps(cam_matrices, guided_cam_matrices,
                 smoothing_radius_grid_cells):
    """Smooths guided and unguided class-activation maps, using Gaussian filter.

    T = number of input tensors to the model

    :param cam_matrices: length-T list of numpy arrays with unguided class-
        activation maps (CAMs).
    :param guided_cam_matrices: length-T list of numpy arrays with guided CAMs.
    :param smoothing_radius_grid_cells: e-folding radius (number of grid cells).
    :return: cam_matrices: Smoothed version of input.
    :return: guided_cam_matrices: Smoothed version of input.
    """

    print((
        'Smoothing guided and unguided CAMs with Gaussian filter (e-folding '
        'radius of {0:.1f} grid cells)...'
    ).format(
        smoothing_radius_grid_cells
    ))

    num_matrices = len(cam_matrices)

    for j in range(num_matrices):
        if cam_matrices[j] is None:
            continue

        num_examples = cam_matrices[j].shape[0]
        this_num_channels = guided_cam_matrices[j].shape[-1]

        for i in range(num_examples):
            cam_matrices[j][i, ...] = general_utils.apply_gaussian_filter(
                input_matrix=cam_matrices[j][i, ...],
                e_folding_radius_grid_cells=smoothing_radius_grid_cells
            )

            for k in range(this_num_channels):
                guided_cam_matrices[j][i, ..., k] = (
                    general_utils.apply_gaussian_filter(
                        input_matrix=guided_cam_matrices[j][i, ..., k],
                        e_folding_radius_grid_cells=smoothing_radius_grid_cells
                    )
                )

    return cam_matrices, guided_cam_matrices


def _run(input_file_name, colour_map_name, max_unguided_value, max_guided_value,
         num_unguided_contours, half_num_guided_contours,
         smoothing_radius_grid_cells, allow_whitespace, plot_panel_names,
         add_titles, label_colour_bars, colour_bar_length, top_output_dir_name):
    """Plots Grad-CAM output (guided and unguided class-activation maps).

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param colour_map_name: Same.
    :param max_unguided_value: Same.
    :param max_guided_value: Same.
    :param num_unguided_contours: Same.
    :param half_num_guided_contours: Same.
    :param smoothing_radius_grid_cells: Same.
    :param allow_whitespace: Same.
    :param plot_panel_names: Same.
    :param add_titles: Same.
    :param label_colour_bars: Same.
    :param colour_bar_length: Same.
    :param top_output_dir_name: Same.
    """

    if smoothing_radius_grid_cells <= 0:
        smoothing_radius_grid_cells = None

    unguided_cam_dir_name = '{0:s}/main_gradcam'.format(top_output_dir_name)
    guided_cam_dir_name = '{0:s}/guided_gradcam'.format(top_output_dir_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=unguided_cam_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=guided_cam_dir_name)

    # Check input args.
    colour_map_object = pyplot.get_cmap(colour_map_name)
    error_checking.assert_is_greater(max_unguided_value, 0.)
    error_checking.assert_is_greater(max_guided_value, 0.)
    error_checking.assert_is_geq(num_unguided_contours, 10)
    error_checking.assert_is_geq(half_num_guided_contours, 5)

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    gradcam_dict, pmm_flag = gradcam.read_file(input_file_name)

    if pmm_flag:
        predictor_matrices = gradcam_dict.pop(
            gradcam.MEAN_PREDICTOR_MATRICES_KEY)
        cam_matrices = gradcam_dict.pop(gradcam.MEAN_CAM_MATRICES_KEY)
        guided_cam_matrices = gradcam_dict.pop(
            gradcam.MEAN_GUIDED_CAM_MATRICES_KEY)

        full_storm_id_strings = [None]
        storm_times_unix_sec = [None]

        for j in range(len(predictor_matrices)):
            predictor_matrices[j] = numpy.expand_dims(
                predictor_matrices[j], axis=0
            )

            if cam_matrices[j] is None:
                continue

            cam_matrices[j] = numpy.expand_dims(
                cam_matrices[j], axis=0
            )
            guided_cam_matrices[j] = numpy.expand_dims(
                guided_cam_matrices[j], axis=0
            )
    else:
        predictor_matrices = gradcam_dict.pop(gradcam.PREDICTOR_MATRICES_KEY)
        cam_matrices = gradcam_dict.pop(gradcam.CAM_MATRICES_KEY)
        guided_cam_matrices = gradcam_dict.pop(gradcam.GUIDED_CAM_MATRICES_KEY)

        full_storm_id_strings = gradcam_dict[gradcam.FULL_STORM_IDS_KEY]
        storm_times_unix_sec = gradcam_dict[gradcam.STORM_TIMES_KEY]

    if smoothing_radius_grid_cells is not None:
        cam_matrices, guided_cam_matrices = _smooth_maps(
            cam_matrices=cam_matrices, guided_cam_matrices=guided_cam_matrices,
            smoothing_radius_grid_cells=smoothing_radius_grid_cells)

    # Read metadata for CNN.
    model_file_name = gradcam_dict[gradcam.MODEL_FILE_KEY]
    model_metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0]
    )

    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_model_metadata(model_metafile_name)
    print(SEPARATOR_STRING)

    num_examples = predictor_matrices[0].shape[0]
    num_matrices = len(predictor_matrices)

    for i in range(num_examples):
        this_handle_dict = plot_examples.plot_one_example(
            list_of_predictor_matrices=predictor_matrices,
            model_metadata_dict=model_metadata_dict, pmm_flag=pmm_flag,
            example_index=i, plot_sounding=False,
            allow_whitespace=allow_whitespace,
            plot_panel_names=plot_panel_names, add_titles=add_titles,
            label_colour_bars=label_colour_bars,
            colour_bar_length=colour_bar_length)

        these_figure_objects = this_handle_dict[plot_examples.RADAR_FIGURES_KEY]
        these_axes_object_matrices = this_handle_dict[
            plot_examples.RADAR_AXES_KEY]

        for j in range(num_matrices):
            if cam_matrices[j] is None:
                continue

            this_num_spatial_dim = len(predictor_matrices[j].shape) - 2

            if this_num_spatial_dim == 3:
                _plot_3d_radar_cam(
                    colour_map_object=colour_map_object,
                    max_unguided_value=max_unguided_value,
                    num_unguided_contours=num_unguided_contours,
                    max_guided_value=max_guided_value,
                    half_num_guided_contours=half_num_guided_contours,
                    label_colour_bars=label_colour_bars,
                    colour_bar_length=colour_bar_length,
                    figure_objects=these_figure_objects,
                    axes_object_matrices=these_axes_object_matrices,
                    model_metadata_dict=model_metadata_dict,
                    output_dir_name=unguided_cam_dir_name,
                    cam_matrix=cam_matrices[j][i, ...],
                    full_storm_id_string=full_storm_id_strings[i],
                    storm_time_unix_sec=storm_times_unix_sec[i]
                )
            else:
                _plot_2d_radar_cam(
                    colour_map_object=colour_map_object,
                    max_unguided_value=max_unguided_value,
                    num_unguided_contours=num_unguided_contours,
                    max_guided_value=max_guided_value,
                    half_num_guided_contours=half_num_guided_contours,
                    label_colour_bars=label_colour_bars,
                    colour_bar_length=colour_bar_length,
                    figure_objects=these_figure_objects,
                    axes_object_matrices=these_axes_object_matrices,
                    model_metadata_dict=model_metadata_dict,
                    output_dir_name=unguided_cam_dir_name,
                    cam_matrix=cam_matrices[j][i, ...],
                    full_storm_id_string=full_storm_id_strings[i],
                    storm_time_unix_sec=storm_times_unix_sec[i]
                )

        this_handle_dict = plot_examples.plot_one_example(
            list_of_predictor_matrices=predictor_matrices,
            model_metadata_dict=model_metadata_dict, pmm_flag=pmm_flag,
            example_index=i, plot_sounding=False,
            allow_whitespace=allow_whitespace,
            plot_panel_names=plot_panel_names, add_titles=add_titles,
            label_colour_bars=label_colour_bars,
            colour_bar_length=colour_bar_length)

        these_figure_objects = this_handle_dict[plot_examples.RADAR_FIGURES_KEY]
        these_axes_object_matrices = this_handle_dict[
            plot_examples.RADAR_AXES_KEY]

        for j in range(num_matrices):
            if guided_cam_matrices[j] is None:
                continue

            this_num_spatial_dim = len(predictor_matrices[j].shape) - 2

            if this_num_spatial_dim == 3:
                _plot_3d_radar_cam(
                    colour_map_object=colour_map_object,
                    max_unguided_value=max_unguided_value,
                    num_unguided_contours=num_unguided_contours,
                    max_guided_value=max_guided_value,
                    half_num_guided_contours=half_num_guided_contours,
                    label_colour_bars=label_colour_bars,
                    colour_bar_length=colour_bar_length,
                    figure_objects=these_figure_objects,
                    axes_object_matrices=these_axes_object_matrices,
                    model_metadata_dict=model_metadata_dict,
                    output_dir_name=guided_cam_dir_name,
                    guided_cam_matrix=guided_cam_matrices[j][i, ...],
                    full_storm_id_string=full_storm_id_strings[i],
                    storm_time_unix_sec=storm_times_unix_sec[i]
                )
            else:
                _plot_2d_radar_cam(
                    colour_map_object=colour_map_object,
                    max_unguided_value=max_unguided_value,
                    num_unguided_contours=num_unguided_contours,
                    max_guided_value=max_guided_value,
                    half_num_guided_contours=half_num_guided_contours,
                    label_colour_bars=label_colour_bars,
                    colour_bar_length=colour_bar_length,
                    figure_objects=these_figure_objects,
                    axes_object_matrices=these_axes_object_matrices,
                    model_metadata_dict=model_metadata_dict,
                    output_dir_name=guided_cam_dir_name,
                    guided_cam_matrix=guided_cam_matrices[j][i, ...],
                    full_storm_id_string=full_storm_id_strings[i],
                    storm_time_unix_sec=storm_times_unix_sec[i]
                )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        colour_map_name=getattr(INPUT_ARG_OBJECT, COLOUR_MAP_ARG_NAME),
        max_unguided_value=getattr(
            INPUT_ARG_OBJECT, MAX_UNGUIDED_VALUE_ARG_NAME),
        num_unguided_contours=getattr(
            INPUT_ARG_OBJECT, NUM_UNGUIDED_CONTOURS_ARG_NAME),
        max_guided_value=getattr(INPUT_ARG_OBJECT, MAX_GUIDED_VALUE_ARG_NAME),
        half_num_guided_contours=getattr(
            INPUT_ARG_OBJECT, HALF_NUM_GUIDED_CONTOURS_ARG_NAME),
        smoothing_radius_grid_cells=getattr(
            INPUT_ARG_OBJECT, SMOOTHING_RADIUS_ARG_NAME),
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
