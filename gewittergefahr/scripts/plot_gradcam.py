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
from gewittergefahr.plotting import cam_plotting
from gewittergefahr.plotting import saliency_plotting
from gewittergefahr.scripts import plot_input_examples as plot_examples

# TODO(thunderhoser): Make this script deal with soundings.

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

REGION_COLOUR = numpy.full(3, 0.)
REGION_LINE_WIDTH = 3

NUM_CONTOURS = 12
HALF_NUM_CONTOURS = 10
FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_file_name'
COLOUR_MAP_ARG_NAME = 'colour_map_name'
MAX_PERCENTILE_ARG_NAME = 'max_colour_percentile'
SMOOTHING_HW_SIZE_ARG_NAME = 'smoothing_half_window_size'
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

MAX_PERCENTILE_HELP_STRING = (
    'Determines max value in colour scheme for each class-activation map (CAM).'
    '  The max value for the [i]th example will be the [q]th percentile of all '
    'class activations for the [i]th example, where q = `{0:s}`.'
).format(MAX_PERCENTILE_ARG_NAME)

SMOOTHING_HW_SIZE_HELP_STRING = (
    'Number of grid cells in half-window for median smoother.  If you do not '
    'want to smooth class-activation maps, leave this alone.')

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
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False,
    default=99., help=MAX_PERCENTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SMOOTHING_HW_SIZE_ARG_NAME, type=int, required=False,
    default=-1, help=SMOOTHING_HW_SIZE_HELP_STRING)

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
        colour_map_object, max_contour_value, figure_objects,
        axes_object_matrices, model_metadata_dict, output_dir_name,
        cam_matrix=None, guided_cam_matrix=None, full_storm_id_string=None,
        storm_time_unix_sec=None):
    """Plots class-activation map for 3-D radar data.

    M = number of rows in spatial grid
    N = number of columns in spatial grid
    H = number of heights in spatial grid
    F = number of radar fields

    If this method is plotting a composite rather than single example (storm
    object), `full_storm_id_string` and `storm_time_unix_sec` can be None.

    :param colour_map_object: See documentation at top of file.
    :param max_contour_value: Max contour value for class activation.
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

    for j in range(loop_max):
        if cam_matrix is None:
            saliency_plotting.plot_many_2d_grids_with_contours(
                saliency_matrix_3d=numpy.flip(
                    guided_cam_matrix[..., j], axis=0
                ),
                axes_object_matrix=axes_object_matrices[j],
                colour_map_object=colour_map_object,
                max_absolute_contour_level=max_contour_value,
                contour_interval=max_contour_value / HALF_NUM_CONTOURS)
        else:
            cam_plotting.plot_many_2d_grids(
                class_activation_matrix_3d=numpy.flip(cam_matrix, axis=0),
                axes_object_matrix=axes_object_matrices[j],
                colour_map_object=colour_map_object,
                max_contour_level=max_contour_value,
                contour_interval=max_contour_value / NUM_CONTOURS)

        this_title_object = figure_objects[j]._suptitle

        if this_title_object is not None:
            this_title_string = '{0:s} ... max {1:s} = {2:.2e}'.format(
                this_title_object.get_text(),
                'absolute guided activation' if cam_matrix is None
                else 'class activation',
                max_contour_value
            )

            figure_objects[j].suptitle(
                this_title_string,
                fontsize=plot_examples.DEFAULT_TITLE_FONT_SIZE)

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
        colour_map_object, max_contour_value, figure_objects,
        axes_object_matrices, model_metadata_dict, output_dir_name,
        cam_matrix=None, guided_cam_matrix=None, full_storm_id_string=None,
        storm_time_unix_sec=None):
    """Plots class-activation map for 2-D radar data.

    M = number of rows in spatial grid
    N = number of columns in spatial grid
    F = number of radar fields

    If this method is plotting a composite rather than single example (storm
    object), `full_storm_id_string` and `storm_time_unix_sec` can be None.

    :param colour_map_object: See doc for `_plot_3d_radar_cam`.
    :param max_contour_value: Same.
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

    if cam_matrix is None:
        saliency_plotting.plot_many_2d_grids_with_contours(
            saliency_matrix_3d=numpy.flip(guided_cam_matrix, axis=0),
            axes_object_matrix=axes_object_matrices[figure_index],
            colour_map_object=colour_map_object,
            max_absolute_contour_level=max_contour_value,
            contour_interval=max_contour_value / HALF_NUM_CONTOURS,
            row_major=False)
    else:
        this_matrix = numpy.expand_dims(cam_matrix, axis=-1)
        this_matrix = numpy.repeat(this_matrix, repeats=num_channels, axis=-1)

        cam_plotting.plot_many_2d_grids(
            class_activation_matrix_3d=numpy.flip(this_matrix, axis=0),
            axes_object_matrix=axes_object_matrices[figure_index],
            colour_map_object=colour_map_object,
            max_contour_level=max_contour_value,
            contour_interval=max_contour_value / NUM_CONTOURS,
            row_major=False)

    this_title_object = figure_objects[figure_index]._suptitle

    if this_title_object is not None:
        this_title_string = '{0:s} ... max {1:s} = {2:.2e}'.format(
            this_title_object.get_text(),
            'absolute guided activation' if cam_matrix is None
            else 'class activation',
            max_contour_value
        )

        figure_objects[figure_index].suptitle(
            this_title_string,
            fontsize=plot_examples.DEFAULT_TITLE_FONT_SIZE)

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


def _smooth_maps(cam_matrices, guided_cam_matrices, smoothing_half_window_size):
    """Smooths guided and unguided class-activation maps, using median filter.

    T = number of input tensors to the model

    :param cam_matrices: length-T list of numpy arrays with unguided class-
        activation maps (CAMs).
    :param guided_cam_matrices: length-T list of numpy arrays with guided CAMs.
    :param smoothing_half_window_size: Number of grid cells in half-window for
        median filter.
    :return: cam_matrices: Smoothed version of input.
    :return: guided_cam_matrices: Smoothed version of input.
    """

    print((
        'Smoothing guided and unguided CAMs with {0:d}-by-{0:d} median '
        'filter...'
    ).format(
        2 * smoothing_half_window_size + 1
    ))

    num_matrices = len(cam_matrices)

    for j in range(num_matrices):
        if cam_matrices[j] is None:
            continue

        num_examples = cam_matrices[j].shape[0]
        this_num_channels = guided_cam_matrices[j].shape[-1]

        for i in range(num_examples):
            cam_matrices[j][i, ...] = general_utils.apply_median_filter(
                input_matrix=cam_matrices[j][i, ...],
                num_cells_in_half_window=smoothing_half_window_size
            )

            for k in range(this_num_channels):
                guided_cam_matrices[j][i, ..., k] = (
                    general_utils.apply_median_filter(
                        input_matrix=guided_cam_matrices[j][i, ..., k],
                        num_cells_in_half_window=smoothing_half_window_size
                    )
                )

    return cam_matrices, guided_cam_matrices


def _run(input_file_name, colour_map_name, max_colour_percentile,
         smoothing_half_window_size, allow_whitespace, plot_panel_names,
         add_titles, label_colour_bars, colour_bar_length, top_output_dir_name):
    """Plots Grad-CAM output (guided and unguided class-activation maps).

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param colour_map_name: Same.
    :param max_colour_percentile: Same.
    :param smoothing_half_window_size: Same.
    :param allow_whitespace: Same.
    :param plot_panel_names: Same.
    :param add_titles: Same.
    :param label_colour_bars: Same.
    :param colour_bar_length: Same.
    :param top_output_dir_name: Same.
    """

    if smoothing_half_window_size < 1:
        smoothing_half_window_size = None

    unguided_cam_dir_name = '{0:s}/main_gradcam'.format(top_output_dir_name)
    guided_cam_dir_name = '{0:s}/guided_gradcam'.format(top_output_dir_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=unguided_cam_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=guided_cam_dir_name)

    # Check input args.
    error_checking.assert_is_geq(max_colour_percentile, 0.)
    error_checking.assert_is_leq(max_colour_percentile, 100.)
    colour_map_object = pyplot.get_cmap(colour_map_name)

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

    if smoothing_half_window_size is not None:
        cam_matrices, guided_cam_matrices = _smooth_maps(
            cam_matrices=cam_matrices, guided_cam_matrices=guided_cam_matrices,
            smoothing_half_window_size=smoothing_half_window_size)

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

        these_activations = numpy.array([])

        for j in range(num_matrices):
            if cam_matrices[j] is None:
                continue

            these_activations = numpy.concatenate((
                these_activations, numpy.ravel(cam_matrices[j][i, ...])
            ))

        this_max_contour_value = numpy.percentile(
            these_activations, max_colour_percentile)

        for j in range(num_matrices):
            if cam_matrices[j] is None:
                continue

            this_num_spatial_dim = len(predictor_matrices[j].shape) - 2

            if this_num_spatial_dim == 3:
                _plot_3d_radar_cam(
                    colour_map_object=colour_map_object,
                    max_contour_value=this_max_contour_value,
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
                    max_contour_value=this_max_contour_value,
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

        these_activations = numpy.array([])

        for j in range(num_matrices):
            if guided_cam_matrices[j] is None:
                continue

            these_activations = numpy.concatenate((
                these_activations, numpy.ravel(guided_cam_matrices[j][i, ...])
            ))

        this_max_contour_value = numpy.percentile(
            numpy.absolute(these_activations), max_colour_percentile
        )

        for j in range(num_matrices):
            if guided_cam_matrices[j] is None:
                continue

            this_num_spatial_dim = len(predictor_matrices[j].shape) - 2

            if this_num_spatial_dim == 3:
                _plot_3d_radar_cam(
                    colour_map_object=colour_map_object,
                    max_contour_value=this_max_contour_value,
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
                    max_contour_value=this_max_contour_value,
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
        max_colour_percentile=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME),
        smoothing_half_window_size=getattr(
            INPUT_ARG_OBJECT, SMOOTHING_HW_SIZE_ARG_NAME),
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
