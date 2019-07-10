"""Plots Grad-CAM output (class-activation maps)."""

import os
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import monte_carlo
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import gradcam
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import radar_plotting
from gewittergefahr.plotting import cam_plotting
from gewittergefahr.plotting import saliency_plotting
from gewittergefahr.plotting import significance_plotting

# TODO(thunderhoser): Use threshold counts at some point.
# TODO(thunderhoser): Make this script deal with soundings.

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_CONTOURS = 12
HALF_NUM_CONTOURS = 10

TITLE_FONT_SIZE = 20
FONT_SIZE_WITH_COLOUR_BARS = 16
FONT_SIZE_SANS_COLOUR_BARS = 20
FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_file_name'
PLOT_SIGNIFICANCE_ARG_NAME = 'plot_significance'
COLOUR_MAP_ARG_NAME = 'cam_colour_map_name'
MAX_PERCENTILE_ARG_NAME = 'max_colour_prctile_for_cam'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `gradcam.read_standard_file` or'
    ' `gradcam.read_pmm_file`.')

PLOT_SIGNIFICANCE_HELP_STRING = (
    'Boolean flag.  If 1, will plot stippling for significance.  This applies '
    'only if the saliency map contains PMM (proability-matched means) and '
    'results of a Monte Carlo comparison.')

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
    '--' + PLOT_SIGNIFICANCE_ARG_NAME, type=int, required=False, default=0,
    help=PLOT_SIGNIFICANCE_HELP_STRING)

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
        radar_matrix, model_metadata_dict, cam_colour_map_object,
        max_colour_prctile_for_cam, output_dir_name,
        class_activation_matrix=None, guided_class_activation_matrix=None,
        full_id_strings=None, storm_times_unix_sec=None, monte_carlo_dict=None,
        monte_carlo_index=None):
    """Plots class-activation maps for 3-D radar data.

    E = number of examples
    M = number of rows in spatial grid
    N = number of columns in spatial grid
    H = number of heights in spatial grid
    F = number of radar fields

    This method will plot either `class_activation_matrix` or
    `guided_class_activation_matrix`, not both.

    If `full_id_strings is None` and `storm_times_unix_sec is None`, will assume
    that the input matrices contain probability-matched means.

    :param radar_matrix: E-by-M-by-N-by-H-by-F numpy array of radar values.
    :param model_metadata_dict: Dictionary returned by
        `cnn.read_model_metadata`.
    :param cam_colour_map_object: See documentation at top of file.
    :param max_colour_prctile_for_cam: Same.
    :param output_dir_name: Same.
    :param class_activation_matrix: E-by-M-by-N-by-H numpy array of class
        activations.
    :param guided_class_activation_matrix: E-by-M-by-N-by-H-by-F numpy array of
        guided class activations.
    :param full_id_strings: length-E list of full storm IDs.
    :param storm_times_unix_sec: length-E numpy array of storm times.
    :param monte_carlo_dict: See doc for `monte_carlo.check_output`.  If this is
        None, will *not* plot stippling for significance.
    :param monte_carlo_index: [used only if `monte_carlo_dict is not None`]
        Index of input tensor being plotted.  If this is q, plotting the [q]th
        input tensor to the model.
    """

    pmm_flag = full_id_strings is None and storm_times_unix_sec is None

    if monte_carlo_dict is not None:
        i = monte_carlo_index

        significance_matrix = numpy.logical_or(
            monte_carlo_dict[monte_carlo.TRIAL_PMM_MATRICES_KEY][i] <
            monte_carlo_dict[monte_carlo.MIN_MATRICES_KEY][i],
            monte_carlo_dict[monte_carlo.TRIAL_PMM_MATRICES_KEY][i] >
            monte_carlo_dict[monte_carlo.MAX_MATRICES_KEY][i]
        )

    num_examples = radar_matrix.shape[0]
    num_heights = radar_matrix.shape[-2]
    num_fields = radar_matrix.shape[-1]
    num_panel_rows = int(numpy.floor(
        numpy.sqrt(num_heights)
    ))

    if class_activation_matrix is None:
        quantity_string = 'max absolute value'
        pathless_file_name_prefix = 'guided-gradcam'
    else:
        quantity_string = 'max class activation'
        pathless_file_name_prefix = 'gradcam'

    conv_2d3d = model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]
    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]

    for i in range(num_examples):
        for k in range(num_fields):
            if conv_2d3d:
                this_field_name = radar_utils.REFL_NAME
            else:
                this_field_name = training_option_dict[
                    trainval_io.RADAR_FIELDS_KEY][k]

            _, this_axes_object_matrix = (
                radar_plotting.plot_3d_grid_without_coords(
                    field_matrix=numpy.flip(radar_matrix[i, ..., k], axis=0),
                    field_name=this_field_name,
                    grid_point_heights_metres=training_option_dict[
                        trainval_io.RADAR_HEIGHTS_KEY],
                    ground_relative=True, num_panel_rows=num_panel_rows,
                    font_size=FONT_SIZE_SANS_COLOUR_BARS)
            )

            if class_activation_matrix is None:
                this_matrix = guided_class_activation_matrix[i, ..., k]
                this_max_contour_level = numpy.percentile(
                    numpy.absolute(this_matrix), max_colour_prctile_for_cam
                )

                saliency_plotting.plot_many_2d_grids_with_contours(
                    saliency_matrix_3d=numpy.flip(this_matrix, axis=0),
                    axes_object_matrix=this_axes_object_matrix,
                    colour_map_object=cam_colour_map_object,
                    max_absolute_contour_level=this_max_contour_level,
                    contour_interval=this_max_contour_level / HALF_NUM_CONTOURS)

                if monte_carlo_dict is not None:
                    significance_plotting.plot_many_2d_grids_without_coords(
                        significance_matrix=numpy.flip(
                            significance_matrix[i, ..., k], axis=0),
                        axes_object_matrix=this_axes_object_matrix
                    )
            else:
                this_matrix = class_activation_matrix[i, ...]
                this_max_contour_level = numpy.percentile(
                    this_matrix, max_colour_prctile_for_cam)

                cam_plotting.plot_many_2d_grids(
                    class_activation_matrix_3d=numpy.flip(this_matrix, axis=0),
                    axes_object_matrix=this_axes_object_matrix,
                    colour_map_object=cam_colour_map_object,
                    max_contour_level=this_max_contour_level,
                    contour_interval=this_max_contour_level / NUM_CONTOURS)

                if monte_carlo_dict is not None:
                    significance_plotting.plot_many_2d_grids_without_coords(
                        significance_matrix=numpy.flip(
                            significance_matrix[i, ...], axis=0),
                        axes_object_matrix=this_axes_object_matrix
                    )

            this_colour_map_object, this_colour_norm_object = (
                radar_plotting.get_default_colour_scheme(this_field_name)
            )

            plotting_utils.plot_colour_bar(
                axes_object_or_matrix=this_axes_object_matrix,
                data_matrix=radar_matrix[i, ..., k],
                colour_map_object=this_colour_map_object,
                colour_norm_object=this_colour_norm_object,
                orientation_string='horizontal', extend_min=True,
                extend_max=True)

            if pmm_flag:
                this_title_string = 'Probability-matched mean'
                this_figure_file_name = '{0:s}/{1:s}_pmm_{2:s}.jpg'.format(
                    output_dir_name, pathless_file_name_prefix,
                    this_field_name.replace('_', '-')
                )

            else:
                this_storm_time_string = time_conversion.unix_sec_to_string(
                    storm_times_unix_sec[i], TIME_FORMAT)

                this_title_string = 'Storm "{0:s}" at {1:s}'.format(
                    full_id_strings[i], this_storm_time_string)

                this_figure_file_name = (
                    '{0:s}/{1:s}_{2:s}_{3:s}_{4:s}.jpg'
                ).format(
                    output_dir_name, pathless_file_name_prefix,
                    full_id_strings[i].replace('_', '-'),
                    this_storm_time_string, this_field_name.replace('_', '-')
                )

            this_title_string += ' ({0:s} = {1:.3f})'.format(
                quantity_string, this_max_contour_level)
            pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)

            print('Saving figure to file: "{0:s}"...'.format(
                this_figure_file_name))

            pyplot.savefig(this_figure_file_name, dpi=FIGURE_RESOLUTION_DPI)
            pyplot.close()


def _plot_2d_radar_cams(
        radar_matrix, model_metadata_dict, cam_colour_map_object,
        max_colour_prctile_for_cam, output_dir_name,
        class_activation_matrix=None, guided_class_activation_matrix=None,
        full_id_strings=None, storm_times_unix_sec=None, monte_carlo_dict=None,
        monte_carlo_index=None):
    """Plots class-activation maps for 2-D radar data.

    E = number of examples
    M = number of rows in spatial grid
    N = number of columns in spatial grid
    C = number of channels (field/height pairs)

    This method will plot either `class_activation_matrix` or
    `list_of_guided_cam_matrices`, not both.

    If `full_id_strings is None` and `storm_times_unix_sec is None`, will assume
    that the input matrices contain probability-matched means.

    :param radar_matrix: E-by-M-by-N-by-C numpy array of radar values.
    :param model_metadata_dict: See doc for `_plot_3d_radar_cams`.
    :param cam_colour_map_object: Same.
    :param max_colour_prctile_for_cam: Same.
    :param output_dir_name: Same.
    :param class_activation_matrix: E-by-M-by-N numpy array of class
        activations.
    :param guided_class_activation_matrix: E-by-M-by-N-by-C numpy array of
        guided class activations.
    :param full_id_strings: See doc for `_plot_3d_radar_cams`.
    :param storm_times_unix_sec: Same.
    :param monte_carlo_dict: Same.
    :param monte_carlo_index: Same.
    """

    pmm_flag = full_id_strings is None and storm_times_unix_sec is None

    if monte_carlo_dict is not None:
        i = monte_carlo_index

        significance_matrix = numpy.logical_or(
            monte_carlo_dict[monte_carlo.TRIAL_PMM_MATRICES_KEY][i] <
            monte_carlo_dict[monte_carlo.MIN_MATRICES_KEY][i],
            monte_carlo_dict[monte_carlo.TRIAL_PMM_MATRICES_KEY][i] >
            monte_carlo_dict[monte_carlo.MAX_MATRICES_KEY][i]
        )

    conv_2d3d = model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    list_of_layer_operation_dicts = model_metadata_dict[
        cnn.LAYER_OPERATIONS_KEY]

    if list_of_layer_operation_dicts is None:
        field_name_by_panel = training_option_dict[trainval_io.RADAR_FIELDS_KEY]

        if conv_2d3d:
            heights_m_agl = numpy.full(len(field_name_by_panel), 0, dtype=int)
        else:
            heights_m_agl = training_option_dict[trainval_io.RADAR_HEIGHTS_KEY]

        panel_names = radar_plotting.radar_fields_and_heights_to_panel_names(
            field_names=field_name_by_panel, heights_m_agl=heights_m_agl)

        plot_cbar_by_panel = numpy.full(len(panel_names), True, dtype=bool)
    else:
        field_name_by_panel, panel_names = (
            radar_plotting.layer_ops_to_field_and_panel_names(
                list_of_layer_operation_dicts)
        )

        plot_cbar_by_panel = numpy.full(len(panel_names), False, dtype=bool)
        plot_cbar_by_panel[2::3] = True

    num_examples = radar_matrix.shape[0]
    num_channels = radar_matrix.shape[-1]
    num_panel_rows = int(numpy.floor(
        numpy.sqrt(num_channels)
    ))

    if class_activation_matrix is None:
        quantity_string = 'max absolute value'
        pathless_file_name_prefix = 'guided-gradcam'
    else:
        quantity_string = 'max class activation'
        pathless_file_name_prefix = 'gradcam'

    for i in range(num_examples):
        _, this_axes_object_matrix = (
            radar_plotting.plot_many_2d_grids_without_coords(
                field_matrix=numpy.flip(radar_matrix[i, ...], axis=0),
                field_name_by_panel=field_name_by_panel,
                panel_names=panel_names, num_panel_rows=num_panel_rows,
                plot_colour_bar_by_panel=plot_cbar_by_panel,
                font_size=FONT_SIZE_WITH_COLOUR_BARS, row_major=False)
        )

        if class_activation_matrix is None:
            this_matrix = guided_class_activation_matrix[i, ...]
            this_max_contour_level = numpy.percentile(
                numpy.absolute(this_matrix), max_colour_prctile_for_cam
            )

            saliency_plotting.plot_many_2d_grids_with_contours(
                saliency_matrix_3d=numpy.flip(this_matrix, axis=0),
                axes_object_matrix=this_axes_object_matrix,
                colour_map_object=cam_colour_map_object,
                max_absolute_contour_level=this_max_contour_level,
                contour_interval=this_max_contour_level / HALF_NUM_CONTOURS,
                row_major=False)

            if monte_carlo_dict is not None:
                significance_plotting.plot_many_2d_grids_without_coords(
                    significance_matrix=numpy.flip(
                        significance_matrix[i, ...], axis=0),
                    axes_object_matrix=this_axes_object_matrix, row_major=False
                )

        else:
            this_matrix = numpy.expand_dims(
                class_activation_matrix[i, ...], axis=-1)
            this_matrix = numpy.repeat(
                this_matrix, repeats=num_channels, axis=-1)

            this_max_contour_level = numpy.percentile(
                this_matrix, max_colour_prctile_for_cam)

            cam_plotting.plot_many_2d_grids(
                class_activation_matrix_3d=numpy.flip(this_matrix, axis=0),
                axes_object_matrix=this_axes_object_matrix,
                colour_map_object=cam_colour_map_object,
                max_contour_level=this_max_contour_level,
                contour_interval=this_max_contour_level / NUM_CONTOURS,
                row_major=False)

            if monte_carlo_dict is not None:
                this_matrix = numpy.expand_dims(
                    significance_matrix[i, ...], axis=-1)
                this_matrix = numpy.repeat(
                    this_matrix, repeats=num_channels, axis=-1)

                significance_plotting.plot_many_2d_grids_without_coords(
                    significance_matrix=numpy.flip(this_matrix, axis=0),
                    axes_object_matrix=this_axes_object_matrix, row_major=False
                )

        if pmm_flag:
            this_title_string = 'Probability-matched mean'
            this_figure_file_name = '{0:s}/{1:s}_pmm_radar.jpg'.format(
                output_dir_name, pathless_file_name_prefix)

        else:
            this_storm_time_string = time_conversion.unix_sec_to_string(
                storm_times_unix_sec[i], TIME_FORMAT)

            this_title_string = 'Storm "{0:s}" at {1:s}'.format(
                full_id_strings[i], this_storm_time_string)

            this_figure_file_name = '{0:s}/{1:s}_{2:s}_{3:s}_radar.jpg'.format(
                output_dir_name, pathless_file_name_prefix,
                full_id_strings[i].replace('_', '-'), this_storm_time_string
            )

        this_title_string += ' ({0:s} = {1:.3f})'.format(
            quantity_string, this_max_contour_level)
        pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)

        print('Saving figure to file: "{0:s}"...'.format(this_figure_file_name))
        pyplot.savefig(this_figure_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()


def _run(input_file_name, plot_significance, cam_colour_map_name,
         max_colour_prctile_for_cam, top_output_dir_name):
    """Plots Grad-CAM output (class-activation maps).

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param plot_significance: Same.
    :param cam_colour_map_name: Same.
    :param max_colour_prctile_for_cam: Same.
    :param top_output_dir_name: Same.
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

    print('Reading data from: "{0:s}"...'.format(input_file_name))

    try:
        gradcam_dict = gradcam.read_standard_file(input_file_name)
        list_of_input_matrices = gradcam_dict.pop(gradcam.INPUT_MATRICES_KEY)
        list_of_cam_matrices = gradcam_dict.pop(gradcam.CAM_MATRICES_KEY)
        list_of_guided_cam_matrices = gradcam_dict.pop(
            gradcam.GUIDED_CAM_MATRICES_KEY)

        gradcam_metadata_dict = gradcam_dict
        full_id_strings = gradcam_metadata_dict[gradcam.FULL_IDS_KEY]
        storm_times_unix_sec = gradcam_metadata_dict[gradcam.STORM_TIMES_KEY]

    except ValueError:
        gradcam_dict = gradcam.read_pmm_file(input_file_name)
        list_of_input_matrices = gradcam_dict[gradcam.MEAN_INPUT_MATRICES_KEY]
        list_of_cam_matrices = gradcam_dict[gradcam.MEAN_CAM_MATRICES_KEY]
        list_of_guided_cam_matrices = gradcam_dict[
            gradcam.MEAN_GUIDED_CAM_MATRICES_KEY]

        for i in range(len(list_of_input_matrices)):
            list_of_input_matrices[i] = numpy.expand_dims(
                list_of_input_matrices[i], axis=0
            )

            if list_of_cam_matrices[i] is None:
                continue

            list_of_cam_matrices[i] = numpy.expand_dims(
                list_of_cam_matrices[i], axis=0
            )
            list_of_guided_cam_matrices[i] = numpy.expand_dims(
                list_of_guided_cam_matrices[i], axis=0
            )

        gradcam_metadata_dict = gradcam_dict
        full_id_strings = None
        storm_times_unix_sec = None

    # Read metadata for CNN.
    model_file_name = gradcam_metadata_dict[gradcam.MODEL_FILE_KEY]
    model_metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0]
    )

    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_model_metadata(model_metafile_name)
    print(SEPARATOR_STRING)

    cam_monte_carlo_dict = (
        gradcam_metadata_dict[gradcam.CAM_MONTE_CARLO_KEY]
        if plot_significance and
        gradcam.CAM_MONTE_CARLO_KEY in gradcam_metadata_dict
        else None
    )

    guided_cam_monte_carlo_dict = (
        gradcam_metadata_dict[gradcam.GUIDED_CAM_MONTE_CARLO_KEY]
        if plot_significance and
        gradcam.GUIDED_CAM_MONTE_CARLO_KEY in gradcam_metadata_dict
        else None
    )

    # Do plotting.
    for i in range(len(list_of_input_matrices)):
        if list_of_cam_matrices[i] is None:
            continue

        this_num_spatial_dim = len(list_of_input_matrices[i].shape) - 2

        if this_num_spatial_dim == 3:
            _plot_3d_radar_cams(
                radar_matrix=list_of_input_matrices[i],
                class_activation_matrix=list_of_cam_matrices[i],
                model_metadata_dict=model_metadata_dict,
                cam_colour_map_object=cam_colour_map_object,
                max_colour_prctile_for_cam=max_colour_prctile_for_cam,
                output_dir_name=main_gradcam_dir_name,
                full_id_strings=full_id_strings,
                storm_times_unix_sec=storm_times_unix_sec,
                monte_carlo_dict=cam_monte_carlo_dict, monte_carlo_index=i)

            print(SEPARATOR_STRING)

            _plot_3d_radar_cams(
                radar_matrix=list_of_input_matrices[i],
                guided_class_activation_matrix=list_of_guided_cam_matrices[i],
                model_metadata_dict=model_metadata_dict,
                cam_colour_map_object=cam_colour_map_object,
                max_colour_prctile_for_cam=max_colour_prctile_for_cam,
                output_dir_name=guided_gradcam_dir_name,
                full_id_strings=full_id_strings,
                storm_times_unix_sec=storm_times_unix_sec,
                monte_carlo_dict=guided_cam_monte_carlo_dict,
                monte_carlo_index=i)

            print(SEPARATOR_STRING)

        if this_num_spatial_dim == 2:
            _plot_2d_radar_cams(
                radar_matrix=list_of_input_matrices[i],
                class_activation_matrix=list_of_cam_matrices[i],
                model_metadata_dict=model_metadata_dict,
                cam_colour_map_object=cam_colour_map_object,
                max_colour_prctile_for_cam=max_colour_prctile_for_cam,
                output_dir_name=main_gradcam_dir_name,
                full_id_strings=full_id_strings,
                storm_times_unix_sec=storm_times_unix_sec,
                monte_carlo_dict=cam_monte_carlo_dict, monte_carlo_index=i)

            print(SEPARATOR_STRING)

            _plot_2d_radar_cams(
                radar_matrix=list_of_input_matrices[i],
                guided_class_activation_matrix=list_of_guided_cam_matrices[i],
                model_metadata_dict=model_metadata_dict,
                cam_colour_map_object=cam_colour_map_object,
                max_colour_prctile_for_cam=max_colour_prctile_for_cam,
                output_dir_name=guided_gradcam_dir_name,
                full_id_strings=full_id_strings,
                storm_times_unix_sec=storm_times_unix_sec,
                monte_carlo_dict=guided_cam_monte_carlo_dict,
                monte_carlo_index=i)

            print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        plot_significance=bool(getattr(
            INPUT_ARG_OBJECT, PLOT_SIGNIFICANCE_ARG_NAME)),
        cam_colour_map_name=getattr(INPUT_ARG_OBJECT, COLOUR_MAP_ARG_NAME),
        max_colour_prctile_for_cam=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
