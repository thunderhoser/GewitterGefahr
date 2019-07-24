"""Plots Grad-CAM output (class-activation maps)."""

import os
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import monte_carlo
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import gradcam
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import cam_plotting
from gewittergefahr.plotting import saliency_plotting
from gewittergefahr.plotting import significance_plotting
from gewittergefahr.scripts import plot_input_examples

# TODO(thunderhoser): Use threshold counts at some point.
# TODO(thunderhoser): Make this script deal with soundings.

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

REGION_COLOUR = numpy.full(3, 0.)
REGION_LINE_WIDTH = 3

NUM_CONTOURS = 12
HALF_NUM_CONTOURS = 10
FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_file_name'
ALLOW_WHITESPACE_ARG_NAME = 'allow_whitespace'
PLOT_SIGNIFICANCE_ARG_NAME = 'plot_significance'
PLOT_REGIONS_ARG_NAME = 'plot_regions_of_interest'
COLOUR_MAP_ARG_NAME = 'colour_map_name'
MAX_PERCENTILE_ARG_NAME = 'max_colour_percentile'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `gradcam.read_standard_file` or'
    ' `gradcam.read_pmm_file`.')

ALLOW_WHITESPACE_HELP_STRING = (
    'Boolean flag.  If 0, will plot with no whitespace between panels or around'
    ' outside of image.')

PLOT_SIGNIFICANCE_HELP_STRING = (
    'Boolean flag.  If 1, will plot stippling for significance.  This applies '
    'only if the saliency map contains PMM (proability-matched means) and '
    'results of a Monte Carlo comparison.')

PLOT_REGIONS_HELP_STRING = (
    'Boolean flag.  If 1, will plot regions of interest (as polygons on top of '
    'class-activation maps).')

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
    '--' + ALLOW_WHITESPACE_ARG_NAME, type=int, required=False, default=1,
    help=ALLOW_WHITESPACE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_SIGNIFICANCE_ARG_NAME, type=int, required=False, default=0,
    help=PLOT_SIGNIFICANCE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_REGIONS_ARG_NAME, type=int, required=False, default=0,
    help=PLOT_REGIONS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + COLOUR_MAP_ARG_NAME, type=str, required=False, default='gist_yarg',
    help=COLOUR_MAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False,
    default=99, help=MAX_PERCENTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _plot_3d_radar_cam(
        colour_map_object, max_colour_percentile, figure_objects,
        axes_object_matrices, model_metadata_dict, output_dir_name,
        cam_matrix=None, guided_cam_matrix=None, significance_matrix=None,
        full_storm_id_string=None, storm_time_unix_sec=None):
    """Plots guided or unguided class-activation map for 3-D radar data.

    This method will plot either `cam_matrix` or `guided_cam_matrix`, not both.

    M = number of rows in spatial grid
    N = number of columns in spatial grid
    H = number of heights in spatial grid
    F = number of radar fields

    If this method is plotting a composite rather than single example (storm
    object), `full_storm_id_string` and `storm_time_unix_sec` can be None.

    :param colour_map_object: See documentation at top of file.
    :param max_colour_percentile: Same.
    :param figure_objects: See doc for `plot_input_examples._plot_3d_example`.
    :param axes_object_matrices: Same.
    :param model_metadata_dict: Dictionary returned by
        `cnn.read_model_metadata`.
    :param output_dir_name: Path to output directory.  Figure(s) will be saved
        here.
    :param cam_matrix: M-by-N-by-H numpy array of class activations.
    :param guided_cam_matrix: M-by-N-by-H-by-F numpy array of guided-CAM output.
    :param significance_matrix: Boolean numpy array with the same dimensions as
        the array being plotted (`cam_matrix` or `guided_cam_matrix`),
        indicating where differences with some other CAM are significant.
    :param full_storm_id_string: Full storm ID.
    :param storm_time_unix_sec: Storm time.
    """

    if cam_matrix is None:
        quantity_string = 'max abs value'
    else:
        quantity_string = 'max activation'

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
            this_matrix = guided_cam_matrix[..., j]
        else:
            this_matrix = cam_matrix

        this_max_contour_level = numpy.percentile(
            numpy.absolute(this_matrix), max_colour_percentile
        )

        if cam_matrix is None:
            saliency_plotting.plot_many_2d_grids_with_contours(
                saliency_matrix_3d=numpy.flip(this_matrix, axis=0),
                axes_object_matrix=axes_object_matrices[j],
                colour_map_object=colour_map_object,
                max_absolute_contour_level=this_max_contour_level,
                contour_interval=this_max_contour_level / HALF_NUM_CONTOURS)
        else:
            cam_plotting.plot_many_2d_grids(
                class_activation_matrix_3d=numpy.flip(this_matrix, axis=0),
                axes_object_matrix=axes_object_matrices[j],
                colour_map_object=colour_map_object,
                max_contour_level=this_max_contour_level,
                contour_interval=this_max_contour_level / NUM_CONTOURS)

        if significance_matrix is not None:
            if cam_matrix is None:
                this_matrix = significance_matrix[..., j]
            else:
                this_matrix = significance_matrix

            significance_plotting.plot_many_2d_grids_without_coords(
                significance_matrix=numpy.flip(this_matrix, axis=0),
                axes_object_matrix=axes_object_matrices[j]
            )

        this_title_object = figure_objects[j]._suptitle

        if this_title_object is not None:
            this_title_string = '{0:s} ({1:s} = {2:.2e})'.format(
                this_title_object.get_text(), quantity_string,
                this_max_contour_level)

            figure_objects[j].suptitle(
                this_title_string, fontsize=plot_input_examples.TITLE_FONT_SIZE)

        this_file_name = plot_input_examples.metadata_to_file_name(
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
        colour_map_object, max_colour_percentile, figure_objects,
        axes_object_matrices, model_metadata_dict, output_dir_name,
        cam_matrix=None, guided_cam_matrix=None, significance_matrix=None,
        full_storm_id_string=None, storm_time_unix_sec=None):
    """Plots guided or unguided class-activation map for 2-D radar data.

    M = number of rows in spatial grid
    N = number of columns in spatial grid
    C = number of radar channels

    :param colour_map_object: See doc for `_plot_3d_radar_cam`.
    :param max_colour_percentile: Same.
    :param figure_objects: Same.
    :param axes_object_matrices: Same.
    :param model_metadata_dict: Same.
    :param output_dir_name: Same.
    :param cam_matrix: M-by-N numpy array of class activations.
    :param guided_cam_matrix: M-by-N-by-C numpy array of guided-CAM output.
    :param significance_matrix: See doc for `_plot_3d_radar_cam`.
    :param full_storm_id_string: Same.
    :param storm_time_unix_sec: Same.
    """

    if cam_matrix is None:
        quantity_string = 'max abs value'
    else:
        quantity_string = 'max activation'

    pmm_flag = full_storm_id_string is None and storm_time_unix_sec is None
    conv_2d3d = model_metadata_dict[cnn.CONV_2D3D_KEY]
    figure_index = 1 if conv_2d3d else 0

    list_of_layer_operation_dicts = model_metadata_dict[
        cnn.LAYER_OPERATIONS_KEY]

    if list_of_layer_operation_dicts is None:
        training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
        radar_field_names = training_option_dict[trainval_io.RADAR_FIELDS_KEY]
        num_channels = len(radar_field_names)
    else:
        num_channels = len(list_of_layer_operation_dicts)

    if cam_matrix is None:
        this_matrix = guided_cam_matrix
    else:
        this_matrix = numpy.expand_dims(cam_matrix, axis=-1)
        this_matrix = numpy.repeat(this_matrix, repeats=num_channels, axis=-1)

    if list_of_layer_operation_dicts is not None:
        this_matrix = this_matrix[
            ..., plot_input_examples.LAYER_OP_INDICES_TO_KEEP
        ]

    max_contour_level = numpy.percentile(
        numpy.absolute(this_matrix), max_colour_percentile
    )

    if cam_matrix is None:
        saliency_plotting.plot_many_2d_grids_with_contours(
            saliency_matrix_3d=numpy.flip(this_matrix, axis=0),
            axes_object_matrix=axes_object_matrices[figure_index],
            colour_map_object=colour_map_object,
            max_absolute_contour_level=max_contour_level,
            contour_interval=max_contour_level / HALF_NUM_CONTOURS,
            row_major=False)
    else:
        cam_plotting.plot_many_2d_grids(
            class_activation_matrix_3d=numpy.flip(this_matrix, axis=0),
            axes_object_matrix=axes_object_matrices[figure_index],
            colour_map_object=colour_map_object,
            max_contour_level=max_contour_level,
            contour_interval=max_contour_level / NUM_CONTOURS,
            row_major=False)

    if significance_matrix is not None:
        if cam_matrix is None:
            this_matrix = significance_matrix
        else:
            this_matrix = numpy.expand_dims(significance_matrix, axis=-1)
            this_matrix = numpy.repeat(
                this_matrix, repeats=num_channels, axis=-1)

        if list_of_layer_operation_dicts is not None:
            this_matrix = this_matrix[
                ..., plot_input_examples.LAYER_OP_INDICES_TO_KEEP
            ]

        significance_plotting.plot_many_2d_grids_without_coords(
            significance_matrix=numpy.flip(this_matrix, axis=0),
            axes_object_matrix=axes_object_matrices[figure_index],
            row_major=False
        )

    this_title_object = figure_objects[figure_index]._suptitle

    if this_title_object is not None:
        this_title_string = '{0:s} ({1:s} = {2:.2e})'.format(
            this_title_object.get_text(), quantity_string, max_contour_level
        )

        figure_objects[figure_index].suptitle(
            this_title_string, fontsize=plot_input_examples.TITLE_FONT_SIZE)

    output_file_name = plot_input_examples.metadata_to_file_name(
        output_dir_name=output_dir_name, is_sounding=False, pmm_flag=pmm_flag,
        full_storm_id_string=full_storm_id_string,
        storm_time_unix_sec=storm_time_unix_sec,
        radar_field_name='shear' if conv_2d3d else None)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_objects[figure_index].savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_objects[figure_index])


def _plot_2d_regions(
        figure_objects, axes_object_matrices, model_metadata_dict,
        list_of_polygon_objects, output_dir_name, full_storm_id_string=None,
        storm_time_unix_sec=None):
    """Plots regions of interest for 2-D radar data.

    :param figure_objects: See doc for `_plot_3d_radar_cam`.
    :param axes_object_matrices: Same.
    :param model_metadata_dict: Same.
    :param list_of_polygon_objects: List of polygons (instances of
        `shapely.geometry.Polygon`), demarcating regions of interest.
    :param output_dir_name: See doc for `_plot_3d_radar_cam`.
    :param full_storm_id_string: Same.
    :param storm_time_unix_sec: Same.
    """

    conv_2d3d = model_metadata_dict[cnn.CONV_2D3D_KEY]
    figure_index = 1 if conv_2d3d else 0

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    num_grid_rows = training_option_dict[trainval_io.NUM_ROWS_KEY]
    num_grid_rows *= 1 + int(conv_2d3d)

    list_of_layer_operation_dicts = model_metadata_dict[
        cnn.LAYER_OPERATIONS_KEY]

    if list_of_layer_operation_dicts is None:
        num_channels = len(training_option_dict[trainval_io.RADAR_FIELDS_KEY])
    else:
        num_channels = len(list_of_layer_operation_dicts)

    for this_polygon_object in list_of_polygon_objects:
        for k in range(num_channels):
            i, j = numpy.unravel_index(
                k, axes_object_matrices[figure_index].shape, order='F'
            )

            these_grid_columns = numpy.array(
                this_polygon_object.exterior.xy[0]
            )
            these_grid_rows = num_grid_rows - numpy.array(
                this_polygon_object.exterior.xy[1]
            )

            axes_object_matrices[figure_index][i, j].plot(
                these_grid_columns, these_grid_rows,
                color=plotting_utils.colour_from_numpy_to_tuple(
                    REGION_COLOUR),
                linestyle='solid', linewidth=REGION_LINE_WIDTH
            )

    pmm_flag = full_storm_id_string is None and storm_time_unix_sec is None

    output_file_name = plot_input_examples.metadata_to_file_name(
        output_dir_name=output_dir_name, is_sounding=False, pmm_flag=pmm_flag,
        full_storm_id_string=full_storm_id_string,
        storm_time_unix_sec=storm_time_unix_sec,
        radar_field_name='shear' if conv_2d3d else None)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_objects[figure_index].savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_objects[figure_index])


def _run(input_file_name, allow_whitespace, plot_significance,
         plot_regions_of_interest, colour_map_name, max_colour_percentile,
         top_output_dir_name):
    """Plots Grad-CAM output (class-activation maps).

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param allow_whitespace: Same.
    :param plot_significance: Same.
    :param plot_regions_of_interest: Same.
    :param colour_map_name: Same.
    :param max_colour_percentile: Same.
    :param top_output_dir_name: Same.
    """

    if plot_significance:
        plot_regions_of_interest = False

    unguided_cam_dir_name = '{0:s}/main_gradcam'.format(top_output_dir_name)
    guided_cam_dir_name = '{0:s}/guided_gradcam'.format(top_output_dir_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=unguided_cam_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=guided_cam_dir_name)

    # Check input args.
    error_checking.assert_is_geq(max_colour_percentile, 0.)
    error_checking.assert_is_leq(max_colour_percentile, 100.)
    colour_map_object = pyplot.cm.get_cmap(colour_map_name)

    print('Reading data from: "{0:s}"...'.format(input_file_name))

    try:
        gradcam_dict = gradcam.read_standard_file(input_file_name)
        list_of_input_matrices = gradcam_dict[gradcam.INPUT_MATRICES_KEY]
        list_of_cam_matrices = gradcam_dict[gradcam.CAM_MATRICES_KEY]
        list_of_guided_cam_matrices = gradcam_dict[
            gradcam.GUIDED_CAM_MATRICES_KEY]

        full_storm_id_strings = gradcam_dict[gradcam.FULL_IDS_KEY]
        storm_times_unix_sec = gradcam_dict[gradcam.STORM_TIMES_KEY]

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

        full_storm_id_strings = [None]
        storm_times_unix_sec = [None]

    pmm_flag = (
        full_storm_id_strings[0] is None and storm_times_unix_sec[0] is None
    )

    # Read metadata for CNN.
    model_file_name = gradcam_dict[gradcam.MODEL_FILE_KEY]
    model_metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0]
    )

    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_model_metadata(model_metafile_name)
    print(SEPARATOR_STRING)

    cam_monte_carlo_dict = (
        gradcam_dict[gradcam.CAM_MONTE_CARLO_KEY]
        if plot_significance else None
    )

    guided_cam_monte_carlo_dict = (
        gradcam_dict[gradcam.GUIDED_CAM_MONTE_CARLO_KEY]
        if plot_significance else None
    )

    region_dict = (
        gradcam_dict[gradcam.REGION_DICT_KEY]
        if plot_regions_of_interest else None
    )

    num_examples = list_of_input_matrices[0].shape[0]
    num_input_matrices = len(list_of_input_matrices)

    for i in range(num_examples):
        this_handle_dict = plot_input_examples.plot_one_example(
            list_of_predictor_matrices=list_of_input_matrices,
            model_metadata_dict=model_metadata_dict, plot_sounding=False,
            allow_whitespace=allow_whitespace, pmm_flag=pmm_flag,
            example_index=i, full_storm_id_string=full_storm_id_strings[i],
            storm_time_unix_sec=storm_times_unix_sec[i]
        )

        these_figure_objects = this_handle_dict[
            plot_input_examples.RADAR_FIGURES_KEY]
        these_axes_object_matrices = this_handle_dict[
            plot_input_examples.RADAR_AXES_KEY]

        for j in range(num_input_matrices):
            if list_of_cam_matrices[j] is None:
                continue

            if cam_monte_carlo_dict is None:
                this_significance_matrix = None
            else:
                this_significance_matrix = numpy.logical_or(
                    cam_monte_carlo_dict[
                        monte_carlo.TRIAL_PMM_MATRICES_KEY][j][i, ...] <
                    cam_monte_carlo_dict[
                        monte_carlo.MIN_MATRICES_KEY][j][i, ...],
                    cam_monte_carlo_dict[
                        monte_carlo.TRIAL_PMM_MATRICES_KEY][j][i, ...] >
                    cam_monte_carlo_dict[
                        monte_carlo.MAX_MATRICES_KEY][j][i, ...]
                )

            this_num_spatial_dim = len(list_of_input_matrices[j].shape) - 2

            if this_num_spatial_dim == 3:
                _plot_3d_radar_cam(
                    colour_map_object=colour_map_object,
                    max_colour_percentile=max_colour_percentile,
                    figure_objects=these_figure_objects,
                    axes_object_matrices=these_axes_object_matrices,
                    model_metadata_dict=model_metadata_dict,
                    output_dir_name=unguided_cam_dir_name,
                    cam_matrix=list_of_cam_matrices[j][i, ...],
                    significance_matrix=this_significance_matrix,
                    full_storm_id_string=full_storm_id_strings[i],
                    storm_time_unix_sec=storm_times_unix_sec[i]
                )
            else:
                if region_dict is None:
                    _plot_2d_radar_cam(
                        colour_map_object=colour_map_object,
                        max_colour_percentile=max_colour_percentile,
                        figure_objects=these_figure_objects,
                        axes_object_matrices=these_axes_object_matrices,
                        model_metadata_dict=model_metadata_dict,
                        output_dir_name=unguided_cam_dir_name,
                        cam_matrix=list_of_cam_matrices[j][i, ...],
                        significance_matrix=this_significance_matrix,
                        full_storm_id_string=full_storm_id_strings[i],
                        storm_time_unix_sec=storm_times_unix_sec[i]
                    )
                else:
                    _plot_2d_regions(
                        figure_objects=these_figure_objects,
                        axes_object_matrices=these_axes_object_matrices,
                        model_metadata_dict=model_metadata_dict,
                        list_of_polygon_objects=
                        region_dict[gradcam.POLYGON_OBJECTS_KEY][j][i],
                        output_dir_name=unguided_cam_dir_name,
                        full_storm_id_string=full_storm_id_strings[i],
                        storm_time_unix_sec=storm_times_unix_sec[i]
                    )

        this_handle_dict = plot_input_examples.plot_one_example(
            list_of_predictor_matrices=list_of_input_matrices,
            model_metadata_dict=model_metadata_dict, plot_sounding=False,
            allow_whitespace=allow_whitespace, pmm_flag=pmm_flag,
            example_index=i, full_storm_id_string=full_storm_id_strings[i],
            storm_time_unix_sec=storm_times_unix_sec[i]
        )

        these_figure_objects = this_handle_dict[
            plot_input_examples.RADAR_FIGURES_KEY]
        these_axes_object_matrices = this_handle_dict[
            plot_input_examples.RADAR_AXES_KEY]

        for j in range(num_input_matrices):
            if list_of_guided_cam_matrices[j] is None:
                continue

            if guided_cam_monte_carlo_dict is None:
                this_significance_matrix = None
            else:
                this_significance_matrix = numpy.logical_or(
                    guided_cam_monte_carlo_dict[
                        monte_carlo.TRIAL_PMM_MATRICES_KEY][j][i, ...] <
                    guided_cam_monte_carlo_dict[
                        monte_carlo.MIN_MATRICES_KEY][j][i, ...],
                    guided_cam_monte_carlo_dict[
                        monte_carlo.TRIAL_PMM_MATRICES_KEY][j][i, ...] >
                    guided_cam_monte_carlo_dict[
                        monte_carlo.MAX_MATRICES_KEY][j][i, ...]
                )

            this_num_spatial_dim = len(list_of_input_matrices[j].shape) - 2

            if this_num_spatial_dim == 3:
                _plot_3d_radar_cam(
                    colour_map_object=colour_map_object,
                    max_colour_percentile=max_colour_percentile,
                    figure_objects=these_figure_objects,
                    axes_object_matrices=these_axes_object_matrices,
                    model_metadata_dict=model_metadata_dict,
                    output_dir_name=guided_cam_dir_name,
                    guided_cam_matrix=list_of_guided_cam_matrices[j][i, ...],
                    significance_matrix=this_significance_matrix,
                    full_storm_id_string=full_storm_id_strings[i],
                    storm_time_unix_sec=storm_times_unix_sec[i]
                )
            else:
                _plot_2d_radar_cam(
                    colour_map_object=colour_map_object,
                    max_colour_percentile=max_colour_percentile,
                    figure_objects=these_figure_objects,
                    axes_object_matrices=these_axes_object_matrices,
                    model_metadata_dict=model_metadata_dict,
                    output_dir_name=guided_cam_dir_name,
                    guided_cam_matrix=list_of_guided_cam_matrices[j][i, ...],
                    significance_matrix=this_significance_matrix,
                    full_storm_id_string=full_storm_id_strings[i],
                    storm_time_unix_sec=storm_times_unix_sec[i]
                )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        allow_whitespace=bool(getattr(
            INPUT_ARG_OBJECT, ALLOW_WHITESPACE_ARG_NAME
        )),
        plot_significance=bool(getattr(
            INPUT_ARG_OBJECT, PLOT_SIGNIFICANCE_ARG_NAME)),
        plot_regions_of_interest=bool(getattr(
            INPUT_ARG_OBJECT, PLOT_REGIONS_ARG_NAME)),
        colour_map_name=getattr(INPUT_ARG_OBJECT, COLOUR_MAP_ARG_NAME),
        max_colour_percentile=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
