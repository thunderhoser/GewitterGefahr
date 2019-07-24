"""Plots results of backwards optimization."""

import os.path
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import monte_carlo
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.deep_learning import backwards_optimization as backwards_opt
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import radar_plotting
from gewittergefahr.plotting import sounding_plotting
from gewittergefahr.plotting import significance_plotting
from gewittergefahr.scripts import plot_input_examples

# TODO(thunderhoser): A lot of this code should be in plot_input_examples.py.

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TITLE_FONT_SIZE = 12
FONT_SIZE_WITH_COLOUR_BARS = 16
FONT_SIZE_SANS_COLOUR_BARS = 20
FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_file_name'
PLOT_SIGNIFICANCE_ARG_NAME = 'plot_significance'
COLOUR_MAP_ARG_NAME = 'diff_colour_map_name'
MAX_PERCENTILE_ARG_NAME = 'max_colour_percentile'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `backwards_opt.read_standard_file` or'
    ' `backwards_opt.read_pmm_file`.')

PLOT_SIGNIFICANCE_HELP_STRING = (
    'Boolean flag.  If 1, will plot stippling for significance.  This applies '
    'only if the saliency map contains PMM (proability-matched means) and '
    'results of a Monte Carlo comparison.')

COLOUR_MAP_HELP_STRING = (
    'Name of colour map.  Differences for each predictor will be plotted with '
    'the same colour map.  For example, if name is "Greys", the colour map used'
    ' will be `pyplot.cm.Greys`.  This argument supports only pyplot colour '
    'maps.')

MAX_PERCENTILE_HELP_STRING = (
    'Determines max absolute value for each difference map.  The max absolute '
    'value for example e and predictor p will be the [q]th percentile of all '
    'differences for example, where q = `{0:s}`.'
).format(MAX_PERCENTILE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_SIGNIFICANCE_ARG_NAME, type=int, required=False, default=0,
    help=PLOT_SIGNIFICANCE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + COLOUR_MAP_ARG_NAME, type=str, required=False, default='bwr',
    help=COLOUR_MAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False,
    default=99., help=MAX_PERCENTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _plot_3d_radar_difference(
        difference_matrix, colour_map_object, max_colour_percentile,
        model_metadata_dict, backwards_opt_dict, output_dir_name,
        example_index=None, significance_matrix=None):
    """Plots difference (after minus before optimization) for 3-D radar data.

    M = number of rows in spatial grid
    N = number of columns in spatial grid
    H = number of heights in spatial grid
    F = number of fields

    :param difference_matrix: M-by-N-by-H-by-F numpy array of differences (after
        minus before optimization).
    :param colour_map_object: See documentation at top of file.
    :param max_colour_percentile: Same.
    :param model_metadata_dict: Dictionary returned by
        `cnn.read_model_metadata`.
    :param backwards_opt_dict: Dictionary returned by
        `backwards_optimization.read_standard_file` or
        `backwards_optimization.read_pmm_file`, containing metadata.
    :param output_dir_name: Name of output directory.  Figure(s) will be saved
        here.
    :param example_index: This method will plot only the [i]th example, where
        i = `example_index`.  This will be used to find metadata for the given
        example in `backwards_opt_dict`.  If `backwards_opt_dict` contains PMM
        (probability-matched means), leave this argument alone.
    :param significance_matrix: M-by-N-by-H-by-F numpy array of Boolean flags,
        indicating where these differences are significantly different than
        differences from another backwards optimization.
    """

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    radar_heights_m_agl = training_option_dict[trainval_io.RADAR_HEIGHTS_KEY]
    num_heights = len(radar_heights_m_agl)

    num_panel_rows = int(numpy.floor(
        numpy.sqrt(num_heights)
    ))

    pmm_flag = backwards_opt.MEAN_FINAL_ACTIVATION_KEY in backwards_opt_dict
    if pmm_flag:
        initial_activation = backwards_opt_dict[
            backwards_opt.MEAN_INITIAL_ACTIVATION_KEY]
        final_activation = backwards_opt_dict[
            backwards_opt.MEAN_FINAL_ACTIVATION_KEY]

        full_storm_id_string = None
        storm_time_string = None
    else:
        initial_activation = backwards_opt_dict[
            backwards_opt.INITIAL_ACTIVATIONS_KEY][example_index]
        final_activation = backwards_opt_dict[
            backwards_opt.FINAL_ACTIVATIONS_KEY][example_index]

        full_storm_id_string = backwards_opt_dict[
            backwards_opt.FULL_IDS_KEY][example_index]

        storm_time_string = time_conversion.unix_sec_to_string(
            backwards_opt_dict[backwards_opt.STORM_TIMES_KEY][example_index],
            plot_input_examples.TIME_FORMAT
        )

    conv_2d3d = model_metadata_dict[cnn.CONV_2D3D_KEY]
    if conv_2d3d:
        radar_field_names = [radar_utils.REFL_NAME]
    else:
        radar_field_names = training_option_dict[trainval_io.RADAR_FIELDS_KEY]

    num_fields = len(radar_field_names)

    for j in range(num_fields):
        this_max_colour_value = numpy.percentile(
            numpy.absolute(difference_matrix[..., j]), max_colour_percentile
        )

        this_colour_norm_object = matplotlib.colors.Normalize(
            vmin=-1 * this_max_colour_value, vmax=this_max_colour_value,
            clip=False)

        # TODO(thunderhoser): Deal with change of units.
        this_figure_object, this_axes_object_matrix = (
            radar_plotting.plot_3d_grid_without_coords(
                field_matrix=numpy.flip(difference_matrix[..., j], axis=0),
                field_name=radar_field_names[j],
                grid_point_heights_metres=radar_heights_m_agl,
                ground_relative=True, num_panel_rows=num_panel_rows,
                font_size=FONT_SIZE_SANS_COLOUR_BARS,
                colour_map_object=colour_map_object,
                colour_norm_object=this_colour_norm_object)
        )

        if significance_matrix is not None:
            this_matrix = numpy.flip(significance_matrix[..., j], axis=0)

            significance_plotting.plot_many_2d_grids_without_coords(
                significance_matrix=this_matrix,
                axes_object_matrix=this_axes_object_matrix)

        plotting_utils.plot_colour_bar(
            axes_object_or_matrix=this_axes_object_matrix,
            data_matrix=difference_matrix[..., j],
            colour_map_object=colour_map_object,
            colour_norm_object=this_colour_norm_object,
            orientation_string='horizontal', extend_min=True,
            extend_max=True)

        if pmm_flag:
            this_title_string = 'PMM'
        else:
            this_title_string = 'Storm "{0:s}" at {1:s}'.format(
                full_storm_id_string, storm_time_string)

        this_title_string += (
            '; {0:s}; activation from {1:.2e} to {2:.2e}'
        ).format(
            radar_field_names[j], initial_activation, final_activation
        )

        this_figure_object.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)

        this_file_name = plot_input_examples.metadata_to_radar_fig_file_name(
            output_dir_name=output_dir_name, pmm_flag=pmm_flag,
            full_storm_id_string=full_storm_id_string,
            storm_time_string=storm_time_string,
            radar_field_name=radar_field_names[j]
        )

        print('Saving figure to: "{0:s}"...'.format(this_file_name))
        this_figure_object.savefig(
            this_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
            bbox_inches='tight'
        )
        pyplot.close(this_figure_object)


def _plot_2d_radar_difference(
        difference_matrix, colour_map_object, max_colour_percentile,
        model_metadata_dict, backwards_opt_dict, output_dir_name,
        example_index=None, significance_matrix=None):
    """Plots difference (after minus before optimization) for 2-D radar data.

    M = number of rows in spatial grid
    N = number of columns in spatial grid
    C = number of channels

    :param difference_matrix: M-by-N-by-C numpy array of differences (after
        minus before optimization).
    :param colour_map_object: See doc for `_plot_3d_radar_difference`.
    :param max_colour_percentile: Same.
    :param model_metadata_dict: Same.
    :param backwards_opt_dict: Same.
    :param output_dir_name: Same.
    :param example_index: Same.
    :param significance_matrix: M-by-N-by-C numpy array of Boolean flags,
        indicating where these differences are significantly different than
        differences from another backwards optimization.
    """

    pmm_flag = backwards_opt.MEAN_FINAL_ACTIVATION_KEY in backwards_opt_dict
    if pmm_flag:
        initial_activation = backwards_opt_dict[
            backwards_opt.MEAN_INITIAL_ACTIVATION_KEY]
        final_activation = backwards_opt_dict[
            backwards_opt.MEAN_FINAL_ACTIVATION_KEY]

        full_storm_id_string = None
        storm_time_string = None
    else:
        initial_activation = backwards_opt_dict[
            backwards_opt.INITIAL_ACTIVATIONS_KEY][example_index]
        final_activation = backwards_opt_dict[
            backwards_opt.FINAL_ACTIVATIONS_KEY][example_index]

        full_storm_id_string = backwards_opt_dict[
            backwards_opt.FULL_IDS_KEY][example_index]

        storm_time_string = time_conversion.unix_sec_to_string(
            backwards_opt_dict[backwards_opt.STORM_TIMES_KEY][example_index],
            plot_input_examples.TIME_FORMAT
        )

    conv_2d3d = model_metadata_dict[cnn.CONV_2D3D_KEY]
    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]

    if conv_2d3d:
        num_fields = len(training_option_dict[trainval_io.RADAR_FIELDS_KEY])
        radar_heights_m_agl = numpy.full(
            num_fields, radar_utils.SHEAR_HEIGHT_M_ASL, dtype=int)
    else:
        radar_heights_m_agl = training_option_dict[
            trainval_io.RADAR_HEIGHTS_KEY]

    list_of_layer_operation_dicts = model_metadata_dict[
        cnn.LAYER_OPERATIONS_KEY]

    if list_of_layer_operation_dicts is None:
        field_name_by_panel = training_option_dict[trainval_io.RADAR_FIELDS_KEY]

        panel_names = radar_plotting.radar_fields_and_heights_to_panel_names(
            field_names=field_name_by_panel, heights_m_agl=radar_heights_m_agl)
    else:
        field_name_by_panel, panel_names = (
            radar_plotting.layer_ops_to_field_and_panel_names(
                list_of_layer_operation_dicts=list_of_layer_operation_dicts
            )
        )

    num_panels = len(field_name_by_panel)
    plot_cbar_by_panel = numpy.full(num_panels, True, dtype=bool)
    cmap_object_by_panel = [colour_map_object] * num_panels
    cnorm_object_by_panel = [None] * num_panels

    for j in range(num_panels):
        this_max_colour_value = numpy.percentile(
            numpy.absolute(difference_matrix[..., j]), max_colour_percentile
        )

        cnorm_object_by_panel[j] = matplotlib.colors.Normalize(
            vmin=-1 * this_max_colour_value, vmax=this_max_colour_value,
            clip=False)

    num_panel_rows = int(numpy.floor(
        numpy.sqrt(num_panels)
    ))

    figure_object, axes_object_matrix = (
        radar_plotting.plot_many_2d_grids_without_coords(
            field_matrix=numpy.flip(difference_matrix, axis=0),
            field_name_by_panel=field_name_by_panel,
            num_panel_rows=num_panel_rows,
            panel_names=panel_names, row_major=False,
            colour_map_object_by_panel=cmap_object_by_panel,
            colour_norm_object_by_panel=cnorm_object_by_panel,
            plot_colour_bar_by_panel=plot_cbar_by_panel,
            font_size=FONT_SIZE_WITH_COLOUR_BARS)
    )

    if significance_matrix is not None:
        significance_plotting.plot_many_2d_grids_without_coords(
            significance_matrix=numpy.flip(significance_matrix, axis=0),
            axes_object_matrix=axes_object_matrix, row_major=False
        )

    if pmm_flag:
        this_title_string = 'PMM'
    else:
        this_title_string = 'Storm "{0:s}" at {1:s}'.format(
            full_storm_id_string, storm_time_string)

    this_title_string += '; activation from {0:.2e} to {1:.2e}'.format(
        initial_activation, final_activation)
    figure_object.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)

    output_file_name = plot_input_examples.metadata_to_radar_fig_file_name(
        output_dir_name=output_dir_name, pmm_flag=pmm_flag,
        full_storm_id_string=full_storm_id_string,
        storm_time_string=storm_time_string,
        radar_field_name='shear' if conv_2d3d else None)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _plot_bwo_for_soundings(
        optimized_sounding_matrix, training_option_dict, pmm_flag,
        backwards_opt_dict, top_output_dir_name, input_sounding_matrix=None):
    """Plots BWO results for soundings.

    E = number of examples (storm objects)
    H = number of sounding heights
    F = number of sounding fields

    :param optimized_sounding_matrix: E-by-H-by-F numpy array of sounding values
        (predictors).
    :param training_option_dict: See doc for `_plot_bwo_for_2d3d_radar`.
    :param pmm_flag: Same.
    :param backwards_opt_dict: Same.
    :param top_output_dir_name: Same.
    :param input_sounding_matrix: Same as `optimized_sounding_matrix` but with
        non-optimized input.
    """

    before_optimization_dir_name = '{0:s}/before_optimization'.format(
        top_output_dir_name)
    after_optimization_dir_name = '{0:s}/after_optimization'.format(
        top_output_dir_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=before_optimization_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=after_optimization_dir_name)

    if pmm_flag:
        full_id_strings = None
        storm_times_unix_sec = None
        have_storm_ids = False

        initial_activations = numpy.array([
            backwards_opt_dict[backwards_opt.MEAN_INITIAL_ACTIVATION_KEY]
        ])
        final_activations = numpy.array([
            backwards_opt_dict[backwards_opt.MEAN_FINAL_ACTIVATION_KEY]
        ])
    else:
        full_id_strings = backwards_opt_dict[backwards_opt.FULL_IDS_KEY]
        storm_times_unix_sec = backwards_opt_dict[backwards_opt.STORM_TIMES_KEY]

        have_storm_ids = not (
            full_id_strings is None or storm_times_unix_sec is None
        )

        initial_activations = backwards_opt_dict[
            backwards_opt.INITIAL_ACTIVATIONS_KEY]
        final_activations = backwards_opt_dict[
            backwards_opt.FINAL_ACTIVATIONS_KEY]

    num_examples = optimized_sounding_matrix.shape[0]

    list_of_optimized_metpy_dicts = dl_utils.soundings_to_metpy_dictionaries(
        sounding_matrix=optimized_sounding_matrix,
        field_names=training_option_dict[trainval_io.SOUNDING_FIELDS_KEY],
        height_levels_m_agl=training_option_dict[
            trainval_io.SOUNDING_HEIGHTS_KEY],
        storm_elevations_m_asl=numpy.zeros(num_examples)
    )

    if input_sounding_matrix is None:
        list_of_input_metpy_dicts = None
    else:
        list_of_input_metpy_dicts = dl_utils.soundings_to_metpy_dictionaries(
            sounding_matrix=input_sounding_matrix,
            field_names=training_option_dict[trainval_io.SOUNDING_FIELDS_KEY],
            height_levels_m_agl=training_option_dict[
                trainval_io.SOUNDING_HEIGHTS_KEY],
            storm_elevations_m_asl=numpy.zeros(num_examples)
        )

    for i in range(num_examples):
        if pmm_flag:
            this_base_title_string = 'Probability-matched mean'
            this_base_pathless_file_name = 'pmm'
        else:
            if have_storm_ids:
                this_storm_time_string = time_conversion.unix_sec_to_string(
                    storm_times_unix_sec[i], TIME_FORMAT)

                this_base_title_string = 'Storm "{0:s}" at {1:s}'.format(
                    full_id_strings[i], this_storm_time_string)

                this_base_pathless_file_name = '{0:s}_{1:s}'.format(
                    full_id_strings[i].replace('_', '-'),
                    this_storm_time_string)
            else:
                this_base_title_string = 'Example {0:d}'.format(i + 1)
                this_base_pathless_file_name = 'example{0:06d}'.format(i)

        this_title_string = '{0:s} (AFTER; activation = {1:.2e})'.format(
            this_base_title_string, final_activations[i]
        )

        this_file_name = '{0:s}/{1:s}_after-optimization_sounding.jpg'.format(
            after_optimization_dir_name, this_base_pathless_file_name)

        sounding_plotting.plot_sounding(
            sounding_dict_for_metpy=list_of_optimized_metpy_dicts[i],
            title_string=this_title_string)

        print('Saving figure to: "{0:s}"...'.format(this_file_name))
        pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()

        if input_sounding_matrix is None:
            continue

        this_title_string = '{0:s} (BEFORE; activation = {1:.2e})'.format(
            this_base_title_string, initial_activations[i]
        )

        this_file_name = '{0:s}/{1:s}_before-optimization_sounding.jpg'.format(
            before_optimization_dir_name, this_base_pathless_file_name)

        sounding_plotting.plot_sounding(
            sounding_dict_for_metpy=list_of_input_metpy_dicts[i],
            title_string=this_title_string)

        print('Saving figure to: "{0:s}"...'.format(this_file_name))
        pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()


def _run(input_file_name, plot_significance, diff_colour_map_name,
         max_colour_percentile, top_output_dir_name):
    """Plots results of backwards optimization.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param plot_significance: Same.
    :param diff_colour_map_name: Same.
    :param max_colour_percentile: Same.
    :param top_output_dir_name: Same.
    """

    before_optimization_dir_name = '{0:s}/before_optimization'.format(
        top_output_dir_name)
    after_optimization_dir_name = '{0:s}/after_optimization'.format(
        top_output_dir_name)
    difference_dir_name = '{0:s}/difference'.format(top_output_dir_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=before_optimization_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=after_optimization_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=difference_dir_name)

    error_checking.assert_is_geq(max_colour_percentile, 0.)
    error_checking.assert_is_leq(max_colour_percentile, 100.)
    diff_colour_map_object = pyplot.cm.get_cmap(diff_colour_map_name)

    print('Reading data from: "{0:s}"...'.format(input_file_name))

    try:
        backwards_opt_dict = backwards_opt.read_standard_file(input_file_name)
        list_of_optimized_matrices = backwards_opt_dict[
            backwards_opt.OPTIMIZED_MATRICES_KEY]
        list_of_input_matrices = backwards_opt_dict[
            backwards_opt.INIT_FUNCTION_KEY]

        full_storm_id_strings = backwards_opt_dict[backwards_opt.FULL_IDS_KEY]
        storm_times_unix_sec = backwards_opt_dict[backwards_opt.STORM_TIMES_KEY]

        storm_time_strings = [
            time_conversion.unix_sec_to_string(
                t, plot_input_examples.TIME_FORMAT)
            for t in storm_times_unix_sec
        ]

    except ValueError:
        backwards_opt_dict = backwards_opt.read_pmm_file(input_file_name)
        list_of_input_matrices = backwards_opt_dict[
            backwards_opt.MEAN_INPUT_MATRICES_KEY]
        list_of_optimized_matrices = backwards_opt_dict[
            backwards_opt.MEAN_OPTIMIZED_MATRICES_KEY]

        for i in range(len(list_of_input_matrices)):
            list_of_input_matrices[i] = numpy.expand_dims(
                list_of_input_matrices[i], axis=0
            )
            list_of_optimized_matrices[i] = numpy.expand_dims(
                list_of_optimized_matrices[i], axis=0
            )

        full_storm_id_strings = [None]
        storm_times_unix_sec = [None]
        storm_time_strings = [None]

    pmm_flag = (
        full_storm_id_strings[0] is None and storm_time_strings[0] is None
    )

    model_file_name = backwards_opt_dict[backwards_opt.MODEL_FILE_KEY]
    model_metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0]
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_model_metadata(model_metafile_name)
    print(SEPARATOR_STRING)

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    include_soundings = (
        training_option_dict[trainval_io.SOUNDING_FIELDS_KEY] is not None
    )

    if include_soundings:
        _plot_bwo_for_soundings(
            input_sounding_matrix=list_of_input_matrices[-1],
            optimized_sounding_matrix=list_of_optimized_matrices[-1],
            training_option_dict=training_option_dict, pmm_flag=pmm_flag,
            backwards_opt_dict=backwards_opt_dict,
            top_output_dir_name=top_output_dir_name)

        print(SEPARATOR_STRING)

    # TODO(thunderhoser): Make sure to not plot soundings here.
    plot_input_examples.plot_examples(
        list_of_predictor_matrices=list_of_input_matrices,
        model_metadata_dict=model_metadata_dict,
        output_dir_name=before_optimization_dir_name,
        allow_whitespace=True, pmm_flag=pmm_flag,
        full_storm_id_strings=full_storm_id_strings,
        storm_times_unix_sec=storm_times_unix_sec)
    print(SEPARATOR_STRING)

    plot_input_examples.plot_examples(
        list_of_predictor_matrices=list_of_optimized_matrices,
        model_metadata_dict=model_metadata_dict,
        output_dir_name=after_optimization_dir_name,
        allow_whitespace=True, pmm_flag=pmm_flag,
        full_storm_id_strings=full_storm_id_strings,
        storm_times_unix_sec=storm_times_unix_sec)
    print(SEPARATOR_STRING)

    monte_carlo_dict = (
        backwards_opt_dict[backwards_opt.MONTE_CARLO_DICT_KEY]
        if plot_significance and
        backwards_opt.MONTE_CARLO_DICT_KEY in backwards_opt_dict
        else None
    )

    num_examples = list_of_optimized_matrices[0].shape[0]
    num_radar_matrices = (
        len(list_of_optimized_matrices) - int(include_soundings)
    )

    for i in range(num_examples):
        # TODO(thunderhoser): Make BWO file always store initial matrices, even
        # if they are created by a function.

        for j in range(num_radar_matrices):
            if monte_carlo_dict is None:
                this_significance_matrix = None
            else:
                this_significance_matrix = numpy.logical_or(
                    monte_carlo_dict[
                        monte_carlo.TRIAL_PMM_MATRICES_KEY][j][i, ...] <
                    monte_carlo_dict[monte_carlo.MIN_MATRICES_KEY][j][i, ...],
                    monte_carlo_dict[
                        monte_carlo.TRIAL_PMM_MATRICES_KEY][j][i, ...] >
                    monte_carlo_dict[monte_carlo.MAX_MATRICES_KEY][j][i, ...]
                )

            this_difference_matrix = (
                list_of_optimized_matrices[j][i, ...] -
                list_of_input_matrices[j][i, ...]
            )

            this_num_spatial_dim = len(list_of_input_matrices[j].shape) - 2

            if this_num_spatial_dim == 3:
                _plot_3d_radar_difference(
                    difference_matrix=this_difference_matrix,
                    colour_map_object=diff_colour_map_object,
                    max_colour_percentile=max_colour_percentile,
                    model_metadata_dict=model_metadata_dict,
                    backwards_opt_dict=backwards_opt_dict,
                    output_dir_name=difference_dir_name, example_index=i,
                    significance_matrix=this_significance_matrix)
            else:
                _plot_2d_radar_difference(
                    difference_matrix=this_difference_matrix,
                    colour_map_object=diff_colour_map_object,
                    max_colour_percentile=max_colour_percentile,
                    model_metadata_dict=model_metadata_dict,
                    backwards_opt_dict=backwards_opt_dict,
                    output_dir_name=difference_dir_name, example_index=i,
                    significance_matrix=this_significance_matrix)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        plot_significance=bool(getattr(
            INPUT_ARG_OBJECT, PLOT_SIGNIFICANCE_ARG_NAME
        )),
        diff_colour_map_name=getattr(INPUT_ARG_OBJECT, COLOUR_MAP_ARG_NAME),
        max_colour_percentile=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
