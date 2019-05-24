"""Plots results of novelty detection."""

import os.path
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.deep_learning import novelty_detection
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import radar_plotting

# TODO(thunderhoser): This file contains a lot of duplicated code for
# determining output paths and titles.

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TITLE_FONT_SIZE = 20
FONT_SIZE_WITH_COLOUR_BARS = 16
FONT_SIZE_SANS_COLOUR_BARS = 20
FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_file_name'
COLOUR_MAP_ARG_NAME = 'diff_colour_map_name'
MAX_PERCENTILE_ARG_NAME = 'max_colour_percentile_for_diff'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by '
    '`novelty_detection.read_standard_file` or '
    '`novelty_detection.read_pmm_file`.')

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
    '--' + COLOUR_MAP_ARG_NAME, type=str, required=False, default='bwr',
    help=COLOUR_MAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False,
    default=99., help=MAX_PERCENTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _plot_3d_radar(
        training_option_dict, output_dir_name, pmm_flag,
        diff_colour_map_object=None, max_colour_percentile_for_diff=None,
        full_id_strings=None, storm_time_strings=None, novel_radar_matrix=None,
        novel_radar_matrix_upconv=None, novel_radar_matrix_upconv_svd=None):
    """Plots results of novelty detection for 3-D radar fields.

    E = number of examples (storm objects)
    M = number of rows in spatial grid
    N = number of columns in spatial grid
    H = number of heights in spatial grid
    F = number of fields

    If `novel_radar_matrix` is the only matrix given, this method will plot the
    original (not reconstructed) radar fields.

    If `novel_radar_matrix_upconv` is the only matrix given, will plot
    upconvnet-reconstructed fields.

    If `novel_radar_matrix_upconv_svd` is the only matrix given, will plot
    upconvnet-and-SVD-reconstructed fields.

    If both `novel_radar_matrix_upconv` and `novel_radar_matrix_upconv_svd` are
    given, will plot novelty fields (upconvnet/SVD reconstruction minus
    upconvnet reconstruction).

    :param training_option_dict: See doc for `cnn.read_model_metadata`.
    :param output_dir_name: Name of output directory (figures will be saved
        here).
    :param pmm_flag: Boolean flag.  If True, the input matrices contain
        probability-matched means.
    :param diff_colour_map_object:
        [used only if both `novel_radar_matrix_upconv` and
        `novel_radar_matrix_upconv_svd` are given]

        See documentation at top of file.

    :param max_colour_percentile_for_diff: Same.
    :param full_id_strings: [optional and used only if `pmm_flag = False`]
        length-E list of full storm IDs.
    :param storm_time_strings: [optional and used only if `pmm_flag = False`]
        length-E list of storm times.
    :param novel_radar_matrix: E-by-M-by-N-by-H-by-F numpy array of original
        (not reconstructed) radar fields.
    :param novel_radar_matrix_upconv: E-by-M-by-N-by-H-by-F numpy array of
        upconvnet-reconstructed radar fields.
    :param novel_radar_matrix_upconv_svd: E-by-M-by-N-by-H-by-F numpy array of
        upconvnet-and-SVD-reconstructed radar fields.
    """

    if pmm_flag:
        have_storm_ids = False
    else:
        have_storm_ids = not (
            full_id_strings is None or storm_time_strings is None
        )

    plot_difference = False

    if novel_radar_matrix is not None:
        plot_type_abbrev = 'actual'
        plot_type_verbose = 'actual'
        radar_matrix_to_plot = novel_radar_matrix
    else:
        if (novel_radar_matrix_upconv is not None and
                novel_radar_matrix_upconv_svd is not None):

            plot_difference = True
            plot_type_abbrev = 'novelty'
            plot_type_verbose = 'novelty'
            radar_matrix_to_plot = (
                novel_radar_matrix_upconv - novel_radar_matrix_upconv_svd
            )

        else:
            if novel_radar_matrix_upconv is not None:
                plot_type_abbrev = 'upconv'
                plot_type_verbose = 'upconvnet reconstruction'
                radar_matrix_to_plot = novel_radar_matrix_upconv
            else:
                plot_type_abbrev = 'upconv-svd'
                plot_type_verbose = 'upconvnet/SVD reconstruction'
                radar_matrix_to_plot = novel_radar_matrix_upconv_svd

    radar_field_names = training_option_dict[trainval_io.RADAR_FIELDS_KEY]
    radar_heights_m_agl = training_option_dict[trainval_io.RADAR_HEIGHTS_KEY]

    num_storms = novel_radar_matrix.shape[0]
    num_heights = novel_radar_matrix.shape[-2]
    num_panel_rows = int(numpy.floor(
        numpy.sqrt(num_heights)
    ))

    for i in range(num_storms):
        if pmm_flag:
            this_title_string = 'Probability-matched mean'
            this_base_file_name = 'pmm'
        else:
            if have_storm_ids:
                this_title_string = 'Storm "{0:s}" at {1:s}'.format(
                    full_id_strings[i], storm_time_strings[i]
                )

                this_base_file_name = '{0:s}_{1:s}'.format(
                    full_id_strings[i].replace('_', '-'), storm_time_strings[i]
                )
            else:
                this_title_string = 'Example {0:d}'.format(i + 1)
                this_base_file_name = 'example{0:06d}'.format(i)

        this_title_string += ' ({0:s})'.format(plot_type_verbose)

        for j in range(len(radar_field_names)):
            this_file_name = '{0:s}/{1:s}_{2:s}_{3:s}.jpg'.format(
                output_dir_name, this_base_file_name, plot_type_abbrev,
                radar_field_names[j].replace('_', '-')
            )

            if plot_difference:
                this_colour_map_object = diff_colour_map_object

                this_max_value = numpy.percentile(
                    numpy.absolute(radar_matrix_to_plot[i, ..., j]),
                    max_colour_percentile_for_diff)

                this_colour_norm_object = matplotlib.colors.Normalize(
                    vmin=-1 * this_max_value, vmax=this_max_value, clip=False)
            else:
                this_colour_map_object, this_colour_norm_object = (
                    radar_plotting.get_default_colour_scheme(
                        radar_field_names[j])
                )

            _, these_axes_objects = radar_plotting.plot_3d_grid_without_coords(
                field_matrix=numpy.flip(
                    radar_matrix_to_plot[i, ..., j], axis=0),
                field_name=radar_field_names[j],
                grid_point_heights_metres=radar_heights_m_agl,
                ground_relative=True, num_panel_rows=num_panel_rows,
                font_size=FONT_SIZE_SANS_COLOUR_BARS,
                colour_map_object=this_colour_map_object,
                colour_norm_object=this_colour_norm_object)

            plotting_utils.add_colour_bar(
                axes_object_or_list=these_axes_objects,
                values_to_colour=radar_matrix_to_plot[i, ..., j],
                colour_map=this_colour_map_object,
                colour_norm_object=this_colour_norm_object,
                orientation='horizontal', extend_min=True, extend_max=True)

            pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)
            print('Saving figure to: "{0:s}"...'.format(this_file_name))
            pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
            pyplot.close()


def _plot_2d_radar(
        model_metadata_dict, output_dir_name, pmm_flag,
        diff_colour_map_object=None, max_colour_percentile_for_diff=None,
        full_id_strings=None, storm_time_strings=None, novel_radar_matrix=None,
        novel_radar_matrix_upconv=None, novel_radar_matrix_upconv_svd=None):
    """Plots results of novelty detection for 2-D radar fields.

    E = number of examples (storm objects)
    M = number of rows in spatial grid
    N = number of columns in spatial grid
    C = number of channels (field/height pairs)

    This method handles the 3 input matrices in the same way as
    `_plot_3d_radar`.

    :param model_metadata_dict: Dictionary returned by
        `cnn.read_model_metadata`.
    :param output_dir_name: Same.
    :param pmm_flag: Same.
    :param diff_colour_map_object: Same.
    :param max_colour_percentile_for_diff: Same.
    :param full_id_strings: Same.
    :param storm_time_strings: Same.
    :param novel_radar_matrix: E-by-M-by-N-by-C numpy array of original
        (not reconstructed) radar fields.
    :param novel_radar_matrix_upconv: E-by-M-by-N-by-C numpy array of
        upconvnet-reconstructed radar fields.
    :param novel_radar_matrix_upconv_svd: E-by-M-by-N-by-C numpy array of
        upconvnet-and-SVD-reconstructed radar fields.
    """

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    list_of_layer_operation_dicts = model_metadata_dict[
        cnn.LAYER_OPERATIONS_KEY]

    if pmm_flag:
        have_storm_ids = False
    else:
        have_storm_ids = not (
            full_id_strings is None or storm_time_strings is None
        )

    plot_difference = False

    if novel_radar_matrix is not None:
        plot_type_abbrev = 'actual'
        plot_type_verbose = 'actual'
        radar_matrix_to_plot = novel_radar_matrix
    else:
        if (novel_radar_matrix_upconv is not None and
                novel_radar_matrix_upconv_svd is not None):

            plot_difference = True
            plot_type_abbrev = 'novelty'
            plot_type_verbose = 'novelty'
            radar_matrix_to_plot = (
                novel_radar_matrix_upconv - novel_radar_matrix_upconv_svd
            )

        else:
            if novel_radar_matrix_upconv is not None:
                plot_type_abbrev = 'upconv'
                plot_type_verbose = 'upconvnet reconstruction'
                radar_matrix_to_plot = novel_radar_matrix_upconv
            else:
                plot_type_abbrev = 'upconv-svd'
                plot_type_verbose = 'upconvnet/SVD reconstruction'
                radar_matrix_to_plot = novel_radar_matrix_upconv_svd

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

        plot_colour_bar_by_panel = numpy.full(
            len(panel_names), True, dtype=bool)

    else:
        field_name_by_panel, panel_names = (
            radar_plotting.layer_ops_to_field_and_panel_names(
                list_of_layer_operation_dicts)
        )

        plot_colour_bar_by_panel = numpy.full(
            len(panel_names), False, dtype=bool)
        plot_colour_bar_by_panel[2::3] = True

    num_panels = len(panel_names)
    num_storms = radar_matrix_to_plot.shape[0]
    num_channels = radar_matrix_to_plot.shape[-1]
    num_panel_rows = int(numpy.floor(
        numpy.sqrt(num_channels)
    ))

    for i in range(num_storms):
        if pmm_flag:
            this_title_string = 'Probability-matched mean'
            this_file_name = 'pmm'
        else:
            if have_storm_ids:
                this_title_string = 'Storm "{0:s}" at {1:s}'.format(
                    full_id_strings[i], storm_time_strings[i]
                )

                this_file_name = '{0:s}_{1:s}'.format(
                    full_id_strings[i].replace('_', '-'), storm_time_strings[i]
                )
            else:
                this_title_string = 'Example {0:d}'.format(i + 1)
                this_file_name = 'example{0:06d}'.format(i)

        this_title_string += ' ({0:s})'.format(plot_type_verbose)
        this_file_name = '{0:s}/{1:s}_{2:s}_radar.jpg'.format(
            output_dir_name, this_file_name, plot_type_abbrev)

        if plot_difference:
            this_cmap_object_by_panel = [diff_colour_map_object] * num_panels
            this_cnorm_object_by_panel = [None] * num_panels

            if list_of_layer_operation_dicts is None:
                for j in range(num_panels):
                    this_max_value = numpy.percentile(
                        numpy.absolute(radar_matrix_to_plot[i, ..., j]),
                        max_colour_percentile_for_diff)

                    this_cnorm_object_by_panel[j] = matplotlib.colors.Normalize(
                        vmin=-1 * this_max_value, vmax=this_max_value,
                        clip=False)
            else:
                unique_field_names = numpy.unique(
                    numpy.array(field_name_by_panel)
                )

                for this_field_name in unique_field_names:
                    these_panel_indices = numpy.where(
                        numpy.array(field_name_by_panel) == this_field_name
                    )[0]

                    this_diff_matrix = radar_matrix_to_plot[
                        i, ..., these_panel_indices]

                    this_max_value = numpy.percentile(
                        numpy.absolute(this_diff_matrix),
                        max_colour_percentile_for_diff)

                    for this_index in these_panel_indices:
                        this_cnorm_object_by_panel[this_index] = (
                            matplotlib.colors.Normalize(
                                vmin=-1 * this_max_value, vmax=this_max_value,
                                clip=False)
                        )
        else:
            this_cmap_object_by_panel = None
            this_cnorm_object_by_panel = None

        radar_plotting.plot_many_2d_grids_without_coords(
            field_matrix=numpy.flip(radar_matrix_to_plot[i, ...], axis=0),
            field_name_by_panel=field_name_by_panel,
            num_panel_rows=num_panel_rows, panel_names=panel_names,
            colour_map_object_by_panel=this_cmap_object_by_panel,
            colour_norm_object_by_panel=this_cnorm_object_by_panel,
            plot_colour_bar_by_panel=plot_colour_bar_by_panel,
            font_size=FONT_SIZE_WITH_COLOUR_BARS, row_major=False)

        pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)
        print('Saving figure to: "{0:s}"...'.format(this_file_name))
        pyplot.savefig(this_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()


def _run(input_file_name, diff_colour_map_name, max_colour_percentile_for_diff,
         top_output_dir_name):
    """Plots results of novelty detection.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param diff_colour_map_name: Same.
    :param max_colour_percentile_for_diff: Same.
    :param top_output_dir_name: Same.
    """

    actual_dir_name = '{0:s}/actual'.format(top_output_dir_name)
    upconv_dir_name = '{0:s}/upconvnet_reconstruction'.format(
        top_output_dir_name)
    upconv_svd_dir_name = '{0:s}/upconvnet_svd_reconstruction'.format(
        top_output_dir_name)
    difference_dir_name = '{0:s}/difference'.format(top_output_dir_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=actual_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=upconv_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=upconv_svd_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=difference_dir_name)

    pmm_flag = False

    error_checking.assert_is_geq(max_colour_percentile_for_diff, 0.)
    error_checking.assert_is_leq(max_colour_percentile_for_diff, 100.)
    diff_colour_map_object = pyplot.cm.get_cmap(diff_colour_map_name)

    print('Reading data from: "{0:s}"...'.format(input_file_name))

    try:
        novelty_dict = novelty_detection.read_standard_file(input_file_name)

        novel_indices = novelty_dict[novelty_detection.NOVEL_INDICES_KEY]
        novel_radar_matrix = novelty_dict.pop(
            novelty_detection.TRIAL_INPUTS_KEY
        )[0][novel_indices, ...]

        novel_radar_matrix_upconv = novelty_dict.pop(
            novelty_detection.NOVEL_IMAGES_UPCONV_KEY)
        novel_radar_matrix_upconv_svd = novelty_dict.pop(
            novelty_detection.NOVEL_IMAGES_UPCONV_SVD_KEY)

        novelty_metadata_dict = novelty_dict
        full_id_strings = novelty_metadata_dict[novelty_detection.TRIAL_IDS_KEY]

        storm_time_strings = [
            time_conversion.unix_sec_to_string(t, TIME_FORMAT) for t in
            novelty_metadata_dict[novelty_detection.TRIAL_STORM_TIMES_KEY]
        ]

    except ValueError:
        pmm_flag = True
        novelty_dict = novelty_detection.read_pmm_file(input_file_name)

        novel_radar_matrix = numpy.expand_dims(
            novelty_dict.pop(novelty_detection.MEAN_NOVEL_IMAGE_KEY), axis=0)
        novel_radar_matrix_upconv = numpy.expand_dims(
            novelty_dict.pop(novelty_detection.MEAN_NOVEL_IMAGE_UPCONV_KEY),
            axis=0)
        novel_radar_matrix_upconv_svd = numpy.expand_dims(
            novelty_dict.pop(novelty_detection.MEAN_NOVEL_IMAGE_UPCONV_SVD_KEY),
            axis=0)

        orig_novelty_file_name = novelty_dict[
            novelty_detection.STANDARD_FILE_NAME_KEY]

        print('Reading metadata from: "{0:s}"...'.format(
            orig_novelty_file_name))

        novelty_metadata_dict = novelty_detection.read_standard_file(
            orig_novelty_file_name)

        novelty_metadata_dict.pop(novelty_detection.TRIAL_INPUTS_KEY)
        novelty_metadata_dict.pop(novelty_detection.NOVEL_IMAGES_UPCONV_KEY)
        novelty_metadata_dict.pop(novelty_detection.NOVEL_IMAGES_UPCONV_SVD_KEY)

        full_id_strings = None
        storm_time_strings = None

    novelty_metadata_dict.pop(novelty_detection.BASELINE_INPUTS_KEY)

    cnn_file_name = novelty_metadata_dict[novelty_detection.CNN_FILE_KEY]
    cnn_metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(cnn_file_name)[0]
    )

    print('Reading CNN metadata from: "{0:s}"...'.format(cnn_metafile_name))
    model_metadata_dict = cnn.read_model_metadata(cnn_metafile_name)
    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]

    print(SEPARATOR_STRING)
    num_radar_dimensions = len(novel_radar_matrix.shape) - 2

    if num_radar_dimensions == 3:
        _plot_3d_radar(
            training_option_dict=training_option_dict,
            output_dir_name=actual_dir_name, pmm_flag=pmm_flag,
            full_id_strings=full_id_strings,
            storm_time_strings=storm_time_strings,
            novel_radar_matrix=novel_radar_matrix)
        print(SEPARATOR_STRING)

        _plot_3d_radar(
            training_option_dict=training_option_dict,
            output_dir_name=upconv_dir_name, pmm_flag=pmm_flag,
            full_id_strings=full_id_strings,
            storm_time_strings=storm_time_strings,
            novel_radar_matrix_upconv=novel_radar_matrix_upconv)
        print(SEPARATOR_STRING)

        _plot_3d_radar(
            training_option_dict=training_option_dict,
            output_dir_name=upconv_svd_dir_name, pmm_flag=pmm_flag,
            full_id_strings=full_id_strings,
            storm_time_strings=storm_time_strings,
            novel_radar_matrix_upconv_svd=novel_radar_matrix_upconv_svd)
        print(SEPARATOR_STRING)

        _plot_3d_radar(
            training_option_dict=training_option_dict,
            diff_colour_map_object=diff_colour_map_object,
            max_colour_percentile_for_diff=max_colour_percentile_for_diff,
            output_dir_name=difference_dir_name, pmm_flag=pmm_flag,
            full_id_strings=full_id_strings,
            storm_time_strings=storm_time_strings,
            novel_radar_matrix_upconv=novel_radar_matrix_upconv,
            novel_radar_matrix_upconv_svd=novel_radar_matrix_upconv_svd)

        return

    _plot_2d_radar(
        model_metadata_dict=model_metadata_dict,
        output_dir_name=actual_dir_name, pmm_flag=pmm_flag,
        full_id_strings=full_id_strings, storm_time_strings=storm_time_strings,
        novel_radar_matrix=novel_radar_matrix)
    print(SEPARATOR_STRING)

    _plot_2d_radar(
        model_metadata_dict=model_metadata_dict,
        output_dir_name=upconv_dir_name, pmm_flag=pmm_flag,
        full_id_strings=full_id_strings, storm_time_strings=storm_time_strings,
        novel_radar_matrix_upconv=novel_radar_matrix_upconv)
    print(SEPARATOR_STRING)

    _plot_2d_radar(
        model_metadata_dict=model_metadata_dict,
        output_dir_name=upconv_svd_dir_name, pmm_flag=pmm_flag,
        full_id_strings=full_id_strings, storm_time_strings=storm_time_strings,
        novel_radar_matrix_upconv_svd=novel_radar_matrix_upconv_svd)
    print(SEPARATOR_STRING)

    _plot_2d_radar(
        model_metadata_dict=model_metadata_dict,
        diff_colour_map_object=diff_colour_map_object,
        max_colour_percentile_for_diff=max_colour_percentile_for_diff,
        output_dir_name=difference_dir_name, pmm_flag=pmm_flag,
        full_id_strings=full_id_strings, storm_time_strings=storm_time_strings,
        novel_radar_matrix_upconv=novel_radar_matrix_upconv,
        novel_radar_matrix_upconv_svd=novel_radar_matrix_upconv_svd)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        diff_colour_map_name=getattr(INPUT_ARG_OBJECT, COLOUR_MAP_ARG_NAME),
        max_colour_percentile_for_diff=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
