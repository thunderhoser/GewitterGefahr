"""Plots saliency maps."""

import os
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import monte_carlo
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import saliency_maps
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.plotting import sounding_plotting
from gewittergefahr.plotting import saliency_plotting
from gewittergefahr.plotting import significance_plotting
from gewittergefahr.plotting import imagemagick_utils
from gewittergefahr.scripts import plot_input_examples

# TODO(thunderhoser): Use threshold counts at some point.

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

HALF_NUM_CONTOURS = 10
FIGURE_RESOLUTION_DPI = 300
SOUNDING_IMAGE_SIZE_PX = int(1e7)

INPUT_FILE_ARG_NAME = 'input_file_name'
PLOT_SIGNIFICANCE_ARG_NAME = 'plot_significance'
COLOUR_MAP_ARG_NAME = 'colour_map_name'
MAX_PERCENTILE_ARG_NAME = 'max_colour_percentile'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'
PLOT_SOUNDINGS_ARG_NAME = plot_input_examples.PLOT_SOUNDINGS_ARG_NAME
ALLOW_WHITESPACE_ARG_NAME = plot_input_examples.ALLOW_WHITESPACE_ARG_NAME
PLOT_PANEL_NAMES_ARG_NAME = plot_input_examples.PLOT_PANEL_NAMES_ARG_NAME
ADD_TITLES_ARG_NAME = plot_input_examples.ADD_TITLES_ARG_NAME
LABEL_CBARS_ARG_NAME = plot_input_examples.LABEL_CBARS_ARG_NAME
CBAR_LENGTH_ARG_NAME = plot_input_examples.CBAR_LENGTH_ARG_NAME

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `saliency_maps.read_standard_file` or'
    ' `saliency_maps.read_pmm_file`.')

PLOT_SIGNIFICANCE_HELP_STRING = (
    'Boolean flag.  If 1, will plot stippling for significance.  This applies '
    'only if the saliency map contains PMM (proability-matched means) and '
    'results of a Monte Carlo comparison.')

COLOUR_MAP_HELP_STRING = (
    'Name of colour map.  Saliency for each predictor will be plotted with the '
    'same colour map.  For example, if name is "Greys", the colour map used '
    'will be `pyplot.cm.Greys`.  This argument supports only pyplot colour '
    'maps.')

MAX_PERCENTILE_HELP_STRING = (
    'Used to set max absolute value for each saliency map.  The max absolute '
    'value for example e and predictor p will be the [q]th percentile of all '
    'saliency values for example e, where q = `{0:s}`.'
).format(MAX_PERCENTILE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Figures will be saved here.')

PLOT_SOUNDINGS_HELP_STRING = plot_input_examples.PLOT_SOUNDINGS_HELP_STRING
ALLOW_WHITESPACE_HELP_STRING = plot_input_examples.ALLOW_WHITESPACE_HELP_STRING
PLOT_PANEL_NAMES_HELP_STRING = plot_input_examples.PLOT_PANEL_NAMES_HELP_STRING
ADD_TITLES_HELP_STRING = plot_input_examples.ADD_TITLES_HELP_STRING
LABEL_CBARS_HELP_STRING = plot_input_examples.LABEL_CBARS_HELP_STRING
CBAR_LENGTH_HELP_STRING = plot_input_examples.CBAR_LENGTH_HELP_STRING

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_SIGNIFICANCE_ARG_NAME, type=int, required=False, default=0,
    help=PLOT_SIGNIFICANCE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + COLOUR_MAP_ARG_NAME, type=str, required=False, default='binary',
    help=COLOUR_MAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False,
    default=99., help=MAX_PERCENTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_SOUNDINGS_ARG_NAME, type=int, required=False, default=1,
    help=PLOT_SOUNDINGS_HELP_STRING)

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


def _plot_3d_radar_saliency(
        saliency_matrix, colour_map_object, max_colour_value, figure_objects,
        axes_object_matrices, model_metadata_dict, output_dir_name,
        significance_matrix=None, full_storm_id_string=None,
        storm_time_unix_sec=None):
    """Plots saliency map for 3-D radar data.

    M = number of rows in spatial grid
    N = number of columns in spatial grid
    H = number of heights in spatial grid
    F = number of radar fields

    If this method is plotting a composite rather than single example (storm
    object), `full_storm_id_string` and `storm_time_unix_sec` can be None.

    :param saliency_matrix: M-by-N-by-H-by-F numpy array of saliency values.
    :param colour_map_object: See documentation at top of file.
    :param max_colour_value: Max value in colour scheme for saliency.
    :param figure_objects: See doc for
        `plot_input_examples._plot_3d_radar_scan`.
    :param axes_object_matrices: Same.
    :param model_metadata_dict: Dictionary returned by
        `cnn.read_model_metadata`.
    :param output_dir_name: Path to output directory.  Figure(s) will be saved
        here.
    :param significance_matrix: M-by-N-by-H-by-F numpy array of Boolean flags,
        indicating where differences with some other saliency map are
        significant.
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
        saliency_plotting.plot_many_2d_grids_with_contours(
            saliency_matrix_3d=numpy.flip(saliency_matrix[..., j], axis=0),
            axes_object_matrix=axes_object_matrices[j],
            colour_map_object=colour_map_object,
            max_absolute_contour_level=max_colour_value,
            contour_interval=max_colour_value / HALF_NUM_CONTOURS)

        if significance_matrix is not None:
            this_matrix = numpy.flip(significance_matrix[..., j], axis=0)

            significance_plotting.plot_many_2d_grids_without_coords(
                significance_matrix=this_matrix,
                axes_object_matrix=axes_object_matrices[j]
            )

        this_title_object = figure_objects[j]._suptitle

        if this_title_object is not None:
            this_title_string = (
                '{0:s} ... max absolute saliency = {1:.2e}'
            ).format(this_title_object.get_text(), max_colour_value)

            figure_objects[j].suptitle(
                this_title_string,
                fontsize=plot_input_examples.DEFAULT_TITLE_FONT_SIZE)

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


def _plot_2d_radar_saliency(
        saliency_matrix, colour_map_object, max_colour_value, figure_objects,
        axes_object_matrices, model_metadata_dict, output_dir_name,
        significance_matrix=None, full_storm_id_string=None,
        storm_time_unix_sec=None):
    """Plots saliency map for 2-D radar data.

    M = number of rows in spatial grid
    N = number of columns in spatial grid
    C = number of radar channels

    If this method is plotting a composite rather than single example (storm
    object), `full_storm_id_string` and `storm_time_unix_sec` can be None.

    :param saliency_matrix: M-by-N-by-C numpy array of saliency values.
    :param colour_map_object: See documentation at top of file.
    :param max_colour_value: Max value in colour scheme for saliency.
    :param figure_objects: See doc for
        `plot_input_examples._plot_2d_radar_scan`.
    :param axes_object_matrices: Same.
    :param model_metadata_dict: Dictionary returned by
        `cnn.read_model_metadata`.
    :param output_dir_name: Path to output directory.  Figure(s) will be saved
        here.
    :param significance_matrix: M-by-N-by-H numpy array of Boolean flags,
        indicating where differences with some other saliency map are
        significant.
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

    saliency_plotting.plot_many_2d_grids_with_contours(
        saliency_matrix_3d=numpy.flip(saliency_matrix, axis=0),
        axes_object_matrix=axes_object_matrices[figure_index],
        colour_map_object=colour_map_object,
        max_absolute_contour_level=max_colour_value,
        contour_interval=max_colour_value / HALF_NUM_CONTOURS,
        row_major=False)

    if significance_matrix is not None:
        significance_plotting.plot_many_2d_grids_without_coords(
            significance_matrix=numpy.flip(significance_matrix, axis=0),
            axes_object_matrix=axes_object_matrices[figure_index],
            row_major=False)

    this_title_object = figure_objects[figure_index]._suptitle

    if this_title_object is not None:
        this_title_string = '{0:s} ... max absolute saliency = {1:.2e}'.format(
            this_title_object.get_text(), max_colour_value
        )

        figure_objects[figure_index].suptitle(
            this_title_string,
            fontsize=plot_input_examples.DEFAULT_TITLE_FONT_SIZE)

    output_file_name = plot_input_examples.metadata_to_file_name(
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


def _plot_sounding_saliency(
        saliency_matrix, colour_map_object, max_colour_value, sounding_matrix,
        saliency_dict, model_metadata_dict, add_title, output_dir_name,
        pmm_flag, example_index=None):
    """Plots saliency for sounding.

    H = number of sounding heights
    F = number of sounding fields

    If plotting a composite rather than one example, `full_storm_id_string` and
    `storm_time_unix_sec` can be None.

    :param saliency_matrix: H-by-F numpy array of saliency values.
    :param colour_map_object: See documentation at top of file.
    :param max_colour_value: Max value in colour scheme for saliency.
    :param sounding_matrix: H-by-F numpy array of actual sounding values.
    :param saliency_dict: Dictionary returned from
        `saliency_maps.read_standard_file` or `saliency_maps.read_pmm_file`.
    :param model_metadata_dict: Dictionary returned by
        `cnn.read_model_metadata`.
    :param add_title: Boolean flag.
    :param output_dir_name: Path to output directory.  Figure will be saved
        here.
    :param pmm_flag: Boolean flag.  If True, will plot composite rather than one
        example.
    :param example_index: [used only if `pmm_flag == False`]
        Will plot the [i]th example, where i = `example_index`.
    """

    if pmm_flag:
        example_index = 0

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    sounding_field_names = training_option_dict[trainval_io.SOUNDING_FIELDS_KEY]
    sounding_heights_m_agl = training_option_dict[
        trainval_io.SOUNDING_HEIGHTS_KEY]

    sounding_matrix = numpy.expand_dims(sounding_matrix, axis=0)

    if saliency_maps.SOUNDING_PRESSURES_KEY in saliency_dict:
        pressure_matrix_pascals = numpy.expand_dims(
            saliency_dict[saliency_maps.SOUNDING_PRESSURES_KEY], axis=-1
        )

        pressure_matrix_pascals = pressure_matrix_pascals[[example_index], ...]
        sounding_matrix = numpy.concatenate(
            (sounding_matrix, pressure_matrix_pascals), axis=-1
        )

        sounding_dict_for_metpy = dl_utils.soundings_to_metpy_dictionaries(
            sounding_matrix=sounding_matrix,
            field_names=sounding_field_names + [soundings.PRESSURE_NAME]
        )[0]
    else:
        sounding_dict_for_metpy = dl_utils.soundings_to_metpy_dictionaries(
            sounding_matrix=sounding_matrix, field_names=sounding_field_names,
            height_levels_m_agl=sounding_heights_m_agl,
            storm_elevations_m_asl=numpy.array([0.])
        )[0]

    if pmm_flag:
        full_storm_id_string = None
        storm_time_unix_sec = None
    else:
        full_storm_id_string = saliency_dict[saliency_maps.FULL_IDS_KEY][
            example_index]
        storm_time_unix_sec = saliency_dict[saliency_maps.STORM_TIMES_KEY][
            example_index]

    if add_title:
        title_string = 'Max absolute saliency = {0:.2e}'.format(
            max_colour_value)
    else:
        title_string = None

    sounding_plotting.plot_sounding(
        sounding_dict_for_metpy=sounding_dict_for_metpy,
        title_string=title_string)

    left_panel_file_name = plot_input_examples.metadata_to_file_name(
        output_dir_name=output_dir_name, is_sounding=False, pmm_flag=pmm_flag,
        full_storm_id_string=full_storm_id_string,
        storm_time_unix_sec=storm_time_unix_sec,
        radar_field_name='sounding-actual')

    print('Saving figure to file: "{0:s}"...'.format(left_panel_file_name))
    pyplot.savefig(left_panel_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(
        input_file_name=left_panel_file_name,
        output_file_name=left_panel_file_name)

    saliency_plotting.plot_saliency_for_sounding(
        saliency_matrix=saliency_matrix,
        sounding_field_names=sounding_field_names,
        pressure_levels_mb=sounding_dict_for_metpy[
            soundings.PRESSURE_COLUMN_METPY],
        colour_map_object=colour_map_object,
        max_absolute_colour_value=max_colour_value)

    right_panel_file_name = plot_input_examples.metadata_to_file_name(
        output_dir_name=output_dir_name, is_sounding=False, pmm_flag=pmm_flag,
        full_storm_id_string=full_storm_id_string,
        storm_time_unix_sec=storm_time_unix_sec,
        radar_field_name='sounding-saliency')

    print('Saving figure to file: "{0:s}"...'.format(right_panel_file_name))
    pyplot.savefig(right_panel_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(
        input_file_name=right_panel_file_name,
        output_file_name=right_panel_file_name)

    concat_file_name = plot_input_examples.metadata_to_file_name(
        output_dir_name=output_dir_name, is_sounding=True, pmm_flag=pmm_flag,
        full_storm_id_string=full_storm_id_string,
        storm_time_unix_sec=storm_time_unix_sec)

    print('Concatenating figures to: "{0:s}"...\n'.format(concat_file_name))
    imagemagick_utils.concatenate_images(
        input_file_names=[left_panel_file_name, right_panel_file_name],
        output_file_name=concat_file_name, num_panel_rows=1,
        num_panel_columns=2)

    imagemagick_utils.resize_image(
        input_file_name=concat_file_name, output_file_name=concat_file_name,
        output_size_pixels=SOUNDING_IMAGE_SIZE_PX)

    os.remove(left_panel_file_name)
    os.remove(right_panel_file_name)


def _run(input_file_name, plot_significance, colour_map_name,
         max_colour_percentile, plot_soundings, allow_whitespace,
         plot_panel_names, add_titles, label_colour_bars, colour_bar_length,
         output_dir_name):
    """Plots saliency maps.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param plot_significance: Same.
    :param colour_map_name: Same.
    :param max_colour_percentile: Same.
    :param plot_soundings: Same.
    :param allow_whitespace: Same.
    :param plot_panel_names: Same.
    :param add_titles: Same.
    :param label_colour_bars: Same.
    :param colour_bar_length: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    error_checking.assert_is_geq(max_colour_percentile, 0.)
    error_checking.assert_is_leq(max_colour_percentile, 100.)
    colour_map_object = pyplot.cm.get_cmap(colour_map_name)

    print('Reading data from: "{0:s}"...'.format(input_file_name))

    try:
        saliency_dict = saliency_maps.read_standard_file(input_file_name)
        list_of_input_matrices = saliency_dict.pop(
            saliency_maps.INPUT_MATRICES_KEY)
        list_of_saliency_matrices = saliency_dict.pop(
            saliency_maps.SALIENCY_MATRICES_KEY)

        full_storm_id_strings = saliency_dict[saliency_maps.FULL_IDS_KEY]
        storm_times_unix_sec = saliency_dict[saliency_maps.STORM_TIMES_KEY]

    except ValueError:
        saliency_dict = saliency_maps.read_pmm_file(input_file_name)
        list_of_input_matrices = saliency_dict.pop(
            saliency_maps.MEAN_INPUT_MATRICES_KEY)
        list_of_saliency_matrices = saliency_dict.pop(
            saliency_maps.MEAN_SALIENCY_MATRICES_KEY)

        for i in range(len(list_of_input_matrices)):
            list_of_input_matrices[i] = numpy.expand_dims(
                list_of_input_matrices[i], axis=0
            )
            list_of_saliency_matrices[i] = numpy.expand_dims(
                list_of_saliency_matrices[i], axis=0
            )

        full_storm_id_strings = [None]
        storm_times_unix_sec = [None]

    pmm_flag = (
        full_storm_id_strings[0] is None and storm_times_unix_sec[0] is None
    )

    num_examples = list_of_input_matrices[0].shape[0]
    max_colour_value_by_example = numpy.full(num_examples, numpy.nan)

    for i in range(num_examples):
        these_saliency_values = numpy.concatenate(
            [numpy.ravel(s[i, ...]) for s in list_of_saliency_matrices]
        )
        max_colour_value_by_example[i] = numpy.percentile(
            numpy.absolute(these_saliency_values), max_colour_percentile
        )

    model_file_name = saliency_dict[saliency_maps.MODEL_FILE_KEY]
    model_metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0]
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_model_metadata(model_metafile_name)
    print(SEPARATOR_STRING)

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    has_soundings = (
        training_option_dict[trainval_io.SOUNDING_FIELDS_KEY] is not None
    )
    num_radar_matrices = len(list_of_input_matrices) - int(has_soundings)

    monte_carlo_dict = (
        saliency_dict[saliency_maps.MONTE_CARLO_DICT_KEY]
        if plot_significance and
        saliency_maps.MONTE_CARLO_DICT_KEY in saliency_dict
        else None
    )

    for i in range(num_examples):
        if has_soundings and plot_soundings:
            _plot_sounding_saliency(
                saliency_matrix=list_of_saliency_matrices[-1][i, ...],
                colour_map_object=colour_map_object,
                max_colour_value=max_colour_value_by_example[i],
                sounding_matrix=list_of_input_matrices[-1][i, ...],
                saliency_dict=saliency_dict,
                model_metadata_dict=model_metadata_dict,
                add_title=add_titles, output_dir_name=output_dir_name,
                pmm_flag=pmm_flag, example_index=i)

        this_handle_dict = plot_input_examples.plot_one_example(
            list_of_predictor_matrices=list_of_input_matrices,
            model_metadata_dict=model_metadata_dict, pmm_flag=pmm_flag,
            example_index=i, plot_sounding=False,
            allow_whitespace=allow_whitespace,
            plot_panel_names=plot_panel_names, add_titles=add_titles,
            label_colour_bars=label_colour_bars,
            colour_bar_length=colour_bar_length)

        these_figure_objects = this_handle_dict[
            plot_input_examples.RADAR_FIGURES_KEY]
        these_axes_object_matrices = this_handle_dict[
            plot_input_examples.RADAR_AXES_KEY]

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

            this_num_spatial_dim = len(list_of_input_matrices[j].shape) - 2

            if this_num_spatial_dim == 3:
                _plot_3d_radar_saliency(
                    saliency_matrix=list_of_saliency_matrices[j][i, ...],
                    colour_map_object=colour_map_object,
                    max_colour_value=max_colour_value_by_example[i],
                    figure_objects=these_figure_objects,
                    axes_object_matrices=these_axes_object_matrices,
                    model_metadata_dict=model_metadata_dict,
                    output_dir_name=output_dir_name,
                    significance_matrix=this_significance_matrix,
                    full_storm_id_string=full_storm_id_strings[i],
                    storm_time_unix_sec=storm_times_unix_sec[i]
                )
            else:
                _plot_2d_radar_saliency(
                    saliency_matrix=list_of_saliency_matrices[j][i, ...],
                    colour_map_object=colour_map_object,
                    max_colour_value=max_colour_value_by_example[i],
                    figure_objects=these_figure_objects,
                    axes_object_matrices=these_axes_object_matrices,
                    model_metadata_dict=model_metadata_dict,
                    output_dir_name=output_dir_name,
                    significance_matrix=this_significance_matrix,
                    full_storm_id_string=full_storm_id_strings[i],
                    storm_time_unix_sec=storm_times_unix_sec[i]
                )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        plot_significance=bool(getattr(
            INPUT_ARG_OBJECT, PLOT_SIGNIFICANCE_ARG_NAME
        )),
        colour_map_name=getattr(INPUT_ARG_OBJECT, COLOUR_MAP_ARG_NAME),
        max_colour_percentile=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME),
        plot_soundings=bool(getattr(INPUT_ARG_OBJECT, PLOT_SOUNDINGS_ARG_NAME)),
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
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
