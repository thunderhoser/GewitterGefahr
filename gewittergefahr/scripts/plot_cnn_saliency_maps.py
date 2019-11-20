"""Plots saliency maps."""

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
from gewittergefahr.deep_learning import saliency_maps
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import saliency_plotting
from gewittergefahr.plotting import significance_plotting
from gewittergefahr.plotting import imagemagick_utils
from gewittergefahr.scripts import plot_input_examples as plot_examples

PASCALS_TO_MB = 0.01
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MAX_COLOUR_PERCENTILE = 99.
COLOUR_BAR_FONT_SIZE = plot_examples.DEFAULT_CBAR_FONT_SIZE
FIGURE_RESOLUTION_DPI = 300
SOUNDING_IMAGE_SIZE_PX = int(1e7)

INPUT_FILE_ARG_NAME = 'input_file_name'
COLOUR_MAP_ARG_NAME = 'colour_map_name'
MAX_COLOUR_VALUE_ARG_NAME = 'max_colour_value'
HALF_NUM_CONTOURS_ARG_NAME = 'half_num_contours'
SMOOTHING_RADIUS_ARG_NAME = 'smoothing_radius_grid_cells'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'
PLOT_SOUNDINGS_ARG_NAME = plot_examples.PLOT_SOUNDINGS_ARG_NAME
ALLOW_WHITESPACE_ARG_NAME = plot_examples.ALLOW_WHITESPACE_ARG_NAME
PLOT_PANEL_NAMES_ARG_NAME = plot_examples.PLOT_PANEL_NAMES_ARG_NAME
ADD_TITLES_ARG_NAME = plot_examples.ADD_TITLES_ARG_NAME
LABEL_CBARS_ARG_NAME = plot_examples.LABEL_CBARS_ARG_NAME
CBAR_LENGTH_ARG_NAME = plot_examples.CBAR_LENGTH_ARG_NAME

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `saliency_maps.read_file`.')

COLOUR_MAP_HELP_STRING = (
    'Name of colour map.  Saliency for each predictor will be plotted with the '
    'same colour map.  For example, if name is "Greys", the colour map used '
    'will be `pyplot.cm.Greys`.  This argument supports only pyplot colour '
    'maps.')

MAX_COLOUR_VALUE_HELP_STRING = (
    'Max saliency value in colour scheme.  Keep in mind that the colour scheme '
    'encodes *absolute* value, with positive values in solid contours and '
    'negative values in dashed contours.  To use a data-dependent default '
    'value, make this argument negative.')

HALF_NUM_CONTOURS_HELP_STRING = (
    'Number of contours on each side of zero (positive and negative).')

SMOOTHING_RADIUS_HELP_STRING = (
    'e-folding radius for Gaussian smoother (num grid cells).  If you do not '
    'want to smooth saliency maps, make this non-positive.')

OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Figures will be saved here.')

PLOT_SOUNDINGS_HELP_STRING = plot_examples.PLOT_SOUNDINGS_HELP_STRING
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
    '--' + COLOUR_MAP_ARG_NAME, type=str, required=False, default='binary',
    help=COLOUR_MAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_COLOUR_VALUE_ARG_NAME, type=float, required=False,
    default=1.25, help=MAX_COLOUR_VALUE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + HALF_NUM_CONTOURS_ARG_NAME, type=int, required=False,
    default=10, help=HALF_NUM_CONTOURS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SMOOTHING_RADIUS_ARG_NAME, type=float, required=False,
    default=2., help=SMOOTHING_RADIUS_HELP_STRING)

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
        saliency_matrix, colour_map_object, max_colour_value, half_num_contours,
        label_colour_bars, colour_bar_length, figure_objects,
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
    :param max_colour_value: Same.
    :param half_num_contours: Same.
    :param label_colour_bars: Same.
    :param colour_bar_length: Same.
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

    if max_colour_value is None:
        max_colour_value = numpy.percentile(
            numpy.absolute(saliency_matrix), MAX_COLOUR_PERCENTILE
        )

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
            contour_interval=max_colour_value / half_num_contours)

        if significance_matrix is not None:
            this_matrix = numpy.flip(significance_matrix[..., j], axis=0)

            significance_plotting.plot_many_2d_grids_without_coords(
                significance_matrix=this_matrix,
                axes_object_matrix=axes_object_matrices[j]
            )

        this_colour_bar_object = plotting_utils.plot_linear_colour_bar(
            axes_object_or_matrix=axes_object_matrices[j],
            data_matrix=saliency_matrix[..., j],
            colour_map_object=colour_map_object, min_value=0.,
            max_value=max_colour_value, orientation_string='horizontal',
            fraction_of_axis_length=colour_bar_length,
            extend_min=False, extend_max=True, font_size=COLOUR_BAR_FONT_SIZE)

        if label_colour_bars:
            this_colour_bar_object.set_label(
                'Absolute saliency', fontsize=COLOUR_BAR_FONT_SIZE)

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


def _plot_2d_radar_saliency(
        saliency_matrix, colour_map_object, max_colour_value, half_num_contours,
        label_colour_bars, colour_bar_length, figure_objects,
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
    :param max_colour_value: Same.
    :param half_num_contours: Same.
    :param label_colour_bars: Same.
    :param colour_bar_length: Same.
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

    if max_colour_value is None:
        max_colour_value = numpy.percentile(
            numpy.absolute(saliency_matrix), MAX_COLOUR_PERCENTILE
        )

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
        contour_interval=max_colour_value / half_num_contours,
        row_major=False)

    if significance_matrix is not None:
        significance_plotting.plot_many_2d_grids_without_coords(
            significance_matrix=numpy.flip(significance_matrix, axis=0),
            axes_object_matrix=axes_object_matrices[figure_index],
            row_major=False)

    colour_bar_object = plotting_utils.plot_linear_colour_bar(
        axes_object_or_matrix=axes_object_matrices[figure_index],
        data_matrix=saliency_matrix,
        colour_map_object=colour_map_object, min_value=0.,
        max_value=max_colour_value, orientation_string='horizontal',
        fraction_of_axis_length=colour_bar_length / (1 + int(conv_2d3d)),
        extend_min=False, extend_max=True, font_size=COLOUR_BAR_FONT_SIZE)

    if label_colour_bars:
        colour_bar_object.set_label(
            'Absolute saliency', fontsize=COLOUR_BAR_FONT_SIZE)

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


def _plot_sounding_saliency(
        saliency_matrix, colour_map_object, max_colour_value,
        sounding_figure_object, sounding_axes_object,
        sounding_pressures_pascals, saliency_dict, model_metadata_dict,
        add_title, output_dir_name, pmm_flag, example_index=None):
    """Plots saliency for sounding.

    H = number of sounding heights
    F = number of sounding fields

    :param saliency_matrix: H-by-F numpy array of saliency values.
    :param colour_map_object: See documentation at top of file.
    :param max_colour_value: Same.
    :param sounding_figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`) for sounding itself.
    :param sounding_axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`) for sounding itself.
    :param sounding_pressures_pascals: length-H numpy array of sounding
        pressures.
    :param saliency_dict: Dictionary returned by `saliency_maps.read_file`.
    :param model_metadata_dict: Dictionary returned by
        `cnn.read_model_metadata`.
    :param add_title: Boolean flag.
    :param output_dir_name: Name of output directory.  Figures will be saved
        here.
    :param pmm_flag: Boolean flag.  If True, plotting PMM composite rather than
        one example.
    :param example_index: [used only if `pmm_flag == False`]
        Plotting the [i]th example, where i = `example_index`.
    """

    if max_colour_value is None:
        max_colour_value = numpy.percentile(
            numpy.absolute(saliency_matrix), MAX_COLOUR_PERCENTILE
        )

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    sounding_field_names = training_option_dict[trainval_io.SOUNDING_FIELDS_KEY]

    if pmm_flag:
        full_storm_id_string = None
        storm_time_unix_sec = None
    else:
        full_storm_id_string = saliency_dict[saliency_maps.FULL_STORM_IDS_KEY][
            example_index]
        storm_time_unix_sec = saliency_dict[saliency_maps.STORM_TIMES_KEY][
            example_index]

    if add_title:
        title_string = 'Max absolute saliency = {0:.2e}'.format(
            max_colour_value)
        sounding_axes_object.set_title(title_string)

    left_panel_file_name = plot_examples.metadata_to_file_name(
        output_dir_name=output_dir_name, is_sounding=False, pmm_flag=pmm_flag,
        full_storm_id_string=full_storm_id_string,
        storm_time_unix_sec=storm_time_unix_sec,
        radar_field_name='sounding-actual')

    print('Saving figure to file: "{0:s}"...'.format(left_panel_file_name))
    sounding_figure_object.savefig(
        left_panel_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight')
    pyplot.close(sounding_figure_object)

    saliency_plotting.plot_saliency_for_sounding(
        saliency_matrix=saliency_matrix,
        sounding_field_names=sounding_field_names,
        pressure_levels_mb=PASCALS_TO_MB * sounding_pressures_pascals,
        colour_map_object=colour_map_object,
        max_absolute_colour_value=max_colour_value)

    right_panel_file_name = plot_examples.metadata_to_file_name(
        output_dir_name=output_dir_name, is_sounding=False, pmm_flag=pmm_flag,
        full_storm_id_string=full_storm_id_string,
        storm_time_unix_sec=storm_time_unix_sec,
        radar_field_name='sounding-saliency')

    print('Saving figure to file: "{0:s}"...'.format(right_panel_file_name))
    pyplot.savefig(right_panel_file_name, dpi=FIGURE_RESOLUTION_DPI,
                   pad_inches=0, bbox_inches='tight')
    pyplot.close()

    concat_file_name = plot_examples.metadata_to_file_name(
        output_dir_name=output_dir_name, is_sounding=True, pmm_flag=pmm_flag,
        full_storm_id_string=full_storm_id_string,
        storm_time_unix_sec=storm_time_unix_sec)

    print('Concatenating figures to: "{0:s}"...\n'.format(concat_file_name))
    imagemagick_utils.concatenate_images(
        input_file_names=[left_panel_file_name, right_panel_file_name],
        output_file_name=concat_file_name,
        num_panel_rows=1, num_panel_columns=2
    )

    imagemagick_utils.resize_image(
        input_file_name=concat_file_name, output_file_name=concat_file_name,
        output_size_pixels=SOUNDING_IMAGE_SIZE_PX)

    os.remove(left_panel_file_name)
    os.remove(right_panel_file_name)


def _smooth_maps(saliency_matrices, smoothing_radius_grid_cells):
    """Smooths saliency maps via Gaussian filter.

    T = number of input tensors to the model

    :param saliency_matrices: length-T list of numpy arrays.
    :param smoothing_radius_grid_cells: e-folding radius (number of grid cells).
    :return: saliency_matrices: Smoothed version of input.
    """

    print((
        'Smoothing saliency maps with Gaussian filter (e-folding radius of '
        '{0:.1f} grid cells)...'
    ).format(
        smoothing_radius_grid_cells
    ))

    num_matrices = len(saliency_matrices)
    num_examples = saliency_matrices[0].shape[0]

    for j in range(num_matrices):
        this_num_channels = saliency_matrices[j].shape[-1]

        for i in range(num_examples):
            for k in range(this_num_channels):
                saliency_matrices[j][i, ..., k] = (
                    general_utils.apply_gaussian_filter(
                        input_matrix=saliency_matrices[j][i, ..., k],
                        e_folding_radius_grid_cells=smoothing_radius_grid_cells
                    )
                )

    return saliency_matrices


def _run(input_file_name, colour_map_name, max_colour_value, half_num_contours,
         smoothing_radius_grid_cells, plot_soundings, allow_whitespace,
         plot_panel_names, add_titles, label_colour_bars, colour_bar_length,
         output_dir_name):
    """Plots saliency maps.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param colour_map_name: Same.
    :param max_colour_value: Same.
    :param half_num_contours: Same.
    :param smoothing_radius_grid_cells: Same.
    :param plot_soundings: Same.
    :param allow_whitespace: Same.
    :param plot_panel_names: Same.
    :param add_titles: Same.
    :param label_colour_bars: Same.
    :param colour_bar_length: Same.
    :param output_dir_name: Same.
    """

    if max_colour_value <= 0:
        max_colour_value = None
    if smoothing_radius_grid_cells <= 0:
        smoothing_radius_grid_cells = None

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    colour_map_object = pyplot.cm.get_cmap(colour_map_name)
    error_checking.assert_is_geq(half_num_contours, 5)

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    saliency_dict, pmm_flag = saliency_maps.read_file(input_file_name)

    if pmm_flag:
        predictor_matrices = saliency_dict.pop(
            saliency_maps.MEAN_PREDICTOR_MATRICES_KEY)
        saliency_matrices = saliency_dict.pop(
            saliency_maps.MEAN_SALIENCY_MATRICES_KEY)

        full_storm_id_strings = [None]
        storm_times_unix_sec = [None]

        mean_sounding_pressures_pa = saliency_dict[
            saliency_maps.MEAN_SOUNDING_PRESSURES_KEY]
        sounding_pressure_matrix_pa = numpy.reshape(
            mean_sounding_pressures_pa, (1, len(mean_sounding_pressures_pa))
        )

        for i in range(len(predictor_matrices)):
            predictor_matrices[i] = numpy.expand_dims(
                predictor_matrices[i], axis=0
            )
            saliency_matrices[i] = numpy.expand_dims(
                saliency_matrices[i], axis=0
            )
    else:
        predictor_matrices = saliency_dict.pop(
            saliency_maps.PREDICTOR_MATRICES_KEY)
        saliency_matrices = saliency_dict.pop(
            saliency_maps.SALIENCY_MATRICES_KEY)

        full_storm_id_strings = saliency_dict[saliency_maps.FULL_STORM_IDS_KEY]
        storm_times_unix_sec = saliency_dict[saliency_maps.STORM_TIMES_KEY]
        sounding_pressure_matrix_pa = saliency_dict[
            saliency_maps.SOUNDING_PRESSURES_KEY]

    if smoothing_radius_grid_cells is not None:
        saliency_matrices = _smooth_maps(
            saliency_matrices=saliency_matrices,
            smoothing_radius_grid_cells=smoothing_radius_grid_cells)

    model_file_name = saliency_dict[saliency_maps.MODEL_FILE_KEY]
    model_metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0]
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_model_metadata(model_metafile_name)
    print(SEPARATOR_STRING)

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    num_radar_matrices = len(predictor_matrices)

    if training_option_dict[trainval_io.SOUNDING_FIELDS_KEY] is None:
        plot_soundings = False
    else:
        num_radar_matrices -= 1

    num_examples = predictor_matrices[0].shape[0]

    for i in range(num_examples):
        this_handle_dict = plot_examples.plot_one_example(
            list_of_predictor_matrices=predictor_matrices,
            model_metadata_dict=model_metadata_dict, pmm_flag=pmm_flag,
            example_index=i, plot_sounding=plot_soundings,
            sounding_pressures_pascals=sounding_pressure_matrix_pa[i, ...],
            allow_whitespace=allow_whitespace,
            plot_panel_names=plot_panel_names, add_titles=add_titles,
            label_colour_bars=label_colour_bars,
            colour_bar_length=colour_bar_length)

        if plot_soundings:
            _plot_sounding_saliency(
                saliency_matrix=saliency_matrices[-1][i, ...],
                colour_map_object=colour_map_object,
                max_colour_value=max_colour_value,
                sounding_figure_object=this_handle_dict[
                    plot_examples.SOUNDING_FIGURE_KEY],
                sounding_axes_object=this_handle_dict[
                    plot_examples.SOUNDING_AXES_KEY],
                sounding_pressures_pascals=sounding_pressure_matrix_pa[i, ...],
                saliency_dict=saliency_dict,
                model_metadata_dict=model_metadata_dict, add_title=add_titles,
                output_dir_name=output_dir_name, pmm_flag=pmm_flag,
                example_index=i)

        these_figure_objects = this_handle_dict[plot_examples.RADAR_FIGURES_KEY]
        these_axes_object_matrices = this_handle_dict[
            plot_examples.RADAR_AXES_KEY]

        for j in range(num_radar_matrices):
            this_num_spatial_dim = len(predictor_matrices[j].shape) - 2

            if this_num_spatial_dim == 3:
                _plot_3d_radar_saliency(
                    saliency_matrix=saliency_matrices[j][i, ...],
                    colour_map_object=colour_map_object,
                    max_colour_value=max_colour_value,
                    half_num_contours=half_num_contours,
                    label_colour_bars=label_colour_bars,
                    colour_bar_length=colour_bar_length,
                    figure_objects=these_figure_objects,
                    axes_object_matrices=these_axes_object_matrices,
                    model_metadata_dict=model_metadata_dict,
                    output_dir_name=output_dir_name,
                    significance_matrix=None,
                    full_storm_id_string=full_storm_id_strings[i],
                    storm_time_unix_sec=storm_times_unix_sec[i]
                )
            else:
                _plot_2d_radar_saliency(
                    saliency_matrix=saliency_matrices[j][i, ...],
                    colour_map_object=colour_map_object,
                    max_colour_value=max_colour_value,
                    half_num_contours=half_num_contours,
                    label_colour_bars=label_colour_bars,
                    colour_bar_length=colour_bar_length,
                    figure_objects=these_figure_objects,
                    axes_object_matrices=these_axes_object_matrices,
                    model_metadata_dict=model_metadata_dict,
                    output_dir_name=output_dir_name,
                    significance_matrix=None,
                    full_storm_id_string=full_storm_id_strings[i],
                    storm_time_unix_sec=storm_times_unix_sec[i]
                )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        colour_map_name=getattr(INPUT_ARG_OBJECT, COLOUR_MAP_ARG_NAME),
        max_colour_value=getattr(INPUT_ARG_OBJECT, MAX_COLOUR_VALUE_ARG_NAME),
        half_num_contours=getattr(INPUT_ARG_OBJECT, HALF_NUM_CONTOURS_ARG_NAME),
        smoothing_radius_grid_cells=getattr(
            INPUT_ARG_OBJECT, SMOOTHING_RADIUS_ARG_NAME),
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
