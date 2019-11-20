"""Makes figure with backwards-optimization results."""

import os
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.deep_learning import backwards_optimization as backwards_opt
from gewittergefahr.plotting import imagemagick_utils
from gewittergefahr.scripts import plot_input_examples as plot_examples

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

RADAR_HEIGHTS_M_AGL = numpy.array([2000, 6000, 10000], dtype=int)
RADAR_FIELD_NAMES = [
    radar_utils.REFL_NAME, radar_utils.VORTICITY_NAME,
    radar_utils.SPECTRUM_WIDTH_NAME
]

DIFF_COLOUR_MAP_OBJECT = pyplot.get_cmap('seismic')
MAX_DIFF_PERCENTILE = 99.

COLOUR_BAR_LENGTH = 0.25
PANEL_NAME_FONT_SIZE = 30
COLOUR_BAR_FONT_SIZE = 25
SOUNDING_FONT_SIZE = 30

CONVERT_EXE_NAME = '/usr/bin/convert'
TITLE_FONT_SIZE = 150
TITLE_FONT_NAME = 'DejaVu-Sans-Bold'

FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)
SOUNDING_FIGURE_SIZE_PX = int(6e8)

INPUT_FILE_ARG_NAME = 'input_bwo_file_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file (will be read by `backwards_optimization.read_file`).')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _read_bwo_file(bwo_file_name):
    """Reads backwards-optimization results from file.
    
    :param bwo_file_name: Path to input file (will be read by
        `backwards_optimization.read_file`).
    :return: bwo_dictionary: Dictionary returned by
        `backwards_optimization.read_file`.
    :return: model_metadata_dict: Dictionary returned by
        `cnn.read_model_metadata`.
    """

    print('Reading data from: "{0:s}"...'.format(bwo_file_name))
    bwo_dictionary = backwards_opt.read_file(bwo_file_name)[0]

    mean_before_matrices = [
        numpy.expand_dims(a, axis=0) for a in
        bwo_dictionary[backwards_opt.MEAN_INPUT_MATRICES_KEY]
    ]
    mean_after_matrices = [
        numpy.expand_dims(a, axis=0) for a in
        bwo_dictionary[backwards_opt.MEAN_OUTPUT_MATRICES_KEY]
    ]

    model_file_name = bwo_dictionary[backwards_opt.MODEL_FILE_KEY]
    model_metafile_name = cnn.find_metafile(model_file_name)

    print('Reading CNN metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_model_metadata(model_metafile_name)
    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]

    good_indices = numpy.array([
        numpy.where(
            training_option_dict[trainval_io.RADAR_HEIGHTS_KEY] == h
        )[0][0]
        for h in RADAR_HEIGHTS_M_AGL
    ], dtype=int)

    mean_before_matrices[0] = mean_before_matrices[0][
        ..., good_indices, :]
    mean_after_matrices[0] = mean_after_matrices[0][
        ..., good_indices, :]

    good_indices = numpy.array([
        training_option_dict[trainval_io.RADAR_FIELDS_KEY].index(f)
        for f in RADAR_FIELD_NAMES
    ], dtype=int)

    mean_before_matrices[0] = mean_before_matrices[0][
        ..., good_indices]
    mean_after_matrices[0] = mean_after_matrices[0][
        ..., good_indices]

    bwo_dictionary[backwards_opt.MEAN_INPUT_MATRICES_KEY] = mean_before_matrices
    bwo_dictionary[backwards_opt.MEAN_OUTPUT_MATRICES_KEY] = mean_after_matrices

    training_option_dict[trainval_io.RADAR_HEIGHTS_KEY] = RADAR_HEIGHTS_M_AGL
    training_option_dict[trainval_io.RADAR_FIELDS_KEY] = RADAR_FIELD_NAMES
    model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY] = training_option_dict

    return bwo_dictionary, model_metadata_dict


def _overlay_text(
        image_file_name, x_offset_from_center_px, y_offset_from_top_px,
        text_string):
    """Overlays text on image.

    :param image_file_name: Path to image file.
    :param x_offset_from_center_px: Center-relative x-coordinate (pixels).
    :param y_offset_from_top_px: Top-relative y-coordinate (pixels).
    :param text_string: String to overlay.
    :raises: ValueError: if ImageMagick command (which is ultimately a Unix
        command) fails.
    """

    command_string = (
        '"{0:s}" "{1:s}" -gravity north -pointsize {2:d} -font "{3:s}" '
        '-fill "rgb(0, 0, 0)" -annotate {4:+d}{5:+d} "{6:s}" "{1:s}"'
    ).format(
        CONVERT_EXE_NAME, image_file_name, TITLE_FONT_SIZE, TITLE_FONT_NAME,
        x_offset_from_center_px, y_offset_from_top_px, text_string
    )

    exit_code = os.system(command_string)
    if exit_code == 0:
        return

    raise ValueError(imagemagick_utils.ERROR_STRING)


def _write_sounding_figure(figure_object, title_string, output_file_name):
    """Writes sounding figure to file.

    :param figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :param title_string: Figure title.
    :param output_file_name: Path to output file (figure will be saved as image
        here).
    """

    print('Saving figure to: "{0:s}"...'.format(output_file_name))

    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight')
    pyplot.close(figure_object)

    imagemagick_utils.resize_image(
        input_file_name=output_file_name, output_file_name=output_file_name,
        output_size_pixels=SOUNDING_FIGURE_SIZE_PX)

    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name, output_file_name=output_file_name,
        border_width_pixels=TITLE_FONT_SIZE + 25)

    _overlay_text(
        image_file_name=output_file_name, text_string=title_string,
        x_offset_from_center_px=0, y_offset_from_top_px=0)

    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name, output_file_name=output_file_name,
        border_width_pixels=10)


def _write_radar_figures(figure_objects, field_names, composite_name,
                         concat_title_string, output_dir_name):
    """Writes radar figures to file.

    F = number of radar fields

    :param figure_objects: length-F list of figure handles (each an instance of
        `matplotlib.figure.Figure`).
    :param field_names: length-F list of field names.
    :param composite_name: Name of composite (e.g., "after" or "difference").
    :param concat_title_string: Title for concatenated figure.
    :param output_dir_name: Name of output directory (figures will be saved
        here).
    :return: concat_figure_file_name: Path to image file with concatenated
        figure.
    """

    num_fields = len(field_names)
    panel_file_names = [None] * num_fields

    for k in range(num_fields):
        panel_file_names[k] = '{0:s}/{1:s}_{2:s}.jpg'.format(
            output_dir_name, field_names[k].replace('_', '-'), composite_name
        )

        print('Saving figure to: "{0:s}"...'.format(panel_file_names[k]))

        figure_objects[k].savefig(
            panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_objects[k])

    concat_figure_file_name = '{0:s}/radar_{1:s}.jpg'.format(
        output_dir_name, composite_name)

    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_figure_file_name,
        num_panel_rows=1, num_panel_columns=num_fields, border_width_pixels=50)

    imagemagick_utils.resize_image(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX)

    imagemagick_utils.trim_whitespace(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name,
        border_width_pixels=TITLE_FONT_SIZE + 25)

    _overlay_text(
        image_file_name=concat_figure_file_name,
        text_string=concat_title_string,
        x_offset_from_center_px=0, y_offset_from_top_px=0)

    imagemagick_utils.trim_whitespace(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name,
        border_width_pixels=10)

    return concat_figure_file_name


def _run(bwo_file_name, output_dir_name):
    """Makes figure with backwards-optimization results.

    This is effectively the main method.

    :param bwo_file_name: See documentation at top of file.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    bwo_dictionary, model_metadata_dict = _read_bwo_file(bwo_file_name)
    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    radar_field_names = training_option_dict[trainval_io.RADAR_FIELDS_KEY]

    mean_before_matrices = bwo_dictionary[backwards_opt.MEAN_INPUT_MATRICES_KEY]
    mean_after_matrices = bwo_dictionary[backwards_opt.MEAN_OUTPUT_MATRICES_KEY]
    mean_sounding_pressures_pa = bwo_dictionary[
        backwards_opt.MEAN_SOUNDING_PRESSURES_KEY]

    num_radar_heights = mean_before_matrices[0].shape[-1]

    # Plot sounding before optimization.
    handle_dict = plot_examples.plot_one_example(
        list_of_predictor_matrices=mean_before_matrices,
        model_metadata_dict=model_metadata_dict,
        pmm_flag=True, plot_sounding=True,
        sounding_pressures_pascals=mean_sounding_pressures_pa,
        allow_whitespace=True, plot_panel_names=True,
        panel_name_font_size=PANEL_NAME_FONT_SIZE, add_titles=False,
        label_colour_bars=True, colour_bar_length=COLOUR_BAR_LENGTH,
        colour_bar_font_size=COLOUR_BAR_FONT_SIZE,
        sounding_font_size=SOUNDING_FONT_SIZE, num_panel_rows=num_radar_heights,
        plot_radar_diffs=False)

    panel_file_names = [None] * 4
    panel_file_names[2] = '{0:s}/sounding_before.jpg'.format(output_dir_name)

    _write_sounding_figure(
        figure_object=handle_dict[plot_examples.SOUNDING_FIGURE_KEY],
        title_string='Sounding before optimization',
        output_file_name=panel_file_names[2]
    )
    print(SEPARATOR_STRING)

    # Plot radar and sounding after optimization.
    handle_dict = plot_examples.plot_one_example(
        list_of_predictor_matrices=mean_after_matrices,
        model_metadata_dict=model_metadata_dict,
        pmm_flag=True, plot_sounding=True,
        sounding_pressures_pascals=mean_sounding_pressures_pa,
        allow_whitespace=True, plot_panel_names=True,
        panel_name_font_size=PANEL_NAME_FONT_SIZE, add_titles=False,
        label_colour_bars=True, colour_bar_length=COLOUR_BAR_LENGTH,
        colour_bar_font_size=COLOUR_BAR_FONT_SIZE,
        sounding_font_size=SOUNDING_FONT_SIZE, num_panel_rows=num_radar_heights,
        plot_radar_diffs=False)

    panel_file_names[3] = '{0:s}/sounding_after.jpg'.format(output_dir_name)

    _write_sounding_figure(
        figure_object=handle_dict[plot_examples.SOUNDING_FIGURE_KEY],
        title_string='Sounding after optimization',
        output_file_name=panel_file_names[3]
    )

    panel_file_names[0] = _write_radar_figures(
        figure_objects=handle_dict[plot_examples.RADAR_FIGURES_KEY],
        field_names=radar_field_names, composite_name='after',
        concat_title_string='Radar after optimization',
        output_dir_name=output_dir_name)

    print(SEPARATOR_STRING)

    mean_difference_matrices = [
        a - b for a, b in zip(mean_after_matrices, mean_before_matrices)
    ]

    handle_dict = plot_examples.plot_one_example(
        list_of_predictor_matrices=mean_difference_matrices,
        model_metadata_dict=model_metadata_dict,
        pmm_flag=True, plot_sounding=False,
        allow_whitespace=True, plot_panel_names=True,
        panel_name_font_size=PANEL_NAME_FONT_SIZE, add_titles=False,
        label_colour_bars=True, colour_bar_length=COLOUR_BAR_LENGTH,
        colour_bar_font_size=COLOUR_BAR_FONT_SIZE,
        num_panel_rows=num_radar_heights, plot_radar_diffs=True,
        diff_colour_map_object=DIFF_COLOUR_MAP_OBJECT,
        max_diff_percentile=MAX_DIFF_PERCENTILE)

    panel_file_names[1] = _write_radar_figures(
        figure_objects=handle_dict[plot_examples.RADAR_FIGURES_KEY],
        field_names=radar_field_names, composite_name='difference',
        concat_title_string='Radar difference',
        output_dir_name=output_dir_name)

    print(SEPARATOR_STRING)

    figure_file_name = '{0:s}/bwo_concat.jpg'.format(output_dir_name)
    print('Concatenating panels to: "{0:s}"...'.format(figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=figure_file_name, border_width_pixels=100,
        num_panel_rows=2, num_panel_columns=2,
        extra_args_string='-gravity Center')

    imagemagick_utils.trim_whitespace(
        input_file_name=figure_file_name, output_file_name=figure_file_name,
        border_width_pixels=10)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        bwo_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
