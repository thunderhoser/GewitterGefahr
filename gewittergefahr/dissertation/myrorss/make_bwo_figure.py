"""Makes figure with backwards-optimization results for MYRORSS model."""

import os
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import backwards_optimization as backwards_opt
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.plotting import imagemagick_utils
from gewittergefahr.scripts import plot_input_examples as plot_examples

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

REFL_HEIGHTS_M_AGL = numpy.array([2000, 6000, 10000], dtype=int)

MODEL_FILE_KEY = backwards_opt.MODEL_FILE_KEY
MEAN_INPUT_MATRICES_KEY = backwards_opt.MEAN_INPUT_MATRICES_KEY
MEAN_OUTPUT_MATRICES_KEY = backwards_opt.MEAN_OUTPUT_MATRICES_KEY

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
SOUNDING_FIGURE_SIZE_PX = int(8e6)

INPUT_FILE_ARG_NAME = 'input_bwo_file_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file (will be read by `backwards_optimization.read_file`).'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


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
        bwo_dictionary[MEAN_INPUT_MATRICES_KEY]
    ]
    mean_after_matrices = [
        numpy.expand_dims(a, axis=0) for a in
        bwo_dictionary[MEAN_OUTPUT_MATRICES_KEY]
    ]

    model_file_name = bwo_dictionary[MODEL_FILE_KEY]
    model_metafile_name = cnn.find_metafile(model_file_name)

    print('Reading CNN metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_model_metadata(model_metafile_name)
    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    training_option_dict[trainval_io.UPSAMPLE_REFLECTIVITY_KEY] = False

    all_refl_heights_m_agl = training_option_dict[trainval_io.RADAR_HEIGHTS_KEY]
    good_flags = numpy.array(
        [h in REFL_HEIGHTS_M_AGL for h in all_refl_heights_m_agl], dtype=bool
    )
    good_indices = numpy.where(good_flags)[0]

    mean_before_matrices[0] = mean_before_matrices[0][..., good_indices, :]
    mean_after_matrices[0] = mean_after_matrices[0][..., good_indices, :]

    bwo_dictionary[MEAN_INPUT_MATRICES_KEY] = mean_before_matrices
    bwo_dictionary[MEAN_OUTPUT_MATRICES_KEY] = mean_after_matrices

    training_option_dict[trainval_io.RADAR_HEIGHTS_KEY] = REFL_HEIGHTS_M_AGL
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
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    imagemagick_utils.resize_image(
        input_file_name=output_file_name, output_file_name=output_file_name,
        output_size_pixels=SOUNDING_FIGURE_SIZE_PX
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name, output_file_name=output_file_name,
        border_width_pixels=TITLE_FONT_SIZE + 25
    )
    _overlay_text(
        image_file_name=output_file_name, text_string=title_string,
        x_offset_from_center_px=0, y_offset_from_top_px=0
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name, output_file_name=output_file_name,
        border_width_pixels=10
    )


def _write_radar_figures(
        refl_figure_object, shear_figure_object, composite_name,
        concat_title_string, output_dir_name):
    """Writes radar figures to file.

    :param refl_figure_object: Handle for reflectivity figure (instance of
        `matplotlib.figure.Figure`).
    :param shear_figure_object: Handle for azimuthal-shear figure (instance of
        `matplotlib.figure.Figure`).
    :param composite_name: Name of composite (e.g., "after" or "difference").
    :param concat_title_string: Title for concatenated figure.
    :param output_dir_name: Name of output directory (figures will be saved
        here).
    :return: concat_figure_file_name: Path to image file with concatenated
        figure.
    """

    refl_figure_file_name = '{0:s}/reflectivity_{1:s}.jpg'.format(
        output_dir_name, composite_name
    )
    print('Saving figure to: "{0:s}"...'.format(refl_figure_file_name))

    refl_figure_object.savefig(
        refl_figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(refl_figure_object)

    shear_figure_file_name = '{0:s}/shear_{1:s}.jpg'.format(
        output_dir_name, composite_name
    )
    print('Saving figure to: "{0:s}"...'.format(shear_figure_file_name))

    shear_figure_object.savefig(
        shear_figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(shear_figure_object)

    concat_figure_file_name = '{0:s}/radar_{1:s}.jpg'.format(
        output_dir_name, composite_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=[refl_figure_file_name, shear_figure_file_name],
        output_file_name=concat_figure_file_name,
        num_panel_rows=1, num_panel_columns=2, border_width_pixels=50,
        extra_args_string='-gravity south'
    )
    imagemagick_utils.resize_image(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name,
        border_width_pixels=TITLE_FONT_SIZE + 25
    )
    _overlay_text(
        image_file_name=concat_figure_file_name,
        x_offset_from_center_px=0, y_offset_from_top_px=0,
        text_string=concat_title_string
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name,
        border_width_pixels=10
    )

    return concat_figure_file_name


def _run(bwo_file_name, output_dir_name):
    """Makes figure with backwards-optimization results for MYRORSS model.

    This is effectively the main method.

    :param bwo_file_name: See documentation at top of file.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    bwo_dictionary, model_metadata_dict = _read_bwo_file(bwo_file_name)

    mean_before_matrices = bwo_dictionary[backwards_opt.MEAN_INPUT_MATRICES_KEY]
    mean_after_matrices = bwo_dictionary[backwards_opt.MEAN_OUTPUT_MATRICES_KEY]
    mean_sounding_pressures_pa = (
        bwo_dictionary[backwards_opt.MEAN_SOUNDING_PRESSURES_KEY]
    )

    num_radar_heights = mean_before_matrices[0].shape[-2]

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
        sounding_font_size=SOUNDING_FONT_SIZE,
        num_panel_rows=num_radar_heights, plot_radar_diffs=False
    )

    panel_file_names = [None] * 4
    panel_file_names[2] = '{0:s}/sounding_before.jpg'.format(output_dir_name)

    _write_sounding_figure(
        figure_object=handle_dict[plot_examples.SOUNDING_FIGURE_KEY],
        title_string='(c) Original sounding',
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
        sounding_font_size=SOUNDING_FONT_SIZE,
        num_panel_rows=num_radar_heights, plot_radar_diffs=False
    )

    panel_file_names[3] = '{0:s}/sounding_after.jpg'.format(output_dir_name)

    _write_sounding_figure(
        figure_object=handle_dict[plot_examples.SOUNDING_FIGURE_KEY],
        title_string='(d) Synthetic sounding',
        output_file_name=panel_file_names[3]
    )

    panel_file_names[0] = _write_radar_figures(
        refl_figure_object=handle_dict[plot_examples.RADAR_FIGURES_KEY][0],
        shear_figure_object=handle_dict[plot_examples.RADAR_FIGURES_KEY][1],
        composite_name='after',
        concat_title_string='(a) Synthetic radar image',
        output_dir_name=output_dir_name
    )

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
        max_diff_percentile=MAX_DIFF_PERCENTILE
    )

    panel_file_names[1] = _write_radar_figures(
        refl_figure_object=handle_dict[plot_examples.RADAR_FIGURES_KEY][0],
        shear_figure_object=handle_dict[plot_examples.RADAR_FIGURES_KEY][1],
        composite_name='difference',
        concat_title_string='(b) Radar difference',
        output_dir_name=output_dir_name
    )

    print(SEPARATOR_STRING)

    figure_file_name = '{0:s}/bwo_concat.jpg'.format(output_dir_name)
    print('Concatenating panels to: "{0:s}"...'.format(figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=figure_file_name, border_width_pixels=100,
        num_panel_rows=2, num_panel_columns=2,
        extra_args_string='-gravity Center'
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=figure_file_name, output_file_name=figure_file_name,
        border_width_pixels=10
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        bwo_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
