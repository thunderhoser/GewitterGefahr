"""Makes figure with PMM composites of extreme examples for MYRORSS model.

PMM = probability-matched means

"Extreme examples" include best hits, best correct nulls, worst misses, worst
false alarms, high-probability examples (regardless of true label), and
low-probability examples (regardless of true label).
"""

import pickle
import os.path
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import model_interpretation
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.plotting import imagemagick_utils
from gewittergefahr.scripts import plot_input_examples as plot_examples

REFL_HEIGHTS_M_AGL = numpy.array([2000, 6000, 10000], dtype=int)

MODEL_FILE_KEY = model_interpretation.MODEL_FILE_KEY
MEAN_PREDICTOR_MATRICES_KEY = model_interpretation.MEAN_PREDICTOR_MATRICES_KEY
MEAN_SOUNDING_PRESSURES_KEY = model_interpretation.MEAN_SOUNDING_PRESSURES_KEY

COLOUR_BAR_LENGTH = 0.25
PANEL_NAME_FONT_SIZE = 30
COLOUR_BAR_FONT_SIZE = 25
SOUNDING_FONT_SIZE = 30

CONVERT_EXE_NAME = '/usr/bin/convert'
TITLE_FONT_SIZE = 150
TITLE_FONT_TYPE = 'DejaVu-Sans-Bold'

FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

INPUT_FILES_ARG_NAME = 'input_composite_file_names'
COMPOSITE_NAMES_ARG_NAME = 'composite_names'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILES_HELP_STRING = (
    'List of paths to input files.  Each should contain a PMM composite over '
    'many examples (storm objects).  Specifically, each should be a Pickle file'
    ' with one dictionary, containing the keys "{0:s}" and "{1:s}".'
).format(MEAN_PREDICTOR_MATRICES_KEY, MODEL_FILE_KEY)

COMPOSITE_NAMES_HELP_STRING = (
    'List of PMM-composite names (one per input file).  The list should be '
    'space-separated.  In each list item, underscores will be replaced with '
    'spaces.')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_FILES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + COMPOSITE_NAMES_ARG_NAME, type=str, nargs='+', required=True,
    help=COMPOSITE_NAMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _read_composite(pickle_file_name):
    """Reads PMM composite of examples (storm objects) from Pickle file.

    T = number of input tensors to model
    H_s = number of sounding heights

    :param pickle_file_name: Path to input file.
    :return: mean_predictor_matrices: length-T of numpy arrays, where the [i]th
        item has dimensions of the [i]th input tensor to the model.
    :return: model_metadata_dict: Dictionary returned by
        `cnn.read_model_metadata`.
    :return: mean_sounding_pressures_pa: numpy array (length H_s) of
        sounding pressures.
    """

    print('Reading data from: "{0:s}"...'.format(pickle_file_name))
    file_handle = open(pickle_file_name, 'rb')
    composite_dict = pickle.load(file_handle)
    file_handle.close()

    mean_predictor_matrices = composite_dict[MEAN_PREDICTOR_MATRICES_KEY]
    mean_sounding_pressures_pa = composite_dict[MEAN_SOUNDING_PRESSURES_KEY]

    for i in range(len(mean_predictor_matrices)):
        mean_predictor_matrices[i] = numpy.expand_dims(
            mean_predictor_matrices[i], axis=0
        )

    model_file_name = composite_dict[MODEL_FILE_KEY]
    model_metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0]
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_model_metadata(model_metafile_name)
    model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY][
        trainval_io.UPSAMPLE_REFLECTIVITY_KEY
    ] = False

    all_refl_heights_m_agl = model_metadata_dict[
        cnn.TRAINING_OPTION_DICT_KEY][trainval_io.RADAR_HEIGHTS_KEY]

    good_flags = numpy.array(
        [h in REFL_HEIGHTS_M_AGL for h in all_refl_heights_m_agl], dtype=bool
    )
    good_indices = numpy.where(good_flags)[0]

    mean_predictor_matrices[0] = mean_predictor_matrices[0][
        ..., good_indices, :]

    model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY][
        trainval_io.RADAR_HEIGHTS_KEY
    ] = REFL_HEIGHTS_M_AGL

    return (mean_predictor_matrices, model_metadata_dict,
            mean_sounding_pressures_pa)


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
        CONVERT_EXE_NAME, image_file_name, TITLE_FONT_SIZE, TITLE_FONT_TYPE,
        x_offset_from_center_px, y_offset_from_top_px, text_string
    )

    exit_code = os.system(command_string)
    if exit_code == 0:
        return

    raise ValueError(imagemagick_utils.ERROR_STRING)


def _plot_composite(
        composite_file_name, composite_name_abbrev, composite_name_verbose,
        output_dir_name):
    """Plots one composite.

    :param composite_file_name: Path to input file.  Will be read by
        `_read_composite`.
    :param composite_name_abbrev: Abbreviated name for composite.  Will be used
        in names of output files.
    :param composite_name_verbose: Verbose name for composite.  Will be used as
        figure title.
    :param output_dir_name: Path to output directory.  Figures will be saved
        here.
    :return: radar_figure_file_name: Path to file with radar figure for this
        composite.
    :return: sounding_figure_file_name: Path to file with sounding figure for
        this composite.
    """

    mean_predictor_matrices, model_metadata_dict, mean_sounding_pressures_pa = (
        _read_composite(composite_file_name)
    )

    refl_heights_m_agl = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY][
        trainval_io.RADAR_HEIGHTS_KEY]
    num_refl_heights = len(refl_heights_m_agl)

    handle_dict = plot_examples.plot_one_example(
        list_of_predictor_matrices=mean_predictor_matrices,
        model_metadata_dict=model_metadata_dict,
        pmm_flag=True, plot_sounding=True,
        sounding_pressures_pascals=mean_sounding_pressures_pa,
        allow_whitespace=True, plot_panel_names=True,
        panel_name_font_size=PANEL_NAME_FONT_SIZE,
        add_titles=False, label_colour_bars=True,
        colour_bar_length=COLOUR_BAR_LENGTH,
        colour_bar_font_size=COLOUR_BAR_FONT_SIZE,
        sounding_font_size=SOUNDING_FONT_SIZE, num_panel_rows=num_refl_heights)

    sounding_figure_object = handle_dict[plot_examples.SOUNDING_FIGURE_KEY]
    sounding_figure_file_name = '{0:s}/{1:s}_sounding.jpg'.format(
        output_dir_name, composite_name_abbrev)

    print('Saving figure to: "{0:s}"...'.format(sounding_figure_file_name))
    sounding_figure_object.savefig(
        sounding_figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight')
    pyplot.close(sounding_figure_object)

    imagemagick_utils.resize_image(
        input_file_name=sounding_figure_file_name,
        output_file_name=sounding_figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX)

    imagemagick_utils.trim_whitespace(
        input_file_name=sounding_figure_file_name,
        output_file_name=sounding_figure_file_name,
        border_width_pixels=TITLE_FONT_SIZE + 25)

    _overlay_text(
        image_file_name=sounding_figure_file_name,
        x_offset_from_center_px=0, y_offset_from_top_px=0,
        text_string=composite_name_verbose)

    imagemagick_utils.trim_whitespace(
        input_file_name=sounding_figure_file_name,
        output_file_name=sounding_figure_file_name,
        border_width_pixels=10)

    refl_figure_object = handle_dict[plot_examples.RADAR_FIGURES_KEY][0]
    refl_figure_file_name = '{0:s}/{1:s}_reflectivity.jpg'.format(
        output_dir_name, composite_name_abbrev)

    print('Saving figure to: "{0:s}"...'.format(refl_figure_file_name))
    refl_figure_object.savefig(
        refl_figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight')
    pyplot.close(refl_figure_object)

    shear_figure_object = handle_dict[plot_examples.RADAR_FIGURES_KEY][1]
    shear_figure_file_name = '{0:s}/{1:s}_shear.jpg'.format(
        output_dir_name, composite_name_abbrev)

    print('Saving figure to: "{0:s}"...'.format(shear_figure_file_name))
    shear_figure_object.savefig(
        shear_figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight')
    pyplot.close(shear_figure_object)

    radar_figure_file_name = '{0:s}/{1:s}_radar.jpg'.format(
        output_dir_name, composite_name_abbrev)

    print('Concatenating panels to: "{0:s}"...'.format(radar_figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=[refl_figure_file_name, shear_figure_file_name],
        output_file_name=radar_figure_file_name,
        num_panel_rows=1, num_panel_columns=2, border_width_pixels=50,
        extra_args_string='-gravity south')

    imagemagick_utils.resize_image(
        input_file_name=radar_figure_file_name,
        output_file_name=radar_figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX)

    imagemagick_utils.trim_whitespace(
        input_file_name=radar_figure_file_name,
        output_file_name=radar_figure_file_name,
        border_width_pixels=TITLE_FONT_SIZE + 25)

    _overlay_text(
        image_file_name=radar_figure_file_name,
        x_offset_from_center_px=0, y_offset_from_top_px=0,
        text_string=composite_name_verbose)

    imagemagick_utils.trim_whitespace(
        input_file_name=radar_figure_file_name,
        output_file_name=radar_figure_file_name,
        border_width_pixels=10)

    return radar_figure_file_name, sounding_figure_file_name


def _run(composite_file_names, composite_names, output_dir_name):
    """Makes figure with extreme examples for MYRORSS model.

    This is effectively the main method.

    :param composite_file_names: See documentation at top of file.
    :param composite_names: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    num_composites = len(composite_file_names)
    expected_dim = numpy.array([num_composites], dtype=int)
    error_checking.assert_is_numpy_array(
        numpy.array(composite_names), exact_dimensions=expected_dim
    )

    composite_names_abbrev = [
        n.replace('_', '-').lower() for n in composite_names
    ]
    composite_names_verbose = [n.replace('_', ' ') for n in composite_names]

    radar_panel_file_names = [None] * num_composites
    sounding_panel_file_names = [None] * num_composites

    for i in range(num_composites):
        radar_panel_file_names[i], sounding_panel_file_names[i] = (
            _plot_composite(
                composite_file_name=composite_file_names[i],
                composite_name_abbrev=composite_names_abbrev[i],
                composite_name_verbose=composite_names_verbose[i],
                output_dir_name=output_dir_name)
        )

        print('\n')

    radar_figure_file_name = '{0:s}/radar_concat.jpg'.format(output_dir_name)
    print('Concatenating panels to: "{0:s}"...'.format(radar_figure_file_name))

    num_panel_rows = int(numpy.floor(
        numpy.sqrt(num_composites)
    ))
    num_panel_columns = int(numpy.ceil(
        float(num_composites) / num_panel_rows
    ))

    imagemagick_utils.concatenate_images(
        input_file_names=radar_panel_file_names,
        output_file_name=radar_figure_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns)

    imagemagick_utils.trim_whitespace(
        input_file_name=radar_figure_file_name,
        output_file_name=radar_figure_file_name)

    imagemagick_utils.resize_image(
        input_file_name=radar_figure_file_name,
        output_file_name=radar_figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX)

    sounding_figure_file_name = '{0:s}/sounding_concat.jpg'.format(
        output_dir_name)
    print('Concatenating panels to: "{0:s}"...'.format(
        sounding_figure_file_name
    ))

    imagemagick_utils.concatenate_images(
        input_file_names=sounding_panel_file_names,
        output_file_name=sounding_figure_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns)

    imagemagick_utils.trim_whitespace(
        input_file_name=sounding_figure_file_name,
        output_file_name=sounding_figure_file_name)

    imagemagick_utils.resize_image(
        input_file_name=sounding_figure_file_name,
        output_file_name=sounding_figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        composite_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        composite_names=getattr(INPUT_ARG_OBJECT, COMPOSITE_NAMES_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
