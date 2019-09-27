"""Makes figure with PMM composites of extreme examples for GridRad model.

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
from gewittergefahr.scripts import plot_input_examples

RADAR_HEIGHTS_M_AGL = numpy.array([2000, 6000, 10000], dtype=int)

MODEL_FILE_KEY = model_interpretation.MODEL_FILE_KEY
MEAN_INPUT_MATRICES_KEY = model_interpretation.MEAN_INPUT_MATRICES_KEY

FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

INPUT_FILES_ARG_NAME = 'input_composite_file_names'
COMPOSITE_NAMES_ARG_NAME = 'composite_names'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILES_HELP_STRING = (
    'List of paths to input files.  Each should contain a PMM composite over '
    'many examples (storm objects).  Specifically, each should be a Pickle file'
    ' with one dictionary, containing the keys "{0:s}" and "{1:s}".'
).format(MEAN_INPUT_MATRICES_KEY, MODEL_FILE_KEY)

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


def _read_pmm_composite(pickle_file_name):
    """Reads PMM composite of examples (storm objects) from Pickle file.

    T = number of input tensors to model

    :param pickle_file_name: Path to input file.
    :return: list_of_mean_input_matrices: length-T of numpy arrays, where the
        [i]th item has dimensions of the [i]th input tensor to the model.
    :return: model_metadata_dict: Dictionary returned by
        `cnn.read_model_metadata`.
    """

    print('Reading data from: "{0:s}"...'.format(pickle_file_name))
    file_handle = open(pickle_file_name, 'rb')
    composite_dict = pickle.load(file_handle)
    file_handle.close()

    list_of_mean_input_matrices = composite_dict[MEAN_INPUT_MATRICES_KEY]
    for i in range(len(list_of_mean_input_matrices)):
        list_of_mean_input_matrices[i] = numpy.expand_dims(
            list_of_mean_input_matrices[i], axis=0
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

    all_radar_heights_m_agl = model_metadata_dict[
        cnn.TRAINING_OPTION_DICT_KEY][trainval_io.RADAR_HEIGHTS_KEY]

    good_flags = numpy.array(
        [h in RADAR_HEIGHTS_M_AGL for h in all_radar_heights_m_agl], dtype=bool
    )
    good_indices = numpy.where(good_flags)[0]

    list_of_mean_input_matrices[0] = list_of_mean_input_matrices[0][
        ..., good_indices, :]

    model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY][
        trainval_io.RADAR_HEIGHTS_KEY
    ] = RADAR_HEIGHTS_M_AGL

    return list_of_mean_input_matrices, model_metadata_dict


def _run(composite_file_names, composite_names, output_dir_name):
    """Makes figure with extreme examples for GridRad model.

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

    for i in range(num_composites):
        list_of_mean_input_matrices, model_metadata_dict = _read_pmm_composite(
            composite_file_names[i]
        )

        print('\n')

        radar_field_names = model_metadata_dict[
            cnn.TRAINING_OPTION_DICT_KEY][trainval_io.RADAR_FIELDS_KEY]
        num_radar_fields = len(radar_field_names)

        handle_dict = plot_input_examples.plot_one_example(
            list_of_predictor_matrices=list_of_mean_input_matrices,
            model_metadata_dict=model_metadata_dict, pmm_flag=True,
            plot_sounding=True, allow_whitespace=True, plot_panel_names=True,
            panel_name_font_size=45,
            add_titles=False, label_colour_bars=True, colour_bar_length=0.8,
            colour_bar_font_size=45, sounding_font_size=45)

        sounding_figure_object = handle_dict[
            plot_input_examples.SOUNDING_FIGURE_KEY]
        pyplot.close(sounding_figure_object)

        radar_figure_objects = handle_dict[
            plot_input_examples.RADAR_FIGURES_KEY]
        radar_panel_file_names = [None] * num_radar_fields

        for j in range(num_radar_fields):
            radar_panel_file_names[j] = '{0:s}/{1:s}_{2:s}.jpg'.format(
                output_dir_name, composite_names_abbrev[i],
                radar_field_names[j].replace('_', '-')
            )

            print('Saving figure to: "{0:s}"...'.format(
                radar_panel_file_names[j]
            ))

            radar_figure_objects[j].savefig(
                radar_panel_file_names[j], dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(radar_figure_objects[j])

        radar_concat_file_name = '{0:s}/{1:s}_radar.jpg'.format(
            output_dir_name, composite_names_abbrev[i]
        )

        print('Concatenating panels to: "{0:s}"...'.format(
            radar_concat_file_name
        ))

        imagemagick_utils.concatenate_images(
            input_file_names=radar_panel_file_names,
            output_file_name=radar_concat_file_name,
            num_panel_rows=1, num_panel_columns=num_radar_fields)

        imagemagick_utils.resize_image(
            input_file_name=radar_concat_file_name,
            output_file_name=radar_concat_file_name,
            output_size_pixels=CONCAT_FIGURE_SIZE_PX)

        print('\n')


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        composite_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        composite_names=getattr(INPUT_ARG_OBJECT, COMPOSITE_NAMES_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
