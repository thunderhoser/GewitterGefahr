"""Plots data augmentation."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.scripts import plot_input_examples as plot_examples

RADAR_FIELD_NAME = radar_utils.REFL_NAME
RADAR_HEIGHT_M_AGL = 3000
NORMALIZATION_TYPE_STRING = dl_utils.Z_NORMALIZATION_TYPE_STRING

X_TRANSLATIONS_PX = numpy.array([3], dtype=int)
Y_TRANSLATIONS_PX = numpy.array([3], dtype=int)
CCW_ROTATION_ANGLES_DEG = numpy.array([30.])
NOISE_STANDARD_DEVIATION = 0.1

TITLE_FONT_SIZE = 30
FIGURE_RESOLUTION_DPI = 300
FILE_NAME_TIME_FORMAT = '%Y-%m-%d-%H%M%S'

EXAMPLE_FILE_ARG_NAME = 'input_example_file_name'
EXAMPLE_INDICES_ARG_NAME = 'example_indices'
NORMALIZATION_FILE_ARG_NAME = 'normalization_file_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

EXAMPLE_FILE_HELP_STRING = (
    'Path to example file.  Radar images will be read from here by '
    '`input_examples.read_example_file`.')

EXAMPLE_INDICES_HELP_STRING = (
    '1-D list of example indices in file.  This script will create one figure '
    'for each example.')

NORMALIZATION_FILE_HELP_STRING = (
    'Path to normalization file.  Will be read by `deep_learning_utils.'
    'read_normalization_params_from_file` and used to normalize radar images.')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_FILE_ARG_NAME, type=str, required=True,
    help=EXAMPLE_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_INDICES_ARG_NAME, type=int, nargs='+', required=True,
    help=EXAMPLE_INDICES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NORMALIZATION_FILE_ARG_NAME, type=str, required=True,
    help=NORMALIZATION_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _plot_one_example(
        orig_radar_matrix, translated_radar_matrix, rotated_radar_matrix,
        noised_radar_matrix, output_dir_name, full_storm_id_string,
        storm_time_unix_sec):
    """Plots original and augmented radar images for one example.

    M = number of rows in grid
    N = number of columns in grid

    :param orig_radar_matrix: M-by-N-by-1-by-1 numpy array with original values.
    :param translated_radar_matrix: Same but with translated values.
    :param rotated_radar_matrix: Same but with rotated values.
    :param noised_radar_matrix: Same but with noised values.
    :param output_dir_name: Name of output directory (figure will be saved
        here).
    :param full_storm_id_string: Storm ID.
    :param storm_time_unix_sec: Storm time.
    """

    dummy_heights_m_agl = numpy.array([1000, 2000, 3000, 4000], dtype=int)
    concat_radar_matrix = numpy.concatenate((
        orig_radar_matrix, translated_radar_matrix, rotated_radar_matrix,
        noised_radar_matrix
    ), axis=-2)

    training_option_dict = {
        trainval_io.SOUNDING_FIELDS_KEY: None,
        trainval_io.RADAR_FIELDS_KEY: [RADAR_FIELD_NAME],
        trainval_io.RADAR_HEIGHTS_KEY: dummy_heights_m_agl
    }

    model_metadata_dict = {cnn.TRAINING_OPTION_DICT_KEY: training_option_dict}

    handle_dict = plot_examples.plot_one_example(
        list_of_predictor_matrices=[concat_radar_matrix],
        model_metadata_dict=model_metadata_dict,
        pmm_flag=True, plot_sounding=False, allow_whitespace=True,
        plot_panel_names=False, add_titles=False, label_colour_bars=True,
        num_panel_rows=2)

    figure_object = handle_dict[plot_examples.RADAR_FIGURES_KEY][0]
    axes_object_matrix = handle_dict[plot_examples.RADAR_AXES_KEY]

    axes_object_matrix[0, 0].set_title('(a) Original', fontsize=TITLE_FONT_SIZE)
    axes_object_matrix[0, 1].set_title(
        '(b) Translated', fontsize=TITLE_FONT_SIZE
    )
    axes_object_matrix[1, 0].set_title('(c) Rotated', fontsize=TITLE_FONT_SIZE)
    axes_object_matrix[1, 1].set_title('(d) Noised', fontsize=TITLE_FONT_SIZE)

    output_file_name = '{0:s}/storm={1:s}_time={2:s}'.format(
        output_dir_name, full_storm_id_string.replace('_', '-'),
        time_conversion.unix_sec_to_string(
            storm_time_unix_sec, FILE_NAME_TIME_FORMAT)
    )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(example_file_name, example_indices, normalization_file_name,
         output_dir_name):
    """Plots data augmentation.

    This is effectively the main method.

    :param example_file_name: See documentation at top of file.
    :param example_indices: Same.
    :param normalization_file_name: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    print('Reading data from: "{0:s}"...'.format(example_file_name))
    example_dict = input_examples.read_example_file(
        netcdf_file_name=example_file_name,
        read_all_target_vars=True, include_soundings=False,
        radar_field_names_to_keep=[RADAR_FIELD_NAME],
        radar_heights_to_keep_m_agl=numpy.array([RADAR_HEIGHT_M_AGL], dtype=int)
    )

    if input_examples.REFL_IMAGE_MATRIX_KEY in example_dict:
        orig_radar_matrix = example_dict[input_examples.REFL_IMAGE_MATRIX_KEY]
    else:
        orig_radar_matrix = example_dict[input_examples.RADAR_IMAGE_MATRIX_KEY]

    num_examples = orig_radar_matrix.shape[0]
    error_checking.assert_is_geq_numpy_array(example_indices, 0)
    error_checking.assert_is_less_than_numpy_array(
        example_indices, num_examples)

    orig_radar_matrix = orig_radar_matrix[example_indices, ...]
    full_storm_id_strings = [
        example_dict[input_examples.FULL_IDS_KEY][k] for k in example_indices
    ]
    storm_times_unix_sec = example_dict[input_examples.STORM_TIMES_KEY][
        example_indices]

    orig_radar_matrix = dl_utils.normalize_radar_images(
        radar_image_matrix=orig_radar_matrix, field_names=[RADAR_FIELD_NAME],
        normalization_type_string=NORMALIZATION_TYPE_STRING,
        normalization_param_file_name=normalization_file_name)

    augmented_radar_matrix = trainval_io._augment_radar_images(
        list_of_predictor_matrices=[orig_radar_matrix], target_array=None,
        x_translations_pixels=X_TRANSLATIONS_PX,
        y_translations_pixels=Y_TRANSLATIONS_PX,
        ccw_rotation_angles_deg=CCW_ROTATION_ANGLES_DEG,
        noise_standard_deviation=NOISE_STANDARD_DEVIATION,
        num_noisings=1, flip_in_x=False, flip_in_y=False
    )[0][0]

    augmented_radar_matrix = augmented_radar_matrix[num_examples:, ...]
    translated_radar_matrix = augmented_radar_matrix[:num_examples, ...]
    augmented_radar_matrix = augmented_radar_matrix[num_examples:, ...]
    rotated_radar_matrix = augmented_radar_matrix[:num_examples, ...]
    noised_radar_matrix = augmented_radar_matrix[num_examples:, ...]

    for i in range(num_examples):
        _plot_one_example(
            orig_radar_matrix=orig_radar_matrix[i, ...],
            translated_radar_matrix=translated_radar_matrix[i, ...],
            rotated_radar_matrix=rotated_radar_matrix[i, ...],
            noised_radar_matrix=noised_radar_matrix[i, ...],
            output_dir_name=output_dir_name,
            full_storm_id_string=full_storm_id_strings[i],
            storm_time_unix_sec=storm_times_unix_sec[i]
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        example_file_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_FILE_ARG_NAME),
        example_indices=numpy.array(
            getattr(INPUT_ARG_OBJECT, EXAMPLE_INDICES_ARG_NAME), dtype=int
        ),
        normalization_file_name=getattr(
            INPUT_ARG_OBJECT, NORMALIZATION_FILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
