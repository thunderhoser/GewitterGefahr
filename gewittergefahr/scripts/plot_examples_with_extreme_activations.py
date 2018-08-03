"""Plots examples (storm objects) with extremely high or low CNN activations.

CNN = convolutional neural network
"""

import os.path
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import storm_images
from gewittergefahr.deep_learning import model_activation
from gewittergefahr.deep_learning import deployment_io
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.plotting import radar_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

LARGE_INTEGER = int(1e10)
TIME_FORMAT = '%Y-%m-%d-%H%M%S'

NUM_PANEL_ROWS = 3
DOTS_PER_INCH = 600

FIELD_NAMES_2D_KEY = 'field_name_by_pair'
HEIGHTS_2D_KEY = 'height_by_pair_m_asl'
FIELD_NAMES_3D_KEY = 'radar_field_names'
HEIGHTS_3D_KEY = 'radar_heights_m_asl'
STORM_IDS_KEY = 'storm_ids'
STORM_TIMES_KEY = 'storm_times_unix_sec'
STORM_ACTIVATIONS_KEY = 'storm_activations'
RADAR_IMAGE_MATRIX_KEY = 'radar_image_matrix'
SOUNDING_MATRIX_KEY = 'sounding_matrix'

ACTIVATION_FILE_ARG_NAME = 'input_activation_file_name'
NUM_HIGH_ARG_NAME = 'num_high_activation_examples'
NUM_LOW_ARG_NAME = 'num_low_activation_examples'
RADAR_IMAGE_DIR_ARG_NAME = 'input_radar_image_dir_name'
SOUNDING_DIR_ARG_NAME = 'input_sounding_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

ACTIVATION_FILE_HELP_STRING = (
    'Path to input file, containing activation of a single model component for '
    'each example.  This file will be read by `model_activation.read_file`.  If'
    ' the file contains activations for more than one model component, this '
    'script will error out.')
NUM_HIGH_HELP_STRING = (
    'Number of high-activation examples (storm objects) to plot.  The examples '
    'with the `{0:s}` highest activations will be plotted.'
).format(NUM_HIGH_ARG_NAME)
NUM_LOW_HELP_STRING = (
    'Number of low-activation examples (storm objects) to plot.  The examples '
    'with the `{0:s}` lowest activations will be plotted.'
).format(NUM_LOW_ARG_NAME)
RADAR_IMAGE_DIR_HELP_STRING = (
    'Name of top-level directory with storm-centered radar images.  Files '
    'therein will be found by `storm_images.find_storm_image_file` and read by '
    '`storm_images.read_storm_images`.')
SOUNDING_DIR_HELP_STRING = (
    'Name of top-level directory with storm-centered soundings.  Files therein '
    'will be found by `soundings_only.find_sounding_file` and read by '
    '`soundings_only.read_soundings`.')
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.')

DEFAULT_TOP_RADAR_IMAGE_DIR_NAME = (
    '/condo/swatcommon/common/gridrad_final/myrorss_format/tracks/reanalyzed/'
    'storm_images_with_rdp')
DEFAULT_TOP_SOUNDING_DIR_NAME = (
    '/condo/swatcommon/common/gridrad_final/myrorss_format/tracks/reanalyzed/'
    'soundings')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + ACTIVATION_FILE_ARG_NAME, type=str, required=True,
    help=ACTIVATION_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_HIGH_ARG_NAME, type=int, required=False, default=100,
    help=NUM_HIGH_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_LOW_ARG_NAME, type=int, required=False, default=100,
    help=NUM_LOW_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_IMAGE_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_TOP_RADAR_IMAGE_DIR_NAME,
    help=RADAR_IMAGE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SOUNDING_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_TOP_SOUNDING_DIR_NAME, help=SOUNDING_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _read_storm_objects(
        top_radar_image_dir_name, top_sounding_dir_name,
        activation_metadata_dict, storm_ids, storm_times_unix_sec,
        storm_activations):
    """Reads radar and sounding data for each storm object.

    E = number of examples (storm objects)
    M = number of rows in each storm-centered grid
    N = number of columns in each storm-centered grid
    F = number of radar fields
    H = number of radar heights
    C = number of field/height pairs

    :param top_radar_image_dir_name: See documentation at top of file.
    :param top_sounding_dir_name: Same.
    :param activation_metadata_dict: Dictionary returned by
        `model_activation.read_file`.
    :param storm_ids: length-E list of storm IDs.
    :param storm_times_unix_sec: length-E numpy array of storm times.
    :param storm_activations: length-E numpy array of storm activations.
    :return: storm_object_dict: Dictionary with the following keys.
    storm_object_dict['radar_field_names']: length-F list with names of radar
        fields.  If radar images are 2-D, this is `None`.
    storm_object_dict['radar_heights_m_asl']: length-H numpy array of radar
        heights (metres above sea level).  If radar images are 2-D, this is
        `None`.
    storm_object_dict['field_name_by_pair']: length-C list with names of radar
        fields.  If radar images are 3-D, this is `None`.
    storm_object_dict['height_by_pair_m_asl']: length-C numpy array of radar
        heights (metres above sea level).  If radar images are 3-D, this is
        `None`.
    storm_object_dict['storm_ids']: length-E list of storm IDs.
    storm_object_dict['storm_times_unix_sec']: length-E numpy array of storm
        times.
    storm_object_dict['radar_image_matrix']: numpy array
        (either E x M x N x C or E x M x N x H x F) of radar values.
    storm_object_dict['sounding_matrix']: 3-D numpy array of sounding values.
    """

    # Convert input args.
    storm_spc_date_strings_numpy = numpy.array(
        [time_conversion.time_to_spc_date_string(t)
         for t in storm_times_unix_sec],
        dtype=object)
    unique_spc_date_strings_numpy = numpy.unique(storm_spc_date_strings_numpy)

    # Read metadata for machine-learning model.
    model_file_name = activation_metadata_dict[
        model_activation.MODEL_FILE_NAME_KEY]
    model_metadata_file_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0])

    print 'Reading metadata from: "{0:s}"...'.format(model_metadata_file_name)
    model_metadata_dict = cnn.read_model_metadata(model_metadata_file_name)

    # Find files with storm-centered radar images.
    num_radar_dimensions = len(
        model_metadata_dict[cnn.TRAINING_FILE_NAMES_KEY].shape)
    num_spc_dates = len(unique_spc_date_strings_numpy)

    radar_image_matrix = None
    sounding_matrix = None
    radar_field_names = None
    radar_heights_m_asl = None
    field_name_by_pair = None
    height_by_pair_m_asl = None

    read_soundings = (
        model_metadata_dict[cnn.SOUNDING_FIELD_NAMES_KEY] is not None)

    for i in range(num_spc_dates):
        if num_radar_dimensions == 2:
            this_radar_file_name_matrix = trainval_io.find_radar_files_2d(
                top_directory_name=top_radar_image_dir_name,
                radar_source=model_metadata_dict[cnn.RADAR_SOURCE_KEY],
                radar_field_names=model_metadata_dict[
                    cnn.RADAR_FIELD_NAMES_KEY],
                first_file_time_unix_sec=
                time_conversion.spc_date_string_to_unix_sec(
                    unique_spc_date_strings_numpy[i]),
                last_file_time_unix_sec=
                time_conversion.spc_date_string_to_unix_sec(
                    unique_spc_date_strings_numpy[i]),
                one_file_per_time_step=False, shuffle_times=False,
                radar_heights_m_asl=model_metadata_dict[cnn.RADAR_HEIGHTS_KEY],
                reflectivity_heights_m_asl=model_metadata_dict[
                    cnn.REFLECTIVITY_HEIGHTS_KEY])[0]
            print MINOR_SEPARATOR_STRING

            if i == 0:
                field_name_by_pair = [
                    storm_images.image_file_name_to_field(f) for f in
                    this_radar_file_name_matrix[0, 0, 0, :]
                ]
                height_by_pair_m_asl = numpy.array([
                    storm_images.image_file_name_to_height(f) for f in
                    this_radar_file_name_matrix[0, 0, 0, :]
                ], dtype=int)

            example_dict = deployment_io.create_storm_images_2d(
                radar_file_name_matrix=this_radar_file_name_matrix,
                num_examples_per_file_time=LARGE_INTEGER, return_target=False,
                target_name=model_metadata_dict[cnn.TARGET_NAME_KEY],
                radar_normalization_dict=model_metadata_dict[
                    cnn.RADAR_NORMALIZATION_DICT_KEY],
                sounding_field_names=model_metadata_dict[
                    cnn.SOUNDING_FIELD_NAMES_KEY],
                top_sounding_dir_name=top_sounding_dir_name,
                sounding_lag_time_for_convective_contamination_sec=
                model_metadata_dict[cnn.SOUNDING_LAG_TIME_KEY],
                sounding_normalization_dict=model_metadata_dict[
                    cnn.SOUNDING_NORMALIZATION_DICT_KEY])

        else:
            this_radar_file_name_matrix = trainval_io.find_radar_files_3d(
                top_directory_name=top_radar_image_dir_name,
                radar_source=model_metadata_dict[cnn.RADAR_SOURCE_KEY],
                radar_field_names=model_metadata_dict[
                    cnn.RADAR_FIELD_NAMES_KEY],
                radar_heights_m_asl=model_metadata_dict[cnn.RADAR_HEIGHTS_KEY],
                first_file_time_unix_sec=
                time_conversion.spc_date_string_to_unix_sec(
                    unique_spc_date_strings_numpy[i]),
                last_file_time_unix_sec=
                time_conversion.spc_date_string_to_unix_sec(
                    unique_spc_date_strings_numpy[i]),
                one_file_per_time_step=False, shuffle_times=False)[0]
            print MINOR_SEPARATOR_STRING

            if i == 0:
                radar_field_names = [
                    storm_images.image_file_name_to_field(f) for f in
                    this_radar_file_name_matrix[0, 0, 0, 0, :]
                ]
                radar_heights_m_asl = numpy.array([
                    storm_images.image_file_name_to_height(f) for f in
                    this_radar_file_name_matrix[0, 0, 0, :, 0]
                ], dtype=int)

            example_dict = deployment_io.create_storm_images_3d(
                radar_file_name_matrix=this_radar_file_name_matrix,
                num_examples_per_file_time=LARGE_INTEGER, return_target=False,
                target_name=model_metadata_dict[cnn.TARGET_NAME_KEY],
                radar_normalization_dict=model_metadata_dict[
                    cnn.RADAR_NORMALIZATION_DICT_KEY],
                refl_masking_threshold_dbz=model_metadata_dict[
                    cnn.REFL_MASKING_THRESHOLD_KEY],
                return_rotation_divergence_product=False,
                sounding_field_names=model_metadata_dict[
                    cnn.SOUNDING_FIELD_NAMES_KEY],
                top_sounding_dir_name=top_sounding_dir_name,
                sounding_lag_time_for_convective_contamination_sec=
                model_metadata_dict[cnn.SOUNDING_LAG_TIME_KEY],
                sounding_normalization_dict=model_metadata_dict[
                    cnn.SOUNDING_NORMALIZATION_DICT_KEY])

        these_indices = numpy.where(
            storm_spc_date_strings_numpy == unique_spc_date_strings_numpy[i])[0]
        these_indices = storm_images.find_storm_objects(
            all_storm_ids=example_dict[deployment_io.STORM_IDS_KEY],
            all_valid_times_unix_sec=example_dict[
                deployment_io.STORM_TIMES_KEY],
            storm_ids_to_keep=storm_ids[these_indices],
            valid_times_to_keep_unix_sec=storm_times_unix_sec[these_indices])

        if radar_image_matrix is None:
            radar_image_matrix = example_dict[
                deployment_io.RADAR_IMAGE_MATRIX_KEY][these_indices, ...] + 0.
            if read_soundings:
                sounding_matrix = example_dict[
                    deployment_io.SOUNDING_MATRIX_KEY][these_indices, ...] + 0.
        else:
            radar_image_matrix = numpy.concatenate(
                (radar_image_matrix,
                 example_dict[deployment_io.RADAR_IMAGE_MATRIX_KEY][
                     these_indices, ...]),
                axis=0)
            if read_soundings:
                sounding_matrix = numpy.concatenate(
                    (sounding_matrix,
                     example_dict[deployment_io.SOUNDING_MATRIX_KEY][
                         these_indices, ...]),
                    axis=0)

        if i != num_spc_dates - 1:
            print MINOR_SEPARATOR_STRING

    return {
        FIELD_NAMES_2D_KEY: field_name_by_pair,
        HEIGHTS_2D_KEY: height_by_pair_m_asl,
        FIELD_NAMES_3D_KEY: radar_field_names,
        HEIGHTS_3D_KEY: radar_heights_m_asl,
        STORM_IDS_KEY: storm_ids,
        STORM_TIMES_KEY: storm_times_unix_sec,
        STORM_ACTIVATIONS_KEY: storm_activations,
        RADAR_IMAGE_MATRIX_KEY: radar_image_matrix,
        SOUNDING_MATRIX_KEY: sounding_matrix
    }


def _plot_storm_objects(storm_object_dict, output_dir_name):
    """Plots radar and sounding data for each storm object.

    :param storm_object_dict: Dictionary created by `_read_storm_objects`.
    :param output_dir_name: Name of output directory.  Figures will be saved
        here.
    """

    field_name_by_pair = storm_object_dict[FIELD_NAMES_2D_KEY]
    height_by_pair_m_asl = storm_object_dict[HEIGHTS_2D_KEY]
    radar_field_names = storm_object_dict[FIELD_NAMES_3D_KEY]
    radar_heights_m_asl = storm_object_dict[HEIGHTS_3D_KEY]
    storm_ids = storm_object_dict[STORM_IDS_KEY]
    storm_times_unix_sec = storm_object_dict[STORM_TIMES_KEY]
    storm_activations = storm_object_dict[STORM_ACTIVATIONS_KEY]
    radar_image_matrix = storm_object_dict[RADAR_IMAGE_MATRIX_KEY]

    num_radar_dimensions = 2 + int(radar_field_names is not None)
    num_storm_objects = len(storm_ids)

    for i in range(num_storm_objects):
        this_time_string = time_conversion.unix_sec_to_string(
            storm_times_unix_sec[i], TIME_FORMAT)
        this_title_string = (
            'Storm "{0:s}" at {1:s} (activation = {2:.2e})'
        ).format(storm_ids[i], this_time_string, storm_activations[i])
        this_base_file_name = '{0:s}/storm={1:s}_{2:s}'.format(
            output_dir_name, storm_ids[i].replace('_', '-'), this_time_string)

        if num_radar_dimensions == 2:
            j_max = 1
        else:
            j_max = len(radar_field_names)

        for j in range(j_max):
            if num_radar_dimensions == 2:
                radar_plotting.plot_many_2d_grids_without_coords(
                    field_matrix=radar_image_matrix[i, ...],
                    field_name_by_pair=field_name_by_pair,
                    height_by_pair_m_asl=height_by_pair_m_asl,
                    num_panel_rows=NUM_PANEL_ROWS)

                this_figure_file_name = '{0:s}.jpg'.format(this_base_file_name)
            else:
                radar_plotting.plot_3d_grid_without_coords(
                    field_matrix=radar_image_matrix[i, ..., j],
                    field_name=radar_field_names[j],
                    grid_point_heights_m_asl=radar_heights_m_asl,
                    num_panel_rows=NUM_PANEL_ROWS)

                this_figure_file_name = '{0:s}_{1:s}.jpg'.format(
                    this_base_file_name, radar_field_names[j].replace('_', '-'))

            pyplot.title(this_title_string)
            print 'Saving figure to: "{0:s}"...'.format(this_figure_file_name)
            pyplot.savefig(this_figure_file_name, dpi=DOTS_PER_INCH)
            pyplot.close()


def _run(
        input_activation_file_name, num_high_activation_examples,
        num_low_activation_examples, top_radar_image_dir_name,
        top_sounding_dir_name, output_dir_name):
    """Plots examples (storm objects) with extremely high/low CNN activations.

    This is effectively the main method.

    :param input_activation_file_name: See documentation at top of file.
    :param num_high_activation_examples: Same.
    :param num_low_activation_examples: Same.
    :param top_radar_image_dir_name: Same.
    :param top_sounding_dir_name: Same.
    :param output_dir_name: Same.
    """

    # Check input args.
    error_checking.assert_is_geq(num_high_activation_examples, 0)
    error_checking.assert_is_geq(num_low_activation_examples, 0)
    error_checking.assert_is_greater(
        num_high_activation_examples + num_low_activation_examples, 0)

    # Read activations.
    print 'Reading activations from: "{0:s}"...'.format(
        input_activation_file_name)
    activation_matrix, activation_metadata_dict = model_activation.read_file(
        input_activation_file_name)

    num_model_components = activation_matrix.shape[1]
    if num_model_components > 1:
        error_string = (
            'The file should contain activations for only one model component, '
            'not {0:d}.'
        ).format(num_model_components)
        raise ValueError(error_string)

    storm_activations = activation_matrix[:, 0]
    storm_ids = activation_metadata_dict[model_activation.STORM_IDS_KEY]
    storm_times_unix_sec = activation_metadata_dict[
        model_activation.STORM_TIMES_KEY]

    num_storm_objects = len(storm_ids)
    error_checking.assert_is_leq(
        num_high_activation_examples + num_low_activation_examples,
        num_storm_objects)

    # Find high- and low-activation examples.
    sort_indices = numpy.argsort(storm_activations)

    if num_high_activation_examples > 0:
        print (
            'Finding the {0:d} examples (storm objects) with highest '
            'activation...'
        ).format(num_high_activation_examples)
        high_indices = sort_indices[::-1][:num_high_activation_examples]

        for this_index in high_indices:
            print (
                'Storm ID = "{0:s}" ... time = {1:s} ... activation = {2:.2e}'
            ).format(storm_ids[this_index],
                     time_conversion.unix_sec_to_string(
                         storm_times_unix_sec[this_index], TIME_FORMAT),
                     storm_activations[this_index])
    else:
        high_indices = numpy.array([], dtype=int)

    if num_low_activation_examples > 0:
        print (
            'Finding the {0:d} examples (storm objects) with lowest '
            'activation...'
        ).format(num_low_activation_examples)
        low_indices = sort_indices[:num_low_activation_examples]

        for this_index in low_indices:
            print (
                'Storm ID = "{0:s}" ... time = {1:s} ... activation = {2:.2e}'
            ).format(storm_ids[this_index],
                     time_conversion.unix_sec_to_string(
                         storm_times_unix_sec[this_index], TIME_FORMAT),
                     storm_activations[this_index])
    else:
        low_indices = numpy.array([], dtype=int)

    # Read data for low- and high-activation examples.
    hilo_indices = numpy.concatenate((high_indices, low_indices))
    storm_activations = storm_activations[hilo_indices]
    storm_ids = [storm_ids[k] for k in hilo_indices]
    storm_times_unix_sec = storm_times_unix_sec[hilo_indices]

    print SEPARATOR_STRING
    storm_object_dict = _read_storm_objects(
        top_radar_image_dir_name=top_radar_image_dir_name,
        top_sounding_dir_name=top_sounding_dir_name,
        activation_metadata_dict=activation_metadata_dict,
        storm_ids=storm_ids, storm_times_unix_sec=storm_times_unix_sec,
        storm_activations=storm_activations)
    print SEPARATOR_STRING

    _plot_storm_objects(
        storm_object_dict=storm_object_dict, output_dir_name=output_dir_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_activation_file_name=getattr(
            INPUT_ARG_OBJECT, ACTIVATION_FILE_ARG_NAME),
        num_high_activation_examples=getattr(
            INPUT_ARG_OBJECT, NUM_HIGH_ARG_NAME),
        num_low_activation_examples=getattr(INPUT_ARG_OBJECT, NUM_LOW_ARG_NAME),
        top_radar_image_dir_name=getattr(
            INPUT_ARG_OBJECT, RADAR_IMAGE_DIR_ARG_NAME),
        top_sounding_dir_name=getattr(INPUT_ARG_OBJECT, SOUNDING_DIR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME))
