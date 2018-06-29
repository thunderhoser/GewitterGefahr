"""Extracts features (output of the last "Flatten" layer) from a CNN.

CNN = convolutional neural network
"""

import os.path
import argparse
import numpy
from keras import backend as K
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import training_validation_io as trainval_io

# TODO(thunderhoser): Allow generators to stop when they run out of files,
# rather than looping through files again.

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

INPUT_TIME_FORMAT = '%Y-%m-%d-%H%M%S'
NUM_VALUES_PER_BATCH = int(1e8)

MODEL_FILE_ARG_NAME = 'input_model_file_name'
RADAR_DIRECTORY_ARG_NAME = 'input_storm_radar_image_dir_name'
SOUNDING_DIRECTORY_ARG_NAME = 'input_sounding_dir_name'
TARGET_DIRECTORY_ARG_NAME = 'input_target_dir_name'
FIRST_STORM_TIME_ARG_NAME = 'first_storm_time_string'
LAST_STORM_TIME_ARG_NAME = 'last_storm_time_string'
ONE_FILE_PER_TIME_STEP_ARG_NAME = 'one_file_per_time_step'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
OUTPUT_FILE_ARG_NAME = 'output_netcdf_file_name'

MODEL_FILE_HELP_STRING = (
    'Path to input file (readable by `cnn.read_model`), containing the trained '
    'CNN.')
RADAR_DIRECTORY_HELP_STRING = (
    'Name of top-level directory with storm-centered radar images.  Files '
    'therein will be found by `training_validation_io.find_radar_files_2d` or '
    '`training_validation_io.find_radar_files_3d`.')
SOUNDING_DIRECTORY_HELP_STRING = (
    'Name of top-level directory with storm-centered soundings.  Files therein '
    'will be found by `training_validation_io.find_sounding_files`.')
TARGET_DIRECTORY_HELP_STRING = (
    'Name of top-level directory with labels (target values).  Files therein '
    'will be found by `labels.find_label_file`.')
STORM_TIME_HELP_STRING = (
    'Storm time (format "yyyy-mm-dd-HHMMSS").  `{0:s}` storm objects will be '
    'drawn randomly from `{1:s}`...`{2:s}`.  The features and target value for '
    'each storm object will be written to `{3:s}`.'
).format(NUM_EXAMPLES_ARG_NAME, FIRST_STORM_TIME_ARG_NAME,
         LAST_STORM_TIME_ARG_NAME, OUTPUT_FILE_ARG_NAME)
ONE_FILE_PER_TIME_STEP_HELP_STRING = (
    'Boolean flag.  If 1 (0), this script will read data from one set of files '
    'per time step (SPC date).')
NUM_EXAMPLES_HELP_STRING = (
    'Number of storm objects.  See discussion for `{0:s}` and `{1:s}`.'
).format(FIRST_STORM_TIME_ARG_NAME, LAST_STORM_TIME_ARG_NAME)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file (will be written by `cnn.write_features`).')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_DIRECTORY_ARG_NAME, type=str, required=True,
    help=RADAR_DIRECTORY_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SOUNDING_DIRECTORY_ARG_NAME, type=str, required=False,
    default='', help=SOUNDING_DIRECTORY_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_DIRECTORY_ARG_NAME, type=str, required=True,
    help=TARGET_DIRECTORY_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_STORM_TIME_ARG_NAME, type=str, required=True,
    help=STORM_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_STORM_TIME_ARG_NAME, type=str, required=True,
    help=STORM_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + ONE_FILE_PER_TIME_STEP_ARG_NAME, type=int, required=False, default=0,
    help=ONE_FILE_PER_TIME_STEP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=True,
    help=NUM_EXAMPLES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _extract_2d_cnn_features(
        model_object, top_storm_radar_image_dir_name, top_sounding_dir_name,
        top_target_dir_name, first_storm_time_unix_sec,
        last_storm_time_unix_sec, one_file_per_time_step,
        num_examples_per_batch, num_examples_total, output_netcdf_file_name,
        model_metadata_dict):
    """Extracts features (output of the last "Flatten" layer) from a 2-D CNN.

    :param model_object: Trained model (instance of `keras.models.Sequential`).
    :param top_storm_radar_image_dir_name: See documentation at top of file.
    :param top_sounding_dir_name: Same.
    :param top_target_dir_name: Same.
    :param first_storm_time_unix_sec: Same.
    :param last_storm_time_unix_sec: Same.
    :param one_file_per_time_step: Same.
    :param num_examples_per_batch: Number of examples (storm objects) per batch.
        This is the number of storm objects that will be written to
        `output_netcdf_file_name`.  This method will continue until
        `num_examples_total` storm objects have been written.
    :param num_examples_total: See above.
    :param output_netcdf_file_name: Path to output file.
    :param model_metadata_dict: Dictionary created by `cnn.read_model_metadata`.
    """

    radar_file_name_matrix, _ = trainval_io.find_radar_files_2d(
        top_directory_name=top_storm_radar_image_dir_name,
        radar_source=model_metadata_dict[cnn.RADAR_SOURCE_KEY],
        radar_field_names=model_metadata_dict[cnn.RADAR_FIELD_NAMES_KEY],
        first_file_time_unix_sec=first_storm_time_unix_sec,
        last_file_time_unix_sec=last_storm_time_unix_sec,
        one_file_per_time_step=one_file_per_time_step,
        radar_heights_m_asl=model_metadata_dict[cnn.RADAR_HEIGHTS_KEY],
        reflectivity_heights_m_asl=model_metadata_dict[
            cnn.REFLECTIVITY_HEIGHTS_KEY])
    print SEPARATOR_STRING

    generator_object = trainval_io.storm_image_generator_2d(
        radar_file_name_matrix=radar_file_name_matrix,
        top_target_directory_name=top_target_dir_name,
        num_examples_per_batch=num_examples_per_batch,
        num_examples_per_file_time=num_examples_per_batch,
        target_name=model_metadata_dict[cnn.TARGET_NAME_KEY],
        binarize_target=model_metadata_dict[cnn.BINARIZE_TARGET_KEY],
        radar_normalization_dict=model_metadata_dict[
            cnn.RADAR_NORMALIZATION_DICT_KEY],
        sampling_fraction_by_class_dict=model_metadata_dict[
            cnn.TRAINING_FRACTION_BY_CLASS_KEY],
        sounding_field_names=model_metadata_dict[
            cnn.SOUNDING_FIELD_NAMES_KEY],
        top_sounding_dir_name=top_sounding_dir_name,
        sounding_lag_time_for_convective_contamination_sec=
        model_metadata_dict[cnn.SOUNDING_LAG_TIME_KEY],
        sounding_normalization_dict=model_metadata_dict[
            cnn.SOUNDING_NORMALIZATION_DICT_KEY])

    num_examples_read = 0
    use_soundings = model_metadata_dict[
        cnn.SOUNDING_FIELD_NAMES_KEY] is not None

    while num_examples_read < num_examples_total:
        print (
            'Have extracted features for {0:d} of {1:d} storm objects...'
        ).format(num_examples_read, num_examples_total)
        print MINOR_SEPARATOR_STRING

        if use_soundings:
            this_list_of_predictor_matrices, this_target_matrix = next(
                generator_object)
            this_radar_image_matrix = this_list_of_predictor_matrices[0]
            this_sounding_matrix = this_list_of_predictor_matrices[1]
        else:
            this_radar_image_matrix, this_target_matrix = next(generator_object)
            this_sounding_matrix = None

        print SEPARATOR_STRING

        num_examples_in_this_batch = min(
            [num_examples_per_batch, num_examples_total - num_examples_read])
        this_radar_image_matrix = this_radar_image_matrix[
            :num_examples_in_this_batch, ...]
        these_target_values = numpy.argmax(
            this_target_matrix[:num_examples_in_this_batch, ...], axis=1)
        if use_soundings:
            this_sounding_matrix = this_sounding_matrix[
                :num_examples_in_this_batch, ...]

        this_feature_matrix = cnn.apply_2d_cnn(
            model_object=model_object,
            radar_image_matrix=this_radar_image_matrix,
            sounding_matrix=this_sounding_matrix, return_features=True)

        print 'Writing features and target values to: "{0:s}"...'.format(
            output_netcdf_file_name)
        cnn.write_features(
            netcdf_file_name=output_netcdf_file_name,
            feature_matrix=this_feature_matrix,
            target_values=these_target_values,
            append_to_file=num_examples_read > 0)

        num_examples_read += num_examples_in_this_batch


def _extract_3d_cnn_features(
        model_object, top_storm_radar_image_dir_name, top_sounding_dir_name,
        top_target_dir_name, first_storm_time_unix_sec,
        last_storm_time_unix_sec, one_file_per_time_step,
        num_examples_per_batch, num_examples_total, output_netcdf_file_name,
        model_metadata_dict):
    """Extracts features (output of the last "Flatten" layer) from a 3-D CNN.

    :param model_object: Trained model (instance of `keras.models.Sequential`).
    :param top_storm_radar_image_dir_name: See documentation at top of file.
    :param top_sounding_dir_name: Same.
    :param top_target_dir_name: Same.
    :param first_storm_time_unix_sec: Same.
    :param last_storm_time_unix_sec: Same.
    :param one_file_per_time_step: Same.
    :param num_examples_per_batch: See doc for `_extract_2d_cnn_features`.
    :param num_examples_total: Same.
    :param output_netcdf_file_name: See documentation at top of file.
    :param model_metadata_dict: Dictionary created by `cnn.read_model_metadata`.
    """

    radar_file_name_matrix, _ = trainval_io.find_radar_files_3d(
        top_directory_name=top_storm_radar_image_dir_name,
        radar_source=model_metadata_dict[cnn.RADAR_SOURCE_KEY],
        radar_field_names=model_metadata_dict[cnn.RADAR_FIELD_NAMES_KEY],
        radar_heights_m_asl=model_metadata_dict[cnn.RADAR_HEIGHTS_KEY],
        first_file_time_unix_sec=first_storm_time_unix_sec,
        last_file_time_unix_sec=last_storm_time_unix_sec,
        one_file_per_time_step=one_file_per_time_step)
    print SEPARATOR_STRING
    print num_examples_per_batch

    generator_object = trainval_io.storm_image_generator_3d(
        radar_file_name_matrix=radar_file_name_matrix,
        top_target_directory_name=top_target_dir_name,
        num_examples_per_batch=num_examples_per_batch,
        num_examples_per_file_time=num_examples_per_batch,
        target_name=model_metadata_dict[cnn.TARGET_NAME_KEY],
        binarize_target=model_metadata_dict[cnn.BINARIZE_TARGET_KEY],
        radar_normalization_dict=model_metadata_dict[
            cnn.RADAR_NORMALIZATION_DICT_KEY],
        sampling_fraction_by_class_dict=model_metadata_dict[
            cnn.TRAINING_FRACTION_BY_CLASS_KEY],
        sounding_field_names=model_metadata_dict[
            cnn.SOUNDING_FIELD_NAMES_KEY],
        top_sounding_dir_name=top_sounding_dir_name,
        sounding_lag_time_for_convective_contamination_sec=
        model_metadata_dict[cnn.SOUNDING_LAG_TIME_KEY],
        sounding_normalization_dict=model_metadata_dict[
            cnn.SOUNDING_NORMALIZATION_DICT_KEY])

    num_examples_read = 0
    use_soundings = model_metadata_dict[
        cnn.SOUNDING_FIELD_NAMES_KEY] is not None

    while num_examples_read < num_examples_total:
        print (
            'Have extracted features for {0:d} of {1:d} storm objects...'
        ).format(num_examples_read, num_examples_total)
        print MINOR_SEPARATOR_STRING

        if use_soundings:
            this_list_of_predictor_matrices, this_target_matrix = next(
                generator_object)
            this_radar_image_matrix = this_list_of_predictor_matrices[0]
            this_sounding_matrix = this_list_of_predictor_matrices[1]
        else:
            this_radar_image_matrix, this_target_matrix = next(generator_object)
            this_sounding_matrix = None

        print SEPARATOR_STRING

        num_examples_in_this_batch = min(
            [num_examples_per_batch, num_examples_total - num_examples_read])
        this_radar_image_matrix = this_radar_image_matrix[
            :num_examples_in_this_batch, ...]
        these_target_values = numpy.argmax(
            this_target_matrix[:num_examples_in_this_batch, ...], axis=1)
        if use_soundings:
            this_sounding_matrix = this_sounding_matrix[
                :num_examples_in_this_batch, ...]

        this_feature_matrix = cnn.apply_3d_cnn(
            model_object=model_object,
            radar_image_matrix=this_radar_image_matrix,
            sounding_matrix=this_sounding_matrix, return_features=True)

        print 'Writing features and target values to: "{0:s}"...'.format(
            output_netcdf_file_name)
        cnn.write_features(
            netcdf_file_name=output_netcdf_file_name,
            feature_matrix=this_feature_matrix,
            target_values=these_target_values,
            append_to_file=num_examples_read > 0)

        num_examples_read += num_examples_in_this_batch


def _extract_2d3d_cnn_features(
        model_object, top_storm_radar_image_dir_name, top_sounding_dir_name,
        top_target_dir_name, first_storm_time_unix_sec,
        last_storm_time_unix_sec, one_file_per_time_step,
        num_examples_per_batch, num_examples_total, output_netcdf_file_name,
        model_metadata_dict):
    """Extracts features (output of the last "Flatten" layer) from a 2D/3D CNN.

    :param model_object: Trained model (instance of `keras.models.Sequential`).
    :param top_storm_radar_image_dir_name: See documentation at top of file.
    :param top_sounding_dir_name: Same.
    :param top_target_dir_name: Same.
    :param first_storm_time_unix_sec: Same.
    :param last_storm_time_unix_sec: Same.
    :param one_file_per_time_step: Same.
    :param num_examples_per_batch: See doc for `_extract_2d_cnn_features`.
    :param num_examples_total: Same.
    :param output_netcdf_file_name: See documentation at top of file.
    :param model_metadata_dict: Dictionary created by `cnn.read_model_metadata`.
    """

    radar_file_name_matrix, _ = trainval_io.find_radar_files_2d(
        top_directory_name=top_storm_radar_image_dir_name,
        radar_source=model_metadata_dict[cnn.RADAR_SOURCE_KEY],
        radar_field_names=model_metadata_dict[cnn.RADAR_FIELD_NAMES_KEY],
        first_file_time_unix_sec=first_storm_time_unix_sec,
        last_file_time_unix_sec=last_storm_time_unix_sec,
        one_file_per_time_step=one_file_per_time_step,
        radar_heights_m_asl=model_metadata_dict[cnn.RADAR_HEIGHTS_KEY],
        reflectivity_heights_m_asl=model_metadata_dict[
            cnn.REFLECTIVITY_HEIGHTS_KEY])
    print SEPARATOR_STRING

    generator_object = trainval_io.storm_image_generator_2d3d_myrorss(
        radar_file_name_matrix=radar_file_name_matrix,
        top_target_directory_name=top_target_dir_name,
        num_examples_per_batch=num_examples_per_batch,
        num_examples_per_file_time=num_examples_per_batch,
        target_name=model_metadata_dict[cnn.TARGET_NAME_KEY],
        binarize_target=model_metadata_dict[cnn.BINARIZE_TARGET_KEY],
        radar_normalization_dict=model_metadata_dict[
            cnn.RADAR_NORMALIZATION_DICT_KEY],
        sampling_fraction_by_class_dict=model_metadata_dict[
            cnn.TRAINING_FRACTION_BY_CLASS_KEY],
        sounding_field_names=model_metadata_dict[
            cnn.SOUNDING_FIELD_NAMES_KEY],
        top_sounding_dir_name=top_sounding_dir_name,
        sounding_lag_time_for_convective_contamination_sec=
        model_metadata_dict[cnn.SOUNDING_LAG_TIME_KEY],
        sounding_normalization_dict=model_metadata_dict[
            cnn.SOUNDING_NORMALIZATION_DICT_KEY])

    num_examples_read = 0
    use_soundings = model_metadata_dict[
        cnn.SOUNDING_FIELD_NAMES_KEY] is not None

    while num_examples_read < num_examples_total:
        print (
            'Have extracted features for {0:d} of {1:d} storm objects...'
        ).format(num_examples_read, num_examples_total)
        print MINOR_SEPARATOR_STRING

        this_list_of_predictor_matrices, this_target_matrix = next(
            generator_object)
        print SEPARATOR_STRING

        num_examples_in_this_batch = min(
            [num_examples_per_batch, num_examples_total - num_examples_read])

        this_reflectivity_matrix_dbz = this_list_of_predictor_matrices[0][
            :num_examples_in_this_batch, ...]
        this_azimuthal_shear_matrix_s01 = this_list_of_predictor_matrices[
            1][:num_examples_in_this_batch, ...]
        these_target_values = numpy.argmax(
            this_target_matrix[:num_examples_in_this_batch, ...], axis=1)
        if use_soundings:
            this_sounding_matrix = this_list_of_predictor_matrices[
                2][:num_examples_in_this_batch, ...]
        else:
            this_sounding_matrix = None

        this_feature_matrix = cnn.apply_2d3d_cnn(
            model_object=model_object,
            reflectivity_image_matrix_dbz=this_reflectivity_matrix_dbz,
            azimuthal_shear_image_matrix_s01=this_azimuthal_shear_matrix_s01,
            sounding_matrix=this_sounding_matrix, return_features=True)

        print 'Writing features and target values to: "{0:s}"...'.format(
            output_netcdf_file_name)
        cnn.write_features(
            netcdf_file_name=output_netcdf_file_name,
            feature_matrix=this_feature_matrix,
            target_values=these_target_values,
            append_to_file=num_examples_read > 0)

        num_examples_read += num_examples_in_this_batch


def _extract_features(
        model_file_name, top_storm_radar_image_dir_name, top_sounding_dir_name,
        top_target_dir_name, first_storm_time_string, last_storm_time_string,
        one_file_per_time_step, num_examples, output_netcdf_file_name):
    """Extracts features (output of the last "Flatten" layer) from a CNN.

    :param model_file_name: See documentation at top of file.
    :param top_storm_radar_image_dir_name: Same.
    :param top_sounding_dir_name: Same.
    :param top_target_dir_name: Same.
    :param first_storm_time_string: Same.
    :param last_storm_time_string: Same.
    :param one_file_per_time_step: Same.
    :param num_examples: Same.
    :param output_netcdf_file_name: Same.
    """

    first_storm_time_unix_sec = time_conversion.string_to_unix_sec(
        first_storm_time_string, INPUT_TIME_FORMAT)
    last_storm_time_unix_sec = time_conversion.string_to_unix_sec(
        last_storm_time_string, INPUT_TIME_FORMAT)

    print 'Reading model from: "{0:s}"...'.format(model_file_name)
    model_object = cnn.read_model(model_file_name)

    intermediate_model_object = cnn.model_to_feature_generator(model_object)
    num_features = numpy.array(
        intermediate_model_object.layers[-1].output_shape)[-1]
    num_examples_per_batch = int(numpy.round(
        float(NUM_VALUES_PER_BATCH) / num_features))

    model_directory_name, _ = os.path.split(model_file_name)
    metadata_file_name = '{0:s}/model_metadata.p'.format(model_directory_name)
    print 'Reading metadata from: "{0:s}"...'.format(metadata_file_name)
    model_metadata_dict = cnn.read_model_metadata(metadata_file_name)

    if model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]:
        _extract_2d3d_cnn_features(
            model_object=model_object,
            top_storm_radar_image_dir_name=top_storm_radar_image_dir_name,
            top_sounding_dir_name=top_sounding_dir_name,
            top_target_dir_name=top_target_dir_name,
            first_storm_time_unix_sec=first_storm_time_unix_sec,
            last_storm_time_unix_sec=last_storm_time_unix_sec,
            one_file_per_time_step=one_file_per_time_step,
            num_examples_per_batch=num_examples_per_batch,
            num_examples_total=num_examples,
            output_netcdf_file_name=output_netcdf_file_name,
            model_metadata_dict=model_metadata_dict)
    else:
        num_radar_dimensions = len(
            model_metadata_dict[cnn.TRAINING_FILE_NAME_MATRIX_KEY].shape)

        if num_radar_dimensions == 2:
            _extract_2d_cnn_features(
                model_object=model_object,
                top_storm_radar_image_dir_name=top_storm_radar_image_dir_name,
                top_sounding_dir_name=top_sounding_dir_name,
                top_target_dir_name=top_target_dir_name,
                first_storm_time_unix_sec=first_storm_time_unix_sec,
                last_storm_time_unix_sec=last_storm_time_unix_sec,
                one_file_per_time_step=one_file_per_time_step,
                num_examples_per_batch=num_examples_per_batch,
                num_examples_total=num_examples,
                output_netcdf_file_name=output_netcdf_file_name,
                model_metadata_dict=model_metadata_dict)
        else:
            _extract_3d_cnn_features(
                model_object=model_object,
                top_storm_radar_image_dir_name=top_storm_radar_image_dir_name,
                top_sounding_dir_name=top_sounding_dir_name,
                top_target_dir_name=top_target_dir_name,
                first_storm_time_unix_sec=first_storm_time_unix_sec,
                last_storm_time_unix_sec=last_storm_time_unix_sec,
                one_file_per_time_step=one_file_per_time_step,
                num_examples_per_batch=num_examples_per_batch,
                num_examples_total=num_examples,
                output_netcdf_file_name=output_netcdf_file_name,
                model_metadata_dict=model_metadata_dict)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _extract_features(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        top_storm_radar_image_dir_name=getattr(
            INPUT_ARG_OBJECT, RADAR_DIRECTORY_ARG_NAME),
        top_sounding_dir_name=getattr(
            INPUT_ARG_OBJECT, SOUNDING_DIRECTORY_ARG_NAME),
        top_target_dir_name=getattr(
            INPUT_ARG_OBJECT, TARGET_DIRECTORY_ARG_NAME),
        first_storm_time_string=getattr(
            INPUT_ARG_OBJECT, FIRST_STORM_TIME_ARG_NAME),
        last_storm_time_string=getattr(
            INPUT_ARG_OBJECT, LAST_STORM_TIME_ARG_NAME),
        one_file_per_time_step=bool(getattr(
            INPUT_ARG_OBJECT, ONE_FILE_PER_TIME_STEP_ARG_NAME)),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        output_netcdf_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME))
