"""Writes CNN features (output of the last "Flatten" layer) to CSV file.

CNN = convolutional neural network
"""

import os.path
import pickle
import argparse
import numpy
from keras import backend as K
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import deployment_io
from gewittergefahr.deep_learning import training_validation_io as trainval_io

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

INPUT_TIME_FORMAT = '%Y-%m-%d-%H%M%S'

MODEL_FILE_ARG_NAME = 'input_model_file_name'
RADAR_DIRECTORY_ARG_NAME = 'input_storm_radar_image_dir_name'
SOUNDING_DIRECTORY_ARG_NAME = 'input_sounding_dir_name'
TARGET_DIRECTORY_ARG_NAME = 'input_target_dir_name'
FIRST_STORM_TIME_ARG_NAME = 'first_storm_time_string'
LAST_STORM_TIME_ARG_NAME = 'last_storm_time_string'
ONE_FILE_PER_TIME_STEP_ARG_NAME = 'one_file_per_time_step'
NUM_EXAMPLES_PER_FILE_TIME_ARG_NAME = 'num_examples_per_file_time'
NUM_EXAMPLES_TOTAL_ARG_NAME = 'num_examples_total'
OUTPUT_FILE_ARG_NAME = 'output_pickle_file_name'

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
    'Storm time (format "yyyy-mm-dd-HHMMSS").  Storm times will be drawn '
    'randomly from `{0:s}`...`{1:s}`.  For each time drawn, a max of `{2:d}` '
    'storm objects will be used.  A max of `{3:d}` storm objects over the '
    'entire period will be used.  The features and target value will be written'
    ' for each storm object used.'
).format(FIRST_STORM_TIME_ARG_NAME, LAST_STORM_TIME_ARG_NAME,
         NUM_EXAMPLES_PER_FILE_TIME_ARG_NAME, NUM_EXAMPLES_TOTAL_ARG_NAME)
ONE_FILE_PER_TIME_STEP_HELP_STRING = (
    'Boolean flag.  If 1 (0), the model will be applied to one set of files per'
    ' time step (SPC date).')
NUM_EXAMPLES_PER_FILE_TIME_HELP_STRING = (
    'See discussion for `{0:s}` and `{1:s}`.'
).format(FIRST_STORM_TIME_ARG_NAME, LAST_STORM_TIME_ARG_NAME)
NUM_EXAMPLES_TOTAL_HELP_STRING = NUM_EXAMPLES_PER_FILE_TIME_HELP_STRING
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  For each storm object used, CNN features (output of '
    'the last "Flatten" layer) and the target value will be written to this '
    'file.')

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
    '--' + NUM_EXAMPLES_PER_FILE_TIME_ARG_NAME, type=int, required=True,
    help=NUM_EXAMPLES_PER_FILE_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_TOTAL_ARG_NAME, type=int, required=True,
    help=NUM_EXAMPLES_TOTAL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _get_2d_cnn_features(
        model_object, top_storm_radar_image_dir_name, top_sounding_dir_name,
        top_target_dir_name, first_storm_time_unix_sec,
        last_storm_time_unix_sec, one_file_per_time_step,
        num_examples_per_file_time, num_examples_total, model_metadata_dict):
    """Returns features (output of the last "Flatten" layer) from 2-D CNN.

    E = number of storm objects
    Z = number of features

    :param model_object: Trained model (instance of `keras.models.Sequential`).
    :param top_storm_radar_image_dir_name: See documentation at top of file.
    :param top_sounding_dir_name: Same.
    :param top_target_dir_name: Same.
    :param first_storm_time_unix_sec: Same.
    :param last_storm_time_unix_sec: Same.
    :param one_file_per_time_step: Same.
    :param num_examples_per_file_time: Same.
    :param num_examples_total: Same.
    :param model_metadata_dict: Dictionary created by `cnn.read_model_metadata`.
    :return: feature_matrix: E-by-Z numpy array of features.
    :return: target_values: length-E numpy array of target values.  If
        target_values[i] = k, the [i]th storm object belongs to the [k]th class.
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

    feature_matrix = None
    target_values = numpy.array([], dtype=int)
    num_radar_times = radar_file_name_matrix.shape[0]

    for i in range(num_radar_times):
        print (
            'Have created feature vector for {0:d} of {1:d} storm objects...\n'
        ).format(len(target_values), num_examples_total)

        if len(target_values) > num_examples_total:
            break

        (this_radar_image_matrix, this_sounding_matrix, these_target_values
        ) = deployment_io.create_storm_images_2d(
            radar_file_name_matrix=radar_file_name_matrix[[i], ...],
            num_examples_per_file_time=num_examples_per_file_time,
            return_target=True,
            target_name=model_metadata_dict[cnn.TARGET_NAME_KEY],
            binarize_target=model_metadata_dict[cnn.BINARIZE_TARGET_KEY],
            top_target_directory_name=top_target_dir_name,
            radar_normalization_dict=model_metadata_dict[
                cnn.RADAR_NORMALIZATION_DICT_KEY],
            sounding_field_names=model_metadata_dict[
                cnn.SOUNDING_FIELD_NAMES_KEY],
            top_sounding_dir_name=top_sounding_dir_name,
            sounding_lag_time_for_convective_contamination_sec=
            model_metadata_dict[cnn.SOUNDING_LAG_TIME_KEY],
            sounding_normalization_dict=model_metadata_dict[
                cnn.SOUNDING_NORMALIZATION_DICT_KEY])

        print MINOR_SEPARATOR_STRING
        if this_radar_image_matrix is None:
            continue

        this_feature_matrix = cnn.apply_2d_cnn(
            model_object=model_object,
            radar_image_matrix=this_radar_image_matrix,
            sounding_matrix=this_sounding_matrix, return_features=True)

        target_values = numpy.concatenate((target_values, these_target_values))
        if feature_matrix is None:
            feature_matrix = this_feature_matrix + 0.
        else:
            feature_matrix = numpy.concatenate(
                (feature_matrix, this_feature_matrix), axis=0)

    if len(target_values) > num_examples_total:
        feature_matrix = feature_matrix[:num_examples_total, ...]
        target_values = target_values[:num_examples_total]

    return feature_matrix, target_values


def _get_3d_cnn_features(
        model_object, top_storm_radar_image_dir_name, top_sounding_dir_name,
        top_target_dir_name, first_storm_time_unix_sec,
        last_storm_time_unix_sec, one_file_per_time_step,
        num_examples_per_file_time, num_examples_total, model_metadata_dict):
    """Returns features (output of the last "Flatten" layer) from 3-D CNN.

    :param model_object: Trained model (instance of `keras.models.Sequential`).
    :param top_storm_radar_image_dir_name: See documentation at top of file.
    :param top_sounding_dir_name: Same.
    :param top_target_dir_name: Same.
    :param first_storm_time_unix_sec: Same.
    :param last_storm_time_unix_sec: Same.
    :param one_file_per_time_step: Same.
    :param num_examples_per_file_time: Same.
    :param num_examples_total: Same.
    :param model_metadata_dict: Dictionary created by `cnn.read_model_metadata`.
    :return: feature_matrix: See doc for `_get_2d_cnn_features`.
    :return: target_values: Same.
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

    feature_matrix = None
    target_values = numpy.array([], dtype=int)
    num_radar_times = radar_file_name_matrix.shape[0]

    for i in range(num_radar_times):
        print (
            'Have created feature vector for {0:d} of {1:d} storm objects...\n'
        ).format(len(target_values), num_examples_total)

        if len(target_values) > num_examples_total:
            break

        (this_radar_image_matrix, this_sounding_matrix, these_target_values
        ) = deployment_io.create_storm_images_3d(
            radar_file_name_matrix=radar_file_name_matrix[[i], ...],
            num_examples_per_file_time=num_examples_per_file_time,
            return_target=True,
            target_name=model_metadata_dict[cnn.TARGET_NAME_KEY],
            binarize_target=model_metadata_dict[cnn.BINARIZE_TARGET_KEY],
            top_target_directory_name=top_target_dir_name,
            radar_normalization_dict=model_metadata_dict[
                cnn.RADAR_NORMALIZATION_DICT_KEY],
            sounding_field_names=model_metadata_dict[
                cnn.SOUNDING_FIELD_NAMES_KEY],
            top_sounding_dir_name=top_sounding_dir_name,
            sounding_lag_time_for_convective_contamination_sec=
            model_metadata_dict[cnn.SOUNDING_LAG_TIME_KEY],
            sounding_normalization_dict=model_metadata_dict[
                cnn.SOUNDING_NORMALIZATION_DICT_KEY])

        print MINOR_SEPARATOR_STRING
        if this_radar_image_matrix is None:
            continue

        this_feature_matrix = cnn.apply_3d_cnn(
            model_object=model_object,
            radar_image_matrix=this_radar_image_matrix,
            sounding_matrix=this_sounding_matrix, return_features=True)

        target_values = numpy.concatenate((target_values, these_target_values))
        if feature_matrix is None:
            feature_matrix = this_feature_matrix + 0.
        else:
            feature_matrix = numpy.concatenate(
                (feature_matrix, this_feature_matrix), axis=0)

    if len(target_values) > num_examples_total:
        feature_matrix = feature_matrix[:num_examples_total, ...]
        target_values = target_values[:num_examples_total]

    return feature_matrix, target_values


def _get_2d3d_cnn_features(
        model_object, top_storm_radar_image_dir_name, top_sounding_dir_name,
        top_target_dir_name, first_storm_time_unix_sec,
        last_storm_time_unix_sec, one_file_per_time_step,
        num_examples_per_file_time, num_examples_total, model_metadata_dict):
    """Returns features (output of the last "Flatten" layer) from 2D/3D CNN.

    :param model_object: Trained model (instance of `keras.models.Sequential`).
    :param top_storm_radar_image_dir_name: See documentation at top of file.
    :param top_sounding_dir_name: Same.
    :param top_target_dir_name: Same.
    :param first_storm_time_unix_sec: Same.
    :param last_storm_time_unix_sec: Same.
    :param one_file_per_time_step: Same.
    :param num_examples_per_file_time: Same.
    :param num_examples_total: Same.
    :param model_metadata_dict: Dictionary created by `cnn.read_model_metadata`.
    :return: feature_matrix: See doc for `_get_2d_cnn_features`.
    :return: target_values: Same.
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

    feature_matrix = None
    target_values = numpy.array([], dtype=int)
    num_radar_times = radar_file_name_matrix.shape[0]

    for i in range(num_radar_times):
        print (
            'Have created feature vector for {0:d} of {1:d} storm objects...\n'
        ).format(len(target_values), num_examples_total)

        if len(target_values) > num_examples_total:
            break

        (this_reflectivity_matrix_dbz, this_azimuthal_shear_matrix_s01,
         this_sounding_matrix, these_target_values
        ) = deployment_io.create_storm_images_2d3d_myrorss(
            radar_file_name_matrix=radar_file_name_matrix[[i], ...],
            num_examples_per_file_time=num_examples_per_file_time,
            return_target=True,
            target_name=model_metadata_dict[cnn.TARGET_NAME_KEY],
            binarize_target=model_metadata_dict[cnn.BINARIZE_TARGET_KEY],
            top_target_directory_name=top_target_dir_name,
            radar_normalization_dict=model_metadata_dict[
                cnn.RADAR_NORMALIZATION_DICT_KEY],
            sounding_field_names=model_metadata_dict[
                cnn.SOUNDING_FIELD_NAMES_KEY],
            top_sounding_dir_name=top_sounding_dir_name,
            sounding_lag_time_for_convective_contamination_sec=
            model_metadata_dict[cnn.SOUNDING_LAG_TIME_KEY],
            sounding_normalization_dict=model_metadata_dict[
                cnn.SOUNDING_NORMALIZATION_DICT_KEY])

        print MINOR_SEPARATOR_STRING
        if this_reflectivity_matrix_dbz is None:
            continue

        this_feature_matrix = cnn.apply_2d3d_cnn(
            model_object=model_object,
            reflectivity_image_matrix_dbz=this_reflectivity_matrix_dbz,
            azimuthal_shear_image_matrix_s01=this_azimuthal_shear_matrix_s01,
            sounding_matrix=this_sounding_matrix, return_features=True)

        target_values = numpy.concatenate((target_values, these_target_values))
        if feature_matrix is None:
            feature_matrix = this_feature_matrix + 0.
        else:
            feature_matrix = numpy.concatenate(
                (feature_matrix, this_feature_matrix), axis=0)

    if len(target_values) > num_examples_total:
        feature_matrix = feature_matrix[:num_examples_total, ...]
        target_values = target_values[:num_examples_total]

    return feature_matrix, target_values


def _write_features(
        model_file_name, top_storm_radar_image_dir_name, top_sounding_dir_name,
        top_target_dir_name, first_storm_time_string, last_storm_time_string,
        one_file_per_time_step, num_examples_per_file_time, num_examples_total,
        output_pickle_file_name):
    """Writes CNN features (output of the last "Flatten" layer) to CSV file.

    :param model_file_name: See documentation at top of file.
    :param top_storm_radar_image_dir_name: Same.
    :param top_sounding_dir_name: Same.
    :param top_target_dir_name: Same.
    :param first_storm_time_string: Same.
    :param last_storm_time_string: Same.
    :param one_file_per_time_step: Same.
    :param num_examples_per_file_time: Same.
    :param num_examples_total: Same.
    :param output_pickle_file_name: Same.
    """

    first_storm_time_unix_sec = time_conversion.string_to_unix_sec(
        first_storm_time_string, INPUT_TIME_FORMAT)
    last_storm_time_unix_sec = time_conversion.string_to_unix_sec(
        last_storm_time_string, INPUT_TIME_FORMAT)
    file_system_utils.mkdir_recursive_if_necessary(
        file_name=output_pickle_file_name)

    print 'Reading model from: "{0:s}"...'.format(model_file_name)
    model_object = cnn.read_model(model_file_name)

    model_directory_name, _ = os.path.split(model_file_name)
    metadata_file_name = '{0:s}/model_metadata.p'.format(model_directory_name)

    print 'Reading metadata from: "{0:s}"...'.format(metadata_file_name)
    model_metadata_dict = cnn.read_model_metadata(metadata_file_name)

    if model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]:
        feature_matrix, target_values = _get_2d3d_cnn_features(
            model_object=model_object,
            top_storm_radar_image_dir_name=top_storm_radar_image_dir_name,
            top_sounding_dir_name=top_sounding_dir_name,
            top_target_dir_name=top_target_dir_name,
            first_storm_time_unix_sec=first_storm_time_unix_sec,
            last_storm_time_unix_sec=last_storm_time_unix_sec,
            one_file_per_time_step=one_file_per_time_step,
            num_examples_per_file_time=num_examples_per_file_time,
            num_examples_total=num_examples_total,
            model_metadata_dict=model_metadata_dict)
    else:
        num_radar_dimensions = len(
            model_metadata_dict[cnn.TRAINING_FILE_NAME_MATRIX_KEY].shape)
        if num_radar_dimensions == 2:
            feature_matrix, target_values = _get_2d_cnn_features(
                model_object=model_object,
                top_storm_radar_image_dir_name=top_storm_radar_image_dir_name,
                top_sounding_dir_name=top_sounding_dir_name,
                top_target_dir_name=top_target_dir_name,
                first_storm_time_unix_sec=first_storm_time_unix_sec,
                last_storm_time_unix_sec=last_storm_time_unix_sec,
                one_file_per_time_step=one_file_per_time_step,
                num_examples_per_file_time=num_examples_per_file_time,
                num_examples_total=num_examples_total,
                model_metadata_dict=model_metadata_dict)
        else:
            feature_matrix, target_values = _get_3d_cnn_features(
                model_object=model_object,
                top_storm_radar_image_dir_name=top_storm_radar_image_dir_name,
                top_sounding_dir_name=top_sounding_dir_name,
                top_target_dir_name=top_target_dir_name,
                first_storm_time_unix_sec=first_storm_time_unix_sec,
                last_storm_time_unix_sec=last_storm_time_unix_sec,
                one_file_per_time_step=one_file_per_time_step,
                num_examples_per_file_time=num_examples_per_file_time,
                num_examples_total=num_examples_total,
                model_metadata_dict=model_metadata_dict)

    print SEPARATOR_STRING
    print (
        'Writing {0:d}-by-{1:d} feature matrix and target values to: "{2:s}"...'
    ).format(feature_matrix.shape[0], feature_matrix.shape[1],
             output_pickle_file_name)

    pickle_file_handle = open(output_pickle_file_name, 'wb')
    pickle.dump(feature_matrix, pickle_file_handle)
    pickle.dump(target_values, pickle_file_handle)
    pickle_file_handle.close()


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _write_features(
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
        num_examples_per_file_time=getattr(
            INPUT_ARG_OBJECT, NUM_EXAMPLES_PER_FILE_TIME_ARG_NAME),
        num_examples_total=getattr(
            INPUT_ARG_OBJECT, NUM_EXAMPLES_TOTAL_ARG_NAME),
        output_pickle_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME))
