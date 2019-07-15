"""Applies CNN to one example file."""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import numpy
import keras.backend as K
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import linkage
from gewittergefahr.gg_utils import target_val_utils
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import testing_io
from gewittergefahr.deep_learning import prediction_io
from gewittergefahr.deep_learning import training_validation_io as trainval_io

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
    allow_soft_placement=False
)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

LARGE_INTEGER = int(1e10)
INPUT_TIME_FORMAT = '%Y-%m-%d-%H%M%S'

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_FILE_ARG_NAME = 'input_example_file_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained CNN.  Will be read by `cnn.read_model`.')

EXAMPLE_FILE_HELP_STRING = (
    'Path to example file (will be read by `input_examples.read_example_file`).'
    '  The CNN will be applied to all examples in this file in the time period '
    '`{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

TIME_HELP_STRING = (
    'Time (format "yyyy-mm-dd-HHMMSS").  The CNN will be applied to all '
    'examples in the file `{0:s}` in the time period `{1:s}`...`{2:s}`.'
).format(EXAMPLE_FILE_ARG_NAME, FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Predictions will be written here by '
    '`predictions_io.write_ungridded_predictions`, to an exact location '
    'determined by `prediction_io.find_file`.')

DEFAULT_MODEL_FILE_NAME = (
    '/condo/swatwork/ralager/data_aug_experiment_gridrad/'
    'rotation-angles-deg=15-m15-30-m30_noise-stdev=0.05_flip=0_'
    'x-translations-px=m3-0-3-3-3-0-m3-m3_y-translations-px=3-3-3-0-m3-m3-m3-0/'
    'model.h5')

DEFAULT_EXAMPLE_FILE_NAME = (
    '/condo/swatcommon/common/gridrad_final/myrorss_format/tracks/'
    'correct_echo_tops/reanalyzed/input_examples/2018/'
    'input_examples_20180403.nc')

DEFAULT_OUTPUT_DIR_NAME = '{0:s}/predictions'.format(DEFAULT_MODEL_FILE_NAME)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=False,
    default=DEFAULT_MODEL_FILE_NAME, help=MODEL_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_FILE_ARG_NAME, type=str, required=False,
    default=DEFAULT_EXAMPLE_FILE_NAME, help=EXAMPLE_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_OUTPUT_DIR_NAME, help=OUTPUT_DIR_HELP_STRING)


def _run(model_file_name, example_file_name, first_time_string,
         last_time_string, top_output_dir_name):
    """Applies CNN to one example file.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param example_file_name: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param top_output_dir_name: Same.
    """

    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = cnn.read_model(model_file_name)

    model_directory_name, _ = os.path.split(model_file_name)
    model_metafile_name = '{0:s}/model_metadata.p'.format(model_directory_name)

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_model_metadata(model_metafile_name)
    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)

    training_option_dict[trainval_io.SAMPLING_FRACTIONS_KEY] = None
    training_option_dict[trainval_io.EXAMPLE_FILES_KEY] = [example_file_name]
    training_option_dict[trainval_io.FIRST_STORM_TIME_KEY] = first_time_unix_sec
    training_option_dict[trainval_io.LAST_STORM_TIME_KEY] = last_time_unix_sec

    if model_metadata_dict[cnn.LAYER_OPERATIONS_KEY] is not None:
        generator_object = testing_io.gridrad_generator_2d_reduced(
            option_dict=training_option_dict,
            list_of_operation_dicts=model_metadata_dict[
                cnn.LAYER_OPERATIONS_KEY],
            num_examples_total=LARGE_INTEGER
        )

    elif model_metadata_dict[cnn.CONV_2D3D_KEY]:
        generator_object = testing_io.myrorss_generator_2d3d(
            option_dict=training_option_dict, num_examples_total=LARGE_INTEGER)
    else:
        generator_object = testing_io.generator_2d_or_3d(
            option_dict=training_option_dict, num_examples_total=LARGE_INTEGER)

    include_soundings = (
        training_option_dict[trainval_io.SOUNDING_FIELDS_KEY] is not None
    )

    try:
        storm_object_dict = next(generator_object)
    except StopIteration:
        storm_object_dict = None

    print(SEPARATOR_STRING)

    if storm_object_dict is not None:
        observed_labels = storm_object_dict[testing_io.TARGET_ARRAY_KEY]
        list_of_predictor_matrices = storm_object_dict[
            testing_io.INPUT_MATRICES_KEY]

        if include_soundings:
            sounding_matrix = list_of_predictor_matrices[-1]
        else:
            sounding_matrix = None

        if model_metadata_dict[cnn.CONV_2D3D_KEY]:
            if training_option_dict[trainval_io.UPSAMPLE_REFLECTIVITY_KEY]:
                class_probability_matrix = cnn.apply_2d_or_3d_cnn(
                    model_object=model_object,
                    radar_image_matrix=list_of_predictor_matrices[0],
                    sounding_matrix=sounding_matrix, verbose=True)
            else:
                class_probability_matrix = cnn.apply_2d3d_cnn(
                    model_object=model_object,
                    reflectivity_matrix_dbz=list_of_predictor_matrices[0],
                    azimuthal_shear_matrix_s01=list_of_predictor_matrices[1],
                    sounding_matrix=sounding_matrix, verbose=True)
        else:
            class_probability_matrix = cnn.apply_2d_or_3d_cnn(
                model_object=model_object,
                radar_image_matrix=list_of_predictor_matrices[0],
                sounding_matrix=sounding_matrix, verbose=True)

        print(SEPARATOR_STRING)
        num_examples = class_probability_matrix.shape[0]

        for k in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            print('{0:d}th percentile of {1:d} forecast probs = {2:.4f}'.format(
                k, num_examples,
                numpy.percentile(class_probability_matrix[:, 1], k)
            ))

        print('\n')

    target_param_dict = target_val_utils.target_name_to_params(
        training_option_dict[trainval_io.TARGET_NAME_KEY]
    )

    event_type_string = target_param_dict[target_val_utils.EVENT_TYPE_KEY]
    if event_type_string == linkage.TORNADO_EVENT_STRING:
        genesis_only = False
    elif event_type_string == linkage.TORNADOGENESIS_EVENT_STRING:
        genesis_only = True
    else:
        genesis_only = None

    target_name = target_val_utils.target_params_to_name(
        min_lead_time_sec=target_param_dict[target_val_utils.MIN_LEAD_TIME_KEY],
        max_lead_time_sec=target_param_dict[target_val_utils.MAX_LEAD_TIME_KEY],
        min_link_distance_metres=target_param_dict[
            target_val_utils.MIN_LINKAGE_DISTANCE_KEY],
        max_link_distance_metres=10000., genesis_only=genesis_only
    )

    output_file_name = prediction_io.find_file(
        top_prediction_dir_name=top_output_dir_name,
        first_init_time_unix_sec=first_time_unix_sec,
        last_init_time_unix_sec=last_time_unix_sec,
        gridded=False, raise_error_if_missing=False
    )

    print('Writing "{0:s}" predictions to: "{1:s}"...'.format(
        target_name, output_file_name))

    if storm_object_dict is None:
        num_output_neurons = (
            model_object.layers[-1].output.get_shape().as_list()[-1]
        )

        num_classes = max([num_output_neurons, 2])
        class_probability_matrix = numpy.full((0, num_classes), numpy.nan)

        prediction_io.write_ungridded_predictions(
            netcdf_file_name=output_file_name,
            class_probability_matrix=class_probability_matrix,
            storm_ids=[], storm_times_unix_sec=numpy.array([], dtype=int),
            target_name=target_name, observed_labels=numpy.array([], dtype=int)
        )

        return

    prediction_io.write_ungridded_predictions(
        netcdf_file_name=output_file_name,
        class_probability_matrix=class_probability_matrix,
        storm_ids=storm_object_dict[testing_io.FULL_IDS_KEY],
        storm_times_unix_sec=storm_object_dict[testing_io.STORM_TIMES_KEY],
        target_name=target_name, observed_labels=observed_labels)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        example_file_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_FILE_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
