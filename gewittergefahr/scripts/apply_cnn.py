"""Applies CNN to one example file."""

import os.path
import argparse
import numpy
import netCDF4
import keras.backend as K
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import target_val_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import testing_io
from gewittergefahr.deep_learning import training_validation_io as trainval_io

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

LARGE_INTEGER = int(1e10)
INPUT_TIME_FORMAT = '%Y-%m-%d-%H%M%S'

TARGET_NAME_KEY = 'target_name'
EXAMPLE_DIMENSION_KEY = 'storm_object'
CLASS_DIMENSION_KEY = 'class'
STORM_ID_CHAR_DIM_KEY = 'storm_id_character'

STORM_IDS_KEY = 'storm_ids'
STORM_TIMES_KEY = 'storm_times_unix_sec'
PROBABILITY_MATRIX_KEY = 'class_probability_matrix'
OBSERVED_LABELS_KEY = 'observed_labels'

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_FILE_ARG_NAME = 'input_example_file_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
OUTPUT_FILE_ARG_NAME = 'output_prediction_file_name'

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

OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Predictions will be written here by '
    '`_write_predictions`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_FILE_ARG_NAME, type=str, required=True,
    help=EXAMPLE_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _write_predictions(
        netcdf_file_name, class_probability_matrix, storm_ids,
        storm_times_unix_sec, target_name, observed_labels=None):
    """Writes predictions to NetCDF file.

    K = number of classes
    E = number of examples (storm objects)

    :param netcdf_file_name: Path to output file.
    :param class_probability_matrix: E-by-K numpy array of forecast
        probabilities.
    :param storm_ids: length-E list of storm IDs (strings).
    :param storm_times_unix_sec: length-E numpy array of valid times.
    :param target_name: Name of target variable.
    :param observed_labels: [this may be None]
        length-E numpy array of observed labels (integers in 0...[K - 1]).
    """

    # TODO(thunderhoser): Move this method to another file.

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET')

    dataset_object.setncattr(TARGET_NAME_KEY, target_name)
    dataset_object.createDimension(
        EXAMPLE_DIMENSION_KEY, class_probability_matrix.shape[0]
    )
    dataset_object.createDimension(
        CLASS_DIMENSION_KEY, class_probability_matrix.shape[1]
    )

    num_id_characters = 1 + numpy.max(numpy.array([
        len(s) for s in storm_ids
    ]))
    dataset_object.createDimension(STORM_ID_CHAR_DIM_KEY, num_id_characters)

    # Add storm IDs.
    this_string_format = 'S{0:d}'.format(num_id_characters)
    storm_ids_char_array = netCDF4.stringtochar(numpy.array(
        storm_ids, dtype=this_string_format
    ))

    dataset_object.createVariable(
        STORM_IDS_KEY, datatype='S1',
        dimensions=(EXAMPLE_DIMENSION_KEY, STORM_ID_CHAR_DIM_KEY)
    )
    dataset_object.variables[STORM_IDS_KEY][:] = numpy.array(
        storm_ids_char_array)

    # Add storm times.
    dataset_object.createVariable(
        STORM_TIMES_KEY, datatype=numpy.int32, dimensions=EXAMPLE_DIMENSION_KEY
    )
    dataset_object.variables[STORM_TIMES_KEY][:] = storm_times_unix_sec

    # Add probabilities.
    dataset_object.createVariable(
        PROBABILITY_MATRIX_KEY, datatype=numpy.float32,
        dimensions=(EXAMPLE_DIMENSION_KEY, CLASS_DIMENSION_KEY)
    )
    dataset_object.variables[PROBABILITY_MATRIX_KEY][:] = (
        class_probability_matrix
    )

    if observed_labels is not None:
        dataset_object.createVariable(
            OBSERVED_LABELS_KEY, datatype=numpy.int32,
            dimensions=EXAMPLE_DIMENSION_KEY
        )
        dataset_object.variables[OBSERVED_LABELS_KEY][:] = observed_labels

    dataset_object.close()


def _run(model_file_name, example_file_name, first_time_string,
         last_time_string, output_file_name):
    """Applies CNN to one example file.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param example_file_name: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param output_file_name: Same.
    """

    print 'Reading model from: "{0:s}"...'.format(model_file_name)
    model_object = cnn.read_model(model_file_name)

    model_directory_name, _ = os.path.split(model_file_name)
    model_metafile_name = '{0:s}/model_metadata.p'.format(model_directory_name)

    print 'Reading metadata from: "{0:s}"...'.format(model_metafile_name)
    model_metadata_dict = cnn.read_model_metadata(model_metafile_name)
    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]

    training_option_dict[trainval_io.SAMPLING_FRACTIONS_KEY] = None
    training_option_dict[trainval_io.EXAMPLE_FILES_KEY] = [example_file_name]
    training_option_dict[trainval_io.FIRST_STORM_TIME_KEY] = (
        time_conversion.string_to_unix_sec(first_time_string, INPUT_TIME_FORMAT)
    )
    training_option_dict[trainval_io.LAST_STORM_TIME_KEY] = (
        time_conversion.string_to_unix_sec(last_time_string, INPUT_TIME_FORMAT)
    )

    if model_metadata_dict[cnn.LAYER_OPERATIONS_KEY] is not None:
        generator_object = testing_io.gridrad_generator_2d_reduced(
            option_dict=training_option_dict,
            list_of_operation_dicts=model_metadata_dict[
                cnn.LAYER_OPERATIONS_KEY],
            num_examples_total=LARGE_INTEGER
        )

    elif model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]:
        generator_object = testing_io.myrorss_generator_2d3d(
            option_dict=training_option_dict, num_examples_total=LARGE_INTEGER)
    else:
        generator_object = testing_io.generator_2d_or_3d(
            option_dict=training_option_dict, num_examples_total=LARGE_INTEGER)

    storm_object_dict = next(generator_object)
    print SEPARATOR_STRING

    observed_labels = storm_object_dict[testing_io.TARGET_ARRAY_KEY]
    list_of_predictor_matrices = storm_object_dict[
        testing_io.INPUT_MATRICES_KEY]

    if model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]:
        if len(list_of_predictor_matrices) == 3:
            this_sounding_matrix = list_of_predictor_matrices[2]
        else:
            this_sounding_matrix = None

        class_probability_matrix = cnn.apply_2d3d_cnn(
            model_object=model_object,
            reflectivity_matrix_dbz=list_of_predictor_matrices[0],
            azimuthal_shear_matrix_s01=list_of_predictor_matrices[1],
            sounding_matrix=this_sounding_matrix, verbose=True)

    else:
        if len(list_of_predictor_matrices) == 2:
            this_sounding_matrix = list_of_predictor_matrices[1]
        else:
            this_sounding_matrix = None

        class_probability_matrix = cnn.apply_2d_or_3d_cnn(
            model_object=model_object,
            radar_image_matrix=list_of_predictor_matrices[0],
            sounding_matrix=this_sounding_matrix, verbose=True)

    print SEPARATOR_STRING
    num_examples = class_probability_matrix.shape[0]

    for k in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        print '{0:d}th percentile of {1:d} forecast probs = {2:.4f}'.format(
            k, num_examples, numpy.percentile(class_probability_matrix[:, 1], k)
        )

    print '\n'

    # for i in range(num_examples):
    #     print 'Observed label = {0:d} ... forecast prob = {1:.4f}'.format(
    #         observed_labels[i], class_probability_matrix[i, 1]
    #     )
    #
    # print '\n'

    target_param_dict = target_val_utils.target_name_to_params(
        model_metadata_dict[cnn.TARGET_NAME_KEY]
    )

    target_name = target_val_utils.target_params_to_name(
        min_lead_time_sec=target_param_dict[target_val_utils.MIN_LEAD_TIME_KEY],
        max_lead_time_sec=target_param_dict[target_val_utils.MAX_LEAD_TIME_KEY],
        min_link_distance_metres=target_param_dict[
            target_val_utils.MIN_LINKAGE_DISTANCE_KEY],
        max_link_distance_metres=10000.
    )

    print 'Writing "{0:s}" predictions to: "{1:s}"...'.format(
        target_name, output_file_name)

    _write_predictions(
        netcdf_file_name=output_file_name,
        class_probability_matrix=class_probability_matrix,
        storm_ids=storm_object_dict[testing_io.STORM_IDS_KEY],
        storm_times_unix_sec=storm_object_dict[testing_io.STORM_TIMES_KEY],
        target_name=target_name, observed_labels=observed_labels)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        example_file_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_FILE_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
