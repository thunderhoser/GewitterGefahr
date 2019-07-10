"""Finds extreme examples (storm objects), based on model activations.

There are 6 types of extreme examples:

- highest activations
- lowest activations
- best hits (tornadic storms with highest activations)
- worst misses (tornadic storms with lowest activations)
- worst false alarms (non-tornadic storms with highest activations)
- best correct nulls (non-tornadic storms with lowest activations)

In the above discussion, I assume that the target phenomenon is tornadoes
(target = 1 if tornadic, 0 if not).  The target phenomenon can be anything, as
long as the target variable is binary.
"""

import os.path
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import target_val_utils
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import model_activation
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import training_validation_io as trainval_io

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

FULL_IDS_KEY = model_activation.FULL_IDS_KEY
STORM_TIMES_KEY = model_activation.STORM_TIMES_KEY
STORM_ACTIVATIONS_KEY = 'storm_activations'
TARGET_VALUES_KEY = 'storm_target_values'

ACTIVATION_FILE_ARG_NAME = 'input_activation_file_name'
NUM_LOW_ARG_NAME = 'num_low_activation_examples'
NUM_HIGH_ARG_NAME = 'num_high_activation_examples'
NUM_HITS_ARG_NAME = 'num_hits'
NUM_MISSES_ARG_NAME = 'num_misses'
NUM_FALSE_ALARMS_ARG_NAME = 'num_false_alarms'
NUM_CORRECT_NULLS_ARG_NAME = 'num_correct_nulls'
TARGET_DIR_ARG_NAME = 'input_target_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

ACTIVATION_FILE_HELP_STRING = (
    'Path to input file, containing the activation of one model component for '
    'each example.  Will be read by `model_activation.read_file`.  If the file '
    'contains activations for more than one model component, this script will '
    'error out.')

NUM_LOW_HELP_STRING = 'Number of low-activation examples to keep.'

NUM_HIGH_HELP_STRING = 'Number of high-activation examples to keep.'

NUM_HITS_HELP_STRING = (
    'Number of best hits (highest-activated examples with target = 1) to keep.')

NUM_MISSES_HELP_STRING = (
    'Number of worst misses (lowest-activated examples with target = 1) to '
    'keep.')

NUM_FALSE_ALARMS_HELP_STRING = (
    'Number of worst false alarms (highest-activated examples with target = 0) '
    'to keep.')

NUM_CORRECT_NULLS_HELP_STRING = (
    'Number of best correct nulls (lowest-activated examples with target = 0) '
    'to keep.')

TARGET_DIR_HELP_STRING = (
    'Name of top-level directory with target values (storm-hazard labels).  '
    'Files therein will be found by `target_val_utils.find_target_file` and '
    'read by `target_val_utils.read_target_values`.')

OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  For each type of extreme example (high '
    'activation, low activation, best hit, worst miss, worst false alarm, best '
    'correct null), a single Pickle file will be saved here with storm ID-time '
    'pairs.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + ACTIVATION_FILE_ARG_NAME, type=str, required=True,
    help=ACTIVATION_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_LOW_ARG_NAME, type=int, required=False, default=100,
    help=NUM_LOW_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_HIGH_ARG_NAME, type=int, required=False, default=100,
    help=NUM_HIGH_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_HITS_ARG_NAME, type=int, required=False, default=100,
    help=NUM_HITS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_MISSES_ARG_NAME, type=int, required=False, default=100,
    help=NUM_MISSES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_FALSE_ALARMS_ARG_NAME, type=int, required=False, default=100,
    help=NUM_FALSE_ALARMS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_CORRECT_NULLS_ARG_NAME, type=int, required=False, default=100,
    help=NUM_CORRECT_NULLS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_DIR_ARG_NAME, type=str, required=True,
    help=TARGET_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _read_target_values(
        top_target_dir_name, storm_activations, activation_metadata_dict):
    """Reads target value for each storm object.

    E = number of examples (storm objects)

    :param top_target_dir_name: See documentation at top of file.
    :param storm_activations: length-E numpy array of activations.
    :param activation_metadata_dict: Dictionary returned by
        `model_activation.read_file`.
    :return: target_dict: Dictionary with the following keys.
    target_dict['full_id_strings']: length-E list of full storm IDs.
    target_dict['storm_times_unix_sec']: length-E numpy array of storm times.
    target_dict['storm_activations']: length-E numpy array of model activations.
    target_dict['storm_target_values']: length-E numpy array of target values.

    :raises: ValueError: if the target variable is multiclass and not binarized.
    """

    # Convert input args.
    full_id_strings = activation_metadata_dict[model_activation.FULL_IDS_KEY]
    storm_times_unix_sec = activation_metadata_dict[
        model_activation.STORM_TIMES_KEY]

    storm_spc_date_strings_numpy = numpy.array([
        time_conversion.time_to_spc_date_string(t) for t in storm_times_unix_sec
    ], dtype=object)

    unique_spc_date_strings_numpy = numpy.unique(storm_spc_date_strings_numpy)

    # Read metadata for machine-learning model.
    model_file_name = activation_metadata_dict[
        model_activation.MODEL_FILE_NAME_KEY
    ]
    model_metadata_file_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0]
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metadata_file_name))
    model_metadata_dict = cnn.read_model_metadata(model_metadata_file_name)
    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]

    target_name = training_option_dict[trainval_io.TARGET_NAME_KEY]
    num_classes = target_val_utils.target_name_to_num_classes(
        target_name=target_name, include_dead_storms=False)

    binarize_target = (
        training_option_dict[trainval_io.BINARIZE_TARGET_KEY] and
        num_classes > 2
    )

    if num_classes > 2 and not binarize_target:
        error_string = (
            'The target variable ("{0:s}") is multiclass, which this script '
            'cannot handle.'
        ).format(target_name)

        raise ValueError(error_string)

    event_type_string = target_val_utils.target_name_to_params(target_name)[
        target_val_utils.EVENT_TYPE_KEY
    ]

    # Read target values.
    storm_target_values = numpy.array([], dtype=int)
    id_sort_indices = numpy.array([], dtype=int)
    num_spc_dates = len(unique_spc_date_strings_numpy)

    for i in range(num_spc_dates):
        this_target_file_name = target_val_utils.find_target_file(
            top_directory_name=top_target_dir_name,
            event_type_string=event_type_string,
            spc_date_string=unique_spc_date_strings_numpy[i]
        )

        print('Reading data from: "{0:s}"...'.format(this_target_file_name))
        this_target_value_dict = target_val_utils.read_target_values(
            netcdf_file_name=this_target_file_name, target_names=[target_name]
        )

        these_indices = numpy.where(
            storm_spc_date_strings_numpy == unique_spc_date_strings_numpy[i]
        )[0]
        id_sort_indices = numpy.concatenate((id_sort_indices, these_indices))

        these_indices = tracking_utils.find_storm_objects(
            all_id_strings=this_target_value_dict[
                target_val_utils.FULL_IDS_KEY],
            all_times_unix_sec=this_target_value_dict[
                target_val_utils.VALID_TIMES_KEY],
            id_strings_to_keep=[full_id_strings[k] for k in these_indices],
            times_to_keep_unix_sec=storm_times_unix_sec[these_indices]
        )

        if len(these_indices) == 0:
            continue

        these_target_values = numpy.squeeze(
            this_target_value_dict[target_val_utils.TARGET_MATRIX_KEY][
                these_indices, :]
        )

        storm_target_values = numpy.concatenate((
            storm_target_values, these_target_values
        ))

    good_indices = numpy.where(
        storm_target_values != target_val_utils.INVALID_STORM_INTEGER
    )[0]

    storm_target_values = storm_target_values[good_indices]
    id_sort_indices = id_sort_indices[good_indices]

    if binarize_target:
        storm_target_values = (
            storm_target_values == num_classes - 1
        ).astype(int)

    return {
        FULL_IDS_KEY: [full_id_strings[k] for k in id_sort_indices],
        STORM_TIMES_KEY: storm_times_unix_sec[id_sort_indices],
        STORM_ACTIVATIONS_KEY: storm_activations[id_sort_indices],
        TARGET_VALUES_KEY: storm_target_values
    }


def _run(
        input_activation_file_name, num_low_activation_examples,
        num_high_activation_examples, num_hits, num_misses, num_false_alarms,
        num_correct_nulls, top_target_dir_name, output_dir_name):
    """Finds extreme examples (storm objects), based on model activations.

    This is effectively the main method.

    :param input_activation_file_name: See documentation at top of file.
    :param num_low_activation_examples: Same.
    :param num_high_activation_examples: Same.
    :param num_hits: Same.
    :param num_misses: Same.
    :param num_false_alarms: Same.
    :param num_correct_nulls: Same.
    :param top_target_dir_name: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if the activation file contains activations for more
        than one model component.
    """

    # Check input args.
    example_counts = numpy.array([
        num_low_activation_examples, num_high_activation_examples, num_hits,
        num_misses, num_false_alarms, num_correct_nulls
    ], dtype=int)

    error_checking.assert_is_greater(numpy.sum(example_counts), 0)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    # Read activations.
    print('Reading activations from: "{0:s}"...'.format(
        input_activation_file_name
    ))
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
    full_id_strings = activation_metadata_dict[model_activation.FULL_IDS_KEY]
    storm_times_unix_sec = activation_metadata_dict[
        model_activation.STORM_TIMES_KEY]

    num_storm_objects = len(full_id_strings)
    error_checking.assert_is_leq(numpy.sum(example_counts), num_storm_objects)

    # Find high- and low-activation examples.
    if num_low_activation_examples + num_high_activation_examples > 0:
        high_indices, low_indices = (
            model_activation.get_hilo_activation_examples(
                storm_activations=storm_activations,
                num_low_activation_examples=num_low_activation_examples,
                num_high_activation_examples=num_high_activation_examples)
        )
    else:
        high_indices = numpy.array([], dtype=int)
        low_indices = numpy.array([], dtype=int)

    # Write high-activation examples to file.
    if len(high_indices) > 0:
        high_activation_file_name = '{0:s}/high_activation_examples.p'.format(
            output_dir_name)

        print((
            'Writing IDs and times for high-activation examples to: "{0:s}"...'
        ).format(high_activation_file_name))

        this_activation_matrix = numpy.reshape(
            storm_activations[high_indices], (len(high_indices), 1)
        )

        model_activation.write_file(
            pickle_file_name=high_activation_file_name,
            activation_matrix=this_activation_matrix,
            full_id_strings=[full_id_strings[k] for k in high_indices],
            storm_times_unix_sec=storm_times_unix_sec[high_indices],
            model_file_name=activation_metadata_dict[
                model_activation.MODEL_FILE_NAME_KEY],
            component_type_string=activation_metadata_dict[
                model_activation.COMPONENT_TYPE_KEY],
            target_class=activation_metadata_dict[
                model_activation.TARGET_CLASS_KEY],
            layer_name=activation_metadata_dict[
                model_activation.LAYER_NAME_KEY],
            neuron_index_matrix=activation_metadata_dict[
                model_activation.NEURON_INDICES_KEY],
            channel_indices=activation_metadata_dict[
                model_activation.CHANNEL_INDICES_KEY]
        )

    # Write low-activation examples to file
    if len(low_indices) > 0:
        low_activation_file_name = '{0:s}/low_activation_examples.p'.format(
            output_dir_name)

        print((
            'Writing IDs and times for low-activation examples to: "{0:s}"...'
        ).format(low_activation_file_name))

        this_activation_matrix = numpy.reshape(
            storm_activations[low_indices], (len(low_indices), 1)
        )

        model_activation.write_file(
            pickle_file_name=low_activation_file_name,
            activation_matrix=this_activation_matrix,
            full_id_strings=[full_id_strings[k] for k in low_indices],
            storm_times_unix_sec=storm_times_unix_sec[low_indices],
            model_file_name=activation_metadata_dict[
                model_activation.MODEL_FILE_NAME_KEY],
            component_type_string=activation_metadata_dict[
                model_activation.COMPONENT_TYPE_KEY],
            target_class=activation_metadata_dict[
                model_activation.TARGET_CLASS_KEY],
            layer_name=activation_metadata_dict[
                model_activation.LAYER_NAME_KEY],
            neuron_index_matrix=activation_metadata_dict[
                model_activation.NEURON_INDICES_KEY],
            channel_indices=activation_metadata_dict[
                model_activation.CHANNEL_INDICES_KEY]
        )

    if num_hits + num_misses + num_false_alarms + num_correct_nulls == 0:
        return

    print(SEPARATOR_STRING)
    target_value_dict = _read_target_values(
        top_target_dir_name=top_target_dir_name,
        storm_activations=storm_activations,
        activation_metadata_dict=activation_metadata_dict)
    print(SEPARATOR_STRING)

    full_id_strings = target_value_dict[FULL_IDS_KEY]
    storm_times_unix_sec = target_value_dict[STORM_TIMES_KEY]
    storm_activations = target_value_dict[STORM_ACTIVATIONS_KEY]
    storm_target_values = target_value_dict[TARGET_VALUES_KEY]

    ct_extreme_dict = model_activation.get_contingency_table_extremes(
        storm_activations=storm_activations,
        storm_target_values=storm_target_values, num_hits=num_hits,
        num_misses=num_misses, num_false_alarms=num_false_alarms,
        num_correct_nulls=num_correct_nulls)

    hit_indices = ct_extreme_dict[model_activation.HIT_INDICES_KEY]
    miss_indices = ct_extreme_dict[model_activation.MISS_INDICES_KEY]
    false_alarm_indices = ct_extreme_dict[
        model_activation.FALSE_ALARM_INDICES_KEY]
    correct_null_indices = ct_extreme_dict[
        model_activation.CORRECT_NULL_INDICES_KEY]

    # Write best hits to file.
    if len(hit_indices) > 0:
        best_hit_file_name = '{0:s}/best_hits.p'.format(output_dir_name)
        print('Writing IDs and times for best hits to: "{0:s}"...'.format(
            best_hit_file_name))

        this_activation_matrix = numpy.reshape(
            storm_activations[hit_indices], (len(hit_indices), 1)
        )

        model_activation.write_file(
            pickle_file_name=best_hit_file_name,
            activation_matrix=this_activation_matrix,
            full_id_strings=[full_id_strings[k] for k in hit_indices],
            storm_times_unix_sec=storm_times_unix_sec[hit_indices],
            model_file_name=activation_metadata_dict[
                model_activation.MODEL_FILE_NAME_KEY],
            component_type_string=activation_metadata_dict[
                model_activation.COMPONENT_TYPE_KEY],
            target_class=activation_metadata_dict[
                model_activation.TARGET_CLASS_KEY],
            layer_name=activation_metadata_dict[
                model_activation.LAYER_NAME_KEY],
            neuron_index_matrix=activation_metadata_dict[
                model_activation.NEURON_INDICES_KEY],
            channel_indices=activation_metadata_dict[
                model_activation.CHANNEL_INDICES_KEY]
        )

    # Write worst misses to file.
    if len(miss_indices) > 0:
        worst_miss_file_name = '{0:s}/worst_misses.p'.format(output_dir_name)
        print('Writing IDs and times for worst misses to: "{0:s}"...'.format(
            worst_miss_file_name))

        this_activation_matrix = numpy.reshape(
            storm_activations[miss_indices], (len(miss_indices), 1)
        )

        model_activation.write_file(
            pickle_file_name=worst_miss_file_name,
            activation_matrix=this_activation_matrix,
            full_id_strings=[full_id_strings[k] for k in miss_indices],
            storm_times_unix_sec=storm_times_unix_sec[miss_indices],
            model_file_name=activation_metadata_dict[
                model_activation.MODEL_FILE_NAME_KEY],
            component_type_string=activation_metadata_dict[
                model_activation.COMPONENT_TYPE_KEY],
            target_class=activation_metadata_dict[
                model_activation.TARGET_CLASS_KEY],
            layer_name=activation_metadata_dict[
                model_activation.LAYER_NAME_KEY],
            neuron_index_matrix=activation_metadata_dict[
                model_activation.NEURON_INDICES_KEY],
            channel_indices=activation_metadata_dict[
                model_activation.CHANNEL_INDICES_KEY]
        )

    # Write worst false alarms to file.
    if len(false_alarm_indices) > 0:
        worst_fa_file_name = '{0:s}/worst_false_alarms.p'.format(
            output_dir_name)

        print((
            'Writing IDs and times for worst false alarms to: "{0:s}"...'
        ).format(worst_fa_file_name))

        this_activation_matrix = numpy.reshape(
            storm_activations[false_alarm_indices],
            (len(false_alarm_indices), 1)
        )

        model_activation.write_file(
            pickle_file_name=worst_fa_file_name,
            activation_matrix=this_activation_matrix,
            full_id_strings=[full_id_strings[k] for k in false_alarm_indices],
            storm_times_unix_sec=storm_times_unix_sec[false_alarm_indices],
            model_file_name=activation_metadata_dict[
                model_activation.MODEL_FILE_NAME_KEY],
            component_type_string=activation_metadata_dict[
                model_activation.COMPONENT_TYPE_KEY],
            target_class=activation_metadata_dict[
                model_activation.TARGET_CLASS_KEY],
            layer_name=activation_metadata_dict[
                model_activation.LAYER_NAME_KEY],
            neuron_index_matrix=activation_metadata_dict[
                model_activation.NEURON_INDICES_KEY],
            channel_indices=activation_metadata_dict[
                model_activation.CHANNEL_INDICES_KEY]
        )

    # Write best correct nulls to file.
    if len(correct_null_indices) > 0:
        best_cn_file_name = '{0:s}/best_correct_nulls.p'.format(
            output_dir_name)

        print((
            'Writing IDs and times for best correct nulls to: "{0:s}"...'
        ).format(best_cn_file_name))

        this_activation_matrix = numpy.reshape(
            storm_activations[correct_null_indices],
            (len(correct_null_indices), 1)
        )

        model_activation.write_file(
            pickle_file_name=best_cn_file_name,
            activation_matrix=this_activation_matrix,
            full_id_strings=[full_id_strings[k] for k in correct_null_indices],
            storm_times_unix_sec=storm_times_unix_sec[correct_null_indices],
            model_file_name=activation_metadata_dict[
                model_activation.MODEL_FILE_NAME_KEY],
            component_type_string=activation_metadata_dict[
                model_activation.COMPONENT_TYPE_KEY],
            target_class=activation_metadata_dict[
                model_activation.TARGET_CLASS_KEY],
            layer_name=activation_metadata_dict[
                model_activation.LAYER_NAME_KEY],
            neuron_index_matrix=activation_metadata_dict[
                model_activation.NEURON_INDICES_KEY],
            channel_indices=activation_metadata_dict[
                model_activation.CHANNEL_INDICES_KEY]
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_activation_file_name=getattr(
            INPUT_ARG_OBJECT, ACTIVATION_FILE_ARG_NAME),
        num_low_activation_examples=getattr(INPUT_ARG_OBJECT, NUM_LOW_ARG_NAME),
        num_high_activation_examples=getattr(
            INPUT_ARG_OBJECT, NUM_HIGH_ARG_NAME),
        num_hits=getattr(INPUT_ARG_OBJECT, NUM_HITS_ARG_NAME),
        num_misses=getattr(INPUT_ARG_OBJECT, NUM_MISSES_ARG_NAME),
        num_false_alarms=getattr(INPUT_ARG_OBJECT, NUM_FALSE_ALARMS_ARG_NAME),
        num_correct_nulls=getattr(INPUT_ARG_OBJECT, NUM_CORRECT_NULLS_ARG_NAME),
        top_target_dir_name=getattr(INPUT_ARG_OBJECT, TARGET_DIR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
