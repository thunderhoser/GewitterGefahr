"""Finds extreme examples vis-a-vis two models.

The 6 sets of extreme examples are as follows:

- high probability
- low probability
- best hits (positive examples with high probability)
- worst false alarms (negative examples with high probability)
- worst misses (positive examples with low probability)
- best correct nulls (negative examples with low probability)
"""

import time
import argparse
import numpy
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import prediction_io
from gewittergefahr.deep_learning import model_activation
from gewittergefahr.deep_learning import model_interpretation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

CLASS_COMPONENT_STRING = model_interpretation.CLASS_COMPONENT_TYPE_STRING

PREDICTION_FILES_ARG_NAME = 'input_prediction_file_names'
MATCH_DIR_ARG_NAME = 'input_match_dir_name'
UNIQUE_CELLS_ARG_NAME = 'unique_storm_cells'
NUM_HITS_ARG_NAME = 'num_hits'
NUM_MISSES_ARG_NAME = 'num_misses'
NUM_FALSE_ALARMS_ARG_NAME = 'num_false_alarms'
NUM_CORRECT_NULLS_ARG_NAME = 'num_correct_nulls'
NUM_DISAGREEMENTS_ARG_NAME = 'num_disagreements'
OUTPUT_DIRS_ARG_NAME = 'output_dir_names'

PREDICTION_FILES_HELP_STRING = (
    'Paths to prediction files (one for each model).  Will be read by '
    '`prediction_io.read_ungridded_predictions`.')

MATCH_DIR_HELP_STRING = (
    'Name of top-level directory with matches (from storm objects in first '
    'prediction file to those in second).  Files therein will be found by '
    '`storm_tracking_io.find_match_file` and read by '
    '`storm_tracking_io.read_matches`.')

UNIQUE_CELLS_HELP_STRING = (
    'Boolean flag.  If 1, each set will contain no more than one example per '
    'storm cell.  If 0, each set may contain multiple examples from the same '
    'storm cell.')

NUM_HITS_HELP_STRING = (
    'Number of best hits to keep (based on average probability from the two '
    'models).')

NUM_MISSES_HELP_STRING = (
    'Number of worst misses to keep (based on average probability from the two '
    'models).')

NUM_FALSE_ALARMS_HELP_STRING = (
    'Number of worst false alarms to keep (based on average probability from '
    'the two models).')

NUM_CORRECT_NULLS_HELP_STRING = (
    'Number of best correct nulls to keep (based on average probability from '
    'the two models).')

NUM_DISAGREEMENTS_HELP_STRING = (
    'Number of disagreements to keep.  Will keep the K examples with the '
    'greatest p_2 - p_1 and the K examples with the greatest p_1 - p_2, where '
    'K = `{0:s}`; p_1 = probability from the first model; and p_2 = probability'
    ' from the second model.'
).format(NUM_DISAGREEMENTS_ARG_NAME)

OUTPUT_DIRS_HELP_STRING = (
    'Names of output directories.  For each set of extreme examples, a file '
    'will be written to each directory by `model_activation.write_file`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_FILES_ARG_NAME, type=str, nargs=2, required=True,
    help=PREDICTION_FILES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MATCH_DIR_ARG_NAME, type=str, required=True,
    help=MATCH_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + UNIQUE_CELLS_ARG_NAME, type=int, required=False, default=0,
    help=UNIQUE_CELLS_HELP_STRING)

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
    '--' + NUM_DISAGREEMENTS_ARG_NAME, type=int, required=False, default=100,
    help=NUM_DISAGREEMENTS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIRS_ARG_NAME, type=str, nargs=2, required=True,
    help=OUTPUT_DIRS_HELP_STRING)


def _find_examples_in_prediction_dict(
        prediction_dict, full_storm_id_strings, storm_times_unix_sec,
        allow_missing):
    """Finds examples (with given ID-timem pairs) in dictionary w/ predictions.

    E = number of desired examples

    :param prediction_dict: Dictionary returned by
        `prediction_io.read_ungridded_predictions`.
    :param full_storm_id_strings: length-E list of storm IDs.
    :param storm_times_unix_sec: length-E numpy array of valid times.
    :param allow_missing: Boolean flag.  If True, will allow for missing storm
        objects.
    :return: indices_in_dict: length-E numpy array of indices.  If
        `k in indices_in_dict`, the [k]th example in the dictionary is one of
        the desired examples.
    """

    if len(numpy.unique(storm_times_unix_sec)) == 1:
        indices_in_dict = numpy.where(
            prediction_dict[prediction_io.STORM_TIMES_KEY] ==
            storm_times_unix_sec[0]
        )[0]

        subindices = tracking_utils.find_storm_objects(
            all_id_strings=[
                prediction_dict[prediction_io.STORM_IDS_KEY][k]
                for k in indices_in_dict
            ],
            all_times_unix_sec=
            prediction_dict[prediction_io.STORM_TIMES_KEY][indices_in_dict],
            id_strings_to_keep=full_storm_id_strings,
            times_to_keep_unix_sec=storm_times_unix_sec,
            allow_missing=allow_missing
        )

        subindices = subindices[subindices >= 0]
        indices_in_dict = indices_in_dict[subindices]
        return indices_in_dict

    indices_in_dict = tracking_utils.find_storm_objects(
        all_id_strings=prediction_dict[prediction_io.STORM_IDS_KEY],
        all_times_unix_sec=prediction_dict[prediction_io.STORM_TIMES_KEY],
        id_strings_to_keep=full_storm_id_strings,
        times_to_keep_unix_sec=storm_times_unix_sec,
        allow_missing=allow_missing
    )

    return indices_in_dict[indices_in_dict >= 0]


def _match_storm_objects_one_time(
        first_prediction_dict, second_prediction_dict, match_dict,
        allow_missing=True):
    """Matches storm objects at one time step.

    E = number of matched storm objects

    :param first_prediction_dict: See doc for `_match_storm_objects`.
    :param second_prediction_dict: Same.
    :param match_dict: Dictionary returned by `storm_tracking_io.read_matches`.
    :param allow_missing: Boolean flag.  If True, will allow for missing storm
        objects.
    :return: first_indices: length-E numpy array with indices of matched storm
        objects in `first_prediction_dict`.
    :return: second_indices: Same but for `second_prediction_dict`.
    """

    first_id_time_pairs = [
        k for k in match_dict.keys() if match_dict[k] is not None
    ]

    if len(first_id_time_pairs) == 0:
        empty_array = numpy.array([], dtype=int)
        return empty_array, empty_array

    first_full_id_strings = [p[0] for p in first_id_time_pairs]
    first_times_unix_sec = numpy.array(
        [p[1] for p in first_id_time_pairs], dtype=int
    )

    first_indices = _find_examples_in_prediction_dict(
        prediction_dict=first_prediction_dict,
        full_storm_id_strings=first_full_id_strings,
        storm_times_unix_sec=first_times_unix_sec, allow_missing=allow_missing)

    second_full_id_strings = [
        match_dict[p][0] for p in first_id_time_pairs
    ]
    second_times_unix_sec = numpy.array([
        match_dict[p][1] for p in first_id_time_pairs
    ], dtype=int)

    second_indices = _find_examples_in_prediction_dict(
        prediction_dict=second_prediction_dict,
        full_storm_id_strings=second_full_id_strings,
        storm_times_unix_sec=second_times_unix_sec, allow_missing=allow_missing)

    good_subindices = numpy.where(
        first_prediction_dict[prediction_io.OBSERVED_LABELS_KEY][first_indices]
        ==
        second_prediction_dict[prediction_io.OBSERVED_LABELS_KEY][
            second_indices]
    )[0]

    if len(good_subindices) != len(first_indices):
        print(first_indices)
        print('\n')
        print(second_indices)
        print('\n')

        print((
            '{0:d} of {1:d} storm-object pairs have different labels!\n'
        ).format(
            len(first_indices) - len(good_subindices), len(first_indices)
        ))

    return first_indices[good_subindices], second_indices[good_subindices]


def _match_storm_objects(first_prediction_dict, second_prediction_dict,
                         top_match_dir_name):
    """Matches storm objects between first and second prediction files.

    F = number of storm objects in first prediction file

    :param first_prediction_dict: Dictionary returned by
        `prediction_io.read_ungridded_predictions` for first prediction file.
    :param second_prediction_dict: Same but for second prediction file.
    :param top_match_dir_name: See documentation at top of file.
    :return: first_prediction_dict: Same as input, but containing only storm
        objects matched with one in the second file.
    :return: second_prediction_dict: Same as input, but containing only storm
        objects matched with one in the first file.  Both dictionaries have
        storm objects in the same order.
    """

    first_storm_times_unix_sec = first_prediction_dict[
        prediction_io.STORM_TIMES_KEY]
    first_unique_times_unix_sec = numpy.unique(first_storm_times_unix_sec)

    first_indices = numpy.array([], dtype=int)
    second_indices = numpy.array([], dtype=int)

    for i in range(len(first_unique_times_unix_sec)):
        this_match_file_name = tracking_io.find_match_file(
            top_directory_name=top_match_dir_name,
            valid_time_unix_sec=first_unique_times_unix_sec[i],
            raise_error_if_missing=True)

        print('Reading data from: "{0:s}"...'.format(this_match_file_name))
        this_match_dict = tracking_io.read_matches(this_match_file_name)[0]

        exec_start_time_unix_sec = time.time()

        these_first_indices, these_second_indices = (
            _match_storm_objects_one_time(
                first_prediction_dict=first_prediction_dict,
                second_prediction_dict=second_prediction_dict,
                match_dict=this_match_dict)
        )

        print('Time elapsed matching storm objects = {0:.4f} seconds'.format(
            time.time() - exec_start_time_unix_sec
        ))

        first_indices = numpy.concatenate((first_indices, these_first_indices))
        second_indices = numpy.concatenate((
            second_indices, these_second_indices
        ))

    first_prediction_dict = prediction_io.subset_ungridded_predictions(
        prediction_dict=first_prediction_dict,
        desired_storm_indices=first_indices)

    second_prediction_dict = prediction_io.subset_ungridded_predictions(
        prediction_dict=second_prediction_dict,
        desired_storm_indices=second_indices)

    return first_prediction_dict, second_prediction_dict


def _run(prediction_file_names, top_match_dir_name, unique_storm_cells,
         num_hits, num_misses, num_false_alarms, num_correct_nulls,
         num_disagreements, output_dir_names):
    """Finds extreme examples vis-a-vis two models.

    This is effectively the main method.

    :param prediction_file_names: See documentation at top of file.
    :param top_match_dir_name: Same.
    :param unique_storm_cells: Same.
    :param num_hits: Same.
    :param num_misses: Same.
    :param num_false_alarms: Same.
    :param num_correct_nulls: Same.
    :param num_disagreements: Same.
    :param output_dir_names: Same.
    """

    # TODO(thunderhoser): Throw error if multiclass predictions are read.

    # Check input args.
    example_counts = numpy.array([
        num_hits, num_misses, num_false_alarms, num_correct_nulls,
        num_disagreements
    ], dtype=int)

    error_checking.assert_is_geq_numpy_array(example_counts, 0)

    first_output_dir_name = output_dir_names[0]
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=first_output_dir_name)

    second_output_dir_name = output_dir_names[1]
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=second_output_dir_name)

    # Match storm objects between the two prediction files.
    print('Reading data from: "{0:s}"...'.format(prediction_file_names[0]))
    first_prediction_dict = prediction_io.read_ungridded_predictions(
        prediction_file_names[0]
    )

    print('Reading data from: "{0:s}"...'.format(prediction_file_names[1]))
    second_prediction_dict = prediction_io.read_ungridded_predictions(
        prediction_file_names[1]
    )

    print(SEPARATOR_STRING)
    first_prediction_dict, second_prediction_dict = _match_storm_objects(
        first_prediction_dict=first_prediction_dict,
        second_prediction_dict=second_prediction_dict,
        top_match_dir_name=top_match_dir_name)
    print(SEPARATOR_STRING)

    observed_labels = first_prediction_dict[prediction_io.OBSERVED_LABELS_KEY]

    first_model_file_name = first_prediction_dict[prediction_io.MODEL_FILE_KEY]
    first_full_id_strings = first_prediction_dict[prediction_io.STORM_IDS_KEY]
    first_storm_times_unix_sec = first_prediction_dict[
        prediction_io.STORM_TIMES_KEY]
    first_probabilities = first_prediction_dict[
        prediction_io.PROBABILITY_MATRIX_KEY][:, 1]

    second_model_file_name = second_prediction_dict[
        prediction_io.MODEL_FILE_KEY]
    second_full_id_strings = second_prediction_dict[prediction_io.STORM_IDS_KEY]
    second_storm_times_unix_sec = second_prediction_dict[
        prediction_io.STORM_TIMES_KEY]
    second_probabilities = second_prediction_dict[
        prediction_io.PROBABILITY_MATRIX_KEY][:, 1]

    if num_disagreements > 0:
        second_high_indices, first_high_indices = (
            model_activation.get_hilo_activation_examples(
                storm_activations=second_probabilities - first_probabilities,
                num_low_activation_examples=num_disagreements,
                num_high_activation_examples=num_disagreements,
                unique_storm_cells=unique_storm_cells,
                full_storm_id_strings=first_full_id_strings
            )
        )

        # Print summary to command window.
        this_mean_diff = numpy.mean(
            second_probabilities[second_high_indices] -
            first_probabilities[second_high_indices]
        )

        print((
            'Average prob difference for {0:d} worst disagreements with second '
            'model higher: {1:.3f}'
        ).format(
            num_disagreements, this_mean_diff
        ))

        this_mean_diff = numpy.mean(
            second_probabilities[first_high_indices] -
            first_probabilities[first_high_indices]
        )

        print((
            'Average prob difference for {0:d} worst disagreements with first '
            'model higher: {1:.3f}'
        ).format(
            num_disagreements, this_mean_diff
        ))

        # Write file.
        this_activation_file_name = '{0:s}/low_disagreement_examples.p'.format(
            first_output_dir_name)

        print((
            'Writing disagreements (second model higher) to: "{0:s}"...'
        ).format(
            this_activation_file_name
        ))

        this_activation_matrix = numpy.reshape(
            first_probabilities[second_high_indices],
            (len(second_high_indices), 1)
        )

        model_activation.write_file(
            pickle_file_name=this_activation_file_name,
            activation_matrix=this_activation_matrix,
            full_id_strings=[
                first_full_id_strings[j] for j in second_high_indices
            ],
            storm_times_unix_sec=first_storm_times_unix_sec[
                second_high_indices],
            model_file_name=first_model_file_name,
            component_type_string=CLASS_COMPONENT_STRING, target_class=1
        )

        # Write file.
        this_activation_file_name = '{0:s}/high_disagreement_examples.p'.format(
            second_output_dir_name)

        print((
            'Writing disagreements (second model higher) to: "{0:s}"...'
        ).format(
            this_activation_file_name
        ))

        this_activation_matrix = numpy.reshape(
            second_probabilities[second_high_indices],
            (len(second_high_indices), 1)
        )

        model_activation.write_file(
            pickle_file_name=this_activation_file_name,
            activation_matrix=this_activation_matrix,
            full_id_strings=[
                second_full_id_strings[j] for j in second_high_indices
            ],
            storm_times_unix_sec=second_storm_times_unix_sec[
                second_high_indices],
            model_file_name=second_model_file_name,
            component_type_string=CLASS_COMPONENT_STRING, target_class=1
        )

        # Write file.
        this_activation_file_name = '{0:s}/high_disagreement_examples.p'.format(
            first_output_dir_name)

        print((
            'Writing disagreements (first model higher) to: "{0:s}"...'
        ).format(
            this_activation_file_name
        ))

        this_activation_matrix = numpy.reshape(
            first_probabilities[first_high_indices],
            (len(first_high_indices), 1)
        )

        model_activation.write_file(
            pickle_file_name=this_activation_file_name,
            activation_matrix=this_activation_matrix,
            full_id_strings=[
                first_full_id_strings[j] for j in first_high_indices
            ],
            storm_times_unix_sec=first_storm_times_unix_sec[first_high_indices],
            model_file_name=first_model_file_name,
            component_type_string=CLASS_COMPONENT_STRING, target_class=1
        )

        # Write file.
        this_activation_file_name = '{0:s}/low_disagreement_examples.p'.format(
            second_output_dir_name)

        print((
            'Writing disagreements (first model higher) to: "{0:s}"...'
        ).format(
            this_activation_file_name
        ))

        this_activation_matrix = numpy.reshape(
            second_probabilities[first_high_indices],
            (len(first_high_indices), 1)
        )

        model_activation.write_file(
            pickle_file_name=this_activation_file_name,
            activation_matrix=this_activation_matrix,
            full_id_strings=[
                second_full_id_strings[j] for j in first_high_indices
            ],
            storm_times_unix_sec=second_storm_times_unix_sec[
                first_high_indices],
            model_file_name=second_model_file_name,
            component_type_string=CLASS_COMPONENT_STRING, target_class=1
        )

    if num_hits + num_misses + num_false_alarms + num_correct_nulls == 0:
        return

    mean_probabilities = 0.5 * (first_probabilities + second_probabilities)

    ct_extreme_dict = model_activation.get_contingency_table_extremes(
        storm_activations=mean_probabilities,
        storm_target_values=observed_labels,
        num_hits=num_hits, num_misses=num_misses,
        num_false_alarms=num_false_alarms, num_correct_nulls=num_correct_nulls,
        unique_storm_cells=unique_storm_cells,
        full_storm_id_strings=first_full_id_strings)

    hit_indices = ct_extreme_dict[model_activation.HIT_INDICES_KEY]
    miss_indices = ct_extreme_dict[model_activation.MISS_INDICES_KEY]
    false_alarm_indices = ct_extreme_dict[
        model_activation.FALSE_ALARM_INDICES_KEY]
    correct_null_indices = ct_extreme_dict[
        model_activation.CORRECT_NULL_INDICES_KEY]

    if num_hits > 0:
        print((
            'Mean probability from first and second model for {0:d} best hits: '
            '{1:.3f}, {2:.3f}'
        ).format(
            num_hits, numpy.mean(first_probabilities[hit_indices]),
            numpy.mean(second_probabilities[hit_indices])
        ))

        this_activation_file_name = '{0:s}/best_hits.p'.format(
            first_output_dir_name)

        print('Writing best hits to: "{0:s}"...'.format(
            this_activation_file_name
        ))
        this_activation_matrix = numpy.reshape(
            first_probabilities[hit_indices], (len(hit_indices), 1)
        )

        model_activation.write_file(
            pickle_file_name=this_activation_file_name,
            activation_matrix=this_activation_matrix,
            full_id_strings=[first_full_id_strings[j] for j in hit_indices],
            storm_times_unix_sec=first_storm_times_unix_sec[hit_indices],
            model_file_name=first_model_file_name,
            component_type_string=CLASS_COMPONENT_STRING, target_class=1
        )

        this_activation_file_name = '{0:s}/best_hits.p'.format(
            second_output_dir_name)

        print('Writing best hits to: "{0:s}"...'.format(
            this_activation_file_name
        ))
        this_activation_matrix = numpy.reshape(
            second_probabilities[hit_indices], (len(hit_indices), 1)
        )

        model_activation.write_file(
            pickle_file_name=this_activation_file_name,
            activation_matrix=this_activation_matrix,
            full_id_strings=[second_full_id_strings[j] for j in hit_indices],
            storm_times_unix_sec=second_storm_times_unix_sec[hit_indices],
            model_file_name=second_model_file_name,
            component_type_string=CLASS_COMPONENT_STRING, target_class=1
        )

    if miss_indices > 0:
        print((
            'Mean probability from first and second model for {0:d} worst '
            'misses: {1:.3f}, {2:.3f}'
        ).format(
            num_misses, numpy.mean(first_probabilities[miss_indices]),
            numpy.mean(second_probabilities[miss_indices])
        ))

        this_activation_file_name = '{0:s}/worst_misses.p'.format(
            first_output_dir_name)

        print('Writing worst misses to: "{0:s}"...'.format(
            this_activation_file_name
        ))
        this_activation_matrix = numpy.reshape(
            first_probabilities[miss_indices], (len(miss_indices), 1)
        )

        model_activation.write_file(
            pickle_file_name=this_activation_file_name,
            activation_matrix=this_activation_matrix,
            full_id_strings=[first_full_id_strings[j] for j in miss_indices],
            storm_times_unix_sec=first_storm_times_unix_sec[miss_indices],
            model_file_name=first_model_file_name,
            component_type_string=CLASS_COMPONENT_STRING, target_class=1
        )

        this_activation_file_name = '{0:s}/worst_misses.p'.format(
            second_output_dir_name)

        print('Writing worst misses to: "{0:s}"...'.format(
            this_activation_file_name
        ))
        this_activation_matrix = numpy.reshape(
            second_probabilities[miss_indices], (len(miss_indices), 1)
        )

        model_activation.write_file(
            pickle_file_name=this_activation_file_name,
            activation_matrix=this_activation_matrix,
            full_id_strings=[second_full_id_strings[j] for j in miss_indices],
            storm_times_unix_sec=second_storm_times_unix_sec[miss_indices],
            model_file_name=second_model_file_name,
            component_type_string=CLASS_COMPONENT_STRING, target_class=1
        )

    if num_false_alarms > 0:
        print((
            'Mean probability from first and second model for {0:d} worst '
            'false alarms: {1:.3f}, {2:.3f}'
        ).format(
            num_false_alarms,
            numpy.mean(first_probabilities[false_alarm_indices]),
            numpy.mean(second_probabilities[false_alarm_indices])
        ))

        this_activation_file_name = '{0:s}/worst_false_alarms.p'.format(
            first_output_dir_name)

        print('Writing worst false alarms to: "{0:s}"...'.format(
            this_activation_file_name
        ))
        this_activation_matrix = numpy.reshape(
            first_probabilities[false_alarm_indices],
            (len(false_alarm_indices), 1)
        )

        model_activation.write_file(
            pickle_file_name=this_activation_file_name,
            activation_matrix=this_activation_matrix,
            full_id_strings=[
                first_full_id_strings[j] for j in false_alarm_indices
            ],
            storm_times_unix_sec=first_storm_times_unix_sec[
                false_alarm_indices],
            model_file_name=first_model_file_name,
            component_type_string=CLASS_COMPONENT_STRING, target_class=1
        )

        this_activation_file_name = '{0:s}/worst_false_alarms.p'.format(
            second_output_dir_name)

        print('Writing worst false alarms to: "{0:s}"...'.format(
            this_activation_file_name
        ))
        this_activation_matrix = numpy.reshape(
            second_probabilities[false_alarm_indices],
            (len(false_alarm_indices), 1)
        )

        model_activation.write_file(
            pickle_file_name=this_activation_file_name,
            activation_matrix=this_activation_matrix,
            full_id_strings=[
                second_full_id_strings[j] for j in false_alarm_indices
            ],
            storm_times_unix_sec=second_storm_times_unix_sec[
                false_alarm_indices],
            model_file_name=second_model_file_name,
            component_type_string=CLASS_COMPONENT_STRING, target_class=1
        )

    if num_correct_nulls > 0:
        print((
            'Mean probability from first and second model for {0:d} best '
            'correct nulls: {1:.3f}, {2:.3f}'
        ).format(
            num_correct_nulls,
            numpy.mean(first_probabilities[correct_null_indices]),
            numpy.mean(second_probabilities[correct_null_indices])
        ))

        this_activation_file_name = '{0:s}/best_correct_nulls.p'.format(
            first_output_dir_name)

        print('Writing best correct nulls to: "{0:s}"...'.format(
            this_activation_file_name
        ))
        this_activation_matrix = numpy.reshape(
            first_probabilities[correct_null_indices],
            (len(correct_null_indices), 1)
        )

        model_activation.write_file(
            pickle_file_name=this_activation_file_name,
            activation_matrix=this_activation_matrix,
            full_id_strings=[
                first_full_id_strings[j] for j in correct_null_indices
            ],
            storm_times_unix_sec=first_storm_times_unix_sec[
                correct_null_indices],
            model_file_name=first_model_file_name,
            component_type_string=CLASS_COMPONENT_STRING, target_class=1
        )

        this_activation_file_name = '{0:s}/best_correct_nulls.p'.format(
            second_output_dir_name)

        print('Writing best correct nulls to: "{0:s}"...'.format(
            this_activation_file_name
        ))
        this_activation_matrix = numpy.reshape(
            second_probabilities[correct_null_indices],
            (len(correct_null_indices), 1)
        )

        model_activation.write_file(
            pickle_file_name=this_activation_file_name,
            activation_matrix=this_activation_matrix,
            full_id_strings=[
                second_full_id_strings[j] for j in correct_null_indices
            ],
            storm_times_unix_sec=second_storm_times_unix_sec[
                correct_null_indices],
            model_file_name=second_model_file_name,
            component_type_string=CLASS_COMPONENT_STRING, target_class=1
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_names=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FILES_ARG_NAME),
        top_match_dir_name=getattr(INPUT_ARG_OBJECT, MATCH_DIR_ARG_NAME),
        unique_storm_cells=bool(getattr(
            INPUT_ARG_OBJECT, UNIQUE_CELLS_ARG_NAME
        )),
        num_hits=getattr(INPUT_ARG_OBJECT, NUM_HITS_ARG_NAME),
        num_misses=getattr(INPUT_ARG_OBJECT, NUM_MISSES_ARG_NAME),
        num_false_alarms=getattr(INPUT_ARG_OBJECT, NUM_FALSE_ALARMS_ARG_NAME),
        num_correct_nulls=getattr(INPUT_ARG_OBJECT, NUM_CORRECT_NULLS_ARG_NAME),
        num_disagreements=getattr(INPUT_ARG_OBJECT, NUM_DISAGREEMENTS_ARG_NAME),
        output_dir_names=getattr(INPUT_ARG_OBJECT, OUTPUT_DIRS_ARG_NAME)
    )
