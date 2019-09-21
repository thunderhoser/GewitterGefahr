"""Evaluates CNN predictions."""

import argparse
import numpy
import pandas
from gewittergefahr.gg_utils import model_evaluation as model_eval
from gewittergefahr.gg_utils import bootstrapping
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import prediction_io
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

FORECAST_PRECISION = 1e-4
DUMMY_TARGET_NAME = 'tornado_lead-time=0000-3600sec_distance=00000-10000m'

PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
PROB_THRESHOLD_ARG_NAME = 'best_prob_threshold'
NUM_BOOTSTRAP_ARG_NAME = 'num_bootstrap_reps'
DS_FRACTIONS_ARG_NAME = 'downsampling_fractions'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

PREDICTION_FILE_HELP_STRING = (
    'Path to input file, containing CNN predictions.  Will be read by '
    '`prediction_io.read_ungridded_predictions`.')

PROB_THRESHOLD_HELP_STRING = (
    'Probability threshold (used to convert forecasts to deterministic).  If '
    'you make this negative, threshold will be determined on the fly (that '
    'which maximizes CSI).')

NUM_BOOTSTRAP_HELP_STRING = (
    'Number of bootstrap replicates.  If you do not want bootstrapping, leave '
    'this alone.')

DS_FRACTIONS_HELP_STRING = (
    'List of downsampling fractions.  This should contain two values (class 0, '
    'then class 1), summing to 1.0.  If you do not want downsampling, leave '
    'this alone.')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Results will be written here by '
    '`model_evaluation.write_evaluation`, to an exact location determined by '
    '`model_evaluation.find_file_from_prediction_file`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_FILE_ARG_NAME, type=str, required=True,
    help=PREDICTION_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PROB_THRESHOLD_ARG_NAME, type=float, required=False, default=-1.,
    help=PROB_THRESHOLD_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_BOOTSTRAP_ARG_NAME, type=int, required=False, default=1,
    help=NUM_BOOTSTRAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + DS_FRACTIONS_ARG_NAME, type=float, nargs=2, required=False,
    default=[-1, -1], help=DS_FRACTIONS_HELP_STRING
)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _compute_scores(
        forecast_probabilities, observed_labels, num_bootstrap_reps,
        output_file_name, best_prob_threshold=None, downsampling_dict=None):
    """Computes evaluation scores.

    E = number of examples (storm objects)

    :param forecast_probabilities: length-E numpy array of forecast event
        probabilities.
    :param observed_labels: length-E numpy array of observations (1 for event,
        0 for non-event).
    :param num_bootstrap_reps: Number of bootstrap replicates.
    :param output_file_name: Path to output file (will be written by
        `model_evaluation.write_evaluation`).
    :param best_prob_threshold: Best probability threshold.  If None, will be
        determined on the fly.
    :param downsampling_dict: Dictionary with downsampling fractions.  See doc
        for `deep_learning_utils.sample_by_class`.  If this is None,
        downsampling will not be used.
    """

    num_examples = len(observed_labels)
    num_examples_by_class = numpy.unique(
        observed_labels, return_counts=True
    )[-1]

    print('Number of examples by class (no downsampling): {0:s}'.format(
        str(num_examples_by_class)
    ))

    positive_example_indices = numpy.where(observed_labels == 1)[0]
    negative_example_indices = numpy.where(observed_labels == 0)[0]

    if downsampling_dict is None:
        these_indices = numpy.linspace(
            0, num_examples - 1, num=num_examples, dtype=int)
    else:
        these_indices = dl_utils.sample_by_class(
            sampling_fraction_by_class_dict=downsampling_dict,
            target_name=DUMMY_TARGET_NAME, target_values=observed_labels,
            num_examples_total=num_examples)

        this_num_ex_by_class = numpy.unique(
            observed_labels[these_indices], return_counts=True
        )[-1]

        print('Number of examples by class (after downsampling): {0:s}'.format(
            str(this_num_ex_by_class)
        ))

    all_prob_thresholds = model_eval.get_binarization_thresholds(
        threshold_arg=model_eval.THRESHOLD_ARG_FOR_UNIQUE_FORECASTS,
        forecast_probabilities=forecast_probabilities[these_indices],
        forecast_precision=FORECAST_PRECISION)

    if best_prob_threshold is None:
        best_prob_threshold, best_csi = (
            model_eval.find_best_binarization_threshold(
                forecast_probabilities=forecast_probabilities[these_indices],
                observed_labels=observed_labels[these_indices],
                threshold_arg=all_prob_thresholds,
                criterion_function=model_eval.get_csi,
                optimization_direction=model_eval.MAX_OPTIMIZATION_STRING)
        )
    else:
        these_forecast_labels = model_eval.binarize_forecast_probs(
            forecast_probabilities=forecast_probabilities[these_indices],
            binarization_threshold=best_prob_threshold)

        this_contingency_dict = model_eval.get_contingency_table(
            forecast_labels=these_forecast_labels,
            observed_labels=observed_labels[these_indices]
        )

        best_csi = model_eval.get_csi(this_contingency_dict)

    print((
        'Best probability threshold = {0:.4f} ... corresponding CSI = {1:.4f}'
    ).format(
        best_prob_threshold, best_csi
    ))

    list_of_evaluation_tables = []

    for i in range(num_bootstrap_reps):
        print((
            'Computing scores for {0:d}th of {1:d} bootstrap replicates...'
        ).format(
            i + 1, num_bootstrap_reps
        ))

        if num_bootstrap_reps == 1:
            if downsampling_dict is None:
                these_indices = numpy.linspace(
                    0, num_examples - 1, num=num_examples, dtype=int)
            else:
                these_indices = dl_utils.sample_by_class(
                    sampling_fraction_by_class_dict=downsampling_dict,
                    target_name=DUMMY_TARGET_NAME,
                    target_values=observed_labels,
                    num_examples_total=num_examples)
        else:
            if len(positive_example_indices) > 0:
                these_positive_indices = bootstrapping.draw_sample(
                    positive_example_indices
                )[0]
            else:
                these_positive_indices = numpy.array([], dtype=int)

            these_negative_indices = bootstrapping.draw_sample(
                negative_example_indices
            )[0]

            these_indices = numpy.concatenate((
                these_positive_indices, these_negative_indices))

            if downsampling_dict is not None:
                these_subindices = dl_utils.sample_by_class(
                    sampling_fraction_by_class_dict=downsampling_dict,
                    target_name=DUMMY_TARGET_NAME,
                    target_values=observed_labels[these_indices],
                    num_examples_total=num_examples)

                these_indices = these_indices[these_subindices]

        if downsampling_dict is not None:
            this_num_ex_by_class = numpy.unique(
                observed_labels[these_indices], return_counts=True
            )[-1]

            print('Number of examples by class: {0:s}'.format(
                str(this_num_ex_by_class)
            ))

        this_evaluation_table = model_eval.run_evaluation(
            forecast_probabilities=forecast_probabilities[these_indices],
            observed_labels=observed_labels[these_indices],
            best_prob_threshold=best_prob_threshold,
            all_prob_thresholds=all_prob_thresholds,
            climatology=numpy.mean(observed_labels[these_indices])
        )

        list_of_evaluation_tables.append(this_evaluation_table)

        if i == num_bootstrap_reps - 1:
            print(SEPARATOR_STRING)
        else:
            print(MINOR_SEPARATOR_STRING)

        if i == 0:
            continue

        list_of_evaluation_tables[-1] = list_of_evaluation_tables[-1].align(
            list_of_evaluation_tables[0], axis=1
        )[0]

    evaluation_table = pandas.concat(
        list_of_evaluation_tables, axis=0, ignore_index=True)

    print('Writing results to: "{0:s}"...'.format(output_file_name))

    model_eval.write_evaluation(
        pickle_file_name=output_file_name,
        forecast_probabilities=forecast_probabilities,
        observed_labels=observed_labels,
        best_prob_threshold=best_prob_threshold,
        all_prob_thresholds=all_prob_thresholds,
        evaluation_table=evaluation_table)


def _run(prediction_file_name, best_prob_threshold, num_bootstrap_reps,
         downsampling_fractions, output_dir_name):
    """Evaluates CNN predictions.

    This is effectively the main method.

    :param prediction_file_name: Same.
    :param best_prob_threshold: Same.
    :param num_bootstrap_reps: Same.
    :param downsampling_fractions: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if file contains no examples (storm objects).
    :raises: ValueError: if file contains multi-class predictions.
    """

    num_bootstrap_reps = max([num_bootstrap_reps, 1])
    if best_prob_threshold < 0:
        best_prob_threshold = None

    output_file_name = model_eval.find_file_from_prediction_file(
        input_prediction_file_name=prediction_file_name,
        output_dir_name=output_dir_name, raise_error_if_missing=False)

    file_system_utils.mkdir_recursive_if_necessary(
        file_name=output_file_name)

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_ungridded_predictions(
        prediction_file_name)

    observed_labels = prediction_dict[prediction_io.OBSERVED_LABELS_KEY]
    class_probability_matrix = prediction_dict[
        prediction_io.PROBABILITY_MATRIX_KEY]

    num_examples = len(observed_labels)
    num_classes = class_probability_matrix.shape[1]

    if num_examples == 0:
        raise ValueError('File contains no examples (storm objects).')

    if num_classes > 2:
        error_string = (
            'This script handles only binary, not {0:d}-class, classification.'
        ).format(num_classes)

        raise ValueError(error_string)

    forecast_probabilities = class_probability_matrix[:, -1]

    if numpy.any(downsampling_fractions <= 0):
        downsampling_dict = None
    else:
        downsampling_dict = {
            0: downsampling_fractions[0],
            1: downsampling_fractions[1]
        }

    _compute_scores(
        forecast_probabilities=forecast_probabilities,
        observed_labels=observed_labels, num_bootstrap_reps=num_bootstrap_reps,
        best_prob_threshold=best_prob_threshold,
        downsampling_dict=downsampling_dict, output_file_name=output_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FILE_ARG_NAME),
        best_prob_threshold=getattr(INPUT_ARG_OBJECT, PROB_THRESHOLD_ARG_NAME),
        num_bootstrap_reps=getattr(INPUT_ARG_OBJECT, NUM_BOOTSTRAP_ARG_NAME),
        downsampling_fractions=numpy.array(
            getattr(INPUT_ARG_OBJECT, DS_FRACTIONS_ARG_NAME), dtype=float
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
