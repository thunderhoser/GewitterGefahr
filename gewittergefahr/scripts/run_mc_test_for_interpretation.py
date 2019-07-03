"""Runs Monte Carlo significance test for interpretation output."""

import copy
import argparse
import numpy
from gewittergefahr.gg_utils import monte_carlo
from gewittergefahr.gg_utils import prob_matched_means as pmm
from gewittergefahr.deep_learning import gradcam
from gewittergefahr.deep_learning import saliency_maps
from gewittergefahr.deep_learning import backwards_optimization as backwards_opt

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

SALIENCY_STRING = 'saliency'
GRADCAM_STRING = 'gradcam'
BACKWARDS_OPT_STRING = 'backwards_opt'

VALID_INTERPRETATION_TYPE_STRINGS = [
    SALIENCY_STRING, GRADCAM_STRING, BACKWARDS_OPT_STRING
]

INTERPRETATION_TYPE_ARG_NAME = 'interpretation_type_string'
BASELINE_FILE_ARG_NAME = 'baseline_file_name'
TRIAL_FILE_ARG_NAME = 'trial_file_name'
MAX_PERCENTILE_ARG_NAME = 'max_pmm_percentile_level'
NUM_ITERATIONS_ARG_NAME = 'num_iterations'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INTERPRETATION_TYPE_HELP_STRING = (
    'Interpretation type.  Must be in the following list:\n{0:s}'
).format(str(VALID_INTERPRETATION_TYPE_STRINGS))

BASELINE_FILE_HELP_STRING = (
    'Path to file with interpretation maps for baseline set.  Will be read by '
    '`saliency_maps.read_standard_file`, `gradcam.read_standard_file`, or '
    '`backwards_optimization.read_standard_file`.')

TRIAL_FILE_HELP_STRING = (
    'Path to file with interpretation maps for baseline set.  Will be read by '
    '`saliency_maps.read_standard_file`, `gradcam.read_standard_file`, or '
    '`backwards_optimization.read_standard_file`.  This script will find grid '
    'points in the trial set where interpretation value is significantly '
    'different than in the baseline set.')

MAX_PERCENTILE_HELP_STRING = (
    'Max percentile used in PMM procedure.  See '
    '`prob_matched_means.run_pmm_one_variable` for details.')

NUM_ITERATIONS_HELP_STRING = 'Number of Monte Carlo iterations.'

CONFIDENCE_LEVEL_HELP_STRING = (
    'Confidence level for statistical significance.  Grid point G in the trial '
    'set will be deemed sig diff than baseline set iff its interpretation value'
    ' is outside this confidence interval from the Monte Carlo iterations.  For'
    ' example, if the confidence level is 0.95, grid point G will be '
    'significant iff its value is outside the [2.5th, 97.5th] percentiles from '
    'Monte Carlo iterations.')

OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Results will be written here by '
    '`saliency_maps.write_pmm_file`, `gradcam.write_pmm_file`, or '
    '`backwards_optimization.write_pmm_file`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INTERPRETATION_TYPE_ARG_NAME, type=str, required=True,
    help=INTERPRETATION_TYPE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + BASELINE_FILE_ARG_NAME, type=str, required=True,
    help=BASELINE_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRIAL_FILE_ARG_NAME, type=str, required=True,
    help=TRIAL_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False,
    default=pmm.DEFAULT_MAX_PERCENTILE_LEVEL, help=MAX_PERCENTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_ITERATIONS_ARG_NAME, type=int, required=False, default=5000,
    help=NUM_ITERATIONS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=0.95,
    help=CONFIDENCE_LEVEL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _run(interpretation_type_string, baseline_file_name, trial_file_name,
         max_pmm_percentile_level, num_iterations, confidence_level,
         output_file_name):
    """Runs Monte Carlo significance test for interpretation output.

    This is effectively the main method.

    :param interpretation_type_string: See documentation at top of file.
    :param baseline_file_name: Same.
    :param trial_file_name: Same.
    :param max_pmm_percentile_level: Same.
    :param num_iterations: Same.
    :param confidence_level: Same.
    :param output_file_name: Same.
    :raises: ValueError: if
        `interpretation_type_string not in VALID_INTERPRETATION_TYPE_STRINGS`.
    """

    if interpretation_type_string not in VALID_INTERPRETATION_TYPE_STRINGS:
        error_string = (
            '\n{0:s}\nValid interpretation types (listed above) do not include '
            '"{1:s}".'
        ).format(
            str(VALID_INTERPRETATION_TYPE_STRINGS), interpretation_type_string
        )

        raise ValueError(error_string)

    print('Reading baseline set from: "{0:s}"...'.format(baseline_file_name))

    if interpretation_type_string == SALIENCY_STRING:
        baseline_dict = saliency_maps.read_standard_file(baseline_file_name)
    elif interpretation_type_string == GRADCAM_STRING:
        baseline_dict = gradcam.read_standard_file(baseline_file_name)
    else:
        baseline_dict = backwards_opt.read_standard_file(baseline_file_name)

    print('Reading trial set from: "{0:s}"...'.format(trial_file_name))
    monte_carlo_dict = None
    gradcam_monte_carlo_dict = None
    ggradcam_monte_carlo_dict = None

    if interpretation_type_string == SALIENCY_STRING:
        trial_dict = saliency_maps.read_standard_file(trial_file_name)

        monte_carlo_dict = monte_carlo.run_monte_carlo_test(
            list_of_baseline_matrices=baseline_dict[
                saliency_maps.SALIENCY_MATRICES_KEY],
            list_of_trial_matrices=trial_dict[
                saliency_maps.SALIENCY_MATRICES_KEY],
            max_pmm_percentile_level=max_pmm_percentile_level,
            num_iterations=num_iterations, confidence_level=confidence_level)

        monte_carlo_dict[monte_carlo.BASELINE_FILE_KEY] = baseline_file_name
        list_of_input_matrices = trial_dict[saliency_maps.INPUT_MATRICES_KEY]

    elif interpretation_type_string == GRADCAM_STRING:
        trial_dict = gradcam.read_standard_file(trial_file_name)

        gradcam_monte_carlo_dict = monte_carlo.run_monte_carlo_test(
            list_of_baseline_matrices=[
                baseline_dict[gradcam.CLASS_ACTIVATIONS_KEY]
            ],
            list_of_trial_matrices=[trial_dict[gradcam.CLASS_ACTIVATIONS_KEY]],
            max_pmm_percentile_level=max_pmm_percentile_level,
            num_iterations=num_iterations, confidence_level=confidence_level)

        ggradcam_monte_carlo_dict = monte_carlo.run_monte_carlo_test(
            list_of_baseline_matrices=[
                baseline_dict[gradcam.GUIDED_GRADCAM_KEY]
            ],
            list_of_trial_matrices=[trial_dict[gradcam.GUIDED_GRADCAM_KEY]],
            max_pmm_percentile_level=max_pmm_percentile_level,
            num_iterations=num_iterations, confidence_level=confidence_level)

        gradcam_monte_carlo_dict[
            monte_carlo.BASELINE_FILE_KEY] = baseline_file_name
        ggradcam_monte_carlo_dict[
            monte_carlo.BASELINE_FILE_KEY] = baseline_file_name
        list_of_input_matrices = trial_dict[gradcam.INPUT_MATRICES_KEY]

    else:
        trial_dict = backwards_opt.read_standard_file(trial_file_name)

        monte_carlo_dict = monte_carlo.run_monte_carlo_test(
            list_of_baseline_matrices=baseline_dict[
                backwards_opt.OPTIMIZED_MATRICES_KEY],
            list_of_trial_matrices=trial_dict[
                backwards_opt.OPTIMIZED_MATRICES_KEY],
            max_pmm_percentile_level=max_pmm_percentile_level,
            num_iterations=num_iterations, confidence_level=confidence_level)

        monte_carlo_dict[monte_carlo.BASELINE_FILE_KEY] = baseline_file_name
        list_of_input_matrices = trial_dict[backwards_opt.INIT_FUNCTION_KEY]

    print(SEPARATOR_STRING)

    num_matrices = len(list_of_input_matrices)
    list_of_mean_input_matrices = [None] * num_matrices

    for i in range(num_matrices):
        list_of_mean_input_matrices[i] = pmm.run_pmm_many_variables(
            input_matrix=list_of_input_matrices[i],
            max_percentile_level=max_pmm_percentile_level
        )[0]

    pmm_metadata_dict = pmm.check_input_args(
        input_matrix=list_of_input_matrices[0],
        max_percentile_level=max_pmm_percentile_level,
        threshold_var_index=None, threshold_value=None,
        threshold_type_string=None)

    print('Writing results to: "{0:s}"...'.format(output_file_name))

    if interpretation_type_string == SALIENCY_STRING:
        saliency_maps.write_pmm_file(
            pickle_file_name=output_file_name,
            list_of_mean_input_matrices=list_of_mean_input_matrices,
            list_of_mean_saliency_matrices=copy.deepcopy(
                monte_carlo_dict[monte_carlo.TRIAL_PMM_MATRICES_KEY]
            ),
            threshold_count_matrix=None,
            model_file_name=trial_dict[saliency_maps.MODEL_FILE_KEY],
            standard_saliency_file_name=trial_file_name,
            pmm_metadata_dict=pmm_metadata_dict,
            monte_carlo_dict=monte_carlo_dict)

    elif interpretation_type_string == GRADCAM_STRING:
        gradcam.write_pmm_file(
            pickle_file_name=output_file_name,
            list_of_mean_input_matrices=list_of_mean_input_matrices,
            mean_class_activation_matrix=copy.deepcopy(
                gradcam_monte_carlo_dict[monte_carlo.TRIAL_PMM_MATRICES_KEY][0]
            ),
            mean_ggradcam_output_matrix=copy.deepcopy(
                ggradcam_monte_carlo_dict[monte_carlo.TRIAL_PMM_MATRICES_KEY][0]
            ),
            threshold_count_matrix=None,
            model_file_name=trial_dict[gradcam.MODEL_FILE_KEY],
            standard_gradcam_file_name=trial_file_name,
            pmm_metadata_dict=pmm_metadata_dict,
            gradcam_monte_carlo_dict=gradcam_monte_carlo_dict,
            ggradcam_monte_carlo_dict=ggradcam_monte_carlo_dict)

    else:
        backwards_opt.write_pmm_file(
            pickle_file_name=output_file_name,
            list_of_mean_input_matrices=list_of_mean_input_matrices,
            list_of_mean_optimized_matrices=copy.deepcopy(
                monte_carlo_dict[monte_carlo.TRIAL_PMM_MATRICES_KEY]
            ),
            mean_initial_activation=numpy.mean(
                trial_dict[backwards_opt.INITIAL_ACTIVATIONS_KEY]
            ),
            mean_final_activation=numpy.mean(
                trial_dict[backwards_opt.FINAL_ACTIVATIONS_KEY]
            ),
            threshold_count_matrix=None,
            model_file_name=trial_dict[backwards_opt.MODEL_FILE_KEY],
            standard_bwo_file_name=trial_file_name,
            pmm_metadata_dict=pmm_metadata_dict,
            monte_carlo_dict=monte_carlo_dict)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        interpretation_type_string=getattr(
            INPUT_ARG_OBJECT, INTERPRETATION_TYPE_ARG_NAME),
        baseline_file_name=getattr(INPUT_ARG_OBJECT, BASELINE_FILE_ARG_NAME),
        trial_file_name=getattr(INPUT_ARG_OBJECT, TRIAL_FILE_ARG_NAME),
        max_pmm_percentile_level=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME),
        num_iterations=getattr(INPUT_ARG_OBJECT, NUM_ITERATIONS_ARG_NAME),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME),
    )
