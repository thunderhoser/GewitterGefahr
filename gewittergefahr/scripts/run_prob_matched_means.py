"""Runs probability-matched means (PMM).

Specifically, this script uses PMM to composite one of the following:

- set of saliency maps
- set of class-activation maps (CAMs)

When compositing saliency maps or CAMs, this script also composites the
accompanying predictor fields (input to the convnet that generated the saliency
maps or CAMs), again using PMM.
"""

import argparse
import numpy
from gewittergefahr.gg_utils import prob_matched_means as pmm
from gewittergefahr.deep_learning import saliency_maps
from gewittergefahr.deep_learning import gradcam

SALIENCY_FILE_ARG_NAME = 'input_saliency_file_name'
GRADCAM_FILE_ARG_NAME = 'input_gradcam_file_name'
MAX_PERCENTILE_ARG_NAME = 'max_percentile_level'
THRESHOLD_INDEX_ARG_NAME = 'radar_channel_idx_for_thres'
THRESHOLD_VALUE_ARG_NAME = 'threshold_value'
THRESHOLD_TYPE_ARG_NAME = 'threshold_type_string'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

SALIENCY_FILE_HELP_STRING = (
    'Path to saliency file (will be read by `saliency_maps.read_standard_file`)'
    '.  If you want to composite class-activation maps instead, leave this '
    'argument alone.')

GRADCAM_FILE_HELP_STRING = (
    'Path to Grad-CAM file (will be read by `gradcam.read_standard_file`).  If '
    'you want to composite saliency maps instead of class-activation maps, '
    'leave this argument alone.')

MAX_PERCENTILE_HELP_STRING = (
    'Max percentile used in PMM procedure.  See '
    '`prob_matched_means.run_pmm_one_variable` for details.')

THRESHOLD_INDEX_HELP_STRING = (
    'Index of radar channel used for thresholding in PMM procedure.  See '
    '`prob_matched_means.run_pmm_one_variable` for details.  If you do not want'
    ' thresholding, leave this argument alone.')

THRESHOLD_VALUE_HELP_STRING = (
    'Threshold value used in PMM procedure.  See '
    '`prob_matched_means.run_pmm_one_variable` for details.  If you do not want'
    ' thresholding, leave this argument alone.')

THRESHOLD_TYPE_HELP_STRING = (
    'Thresholding type used in PMM procedure.  See '
    '`prob_matched_means.run_pmm_one_variable` for details.  If you do not want'
    ' thresholding, leave this argument alone.')

OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  If PMM is run on saliency maps, output file will be '
    'written by `saliency_maps.write_pmm_file`.  If run on class-activation'
    'maps, output file will be written by `gradcam.write_pmm_file`.')

DEFAULT_THRESHOLD_VALUE = 0.
DEFAULT_THRESHOLD_TYPE_STRING = pmm.MINIMUM_STRING + ''

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + SALIENCY_FILE_ARG_NAME, type=str, required=False, default='',
    help=SALIENCY_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + GRADCAM_FILE_ARG_NAME, type=str, required=False, default='',
    help=GRADCAM_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False,
    default=pmm.DEFAULT_MAX_PERCENTILE_LEVEL, help=MAX_PERCENTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + THRESHOLD_INDEX_ARG_NAME, type=int, required=False, default=-1,
    help=THRESHOLD_INDEX_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + THRESHOLD_VALUE_ARG_NAME, type=float, required=False,
    default=DEFAULT_THRESHOLD_VALUE, help=THRESHOLD_VALUE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + THRESHOLD_TYPE_ARG_NAME, type=str, required=False,
    default=DEFAULT_THRESHOLD_TYPE_STRING, help=THRESHOLD_TYPE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=False, default='',
    help=OUTPUT_FILE_HELP_STRING)


def _run(input_saliency_file_name, input_gradcam_file_name,
         max_percentile_level,
         radar_channel_idx_for_thres, threshold_value, threshold_type_string,
         output_file_name):
    """Runs probability-matched means (PMM).

    This is effectively the main method.

    :param input_saliency_file_name: See documentation at top of file.
    :param input_gradcam_file_name: Same.
    :param max_percentile_level: Same.
    :param radar_channel_idx_for_thres: Same.
    :param threshold_value: Same.
    :param threshold_type_string: Same.
    :param output_file_name: Same.
    """

    if input_saliency_file_name in ['', 'None']:
        input_saliency_file_name = None
    if input_gradcam_file_name in ['', 'None']:
        input_gradcam_file_name = None

    if radar_channel_idx_for_thres < 0:
        radar_channel_idx_for_thres = None
        threshold_value = None
        threshold_type_string = None

    if input_saliency_file_name is None:
        print 'Reading data from: "{0:s}"...'.format(input_gradcam_file_name)
        gradcam_dict = gradcam.read_standard_file(input_gradcam_file_name)
        list_of_input_matrices = gradcam_dict[gradcam.INPUT_MATRICES_KEY]
    else:
        print 'Reading data from: "{0:s}"...'.format(input_saliency_file_name)
        list_of_input_matrices, list_of_saliency_matrices = (
            saliency_maps.read_standard_file(input_saliency_file_name)[:2]
        )

    print 'Running PMM on denormalized predictor matrices...'

    num_input_matrices = len(list_of_input_matrices)
    list_of_mean_input_matrices = [None] * num_input_matrices
    pmm_metadata_dict = None
    threshold_count_matrix = None

    for i in range(num_input_matrices):
        if i == 0:
            list_of_mean_input_matrices[i], threshold_count_matrix = (
                pmm.run_pmm_many_variables(
                    input_matrix=list_of_input_matrices[i],
                    max_percentile_level=max_percentile_level,
                    threshold_var_index=radar_channel_idx_for_thres,
                    threshold_value=threshold_value,
                    threshold_type_string=threshold_type_string)
            )

            pmm_metadata_dict = pmm.check_input_args(
                input_matrix=list_of_input_matrices[i],
                max_percentile_level=max_percentile_level,
                threshold_var_index=radar_channel_idx_for_thres,
                threshold_value=threshold_value,
                threshold_type_string=threshold_type_string)
        else:
            list_of_mean_input_matrices[i] = pmm.run_pmm_many_variables(
                input_matrix=list_of_input_matrices[i],
                max_percentile_level=max_percentile_level
            )[0]

    this_mean_matrix = list_of_mean_input_matrices[0][..., 0]
    this_num_rows = this_mean_matrix.shape[0]
    this_num_columns = this_mean_matrix.shape[1]

    for i in range(this_num_rows):
        for j in range(this_num_columns):
            print this_mean_matrix[i, j]
        print '\n'

    if input_saliency_file_name is None:
        class_activation_matrix = gradcam_dict[gradcam.CLASS_ACTIVATIONS_KEY]
        ggradcam_output_matrix = gradcam_dict[gradcam.GUIDED_GRADCAM_KEY]

        print 'Running PMM on class-activation matrices...'
        class_activation_matrix = numpy.expand_dims(
            class_activation_matrix, axis=-1)

        mean_class_activation_matrix = pmm.run_pmm_many_variables(
            input_matrix=class_activation_matrix,
            max_percentile_level=max_percentile_level
        )[0]

        mean_class_activation_matrix = mean_class_activation_matrix[..., 0]

        print 'Running PMM on output matrices from guided Grad-CAM...'
        mean_ggradcam_output_matrix = pmm.run_pmm_many_variables(
            input_matrix=ggradcam_output_matrix,
            max_percentile_level=max_percentile_level
        )[0]

        print 'Writing output to: "{0:s}"...'.format(output_file_name)
        gradcam.write_pmm_file(
            pickle_file_name=output_file_name,
            list_of_mean_input_matrices=list_of_mean_input_matrices,
            mean_class_activation_matrix=mean_class_activation_matrix,
            mean_ggradcam_output_matrix=mean_ggradcam_output_matrix,
            threshold_count_matrix=threshold_count_matrix,
            standard_gradcam_file_name=input_gradcam_file_name,
            pmm_metadata_dict=pmm_metadata_dict)

        return

    print 'Running PMM on saliency matrices...'

    num_input_matrices = len(list_of_input_matrices)
    list_of_mean_saliency_matrices = [None] * num_input_matrices

    for i in range(num_input_matrices):
        list_of_mean_saliency_matrices[i] = pmm.run_pmm_many_variables(
            input_matrix=list_of_saliency_matrices[i],
            max_percentile_level=max_percentile_level
        )[0]

    print 'Writing output to: "{0:s}"...'.format(output_file_name)
    saliency_maps.write_pmm_file(
        pickle_file_name=output_file_name,
        list_of_mean_input_matrices=list_of_mean_input_matrices,
        list_of_mean_saliency_matrices=list_of_mean_saliency_matrices,
        threshold_count_matrix=threshold_count_matrix,
        standard_saliency_file_name=input_saliency_file_name,
        pmm_metadata_dict=pmm_metadata_dict)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_saliency_file_name=getattr(INPUT_ARG_OBJECT,
                                         SALIENCY_FILE_ARG_NAME),
        input_gradcam_file_name=getattr(INPUT_ARG_OBJECT,
                                        GRADCAM_FILE_ARG_NAME),
        max_percentile_level=getattr(INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME),
        radar_channel_idx_for_thres=getattr(
            INPUT_ARG_OBJECT, THRESHOLD_INDEX_ARG_NAME),
        threshold_value=getattr(INPUT_ARG_OBJECT, THRESHOLD_VALUE_ARG_NAME),
        threshold_type_string=getattr(
            INPUT_ARG_OBJECT, THRESHOLD_TYPE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
