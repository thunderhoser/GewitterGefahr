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
from gewittergefahr.deep_learning import backwards_optimization as backwards_opt
from gewittergefahr.deep_learning import novelty_detection

NONE_STRINGS = ['', 'None']

SALIENCY_FILE_ARG_NAME = 'input_saliency_file_name'
GRADCAM_FILE_ARG_NAME = 'input_gradcam_file_name'
BWO_FILE_ARG_NAME = 'input_bwo_file_name'
NOVELTY_FILE_ARG_NAME = 'input_novelty_file_name'
MAX_PERCENTILE_ARG_NAME = 'max_percentile_level'
THRESHOLD_INDEX_ARG_NAME = 'radar_channel_idx_for_thres'
THRESHOLD_VALUE_ARG_NAME = 'threshold_value'
THRESHOLD_TYPE_ARG_NAME = 'threshold_type_string'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

SALIENCY_FILE_HELP_STRING = (
    'Path to saliency file (will be read by `saliency_maps.read_standard_file`)'
    '.  If you are compositing something other than saliency maps, leave this '
    'argument alone.')

GRADCAM_FILE_HELP_STRING = (
    'Path to Grad-CAM file (will be read by `gradcam.read_standard_file`).  If '
    'you are compositing something other than class-activation maps, leave this'
    ' argument alone.')

BWO_FILE_HELP_STRING = (
    'Path to backwards-optimization file (will be read by '
    '`backwards_optimization.read_standard_file`).  If you are compositing '
    'something other than backwards-optimization results, leave this argument'
    'alone.')

NOVELTY_FILE_HELP_STRING = (
    'Path to novelty-detection file (will be read by '
    '`novelty_detection.read_standard_file`).  If you are compositing '
    'something other than novelty-detection results, leave this argument '
    'alone.')

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
    '--' + BWO_FILE_ARG_NAME, type=str, required=False, default='',
    help=BWO_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NOVELTY_FILE_ARG_NAME, type=str, required=False,
    default='', help=NOVELTY_FILE_HELP_STRING)

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


def _run(input_saliency_file_name, input_gradcam_file_name, input_bwo_file_name,
         input_novelty_file_name, max_percentile_level,
         radar_channel_idx_for_thres, threshold_value, threshold_type_string,
         output_file_name):
    """Runs probability-matched means (PMM).

    This is effectively the main method.

    :param input_saliency_file_name: See documentation at top of file.
    :param input_gradcam_file_name: Same.
    :param input_bwo_file_name: Same.
    :param input_novelty_file_name: Same.
    :param max_percentile_level: Same.
    :param radar_channel_idx_for_thres: Same.
    :param threshold_value: Same.
    :param threshold_type_string: Same.
    :param output_file_name: Same.
    """

    if input_saliency_file_name not in NONE_STRINGS:
        input_gradcam_file_name = None
        input_bwo_file_name = None
        input_novelty_file_name = None
    elif input_gradcam_file_name not in NONE_STRINGS:
        input_saliency_file_name = None
        input_bwo_file_name = None
        input_novelty_file_name = None
    elif input_bwo_file_name not in NONE_STRINGS:
        input_saliency_file_name = None
        input_gradcam_file_name = None
        input_novelty_file_name = None
    else:
        input_saliency_file_name = None
        input_gradcam_file_name = None
        input_bwo_file_name = None

    if radar_channel_idx_for_thres < 0:
        radar_channel_idx_for_thres = None
        threshold_value = None
        threshold_type_string = None

    if input_saliency_file_name is not None:
        print('Reading data from: "{0:s}"...'.format(input_saliency_file_name))

        saliency_dict = saliency_maps.read_standard_file(
            input_saliency_file_name)
        list_of_input_matrices = saliency_dict[saliency_maps.INPUT_MATRICES_KEY]

    elif input_gradcam_file_name is not None:
        print('Reading data from: "{0:s}"...'.format(input_gradcam_file_name))

        gradcam_dict = gradcam.read_standard_file(input_gradcam_file_name)
        list_of_input_matrices = gradcam_dict[gradcam.INPUT_MATRICES_KEY]

    elif input_bwo_file_name is not None:
        print('Reading data from: "{0:s}"...'.format(input_bwo_file_name))

        bwo_dictionary = backwards_opt.read_standard_file(input_bwo_file_name)
        list_of_input_matrices = bwo_dictionary[backwards_opt.INIT_FUNCTION_KEY]

    else:
        print('Reading data from: "{0:s}"...'.format(input_novelty_file_name))
        novelty_dict = novelty_detection.read_standard_file(
            input_novelty_file_name)

        list_of_input_matrices = novelty_dict[
            novelty_detection.TRIAL_INPUTS_KEY]
        novel_indices = novelty_dict[novelty_detection.NOVEL_INDICES_KEY]

        list_of_input_matrices = [
            a[novel_indices, ...] for a in list_of_input_matrices
        ]

    print('Running PMM on denormalized predictor matrices...')

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

    if input_saliency_file_name is not None:
        print('Running PMM on saliency matrices...')
        list_of_saliency_matrices = saliency_dict[
            saliency_maps.SALIENCY_MATRICES_KEY]

        num_input_matrices = len(list_of_input_matrices)
        list_of_mean_saliency_matrices = [None] * num_input_matrices

        for i in range(num_input_matrices):
            list_of_mean_saliency_matrices[i] = pmm.run_pmm_many_variables(
                input_matrix=list_of_saliency_matrices[i],
                max_percentile_level=max_percentile_level
            )[0]

        print('Writing output to: "{0:s}"...'.format(output_file_name))
        saliency_maps.write_pmm_file(
            pickle_file_name=output_file_name,
            list_of_mean_input_matrices=list_of_mean_input_matrices,
            list_of_mean_saliency_matrices=list_of_mean_saliency_matrices,
            threshold_count_matrix=threshold_count_matrix,
            model_file_name=saliency_dict[saliency_maps.MODEL_FILE_KEY],
            standard_saliency_file_name=input_saliency_file_name,
            pmm_metadata_dict=pmm_metadata_dict)

        return

    if input_gradcam_file_name is not None:
        print('Running PMM on class-activation matrices...')

        class_activation_matrix = gradcam_dict[gradcam.CLASS_ACTIVATIONS_KEY]
        ggradcam_output_matrix = gradcam_dict[gradcam.GUIDED_GRADCAM_KEY]
        class_activation_matrix = numpy.expand_dims(
            class_activation_matrix, axis=-1)

        mean_class_activation_matrix = pmm.run_pmm_many_variables(
            input_matrix=class_activation_matrix,
            max_percentile_level=max_percentile_level
        )[0]

        mean_class_activation_matrix = mean_class_activation_matrix[..., 0]

        print('Running PMM on output matrices from guided Grad-CAM...')
        mean_ggradcam_output_matrix = pmm.run_pmm_many_variables(
            input_matrix=ggradcam_output_matrix,
            max_percentile_level=max_percentile_level
        )[0]

        print('Writing output to: "{0:s}"...'.format(output_file_name))
        gradcam.write_pmm_file(
            pickle_file_name=output_file_name,
            list_of_mean_input_matrices=list_of_mean_input_matrices,
            mean_class_activation_matrix=mean_class_activation_matrix,
            mean_ggradcam_output_matrix=mean_ggradcam_output_matrix,
            threshold_count_matrix=threshold_count_matrix,
            model_file_name=gradcam_dict[gradcam.MODEL_FILE_KEY],
            standard_gradcam_file_name=input_gradcam_file_name,
            pmm_metadata_dict=pmm_metadata_dict)

        return

    if input_bwo_file_name is not None:
        print('Running PMM on backwards-optimization output...')
        list_of_optimized_matrices = bwo_dictionary[
            backwards_opt.OPTIMIZED_MATRICES_KEY]

        num_input_matrices = len(list_of_input_matrices)
        list_of_mean_optimized_matrices = [None] * num_input_matrices

        for i in range(num_input_matrices):
            list_of_mean_optimized_matrices[i] = pmm.run_pmm_many_variables(
                input_matrix=list_of_optimized_matrices[i],
                max_percentile_level=max_percentile_level
            )[0]

        mean_initial_activation = numpy.mean(
            bwo_dictionary[backwards_opt.INITIAL_ACTIVATIONS_KEY]
        )
        mean_final_activation = numpy.mean(
            bwo_dictionary[backwards_opt.FINAL_ACTIVATIONS_KEY]
        )

        print('Writing output to: "{0:s}"...'.format(output_file_name))
        backwards_opt.write_pmm_file(
            pickle_file_name=output_file_name,
            list_of_mean_input_matrices=list_of_mean_input_matrices,
            list_of_mean_optimized_matrices=list_of_mean_optimized_matrices,
            mean_initial_activation=mean_initial_activation,
            mean_final_activation=mean_final_activation,
            threshold_count_matrix=threshold_count_matrix,
            model_file_name=bwo_dictionary[backwards_opt.MODEL_FILE_KEY],
            standard_bwo_file_name=input_bwo_file_name,
            pmm_metadata_dict=pmm_metadata_dict)

        return

    print('Running PMM on novelty-detection output...')

    mean_novel_image_matrix_upconv = pmm.run_pmm_many_variables(
        input_matrix=novelty_dict[novelty_detection.NOVEL_IMAGES_UPCONV_KEY],
        max_percentile_level=max_percentile_level
    )[0]

    mean_novel_image_matrix_upconv_svd = pmm.run_pmm_many_variables(
        input_matrix=novelty_dict[
            novelty_detection.NOVEL_IMAGES_UPCONV_SVD_KEY],
        max_percentile_level=max_percentile_level
    )[0]

    print('Writing output to: "{0:s}"...'.format(output_file_name))
    novelty_detection.write_pmm_file(
        pickle_file_name=output_file_name,
        mean_novel_image_matrix=list_of_mean_input_matrices[0],
        mean_novel_image_matrix_upconv=mean_novel_image_matrix_upconv,
        mean_novel_image_matrix_upconv_svd=mean_novel_image_matrix_upconv_svd,
        threshold_count_matrix=threshold_count_matrix,
        standard_novelty_file_name=input_novelty_file_name,
        pmm_metadata_dict=pmm_metadata_dict)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_saliency_file_name=getattr(
            INPUT_ARG_OBJECT, SALIENCY_FILE_ARG_NAME),
        input_gradcam_file_name=getattr(
            INPUT_ARG_OBJECT, GRADCAM_FILE_ARG_NAME),
        input_bwo_file_name=getattr(INPUT_ARG_OBJECT, BWO_FILE_ARG_NAME),
        input_novelty_file_name=getattr(
            INPUT_ARG_OBJECT, NOVELTY_FILE_ARG_NAME),
        max_percentile_level=getattr(INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME),
        radar_channel_idx_for_thres=getattr(
            INPUT_ARG_OBJECT, THRESHOLD_INDEX_ARG_NAME),
        threshold_value=getattr(INPUT_ARG_OBJECT, THRESHOLD_VALUE_ARG_NAME),
        threshold_type_string=getattr(
            INPUT_ARG_OBJECT, THRESHOLD_TYPE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
