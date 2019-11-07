"""Runs probability-matched means (PMM).

Specifically, this script applies PMM to inputs (predictors) and outputs from
one of the following interpretation methods:

- saliency maps
- class-activation maps
- backwards optimization
- novelty detection
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
OUTPUT_FILE_ARG_NAME = 'output_file_name'

SALIENCY_FILE_HELP_STRING = (
    'Path to saliency file (will be read by `saliency_maps.read_file`).  If you'
    ' are compositing something other than saliency maps, leave this argument '
    'alone.')

GRADCAM_FILE_HELP_STRING = (
    'Path to Grad-CAM file (will be read by `gradcam.read_file`).  If you are '
    'compositing something other than class-activation maps, leave this '
    'argument alone.')

BWO_FILE_HELP_STRING = (
    'Path to backwards-optimization file (will be read by '
    '`backwards_optimization.read_file`).  If you are compositing something '
    'other than backwards-optimization results, leave this argument alone.')

NOVELTY_FILE_HELP_STRING = (
    'Path to novelty-detection file (will be read by '
    '`novelty_detection.read_file`).  If you are compositing something other '
    'than novelty-detection results, leave this argument alone.')

MAX_PERCENTILE_HELP_STRING = (
    'Max percentile used in PMM procedure.  See '
    '`prob_matched_means.run_pmm_one_variable` for details.')

OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Will be written by `saliency_maps.write_pmm_file`, '
    '`gradcam.write_pmm_file`, `backwards_optimization.write_pmm_file`, or '
    '`novelty_detection.write_pmm_file`.')

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
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=False, default='',
    help=OUTPUT_FILE_HELP_STRING)


def _composite_predictors(
        predictor_matrices, max_percentile_level,
        sounding_pressure_matrix_pa=None):
    """Runs PMM on predictors.

    T = number of input tensors to the model
    E = number of examples
    H_s = number of sounding heights

    :param predictor_matrices: length-T list of numpy arrays, each containing
        one type of predictor.
    :param max_percentile_level: See documentation at top of file.
    :param sounding_pressure_matrix_pa: numpy array (E x H_s) of sounding
        pressures.  This may be None, in which case the method will not bother
        trying to composite sounding pressures.
    :return: mean_predictor_matrices: length-T list of numpy arrays, where
        mean_predictor_matrices[i] is a composite over all examples in
        predictor_matrices[i].
    :return: mean_sounding_pressures_pa: numpy array (length H_s) of
        sounding pressures.  If `sounding_pressure_matrix_pa is None`, this
        is also None.
    """

    num_matrices = len(predictor_matrices)
    mean_predictor_matrices = [None] * num_matrices

    for i in range(num_matrices):
        mean_predictor_matrices[i] = pmm.run_pmm_many_variables(
            input_matrix=predictor_matrices[i],
            max_percentile_level=max_percentile_level)

    if sounding_pressure_matrix_pa is None:
        mean_sounding_pressures_pa = None
    else:
        this_input_matrix = numpy.expand_dims(
            sounding_pressure_matrix_pa, axis=-1)

        mean_sounding_pressures_pa = pmm.run_pmm_many_variables(
            input_matrix=this_input_matrix,
            max_percentile_level=max_percentile_level
        )[..., 0]

    return mean_predictor_matrices, mean_sounding_pressures_pa


def _composite_saliency_maps(
        input_file_name, max_percentile_level, output_file_name):
    """Composites predictors and resulting saliency maps.

    :param input_file_name: Path to input file.  Will be read by
        `saliency_maps.read_file`.
    :param max_percentile_level: See documentation at top of file.
    :param output_file_name: Path to output file.  Will be written by
        `saliency_maps.write_pmm_file`.
    """

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    saliency_dict = saliency_maps.read_file(input_file_name)[0]

    predictor_matrices = saliency_dict[saliency_maps.PREDICTOR_MATRICES_KEY]
    saliency_matrices = saliency_dict[saliency_maps.SALIENCY_MATRICES_KEY]
    sounding_pressure_matrix_pa = saliency_dict[
        saliency_maps.SOUNDING_PRESSURES_KEY]

    print('Compositing predictor matrices...')
    mean_predictor_matrices, mean_sounding_pressures_pa = _composite_predictors(
        predictor_matrices=predictor_matrices,
        max_percentile_level=max_percentile_level,
        sounding_pressure_matrix_pa=sounding_pressure_matrix_pa)

    print('Compositing saliency maps...')
    num_matrices = len(predictor_matrices)
    mean_saliency_matrices = [None] * num_matrices

    for i in range(num_matrices):
        mean_saliency_matrices[i] = pmm.run_pmm_many_variables(
            input_matrix=saliency_matrices[i],
            max_percentile_level=max_percentile_level)

    print('Writing output to: "{0:s}"...'.format(output_file_name))
    saliency_maps.write_pmm_file(
        pickle_file_name=output_file_name,
        mean_denorm_predictor_matrices=mean_predictor_matrices,
        mean_saliency_matrices=mean_saliency_matrices,
        model_file_name=saliency_dict[saliency_maps.MODEL_FILE_KEY],
        non_pmm_file_name=input_file_name,
        pmm_max_percentile_level=max_percentile_level,
        mean_sounding_pressures_pa=mean_sounding_pressures_pa)


def _composite_gradcam(input_file_name, max_percentile_level, output_file_name):
    """Composites predictors and resulting class-activation maps.

    :param input_file_name: Path to input file.  Will be read by
        `gradcam.read_file`.
    :param max_percentile_level: See documentation at top of file.
    :param output_file_name: Path to output file.  Will be written by
        `gradcam.write_pmm_file`.
    """

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    gradcam_dict = gradcam.read_file(input_file_name)[0]

    predictor_matrices = gradcam_dict[gradcam.PREDICTOR_MATRICES_KEY]
    cam_matrices = gradcam_dict[gradcam.CAM_MATRICES_KEY]
    guided_cam_matrices = gradcam_dict[gradcam.GUIDED_CAM_MATRICES_KEY]
    sounding_pressure_matrix_pa = gradcam_dict[
        gradcam.SOUNDING_PRESSURES_KEY]

    print('Compositing predictor matrices...')
    mean_predictor_matrices, mean_sounding_pressures_pa = _composite_predictors(
        predictor_matrices=predictor_matrices,
        max_percentile_level=max_percentile_level,
        sounding_pressure_matrix_pa=sounding_pressure_matrix_pa)

    print('Compositing class-activation maps...')
    num_matrices = len(predictor_matrices)
    mean_cam_matrices = [None] * num_matrices
    mean_guided_cam_matrices = [None] * num_matrices

    for i in range(num_matrices):
        if cam_matrices[i] is None:
            continue

        mean_cam_matrices[i] = pmm.run_pmm_many_variables(
            input_matrix=numpy.expand_dims(cam_matrices[i], axis=-1),
            max_percentile_level=max_percentile_level
        )[..., 0]

        mean_guided_cam_matrices[i] = pmm.run_pmm_many_variables(
            input_matrix=guided_cam_matrices[i],
            max_percentile_level=max_percentile_level)

    print('Writing output to: "{0:s}"...'.format(output_file_name))
    gradcam.write_pmm_file(
        pickle_file_name=output_file_name,
        mean_denorm_predictor_matrices=mean_predictor_matrices,
        mean_cam_matrices=mean_cam_matrices,
        mean_guided_cam_matrices=mean_guided_cam_matrices,
        model_file_name=gradcam_dict[gradcam.MODEL_FILE_KEY],
        non_pmm_file_name=input_file_name,
        pmm_max_percentile_level=max_percentile_level,
        mean_sounding_pressures_pa=mean_sounding_pressures_pa)


def _composite_backwards_opt(
        input_file_name, max_percentile_level, output_file_name):
    """Composites inputs and outputs for backwards optimization.

    :param input_file_name: Path to input file.  Will be read by
        `backwards_optimization.read_file`.
    :param max_percentile_level: See documentation at top of file.
    :param output_file_name: Path to output file.  Will be written by
        `backwards_optimization.write_pmm_file`.
    """

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    bwo_dictionary = backwards_opt.read_file(input_file_name)[0]

    input_matrices = bwo_dictionary[backwards_opt.INPUT_MATRICES_KEY]
    output_matrices = bwo_dictionary[backwards_opt.OUTPUT_MATRICES_KEY]
    sounding_pressure_matrix_pa = bwo_dictionary[
        backwards_opt.SOUNDING_PRESSURES_KEY]

    print('Compositing non-optimized predictors...')
    mean_input_matrices, mean_sounding_pressures_pa = _composite_predictors(
        predictor_matrices=input_matrices,
        max_percentile_level=max_percentile_level,
        sounding_pressure_matrix_pa=sounding_pressure_matrix_pa)

    print('Compositing optimized predictors...')
    mean_output_matrices = _composite_predictors(
        predictor_matrices=output_matrices,
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
        mean_denorm_input_matrices=mean_input_matrices,
        mean_denorm_output_matrices=mean_output_matrices,
        mean_initial_activation=mean_initial_activation,
        mean_final_activation=mean_final_activation,
        model_file_name=bwo_dictionary[backwards_opt.MODEL_FILE_KEY],
        non_pmm_file_name=input_file_name,
        pmm_max_percentile_level=max_percentile_level,
        mean_sounding_pressures_pa=mean_sounding_pressures_pa)


def _composite_novelty(
        input_file_name, max_percentile_level, output_file_name):
    """Composites inputs and outputs for novelty detection.

    :param input_file_name: Path to input file.  Will be read by
        `novelty_detection.read_standard_file`.
    :param max_percentile_level: See documentation at top of file.
    :param output_file_name: Path to output file.  Will be written by
        `novelty_detection.write_pmm_file`.
    """

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    novelty_dict = novelty_detection.read_file(input_file_name)[0]

    print('Compositing baseline radar images...')
    mean_radar_matrix_baseline = pmm.run_pmm_many_variables(
        input_matrix=novelty_dict[novelty_detection.BASELINE_MATRIX_KEY],
        max_percentile_level=max_percentile_level
    )

    print('Compositing novel radar images (in trial set)...')
    novel_indices = novelty_dict[novelty_detection.NOVEL_INDICES_KEY]
    radar_matrix_novel = novelty_dict[novelty_detection.TRIAL_MATRIX_KEY][
        novel_indices, ...]

    mean_radar_matrix_novel = pmm.run_pmm_many_variables(
        input_matrix=radar_matrix_novel,
        max_percentile_level=max_percentile_level)

    print('Compositing reconstructions of novel radar images...')
    mean_radar_matrix_upconv = pmm.run_pmm_many_variables(
        input_matrix=novelty_dict[novelty_detection.UPCONV_MATRIX_KEY],
        max_percentile_level=max_percentile_level
    )

    mean_radar_matrix_upconv_svd = pmm.run_pmm_many_variables(
        input_matrix=novelty_dict[novelty_detection.UPCONV_SVD_MATRIX_KEY],
        max_percentile_level=max_percentile_level
    )

    print('Writing output to: "{0:s}"...'.format(output_file_name))
    novelty_detection.write_pmm_file(
        pickle_file_name=output_file_name,
        mean_denorm_radar_matrix_baseline=mean_radar_matrix_baseline,
        mean_denorm_radar_matrix_novel=mean_radar_matrix_novel,
        mean_denorm_radar_matrix_upconv=mean_radar_matrix_upconv,
        mean_denorm_radar_matrix_upconv_svd=mean_radar_matrix_upconv_svd,
        cnn_file_name=novelty_dict[novelty_detection.CNN_FILE_KEY],
        non_pmm_file_name=input_file_name,
        pmm_max_percentile_level=max_percentile_level)


def _run(input_saliency_file_name, input_gradcam_file_name, input_bwo_file_name,
         input_novelty_file_name, max_percentile_level, output_file_name):
    """Runs probability-matched means (PMM).

    This is effectively the main method.

    :param input_saliency_file_name: See documentation at top of file.
    :param input_gradcam_file_name: Same.
    :param input_bwo_file_name: Same.
    :param input_novelty_file_name: Same.
    :param max_percentile_level: Same.
    :param output_file_name: Same.
    """

    if input_saliency_file_name not in NONE_STRINGS:
        _composite_saliency_maps(
            input_file_name=input_saliency_file_name,
            max_percentile_level=max_percentile_level,
            output_file_name=output_file_name)

        return

    if input_gradcam_file_name not in NONE_STRINGS:
        _composite_gradcam(
            input_file_name=input_gradcam_file_name,
            max_percentile_level=max_percentile_level,
            output_file_name=output_file_name)

        return

    if input_bwo_file_name not in NONE_STRINGS:
        _composite_backwards_opt(
            input_file_name=input_bwo_file_name,
            max_percentile_level=max_percentile_level,
            output_file_name=output_file_name)

        return

    _composite_novelty(
        input_file_name=input_novelty_file_name,
        max_percentile_level=max_percentile_level,
        output_file_name=output_file_name)


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
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
