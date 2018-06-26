"""Evaluates predictions from a convolutional neural network (CNN).

The CNN should be evaluated on independent data (i.e., data used for neither
training nor validation -- the easiest way to ensure independence is to use data
from different years), but this is not enforced by the code.
"""

import os.path
import argparse
import numpy
from keras import backend as K
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from sklearn.metrics import roc_auc_score
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import labels
from gewittergefahr.gg_utils import model_evaluation as model_eval
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import deployment_io
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.plotting import model_eval_plotting

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

INPUT_TIME_FORMAT = '%Y-%m-%d-%H%M%S'
FORECAST_PRECISION_FOR_THRESHOLDS = 1e-4

DOTS_PER_INCH = 300
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

MODEL_FILE_ARG_NAME = 'input_model_file_name'
RADAR_DIRECTORY_ARG_NAME = 'input_storm_radar_image_dir_name'
SOUNDING_DIRECTORY_ARG_NAME = 'input_sounding_dir_name'
TARGET_DIRECTORY_ARG_NAME = 'input_target_dir_name'
ONE_FILE_PER_TIME_STEP_ARG_NAME = 'one_file_per_time_step'
NUM_EXAMPLES_PER_FILE_TIME_ARG_NAME = 'num_examples_per_file_time'
FIRST_EVAL_TIME_ARG_NAME = 'first_eval_time_string'
LAST_EVAL_TIME_ARG_NAME = 'last_eval_time_string'
NUM_STORM_OBJECTS_ARG_NAME = 'num_storm_objects'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

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
ONE_FILE_PER_TIME_STEP_HELP_STRING = (
    'Boolean flag.  If 1 (0), the model will be evaluated with one set of files'
    ' per time step (SPC date).')
NUM_EXAMPLES_PER_FILE_TIME_HELP_STRING = (
    'Number of examples (storm objects) per file time.  If `{0:s}` = True, this'
    ' is the number of examples per time step.  If `{0:s}` = False, this is '
    'number of examples per SPC date.'
).format(ONE_FILE_PER_TIME_STEP_ARG_NAME)
EVAL_TIME_HELP_STRING = (
    'Evaluation time (format "yyyy-mm-dd-HHMMSS").  Evaluation times will be '
    'drawn randomly from `{0:s}`...`{1:s}`.  For each time drawn, all storm '
    'objects will be used.  A forecast-observation pair will be created for '
    'each storm object.'
).format(FIRST_EVAL_TIME_ARG_NAME, LAST_EVAL_TIME_ARG_NAME)
NUM_STORM_OBJECTS_HELP_STRING = (
    'Number of storm objects to draw randomly from `{0:s}`...`{1:s}`.'
).format(FIRST_EVAL_TIME_ARG_NAME, LAST_EVAL_TIME_ARG_NAME)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Evaluation results will be saved here.')

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
    '--' + ONE_FILE_PER_TIME_STEP_ARG_NAME, type=int, required=False, default=0,
    help=ONE_FILE_PER_TIME_STEP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_PER_FILE_TIME_ARG_NAME, type=int, required=True,
    help=NUM_EXAMPLES_PER_FILE_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_EVAL_TIME_ARG_NAME, type=str, required=True,
    help=EVAL_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_EVAL_TIME_ARG_NAME, type=str, required=True,
    help=EVAL_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_STORM_OBJECTS_ARG_NAME, type=int, required=True,
    help=NUM_STORM_OBJECTS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _create_roc_curve(
        forecast_probabilities, observed_labels, output_dir_name):
    """Creates ROC curve.

    N = number of forecast-observation pairs

    :param forecast_probabilities: See documentation for
        `_create_forecast_observation_pairs.`
    :param observed_labels: Same.
    :param output_dir_name: Path to output directory (figure will be saved
        here).
    :return: auc: Area under ROC curve, calculated by GewitterGefahr.
    :return: scikit_learn_auc: Area under ROC curve, calculated by scikit-learn.
    """

    pofd_by_threshold, pod_by_threshold = model_eval.get_points_in_roc_curve(
        forecast_probabilities=forecast_probabilities,
        observed_labels=observed_labels,
        threshold_arg=model_eval.THRESHOLD_ARG_FOR_UNIQUE_FORECASTS,
        unique_forecast_precision=FORECAST_PRECISION_FOR_THRESHOLDS)

    auc = model_eval.get_area_under_roc_curve(
        pofd_by_threshold=pofd_by_threshold,
        pod_by_threshold=pod_by_threshold)
    scikit_learn_auc = roc_auc_score(
        y_true=observed_labels, y_score=forecast_probabilities)

    title_string = 'AUC = {0:.4f} ... scikit-learn AUC = {1:.4f}'.format(
        auc, scikit_learn_auc)
    print title_string

    figure_file_name = '{0:s}/roc_curve.jpg'.format(output_dir_name)
    print 'Saving ROC curve to: "{0:s}"...\n'.format(figure_file_name)

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))
    model_eval_plotting.plot_roc_curve(
        axes_object=axes_object, pod_by_threshold=pod_by_threshold,
        pofd_by_threshold=pofd_by_threshold)

    pyplot.title(title_string)
    pyplot.savefig(figure_file_name, dpi=DOTS_PER_INCH)
    pyplot.close()

    return auc, scikit_learn_auc


def _create_performance_diagram(
        forecast_probabilities, observed_labels, output_dir_name):
    """Creates performance diagram.

    :param forecast_probabilities: See documentation for
        `_create_forecast_observation_pairs.`
    :param observed_labels: Same.
    :param output_dir_name: Same.
    """

    success_ratio_by_threshold, pod_by_threshold = (
        model_eval.get_points_in_performance_diagram(
            forecast_probabilities=forecast_probabilities,
            observed_labels=observed_labels,
            threshold_arg=model_eval.THRESHOLD_ARG_FOR_UNIQUE_FORECASTS,
            unique_forecast_precision=FORECAST_PRECISION_FOR_THRESHOLDS))

    figure_file_name = '{0:s}/performance_diagram.jpg'.format(output_dir_name)
    print 'Saving performance diagram to: "{0:s}"...\n'.format(figure_file_name)

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))
    model_eval_plotting.plot_performance_diagram(
        axes_object=axes_object, pod_by_threshold=pod_by_threshold,
        success_ratio_by_threshold=success_ratio_by_threshold)

    pyplot.savefig(figure_file_name, dpi=DOTS_PER_INCH)
    pyplot.close()


def _create_attributes_diagram(
        forecast_probabilities, observed_labels, output_dir_name):
    """Creates attributes diagram.

    :param forecast_probabilities: See documentation for
        `_create_forecast_observation_pairs.`
    :param observed_labels: Same.
    :param output_dir_name: Same.
    :return: bss_dict: Dictionary created by
        `model_evaluation.get_brier_skill_score`.
    """

    mean_forecast_by_bin, class_frequency_by_bin, num_examples_by_bin = (
        model_eval.get_points_in_reliability_curve(
            forecast_probabilities=forecast_probabilities,
            observed_labels=observed_labels))

    climatology = numpy.mean(observed_labels)
    bss_dict = model_eval.get_brier_skill_score(
        mean_forecast_prob_by_bin=mean_forecast_by_bin,
        mean_observed_label_by_bin=class_frequency_by_bin,
        num_examples_by_bin=num_examples_by_bin, climatology=climatology)

    print (
        'Climatology = {0:.4f} ... reliability = {1:.4f} ... resolution = '
        '{2:.4f} ... BSS = {3:.4f}'
    ).format(climatology, bss_dict[model_eval.RELIABILITY_KEY],
             bss_dict[model_eval.RESOLUTION_KEY],
             bss_dict[model_eval.BRIER_SKILL_SCORE_KEY])

    figure_file_name = '{0:s}/reliability_curve.jpg'.format(output_dir_name)
    print 'Saving reliability curve to: "{0:s}"...\n'.format(figure_file_name)

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))
    model_eval_plotting.plot_reliability_curve(
        axes_object=axes_object, mean_forecast_prob_by_bin=mean_forecast_by_bin,
        mean_observed_label_by_bin=class_frequency_by_bin)

    title_string = 'REL = {0:.4f} ... RES = {1:.4f} ... BSS = {2:.4f}'.format(
        bss_dict[model_eval.RELIABILITY_KEY],
        bss_dict[model_eval.RESOLUTION_KEY],
        bss_dict[model_eval.BRIER_SKILL_SCORE_KEY])
    pyplot.title(title_string)
    pyplot.savefig(figure_file_name, dpi=DOTS_PER_INCH)
    pyplot.close()

    figure_file_name = '{0:s}/attributes_diagram.jpg'.format(output_dir_name)
    print 'Saving attributes diagram to: "{0:s}"...\n'.format(figure_file_name)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))
    model_eval_plotting.plot_attributes_diagram(
        figure_object=figure_object, axes_object=axes_object,
        mean_forecast_prob_by_bin=mean_forecast_by_bin,
        mean_observed_label_by_bin=class_frequency_by_bin,
        num_examples_by_bin=num_examples_by_bin)

    pyplot.savefig(figure_file_name, dpi=DOTS_PER_INCH)
    pyplot.close()

    return bss_dict


def _create_forecast_observation_pairs_2d(
        model_object, top_storm_radar_image_dir_name, top_sounding_dir_name,
        top_target_dir_name, one_file_per_time_step, num_examples_per_file_time,
        first_eval_time_unix_sec, last_eval_time_unix_sec, num_storm_objects,
        model_metadata_dict):
    """Creates forecast-observation pairs for a network with 2-D convolution.

    N = number of storm objects

    :param model_object: Trained model (instance of `keras.models.Sequential`).
    :param top_storm_radar_image_dir_name: See documentation at top of file.
    :param top_sounding_dir_name: Same.
    :param top_target_dir_name: Same.
    :param one_file_per_time_step: Same.
    :param num_examples_per_file_time: Same.
    :param first_eval_time_unix_sec: Same.
    :param last_eval_time_unix_sec: Same.
    :param num_storm_objects: Same.
    :param model_metadata_dict: Dictionary created by `cnn_read_model_metadata`.
    :return: forecast_probabilities: length-N numpy array of forecast event
        probabilities.
    :return: observed_labels: length-N numpy array of observed labels (1 for
        "yes", 0 for "no").
    """

    radar_file_name_matrix, _ = trainval_io.find_radar_files_2d(
        top_directory_name=top_storm_radar_image_dir_name,
        radar_source=model_metadata_dict[cnn.RADAR_SOURCE_KEY],
        radar_field_names=model_metadata_dict[cnn.RADAR_FIELD_NAMES_KEY],
        first_file_time_unix_sec=first_eval_time_unix_sec,
        last_file_time_unix_sec=last_eval_time_unix_sec,
        one_file_per_time_step=one_file_per_time_step,
        radar_heights_m_asl=model_metadata_dict[cnn.RADAR_HEIGHTS_KEY],
        reflectivity_heights_m_asl=model_metadata_dict[
            cnn.REFLECTIVITY_HEIGHTS_KEY])

    print SEPARATOR_STRING

    forecast_probabilities = numpy.array([])
    observed_labels = numpy.array([], dtype=int)
    num_radar_times = radar_file_name_matrix.shape[0]

    for i in range(num_radar_times):
        print (
            'Have created forecast-observation pair for {0:d} of {1:d} storm '
            'objects...\n'
        ).format(len(observed_labels), num_storm_objects)

        if len(observed_labels) > num_storm_objects:
            break

        (this_radar_image_matrix, this_sounding_matrix, these_observed_labels
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

        this_probability_matrix = cnn.apply_2d_cnn(
            model_object=model_object,
            radar_image_matrix=this_radar_image_matrix,
            sounding_matrix=this_sounding_matrix)

        observed_labels = numpy.concatenate((
            observed_labels, these_observed_labels))
        forecast_probabilities = numpy.concatenate((
            forecast_probabilities, this_probability_matrix[:, 1]))

    if len(observed_labels) > num_storm_objects:
        forecast_probabilities = forecast_probabilities[:num_storm_objects]
        observed_labels = observed_labels[:num_storm_objects]

    return forecast_probabilities, observed_labels


def _create_forecast_observation_pairs_3d(
        model_object, top_storm_radar_image_dir_name, top_sounding_dir_name,
        top_target_dir_name, one_file_per_time_step, num_examples_per_file_time,
        first_eval_time_unix_sec, last_eval_time_unix_sec, num_storm_objects,
        model_metadata_dict):
    """Creates forecast-observation pairs for a network with 3-D convolution.

    :param model_object: See doc for `_create_forecast_observation_pairs_2d`.
    :param top_storm_radar_image_dir_name: Same.
    :param top_sounding_dir_name: Same.
    :param top_target_dir_name: Same.
    :param one_file_per_time_step: Same.
    :param num_examples_per_file_time: Same.
    :param first_eval_time_unix_sec: Same.
    :param last_eval_time_unix_sec: Same.
    :param num_storm_objects: Same.
    :param model_metadata_dict: Same.
    :return: forecast_probabilities: Same.
    :return: observed_labels: Same.
    """

    radar_file_name_matrix, _ = trainval_io.find_radar_files_3d(
        top_directory_name=top_storm_radar_image_dir_name,
        radar_source=model_metadata_dict[cnn.RADAR_SOURCE_KEY],
        radar_field_names=model_metadata_dict[cnn.RADAR_FIELD_NAMES_KEY],
        radar_heights_m_asl=model_metadata_dict[cnn.RADAR_HEIGHTS_KEY],
        first_file_time_unix_sec=first_eval_time_unix_sec,
        last_file_time_unix_sec=last_eval_time_unix_sec,
        one_file_per_time_step=one_file_per_time_step)

    print SEPARATOR_STRING

    forecast_probabilities = numpy.array([])
    observed_labels = numpy.array([], dtype=int)
    num_radar_times = radar_file_name_matrix.shape[0]

    for i in range(num_radar_times):
        print (
            'Have created forecast-observation pair for {0:d} of {1:d} storm '
            'objects...\n'
        ).format(len(observed_labels), num_storm_objects)

        if len(observed_labels) > num_storm_objects:
            break

        (this_radar_image_matrix, this_sounding_matrix, these_observed_labels
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

        this_probability_matrix = cnn.apply_3d_cnn(
            model_object=model_object,
            radar_image_matrix=this_radar_image_matrix,
            sounding_matrix=this_sounding_matrix)

        observed_labels = numpy.concatenate((
            observed_labels, these_observed_labels))
        forecast_probabilities = numpy.concatenate((
            forecast_probabilities, this_probability_matrix[:, 1]))

    if len(observed_labels) > num_storm_objects:
        forecast_probabilities = forecast_probabilities[:num_storm_objects]
        observed_labels = observed_labels[:num_storm_objects]

    return forecast_probabilities, observed_labels


def _create_forecast_observation_pairs_2d3d(
        model_object, top_storm_radar_image_dir_name, top_sounding_dir_name,
        top_target_dir_name, one_file_per_time_step, num_examples_per_file_time,
        first_eval_time_unix_sec, last_eval_time_unix_sec, num_storm_objects,
        model_metadata_dict):
    """Creates forecast-observation pairs for a network with 2D/3D convolution.

    :param model_object: See doc for `_create_forecast_observation_pairs_2d`.
    :param top_storm_radar_image_dir_name: Same.
    :param top_sounding_dir_name: Same.
    :param top_target_dir_name: Same.
    :param one_file_per_time_step: Same.
    :param num_examples_per_file_time: Same.
    :param first_eval_time_unix_sec: Same.
    :param last_eval_time_unix_sec: Same.
    :param num_storm_objects: Same.
    :param model_metadata_dict: Same.
    :return: forecast_probabilities: Same.
    :return: observed_labels: Same.
    """

    radar_file_name_matrix, _ = trainval_io.find_radar_files_2d(
        top_directory_name=top_storm_radar_image_dir_name,
        radar_source=model_metadata_dict[cnn.RADAR_SOURCE_KEY],
        radar_field_names=model_metadata_dict[cnn.RADAR_FIELD_NAMES_KEY],
        first_file_time_unix_sec=first_eval_time_unix_sec,
        last_file_time_unix_sec=last_eval_time_unix_sec,
        one_file_per_time_step=one_file_per_time_step,
        radar_heights_m_asl=model_metadata_dict[cnn.RADAR_HEIGHTS_KEY],
        reflectivity_heights_m_asl=model_metadata_dict[
            cnn.REFLECTIVITY_HEIGHTS_KEY])

    print SEPARATOR_STRING

    forecast_probabilities = numpy.array([])
    observed_labels = numpy.array([], dtype=int)
    num_radar_times = radar_file_name_matrix.shape[0]

    for i in range(num_radar_times):
        print (
            'Have created forecast-observation pair for {0:d} of {1:d} storm '
            'objects...\n'
        ).format(len(observed_labels), num_storm_objects)

        if len(observed_labels) > num_storm_objects:
            break

        (this_reflectivity_matrix_dbz, this_azimuthal_shear_matrix_s01,
         this_sounding_matrix, these_observed_labels
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

        this_probability_matrix = cnn.apply_2d3d_cnn(
            model_object=model_object,
            reflectivity_image_matrix_dbz=this_reflectivity_matrix_dbz,
            azimuthal_shear_image_matrix_s01=this_azimuthal_shear_matrix_s01,
            sounding_matrix=this_sounding_matrix)

        observed_labels = numpy.concatenate((
            observed_labels, these_observed_labels))
        forecast_probabilities = numpy.concatenate((
            forecast_probabilities, this_probability_matrix[:, 1]))

    if len(observed_labels) > num_storm_objects:
        forecast_probabilities = forecast_probabilities[:num_storm_objects]
        observed_labels = observed_labels[:num_storm_objects]

    return forecast_probabilities, observed_labels


def _evaluate_model(
        model_file_name, top_storm_radar_image_dir_name, top_sounding_dir_name,
        top_target_dir_name, one_file_per_time_step, num_examples_per_file_time,
        first_eval_time_string, last_eval_time_string, num_storm_objects,
        output_dir_name):
    """Evaluates predictions from a convolutional neural network (CNN).

    :param model_file_name: See documentation at top of file.
    :param top_storm_radar_image_dir_name: Same.
    :param top_sounding_dir_name: Same.
    :param top_target_dir_name: Same.
    :param one_file_per_time_step: Same.
    :param num_examples_per_file_time: Same.
    :param first_eval_time_string: Same.
    :param last_eval_time_string: Same.
    :param num_storm_objects: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if the target variable is non-binary.  This script is
        designed for binary classification only.
    """

    first_eval_time_unix_sec = time_conversion.string_to_unix_sec(
        first_eval_time_string, INPUT_TIME_FORMAT)
    last_eval_time_unix_sec = time_conversion.string_to_unix_sec(
        last_eval_time_string, INPUT_TIME_FORMAT)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    print 'Reading model from: "{0:s}"...'.format(model_file_name)
    model_object = cnn.read_model(model_file_name)
    model_directory_name, _ = os.path.split(model_file_name)
    metadata_file_name = '{0:s}/model_metadata.p'.format(model_directory_name)

    print 'Reading metadata from: "{0:s}"...'.format(metadata_file_name)
    model_metadata_dict = cnn.read_model_metadata(metadata_file_name)

    if not model_metadata_dict[cnn.BINARIZE_TARGET_KEY]:
        num_classes = labels.column_name_to_num_classes(
            model_metadata_dict[cnn.TARGET_NAME_KEY])
        if num_classes > 2:
            error_string = (
                'The target variable ("{0:s}") has {1:d} classes.  This script '
                'is designed for binary classification only.'
            ).format(model_metadata_dict[cnn.TARGET_NAME_KEY], num_classes)
            raise ValueError(error_string)

    if model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]:
        forecast_probabilities, observed_labels = (
            _create_forecast_observation_pairs_2d3d(
                model_object=model_object,
                top_storm_radar_image_dir_name=top_storm_radar_image_dir_name,
                top_sounding_dir_name=top_sounding_dir_name,
                top_target_dir_name=top_target_dir_name,
                one_file_per_time_step=one_file_per_time_step,
                num_examples_per_file_time=num_examples_per_file_time,
                first_eval_time_unix_sec=first_eval_time_unix_sec,
                last_eval_time_unix_sec=last_eval_time_unix_sec,
                num_storm_objects=num_storm_objects,
                model_metadata_dict=model_metadata_dict))
    else:
        num_radar_dimensions = len(
            model_metadata_dict[cnn.TRAINING_FILE_NAME_MATRIX_KEY].shape)
        if num_radar_dimensions == 2:
            forecast_probabilities, observed_labels = (
                _create_forecast_observation_pairs_2d(
                    model_object=model_object,
                    top_storm_radar_image_dir_name=
                    top_storm_radar_image_dir_name,
                    top_sounding_dir_name=top_sounding_dir_name,
                    top_target_dir_name=top_target_dir_name,
                    one_file_per_time_step=one_file_per_time_step,
                    num_examples_per_file_time=num_examples_per_file_time,
                    first_eval_time_unix_sec=first_eval_time_unix_sec,
                    last_eval_time_unix_sec=last_eval_time_unix_sec,
                    num_storm_objects=num_storm_objects,
                    model_metadata_dict=model_metadata_dict))
        else:
            forecast_probabilities, observed_labels = (
                _create_forecast_observation_pairs_3d(
                    model_object=model_object,
                    top_storm_radar_image_dir_name=
                    top_storm_radar_image_dir_name,
                    top_sounding_dir_name=top_sounding_dir_name,
                    top_target_dir_name=top_target_dir_name,
                    one_file_per_time_step=one_file_per_time_step,
                    num_examples_per_file_time=num_examples_per_file_time,
                    first_eval_time_unix_sec=first_eval_time_unix_sec,
                    last_eval_time_unix_sec=last_eval_time_unix_sec,
                    num_storm_objects=num_storm_objects,
                    model_metadata_dict=model_metadata_dict))

    print SEPARATOR_STRING

    # TODO(thunderhoser): Make binarization threshold an input argument to the
    # script.
    binarization_threshold, best_csi = (
        model_eval.find_best_binarization_threshold(
            forecast_probabilities=forecast_probabilities,
            observed_labels=observed_labels,
            threshold_arg=model_eval.THRESHOLD_ARG_FOR_UNIQUE_FORECASTS,
            criterion_function=model_eval.get_csi,
            optimization_direction=model_eval.MAX_OPTIMIZATION_DIRECTION,
            unique_forecast_precision=FORECAST_PRECISION_FOR_THRESHOLDS))

    print (
        'Best binarization threshold = {0:.4f} ... corresponding CSI = {1:.4f}'
    ).format(binarization_threshold, best_csi)

    print 'Binarizing forecast probabilities...'
    forecast_labels = model_eval.binarize_forecast_probs(
        forecast_probabilities=forecast_probabilities,
        binarization_threshold=binarization_threshold)

    print 'Creating contingency table...'
    contingency_table_as_dict = model_eval.get_contingency_table(
        forecast_labels=forecast_labels, observed_labels=observed_labels)
    print '{0:s}\n'.format(str(contingency_table_as_dict))

    print 'Computing performance metrics...'
    pod = model_eval.get_pod(contingency_table_as_dict)
    pofd = model_eval.get_pofd(contingency_table_as_dict)
    success_ratio = model_eval.get_success_ratio(contingency_table_as_dict)
    focn = model_eval.get_focn(contingency_table_as_dict)
    accuracy = model_eval.get_accuracy(contingency_table_as_dict)
    csi = model_eval.get_csi(contingency_table_as_dict)
    frequency_bias = model_eval.get_frequency_bias(contingency_table_as_dict)
    peirce_score = model_eval.get_peirce_score(contingency_table_as_dict)
    heidke_score = model_eval.get_heidke_score(contingency_table_as_dict)

    print (
        'POD = {0:.4f} ... POFD = {1:.4f} ... success ratio = {2:.4f} ... '
        'FOCN = {3:.4f} ... accuracy = {4:.4f} ... CSI = {5:.4f} ... frequency '
        'bias = {6:.4f} ... Peirce score = {7:.4f} ... Heidke score = {8:.4f}\n'
    ).format(pod, pofd, success_ratio, focn, accuracy, csi, frequency_bias,
             peirce_score, heidke_score)

    auc, scikit_learn_auc = _create_roc_curve(
        forecast_probabilities=forecast_probabilities,
        observed_labels=observed_labels, output_dir_name=output_dir_name)
    print '\n'

    bss_dict = _create_attributes_diagram(
        forecast_probabilities=forecast_probabilities,
        observed_labels=observed_labels, output_dir_name=output_dir_name)
    print '\n'

    evaluation_file_name = '{0:s}/model_evaluation.p'.format(output_dir_name)
    print 'Writing evaluation results to: "{0:s}"...\n'.format(
        evaluation_file_name)
    model_eval.write_results(
        forecast_probabilities=forecast_probabilities,
        observed_labels=observed_labels,
        binarization_threshold=binarization_threshold, pod=pod, pofd=pofd,
        success_ratio=success_ratio, focn=focn, accuracy=accuracy, csi=csi,
        frequency_bias=frequency_bias, peirce_score=peirce_score,
        heidke_score=heidke_score, auc=auc, scikit_learn_auc=scikit_learn_auc,
        bss_dict=bss_dict, pickle_file_name=evaluation_file_name)

    _create_performance_diagram(
        forecast_probabilities=forecast_probabilities,
        observed_labels=observed_labels, output_dir_name=output_dir_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _evaluate_model(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        top_storm_radar_image_dir_name=getattr(
            INPUT_ARG_OBJECT, RADAR_DIRECTORY_ARG_NAME),
        top_sounding_dir_name=getattr(
            INPUT_ARG_OBJECT, SOUNDING_DIRECTORY_ARG_NAME),
        top_target_dir_name=getattr(
            INPUT_ARG_OBJECT, TARGET_DIRECTORY_ARG_NAME),
        one_file_per_time_step=bool(getattr(
            INPUT_ARG_OBJECT, ONE_FILE_PER_TIME_STEP_ARG_NAME)),
        num_examples_per_file_time=getattr(
            INPUT_ARG_OBJECT, NUM_EXAMPLES_PER_FILE_TIME_ARG_NAME),
        first_eval_time_string=getattr(
            INPUT_ARG_OBJECT, FIRST_EVAL_TIME_ARG_NAME),
        last_eval_time_string=getattr(
            INPUT_ARG_OBJECT, LAST_EVAL_TIME_ARG_NAME),
        num_storm_objects=getattr(INPUT_ARG_OBJECT, NUM_STORM_OBJECTS_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME))
