"""Finds best RDP value for pre-model filtering.

RDP = rotation-divergence product

Specifically, I want to use the RDP to filter out non-tornadic storms, which
will balance the training data for machine learning.  The training data are
dominated by non-tornadic storms (95% in the GridRad dataset, > 99% in the
MYRORSS dataset), which is problematic because machine-learning models usually
do not perform well on rare events.

There are other ways to balance the training set (e.g., downsampling), but this
cannot be applied to training or validation data.  If I could develop a good
pre-model filter -- i.e., one that filters out a huge number of non-tornadic
storms and falsely removes only a few tornadic storms -- I could balance the
training, validation, and testing sets.  My hope is that this will yield better
performance on validation/testing data than downsampling.
"""

import os.path
import argparse
import numpy
from gewittergefahr.gg_utils import labels
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import model_evaluation as model_eval
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import storm_images

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_TIME_FORMAT = '%Y-%m-%d-%H%M%S'
DUMMY_RADAR_FIELD_NAME = radar_utils.REFL_NAME
DUMMY_RADAR_HEIGHT_M_ASL = 1000

STORM_IMAGE_DIR_ARG_NAME = 'input_storm_image_dir_name'
TARGET_DIRECTORY_ARG_NAME = 'input_target_dir_name'
TARGET_NAME_ARG_NAME = 'target_name'
FIRST_VALIDN_TIME_ARG_NAME = 'first_validn_time_string'
LAST_VALIDN_TIME_ARG_NAME = 'last_validn_time_string'
FIRST_TESTING_TIME_ARG_NAME = 'first_testing_time_string'
LAST_TESTING_TIME_ARG_NAME = 'last_testing_time_string'
MAX_FOM_ARG_NAME = 'max_fom'
RDP_INTERVAL_ARG_NAME = 'rdp_interval_s02'

STORM_IMAGE_DIR_HELP_STRING = (
    'Name of top-level directory for storm-centered radar images.  Files '
    'therein, which will be found by `storm_images.find_storm_image_file` and '
    'read by `storm_images.read_storm_image_file`, should also contain the RDP '
    'for each storm object.')
TARGET_DIRECTORY_HELP_STRING = (
    'Name of top-level directory with labels (target values).  Files therein '
    'will be found by `labels.find_label_file`.')
TARGET_NAME_HELP_STRING = (
    'Name of target variable (must be accepted by '
    '`labels.column_name_to_label_params`).')
VALIDN_TIME_HELP_STRING = (
    'Time (format "yyyy-mm-dd-HHMMSS").  Each RDP threshold will be validated '
    'on all storm objects from `{0:s}`...`{1:s}`.  The best RDP threshold on '
    'validation data -- i.e., that with the highest NPV (negative predictive '
    'value), subject to FOM (frequency of misses) <= `{2:s}` -- will be '
    'selected.'
).format(FIRST_VALIDN_TIME_ARG_NAME, LAST_VALIDN_TIME_ARG_NAME,
         MAX_FOM_ARG_NAME)
TESTING_TIME_HELP_STRING = (
    'Time (format "yyyy-mm-dd-HHMMSS").  The selected RDP threshold will be '
    'tested on all storm objects from `{0:s}`...`{1:s}`.'
).format(FIRST_TESTING_TIME_ARG_NAME, LAST_TESTING_TIME_ARG_NAME)
MAX_FOM_HELP_STRING = 'See help string for `{0:s}` and `{1:s}`.'.format(
    FIRST_VALIDN_TIME_ARG_NAME, LAST_VALIDN_TIME_ARG_NAME)
RDP_INTERVAL_HELP_STRING = (
    'Interval between successive RDP thresholds.  All thresholds from '
    '0...max_rdp will be tried, where max_rdp is the maximum in the validation '
    'set.  The interval between successive thresholds will be `{0:s}`.'
).format(RDP_INTERVAL_ARG_NAME)

DEFAULT_TOP_STORM_IMAGE_DIR_NAME = (
    '/condo/swatcommon/common/gridrad_final/myrorss_format/tracks/reanalyzed/'
    'storm_images_with_rdp')
DEFAULT_TOP_TARGET_DIR_NAME = (
    '/condo/swatcommon/common/gridrad_final/myrorss_format/tracks/reanalyzed/'
    'tornado_linkages/reanalyzed/labels')
DEFAULT_TARGET_NAME = 'tornado_lead-time=0000-3600sec_distance=00000-10000m'
DEFAULT_FIRST_VALIDN_TIME_STRING = '2011-01-01-000000'
DEFAULT_LAST_VALIDN_TIME_STRING = '2014-12-31-000000'
DEFAULT_FIRST_TESTING_TIME_STRING = '2015-01-01-000000'
DEFAULT_LAST_TESTING_TIME_STRING = '2018-01-01-000000'
DEFAULT_MAX_FOM = 0.01
DEFAULT_RDP_INTERVAL_S02 = 1e-7

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + STORM_IMAGE_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_TOP_STORM_IMAGE_DIR_NAME,
    help=STORM_IMAGE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_DIRECTORY_ARG_NAME, type=str, required=False,
    default=DEFAULT_TOP_TARGET_DIR_NAME, help=TARGET_DIRECTORY_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_NAME_ARG_NAME, type=str, required=False,
    default=DEFAULT_TARGET_NAME, help=TARGET_NAME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_VALIDN_TIME_ARG_NAME, type=str, required=False,
    default=DEFAULT_FIRST_VALIDN_TIME_STRING, help=VALIDN_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_VALIDN_TIME_ARG_NAME, type=str, required=False,
    default=DEFAULT_LAST_VALIDN_TIME_STRING, help=VALIDN_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TESTING_TIME_ARG_NAME, type=str, required=False,
    default=DEFAULT_FIRST_TESTING_TIME_STRING, help=TESTING_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TESTING_TIME_ARG_NAME, type=str, required=False,
    default=DEFAULT_LAST_TESTING_TIME_STRING, help=TESTING_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_FOM_ARG_NAME, type=float, required=False,
    default=DEFAULT_MAX_FOM, help=MAX_FOM_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RDP_INTERVAL_ARG_NAME, type=float, required=False,
    default=DEFAULT_RDP_INTERVAL_S02, help=RDP_INTERVAL_HELP_STRING)


def _read_rdp_and_targets_one_spc_date(
        storm_image_file_name, target_file_name, target_name,
        min_storm_time_unix_sec, max_storm_time_unix_sec):
    """Reads RDP and target values for one SPC date.

    N = number of storm objects

    :param storm_image_file_name: Path to file with storm-centered radar images
        (and also RDP values).
    :param target_file_name: Path to file with target values.
    :param target_name: Name of target values.
    :param min_storm_time_unix_sec: Minimum storm time.  Will keep only storm
        objects from `min_storm_time_unix_sec`...`max_storm_time_unix_sec`.
    :param max_storm_time_unix_sec: Same.
    :return: rdp_values_s02: length-N numpy array of RDP values (units of s^-2).
    :return: target_values: length-N numpy array of target values (integer class
        labels).
    """

    print 'Reading data from: "{0:s}"...'.format(storm_image_file_name)
    storm_image_dict = storm_images.read_storm_images(
        netcdf_file_name=storm_image_file_name, return_images=False)

    indices_to_keep = numpy.where(numpy.logical_and(
        storm_image_dict[storm_images.VALID_TIMES_KEY] >=
        min_storm_time_unix_sec,
        storm_image_dict[storm_images.VALID_TIMES_KEY] <=
        max_storm_time_unix_sec
    ))[0]

    storm_ids = [storm_image_dict[storm_images.STORM_IDS_KEY][k]
                 for k in indices_to_keep]
    storm_times_unix_sec = storm_image_dict[
        storm_images.VALID_TIMES_KEY][indices_to_keep]
    rdp_values_s02 = storm_image_dict[
        storm_images.ROTATION_DIVERGENCE_PRODUCTS_KEY][indices_to_keep]

    print 'Reading data from: "{0:s}"...'.format(target_file_name)
    storm_label_dict = labels.read_labels_from_netcdf(
        netcdf_file_name=target_file_name, label_name=target_name)

    print 'Matching storm objects from the two files...'
    these_indices = storm_images.find_storm_objects(
        all_storm_ids=storm_label_dict[labels.STORM_IDS_KEY],
        all_valid_times_unix_sec=storm_label_dict[labels.VALID_TIMES_KEY],
        storm_ids_to_keep=storm_ids,
        valid_times_to_keep_unix_sec=storm_times_unix_sec)
    target_values = storm_label_dict[labels.LABEL_VALUES_KEY][these_indices]

    indices_to_keep = numpy.where(numpy.logical_and(
        target_values >= 0, numpy.invert(numpy.isnan(rdp_values_s02))
    ))[0]

    return rdp_values_s02[indices_to_keep], target_values[indices_to_keep]


def _do_validation(
        validation_image_file_names, validation_target_file_names, target_name,
        first_validn_time_unix_sec, last_validn_time_unix_sec, max_fom,
        rdp_interval_s02):
    """Tries various RDP thresholds and selects the best on validation data.

    V = number of validation days

    :param validation_image_file_names: length-V list of paths to storm-image
        files.
    :param validation_target_file_names: length-V list of paths to target files.
    :param target_name: Name of target variable.
    :param first_validn_time_unix_sec: Start of validation period.
    :param last_validn_time_unix_sec: End of validation period.
    :param max_fom: See documentation at top of file.
    :param rdp_interval_s02: Same.
    :return: best_rdp_threshold_s02: Best RDP threshold (units of s^-2).  For
        the definition of "best" threshold, see documentation at top of file.
    :return: best_npv: Negative predictive value with the given threshold.
    :return: best_fom: Frequency of misses with the given threshold.
    """

    num_validation_days = len(validation_image_file_names)
    rdp_by_storm_object_s02 = numpy.array([])
    target_value_by_storm_object = numpy.array([], dtype=int)

    for i in range(num_validation_days):
        (these_rdp_values_s02, these_target_values
        ) = _read_rdp_and_targets_one_spc_date(
            storm_image_file_name=validation_image_file_names[i],
            target_file_name=validation_target_file_names[i],
            target_name=target_name,
            min_storm_time_unix_sec=first_validn_time_unix_sec,
            max_storm_time_unix_sec=last_validn_time_unix_sec)

        print '\n'
        rdp_by_storm_object_s02 = numpy.concatenate((
            rdp_by_storm_object_s02, these_rdp_values_s02))
        target_value_by_storm_object = numpy.concatenate((
            target_value_by_storm_object, these_target_values))

    num_storm_objects = len(rdp_by_storm_object_s02)
    minimum_rdp_s02 = numpy.min(rdp_by_storm_object_s02)
    maximum_rdp_s02 = numpy.max(rdp_by_storm_object_s02)
    print (
        'There are {0:d} storm objects, with min and max RDP values of {1:.2e} '
        'and {2:.2e} s^-2.'
    ).format(num_storm_objects, minimum_rdp_s02, maximum_rdp_s02)

    minimum_rdp_s02 = number_rounding.floor_to_nearest(
        minimum_rdp_s02, rdp_interval_s02)
    maximum_rdp_s02 = number_rounding.ceiling_to_nearest(
        maximum_rdp_s02, rdp_interval_s02)
    num_thresholds = 1 + int(numpy.round(
        (maximum_rdp_s02 - minimum_rdp_s02) / rdp_interval_s02))
    rdp_thresholds_s02 = numpy.linspace(
        minimum_rdp_s02, maximum_rdp_s02, num=num_thresholds)

    npv_by_threshold = numpy.full(num_thresholds, numpy.nan)
    fom_by_threshold = numpy.full(num_thresholds, numpy.nan)
    print SEPARATOR_STRING

    for j in range(num_thresholds):
        these_predicted_target_values = numpy.full(
            num_storm_objects, 0, dtype=int)
        these_indices = numpy.where(
            rdp_by_storm_object_s02 >= rdp_thresholds_s02[j])[0]
        these_predicted_target_values[these_indices] = 1

        this_contingency_table_as_dict = model_eval.get_contingency_table(
            forecast_labels=these_predicted_target_values,
            observed_labels=target_value_by_storm_object)
        npv_by_threshold[j] = model_eval.get_npv(this_contingency_table_as_dict)
        fom_by_threshold[j] = model_eval.get_fom(this_contingency_table_as_dict)

        print (
            'Threshold = {0:.2e} s^-2 ... num false negatives = {1:d} ... num '
            'true negatives = {2:d} ... NPV = {3:.4f} ... FOM = {4:.4f}'
        ).format(
            rdp_thresholds_s02[j],
            this_contingency_table_as_dict[model_eval.NUM_FALSE_NEGATIVES_KEY],
            this_contingency_table_as_dict[model_eval.NUM_TRUE_NEGATIVES_KEY],
            npv_by_threshold[j], fom_by_threshold[j])

    print SEPARATOR_STRING

    bad_indices = numpy.where(fom_by_threshold > max_fom)[0]
    npv_by_threshold[bad_indices] = numpy.nan
    best_threshold_index = numpy.nanargmax(npv_by_threshold)

    print (
        'Best threshold = {0:.2e} s^-2 ... NPV = {1:.4f} ... FOM = {2:.4f}'
    ).format(rdp_thresholds_s02[best_threshold_index],
             npv_by_threshold[best_threshold_index],
             fom_by_threshold[best_threshold_index])

    return (rdp_thresholds_s02[best_threshold_index],
            npv_by_threshold[best_threshold_index],
            fom_by_threshold[best_threshold_index])


def _do_testing(
        testing_image_file_names, testing_target_file_names, target_name,
        first_testing_time_unix_sec, last_testing_time_unix_sec,
        best_rdp_threshold_s02):
    """Evaluates selected RDP threshold on testing data.

    T = number of testing days

    :param testing_image_file_names: length-T list of paths to storm-image
        files.
    :param testing_target_file_names: length-T list of paths to target files.
    :param target_name: Name of target variable.
    :param first_testing_time_unix_sec: Start of testing period.
    :param last_testing_time_unix_sec: End of testing period.
    :param best_rdp_threshold_s02: RDP threshold (units of s^-2) selected in
        validation.
    :return: testing_npv: Negative predictive value on testing data.
    :return: testing_fom: Frequency of misses on testing data.
    """

    num_testing_days = len(testing_image_file_names)
    rdp_by_storm_object_s02 = numpy.array([])
    target_value_by_storm_object = numpy.array([], dtype=int)

    for i in range(num_testing_days):
        (these_rdp_values_s02, these_target_values
        ) = _read_rdp_and_targets_one_spc_date(
            storm_image_file_name=testing_image_file_names[i],
            target_file_name=testing_target_file_names[i],
            target_name=target_name,
            min_storm_time_unix_sec=first_testing_time_unix_sec,
            max_storm_time_unix_sec=last_testing_time_unix_sec)

        print '\n'
        rdp_by_storm_object_s02 = numpy.concatenate((
            rdp_by_storm_object_s02, these_rdp_values_s02))
        target_value_by_storm_object = numpy.concatenate((
            target_value_by_storm_object, these_target_values))

    num_storm_objects = len(rdp_by_storm_object_s02)
    predicted_target_values = numpy.full(num_storm_objects, 0, dtype=int)
    these_indices = numpy.where(
        rdp_by_storm_object_s02 >= best_rdp_threshold_s02)[0]
    predicted_target_values[these_indices] = 1

    contingency_table_as_dict = model_eval.get_contingency_table(
        forecast_labels=predicted_target_values,
        observed_labels=target_value_by_storm_object)
    testing_npv = model_eval.get_npv(contingency_table_as_dict)
    testing_fom = model_eval.get_fom(contingency_table_as_dict)

    print (
        'Testing results: num false negatives = {0:d} ... num true negatives = '
        '{1:d} ... NPV = {2:.4f} ... FOM = {3:.4f}'
    ).format(
        contingency_table_as_dict[model_eval.NUM_FALSE_NEGATIVES_KEY],
        contingency_table_as_dict[model_eval.NUM_TRUE_NEGATIVES_KEY],
        testing_npv, testing_fom)

    return testing_npv, testing_fom


def _run(
        top_storm_image_dir_name, top_target_dir_name, target_name,
        first_validn_time_string, last_validn_time_string,
        first_testing_time_string, last_testing_time_string, max_fom,
        rdp_interval_s02):
    """Tests various RDP thresholds as pre-model filters.

    This is essentially the main method.

    :param top_storm_image_dir_name: See documentation at top of file.
    :param top_target_dir_name: Same.
    :param target_name: Same.
    :param first_validn_time_string: Same.
    :param last_validn_time_string: Same.
    :param first_testing_time_string: Same.
    :param last_testing_time_string: Same.
    :param max_fom: Same.
    :param rdp_interval_s02: Same.
    :raises: ValueError: if the target variable is non-binary.
    """

    # Check input arguments.
    num_classes = labels.column_name_to_num_classes(target_name)
    if num_classes != 2:
        error_string = (
            'This script works only for binary target variables -- not "{0:s}",'
            ' which has {1:d} classes.'
        ).format(target_name, num_classes)
        raise ValueError(error_string)

    first_validn_time_unix_sec = time_conversion.string_to_unix_sec(
        first_validn_time_string, INPUT_TIME_FORMAT)
    last_validn_time_unix_sec = time_conversion.string_to_unix_sec(
        last_validn_time_string, INPUT_TIME_FORMAT)
    error_checking.assert_is_greater(
        last_validn_time_unix_sec, first_validn_time_unix_sec)

    first_testing_time_unix_sec = time_conversion.string_to_unix_sec(
        first_testing_time_string, INPUT_TIME_FORMAT)
    last_testing_time_unix_sec = time_conversion.string_to_unix_sec(
        last_testing_time_string, INPUT_TIME_FORMAT)
    error_checking.assert_is_greater(
        last_testing_time_unix_sec, first_testing_time_unix_sec)

    # Find files.
    print 'Finding storm-image files for validation and testing...'

    first_validn_spc_date_string = time_conversion.time_to_spc_date_string(
        first_validn_time_unix_sec)
    last_validn_spc_date_string = time_conversion.time_to_spc_date_string(
        last_validn_time_unix_sec)
    these_spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string=first_validn_spc_date_string,
        last_spc_date_string=last_validn_spc_date_string)

    validation_image_file_names = []
    validation_spc_date_strings = []
    for this_spc_date_string in these_spc_date_strings:
        this_file_name = storm_images.find_storm_image_file(
            top_directory_name=top_storm_image_dir_name,
            spc_date_string=this_spc_date_string,
            radar_source=radar_utils.GRIDRAD_SOURCE_ID,
            radar_field_name=DUMMY_RADAR_FIELD_NAME,
            radar_height_m_asl=DUMMY_RADAR_HEIGHT_M_ASL,
            raise_error_if_missing=False)
        if not os.path.isfile(this_file_name):
            continue

        validation_image_file_names.append(this_file_name)
        validation_spc_date_strings.append(this_spc_date_string)

    first_testing_spc_date_string = time_conversion.time_to_spc_date_string(
        first_testing_time_unix_sec)
    last_testing_spc_date_string = time_conversion.time_to_spc_date_string(
        last_testing_time_unix_sec)
    these_spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string=first_testing_spc_date_string,
        last_spc_date_string=last_testing_spc_date_string)

    testing_image_file_names = []
    testing_spc_date_strings = []
    for this_spc_date_string in these_spc_date_strings:
        this_file_name = storm_images.find_storm_image_file(
            top_directory_name=top_storm_image_dir_name,
            spc_date_string=this_spc_date_string,
            radar_source=radar_utils.GRIDRAD_SOURCE_ID,
            radar_field_name=DUMMY_RADAR_FIELD_NAME,
            radar_height_m_asl=DUMMY_RADAR_HEIGHT_M_ASL,
            raise_error_if_missing=False)
        if not os.path.isfile(this_file_name):
            continue

        testing_image_file_names.append(this_file_name)
        testing_spc_date_strings.append(this_spc_date_string)

    num_validation_days = len(validation_image_file_names)
    num_testing_days = len(testing_image_file_names)
    print ('Found {0:d} validation files, {1:d} testing files (one file per SPC'
           ' date).').format(num_validation_days, num_testing_days)

    print (
        'Finding target files (with "{0:s}") for validation and testing...'
    ).format(target_name)

    event_type_string = labels.column_name_to_label_params(
        target_name)[labels.EVENT_TYPE_KEY]
    validation_target_file_names = [''] * num_validation_days
    for i in range(num_validation_days):
        validation_target_file_names[i] = labels.find_label_file(
            top_directory_name=top_target_dir_name,
            event_type_string=event_type_string, file_extension='.nc',
            spc_date_string=validation_spc_date_strings[i])

    testing_target_file_names = [''] * num_testing_days
    for i in range(num_testing_days):
        testing_target_file_names[i] = labels.find_label_file(
            top_directory_name=top_target_dir_name,
            event_type_string=event_type_string, file_extension='.nc',
            spc_date_string=testing_spc_date_strings[i])

    # Do validation.
    print SEPARATOR_STRING
    best_rdp_threshold_s02, _, _ = _do_validation(
        validation_image_file_names=validation_image_file_names,
        validation_target_file_names=validation_target_file_names,
        target_name=target_name,
        first_validn_time_unix_sec=first_validn_time_unix_sec,
        last_validn_time_unix_sec=last_validn_time_unix_sec, max_fom=max_fom,
        rdp_interval_s02=rdp_interval_s02)
    print SEPARATOR_STRING

    _do_testing(
        testing_image_file_names=testing_image_file_names,
        testing_target_file_names=testing_target_file_names,
        target_name=target_name,
        first_testing_time_unix_sec=first_testing_time_unix_sec,
        last_testing_time_unix_sec=last_testing_time_unix_sec,
        best_rdp_threshold_s02=best_rdp_threshold_s02)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_storm_image_dir_name=getattr(
            INPUT_ARG_OBJECT, STORM_IMAGE_DIR_ARG_NAME),
        top_target_dir_name=getattr(
            INPUT_ARG_OBJECT, TARGET_DIRECTORY_ARG_NAME),
        target_name=getattr(INPUT_ARG_OBJECT, TARGET_NAME_ARG_NAME),
        first_validn_time_string=getattr(
            INPUT_ARG_OBJECT, FIRST_VALIDN_TIME_ARG_NAME),
        last_validn_time_string=getattr(
            INPUT_ARG_OBJECT, LAST_VALIDN_TIME_ARG_NAME),
        first_testing_time_string=getattr(
            INPUT_ARG_OBJECT, FIRST_TESTING_TIME_ARG_NAME),
        last_testing_time_string=getattr(
            INPUT_ARG_OBJECT, LAST_TESTING_TIME_ARG_NAME),
        max_fom=getattr(INPUT_ARG_OBJECT, MAX_FOM_ARG_NAME),
        rdp_interval_s02=getattr(INPUT_ARG_OBJECT, RDP_INTERVAL_ARG_NAME))
