"""Downsamples storm objects, based on target values."""

import os.path
import argparse
import numpy
import pandas
from gewittergefahr.gg_utils import labels
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils

LARGE_INTEGER = int(1e12)
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_DIR_ARG_NAME = 'input_target_dir_name'
TARGET_NAME_ARG_NAME = 'target_name'
FIRST_DATE_ARG_NAME = 'first_spc_date_string'
LAST_DATE_ARG_NAME = 'last_spc_date_string'
CLASS_FRACTION_KEYS_ARG_NAME = 'class_fraction_keys'
CLASS_FRACTION_VALUES_ARG_NAME = 'class_fraction_values'
OUTPUT_DIR_ARG_NAME = 'output_target_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of top-level input directory with target values.  Files therein will '
    'be located by `labels.find_label_file` and read by '
    '`labels.read_labels_from_netcdf`.')

TARGET_NAME_HELP_STRING = (
    'Name of target variable on which to base downsampling.')

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  Downsampling will be based on all storm '
    'objects from `{0:s}`...`{1:s}` and applied to the same storm objects.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

CLASS_FRACTION_KEYS_HELP_STRING = (
    'List of keys.  Each key is the integer encoding a class (-2 for "dead '
    'storm").  Corresponding values in `{0:s}` are sampling fractions.'
).format(CLASS_FRACTION_VALUES_ARG_NAME)

CLASS_FRACTION_VALUES_HELP_STRING = 'See doc for `{0:s}`.'.format(
    CLASS_FRACTION_KEYS_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory for downsampled target values.  New '
    'files will be written by `labels.write_labels_to_netcdf`, to locations in '
    'this directory determined by `labels.find_label_file`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_NAME_ARG_NAME, type=str, required=True,
    help=TARGET_NAME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CLASS_FRACTION_KEYS_ARG_NAME, type=int, nargs='+',
    required=True, help=CLASS_FRACTION_KEYS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CLASS_FRACTION_VALUES_ARG_NAME, type=float, nargs='+',
    required=True, help=CLASS_FRACTION_VALUES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(top_input_dir_name, target_name, first_spc_date_string,
         last_spc_date_string, class_fraction_keys, class_fraction_values,
         top_output_dir_name):
    """Downsamples storm objects, based on target values.

    This is effectively the main method.

    :param top_input_dir_name: See documentation at top of file.
    :param target_name: Same.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param class_fraction_keys: Same.
    :param class_fraction_values: Same.
    :param top_output_dir_name: Same.
    """

    spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string)

    class_fraction_dict = dict(zip(class_fraction_keys, class_fraction_values))
    target_param_dict = labels.column_name_to_label_params(target_name)

    input_target_file_names = []
    spc_date_string_by_file = []

    for this_spc_date_string in spc_date_strings:
        this_file_name = labels.find_label_file(
            top_directory_name=top_input_dir_name,
            event_type_string=target_param_dict[labels.EVENT_TYPE_KEY],
            file_extension='.nc', spc_date_string=this_spc_date_string,
            raise_error_if_missing=False)
        if not os.path.isfile(this_file_name):
            continue

        input_target_file_names.append(this_file_name)
        spc_date_string_by_file.append(this_spc_date_string)

    num_files = len(input_target_file_names)
    spc_date_strings = spc_date_string_by_file
    target_dict_by_file = [None] * num_files

    storm_ids = []
    storm_times_unix_sec = numpy.array([], dtype=int)
    target_values = numpy.array([], dtype=int)

    for i in range(num_files):
        print 'Reading "{0:s}" from: "{1:s}"...'.format(
            target_name, input_target_file_names[i])
        target_dict_by_file[i] = labels.read_labels_from_netcdf(
            netcdf_file_name=input_target_file_names[i], label_name=target_name)

        storm_ids += target_dict_by_file[i][labels.STORM_IDS_KEY]
        storm_times_unix_sec = numpy.concatenate((
            storm_times_unix_sec, target_dict_by_file[i][labels.VALID_TIMES_KEY]
        ))
        target_values = numpy.concatenate((
            target_values, target_dict_by_file[i][labels.LABEL_VALUES_KEY]
        ))

    print SEPARATOR_STRING

    good_indices = numpy.where(target_values != labels.INVALID_STORM_INTEGER)[0]
    storm_ids = [storm_ids[k] for k in good_indices]
    storm_times_unix_sec = storm_times_unix_sec[good_indices]
    target_values = target_values[good_indices]

    unique_target_values, unique_counts = numpy.unique(
        target_values, return_counts=True)
    for k in range(len(unique_target_values)):
        print '{0:d} examples with target class = {1:d}'.format(
            unique_counts[k], unique_target_values[k])
    print '\n'

    first_indices_to_keep = dl_utils.sample_by_class(
        sampling_fraction_by_class_dict=class_fraction_dict,
        target_name=target_name, target_values=target_values,
        num_examples_total=LARGE_INTEGER)

    num_classes = labels.column_name_to_num_classes(
        column_name=target_name, include_dead_storms=False)
    highest_class_indices = numpy.where(
        target_values[first_indices_to_keep] == num_classes - 1
    )[0]
    highest_class_indices = first_indices_to_keep[highest_class_indices]

    second_indices_to_keep = tracking_utils.find_storm_cells(
        storm_id_by_object=storm_ids,
        desired_storm_ids=[storm_ids[k] for k in highest_class_indices])

    print first_indices_to_keep.shape
    print second_indices_to_keep.shape
    print first_indices_to_keep.dtype
    print second_indices_to_keep.dtype

    indices_to_keep = numpy.unique(numpy.concatenate(
        first_indices_to_keep, second_indices_to_keep))

    storm_ids = [storm_ids[k] for k in indices_to_keep]
    storm_times_unix_sec = storm_times_unix_sec[indices_to_keep]
    target_values = target_values[indices_to_keep]

    unique_target_values, unique_counts = numpy.unique(
        target_values, return_counts=True)
    for k in range(len(unique_target_values)):
        print '{0:d} examples with target class = {1:d}'.format(
            unique_counts[k], unique_target_values[k])
    print '\n'

    for i in range(num_files):
        these_indices = tracking_utils.find_storm_objects(
            all_storm_ids=target_dict_by_file[i][labels.STORM_IDS_KEY],
            all_times_unix_sec=target_dict_by_file[i][labels.VALID_TIMES_KEY],
            storm_ids_to_keep=storm_ids,
            times_to_keep_unix_sec=storm_times_unix_sec, allow_missing=True)

        these_indices = these_indices[these_indices >= 0]
        if len(these_indices) == 0:
            continue

        this_output_dict = {
            tracking_utils.STORM_ID_COLUMN: [
                target_dict_by_file[i][labels.STORM_IDS_KEY][k]
                for k in these_indices
            ],
            tracking_utils.TIME_COLUMN:
                target_dict_by_file[i][labels.VALID_TIMES_KEY][these_indices],
            target_name:
                target_dict_by_file[i][labels.LABEL_VALUES_KEY][these_indices]
        }
        this_output_table = pandas.DataFrame.from_dict(this_output_dict)

        this_new_file_name = labels.find_label_file(
            top_directory_name=top_output_dir_name,
            event_type_string=target_param_dict[labels.EVENT_TYPE_KEY],
            file_extension='.nc', spc_date_string=spc_date_strings[i],
            raise_error_if_missing=False)

        print (
            'Writing {0:d} downsampled storm objects (out of {1:d} total) to: '
            '"{2:s}"...'
        ).format(
            len(this_output_table.index),
            len(target_dict_by_file[i][labels.STORM_IDS_KEY]),
            this_new_file_name
        )

        labels.write_labels_to_netcdf(
            storm_to_events_table=this_output_table, label_names=[target_name],
            netcdf_file_name=this_new_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        target_name=getattr(INPUT_ARG_OBJECT, TARGET_NAME_ARG_NAME),
        first_spc_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        class_fraction_keys=numpy.array(
            getattr(INPUT_ARG_OBJECT, CLASS_FRACTION_KEYS_ARG_NAME), dtype=int),
        class_fraction_values=numpy.array(
            getattr(INPUT_ARG_OBJECT, CLASS_FRACTION_VALUES_ARG_NAME),
            dtype=float),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
