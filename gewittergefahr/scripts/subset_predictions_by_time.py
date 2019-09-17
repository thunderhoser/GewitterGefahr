"""Subsets ungridded predictions by time."""

import os.path
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import prediction_io

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_MONTHS_IN_YEAR = 12
NUM_HOURS_IN_DAY = 24

VALID_MONTH_COUNTS = numpy.array([1, 3], dtype=int)
VALID_HOUR_COUNTS = numpy.array([1, 3, 6], dtype=int)

INPUT_FILE_ARG_NAME = 'input_file_name'
NUM_MONTHS_ARG_NAME = 'num_months_per_chunk'
NUM_HOURS_ARG_NAME = 'num_hours_per_chunk'
OUTPUT_DIR_ARG_NAME = 'top_output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by '
    '`prediction_io.read_ungridded_predictions`.')

# TODO(thunderhoser): Don't need these arguments anymore.
NUM_MONTHS_HELP_STRING = (
    'Number of months in each chunk.  Must be in the following list.\n{0:s}'
).format(str(VALID_MONTH_COUNTS))

NUM_HOURS_HELP_STRING = (
    'Number of hours in each chunk.  Must be in the following list.\n{0:s}'
).format(str(VALID_HOUR_COUNTS))

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory.  Output files (each subset by month or'
    ' hour) will be saved to subdirectories herein.  Exact file locations will '
    'be determined by `prediction_io.find_file`, and files will be written by '
    '`prediction_io.write_ungridded_predictions`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_MONTHS_ARG_NAME, type=int, required=False, default=1,
    help=NUM_MONTHS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_HOURS_ARG_NAME, type=int, required=False, default=1,
    help=NUM_HOURS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _get_months_in_each_chunk(num_months_per_chunk):
    """Returns list of months in each chunk.

    :param: num_months_per_chunk: Number of months per chunk.
    :return: chunk_to_months_dict: Dictionary, where each key is an index and
        the corresponding value is a 1-D numpy array of months (from 1...12).
    :raises: ValueError: if `num_months_per_chunk not in VALID_MONTH_COUNTS`.
    """

    if num_months_per_chunk not in VALID_MONTH_COUNTS:
        error_string = (
            '\n{0:s}\nValid numbers of months per chunk (listed above) do not '
            'include {1:d}.'
        ).format(str(VALID_MONTH_COUNTS), num_months_per_chunk)

        raise ValueError(error_string)

    chunk_to_months_dict = dict()

    if num_months_per_chunk == 1:
        for i in range(NUM_MONTHS_IN_YEAR):
            chunk_to_months_dict[i] = numpy.array([i + 1], dtype=int)
    else:
        chunk_to_months_dict[0] = numpy.array([12, 1, 2], dtype=int)
        chunk_to_months_dict[1] = numpy.array([3, 4, 5], dtype=int)
        chunk_to_months_dict[2] = numpy.array([6, 7, 8], dtype=int)
        chunk_to_months_dict[3] = numpy.array([9, 10, 11], dtype=int)

    num_chunks = len(chunk_to_months_dict.keys())

    for i in range(num_chunks):
        print('Months in {0:d}th chunk = {1:s}'.format(
            i + 1, str(chunk_to_months_dict[i])
        ))

    return chunk_to_months_dict


def _get_hours_in_each_chunk(num_hours_per_chunk):
    """Returns list of hours in each chunk.

    :param: num_hours_per_chunk: Number of hours per chunk.
    :return: chunk_to_hours_dict: Dictionary, where each key is an index and
        the corresponding value is a 1-D numpy array of hours (from 0...23).
    :raises: ValueError: if `num_hours_per_chunk not in VALID_HOUR_COUNTS`.
    """

    if num_hours_per_chunk not in VALID_HOUR_COUNTS:
        error_string = (
            '\n{0:s}\nValid numbers of hours per chunk (listed above) do not '
            'include {1:d}.'
        ).format(str(VALID_HOUR_COUNTS), num_hours_per_chunk)

        raise ValueError(error_string)

    chunk_to_hours_dict = dict()
    num_hourly_chunks = int(numpy.round(
        NUM_HOURS_IN_DAY / num_hours_per_chunk
    ))

    for i in range(num_hourly_chunks):
        chunk_to_hours_dict[i] = numpy.linspace(
            i * num_hours_per_chunk, (i + 1) * num_hours_per_chunk - 1,
            num=num_hours_per_chunk, dtype=int
        )

        print('Hours in {0:d}th chunk = {1:s}'.format(
            i + 1, str(chunk_to_hours_dict[i])
        ))

    return chunk_to_hours_dict


def _subset_prediction_dict(prediction_dict, desired_storm_indices):
    """Subsets dictionary with predictions.

    :param prediction_dict: Dictionary returned by
        `prediction_io.read_ungridded_predictions`.
    :param desired_storm_indices: 1-D numpy array with indices of desired storm
        objects.
    :return: small_prediction_dict: Same as input but maybe with fewer storm
        objects.
    """

    small_prediction_dict = dict()

    for this_key in prediction_dict:
        if isinstance(prediction_dict[this_key], list):
            small_prediction_dict[this_key] = [
                prediction_dict[this_key][k] for k in desired_storm_indices
            ]
        elif isinstance(prediction_dict[this_key], numpy.ndarray):
            small_prediction_dict[this_key] = prediction_dict[this_key][
                desired_storm_indices, ...]
        else:
            small_prediction_dict[this_key] = prediction_dict[this_key]

    return small_prediction_dict


def _find_storm_objects_in_months(desired_months, prediction_dict,
                                  storm_months=None):
    """Finds storm objects in desired months.

    N = number of storm objects

    :param desired_months: 1-D numpy array of desired months (range 1...12).
    :param prediction_dict: Dictionary returned by
        `prediction_io.read_ungridded_predictions` (with N storm objects).
    :param storm_months: length-N numpy array of valid months.  If this is None,
        months will be determined on the fly from `prediction_dict`.
    :return: small_prediction_dict: Same as input but maybe with fewer storm
        objects.
    :return: storm_months: See input doc.
    """

    if storm_months is None:
        storm_times_unix_sec = prediction_dict[prediction_io.STORM_TIMES_KEY]

        storm_months = numpy.array([
            int(time_conversion.unix_sec_to_string(t, '%m'))
            for t in storm_times_unix_sec
        ], dtype=int)

    desired_storm_flags = numpy.array(
        [m in desired_months for m in storm_months], dtype=bool
    )
    desired_storm_indices = numpy.where(desired_storm_flags)[0]

    print('{0:d} of {1:d} storm objects are in months {2:s}!'.format(
        len(desired_storm_indices), len(desired_storm_flags),
        str(desired_months)
    ))

    small_prediction_dict = _subset_prediction_dict(
        prediction_dict=prediction_dict,
        desired_storm_indices=desired_storm_indices)

    return small_prediction_dict, storm_months


def _find_storm_objects_in_hours(desired_hours, prediction_dict,
                                 storm_hours=None):
    """Finds storm objects in desired hours.

    N = number of storm objects

    :param desired_hours: 1-D numpy array of desired hours (range 0...23).
    :param prediction_dict: Dictionary returned by
        `prediction_io.read_ungridded_predictions` (with N storm objects).
    :param storm_hours: length-N numpy array of valid hours.  If this is None,
        hours will be determined on the fly from `prediction_dict`.
    :return: small_prediction_dict: Same as input but maybe with fewer storm
        objects.
    :return: storm_hours: See input doc.
    """

    if storm_hours is None:
        storm_times_unix_sec = prediction_dict[prediction_io.STORM_TIMES_KEY]

        storm_hours = numpy.array([
            int(time_conversion.unix_sec_to_string(t, '%H'))
            for t in storm_times_unix_sec
        ], dtype=int)

    desired_storm_flags = numpy.array(
        [h in desired_hours for h in storm_hours], dtype=bool
    )
    desired_storm_indices = numpy.where(desired_storm_flags)[0]

    print('{0:d} of {1:d} storm objects are in hours {2:s}!'.format(
        len(desired_storm_indices), len(desired_storm_flags),
        str(desired_hours)
    ))

    small_prediction_dict = _subset_prediction_dict(
        prediction_dict=prediction_dict,
        desired_storm_indices=desired_storm_indices)

    return small_prediction_dict, storm_hours


def _run(input_file_name, num_months_per_chunk, num_hours_per_chunk,
         top_output_dir_name):
    """Subsets ungridded predictions by time.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param num_months_per_chunk: Same.
    :param num_hours_per_chunk: Same.
    :param top_output_dir_name: Same.
    """

    pathless_input_file_name = os.path.split(input_file_name)[-1]

    chunk_to_months_dict = _get_months_in_each_chunk(num_months_per_chunk)
    print(SEPARATOR_STRING)

    print('Reading input data from: "{0:s}"...'.format(input_file_name))
    prediction_dict = prediction_io.read_ungridded_predictions(input_file_name)

    storm_months = None
    num_monthly_chunks = len(chunk_to_months_dict.keys())

    for i in range(num_monthly_chunks):
        this_prediction_dict, storm_months = _find_storm_objects_in_months(
            desired_months=chunk_to_months_dict[i],
            prediction_dict=prediction_dict, storm_months=storm_months)

        this_subdir_name = '-'.join([
            '{0:02d}'.format(m) for m in chunk_to_months_dict[i]
        ])
        this_output_file_name = '{0:s}/months={1:s}/{2:s}'.format(
            top_output_dir_name, this_subdir_name, pathless_input_file_name)

        file_system_utils.mkdir_recursive_if_necessary(
            file_name=this_output_file_name)

        print('Writing temporal subset to: "{0:s}"...'.format(
            this_output_file_name
        ))

        prediction_io.write_ungridded_predictions(
            netcdf_file_name=this_output_file_name,
            class_probability_matrix=this_prediction_dict[
                prediction_io.PROBABILITY_MATRIX_KEY],
            storm_ids=this_prediction_dict[prediction_io.STORM_IDS_KEY],
            storm_times_unix_sec=this_prediction_dict[
                prediction_io.STORM_TIMES_KEY],
            target_name=this_prediction_dict[prediction_io.TARGET_NAME_KEY],
            observed_labels=this_prediction_dict[
                prediction_io.OBSERVED_LABELS_KEY]
        )

        print(SEPARATOR_STRING)

    chunk_to_hours_dict = _get_hours_in_each_chunk(num_hours_per_chunk)
    print(SEPARATOR_STRING)

    storm_hours = None
    num_hourly_chunks = len(chunk_to_hours_dict.keys())

    for i in range(num_hourly_chunks):
        this_prediction_dict, storm_hours = _find_storm_objects_in_hours(
            desired_hours=chunk_to_hours_dict[i],
            prediction_dict=prediction_dict, storm_hours=storm_hours)

        this_subdir_name = '-'.join([
            '{0:02d}'.format(h) for h in chunk_to_hours_dict[i]
        ])
        this_output_file_name = '{0:s}/hours={1:s}/{2:s}'.format(
            top_output_dir_name, this_subdir_name, pathless_input_file_name)

        file_system_utils.mkdir_recursive_if_necessary(
            file_name=this_output_file_name)

        print('Writing temporal subset to: "{0:s}"...'.format(
            this_output_file_name
        ))

        prediction_io.write_ungridded_predictions(
            netcdf_file_name=this_output_file_name,
            class_probability_matrix=this_prediction_dict[
                prediction_io.PROBABILITY_MATRIX_KEY],
            storm_ids=this_prediction_dict[prediction_io.STORM_IDS_KEY],
            storm_times_unix_sec=this_prediction_dict[
                prediction_io.STORM_TIMES_KEY],
            target_name=this_prediction_dict[prediction_io.TARGET_NAME_KEY],
            observed_labels=this_prediction_dict[
                prediction_io.OBSERVED_LABELS_KEY]
        )

        if i != num_hourly_chunks - 1:
            print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        num_months_per_chunk=getattr(INPUT_ARG_OBJECT, NUM_MONTHS_ARG_NAME),
        num_hours_per_chunk=getattr(INPUT_ARG_OBJECT, NUM_HOURS_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
