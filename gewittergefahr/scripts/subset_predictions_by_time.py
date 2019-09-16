"""Subsets ungridded predictions by time."""

import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.deep_learning import prediction_io

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


def _run(input_file_name, num_months_per_chunk, num_hours_per_chunk,
         top_output_dir_name):
    """Subsets ungridded predictions by time.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param num_months_per_chunk: Same.
    :param num_hours_per_chunk: Same.
    :param top_output_dir_name: Same.
    :raises: ValueError: if `num_months_per_chunk not in VALID_MONTH_COUNTS`.
    :raises: ValueError: if `num_hours_per_chunk not in VALID_HOUR_COUNTS`.
    """

    if num_months_per_chunk not in VALID_MONTH_COUNTS:
        error_string = (
            '\n{0:s}\nValid numbers of months per chunk (listed above) do not '
            'include {1:d}.'
        ).format(str(VALID_MONTH_COUNTS), num_months_per_chunk)

        raise ValueError(error_string)

    if num_hours_per_chunk not in VALID_HOUR_COUNTS:
        error_string = (
            '\n{0:s}\nValid numbers of hours per chunk (listed above) do not '
            'include {1:d}.'
        ).format(str(VALID_HOUR_COUNTS), num_hours_per_chunk)

        raise ValueError(error_string)

    chunk_to_months = dict()

    if num_months_per_chunk == 1:
        for i in range(NUM_MONTHS_IN_YEAR):
            chunk_to_months[i] = numpy.array([i + 1], dtype=int)
    else:
        chunk_to_months[0] = numpy.array([12, 1, 2], dtype=int)
        chunk_to_months[1] = numpy.array([3, 4, 5], dtype=int)
        chunk_to_months[2] = numpy.array([6, 7, 8], dtype=int)
        chunk_to_months[3] = numpy.array([9, 10, 11], dtype=int)

    chunk_to_hours = dict()
    this_num_chunks = int(numpy.round(
        NUM_HOURS_IN_DAY / num_hours_per_chunk
    ))

    # TODO(thunderhoser): Finish things here.


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        num_months_per_chunk=getattr(INPUT_ARG_OBJECT, NUM_MONTHS_ARG_NAME),
        num_hours_per_chunk=getattr(INPUT_ARG_OBJECT, NUM_HOURS_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
