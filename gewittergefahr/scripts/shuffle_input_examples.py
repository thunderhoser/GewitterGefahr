"""Shuffles input examples in time and writes them to new file."""

import random
import os.path
import argparse
import numpy
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import input_examples

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_DIR_ARG_NAME = 'input_example_dir_name'
FIRST_DATE_ARG_NAME = 'first_spc_date_string'
LAST_DATE_ARG_NAME = 'last_spc_date_string'
OUTPUT_DIR_ARG_NAME = 'output_example_dir_name'
RADAR_FIELDS_ARG_NAME = 'radar_field_names'
FIRST_BATCH_NUM_ARG_NAME = 'first_output_batch_number'
NUM_EXAMPLES_PER_CHUNK_ARG_NAME = 'num_examples_per_out_chunk'
NUM_EXAMPLES_PER_OUT_FILE_ARG_NAME = 'num_examples_per_out_file'

INPUT_DIR_HELP_STRING = (
    'Name of top-level directory with input files (containing unshuffled '
    'examples).  Files therein will be found by '
    '`input_examples.find_example_file` and read by '
    '`input_examples.read_example_file`.')

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  This script will shuffle examples from the '
    'time period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level directory for output files (containing shuffled '
    'examples).  Files will be written by `input_examples.write_example_file`, '
    'to locations in this directory determined by '
    '`input_examples.find_example_file`.')

RADAR_FIELDS_HELP_STRING = (
    'List of radar fields to output.  Each field must be accepted by '
    '`radar_utils.check_field_name`.  If you leave this argument, all radar '
    'fields will be output.')

FIRST_BATCH_NUM_HELP_STRING = (
    'First batch number (integer).  Used to determine locations of output '
    'files.')

NUM_EXAMPLES_PER_CHUNK_HELP_STRING = (
    'Number of examples per output chunk (all written to the same file).')

NUM_EXAMPLES_PER_OUT_FILE_HELP_STRING = (
    'Number of examples written to each output file.')

DEFAULT_NUM_EXAMPLES_PER_CHUNK = 8
DEFAULT_NUM_EXAMPLES_PER_OUT_FILE = 256

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_FIELDS_ARG_NAME, type=str, nargs='+', required=False,
    default=[''], help=RADAR_FIELDS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_BATCH_NUM_ARG_NAME, type=int, required=True,
    help=FIRST_BATCH_NUM_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_PER_CHUNK_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_EXAMPLES_PER_CHUNK,
    help=NUM_EXAMPLES_PER_CHUNK_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_PER_OUT_FILE_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_EXAMPLES_PER_OUT_FILE,
    help=NUM_EXAMPLES_PER_OUT_FILE_HELP_STRING)


def _find_input_files(
        top_input_dir_name, first_spc_date_string, last_spc_date_string):
    """Finds input files (containing unshuffled examples).

    :param top_input_dir_name: See documentation at top of file.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :return: input_example_file_names: 1-D list of paths to input files.
    :return: num_input_examples: Total number of examples in these files.
    """

    input_example_file_names = input_examples.find_many_example_files(
        top_directory_name=top_input_dir_name, shuffled=False,
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string,
        raise_error_if_any_missing=False)

    num_input_examples = 0

    for this_file_name in input_example_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_example_dict = input_examples.read_example_file(
            netcdf_file_name=this_file_name, metadata_only=True)

        num_input_examples += len(
            this_example_dict[input_examples.FULL_IDS_KEY]
        )

    return input_example_file_names, num_input_examples


def _set_output_locations(
        top_output_dir_name, num_input_examples, num_examples_per_out_file,
        first_output_batch_number):
    """Sets locations of output files.

    :param top_output_dir_name: See documentation at top of file.
    :param num_input_examples: Total number of examples in input files.
    :param num_examples_per_out_file: See documentation at top of file.
    :param first_output_batch_number: Same.
    :return: output_example_file_names: 1-D list of paths to output files.
    """

    num_output_files = int(
        numpy.ceil(float(num_input_examples) / num_examples_per_out_file)
    )

    print((
        'Num input examples = {0:d} ... num examples per output file = {1:d} '
        '... num output files = {2:d}'
    ).format(num_input_examples, num_examples_per_out_file, num_output_files))

    output_example_file_names = [
        input_examples.find_example_file(
            top_directory_name=top_output_dir_name, shuffled=True,
            batch_number=first_output_batch_number + i,
            raise_error_if_missing=False
        ) for i in range(num_output_files)
    ]

    for this_file_name in output_example_file_names:
        if not os.path.isfile(this_file_name):
            continue

        print('Deleting output file: "{0:s}"...'.format(this_file_name))
        os.remove(this_file_name)

    return output_example_file_names


def _shuffle_one_input_file(
        input_example_file_name, radar_field_names, num_examples_per_out_chunk,
        output_example_file_names):
    """Shuffles examples from one input file to many output files.

    :param input_example_file_name: Path to input file.
    :param radar_field_names: See documentation at top of file.
    :param num_examples_per_out_chunk: Same.
    :param output_example_file_names: 1-D list of paths to output files.
    """

    print('Reading data from: "{0:s}"...'.format(input_example_file_name))
    example_dict = input_examples.read_example_file(
        netcdf_file_name=input_example_file_name,
        radar_field_names_to_keep=radar_field_names)

    num_examples = len(example_dict[input_examples.FULL_IDS_KEY])
    shuffled_indices = numpy.linspace(
        0, num_examples - 1, num=num_examples, dtype=int)
    numpy.random.shuffle(shuffled_indices)

    example_dict = input_examples.subset_examples(
        example_dict=example_dict, indices_to_keep=shuffled_indices)

    for j in range(0, num_examples, num_examples_per_out_chunk):
        this_first_index = j
        this_last_index = min(
            [j + num_examples_per_out_chunk - 1, num_examples - 1]
        )

        these_indices = numpy.linspace(
            this_first_index, this_last_index,
            num=this_last_index - this_first_index + 1, dtype=int)

        this_example_dict = input_examples.subset_examples(
            example_dict=example_dict, indices_to_keep=these_indices,
            create_new_dict=True)

        this_output_file_name = random.choice(output_example_file_names)
        print('Writing shuffled examples to: "{0:s}"...'.format(
            this_output_file_name))

        input_examples.write_example_file(
            netcdf_file_name=this_output_file_name,
            example_dict=this_example_dict,
            append_to_file=os.path.isfile(this_output_file_name)
        )


def _run(top_input_dir_name, first_spc_date_string, last_spc_date_string,
         top_output_dir_name, radar_field_names, first_output_batch_number,
         num_examples_per_out_chunk, num_examples_per_out_file):
    """Shuffles input examples in time and writes them to new file.

    This is effectively the main method.

    :param top_input_dir_name: See documentation at top of file.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param top_output_dir_name: Same.
    :param radar_field_names: Same.
    :param first_output_batch_number: Same.
    :param num_examples_per_out_chunk: Same.
    :param num_examples_per_out_file: Same.
    """

    if radar_field_names[0] in ['', 'None']:
        radar_field_names = None

    error_checking.assert_is_geq(num_examples_per_out_chunk, 2)
    error_checking.assert_is_geq(num_examples_per_out_file, 100)

    input_example_file_names, num_input_examples = _find_input_files(
        top_input_dir_name=top_input_dir_name,
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string)
    print(SEPARATOR_STRING)

    output_example_file_names = _set_output_locations(
        top_output_dir_name=top_output_dir_name,
        num_input_examples=num_input_examples,
        num_examples_per_out_file=num_examples_per_out_file,
        first_output_batch_number=first_output_batch_number)
    print(SEPARATOR_STRING)

    for this_file_name in input_example_file_names:
        _shuffle_one_input_file(
            input_example_file_name=this_file_name,
            radar_field_names=radar_field_names,
            num_examples_per_out_chunk=num_examples_per_out_chunk,
            output_example_file_names=output_example_file_names)
        print('\n')


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_spc_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME),
        radar_field_names=getattr(INPUT_ARG_OBJECT, RADAR_FIELDS_ARG_NAME),
        first_output_batch_number=getattr(
            INPUT_ARG_OBJECT, FIRST_BATCH_NUM_ARG_NAME),
        num_examples_per_out_chunk=getattr(
            INPUT_ARG_OBJECT, NUM_EXAMPLES_PER_CHUNK_ARG_NAME),
        num_examples_per_out_file=getattr(
            INPUT_ARG_OBJECT, NUM_EXAMPLES_PER_OUT_FILE_ARG_NAME)
    )
