"""Extracts desired examples and writes them to one file."""

import argparse
import numpy
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.deep_learning import input_examples

INPUT_DIR_ARG_NAME = 'input_example_dir_name'
STORM_METAFILE_ARG_NAME = 'input_storm_metafile_name'
NUM_EXAMPLES_ARG_NAME = 'num_examples_in_subset'
SUBSET_RANDOMLY_ARG_NAME = 'subset_randomly'
OUTPUT_FILE_ARG_NAME = 'output_example_file_name'

INPUT_DIR_HELP_STRING = (
    'Name of top-level input directory, containing all examples in daily files.'
    '  Files therein will be found by `input_examples.find_example_file` and '
    'read by `input_examples.read_example_file`.'
)
STORM_METAFILE_HELP_STRING = (
    'Path to Pickle file with storm metadata (ID and valid time for each '
    'desired example).  Will be read by `storm_tracking_io.read_ids_and_times`.'
)
NUM_EXAMPLES_HELP_STRING = (
    'Will write only N examples to file (selected from those in `{0:s}`), where'
    ' N = `{1:s}`.  If you want to write all examples to file, leave this '
    'alone.'
).format(STORM_METAFILE_ARG_NAME, NUM_EXAMPLES_ARG_NAME)

SUBSET_RANDOMLY_HELP_STRING = (
    '[used only if `{0:s}` is specified] Boolean flag.  If 1 (0), will select N'
    ' random examples (the first N examples) from file `{1:s}`.'
).format(NUM_EXAMPLES_ARG_NAME, STORM_METAFILE_ARG_NAME)

OUTPUT_FILE_HELP_STRING = (
    'Path to output file (will be written by '
    '`input_examples.write_example_file`).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + STORM_METAFILE_ARG_NAME, type=str, required=True,
    help=STORM_METAFILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_EXAMPLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SUBSET_RANDOMLY_ARG_NAME, type=int, required=False, default=0,
    help=SUBSET_RANDOMLY_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_example_dir_name, storm_metafile_name, num_examples_in_subset,
         subset_randomly, output_example_file_name):
    """Extracts desired examples and writes them to one file.

    This is effectively the main method.

    :param input_example_dir_name: See documentation at top of file.
    :param storm_metafile_name: Same.
    :param num_examples_in_subset: Same.
    :param subset_randomly: Same.
    :param output_example_file_name: Same.
    """

    print('Reading storm metadata from: "{0:s}"...'.format(storm_metafile_name))
    example_id_strings, example_times_unix_sec = (
        tracking_io.read_ids_and_times(storm_metafile_name)
    )

    if not 0 < num_examples_in_subset < len(example_id_strings):
        num_examples_in_subset = None

    if num_examples_in_subset is not None:
        if subset_randomly:
            these_indices = numpy.linspace(
                0, len(example_id_strings) - 1, num=len(example_id_strings),
                dtype=int
            )
            these_indices = numpy.random.choice(
                these_indices, size=num_examples_in_subset, replace=False
            )

            example_id_strings = [example_id_strings[k] for k in these_indices]
            example_times_unix_sec = example_times_unix_sec[these_indices]
        else:
            example_id_strings = example_id_strings[:num_examples_in_subset]
            example_times_unix_sec = (
                example_times_unix_sec[:num_examples_in_subset]
            )

    example_spc_date_strings = numpy.array([
        time_conversion.time_to_spc_date_string(t)
        for t in example_times_unix_sec
    ])
    spc_date_strings = numpy.unique(example_spc_date_strings)

    example_file_name_by_day = [
        input_examples.find_example_file(
            top_directory_name=input_example_dir_name, shuffled=False,
            spc_date_string=d, raise_error_if_missing=True
        ) for d in spc_date_strings
    ]

    num_days = len(spc_date_strings)

    for i in range(num_days):
        print('Reading data from: "{0:s}"...'.format(
            example_file_name_by_day[i]
        ))
        all_example_dict = input_examples.read_example_file(
            netcdf_file_name=example_file_name_by_day[i],
            read_all_target_vars=True
        )

        these_indices = numpy.where(
            example_spc_date_strings == spc_date_strings[i]
        )[0]

        desired_indices = tracking_utils.find_storm_objects(
            all_id_strings=all_example_dict[input_examples.FULL_IDS_KEY],
            all_times_unix_sec=
            all_example_dict[input_examples.STORM_TIMES_KEY],
            id_strings_to_keep=[example_id_strings[k] for k in these_indices],
            times_to_keep_unix_sec=example_times_unix_sec[these_indices],
            allow_missing=False
        )

        desired_example_dict = input_examples.subset_examples(
            example_dict=all_example_dict, indices_to_keep=desired_indices
        )

        print('Writing {0:d} desired examples to: "{1:s}"...'.format(
            len(desired_indices), output_example_file_name
        ))
        input_examples.write_example_file(
            netcdf_file_name=output_example_file_name,
            example_dict=desired_example_dict, append_to_file=i > 0
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_example_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        storm_metafile_name=getattr(INPUT_ARG_OBJECT, STORM_METAFILE_ARG_NAME),
        num_examples_in_subset=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        subset_randomly=bool(
            getattr(INPUT_ARG_OBJECT, SUBSET_RANDOMLY_ARG_NAME)
        ),
        output_example_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
