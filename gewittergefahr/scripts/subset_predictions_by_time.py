"""Subsets ungridded predictions by time."""

import argparse
from gewittergefahr.gg_utils import temporal_subsetting
from gewittergefahr.deep_learning import prediction_io

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_FILE_ARG_NAME = 'input_file_name'
NUM_MONTHS_ARG_NAME = 'num_months_per_chunk'
NUM_HOURS_ARG_NAME = 'num_hours_per_chunk'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file (with predictions to be subset).  Will be read by '
    '`prediction_io.read_ungridded_predictions`.')

NUM_MONTHS_HELP_STRING = (
    'Number of months in each chunk.  Must be in the list below.  Or, if you do'
    ' not want to subset by month, make this negative.\n{0:s}'
).format(str(temporal_subsetting.VALID_MONTH_COUNTS))

NUM_HOURS_HELP_STRING = (
    'Number of hours in each chunk.  Must be in the list below.  Or, if you do'
    ' not want to subset by hour, make this negative.\n{0:s}'
).format(str(temporal_subsetting.VALID_HOUR_COUNTS))

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory.  Output files (each subset by month or'
    ' hour) will be saved to subdirectories herein.  Exact file locations will '
    'be determined by `prediction_io.find_ungridded_file`, and files will be '
    'written by `prediction_io.write_ungridded_predictions`.')

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
         output_dir_name):
    """Subsets ungridded predictions by time.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param num_months_per_chunk: Same.
    :param num_hours_per_chunk: Same.
    :param output_dir_name: Same.
    """

    if num_months_per_chunk > 0:
        chunk_to_months_dict = temporal_subsetting.get_monthly_chunks(
            num_months_per_chunk=num_months_per_chunk, verbose=True)

        num_monthly_chunks = len(chunk_to_months_dict.keys())
        print(SEPARATOR_STRING)
    else:
        num_monthly_chunks = 0

    if num_hours_per_chunk > 0:
        chunk_to_hours_dict = temporal_subsetting.get_hourly_chunks(
            num_hours_per_chunk=num_hours_per_chunk, verbose=True)

        num_hourly_chunks = len(chunk_to_hours_dict.keys())
        print(SEPARATOR_STRING)
    else:
        num_hourly_chunks = 0

    print('Reading input data from: "{0:s}"...'.format(input_file_name))
    prediction_dict = prediction_io.read_ungridded_predictions(input_file_name)
    storm_times_unix_sec = prediction_dict[prediction_io.STORM_TIMES_KEY]

    storm_months = None

    for i in range(num_monthly_chunks):
        these_storm_indices, storm_months = (
            temporal_subsetting.get_events_in_months(
                event_months=storm_months,
                event_times_unix_sec=storm_times_unix_sec,
                desired_months=chunk_to_months_dict[i], verbose=True)
        )

        this_prediction_dict = prediction_io.subset_ungridded_predictions(
            prediction_dict=prediction_dict,
            desired_storm_indices=these_storm_indices)

        this_output_file_name = prediction_io.find_ungridded_file(
            directory_name=output_dir_name,
            months_in_subset=chunk_to_months_dict[i],
            raise_error_if_missing=False)

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

    storm_hours = None

    for i in range(num_hourly_chunks):
        these_storm_indices, storm_hours = (
            temporal_subsetting.get_events_in_hours(
                event_hours=storm_hours,
                event_times_unix_sec=storm_times_unix_sec,
                desired_hours=chunk_to_hours_dict[i], verbose=True)
        )

        this_prediction_dict = prediction_io.subset_ungridded_predictions(
            prediction_dict=prediction_dict,
            desired_storm_indices=these_storm_indices)

        this_output_file_name = prediction_io.find_ungridded_file(
            directory_name=output_dir_name,
            hours_in_subset=chunk_to_hours_dict[i],
            raise_error_if_missing=False)

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
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
