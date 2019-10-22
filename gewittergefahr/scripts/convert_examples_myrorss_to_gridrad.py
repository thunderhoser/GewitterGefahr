"""Converts examples from MYRORSS to GridRad format."""

import os.path
import argparse
import numpy
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.deep_learning import training_validation_io as trainval_io

AZ_SHEAR_TO_VORTICITY = 0.5
MAX_LL_SHEAR_HEIGHT_M_AGL = 2000
REFL_HEIGHTS_M_AGL = numpy.array(
    [1000, 2000, 3000, 4000, 5000, 6000], dtype=int
)
NEW_RADAR_HEIGHTS_M_AGL = numpy.array(
    [0, 1000, 2000, 3000, 4000, 5000, 6000], dtype=int
)

INPUT_DIR_ARG_NAME = 'input_example_dir_name'
FIRST_DATE_ARG_NAME = 'first_spc_date_string'
LAST_DATE_ARG_NAME = 'last_spc_date_string'
NUM_EX_PER_BATCH_ARG_NAME = 'num_examples_per_batch'
OUTPUT_DIR_ARG_NAME = 'output_example_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of top-level directory with original examples (in MYRORSS format).  '
    'Files therein will be found by `input_examples.find_example_file` and read'
    ' by `input_examples.read_example_file`.')

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  Examples will be converted for all SPC '
    'dates in period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

NUM_EX_PER_BATCH_HELP_STRING = (
    'Number of examples per batch.  Examples will read and written in batches '
    'of this size.')

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level directory for new examples (in GridRad format).  Files '
    'will be written by `input_examples.write_example_file` to locations '
    'therein determined by `input_examples.find_example_file`.')

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
    '--' + NUM_EX_PER_BATCH_ARG_NAME, type=int, required=False, default=1000,
    help=NUM_EX_PER_BATCH_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _convert_one_file_selected_examples(
        input_file_name, output_file_name, full_storm_id_strings,
        storm_times_unix_sec, append_to_file):
    """Converts selected examples in one file from MYRORSS to GridRad format.

    E = number of examples

    :param input_file_name: See doc for `_convert_one_file`.
    :param output_file_name: Same.
    :param full_storm_id_strings: length-E list of storm IDs.
    :param storm_times_unix_sec: length-E numpy array of storm times.
    :param append_to_file: Boolean flag.  If True, will append new examples to
        output file.  If False, will overwrite output file.
    """

    print('Reading MYRORSS examples from: "{0:s}"...'.format(input_file_name))
    example_dict = input_examples.read_specific_examples(
        netcdf_file_name=input_file_name, read_all_target_vars=True,
        full_storm_id_strings=full_storm_id_strings,
        storm_times_unix_sec=storm_times_unix_sec,
        radar_heights_to_keep_m_agl=REFL_HEIGHTS_M_AGL)

    # Add surface reflectivity, then double horizontal resolution.
    reflectivity_matrix_dbz = example_dict[
        input_examples.REFL_IMAGE_MATRIX_KEY][..., 0]

    reflectivity_matrix_dbz = numpy.concatenate(
        (reflectivity_matrix_dbz, reflectivity_matrix_dbz[..., [0]]), axis=-1
    )

    reflectivity_matrix_dbz = trainval_io.upsample_reflectivity(
        reflectivity_matrix_dbz=reflectivity_matrix_dbz, upsampling_factor=2)

    # Create vorticity matrix.
    shear_field_names = example_dict[input_examples.RADAR_FIELDS_KEY]
    ll_shear_index = shear_field_names.index(radar_utils.LOW_LEVEL_SHEAR_NAME)
    ml_shear_index = shear_field_names.index(radar_utils.MID_LEVEL_SHEAR_NAME)

    ll_shear_matrix_s01 = example_dict[
        input_examples.AZ_SHEAR_IMAGE_MATRIX_KEY
    ][..., ll_shear_index]

    ml_shear_matrix_s01 = example_dict[
        input_examples.AZ_SHEAR_IMAGE_MATRIX_KEY
    ][..., ml_shear_index]

    num_radar_heights = len(NEW_RADAR_HEIGHTS_M_AGL)
    these_dimensions = numpy.array(
        ll_shear_matrix_s01.shape + (num_radar_heights,), dtype=int
    )
    vorticity_matrix_s01 = numpy.full(these_dimensions, numpy.nan)

    for k in range(num_radar_heights):
        if NEW_RADAR_HEIGHTS_M_AGL[k] > MAX_LL_SHEAR_HEIGHT_M_AGL:
            vorticity_matrix_s01[..., k] = ml_shear_matrix_s01
        else:
            vorticity_matrix_s01[..., k] = ll_shear_matrix_s01

    vorticity_matrix_s01 *= AZ_SHEAR_TO_VORTICITY
    radar_matrix = numpy.stack(
        (reflectivity_matrix_dbz, vorticity_matrix_s01), axis=-1
    )

    example_dict[input_examples.RADAR_IMAGE_MATRIX_KEY] = radar_matrix
    example_dict[input_examples.RADAR_HEIGHTS_KEY] = NEW_RADAR_HEIGHTS_M_AGL
    example_dict[input_examples.RADAR_FIELDS_KEY] = [
        radar_utils.REFL_NAME, radar_utils.VORTICITY_NAME
    ]
    example_dict[input_examples.ROTATED_GRID_SPACING_KEY] *= 0.5

    example_dict.pop(input_examples.REFL_IMAGE_MATRIX_KEY, None)
    example_dict.pop(input_examples.AZ_SHEAR_IMAGE_MATRIX_KEY, None)

    print('Writing examples in GridRad format to: "{0:s}"...'.format(
        output_file_name
    ))

    input_examples.write_example_file(
        netcdf_file_name=output_file_name, example_dict=example_dict,
        append_to_file=append_to_file)


def _convert_one_file(input_file_name, output_file_name,
                      num_examples_per_batch):
    """Converts examples in one file from MYRORSS to GridRad format.

    :param input_file_name: Path to input file (with MYRORSS examples).  Will be
        read by `input_examples.read_example_file`.
    :param output_file_name: Path to output file (with the same examples but in
        GridRad format).  Will be written by
        `input_examples.write_example_file`.
    :param num_examples_per_batch: See documentation at top of file.
    """

    print('Reading metadata from: "{0:s}"...'.format(input_file_name))
    example_dict = input_examples.read_example_file(
        netcdf_file_name=input_file_name, read_all_target_vars=True,
        metadata_only=True)

    full_storm_id_strings = example_dict[input_examples.FULL_IDS_KEY]
    storm_times_unix_sec = example_dict[input_examples.STORM_TIMES_KEY]
    num_examples = len(full_storm_id_strings)

    for i in range(0, num_examples, num_examples_per_batch):
        this_first_index = i
        this_last_index = min(
            [i + num_examples_per_batch - 1, num_examples - 1]
        )

        _convert_one_file_selected_examples(
            input_file_name=input_file_name,
            output_file_name=output_file_name,
            full_storm_id_strings=
            full_storm_id_strings[this_first_index:(this_last_index + 1)],
            storm_times_unix_sec=
            storm_times_unix_sec[this_first_index:(this_last_index + 1)],
            append_to_file=i > 0
        )


def _run(top_input_dir_name, first_spc_date_string, last_spc_date_string,
         num_examples_per_batch, top_output_dir_name):
    """Converts examples from MYRORSS to GridRad format.

    This is effectively the main method.

    :param top_input_dir_name: See documentation at top of file.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param num_examples_per_batch: Same.
    :param top_output_dir_name: Same.
    """

    spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string)

    input_file_names = [
        input_examples.find_example_file(
            top_directory_name=top_input_dir_name, shuffled=False,
            spc_date_string=d, raise_error_if_missing=False
        )
        for d in spc_date_strings
    ]

    output_file_names = [
        input_examples.find_example_file(
            top_directory_name=top_output_dir_name, shuffled=False,
            spc_date_string=d, raise_error_if_missing=False
        )
        for d in spc_date_strings
    ]

    num_spc_dates = len(spc_date_strings)

    for i in range(num_spc_dates):
        if not os.path.isfile(input_file_names[i]):
            continue

        _convert_one_file(
            input_file_name=input_file_names[i],
            output_file_name=output_file_names[i],
            num_examples_per_batch=num_examples_per_batch
        )

        print('\n')


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_spc_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        num_examples_per_batch=getattr(
            INPUT_ARG_OBJECT, NUM_EX_PER_BATCH_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
