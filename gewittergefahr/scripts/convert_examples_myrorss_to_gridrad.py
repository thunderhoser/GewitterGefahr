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
OUTPUT_DIR_ARG_NAME = 'output_example_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of top-level directory with original examples (in MYRORSS format).  '
    'Files therein will be found by `input_examples.find_example_file` and read'
    ' by `input_examples.read_example_file`.')

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  Examples will be converted for all SPC '
    'dates in period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

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
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _convert_one_file(input_file_name, output_file_name):
    """Converts examples in one file from MYRORSS to GridRad format.

    :param input_file_name: Path to input file (with MYRORSS examples).  Will be
        read by `input_examples.read_example_file`.
    :param output_file_name: Path to output file (with the same examples but in
        GridRad format).  Will be written by
        `input_examples.write_example_file`.
    """

    print('Reading MYRORSS examples from: "{0:s}"...'.format(input_file_name))
    example_dict = input_examples.read_example_file(
        netcdf_file_name=input_file_name, read_all_target_vars=True,
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
        append_to_file=False)


def _run(top_input_dir_name, first_spc_date_string, last_spc_date_string,
         top_output_dir_name):
    """Converts examples from MYRORSS to GridRad format.

    This is effectively the main method.

    :param top_input_dir_name: See documentation at top of file.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
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
            output_file_name=output_file_names[i]
        )

        print('\n')


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_spc_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
