"""Converts storm-centered radar images from GridRad to MYRORSS format."""

import argparse
import numpy
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.deep_learning import training_validation_io as trainval_io

LL_SHEAR_OPERATION_DICT = {
    input_examples.RADAR_FIELD_KEY: radar_utils.VORTICITY_NAME,
    input_examples.MIN_HEIGHT_KEY: 0,
    input_examples.MAX_HEIGHT_KEY: 2000,
    input_examples.OPERATION_NAME_KEY: input_examples.MAX_OPERATION_NAME
}

ML_SHEAR_OPERATION_DICT = {
    input_examples.RADAR_FIELD_KEY: radar_utils.VORTICITY_NAME,
    input_examples.MIN_HEIGHT_KEY: 3000,
    input_examples.MAX_HEIGHT_KEY: 6000,
    input_examples.OPERATION_NAME_KEY: input_examples.MAX_OPERATION_NAME
}

VORTICITY_TO_AZ_SHEAR = 2.

INPUT_DIR_ARG_NAME = 'input_storm_image_dir_name'
FIRST_DATE_ARG_NAME = 'first_spc_date_string'
LAST_DATE_ARG_NAME = 'last_spc_date_string'
RESOLUTION_FACTOR_ARG_NAME = 'resolution_factor'
OUTPUT_DIR_ARG_NAME = 'output_storm_image_dir_name'

# TODO(thunderhoser): Allow this to be done for shuffled example files?

INPUT_DIR_HELP_STRING = (
    'Name of top-level directory with input images (in GridRad format).  Files '
    'therein will be found by `input_examples.find_example_file` and read by '
    '`input_examples.read_examples`.')

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  Images will be converted for all SPC dates '
    'in period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

NUM_ROWS_HELP_STRING = (
    'Number of rows in new (MYRORSS-style) reflectivity grid.  Azimuthal-shear '
    'grid will have twice as many rows.')

RESOLUTION_FACTOR_HELP_STRING = (
    'Resolution factor.  Horizontal resolution of reflectivity grids will be '
    'upsampled this factor.  Horizontal resolution of az-shear grids will be '
    'upsampled by twice this factor.')

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level directory for output images (in MYRORSS format).  Files '
    'will be written by `input_examples.write_examples` to locations therein '
    'determined by `input_examples.find_example_file`.')

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
    '--' + RESOLUTION_FACTOR_ARG_NAME, type=int, required=True,
    help=RESOLUTION_FACTOR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _convert_one_file(input_file_name, resolution_factor):
    """Converts radar images in one file from GridRad to MYRORSS format.

    :param input_file_name: Path to input file (with GridRad examples).  Will be
        read by `input_examples.read_example_file`.
    :param resolution_factor: See documentation at top of file.
    """

    print('Reading GridRad examples from: "{0:s}"...'.format(input_file_name))
    example_dict = input_examples.read_example_file(
        netcdf_file_name=input_file_name, read_all_target_vars=True)

    refl_heights_m_agl = example_dict[input_examples.RADAR_HEIGHTS_KEY] + 0
    refl_index = example_dict[input_examples.RADAR_FIELDS_KEY].index(
        radar_utils.REFL_NAME)

    reflectivity_matrix_dbz = trainval_io.upsample_reflectivity(
        reflectivity_matrix_dbz=example_dict[
            input_examples.RADAR_IMAGE_MATRIX_KEY][..., refl_index],
        upsampling_factor=resolution_factor
    )

    reflectivity_matrix_dbz = numpy.expand_dims(
        reflectivity_matrix_dbz, axis=-1)

    example_dict = input_examples.reduce_examples_3d_to_2d(
        example_dict=example_dict,
        list_of_operation_dicts=[
            LL_SHEAR_OPERATION_DICT, ML_SHEAR_OPERATION_DICT
        ]
    )

    field_names = example_dict[input_examples.RADAR_FIELDS_KEY]
    min_heights_m_asl = example_dict[input_examples.MIN_RADAR_HEIGHTS_KEY]

    ll_shear_index = numpy.where(numpy.logical_and(
        numpy.array(field_names) == radar_utils.VORTICITY_NAME,
        min_heights_m_asl ==
        LL_SHEAR_OPERATION_DICT[input_examples.MIN_HEIGHT_KEY]
    ))[0]

    ll_shear_matrix_s01 = trainval_io.upsample_reflectivity(
        reflectivity_matrix_dbz=example_dict[
            input_examples.RADAR_IMAGE_MATRIX_KEY][..., ll_shear_index],
        upsampling_factor=resolution_factor * 2
    )

    ml_shear_index = numpy.where(numpy.logical_and(
        numpy.array(field_names) == radar_utils.VORTICITY_NAME,
        min_heights_m_asl ==
        ML_SHEAR_OPERATION_DICT[input_examples.MIN_HEIGHT_KEY]
    ))[0]

    ml_shear_matrix_s01 = trainval_io.upsample_reflectivity(
        reflectivity_matrix_dbz=example_dict[
            input_examples.RADAR_IMAGE_MATRIX_KEY][..., ml_shear_index],
        upsampling_factor=resolution_factor * 2
    )

    azimuthal_shear_matrix_s01 = VORTICITY_TO_AZ_SHEAR * numpy.concatenate(
        (ll_shear_matrix_s01, ml_shear_matrix_s01), axis=-1
    )

    example_dict[input_examples.REFL_IMAGE_MATRIX_KEY] = reflectivity_matrix_dbz
    example_dict[
        input_examples.AZ_SHEAR_IMAGE_MATRIX_KEY] = azimuthal_shear_matrix_s01

    example_dict[input_examples.RADAR_HEIGHTS_KEY] = refl_heights_m_agl
    example_dict[input_examples.RADAR_FIELDS_KEY] = [
        radar_utils.LOW_LEVEL_SHEAR_NAME, radar_utils.MID_LEVEL_SHEAR_NAME
    ]
    example_dict[input_examples.ROTATED_GRID_SPACING_KEY] *= resolution_factor

    example_dict.pop(input_examples.RADAR_IMAGE_MATRIX_KEY, None)
    example_dict.pop(input_examples.MIN_RADAR_HEIGHTS_KEY, None)
    example_dict.pop(input_examples.MAX_RADAR_HEIGHTS_KEY, None)
    example_dict.pop(input_examples.RADAR_LAYER_OPERATION_NAMES_KEY, None)

    print(example_dict.keys())


def _run(top_input_dir_name, first_spc_date_string, last_spc_date_string,
         resolution_factor, top_output_dir_name):
    """Converts storm-centered radar images from GridRad to MYRORSS format.

    This is effectively the main method.

    :param top_input_dir_name: See documentation at top of file.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param resolution_factor: Same.
    :param top_output_dir_name: Same.
    """

    spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string)

    input_example_file_names = input_examples.find_many_example_files(
        top_directory_name=top_input_dir_name, shuffled=False,
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string,
        raise_error_if_any_missing=True)

    num_spc_dates = len(spc_date_strings)

    for i in range(num_spc_dates):
        _convert_one_file(input_file_name=input_example_file_names[i],
                          resolution_factor=resolution_factor)

        break


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_spc_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        resolution_factor=getattr(
            INPUT_ARG_OBJECT, RESOLUTION_FACTOR_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
