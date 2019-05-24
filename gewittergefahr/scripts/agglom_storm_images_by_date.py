"""Agglomerates storm-centered radar images by SPC date.

In other words, this script converts the file structure from one file per
field/height pair per time step to one file per field/height pair per SPC date.
"""

import argparse
import numpy
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.deep_learning import storm_images

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

STORM_IMAGE_DIR_ARG_NAME = 'storm_image_dir_name'
RADAR_SOURCE_ARG_NAME = 'radar_source'
SPC_DATE_ARG_NAME = 'spc_date_string'
RADAR_FIELD_NAMES_ARG_NAME = 'radar_field_names'
RADAR_HEIGHTS_ARG_NAME = 'radar_heights_m_agl'
REFL_HEIGHTS_ARG_NAME = 'refl_heights_m_agl'

STORM_IMAGE_DIR_HELP_STRING = (
    'Name of top-level with storm-centered radar images.  One-time files '
    'therein will be found by `storm_images.find_storm_image_file`; one-time '
    'files will be read therefrom by `storm_images.write_storm_images`; and '
    'one-day files will be written thereto by `storm_images.read_storm_images`.'
)

RADAR_SOURCE_HELP_STRING = (
    'Data source.  Must belong to the following list.\n{0:s}'
).format(str(radar_utils.DATA_SOURCE_IDS))

SPC_DATE_HELP_STRING = (
    'SPC (Storm Prediction Center) date in format "yyyymmdd".  Files will be '
    'agglomerated for each field, height, and storm object on this date.'
)

RADAR_FIELD_NAMES_HELP_STRING = (
    'List with names of radar fields.  Each must belong to the following list.'
    '\n{0:s}'
).format(str(radar_utils.RADAR_FIELD_NAMES))

RADAR_HEIGHTS_HELP_STRING = (
    '[used only if {0:s} = "{1:s}"] List of radar heights (metres above ground '
    'level).'
).format(RADAR_SOURCE_ARG_NAME, radar_utils.GRIDRAD_SOURCE_ID)

REFL_HEIGHTS_HELP_STRING = (
    '[used only if {0:s} != "{1:s}"] List of reflectivity heights (metres above'
    ' ground level).'
).format(RADAR_SOURCE_ARG_NAME, radar_utils.GRIDRAD_SOURCE_ID)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + STORM_IMAGE_DIR_ARG_NAME, type=str, required=True,
    help=STORM_IMAGE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_SOURCE_ARG_NAME, type=str, required=True,
    help=RADAR_SOURCE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_FIELD_NAMES_ARG_NAME, type=str, nargs='+', required=True,
    help=RADAR_FIELD_NAMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_HEIGHTS_ARG_NAME, type=int, nargs='+', required=False,
    default=storm_images.DEFAULT_RADAR_HEIGHTS_M_AGL,
    help=RADAR_HEIGHTS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + REFL_HEIGHTS_ARG_NAME, type=int, nargs='+', required=False,
    default=storm_images.DEFAULT_RADAR_HEIGHTS_M_AGL,
    help=REFL_HEIGHTS_HELP_STRING)


def _run(
        top_storm_image_dir_name, radar_source, spc_date_string,
        radar_field_names, radar_heights_m_agl, refl_heights_m_agl):
    """Agglomerates storm-centered radar images by SPC date.

    This is effectively the main method.

    :param top_storm_image_dir_name: See documentation at top of file.
    :param radar_source: Same.
    :param spc_date_string: Same.
    :param radar_field_names: Same.
    :param radar_heights_m_agl: Same.
    :param refl_heights_m_agl: Same.
    """

    start_time_unix_sec = time_conversion.get_start_of_spc_date(spc_date_string)
    end_time_unix_sec = time_conversion.get_end_of_spc_date(spc_date_string)

    if radar_source == radar_utils.GRIDRAD_SOURCE_ID:
        file_dict = storm_images.find_many_files_gridrad(
            top_directory_name=top_storm_image_dir_name,
            radar_field_names=radar_field_names,
            radar_heights_m_agl=radar_heights_m_agl,
            start_time_unix_sec=start_time_unix_sec,
            end_time_unix_sec=end_time_unix_sec,
            one_file_per_time_step=True)

        image_file_name_matrix = file_dict[storm_images.IMAGE_FILE_NAMES_KEY]

        num_times = image_file_name_matrix.shape[0]
        num_field_height_pairs = (
            image_file_name_matrix.shape[1] * image_file_name_matrix.shape[2]
        )
        image_file_name_matrix = numpy.reshape(
            image_file_name_matrix, (num_times, num_field_height_pairs))

    else:
        file_dict = storm_images.find_many_files_myrorss_or_mrms(
            top_directory_name=top_storm_image_dir_name,
            radar_source=radar_source, radar_field_names=radar_field_names,
            start_time_unix_sec=start_time_unix_sec,
            end_time_unix_sec=end_time_unix_sec, one_file_per_time_step=True,
            reflectivity_heights_m_agl=refl_heights_m_agl,
            raise_error_if_all_missing=True,
            raise_error_if_any_missing=True)

        image_file_name_matrix = file_dict[storm_images.IMAGE_FILE_NAMES_KEY]
        num_times = image_file_name_matrix.shape[0]
        num_field_height_pairs = image_file_name_matrix.shape[1]

    print(SEPARATOR_STRING)

    for j in range(num_field_height_pairs):
        storm_image_dict = None

        for i in range(num_times):
            if image_file_name_matrix[i, j] == '':
                continue

            print('Reading data from: "{0:s}"...'.format(
                image_file_name_matrix[i, j]))

            if storm_image_dict is None:
                storm_image_dict = storm_images.read_storm_images(
                    netcdf_file_name=image_file_name_matrix[i, j]
                )
            else:
                this_storm_image_dict = storm_images.read_storm_images(
                    netcdf_file_name=image_file_name_matrix[i, j]
                )

                storm_image_dict[storm_images.FULL_IDS_KEY] += (
                    this_storm_image_dict[storm_images.FULL_IDS_KEY]
                )

                storm_image_dict[storm_images.VALID_TIMES_KEY] = (
                    numpy.concatenate((
                        storm_image_dict[storm_images.VALID_TIMES_KEY],
                        this_storm_image_dict[storm_images.VALID_TIMES_KEY]
                    ))
                )

                storm_image_dict[storm_images.STORM_IMAGE_MATRIX_KEY] = (
                    numpy.concatenate((
                        storm_image_dict[storm_images.STORM_IMAGE_MATRIX_KEY],
                        this_storm_image_dict[
                            storm_images.STORM_IMAGE_MATRIX_KEY]
                    ), axis=0)
                )

        this_file_name = storm_images.find_storm_image_file(
            top_directory_name=top_storm_image_dir_name,
            spc_date_string=spc_date_string, radar_source=radar_source,
            radar_field_name=storm_image_dict[
                storm_images.RADAR_FIELD_NAME_KEY],
            radar_height_m_agl=storm_image_dict[storm_images.RADAR_HEIGHT_KEY],
            raise_error_if_missing=False)

        print('Writing data to: "{0:s}"...'.format(this_file_name))

        storm_images.write_storm_images(
            netcdf_file_name=this_file_name,
            storm_image_matrix=storm_image_dict[
                storm_images.STORM_IMAGE_MATRIX_KEY],
            full_id_strings=storm_image_dict[storm_images.FULL_IDS_KEY],
            valid_times_unix_sec=storm_image_dict[storm_images.VALID_TIMES_KEY],
            radar_field_name=storm_image_dict[
                storm_images.RADAR_FIELD_NAME_KEY],
            radar_height_m_agl=storm_image_dict[storm_images.RADAR_HEIGHT_KEY],
            rotated_grids=storm_image_dict[storm_images.ROTATED_GRIDS_KEY],
            rotated_grid_spacing_metres=storm_image_dict[
                storm_images.ROTATED_GRID_SPACING_KEY]
        )

        print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_storm_image_dir_name=getattr(
            INPUT_ARG_OBJECT, STORM_IMAGE_DIR_ARG_NAME),
        radar_source=getattr(INPUT_ARG_OBJECT, RADAR_SOURCE_ARG_NAME),
        spc_date_string=getattr(INPUT_ARG_OBJECT, SPC_DATE_ARG_NAME),
        radar_field_names=getattr(INPUT_ARG_OBJECT, RADAR_FIELD_NAMES_ARG_NAME),
        radar_heights_m_agl=numpy.array(
            getattr(INPUT_ARG_OBJECT, RADAR_HEIGHTS_ARG_NAME), dtype=int),
        refl_heights_m_agl=numpy.array(
            getattr(INPUT_ARG_OBJECT, REFL_HEIGHTS_ARG_NAME), dtype=int)
    )
