"""Separates storm-centered MYRORSS images associated with positive targets.

Specifically, images associated with positive target values are written to a new
directory.  For rare events like tornadoes and severe convective wind, positive
target values (anything other than -2 ["dead storm"] and 0 [lowest category])
are the minority, so this has the advantage of separating the most interesting
cases in smaller files, where it takes much less time to access them.  (For
example, reading 10 images from a file with 10^5 images takes much longer than
reading 10 images from a file with 100 images -- despite the fact that storm-
centered radar images are stored in NetCDF files, which allow random access.)
"""

import argparse
import numpy
from gewittergefahr.gg_utils import labels
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.deep_learning import storm_images

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
INPUT_TIME_FORMAT = '%Y-%m-%d-%H%M%S'

REFLECTIVITY_HEIGHTS_M_ASL = numpy.linspace(1000, 12000, num=12, dtype=int)
RADAR_FIELD_NAMES = [
    radar_utils.ECHO_TOP_50DBZ_NAME, radar_utils.ECHO_TOP_18DBZ_NAME,
    radar_utils.LOW_LEVEL_SHEAR_NAME, radar_utils.MESH_NAME,
    radar_utils.MID_LEVEL_SHEAR_NAME, radar_utils.REFL_0CELSIUS_NAME,
    radar_utils.REFL_COLUMN_MAX_NAME, radar_utils.REFL_NAME,
    radar_utils.REFL_LOWEST_ALTITUDE_NAME, radar_utils.REFL_M10CELSIUS_NAME,
    radar_utils.REFL_M20CELSIUS_NAME, radar_utils.SHI_NAME, radar_utils.VIL_NAME
]

INPUT_RADAR_DIR_ARG_NAME = 'input_storm_radar_image_dir_name'
OUTPUT_RADAR_DIR_ARG_NAME = 'output_storm_radar_image_dir_name'
TARGET_DIRECTORY_ARG_NAME = 'input_target_dir_name'
ONE_FILE_PER_TIME_STEP_ARG_NAME = 'one_file_per_time_step'
TARGET_NAMES_ARG_NAME = 'target_names'
FIRST_STORM_TIME_ARG_NAME = 'first_storm_time_string'
LAST_STORM_TIME_ARG_NAME = 'last_storm_time_string'

INPUT_RADAR_DIR_HELP_STRING = (
    'Name of top-level input directory.  Files therein will be found by '
    '`storm_images.find_many_files_myrorss_or_mrms`.')
OUTPUT_RADAR_DIR_HELP_STRING = (
    'Name of top-level output directory.  Images associated with positive '
    'target values will be written therein by '
    '`storm_images.write_storm_images`.')
TARGET_DIRECTORY_HELP_STRING = (
    'Name of top-level directory with target values.  Files therein will be '
    'found by `labels.find_label_file`.')
ONE_FILE_PER_TIME_STEP_HELP_STRING = (
    'Boolean flag.  If 1 (0), this script will read and write files with one '
    'time step (SPC date) each.')
TARGET_NAMES_HELP_STRING = (
    'Names of target variables (each must be accepted by '
    '`labels.column_name_to_label_params`).  Any storm object with a positive '
    'target value for *any* target variable will be written to the new '
    'directory.')
STORM_TIME_HELP_STRING = (
    'Storm time (format "yyyy-mm-dd-HHMMSS").  This script will read and write '
    'files for all storm times from `{0:s}`...`{1:s}`.  If `{2:s}` = False, '
    '`{0:s}` and `{1:s}` can be any time on the first and last SPC date, '
    'respectively.'
).format(FIRST_STORM_TIME_ARG_NAME, LAST_STORM_TIME_ARG_NAME,
         ONE_FILE_PER_TIME_STEP_ARG_NAME)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_RADAR_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_RADAR_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_RADAR_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_RADAR_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_DIRECTORY_ARG_NAME, type=str, required=True,
    help=TARGET_DIRECTORY_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + ONE_FILE_PER_TIME_STEP_ARG_NAME, type=int, required=False, default=0,
    help=ONE_FILE_PER_TIME_STEP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_NAMES_ARG_NAME, type=str, nargs='+', required=True,
    help=TARGET_NAMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_STORM_TIME_ARG_NAME, type=str, required=True,
    help=STORM_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_STORM_TIME_ARG_NAME, type=str, required=True,
    help=STORM_TIME_HELP_STRING)


def _separate_files(
        top_input_radar_image_dir_name, top_output_radar_image_dir_name,
        top_target_dir_name, one_file_per_time_step, target_names,
        first_storm_time_string, last_storm_time_string):
    """Separates MYRORSS images associated with positive target values.

    :param top_input_radar_image_dir_name: See documentation at top of file.
    :param top_output_radar_image_dir_name: Same.
    :param top_target_dir_name: Same.
    :param one_file_per_time_step: Same.
    :param target_names: Same.
    :param first_storm_time_string: Same.
    :param last_storm_time_string: Same.
    """

    first_storm_time_unix_sec = time_conversion.string_to_unix_sec(
        first_storm_time_string, INPUT_TIME_FORMAT)
    last_storm_time_unix_sec = time_conversion.string_to_unix_sec(
        last_storm_time_string, INPUT_TIME_FORMAT)

    (orig_storm_image_file_name_matrix, _
    ) = storm_images.find_many_files_myrorss_or_mrms(
        top_directory_name=top_input_radar_image_dir_name,
        radar_source=radar_utils.MYRORSS_SOURCE_ID,
        radar_field_names=RADAR_FIELD_NAMES,
        start_time_unix_sec=first_storm_time_unix_sec,
        end_time_unix_sec=last_storm_time_unix_sec,
        one_file_per_time_step=one_file_per_time_step,
        reflectivity_heights_m_asl=REFLECTIVITY_HEIGHTS_M_ASL)
    print SEPARATOR_STRING

    num_storm_times = orig_storm_image_file_name_matrix.shape[0]
    num_field_height_pairs = orig_storm_image_file_name_matrix.shape[1]
    num_target_variables = len(target_names)
    target_file_names = [''] * num_storm_times

    print 'Finding target files to match storm-image files...'
    for i in range(num_storm_times):
        target_file_names[i] = storm_images.find_storm_label_file(
            storm_image_file_name=orig_storm_image_file_name_matrix[i, 0],
            top_label_directory_name=top_target_dir_name,
            label_name=target_names[0],
            one_file_per_spc_date=not one_file_per_time_step,
            raise_error_if_missing=True)

    for i in range(num_storm_times):
        these_indices_to_keep = numpy.array([], dtype=int)

        for j in range(num_target_variables):
            print 'Reading target variable "{0:s}" from: "{1:s}"...'.format(
                target_names[j], target_file_names[i])
            this_storm_label_dict = labels.read_labels_from_netcdf(
                netcdf_file_name=target_file_names[i],
                label_name=target_names[j])

            these_new_indices = numpy.where(
                this_storm_label_dict[labels.LABEL_VALUES_KEY] > 0)[0]
            these_indices_to_keep = numpy.concatenate((
                these_indices_to_keep, these_new_indices))
            if j != num_target_variables - 1:
                continue

            these_indices_to_keep = numpy.unique(these_indices_to_keep)
            these_storm_ids_to_keep = [
                this_storm_label_dict[labels.STORM_IDS_KEY][k]
                for k in these_indices_to_keep]
            these_storm_times_to_keep_unix_sec = this_storm_label_dict[
                labels.VALID_TIMES_KEY][these_indices_to_keep]

        print (
            'There are {0:d} storm objects with positive values of any target '
            'variable.'
        ).format(len(these_storm_ids_to_keep))

        for j in range(num_field_height_pairs):
            print 'Reading data from: "{0:s}"...'.format(
                orig_storm_image_file_name_matrix[i, j])
            this_storm_image_dict = storm_images.read_storm_images(
                netcdf_file_name=orig_storm_image_file_name_matrix[i, j],
                return_images=True, storm_ids_to_keep=these_storm_ids_to_keep,
                valid_times_to_keep_unix_sec=these_storm_times_to_keep_unix_sec)

            _, this_spc_date_string = storm_images.image_file_name_to_time(
                orig_storm_image_file_name_matrix[i, j])
            this_new_storm_image_file_name = storm_images.find_storm_image_file(
                top_directory_name=top_output_radar_image_dir_name,
                spc_date_string=this_spc_date_string,
                radar_source=radar_utils.MYRORSS_SOURCE_ID,
                radar_field_name=this_storm_image_dict[
                    storm_images.RADAR_FIELD_NAME_KEY],
                radar_height_m_asl=this_storm_image_dict[
                    storm_images.RADAR_HEIGHT_KEY],
                unix_time_sec=None, raise_error_if_missing=False)

            print 'Writing {0:d} storm objects to: "{1:s}"...'.format(
                len(this_storm_image_dict[storm_images.STORM_IDS_KEY]),
                this_new_storm_image_file_name)

            storm_images.write_storm_images(
                netcdf_file_name=this_new_storm_image_file_name,
                storm_image_matrix=this_storm_image_dict[
                    storm_images.STORM_IMAGE_MATRIX_KEY],
                storm_ids=this_storm_image_dict[storm_images.STORM_IDS_KEY],
                valid_times_unix_sec=this_storm_image_dict[
                    storm_images.VALID_TIMES_KEY],
                radar_field_name=this_storm_image_dict[
                    storm_images.RADAR_FIELD_NAME_KEY],
                radar_height_m_asl=this_storm_image_dict[
                    storm_images.RADAR_HEIGHT_KEY])


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _separate_files(
        top_input_radar_image_dir_name=getattr(
            INPUT_ARG_OBJECT, INPUT_RADAR_DIR_ARG_NAME),
        top_output_radar_image_dir_name=getattr(
            INPUT_ARG_OBJECT, OUTPUT_RADAR_DIR_ARG_NAME),
        top_target_dir_name=getattr(
            INPUT_ARG_OBJECT, TARGET_DIRECTORY_ARG_NAME),
        one_file_per_time_step=bool(getattr(
            INPUT_ARG_OBJECT, ONE_FILE_PER_TIME_STEP_ARG_NAME)),
        target_names=getattr(INPUT_ARG_OBJECT, TARGET_NAMES_ARG_NAME),
        first_storm_time_string=getattr(
            INPUT_ARG_OBJECT, FIRST_STORM_TIME_ARG_NAME),
        last_storm_time_string=getattr(
            INPUT_ARG_OBJECT, LAST_STORM_TIME_ARG_NAME))
