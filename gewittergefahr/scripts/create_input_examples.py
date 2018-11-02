"""Creates input examples and writes them to unshuffled files.

Basically, this script is a wrapper for `input_examples.create_examples`.
"""

import copy
import argparse
import numpy
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import storm_images
from gewittergefahr.deep_learning import input_examples

# TODO(thunderhoser): Allow downsampling based on multiple target variables.

STORM_IMAGE_DIR_ARG_NAME = 'input_storm_image_dir_name'
RADAR_SOURCE_ARG_NAME = 'radar_source'
NUM_RADAR_DIM_ARG_NAME = 'num_radar_dimensions'
RADAR_FIELDS_ARG_NAME = 'radar_field_names'
RADAR_HEIGHTS_ARG_NAME = 'radar_heights_m_agl'
FIRST_DATE_ARG_NAME = 'first_spc_date_string'
LAST_DATE_ARG_NAME = 'last_spc_date_string'
TARGET_DIR_ARG_NAME = 'input_target_dir_name'
TARGET_NAME_ARG_NAME = 'target_name'
SOUNDING_DIR_ARG_NAME = 'input_sounding_dir_name'
LAG_TIME_ARG_NAME = 'sounding_lag_time_sec'
NUM_EXAMPLES_PER_IN_FILE_ARG_NAME = 'num_examples_per_in_file'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'
CLASS_FRACTION_KEYS_ARG_NAME = 'class_fraction_keys'
CLASS_FRACTION_VALUES_ARG_NAME = 'class_fraction_values'

STORM_IMAGE_DIR_HELP_STRING = (
    'Name of top-level directory with storm-centered radar images.  Files '
    'therein will be located by `storm_images.find_storm_image_file` and read '
    'by `storm_images.read_storm_images`.'
)

RADAR_SOURCE_HELP_STRING = (
    'Radar source (must be accepted by `radar_utils.check_data_source`).'
)

NUM_RADAR_DIM_HELP_STRING = (
    'Number of radar dimensions.  If 2, this script will create examples with '
    'only 2-D radar images.  If 3, will create examples with only 3-D radar '
    'images.  If -1, will create examples with both 2-D azimuthal-shear and 3-D'
    ' reflectivity images.  However, the latter is possible only if `{0:s}` = '
    '"{1:s}".'
).format(RADAR_SOURCE_ARG_NAME, radar_utils.MYRORSS_SOURCE_ID)

RADAR_FIELDS_HELP_STRING = (
    'List of radar fields.  Each item must be accepted by '
    '`radar_utils.check_field_name`.'
)

RADAR_HEIGHTS_HELP_STRING = (
    'Radar heights (metres above ground level).  If `{0:s}` = 3, these heights '
    'will apply to every field in `{1:s}`.  Otherwise, these heights will apply'
    ' only to the reflectivity field.'
).format(NUM_RADAR_DIM_ARG_NAME, RADAR_FIELDS_ARG_NAME)

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  This script will deal with the time period '
    '`{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

TARGET_DIR_HELP_STRING = (
    'Name of top-level directory with target values (storm-hazard indicators).'
    '  Files therein will be located by `labels.find_label_file` and read by '
    '`labels.read_labels_from_netcdf`.'
)

TARGET_NAME_HELP_STRING = (
    'Name of target variable.  Must be accepted by '
    '`labels.column_name_to_label_params`.'
)

SOUNDING_DIR_HELP_STRING = (
    'Name of top-level directory with storm-centered soundings.  Files therein '
    'will be located by `soundings.find_sounding_file` and read by '
    '`soundings.read_soundings`.  If you do not want to use soundings, make '
    'this the empty string ("").'
)

LAG_TIME_HELP_STRING = (
    'Lag time used to create soundings (see '
    '`soundings.interp_soundings_to_storm_objects` for more on why this '
    'exists).'
)

NUM_EXAMPLES_PER_IN_FILE_HELP_STRING = (
    'Number of examples to read from each input file.'
)

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level directory.  Files will be written here by '
    '`input_examples.write_example_file`, to locations determined by '
    '`input_examples.find_example_file`.'
)

CLASS_FRACTION_KEYS_HELP_STRING = (
    'List of keys used to create input `class_to_sampling_fraction_dict` for '
    '`input_examples.create_examples`.  If you do not want class-conditional '
    'sampling, leave this alone.'
)

CLASS_FRACTION_VALUES_HELP_STRING = (
    'List of values used to create input `class_to_sampling_fraction_dict` for '
    '`input_examples.create_examples`.  If you do not want class-conditional '
    'sampling, leave this alone.'
)

DEFAULT_TOP_STORM_IMAGE_DIR_NAME = (
    '/condo/swatcommon/common/gridrad_final/myrorss_format/tracks/'
    'correct_echo_tops/reanalyzed/storm_images'
)
DEFAULT_TOP_TARGET_DIR_NAME = (
    '/condo/swatcommon/common/gridrad_final/myrorss_format/tracks/'
    'correct_echo_tops/reanalyzed/tornado_linkages/reanalyzed/labels'
)
DEFAULT_TOP_SOUNDING_DIR_NAME = (
    '/condo/swatcommon/common/gridrad_final/myrorss_format/tracks/'
    'correct_echo_tops/reanalyzed/soundings'
)

DEFAULT_RADAR_HEIGHTS_M_AGL = storm_images.DEFAULT_RADAR_HEIGHTS_M_AGL + 0
DEFAULT_TARGET_NAME = 'tornado_lead-time=0000-3600sec_distance=00000-10000m'
DEFAULT_NUM_EXAMPLES_PER_IN_FILE = 10000

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + STORM_IMAGE_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_TOP_STORM_IMAGE_DIR_NAME, help=STORM_IMAGE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_SOURCE_ARG_NAME, type=str, required=False,
    default=radar_utils.GRIDRAD_SOURCE_ID, help=RADAR_SOURCE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_RADAR_DIM_ARG_NAME, type=int, required=True,
    help=NUM_RADAR_DIM_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_FIELDS_ARG_NAME, type=str, nargs='+', required=True,
    help=RADAR_FIELDS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_HEIGHTS_ARG_NAME, type=int, nargs='+', required=False,
    default=DEFAULT_RADAR_HEIGHTS_M_AGL, help=RADAR_HEIGHTS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_TOP_TARGET_DIR_NAME, help=TARGET_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_NAME_ARG_NAME, type=str, required=False,
    default=DEFAULT_TARGET_NAME, help=TARGET_NAME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SOUNDING_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_TOP_SOUNDING_DIR_NAME, help=SOUNDING_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAG_TIME_ARG_NAME, type=int, required=False,
    default=soundings.DEFAULT_LAG_TIME_FOR_CONVECTIVE_CONTAMINATION_SEC,
    help=LAG_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_PER_IN_FILE_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_EXAMPLES_PER_IN_FILE,
    help=NUM_EXAMPLES_PER_IN_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CLASS_FRACTION_KEYS_ARG_NAME, type=int, nargs='+',
    required=False, default=[0], help=CLASS_FRACTION_KEYS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CLASS_FRACTION_VALUES_ARG_NAME, type=float, nargs='+',
    required=False, default=[0.], help=CLASS_FRACTION_VALUES_HELP_STRING)


def _run(top_storm_image_dir_name, radar_source, num_radar_dimensions,
         radar_field_names, radar_heights_m_agl, first_spc_date_string,
         last_spc_date_string, top_target_dir_name, target_name,
         top_sounding_dir_name, sounding_lag_time_sec, num_examples_per_in_file,
         top_output_dir_name, class_fraction_keys, class_fraction_values):
    """Runs `input_examples.shuffle_and_write_examples`.

    This is effectively the main method.

    :param top_storm_image_dir_name: See documentation at top of file.
    :param radar_source: Same.
    :param num_radar_dimensions: Same.
    :param radar_field_names: Same.
    :param radar_heights_m_agl: Same.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param top_target_dir_name: Same.
    :param target_name: Same.
    :param top_sounding_dir_name: Same.
    :param sounding_lag_time_sec: Same.
    :param num_examples_per_in_file: Same.
    :param top_output_dir_name: Same.
    :param class_fraction_keys: Same.
    :param class_fraction_values: Same.
    """

    if len(class_fraction_keys) > 1:
        class_to_sampling_fraction_dict = dict(zip(
            class_fraction_keys, class_fraction_values))
    else:
        class_to_sampling_fraction_dict = None

    include_soundings = top_sounding_dir_name != ''
    radar_file_name_matrix = None
    az_shear_file_name_matrix = None
    reflectivity_file_name_matrix = None

    if num_radar_dimensions < 0:
        az_shear_file_name_matrix, reflectivity_file_name_matrix = (
            input_examples.find_storm_images_2d3d_myrorss(
                top_directory_name=top_storm_image_dir_name,
                first_spc_date_string=first_spc_date_string,
                last_spc_date_string=last_spc_date_string,
                reflectivity_heights_m_agl=radar_heights_m_agl)
        )

        main_file_name_matrix = copy.deepcopy(reflectivity_file_name_matrix)
    else:
        error_checking.assert_is_geq(num_radar_dimensions, 2)
        error_checking.assert_is_leq(num_radar_dimensions, 3)

        if num_radar_dimensions == 2:
            radar_file_name_matrix = input_examples.find_storm_images_2d(
                top_directory_name=top_storm_image_dir_name,
                radar_source=radar_source, radar_field_names=radar_field_names,
                reflectivity_heights_m_agl=radar_heights_m_agl,
                first_spc_date_string=first_spc_date_string,
                last_spc_date_string=last_spc_date_string)
        else:
            radar_file_name_matrix = input_examples.find_storm_images_3d(
                top_directory_name=top_storm_image_dir_name,
                radar_source=radar_source, radar_field_names=radar_field_names,
                radar_heights_m_agl=radar_heights_m_agl,
                first_spc_date_string=first_spc_date_string,
                last_spc_date_string=last_spc_date_string)

        main_file_name_matrix = copy.deepcopy(radar_file_name_matrix)

    target_file_names = input_examples.find_target_files(
        top_target_dir_name=top_target_dir_name,
        radar_file_name_matrix=main_file_name_matrix, target_name=target_name)

    if include_soundings:
        sounding_file_names = input_examples.find_sounding_files(
            top_sounding_dir_name=top_sounding_dir_name,
            radar_file_name_matrix=main_file_name_matrix,
            target_name=target_name,
            lag_time_for_convective_contamination_sec=sounding_lag_time_sec)
    else:
        sounding_file_names = None

    input_examples.create_examples(
        target_file_names=target_file_names, target_name=target_name,
        num_examples_per_in_file=num_examples_per_in_file,
        top_output_dir_name=top_output_dir_name,
        radar_file_name_matrix=radar_file_name_matrix,
        reflectivity_file_name_matrix=reflectivity_file_name_matrix,
        az_shear_file_name_matrix=az_shear_file_name_matrix,
        class_to_sampling_fraction_dict=class_to_sampling_fraction_dict,
        sounding_file_names=sounding_file_names)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_storm_image_dir_name=getattr(
            INPUT_ARG_OBJECT, STORM_IMAGE_DIR_ARG_NAME),
        radar_source=getattr(INPUT_ARG_OBJECT, RADAR_SOURCE_ARG_NAME),
        num_radar_dimensions=getattr(INPUT_ARG_OBJECT, NUM_RADAR_DIM_ARG_NAME),
        radar_field_names=getattr(INPUT_ARG_OBJECT, RADAR_FIELDS_ARG_NAME),
        radar_heights_m_agl=numpy.array(
            getattr(INPUT_ARG_OBJECT, RADAR_HEIGHTS_ARG_NAME), dtype=int),
        first_spc_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_spc_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        top_target_dir_name=getattr(INPUT_ARG_OBJECT, TARGET_DIR_ARG_NAME),
        target_name=getattr(INPUT_ARG_OBJECT, TARGET_NAME_ARG_NAME),
        top_sounding_dir_name=getattr(INPUT_ARG_OBJECT, SOUNDING_DIR_ARG_NAME),
        sounding_lag_time_sec=getattr(INPUT_ARG_OBJECT, LAG_TIME_ARG_NAME),
        num_examples_per_in_file=getattr(
            INPUT_ARG_OBJECT, NUM_EXAMPLES_PER_IN_FILE_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME),
        class_fraction_keys=numpy.array(
            getattr(INPUT_ARG_OBJECT, CLASS_FRACTION_KEYS_ARG_NAME), dtype=int),
        class_fraction_values=numpy.array(
            getattr(INPUT_ARG_OBJECT, CLASS_FRACTION_VALUES_ARG_NAME),
            dtype=float)
    )
