"""Extracts storm-centered radar images from MYRORSS data."""

import argparse
import numpy
from gewittergefahr.gg_io import myrorss_io
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import target_val_utils
from gewittergefahr.deep_learning import storm_images

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

ALL_AZ_SHEAR_FIELD_NAMES = [
    radar_utils.LOW_LEVEL_SHEAR_NAME, radar_utils.MID_LEVEL_SHEAR_NAME
]

NUM_ROWS_ARG_NAME = 'num_rows_per_image'
NUM_COLUMNS_ARG_NAME = 'num_columns_per_image'
ROTATE_GRIDS_ARG_NAME = 'rotate_grids'
GRID_SPACING_ARG_NAME = 'rotated_grid_spacing_metres'
RADAR_FIELD_NAMES_ARG_NAME = 'radar_field_names'
REFL_HEIGHTS_ARG_NAME = 'refl_heights_m_agl'
SPC_DATE_ARG_NAME = 'spc_date_string'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
TARRED_MYRORSS_DIR_ARG_NAME = 'input_tarred_myrorss_dir_name'
UNTARRED_MYRORSS_DIR_ARG_NAME = 'input_untarred_myrorss_dir_name'
TRACKING_DIR_ARG_NAME = 'input_tracking_dir_name'
ELEVATION_DIR_ARG_NAME = 'input_elevation_dir_name'
TRACKING_SCALE_ARG_NAME = 'tracking_scale_metres2'
TARGET_NAME_ARG_NAME = 'target_name'
TARGET_DIR_ARG_NAME = 'input_target_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

NUM_ROWS_HELP_STRING = (
    'Number of pixel rows in each storm-centered radar image.')

NUM_COLUMNS_HELP_STRING = (
    'Number of pixel columns in each storm-centered radar image.')

ROTATE_GRIDS_HELP_STRING = (
    'Boolean flag.  If 1, each grid will be rotated so that storm motion is in '
    'the +x-direction; thus, storm-centered grids will be equidistant.  If '
    'False, each storm-centered grid will be a contiguous rectangle extracted '
    'from the full grid; thus, storm-centered grids will be lat-long.')

GRID_SPACING_HELP_STRING = (
    '[used only if `{0:s}` = 1] Spacing between grid points in adjacent rows or'
    ' columns.'
).format(ROTATE_GRIDS_ARG_NAME)

RADAR_FIELD_NAMES_HELP_STRING = (
    'List with names of radar fields.  For more details, see documentation for'
    ' `{0:s}`.').format(SPC_DATE_ARG_NAME)

REFL_HEIGHTS_HELP_STRING = (
    'List of radar heights (metres above ground level).  These apply only to '
    'the field "{0:s}" (all other fields are 2-D).  For more details, see '
    'documentation for `{1:s}`.'
).format(radar_utils.REFL_NAME, SPC_DATE_ARG_NAME)

SPC_DATE_HELP_STRING = (
    'SPC date (format "yyyymmdd").  Will create image for each storm object on '
    'this date.  If you want to use `{0:s}` and `{1:s}` instead, leave this '
    'empty.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

TIME_HELP_STRING = (
    'Time (format "yyyy-mm-dd-HHMMSS").  Will create image for each storm '
    'object from `{0:s}`...`{1:s}`.  If you want to use `{2:s}` instead, leave '
    'these arguments empty.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME, SPC_DATE_ARG_NAME)

TARRED_MYRORSS_DIR_HELP_STRING = (
    'Name of top-level directory with tarred MYRORSS data (one tar file per SPC'
    ' date).  If data are already untarred, make this empty.')

UNTARRED_MYRORSS_DIR_HELP_STRING = (
    'Name of top-level directory for untarred MYRORSS data.  Tar files will be '
    'temporarily extracted here, but extracted contents will be deleted at the '
    'end of the script.')

TRACKING_DIR_HELP_STRING = (
    'Name of top-level directory with storm-tracking data.')

ELEVATION_DIR_HELP_STRING = (
    'Name of directory with elevation data (used by the Python package "srtm").'
)

TRACKING_SCALE_HELP_STRING = (
    'Tracking scale (minimum storm area).  This argument is used to find the '
    'specific tracking files in `{0:s}`.').format(TRACKING_DIR_ARG_NAME)

TARGET_NAME_HELP_STRING = (
    'Name of target variable (must be accepted by '
    '`target_val_utils.target_name_to_params`).  If specified, images will be '
    'created only for labeled storm objects (those with a target value).  If '
    '*not* specified (i.e., if you leave this argument), images will be created'
    ' for all storm objects.')

TARGET_DIR_HELP_STRING = (
    '[used only if `{0:s}` is not empty] Name of top-level directory.  Files '
    'therein will be located by `target_val_utils.find_target_file` and read by'
    ' `target_val_utils.read_target_values`.'
).format(TARGET_NAME_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level directory for storm-centered radar images.')

DEFAULT_TARRED_DIR_NAME = '/condo/swatcommon/common/myrorss'
DEFAULT_UNTARRED_DIR_NAME = '/condo/swatwork/ralager/myrorss_temp'
DEFAULT_ELEVATION_DIR_NAME = '/condo/swatwork/ralager/elevation'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_ROWS_ARG_NAME, type=int, required=False,
    default=storm_images.DEFAULT_NUM_IMAGE_ROWS, help=NUM_ROWS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_COLUMNS_ARG_NAME, type=int, required=False,
    default=storm_images.DEFAULT_NUM_IMAGE_COLUMNS,
    help=NUM_COLUMNS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + ROTATE_GRIDS_ARG_NAME, type=int, required=False,
    default=1, help=ROTATE_GRIDS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + GRID_SPACING_ARG_NAME, type=float, required=False,
    default=storm_images.DEFAULT_ROTATED_GRID_SPACING_METRES,
    help=GRID_SPACING_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_FIELD_NAMES_ARG_NAME, type=str, nargs='+', required=False,
    default=storm_images.DEFAULT_MYRORSS_MRMS_FIELD_NAMES,
    help=RADAR_FIELD_NAMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + REFL_HEIGHTS_ARG_NAME, type=int, nargs='+', required=False,
    default=storm_images.DEFAULT_RADAR_HEIGHTS_M_AGL,
    help=REFL_HEIGHTS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SPC_DATE_ARG_NAME, type=str, required=False, default='',
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=False, default='',
    help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=False, default='',
    help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TARRED_MYRORSS_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_TARRED_DIR_NAME, help=TARRED_MYRORSS_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + UNTARRED_MYRORSS_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_UNTARRED_DIR_NAME, help=UNTARRED_MYRORSS_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_DIR_ARG_NAME, type=str, required=True,
    help=TRACKING_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + ELEVATION_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_ELEVATION_DIR_NAME, help=ELEVATION_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRACKING_SCALE_ARG_NAME, type=int, required=False,
    default=echo_top_tracking.DUMMY_TRACKING_SCALE_METRES2,
    help=TRACKING_SCALE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_NAME_ARG_NAME, type=str, required=False, default='',
    help=TARGET_NAME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_DIR_ARG_NAME, type=str, required=False, default='',
    help=TARGET_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _extract_storm_images(
        num_image_rows, num_image_columns, rotate_grids,
        rotated_grid_spacing_metres, radar_field_names, refl_heights_m_agl,
        spc_date_string, first_time_string, last_time_string,
        tarred_myrorss_dir_name, untarred_myrorss_dir_name,
        top_tracking_dir_name, elevation_dir_name, tracking_scale_metres2,
        target_name, top_target_dir_name, top_output_dir_name):
    """Extracts storm-centered img for each field/height pair and storm object.

    :param num_image_rows: See documentation at top of file.
    :param num_image_columns: Same.
    :param rotate_grids: Same.
    :param rotated_grid_spacing_metres: Same.
    :param radar_field_names: Same.
    :param refl_heights_m_agl: Same.
    :param spc_date_string: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param tarred_myrorss_dir_name: Same.
    :param untarred_myrorss_dir_name: Same.
    :param top_tracking_dir_name: Same.
    :param elevation_dir_name: Same.
    :param tracking_scale_metres2: Same.
    :param target_name: Same.
    :param top_target_dir_name: Same.
    :param top_output_dir_name: Same.
    :raises: ValueError: if `first_time_string` and `last_time_string` have
        different SPC dates.
    """

    if spc_date_string in ['', 'None']:
        first_time_unix_sec = time_conversion.string_to_unix_sec(
            first_time_string, TIME_FORMAT)
        last_time_unix_sec = time_conversion.string_to_unix_sec(
            last_time_string, TIME_FORMAT)

        first_spc_date_string = time_conversion.time_to_spc_date_string(
            first_time_unix_sec)
        last_spc_date_string = time_conversion.time_to_spc_date_string(
            last_time_unix_sec)

        if first_spc_date_string != last_spc_date_string:
            error_string = (
                'First ({0:s}) and last ({1:s}) times have different SPC dates.'
                '  This script can handle only one SPC date.'
            ).format(first_time_string, last_time_string)

            raise ValueError(error_string)

        spc_date_string = first_spc_date_string
    else:
        first_time_unix_sec = 0
        last_time_unix_sec = int(1e12)

    if tarred_myrorss_dir_name in ['', 'None']:
        tarred_myrorss_dir_name = None
    if target_name in ['', 'None']:
        target_name = None

    if target_name is not None:
        target_param_dict = target_val_utils.target_name_to_params(target_name)

        target_file_name = target_val_utils.find_target_file(
            top_directory_name=top_target_dir_name,
            event_type_string=target_param_dict[
                target_val_utils.EVENT_TYPE_KEY],
            spc_date_string=spc_date_string)

        print('Reading data from: "{0:s}"...'.format(target_file_name))
        target_dict = target_val_utils.read_target_values(
            netcdf_file_name=target_file_name, target_names=[target_name]
        )
        print('\n')

    refl_heights_m_asl = radar_utils.get_valid_heights(
        data_source=radar_utils.MYRORSS_SOURCE_ID,
        field_name=radar_utils.REFL_NAME)

    # Untar files.
    if tarred_myrorss_dir_name is not None:
        az_shear_field_names = list(
            set(radar_field_names) & set(ALL_AZ_SHEAR_FIELD_NAMES)
        )

        if len(az_shear_field_names) > 0:
            az_shear_tar_file_name = (
                '{0:s}/{1:s}/azimuthal_shear_only/{2:s}.tar'
            ).format(
                tarred_myrorss_dir_name, spc_date_string[:4], spc_date_string
            )

            myrorss_io.unzip_1day_tar_file(
                tar_file_name=az_shear_tar_file_name,
                field_names=az_shear_field_names,
                spc_date_string=spc_date_string,
                top_target_directory_name=untarred_myrorss_dir_name)
            print(SEPARATOR_STRING)

        non_shear_field_names = list(
            set(radar_field_names) - set(ALL_AZ_SHEAR_FIELD_NAMES)
        )

        if len(non_shear_field_names) > 0:
            non_shear_tar_file_name = '{0:s}/{1:s}/{2:s}.tar'.format(
                tarred_myrorss_dir_name, spc_date_string[:4], spc_date_string
            )

            myrorss_io.unzip_1day_tar_file(
                tar_file_name=non_shear_tar_file_name,
                field_names=non_shear_field_names,
                spc_date_string=spc_date_string,
                top_target_directory_name=untarred_myrorss_dir_name,
                refl_heights_m_asl=refl_heights_m_asl)
            print(SEPARATOR_STRING)

    # Read storm tracks for the given SPC date.
    tracking_file_names = tracking_io.find_files_one_spc_date(
        spc_date_string=spc_date_string,
        source_name=tracking_utils.SEGMOTION_NAME,
        top_tracking_dir_name=top_tracking_dir_name,
        tracking_scale_metres2=tracking_scale_metres2
    )[0]

    file_times_unix_sec = numpy.array(
        [tracking_io.file_name_to_time(f) for f in tracking_file_names],
        dtype=int
    )

    good_indices = numpy.where(numpy.logical_and(
        file_times_unix_sec >= first_time_unix_sec,
        file_times_unix_sec <= last_time_unix_sec
    ))[0]

    tracking_file_names = [tracking_file_names[k] for k in good_indices]

    storm_object_table = tracking_io.read_many_files(
        tracking_file_names
    )[storm_images.STORM_COLUMNS_NEEDED]
    print(SEPARATOR_STRING)

    if target_name is not None:
        print((
            'Removing storm objects without target values (variable = '
            '"{0:s}")...'
        ).format(target_name))

        these_indices = tracking_utils.find_storm_objects(
            all_id_strings=storm_object_table[
                tracking_utils.FULL_ID_COLUMN].values.tolist(),
            all_times_unix_sec=storm_object_table[
                tracking_utils.VALID_TIME_COLUMN].values.astype(int),
            id_strings_to_keep=target_dict[target_val_utils.FULL_IDS_KEY],
            times_to_keep_unix_sec=target_dict[
                target_val_utils.VALID_TIMES_KEY],
            allow_missing=False)

        num_storm_objects_orig = len(storm_object_table.index)
        storm_object_table = storm_object_table.iloc[these_indices]
        num_storm_objects = len(storm_object_table.index)

        print('Removed {0:d} of {1:d} storm objects!\n'.format(
            num_storm_objects_orig - num_storm_objects, num_storm_objects_orig
        ))

    # Extract storm-centered radar images.
    storm_images.extract_storm_images_myrorss_or_mrms(
        storm_object_table=storm_object_table,
        radar_source=radar_utils.MYRORSS_SOURCE_ID,
        top_radar_dir_name=untarred_myrorss_dir_name,
        top_output_dir_name=top_output_dir_name,
        elevation_dir_name=elevation_dir_name,
        num_storm_image_rows=num_image_rows,
        num_storm_image_columns=num_image_columns, rotate_grids=rotate_grids,
        rotated_grid_spacing_metres=rotated_grid_spacing_metres,
        radar_field_names=radar_field_names,
        reflectivity_heights_m_agl=refl_heights_m_agl)
    print(SEPARATOR_STRING)

    # Remove untarred MYRORSS files.
    myrorss_io.remove_unzipped_data_1day(
        spc_date_string=spc_date_string,
        top_directory_name=untarred_myrorss_dir_name,
        field_names=radar_field_names, refl_heights_m_asl=refl_heights_m_asl)
    print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _extract_storm_images(
        num_image_rows=getattr(INPUT_ARG_OBJECT, NUM_ROWS_ARG_NAME),
        num_image_columns=getattr(INPUT_ARG_OBJECT, NUM_COLUMNS_ARG_NAME),
        rotate_grids=bool(getattr(INPUT_ARG_OBJECT, ROTATE_GRIDS_ARG_NAME)),
        rotated_grid_spacing_metres=getattr(
            INPUT_ARG_OBJECT, GRID_SPACING_ARG_NAME),
        radar_field_names=getattr(INPUT_ARG_OBJECT, RADAR_FIELD_NAMES_ARG_NAME),
        refl_heights_m_agl=numpy.array(
            getattr(INPUT_ARG_OBJECT, REFL_HEIGHTS_ARG_NAME), dtype=int),
        spc_date_string=getattr(INPUT_ARG_OBJECT, SPC_DATE_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        tarred_myrorss_dir_name=getattr(
            INPUT_ARG_OBJECT, TARRED_MYRORSS_DIR_ARG_NAME),
        untarred_myrorss_dir_name=getattr(
            INPUT_ARG_OBJECT, UNTARRED_MYRORSS_DIR_ARG_NAME),
        top_tracking_dir_name=getattr(INPUT_ARG_OBJECT, TRACKING_DIR_ARG_NAME),
        elevation_dir_name=getattr(INPUT_ARG_OBJECT, ELEVATION_DIR_ARG_NAME),
        tracking_scale_metres2=getattr(
            INPUT_ARG_OBJECT, TRACKING_SCALE_ARG_NAME),
        target_name=getattr(INPUT_ARG_OBJECT, TARGET_NAME_ARG_NAME),
        top_target_dir_name=getattr(INPUT_ARG_OBJECT, TARGET_DIR_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
