"""Extracts storm-centered radar images from GridRad data."""

import argparse
import numpy
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import target_val_utils
from gewittergefahr.deep_learning import storm_images

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_ROWS_ARG_NAME = 'num_rows_per_image'
NUM_COLUMNS_ARG_NAME = 'num_columns_per_image'
ROTATE_GRIDS_ARG_NAME = 'rotate_grids'
GRID_SPACING_ARG_NAME = 'rotated_grid_spacing_metres'
RADAR_FIELD_NAMES_ARG_NAME = 'radar_field_names'
RADAR_HEIGHTS_ARG_NAME = 'radar_heights_m_agl'
SPC_DATE_ARG_NAME = 'spc_date_string'
RADAR_DIR_ARG_NAME = 'input_radar_dir_name'
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
    ' `{0:s}`.'
).format(SPC_DATE_ARG_NAME)

RADAR_HEIGHTS_HELP_STRING = (
    'List of radar heights (metres above ground level).  For more details, see '
    'documentation for `{0:s}`.'
).format(SPC_DATE_ARG_NAME)

SPC_DATE_HELP_STRING = (
    'SPC (Storm Prediction Center) date in format "yyyymmdd".  An image will be'
    ' created for each field, height, and storm object on this date.  Fields '
    'and heights are determined by defaults in'
    ' `storm_images.extract_storm_images_gridrad`.')

RADAR_DIR_HELP_STRING = (
    'Name of top-level directory with GridRad data (files readable by'
    ' `gridrad_io.read_field_from_full_grid_file`).')

TRACKING_DIR_HELP_STRING = (
    'Name of top-level directory with storm-tracking data.')

ELEVATION_DIR_HELP_STRING = (
    'Name of directory with elevation data (used by the Python package "srtm").'
)

TRACKING_SCALE_HELP_STRING = (
    'Tracking scale (minimum storm area).  This argument is used to find the '
    'specific tracking files in `{0:s}`.'
).format(TRACKING_DIR_ARG_NAME)

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

DEFAULT_RADAR_DIR_NAME = '/condo/swatcommon/common/gridrad_final/native_format'
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
    default=storm_images.DEFAULT_GRIDRAD_FIELD_NAMES,
    help=RADAR_FIELD_NAMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_HEIGHTS_ARG_NAME, type=int, nargs='+', required=False,
    default=storm_images.DEFAULT_RADAR_HEIGHTS_M_AGL,
    help=RADAR_HEIGHTS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SPC_DATE_ARG_NAME, type=str, required=True,
    help=SPC_DATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_RADAR_DIR_NAME, help=RADAR_DIR_HELP_STRING)

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
        rotated_grid_spacing_metres, radar_field_names, radar_heights_m_agl,
        spc_date_string, top_radar_dir_name, top_tracking_dir_name,
        elevation_dir_name, tracking_scale_metres2, target_name,
        top_target_dir_name, top_output_dir_name):
    """Extracts storm-centered radar images from GridRad data.

    :param num_image_rows: See documentation at top of file.
    :param num_image_columns: Same.
    :param rotate_grids: Same.
    :param rotated_grid_spacing_metres: Same.
    :param radar_field_names: Same.
    :param radar_heights_m_agl: Same.
    :param spc_date_string: Same.
    :param top_radar_dir_name: Same.
    :param top_tracking_dir_name: Same.
    :param elevation_dir_name: Same.
    :param tracking_scale_metres2: Same.
    :param target_name: Same.
    :param top_target_dir_name: Same.
    :param top_output_dir_name: Same.
    """

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
            netcdf_file_name=target_file_name, target_name=target_name)

        print('\n')

    # Find storm objects on the given SPC date.
    tracking_file_names = tracking_io.find_files_one_spc_date(
        spc_date_string=spc_date_string,
        source_name=tracking_utils.SEGMOTION_NAME,
        top_tracking_dir_name=top_tracking_dir_name,
        tracking_scale_metres2=tracking_scale_metres2
    )[0]

    # Read storm objects on the given SPC date.
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
    storm_images.extract_storm_images_gridrad(
        storm_object_table=storm_object_table,
        top_radar_dir_name=top_radar_dir_name,
        top_output_dir_name=top_output_dir_name,
        elevation_dir_name=elevation_dir_name,
        num_storm_image_rows=num_image_rows,
        num_storm_image_columns=num_image_columns, rotate_grids=rotate_grids,
        rotated_grid_spacing_metres=rotated_grid_spacing_metres,
        radar_field_names=radar_field_names,
        radar_heights_m_agl=radar_heights_m_agl)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _extract_storm_images(
        num_image_rows=getattr(INPUT_ARG_OBJECT, NUM_ROWS_ARG_NAME),
        num_image_columns=getattr(INPUT_ARG_OBJECT, NUM_COLUMNS_ARG_NAME),
        rotate_grids=bool(getattr(INPUT_ARG_OBJECT, ROTATE_GRIDS_ARG_NAME)),
        rotated_grid_spacing_metres=getattr(
            INPUT_ARG_OBJECT, GRID_SPACING_ARG_NAME),
        radar_field_names=getattr(INPUT_ARG_OBJECT, RADAR_FIELD_NAMES_ARG_NAME),
        radar_heights_m_agl=numpy.array(
            getattr(INPUT_ARG_OBJECT, RADAR_HEIGHTS_ARG_NAME), dtype=int),
        spc_date_string=getattr(INPUT_ARG_OBJECT, SPC_DATE_ARG_NAME),
        top_radar_dir_name=getattr(INPUT_ARG_OBJECT, RADAR_DIR_ARG_NAME),
        top_tracking_dir_name=getattr(INPUT_ARG_OBJECT, TRACKING_DIR_ARG_NAME),
        elevation_dir_name=getattr(INPUT_ARG_OBJECT, ELEVATION_DIR_ARG_NAME),
        tracking_scale_metres2=getattr(
            INPUT_ARG_OBJECT, TRACKING_SCALE_ARG_NAME),
        target_name=getattr(INPUT_ARG_OBJECT, TARGET_NAME_ARG_NAME),
        top_target_dir_name=getattr(INPUT_ARG_OBJECT, TARGET_DIR_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
