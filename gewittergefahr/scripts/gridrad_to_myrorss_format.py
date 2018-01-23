"""Converts GridRad data to MYRORSS format.

Example usage (to compute 40-dBZ echo tops):

gridrad_to_myrorss_format.py --input_gridrad_dir_name="foo"
--target_radar_field_name="echo_top_40dbz_km"

Example usage (to compute composite [column-max] reflectivity):

gridrad_to_myrorss_format.py --input_gridrad_dir_name="foo"
--target_radar_field_name="reflectivity_column_max_dbz"

Example usage (to compute minus-10-Celsius reflectivity):

gridrad_to_myrorss_format.py --input_gridrad_dir_name="foo"
--target_radar_field_name="reflectivity_m10celsius_dbz"
"""

import copy
import glob
import argparse
from gewittergefahr.gg_io import gridrad_io
from gewittergefahr.gg_io import myrorss_and_mrms_io
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import gridrad_utils
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import time_conversion

# TODO(thunderhoser): let `TOP_RUC_DIRECTORY_NAME`, `TOP_RAP_DIRECTORY_NAME`,
# and `TOP_MYRORSS_DIR_NAME` also be input args.

TIME_FORMAT_HOUR = '%Y-%m-%d-%H'
RAP_RUC_CUTOFF_STRING = '2012-05-01-00'
RAP_RUC_CUTOFF_UNIX_SEC = time_conversion.string_to_unix_sec(
    RAP_RUC_CUTOFF_STRING, TIME_FORMAT_HOUR)

TEMPERATURE_LEVEL_KELVINS = 263.15
CRITICAL_REFL_FOR_ECHO_TOPS_DBZ = 40.

SOURCE_FIELD_NAME = radar_utils.REFL_NAME
VALID_TARGET_FIELD_NAMES = [
    radar_utils.REFL_M10CELSIUS_NAME, radar_utils.REFL_COLUMN_MAX_NAME,
    radar_utils.ECHO_TOP_40DBZ_NAME]

TOP_RUC_DIRECTORY_NAME = '/condo/swatwork/ralager/ruc_data'
TOP_RAP_DIRECTORY_NAME = '/condo/swatwork/ralager/rap_data'
TOP_MYRORSS_DIR_NAME = '/condo/swatcommon/common/gridrad/myrorss_format'

GRIDRAD_DIR_INPUT_ARG = 'input_gridrad_dir_name'
TARGET_FIELD_INPUT_ARG = 'target_radar_field_name'

GRIDRAD_DIR_HELP_STRING = (
    'Name of input directory.  All GridRad files in this directory (extension '
    '.nc, no subdirectories) will be converted to MYRORSS format.')
TARGET_FIELD_HELP_STRING = (
    'Name of target radar field (to be written to MYRORSS files).  Valid '
    'options listed below:\n{0:s}').format(VALID_TARGET_FIELD_NAMES)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + GRIDRAD_DIR_INPUT_ARG, type=str, required=True,
    help=GRIDRAD_DIR_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_FIELD_INPUT_ARG, type=str, required=False,
    help=TARGET_FIELD_HELP_STRING, default=radar_utils.ECHO_TOP_40DBZ_NAME)


def _check_target_field(target_field_name):
    """Ensures that target radar field is valid.

    :param target_field_name: Name of target radar field.
    :raises: ValueError: if `target_field_name not in VALID_TARGET_FIELD_NAMES`.
    """

    if target_field_name not in VALID_TARGET_FIELD_NAMES:
        error_string = (
            '\n\n{0:s}\n\nValid target fields (listed above) do not include '
            '"{1:s}".').format(VALID_TARGET_FIELD_NAMES, target_field_name)
        raise ValueError(error_string)


def _convert_to_myrorss_format(input_gridrad_dir_name, target_field_name):
    """Converts GridRad data to MYRORSS format.

    :param input_gridrad_dir_name: Name of input directory.  All GridRad files
        in this directory (extension .nc, no subdirectories) will be converted
        to MYRORSS format.
    :param target_field_name: Name of target radar field (to be written to
        MYRORSS files).
    """

    input_file_names = glob.glob(input_gridrad_dir_name + '/*.nc')
    last_hour_string = 'NaN'
    target_height_matrix_m_asl = None

    for this_input_file_name in input_file_names:
        this_metadata_dict = gridrad_io.read_metadata_from_full_grid_file(
            this_input_file_name)
        this_time_unix_sec = this_metadata_dict[radar_utils.UNIX_TIME_COLUMN]

        (this_reflectivity_matrix_dbz,
         these_grid_point_heights_m_asl,
         these_grid_point_lats_deg,
         these_grid_point_lngs_deg) = gridrad_io.read_field_from_full_grid_file(
             this_input_file_name, field_name=SOURCE_FIELD_NAME,
             metadata_dict=this_metadata_dict)

        if target_field_name == radar_utils.REFL_M10CELSIUS_NAME:
            this_hour_string = time_conversion.unix_sec_to_string(
                this_time_unix_sec, TIME_FORMAT_HOUR)

            if this_hour_string != last_hour_string:
                if this_time_unix_sec >= RAP_RUC_CUTOFF_UNIX_SEC:
                    this_model_name = nwp_model_utils.RAP_MODEL_NAME
                    this_top_grib_directory_name = TOP_RAP_DIRECTORY_NAME
                else:
                    this_model_name = nwp_model_utils.RUC_MODEL_NAME
                    this_top_grib_directory_name = TOP_RUC_DIRECTORY_NAME

                target_height_matrix_m_asl = (
                    gridrad_utils.interp_temperature_sfc_from_nwp(
                        radar_grid_point_lats_deg=these_grid_point_lats_deg,
                        radar_grid_point_lngs_deg=these_grid_point_lngs_deg,
                        unix_time_sec=this_time_unix_sec,
                        temperature_kelvins=TEMPERATURE_LEVEL_KELVINS,
                        model_name=this_model_name,
                        grid_id=nwp_model_utils.ID_FOR_130GRID,
                        top_grib_directory_name=this_top_grib_directory_name))

                last_hour_string = copy.deepcopy(this_hour_string)

            this_field_matrix = gridrad_utils.interp_reflectivity_to_heights(
                reflectivity_matrix_dbz=this_reflectivity_matrix_dbz,
                unique_grid_point_heights_m_asl=these_grid_point_heights_m_asl,
                target_height_matrix_m_asl=target_height_matrix_m_asl)

        elif target_field_name == radar_utils.REFL_COLUMN_MAX_NAME:
            this_field_matrix = gridrad_utils.get_column_max_reflectivity(
                this_reflectivity_matrix_dbz)
        else:
            this_field_matrix = gridrad_utils.get_echo_tops(
                reflectivity_matrix_dbz=this_reflectivity_matrix_dbz,
                unique_grid_point_heights_m_asl=these_grid_point_heights_m_asl,
                critical_reflectivity_dbz=CRITICAL_REFL_FOR_ECHO_TOPS_DBZ)

        this_spc_date_string = time_conversion.time_to_spc_date_string(
            this_time_unix_sec)

        this_myrorss_file_name = myrorss_and_mrms_io.find_raw_file(
            unix_time_sec=this_time_unix_sec,
            spc_date_string=this_spc_date_string, field_name=target_field_name,
            data_source=radar_utils.MYRORSS_SOURCE_ID,
            top_directory_name=TOP_MYRORSS_DIR_NAME,
            raise_error_if_missing=False)
        this_myrorss_file_name = this_myrorss_file_name.replace('.gz', '')

        print 'Writing "{0:s}" field to MYRORSS file: "{1:s}"...'.format(
            target_field_name, this_myrorss_file_name)

        myrorss_and_mrms_io.write_field_to_myrorss_file(
            field_matrix=this_field_matrix,
            netcdf_file_name=this_myrorss_file_name,
            field_name=target_field_name, metadata_dict=this_metadata_dict)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()
    INPUT_GRIDRAD_DIR_NAME = getattr(INPUT_ARG_OBJECT, GRIDRAD_DIR_INPUT_ARG)
    TARGET_FIELD_NAME = getattr(INPUT_ARG_OBJECT, TARGET_FIELD_INPUT_ARG)

    _check_target_field(TARGET_FIELD_NAME)
    _convert_to_myrorss_format(
        input_gridrad_dir_name=INPUT_GRIDRAD_DIR_NAME,
        target_field_name=TARGET_FIELD_NAME)
