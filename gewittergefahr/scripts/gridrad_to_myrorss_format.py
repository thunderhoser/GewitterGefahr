"""Converts GridRad data into MYRORSS format.

Example usage (to compute 40-dBZ echo tops):

gridrad_to_myrorss_format.py \
--input_gridrad_dir_name="${TOP_GRIDRAD_DIR_NAME}" \
--output_myrorss_dir_name="${TOP_MYRORSS_DIR_NAME}" \
--output_field_name="echo_top_40dbz_km"

Example usage (to compute composite, or column-max, reflectivity):

gridrad_to_myrorss_format.py \
--input_gridrad_dir_name="${TOP_GRIDRAD_DIR_NAME}" \
--output_myrorss_dir_name="${TOP_MYRORSS_DIR_NAME}" \
--output_field_name="reflectivity_column_max_dbz"

Example usage (to compute minus-10-Celsius reflectivity):

gridrad_to_myrorss_format.py \
--input_gridrad_dir_name="${TOP_GRIDRAD_DIR_NAME}" \
--output_myrorss_dir_name="${TOP_MYRORSS_DIR_NAME}" \
--input_rap_dir_name="${TOP_RAP_DIR_NAME}" \
--input_ruc_dir_name="${TOP_RUC_DIR_NAME}" \
--output_field_name="reflectivity_m10celsius_dbz"
"""

import os
import copy
import argparse
from gewittergefahr.gg_io import gridrad_io
from gewittergefahr.gg_io import myrorss_and_mrms_io
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import gridrad_utils
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import time_conversion

GRIDRAD_FILE_EXTENSION = '.nc'

TIME_FORMAT_HOUR = '%Y-%m-%d-%H'
RAP_RUC_CUTOFF_TIME_STRING = '2012-05-01-00'
RAP_RUC_CUTOFF_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    RAP_RUC_CUTOFF_TIME_STRING, TIME_FORMAT_HOUR)

TEMPERATURE_LEVEL_KELVINS = 263.15
ECHO_TOP_REFLECTIVITY_DBZ = 40.

INPUT_FIELD_NAME = radar_utils.REFL_NAME
VALID_OUTPUT_FIELD_NAMES = [
    radar_utils.REFL_M10CELSIUS_NAME, radar_utils.REFL_COLUMN_MAX_NAME,
    radar_utils.ECHO_TOP_40DBZ_NAME]

GRIDRAD_DIR_ARG_NAME = 'input_gridrad_dir_name'
MYRORSS_DIR_ARG_NAME = 'output_myrorss_dir_name'
RUC_DIR_ARG_NAME = 'input_ruc_dir_name'
RAP_DIR_ARG_NAME = 'input_rap_dir_name'
OUTPUT_FIELD_ARG_NAME = 'output_field_name'

GRIDRAD_DIR_HELP_STRING = (
    'Name of top-level directory with GridRad data (files readable by'
    ' `gridrad_io.read_field_from_full_grid_file`).  All .nc files in this '
    'directory will be processed, including in subdirectories at any depth.')

MYRORSS_DIR_HELP_STRING = (
    'Name of top-level directory for MYRORSS data (after conversion; files will'
    ' be written by `myrorss_and_mrms_io.write_field_to_myrorss_file`).')

RUC_DIR_HELP_STRING = (
    'Name of top-level directory with RUC (Rapid Update Cycle) files (readable '
    'by `nwp_model_io.read_field_from_grib_file`).  Will be used to compute'
    ' "{0:s}" for times < {1:s}.'
).format(radar_utils.REFL_M10CELSIUS_NAME, RAP_RUC_CUTOFF_TIME_STRING)

RAP_DIR_HELP_STRING = (
    'Name of top-level directory with RAP (Rapid Refresh) files (readable by'
    ' `nwp_model_io.read_field_from_grib_file`).  Will be used to compute'
    ' "{0:s}" for times >= {1:s}.'
).format(radar_utils.REFL_M10CELSIUS_NAME, RAP_RUC_CUTOFF_TIME_STRING)

OUTPUT_FIELD_HELP_STRING = (
    'Name of output field (will be written to MYRORSS files).  Must be in the '
    'following list:\n{0:s}').format(str(VALID_OUTPUT_FIELD_NAMES))

TOP_GRIDRAD_DIR_NAME_DEFAULT = (
    '/condo/swatcommon/common/gridrad_final/native_format')
TOP_MYRORSS_DIR_NAME_DEFAULT = (
    '/condo/swatcommon/common/gridrad_final/myrorss_format')
TOP_RUC_DIR_NAME_DEFAULT = '/condo/swatwork/ralager/ruc_data'
TOP_RAP_DIR_NAME_DEFAULT = '/condo/swatwork/ralager/rap_data'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + GRIDRAD_DIR_ARG_NAME, type=str, required=False,
    default=TOP_GRIDRAD_DIR_NAME_DEFAULT, help=GRIDRAD_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MYRORSS_DIR_ARG_NAME, type=str, required=False,
    default=TOP_MYRORSS_DIR_NAME_DEFAULT, help=MYRORSS_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RUC_DIR_ARG_NAME, type=str, required=False,
    default=TOP_RUC_DIR_NAME_DEFAULT, help=RUC_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RAP_DIR_ARG_NAME, type=str, required=False,
    default=TOP_RAP_DIR_NAME_DEFAULT, help=RAP_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FIELD_ARG_NAME, type=str, required=False,
    default=radar_utils.ECHO_TOP_40DBZ_NAME, help=OUTPUT_FIELD_HELP_STRING)


def _check_output_field(output_field_name):
    """Ensures that output field is valid.

    :param output_field_name: See documentation at top of file.
    :raises: ValueError: if `output_field_name not in VALID_OUTPUT_FIELD_NAMES`.
    """

    if output_field_name not in VALID_OUTPUT_FIELD_NAMES:
        error_string = (
            '\n\n{0:s}\nValid output fields (listed above) do not include '
            '"{1:s}".').format(VALID_OUTPUT_FIELD_NAMES, output_field_name)
        raise ValueError(error_string)


def _find_gridrad_files(top_gridrad_dir_name):
    """Finds GridRad files at any depth in directory.

    :param top_gridrad_dir_name: Name of input directory.
    :return: gridrad_file_names: 1-D list of paths to GridRad files.
    """

    gridrad_file_names = []

    for root_dir_name, _, pathless_file_names in os.walk(
            top_gridrad_dir_name, topdown=True, followlinks=False):

        for this_pathless_file_name in pathless_file_names:
            _, this_extension = os.path.splitext(this_pathless_file_name)
            if this_extension != GRIDRAD_FILE_EXTENSION:
                continue

            gridrad_file_names.append(
                os.path.join(root_dir_name, this_pathless_file_name))

    return gridrad_file_names


def _convert_to_myrorss_format(
        top_gridrad_dir_name, top_myrorss_dir_name, top_ruc_dir_name,
        top_rap_dir_name, output_field_name):
    """Converts GridRad data to MYRORSS format.

    :param top_gridrad_dir_name: See documentation at top of file.
    :param top_myrorss_dir_name: Same.
    :param top_ruc_dir_name: Same.
    :param top_rap_dir_name: Same.
    :param output_field_name: Same.
    """

    gridrad_file_names = _find_gridrad_files(top_gridrad_dir_name)
    last_hour_string = 'NaN'
    target_height_matrix_m_asl = None

    for this_gridrad_file_name in gridrad_file_names:
        this_metadata_dict = gridrad_io.read_metadata_from_full_grid_file(
            this_gridrad_file_name)
        this_time_unix_sec = this_metadata_dict[radar_utils.UNIX_TIME_COLUMN]

        (this_refl_matrix_dbz, these_grid_point_heights_m_asl,
         these_grid_point_latitudes_deg, these_grid_point_longitudes_deg
        ) = gridrad_io.read_field_from_full_grid_file(
            netcdf_file_name=this_gridrad_file_name,
            field_name=INPUT_FIELD_NAME, metadata_dict=this_metadata_dict)

        if output_field_name == radar_utils.REFL_M10CELSIUS_NAME:
            this_hour_string = time_conversion.unix_sec_to_string(
                this_time_unix_sec, TIME_FORMAT_HOUR)

            if this_hour_string != last_hour_string:
                if this_time_unix_sec >= RAP_RUC_CUTOFF_TIME_UNIX_SEC:
                    this_model_name = nwp_model_utils.RAP_MODEL_NAME
                    this_top_model_dir_name = top_rap_dir_name
                else:
                    this_model_name = nwp_model_utils.RUC_MODEL_NAME
                    this_top_model_dir_name = top_ruc_dir_name

                target_height_matrix_m_asl = (
                    gridrad_utils.interp_temperature_surface_from_nwp(
                        radar_grid_point_latitudes_deg=
                        these_grid_point_latitudes_deg,
                        radar_grid_point_longitudes_deg=
                        these_grid_point_longitudes_deg,
                        radar_time_unix_sec=this_time_unix_sec,
                        critical_temperature_kelvins=TEMPERATURE_LEVEL_KELVINS,
                        model_name=this_model_name, use_all_grids=False,
                        grid_id=nwp_model_utils.ID_FOR_130GRID,
                        top_grib_directory_name=this_top_model_dir_name)
                )

                last_hour_string = copy.deepcopy(this_hour_string)

            this_output_matrix = gridrad_utils.interp_reflectivity_to_heights(
                reflectivity_matrix_dbz=this_refl_matrix_dbz,
                grid_point_heights_m_asl=these_grid_point_heights_m_asl,
                target_height_matrix_m_asl=target_height_matrix_m_asl)

        elif output_field_name == radar_utils.REFL_COLUMN_MAX_NAME:
            this_output_matrix = gridrad_utils.get_column_max_reflectivity(
                this_refl_matrix_dbz)
        else:
            this_output_matrix = gridrad_utils.get_echo_tops(
                reflectivity_matrix_dbz=this_refl_matrix_dbz,
                grid_point_heights_m_asl=these_grid_point_heights_m_asl,
                critical_reflectivity_dbz=ECHO_TOP_REFLECTIVITY_DBZ)

        this_spc_date_string = time_conversion.time_to_spc_date_string(
            this_time_unix_sec)

        this_myrorss_file_name = myrorss_and_mrms_io.find_raw_file(
            unix_time_sec=this_time_unix_sec,
            spc_date_string=this_spc_date_string, field_name=output_field_name,
            data_source=radar_utils.MYRORSS_SOURCE_ID,
            top_directory_name=top_myrorss_dir_name,
            raise_error_if_missing=False)
        this_myrorss_file_name = this_myrorss_file_name.replace('.gz', '')

        print 'Writing "{0:s}" to MYRORSS file: "{1:s}"...'.format(
            output_field_name, this_myrorss_file_name)

        myrorss_and_mrms_io.write_field_to_myrorss_file(
            field_matrix=this_output_matrix,
            netcdf_file_name=this_myrorss_file_name,
            field_name=output_field_name, metadata_dict=this_metadata_dict)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()
    TOP_GRIDRAD_DIR_NAME = getattr(INPUT_ARG_OBJECT, GRIDRAD_DIR_ARG_NAME)
    TOP_MYRORSS_DIR_NAME = getattr(INPUT_ARG_OBJECT, MYRORSS_DIR_ARG_NAME)
    TOP_RUC_DIR_NAME = getattr(INPUT_ARG_OBJECT, RUC_DIR_ARG_NAME)
    TOP_RAP_DIR_NAME = getattr(INPUT_ARG_OBJECT, RAP_DIR_ARG_NAME)
    OUTPUT_FIELD_NAME = getattr(INPUT_ARG_OBJECT, OUTPUT_FIELD_ARG_NAME)

    _check_output_field(OUTPUT_FIELD_NAME)
    _convert_to_myrorss_format(
        top_gridrad_dir_name=TOP_GRIDRAD_DIR_NAME,
        top_myrorss_dir_name=TOP_MYRORSS_DIR_NAME,
        top_ruc_dir_name=TOP_RUC_DIR_NAME, top_rap_dir_name=TOP_RAP_DIR_NAME,
        output_field_name=OUTPUT_FIELD_NAME)
