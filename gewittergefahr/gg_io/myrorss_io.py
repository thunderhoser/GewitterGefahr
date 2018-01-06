"""IO methods for radar data from MYRORSS.

--- DEFINITIONS ---

MYRORSS = Multi-year Reanalysis of Remotely Sensed Storms
"""

import shutil
import os.path
import numpy
from gewittergefahr.gg_io import radar_io
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import unzipping
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking

IGNORABLE_RADAR_FIELD_NAMES = [
    radar_utils.LOW_LEVEL_SHEAR_NAME, radar_utils.MID_LEVEL_SHEAR_NAME]

DEFAULT_FIELDS_TO_REMOVE = [
    radar_utils.ECHO_TOP_18DBZ_NAME, radar_utils.ECHO_TOP_50DBZ_NAME,
    radar_utils.LOW_LEVEL_SHEAR_NAME, radar_utils.MID_LEVEL_SHEAR_NAME,
    radar_utils.REFL_NAME, radar_utils.REFL_COLUMN_MAX_NAME,
    radar_utils.MESH_NAME, radar_utils.REFL_0CELSIUS_NAME,
    radar_utils.REFL_M10CELSIUS_NAME, radar_utils.REFL_M20CELSIUS_NAME,
    radar_utils.REFL_LOWEST_ALTITUDE_NAME, radar_utils.SHI_NAME,
    radar_utils.VIL_NAME]

DEFAULT_REFL_HEIGHTS_TO_REMOVE_M_ASL = radar_utils.get_valid_heights_for_field(
    radar_utils.REFL_NAME, data_source=radar_utils.MYRORSS_SOURCE_ID)


def unzip_1day_tar_file(
        tar_file_name, field_names, spc_date_string, top_target_directory_name,
        refl_heights_m_asl=None):
    """Unzips 1-day tar file (containing raw MYRORSS data for one SPC date).

    :param tar_file_name: Path to input file.
    :param field_names: 1-D list with names of radar fields.
    :param spc_date_string: SPC date (format "yyyymmdd").
    :param top_target_directory_name: Name of top-level directory for unzipped
        MYRORSS files.  This method will create a subdirectory therein for the
        SPC date.
    :param refl_heights_m_asl: 1-D numpy array of reflectivity heights (metres
        above sea level).
    :return: target_directory_name: Path to output directory.
    """

    # Verification.
    _ = time_conversion.spc_date_string_to_unix_sec(spc_date_string)
    error_checking.assert_is_string_list(field_names)
    error_checking.assert_is_numpy_array(
        numpy.asarray(field_names), num_dimensions=1)
    error_checking.assert_is_string(top_target_directory_name)

    # Put ignorable radar fields (ones that are allowed to be missing) at the
    # end.  This way, if the tar command errors out due to missing data, it will
    # do so after unzipping all the non-missing data.
    field_names_removed = []
    for this_field_name in IGNORABLE_RADAR_FIELD_NAMES:
        if this_field_name in field_names:
            field_names.remove(this_field_name)
            field_names_removed.append(this_field_name)

    for this_field_name in field_names_removed:
        field_names.append(this_field_name)

    field_to_heights_dict_m_asl = radar_utils.field_and_height_arrays_to_dict(
        field_names=field_names, data_source=radar_utils.MYRORSS_SOURCE_ID,
        refl_heights_m_asl=refl_heights_m_asl)

    target_directory_name = '{0:s}/{1:s}'.format(
        top_target_directory_name, spc_date_string)
    field_names = field_to_heights_dict_m_asl.keys()
    directory_names_to_unzip = []

    for this_field_name in field_names:
        these_heights_m_asl = field_to_heights_dict_m_asl[this_field_name]

        for this_height_m_asl in these_heights_m_asl:
            directory_names_to_unzip.append(
                radar_io.get_relative_dir_for_raw_files(
                    field_name=this_field_name,
                    data_source=radar_utils.MYRORSS_SOURCE_ID,
                    height_m_asl=this_height_m_asl))

    unzipping.unzip_tar(
        tar_file_name,
        target_directory_name=target_directory_name,
        file_and_dir_names_to_unzip=directory_names_to_unzip)

    return target_directory_name


def remove_unzipped_data_1day(
        spc_date_string, top_directory_name,
        field_names=DEFAULT_FIELDS_TO_REMOVE,
        refl_heights_m_asl=DEFAULT_REFL_HEIGHTS_TO_REMOVE_M_ASL):
    """Removes unzipped MYRORSS data for one SPC date.

    Basically, this method cleans up after unzip_1day_tar_file.

    :param spc_date_string: SPC date (format "yyyymmdd").
    :param top_directory_name: Name of top-level directory with unzipped MYRORSS
        files.  This method will find the subdirectory in `top_directory_name`
        for the given SPC date.
    :param field_names: 1-D list with names of radar fields.  Only these will be
        deleted.
    :param refl_heights_m_asl: 1-D numpy array of reflectivity heights (metres
        above sea level).
    """

    spc_date_unix_sec = time_conversion.spc_date_string_to_unix_sec(
        spc_date_string)

    field_to_heights_dict_m_asl = radar_utils.field_and_height_arrays_to_dict(
        field_names=field_names, data_source=radar_utils.MYRORSS_SOURCE_ID,
        refl_heights_m_asl=refl_heights_m_asl)

    for this_field_name in field_to_heights_dict_m_asl.keys():
        these_heights_m_asl = field_to_heights_dict_m_asl[this_field_name]

        for this_height_m_asl in these_heights_m_asl:
            example_file_name = radar_io.find_raw_file(
                unix_time_sec=spc_date_unix_sec,
                spc_date_string=spc_date_string, field_name=this_field_name,
                data_source=radar_utils.MYRORSS_SOURCE_ID,
                top_directory_name=top_directory_name,
                height_m_asl=this_height_m_asl,
                raise_error_if_missing=False)

            example_directory_name, _ = os.path.split(example_file_name)
            directory_name_parts = example_directory_name.split('/')
            remove_all_heights = False

            if this_field_name == radar_utils.REFL_NAME:
                if (set(these_heights_m_asl) ==
                        set(DEFAULT_REFL_HEIGHTS_TO_REMOVE_M_ASL)):
                    remove_all_heights = True
                    dir_name_to_remove = '/'.join(directory_name_parts[:-1])
                else:
                    dir_name_to_remove = '/'.join(directory_name_parts)

            else:
                dir_name_to_remove = '/'.join(directory_name_parts[:-1])

            if os.path.isdir(dir_name_to_remove):
                print 'Removing directory "{0:s}"...'.format(dir_name_to_remove)
                shutil.rmtree(dir_name_to_remove, ignore_errors=True)

            if remove_all_heights:
                break
