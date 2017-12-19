"""IO methods for radar data from MYRORSS.

--- DEFINITIONS ---

MYRORSS = Multi-year Reanalysis of Remotely Sensed Storms
"""

import shutil
import os.path
import numpy
from gewittergefahr.gg_io import radar_io
from gewittergefahr.gg_utils import unzipping
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking

IGNORABLE_RADAR_FIELD_NAMES = [
    radar_io.LOW_LEVEL_SHEAR_NAME, radar_io.MID_LEVEL_SHEAR_NAME]


def unzip_1day_tar_file(
        tar_file_name, field_names, spc_date_string, top_target_directory_name,
        refl_heights_m_agl=None):
    """Unzips 1-day tar file (containing raw MYRORSS data for one SPC date).

    :param tar_file_name: Path to input file.
    :param field_names: 1-D list with names of radar fields.
    :param spc_date_string: SPC date (format "yyyymmdd").
    :param top_target_directory_name: Name of top-level directory for unzipped
        MYRORSS files.  This method will create a subdirectory therein for the
        SPC date.
    :param refl_heights_m_agl: 1-D numpy array of reflectivity heights (metres
        above ground level).
    :return: target_directory_name: Path to output directory.
    """

    error_checking.assert_is_string_list(field_names)
    error_checking.assert_is_numpy_array(
        numpy.asarray(field_names), num_dimensions=1)

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

    field_to_heights_dict_m_agl = radar_io.field_and_height_arrays_to_dict(
        field_names, refl_heights_m_agl=refl_heights_m_agl,
        data_source=radar_io.MYRORSS_SOURCE_ID)

    error_checking.assert_is_string(top_target_directory_name)
    target_directory_name = '{0:s}/{1:s}'.format(
        top_target_directory_name, spc_date_string)

    field_names = field_to_heights_dict_m_agl.keys()
    directory_names_to_unzip = []

    for this_field_name in field_names:
        these_heights_m_agl = field_to_heights_dict_m_agl[this_field_name]

        for this_height_m_agl in these_heights_m_agl:
            directory_names_to_unzip.append(
                radar_io.get_relative_dir_for_raw_files(
                    field_name=this_field_name, height_m_agl=this_height_m_agl,
                    data_source=radar_io.MYRORSS_SOURCE_ID))

    unzipping.unzip_tar(
        tar_file_name,
        target_directory_name=target_directory_name,
        file_and_dir_names_to_unzip=directory_names_to_unzip)

    return target_directory_name


def remove_unzipped_data_1day(spc_date_string, top_directory_name):
    """Removes unzipped MYRORSS data for one SPC date.

    Basically, this method cleans up after unzip_1day_tar_file.

    :param spc_date_string: SPC date (format "yyyymmdd").
    :param top_directory_name: Name of top-level directory for unzipped MYRORSS
        files.  This method will find the subdirectory therein for the SPC date.
    """

    spc_date_unix_sec = time_conversion.spc_date_string_to_unix_sec(
        spc_date_string)

    field_names = radar_io.RADAR_FIELD_NAMES
    field_names.remove(radar_io.STORM_ID_NAME)
    field_to_heights_dict_m_agl = radar_io.field_and_height_arrays_to_dict(
        field_names,
        refl_heights_m_agl=numpy.array([radar_io.DEFAULT_HEIGHT_MYRORSS_M_AGL]),
        data_source=radar_io.MYRORSS_SOURCE_ID)

    for this_field_name in field_to_heights_dict_m_agl.keys():
        example_file_name = radar_io.find_raw_file(
            unix_time_sec=spc_date_unix_sec,
            spc_date_unix_sec=spc_date_unix_sec, field_name=this_field_name,
            height_m_agl=field_to_heights_dict_m_agl[this_field_name][0],
            data_source=radar_io.MYRORSS_SOURCE_ID,
            top_directory_name=top_directory_name, raise_error_if_missing=False)

        field_and_height_dir_name, _ = os.path.split(example_file_name)
        directory_name_parts = field_and_height_dir_name.split('/')
        field_directory_name = '/'.join(directory_name_parts[:-1])

        if os.path.isdir(field_directory_name):
            print 'Removing directory "{0:s}"...'.format(field_directory_name)
            shutil.rmtree(field_directory_name, ignore_errors=True)
