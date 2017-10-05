"""IO methods for radar data from MYRORSS.

--- DEFINITIONS ---

MYRORSS = Multi-year Reanalysis of Remotely Sensed Storms

SPC = Storm Prediction Center

SPC date = a 24-hour period running from 1200-1200 UTC.  If time is discretized
in seconds, the period runs from 120000-115959 UTC.  This is unlike a human
date, which runs from 0000-0000 UTC (or 000000-235959 UTC).
"""

from gewittergefahr.gg_io import radar_io
from gewittergefahr.gg_utils import unzipping
from gewittergefahr.gg_utils import error_checking


def unzip_1day_tar_file(tar_file_name, spc_date_unix_sec=None,
                        top_target_directory_name=None,
                        field_to_heights_dict_m_agl=None):
    """Unzips 1-day tar file (containing raw MYRORSS for one SPC date).

    :param tar_file_name: Path to input file.
    :param spc_date_unix_sec: SPC date in Unix format.
    :param top_target_directory_name: Top-level directory for raw unzipped
        MYRORSS files.  This method will create a subdirectory therein for the
        SPC date.
    :param field_to_heights_dict_m_agl: Dictionary, where each key is the name
        of a radar field and each value is 1-D numpy array of heights (metres
        above ground level).
    :return: target_directory_name: Path to output directory.
    """

    error_checking.assert_is_string(top_target_directory_name)
    target_directory_name = '{0:s}/{1:s}'.format(
        top_target_directory_name,
        radar_io.time_unix_sec_to_spc_date(spc_date_unix_sec))

    field_names = field_to_heights_dict_m_agl.keys()
    directory_names_to_unzip = []
    for this_field_name in field_names:
        these_heights_m_agl = field_to_heights_dict_m_agl[this_field_name]

        for this_height_m_agl in these_heights_m_agl:
            directory_names_to_unzip.append(
                radar_io.get_relative_dir_for_raw_files(
                    field_name=this_field_name, height_m_agl=this_height_m_agl,
                    data_source=radar_io.MYRORSS_SOURCE_ID))

    unzipping.unzip_tar(tar_file_name,
                        target_directory_name=target_directory_name,
                        file_and_dir_names_to_unzip=directory_names_to_unzip)

    return target_directory_name
