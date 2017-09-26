"""IO methods for data from NWP (numerical weather prediction) models.

Strings created by time_unix_sec_to_month_string, time_unix_sec_to_date_string,
time_unix_sec_to_hour_string, and lead_time_hours_to_string are used in
file/directory names on the NCDC (National Climatic Data Center) website.  Thus,
these strings are needed to locate model output for a given initialization time
and lead time.
"""

import os.path
from gewittergefahr.gg_io import grib_io
from gewittergefahr.gg_io import downloads
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking

TIME_FORMAT_MONTH = '%Y%m'
TIME_FORMAT_DATE = '%Y%m%d'
TIME_FORMAT_HOUR = '%Y%m%d_%H00'

TEXT_FILE_EXTENSION = '.txt'


def _lead_time_number_to_string(lead_time_hours):
    """Converts lead time from number to string.

    :param lead_time_hours: Lead time (integer).
    :return: lead_time_hour_string: String describing lead time.  This will have
        3 digits, with leading zeros if necessary.
    """

    error_checking.assert_is_integer(lead_time_hours)
    error_checking.assert_is_non_negative(lead_time_hours)
    return '{0:03d}'.format(lead_time_hours)


def _get_pathless_raw_file_name(init_time_unix_sec, lead_time_hours,
                                model_id=None, file_extension=None):
    """Generates pathless name for raw file.

    :param init_time_unix_sec: Initialization time in Unix format.
    :param lead_time_hours: Lead time (valid time minus init time).
    :param model_id: Model ID for use in pathless file name.  Examples:
        "narr-a_221" for the NARR, "rap_130" for RAP on the 130 grid.
    :param file_extension: File extension.  Examples: "grb" for the NARR, "grb2"
        for the RAP.
    :return: pathless_file_name: Expected pathless file name.
    """

    error_checking.assert_is_string(model_id)
    error_checking.assert_is_string(file_extension)

    return '{0:s}_{1:s}_{2:s}{3:s}'.format(
        model_id, time_conversion.unix_sec_to_string(init_time_unix_sec,
                                                     TIME_FORMAT_HOUR),
        _lead_time_number_to_string(lead_time_hours), file_extension)


def _get_pathless_text_file_name(init_time_unix_sec, lead_time_hours,
                                 model_id=None, variable_id=None):
    """Generates pathless name for text file.

    :param init_time_unix_sec: Initialization time in Unix format.
    :param lead_time_hours: Lead time (valid time minus init time).
    :param model_id: Model ID for use in pathless file name.  Examples:
        "narr-a_221" for the NARR, "rap_130" for RAP on the 130 grid.
    :param variable_id: Variable ID for use in pathless file name.  Example:
        "HGT:500mb" for 500-mb height.
    :return: pathless_file_name: Expected pathless file name.
    """

    error_checking.assert_is_string(model_id)
    error_checking.assert_is_string(variable_id)

    return '{0:s}_{1:s}_{2:s}_{3:s}{4:s}'.format(
        model_id, time_conversion.unix_sec_to_string(init_time_unix_sec,
                                                     TIME_FORMAT_HOUR),
        _lead_time_number_to_string(lead_time_hours),
        variable_id.replace(' ', ''), TEXT_FILE_EXTENSION)


def find_local_raw_file(init_time_unix_sec, lead_time_hours,
                        top_directory_name=None, model_id=None,
                        file_extension=None, raise_error_if_missing=True):
    """Finds raw file on local machine.

    This file should all contain data for one model, one initialization time,
    and one lead time.

    :param init_time_unix_sec: Initialization time in Unix format.
    :param lead_time_hours: Lead time (valid time minus init time).
    :param top_directory_name: Top-level directory with raw files for the given
        model.
    :param model_id: See documentation for _get_pathless_raw_file_name.
    :param file_extension: See documentation for _get_pathless_raw_file_name.
    :param raise_error_if_missing: Boolean flag.  If True and file is missing,
        this method will raise an error.
    :return: raw_file_name: Path to raw file.  If file is missing but
        raise_error_if_missing = False, this will be the *expected* path.
    :raises: ValueError: if raise_error_if_missing = True and file is missing.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    pathless_file_name = _get_pathless_raw_file_name(
        init_time_unix_sec, lead_time_hours, model_id=model_id,
        file_extension=file_extension)

    raw_file_name = '{0:s}/{1:s}/{2:s}'.format(
        top_directory_name,
        time_conversion.unix_sec_to_string(init_time_unix_sec,
                                           TIME_FORMAT_MONTH),
        pathless_file_name)

    if raise_error_if_missing and not os.path.isfile(raw_file_name):
        raise ValueError(
            'Cannot find raw file.  Expected at location: ' + raw_file_name)

    return raw_file_name


def find_local_text_file(init_time_unix_sec, lead_time_hours,
                         top_directory_name=None, model_id=None,
                         variable_id=None, raise_error_if_missing=True):
    """Finds text file on local machine.

    This file should all contain data for one model, one initialization time,
    one lead time, and one variable.

    :param init_time_unix_sec: Initialization time in Unix format.
    :param lead_time_hours: Lead time (valid time minus init time).
    :param top_directory_name: Top-level directory with text files for the given
        model.
    :param model_id: See documentation for _get_pathless_text_file_name.
    :param file_extension: See documentation for _get_pathless_text_file_name.
    :param raise_error_if_missing:
    :param raise_error_if_missing: Boolean flag.  If True and file is missing,
        this method will raise an error.
    :return: text_file_name: Path to text file.  If file is missing but
        raise_error_if_missing = False, this will be the *expected* path.
    :raises: ValueError: if raise_error_if_missing = True and file is missing.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    pathless_file_name = _get_pathless_text_file_name(
        init_time_unix_sec, lead_time_hours, model_id=model_id,
        variable_id=variable_id)

    text_file_name = '{0:s}/{1:s}/{2:s}'.format(
        top_directory_name,
        time_conversion.unix_sec_to_string(init_time_unix_sec,
                                           TIME_FORMAT_MONTH),
        pathless_file_name)

    if raise_error_if_missing and not os.path.isfile(text_file_name):
        raise ValueError(
            'Cannot find text file.  Expected at location: ' + text_file_name)

    return text_file_name


def download_grib_or_grib2_file(init_time_unix_sec, lead_time_hours,
                                top_online_directory_name=None,
                                top_local_directory_name=None, model_id=None,
                                file_extension=None, raise_error_if_fails=True):
    """Downloads grib or grib2 file to local machine.

    :param init_time_unix_sec: Initialization time in Unix format.
    :param lead_time_hours: Lead time (valid time minus init time).
    :param top_online_directory_name: Top-level directory with raw files for
        given model on website.
    :param top_local_directory_name: Top-level directory with raw files for
        given model on local machine.
    :param model_id: See documentation for _get_pathless_raw_file_name.
    :param file_extension: See documentation for _get_pathless_raw_file_name.
    :param raise_error_if_fails: Boolean flag.  If True and download fails, will
        raise error.
    :return: local_file_name: Path to grib or grib2 file on local machine.  If
        download failed but raise_error_if_fails = False, will return None.
    :raises: ValueError: if raise_error_if_fails = True and download failed.
    """

    error_checking.assert_is_string(top_online_directory_name)
    error_checking.assert_is_boolean(raise_error_if_fails)

    pathless_file_name = _get_pathless_raw_file_name(
        init_time_unix_sec, lead_time_hours, model_id=model_id,
        file_extension=file_extension)

    online_file_name = '{0:s}/{1:s}/{2:s}/{3:s}'.format(
        top_online_directory_name,
        time_conversion.unix_sec_to_string(init_time_unix_sec,
                                           TIME_FORMAT_MONTH),
        time_conversion.unix_sec_to_string(init_time_unix_sec,
                                           TIME_FORMAT_DATE),
        pathless_file_name)

    local_file_name = find_local_raw_file(
        init_time_unix_sec, lead_time_hours,
        top_directory_name=top_local_directory_name, model_id=model_id,
        file_extension=file_extension, raise_error_if_missing=False)

    return downloads.download_file_from_url(
        online_file_name, local_file_name,
        raise_error_if_fails=raise_error_if_fails)


def read_variable_from_grib(grib_file_name, init_time_unix_sec=None,
                            lead_time_hours=None, grib_var_name=None,
                            top_local_text_dir_name=None,
                            model_id_for_text_file=None,
                            wgrib_exe_name=grib_io.WGRIB_EXE_NAME_DEFAULT,
                            num_grid_rows=None, num_grid_columns=None,
                            sentinel_value=None, delete_text_file=True):
    """Reads single field from grib file.

    M = number of grid rows
    N = number of grid columns

    :param grib_file_name: Path to input file.
    :param init_time_unix_sec: Initialization time in Unix format.
    :param lead_time_hours: Lead time (valid time minus init time).
    :param grib_var_name: Field name in grib format (example: "HGT:500 mb" for
        500-mb height).
    :param top_local_text_dir_name: Top-level directory with text files
        for the given model.
    :param model_id_for_text_file: See documentation for
        _get_pathless_text_file_name.
    :param wgrib_exe_name: Path to wgrib executable.
    :param num_grid_rows: Number of rows in grid.
    :param num_grid_columns: Number of columns in grid.
    :param sentinel_value: Sentinel value for given model.  All instances of
        `sentinel_value` will be replaced with NaN.
    :param delete_text_file: Boolean flag.  If True, text file will be deleted
        immediately after it is read.
    :return: text_file_name: Path to output file.  If delete_text_file = True,
        this is None.
    """

    error_checking.assert_is_boolean(delete_text_file)

    text_file_name = find_local_text_file(
        init_time_unix_sec, lead_time_hours,
        top_directory_name=top_local_text_dir_name,
        model_id=model_id_for_text_file, variable_id=grib_var_name,
        raise_error_if_missing=False)

    file_system_utils.mkdir_recursive_if_necessary(file_name=text_file_name)
    data_matrix = (
        grib_io.read_variable_from_grib(grib_file_name,
                                        grib_var_name=grib_var_name,
                                        text_file_name=text_file_name,
                                        wgrib_exe_name=wgrib_exe_name,
                                        num_grid_rows=num_grid_rows,
                                        num_grid_columns=num_grid_columns,
                                        sentinel_value=sentinel_value,
                                        delete_text_file=delete_text_file))

    if delete_text_file:
        text_file_name = None

    return data_matrix, text_file_name


def read_variable_from_grib2(grib2_file_name, init_time_unix_sec=None,
                             lead_time_hours=None, grib2_var_name=None,
                             top_local_text_dir_name=None,
                             model_id_for_text_file=None,
                             wgrib2_exe_name=grib_io.WGRIB2_EXE_NAME_DEFAULT,
                             num_grid_rows=None, num_grid_columns=None,
                             sentinel_value=None, delete_text_file=True):
    """Reads single field from grib2 file.

    M = number of grid rows
    N = number of grid columns

    :param grib2_file_name: Path to input file.
    :param init_time_unix_sec: Initialization time in Unix format.
    :param lead_time_hours: Lead time (valid time minus init time).
    :param grib2_var_name: Field name in grib2 format (example: "HGT:500 mb" for
        500-mb height).
    :param top_local_text_dir_name: Top-level directory with text files
        for the given model.
    :param model_id_for_text_file: See documentation for
        _get_pathless_text_file_name.
    :param wgrib2_exe_name: Path to wgrib2 executable.
    :param num_grid_rows: Number of rows in grid.
    :param num_grid_columns: Number of columns in grid.
    :param sentinel_value: Sentinel value for given model.  All instances of
        `sentinel_value` will be replaced with NaN.
    :param delete_text_file: Boolean flag.  If True, text file will be deleted
        immediately after it is read.
    :return: text_file_name: Path to output file.  If delete_text_file = True,
        this is None.
    """

    error_checking.assert_is_boolean(delete_text_file)

    text_file_name = find_local_text_file(
        init_time_unix_sec, lead_time_hours,
        top_directory_name=top_local_text_dir_name,
        model_id=model_id_for_text_file, variable_id=grib2_var_name,
        raise_error_if_missing=False)

    file_system_utils.mkdir_recursive_if_necessary(file_name=text_file_name)
    data_matrix = (
        grib_io.read_variable_from_grib2(grib2_file_name,
                                         grib2_var_name=grib2_var_name,
                                         text_file_name=text_file_name,
                                         wgrib2_exe_name=wgrib2_exe_name,
                                         num_grid_rows=num_grid_rows,
                                         num_grid_columns=num_grid_columns,
                                         sentinel_value=sentinel_value,
                                         delete_text_file=delete_text_file))

    if delete_text_file:
        text_file_name = None

    return data_matrix, text_file_name
