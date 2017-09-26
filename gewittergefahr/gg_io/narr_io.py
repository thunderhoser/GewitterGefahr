"""IO methods for NARR (North American Regional Analysis) data.

Since the NARR is a reanalysis, valid time = initialization time always.  In
other words, lead time is zero.
"""

from gewittergefahr.gg_io import grib_io
from gewittergefahr.gg_io import nwp_model_io

# TODO(thunderhoser): replace main method with named method.

LEAD_TIME_HOURS = 0
NUM_GRID_ROWS = 277
NUM_GRID_COLUMNS = 349
SENTINEL_VALUE = 9.999e20

RAW_FILE_EXTENSION = '.grb'
MODEL_ID_FOR_FILE_NAMES = 'narr-a_221'
TOP_ONLINE_DIRECTORY_NAME = 'https://nomads.ncdc.noaa.gov/data/narr'

# The following constants are used only in the main method.
VALID_TIME_UNIX_SEC = 1404712800
GRIB_VAR_NAME = 'HGT:500 mb'
TOP_LOCAL_DIR_NAME_FOR_GRIB = (
    '/localdata/ryan.lagerquist/gewittergefahr_junk/narr/grib')
TOP_LOCAL_DIR_NAME_FOR_TEXT = (
    '/localdata/ryan.lagerquist/gewittergefahr_junk/narr/text')


def find_local_grib_file(valid_time_unix_sec, top_directory_name=None,
                         raise_error_if_missing=True):
    """Finds grib file on local machine.

    :param valid_time_unix_sec: Valid time in Unix format.
    :param top_directory_name: Top-level directory containing grib files with
        NARR data.
    :param raise_error_if_missing: Boolean flag.  If True and file is missing,
        this method will raise an error.
    :return: grib_file_name: Path to grib file on local machine.  If file is
        missing but raise_error_if_missing = False, this will be the *expected*
        path to the grib file.
    """

    return nwp_model_io.find_local_raw_file(
        valid_time_unix_sec, LEAD_TIME_HOURS,
        top_directory_name=top_directory_name,
        model_id_for_pathless_file_name=MODEL_ID_FOR_FILE_NAMES,
        file_extension=RAW_FILE_EXTENSION,
        raise_error_if_missing=raise_error_if_missing)


def find_local_text_file(valid_time_unix_sec, top_directory_name=None,
                         variable_name=None, raise_error_if_missing=True):
    """Finds text file on local machine.

    :param valid_time_unix_sec: Valid time in Unix format.
    :param top_directory_name: Top-level directory containing text files with
        NARR data.
    :param variable_name: Variable name used in file name.
    :param raise_error_if_missing: Boolean flag.  If True and file is missing,
        this method will raise an error.
    :return: text_file_name: Path to text file on local machine.  If file is
        missing but raise_error_if_missing = False, this will be the *expected*
        path to the text file.
    """

    return nwp_model_io.find_local_text_file(
        valid_time_unix_sec, LEAD_TIME_HOURS,
        top_directory_name=top_directory_name,
        model_id_for_pathless_file_name=MODEL_ID_FOR_FILE_NAMES,
        variable_name=variable_name,
        raise_error_if_missing=raise_error_if_missing)


def download_grib_file(valid_time_unix_sec, top_local_directory_name=None,
                       raise_error_if_fails=True):
    """Downloads grib file to local machine.

    :param valid_time_unix_sec: Valid time in Unix format.
    :param top_local_directory_name: Top-level directory on local machine
        containing grib files with NARR data.
    :param raise_error_if_fails: Boolean flag.  If True and download fails, will
        raise error.
    :return: local_file_name: Path to grib file on local machine.  If download
        failed but raise_error_if_fails = False, will return None.
    """

    return nwp_model_io.download_grib_or_grib2_file(
        valid_time_unix_sec, LEAD_TIME_HOURS,
        top_online_directory_name=TOP_ONLINE_DIRECTORY_NAME,
        top_local_directory_name=top_local_directory_name,
        model_id_for_pathless_file_name=MODEL_ID_FOR_FILE_NAMES,
        file_extension=RAW_FILE_EXTENSION,
        raise_error_if_fails=raise_error_if_fails)


def read_variable_from_grib(grib_file_name, valid_time_unix_sec=None,
                            grib_var_name=None,
                            top_local_dir_name_for_text_file=None,
                            wgrib_exe_name=grib_io.WGRIB_EXE_NAME_DEFAULT,
                            delete_text_file=True):
    """Reads single field from grib file.

    :param grib_file_name: Path to input file.
    :param valid_time_unix_sec: Valid time in Unix format.
    :param grib_var_name: Name of variable being read.  This must be the name
        used in grib files (e.g., "HGT:500 mb" for 500-mb height).
    :param top_local_dir_name_for_text_file: Top-level directory containing text
        files with NARR data.
    :param wgrib_exe_name: Path to wgrib executable.
    :param delete_text_file: Boolean flag.  If True, text file with single field
        will be deleted immediately after reading.
    :return: data_matrix: 277-by-349 numpy array with values of `grib_var_name`.
        x increases while traveling right across the rows, and y increases while
        traveling down the columns.
    :return: text_file_name: Path to output file (text file with field extracted
        from grib file).  If delete_text_file = True, this is None.
    """

    return nwp_model_io.read_variable_from_grib(
        grib_file_name, init_time_unix_sec=valid_time_unix_sec,
        lead_time_hours=LEAD_TIME_HOURS, grib_var_name=grib_var_name,
        top_local_dir_name_for_text_file=top_local_dir_name_for_text_file,
        model_id_for_text_file_name=MODEL_ID_FOR_FILE_NAMES,
        wgrib_exe_name=wgrib_exe_name, num_grid_rows=NUM_GRID_ROWS,
        num_grid_columns=NUM_GRID_COLUMNS, sentinel_value=SENTINEL_VALUE,
        delete_text_file=delete_text_file)


if __name__ == '__main__':
    local_grib_file_name = download_grib_file(
        VALID_TIME_UNIX_SEC,
        top_local_directory_name=TOP_LOCAL_DIR_NAME_FOR_GRIB)
    print local_grib_file_name

    (data_matrix, local_text_file_name) = read_variable_from_grib(
        local_grib_file_name, valid_time_unix_sec=VALID_TIME_UNIX_SEC,
        grib_var_name=GRIB_VAR_NAME,
        top_local_dir_name_for_text_file=TOP_LOCAL_DIR_NAME_FOR_TEXT)
    print data_matrix
