"""IO methods for NARR (North American Regional Reanalysis) data.

Since the NARR is a reanalysis, lead time is always zero (init time = valid
time).
"""

from gewittergefahr.gg_io import grib_io
from gewittergefahr.gg_io import nwp_model_io

# TODO(thunderhoser): replace main method with named method.

# TODO(thunderhoser): put these constants in narr_utils.
LEAD_TIME_HOURS = 0
NUM_GRID_ROWS = 277
NUM_GRID_COLUMNS = 349
SENTINEL_VALUE = 9.999e20

GRIB_TYPE = 'grib1'
MODEL_NAME = 'narr'
TOP_ONLINE_DIRECTORY_NAME = 'https://nomads.ncdc.noaa.gov/data/narr'

# The following constants are used only in the main method.
VALID_TIME_UNIX_SEC = 1404712800
GRIB1_FIELD_NAME = 'HGT:500 mb'
TOP_GRIB_DIRECTORY_NAME = (
    '/localdata/ryan.lagerquist/gewittergefahr_junk/narr/grib')
TOP_SINGLE_FIELD_DIR_NAME = (
    '/localdata/ryan.lagerquist/gewittergefahr_junk/narr/text')


def find_grib_file(valid_time_unix_sec, top_directory_name=None,
                   raise_error_if_missing=True):
    """Finds grib file with NARR data on local machine.

    :param valid_time_unix_sec: Valid time (Unix format).
    :param top_directory_name: Top-level directory with NARR grib files.
    :param raise_error_if_missing: Boolean flag.  If True and file is missing,
        will raise an error.
    :return: grib_file_name: Path to grib file.  If file is missing but
        raise_error_if_missing = False, this will be the *expected* path.
    """

    return nwp_model_io.find_grib_file(
        valid_time_unix_sec, LEAD_TIME_HOURS,
        top_directory_name=top_directory_name, model_name=MODEL_NAME,
        grib_type=GRIB_TYPE, raise_error_if_missing=raise_error_if_missing)


def find_single_field_file(valid_time_unix_sec, top_directory_name=None,
                           grib1_field_name=None, raise_error_if_missing=True):
    """Finds single-field file with NARR data on local machine.

    A "single field" is one variable at one time step and all grid cells.

    :param valid_time_unix_sec: Valid time (Unix format).
    :param top_directory_name: Top-level directory with single-field NARR files.
    :param grib1_field_name: Field name in grib1 format (example: 500-mb height
        is "HGT:500 mb").
    :param raise_error_if_missing: Boolean flag.  If True and file is missing,
        will raise an error.
    :return: single_field_file_name: Path to single-field file.  If file is
        missing but raise_error_if_missing = False, this will be the *expected*
        path.
    """

    return nwp_model_io.find_single_field_file(
        valid_time_unix_sec, LEAD_TIME_HOURS,
        top_directory_name=top_directory_name, model_name=MODEL_NAME,
        grib1_field_name=grib1_field_name,
        raise_error_if_missing=raise_error_if_missing)


def download_grib_file(valid_time_unix_sec, top_local_directory_name=None,
                       raise_error_if_fails=True):
    """Downloads NARR grib file to local machine.

    :param valid_time_unix_sec: Valid time (Unix format).
    :param top_local_directory_name: Top-level directory for NARR grib files on
        local machine.
    :param raise_error_if_fails: Boolean flag.  If True and download fails, will
        raise an error.
    :return: local_file_name: Path to grib file on local machine (after
        downloading).  If download failed but raise_error_if_fails = False, this
        will be None.
    """

    return nwp_model_io.download_grib_file(
        valid_time_unix_sec, LEAD_TIME_HOURS,
        top_online_directory_name=TOP_ONLINE_DIRECTORY_NAME,
        top_local_directory_name=top_local_directory_name,
        model_name=MODEL_NAME, grib_type=GRIB_TYPE,
        raise_error_if_fails=raise_error_if_fails)


def read_field_from_grib_file(grib_file_name, valid_time_unix_sec=None,
                              grib1_field_name=None,
                              top_single_field_dir_name=None,
                              wgrib_exe_name=grib_io.WGRIB_EXE_NAME_DEFAULT,
                              delete_single_field_file=True,
                              raise_error_if_fails=True):
    """Reads single field from NARR grib file.

    A "single field" is one variable at one time step and all grid cells.

    :param grib_file_name: Path to input file.
    :param valid_time_unix_sec: Valid time (Unix format).
    :param top_directory_name: Top-level directory with single-field NARR files.
    :param grib1_field_name: Field name in grib1 format (example: 500-mb height
        is "HGT:500 mb").
    :param wgrib_exe_name: Path to wgrib executable.
    :param delete_single_field_file: Boolean flag.  If True, single-field file
        will be deleted immediately upon reading.
    :param raise_error_if_fails: Boolean flag.  If True and field cannot be
        read, will raise error.  If False and field cannot be read, all return
        variables will be None.
    :return: field_matrix: See documentation for
        `nwp_model_io.read_field_from_grib_file`.
    :return: single_field_file_name: Path to output file (containing single
        field).
    """

    return nwp_model_io.read_field_from_grib_file(
        grib_file_name, init_time_unix_sec=valid_time_unix_sec,
        lead_time_hours=LEAD_TIME_HOURS,
        top_single_field_dir_name=top_single_field_dir_name,
        model_name=MODEL_NAME, grib1_field_name=grib1_field_name,
        wgrib_exe_name=wgrib_exe_name,
        delete_single_field_file=delete_single_field_file,
        raise_error_if_fails=raise_error_if_fails)


if __name__ == '__main__':
    LOCAL_GRIB_FILE_NAME = download_grib_file(
        VALID_TIME_UNIX_SEC,
        top_local_directory_name=TOP_GRIB_DIRECTORY_NAME)
    print LOCAL_GRIB_FILE_NAME

    (DATA_MATRIX, _) = read_field_from_grib_file(
        LOCAL_GRIB_FILE_NAME, valid_time_unix_sec=VALID_TIME_UNIX_SEC,
        grib1_field_name=GRIB1_FIELD_NAME,
        top_single_field_dir_name=TOP_SINGLE_FIELD_DIR_NAME)
    print DATA_MATRIX
