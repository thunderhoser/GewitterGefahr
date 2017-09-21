"""IO methods for data from RAP (Rapid Refresh) model."""

from gewittergefahr.gg_io import grib_io
from gewittergefahr.gg_io import nwp_model_io

# TODO(thunderhoser): add error-checking to all methods.

SENTINEL_VALUE = 9.999e20
RAW_FILE_EXTENSION = 'grb2'

ID_FOR_130GRID = '130'
NUM_ROWS_130GRID = 337
NUM_COLUMNS_130GRID = 451
MODEL_ID_FOR_FILE_NAMES_130GRID = 'rap_130'
TOP_ONLINE_DIR_NAME_130GRID = 'https://nomads.ncdc.noaa.gov/data/rap130'

ID_FOR_252GRID = '252'
NUM_ROWS_252GRID = 225
NUM_COLUMNS_252GRID = 301
MODEL_ID_FOR_FILE_NAMES_252GRID = 'rap_252'
TOP_ONLINE_DIR_NAME_252GRID = 'https://nomads.ncdc.noaa.gov/data/rap252'

NUM_ROWS_COLUMN = 'num_grid_rows'
NUM_COLUMNS_COLUMN = 'num_grid_columns'
MODEL_ID_COLUMN = 'model_id_for_file_names'
TOP_ONLINE_DIR_COLUMN = 'top_online_directory_name'

# TODO(thunderhoser): get rid of main method and constants defined below.

INIT_TIME_UNIX_SEC = 1475031600  # 0300 UTC 28 Sep 2016
LEAD_TIME_HOURS = 10
GRIB2_VAR_NAME = 'HGT:500 mb'

TOP_LOCAL_GRIB_DIR_NAME_130GRID = (
    '/localdata/ryan.lagerquist/gewittergefahr_junk/rap130/grib2')
TOP_LOCAL_TEXT_DIR_NAME_130GRID = (
    '/localdata/ryan.lagerquist/gewittergefahr_junk/rap130/text')

TOP_LOCAL_GRIB_DIR_NAME_252GRID = (
    '/localdata/ryan.lagerquist/gewittergefahr_junk/rap252/grib2')
TOP_LOCAL_TEXT_DIR_NAME_252GRID = (
    '/localdata/ryan.lagerquist/gewittergefahr_junk/rap252/text')


def _get_metadata_for_grid(grid_id=ID_FOR_130GRID):
    """Returns metadata for grid.

    :param grid_id: String ID for grid (either "130" or "252").
    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict.num_grid_rows: Number of grid rows.
    metadata_dict.num_grid_columns: Number of grid columns.
    metadata_dict.model_id_for_file_names: Model ID used in file names (either
        "rap_130" or "rap_252").
    metadata_dict.top_online_directory_name: Top-level web directory with grib2
        files on this grid.
    """

    if grid_id == ID_FOR_130GRID:
        return {NUM_ROWS_COLUMN: NUM_ROWS_130GRID,
                NUM_COLUMNS_COLUMN: NUM_COLUMNS_130GRID,
                MODEL_ID_COLUMN: MODEL_ID_FOR_FILE_NAMES_130GRID,
                TOP_ONLINE_DIR_COLUMN: TOP_ONLINE_DIR_NAME_130GRID}

    return {NUM_ROWS_COLUMN: NUM_ROWS_252GRID,
            NUM_COLUMNS_COLUMN: NUM_COLUMNS_252GRID,
            MODEL_ID_COLUMN: MODEL_ID_FOR_FILE_NAMES_252GRID,
            TOP_ONLINE_DIR_COLUMN: TOP_ONLINE_DIR_NAME_252GRID}


def find_local_grib2_file(init_time_unix_sec, lead_time_hours,
                          grid_id=ID_FOR_130GRID,
                          top_directory_name=None, raise_error_if_missing=True):
    """Finds grib file on local machine.

    :param init_time_unix_sec: Initialization time in Unix format.
    :param lead_time_hours: Lead time (valid time minus init time).
    :param grid_id: String ID for grid (either "130" or "252").
    :param top_directory_name: Top-level directory containing grib2 files with
        RAP data.
    :param raise_error_if_missing: Boolean flag.  If True and file is missing,
        this method will raise an error.
    :return: grib2_file_name: Path to grib2 file on local machine.  If file is
        missing but raise_error_if_missing = False, this will be the *expected*
        path to the grib2 file.
    """

    grid_metadata_dict = _get_metadata_for_grid(grid_id)

    return nwp_model_io.find_local_raw_file(
        init_time_unix_sec, lead_time_hours,
        top_directory_name=top_directory_name,
        model_id_for_pathless_file_name=grid_metadata_dict[MODEL_ID_COLUMN],
        file_extension=RAW_FILE_EXTENSION,
        raise_error_if_missing=raise_error_if_missing)


def find_local_text_file(init_time_unix_sec, lead_time_hours,
                         grid_id=ID_FOR_130GRID, top_directory_name=None,
                         variable_name=None, raise_error_if_missing=True):
    """Finds text file on local machine.

    :param init_time_unix_sec: Initialization time in Unix format.
    :param lead_time_hours: Lead time (valid time minus init time).
    :param grid_id: String ID for grid (either "130" or "252").
    :param top_directory_name: Top-level directory containing text files with
        RAP data.
    :param variable_name: Variable name used in file name.
    :param raise_error_if_missing: Boolean flag.  If True and file is missing,
        this method will raise an error.
    :return: text_file_name: Path to text file on local machine.  If file is
        missing but raise_error_if_missing = False, this will be the *expected*
        path to the text file.
    """

    grid_metadata_dict = _get_metadata_for_grid(grid_id)

    return nwp_model_io.find_local_text_file(
        init_time_unix_sec, lead_time_hours,
        top_directory_name=top_directory_name,
        model_id_for_pathless_file_name=grid_metadata_dict[MODEL_ID_COLUMN],
        variable_name=variable_name,
        raise_error_if_missing=raise_error_if_missing)


def download_grib2_file(init_time_unix_sec, lead_time_hours,
                        grid_id=ID_FOR_130GRID, top_local_directory_name=None,
                        raise_error_if_fails=True):
    """Downloads grib2 file to local machine.

    :param init_time_unix_sec: Initialization time in Unix format.
    :param lead_time_hours: Lead time (valid time minus init time).
    :param grid_id: String ID for grid (either "130" or "252").
    :param top_local_directory_name: Top-level directory on local machine
        containing grib2 files with RAP data.
    :param raise_error_if_fails: Boolean flag.  If True and download fails, will
        raise error.
    :return: local_file_name: Path to grib2 file on local machine.  If download
        failed but raise_error_if_fails = False, will return None.
    """

    grid_metadata_dict = _get_metadata_for_grid(grid_id)

    return nwp_model_io.download_grib_or_grib2_file(
        init_time_unix_sec, lead_time_hours,
        top_online_directory_name=grid_metadata_dict[TOP_ONLINE_DIR_COLUMN],
        top_local_directory_name=top_local_directory_name,
        model_id_for_pathless_file_name=grid_metadata_dict[MODEL_ID_COLUMN],
        file_extension=RAW_FILE_EXTENSION,
        raise_error_if_fails=raise_error_if_fails)


def read_variable_from_grib2(grib2_file_name, init_time_unix_sec=None,
                             lead_time_hours=None, grid_id=ID_FOR_130GRID,
                             grib2_var_name=None,
                             top_local_dir_name_for_text_file=None,
                             wgrib2_exe_name=grib_io.WGRIB2_EXE_NAME_DEFAULT,
                             delete_text_file=True):
    """Reads single field from grib2 file.

    M = number of grid rows
    N = number of grid columns

    :param grib2_file_name: Path to input file.
    :param init_time_unix_sec: Initialization time in Unix format.
    :param lead_time_hours: Lead time (valid time minus init time).
    :param grid_id: String ID for grid (either "130" or "252").
    :param grib2_var_name: Name of variable being read.  This must be the name
        used in grib2 files (e.g., "HGT:500 mb" for 500-mb height).
    :param top_local_dir_name_for_text_file: Top-level directory containing text
        files with RAP data.
    :param wgrib2_exe_name: Path to wgrib2 executable.
    :param delete_text_file: Boolean flag.  If True, text file with single field
        will be deleted immediately after reading.
    :return: data_matrix: M-by-N numpy array with values of `grib2_var_name`.  x
        increases while traveling right across the rows, and y increases while
        traveling down the columns.
    :return: text_file_name: Path to output file (text file with field extracted
        from grib2 file).  If delete_text_file = True, this is None.
    """

    grid_metadata_dict = _get_metadata_for_grid(grid_id)

    return nwp_model_io.read_variable_from_grib2(
        grib2_file_name, init_time_unix_sec=init_time_unix_sec,
        lead_time_hours=lead_time_hours, grib2_var_name=grib2_var_name,
        top_local_dir_name_for_text_file=top_local_dir_name_for_text_file,
        model_id_for_text_file_name=grid_metadata_dict[MODEL_ID_COLUMN],
        wgrib2_exe_name=wgrib2_exe_name,
        num_grid_rows=grid_metadata_dict[NUM_ROWS_COLUMN],
        num_grid_columns=grid_metadata_dict[NUM_COLUMNS_COLUMN],
        sentinel_value=SENTINEL_VALUE, delete_text_file=delete_text_file)


if __name__ == '__main__':
    local_grib2_file_name = download_grib2_file(
        INIT_TIME_UNIX_SEC, LEAD_TIME_HOURS, grid_id=ID_FOR_130GRID,
        top_local_directory_name=TOP_LOCAL_GRIB_DIR_NAME_130GRID,
        raise_error_if_fails=True)
    print local_grib2_file_name

    (data_matrix, local_text_file_name) = read_variable_from_grib2(
        local_grib2_file_name, init_time_unix_sec=INIT_TIME_UNIX_SEC,
        lead_time_hours=LEAD_TIME_HOURS, grid_id=ID_FOR_130GRID,
        grib2_var_name=GRIB2_VAR_NAME,
        top_local_dir_name_for_text_file=TOP_LOCAL_TEXT_DIR_NAME_130GRID)
    print data_matrix
    print '\n'

    local_grib2_file_name = download_grib2_file(
        INIT_TIME_UNIX_SEC, LEAD_TIME_HOURS, grid_id=ID_FOR_252GRID,
        top_local_directory_name=TOP_LOCAL_GRIB_DIR_NAME_252GRID,
        raise_error_if_fails=True)
    print local_grib2_file_name

    (data_matrix, local_text_file_name) = read_variable_from_grib2(
        local_grib2_file_name, init_time_unix_sec=INIT_TIME_UNIX_SEC,
        lead_time_hours=LEAD_TIME_HOURS, grid_id=ID_FOR_252GRID,
        grib2_var_name=GRIB2_VAR_NAME,
        top_local_dir_name_for_text_file=TOP_LOCAL_TEXT_DIR_NAME_252GRID)
    print data_matrix
