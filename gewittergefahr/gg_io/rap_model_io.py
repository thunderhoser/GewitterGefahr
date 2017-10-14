"""IO methods for RAP (Rapid Refresh model) data."""

from gewittergefahr.gg_io import grib_io
from gewittergefahr.gg_io import nwp_model_io
from gewittergefahr.gg_utils import rap_model_utils

# TODO(thunderhoser): replace main method with named method.

# TODO(thunderhoser): put these constants in rap_model_utils.
GRIB_TYPE = 'grib2'
MODEL_NAME = 'rap'
SENTINEL_VALUE = 9.999e20

# The following constants are used only in the main method.
INIT_TIME_UNIX_SEC = 1506567600  # 0300 UTC 28 Sep 2017
LEAD_TIME_HOURS = 10
GRIB2_VAR_NAME = 'HGT:500 mb'

TOP_GRIB_DIRECTORY_NAME_130GRID = (
    '/localdata/ryan.lagerquist/gewittergefahr_junk/rap130/grib2')
TOP_SINGLE_FIELD_DIR_NAME_130GRID = (
    '/localdata/ryan.lagerquist/gewittergefahr_junk/rap130/text')

TOP_GRIB_DIRECTORY_NAME_252GRID = (
    '/localdata/ryan.lagerquist/gewittergefahr_junk/rap252/grib2')
TOP_SINGLE_FIELD_DIR_NAME_252GRID = (
    '/localdata/ryan.lagerquist/gewittergefahr_junk/rap252/text')


def find_grib_file(init_time_unix_sec, lead_time_hours,
                   grid_id=rap_model_utils.ID_FOR_130GRID,
                   top_directory_name=None, raise_error_if_missing=True):
    """Finds grib file with RAP data on local machine.

    :param init_time_unix_sec: Model-initialization time (Unix format).
    :param lead_time_hours: Lead time (valid time minus init time).
    :param grid_id: String ID for grid (examples: "130" or "252").
    :param top_directory_name: Name of top-level directory for grib files with
        RAP data.
    :param raise_error_if_missing: Boolean flag.  If True and file is missing,
        will raise an error.
    :return: grib_file_name: Path to grib file.  If file is missing but
        raise_error_if_missing = False, this will be the *expected* path.
    """

    return nwp_model_io.find_grib_file(
        init_time_unix_sec, lead_time_hours,
        top_directory_name=top_directory_name, model_name=MODEL_NAME,
        grid_id=grid_id, grib_type=GRIB_TYPE,
        raise_error_if_missing=raise_error_if_missing)


def find_single_field_file(init_time_unix_sec, lead_time_hours,
                           grid_id=rap_model_utils.ID_FOR_130GRID,
                           grib1_field_name=None, top_directory_name=None,
                           raise_error_if_missing=True):
    """Finds single-field file with RAP data on local machine.

    A "single field" is one variable at one time step and all grid cells.

    :param init_time_unix_sec: Model-initialization time (Unix format).
    :param lead_time_hours: Lead time (valid time minus init time).
    :param grid_id: String ID for grid (examples: "130" or "252").
    :param grib1_field_name: Field name in grib1 format (example: 500-mb height
        is "HGT:500 mb").
    :param top_directory_name: Name of top-level directory for single-field
        files with RAP data.
    :param raise_error_if_missing: Boolean flag.  If True and file is missing,
        will raise an error.
    :return: single_field_file_name: Path to single-field file.  If file is
        missing but raise_error_if_missing = False, this will be the *expected*
        path.
    """

    return nwp_model_io.find_single_field_file(
        init_time_unix_sec, lead_time_hours,
        top_directory_name=top_directory_name, model_name=MODEL_NAME,
        grid_id=grid_id, grib1_field_name=grib1_field_name,
        raise_error_if_missing=raise_error_if_missing)


def download_grib_file(init_time_unix_sec, lead_time_hours,
                       grid_id=rap_model_utils.ID_FOR_130GRID,
                       top_local_directory_name=None,
                       raise_error_if_fails=True):
    """Downloads RAP grib file to local machine.

    :param init_time_unix_sec: Model-initialization time (Unix format).
    :param lead_time_hours: Lead time (valid time minus init time).
    :param grid_id: String ID for grid (examples: "130" or "252").
    :param top_local_directory_name: Name of top-level directory with RAP grib
        files on local machine.
    :param raise_error_if_fails: Boolean flag.  If True and download fails, will
        raise an error.
    :return: local_file_name: Path to grib file on local machine (after
        downloading).  If download failed but raise_error_if_fails = False, this
        will be None.
    """

    grid_metadata_dict = rap_model_utils.get_metadata_for_grid(grid_id)

    return nwp_model_io.download_grib_file(
        init_time_unix_sec, lead_time_hours,
        top_online_directory_name=
        grid_metadata_dict[rap_model_utils.TOP_ONLINE_DIRECTORY_COLUMN],
        top_local_directory_name=top_local_directory_name,
        model_name=MODEL_NAME, grid_id=grid_id, grib_type=GRIB_TYPE,
        raise_error_if_fails=raise_error_if_fails)


def read_field_from_grib_file(grib_file_name, init_time_unix_sec=None,
                              lead_time_hours=None,
                              grid_id=rap_model_utils.ID_FOR_130GRID,
                              grib1_field_name=None,
                              top_single_field_dir_name=None,
                              wgrib2_exe_name=grib_io.WGRIB2_EXE_NAME_DEFAULT,
                              delete_single_field_file=True,
                              raise_error_if_fails=True):
    """Reads single field from RAP grib file.

    A "single field" is one variable at one time step and all grid cells.

    :param grib_file_name: Path to input file.
    :param init_time_unix_sec: Model-initialization time (Unix format).
    :param lead_time_hours: Lead time (valid time minus init time).
    :param grid_id: String ID for grid (examples: "130" or "252").
    :param grib1_field_name: Field name in grib1 format (example: 500-mb height
        is "HGT:500 mb").
    :param top_single_field_dir_name: Name of top-level directory for single-
        field files with RAP data.
    :param wgrib2_exe_name: Path to wgrib2 executable.
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
        grib_file_name, init_time_unix_sec=init_time_unix_sec,
        lead_time_hours=lead_time_hours,
        top_single_field_dir_name=top_single_field_dir_name,
        model_name=MODEL_NAME, grid_id=grid_id,
        grib1_field_name=grib1_field_name, wgrib2_exe_name=wgrib2_exe_name,
        delete_single_field_file=delete_single_field_file,
        raise_error_if_fails=raise_error_if_fails)


if __name__ == '__main__':
    LOCAL_GRIB_FILE_NAME = download_grib_file(
        INIT_TIME_UNIX_SEC, LEAD_TIME_HOURS,
        grid_id=rap_model_utils.ID_FOR_130GRID,
        top_local_directory_name=TOP_GRIB_DIRECTORY_NAME_130GRID,
        raise_error_if_fails=True)
    print LOCAL_GRIB_FILE_NAME

    DATA_MATRIX, LOCAL_TEXT_FILE_NAME = read_field_from_grib_file(
        LOCAL_GRIB_FILE_NAME, init_time_unix_sec=INIT_TIME_UNIX_SEC,
        lead_time_hours=LEAD_TIME_HOURS, grid_id=rap_model_utils.ID_FOR_130GRID,
        grib1_field_name=GRIB2_VAR_NAME,
        top_single_field_dir_name=TOP_SINGLE_FIELD_DIR_NAME_130GRID)
    print DATA_MATRIX
    print '\n'

    LOCAL_GRIB_FILE_NAME = download_grib_file(
        INIT_TIME_UNIX_SEC, LEAD_TIME_HOURS,
        grid_id=rap_model_utils.ID_FOR_252GRID,
        top_local_directory_name=TOP_GRIB_DIRECTORY_NAME_252GRID,
        raise_error_if_fails=True)
    print LOCAL_GRIB_FILE_NAME

    DATA_MATRIX, LOCAL_TEXT_FILE_NAME = read_field_from_grib_file(
        LOCAL_GRIB_FILE_NAME, init_time_unix_sec=INIT_TIME_UNIX_SEC,
        lead_time_hours=LEAD_TIME_HOURS, grid_id=rap_model_utils.ID_FOR_252GRID,
        grib1_field_name=GRIB2_VAR_NAME,
        top_single_field_dir_name=TOP_SINGLE_FIELD_DIR_NAME_252GRID)
    print DATA_MATRIX
    print '\n'
