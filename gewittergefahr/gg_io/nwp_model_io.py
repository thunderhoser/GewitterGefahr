"""IO methods for NWP (numerical weather prediction) data."""

import os.path
from gewittergefahr.gg_io import grib_io
from gewittergefahr.gg_io import downloads
from gewittergefahr.gg_utils import narr_utils
from gewittergefahr.gg_utils import rap_model_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking

TIME_FORMAT_MONTH = '%Y%m'
TIME_FORMAT_DATE = '%Y%m%d'
TIME_FORMAT_HOUR = '%Y%m%d_%H00'

SINGLE_FIELD_FILE_EXTENSION = '.txt'

RAP_MODEL_NAME = 'rap'
NARR_MODEL_NAME = 'narr'
MODEL_NAMES = [RAP_MODEL_NAME, NARR_MODEL_NAME]

RAP_GRID_IDS = [rap_model_utils.ID_FOR_130GRID, rap_model_utils.ID_FOR_252GRID]
NARR_ID_FOR_GRIB_FILE_NAMES = 'narr-a_221'


def _lead_time_to_string(lead_time_hours):
    """Converts lead time from number to string.

    :param lead_time_hours: Lead time (integer).
    :return: lead_time_hour_string: Lead time as string.  This will have 3
        digits, with leading zeros if necessary.
    """

    return '{0:03d}'.format(lead_time_hours)


def _check_model_name(model_name):
    """Ensures that model name is valid.

    :param model_name: Name of model (examples: "rap" and "narr").
    :raises: ValueError: if model name is not recognized.
    """

    if model_name not in MODEL_NAMES:
        error_string = (
            '\n\n' + str(MODEL_NAMES) +
            '\n\nValid models (listed above) do not include the following: "' +
            model_name + '"')
        raise ValueError(error_string)


def _get_model_id_for_grib_file_names(model_name, grid_id=None):
    """Generates model ID for grib file names.

    :param model_name: Name of model (examples: "rap" and "narr").
    :param grid_id: String ID for RAP grid (either "130" or "252").  If
        model_name != "rap", this can be left as None.
    :return: model_id: Model ID for grib file names (examples: "narr-a_221",
        "rap_130", "rap_252").
    """

    check_grid_id(model_name, grid_id=grid_id)
    if model_name == NARR_MODEL_NAME:
        return NARR_ID_FOR_GRIB_FILE_NAMES

    return '{0:s}_{1:s}'.format(model_name, grid_id)


def _get_pathless_grib_file_name(init_time_unix_sec, lead_time_hours,
                                 model_name=None, grid_id=None, grib_type=None):
    """Generates pathless name for grib file.

    :param init_time_unix_sec: Model-initialization time (Unix format).
    :param lead_time_hours: Lead time (valid time minus init time).
    :param model_name: Name of model (examples: "rap" and "narr").
    :param grid_id: String ID for RAP grid (either "130" or "252").  If
        model_name != "rap", this can be left as None.
    :param grib_type: File type (either "grib1" or "grib2").
    :return: pathless_file_name: Expected pathless file name.
    """

    return '{0:s}_{1:s}_{2:s}{3:s}'.format(
        _get_model_id_for_grib_file_names(model_name, grid_id=grid_id),
        time_conversion.unix_sec_to_string(
            init_time_unix_sec, TIME_FORMAT_HOUR),
        _lead_time_to_string(lead_time_hours),
        grib_io.file_type_to_extension(grib_type))


def _get_pathless_single_field_file_name(init_time_unix_sec, lead_time_hours,
                                         model_name=None, grid_id=None,
                                         grib1_field_name=None):
    """Generates pathless name for file with single field.

    A "single field" is one variable at one time step and all grid cells.

    :param init_time_unix_sec: Model-initialization time (Unix format).
    :param lead_time_hours: Lead time (valid time minus init time).
    :param model_name: Name of model (examples: "rap" and "narr").
    :param grid_id: String ID for RAP grid (either "130" or "252").  If
        model_name != "rap", this can be left as None.
    :param grib1_field_name: Field name in grib1 format (example: 500-mb height
        is "HGT:500 mb").
    :return: pathless_file_name: Expected pathless file name.
    """

    return '{0:s}_{1:s}_{2:s}_{3:s}{4:s}'.format(
        _get_model_id_for_grib_file_names(model_name, grid_id=grid_id),
        time_conversion.unix_sec_to_string(
            init_time_unix_sec, TIME_FORMAT_HOUR),
        _lead_time_to_string(lead_time_hours),
        grib1_field_name.replace(' ', ''), SINGLE_FIELD_FILE_EXTENSION)


def check_grid_id(model_name, grid_id=None):
    """Ensures that grid ID is valid for the given model.

    :param model_name: Name of model (examples: "rap" and "narr").
    :param grid_id: String ID for RAP grid (either "130" or "252").  If
        model_name != "rap", this can be left as None.
    :raises: ValueError: if grid ID is not recognized for the given model.
    """

    _check_model_name(model_name)
    if model_name == RAP_MODEL_NAME and grid_id not in RAP_GRID_IDS:
        error_string = (
            '\n\n' + str(RAP_GRID_IDS) + '\n\nValid grid IDs for ' +
            model_name.upper() +
            ' model (listed above) do not include the following: "' + grid_id +
            '"')
        raise ValueError(error_string)


def find_grib_file(init_time_unix_sec, lead_time_hours, top_directory_name=None,
                   model_name=None, grid_id=None, grib_type=None,
                   raise_error_if_missing=True):
    """Finds grib file on local machine.

    :param init_time_unix_sec: Model-initialization time (Unix format).
    :param lead_time_hours: Lead time (valid time minus init time).
    :param top_directory_name: Name of top-level directory with grib files.
    :param model_name: Name of model (examples: "rap" and "narr").
    :param grid_id: String ID for RAP grid (either "130" or "252").  If
        model_name != "rap", this can be left as None.
    :param grib_type: File type (either "grib1" or "grib2").
    :param raise_error_if_missing: Boolean flag.  If True and file is missing,
        will raise an error.
    :return: grib_file_name: Path to grib file.  If file is missing but
        raise_error_if_missing = False, this will be the *expected* path.
    :raises: ValueError: if raise_error_if_missing = True and file is missing.
    """

    error_checking.assert_is_integer(init_time_unix_sec)
    error_checking.assert_is_integer(lead_time_hours)
    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    pathless_file_name = _get_pathless_grib_file_name(
        init_time_unix_sec, lead_time_hours, model_name=model_name,
        grid_id=grid_id, grib_type=grib_type)

    grib_file_name = '{0:s}/{1:s}/{2:s}'.format(
        top_directory_name,
        time_conversion.unix_sec_to_string(init_time_unix_sec,
                                           TIME_FORMAT_MONTH),
        pathless_file_name)

    if raise_error_if_missing and not os.path.isfile(grib_file_name):
        raise ValueError(
            'Cannot find grib file.  Expected at: ' + grib_file_name)

    return grib_file_name


def find_single_field_file(init_time_unix_sec, lead_time_hours,
                           top_directory_name=None, model_name=None,
                           grid_id=None, grib1_field_name=None,
                           raise_error_if_missing=True):
    """Finds file with single field on local machine.

    A "single field" is one variable at one time step and all grid cells.

    :param init_time_unix_sec: Model-initialization time (Unix format).
    :param lead_time_hours: Lead time (valid time minus init time).
    :param top_directory_name: Name of top-level directory with single-field
        files.
    :param model_name: Name of model (examples: "rap" and "narr").
    :param grid_id: String ID for RAP grid (either "130" or "252").  If
        model_name != "rap", this can be left as None.
    :param grib1_field_name: Field name in grib1 format (example: 500-mb height
        is "HGT:500 mb").
    :param raise_error_if_missing: Boolean flag.  If True and file is missing,
        will raise an error.
    :return: single_field_file_name: Path to single-field file.  If file is
        missing but raise_error_if_missing = False, this will be the *expected*
        path.
    :raises: ValueError: if raise_error_if_missing = True and file is missing.
    """

    error_checking.assert_is_integer(init_time_unix_sec)
    error_checking.assert_is_integer(lead_time_hours)
    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    pathless_file_name = _get_pathless_single_field_file_name(
        init_time_unix_sec, lead_time_hours, model_name=model_name,
        grid_id=grid_id, grib1_field_name=grib1_field_name)

    single_field_file_name = '{0:s}/{1:s}/{2:s}'.format(
        top_directory_name,
        time_conversion.unix_sec_to_string(init_time_unix_sec,
                                           TIME_FORMAT_MONTH),
        pathless_file_name)

    if raise_error_if_missing and not os.path.isfile(single_field_file_name):
        raise ValueError(
            'Cannot find single-field file.  Expected at: ' +
            single_field_file_name)

    return single_field_file_name


def download_grib_file(init_time_unix_sec, lead_time_hours,
                       top_online_directory_name=None,
                       top_local_directory_name=None, model_name=None,
                       grid_id=None, grib_type=None, raise_error_if_fails=True):
    """Downloads grib file to local machine.

    :param init_time_unix_sec: Model-initialization time (Unix format).
    :param lead_time_hours: Lead time (valid time minus init time).
    :param top_online_directory_name: Name of top-level directory with grib
        files on website.
    :param top_local_directory_name: Name of top-level directory with grib files
        on local machine.
    :param model_name: Name of model (examples: "rap" and "narr").
    :param grid_id: String ID for RAP grid (either "130" or "252").  If
        model_name != "rap", this can be left as None.
    :param grib_type: File type (either "grib1" or "grib2").
    :param raise_error_if_fails: Boolean flag.  If True and download fails, will
        raise an error.
    :return: local_file_name: Path to grib file on local machine (after
        downloading).  If download failed but raise_error_if_fails = False, this
        will be None.
    """

    error_checking.assert_is_integer(init_time_unix_sec)
    error_checking.assert_is_integer(lead_time_hours)
    error_checking.assert_is_string(top_online_directory_name)
    error_checking.assert_is_boolean(raise_error_if_fails)

    pathless_file_name = _get_pathless_grib_file_name(
        init_time_unix_sec, lead_time_hours, model_name=model_name,
        grid_id=grid_id, grib_type=grib_type)

    online_file_name = '{0:s}/{1:s}/{2:s}/{3:s}'.format(
        top_online_directory_name,
        time_conversion.unix_sec_to_string(init_time_unix_sec,
                                           TIME_FORMAT_MONTH),
        time_conversion.unix_sec_to_string(init_time_unix_sec,
                                           TIME_FORMAT_DATE),
        pathless_file_name)

    local_file_name = find_grib_file(
        init_time_unix_sec, lead_time_hours,
        top_directory_name=top_local_directory_name, model_name=model_name,
        grid_id=grid_id, grib_type=grib_type, raise_error_if_missing=False)

    return downloads.download_file_via_http(
        online_file_name, local_file_name,
        raise_error_if_fails=raise_error_if_fails)


def read_field_from_grib_file(grib_file_name, init_time_unix_sec=None,
                              lead_time_hours=None,
                              top_single_field_dir_name=None, model_name=None,
                              grid_id=None, grib1_field_name=None,
                              wgrib_exe_name=grib_io.WGRIB_EXE_NAME_DEFAULT,
                              wgrib2_exe_name=grib_io.WGRIB2_EXE_NAME_DEFAULT,
                              delete_single_field_file=True,
                              raise_error_if_fails=True):
    """Reads single field from grib file.

    A "single field" is one variable at one time step and all grid cells.

    :param grib_file_name: Path to input file.
    :param init_time_unix_sec: Model-initialization time (Unix format).
    :param lead_time_hours: Lead time (valid time minus init time).
    :param top_single_field_dir_name: Name of top-level directory for single-
        field files.
    :param model_name: Name of model (examples: "rap" and "narr").
    :param grid_id: String ID for RAP grid (either "130" or "252").  If
        model_name != "rap", this can be left as None.
    :param grib_type: File type (either "grib1" or "grib2").
    :param grib1_field_name: Field name in grib1 format (example: 500-mb height
        is "HGT:500 mb").
    :param wgrib_exe_name: Path to wgrib executable.
    :param wgrib2_exe_name: Path to wgrib2 executable.
    :param delete_single_field_file: Boolean flag.  If True, single-field file
        will be deleted immediately upon reading.
    :param raise_error_if_fails: Boolean flag.  If True and field cannot be
        read, will raise error.  If False and field cannot be read, all return
        variables will be None.
    :return: field_matrix: See documentation for
        `grib_io.read_field_from_grib_file`.
    :return: single_field_file_name: Path to output file (containing single
        field).
    """

    single_field_file_name = find_single_field_file(
        init_time_unix_sec, lead_time_hours,
        top_directory_name=top_single_field_dir_name, model_name=model_name,
        grid_id=grid_id, grib1_field_name=grib1_field_name,
        raise_error_if_missing=False)

    if model_name == NARR_MODEL_NAME:
        num_grid_rows = narr_utils.NUM_GRID_ROWS
        num_grid_columns = narr_utils.NUM_GRID_COLUMNS
        sentinel_value = narr_utils.SENTINEL_VALUE
    else:
        grid_metadata_dict = rap_model_utils.get_metadata_for_grid(grid_id)
        num_grid_rows = grid_metadata_dict[rap_model_utils.NUM_ROWS_COLUMN]
        num_grid_columns = grid_metadata_dict[
            rap_model_utils.NUM_COLUMNS_COLUMN]
        sentinel_value = rap_model_utils.SENTINEL_VALUE

    field_matrix = grib_io.read_field_from_grib_file(
        grib_file_name, grib1_field_name=grib1_field_name,
        single_field_file_name=single_field_file_name,
        wgrib_exe_name=wgrib_exe_name, wgrib2_exe_name=wgrib2_exe_name,
        num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns,
        sentinel_value=sentinel_value,
        delete_single_field_file=delete_single_field_file,
        raise_error_if_fails=raise_error_if_fails)

    if field_matrix is None:
        return None, None
    return field_matrix, single_field_file_name
