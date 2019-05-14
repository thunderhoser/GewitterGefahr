"""IO methods for NWP (numerical weather prediction) data."""

import os
import copy
import numpy
from gewittergefahr.gg_io import grib_io
from gewittergefahr.gg_io import downloads
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking

TIME_FORMAT_MONTH = '%Y%m'
TIME_FORMAT_DATE = '%Y%m%d'
TIME_FORMAT_HOUR = '%Y%m%d_%H00'
NARR_ID_FOR_FILE_NAMES = 'narr-a_221'


def _lead_time_to_string(lead_time_hours):
    """Converts lead time from number to string.

    :param lead_time_hours: Lead time (integer).
    :return: lead_time_hour_string: Lead time (format "HHH").
    """

    error_checking.assert_is_geq(lead_time_hours, 0)
    return '{0:03d}'.format(int(numpy.round(lead_time_hours)))


def _get_pathless_file_name_prefixes(model_name, grid_id=None):
    """Returns possible starts of pathless file names for the given model/grid.

    :param model_name: See doc for `nwp_model_utils.check_grid_name`.
    :param grid_id: Same.
    :return: pathless_file_name_prefixes: 1-D list with possible starts of
        pathless file names.
    """

    grid_id = nwp_model_utils.check_grid_name(
        model_name=model_name, grid_name=grid_id)

    if model_name == nwp_model_utils.NARR_MODEL_NAME:
        return [NARR_ID_FOR_FILE_NAMES]

    if model_name == nwp_model_utils.RAP_MODEL_NAME:
        return ['{0:s}_{1:s}'.format(model_name, grid_id)]

    return ['ruc2_{0:s}'.format(grid_id), 'ruc2anl_{0:s}'.format(grid_id)]


def _get_pathless_grib_file_names(
        init_time_unix_sec, model_name, grid_id=None, lead_time_hours=None):
    """Returns possible pathless file names for the given model/grid.

    :param init_time_unix_sec: Model-initialization time.
    :param model_name: See doc for `nwp_model_utils.check_grid_name`.
    :param grid_id: Same.
    :param lead_time_hours: Lead time (valid time minus init time).
    :return: pathless_file_names: 1-D list of possible pathless file names.
    """

    pathless_file_name_prefixes = _get_pathless_file_name_prefixes(
        model_name=model_name, grid_id=grid_id)

    grib_file_types = nwp_model_utils.model_to_grib_types(model_name)
    if model_name == nwp_model_utils.NARR_MODEL_NAME:
        lead_time_hours = 0

    pathless_file_names = []
    for this_prefix in pathless_file_name_prefixes:
        for this_file_type in grib_file_types:
            this_pathless_file_name = '{0:s}_{1:s}_{2:s}{3:s}'.format(
                this_prefix,
                time_conversion.unix_sec_to_string(
                    init_time_unix_sec, TIME_FORMAT_HOUR),
                _lead_time_to_string(lead_time_hours),
                grib_io.file_type_to_extension(this_file_type))

            pathless_file_names.append(this_pathless_file_name)

    return pathless_file_names


def find_grib_file(
        top_directory_name, init_time_unix_sec, model_name, grid_id=None,
        lead_time_hours=None, raise_error_if_missing=True):
    """Finds grib file.

    :param top_directory_name: Name of top-level directory with grib files.
    :param init_time_unix_sec: Model-initialization time.
    :param model_name: See doc for `nwp_model_utils.check_grid_name`.
    :param grid_id: Same.
    :param lead_time_hours: Lead time.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        raise_error_if_missing = True, this method will error out.
    :return: grib_file_name: Path to grib file.  If file is missing and
        raise_error_if_missing = False, this will be the *expected* path.
    :raises: ValueError: if file is missing and raise_error_if_missing = True.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    pathless_file_names = _get_pathless_grib_file_names(
        init_time_unix_sec=init_time_unix_sec, model_name=model_name,
        grid_id=grid_id, lead_time_hours=lead_time_hours)

    possible_grib_file_names = []
    for this_pathless_file_name in pathless_file_names[::-1]:
        grib_file_name = '{0:s}/{1:s}/{2:s}'.format(
            top_directory_name,
            time_conversion.unix_sec_to_string(
                init_time_unix_sec, TIME_FORMAT_MONTH),
            this_pathless_file_name)

        possible_grib_file_names.append(grib_file_name)
        if os.path.isfile(grib_file_name):
            break

    if raise_error_if_missing and not os.path.isfile(grib_file_name):
        error_string = (
            '\n\n{0:s}\nCannot find grib file.  Expected at one of the above '
            'locations.'
        ).format(str(possible_grib_file_names))
        raise ValueError(error_string)

    return grib_file_name


def find_ruc_file_any_grid(
        top_directory_name, init_time_unix_sec, lead_time_hours,
        raise_error_if_missing=True):
    """Finds RUC (Rapid Update Cycle) file on any grid.

    :param top_directory_name: Name of top-level directory with grib files.
    :param init_time_unix_sec: Model-initialization time.
    :param lead_time_hours: Lead time.
    :param raise_error_if_missing: Boolean flag.  If no file is found and
        raise_error_if_missing = True, this method will error out.
    :return: grib_file_name: Path to grib file.  If no file is found and
        raise_error_if_missing = False, this will be None.
    """

    error_checking.assert_is_boolean(raise_error_if_missing)
    grid_ids = nwp_model_utils.RUC_GRID_NAMES

    for i in range(len(grid_ids)):
        grib_file_name = find_grib_file(
            top_directory_name=top_directory_name,
            init_time_unix_sec=init_time_unix_sec,
            model_name=nwp_model_utils.RUC_MODEL_NAME, grid_id=grid_ids[i],
            lead_time_hours=lead_time_hours,
            raise_error_if_missing=(
                raise_error_if_missing and i == len(grid_ids) - 1))
        if os.path.isfile(grib_file_name):
            return grib_file_name

    return None


def find_rap_file_any_grid(
        top_directory_name, init_time_unix_sec, lead_time_hours,
        raise_error_if_missing=True):
    """Finds RAP (Rapid Refresh) file on any grid.

    :param top_directory_name: See doc for `find_ruc_file_any_grid`.
    :param init_time_unix_sec: Same.
    :param lead_time_hours: Same.
    :param raise_error_if_missing: Same.
    :return: grib_file_name: Same.
    """

    error_checking.assert_is_boolean(raise_error_if_missing)
    grid_ids = nwp_model_utils.RAP_GRID_NAMES

    for i in range(len(grid_ids)):
        grib_file_name = find_grib_file(
            top_directory_name=top_directory_name,
            init_time_unix_sec=init_time_unix_sec,
            model_name=nwp_model_utils.RAP_MODEL_NAME, grid_id=grid_ids[i],
            lead_time_hours=lead_time_hours,
            raise_error_if_missing=(
                raise_error_if_missing and i == len(grid_ids) - 1))
        if os.path.isfile(grib_file_name):
            return grib_file_name

    return None


def download_grib_file(
        top_local_directory_name, init_time_unix_sec, model_name, grid_id=None,
        lead_time_hours=None, raise_error_if_fails=None):
    """Downloads grib file to local machine.

    :param top_local_directory_name: Name of top-level directory for grib files
        on local machine.
    :param init_time_unix_sec: Model-initialization time.
    :param model_name: See doc for `nwp_model_utils.check_grid_name`.
    :param grid_id: Same.
    :param lead_time_hours: Lead time.
    :param raise_error_if_fails: Boolean flag.  If download fails and
        raise_error_if_fails = True, this method will error out.
    :return: local_file_name: Path to grib file on local machine.  If download
        fails and raise_error_if_fails = False, this will be None.
    """

    error_checking.assert_is_boolean(raise_error_if_fails)

    pathless_file_names = _get_pathless_grib_file_names(
        init_time_unix_sec=init_time_unix_sec, model_name=model_name,
        grid_id=grid_id, lead_time_hours=lead_time_hours)
    top_online_dir_names = nwp_model_utils.get_online_directories(
        model_name=model_name, grid_name=grid_id)

    desired_local_file_name = find_grib_file(
        top_directory_name=top_local_directory_name,
        init_time_unix_sec=init_time_unix_sec, model_name=model_name,
        grid_id=grid_id, lead_time_hours=lead_time_hours,
        raise_error_if_missing=False)

    for i in range(len(pathless_file_names)):
        for j in range(len(top_online_dir_names)):
            this_online_file_name = '{0:s}/{1:s}/{2:s}/{3:s}'.format(
                top_online_dir_names[j],
                time_conversion.unix_sec_to_string(
                    init_time_unix_sec, TIME_FORMAT_MONTH),
                time_conversion.unix_sec_to_string(
                    init_time_unix_sec, TIME_FORMAT_DATE),
                pathless_file_names[i])

            raise_error_now = (
                raise_error_if_fails and i == len(pathless_file_names) - 1 and
                j == len(top_online_dir_names) - 1)
            local_file_name = downloads.download_files_via_http(
                online_file_names=[this_online_file_name],
                local_file_names=[desired_local_file_name],
                raise_error_if_fails=raise_error_now)[0]

            if local_file_name is None:
                continue

            extensionless_local_file_name, local_file_extension = (
                os.path.splitext(local_file_name))
            if this_online_file_name.endswith(local_file_extension):
                break

            if local_file_extension == grib_io.GRIB1_FILE_EXTENSION:
                new_local_file_name = '{0:s}{1:s}'.format(
                    extensionless_local_file_name, grib_io.GRIB2_FILE_EXTENSION)
            else:
                new_local_file_name = '{0:s}{1:s}'.format(
                    extensionless_local_file_name, grib_io.GRIB1_FILE_EXTENSION)

            os.rename(local_file_name, new_local_file_name)
            local_file_name = copy.deepcopy(new_local_file_name)
            break

        if local_file_name is not None:
            break

    return local_file_name


def download_ruc_file_any_grid(
        top_local_directory_name, init_time_unix_sec, lead_time_hours,
        raise_error_if_fails=True):
    """Downloads RUC (Rapid Update Cycle) file on any grid.

    :param top_local_directory_name: Name of top-level directory for grib files
        on local machine.
    :param init_time_unix_sec: Model-initialization time.
    :param lead_time_hours: Lead time.
    :param raise_error_if_fails: See doc for `download_grib_file`.
    :return: local_file_name: See doc for `download_grib_file`.
    """

    error_checking.assert_is_boolean(raise_error_if_fails)

    # grid_ids = nwp_model_utils.RUC_GRID_IDS
    grid_ids = [
        nwp_model_utils.NAME_OF_130GRID, nwp_model_utils.NAME_OF_252GRID
    ]

    for i in range(len(grid_ids)):
        local_file_name = download_grib_file(
            top_local_directory_name=top_local_directory_name,
            init_time_unix_sec=init_time_unix_sec,
            model_name=nwp_model_utils.RUC_MODEL_NAME, grid_id=grid_ids[i],
            lead_time_hours=lead_time_hours,
            raise_error_if_fails=(
                raise_error_if_fails and i == len(grid_ids) - 1))

        if local_file_name is not None:
            break

    return local_file_name


def download_rap_file_any_grid(
        top_local_directory_name, init_time_unix_sec, lead_time_hours,
        raise_error_if_fails=True):
    """Downloads RAP (Rapid Refresh) file on any grid.

    :param top_local_directory_name: Name of top-level directory for grib files
        on local machine.
    :param init_time_unix_sec: Model-initialization time.
    :param lead_time_hours: Lead time.
    :param raise_error_if_fails: See doc for `download_grib_file`.
    :return: local_file_name: See doc for `download_grib_file`.
    """

    error_checking.assert_is_boolean(raise_error_if_fails)

    # grid_ids = nwp_model_utils.RAP_GRID_IDS
    grid_ids = [nwp_model_utils.NAME_OF_130GRID, nwp_model_utils.NAME_OF_252GRID]

    for i in range(len(grid_ids)):
        local_file_name = download_grib_file(
            top_local_directory_name=top_local_directory_name,
            init_time_unix_sec=init_time_unix_sec,
            model_name=nwp_model_utils.RAP_MODEL_NAME, grid_id=grid_ids[i],
            lead_time_hours=lead_time_hours,
            raise_error_if_fails=(
                raise_error_if_fails and i == len(grid_ids) - 1))

        if local_file_name is not None:
            break

    return local_file_name


def read_field_from_grib_file(
        grib_file_name, field_name_grib1, model_name, grid_id=None,
        temporary_dir_name=None, wgrib_exe_name=grib_io.WGRIB_EXE_NAME_DEFAULT,
        wgrib2_exe_name=grib_io.WGRIB2_EXE_NAME_DEFAULT,
        raise_error_if_fails=True):
    """Reads field from grib file.

    One field = one variable at one time step.

    :param grib_file_name: Path to input file.
    :param field_name_grib1: See doc for `grib_io.read_field_from_grib_file`.
    :param model_name: See doc for `nwp_model_utils.check_grid_name`.
    :param grid_id: Same.
    :param temporary_dir_name: See doc for `grib_io.read_field_from_grib_file`.
    :param wgrib_exe_name: Same.
    :param wgrib2_exe_name: Same.
    :param raise_error_if_fails: Same.
    :return: field_matrix: Same.
    """

    num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
        model_name=model_name, grid_name=grid_id)

    return grib_io.read_field_from_grib_file(
        grib_file_name=grib_file_name, field_name_grib1=field_name_grib1,
        num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns,
        sentinel_value=nwp_model_utils.SENTINEL_VALUE,
        temporary_dir_name=temporary_dir_name, wgrib_exe_name=wgrib_exe_name,
        wgrib2_exe_name=wgrib2_exe_name,
        raise_error_if_fails=raise_error_if_fails)
