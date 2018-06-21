"""IO methods for grib and grib2 files.

These methods use wgrib and wgrib2, which are command-line tools.  See
README_grib (in the same directory as this module) for installation
instructions.
"""

import os
import subprocess
import tempfile
import warnings
import numpy
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

TOLERANCE_FOR_SENTINEL_VALUES = 10.

GRIB1_FILE_EXTENSION = '.grb'
GRIB1_FILE_TYPE = 'grib1'
WGRIB_EXE_NAME_DEFAULT = '/usr/bin/wgrib'

GRIB2_FILE_EXTENSION = '.grb2'
GRIB2_FILE_TYPE = 'grib2'
WGRIB2_EXE_NAME_DEFAULT = '/usr/bin/wgrib2'
VALID_FILE_TYPES = [GRIB1_FILE_TYPE, GRIB2_FILE_TYPE]

U_WIND_PREFIX = 'UGRD'
V_WIND_PREFIX = 'VGRD'


def _field_name_grib1_to_grib2(field_name_grib1):
    """Converts field name from grib1 to grib2.

    :param field_name_grib1: Field name in grib1 format.
    :return: field_name_grib2: Field name in grib2 format.
    """

    return field_name_grib1.replace('gnd', 'ground').replace('sfc', 'surface')


def _sentinel_value_to_nan(data_matrix, sentinel_value=None):
    """Replaces all instances of sentinel value with NaN.

    :param data_matrix: numpy array (may contain sentinel values).
    :param sentinel_value: Sentinel value (may be None).
    :return: data_matrix: numpy array without sentinel values.
    """

    if sentinel_value is None:
        return data_matrix

    data_vector = numpy.reshape(data_matrix, data_matrix.size)
    sentinel_flags = numpy.isclose(
        data_vector, sentinel_value, atol=TOLERANCE_FOR_SENTINEL_VALUES)

    sentinel_indices = numpy.where(sentinel_flags)[0]
    data_vector[sentinel_indices] = numpy.nan
    return numpy.reshape(data_vector, data_matrix.shape)


def check_file_type(grib_file_type):
    """Ensures that grib file type is valid.

    :param grib_file_type: Either "grib1" or "grib2".
    :raises: ValueError: if `grib_file_type not in VALID_FILE_TYPES`.
    """

    error_checking.assert_is_string(grib_file_type)
    if grib_file_type not in VALID_FILE_TYPES:
        error_string = (
            '\n\n{0:s}\nValid file types (listed above) do not include "{1:s}".'
        ).format(str(VALID_FILE_TYPES), grib_file_type)
        raise ValueError(error_string)


def file_name_to_type(grib_file_name):
    """Determines file type (either grib1 or grib2) from file name.

    :param grib_file_name: Path to input file.
    :return: grib_file_type: Either "grib1" or "grib2".
    :raises: ValueError: if file type is neither grib1 nor grib2.
    """

    error_checking.assert_is_string(grib_file_name)
    if grib_file_name.endswith(GRIB1_FILE_EXTENSION):
        return GRIB1_FILE_TYPE
    if grib_file_name.endswith(GRIB2_FILE_EXTENSION):
        return GRIB2_FILE_TYPE

    error_string = (
        'File type should be either "{0:s}" or "{1:s}".  Instead, got: "{2:s}"'
    ).format(GRIB1_FILE_TYPE, GRIB2_FILE_TYPE, grib_file_name)
    raise ValueError(error_string)


def read_field_from_grib_file(
        grib_file_name, field_name_grib1, num_grid_rows, num_grid_columns,
        sentinel_value=None, temporary_dir_name=None,
        wgrib_exe_name=WGRIB_EXE_NAME_DEFAULT,
        wgrib2_exe_name=WGRIB2_EXE_NAME_DEFAULT, raise_error_if_fails=True):
    """Reads field from grib file.

    One field = one variable at one time step.

    M = number of rows (unique y-coordinates or latitudes of grid points)
    N = number of columns (unique x-coordinates or longitudes of grid points)

    :param grib_file_name: Path to input file.
    :param field_name_grib1: Field name in grib1 format (example: 500-mb height
        is "HGT:500 mb").
    :param num_grid_rows: Number of rows expected in grid.
    :param num_grid_columns: Number of columns expected in grid.
    :param sentinel_value: Sentinel value (all instances will be replaced with
        NaN).
    :param temporary_dir_name: Name of temporary directory.  An intermediate
        text file will be stored here.
    :param wgrib_exe_name: Path to wgrib executable.
    :param wgrib2_exe_name: Path to wgrib2 executable.
    :param raise_error_if_fails: Boolean flag.  If the extraction fails and
        raise_error_if_fails = True, this method will error out.  If the
        extraction fails and raise_error_if_fails = False, this method will
        return None.
    :return: field_matrix: M-by-N numpy array with values of the given field.
        If the grid is regular in x-y coordinates, x increases towards the right
        (in the positive direction of the second axis), while y increases
        downward (in the positive direction of the first axis).  If the grid is
        regular in lat-long, replace "x" and "y" in the previous sentence with
        "long" and "lat," respectively.
    :raises: ValueError: if extraction fails and raise_error_if_fails = True.
    """

    # Error-checking.
    error_checking.assert_is_string(field_name_grib1)
    error_checking.assert_is_integer(num_grid_rows)
    error_checking.assert_is_greater(num_grid_rows, 0)
    error_checking.assert_is_integer(num_grid_columns)
    error_checking.assert_is_greater(num_grid_columns, 0)
    error_checking.assert_file_exists(wgrib_exe_name)
    error_checking.assert_file_exists(wgrib2_exe_name)
    error_checking.assert_is_boolean(raise_error_if_fails)
    if sentinel_value is not None:
        error_checking.assert_is_not_nan(sentinel_value)

    # Housekeeping.
    grib_file_type = file_name_to_type(grib_file_name)

    if temporary_dir_name is not None:
        file_system_utils.mkdir_recursive_if_necessary(
            directory_name=temporary_dir_name)
    temporary_file_name = tempfile.NamedTemporaryFile(
        dir=temporary_dir_name, delete=False).name

    # Extract field to temporary file.
    if grib_file_type == GRIB1_FILE_TYPE:
        command_string = (
            '"{0:s}" "{1:s}" -s | grep -w "{2:s}" | "{0:s}" -i "{1:s}" '
            '-text -nh -o "{3:s}"'
        ).format(wgrib_exe_name, grib_file_name, field_name_grib1,
                 temporary_file_name)
    else:
        command_string = (
            '"{0:s}" "{1:s}" -s | grep -w "{2:s}" | "{0:s}" -i "{1:s}" '
            '-no_header -text "{3:s}"'
        ).format(wgrib2_exe_name, grib_file_name,
                 _field_name_grib1_to_grib2(field_name_grib1),
                 temporary_file_name)

    print command_string

    try:
        subprocess.call(command_string, shell=True)
    except OSError as this_exception:
        os.remove(temporary_file_name)
        if raise_error_if_fails:
            raise

        warning_string = (
            '\n\n{0:s}\n\nCommand (shown above) failed (details shown below).'
            '\n\n{1:s}'
        ).format(command_string, str(this_exception))
        warnings.warn(warning_string)
        return None

    # Read field from temporary file.
    field_vector = numpy.loadtxt(temporary_file_name)
    os.remove(temporary_file_name)

    try:
        field_matrix = numpy.reshape(
            field_vector, (num_grid_rows, num_grid_columns))
    except ValueError as this_exception:
        if raise_error_if_fails:
            raise

        warning_string = (
            '\n\nnumpy.reshape failed (details shown below).\n\n{0:s}'
        ).format(str(this_exception))
        warnings.warn(warning_string)
        return None

    return _sentinel_value_to_nan(
        data_matrix=field_matrix, sentinel_value=sentinel_value)


def is_u_wind_field(field_name_grib1):
    """Determines whether or not field is a u-wind field.

    :param field_name_grib1: Field name in grib1 format.
    :return: is_u_wind_flag: Boolean flag.
    """

    error_checking.assert_is_string(field_name_grib1)
    return field_name_grib1.startswith(U_WIND_PREFIX)


def is_v_wind_field(field_name_grib1):
    """Determines whether or not field is a v-wind field.

    :param field_name_grib1: Field name in grib1 format.
    :return: is_v_wind_flag: Boolean flag.
    """

    error_checking.assert_is_string(field_name_grib1)
    return field_name_grib1.startswith(V_WIND_PREFIX)


def is_wind_field(field_name_grib1):
    """Determines whether or not field is a wind field.

    :param field_name_grib1: Field name in grib1 format.
    :return: is_wind_flag: Boolean flag.
    """

    return is_u_wind_field(field_name_grib1) or is_v_wind_field(
        field_name_grib1)


def switch_uv_in_field_name(field_name_grib1):
    """Switches u-wind and v-wind in field name.

    In other words, if the original field is u-wind, this method converts it to
    the equivalent v-wind field.  If the original field is v-wind, this method
    converts to the equivalent u-wind field.

    :param field_name_grib1: Field name in grib1 format.
    :return: switched_field_name_grib1: See above discussion.
    """

    if not is_wind_field(field_name_grib1):
        return field_name_grib1
    if is_u_wind_field(field_name_grib1):
        return field_name_grib1.replace(U_WIND_PREFIX, V_WIND_PREFIX)
    return field_name_grib1.replace(V_WIND_PREFIX, U_WIND_PREFIX)


def file_type_to_extension(grib_file_type):
    """Converts grib file type to file extension.

    :param grib_file_type: File type (either "grib1" or "grib2").
    :return: grib_file_extension: Expected file extension for the given type.
    """

    check_file_type(grib_file_type)
    if grib_file_type == GRIB1_FILE_TYPE:
        return GRIB1_FILE_EXTENSION
    if grib_file_type == GRIB2_FILE_TYPE:
        return GRIB2_FILE_EXTENSION

    return None
