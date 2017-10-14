"""IO methods for grib and grib2 files.

These methods use wgrib and wgrib2, which are command-line tools for parsing
grib and grib2 files.  See README_grib (in this directory) for installation
instructions.
"""

import os
import subprocess
import warnings
import numpy
from gewittergefahr.gg_io import downloads
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

# TODO(thunderhoser): replace main method with named method.

SENTINEL_TOLERANCE = 10.

GRIB1_FILE_EXTENSION = '.grb'
GRIB1_FILE_TYPE = 'grib1'
WGRIB_EXE_NAME_DEFAULT = '/usr/bin/wgrib'

GRIB2_FILE_EXTENSION = '.grb2'
GRIB2_FILE_TYPE = 'grib2'
WGRIB2_EXE_NAME_DEFAULT = '/usr/bin/wgrib2'

# The following constants are used in the main method only.
NARR_FILE_NAME_ONLINE = (
    'https://nomads.ncdc.noaa.gov/data/narr/201408/20140810/'
    'narr-a_221_20140810_1200_000.grb')
NARR_FILE_NAME_LOCAL = (
    '/localdata/ryan.lagerquist/gewittergefahr_junk/'
    'narr-a_221_20140810_1200_000.grb')
NARR_H500_FILE_NAME = (
    '/localdata/ryan.lagerquist/gewittergefahr_junk/'
    'narr_h500_20140810_1200_000.txt')

RAP_FILE_NAME_ONLINE = (
    'https://nomads.ncdc.noaa.gov/data/rap130/201708/20170822/'
    'rap_130_20170822_0000_006.grb2')
RAP_FILE_NAME_LOCAL = (
    '/localdata/ryan.lagerquist/gewittergefahr_junk/'
    'rap_130_20170822_0000_006.grb2')
RAP_H500_FILE_NAME = (
    '/localdata/ryan.lagerquist/gewittergefahr_junk/'
    'rap_h500_20170822_0000_006.txt')

H500_NAME_GRIB = 'HGT:500 mb'

NUM_ROWS_IN_NARR = 277
NUM_COLUMNS_IN_NARR = 349
NARR_SENTINEL_VALUE = 9.999e20

NUM_ROWS_IN_RAP = 337
NUM_COLUMNS_IN_RAP = 451
RAP_SENTINEL_VALUE = 9.999e20


def _get_file_type(grib_file_name):
    """Determines whether file type is grib1 or grib2.

    :param grib_file_name: Path to input file.
    :return: file_type: Either "grib1" or "grib2".
    :raises: ValueError: if file type is neither grib1 nor grib2.
    """

    error_checking.assert_is_string(grib_file_name)
    if grib_file_name.endswith(GRIB1_FILE_EXTENSION):
        return GRIB1_FILE_TYPE
    if grib_file_name.endswith(GRIB2_FILE_EXTENSION):
        return GRIB2_FILE_TYPE

    error_string = (
        'Expected file extension to be either "' + GRIB1_FILE_EXTENSION +
        '" or "' + GRIB2_FILE_EXTENSION + '".  Instead, got:\n' +
        grib_file_name)
    raise ValueError(error_string)


def _field_name_grib1_to_grib2(grib1_field_name):
    """Converts field name from grib1 to grib2 format.

    :param grib1_field_name: Field name in grib1 format.
    :return: grib2_field_name: Field name in grib2 format.
    """

    return grib1_field_name.replace('gnd', 'ground').replace('sfc', 'surface')


def _replace_sentinels_with_nan(data_matrix, sentinel_value=None):
    """Replaces all occurrences of sentinel value with NaN.

    :param data_matrix: numpy array, which may contain sentinels.
    :param sentinel_value: Sentinel value.
    :return: data_matrix: numpy array without sentinels.
    """

    if sentinel_value is None:
        return data_matrix

    data_vector = numpy.reshape(data_matrix, data_matrix.size)
    sentinel_flags = numpy.isclose(
        data_vector, sentinel_value, atol=SENTINEL_TOLERANCE)

    sentinel_indices = numpy.where(sentinel_flags)[0]
    data_vector[sentinel_indices] = numpy.nan
    return numpy.reshape(data_vector, data_matrix.shape)


def _extract_single_field_to_file(grib_file_name, grib1_field_name=None,
                                  output_file_name=None,
                                  wgrib_exe_name=WGRIB_EXE_NAME_DEFAULT,
                                  wgrib2_exe_name=WGRIB2_EXE_NAME_DEFAULT,
                                  raise_error_if_fails=True):
    """Extracts single field from grib1 or grib2 file; writes to text file.

    A "single field" is one variable at one time step and all grid cells.

    :param grib_file_name: Path to input (grib1 or grib2) file.
    :param grib1_field_name: Field name in grib1 format (example: 500-mb height
        is "HGT:500 mb").
    :param output_file_name: Path to output file.
    :param wgrib_exe_name: Path to wgrib executable.
    :param wgrib2_exe_name: Path to wgrib2 executable.
    :param raise_error_if_fails: Boolean flag.  If command fails and
        raise_error_if_fails = True, will raise an error.
    :return: success: Boolean flag.  If command succeeded, this is True.  If
        command failed and raise_error_if_fails = False, this is False.
    :raises: OSError: if command fails and raise_error_if_fails = True.
    """

    grib_file_type = _get_file_type(grib_file_name)
    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    if grib_file_type == GRIB1_FILE_TYPE:
        command_string = (
            '"' + wgrib_exe_name + '" "' + grib_file_name + '" -s | grep -w "' +
            grib1_field_name + '" | "' + wgrib_exe_name + '" -i "' +
            grib_file_name + '" -text -nh -o "' + output_file_name + '"')
    else:
        command_string = (
            '"' + wgrib2_exe_name + '" "' + grib_file_name + '" -s | grep -w "'
            + _field_name_grib1_to_grib2(grib1_field_name) + '" | "' +
            wgrib2_exe_name + '" -i "' + grib_file_name + '" -no_header -text "'
            + output_file_name + '"')

    try:
        subprocess.call(command_string, shell=True)
    except OSError as this_exception:
        if raise_error_if_fails:
            raise

        warn_string = (
            '\n\n' + command_string +
            '\n\nCommand (shown above) failed (details shown below).\n\n' +
            str(this_exception))
        warnings.warn(warn_string)
        return False

    return True


def _read_single_field_from_file(input_file_name, num_grid_rows=None,
                                 num_grid_columns=None, sentinel_value=None,
                                 raise_error_if_fails=True):
    """Reads single field from text file.

    A "single field" is one variable at one time step and all grid cells.

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)

    :param input_file_name: Path to input file.  File should be in the format
        generated by _extract_single_field_to_file.
    :param num_grid_rows: Number of rows in grid.
    :param num_grid_columns: Number of columns in grid.
    :param sentinel_value: Sentinel value, all occurrences of which will be
        replaced with NaN.
    :param raise_error_if_fails: Boolean flag.  If read fails and
        raise_error_if_fails = True, will raise an error.
    :return: field_matrix: If read fails, this is None.  Otherwise, M-by-N numpy
        array with values of field.  If the grid is regular in x-y, x-coordinate
        increases towards the right and y-coordinate increases towards the
        bottom.  If the grid is regular in lat-long, longitude increases towards
        the right and latitude increases towards the bottom.
    :raises: ValueError: if read fails and raise_error_if_fails = True.
    """

    field_vector = numpy.loadtxt(input_file_name)

    try:
        field_matrix = numpy.reshape(
            field_vector, (num_grid_rows, num_grid_columns))
    except ValueError as this_exception:
        if raise_error_if_fails:
            raise

        warn_string = (
            '\n\n' + str(this_exception) + '\n\nnumpy.reshape failed (probably '
            + 'wrong number of grid points in file -- details shown above).')
        warnings.warn(warn_string)
        return None

    return _replace_sentinels_with_nan(field_matrix, sentinel_value)


def file_type_to_extension(grib_file_type):
    """Converts file type to extension.

    :param grib_file_type: File type (either "grib1" or "grib2").
    :return: grib_file_extension: File extension (either ".grb" or ".grb2").
    :raises: ValueError: if file type is neither "grib1" nor "grib2".
    """

    error_checking.assert_is_string(grib_file_type)
    if grib_file_type == GRIB1_FILE_TYPE:
        return GRIB1_FILE_EXTENSION
    if grib_file_type == GRIB2_FILE_TYPE:
        return GRIB2_FILE_EXTENSION

    error_string = (
        'Expected file type to be either "' + GRIB1_FILE_TYPE + '" or "' +
        GRIB2_FILE_TYPE + '".  Instead, got:\n' + grib_file_type)
    raise ValueError(error_string)


def read_field_from_grib_file(grib_file_name, grib1_field_name=None,
                              single_field_file_name=None,
                              wgrib_exe_name=WGRIB_EXE_NAME_DEFAULT,
                              wgrib2_exe_name=WGRIB2_EXE_NAME_DEFAULT,
                              num_grid_rows=None, num_grid_columns=None,
                              sentinel_value=None,
                              delete_single_field_file=True,
                              raise_error_if_fails=True):
    """Reads single field from grib1 or grib2 file.

    A "single field" is one variable at one time step and all grid cells.

    :param grib_file_name: Path to input (grib1 or grib2) file.
    :param grib1_field_name: Field name in grib1 format (example: 500-mb height
        is "HGT:500 mb").
    :param single_field_file_name: Single field will be extracted from grib file
        to here.
    :param wgrib_exe_name: Path to wgrib executable.
    :param wgrib2_exe_name: Path to wgrib2 executable.
    :param num_grid_rows: Number of rows in grid.
    :param num_grid_columns: Number of columns in grid.
    :param sentinel_value: Sentinel value, all occurrences of which will be
        replaced with NaN.
    :param delete_single_field_file: Boolean flag.  If True, single-field file
        will be deleted immediately upon reading.
    :param raise_error_if_fails: Boolean flag.  If read fails and
        raise_error_if_fails = True, will raise an error.
    :return: field_matrix: See documentation for _read_single_field_from_file.
    """

    error_checking.assert_file_exists(grib_file_name)
    error_checking.assert_is_string(grib1_field_name)
    error_checking.assert_file_exists(wgrib_exe_name)
    error_checking.assert_file_exists(wgrib2_exe_name)
    error_checking.assert_is_integer(num_grid_rows)
    error_checking.assert_is_greater(num_grid_rows, 0)
    error_checking.assert_is_integer(num_grid_columns)
    error_checking.assert_is_greater(num_grid_columns, 0)
    if sentinel_value is not None:
        error_checking.assert_is_not_nan(sentinel_value)

    error_checking.assert_is_boolean(delete_single_field_file)
    error_checking.assert_is_boolean(raise_error_if_fails)

    success = _extract_single_field_to_file(
        grib_file_name, grib1_field_name=grib1_field_name,
        output_file_name=single_field_file_name, wgrib_exe_name=wgrib_exe_name,
        wgrib2_exe_name=wgrib2_exe_name,
        raise_error_if_fails=raise_error_if_fails)
    if not success:
        return None

    field_matrix = _read_single_field_from_file(
        single_field_file_name, num_grid_rows=num_grid_rows,
        num_grid_columns=num_grid_columns, sentinel_value=sentinel_value,
        raise_error_if_fails=raise_error_if_fails)

    if delete_single_field_file and os.path.isfile(single_field_file_name):
        os.remove(single_field_file_name)

    return field_matrix


if __name__ == '__main__':
    downloads.download_file_via_http(NARR_FILE_NAME_ONLINE,
                                     NARR_FILE_NAME_LOCAL)
    downloads.download_file_via_http(RAP_FILE_NAME_ONLINE, RAP_FILE_NAME_LOCAL)

    NARR_H500_MATRIX_METRES = read_field_from_grib_file(
        NARR_FILE_NAME_LOCAL, grib1_field_name=H500_NAME_GRIB,
        single_field_file_name=NARR_H500_FILE_NAME,
        wgrib_exe_name=WGRIB_EXE_NAME_DEFAULT,
        wgrib2_exe_name=WGRIB2_EXE_NAME_DEFAULT, num_grid_rows=NUM_ROWS_IN_NARR,
        num_grid_columns=NUM_COLUMNS_IN_NARR,
        sentinel_value=NARR_SENTINEL_VALUE)
    print NARR_H500_MATRIX_METRES
    print numpy.nanmin(NARR_H500_MATRIX_METRES)
    print numpy.nanmax(NARR_H500_MATRIX_METRES)

    RAP_H500_MATRIX_METRES = read_field_from_grib_file(
        RAP_FILE_NAME_LOCAL, grib1_field_name=H500_NAME_GRIB,
        single_field_file_name=RAP_H500_FILE_NAME,
        wgrib_exe_name=WGRIB_EXE_NAME_DEFAULT,
        wgrib2_exe_name=WGRIB2_EXE_NAME_DEFAULT, num_grid_rows=NUM_ROWS_IN_RAP,
        num_grid_columns=NUM_COLUMNS_IN_RAP, sentinel_value=RAP_SENTINEL_VALUE)
    print RAP_H500_MATRIX_METRES
    print numpy.nanmin(RAP_H500_MATRIX_METRES)
    print numpy.nanmax(RAP_H500_MATRIX_METRES)
