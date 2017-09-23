"""IO methods for MYRORSS data.

DEFINITIONS

MYRORSS = Multi-year Reanalysis of Remotely Sensed Storms

SPC = Storm Prediction Center

SPC date = a 24-hour period, starting and ending at 1200 UTC.  This is unlike a
normal person's day, which starts and ends at 0000 UTC.  An SPC date is
referenced by the calendar day at the beginning of the SPC date.  In other
words, SPC date "Sep 23 2017" actually runs from 1200 UTC 23 Sep 2017 -
1200 UTC 24 Sep 2017.
"""

import collections
import numpy
import pandas
import time
import os
from netCDF4 import Dataset
from gewittergefahr.gg_utils import number_rounding as rounder

# TODO(thunderhoser): add error-checking to all methods.

NW_GRID_POINT_LAT_COLUMN = 'nw_grid_point_lat_deg'
NW_GRID_POINT_LNG_COLUMN = 'nw_grid_point_lng_deg'
LAT_SPACING_COLUMN = 'lat_spacing_deg'
LNG_SPACING_COLUMN = 'lng_spacing_deg'
NUM_LAT_COLUMN = 'num_lat_in_grid'
NUM_LNG_COLUMN = 'num_lng_in_grid'
ELEVATION_COLUMN = 'elevation_m_agl'
UNIX_TIME_COLUMN = 'unix_time_sec'
VAR_NAME_COLUMN = 'var_name'
SENTINEL_VALUE_COLUMN = 'sentinel_values'

NW_GRID_POINT_LAT_COLUMN_ORIG = 'Latitude'
NW_GRID_POINT_LNG_COLUMN_ORIG = 'Longitude'
LAT_SPACING_COLUMN_ORIG = 'LatGridSpacing'
LNG_SPACING_COLUMN_ORIG = 'LonGridSpacing'
NUM_LAT_COLUMN_ORIG = 'Lat'
NUM_LNG_COLUMN_ORIG = 'Lon'
ELEVATION_COLUMN_ORIG = 'Height'
UNIX_TIME_COLUMN_ORIG = 'Time'
VAR_NAME_COLUMN_ORIG = 'TypeName'
SENTINEL_VALUE_COLUMNS_ORIG = ['MissingData', 'RangeFolded']

GRID_ROW_COLUMN = 'grid_row'
GRID_COLUMN_COLUMN = 'grid_column'
GRID_LAT_COLUMN = 'latitude_deg'
GRID_LNG_COLUMN = 'longitude_deg'
NUM_GRID_CELL_COLUMN = 'num_grid_cells'

GRID_ROW_COLUMN_ORIG = 'pixel_x'
GRID_COLUMN_COLUMN_ORIG = 'pixel_y'
NUM_GRID_CELL_COLUMN_ORIG = 'pixel_count'

ECHO_TOP_18DBZ_NAME = 'echo_top_18dbz_km'
ECHO_TOP_50DBZ_NAME = 'echo_top_50dbz_km'
LOW_LEVEL_SHEAR_NAME = 'low_level_shear_s01'
MID_LEVEL_SHEAR_NAME = 'mid_level_shear_s01'
REFL_NAME = 'reflectivity_dbz'
REFL_COLUMN_MAX_NAME = 'reflectivity_column_max_dbz'
MESH_NAME = 'mesh_mm'
REFL_0CELSIUS_NAME = 'reflectivity_0celsius_dbz'
REFL_M10CELSIUS_NAME = 'reflectivity_m10celsius_dbz'
REFL_M20CELSIUS_NAME = 'reflectivity_m20celsius_dbz'
REFL_LOWEST_ALTITUDE_NAME = 'reflectivity_lowest_altitude_dbz'
SHI_NAME = 'shi'
VIL_NAME = 'vil_mm'
STORM_ID_NAME = 'storm_id'

ECHO_TOP_18DBZ_NAME_ORIG = 'EchoTop_18'
ECHO_TOP_50DBZ_NAME_ORIG = 'EchoTop_50'
LOW_LEVEL_SHEAR_NAME_ORIG = 'MergedLLShear'
MID_LEVEL_SHEAR_NAME_ORIG = 'MergedMLShear'
REFL_NAME_ORIG = 'MergedReflectivityQC'
REFL_COLUMN_MAX_NAME_ORIG = 'MergedReflectivityQCComposite'
MESH_NAME_ORIG = 'MESH'
REFL_0CELSIUS_NAME_ORIG = 'Reflectivity_0C'
REFL_M10CELSIUS_NAME_ORIG = 'Reflectivity_-10C'
REFL_M20CELSIUS_NAME_ORIG = 'Reflectivity_-20C'
REFL_LOWEST_ALTITUDE_NAME_ORIG = 'ReflectivityAtLowestAltitude'
SHI_NAME_ORIG = 'SHI'
VIL_NAME_ORIG = 'VIL'
STORM_ID_NAME_ORIG = 'ClusterID'

RADAR_VAR_NAMES = [ECHO_TOP_18DBZ_NAME, ECHO_TOP_50DBZ_NAME,
                   LOW_LEVEL_SHEAR_NAME, MID_LEVEL_SHEAR_NAME, REFL_NAME,
                   REFL_COLUMN_MAX_NAME, MESH_NAME, REFL_0CELSIUS_NAME,
                   REFL_M10CELSIUS_NAME, REFL_M20CELSIUS_NAME,
                   REFL_LOWEST_ALTITUDE_NAME, SHI_NAME, VIL_NAME, STORM_ID_NAME]

RADAR_VAR_NAMES_ORIG = [ECHO_TOP_18DBZ_NAME_ORIG, ECHO_TOP_50DBZ_NAME_ORIG,
                        LOW_LEVEL_SHEAR_NAME_ORIG, MID_LEVEL_SHEAR_NAME_ORIG,
                        REFL_NAME_ORIG, REFL_COLUMN_MAX_NAME_ORIG,
                        MESH_NAME_ORIG, REFL_0CELSIUS_NAME_ORIG,
                        REFL_M10CELSIUS_NAME_ORIG, REFL_M20CELSIUS_NAME_ORIG,
                        REFL_LOWEST_ALTITUDE_NAME_ORIG, SHI_NAME_ORIG,
                        VIL_NAME_ORIG, STORM_ID_NAME_ORIG]

HEIGHT_ARRAY_COLUMN = 'heights_m_agl'
RELATIVE_TOLERANCE = 1e-6

TIME_FORMAT = '%Y%m%d-%H%M%S'
SPC_DATE_FORMAT = '%Y%m%d'
DAYS_TO_SECONDS = 86400
METRES_TO_KM = 1e-3

RAW_FILE_EXTENSION1 = '.netcdf'
RAW_FILE_EXTENSION2 = '.gz'


def _time_unix_sec_to_string(unix_time_sec):
    """Converts time from Unix format to string.

    :param unix_time_sec: Time in Unix format.
    :return: time_string: Time string (format "yyyymmdd-HHMMSS").
    """

    return time.strftime(TIME_FORMAT, time.gmtime(unix_time_sec))


def _variable_to_valid_heights(variable_name):
    """Returns valid heights for given radar variable.

    :param variable_name: Name of radar variable (must be in `RADAR_VAR_NAMES`).
    :return: valid_heights_m_agl: 1-D numpy array of valid heights (metres above
        ground level).
    """

    if variable_name == ECHO_TOP_18DBZ_NAME:
        return numpy.array([250.])
    elif variable_name == ECHO_TOP_50DBZ_NAME:
        return numpy.array([250.])
    elif variable_name == LOW_LEVEL_SHEAR_NAME:
        return numpy.array([0.])
    elif variable_name == MID_LEVEL_SHEAR_NAME:
        return numpy.array([0.])
    elif variable_name == REFL_NAME:
        return numpy.array(
            [250., 500., 750., 1000., 1250., 1500., 1750., 2000., 2250., 2500.,
             2750., 3000., 3500., 4000., 4500., 5000., 5500., 6000., 6500.,
             7000., 7500., 8000., 8500., 9000., 10000., 11000., 12000., 13000.,
             14000., 15000., 16000., 17000., 18000., 19000., 20000.])
    elif variable_name == REFL_COLUMN_MAX_NAME:
        return numpy.array([250.])
    elif variable_name == MESH_NAME:
        return numpy.array([250.])
    elif variable_name == REFL_0CELSIUS_NAME:
        return numpy.array([250.])
    elif variable_name == REFL_M10CELSIUS_NAME:
        return numpy.array([250.])
    elif variable_name == REFL_M20CELSIUS_NAME:
        return numpy.array([250.])
    elif variable_name == REFL_LOWEST_ALTITUDE_NAME:
        return numpy.array([250.])
    elif variable_name == SHI_NAME:
        return numpy.array([250.])
    elif variable_name == VIL_NAME:
        return numpy.array([250.])

    return None


def _var_height_arrays_to_dict(variable_names, refl_heights_m_agl):
    """Converts two arrays (radar variables and heights) to dictionary.

    V = number of variables

    :param variable_names: length-V list with names of radar variables.  Each
        list element must be in `RADAR_VAR_NAMES`.
    :param refl_heights_m_agl: 1-D numpy array of reflectivity heights (metres
        above ground level).  These will be used only for the variable
        "reflectivity_dbz", since all others have only one valid height.
    :return: var_to_heights_dict_m_agl: Dictionary.  Each key is a variable
        name, and each value is a 1-D numpy array of heights (metres above
        ground level).
    """

    var_to_heights_dict_m_agl = {}
    for j in range(len(variable_names)):
        if variable_names[j] == REFL_NAME:
            var_to_heights_dict_m_agl.update(
                {variable_names[j]: refl_heights_m_agl})
        else:
            var_to_heights_dict_m_agl.update(
                {variable_names[j]:
                     _variable_to_valid_heights(variable_names[j])})

    return var_to_heights_dict_m_agl


def _get_directory_in_tar_file(variable_name, height_m_agl):
    """Generates expected name of directory in tar file.

    :param variable_name: Name of radar variable.  Must be in `RADAR_VAR_NAMES`.
    :param height_m_agl: Height (metres above ground level).
    :return: directory_name: Expected name of directory in tar file.
    """

    return '{0:s}/{1:05.2f}'.format(_var_name_new_to_orig(variable_name),
                                    height_m_agl * METRES_TO_KM)


def _get_pathless_raw_file_name(unix_time_sec):
    """Generates pathless name for raw MYRORSS file.

    This file should contain one variable at one height and one time step.

    :param unix_time_sec: Time in Unix format.
    :return: pathless_raw_file_name: Pathless name for MYRORSS file.
    """

    return '{0:s}{1:s}{2:s}'.format(_time_unix_sec_to_string(unix_time_sec),
                                    RAW_FILE_EXTENSION1, RAW_FILE_EXTENSION2)


def _var_name_orig_to_new(variable_name_orig):
    """Converts name of radar variable from original to new format.

    :param variable_name_orig: Original variable name (must be in
        `RADAR_VAR_NAMES_ORIG`).
    :return: variable_name: New variable name (in `RADAR_VAR_NAMES`).
    """

    found_in_orig_flags = [s == variable_name_orig for s in
                           RADAR_VAR_NAMES_ORIG]
    found_in_orig_index = numpy.where(found_in_orig_flags)[0][0]
    return RADAR_VAR_NAMES[found_in_orig_index]


def _var_name_new_to_orig(variable_name):
    """Converts name of radar variable from new to original format.

    :param variable_name: New variable name (must be in `RADAR_VAR_NAMES`).
    :return: variable_name_orig: Original variable name (in
        `RADAR_VAR_NAMES_ORIG`).
    """

    found_in_new_flags = [s == variable_name for s in RADAR_VAR_NAMES]
    found_in_new_index = numpy.where(found_in_new_flags)[0][0]
    return RADAR_VAR_NAMES_ORIG[found_in_new_index]


def _remove_sentinels(sparse_grid_table, var_name, sentinel_values):
    """Removes sentinel values from radar data.

    :param sparse_grid_table: pandas DataFrame in format produced by
        read_sparse_grid_from_netcdf.
    :param var_name: Name of radar variable.
    :param sentinel_values: 1-D numpy array of sentinel values.
    :return: sparse_grid_table: Same as input, but any row with a sentinel value
        has been removed.
    """

    num_rows = len(sparse_grid_table.index)
    sentinel_flags = numpy.full(num_rows, False, dtype=bool)

    for i in range(len(sentinel_values)):
        these_sentinel_flags = numpy.isclose(sparse_grid_table[var_name].values,
                                             sentinel_values[i],
                                             rtol=RELATIVE_TOLERANCE)
        sentinel_flags = numpy.logical_or(sentinel_flags, these_sentinel_flags)

    sentinel_indices = numpy.where(sentinel_flags)[0]
    sparse_grid_table[var_name].values[sentinel_indices] = numpy.nan

    sparse_grid_table.drop(sparse_grid_table.index[sentinel_indices], axis=0,
                           inplace=True)
    return sparse_grid_table


def time_unix_sec_to_spc_date(unix_time_sec):
    """Converts time from Unix format to SPC date.

    :param unix_time_sec: Time in Unix format.
    :return: spc_date_string: SPC date at the given time, in format "yyyymmdd".
    """

    return time.strftime(SPC_DATE_FORMAT,
                         time.gmtime(unix_time_sec - DAYS_TO_SECONDS / 2))


def unzip_1day_tar_file(tar_file_name, spc_date_unix_sec=None,
                        top_target_directory_name=None,
                        var_to_heights_dict_m_agl=None):
    """Unzips tar file with all radar variables for one SPC date.

    :param tar_file_name: Path to input file.
    :param spc_date_unix_sec: SPC date in Unix format.
    :param top_target_directory_name: Top-level output directory.  This method
        will create a subdirectory for the SPC date.
    :param var_to_heights_dict_m_agl: Dictionary with variable-height pairs to
        extract.  For the format of this dictionary, see
        _var_height_arrays_to_dict.
    :return: target_directory_name: Path to output directory.  This will be
        "<top_target_directory_name>/<yyyymmdd>", where <yyyymmdd> is the SPC
        date.
    """

    target_directory_name = '{0:s}/{1:s}'.format(top_target_directory_name,
                                                 time_unix_sec_to_spc_date(
                                                     spc_date_unix_sec))

    unix_command_string = (
        'tar -C "' + target_directory_name + '" -xvf "' + tar_file_name + '"')

    variable_names = var_to_heights_dict_m_agl.keys()
    for j in range(len(variable_names)):
        these_heights_m_agl = var_to_heights_dict_m_agl[variable_names[j]]

        for k in range(len(these_heights_m_agl)):
            this_directory_name = (
                _get_directory_in_tar_file(variable_names[j],
                                           these_heights_m_agl[k]))

            unix_command_string += ' "' + this_directory_name + '"'

    os.system(unix_command_string)
    return target_directory_name


def find_local_raw_file(unix_time_sec=None, spc_date_unix_sec=None,
                        variable_name=None, height_m_agl=None,
                        top_directory_name=None, raise_error_if_missing=True):
    """Finds raw MYRORSS file on local machine.

    This should contain one radar variable at one height and one time step.

    :param unix_time_sec: Time in Unix format.
    :param spc_date_unix_sec: SPC date in Unix format.
    :param variable_name: Variable name in new format (must be in
        `RADAR_VAR_NAMES`).
    :param height_m_agl: Height (metres above ground level).
    :param top_directory_name: Top-level directory for raw MYRORSS files.
    :param raise_error_if_missing: Boolean flag.  If True and file is missing,
        this method will raise an error.
    :return: raw_file_name: File path.  If raise_error_if_missing = False and
        file is missing, this will be the *expected* path.
    :raises: ValueError: if raise_error_if_missing = True and file is missing.
    """

    pathless_file_name = _get_pathless_raw_file_name(unix_time_sec)
    directory_name = '{0:s}/{1:s}/{2:s}'.format(
        top_directory_name, time_unix_sec_to_spc_date(spc_date_unix_sec),
        _get_directory_in_tar_file(variable_name, height_m_agl))

    raw_file_name = '{0:s}/{1:s}'.format(directory_name, pathless_file_name)

    if raise_error_if_missing and not os.path.isfile(raw_file_name):
        raise ValueError(
            'Cannot find raw file.  Expected at location: ' + raw_file_name)

    return raw_file_name


def convert_lng_negative_in_west(input_longitudes_deg):
    """Converts longitudes so that all WH values are negative.

    In other words, all values in western hemisphere are from -180...0 deg E.

    :param input_longitudes_deg: scalar or numpy array of longitudes (deg E).
    :return: output_longitudes_deg: Same as input_longitudes_deg, except that
        all values in western hemisphere are negative.
    """

    was_input_array = isinstance(input_longitudes_deg, collections.Iterable)
    if not was_input_array:
        input_longitudes_deg = numpy.full(1, input_longitudes_deg)

    positive_in_west_flags = input_longitudes_deg > 180
    positive_in_west_indices = numpy.where(positive_in_west_flags)[0]
    input_longitudes_deg[positive_in_west_indices] -= 360
    if was_input_array:
        return input_longitudes_deg

    return input_longitudes_deg[0]


def convert_lng_positive_in_west(input_longitudes_deg):
    """Converts longitudes so that all WH values are positive.

    In other words, all values in western hemisphere are from 180...360 deg E.

    :param input_longitudes_deg: numpy array of longitudes (deg E).
    :return: output_longitudes_deg: Same as input_longitudes_deg, except that
        all values in western hemisphere are positive.
    """

    was_input_array = isinstance(input_longitudes_deg, collections.Iterable)
    if not was_input_array:
        input_longitudes_deg = numpy.full(1, input_longitudes_deg)

    negative_flags = input_longitudes_deg < 0
    negative_indices = numpy.where(negative_flags)[0]
    input_longitudes_deg[negative_indices] += 360
    if was_input_array:
        return input_longitudes_deg

    return input_longitudes_deg[0]


def rowcol_to_latlng(rows, columns, nw_grid_point_lat_deg=None,
                     nw_grid_point_lng_deg=None, lat_spacing_deg=None,
                     lng_spacing_deg=None):
    """Converts MYRORSS coordinates from row-column to lat-long.

    In the following discussion, let P = number of points.

    :param rows: length-P numpy array of rows (increasing from north to south).
    :param columns: length-P numpy array of columns (increasing from west to
        east).
    :param nw_grid_point_lat_deg: Latitude (deg N) at center of
        northwesternmost grid point.
    :param nw_grid_point_lng_deg: Longitude (deg E) at center of
        northwesternmost grid point.
    :param lat_spacing_deg: Spacing (deg N) between adjacent grid rows.
    :param lng_spacing_deg: Spacing (deg E) between adjacent grid columns.
    :return: latitudes_deg: length-P numpy array of latitudes (deg N).
    :return: longitudes_deg: length-P numpy array of longitudes (deg E).
    """

    latitudes_deg = rounder.round_to_nearest(
        nw_grid_point_lat_deg - lat_spacing_deg * rows,
        lat_spacing_deg / 2)

    longitudes_deg = rounder.round_to_nearest(
        nw_grid_point_lng_deg + lng_spacing_deg * columns,
        lng_spacing_deg / 2)

    return latitudes_deg, convert_lng_positive_in_west(longitudes_deg)


def latlng_to_rowcol(latitudes_deg, longitudes_deg, nw_grid_point_lat_deg=None,
                     nw_grid_point_lng_deg=None, lat_spacing_deg=None,
                     lng_spacing_deg=None):
    """Converts MYRORSS coordinates from lat-long to row-column.

    :param latitudes_deg: length-P numpy array of latitudes (deg N).
    :param longitudes_deg: length-P numpy array of longitudes (deg E).
    :param nw_grid_point_lat_deg: Latitude (deg N) at center of
        northwesternmost grid point.
    :param nw_grid_point_lng_deg: Longitude (deg E) at center of
        northwesternmost grid point.
    :param lat_spacing_deg: Spacing (deg N) between adjacent grid rows.
    :param lng_spacing_deg: Spacing (deg E) between adjacent grid columns.
    :return: rows: length-P numpy array of rows (increasing from north to
        south).
    :return: columns: length-P numpy array of columns (increasing from west to
        east).
    """

    columns = rounder.round_to_nearest(
        (longitudes_deg - nw_grid_point_lng_deg) / lng_spacing_deg, 0.5)

    rows = rounder.round_to_nearest(
        (nw_grid_point_lat_deg - latitudes_deg) / lat_spacing_deg, 0.5)

    return rows, columns


def get_center_of_grid(nw_grid_point_lat_deg=None, nw_grid_point_lng_deg=None,
                       lat_spacing_deg=None, lng_spacing_deg=None,
                       num_lat_in_grid=None, num_lng_in_grid=None):
    """Finds center of grid.

    :param nw_grid_point_lat_deg: Latitude (deg N) at center of northwesternmost
        grid point.
    :param nw_grid_point_lng_deg: Longitude (deg E) at center of
        northwesternmost grid point.
    :param lat_spacing_deg: Spacing (deg N) between adjacent grid rows.
    :param lng_spacing_deg: Spacing (deg E) between adjacent grid columns.
    :param num_lat_in_grid: Number of grid rows (unique latitudes).
    :param num_lng_in_grid: Number of grid columns (unique longitudes).
    :return: center_latitude_deg: Latitude (deg N) at center of grid.
    :return: center_longitude_deg: Longitude (deg E) at center of grid.
    """

    min_latitude_deg = nw_grid_point_lat_deg - (
        (num_lat_in_grid - 1) * lat_spacing_deg)

    max_longitude_deg = nw_grid_point_lng_deg + (
        (num_lng_in_grid - 1) * lng_spacing_deg)

    return (numpy.mean(numpy.array([min_latitude_deg, nw_grid_point_lat_deg])),
            numpy.mean(numpy.array([nw_grid_point_lng_deg, max_longitude_deg])))


def read_metadata_from_netcdf(netcdf_file_name):
    """Reads metadata from NetCDF file.

    This file should contain data for one variable at one elevation and one time
    step (e.g., 1-km reflectivity at 1235 UTC).

    :param netcdf_file_name: Path to input file (string).
    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict.nw_grid_point_lat_deg: Latitude (deg N) at center of
        northwesternmost grid point.
    metadata_dict.nw_grid_point_lng_deg: Longitude (deg E) at center of
        northwesternmost grid point.
    metadata_dict.lat_spacing_deg: Spacing (deg N) between adjacent grid rows.
    metadata_dict.lng_spacing_deg: Spacing (deg E) between adjacent grid
        columns.
    metadata_dict.num_lat_in_grid: Number of grid rows (unique latitudes).
    metadata_dict.num_lng_in_grid: Number of grid columns (unique longitudes).
    metadata_dict.elevation_m_agl: Radar elevation (metres above ground level).
    metadata_dict.unix_time_sec: Valid time (seconds since 0000 UTC 1 Jan 1970).
    metadata_dict.var_name: Variable name in new format (e.g.,
        "reflectivity_column_max_dbz" for column-max reflectivity).
    metadata_dict.var_name_orig: Variable name in original format (e.g.,
        "MergedReflectivityQCComposite" for column-max reflectivity).
    metadata_dict.sentinel_values: 1-D numpy array of sentinel values.
    """

    netcdf_dataset = Dataset(netcdf_file_name)
    var_name_orig = str(getattr(netcdf_dataset, VAR_NAME_COLUMN_ORIG))

    metadata_dict = {NW_GRID_POINT_LAT_COLUMN:
                         getattr(netcdf_dataset, NW_GRID_POINT_LAT_COLUMN_ORIG),
                     NW_GRID_POINT_LNG_COLUMN:
                         convert_lng_positive_in_west(
                             getattr(netcdf_dataset,
                                     NW_GRID_POINT_LNG_COLUMN_ORIG)),
                     LAT_SPACING_COLUMN:
                         getattr(netcdf_dataset, LAT_SPACING_COLUMN_ORIG),
                     LNG_SPACING_COLUMN:
                         getattr(netcdf_dataset, LNG_SPACING_COLUMN_ORIG),
                     NUM_LAT_COLUMN:
                         netcdf_dataset.dimensions[NUM_LAT_COLUMN_ORIG].size,
                     NUM_LNG_COLUMN:
                         netcdf_dataset.dimensions[NUM_LNG_COLUMN_ORIG].size,
                     ELEVATION_COLUMN:
                         getattr(netcdf_dataset, ELEVATION_COLUMN_ORIG),
                     UNIX_TIME_COLUMN:
                         getattr(netcdf_dataset, UNIX_TIME_COLUMN_ORIG),
                     VAR_NAME_COLUMN_ORIG: var_name_orig,
                     VAR_NAME_COLUMN: _var_name_orig_to_new(var_name_orig)}

    metadata_dict[NW_GRID_POINT_LAT_COLUMN] = rounder.floor_to_nearest(
        metadata_dict[NW_GRID_POINT_LAT_COLUMN],
        metadata_dict[LAT_SPACING_COLUMN])
    metadata_dict[NW_GRID_POINT_LNG_COLUMN] = rounder.ceiling_to_nearest(
        metadata_dict[NW_GRID_POINT_LNG_COLUMN],
        metadata_dict[LNG_SPACING_COLUMN])

    metadata_dict[NUM_LAT_COLUMN] = int(rounder.round_to_nearest(
        metadata_dict[NUM_LAT_COLUMN], 100)) + 1
    metadata_dict[NUM_LNG_COLUMN] = int(rounder.round_to_nearest(
        metadata_dict[NUM_LNG_COLUMN], 100)) + 1

    sentinel_values = numpy.full(len(SENTINEL_VALUE_COLUMNS_ORIG), numpy.nan)
    for i in range(len(SENTINEL_VALUE_COLUMNS_ORIG)):
        sentinel_values[i] = getattr(netcdf_dataset,
                                     SENTINEL_VALUE_COLUMNS_ORIG[i])

    metadata_dict.update({SENTINEL_VALUE_COLUMN: sentinel_values})
    return metadata_dict


def read_sparse_grid_from_netcdf(netcdf_file_name, var_name_orig,
                                 sentinel_values):
    """Reads sparse grid from NetCDF file.

    This grid should contain one radar variable at one elevation and one time
    step (e.g., 1-km reflectivity at 1235 UTC).

    :param netcdf_file_name: Path to input file (string).
    :param var_name_orig: Variable name in original format (e.g.,
        "MergedReflectivityQCComposite" for column-max reflectivity).
    :param sentinel_values: 1-D numpy array of sentinel values.
    :return: sparse_grid_table: pandas DataFrame with the following columns.
    sparse_grid_table.grid_row: Grid row (increasing from north to south).
    sparse_grid_table.grid_column: Grid column (increasing from west to east).
    sparse_grid_table.<var_name>: Value of radar variable.
    sparse_grid_table.num_grid_cells: Number of grid cells with the same value
        (counting across rows [west to east] first, then down columns [north to
        south]).
    """

    netcdf_dataset = Dataset(netcdf_file_name)
    var_name = _var_name_orig_to_new(var_name_orig)

    sparse_grid_dict = {
        GRID_ROW_COLUMN: netcdf_dataset.variables[GRID_ROW_COLUMN_ORIG][:],
        GRID_COLUMN_COLUMN: netcdf_dataset.variables[GRID_COLUMN_COLUMN_ORIG][
                            :],
        NUM_GRID_CELL_COLUMN: netcdf_dataset.variables[
                                  NUM_GRID_CELL_COLUMN_ORIG][:],
        var_name: netcdf_dataset.variables[var_name_orig][:]}

    sparse_grid_table = pandas.DataFrame.from_dict(sparse_grid_dict)
    return _remove_sentinels(sparse_grid_table, var_name, sentinel_values)
