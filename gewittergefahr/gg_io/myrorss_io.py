"""IO methods for MYRORSS* data.

* MYRORSS = Multi-year Reanalysis of Remotely Sensed Storms
"""

import collections
import numpy
import pandas
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

RADAR_VAR_NAMES = ['echo_top_18dbz_km', 'echo_top_50dbz_km',
                   'low_level_shear_s01', 'mid_level_shear_s01',
                   'reflectivity_dbz', 'reflectivity_column_max_dbz', 'mesh_mm',
                   'reflectivity_0celsius_dbz', 'reflectivity_m10celsius_dbz',
                   'reflectivity_m20celsius_dbz',
                   'reflectivity_lowest_altitude_dbz', 'shi', 'vil_mm',
                   'storm_id']
RADAR_VAR_NAMES_ORIG = ['EchoTop_18', 'EchoTop_50', 'MergedLLShear',
                        'MergedMLShear', 'MergedReflectivityQC',
                        'MergedReflectivityQCComposite', 'MESH',
                        'Reflectivity_0C', 'Reflectivity_-10C',
                        'Reflectivity_-20C', 'ReflectivityAtLowestAltitude',
                        'SHI', 'VIL', 'ClusterID']

RELATIVE_TOLERANCE = 1e-6


def _convert_var_name(var_name_orig):
    """Converts variable from original to new format.

    "Original format" = MYRORSS format; new format = my format.

    :param var_name_orig: Variable name in original format.
    :return: var_name: Variable name in new format.
    """

    orig_var_flags = [s == var_name_orig for s in RADAR_VAR_NAMES_ORIG]
    orig_var_index = numpy.where(orig_var_flags)[0][0]
    return RADAR_VAR_NAMES[orig_var_index]


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
                     VAR_NAME_COLUMN: _convert_var_name(var_name_orig)}

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
    var_name = _convert_var_name(var_name_orig)

    sparse_grid_dict = {
        GRID_ROW_COLUMN: netcdf_dataset.variables[GRID_ROW_COLUMN_ORIG][:],
        GRID_COLUMN_COLUMN: netcdf_dataset.variables[GRID_COLUMN_COLUMN_ORIG][
                            :],
        NUM_GRID_CELL_COLUMN: netcdf_dataset.variables[
                                  NUM_GRID_CELL_COLUMN_ORIG][:],
        var_name: netcdf_dataset.variables[var_name_orig][:]}

    sparse_grid_table = pandas.DataFrame.from_dict(sparse_grid_dict)
    return _remove_sentinels(sparse_grid_table, var_name, sentinel_values)
