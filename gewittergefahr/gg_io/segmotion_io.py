"""IO methods for segmotion* output.

* segmotion, or w2segmotionll, is a storm-tracking algorithm in the WDSS-II
(Warning Decision Support System -- Integrated Information) software package.
"""

import pandas
import pickle
import time
import os
import calendar
import xml.etree.ElementTree as ElementTree
import numpy
from netCDF4 import Dataset
from gewittergefahr.gg_io import myrorss_io
from gewittergefahr.gg_io import myrorss_sparse_to_full as sparse_to_full
from gewittergefahr.gg_utils import polygons

# TODO(thunderhoser): add error-checking to all methods.
# TODO(thunderhoser): replace main method with named high-level method.

XML_FILE_NAME = (
    '/localdata/ryan.lagerquist/software/matlab/wdssii/raw_files/segmotion/'
    'smooth02_30dBZ/20040811/TrackingTable/0050.00/2004-08-11-124818_'
    'TrackingTable_0050.00.xml')

NETCDF_FILE_NAME = (
    '/localdata/ryan.lagerquist/software/matlab/wdssii/raw_files/segmotion/'
    'smooth02_30dBZ/20040811/ClusterID/0050.00/20040811-124818.netcdf')

GZIP_FILE_EXTENSION = '.gz'

TIME_FORMAT_SEGMOTION = '%Y%m%d-%H%M%S'
TIME_FORMAT_DEFAULT = '%Y-%m-%d-%H%M%S'
SENTINEL_VALUE = -9999

STORM_ID_COLUMN = 'storm_id'
EAST_VELOCITY_COLUMN = 'east_velocity_m_s01'
NORTH_VELOCITY_COLUMN = 'north_velocity_m_s01'
START_TIME_COLUMN = 'start_time_unix_sec'
AGE_COLUMN = 'age_sec'

STORM_ID_COLUMN_ORIG = 'RowName'
EAST_VELOCITY_COLUMN_ORIG = 'MotionEast'
NORTH_VELOCITY_COLUMN_ORIG = 'MotionSouth'
START_TIME_COLUMN_ORIG = 'StartTime'
AGE_COLUMN_ORIG = 'Age'

XML_COLUMN_NAMES = [STORM_ID_COLUMN, EAST_VELOCITY_COLUMN,
                    NORTH_VELOCITY_COLUMN, START_TIME_COLUMN, AGE_COLUMN]
XML_COLUMN_NAMES_ORIG = [STORM_ID_COLUMN_ORIG, EAST_VELOCITY_COLUMN_ORIG,
                         NORTH_VELOCITY_COLUMN_ORIG, START_TIME_COLUMN_ORIG,
                         AGE_COLUMN_ORIG]

GRID_POINT_LAT_COLUMN = 'grid_point_latitudes_deg'
GRID_POINT_LNG_COLUMN = 'grid_point_longitudes_deg'
GRID_POINT_ROW_COLUMN = 'grid_point_rows'
GRID_POINT_COLUMN_COLUMN = 'grid_point_columns'

VERTEX_LAT_COLUMN = 'vertex_latitudes_deg'
VERTEX_LNG_COLUMN = 'vertex_longitudes_deg'
VERTEX_ROW_COLUMN = 'vertex_rows'
VERTEX_COLUMN_COLUMN = 'vertex_columns'


def _convert_xml_column_name(column_name_orig):
    """Converts name of XML column from original to new format.

    "Original format" = format in XML file; new format = my format.

    :param column_name_orig: Column name in original format.
    :return: column_name: Column name in new format.
    """

    orig_column_flags = [c == column_name_orig for c in XML_COLUMN_NAMES_ORIG]
    orig_column_index = numpy.where(orig_column_flags)[0][0]
    return XML_COLUMN_NAMES[orig_column_index]


def _time_string_to_unix_sec(time_string):
    """Converts time from string to Unix format (sec since 0000 UTC 1 Jan 1970).

    :param time_string: Time string (format "yyyymmdd-HHMMSS").
    :return: unix_time_sec: Time in Unix format.
    """

    return calendar.timegm(time.strptime(time_string, TIME_FORMAT_SEGMOTION))


def _remove_rows_with_nan(input_table):
    """Removes all rows with at least one NaN from pandas DataFrame.

    :param input_table: pandas DataFrame, which may contain NaN's.
    :return: output_table: Same as input_table, but without NaN's.
    """

    return input_table.loc[input_table.notnull().all(axis=1)]


def _storm_id_matrix_to_coord_lists(numeric_storm_id_matrix,
                                    unique_center_lat_deg,
                                    unique_center_lng_deg):
    """Converts matrix of storm IDs to one coordinate list for each storm cell.

    The coordinate list for each storm cell is a list of grid points inside the
    storm.

    M = number of grid-point latitudes (unique latitudes at centers of grid
        cells)
    N = number of grid-point longitudes (unique longitudes at centers of grid
        cells)
    P = number of grid points in storm cell (different for each storm cell)

    :param numeric_storm_id_matrix: M-by-N numpy array with numeric storm IDs.
    :param unique_center_lat_deg: length-M numpy array of unique latitudes
        (deg N).  If unique_center_lat_deg is increasing (decreasing), latitudes
        in numeric_storm_id_matrix increase (decrease) while going down a
        column.
    :param unique_center_lng_deg: length-N numpy array of unique longitudes
        (deg E).  If unique_center_lng_deg is increasing (decreasing),
        longitudes in numeric_storm_id_matrix increase (decrease) while going
        right across a row.
    :return: polygon_table: pandas DataFrame with the following columns.
    polygon_table.storm_id: String ID for storm cell.
    polygon_table.grid_point_latitudes_deg: length-P numpy array with latitudes
        (deg N) of grid points in storm cell.
    polygon_table.grid_point_longitudes_deg: length-P numpy array with
        longitudes (deg E) of grid points in storm cell.
    polygon_table.grid_point_rows: length-P numpy array with row indices (all
        integers) of grid points in storm cell.
    polygon_table.grid_point_columns: length-P numpy array with column indices
        (all integers) of grid points in storm cell.
    """

    numeric_storm_id_matrix[
        numpy.isnan(numeric_storm_id_matrix)] = SENTINEL_VALUE
    unique_numeric_storm_ids, indices_orig_to_unique = numpy.unique(
        numeric_storm_id_matrix, return_inverse=True)

    unique_storm_ids = [str(int(id)) for id in unique_numeric_storm_ids]
    polygon_dict = {STORM_ID_COLUMN: unique_storm_ids}
    polygon_table = pandas.DataFrame.from_dict(polygon_dict)

    nested_array = polygon_table[
        [STORM_ID_COLUMN, STORM_ID_COLUMN]].values.tolist()
    argument_dict = {GRID_POINT_LAT_COLUMN: nested_array,
                     GRID_POINT_LNG_COLUMN: nested_array,
                     GRID_POINT_ROW_COLUMN: nested_array,
                     GRID_POINT_COLUMN_COLUMN: nested_array}
    polygon_table = polygon_table.assign(**argument_dict)

    num_storms = len(unique_numeric_storm_ids)
    num_lat_in_grid = len(unique_center_lat_deg)
    num_lng_in_grid = len(unique_center_lng_deg)

    for i in range(num_storms):
        if unique_numeric_storm_ids[i] == SENTINEL_VALUE:
            continue

        this_storm_linear_indices = numpy.where(indices_orig_to_unique == i)[0]
        (this_storm_row_indices,
         this_storm_column_indices) = numpy.unravel_index(
            this_storm_linear_indices, (num_lat_in_grid, num_lng_in_grid))

        polygon_table[GRID_POINT_ROW_COLUMN].values[i] = this_storm_row_indices
        polygon_table[GRID_POINT_COLUMN_COLUMN].values[
            i] = this_storm_column_indices
        polygon_table[GRID_POINT_LAT_COLUMN].values[i] = unique_center_lat_deg[
            this_storm_row_indices]
        polygon_table[GRID_POINT_LNG_COLUMN].values[i] = unique_center_lng_deg[
            this_storm_column_indices]

    return polygon_table.loc[
        polygon_table[STORM_ID_COLUMN] != str(int(SENTINEL_VALUE))]


def extract_file_from_gzip(gzip_file_name):
    """Extracts single file from gzip archive.
    
    :param gzip_file_name: Path to input file.
    :return: unzipped_file_name: Path to unzipped file.  Same as
        `gzip_file_name`, except with ".gz" removed from the end.
    :raises: ValueError: gzip_file_name does not end with ".gz".
    """

    if not gzip_file_name.endswith(GZIP_FILE_EXTENSION):
        err_string = (
            'gzip file (' + gzip_file_name + ') does not end with "' +
            GZIP_FILE_EXTENSION + '".  Cannot generate name for unzipped file.')
        raise ValueError(err_string)

    unzipped_file_name = gzip_file_name[:-len(GZIP_FILE_EXTENSION)]
    unix_command_str = 'gunzip -v -c {0:s} > {1:s}'.format(gzip_file_name,
                                                           unzipped_file_name)
    os.system(unix_command_str)

    return unzipped_file_name


def read_stats_from_xml(xml_file_name):
    """Reads storm statistics from XML file.

    :param xml_file_name: Path to input file.
    :return: stats_table: pandas DataFrame with the following columns.
    stats_table.storm_id: String ID for storm cell.
    stats_table.east_velocity_m_s01: Eastward velocity (m/s).
    stats_table.north_velocity_m_s01: Northward velocity (m/s).
    stats_table.start_time_unix_sec: Start time of storm cell (seconds since
        0000 UTC 1 Jan 1970).
    stats_table.age_sec: Age of storm cell (seconds).
    """

    xml_tree = ElementTree.parse(xml_file_name)

    storm_dict = {}
    this_column_name = None
    this_column_name_orig = None
    this_column_values = None

    for this_element in xml_tree.iter():
        if this_element.tag == 'datacolumn':
            if this_column_name_orig in XML_COLUMN_NAMES_ORIG:
                storm_dict.update({this_column_name: this_column_values})

            this_column_name_orig = this_element.attrib['name']
            if this_column_name_orig in XML_COLUMN_NAMES_ORIG:
                this_column_name = _convert_xml_column_name(
                    this_column_name_orig)
                this_column_values = []

            continue

        if this_column_name_orig not in XML_COLUMN_NAMES_ORIG:
            continue

        if this_column_name == STORM_ID_COLUMN:
            this_column_values.append(this_element.attrib['value'])
        elif this_column_name == NORTH_VELOCITY_COLUMN:
            this_column_values.append(-1 * float(this_element.attrib['value']))
        elif this_column_name == EAST_VELOCITY_COLUMN:
            this_column_values.append(float(this_element.attrib['value']))
        elif this_column_name == AGE_COLUMN:
            this_column_values.append(
                int(numpy.round(float(this_element.attrib['value']))))
        elif this_column_name == START_TIME_COLUMN:
            this_column_values.append(
                _time_string_to_unix_sec(this_element.attrib['value']))

    stats_table = pandas.DataFrame.from_dict(storm_dict)
    return _remove_rows_with_nan(stats_table)


def read_polygons_from_netcdf(netcdf_file_name, metadata_dict):
    """Reads storm polygons (outlines of storm cells) from NetCDF file.

    P = number of grid points in storm cell (different for each storm cell)
    V = number of vertices in storm polygon (different for each storm cell)

    :param netcdf_file_name: Path to input file.
    :param metadata_dict: Dictionary with metadata from NetCDF file, in format
        produced by `myrorss_io.read_metadata_from_netcdf`.
    :return: polygon_table: pandas DataFrame with the following columns.
    polygon_table.storm_id: String ID for storm cell.
    polygon_table.grid_point_latitudes_deg: length-P numpy array with latitudes
        (deg N) of grid points in storm cell.
    polygon_table.grid_point_longitudes_deg: length-P numpy array with
        longitudes (deg E) of grid points in storm cell.
    polygon_table.grid_point_rows: length-P numpy array with row indices (all
        integers) of grid points in storm cell.
    polygon_table.grid_point_columns: length-P numpy array with column indices
        (all integers) of grid points in storm cell.
    polygon_table.vertex_latitudes_deg: length-V numpy array with latitudes (deg
        N) of vertices in storm polygon.
    polygon_table.vertex_longitudes_deg: length-V numpy array with longitudes
        (deg E) of vertices in storm polygon.
    polygon_table.vertex_rows: length-V numpy array with row indices (all half-
        integers) of vertices in storm polygon.
    polygon_table.vertex_columns: length-V numpy array with column indices (all
        half-integers) of vertices in storm polygon.
    """

    netcdf_dataset = Dataset(netcdf_file_name)
    storm_id_var_name = metadata_dict[myrorss_io.VAR_NAME_COLUMN]
    storm_id_var_name_orig = metadata_dict[myrorss_io.VAR_NAME_COLUMN_ORIG]

    sparse_grid_dict = {
        myrorss_io.GRID_ROW_COLUMN:
            netcdf_dataset.variables[myrorss_io.GRID_ROW_COLUMN_ORIG][:],
        myrorss_io.GRID_COLUMN_COLUMN:
            netcdf_dataset.variables[myrorss_io.GRID_COLUMN_COLUMN_ORIG][:],
        myrorss_io.NUM_GRID_CELL_COLUMN:
            netcdf_dataset.variables[myrorss_io.NUM_GRID_CELL_COLUMN_ORIG][:],
        storm_id_var_name: netcdf_dataset.variables[storm_id_var_name_orig][:]}

    sparse_grid_table = pandas.DataFrame.from_dict(sparse_grid_dict)
    (numeric_storm_id_matrix, unique_center_lat_deg,
     unique_center_lng_deg) = sparse_to_full.sparse_to_full_grid_wrapper(
        sparse_grid_table, metadata_dict)

    polygon_table = _storm_id_matrix_to_coord_lists(numeric_storm_id_matrix,
                                                    unique_center_lat_deg,
                                                    unique_center_lng_deg)

    nested_array = polygon_table[
        [STORM_ID_COLUMN, STORM_ID_COLUMN]].values.tolist()
    argument_dict = {VERTEX_LAT_COLUMN: nested_array,
                     VERTEX_LNG_COLUMN: nested_array,
                     VERTEX_ROW_COLUMN: nested_array,
                     VERTEX_COLUMN_COLUMN: nested_array}
    polygon_table = polygon_table.assign(**argument_dict)

    num_storms = len(polygon_table.index)
    for i in range(num_storms):
        (polygon_table[VERTEX_ROW_COLUMN].values[i],
         polygon_table[VERTEX_COLUMN_COLUMN].values[
             i]) = polygons.points_in_poly_to_vertices(
            polygon_table[GRID_POINT_ROW_COLUMN].values[i],
            polygon_table[GRID_POINT_COLUMN_COLUMN].values[i])

        (polygon_table[VERTEX_LAT_COLUMN].values[i],
         polygon_table[VERTEX_LNG_COLUMN].values[
             i]) = myrorss_io.rowcol_to_latlng(
            polygon_table[VERTEX_ROW_COLUMN].values[i],
            polygon_table[VERTEX_COLUMN_COLUMN].values[
                i], nw_grid_point_lat_deg=metadata_dict[
                myrorss_io.NW_GRID_POINT_LAT_COLUMN],
            nw_grid_point_lng_deg=metadata_dict[
                myrorss_io.NW_GRID_POINT_LNG_COLUMN],
            lat_spacing_deg=metadata_dict[
                myrorss_io.LAT_SPACING_COLUMN],
            lng_spacing_deg=metadata_dict[
                myrorss_io.LNG_SPACING_COLUMN])

    return polygon_table


def join_stats_and_polygons(stats_table, polygon_table):
    """Joins tables with storm statistics and polygons.

    :param stats_table: pandas DataFrame created by read_stats_from_xml.
    :param polygon_table: pandas DataFrame created by read_polygons_from_netcdf.
    :return: storm_table: pandas DataFrame with columns from both stats_table
        and polygon_table.
    """

    return polygon_table.merge(stats_table, on=STORM_ID_COLUMN, how='inner')


def write_stats_and_polygons_to_pickle(storm_table, pickle_file_name):
    """Writes storm statistics and polygons to Pickle file.

    :param storm_table: pandas DataFrame created by join_stats_and_polygons.
    :param pickle_file_name: Path to output file.
    """

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(storm_table, pickle_file_handle)
    pickle_file_handle.close()


def read_stats_and_polygons_from_pickle(pickle_file_name):
    """Reads storm statistics and polygons from Pickle file.

    :param pickle_file_name: Path to input file (should be written by
        write_stats_and_polygons_to_pickle).
    :return: storm_table: pandas DataFrame with columns produced by
        join_stats_and_polygons.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    storm_table = pickle.load(pickle_file_handle)
    pickle_file_handle.close()
    return storm_table


if __name__ == '__main__':
    stats_table = read_stats_from_xml(XML_FILE_NAME)
    print stats_table

    metadata_dict = myrorss_io.read_metadata_from_netcdf(NETCDF_FILE_NAME)
    polygon_table = read_polygons_from_netcdf(NETCDF_FILE_NAME, metadata_dict)
    print polygon_table

    storm_table = join_stats_and_polygons(stats_table, polygon_table)
    print storm_table
