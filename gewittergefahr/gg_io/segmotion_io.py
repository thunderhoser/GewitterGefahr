"""IO methods for segmotion output.

DEFINITIONS

segmotion (or w2segmotionll) = storm-tracking algorithm in WDSS-II.

WDSS-II = Warning Decision Support System -- Integrated Information, a software
package for the visualization and analysis of thunderstorm-related data.

SPC = Storm Prediction Center

SPC date = a 24-hour period, starting and ending at 1200 UTC.  This is unlike a
normal person's day, which starts and ends at 0000 UTC.  An SPC date is
referenced by the calendar day at the beginning of the SPC date.  In other
words, SPC date "Sep 23 2017" actually runs from 1200 UTC 23 Sep 2017 -
1200 UTC 24 Sep 2017.
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
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import unzipping

# TODO(thunderhoser): add error-checking to all methods.
# TODO(thunderhoser): replace main method with named high-level method.

MIN_BUFFER_DISTS_METRES = numpy.array([numpy.nan, 0., 5000.])
MAX_BUFFER_DISTS_METRES = numpy.array([0., 5000., 10000.])

XML_FILE_NAME = (
    '/localdata/ryan.lagerquist/software/matlab/wdssii/raw_files/segmotion/'
    'smooth02_30dBZ/20040811/TrackingTable/0050.00/2004-08-11-124818_'
    'TrackingTable_0050.00.xml')

NETCDF_FILE_NAME = (
    '/localdata/ryan.lagerquist/software/matlab/wdssii/raw_files/segmotion/'
    'smooth02_30dBZ/20040811/ClusterID/0050.00/20040811-124818.netcdf')

ZIPPED_FILE_EXTENSION = '.gz'
STATS_FILE_EXTENSION = '.xml'
POLYGON_FILE_EXTENSION = '.netcdf'

STATS_DIR_NAME_PART = 'TrackingTable'
POLYGON_DIR_NAME_PART = 'ClusterID'

TIME_FORMAT_SEGMOTION = '%Y%m%d-%H%M%S'
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

BUFFER_VERTEX_LAT_COLUMN_PREFIX = 'vertex_latitudes_deg_buffer_'
BUFFER_VERTEX_LNG_COLUMN_PREFIX = 'vertex_longitudes_deg_buffer_'


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


def _time_unix_sec_to_string(unix_time_sec):
    """Converts time from Unix format to string.

    :param unix_time_sec: Time in Unix format.
    :return: time_string: Time string (format "yyyymmdd-HHMMSS").
    """

    return time.strftime(TIME_FORMAT_SEGMOTION, time.gmtime(unix_time_sec))


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


def _distance_buffers_to_column_names(min_buffer_dists_metres,
                                      max_buffer_dists_metres):
    """Generates column name for each distance buffer.

    N = number of distance buffers

    :param min_buffer_dists_metres: length-N numpy array of minimum distances
        (integers).
    :param max_buffer_dists_metres: length-N numpy array of maximum distances
        (integers).
    :return: buffer_lat_column_names: length-N list of column names for vertex
        latitudes.
    :return: buffer_lng_column_names: length-N list of column names for vertex
        longitudes.
    """

    num_buffers = len(max_buffer_dists_metres)
    buffer_lat_column_names = [''] * num_buffers
    buffer_lng_column_names = [''] * num_buffers

    for j in range(num_buffers):
        if numpy.isnan(min_buffer_dists_metres[j]):
            buffer_lat_column_names[j] = '{0:s}{1:d}m'.format(
                BUFFER_VERTEX_LAT_COLUMN_PREFIX,
                int(max_buffer_dists_metres[j]))
            buffer_lng_column_names[j] = '{0:s}{1:d}m'.format(
                BUFFER_VERTEX_LNG_COLUMN_PREFIX,
                int(max_buffer_dists_metres[j]))
        else:
            buffer_lat_column_names[j] = '{0:s}{1:d}_{2:d}m'.format(
                BUFFER_VERTEX_LAT_COLUMN_PREFIX,
                int(min_buffer_dists_metres[j]),
                int(max_buffer_dists_metres[j]))
            buffer_lng_column_names[j] = '{0:s}{1:d}_{2:d}m'.format(
                BUFFER_VERTEX_LNG_COLUMN_PREFIX,
                int(min_buffer_dists_metres[j]),
                int(max_buffer_dists_metres[j]))

    return buffer_lat_column_names, buffer_lng_column_names


def _get_pathless_stats_file_name(unix_time_sec, zipped=True):
    """Generates pathless name for statistics file.

    This file should contain storm stats (everything except polygons) for one
    time step and one tracking scale.

    :param unix_time_sec: Time in Unix format.
    :param zipped: Boolean flag.  If True, will generate name for zipped file.
        If False, will generate name for unzipped file.
    :return: pathless_stats_file_name: Pathless name for statistics file.
    """

    if zipped:
        return '{0:s}{1:s}{2:s}'.format(
            _time_unix_sec_to_string(unix_time_sec), STATS_FILE_EXTENSION,
            ZIPPED_FILE_EXTENSION)
    else:
        return '{0:s}{1:s}'.format(
            _time_unix_sec_to_string(unix_time_sec), STATS_FILE_EXTENSION)


def _get_pathless_polygon_file_name(unix_time_sec, zipped=True):
    """Generates pathless name for polygon file.

    This file should contain storm outlines (polygons) for one time step and one
    tracking scale.

    :param unix_time_sec: Time in Unix format.
    :param zipped: Boolean flag.  If True, will generate name for zipped file.
        If False, will generate name for unzipped file.
    :return: pathless_polygon_file_name: Pathless name for polygon file.
    """

    if zipped:
        return '{0:s}{1:s}{2:s}'.format(
            _time_unix_sec_to_string(unix_time_sec), POLYGON_FILE_EXTENSION,
            ZIPPED_FILE_EXTENSION)
    else:
        return '{0:s}{1:s}'.format(
            _time_unix_sec_to_string(unix_time_sec), POLYGON_FILE_EXTENSION)


def _get_relative_stats_dir_ordinal_scale(spc_date_string,
                                          tracking_scale_ordinal):
    """Generates expected relative path for stats directory.
    
    This directory should contain storm statistics for the given tracking scale.

    N = number of tracking scales
    
    :param spc_date_string: SPC date in format "yyyymmdd".
    :param tracking_scale_ordinal: Tracking scale (must be ordinal number in
        [0, N - 1]).
    :return: stats_directory_name: Expected relative path for stats directory.
    """

    return '{0:s}/{1:s}/scale_{2:d}'.format(
        spc_date_string, STATS_DIR_NAME_PART, tracking_scale_ordinal)


def _get_relative_stats_dir_physical_scale(spc_date_string,
                                           tracking_scale_metres2):
    """Generates expected relative path for stats directory.
    
    This directory should contain storm statistics for the given tracking scale.

    N = number of tracking scales
    
    :param spc_date_string: SPC date in format "yyyymmdd".
    :param tracking_scale_metres2: Tracking scale.
    :return: stats_directory_name: Expected relative path for stats directory.
    """

    return '{0:s}/{1:s}/scale_{2:d}m2'.format(
        spc_date_string, STATS_DIR_NAME_PART, int(tracking_scale_metres2))


def _get_relative_polygon_dir_ordinal_scale(spc_date_string,
                                            tracking_scale_ordinal):
    """Generates expected relative path for polygon directory.
    
    This directory should contain storm outlines (polygons) for the given
    tracking scale.

    N = number of tracking scales
    
    :param spc_date_string: SPC date in format "yyyymmdd".
    :param tracking_scale_ordinal: Tracking scale (must be ordinal number in
        [0, N - 1]).
    :return: polygon_directory_name: Expected relative path for polygon
        directory.
    """

    return '{0:s}/{1:s}/scale_{2:d}'.format(
        spc_date_string, POLYGON_DIR_NAME_PART, tracking_scale_ordinal)


def _get_relative_polygon_dir_physical_scale(spc_date_string,
                                             tracking_scale_metres2):
    """Generates expected relative path for polygon directory.
    
    This directory should contain storm outlines (polygons) for the given
    tracking scale.

    N = number of tracking scales
    
    :param spc_date_string: SPC date in format "yyyymmdd".
    :param tracking_scale_metres2: Tracking scale.
    :return: polygon_directory_name: Expected relative path for polygon
        directory.
    """

    return '{0:s}/{1:s}/scale_{2:d}m2'.format(
        spc_date_string, POLYGON_DIR_NAME_PART, int(tracking_scale_metres2))


def _rename_raw_dirs_ordinal_to_physical(top_raw_directory_name=None,
                                         spc_date_string=None,
                                         tracking_scales_ordinal=None,
                                         tracking_scales_metres2=None):
    """Renames dirs by changing tracking scale from ordinal number to m^2.

    Each raw directory should contain either stats or polygon files for one
    tracking scale and one SPC date.  These directories exist inside the 1-day
    tar files and are extracted by unzip_1day_tar_file.

    N = number of tracking scales

    :param top_raw_directory_name: Top-level directory for raw (polygon and
        stats) files.
    :param spc_date_string: SPC date in format "yyyymmdd".
    :param tracking_scales_ordinal: length-N numpy array of tracking scales.
        Each element must be an ordinal number in [0, N - 1].
    :param tracking_scales_metres2: length-N numpy array of tracking scales
        (m^2).
    """

    num_scales = len(tracking_scales_metres2)
    for j in range(num_scales):
        orig_stats_dir_name = '{0:s}/{1:s}'.format(
            top_raw_directory_name, _get_relative_stats_dir_ordinal_scale(
                spc_date_string, tracking_scales_ordinal[j]))
        new_stats_dir_name = '{0:s}/{1:s}'.format(
            top_raw_directory_name, _get_relative_stats_dir_physical_scale(
                spc_date_string, tracking_scales_metres2[j]))
        os.rename(orig_stats_dir_name, new_stats_dir_name)

        orig_polygon_dir_name = '{0:s}/{1:s}'.format(
            top_raw_directory_name, _get_relative_polygon_dir_ordinal_scale(
                spc_date_string, tracking_scales_ordinal[j]))
        new_polygon_dir_name = '{0:s}/{1:s}'.format(
            top_raw_directory_name, _get_relative_polygon_dir_physical_scale(
                spc_date_string, tracking_scales_metres2[j]))
        os.rename(orig_polygon_dir_name, new_polygon_dir_name)


def unzip_1day_tar_file(tar_file_name, spc_date_unix_sec=None,
                        top_target_directory_name=None,
                        scales_to_extract_ordinal=None,
                        scales_to_extract_metres2=None):
    """Unzips tar file with all segmotion output for one SPC date.

    N = number of tracking scales in tar file
    n = number of scales to extract

    :param tar_file_name: Path to input file.
    :param spc_date_unix_sec: SPC date in Unix format.
    :param top_target_directory_name: Top-level output directory.  This method
        will create a subdirectory for the SPC date.
    :param scales_to_extract_ordinal: length-n numpy array of tracking scales to
        extract.  Each array element must be an ordinal number in [0, N - 1].
    :param scales_to_extract_metres2: length-n numpy array of tracking scales to
        extract (m^2).
    :return: target_directory_name: Path to output directory.  This will be
        "<top_target_directory_name>/<yyyymmdd>", where <yyyymmdd> is the SPC
        date.
    """

    spc_date_string = myrorss_io.time_unix_sec_to_spc_date(spc_date_unix_sec)

    num_scales_to_extract = len(scales_to_extract_ordinal)
    directory_names_to_unzip = []
    for j in range(num_scales_to_extract):
        directory_names_to_unzip.append(_get_relative_stats_dir_ordinal_scale(
            spc_date_string, scales_to_extract_ordinal[j]))
        directory_names_to_unzip.append(_get_relative_polygon_dir_ordinal_scale(
            spc_date_string, scales_to_extract_ordinal[j]))

    unzipping.unzip_tar(tar_file_name,
                        target_directory_name=top_target_directory_name,
                        file_and_dir_names_to_unzip=directory_names_to_unzip)

    target_directory_name = '{0:s}/{1:s}'.format(top_target_directory_name,
                                                 spc_date_string)

    _rename_raw_dirs_ordinal_to_physical(
        top_raw_directory_name=top_target_directory_name,
        spc_date_string=spc_date_string,
        tracking_scales_ordinal=scales_to_extract_ordinal,
        tracking_scales_metres2=scales_to_extract_metres2)
    return target_directory_name


def find_local_stats_file(unix_time_sec=None, spc_date_unix_sec=None,
                          top_raw_directory_name=None,
                          tracking_scale_metres2=None, zipped=True,
                          raise_error_if_missing=True):
    """Finds statistics file on local machine.

    This file should contain storm stats (everything except polygons) for one
    time step and one tracking scale.

    :param unix_time_sec: Time in Unix format.
    :param spc_date_unix_sec: SPC date in Unix format.
    :param top_raw_directory_name: Top-level directory for raw segmotion files.
    :param tracking_scale_metres2: Tracking scale.
    :param zipped: Boolean flag.  If True, will look for zipped file.  If False,
        will look for unzipped file.
    :param raise_error_if_missing: Boolean flag.  If True and file is missing,
        this method will raise an error.
    :return: stats_file_name: File path.  If raise_error_if_missing = False and
        file is missing, this will be the *expected* path.
    :raises: ValueError: if raise_error_if_missing = True and file is missing.
    """

    pathless_file_name = _get_pathless_stats_file_name(unix_time_sec,
                                                       zipped=zipped)
    spc_date_string = myrorss_io.time_unix_sec_to_spc_date(spc_date_unix_sec)

    directory_name = '{0:s}/{1:s}'.format(
        top_raw_directory_name,
        _get_relative_stats_dir_physical_scale(spc_date_string,
                                               tracking_scale_metres2))
    stats_file_name = '{0:s}/{1:s}'.format(directory_name, pathless_file_name)

    if raise_error_if_missing and not os.path.isfile(stats_file_name):
        raise ValueError(
            'Cannot find storm-statistics file.  Expected at location: ' +
            stats_file_name)

    return stats_file_name


def find_local_polygon_file(unix_time_sec=None, spc_date_unix_sec=None,
                            top_raw_directory_name=None,
                            tracking_scale_metres2=None, zipped=True,
                            raise_error_if_missing=True):
    """Finds polygon file on local machine.

    This file should contain storm outlines (polygons) for one time step and one
    tracking scale.

    :param unix_time_sec: See documentation for find_local_stats_file.
    :param spc_date_unix_sec: See documentation for find_local_stats_file.
    :param top_raw_directory_name: See documentation for find_local_stats_file.
    :param tracking_scale_metres2: See documentation for find_local_stats_file.
    :param zipped: See documentation for find_local_stats_file.
    :param raise_error_if_missing: See documentation for find_local_stats_file.
    :return: polygon_file_name: File path.  If raise_error_if_missing = False
        and file is missing, this will be the *expected* path.
    :raises: ValueError: if raise_error_if_missing = True and file is missing.
    """

    pathless_file_name = _get_pathless_polygon_file_name(unix_time_sec,
                                                         zipped=zipped)
    spc_date_string = myrorss_io.time_unix_sec_to_spc_date(spc_date_unix_sec)

    directory_name = '{0:s}/{1:s}'.format(
        top_raw_directory_name,
        _get_relative_polygon_dir_physical_scale(spc_date_string,
                                                 tracking_scale_metres2))
    polygon_file_name = '{0:s}/{1:s}'.format(directory_name, pathless_file_name)

    if raise_error_if_missing and not os.path.isfile(polygon_file_name):
        raise ValueError(
            'Cannot find polygon file.  Expected at location: ' +
            polygon_file_name)

    return polygon_file_name


def extract_stats_file_from_gzip(unix_time_sec=None, spc_date_unix_sec=None,
                                 top_raw_directory_name=None,
                                 tracking_scale_metres2=None):
    """Extracts statistics file from gzip archive.

    :param unix_time_sec: See documentation for find_local_stats_file.
    :param spc_date_unix_sec: See documentation for find_local_stats_file.
    :param top_raw_directory_name: See documentation for find_local_stats_file.
    :param tracking_scale_metres2: See documentation for find_local_stats_file.
    :return: stats_file_name: Path to extracted file.
    """

    gzip_file_name = find_local_stats_file(
        unix_time_sec=unix_time_sec, spc_date_unix_sec=spc_date_unix_sec,
        top_raw_directory_name=top_raw_directory_name,
        tracking_scale_metres2=tracking_scale_metres2, zipped=True,
        raise_error_if_missing=True)

    stats_file_name = find_local_stats_file(
        unix_time_sec=unix_time_sec, spc_date_unix_sec=spc_date_unix_sec,
        top_raw_directory_name=top_raw_directory_name,
        tracking_scale_metres2=tracking_scale_metres2, zipped=False,
        raise_error_if_missing=False)

    unzipping.unzip_gzip(gzip_file_name, stats_file_name)
    return stats_file_name


def extract_polygon_file_from_gzip(unix_time_sec=None, spc_date_unix_sec=None,
                                   top_raw_directory_name=None,
                                   tracking_scale_metres2=None):
    """Extracts polygon file from gzip archive.

    :param unix_time_sec: See documentation for find_local_stats_file.
    :param spc_date_unix_sec: See documentation for find_local_stats_file.
    :param top_raw_directory_name: See documentation for find_local_stats_file.
    :param tracking_scale_metres2: See documentation for find_local_stats_file.
    :return: polygon_file_name: Path to extracted file.
    """

    gzip_file_name = find_local_polygon_file(
        unix_time_sec=unix_time_sec, spc_date_unix_sec=spc_date_unix_sec,
        top_raw_directory_name=top_raw_directory_name,
        tracking_scale_metres2=tracking_scale_metres2, zipped=True,
        raise_error_if_missing=True)

    polygon_file_name = find_local_polygon_file(
        unix_time_sec=unix_time_sec, spc_date_unix_sec=spc_date_unix_sec,
        top_raw_directory_name=top_raw_directory_name,
        tracking_scale_metres2=tracking_scale_metres2, zipped=False,
        raise_error_if_missing=False)

    unzipping.unzip_gzip(gzip_file_name, polygon_file_name)
    return polygon_file_name


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


def make_buffers_around_polygons(polygon_table, min_buffer_dists_metres=None,
                                 max_buffer_dists_metres=None,
                                 central_latitude_deg=None,
                                 central_longitude_deg=None):
    """Creates one or more buffers around each polygon.

    N = number of buffers
    V = number of vertices in polygon (different for each buffer and storm cell)

    :param polygon_table: pandas DataFrame created by read_polygons_from_netcdf.
    :param min_buffer_dists_metres: length-N numpy array of minimum buffer
        distances.  If min_buffer_dists_metres[i] is NaN, the [i]th buffer
        includes the original polygon.  If min_buffer_dists_metres[i] is
        defined, the [i]th buffer is a "nested" buffer, not including the
        original polygon.
    :param max_buffer_dists_metres: length-N numpy array of maximum buffer
        distances.  Must be all real numbers (no NaN).
    :param central_latitude_deg: Central latitude (deg N) for azimuthal
        equidistant projection.
    :param central_longitude_deg: Central longitude (deg E) for azimuthal
        equidistant projection.
    :return: polygon_table: Same as input data, but with 2*N additional columns.
    polygon_table.vertex_lat_buffer<j>_deg: length-V numpy array with latitudes
        (deg N) of vertices in [j + 1]th buffer around storm.
    polygon_table.vertex_lng_buffer<j>_deg: length-V numpy array with longitudes
        (deg E) of vertices in [j + 1]th buffer around storm.
    """

    projection_object = projections.init_azimuthal_equidistant_projection(
        central_latitude_deg, central_longitude_deg)

    min_buffer_dists_metres = numpy.round(min_buffer_dists_metres)
    max_buffer_dists_metres = numpy.round(max_buffer_dists_metres)

    (buffer_lat_column_names,
     buffer_lng_column_names) = _distance_buffers_to_column_names(
        min_buffer_dists_metres, max_buffer_dists_metres)

    nested_array = polygon_table[
        [STORM_ID_COLUMN, STORM_ID_COLUMN]].values.tolist()

    argument_dict = {}
    num_buffers = len(max_buffer_dists_metres)
    for j in range(num_buffers):
        argument_dict.update({buffer_lat_column_names[j]: nested_array,
                              buffer_lng_column_names[j]: nested_array})
    polygon_table = polygon_table.assign(**argument_dict)

    num_storms = len(polygon_table.index)
    for i in range(num_storms):
        (orig_vertex_x_metres,
         orig_vertex_y_metres) = projections.project_latlng_to_xy(
            polygon_table[VERTEX_LAT_COLUMN].values[i],
            polygon_table[VERTEX_LNG_COLUMN].values[i],
            projection_object=projection_object)

        for j in range(num_buffers):
            (buffer_vertex_x_metres, buffer_vertex_y_metres) = (
                polygons.make_buffer_around_simple_polygon(
                    orig_vertex_x_metres, orig_vertex_y_metres,
                    min_buffer_dist_metres=min_buffer_dists_metres[j],
                    max_buffer_dist_metres=max_buffer_dists_metres[j]))

            (buffer_vertex_lat_deg,
             buffer_vertex_lng_deg) = projections.project_xy_to_latlng(
                buffer_vertex_x_metres, buffer_vertex_y_metres,
                projection_object=projection_object)

            polygon_table[buffer_lat_column_names[j]].values[
                i] = buffer_vertex_lat_deg
            polygon_table[buffer_lng_column_names[j]].values[
                i] = buffer_vertex_lng_deg

    return polygon_table


def join_stats_and_polygons(stats_table, polygon_table):
    """Joins tables with storm statistics and polygons.

    :param stats_table: pandas DataFrame created by read_stats_from_xml.
    :param polygon_table: pandas DataFrame created by read_polygons_from_netcdf
        or make_buffers_around_polygons.
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

    (central_latitude_deg,
     central_longitude_deg) = myrorss_io.get_center_of_grid(
        nw_grid_point_lat_deg=metadata_dict[
            myrorss_io.NW_GRID_POINT_LAT_COLUMN],
        nw_grid_point_lng_deg=metadata_dict[
            myrorss_io.NW_GRID_POINT_LNG_COLUMN],
        lat_spacing_deg=metadata_dict[myrorss_io.LAT_SPACING_COLUMN],
        lng_spacing_deg=metadata_dict[myrorss_io.LNG_SPACING_COLUMN],
        num_lat_in_grid=metadata_dict[myrorss_io.NUM_LAT_COLUMN],
        num_lng_in_grid=metadata_dict[myrorss_io.NUM_LNG_COLUMN])

    polygon_table = make_buffers_around_polygons(
        polygon_table, min_buffer_dists_metres=MIN_BUFFER_DISTS_METRES,
        max_buffer_dists_metres=MAX_BUFFER_DISTS_METRES,
        central_latitude_deg=central_latitude_deg,
        central_longitude_deg=central_longitude_deg)
    print polygon_table

    storm_table = join_stats_and_polygons(stats_table, polygon_table)
    print storm_table
