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

import os
import glob
import pickle
import xml.etree.ElementTree as ElementTree
import numpy
import pandas
from gewittergefahr.gg_io import netcdf_io
from gewittergefahr.gg_io import myrorss_io
from gewittergefahr.gg_io import myrorss_sparse_to_full as sparse_to_full
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import unzipping
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

# TODO(thunderhoser): replace main method with named method.

ZIPPED_FILE_EXTENSION = '.gz'
STATS_FILE_EXTENSION = '.xml'
POLYGON_FILE_EXTENSION = '.netcdf'
PROCESSED_FILE_EXTENSION = '.p'
PROCESSED_FILE_PREFIX = 'segmotion_'
STATS_DIR_NAME_PART = 'TrackingTable'
POLYGON_DIR_NAME_PART = 'ClusterID'

SENTINEL_VALUE = -9999
TIME_FORMAT_ORIG = '%Y%m%d-%H%M%S'
TIME_FORMAT = '%Y-%m-%d-%H%M%S'
TIME_COLUMN = 'unix_time_sec'

STORM_ID_COLUMN = 'storm_id'
EAST_VELOCITY_COLUMN = 'east_velocity_m_s01'
NORTH_VELOCITY_COLUMN = 'north_velocity_m_s01'
AGE_COLUMN = 'age_sec'

STORM_ID_COLUMN_ORIG = 'RowName'
EAST_VELOCITY_COLUMN_ORIG = 'MotionEast'
NORTH_VELOCITY_COLUMN_ORIG = 'MotionSouth'
AGE_COLUMN_ORIG = 'Age'

XML_COLUMN_NAMES = [STORM_ID_COLUMN, EAST_VELOCITY_COLUMN,
                    NORTH_VELOCITY_COLUMN, AGE_COLUMN]
XML_COLUMN_NAMES_ORIG = [STORM_ID_COLUMN_ORIG, EAST_VELOCITY_COLUMN_ORIG,
                         NORTH_VELOCITY_COLUMN_ORIG, AGE_COLUMN_ORIG]

GRID_POINT_LAT_COLUMN = 'grid_point_latitudes_deg'
GRID_POINT_LNG_COLUMN = 'grid_point_longitudes_deg'
GRID_POINT_ROW_COLUMN = 'grid_point_rows'
GRID_POINT_COLUMN_COLUMN = 'grid_point_columns'

CENTROID_LAT_COLUMN = 'centroid_lat_deg'
CENTROID_LNG_COLUMN = 'centroid_lng_deg'
VERTEX_LAT_COLUMN = 'vertex_latitudes_deg'
VERTEX_LNG_COLUMN = 'vertex_longitudes_deg'
VERTEX_ROW_COLUMN = 'vertex_rows'
VERTEX_COLUMN_COLUMN = 'vertex_columns'

BUFFER_VERTEX_LAT_COLUMN_PREFIX = 'vertex_latitudes_deg_buffer_'
BUFFER_VERTEX_LNG_COLUMN_PREFIX = 'vertex_longitudes_deg_buffer_'

# The following constants are used only in the main method.
SPC_DATE_UNIX_SEC = 1092228498
MIN_BUFFER_DISTS_METRES = numpy.array([numpy.nan, 0., 5000.])
MAX_BUFFER_DISTS_METRES = numpy.array([0., 5000., 10000.])

XML_FILE_NAME = (
    '/localdata/ryan.lagerquist/software/matlab/wdssii/raw_files/segmotion/'
    'smooth02_30dBZ/20040811/TrackingTable/0050.00/2004-08-11-124818_'
    'TrackingTable_0050.00.xml')

NETCDF_FILE_NAME = (
    '/localdata/ryan.lagerquist/software/matlab/wdssii/raw_files/segmotion/'
    'smooth02_30dBZ/20040811/ClusterID/0050.00/20040811-124818.netcdf')

PICKLE_FILE_NAME = (
    '/localdata/ryan.lagerquist/gewittergefahr_junk/segmotion/processed/'
    '20040811/scale_50000000m2/segmotion_2004-08-11-124818.p')


def _xml_column_name_orig_to_new(column_name_orig):
    """Converts name of XML column from original (segmotion) to new format.

    :param column_name_orig: Column name in original format.
    :return: column_name: Column name in new format.
    """

    orig_column_flags = [c == column_name_orig for c in XML_COLUMN_NAMES_ORIG]
    orig_column_index = numpy.where(orig_column_flags)[0][0]
    return XML_COLUMN_NAMES[orig_column_index]


def _remove_rows_with_nan(input_table):
    """Removes all rows with at least one NaN from pandas DataFrame.

    :param input_table: pandas DataFrame, which may contain NaN's.
    :return: output_table: Same as input_table, but without NaN's.
    """

    return input_table.loc[input_table.notnull().all(axis=1)]


def _append_spc_date_to_storm_ids(storm_ids_orig, spc_date_string):
    """Appends SPC date to each storm ID.

    This ensures that storm IDs from different days will be unique.

    N = number of storms

    :param storm_ids_orig: length-N list of string IDs.
    :param spc_date_string: SPC date in format "yyyymmdd".
    :return: storm_ids: Same as input, except with date appended to each ID.
    """

    return ['{0:s}_{1:s}'.format(s, spc_date_string) for s in storm_ids_orig]


def _storm_id_matrix_to_coord_lists(numeric_storm_id_matrix):
    """Converts matrix of storm IDs to one coordinate list* for each storm cell.

    * list of grid points inside the storm

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)
    P = number of grid points in a given storm cell

    :param numeric_storm_id_matrix: M-by-N numpy array of numeric storm IDs.
    :return: polygon_table: pandas DataFrame with the following columns.
    polygon_table.storm_id: String ID for storm cell.
    polygon_table.grid_point_rows: length-P numpy array with row indices
        (integers) of grid points in storm cell.
    polygon_table.grid_point_columns: length-P numpy array with column indices
        (integers) of grid points in storm cell.
    """

    numeric_storm_id_matrix[
        numpy.isnan(numeric_storm_id_matrix)] = SENTINEL_VALUE
    unique_numeric_storm_ids, indices_orig_to_unique = numpy.unique(
        numeric_storm_id_matrix, return_inverse=True)

    unique_storm_ids = [str(int(this_id)) for this_id in
                        unique_numeric_storm_ids]
    polygon_dict = {STORM_ID_COLUMN: unique_storm_ids}
    polygon_table = pandas.DataFrame.from_dict(polygon_dict)

    nested_array = polygon_table[
        [STORM_ID_COLUMN, STORM_ID_COLUMN]].values.tolist()
    argument_dict = {GRID_POINT_ROW_COLUMN: nested_array,
                     GRID_POINT_COLUMN_COLUMN: nested_array}
    polygon_table = polygon_table.assign(**argument_dict)

    num_grid_rows = numeric_storm_id_matrix.shape[0]
    num_grid_columns = numeric_storm_id_matrix.shape[1]
    num_storms = len(unique_numeric_storm_ids)

    for i in range(num_storms):
        if unique_numeric_storm_ids[i] == SENTINEL_VALUE:
            continue

        this_storm_linear_indices = numpy.where(indices_orig_to_unique == i)[0]
        (this_storm_row_indices,
         this_storm_column_indices) = numpy.unravel_index(
             this_storm_linear_indices, (num_grid_rows, num_grid_columns))

        polygon_table[GRID_POINT_ROW_COLUMN].values[i] = this_storm_row_indices
        polygon_table[GRID_POINT_COLUMN_COLUMN].values[
            i] = this_storm_column_indices

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

    num_buffers = len(min_buffer_dists_metres)
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
            time_conversion.unix_sec_to_string(unix_time_sec, TIME_FORMAT_ORIG),
            STATS_FILE_EXTENSION, ZIPPED_FILE_EXTENSION)

    return '{0:s}{1:s}'.format(
        time_conversion.unix_sec_to_string(unix_time_sec, TIME_FORMAT_ORIG),
        STATS_FILE_EXTENSION)


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
            time_conversion.unix_sec_to_string(unix_time_sec, TIME_FORMAT_ORIG),
            POLYGON_FILE_EXTENSION, ZIPPED_FILE_EXTENSION)

    return '{0:s}{1:s}'.format(
        time_conversion.unix_sec_to_string(unix_time_sec, TIME_FORMAT_ORIG),
        POLYGON_FILE_EXTENSION)


def _get_pathless_processed_file_name(unix_time_sec):
    """Generates pathless name for processed file.

    This file should contain both statistics and polygons for one time step and
    one tracking scale.

    :param unix_time_sec: Time in Unix format.
    :return: pathless_processed_file_name: Pathless name for processed file.
    """

    return '{0:s}{1:s}{2:s}'.format(
        PROCESSED_FILE_PREFIX,
        time_conversion.unix_sec_to_string(unix_time_sec, TIME_FORMAT),
        PROCESSED_FILE_EXTENSION)


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

    :param spc_date_string: SPC date in format "yyyymmdd".
    :param tracking_scale_metres2: Tracking scale.
    :return: polygon_directory_name: Expected relative path for polygon
        directory.
    """

    return '{0:s}/{1:s}/scale_{2:d}m2'.format(
        spc_date_string, POLYGON_DIR_NAME_PART, int(tracking_scale_metres2))


def _get_relative_processed_directory(spc_date_string, tracking_scale_metres2):
    """Generates expected relative path for directory with processed files.

    :param spc_date_string: SPC date in format "yyyymmdd".
    :param tracking_scale_metres2: Tracking scale.
    :return: processed_directory_name: Expected relative path for directory with
        processed files.
    """

    return '{0:s}/scale_{1:d}m2'.format(
        spc_date_string, int(tracking_scale_metres2))


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

    num_scales = len(tracking_scales_ordinal)

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

    error_checking.assert_file_exists(tar_file_name)

    error_checking.assert_is_geq_numpy_array(scales_to_extract_ordinal, 0)
    error_checking.assert_is_integer_numpy_array(scales_to_extract_ordinal)
    error_checking.assert_is_numpy_array(scales_to_extract_ordinal,
                                         num_dimensions=1)

    num_scales_to_extract = len(scales_to_extract_ordinal)
    error_checking.assert_is_greater_numpy_array(scales_to_extract_metres2, 0.)
    error_checking.assert_is_numpy_array(
        scales_to_extract_metres2,
        exact_dimensions=numpy.array([num_scales_to_extract]))

    spc_date_string = myrorss_io.time_unix_sec_to_spc_date(spc_date_unix_sec)
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

    error_checking.assert_is_string(top_raw_directory_name)
    error_checking.assert_is_greater(tracking_scale_metres2, 0.)
    error_checking.assert_is_boolean(zipped)
    error_checking.assert_is_boolean(raise_error_if_missing)

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

    error_checking.assert_is_string(top_raw_directory_name)
    error_checking.assert_is_greater(tracking_scale_metres2, 0.)
    error_checking.assert_is_boolean(zipped)
    error_checking.assert_is_boolean(raise_error_if_missing)

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


def read_stats_from_xml(xml_file_name, spc_date_unix_sec=None):
    """Reads storm statistics from XML file.

    :param xml_file_name: Path to input file.
    :param spc_date_unix_sec: SPC date in Unix format.
    :return: stats_table: pandas DataFrame with the following columns.
    stats_table.storm_id: String ID for storm cell.
    stats_table.east_velocity_m_s01: Eastward velocity (m/s).
    stats_table.north_velocity_m_s01: Northward velocity (m/s).
    stats_table.age_sec: Age of storm cell (seconds).
    """

    error_checking.assert_file_exists(xml_file_name)
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
                this_column_name = _xml_column_name_orig_to_new(
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

    stats_table = pandas.DataFrame.from_dict(storm_dict)

    spc_date_string = myrorss_io.time_unix_sec_to_spc_date(spc_date_unix_sec)
    storm_ids = _append_spc_date_to_storm_ids(
        stats_table[STORM_ID_COLUMN].values, spc_date_string)

    stats_table = stats_table.assign(**{STORM_ID_COLUMN: storm_ids})
    return _remove_rows_with_nan(stats_table)


def read_polygons_from_netcdf(netcdf_file_name, metadata_dict=None,
                              spc_date_unix_sec=None,
                              raise_error_if_fails=True):
    """Reads storm polygons (outlines of storm cells) from NetCDF file.

    P = number of grid points in storm cell (different for each storm cell)
    V = number of vertices in storm polygon (different for each storm cell)

    If file cannot be opened, returns None.

    :param netcdf_file_name: Path to input file.
    :param metadata_dict: Dictionary with metadata from NetCDF file, in format
        produced by `myrorss_io.read_metadata_from_netcdf`.
    :param spc_date_unix_sec: SPC date in Unix format.
    :param raise_error_if_fails: Boolean flag.  If raise_error_if_fails = True
        and file cannot be opened, will raise error.
    :return: polygon_table: If file cannot be opened and raise_error_if_fails =
        False, this is None.  Otherwise, it is a pandas DataFrame with the
        following columns.
    polygon_table.storm_id: String ID for storm cell.
    polygon_table.unix_time_sec: Time in Unix format.
    polygon_table.centroid_lat_deg: Latitude at centroid of storm cell (deg N).
    polygon_table.centroid_lng_deg: Longitude at centroid of storm cell (deg E).
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

    error_checking.assert_file_exists(netcdf_file_name)
    netcdf_dataset = netcdf_io.open_netcdf(netcdf_file_name,
                                           raise_error_if_fails)
    if netcdf_dataset is None:
        return None

    storm_id_var_name = metadata_dict[myrorss_io.FIELD_NAME_COLUMN]
    storm_id_var_name_orig = metadata_dict[myrorss_io.FIELD_NAME_COLUMN_ORIG]

    sparse_grid_dict = {
        myrorss_io.GRID_ROW_COLUMN:
            netcdf_dataset.variables[myrorss_io.GRID_ROW_COLUMN_ORIG][:],
        myrorss_io.GRID_COLUMN_COLUMN:
            netcdf_dataset.variables[myrorss_io.GRID_COLUMN_COLUMN_ORIG][:],
        myrorss_io.NUM_GRID_CELL_COLUMN:
            netcdf_dataset.variables[myrorss_io.NUM_GRID_CELL_COLUMN_ORIG][:],
        storm_id_var_name: netcdf_dataset.variables[storm_id_var_name_orig][:]}

    sparse_grid_table = pandas.DataFrame.from_dict(sparse_grid_dict)
    (numeric_storm_id_matrix, _, _) = (
        sparse_to_full.sparse_to_full_grid_wrapper(sparse_grid_table,
                                                   metadata_dict))

    polygon_table = _storm_id_matrix_to_coord_lists(numeric_storm_id_matrix)

    num_storms = len(polygon_table.index)
    unix_times_sec = numpy.full(
        num_storms, metadata_dict[myrorss_io.UNIX_TIME_COLUMN], dtype=int)

    spc_date_string = myrorss_io.time_unix_sec_to_spc_date(spc_date_unix_sec)
    storm_ids = _append_spc_date_to_storm_ids(
        polygon_table[STORM_ID_COLUMN].values, spc_date_string)

    simple_array = numpy.full(num_storms, numpy.nan)
    nested_array = polygon_table[
        [STORM_ID_COLUMN, STORM_ID_COLUMN]].values.tolist()

    argument_dict = {GRID_POINT_LAT_COLUMN: nested_array,
                     GRID_POINT_LNG_COLUMN: nested_array,
                     VERTEX_LAT_COLUMN: nested_array,
                     VERTEX_LNG_COLUMN: nested_array,
                     VERTEX_ROW_COLUMN: nested_array,
                     VERTEX_COLUMN_COLUMN: nested_array,
                     CENTROID_LAT_COLUMN: simple_array,
                     CENTROID_LNG_COLUMN: simple_array,
                     TIME_COLUMN: unix_times_sec, STORM_ID_COLUMN: storm_ids}
    polygon_table = polygon_table.assign(**argument_dict)

    for i in range(num_storms):
        (polygon_table[VERTEX_ROW_COLUMN].values[i],
         polygon_table[VERTEX_COLUMN_COLUMN].values[i]) = (
             polygons.points_in_poly_to_vertices(
                 polygon_table[GRID_POINT_ROW_COLUMN].values[i],
                 polygon_table[GRID_POINT_COLUMN_COLUMN].values[i]))

        (polygon_table[GRID_POINT_ROW_COLUMN].values[i],
         polygon_table[GRID_POINT_COLUMN_COLUMN].values[i]) = (
             polygons.simple_polygon_to_grid_points(
                 polygon_table[VERTEX_ROW_COLUMN].values[i],
                 polygon_table[VERTEX_COLUMN_COLUMN].values[i]))

        (polygon_table[GRID_POINT_LAT_COLUMN].values[i],
         polygon_table[GRID_POINT_LNG_COLUMN].values[i]) = (
             myrorss_io.rowcol_to_latlng(
                 polygon_table[GRID_POINT_ROW_COLUMN].values[i],
                 polygon_table[GRID_POINT_COLUMN_COLUMN].values[i],
                 nw_grid_point_lat_deg=
                 metadata_dict[myrorss_io.NW_GRID_POINT_LAT_COLUMN],
                 nw_grid_point_lng_deg=
                 metadata_dict[myrorss_io.NW_GRID_POINT_LNG_COLUMN],
                 lat_spacing_deg=metadata_dict[myrorss_io.LAT_SPACING_COLUMN],
                 lng_spacing_deg=metadata_dict[myrorss_io.LNG_SPACING_COLUMN]))

        (polygon_table[VERTEX_LAT_COLUMN].values[i],
         polygon_table[VERTEX_LNG_COLUMN].values[i]) = (
             myrorss_io.rowcol_to_latlng(
                 polygon_table[VERTEX_ROW_COLUMN].values[i],
                 polygon_table[VERTEX_COLUMN_COLUMN].values[i],
                 nw_grid_point_lat_deg=
                 metadata_dict[myrorss_io.NW_GRID_POINT_LAT_COLUMN],
                 nw_grid_point_lng_deg=
                 metadata_dict[myrorss_io.NW_GRID_POINT_LNG_COLUMN],
                 lat_spacing_deg=metadata_dict[myrorss_io.LAT_SPACING_COLUMN],
                 lng_spacing_deg=metadata_dict[myrorss_io.LNG_SPACING_COLUMN]))

        (polygon_table[CENTROID_LAT_COLUMN].values[i],
         polygon_table[CENTROID_LNG_COLUMN].values[i]) = (
             polygons.get_latlng_centroid(
                 polygon_table[VERTEX_LAT_COLUMN].values[i],
                 polygon_table[VERTEX_LNG_COLUMN].values[i]))

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

    error_checking.assert_is_geq_numpy_array(
        min_buffer_dists_metres, 0., allow_nan=True)
    error_checking.assert_is_numpy_array(min_buffer_dists_metres,
                                         num_dimensions=1)

    num_buffers = len(min_buffer_dists_metres)
    error_checking.assert_is_geq_numpy_array(max_buffer_dists_metres, 0.)
    error_checking.assert_is_numpy_array(
        max_buffer_dists_metres, exact_dimensions=numpy.array([num_buffers]))

    for j in range(num_buffers):
        if numpy.isnan(min_buffer_dists_metres[j]):
            continue
        error_checking.assert_is_greater(max_buffer_dists_metres[j],
                                         min_buffer_dists_metres[j])

    projection_object = projections.init_azimuthal_equidistant_projection(
        central_latitude_deg, central_longitude_deg)

    (buffer_lat_column_names,
     buffer_lng_column_names) = _distance_buffers_to_column_names(
         min_buffer_dists_metres, max_buffer_dists_metres)

    nested_array = polygon_table[
        [STORM_ID_COLUMN, STORM_ID_COLUMN]].values.tolist()

    argument_dict = {}
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


def find_processed_file(unix_time_sec=None, spc_date_unix_sec=None,
                        top_processed_dir_name=None,
                        tracking_scale_metres2=None,
                        raise_error_if_missing=True):
    """Finds processed file on local machine.

    :param unix_time_sec: Time in Unix format.
    :param spc_date_unix_sec: SPC date in Unix format.
    :param top_processed_dir_name: Top-level directory for processed segmotion
        files.
    :param tracking_scale_metres2: Tracking scale (m^2).
    :param raise_error_if_missing: Boolean flag.  If True and file is missing,
        this method will raise an error.
    :return: processed_file_name: Path to processed file.  If
        raise_error_if_missing = False and file is missing, this will be the
        *expected* path.
    :raises: ValueError: if raise_error_if_missing = True and file is missing.
    """

    error_checking.assert_is_string(top_processed_dir_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    pathless_file_name = _get_pathless_processed_file_name(unix_time_sec)
    spc_date_string = myrorss_io.time_unix_sec_to_spc_date(spc_date_unix_sec)
    relative_directory_name = _get_relative_processed_directory(
        spc_date_string, tracking_scale_metres2)

    processed_file_name = '{0:s}/{1:s}/{2:s}'.format(
        top_processed_dir_name, relative_directory_name, pathless_file_name)

    if raise_error_if_missing and not os.path.isfile(processed_file_name):
        raise ValueError('Cannot find processed file.  Expected at location: ' +
                         processed_file_name)

    return processed_file_name


def glob_processed_files_many_spc_dates(spc_dates_unix_sec,
                                        top_processed_dir_name=None,
                                        tracking_scale_metres2=None,
                                        raise_error_if_missing=True):
    """Globs for processed files from many SPC dates.

    N = number of SPC dates

    :param spc_dates_unix_sec: length-N numpy array of SPC dates in Unix format.
    :param top_processed_dir_name: Top-level directory for processed segmotion
        files.
    :param tracking_scale_metres2: Tracking scale (m^2).
    :param raise_error_if_missing: Boolean flag.  If True and there is any SPC
        date with no files, this method will raise an error.
    :return: processed_file_names: 1-D list of paths to processed files.
    :raises: ValueError: if raise_error_if_missing = True and there is any SPC
        date with no files.
    """

    error_checking.assert_is_numpy_array(spc_dates_unix_sec, num_dimensions=1)
    error_checking.assert_is_string(top_processed_dir_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    processed_file_names = []
    num_spc_dates = len(spc_dates_unix_sec)

    for i in range(num_spc_dates):
        this_spc_date_string = myrorss_io.time_unix_sec_to_spc_date(
            spc_dates_unix_sec[i])
        this_directory_name = '{0:s}/{1:s}'.format(
            top_processed_dir_name, _get_relative_processed_directory(
                this_spc_date_string, tracking_scale_metres2))
        this_file_pattern = '{0:s}/*{1:s}'.format(this_directory_name,
                                                  PROCESSED_FILE_EXTENSION)

        these_processed_file_names = glob.glob(this_file_pattern)
        if not these_processed_file_names:
            if not raise_error_if_missing:
                continue

            error_string = (
                'Cannot find processed files for SPC date "' +
                this_spc_date_string + '".  Expected in directory: ' +
                this_directory_name)
            raise ValueError(error_string)

        processed_file_names += these_processed_file_names

    return processed_file_names


def write_processed_file(storm_table, processed_file_name):
    """Writes storm stats and polygons (at one time step and scale) to Pickle.

    :param storm_table: pandas DataFrame created by join_stats_and_polygons.
    :param processed_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        file_name=processed_file_name)
    processed_file_handle = open(processed_file_name, 'wb')
    pickle.dump(storm_table, processed_file_handle)
    processed_file_handle.close()


def read_processed_file(processed_file_name):
    """Reads storm stats and polygons (at one time step and scale) from Pickle.

    :param processed_file_name: Path to input file.
    :return: storm_table: pandas DataFrame with columns generated by
        join_stats_and_polygons.
    """

    processed_file_handle = open(processed_file_name, 'rb')
    storm_table = pickle.load(processed_file_handle)
    processed_file_handle.close()
    return storm_table


if __name__ == '__main__':
    STATS_TABLE = read_stats_from_xml(
        XML_FILE_NAME, spc_date_unix_sec=SPC_DATE_UNIX_SEC)
    print STATS_TABLE

    METADATA_DICT = myrorss_io.read_metadata_from_netcdf(NETCDF_FILE_NAME)
    POLYGON_TABLE = read_polygons_from_netcdf(
        NETCDF_FILE_NAME, metadata_dict=METADATA_DICT,
        spc_date_unix_sec=SPC_DATE_UNIX_SEC)
    print POLYGON_TABLE

    (CENTRAL_LATITUDE_DEG, CENTRAL_LONGITUDE_DEG) = (
        myrorss_io.get_center_of_grid(
            nw_grid_point_lat_deg=
            METADATA_DICT[myrorss_io.NW_GRID_POINT_LAT_COLUMN],
            nw_grid_point_lng_deg=
            METADATA_DICT[myrorss_io.NW_GRID_POINT_LNG_COLUMN],
            lat_spacing_deg=METADATA_DICT[myrorss_io.LAT_SPACING_COLUMN],
            lng_spacing_deg=METADATA_DICT[myrorss_io.LNG_SPACING_COLUMN],
            num_lat_in_grid=METADATA_DICT[myrorss_io.NUM_LAT_COLUMN],
            num_lng_in_grid=METADATA_DICT[myrorss_io.NUM_LNG_COLUMN]))

    POLYGON_TABLE = make_buffers_around_polygons(
        POLYGON_TABLE, min_buffer_dists_metres=MIN_BUFFER_DISTS_METRES,
        max_buffer_dists_metres=MAX_BUFFER_DISTS_METRES,
        central_latitude_deg=CENTRAL_LATITUDE_DEG,
        central_longitude_deg=CENTRAL_LONGITUDE_DEG)
    print POLYGON_TABLE

    STORM_TABLE = join_stats_and_polygons(STATS_TABLE, POLYGON_TABLE)
    print STORM_TABLE

    write_processed_file(STORM_TABLE, PICKLE_FILE_NAME)
