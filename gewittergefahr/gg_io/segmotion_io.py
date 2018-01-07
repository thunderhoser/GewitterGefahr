"""IO methods for segmotion output.

--- DEFINITIONS ---

segmotion (or w2segmotionll) = storm-tracking algorithm in WDSS-II.

WDSS-II = Warning Decision Support System -- Integrated Information, a software
package for the visualization and analysis of thunderstorm-related data.
"""

import os
import glob
import gzip
import tempfile
import shutil
import xml.etree.ElementTree as ElementTree
import numpy
import pandas
from gewittergefahr.gg_io import netcdf_io
from gewittergefahr.gg_io import myrorss_and_mrms_io
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import radar_sparse_to_full as radar_s2f
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import unzipping
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import error_checking

# TODO(thunderhoser): replace main method with named method.

FILE_EXISTS_ERROR_CODE = 17

GZIP_FILE_EXTENSION = '.gz'
STATS_FILE_EXTENSION = '.xml'
POLYGON_FILE_EXTENSION = '.netcdf'
STATS_DIR_NAME_PART = 'PolygonTable'
POLYGON_DIR_NAME_PART = 'ClusterID'

SENTINEL_VALUE = -9999
TIME_FORMAT_IN_FILES = '%Y%m%d-%H%M%S'
TIME_FORMAT_IN_FILES_HOUR_ONLY = '%Y%m%d-%H'

SPC_DATE_START_HOUR = 11
SPC_DATE_END_HOUR = 37
HOURS_TO_SECONDS = 3600

STORM_ID_COLUMN_ORIG = 'RowName'
EAST_VELOCITY_COLUMN_ORIG = 'MotionEast'
NORTH_VELOCITY_COLUMN_ORIG = 'MotionSouth'
AGE_COLUMN_ORIG = 'Age'

XML_COLUMN_NAMES = [
    tracking_utils.STORM_ID_COLUMN, tracking_utils.EAST_VELOCITY_COLUMN,
    tracking_utils.NORTH_VELOCITY_COLUMN, tracking_utils.AGE_COLUMN]
XML_COLUMN_NAMES_ORIG = [
    STORM_ID_COLUMN_ORIG, EAST_VELOCITY_COLUMN_ORIG, NORTH_VELOCITY_COLUMN_ORIG,
    AGE_COLUMN_ORIG]

# The following constants are used only in the main method.
SPC_DATE_STRING = '20040811'
TRACKING_START_TIME_UNIX_SEC = 1092224296  # 113816 UTC 11 Aug 2004
TRACKING_END_TIME_UNIX_SEC = 1092312485  # 120805 UTC 12 Aug 2004
MIN_BUFFER_DISTS_METRES = numpy.array([numpy.nan, 0., 5000.])
MAX_BUFFER_DISTS_METRES = numpy.array([0., 5000., 10000.])

VALID_TIME_STRING = '2004-08-11-124818'
VALID_TIME_FORMAT = '%Y-%m-%d-%H%M%S'
TOP_PROCESSED_DIR_NAME = (
    '/localdata/ryan.lagerquist/gewittergefahr_junk/segmotion/processed')
TRACKING_SCALE_METRES2 = 50000000

XML_FILE_NAME = (
    '/localdata/ryan.lagerquist/software/matlab/wdssii/raw_files/segmotion/'
    'smooth02_30dBZ/20040811/TrackingTable/0050.00/2004-08-11-124818_'
    'TrackingTable_0050.00.xml')

NETCDF_FILE_NAME = (
    '/localdata/ryan.lagerquist/software/matlab/wdssii/raw_files/segmotion/'
    'smooth02_30dBZ/20040811/ClusterID/0050.00/20040811-124818.netcdf')


def _xml_column_name_orig_to_new(column_name_orig):
    """Converts name of XML column from original (segmotion) to new format.

    :param column_name_orig: Column name in original format.
    :return: column_name: Column name in new format.
    """

    orig_column_flags = [c == column_name_orig for c in XML_COLUMN_NAMES_ORIG]
    orig_column_index = numpy.where(orig_column_flags)[0][0]
    return XML_COLUMN_NAMES[orig_column_index]


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
    polygon_dict = {tracking_utils.STORM_ID_COLUMN: unique_storm_ids}
    polygon_table = pandas.DataFrame.from_dict(polygon_dict)

    nested_array = polygon_table[[
        tracking_utils.STORM_ID_COLUMN,
        tracking_utils.STORM_ID_COLUMN]].values.tolist()
    argument_dict = {tracking_utils.GRID_POINT_ROW_COLUMN: nested_array,
                     tracking_utils.GRID_POINT_COLUMN_COLUMN: nested_array}
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

        polygon_table[tracking_utils.GRID_POINT_ROW_COLUMN].values[
            i] = this_storm_row_indices
        polygon_table[tracking_utils.GRID_POINT_COLUMN_COLUMN].values[
            i] = this_storm_column_indices

    return polygon_table.loc[polygon_table[tracking_utils.STORM_ID_COLUMN] !=
                             str(int(SENTINEL_VALUE))]


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
            time_conversion.unix_sec_to_string(
                unix_time_sec, TIME_FORMAT_IN_FILES),
            STATS_FILE_EXTENSION, GZIP_FILE_EXTENSION)

    return '{0:s}{1:s}'.format(
        time_conversion.unix_sec_to_string(unix_time_sec, TIME_FORMAT_IN_FILES),
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
            time_conversion.unix_sec_to_string(
                unix_time_sec, TIME_FORMAT_IN_FILES),
            POLYGON_FILE_EXTENSION, GZIP_FILE_EXTENSION)

    return '{0:s}{1:s}'.format(
        time_conversion.unix_sec_to_string(unix_time_sec, TIME_FORMAT_IN_FILES),
        POLYGON_FILE_EXTENSION)


def _get_relative_stats_dir_ordinal_scale(
        spc_date_string, tracking_scale_ordinal):
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


def _get_relative_stats_dir_physical_scale(
        spc_date_string, tracking_scale_metres2):
    """Generates expected relative path for stats directory.

    This directory should contain storm statistics for the given tracking scale.

    :param spc_date_string: SPC date in format "yyyymmdd".
    :param tracking_scale_metres2: Tracking scale.
    :return: stats_directory_name: Expected relative path for stats directory.
    """

    return '{0:s}/{1:s}/scale_{2:d}m2'.format(
        spc_date_string, STATS_DIR_NAME_PART, int(tracking_scale_metres2))


def _get_relative_polygon_dir_ordinal_scale(
        spc_date_string, tracking_scale_ordinal):
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


def _get_relative_polygon_dir_physical_scale(
        spc_date_string, tracking_scale_metres2):
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


def _rename_raw_dirs_ordinal_to_physical(
        top_raw_directory_name=None, spc_date_string=None,
        tracking_scales_ordinal=None, tracking_scales_metres2=None):
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

    # TODO(thunderhoser): Write rename method in file_system_utils.py that
    # handles case where target already exists.

    num_scales = len(tracking_scales_ordinal)

    for j in range(num_scales):
        orig_stats_dir_name = '{0:s}/{1:s}'.format(
            top_raw_directory_name, _get_relative_stats_dir_ordinal_scale(
                spc_date_string, tracking_scales_ordinal[j]))
        new_stats_dir_name = '{0:s}/{1:s}'.format(
            top_raw_directory_name, _get_relative_stats_dir_physical_scale(
                spc_date_string, tracking_scales_metres2[j]))

        try:
            os.rename(orig_stats_dir_name, new_stats_dir_name)
        except OSError as this_error:
            if this_error.errno == FILE_EXISTS_ERROR_CODE:
                shutil.rmtree(new_stats_dir_name)
                os.rename(orig_stats_dir_name, new_stats_dir_name)
            else:
                raise

        orig_polygon_dir_name = '{0:s}/{1:s}'.format(
            top_raw_directory_name, _get_relative_polygon_dir_ordinal_scale(
                spc_date_string, tracking_scales_ordinal[j]))
        new_polygon_dir_name = '{0:s}/{1:s}'.format(
            top_raw_directory_name, _get_relative_polygon_dir_physical_scale(
                spc_date_string, tracking_scales_metres2[j]))

        try:
            os.rename(orig_polygon_dir_name, new_polygon_dir_name)
        except OSError as this_error:
            if this_error.errno == FILE_EXISTS_ERROR_CODE:
                shutil.rmtree(new_polygon_dir_name)
                os.rename(orig_polygon_dir_name, new_polygon_dir_name)
            else:
                raise


def _open_xml_file(xml_file_name):
    """Opens an XML file, which may or may not be gzipped.

    :param xml_file_name: Path to input file.
    :return: xml_tree: Instance of `xml.etree.ElementTree`.
    """

    gzip_as_input = xml_file_name.endswith(GZIP_FILE_EXTENSION)
    if gzip_as_input:
        gzip_file_object = gzip.open(xml_file_name, 'rb')
        xml_temporary_file_object = tempfile.NamedTemporaryFile(delete=False)
        shutil.copyfileobj(gzip_file_object, xml_temporary_file_object)

        xml_file_name = xml_temporary_file_object.name
        gzip_file_object.close()
        xml_temporary_file_object.close()

    xml_tree = ElementTree.parse(xml_file_name)
    if gzip_as_input:
        os.remove(xml_file_name)

    return xml_tree


def unzip_1day_tar_file(
        tar_file_name, spc_date_string, top_target_dir_name,
        scales_to_extract_metres2):
    """Unzips tar file with segmotion output for one SPC date.

    :param tar_file_name: Path to input file.
    :param spc_date_string: SPC date (format "yyyymmdd").
    :param top_target_dir_name: Name of top-level output directory.
    :param scales_to_extract_metres2: 1-D numpy array of tracking scales to
        extract.
    :return: target_directory_name: Path to output directory.  This will be
        "<top_target_directory_name>/<yyyymmdd>", where <yyyymmdd> is the SPC
        date.
    """

    # Verification.
    _ = time_conversion.spc_date_string_to_unix_sec(spc_date_string)
    error_checking.assert_file_exists(tar_file_name)
    error_checking.assert_is_greater_numpy_array(scales_to_extract_metres2, 0)
    error_checking.assert_is_integer_numpy_array(scales_to_extract_metres2)
    error_checking.assert_is_numpy_array(
        scales_to_extract_metres2, num_dimensions=1)

    num_scales_to_extract = len(scales_to_extract_metres2)
    directory_names_to_unzip = []

    for j in range(num_scales_to_extract):
        this_relative_stats_dir_name = _get_relative_stats_dir_physical_scale(
            spc_date_string, scales_to_extract_metres2[j])
        this_relative_polygon_dir_name = (
            _get_relative_polygon_dir_physical_scale(
                spc_date_string, scales_to_extract_metres2[j]))

        directory_names_to_unzip.append(
            this_relative_stats_dir_name.replace(spc_date_string + '/', ''))
        directory_names_to_unzip.append(
            this_relative_polygon_dir_name.replace(spc_date_string + '/', ''))

    target_directory_name = '{0:s}/{1:s}'.format(
        top_target_dir_name, spc_date_string)
    unzipping.unzip_tar(
        tar_file_name, target_directory_name=target_directory_name,
        file_and_dir_names_to_unzip=directory_names_to_unzip)
    return target_directory_name


def find_local_stats_file(
        unix_time_sec, spc_date_string, top_raw_directory_name,
        tracking_scale_metres2, raise_error_if_missing=True):
    """Finds statistics file on local machine.

    This file should contain storm stats (everything except polygons) for one
    time step and one tracking scale.

    :param unix_time_sec: Valid time.
    :param spc_date_string: SPC date (format "yyyymmdd").
    :param top_raw_directory_name: Name of top-level directory with raw
        segmotion files.
    :param tracking_scale_metres2: Tracking scale.
    :param raise_error_if_missing: Boolean flag.  If True and file is missing,
        this method will raise an error.
    :return: stats_file_name: Path to statistics file.  If
        raise_error_if_missing = False and file is missing, this will be the
        *expected* path.
    :raises: ValueError: if raise_error_if_missing = True and file is missing.
    """

    # Verification.
    _ = time_conversion.spc_date_string_to_unix_sec(spc_date_string)
    error_checking.assert_is_string(top_raw_directory_name)
    error_checking.assert_is_greater(tracking_scale_metres2, 0.)
    error_checking.assert_is_boolean(raise_error_if_missing)

    directory_name = '{0:s}/{1:s}'.format(
        top_raw_directory_name, _get_relative_stats_dir_physical_scale(
            spc_date_string, tracking_scale_metres2))

    pathless_file_name = _get_pathless_stats_file_name(
        unix_time_sec, zipped=True)
    stats_file_name = '{0:s}/{1:s}'.format(directory_name, pathless_file_name)

    if raise_error_if_missing and not os.path.isfile(stats_file_name):
        pathless_file_name = _get_pathless_stats_file_name(
            unix_time_sec, zipped=False)
        stats_file_name = '{0:s}/{1:s}'.format(
            directory_name, pathless_file_name)

    if raise_error_if_missing and not os.path.isfile(stats_file_name):
        raise ValueError(
            'Cannot find storm-statistics file.  Expected at location: ' +
            stats_file_name)

    return stats_file_name


def find_local_polygon_file(
        unix_time_sec, spc_date_string, top_raw_directory_name,
        tracking_scale_metres2, raise_error_if_missing=True):
    """Finds polygon file on local machine.

    This file should contain storm outlines (polygons) for one time step and one
    tracking scale.

    :param unix_time_sec: Valid time.
    :param spc_date_string: SPC date (format "yyyymmdd").
    :param top_raw_directory_name: Name of top-level directory with raw
        segmotion files.
    :param tracking_scale_metres2: Tracking scale.
    :param raise_error_if_missing: Boolean flag.  If True and file is missing,
        this method will raise an error.
    :return: polygon_file_name: Path to polygon file.  If
        raise_error_if_missing = False and file is missing, this will be the
        *expected* path.
    :raises: ValueError: if raise_error_if_missing = True and file is missing.
    """

    # Verification.
    _ = time_conversion.spc_date_string_to_unix_sec(spc_date_string)
    error_checking.assert_is_string(top_raw_directory_name)
    error_checking.assert_is_greater(tracking_scale_metres2, 0.)
    error_checking.assert_is_boolean(raise_error_if_missing)

    directory_name = '{0:s}/{1:s}'.format(
        top_raw_directory_name, _get_relative_polygon_dir_physical_scale(
            spc_date_string, tracking_scale_metres2))

    pathless_file_name = _get_pathless_polygon_file_name(
        unix_time_sec, zipped=True)
    polygon_file_name = '{0:s}/{1:s}'.format(directory_name, pathless_file_name)

    if raise_error_if_missing and not os.path.isfile(polygon_file_name):
        pathless_file_name = _get_pathless_polygon_file_name(
            unix_time_sec, zipped=False)
        polygon_file_name = '{0:s}/{1:s}'.format(
            directory_name, pathless_file_name)

    if raise_error_if_missing and not os.path.isfile(polygon_file_name):
        raise ValueError(
            'Cannot find polygon file.  Expected at location: ' +
            polygon_file_name)

    return polygon_file_name


def find_polygon_files_for_spc_date(
        spc_date_string, top_raw_directory_name, tracking_scale_metres2,
        raise_error_if_missing=True):
    """Finds all polygon files for one SPC date.

    :param spc_date_string: SPC date (format "yyyymmdd").
    :param top_raw_directory_name: Name of top-level directory with raw
        segmotion files.
    :param tracking_scale_metres2: Tracking scale.
    :param raise_error_if_missing: If True and no files can be found, this
        method will raise an error.
    :return: polygon_file_names: 1-D list of paths to polygon files.
    """

    error_checking.assert_is_string(top_raw_directory_name)

    directory_name = '{0:s}/{1:s}'.format(
        top_raw_directory_name, _get_relative_polygon_dir_physical_scale(
            spc_date_string, tracking_scale_metres2))

    first_hour_unix_sec = SPC_DATE_START_HOUR * HOURS_TO_SECONDS + (
        time_conversion.string_to_unix_sec(
            spc_date_string, time_conversion.SPC_DATE_FORMAT))
    last_hour_unix_sec = SPC_DATE_END_HOUR * HOURS_TO_SECONDS + (
        time_conversion.string_to_unix_sec(
            spc_date_string, time_conversion.SPC_DATE_FORMAT))
    hours_in_spc_date_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_hour_unix_sec,
        end_time_unix_sec=last_hour_unix_sec,
        time_interval_sec=HOURS_TO_SECONDS, include_endpoint=True)

    polygon_file_names = []
    for this_hour_unix_sec in hours_in_spc_date_unix_sec:
        this_time_string_seconds = time_conversion.unix_sec_to_string(
            this_hour_unix_sec, TIME_FORMAT_IN_FILES)
        this_time_string_hours = time_conversion.unix_sec_to_string(
            this_hour_unix_sec, TIME_FORMAT_IN_FILES_HOUR_ONLY) + '*'

        this_pathless_file_name_zipped = _get_pathless_polygon_file_name(
            this_hour_unix_sec, zipped=True)
        this_pathless_file_pattern_zipped = (
            this_pathless_file_name_zipped.replace(
                this_time_string_seconds, this_time_string_hours))
        this_file_pattern_zipped = '{0:s}/{1:s}'.format(
            directory_name, this_pathless_file_pattern_zipped)

        these_polygon_file_names_zipped = glob.glob(this_file_pattern_zipped)
        if these_polygon_file_names_zipped:
            polygon_file_names += these_polygon_file_names_zipped

        this_pathless_file_name_unzipped = _get_pathless_polygon_file_name(
            this_hour_unix_sec, zipped=False)
        this_pathless_file_pattern_unzipped = (
            this_pathless_file_name_unzipped.replace(
                this_time_string_seconds, this_time_string_hours))
        this_file_pattern_unzipped = '{0:s}/{1:s}'.format(
            directory_name, this_pathless_file_pattern_unzipped)

        these_polygon_file_names_unzipped = glob.glob(
            this_file_pattern_unzipped)
        for this_file_name_unzipped in these_polygon_file_names_unzipped:
            this_file_name_zipped = (
                this_file_name_unzipped + GZIP_FILE_EXTENSION)
            if this_file_name_zipped in polygon_file_names:
                continue

            polygon_file_names.append(this_file_name_unzipped)

    if raise_error_if_missing and not polygon_file_names:
        raise ValueError(
            'Cannot find any polygon files in directory: ' + directory_name)

    polygon_file_names.sort()
    return polygon_file_names


def get_start_end_times_for_spc_date(
        spc_date_string, top_raw_directory_name, tracking_scale_metres2):
    """Returns first and last tracking times for SPC date.

    :param spc_date_string: SPC date (format "yyyymmdd").
    :param top_raw_directory_name: Name of top-level directory with raw
        segmotion files.
    :param tracking_scale_metres2: Tracking scale.
    :return: start_time_unix_sec: First tracking time for SPC date.
    :return: end_time_unix_sec: Last tracking time for SPC date.
    """

    polygon_file_names = find_polygon_files_for_spc_date(
        spc_date_string=spc_date_string,
        top_raw_directory_name=top_raw_directory_name,
        tracking_scale_metres2=tracking_scale_metres2)

    first_metadata_dict = myrorss_and_mrms_io.read_metadata_from_raw_file(
        polygon_file_names[0], data_source=radar_utils.MYRORSS_SOURCE_ID)
    start_time_unix_sec = first_metadata_dict[radar_utils.UNIX_TIME_COLUMN]

    last_metadata_dict = myrorss_and_mrms_io.read_metadata_from_raw_file(
        polygon_file_names[-1], data_source=radar_utils.MYRORSS_SOURCE_ID)
    end_time_unix_sec = last_metadata_dict[radar_utils.UNIX_TIME_COLUMN]

    return start_time_unix_sec, end_time_unix_sec


def read_stats_from_xml(xml_file_name, spc_date_string):
    """Reads storm statistics from XML file.

    :param xml_file_name: Path to input file.
    :param spc_date_string: SPC date (format "yyyymmdd").
    :return: stats_table: pandas DataFrame with the following columns.
    stats_table.storm_id: String ID for storm cell.
    stats_table.east_velocity_m_s01: Eastward velocity (m/s).
    stats_table.north_velocity_m_s01: Northward velocity (m/s).
    stats_table.age_sec: Age of storm cell (seconds).
    """

    # Verification.
    _ = time_conversion.spc_date_string_to_unix_sec(spc_date_string)
    error_checking.assert_file_exists(xml_file_name)

    xml_tree = _open_xml_file(xml_file_name)
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

        if this_column_name == tracking_utils.STORM_ID_COLUMN:
            this_column_values.append(this_element.attrib['value'])
        elif this_column_name == tracking_utils.NORTH_VELOCITY_COLUMN:
            this_column_values.append(-1 * float(this_element.attrib['value']))
        elif this_column_name == tracking_utils.EAST_VELOCITY_COLUMN:
            this_column_values.append(float(this_element.attrib['value']))
        elif this_column_name == tracking_utils.AGE_COLUMN:
            this_column_values.append(
                int(numpy.round(float(this_element.attrib['value']))))

    stats_table = pandas.DataFrame.from_dict(storm_dict)
    storm_ids = _append_spc_date_to_storm_ids(
        stats_table[tracking_utils.STORM_ID_COLUMN].values, spc_date_string)

    stats_table = stats_table.assign(
        **{tracking_utils.STORM_ID_COLUMN: storm_ids})
    return tracking_utils.remove_nan_rows_from_dataframe(stats_table)


def read_polygons_from_netcdf(
        netcdf_file_name, metadata_dict, spc_date_string,
        tracking_start_time_unix_sec, tracking_end_time_unix_sec,
        raise_error_if_fails=True):
    """Reads storm polygons (outlines of storm cells) from NetCDF file.

    P = number of grid points in storm cell (different for each storm cell)
    V = number of vertices in storm polygon (different for each storm cell)

    If file cannot be opened, returns None.

    :param netcdf_file_name: Path to input file.
    :param metadata_dict: Dictionary with metadata for NetCDF file, created by
        `myrorss_and_mrms_io.read_metadata_from_raw_file`.
    :param spc_date_string: SPC date (format "yyyymmdd").
    :param tracking_start_time_unix_sec: Start time for tracking period.  This
        can be found by `get_start_end_times_for_spc_date`.
    :param tracking_end_time_unix_sec: End time for tracking period.  This can
        be found by `get_start_end_times_for_spc_date`.
    :param raise_error_if_fails: Boolean flag.  If True and file cannot be
        opened, this method will raise an error.
    :return: polygon_table: If file cannot be opened and raise_error_if_fails =
        False, this is None.  Otherwise, it is a pandas DataFrame with the
        following columns.
    polygon_table.storm_id: String ID for storm cell.
    polygon_table.unix_time_sec: Time in Unix format.
    polygon_table.spc_date_unix_sec: SPC date in Unix format.
    polygon_table.tracking_start_time_unix_sec: Start time for tracking period.
    polygon_table.tracking_end_time_unix_sec: End time for tracking period.
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
    polygon_table.polygon_object_latlng: Instance of `shapely.geometry.Polygon`
        with vertices in lat-long coordinates.
    polygon_table.polygon_object_rowcol: Instance of `shapely.geometry.Polygon`
        with vertices in row-column coordinates.
    """

    error_checking.assert_file_exists(netcdf_file_name)
    error_checking.assert_is_integer(tracking_start_time_unix_sec)
    error_checking.assert_is_not_nan(tracking_start_time_unix_sec)
    error_checking.assert_is_integer(tracking_end_time_unix_sec)
    error_checking.assert_is_not_nan(tracking_end_time_unix_sec)

    netcdf_dataset = netcdf_io.open_netcdf(netcdf_file_name,
                                           raise_error_if_fails)
    if netcdf_dataset is None:
        return None

    storm_id_var_name = metadata_dict[radar_utils.FIELD_NAME_COLUMN]
    storm_id_var_name_orig = metadata_dict[
        myrorss_and_mrms_io.FIELD_NAME_COLUMN_ORIG]
    num_values = len(
        netcdf_dataset.variables[myrorss_and_mrms_io.GRID_ROW_COLUMN_ORIG])

    if num_values == 0:
        sparse_grid_dict = {
            myrorss_and_mrms_io.GRID_ROW_COLUMN: numpy.array([], dtype=int),
            myrorss_and_mrms_io.GRID_COLUMN_COLUMN: numpy.array([], dtype=int),
            myrorss_and_mrms_io.NUM_GRID_CELL_COLUMN:
                numpy.array([], dtype=int),
            storm_id_var_name: numpy.array([], dtype=int)}
    else:
        sparse_grid_dict = {
            myrorss_and_mrms_io.GRID_ROW_COLUMN:
                netcdf_dataset.variables[
                    myrorss_and_mrms_io.GRID_ROW_COLUMN_ORIG][:],
            myrorss_and_mrms_io.GRID_COLUMN_COLUMN:
                netcdf_dataset.variables[
                    myrorss_and_mrms_io.GRID_COLUMN_COLUMN_ORIG][:],
            myrorss_and_mrms_io.NUM_GRID_CELL_COLUMN:
                netcdf_dataset.variables[
                    myrorss_and_mrms_io.NUM_GRID_CELL_COLUMN_ORIG][:],
            storm_id_var_name:
                netcdf_dataset.variables[storm_id_var_name_orig][:]}

    netcdf_dataset.close()
    sparse_grid_table = pandas.DataFrame.from_dict(sparse_grid_dict)
    numeric_storm_id_matrix, _, _ = (
        radar_s2f.sparse_to_full_grid(sparse_grid_table, metadata_dict))
    polygon_table = _storm_id_matrix_to_coord_lists(numeric_storm_id_matrix)

    num_storms = len(polygon_table.index)
    unix_times_sec = numpy.full(
        num_storms, metadata_dict[radar_utils.UNIX_TIME_COLUMN], dtype=int)

    spc_date_unix_sec = time_conversion.spc_date_string_to_unix_sec(
        spc_date_string)
    spc_dates_unix_sec = numpy.full(num_storms, spc_date_unix_sec, dtype=int)
    tracking_start_times_unix_sec = numpy.full(
        num_storms, tracking_start_time_unix_sec, dtype=int)
    tracking_end_times_unix_sec = numpy.full(
        num_storms, tracking_end_time_unix_sec, dtype=int)

    storm_ids = _append_spc_date_to_storm_ids(
        polygon_table[tracking_utils.STORM_ID_COLUMN].values, spc_date_string)

    simple_array = numpy.full(num_storms, numpy.nan)
    object_array = numpy.full(num_storms, numpy.nan, dtype=object)
    nested_array = polygon_table[[
        tracking_utils.STORM_ID_COLUMN,
        tracking_utils.STORM_ID_COLUMN]].values.tolist()

    argument_dict = {
        tracking_utils.STORM_ID_COLUMN: storm_ids,
        tracking_utils.TIME_COLUMN: unix_times_sec,
        tracking_utils.SPC_DATE_COLUMN: spc_dates_unix_sec,
        tracking_utils.TRACKING_START_TIME_COLUMN:
            tracking_start_times_unix_sec,
        tracking_utils.TRACKING_END_TIME_COLUMN: tracking_end_times_unix_sec,
        tracking_utils.CENTROID_LAT_COLUMN: simple_array,
        tracking_utils.CENTROID_LNG_COLUMN: simple_array,
        tracking_utils.GRID_POINT_LAT_COLUMN: nested_array,
        tracking_utils.GRID_POINT_LNG_COLUMN: nested_array,
        tracking_utils.POLYGON_OBJECT_LATLNG_COLUMN: object_array,
        tracking_utils.POLYGON_OBJECT_ROWCOL_COLUMN: object_array}
    polygon_table = polygon_table.assign(**argument_dict)

    for i in range(num_storms):
        these_vertex_rows, these_vertex_columns = (
            polygons.grid_points_in_poly_to_vertices(
                polygon_table[tracking_utils.GRID_POINT_ROW_COLUMN].values[i],
                polygon_table[
                    tracking_utils.GRID_POINT_COLUMN_COLUMN].values[i]))

        (polygon_table[tracking_utils.GRID_POINT_ROW_COLUMN].values[i],
         polygon_table[tracking_utils.GRID_POINT_COLUMN_COLUMN].values[i]) = (
             polygons.simple_polygon_to_grid_points(
                 these_vertex_rows, these_vertex_columns))

        (polygon_table[tracking_utils.GRID_POINT_LAT_COLUMN].values[i],
         polygon_table[tracking_utils.GRID_POINT_LNG_COLUMN].values[i]) = (
             radar_utils.rowcol_to_latlng(
                 polygon_table[tracking_utils.GRID_POINT_ROW_COLUMN].values[i],
                 polygon_table[
                     tracking_utils.GRID_POINT_COLUMN_COLUMN].values[i],
                 nw_grid_point_lat_deg=
                 metadata_dict[radar_utils.NW_GRID_POINT_LAT_COLUMN],
                 nw_grid_point_lng_deg=
                 metadata_dict[radar_utils.NW_GRID_POINT_LNG_COLUMN],
                 lat_spacing_deg=metadata_dict[radar_utils.LAT_SPACING_COLUMN],
                 lng_spacing_deg=metadata_dict[radar_utils.LNG_SPACING_COLUMN]))

        these_vertex_lat_deg, these_vertex_lng_deg = (
            radar_utils.rowcol_to_latlng(
                these_vertex_rows, these_vertex_columns,
                nw_grid_point_lat_deg=
                metadata_dict[radar_utils.NW_GRID_POINT_LAT_COLUMN],
                nw_grid_point_lng_deg=
                metadata_dict[radar_utils.NW_GRID_POINT_LNG_COLUMN],
                lat_spacing_deg=metadata_dict[radar_utils.LAT_SPACING_COLUMN],
                lng_spacing_deg=metadata_dict[radar_utils.LNG_SPACING_COLUMN]))

        (polygon_table[tracking_utils.CENTROID_LAT_COLUMN].values[i],
         polygon_table[tracking_utils.CENTROID_LNG_COLUMN].values[i]) = (
             polygons.get_latlng_centroid(
                 these_vertex_lat_deg, these_vertex_lng_deg))

        polygon_table[tracking_utils.POLYGON_OBJECT_ROWCOL_COLUMN].values[i] = (
            polygons.vertex_arrays_to_polygon_object(
                these_vertex_columns, these_vertex_rows))
        polygon_table[tracking_utils.POLYGON_OBJECT_LATLNG_COLUMN].values[i] = (
            polygons.vertex_arrays_to_polygon_object(
                these_vertex_lng_deg, these_vertex_lat_deg))

    return polygon_table


def join_stats_and_polygons(stats_table, polygon_table):
    """Joins tables with storm statistics and polygons.

    :param stats_table: pandas DataFrame created by read_stats_from_xml.
    :param polygon_table: pandas DataFrame created by read_polygons_from_netcdf
        or `tracking_utils.make_buffers_around_storm_objects`.
    :return: storm_table: pandas DataFrame with columns from both stats_table
        and polygon_table.
    """

    return polygon_table.merge(
        stats_table, on=tracking_utils.STORM_ID_COLUMN, how='inner')


if __name__ == '__main__':
    STATS_TABLE = read_stats_from_xml(
        XML_FILE_NAME, spc_date_string=SPC_DATE_STRING)
    print STATS_TABLE

    METADATA_DICT = myrorss_and_mrms_io.read_metadata_from_raw_file(
        NETCDF_FILE_NAME, data_source=radar_utils.MYRORSS_SOURCE_ID)
    POLYGON_TABLE = read_polygons_from_netcdf(
        NETCDF_FILE_NAME, metadata_dict=METADATA_DICT,
        spc_date_string=SPC_DATE_STRING,
        tracking_start_time_unix_sec=TRACKING_START_TIME_UNIX_SEC,
        tracking_end_time_unix_sec=TRACKING_END_TIME_UNIX_SEC)
    print POLYGON_TABLE

    POLYGON_TABLE = tracking_utils.make_buffers_around_storm_objects(
        POLYGON_TABLE, min_distances_metres=MIN_BUFFER_DISTS_METRES,
        max_distances_metres=MAX_BUFFER_DISTS_METRES)
    print POLYGON_TABLE

    STORM_TABLE = join_stats_and_polygons(STATS_TABLE, POLYGON_TABLE)
    print STORM_TABLE

    VALID_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
        VALID_TIME_STRING, VALID_TIME_FORMAT)

    OUTPUT_FILE_NAME = tracking_io.find_processed_file(
        unix_time_sec=VALID_TIME_UNIX_SEC,
        data_source=tracking_utils.SEGMOTION_SOURCE_ID,
        spc_date_string=SPC_DATE_STRING,
        top_processed_dir_name=TOP_PROCESSED_DIR_NAME,
        tracking_scale_metres2=TRACKING_SCALE_METRES2,
        raise_error_if_missing=False)

    tracking_io.write_processed_file(STORM_TABLE, OUTPUT_FILE_NAME)
