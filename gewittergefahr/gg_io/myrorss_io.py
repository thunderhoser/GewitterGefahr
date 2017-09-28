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

import os
import numpy
import pandas
from netCDF4 import Dataset
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import unzipping
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking

NW_GRID_POINT_LAT_COLUMN = 'nw_grid_point_lat_deg'
NW_GRID_POINT_LNG_COLUMN = 'nw_grid_point_lng_deg'
LAT_SPACING_COLUMN = 'lat_spacing_deg'
LNG_SPACING_COLUMN = 'lng_spacing_deg'
NUM_LAT_COLUMN = 'num_lat_in_grid'
NUM_LNG_COLUMN = 'num_lng_in_grid'
HEIGHT_COLUMN = 'height_m_agl'
UNIX_TIME_COLUMN = 'unix_time_sec'
FIELD_NAME_COLUMN = 'field_name'
SENTINEL_VALUE_COLUMN = 'sentinel_values'

NW_GRID_POINT_LAT_COLUMN_ORIG = 'Latitude'
NW_GRID_POINT_LNG_COLUMN_ORIG = 'Longitude'
LAT_SPACING_COLUMN_ORIG = 'LatGridSpacing'
LNG_SPACING_COLUMN_ORIG = 'LonGridSpacing'
NUM_LAT_COLUMN_ORIG = 'Lat'
NUM_LNG_COLUMN_ORIG = 'Lon'
HEIGHT_COLUMN_ORIG = 'Height'
UNIX_TIME_COLUMN_ORIG = 'Time'
FIELD_NAME_COLUMN_ORIG = 'TypeName'
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

RADAR_FIELD_NAMES = [ECHO_TOP_18DBZ_NAME, ECHO_TOP_50DBZ_NAME,
                     LOW_LEVEL_SHEAR_NAME, MID_LEVEL_SHEAR_NAME, REFL_NAME,
                     REFL_COLUMN_MAX_NAME, MESH_NAME, REFL_0CELSIUS_NAME,
                     REFL_M10CELSIUS_NAME, REFL_M20CELSIUS_NAME,
                     REFL_LOWEST_ALTITUDE_NAME, SHI_NAME, VIL_NAME,
                     STORM_ID_NAME]

RADAR_FIELD_NAMES_ORIG = [ECHO_TOP_18DBZ_NAME_ORIG, ECHO_TOP_50DBZ_NAME_ORIG,
                          LOW_LEVEL_SHEAR_NAME_ORIG, MID_LEVEL_SHEAR_NAME_ORIG,
                          REFL_NAME_ORIG, REFL_COLUMN_MAX_NAME_ORIG,
                          MESH_NAME_ORIG, REFL_0CELSIUS_NAME_ORIG,
                          REFL_M10CELSIUS_NAME_ORIG, REFL_M20CELSIUS_NAME_ORIG,
                          REFL_LOWEST_ALTITUDE_NAME_ORIG, SHI_NAME_ORIG,
                          VIL_NAME_ORIG, STORM_ID_NAME_ORIG]

HEIGHT_ARRAY_COLUMN = 'heights_m_agl'
RELATIVE_TOLERANCE = 1e-6

TIME_FORMAT_SECONDS = '%Y%m%d-%H%M%S'
TIME_FORMAT_SPC_DATE = '%Y%m%d'
DAYS_TO_SECONDS = 86400
METRES_TO_KM = 1e-3

UNZIPPED_FILE_EXTENSION = '.netcdf'
ZIPPED_FILE_EXTENSION = '.gz'


def _check_field_name_orig(field_name_orig):
    """Ensures that field name is valid (in `RADAR_FIELD_NAMES_ORIG`).

    :param field_name_orig: Field name in original (MYRORSS) format.
    :raises: ValueError: if `field_name_orig not in RADAR_FIELD_NAMES_ORIG`.
    """

    error_checking.assert_is_string(field_name_orig)

    if field_name_orig not in RADAR_FIELD_NAMES_ORIG:
        error_string = '\n'
        for j in range(len(RADAR_FIELD_NAMES_ORIG)):
            error_string += RADAR_FIELD_NAMES_ORIG[j] + '\n'

        error_string += ('Field names in MYRORSS format (listed above) do not '
                         'include the following: ' + field_name_orig)
        raise ValueError(error_string)


def _check_field_name(field_name):
    """Ensures that field name is valid (in `RADAR_FIELD_NAMES`).

    :param field_name: Field name in new format (my format, not MYRORSS).
    :raises: ValueError: if `field_name not in RADAR_FIELD_NAMES`.
    """

    error_checking.assert_is_string(field_name)

    if field_name not in RADAR_FIELD_NAMES:
        error_string = '\n'
        for j in range(len(RADAR_FIELD_NAMES)):
            error_string += RADAR_FIELD_NAMES[j] + '\n'

        error_string += ('Field names in new format (mine, not MYRORSS) (listed'
                         ' above) do not include the following: ' + field_name)
        raise ValueError(error_string)


def _field_name_orig_to_new(field_name_orig):
    """Converts name of radar field from original (MYRORSS) to new (my) format.

    :param field_name_orig: Field name in MYRORSS format.
    :return: field_name: Field name in my format.
    """

    _check_field_name_orig(field_name_orig)

    found_in_orig_flags = [s == field_name_orig for s in
                           RADAR_FIELD_NAMES_ORIG]
    found_in_orig_index = numpy.where(found_in_orig_flags)[0][0]
    return RADAR_FIELD_NAMES[found_in_orig_index]


def _field_name_new_to_orig(field_name):
    """Converts name of radar field from new (my) to original (MYRORSS) format.

    :param field_name: Field name in my format.
    :return: field_name_orig: Field name in MYRORSS format.
    """

    _check_field_name(field_name)

    found_in_new_flags = [s == field_name for s in RADAR_FIELD_NAMES]
    found_in_new_index = numpy.where(found_in_new_flags)[0][0]
    return RADAR_FIELD_NAMES_ORIG[found_in_new_index]


def _field_to_valid_heights(field_name):
    """Returns valid heights for given radar field.

    :param field_name: Name of radar field.
    :return: valid_heights_m_agl: 1-D numpy integer array of valid heights
        (metres above ground level).
    :raises: ValueError: if `field_name` is "storm_id".
    """

    if field_name == STORM_ID_NAME:
        error_string = (
            'field_name should not be "' + STORM_ID_NAME +
            '".  field_name can be any other string in `RADAR_FIELD_NAMES`.')
        raise ValueError(error_string)

    if field_name == ECHO_TOP_18DBZ_NAME:
        return numpy.array([250])
    elif field_name == ECHO_TOP_50DBZ_NAME:
        return numpy.array([250])
    elif field_name == LOW_LEVEL_SHEAR_NAME:
        return numpy.array([0])
    elif field_name == MID_LEVEL_SHEAR_NAME:
        return numpy.array([0])
    elif field_name == REFL_NAME:
        return numpy.array(
            [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750,
             3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000,
             8500, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000,
             18000, 19000, 20000])
    elif field_name == REFL_COLUMN_MAX_NAME:
        return numpy.array([250])
    elif field_name == MESH_NAME:
        return numpy.array([250])
    elif field_name == REFL_0CELSIUS_NAME:
        return numpy.array([250])
    elif field_name == REFL_M10CELSIUS_NAME:
        return numpy.array([250])
    elif field_name == REFL_M20CELSIUS_NAME:
        return numpy.array([250])
    elif field_name == REFL_LOWEST_ALTITUDE_NAME:
        return numpy.array([250])
    elif field_name == SHI_NAME:
        return numpy.array([250])
    elif field_name == VIL_NAME:
        return numpy.array([250])


def _check_reflectivity_heights(heights_m_agl):
    """Ensures that all reflectivity heights are valid.

    N = number of heights

    :param heights_m_agl: length-N numpy integer array of heights (metres above
        ground level).
    :raises: ValueError: if any element of heights_m_agl is invalid.
    """

    error_checking.assert_is_integer_numpy_array(heights_m_agl)
    error_checking.assert_is_numpy_array(heights_m_agl, num_dimensions=1)

    valid_heights_m_agl = _field_to_valid_heights(REFL_NAME)
    for k in range(len(heights_m_agl)):
        if heights_m_agl[k] in valid_heights_m_agl:
            continue

        error_string = '\n'
        for m in range(len(valid_heights_m_agl)):
            error_string += str(valid_heights_m_agl[m]) + ' m\n'

        error_string += (
            'Known reflectivity heights (listed above) do not include the '
            'following: ' + str(heights_m_agl[k]) + ' m')
        raise ValueError(error_string)


def _field_and_height_arrays_to_dict(field_names, refl_heights_m_agl=None):
    """Converts two arrays (radar fields and heights) to dictionary.

    F = number of radar fields

    :param field_names: length-F list with names of radar fields.  Each name
        must be in `RADAR_FIELD_NAMES`.
    :param refl_heights_m_agl: 1-D numpy integer array of reflectivity heights
        (metres above ground level).  These will be used only for the field
        "reflectivity_dbz", since all others have only one valid height.  If
        `field_names` does not include "reflectivity_dbz", this can be None.
    :return: field_to_heights_dict_m_agl: Dictionary.  Each key is the name of a
        radar field, and each value is a 1-D numpy integer array of heights
        (metres above ground level).
    """

    field_to_heights_dict_m_agl = {}

    for j in range(len(field_names)):
        if field_names[j] == REFL_NAME:
            _check_reflectivity_heights(refl_heights_m_agl)
            field_to_heights_dict_m_agl.update(
                {field_names[j]: refl_heights_m_agl})
        else:
            field_to_heights_dict_m_agl.update(
                {field_names[j]: _field_to_valid_heights(field_names[j])})

    return field_to_heights_dict_m_agl


def _get_relative_field_height_directory(field_name, height_m_agl):
    """Generates expected relative path name for radar field and height.

    :param field_name: Name of radar field (must be in `RADAR_FIELD_NAMES`).
    :param height_m_agl: Height (integer metres above ground level).
    :return: relative_directory_name: Expected relative path name.
    """

    if field_name == REFL_NAME:
        _check_reflectivity_heights(numpy.array([height_m_agl]))
    else:
        height_m_agl = _field_to_valid_heights(field_name)[0]

    return '{0:s}/{1:05.2f}'.format(_field_name_new_to_orig(field_name),
                                    float(height_m_agl) * METRES_TO_KM)


def _get_pathless_raw_file_name(unix_time_sec, zipped=True):
    """Generates pathless name for raw MYRORSS file.

    This file should contain one radar field at one height and one time step.

    :param unix_time_sec: Time in Unix format.
    :param zipped: Boolean flag.  If True, will generate name for zipped file.
        If False, will generate name for unzipped file.
    :return: pathless_raw_file_name: Pathless name for MYRORSS file.
    """

    if zipped:
        return '{0:s}{1:s}{2:s}'.format(
            time_conversion.unix_sec_to_string(unix_time_sec,
                                               TIME_FORMAT_SECONDS),
            UNZIPPED_FILE_EXTENSION, ZIPPED_FILE_EXTENSION)

    return '{0:s}{1:s}'.format(
        time_conversion.unix_sec_to_string(unix_time_sec, TIME_FORMAT_SECONDS),
        UNZIPPED_FILE_EXTENSION)


def _remove_sentinels(sparse_grid_table, field_name, sentinel_values):
    """Removes sentinel values from radar data.

    :param sparse_grid_table: pandas DataFrame in format produced by
        read_sparse_grid_from_netcdf
    :param field_name: Name of radar field (must be in `RADAR_FIELD_NAMES`).
    :param sentinel_values: 1-D numpy array of sentinel values.
    :return: sparse_grid_table: Same as input, except that rows with a sentinel
        value are removed.
    """

    num_rows = len(sparse_grid_table.index)
    sentinel_flags = numpy.full(num_rows, False, dtype=bool)

    for i in range(len(sentinel_values)):
        these_sentinel_flags = numpy.isclose(
            sparse_grid_table[field_name].values, sentinel_values[i],
            rtol=RELATIVE_TOLERANCE)
        sentinel_flags = numpy.logical_or(sentinel_flags, these_sentinel_flags)

    sentinel_indices = numpy.where(sentinel_flags)[0]
    sparse_grid_table[field_name].values[sentinel_indices] = numpy.nan

    sparse_grid_table.drop(sparse_grid_table.index[sentinel_indices], axis=0,
                           inplace=True)
    return sparse_grid_table


def time_unix_sec_to_spc_date(unix_time_sec):
    """Converts time from Unix format to SPC date.

    :param unix_time_sec: Time in Unix format.
    :return: spc_date_string: SPC date at the given time, in format "yyyymmdd".
    """

    return time_conversion.unix_sec_to_string(
        unix_time_sec - DAYS_TO_SECONDS / 2, TIME_FORMAT_SPC_DATE)


def unzip_1day_tar_file(tar_file_name, spc_date_unix_sec=None,
                        top_target_directory_name=None,
                        field_to_heights_dict_m_agl=None):
    """Unzips tar file with all radar fields for one SPC date.

    :param tar_file_name: Path to input file.
    :param spc_date_unix_sec: SPC date in Unix format.
    :param top_target_directory_name: Top-level output directory.  This method
        will create a subdirectory for the SPC date.
    :param field_to_heights_dict_m_agl: Dictionary with field-height pairs to
        extract.  For the exact format of this dictionary, see
        _field_and_height_arrays_to_dict.
    :return: target_directory_name: Path to output directory.  This will be
        "<top_target_directory_name>/<yyyymmdd>", where <yyyymmdd> is the SPC
        date.
    """

    error_checking.assert_is_string(top_target_directory_name)
    target_directory_name = '{0:s}/{1:s}'.format(
        top_target_directory_name, time_unix_sec_to_spc_date(spc_date_unix_sec))

    field_names = field_to_heights_dict_m_agl.keys()
    directory_names_to_unzip = []
    for j in range(len(field_names)):
        these_heights_m_agl = field_to_heights_dict_m_agl[field_names[j]]

        for k in range(len(these_heights_m_agl)):
            directory_names_to_unzip.append(
                _get_relative_field_height_directory(field_names[j],
                                                     these_heights_m_agl[k]))

    unzipping.unzip_tar(tar_file_name,
                        target_directory_name=target_directory_name,
                        file_and_dir_names_to_unzip=directory_names_to_unzip)

    return target_directory_name


def find_local_raw_file(unix_time_sec=None, spc_date_unix_sec=None,
                        field_name=None, height_m_agl=None,
                        top_directory_name=None, zipped=True,
                        raise_error_if_missing=True):
    """Finds raw MYRORSS file on local machine.

    This file should contain one radar field at one height and one time step.

    :param unix_time_sec: Time in Unix format.
    :param spc_date_unix_sec: SPC date in Unix format.
    :param field_name: Name of radar field (must be in `RADAR_FIELD_NAMES`).
    :param height_m_agl: Height (integer metres above ground level).
    :param top_directory_name: Top-level directory for raw MYRORSS files.
    :param zipped: Boolean flag.  If True, will look for zipped file.  If False,
        will look for unzipped file.
    :param raise_error_if_missing: Boolean flag.  If True and file is missing,
        this method will raise an error.
    :return: raw_file_name: Path to raw MYRORSS file.  If raise_error_if_missing
        = False and file is missing, this will be the *expected* path.
    :raises: ValueError: if raise_error_if_missing = True and file is missing.
    """

    _check_field_name(field_name)
    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_boolean(zipped)
    error_checking.assert_is_boolean(raise_error_if_missing)

    pathless_file_name = _get_pathless_raw_file_name(unix_time_sec,
                                                     zipped=zipped)
    directory_name = '{0:s}/{1:s}/{2:s}'.format(
        top_directory_name, time_unix_sec_to_spc_date(spc_date_unix_sec),
        _get_relative_field_height_directory(field_name, height_m_agl))

    raw_file_name = '{0:s}/{1:s}'.format(directory_name, pathless_file_name)
    if raise_error_if_missing and not os.path.isfile(raw_file_name):
        raise ValueError(
            'Cannot find raw file.  Expected at location: ' + raw_file_name)

    return raw_file_name


def extract_netcdf_from_gzip(unix_time_sec=None, spc_date_unix_sec=None,
                             field_name=None, height_m_agl=None,
                             top_directory_name=None):
    """Extracts NetCDF file from gzip archive.

    :param unix_time_sec: Time in Unix format.
    :param spc_date_unix_sec: SPC date in Unix format.
    :param field_name: Name of radar field (must be in `RADAR_FIELD_NAMES`).
    :param height_m_agl: Height (integer metres above ground level).
    :param top_directory_name: Top-level directory for raw MYRORSS files.
    :return: netcdf_file_name: Path to output file.
    """

    gzip_file_name = find_local_raw_file(
        unix_time_sec=unix_time_sec, spc_date_unix_sec=spc_date_unix_sec,
        field_name=field_name, height_m_agl=height_m_agl,
        top_directory_name=top_directory_name, zipped=True,
        raise_error_if_missing=True)

    netcdf_file_name = find_local_raw_file(
        unix_time_sec=unix_time_sec, spc_date_unix_sec=spc_date_unix_sec,
        field_name=field_name, height_m_agl=height_m_agl,
        top_directory_name=top_directory_name, zipped=False,
        raise_error_if_missing=False)

    unzipping.unzip_gzip(gzip_file_name, netcdf_file_name)
    return netcdf_file_name


def rowcol_to_latlng(rows, columns, nw_grid_point_lat_deg=None,
                     nw_grid_point_lng_deg=None, lat_spacing_deg=None,
                     lng_spacing_deg=None):
    """Converts MYRORSS coordinates from row-column to lat-long.

    P = number of points

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

    error_checking.assert_is_real_numpy_array(rows)
    error_checking.assert_is_geq_numpy_array(rows, -0.5, allow_nan=True)
    error_checking.assert_is_numpy_array(rows, num_dimensions=1)
    num_points = len(rows)

    error_checking.assert_is_real_numpy_array(columns)
    error_checking.assert_is_geq_numpy_array(columns, -0.5, allow_nan=True)
    error_checking.assert_is_numpy_array(
        columns, exact_dimensions=numpy.array([num_points]))

    error_checking.assert_is_valid_latitude(nw_grid_point_lat_deg)
    nw_grid_point_lng_deg = lng_conversion.convert_lng_positive_in_west(
        nw_grid_point_lng_deg, allow_nan=False)

    error_checking.assert_is_greater(lat_spacing_deg, 0)
    error_checking.assert_is_greater(lng_spacing_deg, 0)

    latitudes_deg = rounder.round_to_nearest(
        nw_grid_point_lat_deg - lat_spacing_deg * rows,
        lat_spacing_deg / 2)
    longitudes_deg = rounder.round_to_nearest(
        nw_grid_point_lng_deg + lng_spacing_deg * columns,
        lng_spacing_deg / 2)
    return latitudes_deg, lng_conversion.convert_lng_positive_in_west(
        longitudes_deg, allow_nan=False)


def latlng_to_rowcol(latitudes_deg, longitudes_deg, nw_grid_point_lat_deg=None,
                     nw_grid_point_lng_deg=None, lat_spacing_deg=None,
                     lng_spacing_deg=None):
    """Converts MYRORSS coordinates from lat-long to row-column.

    P = number of points

    :param latitudes_deg: length-P numpy array of latitudes (deg N).
    :param longitudes_deg: length-P numpy array of longitudes (deg E).
    :param nw_grid_point_lat_deg: Latitude (deg N) at center of
        northwesternmost grid point.
    :param nw_grid_point_lng_deg: Longitude (deg E) at center of
        northwesternmost grid point.
    :param lat_spacing_deg: Spacing (deg N) between adjacent grid rows.
    :param lng_spacing_deg: Spacing (deg E) between adjacent grid columns.
    :return: rows: length-P numpy array of rows (increasing from north to south).
    :return: columns: length-P numpy array of columns (increasing from west to
        east).
    """

    error_checking.assert_is_valid_lat_numpy_array(latitudes_deg, allow_nan=True)
    error_checking.assert_is_numpy_array(latitudes_deg, num_dimensions=1)
    num_points = len(latitudes_deg)

    longitudes_deg = lng_conversion.convert_lng_positive_in_west(longitudes_deg,
                                                                 allow_nan=True)
    error_checking.assert_is_numpy_array(
        longitudes_deg, exact_dimensions=numpy.array([num_points]))

    error_checking.assert_is_valid_latitude(nw_grid_point_lat_deg)
    nw_grid_point_lng_deg = lng_conversion.convert_lng_positive_in_west(
        nw_grid_point_lng_deg, allow_nan=False)

    error_checking.assert_is_greater(lat_spacing_deg, 0)
    error_checking.assert_is_greater(lng_spacing_deg, 0)

    columns = rounder.round_to_nearest(
        (longitudes_deg - nw_grid_point_lng_deg) / lng_spacing_deg, 0.5)
    rows = rounder.round_to_nearest(
        (nw_grid_point_lat_deg - latitudes_deg) / lat_spacing_deg, 0.5)
    return rows, columns


def get_center_of_grid(nw_grid_point_lat_deg=None, nw_grid_point_lng_deg=None,
                       lat_spacing_deg=None, lng_spacing_deg=None,
                       num_lat_in_grid=None, num_lng_in_grid=None):
    """Finds center of grid.

    :param nw_grid_point_lat_deg: Latitude (deg N) at center of
        northwesternmost grid point.
    :param nw_grid_point_lng_deg: Longitude (deg E) at center of
        northwesternmost grid point.
    :param lat_spacing_deg: Spacing (deg N) between adjacent grid rows.
    :param lng_spacing_deg: Spacing (deg E) between adjacent grid columns.
    :param num_lat_in_grid: Number of grid rows (unique latitudes).
    :param num_lng_in_grid: Number of grid columns (unique longitudes).
    :return: center_latitude_deg: Latitude (deg N) at center of grid.
    :return: center_longitude_deg: Longitude (deg E) at center of grid.
    """

    error_checking.assert_is_valid_latitude(nw_grid_point_lat_deg)
    nw_grid_point_lng_deg = lng_conversion.convert_lng_positive_in_west(
        nw_grid_point_lng_deg, allow_nan=False)

    error_checking.assert_is_greater(lat_spacing_deg, 0)
    error_checking.assert_is_greater(lng_spacing_deg, 0)
    error_checking.assert_is_integer(num_lat_in_grid)
    error_checking.assert_is_greater(num_lat_in_grid, 0)
    error_checking.assert_is_integer(num_lng_in_grid)
    error_checking.assert_is_greater(num_lng_in_grid, 0)

    min_latitude_deg = nw_grid_point_lat_deg - (
        (num_lat_in_grid - 1) * lat_spacing_deg)

    max_longitude_deg = nw_grid_point_lng_deg + (
        (num_lng_in_grid - 1) * lng_spacing_deg)

    return (numpy.mean(numpy.array([min_latitude_deg, nw_grid_point_lat_deg])),
            numpy.mean(numpy.array([nw_grid_point_lng_deg, max_longitude_deg])))


def read_metadata_from_netcdf(netcdf_file_name):
    """Reads metadata from NetCDF file.

    This file should contain one radar field at one height and one time step.

    :param netcdf_file_name: Path to input file.
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
    metadata_dict.height_m_agl: Height of radar scan (integer metres above
        ground level).
    metadata_dict.unix_time_sec: Valid time in Unix format.
    metadata_dict.field_name: Name of radar field (in `RADAR_FIELD_NAMES`).
    metadata_dict.field_name_orig: Name of radar field (in
        `RADAR_FIELD_NAMES_ORIG`).
    metadata_dict.sentinel_values: 1-D numpy array of sentinel values.
    """

    error_checking.assert_file_exists(netcdf_file_name)
    netcdf_dataset = Dataset(netcdf_file_name)
    field_name_orig = str(getattr(netcdf_dataset, FIELD_NAME_COLUMN_ORIG))

    metadata_dict = {
        NW_GRID_POINT_LAT_COLUMN: getattr(netcdf_dataset,
                                          NW_GRID_POINT_LAT_COLUMN_ORIG),
        NW_GRID_POINT_LNG_COLUMN: lng_conversion.convert_lng_positive_in_west(
            getattr(netcdf_dataset, NW_GRID_POINT_LNG_COLUMN_ORIG),
            allow_nan=False),
        LAT_SPACING_COLUMN: getattr(netcdf_dataset, LAT_SPACING_COLUMN_ORIG),
        LNG_SPACING_COLUMN: getattr(netcdf_dataset, LNG_SPACING_COLUMN_ORIG),
        NUM_LAT_COLUMN: netcdf_dataset.dimensions[NUM_LAT_COLUMN_ORIG].size,
        NUM_LNG_COLUMN: netcdf_dataset.dimensions[NUM_LNG_COLUMN_ORIG].size,
        HEIGHT_COLUMN: getattr(netcdf_dataset, HEIGHT_COLUMN_ORIG),
        UNIX_TIME_COLUMN: getattr(netcdf_dataset, UNIX_TIME_COLUMN_ORIG),
        FIELD_NAME_COLUMN_ORIG: field_name_orig,
        FIELD_NAME_COLUMN: _field_name_orig_to_new(field_name_orig)}

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


def read_sparse_grid_from_netcdf(netcdf_file_name, field_name_orig,
                                 sentinel_values):
    """Reads sparse grid from NetCDF file.

    This file should contain one radar field at one height and one time step.

    :param netcdf_file_name: Path to input file.
    :param field_name_orig: Name of radar field (must be in
        `RADAR_FIELD_NAMES_ORIG`).
    :param sentinel_values: 1-D numpy array of sentinel values.
    :return: sparse_grid_table: pandas DataFrame with the following columns.
    sparse_grid_table.grid_row: Grid row (increasing from north to south).
    sparse_grid_table.grid_column: Grid column (increasing from west to east).
    sparse_grid_table.<field_name>: Value of radar field.
    sparse_grid_table.num_grid_cells: Number of grid cells with the same value
        (counting across rows [west to east] first, then down columns [north to
        south]).
    """

    error_checking.assert_file_exists(netcdf_file_name)
    error_checking.assert_is_numpy_array_without_nan(sentinel_values)
    error_checking.assert_is_numpy_array(sentinel_values, num_dimensions=1)

    netcdf_dataset = Dataset(netcdf_file_name)
    field_name = _field_name_orig_to_new(field_name_orig)

    sparse_grid_dict = {
        GRID_ROW_COLUMN: netcdf_dataset.variables[GRID_ROW_COLUMN_ORIG][:],
        GRID_COLUMN_COLUMN:
            netcdf_dataset.variables[GRID_COLUMN_COLUMN_ORIG][:],
        NUM_GRID_CELL_COLUMN:
            netcdf_dataset.variables[NUM_GRID_CELL_COLUMN_ORIG][:],
        field_name: netcdf_dataset.variables[field_name_orig][:]}

    sparse_grid_table = pandas.DataFrame.from_dict(sparse_grid_dict)
    return _remove_sentinels(sparse_grid_table, field_name, sentinel_values)
