"""IO methods for GridRad data.

--- REFERENCE ---

http://gridrad.org
"""

import os.path
import numpy
from gewittergefahr.gg_io import netcdf_io
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking

LATITUDE_TOLERANCE_DEG = 0.01
LONGITUDE_TOLERANCE_DEG = 0.01

YEAR_FORMAT = '%Y'
TIME_FORMAT_IN_FILE_NAMES = '%Y%m%dT%H%M%SZ'
PATHLESS_FILE_NAME_PREFIX = 'nexrad_3d_4_1'
FILE_EXTENSION = '.nc'

KM_TO_METRES = 1000
ZERO_TIME_UNIX_SEC = 978307200

LATITUDE_NAME_ORIG = 'Latitude'
LONGITUDE_NAME_ORIG = 'Longitude'
HEIGHT_NAME_ORIG = 'Altitude'
TIME_NAME_ORIG = 'time'


def _time_from_gridrad_to_unix(gridrad_time_sec):
    """Converts time from GridRad format to Unix format.

    GridRad format = seconds since 0000 UTC 1 Jan 2001
    Unix format = seconds since 0000 UTC 1 Jan 1970

    :param gridrad_time_sec: Time in GridRad format.
    :return: unix_time_sec: Time in Unix format.
    """

    return gridrad_time_sec + ZERO_TIME_UNIX_SEC


def _get_pathless_file_name(unix_time_sec):
    """Determines pathless name of GridRad file.

    :param unix_time_sec: Valid time.
    :return: pathless_file_name: Pathless name of GridRad file.
    """

    return '{0:s}_{1:s}{2:s}'.format(
        PATHLESS_FILE_NAME_PREFIX,
        time_conversion.unix_sec_to_string(
            unix_time_sec, TIME_FORMAT_IN_FILE_NAMES),
        FILE_EXTENSION
    )


def _check_grid_points(
        grid_point_latitudes_deg, grid_point_longitudes_deg, metadata_dict):
    """Ensures that grid is regular in lat-long coordinates.

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)

    :param grid_point_latitudes_deg: length-M numpy array of grid-point
        latitudes (deg N).
    :param grid_point_longitudes_deg: length-N numpy array of grid-point
        longitudes (deg E).
    :param metadata_dict: Dictionary created by
        `read_metadata_from_full_grid_file`.
    :raises: ValueError: if the grid is not regular in lat-long coordinates.
    """

    min_latitude_deg = metadata_dict[radar_utils.NW_GRID_POINT_LAT_COLUMN] - (
        metadata_dict[radar_utils.LAT_SPACING_COLUMN] *
        (metadata_dict[radar_utils.NUM_LAT_COLUMN] - 1)
    )

    expected_latitudes_deg, expected_longitudes_deg = (
        grids.get_latlng_grid_points(
            min_latitude_deg=min_latitude_deg,
            min_longitude_deg=metadata_dict[
                radar_utils.NW_GRID_POINT_LNG_COLUMN],
            lat_spacing_deg=metadata_dict[radar_utils.LAT_SPACING_COLUMN],
            lng_spacing_deg=metadata_dict[radar_utils.LNG_SPACING_COLUMN],
            num_rows=metadata_dict[radar_utils.NUM_LAT_COLUMN],
            num_columns=metadata_dict[radar_utils.NUM_LNG_COLUMN])
    )

    if not numpy.allclose(grid_point_latitudes_deg, expected_latitudes_deg,
                          atol=LATITUDE_TOLERANCE_DEG):

        for i in range(len(grid_point_latitudes_deg)):
            print((
                      'Expected latitude = {0:.4f} deg N ... actual = {1:.4f} deg N'
            ).format(expected_latitudes_deg[i], grid_point_latitudes_deg[i]))

        max_latitude_diff_deg = numpy.max(numpy.absolute(
            expected_latitudes_deg - grid_point_latitudes_deg
        ))

        error_string = (
            '\n\nAs shown above, lat-long grid is irregular.  There is a max '
            'difference of {0:f} deg N between expected and actual latitudes.'
        ).format(max_latitude_diff_deg)

        raise ValueError(error_string)

    if not numpy.allclose(grid_point_longitudes_deg, expected_longitudes_deg,
                          atol=LONGITUDE_TOLERANCE_DEG):

        for i in range(len(grid_point_longitudes_deg)):
            print((
                'Expected longitude = {0:.4f} deg E ... actual = {1:.4f} deg E'
            ).format(expected_longitudes_deg[i], grid_point_longitudes_deg[i]))

        max_longitude_diff_deg = numpy.max(numpy.absolute(
            expected_longitudes_deg - grid_point_longitudes_deg
        ))

        error_string = (
            '\n\nAs shown above, lat-long grid is irregular.  There is a max '
            'difference of {0:f} deg E between expected and actual longitudes.'
        ).format(max_longitude_diff_deg)

        raise ValueError(error_string)


def file_name_to_time(gridrad_file_name):
    """Parses valid time from name of GridRad file.

    :param gridrad_file_name: Path to GridRad file.
    :return: unix_time_sec: Valid time.
    """

    _, pathless_file_name = os.path.split(gridrad_file_name)
    extensionless_file_name, _ = os.path.splitext(pathless_file_name)
    time_string = extensionless_file_name.split('_')[-1]
    return time_conversion.string_to_unix_sec(
        time_string, TIME_FORMAT_IN_FILE_NAMES)


def find_file(unix_time_sec, top_directory_name, raise_error_if_missing=True):
    """Finds GridRad file on local machine.

    Each GridRad file contains all fields at all heights for one valid time.

    :param unix_time_sec: Valid time.
    :param top_directory_name: Name of top-level directory with GridRad.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        raise_error_if_missing = True, will raise error.  If file is missing and
        raise_error_if_missing = False, will return *expected* path to file.
    :return: gridrad_file_name: Path to GridRad file.
    :raises: ValueError: if raise_error_if_missing = True and file is missing.
    """

    error_checking.assert_is_string(top_directory_name)

    spc_date_string = time_conversion.time_to_spc_date_string(unix_time_sec)
    gridrad_file_name = '{0:s}/{1:s}/{2:s}/{3:s}'.format(
        top_directory_name, spc_date_string[:4], spc_date_string,
        _get_pathless_file_name(unix_time_sec)
    )

    if raise_error_if_missing and not os.path.isfile(gridrad_file_name):
        error_string = (
            'Cannot find GridRad file.  Expected at: "{0:s}"'
        ).format(gridrad_file_name)

        raise ValueError(error_string)

    return gridrad_file_name


def read_metadata_from_full_grid_file(
        netcdf_file_name, raise_error_if_fails=True):
    """Reads metadata from full-grid (not sparse-grid) file.

    This file should contain all radar variables for one time step.

    :param netcdf_file_name: Path to input file.
    :param raise_error_if_fails: Boolean flag.  If True and file cannot be
        opened, this method will raise an error.  If False and file cannot be
        opened, will return None.
    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict['nw_grid_point_lat_deg']: Latitude (deg N) of northwesternmost
        grid point.
    metadata_dict['nw_grid_point_lng_deg']: Longitude (deg E) of
        northwesternmost grid point.
    metadata_dict['lat_spacing_deg']: Spacing (deg N) between adjacent rows.
    metadata_dict['lng_spacing_deg']: Spacing (deg E) between adjacent columns.
    metadata_dict['num_lat_in_grid']: Number of rows (unique grid-point
        latitudes).
    metadata_dict['num_lng_in_grid']: Number of columns (unique grid-point
        longitudes).
    metadata_dict['unix_time_sec']: Valid time.
    """

    error_checking.assert_file_exists(netcdf_file_name)
    netcdf_dataset = netcdf_io.open_netcdf(
        netcdf_file_name, raise_error_if_fails)
    if netcdf_dataset is None:
        return None

    grid_point_latitudes_deg = numpy.array(
        netcdf_dataset.variables[LATITUDE_NAME_ORIG])
    grid_point_longitudes_deg = lng_conversion.convert_lng_positive_in_west(
        numpy.array(netcdf_dataset.variables[LONGITUDE_NAME_ORIG]))

    metadata_dict = {
        radar_utils.NW_GRID_POINT_LAT_COLUMN:
            numpy.max(grid_point_latitudes_deg),
        radar_utils.NW_GRID_POINT_LNG_COLUMN:
            numpy.min(grid_point_longitudes_deg),
        radar_utils.LAT_SPACING_COLUMN:
            numpy.mean(numpy.diff(grid_point_latitudes_deg)),
        radar_utils.LNG_SPACING_COLUMN:
            numpy.mean(numpy.diff(grid_point_longitudes_deg)),
        radar_utils.NUM_LAT_COLUMN: len(grid_point_latitudes_deg),
        radar_utils.NUM_LNG_COLUMN: len(grid_point_longitudes_deg),
        radar_utils.UNIX_TIME_COLUMN: _time_from_gridrad_to_unix(
            netcdf_dataset.variables[TIME_NAME_ORIG][0])
    }

    netcdf_dataset.close()
    return metadata_dict


def read_field_from_full_grid_file(
        netcdf_file_name, field_name=None, metadata_dict=None,
        raise_error_if_fails=True):
    """Reads one radar field from full-grid (not sparse-grid) file.

    This file should contain all radar variables for one time step.

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)
    H = number of height levels (unique grid-point heights)

    :param netcdf_file_name: Path to input file.
    :param field_name: Name of radar field.
    :param metadata_dict: Dictionary created by
        read_metadata_from_full_grid_file.
    :param raise_error_if_fails: Boolean flag.  If True and file cannot be
        opened, this method will raise an error.  If False and file cannot be
        opened, will return None for all output variables.
    :return: field_matrix: H-by-M-by-N numpy array with values of radar field.
    :return: grid_point_heights_m_asl: length-H numpy array of height levels
        (integer metres above sea level).  If array is increasing (decreasing),
        height increases (decreases) with the first index of field_matrix.
    :return: grid_point_latitudes_deg: length-M numpy array of grid-point
        latitudes (deg N).  If array is increasing (decreasing), latitude
        increases (decreases) with the second index of field_matrix.
    :return: grid_point_longitudes_deg: length-N numpy array of grid-point
        longitudes (deg N).  If array is increasing (decreasing), latitude
        increases (decreases) with the third index of field_matrix.
    """

    error_checking.assert_file_exists(netcdf_file_name)
    netcdf_dataset = netcdf_io.open_netcdf(
        netcdf_file_name, raise_error_if_fails)
    if netcdf_dataset is None:
        return None, None, None, None

    field_name_orig = radar_utils.field_name_new_to_orig(
        field_name, data_source=radar_utils.GRIDRAD_SOURCE_ID)
    field_matrix = numpy.array(
        netcdf_dataset.variables[field_name_orig][0, :, :, :])

    grid_point_latitudes_deg = numpy.array(
        netcdf_dataset.variables[LATITUDE_NAME_ORIG])
    grid_point_longitudes_deg = lng_conversion.convert_lng_positive_in_west(
        numpy.array(netcdf_dataset.variables[LONGITUDE_NAME_ORIG]))

    _check_grid_points(
        grid_point_latitudes_deg=grid_point_latitudes_deg,
        grid_point_longitudes_deg=grid_point_longitudes_deg,
        metadata_dict=metadata_dict)

    grid_point_heights_m_asl = KM_TO_METRES * numpy.array(
        netcdf_dataset.variables[HEIGHT_NAME_ORIG])
    grid_point_heights_m_asl = numpy.round(grid_point_heights_m_asl).astype(int)

    netcdf_dataset.close()
    return (field_matrix, grid_point_heights_m_asl, grid_point_latitudes_deg,
            grid_point_longitudes_deg)
