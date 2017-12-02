"""IO methods for GridRad data.

--- REFERENCE ---

http://gridrad.org
"""

import numpy
from gewittergefahr.gg_io import netcdf_io
from gewittergefahr.gg_io import radar_io
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking

MIN_GRID_POINT_HEIGHT_COLUMN = 'lowest_grid_point_height_m_asl'
HEIGHT_SPACING_COLUMN = 'height_spacing_metres'
NUM_HEIGHTS_COLUMN = 'num_heights_in_grid'

LATITUDE_NAME_ORIG = 'Latitude'
LONGITUDE_NAME_ORIG = 'Longitude'
HEIGHT_NAME_ORIG = 'Altitude'
TIME_NAME_ORIG = 'time'
REFL_NAME_ORIG = 'ZH'
DIFFERENTIAL_REFL_NAME_ORIG = 'ZDR'
SPEC_DIFF_PHASE_NAME_ORIG = 'KDP'
CORRELATION_COEFF_NAME_ORIG = 'RHV'
SPECTRUM_WIDTH_NAME_ORIG = 'SW'
VORTICITY_NAME_ORIG = 'VOR'
DIVERGENCE_NAME_ORIG = 'DIV'

REFL_NAME = 'reflectivity_dbz'
DIFFERENTIAL_REFL_NAME = 'differential_reflectivity_db'
SPEC_DIFF_PHASE_NAME = 'specific_differential_phase_deg_km01'
CORRELATION_COEFF_NAME = 'correlation_coefficient'
SPECTRUM_WIDTH_NAME = 'spectrum_width_m_s01'
VORTICITY_NAME = 'vorticity_s01'
DIVERGENCE_NAME = 'divergence_s01'

RADAR_FIELD_NAMES = [
    REFL_NAME, DIFFERENTIAL_REFL_NAME, SPEC_DIFF_PHASE_NAME,
    CORRELATION_COEFF_NAME, SPECTRUM_WIDTH_NAME, VORTICITY_NAME,
    DIVERGENCE_NAME]
RADAR_FIELD_NAMES_ORIG = [
    REFL_NAME_ORIG, DIFFERENTIAL_REFL_NAME_ORIG, SPEC_DIFF_PHASE_NAME_ORIG,
    CORRELATION_COEFF_NAME_ORIG, SPECTRUM_WIDTH_NAME_ORIG, VORTICITY_NAME_ORIG,
    DIVERGENCE_NAME_ORIG]

ZERO_TIME_UNIX_SEC = 978307200  # 0000 UTC 1 Jan 2001


def _check_field_name(field_name):
    """Ensures that name of radar field is valid.

    :param field_name: Name of radar field in GewitterGefahr (not GridRad)
        format.
    :raises: ValueError: if name of radar field is invalid.
    """

    error_checking.assert_is_string(field_name)
    if field_name not in RADAR_FIELD_NAMES:
        error_string = (
            '\n\n' + str(RADAR_FIELD_NAMES) +
            '\n\nValid field names (listed above) do not include "' +
            field_name + '".')
        raise ValueError(error_string)


def _field_name_new_to_orig(field_name):
    """Converts field name from new (GewitterGefahr) to orig (GridRad) format.

    :param field_name: Name of radar field in GewitterGefahr format.
    :return: field_name_orig: Field name in GridRad format.
    """

    _check_field_name(field_name)
    found_flags = [s == field_name for s in RADAR_FIELD_NAMES]
    return RADAR_FIELD_NAMES_ORIG[numpy.where(found_flags)[0][0]]


def _time_from_gridrad_to_unix(gridrad_time_sec):
    """Converts time from GridRad format to Unix format.

    GridRad format = seconds since 0000 UTC 1 Jan 2001
    Unix format = seconds since 0000 UTC 1 Jan 1970

    :param gridrad_time_sec: Time in GridRad format.
    :return: unix_time_sec: Time in Unix format.
    """

    return gridrad_time_sec + ZERO_TIME_UNIX_SEC


def read_metadata_from_full_grid_file(netcdf_file_name,
                                      raise_error_if_fails=True):
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
    metadata_dict['lowest_grid_point_height_m_asl']: Height (metres above sea
        level) of lowest grid point.
    metadata_dict['lat_spacing_deg']: Spacing (deg N) between adjacent rows.
    metadata_dict['lng_spacing_deg']: Spacing (deg E) between adjacent columns.
    metadata_dict['height_spacing_metres']: Spacing between adjacent height
        levels.
    metadata_dict['num_lat_in_grid']: Number of rows (unique grid-point
        latitudes).
    metadata_dict['num_lng_in_grid']: Number of columns (unique grid-point
        longitudes).
    metadata_dict['num_heights_in_grid']: Number of height levels (unique grid
        point heights).
    metadata_dict['unix_time_sec']: Valid time.
    """

    error_checking.assert_file_exists(netcdf_file_name)
    netcdf_dataset = netcdf_io.open_netcdf(netcdf_file_name,
                                           raise_error_if_fails)
    if netcdf_dataset is None:
        return None

    grid_point_latitudes_deg = netcdf_dataset.variables[LATITUDE_NAME_ORIG]
    grid_point_longitudes_deg = netcdf_dataset.variables[LONGITUDE_NAME_ORIG]
    grid_point_heights_m_asl = netcdf_dataset.variables[HEIGHT_NAME_ORIG]

    return {
        radar_io.NW_GRID_POINT_LAT_COLUMN: numpy.max(grid_point_latitudes_deg),
        radar_io.NW_GRID_POINT_LNG_COLUMN:
            lng_conversion.convert_lng_positive_in_west(
                numpy.min(grid_point_longitudes_deg)),
        MIN_GRID_POINT_HEIGHT_COLUMN: numpy.min(grid_point_heights_m_asl),
        radar_io.LAT_SPACING_COLUMN: numpy.absolute(
            grid_point_latitudes_deg[1] - grid_point_latitudes_deg[0]),
        radar_io.LNG_SPACING_COLUMN: numpy.absolute(
            grid_point_longitudes_deg[1] - grid_point_longitudes_deg[0]),
        HEIGHT_SPACING_COLUMN: numpy.absolute(
            grid_point_heights_m_asl[1] - grid_point_heights_m_asl[0]),
        radar_io.NUM_LAT_COLUMN: len(grid_point_latitudes_deg),
        radar_io.NUM_LNG_COLUMN: len(grid_point_longitudes_deg),
        NUM_HEIGHTS_COLUMN: len(grid_point_heights_m_asl),
        radar_io.UNIX_TIME_COLUMN: _time_from_gridrad_to_unix(
            netcdf_dataset.variables[TIME_NAME_ORIG][0])
    }


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
    :return: unique_grid_point_heights_m_asl: length-N numpy array of grid-point
        heights (metres above sea level).  If array is increasing
        (decreasing), height increases (decreases) with the first index of
        field_matrix.
    :return: unique_grid_point_lat_deg: length-M numpy array of grid-point
        latitudes (deg N).  If array is increasing (decreasing), latitude
        increases (decreases) with the second index of field_matrix.
    :return: unique_grid_point_lng_deg: length-N numpy array of grid-point
        longitudes (deg E).  If array is increasing (decreasing), longitude
        increases (decreases) with the third index of field_matrix.
    """

    error_checking.assert_file_exists(netcdf_file_name)
    netcdf_dataset = netcdf_io.open_netcdf(netcdf_file_name,
                                           raise_error_if_fails)
    if netcdf_dataset is None:
        return None, None, None, None

    field_name_orig = _field_name_new_to_orig(field_name)
    field_matrix = numpy.array(
        netcdf_dataset.variables[field_name_orig][0, :, :, :])

    min_latitude_deg = metadata_dict[radar_io.NW_GRID_POINT_LAT_COLUMN] - (
        metadata_dict[radar_io.LAT_SPACING_COLUMN] *
        (metadata_dict[radar_io.NUM_LAT_COLUMN] - 1))
    unique_grid_point_lat_deg, unique_grid_point_lng_deg = (
        grids.get_latlng_grid_points(
            min_latitude_deg=min_latitude_deg,
            min_longitude_deg=metadata_dict[radar_io.NW_GRID_POINT_LNG_COLUMN],
            lat_spacing_deg=metadata_dict[radar_io.LAT_SPACING_COLUMN],
            lng_spacing_deg=metadata_dict[radar_io.LNG_SPACING_COLUMN],
            num_rows=metadata_dict[radar_io.NUM_LAT_COLUMN],
            num_columns=metadata_dict[radar_io.NUM_LNG_COLUMN]))

    max_height_m_asl = metadata_dict[MIN_GRID_POINT_HEIGHT_COLUMN] + (
        metadata_dict[HEIGHT_SPACING_COLUMN] *
        (metadata_dict[NUM_HEIGHTS_COLUMN] - 1))
    unique_grid_point_heights_m_asl = numpy.linspace(
        metadata_dict[MIN_GRID_POINT_HEIGHT_COLUMN], max_height_m_asl,
        num=metadata_dict[NUM_HEIGHTS_COLUMN])

    for i in range(metadata_dict[NUM_HEIGHTS_COLUMN]):
        field_matrix[i, :, :] = numpy.flipud(field_matrix[i, :, :])

    return (field_matrix, unique_grid_point_heights_m_asl,
            unique_grid_point_lat_deg[::-1], unique_grid_point_lng_deg)
