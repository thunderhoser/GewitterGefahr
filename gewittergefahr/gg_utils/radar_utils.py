"""Processing methods for radar data."""

import numpy
import scipy.interpolate
from netCDF4 import Dataset
from gewittergefahr.gg_io import radar_io
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

METRES_TO_KM = 0.001
GRID_SPACING_MULTIPLE_DEG = 0.01


def get_echo_top_single_column(
        reflectivities_dbz, heights_m_asl, critical_reflectivity_dbz,
        check_args=False):
    """Finds echo top for a single column (horizontal location).

    "Echo top" = maximum height with reflectivity >= critical value.

    H = number of heights

    :param reflectivities_dbz: length-H numpy array of reflectivities.
    :param heights_m_asl: length-H numpy array of heights (metres above sea
        level).  This method assumes that heights are sorted in ascending order.
    :param critical_reflectivity_dbz: Critical reflectivity.
    :param check_args: Boolean flag.  If True, will check input arguments for
        errors.
    :return: echo_top_m_asl: Echo top.
    """

    error_checking.assert_is_boolean(check_args)
    if check_args:
        error_checking.assert_is_real_numpy_array(reflectivities_dbz)
        error_checking.assert_is_numpy_array(
            reflectivities_dbz, num_dimensions=1)

        num_heights = len(reflectivities_dbz)
        error_checking.assert_is_geq_numpy_array(heights_m_asl, 0.)
        error_checking.assert_is_numpy_array(
            heights_m_asl, exact_dimensions=numpy.array([num_heights]))

        error_checking.assert_is_greater(critical_reflectivity_dbz, 0.)

    critical_indices = numpy.where(
        reflectivities_dbz >= critical_reflectivity_dbz)[0]
    if len(critical_indices) == 0:
        return numpy.nan

    highest_critical_index = critical_indices[-1]
    subcritical_indices = numpy.where(
        reflectivities_dbz < critical_reflectivity_dbz)[0]
    subcritical_indices = subcritical_indices[
        subcritical_indices > highest_critical_index]

    if len(subcritical_indices) == 0:
        try:
            height_spacing_metres = (
                heights_m_asl[highest_critical_index + 1] -
                heights_m_asl[highest_critical_index])
        except IndexError:
            height_spacing_metres = (
                heights_m_asl[highest_critical_index] -
                heights_m_asl[highest_critical_index - 1])

        extrap_height_metres = height_spacing_metres * (
            1. - critical_reflectivity_dbz /
            reflectivities_dbz[highest_critical_index])
        return heights_m_asl[highest_critical_index] + extrap_height_metres

    adjacent_subcritical_index = subcritical_indices[0]
    indices_for_interp = numpy.array(
        [highest_critical_index, adjacent_subcritical_index], dtype=int)

    # if len(critical_indices) > 1:
    #     adjacent_critical_index = critical_indices[-2]
    #     indices_for_interp = numpy.array(
    #         [adjacent_critical_index, highest_critical_index,
    #          adjacent_subcritical_index], dtype=int)
    # else:
    #     indices_for_interp = numpy.array(
    #         [highest_critical_index, adjacent_subcritical_index], dtype=int)

    interp_object = scipy.interpolate.interp1d(
        reflectivities_dbz[indices_for_interp],
        heights_m_asl[indices_for_interp], kind='linear', bounds_error=False,
        fill_value='extrapolate', assume_sorted=False)
    return interp_object(critical_reflectivity_dbz)


def write_field_to_myrorss_file(
        field_matrix, netcdf_file_name, field_name, metadata_dict,
        height_m_asl=None):
    """Writes field to MYRORSS-formatted file.

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)

    :param field_matrix: M-by-N numpy array with one radar variable at one time.
        Latitude should increase down each column, and longitude should increase
        to the right along each row.
    :param netcdf_file_name: Path to output file.
    :param field_name: Name of radar field in GewitterGefahr format.
    :param metadata_dict: Dictionary created by either
        `gridrad_io.read_metadata_from_full_grid_file` or
        `radar_io.read_metadata_from_raw_file`.
    :param height_m_asl: Height of radar field (metres above sea level).
    """

    if field_name == radar_io.REFL_NAME:
        field_to_heights_dict_m_asl = radar_io.field_and_height_arrays_to_dict(
            field_names=[field_name],
            refl_heights_m_asl=numpy.array([height_m_asl]),
            data_source=radar_io.MYRORSS_SOURCE_ID)
    else:
        field_to_heights_dict_m_asl = radar_io.field_and_height_arrays_to_dict(
            field_names=[field_name], data_source=radar_io.MYRORSS_SOURCE_ID)

    field_name = field_to_heights_dict_m_asl.keys()[0]
    radar_height_m_asl = field_to_heights_dict_m_asl[field_name][0]

    if field_name in radar_io.ECHO_TOP_NAMES:
        field_matrix = METRES_TO_KM * field_matrix
    field_name_myrorss = radar_io.field_name_new_to_orig(
        field_name, data_source=radar_io.MYRORSS_SOURCE_ID)

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    netcdf_dataset = Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET')

    netcdf_dataset.setncattr(radar_io.FIELD_NAME_COLUMN_ORIG,
                             field_name_myrorss)
    netcdf_dataset.setncattr('DataType', 'SparseLatLonGrid')

    netcdf_dataset.setncattr(
        radar_io.NW_GRID_POINT_LAT_COLUMN_ORIG,
        metadata_dict[radar_io.NW_GRID_POINT_LAT_COLUMN])
    netcdf_dataset.setncattr(
        radar_io.NW_GRID_POINT_LNG_COLUMN_ORIG,
        lng_conversion.convert_lng_negative_in_west(
            metadata_dict[radar_io.NW_GRID_POINT_LNG_COLUMN]))
    netcdf_dataset.setncattr(
        radar_io.HEIGHT_COLUMN_ORIG,
        METRES_TO_KM * numpy.float(radar_height_m_asl))
    netcdf_dataset.setncattr(
        radar_io.UNIX_TIME_COLUMN_ORIG,
        numpy.int32(metadata_dict[radar_io.UNIX_TIME_COLUMN]))
    netcdf_dataset.setncattr('FractionalTime', 0.)

    netcdf_dataset.setncattr('attributes', ' ColorMap SubType Unit')
    netcdf_dataset.setncattr('ColorMap-unit', 'dimensionless')
    netcdf_dataset.setncattr('ColorMap-value', '')
    netcdf_dataset.setncattr('SubType-unit', 'dimensionless')
    netcdf_dataset.setncattr('SubType-value', numpy.float(radar_height_m_asl))
    netcdf_dataset.setncattr('Unit-unit', 'dimensionless')
    netcdf_dataset.setncattr('Unit-value', 'dimensionless')

    netcdf_dataset.setncattr(
        radar_io.LAT_SPACING_COLUMN_ORIG, rounder.round_to_nearest(
            metadata_dict[radar_io.LAT_SPACING_COLUMN],
            GRID_SPACING_MULTIPLE_DEG))
    netcdf_dataset.setncattr(
        radar_io.LNG_SPACING_COLUMN_ORIG, rounder.round_to_nearest(
            metadata_dict[radar_io.LNG_SPACING_COLUMN],
            GRID_SPACING_MULTIPLE_DEG))
    netcdf_dataset.setncattr(
        radar_io.SENTINEL_VALUE_COLUMNS_ORIG[0], numpy.double(-99000.))
    netcdf_dataset.setncattr(
        radar_io.SENTINEL_VALUE_COLUMNS_ORIG[1], numpy.double(-99001.))

    min_latitude_deg = metadata_dict[radar_io.NW_GRID_POINT_LAT_COLUMN] - (
        metadata_dict[radar_io.LAT_SPACING_COLUMN] *
        (metadata_dict[radar_io.NUM_LAT_COLUMN] - 1))
    unique_grid_point_lats_deg, unique_grid_point_lngs_deg = (
        grids.get_latlng_grid_points(
            min_latitude_deg=min_latitude_deg,
            min_longitude_deg=metadata_dict[radar_io.NW_GRID_POINT_LNG_COLUMN],
            lat_spacing_deg=metadata_dict[radar_io.LAT_SPACING_COLUMN],
            lng_spacing_deg=metadata_dict[radar_io.LNG_SPACING_COLUMN],
            num_rows=metadata_dict[radar_io.NUM_LAT_COLUMN],
            num_columns=metadata_dict[radar_io.NUM_LNG_COLUMN]))

    num_grid_rows = len(unique_grid_point_lats_deg)
    num_grid_columns = len(unique_grid_point_lngs_deg)
    field_vector = numpy.reshape(field_matrix, num_grid_rows * num_grid_columns)

    grid_point_lat_matrix, grid_point_lng_matrix = (
        grids.latlng_vectors_to_matrices(
            unique_grid_point_lats_deg, unique_grid_point_lngs_deg))
    grid_point_lat_vector = numpy.reshape(
        grid_point_lat_matrix, num_grid_rows * num_grid_columns)
    grid_point_lng_vector = numpy.reshape(
        grid_point_lng_matrix, num_grid_rows * num_grid_columns)

    real_value_indices = numpy.where(numpy.invert(numpy.isnan(field_vector)))[0]
    netcdf_dataset.createDimension(
        radar_io.NUM_LAT_COLUMN_ORIG, num_grid_rows - 1)
    netcdf_dataset.createDimension(
        radar_io.NUM_LNG_COLUMN_ORIG, num_grid_columns - 1)
    netcdf_dataset.createDimension(
        radar_io.NUM_PIXELS_COLUMN_ORIG, len(real_value_indices))

    row_index_vector, column_index_vector = radar_io.latlng_to_rowcol(
        grid_point_lat_vector, grid_point_lng_vector,
        nw_grid_point_lat_deg=metadata_dict[radar_io.NW_GRID_POINT_LAT_COLUMN],
        nw_grid_point_lng_deg=metadata_dict[radar_io.NW_GRID_POINT_LNG_COLUMN],
        lat_spacing_deg=metadata_dict[radar_io.LAT_SPACING_COLUMN],
        lng_spacing_deg=metadata_dict[radar_io.LNG_SPACING_COLUMN])

    netcdf_dataset.createVariable(
        field_name_myrorss, numpy.single, (radar_io.NUM_PIXELS_COLUMN_ORIG,))
    netcdf_dataset.createVariable(
        radar_io.GRID_ROW_COLUMN_ORIG, numpy.int16,
        (radar_io.NUM_PIXELS_COLUMN_ORIG,))
    netcdf_dataset.createVariable(
        radar_io.GRID_COLUMN_COLUMN_ORIG, numpy.int16,
        (radar_io.NUM_PIXELS_COLUMN_ORIG,))
    netcdf_dataset.createVariable(
        radar_io.NUM_GRID_CELL_COLUMN_ORIG, numpy.int32,
        (radar_io.NUM_PIXELS_COLUMN_ORIG,))

    netcdf_dataset.variables[field_name_myrorss].setncattr(
        'BackgroundValue', numpy.int32(-99900))
    netcdf_dataset.variables[field_name_myrorss].setncattr(
        'units', 'dimensionless')
    netcdf_dataset.variables[field_name_myrorss].setncattr(
        'NumValidRuns', numpy.int32(len(real_value_indices)))

    netcdf_dataset.variables[field_name_myrorss][:] = field_vector[
        real_value_indices]
    netcdf_dataset.variables[radar_io.GRID_ROW_COLUMN_ORIG][:] = (
        row_index_vector[real_value_indices])
    netcdf_dataset.variables[radar_io.GRID_COLUMN_COLUMN_ORIG][:] = (
        column_index_vector[real_value_indices])
    netcdf_dataset.variables[radar_io.NUM_GRID_CELL_COLUMN_ORIG][:] = (
        numpy.full(len(real_value_indices), 1, dtype=int))

    netcdf_dataset.close()
