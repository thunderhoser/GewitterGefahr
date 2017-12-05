"""Processing methods for GridRad data.

--- DEFINITIONS ---

WDSS-II = Warning Decision Support System -- Integrated Information = a software
package for the analysis and visualization of thunderstorm-related data
(Lakshmanan et al. 2007).

--- REFERENCES ---

http://gridrad.org

Lakshmanan, V., T. Smith, G. Stumpf, and K. Hondl, 2007: The Warning Decision
    Support System -- Integrated Information. Weather and Forecasting, 22 (3),
    596-612.
"""

import numpy
import pandas
import scipy.interpolate
from netCDF4 import Dataset
from gewittergefahr.gg_io import grib_io
from gewittergefahr.gg_io import radar_io
from gewittergefahr.gg_utils import interp
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

GRID_SPACING_MULTIPLE_DEG = 0.01
METRES_TO_KM = 0.001


def _get_field_name_for_echo_tops(critical_reflectivity_dbz,
                                  myrorss_format=False):
    """Creates field name for echo tops.

    :param critical_reflectivity_dbz: Critical reflectivity for echo tops.
    :param myrorss_format: Boolean flag.  If True, field name will be in MYRORSS
        format.  If False, will be in GewitterGefahr format.
    :return: field_name: Field name for echo tops.
    """

    if myrorss_format:
        field_name_for_18dbz_tops = radar_io.ECHO_TOP_18DBZ_NAME_ORIG
    else:
        field_name_for_18dbz_tops = radar_io.ECHO_TOP_18DBZ_NAME

    return field_name_for_18dbz_tops.replace(
        '18', '{0:.1f}'.format(critical_reflectivity_dbz))


def _get_echo_top_single_column(
        reflectivities_dbz=None, heights_m_asl=None,
        critical_reflectivity_dbz=None):
    """Finds echo top for a single column (horizontal location).

    "Echo top" = maximum height with reflectivity >= critical value.

    H = number of heights

    :param reflectivities_dbz: length-H numpy array of reflectivities.
    :param heights_m_asl: length-H numpy array of heights (metres above sea
        level).  This method assumes that heights are sorted in ascending order.
    :param critical_reflectivity_dbz: Critical reflectivity.
    :return: echo_top_m_asl: Echo top.
    """

    critical_flags = reflectivities_dbz >= critical_reflectivity_dbz
    if not numpy.any(critical_flags):
        return numpy.nan

    critical_indices = numpy.where(critical_flags)[0]
    highest_critical_index = critical_indices[-1]

    subcritical_indices = numpy.where(
        reflectivities_dbz < critical_reflectivity_dbz)[0]
    subcritical_indices = subcritical_indices[
        subcritical_indices > highest_critical_index]

    if not subcritical_indices:
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


def interp_temperature_sfc_from_nwp(
        radar_grid_point_lats_deg=None, radar_grid_point_lngs_deg=None,
        unix_time_sec=None, temperature_kelvins=None, model_name=None,
        grid_id=None, top_grib_directory_name=None,
        wgrib_exe_name=grib_io.WGRIB_EXE_NAME_DEFAULT,
        wgrib2_exe_name=grib_io.WGRIB2_EXE_NAME_DEFAULT):
    """Interpolates temperature isosurface from NWP model.

    M = number of rows (unique grid-point latitudes) in radar grid
    N = number of columns (unique grid-point longitudes) in radar grid

    :param radar_grid_point_lats_deg: length-M numpy array of grid-point
        latitudes (deg N).
    :param radar_grid_point_lngs_deg: length-N numpy array of grid-point
        longitudes (deg E).
    :param unix_time_sec: Target time.
    :param temperature_kelvins: Target temperature.
    :param model_name: Name of model.
    :param grid_id: String ID for model grid.
    :param top_grib_directory_name: Name of top-level directory with grib files
        for the given model.
    :param wgrib_exe_name: Path to wgrib executable.
    :param wgrib2_exe_name: Path to wgrib2 executable.
    :return: isosurface_height_matrix_m_asl: length-Q numpy array with heights
        of temperature isosurface (metres above sea level).
    """

    error_checking.assert_is_numpy_array(
        radar_grid_point_lats_deg, num_dimensions=1)
    error_checking.assert_is_valid_lat_numpy_array(radar_grid_point_lats_deg)

    error_checking.assert_is_numpy_array(
        radar_grid_point_lngs_deg, num_dimensions=1)
    lng_conversion.convert_lng_positive_in_west(
        radar_grid_point_lngs_deg, allow_nan=False)

    radar_lat_matrix_deg, radar_lng_matrix_deg = (
        grids.latlng_vectors_to_matrices(
            radar_grid_point_lats_deg, radar_grid_point_lngs_deg))

    num_grid_rows = len(radar_grid_point_lats_deg)
    num_grid_columns = len(radar_grid_point_lngs_deg)
    radar_lat_vector_deg = numpy.reshape(
        radar_lat_matrix_deg, num_grid_rows * num_grid_columns)
    radar_lng_vector_deg = numpy.reshape(
        radar_lng_matrix_deg, num_grid_rows * num_grid_columns)

    query_point_dict = {interp.QUERY_LAT_COLUMN: radar_lat_vector_deg,
                        interp.QUERY_LNG_COLUMN: radar_lng_vector_deg}
    query_point_table = pandas.DataFrame.from_dict(query_point_dict)

    isosurface_height_vector_m_asl = interp.interp_temperature_sfc_from_nwp(
        query_point_table, unix_time_sec=unix_time_sec,
        temperature_kelvins=temperature_kelvins, model_name=model_name,
        grid_id=grid_id, top_grib_directory_name=top_grib_directory_name,
        wgrib_exe_name=wgrib_exe_name, wgrib2_exe_name=wgrib2_exe_name,
        raise_error_if_missing=True)

    return numpy.reshape(
        isosurface_height_vector_m_asl, (num_grid_rows, num_grid_columns))


def interp_reflectivity_to_heights(
        reflectivity_matrix_dbz=None, unique_grid_point_heights_m_asl=None,
        target_height_matrix_m_asl=None):
    """At each horizontal location, interpolates reflectivity to target height.

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)
    H = number of height levels (unique grid-point heights)

    :param reflectivity_matrix_dbz: H-by-M-by-N matrix of reflectivities.
    :param unique_grid_point_heights_m_asl: length-H numpy array of grid-point
        heights (metres above sea level).  If array is increasing
        (decreasing), height increases (decreases) with the first index of
        reflectivity_matrix_dbz.
    :param target_height_matrix_m_asl: M-by-N matrix of target heights (metres
        above sea level).
    :return: interp_refl_matrix_dbz: M-by-N numpy array of interpolated
        reflectivities.  interp_refl_matrix_dbz[i, j] is the reflectivity
        interpolated to target_height_matrix_m_asl[i, j].
    """

    error_checking.assert_is_numpy_array(
        reflectivity_matrix_dbz, num_dimensions=3)
    error_checking.assert_is_real_numpy_array(reflectivity_matrix_dbz)

    num_grid_heights = reflectivity_matrix_dbz.shape[0]
    num_grid_rows = reflectivity_matrix_dbz.shape[1]
    num_grid_columns = reflectivity_matrix_dbz.shape[2]

    error_checking.assert_is_numpy_array(
        unique_grid_point_heights_m_asl,
        exact_dimensions=numpy.array([num_grid_heights]))
    error_checking.assert_is_geq_numpy_array(
        unique_grid_point_heights_m_asl, 0.)

    error_checking.assert_is_numpy_array(
        target_height_matrix_m_asl,
        exact_dimensions=numpy.array([num_grid_rows, num_grid_columns]))
    error_checking.assert_is_real_numpy_array(target_height_matrix_m_asl)

    interp_refl_matrix_dbz = numpy.full(
        (num_grid_rows, num_grid_columns), numpy.nan)
    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            if numpy.isnan(target_height_matrix_m_asl[i, j]):
                continue

            these_reflectivities_dbz = reflectivity_matrix_dbz[:, i, j]
            these_real_indices = numpy.where(
                numpy.invert(numpy.isnan(these_reflectivities_dbz)))[0]
            if len(these_real_indices) < 2:
                continue

            interp_object = scipy.interpolate.interp1d(
                unique_grid_point_heights_m_asl[these_real_indices],
                these_reflectivities_dbz[these_real_indices], kind='linear',
                bounds_error=False, fill_value='extrapolate',
                assume_sorted=True)
            interp_refl_matrix_dbz[i, j] = interp_object(
                target_height_matrix_m_asl[i, j])

    return interp_refl_matrix_dbz


def get_column_max_reflectivity(reflectivity_matrix_dbz):
    """Finds column-max reflectivity at each horizontal location.

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)
    H = number of height levels (unique grid-point heights)

    :param reflectivity_matrix_dbz: H-by-M-by-N matrix of reflectivities.
    :return: column_max_refl_matrix_dbz: M-by-N matrix of column-max
        reflectivities.
    """

    error_checking.assert_is_numpy_array(
        reflectivity_matrix_dbz, num_dimensions=3)
    error_checking.assert_is_real_numpy_array(reflectivity_matrix_dbz)

    return numpy.nanmax(reflectivity_matrix_dbz, axis=0)


def get_echo_tops(
        reflectivity_matrix_dbz=None, unique_grid_point_heights_m_asl=None,
        critical_reflectivity_dbz=None):
    """Finds echo top at each horizontal location.

    "Echo top" = maximum height with >= critical reflectivity.

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)
    H = number of height levels (unique grid-point heights)

    :param reflectivity_matrix_dbz: H-by-M-by-N matrix of reflectivities.
    :param unique_grid_point_heights_m_asl: length-H numpy array of grid-point
        heights (metres above sea level).  Must be sorted in ascending order,
        which means that height must increase with the first index of
        reflectivity_matrix_dbz.
    :param critical_reflectivity_dbz: Critical reflectivity.
    :return: echo_top_matrix_m_asl: M-by-N matrix of echo tops (metres above sea
        level).
    :raises: ValueError: unique_grid_point_heights_m_asl not sorted in ascending
        order.
    """

    error_checking.assert_is_numpy_array(
        reflectivity_matrix_dbz, num_dimensions=3)
    error_checking.assert_is_real_numpy_array(reflectivity_matrix_dbz)
    error_checking.assert_is_greater(critical_reflectivity_dbz, 0.)

    num_grid_heights = reflectivity_matrix_dbz.shape[0]
    num_grid_rows = reflectivity_matrix_dbz.shape[1]
    num_grid_columns = reflectivity_matrix_dbz.shape[2]

    error_checking.assert_is_numpy_array(
        unique_grid_point_heights_m_asl,
        exact_dimensions=numpy.array([num_grid_heights]))
    error_checking.assert_is_geq_numpy_array(
        unique_grid_point_heights_m_asl, 0.)

    sorted_heights_m_asl = numpy.sort(unique_grid_point_heights_m_asl)
    if not numpy.array_equal(sorted_heights_m_asl,
                             unique_grid_point_heights_m_asl):
        raise ValueError('unique_grid_point_heights_m_asl are not sorted in '
                         'ascending order.')

    echo_top_matrix_m_asl = numpy.full(
        (num_grid_rows, num_grid_columns), numpy.nan)
    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            echo_top_matrix_m_asl[i, j] = _get_echo_top_single_column(
                reflectivities_dbz=reflectivity_matrix_dbz[:, i, j],
                heights_m_asl=unique_grid_point_heights_m_asl,
                critical_reflectivity_dbz=critical_reflectivity_dbz)

    return echo_top_matrix_m_asl


def write_field_to_myrorss_file(
        field_matrix=None, netcdf_file_name=None, field_name=None,
        radar_height_m_asl=None, echo_top_level_dbz=None, metadata_dict=None):
    """Writes field to file in MYRORSS format.

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)

    :param field_matrix: M-by-N numpy array, with values of one variable at one
        time.  Latitude should increase in the positive direction of the first
        axis (down the rows), and longitude should increase in the positive
        direction of the second axis (along the columns).
    :param netcdf_file_name: Path to output file.
    :param field_name: Name of radar field in GewitterGefahr format.
    :param radar_height_m_asl: Height of radar field (metres above sea level).
    :param echo_top_level_dbz: Critical reflectivity for echo tops.  Valid only
        if `field_name in radar_io.ECHO_TOP_NAMES`.
    :param metadata_dict: See documentation for
        `gridrad_io.read_metadata_from_full_grid_file`.
    """

    # TODO(thunderhoser): Current method for dealing with echo tops at non-
    # standard levels (other than 18 and 50 dBZ) is hacky.  I should eventually
    # change radar_io.py to handle echo tops at any level.

    if field_name == radar_io.REFL_NAME:
        field_to_heights_dict_m_asl = radar_io.field_and_height_arrays_to_dict(
            [field_name], refl_heights_m_agl=numpy.array([radar_height_m_asl]),
            data_source=radar_io.MYRORSS_SOURCE_ID)
    else:
        field_to_heights_dict_m_asl = radar_io.field_and_height_arrays_to_dict(
            [field_name], refl_heights_m_agl=None,
            data_source=radar_io.MYRORSS_SOURCE_ID)

    field_name = field_to_heights_dict_m_asl.keys()[0]
    radar_height_m_asl = field_to_heights_dict_m_asl[field_name][0]

    if field_name in radar_io.ECHO_TOP_NAMES:
        field_matrix = METRES_TO_KM * field_matrix

        error_checking.assert_is_greater(echo_top_level_dbz, 0.)
        field_name = _get_field_name_for_echo_tops(echo_top_level_dbz, False)
        field_name_myrorss = _get_field_name_for_echo_tops(
            echo_top_level_dbz, True)
    else:
        field_name_myrorss = radar_io.field_name_new_to_orig(
            field_name, radar_io.MYRORSS_SOURCE_ID)

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
        field_name_myrorss, numpy.single, (radar_io.NUM_PIXELS_COLUMN_ORIG, ))
    netcdf_dataset.createVariable(
        radar_io.GRID_ROW_COLUMN_ORIG, numpy.int16,
        (radar_io.NUM_PIXELS_COLUMN_ORIG, ))
    netcdf_dataset.createVariable(
        radar_io.GRID_COLUMN_COLUMN_ORIG, numpy.int16,
        (radar_io.NUM_PIXELS_COLUMN_ORIG, ))
    netcdf_dataset.createVariable(
        radar_io.NUM_GRID_CELL_COLUMN_ORIG, numpy.int32,
        (radar_io.NUM_PIXELS_COLUMN_ORIG, ))

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
