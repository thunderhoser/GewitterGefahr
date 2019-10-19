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
from gewittergefahr.gg_io import grib_io
from gewittergefahr.gg_utils import interp
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking


def fields_and_refl_heights_to_pairs(field_names, heights_m_asl):
    """Converts unique arrays (field names and heights) to non-unique ones.

    F = number of fields
    H = number of heights
    N = F * H = number of field/height pairs

    :param field_names: length-F list with names of radar fields in
        GewitterGefahr format.
    :param heights_m_asl: length-H numpy array of heights (metres above sea
        level).
    :return: field_name_by_pair: length-N list of field names.
    :return: height_by_pair_m_asl: length-N numpy array of corresponding heights
        (metres above sea level).
    """

    error_checking.assert_is_string_list(field_names)
    error_checking.assert_is_numpy_array(
        numpy.array(field_names), num_dimensions=1)

    radar_utils.check_heights(
        data_source=radar_utils.GRIDRAD_SOURCE_ID, heights_m_asl=heights_m_asl)

    field_name_by_pair = []
    height_by_pair_m_asl = numpy.array([], dtype=int)

    for this_field_name in field_names:
        radar_utils.field_name_new_to_orig(
            field_name=this_field_name,
            data_source_name=radar_utils.GRIDRAD_SOURCE_ID)

        field_name_by_pair += [this_field_name] * len(heights_m_asl)
        height_by_pair_m_asl = numpy.concatenate((
            height_by_pair_m_asl, heights_m_asl))

    return field_name_by_pair, height_by_pair_m_asl


def interp_temperature_surface_from_nwp(
        radar_grid_point_latitudes_deg, radar_grid_point_longitudes_deg,
        radar_time_unix_sec, critical_temperature_kelvins, model_name,
        top_grib_directory_name, use_all_grids=True, grid_id=None,
        wgrib_exe_name=grib_io.WGRIB_EXE_NAME_DEFAULT,
        wgrib2_exe_name=grib_io.WGRIB2_EXE_NAME_DEFAULT):
    """Interpolates temperature (isothermal) surface from NWP model.

    M = number of rows (unique grid-point latitudes) in radar grid
    N = number of columns (unique grid-point longitudes) in radar grid

    :param radar_grid_point_latitudes_deg: length-M numpy array of grid-point
        latitudes (deg N).
    :param radar_grid_point_longitudes_deg: length-N numpy array of grid-point
        longitudes (deg E).
    :param radar_time_unix_sec: Radar time.
    :param critical_temperature_kelvins: Temperature of isosurface.
    :param model_name: See doc for `interp.interp_temperature_surface_from_nwp`.
    :param top_grib_directory_name: Same.
    :param use_all_grids: Same.
    :param grid_id: Same.
    :param wgrib_exe_name: Same.
    :param wgrib2_exe_name: Same.
    :return: isosurface_height_matrix_m_asl: M-by-N numpy array with heights of
        temperature isosurface (metres above sea level).
    """

    error_checking.assert_is_numpy_array(
        radar_grid_point_latitudes_deg, num_dimensions=1)
    error_checking.assert_is_valid_lat_numpy_array(
        radar_grid_point_latitudes_deg)

    error_checking.assert_is_numpy_array(
        radar_grid_point_longitudes_deg, num_dimensions=1)
    lng_conversion.convert_lng_positive_in_west(
        radar_grid_point_longitudes_deg, allow_nan=False)

    (radar_latitude_matrix_deg, radar_longitude_matrix_deg
    ) = grids.latlng_vectors_to_matrices(
        unique_latitudes_deg=radar_grid_point_latitudes_deg,
        unique_longitudes_deg=radar_grid_point_longitudes_deg)

    num_grid_rows = len(radar_grid_point_latitudes_deg)
    num_grid_columns = len(radar_grid_point_longitudes_deg)
    radar_latitude_vector_deg = numpy.reshape(
        radar_latitude_matrix_deg, num_grid_rows * num_grid_columns)
    radar_longitude_vector_deg = numpy.reshape(
        radar_longitude_matrix_deg, num_grid_rows * num_grid_columns)

    query_point_dict = {
        interp.QUERY_LAT_COLUMN: radar_latitude_vector_deg,
        interp.QUERY_LNG_COLUMN: radar_longitude_vector_deg
    }
    query_point_table = pandas.DataFrame.from_dict(query_point_dict)

    isosurface_height_vector_m_asl = interp.interp_temperature_surface_from_nwp(
        query_point_table=query_point_table,
        query_time_unix_sec=radar_time_unix_sec,
        critical_temperature_kelvins=critical_temperature_kelvins,
        model_name=model_name, top_grib_directory_name=top_grib_directory_name,
        use_all_grids=use_all_grids, grid_id=grid_id,
        wgrib_exe_name=wgrib_exe_name, wgrib2_exe_name=wgrib2_exe_name,
        raise_error_if_missing=True)

    return numpy.reshape(
        isosurface_height_vector_m_asl, (num_grid_rows, num_grid_columns))


def interp_reflectivity_to_heights(
        reflectivity_matrix_dbz, grid_point_heights_m_asl,
        target_height_matrix_m_asl):
    """At each horizontal location, interpolates reflectivity to target height.

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)
    H = number of height levels (unique grid-point heights)

    :param reflectivity_matrix_dbz: H-by-M-by-N matrix of reflectivities.
    :param grid_point_heights_m_asl: length-H numpy array of grid-point heights
        (metres above sea level).  If array is increasing (decreasing), height
        increases (decreases) with the first index of reflectivity_matrix_dbz.
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
        grid_point_heights_m_asl,
        exact_dimensions=numpy.array([num_grid_heights]))
    error_checking.assert_is_geq_numpy_array(
        grid_point_heights_m_asl, 0.)

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
                grid_point_heights_m_asl[these_real_indices],
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
        reflectivity_matrix_dbz, grid_point_heights_m_asl,
        critical_reflectivity_dbz):
    """Finds echo top at each horizontal location.

    "Echo top" = maximum height with >= critical reflectivity.

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)
    H = number of height levels (unique grid-point heights)

    :param reflectivity_matrix_dbz: H-by-M-by-N matrix of reflectivities.
    :param grid_point_heights_m_asl: length-H numpy array of grid-point heights
        (metres above sea level).  Must be sorted in ascending order, which
        means that height must increase with the first index of
        reflectivity_matrix_dbz.
    :param critical_reflectivity_dbz: Critical reflectivity.
    :return: echo_top_matrix_m_asl: M-by-N matrix of echo tops (metres above sea
        level).
    :raises: ValueError: grid_point_heights_m_asl not sorted in ascending
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
        grid_point_heights_m_asl,
        exact_dimensions=numpy.array([num_grid_heights]))
    error_checking.assert_is_geq_numpy_array(
        grid_point_heights_m_asl, 0.)

    sorted_heights_m_asl = numpy.sort(grid_point_heights_m_asl)
    if not numpy.array_equal(sorted_heights_m_asl,
                             grid_point_heights_m_asl):
        raise ValueError('grid_point_heights_m_asl are not sorted in '
                         'ascending order.')

    echo_top_matrix_m_asl = numpy.full(
        (num_grid_rows, num_grid_columns), numpy.nan)
    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            echo_top_matrix_m_asl[i, j] = (
                radar_utils.get_echo_top_single_column(
                    reflectivities_dbz=reflectivity_matrix_dbz[:, i, j],
                    heights_m_asl=grid_point_heights_m_asl,
                    critical_reflectivity_dbz=critical_reflectivity_dbz,
                    check_args=False))

    return echo_top_matrix_m_asl
