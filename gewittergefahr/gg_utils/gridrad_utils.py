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


def _get_field_name_for_echo_tops(critical_reflectivity_dbz,
                                  myrorss_format=False):
    """Creates field name for echo tops.

    :param critical_reflectivity_dbz: Critical reflectivity for echo tops.
    :param myrorss_format: Boolean flag.  If True, field name will be in MYRORSS
        format.  If False, will be in GewitterGefahr format.
    :return: field_name: Field name for echo tops.
    """

    # TODO(thunderhoser): probably don't need this method anymore.

    if myrorss_format:
        field_name_for_18dbz_tops = radar_utils.ECHO_TOP_18DBZ_NAME_ORIG
    else:
        field_name_for_18dbz_tops = radar_utils.ECHO_TOP_18DBZ_NAME

    return field_name_for_18dbz_tops.replace(
        '18', '{0:.1f}'.format(critical_reflectivity_dbz))


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
            echo_top_matrix_m_asl[i, j] = (
                radar_utils.get_echo_top_single_column(
                    reflectivities_dbz=reflectivity_matrix_dbz[:, i, j],
                    heights_m_asl=unique_grid_point_heights_m_asl,
                    critical_reflectivity_dbz=critical_reflectivity_dbz,
                    check_args=False))

    return echo_top_matrix_m_asl
