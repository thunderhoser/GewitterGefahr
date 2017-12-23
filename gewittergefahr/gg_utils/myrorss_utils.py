"""Processing methods for MYRORSS data.

MYRORSS = Multi-year Reanalysis of Remotely Sensed Storms (Ortega et al. 2012)

--- REFERENCES ---

Ortega, K., and Coauthors, 2012: "The multi-year reanalysis of remotely sensed
    storms (MYRORSS) project". Conference on Severe Local Storms, Nashville, TN,
    American Meteorological Society.
"""

import numpy
from gewittergefahr.gg_io import radar_io
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import gridrad_utils
from gewittergefahr.gg_utils import radar_sparse_to_full as radar_s2f
from gewittergefahr.gg_utils import error_checking

DEFAULT_TOP_INPUT_HEIGHT_FOR_ECHO_TOPS_M_ASL = 20000


def get_echo_tops(unix_time_sec, spc_date_string, top_directory_name,
                  critical_reflectivity_dbz, top_height_to_consider_m_asl=
                  DEFAULT_TOP_INPUT_HEIGHT_FOR_ECHO_TOPS_M_ASL):
    """Finds echo top at each horizontal location.

    "Echo top" is max height with reflectivity >= critical reflectivity.

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)

    :param unix_time_sec: Valid time.
    :param spc_date_string: SPC date (format "yyyymmdd").
    :param top_directory_name: Name of top-level directory with MYRORSS files.
    :param critical_reflectivity_dbz: Critical reflectivity (used to define echo
        top).
    :param top_height_to_consider_m_asl: Top height level to consider (metres
        above sea level).
    :return: echo_top_matrix_m_asl: M-by-N matrix of echo tops (metres above sea
        level).
    """

    error_checking.assert_is_greater(critical_reflectivity_dbz, 0.)
    error_checking.assert_is_integer(top_height_to_consider_m_asl)
    error_checking.assert_is_greater(top_height_to_consider_m_asl, 0)

    unique_grid_point_heights_m_asl = radar_io.get_valid_heights_for_field(
        radar_io.REFL_NAME, data_source=radar_io.MYRORSS_SOURCE_ID)
    unique_grid_point_heights_m_asl = unique_grid_point_heights_m_asl[
        unique_grid_point_heights_m_asl <= top_height_to_consider_m_asl]
    spc_date_unix_sec = time_conversion.spc_date_string_to_unix_sec(
        spc_date_string)

    column_max_refl_file_name = radar_io.find_raw_file(
        unix_time_sec=unix_time_sec, spc_date_unix_sec=spc_date_unix_sec,
        field_name=radar_io.REFL_COLUMN_MAX_NAME,
        data_source=radar_io.MYRORSS_SOURCE_ID,
        top_directory_name=top_directory_name)

    num_grid_heights = len(unique_grid_point_heights_m_asl)
    single_height_refl_file_names = [''] * num_grid_heights
    for k in range(num_grid_heights):
        single_height_refl_file_names[k] = radar_io.find_raw_file(
            unix_time_sec=unix_time_sec, spc_date_unix_sec=spc_date_unix_sec,
            field_name=radar_io.REFL_NAME,
            height_m_agl=unique_grid_point_heights_m_asl[k],
            data_source=radar_io.MYRORSS_SOURCE_ID,
            top_directory_name=top_directory_name)

    print 'Reading "{0:s}" for echo-top calculation...'.format(
        column_max_refl_file_name)

    this_metadata_dict = radar_io.read_metadata_from_raw_file(
        column_max_refl_file_name, data_source=radar_io.MYRORSS_SOURCE_ID)
    this_sparse_grid_table = radar_io.read_data_from_sparse_grid_file(
        column_max_refl_file_name,
        field_name_orig=this_metadata_dict[radar_io.FIELD_NAME_COLUMN_ORIG],
        data_source=radar_io.MYRORSS_SOURCE_ID,
        sentinel_values=this_metadata_dict[radar_io.SENTINEL_VALUE_COLUMN])

    (column_max_refl_matrix_dbz,
     unique_grid_point_lats_deg,
     unique_grid_point_lngs_deg) = radar_s2f.sparse_to_full_grid(
         this_sparse_grid_table, this_metadata_dict)

    num_grid_rows = len(unique_grid_point_lats_deg)
    num_grid_columns = len(unique_grid_point_lngs_deg)
    linear_indices_to_consider = numpy.where(numpy.reshape(
        column_max_refl_matrix_dbz, num_grid_rows * num_grid_columns) >=
                                             critical_reflectivity_dbz)[0]

    print ('Echo-top calculation is needed at only {0:d}/{1:d} horizontal grid '
           'points!').format(len(linear_indices_to_consider),
                             num_grid_rows * num_grid_columns)

    echo_top_matrix_m_asl = numpy.full(
        (num_grid_rows, num_grid_columns), numpy.nan)
    num_horiz_points_to_consider = len(linear_indices_to_consider)
    if num_horiz_points_to_consider == 0:
        return echo_top_matrix_m_asl

    grid_rows_to_consider, grid_columns_to_consider = numpy.unravel_index(
        linear_indices_to_consider, (num_grid_rows, num_grid_columns))
    reflectivity_matrix_dbz = numpy.full(
        (num_grid_heights, num_horiz_points_to_consider), numpy.nan)

    for k in range(num_grid_heights):
        print 'Reading "{0:s}" for echo-top calculation...'.format(
            single_height_refl_file_names[k])

        this_metadata_dict = radar_io.read_metadata_from_raw_file(
            single_height_refl_file_names[k],
            data_source=radar_io.MYRORSS_SOURCE_ID)
        this_sparse_grid_table = radar_io.read_data_from_sparse_grid_file(
            single_height_refl_file_names[k],
            field_name_orig=this_metadata_dict[radar_io.FIELD_NAME_COLUMN_ORIG],
            data_source=radar_io.MYRORSS_SOURCE_ID,
            sentinel_values=this_metadata_dict[radar_io.SENTINEL_VALUE_COLUMN])

        this_reflectivity_matrix_dbz, _, _ = radar_s2f.sparse_to_full_grid(
            this_sparse_grid_table, this_metadata_dict)
        reflectivity_matrix_dbz[k, :] = this_reflectivity_matrix_dbz[
            grid_rows_to_consider, grid_columns_to_consider]

    print 'Computing echo tops at the {0:d} horizontal grid points...'.format(
        num_horiz_points_to_consider)

    for i in range(num_horiz_points_to_consider):
        echo_top_matrix_m_asl[
            grid_rows_to_consider[i], grid_columns_to_consider[i]] = (
                gridrad_utils.get_echo_top_single_column(
                    reflectivities_dbz=reflectivity_matrix_dbz[:, i],
                    heights_m_asl=unique_grid_point_heights_m_asl,
                    critical_reflectivity_dbz=critical_reflectivity_dbz))

    return echo_top_matrix_m_asl
