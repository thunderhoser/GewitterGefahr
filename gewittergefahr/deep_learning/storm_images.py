"""Methods for handling storm images.

A "storm image" is a radar image that shares a center with the storm object (in
other words, the center of the image is the centroid of the storm object).
"""

import os
import time
import copy
import glob
import numpy
import pandas
import netCDF4
from gewittergefahr.gg_io import netcdf_io
from gewittergefahr.gg_io import gridrad_io
from gewittergefahr.gg_io import myrorss_and_mrms_io
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import gridrad_utils
from gewittergefahr.gg_utils import myrorss_and_mrms_utils
from gewittergefahr.gg_utils import radar_sparse_to_full as radar_s2f
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import link_events_to_storms as events2storms
from gewittergefahr.gg_utils import labels
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import interp
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import geodetic_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

PADDING_VALUE = 0
GRID_SPACING_TOLERANCE_DEG = 1e-4
AZ_SHEAR_GRID_SPACING_MULTIPLIER = 2
LABEL_FILE_EXTENSION = '.nc'

DAYS_TO_SECONDS = 86400
GRIDRAD_TIME_INTERVAL_SEC = 300
TIME_FORMAT = '%Y-%m-%d-%H%M%S'
TIME_FORMAT_REGEX = (
    '[0-9][0-9][0-9][0-9]-[0-1][0-9]-[0-3][0-9]-[0-2][0-9][0-5][0-9][0-5][0-9]')

FIRST_STORM_ROW_KEY = 'first_storm_image_row'
LAST_STORM_ROW_KEY = 'last_storm_image_row'
FIRST_STORM_COLUMN_KEY = 'first_storm_image_column'
LAST_STORM_COLUMN_KEY = 'last_storm_image_column'
NUM_TOP_PADDING_ROWS_KEY = 'num_padding_rows_at_top'
NUM_BOTTOM_PADDING_ROWS_KEY = 'num_padding_rows_at_bottom'
NUM_LEFT_PADDING_COLS_KEY = 'num_padding_columns_at_left'
NUM_RIGHT_PADDING_COLS_KEY = 'num_padding_columns_at_right'

ROTATED_GP_LATITUDES_COLUMN = 'rotated_gp_lat_matrix_deg'
ROTATED_GP_LONGITUDES_COLUMN = 'rotated_gp_lng_matrix_deg'

STORM_IMAGE_MATRIX_KEY = 'storm_image_matrix'
STORM_IDS_KEY = 'storm_ids'
VALID_TIMES_KEY = 'valid_times_unix_sec'
RADAR_FIELD_NAME_KEY = 'radar_field_name'
RADAR_HEIGHT_KEY = 'radar_height_m_asl'
ROTATED_GRIDS_KEY = 'rotated_grids'
ROTATED_GRID_SPACING_KEY = 'rotated_grid_spacing_metres'
ROTATION_DIVERGENCE_PRODUCTS_KEY = 'rotation_divergence_products_s02'
HORIZ_RADIUS_FOR_RDP_KEY = 'horiz_radius_for_rdp_metres'
MIN_VORT_HEIGHT_FOR_RDP_KEY = 'min_vort_height_for_rdp_m_asl'

STORM_TO_WINDS_TABLE_KEY = 'storm_to_winds_table'
STORM_TO_TORNADOES_TABLE_KEY = 'storm_to_tornadoes_table'
LABEL_VALUES_KEY = 'label_values'

RADAR_FIELD_NAMES_KEY = 'radar_field_names'
RADAR_HEIGHTS_KEY = 'radar_heights_m_asl'
IMAGE_FILE_NAME_MATRIX_KEY = 'image_file_name_matrix'
RADAR_FIELD_NAME_BY_PAIR_KEY = 'field_name_by_pair'
RADAR_HEIGHT_BY_PAIR_KEY = 'height_by_pair_m_asl'

ROW_DIMENSION_KEY = 'grid_row'
COLUMN_DIMENSION_KEY = 'grid_column'
CHARACTER_DIMENSION_KEY = 'storm_id_character'
STORM_OBJECT_DIMENSION_KEY = 'storm_object'

DEFAULT_NUM_IMAGE_ROWS = 32
DEFAULT_NUM_IMAGE_COLUMNS = 32
DEFAULT_ROTATED_GRID_SPACING_METRES = 1500.
DEFAULT_HORIZ_RADIUS_FOR_RDP_METRES = 10000.
DEFAULT_MIN_VORT_HEIGHT_FOR_RDP_M_ASL = 4000

DEFAULT_RADAR_HEIGHTS_M_ASL = numpy.linspace(1000, 12000, num=12, dtype=int)

DEFAULT_MYRORSS_MRMS_FIELD_NAMES = [
    radar_utils.ECHO_TOP_18DBZ_NAME, radar_utils.ECHO_TOP_50DBZ_NAME,
    radar_utils.LOW_LEVEL_SHEAR_NAME, radar_utils.MID_LEVEL_SHEAR_NAME,
    radar_utils.REFL_NAME, radar_utils.REFL_COLUMN_MAX_NAME,
    radar_utils.REFL_0CELSIUS_NAME, radar_utils.REFL_M10CELSIUS_NAME,
    radar_utils.REFL_M20CELSIUS_NAME, radar_utils.REFL_LOWEST_ALTITUDE_NAME,
    radar_utils.MESH_NAME, radar_utils.SHI_NAME, radar_utils.VIL_NAME]
AZIMUTHAL_SHEAR_FIELD_NAMES = [
    radar_utils.LOW_LEVEL_SHEAR_NAME, radar_utils.MID_LEVEL_SHEAR_NAME]

DEFAULT_GRIDRAD_FIELD_NAMES = [
    radar_utils.REFL_NAME, radar_utils.SPECTRUM_WIDTH_NAME,
    radar_utils.VORTICITY_NAME, radar_utils.DIVERGENCE_NAME]


def _check_extraction_args(
        one_file_per_time_step, num_storm_image_rows, num_storm_image_columns,
        rotate_grids, rotated_grid_spacing_metres):
    """Checks input args for extraction of storm-centered radar images.

    Specifically, this method checks input args for
    `extract_storm_images_myrorss_or_mrms` or `extract_storm_images_gridrad`.

    :param one_file_per_time_step: Boolean flag.  If True, will write one file
        for each field/height pair and time step.  If not, will write one file
        for each field/height pair and SPC date.
    :param num_storm_image_rows: Number of rows in each storm-centered image.
        Must be even.
    :param num_storm_image_columns: Number columns in each storm-centered image.
        Must be even.
    :param rotate_grids: Boolean flag.  If True, each grid will be rotated so
        that storm motion is in the +x-direction; thus, storm-centered grids
        will be equidistant.  If False, each storm-centered grid will be a
        contiguous rectangle extracted from the full grid; thus, storm-centered
        grids will be lat-long.
    :param rotated_grid_spacing_metres: [used only if rotate_grids = True]
        Spacing between grid points in adjacent rows or columns.
    :raises: ValueError: if `num_storm_image_rows` or `num_storm_image_columns`
        is not even.
    """

    error_checking.assert_is_integer(num_storm_image_rows)
    error_checking.assert_is_greater(num_storm_image_rows, 0)
    if num_storm_image_rows != rounder.round_to_nearest(
            num_storm_image_rows, 2):
        error_string = (
            'Number of rows per storm-centered image ({0:d}) should be even.'
        ).format(num_storm_image_rows)
        raise ValueError(error_string)

    error_checking.assert_is_integer(num_storm_image_columns)
    error_checking.assert_is_greater(num_storm_image_columns, 0)
    if num_storm_image_columns != rounder.round_to_nearest(
            num_storm_image_columns, 2):
        error_string = (
            'Number of columns per storm-centered image ({0:d}) should be even.'
        ).format(num_storm_image_columns)
        raise ValueError(error_string)

    error_checking.assert_is_boolean(one_file_per_time_step)
    error_checking.assert_is_boolean(rotate_grids)
    if rotate_grids:
        error_checking.assert_is_greater(rotated_grid_spacing_metres, 0.)


def _rotate_grid_one_storm_object(
        centroid_latitude_deg, centroid_longitude_deg, eastward_motion_m_s01,
        northward_motion_m_s01, num_storm_image_rows, num_storm_image_columns,
        storm_grid_spacing_metres):
    """Generates lat-long coordinates for rotated, storm-centered grid.

    The grid is rotated so that storm motion is in the +x-direction.

    m = number of rows in storm-centered grid (must be even)
    n = number of columns in storm-centered grid (must be even)

    :param centroid_latitude_deg: Latitude (deg N) of storm centroid.
    :param centroid_longitude_deg: Longitude (deg E) of storm centroid.
    :param eastward_motion_m_s01: Eastward component of storm motion (metres per
        second).
    :param northward_motion_m_s01: Northward component of storm motion.
    :param num_storm_image_rows: m in the above discussion.
    :param num_storm_image_columns: n in the above discussion.
    :param storm_grid_spacing_metres: Spacing between grid points in adjacent
        rows or columns.
    :return: grid_point_lat_matrix_deg: m-by-n numpy array with latitudes
        (deg N) of grid points.
    :return: grid_point_lng_matrix_deg: m-by-n numpy array with longitudes
        (deg E) of grid points.
    """

    storm_bearing_deg = geodetic_utils.xy_to_scalar_displacements_and_bearings(
        x_displacements_metres=numpy.array([eastward_motion_m_s01]),
        y_displacements_metres=numpy.array([northward_motion_m_s01])
    )[-1][0]

    this_max_displacement_metres = storm_grid_spacing_metres * (
        num_storm_image_columns / 2 - 0.5)
    this_min_displacement_metres = -1 * this_max_displacement_metres
    x_prime_displacements_metres = numpy.linspace(
        this_min_displacement_metres, this_max_displacement_metres,
        num=num_storm_image_columns)

    this_max_displacement_metres = storm_grid_spacing_metres * (
        num_storm_image_rows / 2 - 0.5)
    this_min_displacement_metres = -1 * this_max_displacement_metres
    y_prime_displacements_metres = numpy.linspace(
        this_min_displacement_metres, this_max_displacement_metres,
        num=num_storm_image_rows)

    (x_prime_displ_matrix_metres, y_prime_displ_matrix_metres
    ) = grids.xy_vectors_to_matrices(
        x_unique_metres=x_prime_displacements_metres,
        y_unique_metres=y_prime_displacements_metres)

    (x_displacement_matrix_metres, y_displacement_matrix_metres
    ) = geodetic_utils.rotate_displacement_vectors(
        x_displacements_metres=x_prime_displ_matrix_metres,
        y_displacements_metres=y_prime_displ_matrix_metres,
        ccw_rotation_angle_deg=-(storm_bearing_deg - 90))

    (scalar_displacement_matrix_metres, bearing_matrix_deg
    ) = geodetic_utils.xy_to_scalar_displacements_and_bearings(
        x_displacements_metres=x_displacement_matrix_metres,
        y_displacements_metres=y_displacement_matrix_metres)

    start_latitude_matrix_deg = numpy.full(
        (num_storm_image_rows, num_storm_image_columns), centroid_latitude_deg)
    start_longitude_matrix_deg = numpy.full(
        (num_storm_image_rows, num_storm_image_columns), centroid_longitude_deg)

    return geodetic_utils.start_points_and_displacements_to_endpoints(
        start_latitudes_deg=start_latitude_matrix_deg,
        start_longitudes_deg=start_longitude_matrix_deg,
        scalar_displacements_metres=scalar_displacement_matrix_metres,
        geodetic_bearings_deg=bearing_matrix_deg)


def _rotate_grids_many_storm_objects(
        storm_object_table, num_storm_image_rows, num_storm_image_columns,
        storm_grid_spacing_metres):
    """Creates rotated, storm-centered grid for each storm object.

    m = number of rows in each storm-centered grid
    n = number of columns in each storm-centered grid

    :param storm_object_table: pandas DataFrame with the following columns.
        Each row is one storm object.
    storm_object_table.centroid_lat_deg: Latitude (deg N) of storm centroid.
    storm_object_table.centroid_lng_deg: Longitude (deg E) of storm centroid.
    storm_object_table.east_velocity_m_s01: Eastward storm velocity (metres per
        second).
    storm_object_table.north_velocity_m_s01: Northward storm velocity.

    :param num_storm_image_rows: m in the above discussion.
    :param num_storm_image_columns: n in the above discussion.
    :param storm_grid_spacing_metres: Spacing between grid points in adjacent
        rows or columns.
    :return: storm_object_table: Same as input, but with two new columns.
    storm_object_table.rotated_gp_lat_matrix_deg: m-by-n numpy array with
        latitudes (deg N) of grid points.
    storm_object_table.rotated_gp_lng_matrix_deg: m-by-n numpy array with
        longitudes (deg E) of grid points.
    """

    num_storm_objects = len(storm_object_table.index)
    list_of_latitude_matrices = [None] * num_storm_objects
    list_of_longitude_matrices = [None] * num_storm_objects

    good_indices = numpy.where(numpy.invert(numpy.logical_or(
        numpy.isnan(
            storm_object_table[tracking_utils.EAST_VELOCITY_COLUMN].values),
        numpy.isnan(
            storm_object_table[tracking_utils.NORTH_VELOCITY_COLUMN].values)
    )))[0]

    for i in range(len(good_indices)):
        if numpy.mod(i, 1000) == 0:
            print (
                'Have created rotated, storm-centered grid for {0:d} of {1:d} '
                'storm objects...'
            ).format(i, num_storm_objects)

        j = good_indices[i]

        (list_of_latitude_matrices[j], list_of_longitude_matrices[j]
        ) = _rotate_grid_one_storm_object(
            centroid_latitude_deg=storm_object_table[
                tracking_utils.CENTROID_LAT_COLUMN].values[j],
            centroid_longitude_deg=storm_object_table[
                tracking_utils.CENTROID_LNG_COLUMN].values[j],
            eastward_motion_m_s01=storm_object_table[
                tracking_utils.EAST_VELOCITY_COLUMN].values[j],
            northward_motion_m_s01=storm_object_table[
                tracking_utils.NORTH_VELOCITY_COLUMN].values[j],
            num_storm_image_rows=num_storm_image_rows,
            num_storm_image_columns=num_storm_image_columns,
            storm_grid_spacing_metres=storm_grid_spacing_metres)

    print (
        'Created rotated, storm-centered grid for {0:d} of {1:d} storm objects '
        '(the others have missing velocities, because they are the first '
        'objects in their respective storm cells).'
    ).format(len(good_indices), num_storm_objects)

    argument_dict = {
        ROTATED_GP_LATITUDES_COLUMN: list_of_latitude_matrices,
        ROTATED_GP_LONGITUDES_COLUMN: list_of_longitude_matrices,
    }
    return storm_object_table.assign(**argument_dict)


def _centroids_latlng_to_rowcol(
        centroid_latitudes_deg, centroid_longitudes_deg, nw_grid_point_lat_deg,
        nw_grid_point_lng_deg, lat_spacing_deg, lng_spacing_deg):
    """Converts storm centroids from lat-long to row-column coordinates.

    L = number of storm objects

    :param centroid_latitudes_deg: length-L numpy array with latitudes (deg N)
        of storm centroids.
    :param centroid_longitudes_deg: length-L numpy array with longitudes (deg E)
        of storm centroids.
    :param nw_grid_point_lat_deg: Latitude (deg N) of northwesternmost grid
        point.
    :param nw_grid_point_lng_deg: Longitude (deg E) of northwesternmost grid
        point.
    :param lat_spacing_deg: Spacing (deg N) between adjacent grid rows.
    :param lng_spacing_deg: Spacing (deg E) between adjacent grid columns.
    :return: centroid_rows: length-L numpy array with row indices (half-
        integers) of storm centroids.
    :return: centroid_columns: length-L numpy array with column indices (half-
        integers) of storm centroids.
    """

    center_row_indices, center_column_indices = radar_utils.latlng_to_rowcol(
        latitudes_deg=centroid_latitudes_deg,
        longitudes_deg=centroid_longitudes_deg,
        nw_grid_point_lat_deg=nw_grid_point_lat_deg,
        nw_grid_point_lng_deg=nw_grid_point_lng_deg,
        lat_spacing_deg=lat_spacing_deg, lng_spacing_deg=lng_spacing_deg)

    return (rounder.round_to_half_integer(center_row_indices),
            rounder.round_to_half_integer(center_column_indices))


def _get_unrotated_storm_image_coords(
        num_full_grid_rows, num_full_grid_columns, num_storm_image_rows,
        num_storm_image_columns, center_row, center_column):
    """Generates row-column coordinates for storm-centered grid.

    :param num_full_grid_rows: Number of rows in full grid.
    :param num_full_grid_columns: Number of columns in full grid.
    :param num_storm_image_rows: Number of rows in storm-centered image
        (subgrid).
    :param num_storm_image_columns: Number of columns in subgrid.
    :param center_row: Row index (half-integer) at center of subgrid.  If
        `center_row = k`, row k in the full grid is at the center of the
        subgrid.
    :param center_column: Column index (half-integer) at center of subgrid.  If
        `center_column = m`, column m in the full grid is at the center of the
        subgrid.
    :return: coord_dict: Dictionary with the following keys.
    coord_dict['first_storm_image_row']: First row in subgrid.  If
        `first_storm_image_row = i`, row 0 in the subgrid = row i in the full
        grid.
    coord_dict['last_storm_image_row']: Last row in subgrid.
    coord_dict['first_storm_image_column']: First column in subgrid.  If
        `first_storm_image_column = j`, column 0 in the subgrid = column j in
        the full grid.
    coord_dict['last_storm_image_column']: Last column in subgrid.
    coord_dict['num_padding_rows_at_top']: Number of padding rows at top of
        subgrid.  This will be non-zero iff the subgrid does not fit inside the
        full grid.
    coord_dict['num_padding_rows_at_bottom']: Number of padding rows at bottom
        of subgrid.
    coord_dict['num_padding_rows_at_left']: Number of padding rows at left side
        of subgrid.
    coord_dict['num_padding_rows_at_right']: Number of padding rows at right
        side of subgrid.
    """

    first_storm_image_row = int(numpy.ceil(
        center_row - num_storm_image_rows / 2))
    last_storm_image_row = int(numpy.floor(
        center_row + num_storm_image_rows / 2))
    first_storm_image_column = int(numpy.ceil(
        center_column - num_storm_image_columns / 2))
    last_storm_image_column = int(numpy.floor(
        center_column + num_storm_image_columns / 2))

    if first_storm_image_row < 0:
        num_padding_rows_at_top = 0 - first_storm_image_row
        first_storm_image_row = 0
    else:
        num_padding_rows_at_top = 0

    if last_storm_image_row > num_full_grid_rows - 1:
        num_padding_rows_at_bottom = last_storm_image_row - (
            num_full_grid_rows - 1)
        last_storm_image_row = num_full_grid_rows - 1
    else:
        num_padding_rows_at_bottom = 0

    if first_storm_image_column < 0:
        num_padding_columns_at_left = 0 - first_storm_image_column
        first_storm_image_column = 0
    else:
        num_padding_columns_at_left = 0

    if last_storm_image_column > num_full_grid_columns - 1:
        num_padding_columns_at_right = last_storm_image_column - (
            num_full_grid_columns - 1)
        last_storm_image_column = num_full_grid_columns - 1
    else:
        num_padding_columns_at_right = 0

    return {
        FIRST_STORM_ROW_KEY: first_storm_image_row,
        LAST_STORM_ROW_KEY: last_storm_image_row,
        FIRST_STORM_COLUMN_KEY: first_storm_image_column,
        LAST_STORM_COLUMN_KEY: last_storm_image_column,
        NUM_TOP_PADDING_ROWS_KEY: num_padding_rows_at_top,
        NUM_BOTTOM_PADDING_ROWS_KEY: num_padding_rows_at_bottom,
        NUM_LEFT_PADDING_COLS_KEY: num_padding_columns_at_left,
        NUM_RIGHT_PADDING_COLS_KEY: num_padding_columns_at_right
    }


def _check_storm_images(
        storm_image_matrix, storm_ids, valid_times_unix_sec, radar_field_name,
        radar_height_m_asl, rotated_grids, rotated_grid_spacing_metres=None,
        rotation_divergence_products_s02=None, horiz_radius_for_rdp_metres=None,
        min_vort_height_for_rdp_m_asl=None):
    """Checks storm images (e.g., created by extract_storm_image) for errors.

    L = number of storm objects
    M = number of rows in each image
    N = number of columns in each image

    :param storm_image_matrix: L-by-M-by-N numpy array with image for each storm
        object.
    :param storm_ids: length-L list of storm IDs (strings).
    :param valid_times_unix_sec: length-L numpy array of valid times.
    :param radar_field_name: Name of radar field (string).
    :param radar_height_m_asl: Height (metres above sea level) of radar field.
    :param rotated_grids: Boolean flag.  If True, each grid has been rotated so
        that storm motion is in the +x-direction.
    :param rotated_grid_spacing_metres: [used only if `rotate_grids = True`]
        Spacing between grid points in adjacent rows or columns.
    :param rotation_divergence_products_s02: length-L numpy array of rotation-
        divergence products (seconds^-2).  This may be `None`.
    :param horiz_radius_for_rdp_metres:
        [used only if `rotation_divergence_products_s02 is not None`]
        See doc for `get_max_rdp_for_each_storm_object`.
    :param min_vort_height_for_rdp_m_asl:
        [used only if `rotation_divergence_products_s02 is not None`]
        See doc for `get_max_rdp_for_each_storm_object`.
    """

    error_checking.assert_is_string_list(storm_ids)
    error_checking.assert_is_numpy_array(
        numpy.array(storm_ids), num_dimensions=1)
    num_storm_objects = len(storm_ids)

    error_checking.assert_is_integer_numpy_array(valid_times_unix_sec)
    error_checking.assert_is_numpy_array(
        valid_times_unix_sec, exact_dimensions=numpy.array([num_storm_objects]))

    radar_utils.check_field_name(radar_field_name)
    error_checking.assert_is_geq(radar_height_m_asl, 0)

    error_checking.assert_is_numpy_array(storm_image_matrix, num_dimensions=3)
    error_checking.assert_is_numpy_array(
        storm_image_matrix, exact_dimensions=numpy.array(
            [num_storm_objects, storm_image_matrix.shape[1],
             storm_image_matrix.shape[2]]))

    error_checking.assert_is_boolean(rotated_grids)
    if rotated_grids:
        error_checking.assert_is_greater(rotated_grid_spacing_metres, 0.)

    if rotation_divergence_products_s02 is not None:
        error_checking.assert_is_numpy_array(
            rotation_divergence_products_s02,
            exact_dimensions=numpy.array([num_storm_objects]))
        error_checking.assert_is_greater(horiz_radius_for_rdp_metres, 0.)
        error_checking.assert_is_integer(min_vort_height_for_rdp_m_asl)
        error_checking.assert_is_greater(min_vort_height_for_rdp_m_asl, 0)


def _check_storm_labels(
        storm_ids, valid_times_unix_sec, storm_to_winds_table,
        storm_to_tornadoes_table):
    """Error-checks storm labels (target variables).

    L = number of storm objects

    :param storm_ids: length-L list of storm IDs (strings).
    :param valid_times_unix_sec: length-L numpy array of valid times.
    :param storm_to_winds_table: pandas DataFrame created by
        `labels.label_wind_speed_for_classification`.  This may be None.
    :param storm_to_tornadoes_table: pandas DataFrame created by
        `labels.label_tornado_occurrence`.  This may be None.
    :return: relevant_storm_to_winds_table: Same as input, but containing only
        the given storm objects (ID-time pairs).  If `storm_to_winds_table is
        None`, this is None.
    :return: relevant_storm_to_tornadoes_table: Same as input, but containing
        only the given storm objects (ID-time pairs).  If
        `storm_to_tornadoes_table is None`, this is None.
    """

    error_checking.assert_is_string_list(storm_ids)
    error_checking.assert_is_numpy_array(
        numpy.array(storm_ids), num_dimensions=1)
    num_storm_objects = len(storm_ids)

    error_checking.assert_is_integer_numpy_array(valid_times_unix_sec)
    error_checking.assert_is_numpy_array(
        valid_times_unix_sec, exact_dimensions=numpy.array([num_storm_objects]))

    if storm_to_winds_table is None:
        relevant_storm_to_winds_table = None
    else:
        relevant_indices = find_storm_objects(
            all_storm_ids=storm_to_winds_table[
                tracking_utils.STORM_ID_COLUMN].values,
            all_valid_times_unix_sec=storm_to_winds_table[
                tracking_utils.TIME_COLUMN].values,
            storm_ids_to_keep=storm_ids,
            valid_times_to_keep_unix_sec=valid_times_unix_sec)
        relevant_storm_to_winds_table = storm_to_winds_table.iloc[
            relevant_indices]

    if storm_to_tornadoes_table is None:
        relevant_storm_to_tornadoes_table = None
    else:
        relevant_indices = find_storm_objects(
            all_storm_ids=storm_to_tornadoes_table[
                tracking_utils.STORM_ID_COLUMN].values,
            all_valid_times_unix_sec=storm_to_tornadoes_table[
                tracking_utils.TIME_COLUMN].values,
            storm_ids_to_keep=storm_ids,
            valid_times_to_keep_unix_sec=valid_times_unix_sec)
        relevant_storm_to_tornadoes_table = storm_to_tornadoes_table.iloc[
            relevant_indices]

    return relevant_storm_to_winds_table, relevant_storm_to_tornadoes_table


def _find_many_files_one_spc_date(
        top_directory_name, start_time_unix_sec, end_time_unix_sec,
        spc_date_string, radar_source, field_name_by_pair, height_by_pair_m_asl,
        raise_error_if_all_missing=True, raise_error_if_any_missing=False):
    """Finds many files containing storm images with MYRORSS or MRMS data.

    These files contain one time step each, all from the same SPC date.

    T = number of time steps
    C = number of field/height pairs

    :param top_directory_name: See documentation for
        `find_many_files_myrorss_or_mrms`.
    :param start_time_unix_sec: Start time.  This method will find all files
        from `start_time_unix_sec`...`end_time_unix_sec` that fit into the given
        SPC date.
    :param end_time_unix_sec: See above.
    :param spc_date_string: SPC date (format "yyyymmdd").
    :param radar_source: Source (either "myrorss" or "mrms").
    :param field_name_by_pair: length-C list with names of radar fields.
    :param height_by_pair_m_asl: length-C numpy array with heights (metres above
        sea level) of radar fields.
    :param raise_error_if_all_missing: Same.
    :param raise_error_if_any_missing: Same.
    :return: image_file_name_matrix: T-by-C numpy array of file paths.
    :return: valid_times_unix_sec: length-T numpy array of valid times.
    """

    # TODO(thunderhoser): Maybe this method should glob for all radar
    # field/height pairs, instead of just one.  Globbing for only one
    # field/height causes some files to be missed.

    num_field_height_pairs = len(field_name_by_pair)
    image_file_name_matrix = None
    valid_times_unix_sec = None

    is_field_az_shear = numpy.array(
        [this_field_name in AZIMUTHAL_SHEAR_FIELD_NAMES
         for this_field_name in field_name_by_pair])
    is_any_field_az_shear = numpy.any(is_field_az_shear)
    glob_index = -1

    for j in range(num_field_height_pairs):
        glob_now = (field_name_by_pair[j] in AZIMUTHAL_SHEAR_FIELD_NAMES
                    or not is_any_field_az_shear)
        if not glob_now:
            continue
        glob_index = j + 0

        this_file_pattern = (
            '{0:s}/{1:s}/{2:s}/{3:s}/{4:s}/{5:05d}_metres_asl/'
            'storm_images_{6:s}.nc'
        ).format(
            top_directory_name, radar_source, spc_date_string[:4],
            spc_date_string, field_name_by_pair[j],
            int(numpy.round(height_by_pair_m_asl[j])),
            TIME_FORMAT_REGEX)

        these_file_names = glob.glob(this_file_pattern)
        image_file_names = []
        valid_times_unix_sec = []

        for this_file_name in these_file_names:
            _, this_pathless_file_name = os.path.split(this_file_name)
            this_extensionless_file_name, _ = os.path.splitext(
                this_pathless_file_name)

            this_time_string = this_extensionless_file_name.split('_')[-1]
            this_time_unix_sec = time_conversion.string_to_unix_sec(
                this_time_string, TIME_FORMAT)

            if start_time_unix_sec <= this_time_unix_sec <= end_time_unix_sec:
                image_file_names.append(this_file_name)
                valid_times_unix_sec.append(this_time_unix_sec)

        num_times = len(image_file_names)
        if num_times == 0:
            if raise_error_if_all_missing:
                error_string = (
                    'Cannot find any files on SPC date "{0:s}" for "{1:s}" '
                    'at {2:d} metres ASL.'
                ).format(
                    spc_date_string, field_name_by_pair[j],
                    int(numpy.round(height_by_pair_m_asl[j])))
                raise ValueError(error_string)

            return None, None

        valid_times_unix_sec = numpy.array(valid_times_unix_sec, dtype=int)
        image_file_name_matrix = numpy.full(
            (num_times, num_field_height_pairs), '', dtype=object)
        image_file_name_matrix[:, j] = numpy.array(
            image_file_names, dtype=object)

        break

    for j in range(num_field_height_pairs):
        if j == glob_index:
            continue

        for i in range(len(valid_times_unix_sec)):
            image_file_name_matrix[i, j] = find_storm_image_file(
                top_directory_name=top_directory_name,
                unix_time_sec=valid_times_unix_sec[i],
                spc_date_string=spc_date_string, radar_source=radar_source,
                radar_field_name=field_name_by_pair[j],
                radar_height_m_asl=height_by_pair_m_asl[j],
                raise_error_if_missing=raise_error_if_any_missing)

            if not os.path.isfile(image_file_name_matrix[i, j]):
                image_file_name_matrix[i, j] = ''

    return image_file_name_matrix, valid_times_unix_sec


def _filter_storm_objects_by_label(
        label_values, num_storm_objects_class_dict, test_mode=False):
    """Filters storm objects by label (target variable).

    L = number of storm objects
    K_x = number of extended classes

    :param label_values: length-L numpy array of integers.  label_values[i] is
        the class for the [i]th storm object.
    :param num_storm_objects_class_dict: Dictionary, where each key is a class
        integer (-2 for dead storms) and each value is the corresponding number
        of storm objects to return.
    :param test_mode: Boolean flag.  Leave this False.
    :return: indices_to_keep: 1-D numpy array with indices of storm objects to
        keep.
    """

    indices_to_keep = numpy.array([], dtype=int)
    for this_class_integer in num_storm_objects_class_dict.keys():
        this_num_storm_objects = num_storm_objects_class_dict[
            this_class_integer]

        these_indices = numpy.where(label_values == this_class_integer)[0]
        this_num_storm_objects = min(
            [this_num_storm_objects, len(these_indices)])
        if this_num_storm_objects == 0:
            continue

        if test_mode:
            these_indices = these_indices[:this_num_storm_objects]
        else:
            these_indices = numpy.random.choice(
                these_indices, size=this_num_storm_objects, replace=False)

        indices_to_keep = numpy.concatenate((indices_to_keep, these_indices))

    return indices_to_keep


def _get_max_rdp_values_one_time(
        divergence_matrix_s01, vorticity_matrix_s01, grid_point_latitudes_deg,
        grid_point_longitudes_deg, grid_point_heights_m_asl,
        min_vorticity_height_m_asl, storm_centroid_latitudes_deg,
        storm_centroid_longitudes_deg, horizontal_radius_metres):
    """Computes max RDP for each storm object at one time step.

    RDP = rotation-divergence product.  For a thorough definition, see
    documentation for `get_max_rdp_for_each_storm_object`.

    M = number of rows (unique latitudes at grid points)
    N = number of columns (unique longitudes at grid points)
    H = number of depths (unique heights at grid points)
    L = number of storm objects

    :param divergence_matrix_s01: H-by-M-by-N numpy array of divergence values
        (units are seconds^-1).
    :param vorticity_matrix_s01: H-by-M-by-N numpy array of vorticity values
        (seconds^-1).
    :param grid_point_latitudes_deg: length-M numpy array with latitudes (deg N)
        of grid points.
    :param grid_point_longitudes_deg: length-N numpy array with longitudes
        (deg E) of grid points.
    :param grid_point_heights_m_asl: length-H numpy array with heights (metres
        above sea level) of grid point.
    :param min_vorticity_height_m_asl: Minimum vorticity height (metres above
        sea level).  All vorticity values below this height are ignored.
    :param storm_centroid_latitudes_deg: length-L numpy array with latitudes
        (deg N) at centroids of storm objects.
    :param storm_centroid_longitudes_deg: length-L numpy array with longitudes
        (deg E) at centroids of storm objects.
    :param horizontal_radius_metres: Horizontal radius for RDP calculations.
        Values > `horizontal_radius_metres` from the centroid of a storm object
        will be ignored.
    :return: max_rdp_values_s02: length-L numpy array with maximum RDP
        (seconds^-2) for each storm object.
    """

    valid_height_indices = numpy.where(
        grid_point_heights_m_asl >= min_vorticity_height_m_asl)[0]
    vorticity_matrix_s01 = vorticity_matrix_s01[valid_height_indices, ...]
    del grid_point_heights_m_asl

    num_storm_objects = len(storm_centroid_latitudes_deg)
    max_rdp_values_s02 = numpy.full(num_storm_objects, numpy.nan)
    grid_point_dict = None

    for i in range(num_storm_objects):
        (these_rows, these_columns, grid_point_dict
        ) = grids.get_latlng_grid_points_in_radius(
            test_latitude_deg=storm_centroid_latitudes_deg[i],
            test_longitude_deg=storm_centroid_longitudes_deg[i],
            effective_radius_metres=horizontal_radius_metres,
            grid_point_latitudes_deg=grid_point_latitudes_deg,
            grid_point_longitudes_deg=grid_point_longitudes_deg,
            grid_point_dict=grid_point_dict)

        max_rdp_values_s02[i] = (
            numpy.nanmax(divergence_matrix_s01[:, these_rows, these_columns]) *
            numpy.nanmax(vorticity_matrix_s01[:, these_rows, these_columns])
        )

    return max_rdp_values_s02


def _extract_rotated_storm_image(
        full_radar_matrix, full_grid_point_latitudes_deg,
        full_grid_point_longitudes_deg, rotated_gp_lat_matrix_deg,
        rotated_gp_lng_matrix_deg):
    """Extracts rotated, storm-centered image from full radar image.

    M = number of rows in full grid
    N = number of columns in full grid
    m = number of rows in storm-centered grid
    n = number of columns in storm-centered grid

    :param full_radar_matrix: M-by-N numpy array of radar values (one variable
        at one height and one time step).  Latitude should increase with row
        index, and longitude should increase with column index.
    :param full_grid_point_latitudes_deg: length-M numpy array with latitudes
        (deg N) of grid points.
    :param full_grid_point_longitudes_deg: length-N numpy array with longitudes
        (deg E) of grid points.
    :param rotated_gp_lat_matrix_deg: m-by-n numpy array with latitudes (deg N)
        of grid points.
    :param rotated_gp_lng_matrix_deg: m-by-n numpy array with longitudes (deg E)
        of grid points.
    :return: storm_centered_radar_matrix: m-by-n numpy array of radar values
        (same variable, height, and time step).
    """

    central_latitude_deg = numpy.mean(rotated_gp_lat_matrix_deg)
    central_longitude_deg = numpy.mean(rotated_gp_lng_matrix_deg)

    projection_object = projections.init_cylindrical_equidistant_projection(
        central_latitude_deg=central_latitude_deg,
        central_longitude_deg=central_longitude_deg,
        true_scale_latitude_deg=central_latitude_deg)

    (rotated_gp_x_matrix_metres, rotated_gp_y_matrix_metres
    ) = projections.project_latlng_to_xy(
        latitudes_deg=rotated_gp_lat_matrix_deg,
        longitudes_deg=rotated_gp_lng_matrix_deg,
        projection_object=projection_object)

    full_grid_points_x_metres, _ = projections.project_latlng_to_xy(
        latitudes_deg=numpy.full(
            full_grid_point_longitudes_deg.shape, central_latitude_deg),
        longitudes_deg=full_grid_point_longitudes_deg,
        projection_object=projection_object)

    _, full_grid_points_y_metres = projections.project_latlng_to_xy(
        latitudes_deg=full_grid_point_latitudes_deg,
        longitudes_deg=numpy.full(
            full_grid_point_latitudes_deg.shape, central_longitude_deg),
        projection_object=projection_object)

    valid_x_indices = numpy.where(numpy.logical_and(
        full_grid_points_x_metres >= numpy.min(rotated_gp_x_matrix_metres),
        full_grid_points_x_metres <= numpy.max(rotated_gp_x_matrix_metres)
    ))[0]
    first_valid_x_index = max([valid_x_indices[0] - 2, 0])
    last_valid_x_index = min(
        [valid_x_indices[-1] + 2,
         len(full_grid_points_x_metres) - 1])
    valid_x_indices = numpy.linspace(
        first_valid_x_index, last_valid_x_index,
        num=last_valid_x_index - first_valid_x_index + 1, dtype=int)

    valid_y_indices = numpy.where(numpy.logical_and(
        full_grid_points_y_metres >= numpy.min(rotated_gp_y_matrix_metres),
        full_grid_points_y_metres <= numpy.max(rotated_gp_y_matrix_metres)
    ))[0]
    first_valid_y_index = max([valid_y_indices[0] - 2, 0])
    last_valid_y_index = min(
        [valid_y_indices[-1] + 2,
         len(full_grid_points_y_metres) - 1])
    valid_y_indices = numpy.linspace(
        first_valid_y_index, last_valid_y_index,
        num=last_valid_y_index - first_valid_y_index + 1, dtype=int)

    exec_start_time_unix_sec = time.time()
    storm_centered_radar_matrix = interp.interp_from_xy_grid_to_points(
        input_matrix=full_radar_matrix[valid_y_indices, valid_x_indices],
        sorted_grid_point_x_metres=full_grid_points_x_metres[valid_x_indices],
        sorted_grid_point_y_metres=full_grid_points_y_metres[valid_y_indices],
        query_x_coords_metres=rotated_gp_x_matrix_metres.ravel(),
        query_y_coords_metres=rotated_gp_y_matrix_metres.ravel(),
        method_string=interp.SPLINE_METHOD_STRING, spline_degree=1,
        extrapolate=True)
    print 'Time elapsed in interpolation = {0:.3f} s'.format(
        time.time() - exec_start_time_unix_sec)
    storm_centered_radar_matrix = numpy.reshape(
        storm_centered_radar_matrix, rotated_gp_lat_matrix_deg.shape)

    invalid_x_flags = numpy.logical_or(
        rotated_gp_x_matrix_metres < numpy.min(full_grid_points_x_metres),
        rotated_gp_x_matrix_metres > numpy.max(full_grid_points_x_metres))
    invalid_y_flags = numpy.logical_or(
        rotated_gp_y_matrix_metres < numpy.min(full_grid_points_y_metres),
        rotated_gp_y_matrix_metres > numpy.max(full_grid_points_y_metres))
    invalid_indices = numpy.where(
        numpy.logical_or(invalid_x_flags, invalid_y_flags))

    storm_centered_radar_matrix[invalid_indices] = 0.
    return numpy.flipud(storm_centered_radar_matrix)


def _extract_unrotated_storm_image(
        full_radar_matrix, center_row, center_column, num_storm_image_rows,
        num_storm_image_columns):
    """Extracts storm-centered image from full radar image.

    M = number of rows in full grid
    N = number of columns in full grid
    m = number of rows in subgrid (must be even)
    n = number of columns in subgrid (must be even)

    The subgrid is rotated so that storm motion is in the +x-direction.

    :param full_radar_matrix: M-by-N numpy array of radar values (one variable
        at one height and one time step).
    :param center_row: Row index (half-integer) at center of subgrid.  If
        `center_row = i`, row i in the full grid is at the center of the
        subgrid.
    :param center_column: Column index (half-integer) at center of subgrid.  If
        `center_column = j`, column j in the full grid is at the center of the
        subgrid.
    :param num_storm_image_rows: m in the above discussion.
    :param num_storm_image_columns: n in the above discussion.
    :return: storm_centered_radar_matrix: m-by-n numpy array of radar values
        (same variable, height, and time step).
    """

    num_full_grid_rows = full_radar_matrix.shape[0]
    num_full_grid_columns = full_radar_matrix.shape[1]

    error_checking.assert_is_geq(center_row, -0.5)
    error_checking.assert_is_leq(center_row, num_full_grid_rows - 0.5)
    error_checking.assert_is_geq(center_column, -0.5)
    error_checking.assert_is_leq(center_column, num_full_grid_columns - 0.5)

    storm_image_coord_dict = _get_unrotated_storm_image_coords(
        num_full_grid_rows=num_full_grid_rows,
        num_full_grid_columns=num_full_grid_columns,
        num_storm_image_rows=num_storm_image_rows,
        num_storm_image_columns=num_storm_image_columns, center_row=center_row,
        center_column=center_column)

    storm_image_rows = numpy.linspace(
        storm_image_coord_dict[FIRST_STORM_ROW_KEY],
        storm_image_coord_dict[LAST_STORM_ROW_KEY],
        num=(storm_image_coord_dict[LAST_STORM_ROW_KEY] -
             storm_image_coord_dict[FIRST_STORM_ROW_KEY] + 1),
        dtype=int)
    storm_image_columns = numpy.linspace(
        storm_image_coord_dict[FIRST_STORM_COLUMN_KEY],
        storm_image_coord_dict[LAST_STORM_COLUMN_KEY],
        num=(storm_image_coord_dict[LAST_STORM_COLUMN_KEY] -
             storm_image_coord_dict[FIRST_STORM_COLUMN_KEY] + 1),
        dtype=int)

    storm_centered_radar_matrix = numpy.take(
        full_radar_matrix, storm_image_rows, axis=0)
    storm_centered_radar_matrix = numpy.take(
        storm_centered_radar_matrix, storm_image_columns, axis=1)

    pad_width_input_arg = (
        (storm_image_coord_dict[NUM_TOP_PADDING_ROWS_KEY],
         storm_image_coord_dict[NUM_BOTTOM_PADDING_ROWS_KEY]),
        (storm_image_coord_dict[NUM_LEFT_PADDING_COLS_KEY],
         storm_image_coord_dict[NUM_RIGHT_PADDING_COLS_KEY]))

    return numpy.pad(
        storm_centered_radar_matrix, pad_width=pad_width_input_arg,
        mode='constant', constant_values=PADDING_VALUE)


def find_storm_objects(
        all_storm_ids, all_valid_times_unix_sec, storm_ids_to_keep,
        valid_times_to_keep_unix_sec):
    """Finds storm objects.

    P = total number of storm objects
    p = number of storm objects to keep

    :param all_storm_ids: length-P list of storm IDs (strings).
    :param all_valid_times_unix_sec: length-P list of valid times.
    :param storm_ids_to_keep: length-p list of storm IDs (strings).
    :param valid_times_to_keep_unix_sec: length-p numpy array of valid times.
    :return: relevant_indices: length-p numpy array of indices.
        all_storm_ids[relevant_indices] yields storm_ids_to_keep, and
        all_valid_times_unix_sec[relevant_indices] yields
        valid_times_to_keep_unix_sec.
    :raises: ValueError: if `all_storm_ids` and `all_valid_times_unix_sec`
        contain any duplicate pairs.
    :raises: ValueError: if `storm_ids_to_keep` and
        `valid_times_to_keep_unix_sec` contain any duplicate pairs.
    :raises: ValueError: if any desired storm object is not found.
    """

    error_checking.assert_is_numpy_array(
        numpy.array(all_storm_ids), num_dimensions=1)
    num_storm_objects_total = len(all_storm_ids)
    error_checking.assert_is_numpy_array(
        all_valid_times_unix_sec,
        exact_dimensions=numpy.array([num_storm_objects_total]))

    error_checking.assert_is_numpy_array(
        numpy.array(storm_ids_to_keep), num_dimensions=1)
    num_storm_objects_to_keep = len(storm_ids_to_keep)
    error_checking.assert_is_numpy_array(
        valid_times_to_keep_unix_sec,
        exact_dimensions=numpy.array([num_storm_objects_to_keep]))

    all_object_ids = [
        '{0:s}_{1:d}'.format(all_storm_ids[i], all_valid_times_unix_sec[i])
        for i in range(num_storm_objects_total)]

    object_ids_to_keep = [
        '{0:s}_{1:d}'.format(storm_ids_to_keep[i],
                             valid_times_to_keep_unix_sec[i])
        for i in range(num_storm_objects_to_keep)]

    this_num_unique = len(set(all_object_ids))
    if this_num_unique != len(all_object_ids):
        error_string = (
            'Only {0:d} of {1:d} original storm objects are unique.'
        ).format(this_num_unique, len(all_object_ids))
        raise ValueError(error_string)

    this_num_unique = len(set(object_ids_to_keep))
    if this_num_unique != len(object_ids_to_keep):
        error_string = (
            'Only {0:d} of {1:d} desired storm objects are unique.'
        ).format(this_num_unique, len(object_ids_to_keep))
        raise ValueError(error_string)

    all_object_ids_numpy = numpy.array(all_object_ids, dtype='object')
    object_ids_to_keep_numpy = numpy.array(object_ids_to_keep, dtype='object')

    sort_indices = numpy.argsort(all_object_ids_numpy)
    relevant_indices = numpy.searchsorted(
        all_object_ids_numpy[sort_indices], object_ids_to_keep_numpy,
        side='left'
    ).astype(int)
    relevant_indices = sort_indices[relevant_indices]

    if not numpy.array_equal(all_object_ids_numpy[relevant_indices],
                             object_ids_to_keep_numpy):
        missing_object_flags = (
            all_object_ids_numpy[relevant_indices] != object_ids_to_keep_numpy)

        error_string = (
            '{0:d} of {1:d} desired storm objects are missing.  Their ID-time '
            'pairs are listed below.\n{2:s}'
        ).format(numpy.sum(missing_object_flags), num_storm_objects_to_keep,
                 str(object_ids_to_keep_numpy[missing_object_flags]))
        raise ValueError(error_string)

    return relevant_indices


def extract_storm_images_myrorss_or_mrms(
        storm_object_table, radar_source, top_radar_dir_name,
        top_output_dir_name, one_file_per_time_step=False,
        num_storm_image_rows=DEFAULT_NUM_IMAGE_ROWS,
        num_storm_image_columns=DEFAULT_NUM_IMAGE_COLUMNS, rotate_grids=True,
        rotated_grid_spacing_metres=DEFAULT_ROTATED_GRID_SPACING_METRES,
        radar_field_names=DEFAULT_MYRORSS_MRMS_FIELD_NAMES,
        reflectivity_heights_m_asl=DEFAULT_RADAR_HEIGHTS_M_ASL):
    """Extracts storm-centered image for each field/height and storm object.

    L = number of storm objects
    C = number of field/height pairs

    If `one_file_per_time_step = True`:
    T = number of time steps with storm objects

    If `one_file_per_time_step = False`:
    T = number of SPC dates with storm objects

    :param storm_object_table: L-row pandas DataFrame with the following
        columns.
    storm_object_table.storm_id: String ID.
    storm_object_table.unix_time_sec: Valid time.
    storm_object_table.spc_date_unix_sec: SPC date.
    storm_object_table.centroid_lat_deg: Latitude (deg N) of storm centroid.
    storm_object_table.centroid_lng_deg: Longitude (deg E) of storm centroid.

    If `rotate_grids = True`, also need the following columns.

    storm_object_table.east_velocity_m_s01: Eastward storm velocity (metres per
        second).
    storm_object_table.north_velocity_m_s01: Northward storm velocity.

    :param radar_source: Data source (either "myrorss" or "mrms").
    :param top_radar_dir_name: Name of top-level directory with radar data from
        the given source.
    :param top_output_dir_name: Name of top-level directory for storm-centered
        radar images.
    :param one_file_per_time_step: See doc for `_check_extraction_args`.
    :param num_storm_image_rows: Same.
    :param num_storm_image_columns: Same.
    :param rotate_grids: Same.
    :param rotated_grid_spacing_metres: Same.
    :param radar_field_names: 1-D list with names of radar fields.
    :param reflectivity_heights_m_asl:
        [used only if "reflectivity_dbz" in radar_field_names]
        1-D numpy array of reflectivity heights (metres above sea level).
    :return: image_file_name_matrix: T-by-C numpy array of paths to output
        files.
    :raises: ValueError: if grid spacing is not uniform across all input (radar)
        files.
    """

    _check_extraction_args(
        one_file_per_time_step=one_file_per_time_step,
        num_storm_image_rows=num_storm_image_rows,
        num_storm_image_columns=num_storm_image_columns,
        rotate_grids=rotate_grids,
        rotated_grid_spacing_metres=rotated_grid_spacing_metres)

    # Find input (radar) files.
    spc_date_strings = [
        time_conversion.time_to_spc_date_string(t)
        for t in storm_object_table[tracking_utils.SPC_DATE_COLUMN].values
    ]

    file_dictionary = myrorss_and_mrms_io.find_many_raw_files(
        desired_times_unix_sec=
        storm_object_table[tracking_utils.TIME_COLUMN].values.astype(int),
        spc_date_strings=spc_date_strings, data_source=radar_source,
        field_names=radar_field_names, top_directory_name=top_radar_dir_name,
        reflectivity_heights_m_asl=reflectivity_heights_m_asl)

    radar_file_name_matrix = file_dictionary[
        myrorss_and_mrms_io.RADAR_FILE_NAMES_KEY]
    valid_times_unix_sec = file_dictionary[myrorss_and_mrms_io.UNIQUE_TIMES_KEY]
    valid_spc_dates_unix_sec = file_dictionary[
        myrorss_and_mrms_io.SPC_DATES_AT_UNIQUE_TIMES_KEY]
    field_name_by_pair = file_dictionary[
        myrorss_and_mrms_io.FIELD_NAME_BY_PAIR_KEY]
    height_by_pair_m_asl = file_dictionary[
        myrorss_and_mrms_io.HEIGHT_BY_PAIR_KEY]

    valid_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT)
        for t in valid_times_unix_sec
    ]
    unique_spc_dates_unix_sec = numpy.unique(valid_spc_dates_unix_sec)
    unique_spc_date_strings = [
        time_conversion.time_to_spc_date_string(t)
        for t in unique_spc_dates_unix_sec
    ]

    # Create rotated, storm-centered grids.
    if rotate_grids:
        storm_object_table = _rotate_grids_many_storm_objects(
            storm_object_table=storm_object_table,
            num_storm_image_rows=num_storm_image_rows,
            num_storm_image_columns=num_storm_image_columns,
            storm_grid_spacing_metres=rotated_grid_spacing_metres)
        print SEPARATOR_STRING

    # Initialize values.
    num_times = len(valid_time_strings)
    num_spc_dates = len(unique_spc_date_strings)
    num_field_height_pairs = len(field_name_by_pair)
    latitude_spacing_deg = None
    longitude_spacing_deg = None

    if one_file_per_time_step:
        image_file_name_matrix = numpy.full(
            (num_times, num_field_height_pairs), '', dtype=object)
    else:
        image_file_name_matrix = numpy.full(
            (num_spc_dates, num_field_height_pairs), '', dtype=object)

    for q in range(num_spc_dates):
        these_time_indices = numpy.where(
            valid_spc_dates_unix_sec == unique_spc_dates_unix_sec[q])[0]

        for j in range(num_field_height_pairs):
            this_date_storm_image_matrix = None
            this_date_storm_ids = []
            this_date_storm_times_unix_sec = numpy.array([], dtype=int)

            for i in these_time_indices:
                if radar_file_name_matrix[i, j] is None:
                    continue

                # Find storm objects at [i]th valid time.
                these_storm_flags = numpy.logical_and(
                    storm_object_table[tracking_utils.TIME_COLUMN].values ==
                    valid_times_unix_sec[i],
                    storm_object_table[tracking_utils.SPC_DATE_COLUMN].values ==
                    valid_spc_dates_unix_sec[i])

                if rotate_grids:
                    east_velocities_m_s01 = storm_object_table[
                        tracking_utils.EAST_VELOCITY_COLUMN].values
                    north_velocities_m_s01 = storm_object_table[
                        tracking_utils.NORTH_VELOCITY_COLUMN].values

                    these_velocity_flags = numpy.logical_and(
                        numpy.invert(numpy.isnan(east_velocities_m_s01)),
                        numpy.invert(numpy.isnan(north_velocities_m_s01)))
                    these_storm_flags = numpy.logical_and(
                        these_storm_flags, these_velocity_flags)

                these_storm_indices = numpy.where(these_storm_flags)[0]
                this_num_storms = len(these_storm_indices)

                these_storm_ids = storm_object_table[
                    tracking_utils.STORM_ID_COLUMN
                ].values[these_storm_indices].tolist()
                these_times_unix_sec = storm_object_table[
                    tracking_utils.TIME_COLUMN
                ].values[these_storm_indices].astype(int)

                if not one_file_per_time_step:
                    this_date_storm_ids += these_storm_ids
                    this_date_storm_times_unix_sec = numpy.concatenate((
                        this_date_storm_times_unix_sec, these_times_unix_sec))

                print (
                    'Extracting storm-centered images for "{0:s}" at {1:d} '
                    'metres ASL and {2:s}...'
                ).format(field_name_by_pair[j],
                         int(numpy.round(height_by_pair_m_asl[j])),
                         valid_time_strings[i])

                # Read [j]th field/height pair at [i]th time step.
                this_metadata_dict = (
                    myrorss_and_mrms_io.read_metadata_from_raw_file(
                        netcdf_file_name=radar_file_name_matrix[i, j],
                        data_source=radar_source))

                this_sparse_grid_table = (
                    myrorss_and_mrms_io.read_data_from_sparse_grid_file(
                        netcdf_file_name=radar_file_name_matrix[i, j],
                        field_name_orig=this_metadata_dict[
                            myrorss_and_mrms_io.FIELD_NAME_COLUMN_ORIG],
                        data_source=radar_source,
                        sentinel_values=this_metadata_dict[
                            radar_utils.SENTINEL_VALUE_COLUMN]))

                (this_full_radar_matrix, these_full_gp_latitudes_deg,
                 these_full_gp_longitudes_deg
                ) = radar_s2f.sparse_to_full_grid(
                    sparse_grid_table=this_sparse_grid_table,
                    metadata_dict=this_metadata_dict)

                this_full_radar_matrix[numpy.isnan(this_full_radar_matrix)] = 0.
                if rotate_grids:
                    this_full_radar_matrix = numpy.flipud(
                        this_full_radar_matrix)
                    these_full_gp_latitudes_deg = (
                        these_full_gp_latitudes_deg[::-1])

                if field_name_by_pair[j] in AZIMUTHAL_SHEAR_FIELD_NAMES:
                    this_num_image_rows = (
                        num_storm_image_rows * AZ_SHEAR_GRID_SPACING_MULTIPLIER)
                    this_num_image_columns = (num_storm_image_columns *
                                              AZ_SHEAR_GRID_SPACING_MULTIPLIER)

                    this_lat_spacing_deg = (
                        AZ_SHEAR_GRID_SPACING_MULTIPLIER *
                        this_metadata_dict[radar_utils.LAT_SPACING_COLUMN])
                    this_lng_spacing_deg = (
                        AZ_SHEAR_GRID_SPACING_MULTIPLIER *
                        this_metadata_dict[radar_utils.LNG_SPACING_COLUMN])
                else:
                    this_num_image_rows = num_storm_image_rows + 0
                    this_num_image_columns = num_storm_image_columns + 0

                    this_lat_spacing_deg = this_metadata_dict[
                        radar_utils.LAT_SPACING_COLUMN]
                    this_lng_spacing_deg = this_metadata_dict[
                        radar_utils.LNG_SPACING_COLUMN]

                this_lat_spacing_deg = rounder.round_to_nearest(
                    this_lat_spacing_deg, GRID_SPACING_TOLERANCE_DEG)
                this_lng_spacing_deg = rounder.round_to_nearest(
                    this_lng_spacing_deg, GRID_SPACING_TOLERANCE_DEG)

                if latitude_spacing_deg is None:
                    latitude_spacing_deg = this_lat_spacing_deg + 0.
                    longitude_spacing_deg = this_lng_spacing_deg + 0.

                if (latitude_spacing_deg != this_lat_spacing_deg or
                        longitude_spacing_deg != this_lng_spacing_deg):
                    error_string = (
                        'First file has grid spacing of {0:.4f} deg N, {1:.4f} '
                        'deg E.  This file ("{2:s}") has spacing of {3:.4f} deg'
                        ' N, {4:.4f} deg E.'
                    ).format(latitude_spacing_deg, longitude_spacing_deg,
                             radar_file_name_matrix[i, j], this_lat_spacing_deg,
                             this_lng_spacing_deg)
                    raise ValueError(error_string)

                this_storm_image_matrix = numpy.full(
                    (this_num_storms, this_num_image_rows,
                     this_num_image_columns), numpy.nan)

                # Extract images for [j]th field/height pair at [i]th time step.
                if rotate_grids:
                    for k in range(this_num_storms):
                        this_storm_image_matrix[
                            k, :, :
                        ] = _extract_rotated_storm_image(
                            full_radar_matrix=this_full_radar_matrix,
                            full_grid_point_latitudes_deg=
                            these_full_gp_latitudes_deg,
                            full_grid_point_longitudes_deg=
                            these_full_gp_longitudes_deg,
                            rotated_gp_lat_matrix_deg=storm_object_table[
                                ROTATED_GP_LATITUDES_COLUMN
                            ].values[these_storm_indices[k]],
                            rotated_gp_lng_matrix_deg=storm_object_table[
                                ROTATED_GP_LONGITUDES_COLUMN
                            ].values[these_storm_indices[k]])
                else:
                    (these_center_rows, these_center_columns
                    ) = _centroids_latlng_to_rowcol(
                        centroid_latitudes_deg=storm_object_table[
                            tracking_utils.CENTROID_LAT_COLUMN
                        ].values[these_storm_indices],
                        centroid_longitudes_deg=storm_object_table[
                            tracking_utils.CENTROID_LNG_COLUMN
                        ].values[these_storm_indices],
                        nw_grid_point_lat_deg=this_metadata_dict[
                            radar_utils.NW_GRID_POINT_LAT_COLUMN],
                        nw_grid_point_lng_deg=this_metadata_dict[
                            radar_utils.NW_GRID_POINT_LNG_COLUMN],
                        lat_spacing_deg=this_metadata_dict[
                            radar_utils.LAT_SPACING_COLUMN],
                        lng_spacing_deg=this_metadata_dict[
                            radar_utils.LNG_SPACING_COLUMN])

                    for k in range(this_num_storms):
                        this_storm_image_matrix[
                            k, :, :
                        ] = _extract_unrotated_storm_image(
                            full_radar_matrix=this_full_radar_matrix,
                            center_row=these_center_rows[k],
                            center_column=these_center_columns[k],
                            num_storm_image_rows=this_num_image_rows,
                            num_storm_image_columns=this_num_image_columns)

                if one_file_per_time_step:

                    # Write imgs for [j]th field/height pair at [i]th time step.
                    image_file_name_matrix[i, j] = find_storm_image_file(
                        top_directory_name=top_output_dir_name,
                        unix_time_sec=valid_times_unix_sec[i],
                        spc_date_string=unique_spc_date_strings[q],
                        radar_source=radar_source,
                        radar_field_name=field_name_by_pair[j],
                        radar_height_m_asl=height_by_pair_m_asl[j],
                        raise_error_if_missing=False)

                    print (
                        'Writing storm-centered images to: "{0:s}"...\n'
                    ).format(image_file_name_matrix[i, j])
                    write_storm_images(
                        netcdf_file_name=image_file_name_matrix[i, j],
                        storm_image_matrix=this_storm_image_matrix,
                        storm_ids=these_storm_ids,
                        valid_times_unix_sec=these_times_unix_sec,
                        radar_field_name=field_name_by_pair[j],
                        radar_height_m_asl=height_by_pair_m_asl[j],
                        rotated_grids=rotate_grids,
                        rotated_grid_spacing_metres=rotated_grid_spacing_metres)

                    continue

                if this_date_storm_image_matrix is None:
                    this_date_storm_image_matrix = this_storm_image_matrix + 0.
                else:
                    this_date_storm_image_matrix = numpy.concatenate(
                        (this_date_storm_image_matrix, this_storm_image_matrix),
                        axis=0)

            if one_file_per_time_step:
                continue

            # Write images for [j]th field/height pair and [q]th SPC date.
            image_file_name_matrix[q, j] = find_storm_image_file(
                top_directory_name=top_output_dir_name,
                spc_date_string=unique_spc_date_strings[q],
                radar_source=radar_source,
                radar_field_name=field_name_by_pair[j],
                radar_height_m_asl=height_by_pair_m_asl[j],
                raise_error_if_missing=False)

            print (
                'Writing storm-centered images to: "{0:s}"...\n'
            ).format(image_file_name_matrix[q, j])
            write_storm_images(
                netcdf_file_name=image_file_name_matrix[q, j],
                storm_image_matrix=this_date_storm_image_matrix,
                storm_ids=this_date_storm_ids,
                valid_times_unix_sec=this_date_storm_times_unix_sec,
                radar_field_name=field_name_by_pair[j],
                radar_height_m_asl=height_by_pair_m_asl[j],
                rotated_grids=rotate_grids,
                rotated_grid_spacing_metres=rotated_grid_spacing_metres)

    return image_file_name_matrix


def extract_storm_images_gridrad(
        storm_object_table, top_radar_dir_name, top_output_dir_name,
        one_file_per_time_step=False,
        num_storm_image_rows=DEFAULT_NUM_IMAGE_ROWS,
        num_storm_image_columns=DEFAULT_NUM_IMAGE_COLUMNS, rotate_grids=True,
        rotated_grid_spacing_metres=DEFAULT_ROTATED_GRID_SPACING_METRES,
        radar_field_names=DEFAULT_GRIDRAD_FIELD_NAMES,
        radar_heights_m_asl=DEFAULT_RADAR_HEIGHTS_M_ASL):
    """Extracts storm-centered image for each field, height, and storm object.

    L = number of storm objects
    F = number of radar fields
    H = number of radar heights

    If `one_file_per_time_step = True`:
    T = number of time steps with storm objects

    If `one_file_per_time_step = False`:
    T = number of SPC dates with storm objects

    :param storm_object_table: See doc for `extract_storm_images_gridrad`
        (except the column "spc_date_unix_sec" is not needed here).
    :param top_radar_dir_name: See doc for `extract_storm_images_gridrad`.
    :param top_output_dir_name: Same.
    :param one_file_per_time_step: See doc for `_check_extraction_args`.
    :param num_storm_image_rows: Same.
    :param num_storm_image_columns: Same.
    :param rotate_grids: Same.
    :param rotated_grid_spacing_metres: Same.
    :param radar_field_names: length-F list with names of radar fields.
    :param radar_heights_m_asl: length-H numpy array of radar heights (metres
        above sea level).
    :return: image_file_name_matrix: T-by-F-by-H numpy array of paths to output
        files.
    :raises: ValueError: if grid spacing is not uniform across all input (radar)
        files.
    """

    # Convert input args.
    gridrad_utils.fields_and_refl_heights_to_pairs(
        field_names=radar_field_names, heights_m_asl=radar_heights_m_asl)
    radar_heights_m_asl = numpy.sort(
        numpy.round(radar_heights_m_asl).astype(int))

    valid_times_unix_sec = numpy.unique(
        storm_object_table[tracking_utils.TIME_COLUMN].values)
    valid_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT)
        for t in valid_times_unix_sec
    ]

    valid_spc_dates_unix_sec = numpy.array(
        [time_conversion.time_to_spc_date_unix_sec(t)
         for t in valid_times_unix_sec], dtype=int)
    unique_spc_dates_unix_sec = numpy.unique(valid_spc_dates_unix_sec)
    unique_spc_date_strings = [
        time_conversion.time_to_spc_date_string(t)
        for t in unique_spc_dates_unix_sec
    ]

    # Find input (radar) files.
    num_times = len(valid_times_unix_sec)
    radar_file_names = [None] * num_times
    for i in range(num_times):
        radar_file_names[i] = gridrad_io.find_file(
            unix_time_sec=valid_times_unix_sec[i],
            top_directory_name=top_radar_dir_name, raise_error_if_missing=True)

    # Create rotated, storm-centered grids.
    if rotate_grids:
        storm_object_table = _rotate_grids_many_storm_objects(
            storm_object_table=storm_object_table,
            num_storm_image_rows=num_storm_image_rows,
            num_storm_image_columns=num_storm_image_columns,
            storm_grid_spacing_metres=rotated_grid_spacing_metres)
        print SEPARATOR_STRING

    # Initialize values.
    num_times = len(valid_times_unix_sec)
    num_spc_dates = len(unique_spc_date_strings)
    num_fields = len(radar_field_names)
    num_heights = len(radar_heights_m_asl)
    latitude_spacing_deg = None
    longitude_spacing_deg = None

    if one_file_per_time_step:
        image_file_name_matrix = numpy.full(
            (num_times, num_fields, num_heights), '', dtype=object)
    else:
        image_file_name_matrix = numpy.full(
            (num_spc_dates, num_fields, num_heights), '', dtype=object)

    for q in range(num_spc_dates):
        these_time_indices = numpy.where(
            valid_spc_dates_unix_sec == unique_spc_dates_unix_sec[q])[0]

        for j in range(num_fields):
            for k in range(num_heights):
                this_date_storm_image_matrix = None
                this_date_storm_ids = []
                this_date_storm_times_unix_sec = numpy.array([], dtype=int)

                for i in these_time_indices:
                    this_metadata_dict = (
                        gridrad_io.read_metadata_from_full_grid_file(
                            radar_file_names[i]))

                    this_lat_spacing_deg = rounder.round_to_nearest(
                        this_metadata_dict[radar_utils.LAT_SPACING_COLUMN],
                        GRID_SPACING_TOLERANCE_DEG)
                    this_lng_spacing_deg = rounder.round_to_nearest(
                        this_metadata_dict[radar_utils.LNG_SPACING_COLUMN],
                        GRID_SPACING_TOLERANCE_DEG)

                    if latitude_spacing_deg is None:
                        latitude_spacing_deg = this_lat_spacing_deg + 0.
                        longitude_spacing_deg = this_lng_spacing_deg + 0.

                    if (latitude_spacing_deg != this_lat_spacing_deg or
                            longitude_spacing_deg != this_lng_spacing_deg):
                        error_string = (
                            'First file has grid spacing of {0:.4f} deg N, '
                            '{1:.4f} deg E.  This file ("{2:s}") has spacing of'
                            ' {3:.4f} deg N, {4:.4f} deg E.'
                        ).format(latitude_spacing_deg, longitude_spacing_deg,
                                 radar_file_names[i], this_lat_spacing_deg,
                                 this_lng_spacing_deg)
                        raise ValueError(error_string)

                    these_storm_flags = (
                        storm_object_table[tracking_utils.TIME_COLUMN].values ==
                        valid_times_unix_sec[i])
                    if rotate_grids:
                        east_velocities_m_s01 = storm_object_table[
                            tracking_utils.EAST_VELOCITY_COLUMN].values
                        north_velocities_m_s01 = storm_object_table[
                            tracking_utils.NORTH_VELOCITY_COLUMN].values

                        these_velocity_flags = numpy.logical_and(
                            numpy.invert(numpy.isnan(east_velocities_m_s01)),
                            numpy.invert(numpy.isnan(north_velocities_m_s01)))
                        these_storm_flags = numpy.logical_and(
                            these_storm_flags, these_velocity_flags)

                    these_storm_indices = numpy.where(these_storm_flags)[0]
                    this_num_storms = len(these_storm_indices)

                    these_storm_ids = storm_object_table[
                        tracking_utils.STORM_ID_COLUMN
                    ].values[these_storm_indices].tolist()
                    these_times_unix_sec = storm_object_table[
                        tracking_utils.TIME_COLUMN
                    ].values[these_storm_indices].astype(int)

                    if not one_file_per_time_step:
                        this_date_storm_ids += these_storm_ids
                        this_date_storm_times_unix_sec = numpy.concatenate((
                            this_date_storm_times_unix_sec,
                            these_times_unix_sec))

                    # Read [j]th field at [i]th time step.
                    print 'Reading "{0:s}" from file: "{1:s}"...'.format(
                        radar_field_names[j], radar_file_names[i])

                    (this_full_radar_matrix, these_full_gp_heights_m_asl,
                     these_full_gp_latitudes_deg, these_full_gp_longitudes_deg
                    ) = gridrad_io.read_field_from_full_grid_file(
                        netcdf_file_name=radar_file_names[i],
                        field_name=radar_field_names[j],
                        metadata_dict=this_metadata_dict)

                    this_height_index = numpy.where(
                        these_full_gp_heights_m_asl == radar_heights_m_asl[k]
                    )[0][0]
                    del these_full_gp_heights_m_asl

                    this_full_radar_matrix = this_full_radar_matrix[
                        this_height_index, ...]
                    if not rotate_grids:
                        this_full_radar_matrix = numpy.flipud(
                            this_full_radar_matrix)
                        these_full_gp_latitudes_deg = (
                            these_full_gp_latitudes_deg[::-1])

                    this_full_radar_matrix[
                        numpy.isnan(this_full_radar_matrix)] = 0.
                    this_storm_image_matrix = numpy.full(
                        (this_num_storms, num_storm_image_rows,
                         num_storm_image_columns), numpy.nan)

                    print (
                        'Extracting storm-centered images for "{0:s}" at {1:d} '
                        'metres ASL and {2:s}...'
                    ).format(radar_field_names[j],
                             int(numpy.round(radar_heights_m_asl[k])),
                             valid_time_strings[i])

                    if rotate_grids:
                        for m in range(this_num_storms):
                            this_storm_image_matrix[
                                m, :, :
                            ] = _extract_rotated_storm_image(
                                full_radar_matrix=this_full_radar_matrix,
                                full_grid_point_latitudes_deg=
                                these_full_gp_latitudes_deg,
                                full_grid_point_longitudes_deg=
                                these_full_gp_longitudes_deg,
                                rotated_gp_lat_matrix_deg=storm_object_table[
                                    ROTATED_GP_LATITUDES_COLUMN
                                ].values[these_storm_indices[m]],
                                rotated_gp_lng_matrix_deg=storm_object_table[
                                    ROTATED_GP_LONGITUDES_COLUMN
                                ].values[these_storm_indices[m]])
                    else:
                        (these_center_rows, these_center_columns
                        ) = _centroids_latlng_to_rowcol(
                            centroid_latitudes_deg=storm_object_table[
                                tracking_utils.CENTROID_LAT_COLUMN
                            ].values[these_storm_indices],
                            centroid_longitudes_deg=storm_object_table[
                                tracking_utils.CENTROID_LNG_COLUMN
                            ].values[these_storm_indices],
                            nw_grid_point_lat_deg=this_metadata_dict[
                                radar_utils.NW_GRID_POINT_LAT_COLUMN],
                            nw_grid_point_lng_deg=this_metadata_dict[
                                radar_utils.NW_GRID_POINT_LNG_COLUMN],
                            lat_spacing_deg=this_metadata_dict[
                                radar_utils.LAT_SPACING_COLUMN],
                            lng_spacing_deg=this_metadata_dict[
                                radar_utils.LNG_SPACING_COLUMN])

                        for m in range(this_num_storms):
                            this_storm_image_matrix[
                                m, :, :
                            ] = _extract_unrotated_storm_image(
                                full_radar_matrix=this_full_radar_matrix,
                                center_row=these_center_rows[m],
                                center_column=these_center_columns[m],
                                num_storm_image_rows=num_storm_image_rows,
                                num_storm_image_columns=
                                num_storm_image_columns)

                    if one_file_per_time_step:
                        image_file_name_matrix[i, j, k] = find_storm_image_file(
                            top_directory_name=top_output_dir_name,
                            unix_time_sec=valid_times_unix_sec[i],
                            spc_date_string=unique_spc_date_strings[q],
                            radar_source=radar_utils.GRIDRAD_SOURCE_ID,
                            radar_field_name=radar_field_names[j],
                            radar_height_m_asl=radar_heights_m_asl[k],
                            raise_error_if_missing=False)

                        print (
                            'Writing storm-centered images to: "{0:s}"...\n'
                        ).format(image_file_name_matrix[i, j, k])
                        write_storm_images(
                            netcdf_file_name=image_file_name_matrix[i, j, k],
                            storm_image_matrix=this_storm_image_matrix,
                            storm_ids=these_storm_ids,
                            valid_times_unix_sec=these_times_unix_sec,
                            radar_field_name=radar_field_names[j],
                            radar_height_m_asl=radar_heights_m_asl[k],
                            rotated_grids=rotate_grids,
                            rotated_grid_spacing_metres=
                            rotated_grid_spacing_metres)

                        continue

                    if this_date_storm_image_matrix is None:
                        this_date_storm_image_matrix = (
                            this_storm_image_matrix + 0.)
                    else:
                        this_date_storm_image_matrix = numpy.concatenate(
                            (this_date_storm_image_matrix,
                             this_storm_image_matrix), axis=0)

                if one_file_per_time_step:
                    continue

                image_file_name_matrix[q, j, k] = find_storm_image_file(
                    top_directory_name=top_output_dir_name,
                    spc_date_string=unique_spc_date_strings[q],
                    radar_source=radar_utils.GRIDRAD_SOURCE_ID,
                    radar_field_name=radar_field_names[j],
                    radar_height_m_asl=radar_heights_m_asl[k],
                    raise_error_if_missing=False)

                print (
                    'Writing storm-centered images to: "{0:s}"...\n'
                ).format(image_file_name_matrix[q, j, k])
                write_storm_images(
                    netcdf_file_name=image_file_name_matrix[q, j, k],
                    storm_image_matrix=this_date_storm_image_matrix,
                    storm_ids=this_date_storm_ids,
                    valid_times_unix_sec=this_date_storm_times_unix_sec,
                    radar_field_name=radar_field_names[j],
                    radar_height_m_asl=radar_heights_m_asl[k],
                    rotated_grids=rotate_grids,
                    rotated_grid_spacing_metres=rotated_grid_spacing_metres)

    return image_file_name_matrix


def get_max_rdp_for_each_storm_object(
        storm_object_table, top_gridrad_dir_name,
        horizontal_radius_metres=DEFAULT_HORIZ_RADIUS_FOR_RDP_METRES,
        min_vorticity_height_m_asl=DEFAULT_MIN_VORT_HEIGHT_FOR_RDP_M_ASL):
    """Computes max rotation-divergence product (RDP) for each storm object.

    This method works only for GridRad data, because unlike MYRORSS, GridRad
    includes 3-D vorticity and divergence fields.

    The "maximum RDP" for each storm object is max vorticity * max divergence,
    as defined below.

    "Max vorticity" = max vorticity at height >= `min_vorticity_height_m_asl`
    and within a horizontal radius of `horizontal_radius_metres` from the
    storm's horizontal centroid.

    "Max divergence" = max divergence at any height, within a horizontal radius
    of `horizontal_radius_metres` from the storm's horizontal centroid.

    L = number of storm objects

    :param storm_object_table: See doc for `extract_storm_images_gridrad`.
    :param top_gridrad_dir_name: Name of top-level directory with GridRad (files
        therein will be located by `gridrad_io.find_file` and read by
        `gridrad_io.read_field_from_full_grid_file`).
    :param horizontal_radius_metres: See general discussion above.
    :param min_vorticity_height_m_asl: See general discussion above.
    :return: max_rdp_values_s02: length-L numpy array with maximum RDP for each
        storm object.  Units are seconds^-2.
    """

    error_checking.assert_is_greater(horizontal_radius_metres, 0.)
    error_checking.assert_is_integer(min_vorticity_height_m_asl)
    error_checking.assert_is_greater(min_vorticity_height_m_asl, 0)

    storm_times_unix_sec = numpy.unique(
        storm_object_table[tracking_utils.TIME_COLUMN].values)
    storm_time_strings = [time_conversion.unix_sec_to_string(t, TIME_FORMAT)
                          for t in storm_times_unix_sec]

    num_storm_times = len(storm_time_strings)
    gridrad_file_names = [None] * num_storm_times
    for i in range(num_storm_times):
        gridrad_file_names[i] = gridrad_io.find_file(
            unix_time_sec=storm_times_unix_sec[i],
            top_directory_name=top_gridrad_dir_name,
            raise_error_if_missing=True)

    num_storm_objects = len(storm_object_table.index)
    max_rdp_values_s02 = numpy.full(num_storm_objects, numpy.nan)

    for i in range(num_storm_times):
        if i != 0:
            print '\n'
        this_metadata_dict = gridrad_io.read_metadata_from_full_grid_file(
            gridrad_file_names[i])

        print 'Reading "{0:s}" from file: "{1:s}"...'.format(
            radar_utils.DIVERGENCE_NAME, gridrad_file_names[i])
        (this_divergence_matrix_s01, these_grid_point_heights_m_asl,
         these_grid_point_latitudes_deg, these_grid_point_longitudes_deg
        ) = gridrad_io.read_field_from_full_grid_file(
            netcdf_file_name=gridrad_file_names[i],
            field_name=radar_utils.DIVERGENCE_NAME,
            metadata_dict=this_metadata_dict, raise_error_if_fails=True)

        these_grid_point_latitudes_deg = these_grid_point_latitudes_deg[::-1]
        this_divergence_matrix_s01 = numpy.flip(
            this_divergence_matrix_s01, axis=1)

        print 'Reading "{0:s}" from file: "{1:s}"...'.format(
            radar_utils.VORTICITY_NAME, gridrad_file_names[i])
        (this_vorticity_matrix_s01, _, _, _
        ) = gridrad_io.read_field_from_full_grid_file(
            netcdf_file_name=gridrad_file_names[i],
            field_name=radar_utils.VORTICITY_NAME,
            metadata_dict=this_metadata_dict, raise_error_if_fails=True)
        this_vorticity_matrix_s01 = numpy.flip(
            this_vorticity_matrix_s01, axis=1)

        print (
            'Computing rotation-divergence product for all storm objects at '
            '{0:s}...'
        ).format(storm_time_strings[i])
        these_storm_indices = numpy.where(
            storm_object_table[tracking_utils.TIME_COLUMN].values ==
            storm_times_unix_sec[i])[0]

        max_rdp_values_s02[these_storm_indices] = _get_max_rdp_values_one_time(
            divergence_matrix_s01=this_divergence_matrix_s01,
            vorticity_matrix_s01=this_vorticity_matrix_s01,
            grid_point_latitudes_deg=these_grid_point_latitudes_deg,
            grid_point_longitudes_deg=these_grid_point_longitudes_deg,
            grid_point_heights_m_asl=these_grid_point_heights_m_asl,
            min_vorticity_height_m_asl=min_vorticity_height_m_asl,
            storm_centroid_latitudes_deg=storm_object_table[
                tracking_utils.CENTROID_LAT_COLUMN].values[these_storm_indices],
            storm_centroid_longitudes_deg=storm_object_table[
                tracking_utils.CENTROID_LNG_COLUMN].values[these_storm_indices],
            horizontal_radius_metres=horizontal_radius_metres)

    return max_rdp_values_s02


def write_storm_images(
        netcdf_file_name, storm_image_matrix, storm_ids, valid_times_unix_sec,
        radar_field_name, radar_height_m_asl, rotated_grids=False,
        rotated_grid_spacing_metres=None, rotation_divergence_products_s02=None,
        horiz_radius_for_rdp_metres=None, min_vort_height_for_rdp_m_asl=None,
        num_storm_objects_per_chunk=1):
    """Writes storm-centered radar images to NetCDF file.

    These images should be created by `extract_storm_image`.

    :param netcdf_file_name: Path to output file.
    :param storm_image_matrix: See documentation for `_check_storm_images`.
    :param storm_ids: Same.
    :param valid_times_unix_sec: Same.
    :param radar_field_name: Same.
    :param radar_height_m_asl: Same.
    :param rotated_grids: Same.
    :param rotated_grid_spacing_metres: Same.
    :param rotation_divergence_products_s02: Same.
    :param horiz_radius_for_rdp_metres: Same.
    :param min_vort_height_for_rdp_m_asl: Same.
    :param num_storm_objects_per_chunk: Number of storm objects per NetCDF
        chunk.  To use default chunking, set this to `None`.
    """

    _check_storm_images(
        storm_image_matrix=storm_image_matrix, storm_ids=storm_ids,
        valid_times_unix_sec=valid_times_unix_sec,
        radar_field_name=radar_field_name,
        radar_height_m_asl=radar_height_m_asl, rotated_grids=rotated_grids,
        rotated_grid_spacing_metres=rotated_grid_spacing_metres,
        rotation_divergence_products_s02=rotation_divergence_products_s02,
        horiz_radius_for_rdp_metres=horiz_radius_for_rdp_metres,
        min_vort_height_for_rdp_m_asl=min_vort_height_for_rdp_m_asl)

    if num_storm_objects_per_chunk is not None:
        error_checking.assert_is_integer(num_storm_objects_per_chunk)
        error_checking.assert_is_geq(num_storm_objects_per_chunk, 1)

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    netcdf_dataset = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET')

    netcdf_dataset.setncattr(RADAR_FIELD_NAME_KEY, radar_field_name)
    netcdf_dataset.setncattr(RADAR_HEIGHT_KEY, radar_height_m_asl)
    netcdf_dataset.setncattr(ROTATED_GRIDS_KEY, int(rotated_grids))
    if rotated_grids:
        netcdf_dataset.setncattr(
            ROTATED_GRID_SPACING_KEY, rotated_grid_spacing_metres)

    if rotation_divergence_products_s02 is not None:
        netcdf_dataset.setncattr(
            HORIZ_RADIUS_FOR_RDP_KEY, horiz_radius_for_rdp_metres)
        netcdf_dataset.setncattr(
            MIN_VORT_HEIGHT_FOR_RDP_KEY, min_vort_height_for_rdp_m_asl)

    num_storm_objects = storm_image_matrix.shape[0]
    num_storm_id_chars = 1
    for i in range(num_storm_objects):
        num_storm_id_chars = max([num_storm_id_chars, len(storm_ids[i])])

    netcdf_dataset.createDimension(
        STORM_OBJECT_DIMENSION_KEY, num_storm_objects)
    netcdf_dataset.createDimension(
        ROW_DIMENSION_KEY, storm_image_matrix.shape[1])
    netcdf_dataset.createDimension(
        COLUMN_DIMENSION_KEY, storm_image_matrix.shape[2])
    netcdf_dataset.createDimension(CHARACTER_DIMENSION_KEY, num_storm_id_chars)

    netcdf_dataset.createVariable(
        STORM_IDS_KEY, datatype='S1',
        dimensions=(STORM_OBJECT_DIMENSION_KEY, CHARACTER_DIMENSION_KEY))

    string_type = 'S{0:d}'.format(num_storm_id_chars)
    storm_ids_as_char_array = netCDF4.stringtochar(numpy.array(
        storm_ids, dtype=string_type))
    netcdf_dataset.variables[STORM_IDS_KEY][:] = numpy.array(
        storm_ids_as_char_array)

    netcdf_dataset.createVariable(
        VALID_TIMES_KEY, datatype=numpy.int32,
        dimensions=STORM_OBJECT_DIMENSION_KEY)
    netcdf_dataset.variables[VALID_TIMES_KEY][:] = valid_times_unix_sec

    if rotation_divergence_products_s02 is not None:
        netcdf_dataset.createVariable(
            ROTATION_DIVERGENCE_PRODUCTS_KEY, datatype=numpy.float32,
            dimensions=STORM_OBJECT_DIMENSION_KEY)
        netcdf_dataset.variables[
            ROTATION_DIVERGENCE_PRODUCTS_KEY
        ][:] = rotation_divergence_products_s02

    if num_storm_objects_per_chunk is None:
        chunk_size_tuple = None
    else:
        chunk_size_tuple = (
            (num_storm_objects_per_chunk,) + storm_image_matrix.shape[1:])

    netcdf_dataset.createVariable(
        STORM_IMAGE_MATRIX_KEY, datatype=numpy.float32,
        dimensions=(STORM_OBJECT_DIMENSION_KEY, ROW_DIMENSION_KEY,
                    COLUMN_DIMENSION_KEY),
        chunksizes=chunk_size_tuple)

    netcdf_dataset.variables[STORM_IMAGE_MATRIX_KEY][:] = storm_image_matrix
    netcdf_dataset.close()


def read_storm_images(
        netcdf_file_name, return_images=True, storm_ids_to_keep=None,
        valid_times_to_keep_unix_sec=None):
    """Reads storm-centered radar images from NetCDF file.

    L = number of storm objects returned

    :param netcdf_file_name: Path to input file.
    :param return_images: Boolean flag.  If True, will return images and
        metadata.  If False, will return only metadata.
    :param storm_ids_to_keep:
        [used only if `return_images = True`]
        length-L list with string ID of storm objects to keep.
    :param valid_times_to_keep_unix_sec:
        [used only if `return_images = True`]
        length-L numpy array with valid times of storm objects to keep.
    :return: storm_image_dict: Dictionary with the following keys.
    storm_image_dict['storm_image_matrix']: See documentation for
        `_check_storm_images`.
    storm_image_dict['storm_ids']: Same.
    storm_image_dict['valid_times_unix_sec']: Same.
    storm_image_dict['radar_field_name']: Same.
    storm_image_dict['radar_height_m_asl']: Same.
    storm_image_dict['rotated_grids']: Same.
    storm_image_dict['rotated_grid_spacing_key']: Same.
    storm_image_dict['rotation_divergence_products_s02']: Same.
    storm_image_dict['horiz_radius_for_rdp_metres']: Same.
    storm_image_dict['min_vort_height_for_rdp_m_asl']: Same.
    """

    error_checking.assert_is_boolean(return_images)
    netcdf_dataset = netcdf_io.open_netcdf(
        netcdf_file_name=netcdf_file_name, raise_error_if_fails=True)

    radar_field_name = str(getattr(netcdf_dataset, RADAR_FIELD_NAME_KEY))
    radar_height_m_asl = getattr(netcdf_dataset, RADAR_HEIGHT_KEY)
    rotated_grids = bool(getattr(netcdf_dataset, ROTATED_GRIDS_KEY))
    if rotated_grids:
        rotated_grid_spacing_metres = getattr(
            netcdf_dataset, ROTATED_GRID_SPACING_KEY)
    else:
        rotated_grid_spacing_metres = None

    if ROTATION_DIVERGENCE_PRODUCTS_KEY in netcdf_dataset.variables:
        horiz_radius_for_rdp_metres = getattr(
            netcdf_dataset, HORIZ_RADIUS_FOR_RDP_KEY)
        min_vort_height_for_rdp_m_asl = getattr(
            netcdf_dataset, MIN_VORT_HEIGHT_FOR_RDP_KEY)
    else:
        horiz_radius_for_rdp_metres = None
        min_vort_height_for_rdp_m_asl = None

    num_storm_objects = netcdf_dataset.variables[STORM_IDS_KEY].shape[0]
    if num_storm_objects == 0:
        storm_ids = []
        valid_times_unix_sec = numpy.array([], dtype=int)
        if ROTATION_DIVERGENCE_PRODUCTS_KEY in netcdf_dataset.variables:
            rotation_divergence_products_s02 = numpy.array([], dtype=float)
        else:
            rotation_divergence_products_s02 = None

    else:
        storm_ids = netCDF4.chartostring(
            netcdf_dataset.variables[STORM_IDS_KEY][:])
        storm_ids = [str(s) for s in storm_ids]
        valid_times_unix_sec = numpy.array(
            netcdf_dataset.variables[VALID_TIMES_KEY][:], dtype=int)

        if ROTATION_DIVERGENCE_PRODUCTS_KEY in netcdf_dataset.variables:
            rotation_divergence_products_s02 = numpy.array(
                netcdf_dataset.variables[ROTATION_DIVERGENCE_PRODUCTS_KEY][:])
        else:
            rotation_divergence_products_s02 = None

    if not return_images:
        return {
            STORM_IDS_KEY: storm_ids,
            VALID_TIMES_KEY: valid_times_unix_sec,
            RADAR_FIELD_NAME_KEY: radar_field_name,
            RADAR_HEIGHT_KEY: radar_height_m_asl,
            ROTATED_GRIDS_KEY: rotated_grids,
            ROTATED_GRID_SPACING_KEY: rotated_grid_spacing_metres,
            ROTATION_DIVERGENCE_PRODUCTS_KEY: rotation_divergence_products_s02,
            HORIZ_RADIUS_FOR_RDP_KEY: horiz_radius_for_rdp_metres,
            MIN_VORT_HEIGHT_FOR_RDP_KEY: min_vort_height_for_rdp_m_asl
        }

    filter_storms = not(
        storm_ids_to_keep is None or valid_times_to_keep_unix_sec is None)
    if filter_storms:
        error_checking.assert_is_string_list(storm_ids_to_keep)
        error_checking.assert_is_numpy_array(
            numpy.array(storm_ids_to_keep), num_dimensions=1)
        num_storm_objects_to_keep = len(storm_ids_to_keep)

        error_checking.assert_is_integer_numpy_array(
            valid_times_to_keep_unix_sec)
        error_checking.assert_is_numpy_array(
            valid_times_to_keep_unix_sec,
            exact_dimensions=numpy.array([num_storm_objects_to_keep]))

        indices_to_keep = find_storm_objects(
            all_storm_ids=storm_ids,
            all_valid_times_unix_sec=valid_times_unix_sec,
            storm_ids_to_keep=storm_ids_to_keep,
            valid_times_to_keep_unix_sec=valid_times_to_keep_unix_sec)

        storm_ids = [storm_ids[i] for i in indices_to_keep]
        valid_times_unix_sec = valid_times_unix_sec[indices_to_keep]
        if ROTATION_DIVERGENCE_PRODUCTS_KEY in netcdf_dataset.variables:
            rotation_divergence_products_s02 = rotation_divergence_products_s02[
                indices_to_keep]
    else:
        indices_to_keep = numpy.linspace(
            0, num_storm_objects - 1, num=num_storm_objects, dtype=int)

    if len(indices_to_keep):
        storm_image_matrix = numpy.array(
            netcdf_dataset.variables[STORM_IMAGE_MATRIX_KEY][
                indices_to_keep, ...])
    else:
        num_rows = netcdf_dataset.dimensions[ROW_DIMENSION_KEY].size
        num_columns = netcdf_dataset.dimensions[COLUMN_DIMENSION_KEY].size
        storm_image_matrix = numpy.full((0, num_rows, num_columns), 0.)

    netcdf_dataset.close()

    return {
        STORM_IMAGE_MATRIX_KEY: storm_image_matrix,
        STORM_IDS_KEY: storm_ids,
        VALID_TIMES_KEY: valid_times_unix_sec,
        RADAR_FIELD_NAME_KEY: radar_field_name,
        RADAR_HEIGHT_KEY: radar_height_m_asl,
        ROTATED_GRIDS_KEY: rotated_grids,
        ROTATED_GRID_SPACING_KEY: rotated_grid_spacing_metres,
        ROTATION_DIVERGENCE_PRODUCTS_KEY: rotation_divergence_products_s02,
        HORIZ_RADIUS_FOR_RDP_KEY: horiz_radius_for_rdp_metres,
        MIN_VORT_HEIGHT_FOR_RDP_KEY: min_vort_height_for_rdp_m_asl
    }


def read_storm_images_and_labels(
        image_file_name, label_file_name, label_name,
        num_storm_objects_class_dict=None):
    """Reads storm-centered radar images and corresponding hazard labels.

    If no desired storm objects are found, this method returns `None`.

    :param image_file_name: Path to file with storm-centered radar images (will
        be read by `read_storm_images`).
    :param label_file_name: Path to file with hazard labels (will be read by
        `labels.read_labels_from_netcdf`).
    :param label_name: Name of hazard label (target variable).
    :param num_storm_objects_class_dict: Dictionary, where each key is a class
        integer (-2 for dead storms) and each value is the corresponding number
        of storm objects desired.
    :return storm_image_dict: Dictionary with the following keys.
    storm_image_dict['storm_image_matrix']: See documentation for
        `_check_storm_images`.
    storm_image_dict['storm_ids']: Same.
    storm_image_dict['valid_times_unix_sec']: Same.
    storm_image_dict['radar_field_name']: Same.
    storm_image_dict['radar_height_m_asl']: Same.
    storm_image_dict['rotation_divergence_products_s02']: Same.
    storm_image_dict['horiz_radius_for_rdp_metres']: Same.
    storm_image_dict['min_vort_height_for_rdp_m_asl']: Same.
    storm_image_dict['label_values']: 1-D numpy array with label for each storm
        object.
    """

    storm_label_dict = labels.read_labels_from_netcdf(
        netcdf_file_name=label_file_name, label_name=label_name)
    storm_to_events_dict = {
        tracking_utils.STORM_ID_COLUMN:
            storm_label_dict[labels.STORM_IDS_KEY],
        tracking_utils.TIME_COLUMN:
            storm_label_dict[labels.VALID_TIMES_KEY],
        label_name: storm_label_dict[labels.LABEL_VALUES_KEY]
    }

    storm_to_winds_table = None
    storm_to_tornadoes_table = None
    parameter_dict = labels.column_name_to_label_params(label_name)
    event_type_string = parameter_dict[labels.EVENT_TYPE_KEY]

    if event_type_string == events2storms.WIND_EVENT_TYPE_STRING:
        storm_to_winds_table = pandas.DataFrame.from_dict(
            storm_to_events_dict)
    else:
        storm_to_tornadoes_table = pandas.DataFrame.from_dict(
            storm_to_events_dict)

    storm_image_dict = read_storm_images(
        netcdf_file_name=image_file_name, return_images=False)

    label_values = extract_storm_labels_with_name(
        storm_ids=storm_image_dict[STORM_IDS_KEY],
        valid_times_unix_sec=storm_image_dict[VALID_TIMES_KEY],
        label_name=label_name, storm_to_winds_table=storm_to_winds_table,
        storm_to_tornadoes_table=storm_to_tornadoes_table)

    if num_storm_objects_class_dict is None:
        indices_to_keep = numpy.linspace(
            0, len(label_values) - 1, num=len(label_values), dtype=int)
    else:
        indices_to_keep = _filter_storm_objects_by_label(
            label_values=label_values,
            num_storm_objects_class_dict=num_storm_objects_class_dict)

    if not len(indices_to_keep):
        return None

    storm_ids_to_keep = [
        storm_image_dict[STORM_IDS_KEY][i] for i in indices_to_keep]
    valid_times_to_keep_unix_sec = storm_image_dict[
        VALID_TIMES_KEY][indices_to_keep]

    storm_image_dict = read_storm_images(
        netcdf_file_name=image_file_name, return_images=True,
        storm_ids_to_keep=storm_ids_to_keep,
        valid_times_to_keep_unix_sec=valid_times_to_keep_unix_sec)
    label_values = label_values[indices_to_keep]

    storm_image_dict.update({LABEL_VALUES_KEY: label_values})
    return storm_image_dict


def find_storm_image_file(
        top_directory_name, spc_date_string, radar_source, radar_field_name,
        radar_height_m_asl, unix_time_sec=None, raise_error_if_missing=True):
    """Finds file with storm-centered radar images.

    If `unix_time_sec is None`, this method finds a file with images for one SPC
    date.  Otherwise, finds a file with images for one time step.

    Both file types should be written by `write_storm_images`.

    :param top_directory_name: Name of top-level directory with storm-image
        files.
    :param spc_date_string: SPC date (format "yyyymmdd").
    :param radar_source: Data source (must be accepted by
        `radar_utils.check_data_source`).
    :param radar_field_name: Name of radar field (string).
    :param radar_height_m_asl: Height (metres above sea level) of radar field.
    :param unix_time_sec: Valid time.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        raise_error_if_missing = True, this method will error out.
    :return: storm_image_file_name: Path to image file.  If file is missing and
        raise_error_if_missing = False, this is the *expected* path.
    :raises: ValueError: if file is missing and raise_error_if_missing = True.
    """

    error_checking.assert_is_string(top_directory_name)
    _ = time_conversion.spc_date_string_to_unix_sec(spc_date_string)
    radar_utils.check_data_source(radar_source)
    error_checking.assert_is_string(radar_field_name)
    error_checking.assert_is_greater(radar_height_m_asl, 0)
    error_checking.assert_is_boolean(raise_error_if_missing)

    if unix_time_sec is None:
        storm_image_file_name = (
            '{0:s}/{1:s}/{2:s}/{3:s}/{4:05d}_metres_asl/storm_images_{5:s}.nc'
        ).format(
            top_directory_name, radar_source, spc_date_string[:4],
            radar_field_name, int(numpy.round(radar_height_m_asl)),
            spc_date_string)
    else:
        storm_image_file_name = (
            '{0:s}/{1:s}/{2:s}/{3:s}/{4:s}/{5:05d}_metres_asl/'
            'storm_images_{6:s}.nc'
        ).format(
            top_directory_name, radar_source, spc_date_string[:4],
            spc_date_string, radar_field_name,
            int(numpy.round(radar_height_m_asl)),
            time_conversion.unix_sec_to_string(unix_time_sec, TIME_FORMAT))

    if raise_error_if_missing and not os.path.isfile(storm_image_file_name):
        error_string = (
            'Cannot find file with storm-centered radar images.  Expected at: '
            '{0:s}').format(storm_image_file_name)
        raise ValueError(error_string)

    return storm_image_file_name


def image_file_name_to_time(storm_image_file_name):
    """Parses time from name of storm-image file.

    :param storm_image_file_name: Path to input file.
    :return: unix_time_sec: Valid time.  If the file contains data for one SPC
        date (rather than one time step), this will be None.
    :return: spc_date_string: SPC date (format "yyyymmdd").
    """

    directory_name, pathless_file_name = os.path.split(storm_image_file_name)
    extensionless_file_name, _ = os.path.splitext(pathless_file_name)
    time_string = extensionless_file_name.split('_')[-1]

    try:
        time_conversion.spc_date_string_to_unix_sec(time_string)
        return None, time_string
    except:
        pass

    unix_time_sec = time_conversion.string_to_unix_sec(time_string, TIME_FORMAT)
    spc_date_string = directory_name.split('/')[-3]
    time_conversion.spc_date_string_to_unix_sec(spc_date_string)

    return unix_time_sec, spc_date_string


def image_file_name_to_field(storm_image_file_name):
    """Parses radar field from name of storm-image file.

    :param storm_image_file_name: Path to input file.
    :return: radar_field_name: Name of radar field.
    :raises: ValueError: if radar field cannot be parsed from file name.
    """

    subdirectory_names = os.path.split(storm_image_file_name)[0].split('/')
    for this_subdir_name in subdirectory_names:
        try:
            radar_utils.check_field_name(this_subdir_name)
            return this_subdir_name
        except:
            pass

    error_string = 'Cannot parse radar field from file name: "{0:s}"'.format(
        storm_image_file_name)
    raise ValueError(error_string)


def image_file_name_to_height(storm_image_file_name):
    """Parses radar height from name of storm-image file.

    :param storm_image_file_name: Path to input file.
    :return: radar_height_m_asl: Radar height (metres above sea level).
    :raises: ValueError: if radar height cannot be parsed from file name.
    """

    keyword = '_metres_asl'
    subdirectory_names = os.path.split(storm_image_file_name)[0].split('/')

    for this_subdir_name in subdirectory_names:
        if keyword in this_subdir_name:
            return int(this_subdir_name.replace(keyword, ''))

    error_string = 'Cannot parse radar height from file name: "{0:s}"'.format(
        storm_image_file_name)
    raise ValueError(error_string)


def find_storm_label_file(
        storm_image_file_name, top_label_directory_name, label_name,
        one_file_per_spc_date=False, raise_error_if_missing=True,
        warn_if_missing=True):
    """Finds file with storm-hazard labels.

    This file should be written by `labels.write_wind_speed_labels` or
    `labels.write_tornado_labels`.

    :param storm_image_file_name: Path to file with storm-centered radar images.
    :param top_label_directory_name: Name of top-level directory with label
        file.
    :param label_name: Name of label.
    :param one_file_per_spc_date: Boolean flag.  If True, will find one label
        file for the corresponding SPC date.  If False, will find one label file
        for the corresponding time step.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        raise_error_if_missing = True, this method will error out.
    :param warn_if_missing: Boolean flag.  If file is missing,
        raise_error_if_missing = False, and warn_if_missing = True, this method
        will print a warning message.
    :return: storm_label_file_name: Path to label file.  If file is missing and
        raise_error_if_missing = False, this is the *expected* path.
    :raises: ValueError: if `one_file_per_spc_date` = False and
        `storm_image_file_name` contains data for one SPC rather than one time
        step.
    """

    error_checking.assert_is_boolean(one_file_per_spc_date)
    error_checking.assert_is_boolean(warn_if_missing)
    unix_time_sec, spc_date_string = image_file_name_to_time(
        storm_image_file_name)
    if one_file_per_spc_date:
        unix_time_sec = None

    parameter_dict = labels.column_name_to_label_params(label_name)
    storm_label_file_name = labels.find_label_file(
        top_directory_name=top_label_directory_name,
        event_type_string=parameter_dict[labels.EVENT_TYPE_KEY],
        file_extension=LABEL_FILE_EXTENSION,
        raise_error_if_missing=raise_error_if_missing,
        unix_time_sec=unix_time_sec, spc_date_string=spc_date_string)

    if not os.path.isfile(storm_label_file_name) and warn_if_missing:
        warning_string = (
            'POTENTIAL PROBLEM.  Cannot find file with storm-hazard labels, '
            'expected at "{0:s}".'
        ).format(storm_label_file_name)
        print warning_string

    return storm_label_file_name


def find_many_files_myrorss_or_mrms(
        top_directory_name, radar_source, radar_field_names,
        start_time_unix_sec, end_time_unix_sec, one_file_per_time_step=True,
        reflectivity_heights_m_asl=None, raise_error_if_all_missing=True,
        raise_error_if_any_missing=False):
    """Finds many files containing storm images with MYRORSS or MRMS data.

    If `one_file_per_time_step = True`, this method will look for files with one
    time step each.  Otherwise, will look for files with one SPC date each.

    T = number of time steps or SPC dates (in general, number of values along
        the time dimension)
    C = number of field/height pairs

    :param top_directory_name: Name of top-level directory with storm-image
        files.
    :param radar_source: Source (must be accepted by
        `radar_utils.check_data_source`).
    :param radar_field_names: 1-D list with names of radar fields.
    :param start_time_unix_sec: See documentation for `find_many_files_gridrad`.
    :param end_time_unix_sec: Same.
    :param one_file_per_time_step: Same.
    :param reflectivity_heights_m_asl: 1-D numpy array of heights for field
        "reflectivity_dbz".
    :param raise_error_if_all_missing: Boolean flag.  If no files are found and
        `raise_error_if_all_missing` = True, this method will error out.
    :param raise_error_if_any_missing: Boolean flag.  If any file is missing and
        `raise_error_if_any_missing` = True, this method will error out.
    :return: file_dict: Dictionary with the following keys.
    file_dict['image_file_name_matrix']: T-by-C numpy array of file paths.
    file_dict['valid_times_unix_sec']: length-T numpy array of valid times.  If
        `one_file_per_time_step is False`, valid_times_unix_sec[i] is just a
        time on the [i]th SPC date.
    file_dict['field_name_by_pair']: length-C list with names of radar fields.
    file_dict['height_by_pair_m_asl']: length-C numpy array of radar heights
        (metres above sea level).
    :raises: ValueError: If no files are found and `raise_error_if_all_missing`
        = True.
    """

    error_checking.assert_is_boolean(one_file_per_time_step)
    error_checking.assert_is_boolean(raise_error_if_all_missing)
    error_checking.assert_is_boolean(raise_error_if_any_missing)

    field_name_by_pair, height_by_pair_m_asl = (
        myrorss_and_mrms_utils.fields_and_refl_heights_to_pairs(
            field_names=radar_field_names, data_source=radar_source,
            refl_heights_m_asl=reflectivity_heights_m_asl))
    num_field_height_pairs = len(field_name_by_pair)

    file_dict = {
        RADAR_FIELD_NAME_BY_PAIR_KEY: field_name_by_pair,
        RADAR_HEIGHT_BY_PAIR_KEY: height_by_pair_m_asl
    }

    first_spc_date_string = time_conversion.time_to_spc_date_string(
        start_time_unix_sec)
    last_spc_date_string = time_conversion.time_to_spc_date_string(
        end_time_unix_sec)
    all_spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string)

    if one_file_per_time_step:
        image_file_name_matrix = None
        valid_times_unix_sec = None

        for i in range(len(all_spc_date_strings)):
            print 'Finding storm-image files for SPC date "{0:s}"...'.format(
                all_spc_date_strings[i])

            this_file_name_matrix, these_times_unix_sec = (
                _find_many_files_one_spc_date(
                    top_directory_name=top_directory_name,
                    start_time_unix_sec=start_time_unix_sec,
                    end_time_unix_sec=end_time_unix_sec,
                    spc_date_string=all_spc_date_strings[i],
                    radar_source=radar_source,
                    field_name_by_pair=field_name_by_pair,
                    height_by_pair_m_asl=height_by_pair_m_asl,
                    raise_error_if_all_missing=False,
                    raise_error_if_any_missing=raise_error_if_any_missing))

            if this_file_name_matrix is None:
                continue

            if image_file_name_matrix is None:
                image_file_name_matrix = copy.deepcopy(this_file_name_matrix)
                valid_times_unix_sec = these_times_unix_sec + 0
            else:
                image_file_name_matrix = numpy.concatenate(
                    (image_file_name_matrix, this_file_name_matrix), axis=0)
                valid_times_unix_sec = numpy.concatenate((
                    valid_times_unix_sec, these_times_unix_sec))

        if raise_error_if_all_missing and image_file_name_matrix is None:
            start_time_string = time_conversion.unix_sec_to_string(
                start_time_unix_sec, TIME_FORMAT)
            end_time_string = time_conversion.unix_sec_to_string(
                end_time_unix_sec, TIME_FORMAT)
            error_string = 'Cannot find any files from {0:s} to {1:s}.'.format(
                start_time_string, end_time_string)
            raise ValueError(error_string)

        file_dict.update({
            IMAGE_FILE_NAME_MATRIX_KEY: image_file_name_matrix,
            VALID_TIMES_KEY: valid_times_unix_sec
        })
        return file_dict

    image_file_name_matrix = None
    valid_spc_date_strings = None
    valid_times_unix_sec = None

    for j in range(num_field_height_pairs):
        print (
            'Finding storm-image files for "{0:s}" at {1:d} metres ASL...'
        ).format(field_name_by_pair[j],
                 int(numpy.round(height_by_pair_m_asl[j])))

        if j == 0:
            image_file_names = []
            valid_spc_date_strings = []

            for i in range(len(all_spc_date_strings)):
                this_file_name = find_storm_image_file(
                    top_directory_name=top_directory_name,
                    spc_date_string=all_spc_date_strings[i],
                    radar_source=radar_source,
                    radar_field_name=field_name_by_pair[j],
                    radar_height_m_asl=height_by_pair_m_asl[j],
                    raise_error_if_missing=raise_error_if_any_missing)

                if not os.path.isfile(this_file_name):
                    continue

                image_file_names.append(this_file_name)
                valid_spc_date_strings.append(all_spc_date_strings[i])

            num_times = len(image_file_names)
            if num_times == 0:
                if raise_error_if_all_missing:
                    error_string = (
                        'Cannot find any files from SPC dates "{0:s}" to '
                        '"{1:s}".'
                    ).format(all_spc_date_strings[0],
                             all_spc_date_strings[-1])
                    raise ValueError(error_string)

                file_dict.update({
                    IMAGE_FILE_NAME_MATRIX_KEY: None,
                    VALID_TIMES_KEY: None
                })
                return file_dict

            image_file_name_matrix = numpy.full(
                (num_times, num_field_height_pairs), '', dtype=object)
            image_file_name_matrix[:, j] = numpy.array(
                image_file_names, dtype=object)

            valid_times_unix_sec = numpy.array(
                [time_conversion.spc_date_string_to_unix_sec(s)
                 for s in valid_spc_date_strings], dtype=int)

        else:
            for i in range(len(valid_spc_date_strings)):
                image_file_name_matrix[i, j] = find_storm_image_file(
                    top_directory_name=top_directory_name,
                    spc_date_string=valid_spc_date_strings[i],
                    radar_source=radar_source,
                    radar_field_name=field_name_by_pair[j],
                    radar_height_m_asl=height_by_pair_m_asl[j],
                    raise_error_if_missing=raise_error_if_any_missing)

                if not os.path.isfile(image_file_name_matrix[i, j]):
                    image_file_name_matrix[i, j] = ''

    file_dict.update({
        IMAGE_FILE_NAME_MATRIX_KEY: image_file_name_matrix,
        VALID_TIMES_KEY: valid_times_unix_sec
    })
    return file_dict


def find_many_files_gridrad(
        top_directory_name, radar_field_names, radar_heights_m_asl,
        start_time_unix_sec, end_time_unix_sec, one_file_per_time_step=True,
        raise_error_if_all_missing=True):
    """Finds many files containing storm images with GridRad data.

    If `one_file_per_time_step = True`, this method will look for files with one
    time step each.  Otherwise, will look for files with one SPC date each.

    T = number of time steps or SPC dates (in general, number of values along
        the time dimension)
    F = number of radar fields
    H = number of radar heights

    :param top_directory_name: Name of top-level directory with storm-image
        files.
    :param radar_field_names: length-F list with names of radar fields.
    :param radar_heights_m_asl: length-H numpy array of radar heights (metres
        above sea level).
    :param start_time_unix_sec: Start time.  This method will find all files
        from `start_time_unix_sec`...`end_time_unix_sec`.  If
        `one_file_per_time_step is False`, this can be any time on the first SPC
        date.
    :param end_time_unix_sec: End time.  If `one_file_per_time_step is False`,
        this can be any time on the last SPC date.
    :param one_file_per_time_step: See general discussion above.
    :param raise_error_if_all_missing: Boolean flag.  If no files are found and
        `raise_error_if_all_missing` = True, this method will error out.
    :return: file_dict: Dictionary with the following keys.
    file_dict['image_file_name_matrix']: T-by-F-by-H numpy array of file paths.
    file_dict['valid_times_unix_sec']: length-T numpy array of valid times.  If
        `one_file_per_time_step is False`, valid_times_unix_sec[i] is just a
        time on the [i]th SPC date.
    file_dict['radar_field_names']: length-F list with names of radar fields.
    file_dict['radar_heights_m_asl']: length-H numpy array of radar heights
        (metres above sea level).
    :raises: ValueError: If no files are found and `raise_error_if_all_missing`
        = True.
    """

    _, _ = gridrad_utils.fields_and_refl_heights_to_pairs(
        field_names=radar_field_names, heights_m_asl=radar_heights_m_asl)
    file_dict = {
        RADAR_FIELD_NAMES_KEY: radar_field_names,
        RADAR_HEIGHTS_KEY: radar_heights_m_asl
    }

    error_checking.assert_is_boolean(one_file_per_time_step)
    error_checking.assert_is_boolean(raise_error_if_all_missing)

    if one_file_per_time_step:
        all_times_unix_sec = time_periods.range_and_interval_to_list(
            start_time_unix_sec=start_time_unix_sec,
            end_time_unix_sec=end_time_unix_sec,
            time_interval_sec=GRIDRAD_TIME_INTERVAL_SEC, include_endpoint=True)

        all_spc_date_strings = [time_conversion.time_to_spc_date_string(t)
                                for t in all_times_unix_sec]

    else:
        first_spc_date_string = time_conversion.time_to_spc_date_string(
            start_time_unix_sec)
        last_spc_date_string = time_conversion.time_to_spc_date_string(
            end_time_unix_sec)

        all_spc_date_strings = time_conversion.get_spc_dates_in_range(
            first_spc_date_string=first_spc_date_string,
            last_spc_date_string=last_spc_date_string)
        all_times_unix_sec = numpy.array(
            [time_conversion.spc_date_string_to_unix_sec(s)
             for s in all_spc_date_strings], dtype=int)

    image_file_name_matrix = None
    valid_times_unix_sec = None
    valid_spc_date_strings = None
    num_fields = len(radar_field_names)
    num_heights = len(radar_heights_m_asl)

    for j in range(num_fields):
        for k in range(num_heights):
            print (
                'Finding storm-image files for "{0:s}" at {1:d} metres ASL...'
            ).format(radar_field_names[j],
                     int(numpy.round(radar_heights_m_asl[k])))

            if j == 0 and k == 0:
                image_file_names = []
                valid_times_unix_sec = []
                valid_spc_date_strings = []

                for i in range(len(all_times_unix_sec)):
                    if one_file_per_time_step:
                        this_time_unix_sec = all_times_unix_sec[i]
                    else:
                        this_time_unix_sec = None

                    this_file_name = find_storm_image_file(
                        top_directory_name=top_directory_name,
                        unix_time_sec=this_time_unix_sec,
                        spc_date_string=all_spc_date_strings[i],
                        radar_source=radar_utils.GRIDRAD_SOURCE_ID,
                        radar_field_name=radar_field_names[j],
                        radar_height_m_asl=radar_heights_m_asl[k],
                        raise_error_if_missing=False)

                    if not os.path.isfile(this_file_name):
                        continue

                    image_file_names.append(this_file_name)
                    valid_times_unix_sec.append(all_times_unix_sec[i])
                    valid_spc_date_strings.append(all_spc_date_strings[i])

                num_times = len(image_file_names)
                if num_times == 0:
                    if raise_error_if_all_missing and one_file_per_time_step:
                        start_time_string = (
                            time_conversion.unix_sec_to_string(
                                start_time_unix_sec, TIME_FORMAT))
                        end_time_string = (
                            time_conversion.unix_sec_to_string(
                                end_time_unix_sec, TIME_FORMAT))
                        error_string = (
                            'Cannot find any files from {0:s} to {1:s}.'
                        ).format(start_time_string, end_time_string)
                        raise ValueError(error_string)

                    if raise_error_if_all_missing:
                        error_string = (
                            'Cannot find any files from SPC dates "{0:s}" to '
                            '"{1:s}".'
                        ).format(all_spc_date_strings[0],
                                 all_spc_date_strings[-1])
                        raise ValueError(error_string)

                    file_dict.update({
                        IMAGE_FILE_NAME_MATRIX_KEY: None,
                        VALID_TIMES_KEY: None
                    })
                    return file_dict

                image_file_name_matrix = numpy.full(
                    (num_times, num_fields, num_heights), '', dtype=object)
                image_file_name_matrix[:, j, k] = numpy.array(
                    image_file_names, dtype=object)
                valid_times_unix_sec = numpy.array(
                    valid_times_unix_sec, dtype=int)

            else:
                for i in range(len(valid_times_unix_sec)):
                    if one_file_per_time_step:
                        this_time_unix_sec = valid_times_unix_sec[i]
                    else:
                        this_time_unix_sec = None

                    image_file_name_matrix[i, j, k] = find_storm_image_file(
                        top_directory_name=top_directory_name,
                        unix_time_sec=this_time_unix_sec,
                        spc_date_string=valid_spc_date_strings[i],
                        radar_source=radar_utils.GRIDRAD_SOURCE_ID,
                        radar_field_name=radar_field_names[j],
                        radar_height_m_asl=radar_heights_m_asl[k],
                        raise_error_if_missing=True)

    file_dict.update({
        IMAGE_FILE_NAME_MATRIX_KEY: image_file_name_matrix,
        VALID_TIMES_KEY: valid_times_unix_sec
    })
    return file_dict


def extract_storm_labels_with_name(
        storm_ids, valid_times_unix_sec, label_name, storm_to_winds_table=None,
        storm_to_tornadoes_table=None):
    """Extracts one label (target variable) for each storm object.

    L = number of storm objects

    :param storm_ids: See documentation for `_check_storm_labels`.
    :param valid_times_unix_sec: Same.
    :param label_name: Name of target variable.
    :param storm_to_winds_table: See doc for `_check_storm_labels`.
    :param storm_to_tornadoes_table: Same.
    :return: label_values: length-L numpy array of integers.  label_values[i] is
        the class for the [i]th storm object.
    """

    metadata_dict = labels.column_name_to_label_params(label_name)
    event_type_string = metadata_dict[labels.EVENT_TYPE_KEY]

    if event_type_string == events2storms.TORNADO_EVENT_TYPE_STRING:
        _, relevant_storm_to_tornadoes_table = _check_storm_labels(
            storm_ids=storm_ids, valid_times_unix_sec=valid_times_unix_sec,
            storm_to_winds_table=None,
            storm_to_tornadoes_table=storm_to_tornadoes_table)

        return relevant_storm_to_tornadoes_table[label_name].values

    relevant_storm_to_winds_table, _ = _check_storm_labels(
        storm_ids=storm_ids, valid_times_unix_sec=valid_times_unix_sec,
        storm_to_winds_table=storm_to_winds_table,
        storm_to_tornadoes_table=None)
    return relevant_storm_to_winds_table[label_name].values
