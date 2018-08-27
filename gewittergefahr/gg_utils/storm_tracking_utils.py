"""Processing methods for storm-tracking data (both polygons and tracks)."""

import copy
import numpy
import pandas
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import geodetic_utils
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import error_checking

TOLERANCE = 1e-6
TIME_FORMAT_FOR_LOG_MESSAGES = '%Y-%m-%d-%H%M%S'

SEGMOTION_SOURCE_ID = 'segmotion'
PROBSEVERE_SOURCE_ID = 'probSevere'
DATA_SOURCE_IDS = [SEGMOTION_SOURCE_ID, PROBSEVERE_SOURCE_ID]

STORM_ID_COLUMN = 'storm_id'
TIME_COLUMN = 'unix_time_sec'
SPC_DATE_COLUMN = 'spc_date_unix_sec'
EAST_VELOCITY_COLUMN = 'east_velocity_m_s01'
NORTH_VELOCITY_COLUMN = 'north_velocity_m_s01'
AGE_COLUMN = 'age_sec'

CENTROID_LAT_COLUMN = 'centroid_lat_deg'
CENTROID_LNG_COLUMN = 'centroid_lng_deg'
GRID_POINT_LAT_COLUMN = 'grid_point_latitudes_deg'
GRID_POINT_LNG_COLUMN = 'grid_point_longitudes_deg'
GRID_POINT_ROW_COLUMN = 'grid_point_rows'
GRID_POINT_COLUMN_COLUMN = 'grid_point_columns'
POLYGON_OBJECT_LATLNG_COLUMN = 'polygon_object_latlng'
POLYGON_OBJECT_ROWCOL_COLUMN = 'polygon_object_rowcol'
TRACKING_START_TIME_COLUMN = 'tracking_start_time_unix_sec'
TRACKING_END_TIME_COLUMN = 'tracking_end_time_unix_sec'
CELL_START_TIME_COLUMN = 'cell_start_time_unix_sec'
CELL_END_TIME_COLUMN = 'cell_end_time_unix_sec'

BUFFER_POLYGON_COLUMN_PREFIX = 'polygon_object_latlng_buffer'
ORIG_STORM_ID_COLUMN = 'original_storm_id'

FLATTENED_INDEX_COLUMN = 'flattened_index'
COLUMNS_TO_CHANGE_WHEN_MERGING_SCALES = [
    CENTROID_LAT_COLUMN, CENTROID_LNG_COLUMN, GRID_POINT_LAT_COLUMN,
    GRID_POINT_LNG_COLUMN, GRID_POINT_ROW_COLUMN, GRID_POINT_COLUMN_COLUMN,
    POLYGON_OBJECT_LATLNG_COLUMN, POLYGON_OBJECT_ROWCOL_COLUMN]


def _get_grid_points_in_storms(
        storm_object_table, num_grid_rows, num_grid_columns):
    """Finds grid points in each storm object.

    N = number of storm objects
    P = number of grid points in a given storm object

    :param storm_object_table: N-row pandas DataFrame with columns documented in
        `storm_tracking_io.write_processed_file`.
    :param num_grid_rows: Number of rows (unique grid-point latitudes).
    :param num_grid_columns: Number of columns (unique grid-point longitudes).
    :return: grid_points_in_storms_table: P-row pandas DataFrame with the
        following columns.
    grid_points_in_storms_table.flattened_index: Flattened index (integer) of
        grid point.
    grid_points_in_storms_table.storm_id: String ID for storm cell.
    """

    error_checking.assert_is_integer(num_grid_rows)
    error_checking.assert_is_greater(num_grid_rows, 0)
    error_checking.assert_is_integer(num_grid_columns)
    error_checking.assert_is_greater(num_grid_columns, 0)

    grid_point_row_indices = numpy.array([])
    grid_point_column_indices = numpy.array([])
    grid_point_storm_ids = []

    num_storms = len(storm_object_table.index)
    for i in range(num_storms):
        grid_point_row_indices = numpy.concatenate((
            grid_point_row_indices,
            storm_object_table[GRID_POINT_ROW_COLUMN].values[i]))
        grid_point_column_indices = numpy.concatenate((
            grid_point_column_indices,
            storm_object_table[GRID_POINT_COLUMN_COLUMN].values[i]))

        this_num_grid_points = len(
            storm_object_table[GRID_POINT_ROW_COLUMN].values[i])
        this_storm_id_list = ([storm_object_table[STORM_ID_COLUMN].values[i]] *
                              this_num_grid_points)
        grid_point_storm_ids += this_storm_id_list

    grid_point_flattened_indices = numpy.ravel_multi_index(
        (grid_point_row_indices.astype(int),
         grid_point_column_indices.astype(int)),
        (num_grid_rows, num_grid_columns))

    grid_points_in_storms_dict = {
        FLATTENED_INDEX_COLUMN: grid_point_flattened_indices,
        STORM_ID_COLUMN: grid_point_storm_ids
    }
    return pandas.DataFrame.from_dict(grid_points_in_storms_dict)


def check_data_source(data_source):
    """Ensures that data source is recognized.

    :param data_source: Data source (string).
    :raises: ValueError: if `data_source not in DATA_SOURCE_IDS`.
    """

    error_checking.assert_is_string(data_source)
    if data_source not in DATA_SOURCE_IDS:
        error_string = (
            '\n' + str(DATA_SOURCE_IDS) +
            '\nValid data sources (listed above) do not include "' +
            data_source + '".')
        raise ValueError(error_string)


def remove_nan_rows_from_dataframe(input_table):
    """Removes NaN rows from pandas DataFrame.

    "NaN row" = a row with one or more NaN values

    :param input_table: pandas DataFrame.
    :return: output_table: Same as input_table, but without NaN rows.
    """

    # TODO(thunderhoser): This method should be in utils.py or something.

    return input_table.loc[input_table.notnull().all(axis=1)]


def distance_buffer_to_column_name(min_distance_metres, max_distance_metres):
    """Generates column name for distance buffer around storm object.

    :param min_distance_metres: Minimum distance around storm object.  If there
        is no minimum distance (i.e., if the storm object is part of the
        buffer), this should be NaN.
    :param max_distance_metres: Maximum distance around storm object.
    :return: column_name: Column name for distance buffer.
    """

    error_checking.assert_is_geq(max_distance_metres, 0.)
    if numpy.isnan(min_distance_metres):
        return '{0:s}_{1:d}m'.format(
            BUFFER_POLYGON_COLUMN_PREFIX, int(max_distance_metres))

    error_checking.assert_is_geq(min_distance_metres, 0.)
    error_checking.assert_is_greater(
        max_distance_metres, min_distance_metres)

    return '{0:s}_{1:d}m_{2:d}m'.format(
        BUFFER_POLYGON_COLUMN_PREFIX, int(min_distance_metres),
        int(max_distance_metres))


def column_name_to_distance_buffer(column_name):
    """Parses distance buffer (around storm object) from column name.

    If distance buffer cannot be found in column name, this method will return
    None for all output vars.

    :param column_name: Column name for distance buffer.
    :return: min_distance_metres: Minimum distance around storm object.  If
        there is no minimum distance (i.e., if the storm object is part of the
        buffer), this is NaN.
    :return: max_distance_metres: Maximum distance around storm object.
    """

    if not column_name.startswith(BUFFER_POLYGON_COLUMN_PREFIX):
        return None, None

    column_name = column_name.replace(BUFFER_POLYGON_COLUMN_PREFIX + '_', '')
    column_name_parts = column_name.split('_')
    if len(column_name_parts) == 1:
        min_buffer_dist_metres = numpy.nan
    elif len(column_name_parts) == 2:
        min_buffer_dist_metres = -1
    else:
        return None, None

    max_distance_part = column_name_parts[-1]
    if not max_distance_part.endswith('m'):
        return None, None
    max_distance_part = max_distance_part.replace('m', '')
    try:
        max_buffer_dist_metres = float(int(max_distance_part))
    except ValueError:
        return None, None

    if numpy.isnan(min_buffer_dist_metres):
        return min_buffer_dist_metres, max_buffer_dist_metres

    min_distance_part = column_name_parts[-2]
    if not min_distance_part.endswith('m'):
        return None, None
    min_distance_part = min_distance_part.replace('m', '')
    try:
        min_buffer_dist_metres = float(int(min_distance_part))
    except ValueError:
        return None, None

    return min_buffer_dist_metres, max_buffer_dist_metres


def get_distance_buffer_columns(storm_object_table):
    """Finds columns with distance buffers around storm objects.

    :param storm_object_table: pandas DataFrame.
    :return: distance_buffer_column_names: 1-D list with names of columns
        corresponding to distance buffers.  Each column should contain
        `shapely.geometry.Polygon` object for one pair of distances (i.e., one
        min distance and one max distance).
    """

    column_names = list(storm_object_table)
    distance_buffer_column_names = None

    for this_column_name in column_names:
        _, this_max_distance_metres = column_name_to_distance_buffer(
            this_column_name)
        if this_max_distance_metres is None:
            continue

        if distance_buffer_column_names is None:
            distance_buffer_column_names = [this_column_name]
        else:
            distance_buffer_column_names.append(this_column_name)

    return distance_buffer_column_names


def find_storm_objects(
        all_storm_ids, all_times_unix_sec, storm_ids_to_keep,
        times_to_keep_unix_sec):
    """Finds storm objects.

    N = total number of storm objects
    n = number of storm objects to keep

    :param all_storm_ids: length-N list of storm IDs (strings).
    :param all_times_unix_sec: length-N numpy array of valid times.
    :param storm_ids_to_keep: length-n list of storm IDs (strings).
    :param times_to_keep_unix_sec: length-n numpy array of valid times.
    :return: relevant_indices: length-n numpy array of indices.
        [all_storm_ids[k] for k in relevant_indices] = storm_ids_to_keep
        all_times_unix_sec[relevant_indices] = times_to_keep_unix_sec
    :raises: ValueError: if `all_storm_ids` and `all_times_unix_sec` contain any
        duplicate pairs.
    :raises: ValueError: if `storm_ids_to_keep` and `times_to_keep_unix_sec`
        contain any duplicate pairs.
    :raises: ValueError: if any desired storm object is not found.
    """

    error_checking.assert_is_numpy_array(
        numpy.array(all_storm_ids), num_dimensions=1)
    num_storm_objects_total = len(all_storm_ids)
    error_checking.assert_is_numpy_array(
        all_times_unix_sec,
        exact_dimensions=numpy.array([num_storm_objects_total]))

    error_checking.assert_is_numpy_array(
        numpy.array(storm_ids_to_keep), num_dimensions=1)
    num_storm_objects_to_keep = len(storm_ids_to_keep)
    error_checking.assert_is_numpy_array(
        times_to_keep_unix_sec,
        exact_dimensions=numpy.array([num_storm_objects_to_keep]))

    all_object_ids = [
        '{0:s}_{1:d}'.format(all_storm_ids[i], all_times_unix_sec[i])
        for i in range(num_storm_objects_total)]

    object_ids_to_keep = [
        '{0:s}_{1:d}'.format(storm_ids_to_keep[i],
                             times_to_keep_unix_sec[i])
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


def merge_storms_at_two_scales(storm_object_table_small_scale=None,
                               storm_object_table_large_scale=None,
                               num_grid_rows=None, num_grid_columns=None):
    """Merges storm objects at two tracking scales, using the probSevere method.

    For each large-scale object L, if there is only one small-scale object S
    inside L, replace S with L.  In other words, if there is only one small-
    scale object inside a large-scale object, the small-scale object is grown to
    the exact size and dimensions of the large-scale object.

    N_s = number of small-scale storm objects
    N_l = number of large-scale storm objects

    :param storm_object_table_small_scale: pandas DataFrame with N_s rows, in
        format specified by `storm_tracking_io.write_processed_file`.
    :param storm_object_table_large_scale: pandas DataFrame with N_l rows, in
        format specified by `storm_tracking_io.write_processed_file`.
    :param num_grid_rows: Number of rows (unique grid-point latitudes) in grid
        used to detect storm objects.
    :param num_grid_columns: Number of columns (unique grid-point longitudes) in
        grid used to detect storm objects.
    :return: storm_object_table_merged: Same as input, except that some
        small-scale objects may have been replaced with a larger-scale object.
    """

    grid_point_table_small_scale = _get_grid_points_in_storms(
        storm_object_table_small_scale, num_grid_rows=num_grid_rows,
        num_grid_columns=num_grid_columns)

    storm_ids_large_scale = storm_object_table_large_scale[
        STORM_ID_COLUMN].values
    num_storms_large_scale = len(storm_ids_large_scale)

    storm_object_table_merged = copy.deepcopy(storm_object_table_small_scale)
    storm_ids_small_scale_grown = []

    for i in range(num_storms_large_scale):
        these_row_indices = storm_object_table_large_scale[
            GRID_POINT_ROW_COLUMN].values[i]
        these_column_indices = storm_object_table_large_scale[
            GRID_POINT_COLUMN_COLUMN].values[i]
        these_flattened_indices = numpy.ravel_multi_index(
            (these_row_indices, these_column_indices),
            (num_grid_rows, num_grid_columns))

        these_point_in_storm_flags = [
            j in these_flattened_indices for j in grid_point_table_small_scale[
                FLATTENED_INDEX_COLUMN].values]
        if not numpy.any(these_point_in_storm_flags):
            continue

        these_point_in_storm_indices = numpy.where(
            these_point_in_storm_flags)[0]
        these_storm_ids_small_scale = numpy.unique(
            grid_point_table_small_scale[STORM_ID_COLUMN].values[
                these_point_in_storm_indices])
        if len(these_storm_ids_small_scale) != 1:
            continue

        this_storm_id_small_scale = these_storm_ids_small_scale[0]
        if this_storm_id_small_scale in storm_ids_small_scale_grown:
            continue  # Prevents small storm from being grown more than once.

        storm_ids_small_scale_grown.append(this_storm_id_small_scale)
        these_storm_flags_small_scale = [
            s == this_storm_id_small_scale for s in
            storm_object_table_small_scale[STORM_ID_COLUMN].values]
        this_storm_index_small_scale = numpy.where(
            these_storm_flags_small_scale)[0][0]

        for this_column in COLUMNS_TO_CHANGE_WHEN_MERGING_SCALES:
            storm_object_table_merged[this_column].values[
                this_storm_index_small_scale] = storm_object_table_large_scale[
                    this_column].values[i]

    return storm_object_table_merged


def make_buffers_around_storm_objects(
        storm_object_table, min_distances_metres, max_distances_metres):
    """Creates one or more distance buffers around each storm object.

    N = number of storm objects
    B = number of buffers around each storm object
    V = number of vertices in a given buffer

    :param storm_object_table: N-row pandas DataFrame with the following
        columns.
    storm_object_table.storm_id: String ID for storm cell.
    storm_object_table.polygon_object_latlng: Instance of
        `shapely.geometry.Polygon`, containing vertices of storm object in
        lat-long coordinates.

    :param min_distances_metres: length-B numpy array of minimum buffer
        distances.  If min_distances_metres[i] is NaN, the storm object is
        included in the [i]th buffer, so the [i]th buffer is inclusive.  If
        min_distances_metres[i] is a real number, the storm object is *not*
        included in the [i]th buffer, so the [i]th buffer is exclusive.
    :param max_distances_metres: length-B numpy array of maximum buffer
        distances.  Must be all real numbers (no NaN).
    :return: storm_object_table: Same as input, but with B additional columns.
        Each additional column (listed below) contains a
        `shapely.geometry.Polygon` instance for each storm object.  Each
        `shapely.geometry.Polygon` instance contains the lat-long vertices of
        one distance buffer around one storm object.
    storm_object_table.polygon_object_latlng_buffer_<D>m: For an inclusive
        buffer of D metres around the storm.
    storm_object_table.polygon_object_latlng_buffer_<d>_<D>m: For an exclusive
        buffer of d...D metres around the storm.
    """

    error_checking.assert_is_geq_numpy_array(
        min_distances_metres, 0., allow_nan=True)
    error_checking.assert_is_numpy_array(
        min_distances_metres, num_dimensions=1)

    num_buffers = len(min_distances_metres)
    error_checking.assert_is_geq_numpy_array(
        max_distances_metres, 0., allow_nan=False)
    error_checking.assert_is_numpy_array(
        max_distances_metres, exact_dimensions=numpy.array([num_buffers]))

    for j in range(num_buffers):
        if numpy.isnan(min_distances_metres[j]):
            continue
        error_checking.assert_is_greater(
            max_distances_metres[j], min_distances_metres[j],
            allow_nan=False)

    num_storm_objects = len(storm_object_table.index)
    centroid_latitudes_deg = numpy.full(num_storm_objects, numpy.nan)
    centroid_longitudes_deg = numpy.full(num_storm_objects, numpy.nan)

    for i in range(num_storm_objects):
        this_centroid_object = storm_object_table[
            POLYGON_OBJECT_LATLNG_COLUMN].values[0].centroid
        centroid_latitudes_deg[i] = this_centroid_object.y
        centroid_longitudes_deg[i] = this_centroid_object.x

    (global_centroid_lat_deg, global_centroid_lng_deg
    ) = geodetic_utils.get_latlng_centroid(
        latitudes_deg=centroid_latitudes_deg,
        longitudes_deg=centroid_longitudes_deg)
    projection_object = projections.init_azimuthal_equidistant_projection(
        global_centroid_lat_deg, global_centroid_lng_deg)

    object_array = numpy.full(num_storm_objects, numpy.nan, dtype=object)
    argument_dict = {}
    buffer_column_names = [''] * num_buffers

    for j in range(num_buffers):
        buffer_column_names[j] = distance_buffer_to_column_name(
            min_distances_metres[j], max_distances_metres[j])
        argument_dict.update({buffer_column_names[j]: object_array})
    storm_object_table = storm_object_table.assign(**argument_dict)

    for i in range(num_storm_objects):
        orig_vertex_dict_latlng = polygons.polygon_object_to_vertex_arrays(
            storm_object_table[POLYGON_OBJECT_LATLNG_COLUMN].values[i])

        (orig_vertex_x_metres,
         orig_vertex_y_metres) = projections.project_latlng_to_xy(
             orig_vertex_dict_latlng[polygons.EXTERIOR_Y_COLUMN],
             orig_vertex_dict_latlng[polygons.EXTERIOR_X_COLUMN],
             projection_object=projection_object)

        for j in range(num_buffers):
            buffer_polygon_object_xy = polygons.buffer_simple_polygon(
                orig_vertex_x_metres, orig_vertex_y_metres,
                min_buffer_dist_metres=min_distances_metres[j],
                max_buffer_dist_metres=max_distances_metres[j])

            buffer_vertex_dict = polygons.polygon_object_to_vertex_arrays(
                buffer_polygon_object_xy)

            (buffer_vertex_dict[polygons.EXTERIOR_Y_COLUMN],
             buffer_vertex_dict[polygons.EXTERIOR_X_COLUMN]) = (
                 projections.project_xy_to_latlng(
                     buffer_vertex_dict[polygons.EXTERIOR_X_COLUMN],
                     buffer_vertex_dict[polygons.EXTERIOR_Y_COLUMN],
                     projection_object=projection_object))

            this_num_holes = len(buffer_vertex_dict[polygons.HOLE_X_COLUMN])
            for k in range(this_num_holes):
                (buffer_vertex_dict[polygons.HOLE_Y_COLUMN][k],
                 buffer_vertex_dict[polygons.HOLE_X_COLUMN][k]) = (
                     projections.project_xy_to_latlng(
                         buffer_vertex_dict[polygons.HOLE_X_COLUMN][k],
                         buffer_vertex_dict[polygons.HOLE_Y_COLUMN][k],
                         projection_object=projection_object))

            buffer_polygon_object_latlng = (
                polygons.vertex_arrays_to_polygon_object(
                    buffer_vertex_dict[polygons.EXTERIOR_X_COLUMN],
                    buffer_vertex_dict[polygons.EXTERIOR_Y_COLUMN],
                    hole_x_coords_list=
                    buffer_vertex_dict[polygons.HOLE_X_COLUMN],
                    hole_y_coords_list=
                    buffer_vertex_dict[polygons.HOLE_Y_COLUMN]))

            storm_object_table[buffer_column_names[j]].values[
                i] = buffer_polygon_object_latlng

    return storm_object_table


def get_original_probsevere_ids(
        best_track_storm_object_table, probsevere_storm_object_table):
    """For each best-track storm object, returns the original probSevere ID.

    N = number of best-track storm objects

    Each input is a pandas DataFrame with columns documented in
    `storm_tracking_io.write_processed_file`.

    :param best_track_storm_object_table: N-row pandas DataFrame with storm
        objects *after* fixing duplicate probSevere IDs and running best-track.
    :param probsevere_storm_object_table: pandas DataFrame with storm
        objects *before* fixing duplicate probSevere IDs and running best-track.
    :return: orig_probsevere_ids: length-N list of original probSevere IDs
        (strings).
    :raises: ValueError: if any best-track object cannot be found in the
        original probSevere table.
    """

    num_best_track_objects = len(best_track_storm_object_table.index)
    orig_probsevere_ids = [None] * num_best_track_objects

    for i in range(num_best_track_objects):
        these_time_indices = numpy.where(
            probsevere_storm_object_table[TIME_COLUMN].values ==
            best_track_storm_object_table[TIME_COLUMN].values[i])[0]

        if not len(these_time_indices):
            this_time_string = time_conversion.unix_sec_to_string(
                best_track_storm_object_table[TIME_COLUMN].values[i],
                TIME_FORMAT_FOR_LOG_MESSAGES)

            error_string = (
                'Cannot find any probSevere objects at {0:s}, even though there'
                ' are best-track objects at this time.'
            ).format(this_time_string)
            raise ValueError(error_string)

        these_latitude_diffs_deg = numpy.absolute(
            probsevere_storm_object_table[CENTROID_LAT_COLUMN].values[
                these_time_indices] -
            best_track_storm_object_table[CENTROID_LAT_COLUMN].values[i])
        these_longitude_diffs_deg = numpy.absolute(
            probsevere_storm_object_table[CENTROID_LNG_COLUMN].values[
                these_time_indices] -
            best_track_storm_object_table[CENTROID_LNG_COLUMN].values[i])

        this_min_latlng_diff_deg = numpy.min(
            these_latitude_diffs_deg + these_longitude_diffs_deg)
        this_nearest_index = numpy.argmin(
            these_latitude_diffs_deg + these_longitude_diffs_deg)
        this_nearest_index = these_time_indices[this_nearest_index]

        if this_min_latlng_diff_deg > TOLERANCE:
            this_time_string = time_conversion.unix_sec_to_string(
                best_track_storm_object_table[TIME_COLUMN].values[i],
                TIME_FORMAT_FOR_LOG_MESSAGES)

            error_string = (
                'Cannot find original probSevere ID for best-track object '
                '"{0:s}" at {1:s}, {2:.4f} deg N, and {3:.4f} deg E.  Nearest '
                'probSevere object at {1:s} is at {4:.4f} deg N and {5:.4f} deg'
                ' E.'
            ).format(best_track_storm_object_table[STORM_ID_COLUMN].values[i],
                     this_time_string,
                     best_track_storm_object_table[CENTROID_LAT_COLUMN].values[
                         i],
                     best_track_storm_object_table[CENTROID_LNG_COLUMN].values[
                         i],
                     probsevere_storm_object_table[CENTROID_LAT_COLUMN].values[
                         this_nearest_index],
                     probsevere_storm_object_table[CENTROID_LNG_COLUMN].values[
                         this_nearest_index])
            raise ValueError(error_string)

        orig_probsevere_ids[i] = probsevere_storm_object_table[
            STORM_ID_COLUMN].values[this_nearest_index]

    return orig_probsevere_ids
