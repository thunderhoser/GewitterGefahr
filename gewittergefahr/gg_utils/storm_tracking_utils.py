"""Processing methods for storm-tracking data (both polygons and tracks)."""

import numpy
import pandas
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import geodetic_utils
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import error_checking

SEGMOTION_NAME = 'segmotion'
PROBSEVERE_NAME = 'probSevere'
DATA_SOURCE_NAMES = [SEGMOTION_NAME, PROBSEVERE_NAME]

FULL_ID_COLUMN = 'full_id_string'
PRIMARY_ID_COLUMN = 'primary_id_string'
SECONDARY_ID_COLUMN = 'secondary_id_string'
FIRST_PREV_SECONDARY_ID_COLUMN = 'first_prev_secondary_id_string'
SECOND_PREV_SECONDARY_ID_COLUMN = 'second_prev_secondary_id_string'
FIRST_NEXT_SECONDARY_ID_COLUMN = 'first_next_secondary_id_string'
SECOND_NEXT_SECONDARY_ID_COLUMN = 'second_next_secondary_id_string'

VALID_TIME_COLUMN = 'valid_time_unix_sec'
SPC_DATE_COLUMN = 'spc_date_string'
TRACKING_START_TIME_COLUMN = 'tracking_start_time_unix_sec'
TRACKING_END_TIME_COLUMN = 'tracking_end_time_unix_sec'
CELL_START_TIME_COLUMN = 'cell_start_time_unix_sec'
CELL_END_TIME_COLUMN = 'cell_end_time_unix_sec'
AGE_COLUMN = 'age_seconds'

CENTROID_LATITUDE_COLUMN = 'centroid_latitude_deg'
CENTROID_LONGITUDE_COLUMN = 'centroid_longitude_deg'
EAST_VELOCITY_COLUMN = 'east_velocity_m_s01'
NORTH_VELOCITY_COLUMN = 'north_velocity_m_s01'

LATITUDES_IN_STORM_COLUMN = 'grid_point_latitudes_deg'
LONGITUDES_IN_STORM_COLUMN = 'grid_point_longitudes_deg'
ROWS_IN_STORM_COLUMN = 'grid_point_rows'
COLUMNS_IN_STORM_COLUMN = 'grid_point_columns'
LATLNG_POLYGON_COLUMN = 'polygon_object_latlng_deg'
ROWCOL_POLYGON_COLUMN = 'polygon_object_rowcol'

BUFFER_COLUMN_PREFIX = 'polygon_object_latlng_deg_buffer'
LINEAR_INDEX_COLUMN = 'linear_index'

CENTROID_X_COLUMN = 'centroid_x_metres'
CENTROID_Y_COLUMN = 'centroid_y_metres'

TRACK_TIMES_COLUMN = 'valid_times_unix_sec'
OBJECT_INDICES_COLUMN = 'object_indices'
TRACK_LATITUDES_COLUMN = 'centroid_latitudes_deg'
TRACK_LONGITUDES_COLUMN = 'centroid_longitudes_deg'
TRACK_X_COORDS_COLUMN = 'centroid_x_coords_metres'
TRACK_Y_COORDS_COLUMN = 'centroid_y_coords_metres'


def check_data_source(source_name):
    """Error-checks data source.

    :param source_name: Data source.
    :raises: ValueError: if `data_source not in DATA_SOURCE_NAMES`.
    """

    error_checking.assert_is_string(source_name)

    if source_name not in DATA_SOURCE_NAMES:
        error_string = (
            '\n{0:s}\nValid data sources (listed above) do not include "{1:s}".'
        ).format(str(DATA_SOURCE_NAMES), source_name)

        raise ValueError(error_string)


def buffer_to_column_name(min_distance_metres, max_distance_metres):
    """Generates column name for distance buffer around storm object.

    :param min_distance_metres: Minimum distance around storm object.  If the
        storm object is included in the buffer, there is no minimum distance and
        this value should be NaN.
    :param max_distance_metres: Max distance around storm object.
    :return: column_name: Column name.
    """

    max_distance_metres = int(numpy.round(max_distance_metres))
    error_checking.assert_is_geq(max_distance_metres, 0)

    if numpy.isnan(min_distance_metres):
        return '{0:s}_{1:d}m'.format(BUFFER_COLUMN_PREFIX, max_distance_metres)

    min_distance_metres = int(numpy.round(min_distance_metres))
    error_checking.assert_is_geq(min_distance_metres, 0)
    error_checking.assert_is_greater(max_distance_metres, min_distance_metres)

    return '{0:s}_{1:d}m_{2:d}m'.format(
        BUFFER_COLUMN_PREFIX, min_distance_metres, max_distance_metres)


def column_name_to_buffer(column_name):
    """Parses distance buffer (min and max distances) from column name.

    If distance buffer cannot be parsed from column name, this method will just
    return None for all output variables.

    :param column_name: Column name.
    :return: min_distance_metres: See doc for `buffer_to_column_name`.
    :return: max_distance_metres: Same.
    """

    if not column_name.startswith(BUFFER_COLUMN_PREFIX):
        return None, None

    buffer_strings = column_name.replace(
        BUFFER_COLUMN_PREFIX + '_', ''
    ).split('_')

    if len(buffer_strings) == 1:
        min_distance_metres = numpy.nan
    elif len(buffer_strings) == 2:
        min_distance_metres = -1
    else:
        return None, None

    max_distance_string = buffer_strings[-1]
    if not max_distance_string.endswith('m'):
        return None, None

    max_distance_string = max_distance_string.replace('m', '')
    try:
        max_distance_metres = float(int(max_distance_string))
    except ValueError:
        return None, None

    if numpy.isnan(min_distance_metres):
        return min_distance_metres, max_distance_metres

    min_distance_part = buffer_strings[-2]
    if not min_distance_part.endswith('m'):
        return None, None

    min_distance_part = min_distance_part.replace('m', '')
    try:
        min_distance_metres = float(int(min_distance_part))
    except ValueError:
        return None, None

    return min_distance_metres, max_distance_metres


def get_buffer_columns(storm_object_table):
    """Finds column names corresponding to distance buffers.

    :param storm_object_table: pandas DataFrame with storm objects.  Each row is
        one storm object.  Columns are listed in `write_file`.
    :return: buffer_column_names: 1-D list of column names corresponding to
        distance buffers.  This may be None (empty list).
    """

    all_column_names = list(storm_object_table)
    buffer_column_names = None

    for this_column_name in all_column_names:
        _, this_max_distance_metres = column_name_to_buffer(this_column_name)
        if this_max_distance_metres is None:
            continue

        if buffer_column_names is None:
            buffer_column_names = [this_column_name]
        else:
            buffer_column_names.append(this_column_name)

    return buffer_column_names


def find_storm_objects(
        all_id_strings, all_times_unix_sec, id_strings_to_keep,
        times_to_keep_unix_sec, allow_missing=False):
    """Finds storm objects.

    N = total number of storm objects
    K = number of storm objects to keep

    :param all_id_strings: length-N list with all storm IDs.
    :param all_times_unix_sec: length-N list with all valid times.
    :param id_strings_to_keep: length-K list of IDs to keep.
    :param times_to_keep_unix_sec: length-K list of valid times to keep.
    :param allow_missing: Boolean flag.  If True, will allow missing storm
        objects (i.e., some objects specified by `id_strings_to_keep` and
        `times_to_keep_unix_sec` may be missing from `all_id_strings` and
        `all_times_unix_sec`).  If False, this method will error out if any
        missing storm objects.
    :return: relevant_indices: length-K numpy array of indices, such that:
        [find_storm_objects[k] for k in relevant_indices] = id_strings_to_keep
        all_times_unix_sec[relevant_indices] = times_to_keep_unix_sec
    :raises: ValueError: if `all_id_strings` and `all_times_unix_sec` contain
        any duplicate pairs.
    :raises: ValueError: if a desired storm object is not found and
        `allow_missing = False`.
    """

    error_checking.assert_is_boolean(allow_missing)
    error_checking.assert_is_numpy_array(
        numpy.array(all_id_strings), num_dimensions=1)

    num_objects_total = len(all_id_strings)
    these_expected_dim = numpy.array([num_objects_total], dtype=int)
    error_checking.assert_is_numpy_array(
        all_times_unix_sec, exact_dimensions=these_expected_dim)

    error_checking.assert_is_numpy_array(
        numpy.array(id_strings_to_keep), num_dimensions=1)

    num_objects_to_keep = len(id_strings_to_keep)
    these_expected_dim = numpy.array([num_objects_to_keep], dtype=int)
    error_checking.assert_is_numpy_array(
        times_to_keep_unix_sec, exact_dimensions=these_expected_dim)

    all_object_id_strings = [
        '{0:s}_{1:d}'.format(all_id_strings[i], all_times_unix_sec[i])
        for i in range(num_objects_total)
    ]

    object_id_strings_to_keep = [
        '{0:s}_{1:d}'.format(id_strings_to_keep[i], times_to_keep_unix_sec[i])
        for i in range(num_objects_to_keep)
    ]

    this_num_unique = len(set(all_object_id_strings))
    if this_num_unique != len(all_object_id_strings):
        error_string = (
            'Only {0:d} of {1:d} original storm objects are unique.'
        ).format(this_num_unique, len(all_object_id_strings))

        raise ValueError(error_string)

    all_object_id_strings = numpy.array(all_object_id_strings, dtype='object')
    object_id_strings_to_keep = numpy.array(
        object_id_strings_to_keep, dtype='object')

    sort_indices = numpy.argsort(all_object_id_strings)
    relevant_indices = numpy.searchsorted(
        all_object_id_strings[sort_indices], object_id_strings_to_keep,
        side='left'
    ).astype(int)

    relevant_indices[relevant_indices < 0] = 0
    relevant_indices[
        relevant_indices >= len(all_object_id_strings)
        ] = len(all_object_id_strings) - 1
    relevant_indices = sort_indices[relevant_indices]

    if allow_missing:
        bad_indices = numpy.where(
            all_object_id_strings[relevant_indices] != object_id_strings_to_keep
        )[0]
        relevant_indices[bad_indices] = -1
        return relevant_indices

    if not numpy.array_equal(all_object_id_strings[relevant_indices],
                             object_id_strings_to_keep):
        missing_object_flags = (
            all_object_id_strings[relevant_indices] != object_id_strings_to_keep
        )

        error_string = (
            '{0:d} of {1:d} desired storm objects are missing.  Their ID-time '
            'pairs are listed below.\n{2:s}'
        ).format(
            numpy.sum(missing_object_flags), num_objects_to_keep,
            str(object_id_strings_to_keep[missing_object_flags])
        )

        raise ValueError(error_string)

    return relevant_indices


def create_distance_buffers(storm_object_table, min_distances_metres,
                            max_distances_metres):
    """Creates one or more distance buffers around each storm object.

    K = number of buffers

    :param storm_object_table: pandas DataFrame with the following columns.
        Each row is one storm object.
    storm_object_table.centroid_latitude_deg: Latitude (deg N) of storm-object
        centroid.
    storm_object_table.centroid_longitude_deg: Longitude (deg E) of storm-object
        centroid.
    storm_object_table.polygon_object_latlng_deg: Instance of
        `shapely.geometry.Polygon`, with x-coords in longitude (deg E) and
        y-coords in latitude (deg N).

    :param min_distances_metres: length-K numpy array of minimum distances.  If
        the storm object is inside the [k]th buffer -- i.e., the [k]th buffer
        has no minimum distance -- then min_distances_metres[k] should be NaN.
    :param max_distances_metres: length-K numpy array of max distances.
    :return: storm_object_table: Same as input but with K additional columns
        (one per distance buffer).  Column names are generated by
        `buffer_to_column_name`, and each value in these columns is a
        `shapely.geometry.Polygon` object, with x-coords in longitude (deg E) and
        y-coords in latitude (deg N).
    """

    num_buffers = len(min_distances_metres)
    these_expected_dim = numpy.array([num_buffers], dtype=int)
    error_checking.assert_is_numpy_array(
        max_distances_metres, exact_dimensions=these_expected_dim)

    global_centroid_lat_deg, global_centroid_lng_deg = (
        geodetic_utils.get_latlng_centroid(
            latitudes_deg=storm_object_table[CENTROID_LATITUDE_COLUMN].values,
            longitudes_deg=storm_object_table[CENTROID_LONGITUDE_COLUMN].values)
    )

    projection_object = projections.init_azimuthal_equidistant_projection(
        central_latitude_deg=global_centroid_lat_deg,
        central_longitude_deg=global_centroid_lng_deg)

    num_storm_objects = len(storm_object_table.index)
    object_array = numpy.full(num_storm_objects, numpy.nan, dtype=object)
    buffer_column_names = [''] * num_buffers

    for j in range(num_buffers):
        buffer_column_names[j] = buffer_to_column_name(
            min_distance_metres=min_distances_metres[j],
            max_distance_metres=max_distances_metres[j])

        storm_object_table = storm_object_table.assign(
            **{buffer_column_names[j]: object_array}
        )

    for i in range(num_storm_objects):
        this_orig_vertex_dict_latlng_deg = (
            polygons.polygon_object_to_vertex_arrays(
                storm_object_table[LATLNG_POLYGON_COLUMN].values[i]
            )
        )

        these_orig_x_metres, these_orig_y_metres = (
            projections.project_latlng_to_xy(
                latitudes_deg=this_orig_vertex_dict_latlng_deg[
                    polygons.EXTERIOR_Y_COLUMN],
                longitudes_deg=this_orig_vertex_dict_latlng_deg[
                    polygons.EXTERIOR_X_COLUMN],
                projection_object=projection_object)
        )

        for j in range(num_buffers):
            this_buffer_poly_object_xy_metres = polygons.buffer_simple_polygon(
                vertex_x_metres=these_orig_x_metres,
                vertex_y_metres=these_orig_y_metres,
                min_buffer_dist_metres=min_distances_metres[j],
                max_buffer_dist_metres=max_distances_metres[j])

            this_buffer_vertex_dict = polygons.polygon_object_to_vertex_arrays(
                this_buffer_poly_object_xy_metres)

            (this_buffer_vertex_dict[polygons.EXTERIOR_Y_COLUMN],
             this_buffer_vertex_dict[polygons.EXTERIOR_X_COLUMN]
            ) = projections.project_xy_to_latlng(
                x_coords_metres=this_buffer_vertex_dict[
                    polygons.EXTERIOR_X_COLUMN],
                y_coords_metres=this_buffer_vertex_dict[
                    polygons.EXTERIOR_Y_COLUMN],
                projection_object=projection_object)

            this_num_holes = len(
                this_buffer_vertex_dict[polygons.HOLE_X_COLUMN]
            )

            for k in range(this_num_holes):
                (this_buffer_vertex_dict[polygons.HOLE_Y_COLUMN][k],
                 this_buffer_vertex_dict[polygons.HOLE_X_COLUMN][k]
                ) = projections.project_xy_to_latlng(
                    x_coords_metres=this_buffer_vertex_dict[
                        polygons.HOLE_X_COLUMN][k],
                    y_coords_metres=this_buffer_vertex_dict[
                        polygons.HOLE_Y_COLUMN][k],
                    projection_object=projection_object)

            this_buffer_poly_object_latlng_deg = (
                polygons.vertex_arrays_to_polygon_object(
                    exterior_x_coords=this_buffer_vertex_dict[
                        polygons.EXTERIOR_X_COLUMN],
                    exterior_y_coords=this_buffer_vertex_dict[
                        polygons.EXTERIOR_Y_COLUMN],
                    hole_x_coords_list=this_buffer_vertex_dict[
                        polygons.HOLE_X_COLUMN],
                    hole_y_coords_list=this_buffer_vertex_dict[
                        polygons.HOLE_Y_COLUMN]
                )
            )

            storm_object_table[buffer_column_names[j]].values[i] = (
                this_buffer_poly_object_latlng_deg
            )

    return storm_object_table


def storm_objects_to_tracks(storm_object_table):
    """Converts table of storm objects to table of storm tracks.

    T = number of time steps (objects) in a given track

    :param storm_object_table: pandas DataFrame with at least the following
        columns.  Each row is one storm object.
    storm_object_table.primary_id_string: ID for corresponding storm cell.
    storm_object_table.valid_time_unix_sec: Valid time of storm object.
    storm_object_table.centroid_latitude_deg: Latitude (deg N) of storm-object
        centroid.
    storm_object_table.centroid_longitude_deg: Longitude (deg E) of storm-object
        centroid.
    storm_object_table.centroid_x_metres: x-coordinate of storm-object centroid.
    storm_object_table.centroid_y_metres: y-coordinate of storm-object centroid.

    :return: storm_track_table: pandas DataFrame with the following columns.
        Each row is one storm track (cell).
    storm_track_table.primary_id_string: ID for storm cell.
    storm_track_table.valid_times_unix_sec: length-T numpy array of valid times.
    storm_track_table.object_indices: length-T numpy array with indices of storm
        objects in track.  These are indices into the rows of
            `storm_object_table`.
    storm_track_table.centroid_latitudes_deg: length-T numpy array of centroid
        latitudes (deg N).
    storm_track_table.centroid_longitudes_deg: length-T numpy array of centroid
        longitudes (deg E).
    storm_track_table.centroid_x_coords_metres: length-T numpy array of centroid
        x-coords.
    storm_track_table.centroid_y_coords_metres: length-T numpy array of centroid
        y-coords.
    """

    object_id_strings = numpy.array(
        storm_object_table[PRIMARY_ID_COLUMN].values)
    track_id_strings, object_to_track_indices = numpy.unique(
        object_id_strings, return_inverse=True)

    storm_track_dict = {PRIMARY_ID_COLUMN: track_id_strings}
    storm_track_table = pandas.DataFrame.from_dict(storm_track_dict)

    nested_array = storm_track_table[[
        PRIMARY_ID_COLUMN, PRIMARY_ID_COLUMN
    ]].values.tolist()

    storm_track_table = storm_track_table.assign(**{
        TRACK_TIMES_COLUMN: nested_array,
        OBJECT_INDICES_COLUMN: nested_array,
        TRACK_LATITUDES_COLUMN: nested_array,
        TRACK_LONGITUDES_COLUMN: nested_array,
        TRACK_X_COORDS_COLUMN: nested_array,
        TRACK_Y_COORDS_COLUMN: nested_array
    })

    num_storm_tracks = len(storm_track_table.index)

    for i in range(num_storm_tracks):
        these_object_indices = numpy.where(object_to_track_indices == i)[0]
        sort_indices = numpy.argsort(
            storm_object_table[VALID_TIME_COLUMN].values[these_object_indices]
        )
        these_object_indices = these_object_indices[sort_indices]

        storm_track_table[TRACK_TIMES_COLUMN].values[i] = (
            storm_object_table[VALID_TIME_COLUMN].values[these_object_indices]
        )
        storm_track_table[OBJECT_INDICES_COLUMN].values[i] = (
            these_object_indices
        )

        storm_track_table[TRACK_LATITUDES_COLUMN].values[i] = (
            storm_object_table[CENTROID_LATITUDE_COLUMN].values[
                these_object_indices]
        )
        storm_track_table[TRACK_LONGITUDES_COLUMN].values[i] = (
            storm_object_table[CENTROID_LONGITUDE_COLUMN].values[
                these_object_indices]
        )

        storm_track_table[TRACK_X_COORDS_COLUMN].values[i] = (
            storm_object_table[CENTROID_X_COLUMN].values[these_object_indices]
        )
        storm_track_table[TRACK_Y_COORDS_COLUMN].values[i] = (
            storm_object_table[CENTROID_Y_COLUMN].values[these_object_indices]
        )

    return storm_track_table
