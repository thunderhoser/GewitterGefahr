"""Methods to create gridded spatial forecasts from storm-cell-based ones."""

import numpy
import pandas
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import geodetic_utils
from gewittergefahr.gg_utils import error_checking

DEFAULT_PROB_RADIUS_FOR_GRID_METRES = 1e4

LATLNG_POLYGON_COLUMN_PREFIX = tracking_io.BUFFER_POLYGON_COLUMN_PREFIX
XY_POLYGON_COLUMN_PREFIX = 'polygon_object_xy_buffer'
FORECAST_COLUMN_PREFIX = 'forecast_probability_buffer'

LATLNG_POLYGON_COLUMN_TYPE = 'latlng'
XY_POLYGON_COLUMN_TYPE = 'xy'
FORECAST_COLUMN_TYPE = 'forecast'
COLUMN_TYPES = [
    LATLNG_POLYGON_COLUMN_TYPE, XY_POLYGON_COLUMN_TYPE, FORECAST_COLUMN_TYPE]

SPEED_COLUMN = 'speed_m_s01'
GEOGRAPHIC_BEARING_COLUMN = 'geographic_bearing_deg'

# TODO(thunderhoser): Still missing main method that puts forecasts on grid.


def _column_name_to_distance_buffer(column_name):
    """Parses distance buffer from column name.

    If distance buffer cannot be found in column name, returns None for all
    outputs.

    :param column_name: Name of column.  This column may contain x-y polygons,
        lat-long polygons, or forecast probabilities.
    :return: min_buffer_dist_metres: Minimum buffer distance.
    :return: max_buffer_dist_metres: Maximum buffer distance.
    """

    this_column_name = column_name.replace(
        XY_POLYGON_COLUMN_PREFIX, LATLNG_POLYGON_COLUMN_PREFIX).replace(
            FORECAST_COLUMN_PREFIX, LATLNG_POLYGON_COLUMN_PREFIX)

    return tracking_io.column_name_to_distance_buffer(this_column_name)


def _distance_buffer_to_column_name(
        min_buffer_dist_metres, max_buffer_dist_metres, column_type):
    """Generates column name for distance buffer.

    :param min_buffer_dist_metres: Minimum buffer distance.
    :param max_buffer_dist_metres: Max buffer distance.
    :param column_type: Column type (may be any string in `COLUMN_TYPES`).
    :return: column_name: Name of column.
    """

    column_name = tracking_io.distance_buffer_to_column_name(
        min_buffer_dist_metres, max_buffer_dist_metres)

    if column_type == LATLNG_POLYGON_COLUMN_TYPE:
        return column_name
    if column_type == XY_POLYGON_COLUMN_TYPE:
        return column_name.replace(
            LATLNG_POLYGON_COLUMN_PREFIX, XY_POLYGON_COLUMN_PREFIX)
    if column_type == FORECAST_COLUMN_TYPE:
        return column_name.replace(
            LATLNG_POLYGON_COLUMN_PREFIX, FORECAST_COLUMN_PREFIX)

    return None


def _get_distance_buffer_columns(storm_object_table, column_type):
    """Returns all column names corresponding to distance buffers.

    :param storm_object_table: pandas DataFrame.
    :param column_type: Column type (may be any string in `COLUMN_TYPES`).
    :return: dist_buffer_column_names: 1-D list of column names corresponding to
        distance buffers.  If there are no columns with distance buffers,
        returns None.
    """

    column_names = list(storm_object_table)
    dist_buffer_column_names = None

    for this_column_name in column_names:
        if (column_type == XY_POLYGON_COLUMN_TYPE
                and XY_POLYGON_COLUMN_PREFIX not in this_column_name):
            continue

        if (column_type == LATLNG_POLYGON_COLUMN_TYPE
                and LATLNG_POLYGON_COLUMN_PREFIX not in this_column_name):
            continue

        if (column_type == FORECAST_COLUMN_TYPE
                and FORECAST_COLUMN_PREFIX not in this_column_name):
            continue

        _, this_max_distance_metres = _column_name_to_distance_buffer(
            this_column_name)
        if this_max_distance_metres is None:
            continue

        if dist_buffer_column_names is None:
            dist_buffer_column_names = [this_column_name]
        else:
            dist_buffer_column_names.append(this_column_name)

    return dist_buffer_column_names


def _check_distance_buffers(min_distances_metres, max_distances_metres):
    """Ensures that distance buffers are unique and abutting.

    B = number of distance buffers

    :param min_distances_metres: length-B numpy array of minimum buffer
        distances.
    :param max_distances_metres: length-B numpy array of max buffer distances.
    :raises: ValueError: if distance buffers are non-unique or non-abutting.
    """

    error_checking.assert_is_numpy_array(
        min_distances_metres, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(
        min_distances_metres, 0., allow_nan=True)
    num_buffers = len(min_distances_metres)

    error_checking.assert_is_numpy_array(
        max_distances_metres, exact_dimensions=numpy.array([num_buffers]))

    sort_indices = numpy.argsort(max_distances_metres)
    sorted_min_distances_metres = numpy.round(
        min_distances_metres[sort_indices])
    sorted_max_distances_metres = numpy.round(
        max_distances_metres[sort_indices])

    for j in range(num_buffers):
        if numpy.isnan(sorted_min_distances_metres[j]):
            error_checking.assert_is_geq(sorted_max_distances_metres[j], 0.)
        else:
            error_checking.assert_is_greater(
                sorted_max_distances_metres[j], sorted_min_distances_metres[j])

        if (j != 0 and sorted_min_distances_metres[j]
                != sorted_max_distances_metres[j - 1]):
            error_string = (
                'Minimum distance for {0:d}th buffer ({1:d} m) does not equal '
                'max distance for {2:d}th buffer ({3:d} m).  This means the two'
                ' distance buffers are not abutting.').format(
                    j + 1, int(sorted_min_distances_metres[j]), j,
                    int(sorted_max_distances_metres[j - 1]))
            raise ValueError(error_string)


def _init_azimuthal_equidistant_projection(
        min_latitude_deg, max_latitude_deg, min_longitude_deg,
        max_longitude_deg):
    """Initializes azimuthal equidistant projection.

    :param min_latitude_deg: Minimum latitude (deg N) covered by projection.
    :param max_latitude_deg: Max latitude (deg N) covered by projection.
    :param min_longitude_deg: Minimum longitude (deg E) covered by projection.
    :param max_longitude_deg: Max longitude (deg E) covered by projection.
    :return: projection_object: Instance of `pyproj.Proj`, which can be used in
        the future to convert from lat-long to x-y and back.
    """

    return projections.init_azimuthal_equidistant_projection(
        central_latitude_deg=(min_latitude_deg + max_latitude_deg) / 2,
        central_longitude_deg=(min_longitude_deg + max_longitude_deg) / 2)


def _polygons_from_latlng_to_xy(storm_object_table, projection_object):
    """Projects distance buffers around each storm object from lat-long to x-y.

    N = number of storm objects
    B = number of distance buffers around each storm object

    :param storm_object_table: N-row pandas DataFrame.  Each row contains the
        polygons for distance buffers around one storm object.  For the [j]th
        distance buffer, required column is given by the following command:

        _distance_buffer_to_column_name(min_buffer_distances_metres[j],
            max_buffer_distances_metres[j], column_type="latlng")
    :param projection_object: Instance of `pyproj.Proj`.  Will be used to
        project from lat-long to x-y.
    :return: storm_object_table: Same as input but with additional columns.  For
        the [j]th distance buffer, new column is given by the following command:

        _distance_buffer_to_column_name(min_buffer_distances_metres[j],
            max_buffer_distances_metres[j], column_type="xy")
    """

    buffer_column_names_latlng = _get_distance_buffer_columns(
        storm_object_table, column_type='latlng')

    num_buffers = len(buffer_column_names_latlng)
    min_buffer_distances_metres = numpy.full(num_buffers, numpy.nan)
    max_buffer_distances_metres = numpy.full(num_buffers, numpy.nan)
    buffer_column_names_xy = [''] * num_buffers

    for j in range(num_buffers):
        min_buffer_distances_metres[j], max_buffer_distances_metres[j] = (
            _column_name_to_distance_buffer(buffer_column_names_latlng[j]))
        buffer_column_names_xy[j] = _distance_buffer_to_column_name(
            min_buffer_distances_metres[j], max_buffer_distances_metres[j],
            column_type='xy')

    num_storm_objects = len(storm_object_table.index)
    object_array = numpy.full(num_storm_objects, numpy.nan, dtype=object)
    for j in range(num_buffers):
        storm_object_table = storm_object_table.assign(
            **{buffer_column_names_xy[j]: object_array})

    for i in range(num_storm_objects):
        for j in range(num_buffers):
            storm_object_table[buffer_column_names_xy[j]].values[i] = (
                polygons.project_latlng_to_xy(
                    storm_object_table[buffer_column_names_latlng[j]].values[i],
                    projection_object=projection_object,
                    false_easting_metres=0., false_northing_metres=0.))

    return storm_object_table


def _normalize_probs_by_polygon_area(
        storm_object_table, prob_radius_for_grid_metres):
    """Normalizes each forecast probability by area of the attached polygon.

    Specifically, this method multiplies each probability by pi * r^2 / A,
    where:

    A = area of the attached polygon
    r = `prob_radius_for_grid_metres`

    Also:

    N = number of storm objects
    B = number of distance buffers around each storm object

    :param storm_object_table: N-row pandas DataFrame.  Each row contains the
        polygons and forecast probabilities for distance buffers around one
        storm object.  For the [j]th distance buffer, required columns are given
        by the following command:

        _distance_buffer_to_column_name(min_buffer_distances_metres[j],
            max_buffer_distances_metres[j], column_type="xy")

        _distance_buffer_to_column_name(min_buffer_distances_metres[j],
            max_buffer_distances_metres[j], column_type="forecast")

    :param prob_radius_for_grid_metres: Effective radius for gridded
        probabilities.  For example, if the gridded value is "probability within
        10 km of a point," this should be 10 000.
    :return: storm_object_table: Same as input, except that probabilities are
        normalized.
    """

    buffer_column_names_xy = _get_distance_buffer_columns(
        storm_object_table, column_type='xy')

    num_buffers = len(buffer_column_names_xy)
    min_buffer_distances_metres = numpy.full(num_buffers, numpy.nan)
    max_buffer_distances_metres = numpy.full(num_buffers, numpy.nan)
    forecast_column_names = [''] * num_buffers

    for j in range(num_buffers):
        min_buffer_distances_metres[j], max_buffer_distances_metres[j] = (
            _column_name_to_distance_buffer(buffer_column_names_xy[j]))
        forecast_column_names[j] = _distance_buffer_to_column_name(
            min_buffer_distances_metres[j], max_buffer_distances_metres[j],
            column_type='forecast')

    num_storm_objects = len(storm_object_table.index)
    prob_area_for_grid_metres2 = numpy.pi * prob_radius_for_grid_metres ** 2

    for j in range(num_buffers):
        these_areas_metres2 = numpy.array([
            storm_object_table[buffer_column_names_xy[j]].values[i].area
            for i in range(num_storm_objects)])

        these_normalized_probs = prob_area_for_grid_metres2 * (
            storm_object_table[forecast_column_names[j]].values /
            these_areas_metres2)
        these_normalized_probs[these_normalized_probs < 0.] = 0.
        these_normalized_probs[these_normalized_probs > 1.] = 1.

        storm_object_table = storm_object_table.assign(
            **{forecast_column_names[j]: these_normalized_probs})

    return storm_object_table


def _storm_motion_from_uv_to_speed_direction(storm_object_table):
    """For each storm object, converts motion from u-v to speed-direction.

    N = number of storm objects

    :param storm_object_table: N-row pandas DataFrame with the following
        columns.
    storm_object_table.east_velocity_m_s01: Eastward velocity of storm object
        (metres per second).
    storm_object_table.north_velocity_m_s01: Northward velocity of storm object
        (metres per second).
    :return: storm_object_table: N-row pandas DataFrame with the following
        columns.
    storm_object_table.speed_m_s01: Storm speed (magnitude of velocity) in m/s.
    storm_object_table.geographic_bearing_deg: Storm bearing in geographic
        degrees (clockwise from due north).
    """

    storm_speeds_m_s01, geodetic_bearings_deg = (
        geodetic_utils.xy_components_to_displacements_and_bearings(
            storm_object_table[tracking_io.EAST_VELOCITY_COLUMN].values,
            storm_object_table[tracking_io.NORTH_VELOCITY_COLUMN].values))

    argument_dict = {SPEED_COLUMN: storm_speeds_m_s01,
                     GEOGRAPHIC_BEARING_COLUMN: geodetic_bearings_deg}
    storm_object_table = storm_object_table.assign(**argument_dict)

    return storm_object_table.drop(
        [tracking_io.EAST_VELOCITY_COLUMN, tracking_io.NORTH_VELOCITY_COLUMN],
        axis=1, inplace=False)


def _extrapolate_polygons(
        storm_object_table, lead_time_seconds, projection_object):
    """For each storm object, extrapolates distance buffers forward in time.

    N = number of storm objects

    :param storm_object_table: N-row pandas DataFrame.  Each row contains data
        for distance buffers around one storm object.  For the [j]th distance
        buffer, required columns are given by the following command:

        _distance_buffer_to_column_name(min_buffer_distances_metres[j],
            max_buffer_distances_metres[j], column_type="latlng")

        Other required columns are listed below.

    storm_object_table.speed_m_s01: Storm speed (magnitude of velocity) in m/s.
    storm_object_table.geographic_bearing_deg: Storm bearing in geographic
        degrees (clockwise from due north).

    :param lead_time_seconds: Lead time.  Polygons will be extrapolated this far
        into the future.
    :param projection_object: Instance of `pyproj.Proj`.  Will be used to
        project extrapolated polygons from lat-long to x-y.
    :return: extrap_storm_object_table: N-row pandas DataFrame.  Each row
        contains extrapolated polygons for distance buffers around one storm
        object.  For the [j]th distance buffer, columns are given by the
        following command:

        _distance_buffer_to_column_name(min_buffer_distances_metres[j],
            max_buffer_distances_metres[j], column_type="xy")
    """

    buffer_column_names_latlng = _get_distance_buffer_columns(
        storm_object_table, column_type='latlng')

    num_buffers = len(buffer_column_names_latlng)
    num_storm_objects = len(storm_object_table.index)
    object_array = numpy.full(num_storm_objects, numpy.nan, dtype=object)
    extrap_storm_object_table = None

    for j in range(num_buffers):
        if extrap_storm_object_table is None:
            extrap_storm_object_table = pandas.DataFrame.from_dict(
                {buffer_column_names_latlng[j]: object_array})
        else:
            extrap_storm_object_table = extrap_storm_object_table.assign(
                **{buffer_column_names_latlng[j]: object_array})

    for j in range(num_buffers):
        these_first_vertex_lat_deg = [
            storm_object_table[buffer_column_names_latlng[j]].values[
                i].exterior.xy[1][0] for i in range(num_storm_objects)]
        these_first_vertex_lng_deg = [
            storm_object_table[buffer_column_names_latlng[j]].values[
                i].exterior.xy[0][0] for i in range(num_storm_objects)]

        these_first_vertex_lat_deg = numpy.array(these_first_vertex_lat_deg)
        these_first_vertex_lng_deg = numpy.array(these_first_vertex_lng_deg)

        these_extrap_lat_deg, these_extrap_lng_deg = (
            geodetic_utils.start_points_and_distances_and_bearings_to_endpoints(
                start_latitudes_deg=these_first_vertex_lat_deg,
                start_longitudes_deg=these_first_vertex_lng_deg,
                displacements_metres=
                storm_object_table[SPEED_COLUMN].values * lead_time_seconds,
                geodetic_bearings_deg=
                storm_object_table[GEOGRAPHIC_BEARING_COLUMN].values))

        these_lat_diffs_deg = these_extrap_lat_deg - these_first_vertex_lat_deg
        these_lng_diffs_deg = these_extrap_lng_deg - these_first_vertex_lng_deg

        for i in range(num_storm_objects):
            these_new_latitudes_deg = these_lat_diffs_deg[i] + numpy.array(
                storm_object_table[
                    buffer_column_names_latlng[j]].values[i].exterior.xy[1])
            these_new_longitudes_deg = these_lng_diffs_deg[i] + numpy.array(
                storm_object_table[
                    buffer_column_names_latlng[j]].values[i].exterior.xy[0])

            extrap_storm_object_table[
                buffer_column_names_latlng[j]
            ].values[i] = polygons.vertex_arrays_to_polygon_object(
                these_new_longitudes_deg, these_new_latitudes_deg)

    return _polygons_from_latlng_to_xy(
        extrap_storm_object_table, projection_object)


def _find_min_value_greater_or_equal(sorted_input_array, test_value):
    """Finds minimum value in array that is >= test value.

    :param sorted_input_array: Input array.  Must be sorted in ascending order.
    :param test_value: Test value (scalar).
    :return: min_value_geq_test: Minimum of value input_array that is >=
        test_value.
    :return: min_index_geq_test: Array index of min_value_geq_test in
        input_array.  If min_index_geq_test = i, this means that
        min_value_geq_test = sorted_input_array[i].
    """

    min_index_geq_test = numpy.searchsorted(
        sorted_input_array, numpy.array([test_value]), side='left')[0]
    return sorted_input_array[min_index_geq_test], min_index_geq_test


def _find_max_value_less_than_or_equal(sorted_input_array, test_value):
    """Finds maximum value in array that is <= test value.

    :param sorted_input_array: Input array.  Must be sorted in ascending order.
    :param test_value: Test value (scalar).
    :return: max_value_leq_test: Max value of input_array that is <= test_value.
    :return: max_index_leq_test: Array index of max_value_leq_test in
        input_array.  If max_index_leq_test = i, this means that
        max_value_leq_test = sorted_input_array[i].
    """

    max_index_leq_test = numpy.searchsorted(
        sorted_input_array, numpy.array([test_value]), side='right')[0] - 1
    return sorted_input_array[max_index_leq_test], max_index_leq_test


def _find_grid_points_in_polygon(
        polygon_object_xy, grid_points_x_metres, grid_points_y_metres):
    """Finds grid points in polygon.

    M = number of rows (unique grid-point y-coordinates)
    N = number of columns (unique grid-point x-coordinates)
    P = number of grid points in polygon

    :param polygon_object_xy: Instance of `shapely.geometry.Polygon` with
        vertices in x-y coordinates (metres).
    :param grid_points_x_metres: length-N numpy array with x-coordinates of grid
        points.  Must be sorted in ascending order.
    :param grid_points_y_metres: length-M numpy array with y-coordinates of grid
        points.  Must be sorted in ascending order.
    :return: rows_in_polygon: length-P integer numpy array of rows in polygon.
    :return: columns_in_polygon: length-P integer numpy array of columns in
        polygon.
    """

    min_x_in_polygon_metres = numpy.min(numpy.array(
        polygon_object_xy.exterior.xy[0]))
    max_x_in_polygon_metres = numpy.max(numpy.array(
        polygon_object_xy.exterior.xy[0]))
    min_y_in_polygon_metres = numpy.min(numpy.array(
        polygon_object_xy.exterior.xy[1]))
    max_y_in_polygon_metres = numpy.max(numpy.array(
        polygon_object_xy.exterior.xy[1]))

    _, min_row_to_test = _find_min_value_greater_or_equal(
        grid_points_y_metres, min_y_in_polygon_metres)
    _, max_row_to_test = _find_max_value_less_than_or_equal(
        grid_points_y_metres, max_y_in_polygon_metres)
    _, min_column_to_test = _find_min_value_greater_or_equal(
        grid_points_x_metres, min_x_in_polygon_metres)
    _, max_column_to_test = _find_max_value_less_than_or_equal(
        grid_points_x_metres, max_x_in_polygon_metres)

    rows_in_polygon = []
    columns_in_polygon = []

    for this_row in range(min_row_to_test, max_row_to_test + 1):
        for this_column in range(min_column_to_test, max_column_to_test + 1):
            this_flag = polygons.is_point_in_or_on_polygon(
                polygon_object_xy,
                query_x_coordinate=grid_points_x_metres[this_column],
                query_y_coordinate=grid_points_y_metres[this_row])
            if not this_flag:
                continue

            rows_in_polygon.append(this_row)
            columns_in_polygon.append(this_column)

    return numpy.array(rows_in_polygon), numpy.array(columns_in_polygon)
