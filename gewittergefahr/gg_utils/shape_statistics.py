"""Methods for computing shape statistics."""

import copy
import numpy
import skimage.measure
from area import polygon__area as polygon_area
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import shape_utils
from gewittergefahr.gg_utils import smoothing_via_iterative_averaging as sia
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import error_checking

# TODO(thunderhoser): may add moments to list of shape statistics.

RADIANS_TO_DEGREES = 180. / numpy.pi
GRID_SPACING_FOR_BINARY_MATRIX_DEFAULT_METRES = 100.

NUM_VERTICES_IN_SMOOTHING_HALF_WINDOW_DEFAULT = (
    sia.NUM_VERTICES_IN_HALF_WINDOW_DEFAULT)
NUM_SMOOTHING_ITERS_DEFAULT = sia.NUM_ITERATIONS_DEFAULT

AREA_NAME = 'area_metres2'
ECCENTRICITY_NAME = 'eccentricity'
EXTENT_NAME = 'extent'
SOLIDITY_NAME = 'solidity'
ORIENTATION_NAME = 'orientation_deg'
MEAN_ABS_CURVATURE_NAME = 'mean_abs_curvature_metres01'
BENDING_ENERGY_NAME = 'bending_energy_metres03'
COMPACTNESS_NAME = 'compactness'

AREA_NAME_ORIG = 'area'
ECCENTRICITY_NAME_ORIG = 'eccentricity'
EXTENT_NAME_ORIG = 'extent'
SOLIDITY_NAME_ORIG = 'solidity'
ORIENTATION_NAME_ORIG = 'orientation'
MEAN_ABS_CURVATURE_NAME_ORIG = 'mean_abs_curvature_metres01'
BENDING_ENERGY_NAME_ORIG = 'bending_energy_metres03'
COMPACTNESS_NAME_ORIG = 'compactness'

REGION_PROPERTY_NAMES = [
    ECCENTRICITY_NAME, EXTENT_NAME, SOLIDITY_NAME, ORIENTATION_NAME]
REGION_PROP_NAMES_ORIG = [
    ECCENTRICITY_NAME_ORIG, EXTENT_NAME_ORIG, SOLIDITY_NAME_ORIG,
    ORIENTATION_NAME_ORIG]
DEFAULT_REGION_PROP_NAMES = copy.deepcopy(REGION_PROPERTY_NAMES)

CURVATURE_BASED_STAT_NAMES = [
    MEAN_ABS_CURVATURE_NAME, BENDING_ENERGY_NAME, COMPACTNESS_NAME]
CURVATURE_BASED_STAT_NAMES_ORIG = [
    MEAN_ABS_CURVATURE_NAME_ORIG, BENDING_ENERGY_NAME_ORIG,
    COMPACTNESS_NAME_ORIG]
DEFAULT_CURVATURE_BASED_STAT_NAMES = copy.deepcopy(CURVATURE_BASED_STAT_NAMES)

STATISTIC_NAMES = (
    REGION_PROPERTY_NAMES + CURVATURE_BASED_STAT_NAMES + [AREA_NAME])
STATISTIC_NAMES_ORIG = (
    REGION_PROP_NAMES_ORIG + CURVATURE_BASED_STAT_NAMES_ORIG + [AREA_NAME_ORIG])
DEFAULT_STATISTIC_NAMES = copy.deepcopy(STATISTIC_NAMES)


def _check_statistic_names(statistic_names):
    """Ensures that statistic names are valid.

    :param statistic_names: 1-D list of statistic names.
    :raises: ValueError: if any element of `statistic_names` is not in
        `STATISTIC_NAMES`.
    """

    error_checking.assert_is_string_list(statistic_names)
    error_checking.assert_is_numpy_array(
        numpy.array(statistic_names), num_dimensions=1)

    for this_name in statistic_names:
        if this_name in STATISTIC_NAMES:
            continue

        error_string = (
            '\n\n' + str(STATISTIC_NAMES) + '\n\nValid statistic names ' +
            '(listed above) do not include the following: "' + this_name + '"')
        raise ValueError(error_string)


def _stat_name_new_to_orig(statistic_name):
    """Converts name of statistic from new to original format.

    New format = GewitterGefahr
    Original format = `skimage.measure.regionprops`

    :param statistic_name: Statistic name in new format.
    :return: statistic_name_orig: Statistic name in original format.
    """

    found_flags = [s == statistic_name for s in STATISTIC_NAMES]
    return STATISTIC_NAMES_ORIG[numpy.where(found_flags)[0][0]]


def _get_region_property_names(statistic_names):
    """Finds region-property names in list of statistic names.

    :param statistic_names: 1-D list of statistic names.
    :return: region_property_names: 1-D list of region-property names.
    """

    region_property_flags = numpy.array(
        [s in REGION_PROPERTY_NAMES for s in statistic_names])
    region_property_indices = numpy.where(region_property_flags)[0]
    return numpy.array(statistic_names)[region_property_indices].tolist()


def _get_curvature_based_stat_names(statistic_names):
    """Finds names of curvature-based stats in list of statistic names.

    :param statistic_names: 1-D list of statistic names.
    :return: curvature_based_stat_names: 1-D list with names of curvature-based
        stats.
    """

    curvature_based_flags = numpy.array(
        [s in CURVATURE_BASED_STAT_NAMES for s in statistic_names])
    curvature_based_indices = numpy.where(curvature_based_flags)[0]
    return numpy.array(statistic_names)[curvature_based_indices].tolist()


def _project_polygon_latlng_to_xy(polygon_object_latlng,
                                  centroid_latitude_deg=None,
                                  centroid_longitude_deg=None):
    """Projects polygon from lat-long to x-y coordinates.
    
    :param polygon_object_latlng: Instance of `shapely.geometry.Polygon`, where
        x-coordinates are actually longitudes and y-coordinates are actually
        latitudes.
    :param centroid_latitude_deg: Latitude (deg N) at polygon centroid.
    :param centroid_longitude_deg: Longitude (deg E) at polygon centroid.
    :return: polygon_object_xy: Instance of `shapely.geometry.Polygon`, where x-
        and y-coordinates are in metres.
    """

    projection_object = projections.init_lambert_conformal_projection(
        standard_latitudes_deg=
        numpy.array([centroid_latitude_deg, centroid_latitude_deg]),
        central_longitude_deg=centroid_longitude_deg)

    vertex_latitudes_deg = numpy.asarray(polygon_object_latlng.exterior.xy[1])
    vertex_longitudes_deg = numpy.asarray(polygon_object_latlng.exterior.xy[0])
    vertex_x_metres, vertex_y_metres = projections.project_latlng_to_xy(
        vertex_latitudes_deg, vertex_longitudes_deg,
        projection_object=projection_object, false_easting_metres=0.,
        false_northing_metres=0.)

    return polygons.vertex_arrays_to_polygon_object(
        vertex_x_metres, vertex_y_metres)


def _xy_polygon_to_binary_matrix(
        polygon_object_xy,
        grid_spacing_metres=GRID_SPACING_FOR_BINARY_MATRIX_DEFAULT_METRES):
    """Converts x-y polygon to binary image matrix.
    
    M = number of rows in x-y grid
    N = number of columns in x-y grid
    
    :param polygon_object_xy: Instance of `shapely.geometry.Polygon`, where x-
        and y-coordinates are in metres.
    :param grid_spacing_metres: Grid spacing (distance between adjacent grid
        points).
    :return: binary_image_matrix_xy: M-by-N Boolean numpy array.  If
        binary_image_matrix[i, j] = True, grid point [i, j] is inside the
        polygon.  Otherwise, grid point [i, j] is outside the polygon.
    """

    vertex_x_metres = numpy.asarray(polygon_object_xy.exterior.xy[0])
    vertex_y_metres = numpy.asarray(polygon_object_xy.exterior.xy[1])

    num_grid_rows = int(numpy.ceil(
        (numpy.max(vertex_y_metres) - numpy.min(vertex_y_metres)) /
        grid_spacing_metres))
    num_grid_columns = int(numpy.ceil(
        (numpy.max(vertex_x_metres) - numpy.min(vertex_x_metres)) /
        grid_spacing_metres))

    vertex_x_metres = (
        num_grid_columns * (vertex_x_metres - numpy.min(vertex_x_metres)) /
        (numpy.max(vertex_x_metres) - numpy.min(vertex_x_metres)))
    vertex_y_metres = (
        num_grid_rows * (vertex_y_metres - numpy.min(vertex_y_metres)) /
        (numpy.max(vertex_y_metres) - numpy.min(vertex_y_metres)))

    num_vertices = len(vertex_x_metres)
    vertex_array_xy_metres = numpy.hstack((
        vertex_y_metres.reshape(num_vertices, 1),
        vertex_x_metres.reshape(num_vertices, 1)))
    return skimage.measure.grid_points_in_poly(
        (num_grid_rows, num_grid_columns), vertex_array_xy_metres)


def get_area_of_simple_polygon(polygon_object_latlng):
    """Computes area of simple polygon.

    :param polygon_object_latlng: Instance of `shapely.geometry.Polygon`, where
        x-coordinates are actually longitudes and y-coordinates are actually
        latitudes.
    :return: area_metres2: Area of polygon exterior.  All holes (interior) will
        be ignored.
    """

    return polygon_area([list(polygon_object_latlng.exterior.coords)])


def get_region_properties(binary_image_matrix,
                          property_names=DEFAULT_REGION_PROP_NAMES):
    """Computes region properties for one shape (polygon).

    M = number of rows in grid
    N = number of columns in grid

    :param binary_image_matrix: M-by-N Boolean numpy array.  If
        binary_image_matrix[i, j] = True, grid point [i, j] is inside the
        polygon.  Otherwise, grid point [i, j] is outside the polygon.
    :param property_names: 1-D list of region properties to compute.
    :return: property_dict: Dictionary, where each key is a string from
        `property_names` and each item is the corresponding value.
    """

    error_checking.assert_is_boolean_numpy_array(binary_image_matrix)
    error_checking.assert_is_numpy_array(binary_image_matrix, num_dimensions=2)

    error_checking.assert_is_string_list(property_names)
    error_checking.assert_is_numpy_array(
        numpy.array(property_names), num_dimensions=1)

    regionprops_object = skimage.measure.regionprops(
        binary_image_matrix.astype(int))[0]
    property_dict = {}

    for this_name in property_names:
        if this_name == ORIENTATION_NAME:
            property_dict.update({this_name: RADIANS_TO_DEGREES * getattr(
                regionprops_object, _stat_name_new_to_orig(this_name))})
        else:
            property_dict.update({this_name: getattr(
                regionprops_object, _stat_name_new_to_orig(this_name))})

    return property_dict


def get_curvature_based_stats(
        polygon_object_xy, statistic_names=DEFAULT_CURVATURE_BASED_STAT_NAMES):
    """Computes curvature-based statistics for one shape (polygon).

    :param polygon_object_xy: Instance of `shapely.geometry.Polygon`, where x-
        and y-coordinates are in metres.  If the polygon is a storm object (or
        anything else with only 90-degree angles), we recommend (but do not
        enforce) that it be smoothed -- using, for example,
        `smoothing_via_iterative_averaging.sia_for_closed_polygon`.
    :param statistic_names: 1-D list of curvature-based statistics to compute.
    :return: statistic_dict: Dictionary, where each key is a string from
        `statistic_names` and each item is the corresponding value.
    """

    vertex_curvatures_metres01 = shape_utils.get_curvature_for_closed_polygon(
        polygon_object_xy)

    statistic_dict = {}
    if MEAN_ABS_CURVATURE_NAME in statistic_names:
        statistic_dict.update(
            {MEAN_ABS_CURVATURE_NAME:
                 numpy.mean(numpy.absolute(vertex_curvatures_metres01))})

    if BENDING_ENERGY_NAME in statistic_names:
        statistic_dict.update(
            {BENDING_ENERGY_NAME: numpy.sum(
                vertex_curvatures_metres01 ** 2) / polygon_object_xy.length})

    if COMPACTNESS_NAME in statistic_names:
        statistic_dict.update(
            {COMPACTNESS_NAME: polygon_object_xy.length ** 2 / (
                4 * numpy.pi * polygon_object_xy.area)})

    return statistic_dict


def get_stats_for_storm_objects(
        storm_object_table, statistic_names=DEFAULT_STATISTIC_NAMES,
        grid_spacing_for_binary_matrix_metres=
        GRID_SPACING_FOR_BINARY_MATRIX_DEFAULT_METRES,
        num_vertices_in_smoothing_half_window=
        NUM_VERTICES_IN_SMOOTHING_HALF_WINDOW_DEFAULT,
        num_smoothing_iterations=NUM_SMOOTHING_ITERS_DEFAULT):
    """Computes shape statistics for one or more storm objects.

    K = number of statistics

    :param storm_object_table: pandas DataFrame with columns documented in
        `storm_tracking_io.write_processed_file`.  May contain additional
        columns.
    :param statistic_names: length-K list of statistics to compute.
    :param grid_spacing_for_binary_matrix_metres: See documentation for
        _xy_polygon_to_binary_matrix.
    :param num_vertices_in_smoothing_half_window: See documentation for
        `smoothing_via_iterative_averaging.sia_for_closed_polygon`.
    :param num_smoothing_iterations: See documentation for
        `smoothing_via_iterative_averaging.sia_for_closed_polygon`.
    :return: storm_object_table: Same as input, but with K additional columns.
        Names of additional columns come from `statistic_names`.
    """

    _check_statistic_names(statistic_names)
    region_property_names = _get_region_property_names(statistic_names)
    curvature_based_stat_names = _get_curvature_based_stat_names(
        statistic_names)

    num_storm_objects = len(storm_object_table.index)
    nan_array = numpy.full(num_storm_objects, numpy.nan)

    argument_dict = {}
    for this_name in statistic_names:
        argument_dict.update({this_name: nan_array})
    storm_object_table = storm_object_table.assign(**argument_dict)

    for i in range(num_storm_objects):
        this_polygon_object_latlng = storm_object_table[
            tracking_io.POLYGON_OBJECT_LATLNG_COLUMN].values[i]
        
        if AREA_NAME in statistic_names:
            storm_object_table[AREA_NAME].values[i] = (
                get_area_of_simple_polygon(this_polygon_object_latlng))

        if region_property_names or curvature_based_stat_names:
            this_polygon_object_xy = _project_polygon_latlng_to_xy(
                this_polygon_object_latlng,
                centroid_latitude_deg=
                storm_object_table[tracking_io.CENTROID_LAT_COLUMN].values[i],
                centroid_longitude_deg=
                storm_object_table[tracking_io.CENTROID_LNG_COLUMN].values[i])

        if region_property_names:
            this_binary_image_matrix = _xy_polygon_to_binary_matrix(
                this_polygon_object_xy, grid_spacing_for_binary_matrix_metres)

            this_region_prop_dict = get_region_properties(
                this_binary_image_matrix, property_names=region_property_names)

            for this_name in region_property_names:
                storm_object_table[this_name].values[i] = this_region_prop_dict[
                    this_name]

        if curvature_based_stat_names:
            these_x_smoothed_metres, these_y_smoothed_metres = (
                sia.sia_for_closed_polygon(
                    this_polygon_object_xy,
                    num_vertices_in_half_window=
                    num_vertices_in_smoothing_half_window,
                    num_iterations=num_smoothing_iterations,
                    check_input_args=i == 0))

            this_polygon_object_xy_smoothed = (
                polygons.vertex_arrays_to_polygon_object(
                    these_x_smoothed_metres, these_y_smoothed_metres))

            this_curvature_based_stat_dict = get_curvature_based_stats(
                this_polygon_object_xy_smoothed,
                statistic_names=curvature_based_stat_names)

            for this_name in curvature_based_stat_names:
                storm_object_table[this_name].values[i] = (
                    this_curvature_based_stat_dict[this_name])

    return storm_object_table
