"""Methods for computing shape statistics."""

import copy
import numpy
import skimage.measure
from area import polygon__area as polygon_area
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import error_checking

# TODO(thunderhoser): may add moments to list of shape statistics.

RADIANS_TO_DEGREES = 180. / numpy.pi
DEFAULT_GRID_SPACING_METRES = 100.

AREA_NAME = 'area_metres2'
ECCENTRICITY_NAME = 'eccentricity'
EXTENT_NAME = 'extent'
SOLIDITY_NAME = 'solidity'
ORIENTATION_NAME = 'orientation_deg'
PERIMETER_NAME = 'perimeter_metres2'

AREA_NAME_ORIG = 'area'
ECCENTRICITY_NAME_ORIG = 'eccentricity'
EXTENT_NAME_ORIG = 'extent'
SOLIDITY_NAME_ORIG = 'solidity'
ORIENTATION_NAME_ORIG = 'orientation'
PERIMETER_NAME_ORIG = 'perimeter'

VALID_REGION_PROPERTY_NAMES = [
    ECCENTRICITY_NAME, EXTENT_NAME, SOLIDITY_NAME, ORIENTATION_NAME,
    PERIMETER_NAME]
VALID_REGION_PROP_NAMES_ORIG = [
    ECCENTRICITY_NAME_ORIG, EXTENT_NAME_ORIG, SOLIDITY_NAME_ORIG,
    ORIENTATION_NAME_ORIG, PERIMETER_NAME_ORIG]
DEFAULT_REGION_PROPERTY_NAMES = copy.deepcopy(VALID_REGION_PROPERTY_NAMES)

VALID_STATISTIC_NAMES = VALID_REGION_PROPERTY_NAMES + [AREA_NAME]
VALID_STAT_NAMES_ORIG = VALID_REGION_PROP_NAMES_ORIG + [AREA_NAME_ORIG]
DEFAULT_STATISTIC_NAMES = copy.deepcopy(VALID_STATISTIC_NAMES)


def _check_statistic_names(statistic_names):
    """Ensures that statistic names are valid.

    :param statistic_names: 1-D list of statistic names.
    :raises: ValueError: if any element of `statistic_names` is not in
        `VALID_STATISTIC_NAMES`.
    """

    error_checking.assert_is_string_list(statistic_names)
    error_checking.assert_is_numpy_array(
        numpy.array(statistic_names), num_dimensions=1)

    for this_name in statistic_names:
        if this_name in VALID_STATISTIC_NAMES:
            continue

        error_string = (
            '\n\n' + str(VALID_STATISTIC_NAMES) + '\n\nValid statistic names ' +
            '(listed above) do not include the following: "' + this_name + '"')
        raise ValueError(error_string)


def _stat_name_new_to_orig(statistic_name):
    """Converts name of statistic from new to original format.

    New format = GewitterGefahr
    Original format = `skimage.measure.regionprops`

    :param statistic_name: Statistic name in new format.
    :return: statistic_name_orig: Statistic name in original format.
    """

    found_flags = [s == statistic_name for s in VALID_STATISTIC_NAMES]
    return VALID_STAT_NAMES_ORIG[numpy.where(found_flags)[0][0]]


def _latlng_polygon_to_binary_xy_matrix(
        polygon_object_latlng, centroid_latitude_deg=None,
        centroid_longitude_deg=None,
        grid_spacing_metres=DEFAULT_GRID_SPACING_METRES):
    """Converts lat-long polygon to binary image matrix in x-y coordinates.

    M = number of rows in x-y grid
    N = number of columns in x-y grid

    :param polygon_object_latlng: Instance of `shapely.geometry.Polygon`, where
        x-coordinates are actually longitudes and y-coordinates are actually
        latitudes.
    :param centroid_latitude_deg: Latitude (deg N) at polygon centroid.
    :param centroid_longitude_deg: Longitude (deg E) at polygon centroid.
    :param grid_spacing_metres: Spacing (distance between adjacent grid points)
        for x-y grid.
    :return: binary_image_matrix_xy: M-by-N Boolean numpy array.  If
        binary_image_matrix[i, j] = True, grid point [i, j] is inside the
        polygon.  Otherwise, grid point [i, j] is outside the polygon.
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

    num_grid_rows = 1 + int(numpy.ceil(
        (numpy.max(vertex_y_metres) - numpy.min(vertex_y_metres)) /
        grid_spacing_metres))
    num_grid_columns = 1 + int(numpy.ceil(
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
                          property_names=DEFAULT_REGION_PROPERTY_NAMES):
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


def get_stats_for_storm_objects(
        storm_object_table, statistic_names=DEFAULT_STATISTIC_NAMES,
        grid_spacing_metres=DEFAULT_GRID_SPACING_METRES):
    """Computes shape statistics for one or more storm objects.

    K = number of statistics

    :param storm_object_table: pandas DataFrame with columns documented in
        `storm_tracking_io.write_processed_file`.  May contain additional
        columns.
    :param statistic_names: length-K list of statistics to compute.
    :param grid_spacing_metres: See documentation for
        _latlng_polygon_to_binary_xy_matrix.
    :return: storm_object_table: Same as input, but with K additional columns.
        Names of additional columns come from `statistic_names`.
    """

    _check_statistic_names(statistic_names)

    region_property_names = set(statistic_names)
    if AREA_NAME in region_property_names:
        region_property_names.remove(AREA_NAME)
    region_property_names = list(region_property_names)

    num_storm_objects = len(storm_object_table.index)
    nan_array = numpy.full(num_storm_objects, numpy.nan)

    argument_dict = {}
    for this_name in statistic_names:
        argument_dict.update({this_name: nan_array})
    storm_object_table = storm_object_table.assign(**argument_dict)

    for i in range(num_storm_objects):
        if AREA_NAME in statistic_names:
            storm_object_table[AREA_NAME].values[i] = (
                get_area_of_simple_polygon(
                    storm_object_table[
                        tracking_io.POLYGON_OBJECT_LATLNG_COLUMN].values[i]))

        this_binary_image_matrix = _latlng_polygon_to_binary_xy_matrix(
            storm_object_table[
                tracking_io.POLYGON_OBJECT_LATLNG_COLUMN].values[i],
            centroid_latitude_deg=
            storm_object_table[tracking_io.CENTROID_LAT_COLUMN].values[i],
            centroid_longitude_deg=
            storm_object_table[tracking_io.CENTROID_LNG_COLUMN].values[i],
            grid_spacing_metres=grid_spacing_metres)

        this_region_prop_dict = get_region_properties(
            this_binary_image_matrix, property_names=region_property_names)
        for this_name in region_property_names:
            storm_object_table[this_name].values[i] = this_region_prop_dict[
                this_name]

    return storm_object_table
