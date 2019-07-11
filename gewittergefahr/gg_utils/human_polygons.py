"""Handles polygons drawn interactively by a human."""

import numpy
import matplotlib.pyplot as pyplot
import netCDF4
from PIL import Image
import shapely.geometry
from roipoly import MultiRoi
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import general_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

x_coords_px = numpy.array([], dtype=float)
y_coords_px = numpy.array([], dtype=float)
figure_object = None

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

MARKER_TYPE = 'D'
MARKER_SIZE = 10
MARKER_EDGE_WIDTH = 1
MARKER_COLOUR = 'k'

SENTINEL_VALUE = -9999.
DUMMY_STORM_ID_STRING = 'pmm'

STORM_ID_KEY = 'full_storm_id_string'
STORM_TIME_KEY = 'storm_time_unix_sec'
POSITIVE_VERTEX_ROWS_KEY = 'positive_vertex_rows'
POSITIVE_VERTEX_COLUMNS_KEY = 'positive_vertex_columns'
POSITIVE_PANEL_ROW_BY_VERTEX_KEY = 'positive_panel_row_by_vertex'
POSITIVE_PANEL_COLUMN_BY_VERTEX_KEY = 'positive_panel_column_by_vertex'
NEGATIVE_VERTEX_ROWS_KEY = 'negative_vertex_rows'
NEGATIVE_VERTEX_COLUMNS_KEY = 'negative_vertex_columns'
NEGATIVE_PANEL_ROW_BY_VERTEX_KEY = 'negative_panel_row_by_vertex'
NEGATIVE_PANEL_COLUMN_BY_VERTEX_KEY = 'negative_panel_column_by_vertex'
POSITIVE_MASK_MATRIX_KEY = 'positive_mask_matrix'
POSITIVE_POLYGON_OBJECTS_KEY = 'positive_objects_grid_coords'
NEGATIVE_MASK_MATRIX_KEY = 'negative_mask_matrix'
NEGATIVE_POLYGON_OBJECTS_KEY = 'negative_objects_grid_coords'

GRID_ROW_DIMENSION_KEY = 'grid_row'
GRID_COLUMN_DIMENSION_KEY = 'grid_column'
PANEL_ROW_DIMENSION_KEY = 'panel_row'
PANEL_COLUMN_DIMENSION_KEY = 'panel_column'
POSITIVE_VERTEX_DIM_KEY = 'positive_polygon_vertex'
NEGATIVE_VERTEX_DIM_KEY = 'negative_polygon_vertex'

POSITIVE_PANEL_ROW_BY_POLY_KEY = 'positive_panel_row_by_polygon'
POSITIVE_PANEL_COLUMN_BY_POLY_KEY = 'positive_panel_column_by_polygon'
NEGATIVE_PANEL_ROW_BY_POLY_KEY = 'negative_panel_row_by_polygon'
NEGATIVE_PANEL_COLUMN_BY_POLY_KEY = 'negative_panel_column_by_polygon'

POINT_DIMENSION_KEY = 'point'
GRID_ROW_BY_POINT_KEY = 'grid_row_by_point'
GRID_COLUMN_BY_POINT_KEY = 'grid_column_by_point'
PANEL_ROW_BY_POINT_KEY = 'panel_row_by_point'
PANEL_COLUMN_BY_POINT_KEY = 'panel_column_by_point'


def _check_polygons(
        polygon_objects_grid_coords, num_panel_rows, num_panel_columns,
        panel_row_by_polygon, panel_column_by_polygon):
    """Error-checks list of polygons.

    :param polygon_objects_grid_coords: See doc for
        `polygons_from_pixel_to_grid_coords`.
    :param num_panel_rows: Same.
    :param num_panel_columns: Same.
    :param panel_row_by_polygon: Same.
    :param panel_column_by_polygon: Same.
    """

    error_checking.assert_is_integer(num_panel_rows)
    error_checking.assert_is_greater(num_panel_rows, 0)
    error_checking.assert_is_integer(num_panel_columns)
    error_checking.assert_is_greater(num_panel_columns, 0)

    num_polygons = len(polygon_objects_grid_coords)
    if num_polygons == 0:
        return

    error_checking.assert_is_numpy_array(
        numpy.array(polygon_objects_grid_coords, dtype=object), num_dimensions=1
    )

    these_expected_dim = numpy.array([num_polygons], dtype=int)

    error_checking.assert_is_integer_numpy_array(panel_row_by_polygon)
    error_checking.assert_is_numpy_array(
        panel_row_by_polygon, exact_dimensions=these_expected_dim)
    error_checking.assert_is_geq_numpy_array(panel_row_by_polygon, 0)
    error_checking.assert_is_less_than_numpy_array(
        panel_row_by_polygon, num_panel_rows)

    error_checking.assert_is_integer_numpy_array(panel_column_by_polygon)
    error_checking.assert_is_numpy_array(
        panel_column_by_polygon, exact_dimensions=these_expected_dim)
    error_checking.assert_is_geq_numpy_array(panel_column_by_polygon, 0)
    error_checking.assert_is_less_than_numpy_array(
        panel_column_by_polygon, num_panel_columns)


def _polygon_list_to_vertex_list(polygon_objects_grid_coords):
    """Converts list of polygons to list of vertices.

    V = total number of vertices

    :param polygon_objects_grid_coords: List of polygons created by
        `polygons_from_pixel_to_grid_coords`.
    :return: vertex_rows: length-V numpy array of row coordinates.
    :return: vertex_columns: length-V numpy array of column coordinates.
    :return: vertex_to_polygon_indices: length-V numpy array of indices.
        If vertex_to_polygon_indices[i] = j, the [i]th vertex comes from the
        [j]th input polygon.
    """

    error_checking.assert_is_list(polygon_objects_grid_coords)
    num_polygons = len(polygon_objects_grid_coords)

    if num_polygons > 0:
        error_checking.assert_is_numpy_array(
            numpy.array(polygon_objects_grid_coords, dtype=object),
            num_dimensions=1
        )

    vertex_rows = []
    vertex_columns = []
    vertex_to_polygon_indices = []

    for i in range(num_polygons):
        this_num_vertices = len(polygon_objects_grid_coords[i].exterior.xy[1])

        vertex_rows += polygon_objects_grid_coords[i].exterior.xy[1]
        vertex_columns += polygon_objects_grid_coords[i].exterior.xy[0]
        vertex_to_polygon_indices += [i] * this_num_vertices

        if i == num_polygons - 1:
            continue

        vertex_rows += [numpy.nan]
        vertex_columns += [numpy.nan]
        vertex_to_polygon_indices += [-1]

    return (
        numpy.array(vertex_rows), numpy.array(vertex_columns),
        numpy.array(vertex_to_polygon_indices, dtype=int)
    )


def _vertex_list_to_polygon_list(vertex_rows, vertex_columns):
    """This method is the inverse of `_polygon_list_to_vertex_list`.

    P = number of polygons

    :param vertex_rows: See doc for `_polygon_list_to_vertex_list`.
    :param vertex_columns: Same.
    :return: polygon_objects_grid_coords: Same.
    :return: polygon_to_first_vertex_indices: length-P numpy array of indices.
        If polygon_to_first_vertex_indices[j] = i, the first vertex in the
        [j]th polygon is the [i]th vertex in the input arrays.
    :raises: ValueError: if row and column lists have NaN's at different
        locations.
    """

    if len(vertex_rows) == 0:
        return [], numpy.array([], dtype=int)

    nan_row_indices = numpy.where(numpy.isnan(vertex_rows))[0]
    nan_column_indices = numpy.where(numpy.isnan(vertex_columns))[0]

    if not numpy.array_equal(nan_row_indices, nan_column_indices):
        error_string = (
            'Row ({0:s}) and column ({1:s}) lists have NaN''s at different '
            'locations.'
        ).format(str(nan_row_indices), str(nan_column_indices))

        raise ValueError(error_string)

    polygon_to_first_vertex_indices = numpy.concatenate((
        numpy.array([0], dtype=int),
        nan_row_indices + 1
    ))

    vertex_rows_by_polygon = general_utils.split_array_by_nan(vertex_rows)
    vertex_columns_by_polygon = general_utils.split_array_by_nan(vertex_columns)

    num_polygons = len(vertex_rows_by_polygon)
    polygon_objects_grid_coords = []

    for i in range(num_polygons):
        this_polygon_object = polygons.vertex_arrays_to_polygon_object(
            exterior_x_coords=vertex_columns_by_polygon[i],
            exterior_y_coords=vertex_rows_by_polygon[i]
        )

        polygon_objects_grid_coords.append(this_polygon_object)

    return polygon_objects_grid_coords, polygon_to_first_vertex_indices


def _polygons_to_mask_one_panel(polygon_objects_grid_coords, num_grid_rows,
                                num_grid_columns):
    """Converts list of polygons to binary mask.

    M = number of rows in grid
    N = number of columns in grid

    :param polygon_objects_grid_coords: See doc for
        `polygons_from_pixel_to_grid_coords`.
    :param num_grid_rows: Same.
    :param num_grid_columns: Same.
    :return: mask_matrix: M-by-N numpy array of Boolean flags.  If
        mask_matrix[i, j] == True, grid point [i, j] is in/on at least one of
        the polygons.
    """

    mask_matrix = numpy.full(
        (num_grid_rows, num_grid_columns), False, dtype=bool
    )

    num_polygons = len(polygon_objects_grid_coords)
    if num_polygons == 0:
        return mask_matrix

    # TODO(thunderhoser): This triple for-loop is probably inefficient.
    for k in range(num_polygons):
        these_grid_columns = numpy.array(
            polygon_objects_grid_coords[k].exterior.xy[0]
        )

        error_checking.assert_is_geq_numpy_array(these_grid_columns, -0.5)
        error_checking.assert_is_leq_numpy_array(
            these_grid_columns, num_grid_columns - 0.5)

        these_grid_rows = numpy.array(
            polygon_objects_grid_coords[k].exterior.xy[1]
        )

        error_checking.assert_is_geq_numpy_array(these_grid_rows, -0.5)
        error_checking.assert_is_leq_numpy_array(
            these_grid_rows, num_grid_rows - 0.5)

        for i in range(num_grid_rows):
            for j in range(num_grid_columns):
                if mask_matrix[i, j]:
                    continue

                mask_matrix[i, j] = polygons.point_in_or_on_polygon(
                    polygon_object=polygon_objects_grid_coords[k],
                    query_x_coordinate=j, query_y_coordinate=i)

    return mask_matrix


def _click_handler(event_object):
    """Handles human mouse clicks on a matplotlib figure.

    N = number of mouse clicks handled

    :param event_object: Event-handler (I have no idea what type of object this
        is).
    :return: x_coords_px: length-N numpy array with x-coordinates of mouse
        clicks.
    :return: y_coords_px: length-N numpy array with y-coordinates of mouse
        clicks.
    """

    global x_coords_px
    global y_coords_px
    global figure_object

    x_coords_px = numpy.concatenate((
        x_coords_px, numpy.full(1, event_object.xdata)
    ))

    y_coords_px = numpy.concatenate((
        y_coords_px, numpy.full(1, event_object.ydata)
    ))

    pyplot.plot(
        event_object.xdata, event_object.ydata, linestyle='None',
        marker=MARKER_TYPE, markerfacecolor=MARKER_COLOUR,
        markeredgecolor=MARKER_COLOUR, markersize=MARKER_SIZE,
        markeredgewidth=MARKER_EDGE_WIDTH)

    figure_object.canvas.draw()

    return x_coords_px, y_coords_px


def capture_polygons(image_file_name, instruction_string=''):
    """This interactiv method allows you to draw polygons and captures vertices.

    N = number of polygons drawn

    :param image_file_name: Path to image file.  This method will display the
        image in a figure window and allow you to draw polygons on top.
    :param instruction_string: String with instructions for the user.
    :return: polygon_objects_pixel_coords: length-N list of polygons (instances
        of `shapely.geometry.Polygon`), each containing vertices in pixel
        coordinates.
    :return: num_pixel_rows: Number of pixel rows in the image.
    :return: num_pixel_columns: Number of pixel columns in the image.
    """

    error_checking.assert_file_exists(image_file_name)
    error_checking.assert_is_string(instruction_string)

    image_matrix = Image.open(image_file_name)
    num_pixel_columns, num_pixel_rows = image_matrix.size

    pyplot.imshow(image_matrix)
    pyplot.title(instruction_string)
    pyplot.show(block=False)

    multi_roi_object = MultiRoi()

    string_keys = list(multi_roi_object.rois.keys())
    integer_keys = numpy.array([int(k) for k in string_keys], dtype=int)

    sort_indices = numpy.argsort(integer_keys)
    integer_keys = integer_keys[sort_indices]
    string_keys = [string_keys[k] for k in sort_indices]

    num_polygons = len(integer_keys)
    polygon_objects_pixel_coords = []

    for i in range(num_polygons):
        this_roi_object = multi_roi_object.rois[string_keys[i]]

        these_x_coords = numpy.array(
            [this_roi_object.x[0]] + list(reversed(this_roi_object.x))
        )
        these_y_coords = numpy.array(
            [this_roi_object.y[0]] + list(reversed(this_roi_object.y))
        )

        this_polygon_object = polygons.vertex_arrays_to_polygon_object(
            exterior_x_coords=these_x_coords, exterior_y_coords=these_y_coords)

        polygon_objects_pixel_coords.append(this_polygon_object)

    return polygon_objects_pixel_coords, num_pixel_rows, num_pixel_columns


def capture_mouse_clicks(image_file_name, instruction_string=''):
    """This interactive method captures coordinates of human mouse clicks.

    N = number of mouse clicks

    :param image_file_name: Path to image file.  This method will display the
        image in a figure window and allow you to click on top.
    :param instruction_string: String with instructions for the user.
    :return: point_objects_pixel_coords: length-N list of points (instances
        of `shapely.geometry.Point`), each containing a click location in pixel
        coordinates.
    :return: num_pixel_rows: Number of pixel rows in the image.
    :return: num_pixel_columns: Number of pixel columns in the image.
    """

    error_checking.assert_file_exists(image_file_name)
    error_checking.assert_is_string(instruction_string)

    image_matrix = Image.open(image_file_name)
    num_pixel_columns, num_pixel_rows = image_matrix.size

    global figure_object
    figure_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )[0]

    pyplot.imshow(image_matrix)
    pyplot.title(instruction_string)

    connection_id = figure_object.canvas.mpl_connect(
        'button_press_event', _click_handler)
    pyplot.show()
    figure_object.canvas.mpl_disconnect(connection_id)

    point_objects_pixel_coords = [
        shapely.geometry.Point(this_x, this_y)
        for this_x, this_y in zip(x_coords_px, y_coords_px)
    ]

    return point_objects_pixel_coords, num_pixel_rows, num_pixel_columns


def pixel_rows_to_grid_rows(
        pixel_row_by_vertex, num_pixel_rows, num_panel_rows, num_grid_rows,
        assert_same_panel):
    """Converts pixel rows to grid rows.

    V = number of vertices in object

    :param pixel_row_by_vertex: length-V numpy array with row coordinates of
        vertices in pixel space.
    :param num_pixel_rows: Total number of pixel rows in image.
    :param num_panel_rows: Total number of panel rows in image.
    :param num_grid_rows: Total number of rows in grid (one grid per panel).
    :param assert_same_panel: Boolean flag.  If True, all vertices must be in
        the same panel.
    :return: grid_row_by_vertex: length-V numpy array with row coordinates
        (floats) of vertices in grid space.
    :return: panel_row_by_vertex: length-V numpy array with row coordinates
        (integers) of vertices in panel space.
    """

    error_checking.assert_is_integer(num_pixel_rows)
    error_checking.assert_is_greater(num_pixel_rows, 0)
    error_checking.assert_is_integer(num_panel_rows)
    error_checking.assert_is_greater(num_panel_rows, 0)
    error_checking.assert_is_integer(num_grid_rows)
    error_checking.assert_is_greater(num_grid_rows, 0)
    error_checking.assert_is_boolean(assert_same_panel)

    error_checking.assert_is_numpy_array(pixel_row_by_vertex, num_dimensions=1)

    pixel_row_by_vertex += 0.5
    error_checking.assert_is_geq_numpy_array(pixel_row_by_vertex, 0.)
    error_checking.assert_is_leq_numpy_array(
        pixel_row_by_vertex, num_pixel_rows)

    panel_row_to_first_px_row = {}
    for i in range(num_panel_rows):
        panel_row_to_first_px_row[i] = (
            i * float(num_pixel_rows) / num_panel_rows
        )

    panel_row_by_vertex = numpy.floor(
        pixel_row_by_vertex * float(num_panel_rows) / num_pixel_rows
    ).astype(int)

    panel_row_by_vertex[
        panel_row_by_vertex == num_panel_rows
    ] = num_panel_rows - 1

    if assert_same_panel and len(numpy.unique(panel_row_by_vertex)) > 1:
        error_string = (
            'Object is in multiple panels.  Panel rows listed below.\n{0:s}'
        ).format(str(panel_row_by_vertex))

        raise ValueError(error_string)

    num_vertices = len(pixel_row_by_vertex)
    for i in range(num_vertices):
        pixel_row_by_vertex[i] = (
            pixel_row_by_vertex[i] -
            panel_row_to_first_px_row[panel_row_by_vertex[i]]
        )

    grid_row_by_vertex = -0.5 + (
        pixel_row_by_vertex *
        float(num_grid_rows * num_panel_rows) / num_pixel_rows
    )

    grid_row_by_vertex = num_grid_rows - 1 - grid_row_by_vertex

    return grid_row_by_vertex, panel_row_by_vertex


def pixel_columns_to_grid_columns(
        pixel_column_by_vertex, num_pixel_columns, num_panel_columns,
        num_grid_columns, assert_same_panel):
    """Converts pixel columns to grid columns.

    V = number of vertices in object

    :param pixel_column_by_vertex: length-V numpy array with column coordinates
        of vertices in pixel space.
    :param num_pixel_columns: Total number of pixel columns in image.
    :param num_panel_columns: Total number of panel columns in image.
    :param num_grid_columns: Total number of columns in grid (one grid per
        panel).
    :param assert_same_panel: Boolean flag.  If True, all vertices must be in
        the same panel.
    :return: grid_column_by_vertex: length-V numpy array with column coordinates
        (floats) of vertices in grid space.
    :return: panel_column_by_vertex: length-V numpy array with column
        coordinates (integers) of vertices in panel space.
    """

    error_checking.assert_is_integer(num_pixel_columns)
    error_checking.assert_is_greater(num_pixel_columns, 0)
    error_checking.assert_is_integer(num_panel_columns)
    error_checking.assert_is_greater(num_panel_columns, 0)
    error_checking.assert_is_integer(num_grid_columns)
    error_checking.assert_is_greater(num_grid_columns, 0)
    error_checking.assert_is_boolean(assert_same_panel)

    error_checking.assert_is_numpy_array(
        pixel_column_by_vertex, num_dimensions=1)

    pixel_column_by_vertex += 0.5
    error_checking.assert_is_geq_numpy_array(pixel_column_by_vertex, 0.)
    error_checking.assert_is_leq_numpy_array(
        pixel_column_by_vertex, num_pixel_columns)

    panel_column_to_first_px_column = {}
    for j in range(num_panel_columns):
        panel_column_to_first_px_column[j] = (
            j * float(num_pixel_columns) / num_panel_columns
        )

    panel_column_by_vertex = numpy.floor(
        pixel_column_by_vertex * float(num_panel_columns) / num_pixel_columns
    ).astype(int)

    panel_column_by_vertex[
        panel_column_by_vertex == num_panel_columns
    ] = num_panel_columns - 1

    if assert_same_panel and len(numpy.unique(panel_column_by_vertex)) > 1:
        error_string = (
            'Object is in multiple panels.  Panel columns listed below.\n{0:s}'
        ).format(str(panel_column_by_vertex))

        raise ValueError(error_string)

    num_vertices = len(pixel_column_by_vertex)
    for i in range(num_vertices):
        pixel_column_by_vertex[i] = (
            pixel_column_by_vertex[i] -
            panel_column_to_first_px_column[panel_column_by_vertex[i]]
        )

    grid_column_by_vertex = -0.5 + (
        pixel_column_by_vertex *
        float(num_grid_columns * num_panel_columns) / num_pixel_columns
    )

    return grid_column_by_vertex, panel_column_by_vertex


def polygons_from_pixel_to_grid_coords(
        polygon_objects_pixel_coords, num_grid_rows, num_grid_columns,
        num_pixel_rows, num_pixel_columns, num_panel_rows, num_panel_columns):
    """Converts polygons from pixel coordinates to grid coordinates.

    The input args `num_grid_rows` and `num_grid_columns` are the number of rows
    and columns in the data grid.  These are *not* the same as the number of
    pixel rows and columns in the image.

    This method assumes that the image contains one or more panels with gridded
    data.  Each panel may contain a different variable, but they must all
    contain the same grid, with the same aspect ratio and no whitespace border
    (between the panels or around the outside of the image).

    N = number of polygons

    :param polygon_objects_pixel_coords: length-N list of polygons (instances of
        `shapely.geometry.Polygon`) with vertices in pixel coordinates, where
        the top-left corner is x = y = 0.
    :param num_grid_rows: Number of rows in grid.
    :param num_grid_columns: Number of columns in grid.
    :param num_pixel_rows: Number of pixel rows in image.
    :param num_pixel_columns: Number of pixel columns in image.
    :param num_panel_rows: Number of panel rows in image.
    :param num_panel_columns: Number of panel columns in image.
    :return: polygon_objects_grid_coords: length-N list of polygons
        (instances of `shapely.geometry.Polygon`) with vertices in grid
        coordinates, where the bottom-left corner is x = y = 0.
    :return: panel_row_by_polygon: length-N numpy array of panel rows.  If
        panel_rows[k] = i, the [k]th polygon corresponds to a grid in the [i]th
        panel row, where the top row is the 0th.
    :return: panel_column_by_polygon: length-N numpy array of panel columns.  If
        panel_columns[k] = j, the [k]th polygon corresponds to a grid in the
        [j]th panel column, where the left column is the 0th.
    :raises: ValueError: if one polygon is in multiple panels.
    """

    error_checking.assert_is_list(polygon_objects_pixel_coords)

    num_polygons = len(polygon_objects_pixel_coords)
    if num_polygons == 0:
        empty_array = numpy.array([], dtype=int)
        return polygon_objects_pixel_coords, empty_array, empty_array

    error_checking.assert_is_numpy_array(
        numpy.array(polygon_objects_pixel_coords, dtype=object),
        num_dimensions=1
    )

    polygon_objects_grid_coords = [None] * num_polygons
    panel_row_by_polygon = [None] * num_polygons
    panel_column_by_polygon = [None] * num_polygons

    for k in range(num_polygons):
        these_grid_columns, these_panel_columns = pixel_columns_to_grid_columns(
            pixel_column_by_vertex=numpy.array(
                polygon_objects_pixel_coords[k].exterior.xy[0]
            ),
            num_pixel_columns=num_pixel_columns,
            num_panel_columns=num_panel_columns,
            num_grid_columns=num_grid_columns, assert_same_panel=True)

        panel_column_by_polygon[k] = these_panel_columns[0]

        these_grid_rows, these_panel_rows = pixel_rows_to_grid_rows(
            pixel_row_by_vertex=numpy.array(
                polygon_objects_pixel_coords[k].exterior.xy[1]
            ),
            num_pixel_rows=num_pixel_rows, num_panel_rows=num_panel_rows,
            num_grid_rows=num_grid_rows, assert_same_panel=True)

        panel_row_by_polygon[k] = these_panel_rows[0]

        polygon_objects_grid_coords[k] = (
            polygons.vertex_arrays_to_polygon_object(
                exterior_x_coords=these_grid_columns,
                exterior_y_coords=these_grid_rows)
        )

    panel_row_by_polygon = numpy.array(panel_row_by_polygon, dtype=int)
    panel_column_by_polygon = numpy.array(panel_column_by_polygon, dtype=int)

    return (polygon_objects_grid_coords, panel_row_by_polygon,
            panel_column_by_polygon)


def polygons_to_mask(
        polygon_objects_grid_coords, num_grid_rows, num_grid_columns,
        num_panel_rows, num_panel_columns, panel_row_by_polygon,
        panel_column_by_polygon):
    """Converts list of polygons to one binary mask for each panel.

    M = number of rows in grid
    N = number of columns in grid
    J = number of panel rows in image
    K = number of panel columns in image

    :param polygon_objects_grid_coords: See doc for
        `polygons_from_pixel_to_grid_coords`.
    :param num_grid_rows: Same.
    :param num_grid_columns: Same.
    :param num_panel_rows: Same.
    :param num_panel_columns: Same.
    :param panel_row_by_polygon: Same.
    :param panel_column_by_polygon: Same.
    :return: mask_matrix: J-by-K-by-M-by-N numpy array of Boolean flags.  If
        mask_matrix[j, k, m, n] == True, grid point [m, n] in panel [j, k] is
        in/on at least one of the polygons.
    """

    error_checking.assert_is_integer(num_grid_rows)
    error_checking.assert_is_greater(num_grid_rows, 0)
    error_checking.assert_is_integer(num_grid_columns)
    error_checking.assert_is_greater(num_grid_columns, 0)

    _check_polygons(
        polygon_objects_grid_coords=polygon_objects_grid_coords,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns,
        panel_row_by_polygon=panel_row_by_polygon,
        panel_column_by_polygon=panel_column_by_polygon)

    mask_matrix = numpy.full(
        (num_panel_rows, num_panel_columns, num_grid_rows, num_grid_columns),
        False, dtype=bool
    )

    num_polygons = len(polygon_objects_grid_coords)
    if num_polygons == 0:
        return mask_matrix

    panel_coord_matrix = numpy.hstack((
        numpy.reshape(panel_row_by_polygon, (num_polygons, 1)),
        numpy.reshape(panel_column_by_polygon, (num_polygons, 1))
    ))

    panel_coord_matrix = numpy.unique(panel_coord_matrix.astype(int), axis=0)

    for i in range(panel_coord_matrix.shape[0]):
        this_panel_row = panel_coord_matrix[i, 0]
        this_panel_column = panel_coord_matrix[i, 1]

        these_polygon_indices = numpy.where(numpy.logical_and(
            panel_row_by_polygon == this_panel_row,
            panel_column_by_polygon == this_panel_column
        ))[0]

        these_polygon_objects = [
            polygon_objects_grid_coords[k] for k in these_polygon_indices
        ]

        mask_matrix[this_panel_row, this_panel_column, ...] = (
            _polygons_to_mask_one_panel(
                polygon_objects_grid_coords=these_polygon_objects,
                num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns)
        )

    return mask_matrix


def write_polygons(
        output_file_name, positive_objects_grid_coords,
        positive_panel_row_by_polygon, positive_panel_column_by_polygon,
        positive_mask_matrix, negative_objects_grid_coords=None,
        negative_panel_row_by_polygon=None,
        negative_panel_column_by_polygon=None, negative_mask_matrix=None,
        full_storm_id_string=None, storm_time_unix_sec=None):
    """Writes human polygons for one image to NetCDF file.

    P = number of positive regions of interest
    N = number of negative regions of interest

    :param output_file_name: Path to output (NetCDF) file.
    :param positive_objects_grid_coords: length-P list of polygons created by
        `polygons_from_pixel_to_grid_coords`, containing positive regions of
        interest.
    :param positive_panel_row_by_polygon: length-P numpy array of corresponding
        panel rows (non-negative integers).
    :param positive_panel_column_by_polygon: length-P numpy array of
        corresponding panel columns (non-negative integers).
    :param positive_mask_matrix: Binary mask for positive regions of interest,
        created by `polygons_to_mask`.
    :param negative_objects_grid_coords: length-N list of polygons created by
        `polygons_from_pixel_to_grid_coords`, containing negative regions of
        interest.
    :param negative_panel_row_by_polygon: length-N numpy array of corresponding
        panel rows (non-negative integers).
    :param negative_panel_column_by_polygon: length-N numpy array of
        corresponding panel columns (non-negative integers).
    :param negative_mask_matrix: Binary mask for negative regions of interest,
        created by `polygons_to_mask`.
    :param full_storm_id_string: Full storm ID (if polygons were drawn for
        composite, this should be None).
    :param storm_time_unix_sec: Valid time (if polygons were drawn for
        composite, this should be None).
    """

    is_composite = (
        full_storm_id_string is None and storm_time_unix_sec is None
    )

    if is_composite:
        full_storm_id_string = DUMMY_STORM_ID_STRING
        storm_time_unix_sec = -1

    error_checking.assert_is_string(full_storm_id_string)
    error_checking.assert_is_integer(storm_time_unix_sec)

    error_checking.assert_is_boolean_numpy_array(positive_mask_matrix)
    error_checking.assert_is_numpy_array(positive_mask_matrix, num_dimensions=4)

    _check_polygons(
        polygon_objects_grid_coords=positive_objects_grid_coords,
        num_panel_rows=positive_mask_matrix.shape[0],
        num_panel_columns=positive_mask_matrix.shape[1],
        panel_row_by_polygon=positive_panel_row_by_polygon,
        panel_column_by_polygon=positive_panel_column_by_polygon)

    if negative_objects_grid_coords is None:
        negative_objects_grid_coords = []
        negative_panel_row_by_polygon = numpy.array([], dtype=int)
        negative_panel_column_by_polygon = numpy.array([], dtype=int)
        negative_mask_matrix = numpy.full(
            positive_mask_matrix.shape, False, dtype=bool)

    error_checking.assert_is_boolean_numpy_array(negative_mask_matrix)
    error_checking.assert_is_numpy_array(
        negative_mask_matrix,
        exact_dimensions=numpy.array(positive_mask_matrix.shape, dtype=int)
    )

    _check_polygons(
        polygon_objects_grid_coords=negative_objects_grid_coords,
        num_panel_rows=negative_mask_matrix.shape[0],
        num_panel_columns=negative_mask_matrix.shape[1],
        panel_row_by_polygon=negative_panel_row_by_polygon,
        panel_column_by_polygon=negative_panel_column_by_polygon)

    (positive_vertex_rows, positive_vertex_columns, these_vertex_to_poly_indices
    ) = _polygon_list_to_vertex_list(positive_objects_grid_coords)

    positive_panel_row_by_vertex = numpy.array([
        positive_panel_row_by_polygon[k] if k >= 0 else numpy.nan
        for k in these_vertex_to_poly_indices
    ])

    positive_panel_column_by_vertex = numpy.array([
        positive_panel_column_by_polygon[k] if k >= 0 else numpy.nan
        for k in these_vertex_to_poly_indices
    ])

    if len(positive_vertex_rows) == 0:
        positive_vertex_rows = numpy.full(1, SENTINEL_VALUE - 1)
        positive_vertex_columns = numpy.full(1, SENTINEL_VALUE - 1)
        positive_panel_row_by_vertex = numpy.full(1, -1, dtype=int)
        positive_panel_column_by_vertex = numpy.full(1, -1, dtype=int)

    (negative_vertex_rows, negative_vertex_columns, these_vertex_to_poly_indices
    ) = _polygon_list_to_vertex_list(negative_objects_grid_coords)

    negative_panel_row_by_vertex = numpy.array([
        negative_panel_row_by_polygon[k] if k >= 0 else numpy.nan
        for k in these_vertex_to_poly_indices
    ])

    negative_panel_column_by_vertex = numpy.array([
        negative_panel_column_by_polygon[k] if k >= 0 else numpy.nan
        for k in these_vertex_to_poly_indices
    ])

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)
    dataset_object = netCDF4.Dataset(
        output_file_name, 'w', format='NETCDF3_64BIT_OFFSET')

    dataset_object.setncattr(STORM_ID_KEY, full_storm_id_string)
    dataset_object.setncattr(STORM_TIME_KEY, storm_time_unix_sec)

    dataset_object.createDimension(
        PANEL_ROW_DIMENSION_KEY, positive_mask_matrix.shape[0]
    )
    dataset_object.createDimension(
        PANEL_COLUMN_DIMENSION_KEY, positive_mask_matrix.shape[1]
    )
    dataset_object.createDimension(
        GRID_ROW_DIMENSION_KEY, positive_mask_matrix.shape[2]
    )
    dataset_object.createDimension(
        GRID_COLUMN_DIMENSION_KEY, positive_mask_matrix.shape[3]
    )
    dataset_object.createDimension(
        POSITIVE_VERTEX_DIM_KEY, len(positive_vertex_rows)
    )
    dataset_object.createDimension(
        NEGATIVE_VERTEX_DIM_KEY, len(negative_vertex_rows)
    )

    dataset_object.createVariable(
        POSITIVE_VERTEX_ROWS_KEY, datatype=numpy.float32,
        dimensions=POSITIVE_VERTEX_DIM_KEY
    )
    dataset_object.variables[POSITIVE_VERTEX_ROWS_KEY][:] = positive_vertex_rows

    dataset_object.createVariable(
        POSITIVE_VERTEX_COLUMNS_KEY, datatype=numpy.float32,
        dimensions=POSITIVE_VERTEX_DIM_KEY
    )
    dataset_object.variables[POSITIVE_VERTEX_COLUMNS_KEY][:] = (
        positive_vertex_columns
    )

    dataset_object.createVariable(
        POSITIVE_PANEL_ROW_BY_VERTEX_KEY, datatype=numpy.int32,
        dimensions=POSITIVE_VERTEX_DIM_KEY
    )
    dataset_object.variables[
        POSITIVE_PANEL_ROW_BY_VERTEX_KEY][:] = positive_panel_row_by_vertex

    dataset_object.createVariable(
        POSITIVE_PANEL_COLUMN_BY_VERTEX_KEY, datatype=numpy.int32,
        dimensions=POSITIVE_VERTEX_DIM_KEY
    )
    dataset_object.variables[
        POSITIVE_PANEL_COLUMN_BY_VERTEX_KEY
    ][:] = positive_panel_column_by_vertex

    dataset_object.createVariable(
        NEGATIVE_VERTEX_ROWS_KEY, datatype=numpy.float32,
        dimensions=NEGATIVE_VERTEX_DIM_KEY
    )
    dataset_object.variables[NEGATIVE_VERTEX_ROWS_KEY][:] = negative_vertex_rows

    dataset_object.createVariable(
        NEGATIVE_VERTEX_COLUMNS_KEY, datatype=numpy.float32,
        dimensions=NEGATIVE_VERTEX_DIM_KEY
    )
    dataset_object.variables[NEGATIVE_VERTEX_COLUMNS_KEY][:] = (
        negative_vertex_columns
    )

    dataset_object.createVariable(
        NEGATIVE_PANEL_ROW_BY_VERTEX_KEY, datatype=numpy.int32,
        dimensions=NEGATIVE_VERTEX_DIM_KEY
    )
    dataset_object.variables[
        NEGATIVE_PANEL_ROW_BY_VERTEX_KEY][:] = negative_panel_row_by_vertex

    dataset_object.createVariable(
        NEGATIVE_PANEL_COLUMN_BY_VERTEX_KEY, datatype=numpy.int32,
        dimensions=NEGATIVE_VERTEX_DIM_KEY
    )
    dataset_object.variables[
        NEGATIVE_PANEL_COLUMN_BY_VERTEX_KEY
    ][:] = negative_panel_column_by_vertex

    dataset_object.createVariable(
        POSITIVE_MASK_MATRIX_KEY, datatype=numpy.int32,
        dimensions=(PANEL_ROW_DIMENSION_KEY, PANEL_COLUMN_DIMENSION_KEY,
                    GRID_ROW_DIMENSION_KEY, GRID_COLUMN_DIMENSION_KEY)
    )
    dataset_object.variables[POSITIVE_MASK_MATRIX_KEY][:] = (
        positive_mask_matrix.astype(int)
    )

    dataset_object.createVariable(
        NEGATIVE_MASK_MATRIX_KEY, datatype=numpy.int32,
        dimensions=(PANEL_ROW_DIMENSION_KEY, PANEL_COLUMN_DIMENSION_KEY,
                    GRID_ROW_DIMENSION_KEY, GRID_COLUMN_DIMENSION_KEY)
    )
    dataset_object.variables[NEGATIVE_MASK_MATRIX_KEY][:] = (
        negative_mask_matrix.astype(int)
    )

    dataset_object.close()


def read_polygons(netcdf_file_name):
    """Reads human polygons for one image from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: polygon_dict: Dictionary with the following keys.
    polygon_dict['full_storm_id_string']: See input doc for `write_polygons`.
    polygon_dict['storm_time_unix_sec']: Same.
    polygon_dict['positive_objects_grid_coords']: Same.
    polygon_dict['positive_panel_row_by_polygon']: Same.
    polygon_dict['positive_panel_column_by_polygon']: Same.
    polygon_dict['positive_mask_matrix']: Same.
    polygon_dict['negative_objects_grid_coords']: Same.
    polygon_dict['negative_panel_row_by_polygon']: Same.
    polygon_dict['negative_panel_column_by_polygon']: Same.
    polygon_dict['negative_mask_matrix']: Same.
    """

    error_checking.assert_file_exists(netcdf_file_name)
    dataset_object = netCDF4.Dataset(netcdf_file_name)

    polygon_dict = {
        STORM_ID_KEY: str(getattr(dataset_object, STORM_ID_KEY)),
        STORM_TIME_KEY: int(numpy.round(
            getattr(dataset_object, STORM_TIME_KEY)
        )),
        POSITIVE_MASK_MATRIX_KEY: numpy.array(
            dataset_object.variables[POSITIVE_MASK_MATRIX_KEY][:], dtype=bool
        ),
        NEGATIVE_MASK_MATRIX_KEY: numpy.array(
            dataset_object.variables[NEGATIVE_MASK_MATRIX_KEY][:], dtype=bool
        )
    }

    if polygon_dict[STORM_ID_KEY] == DUMMY_STORM_ID_STRING:
        polygon_dict[STORM_ID_KEY] = None
        polygon_dict[STORM_TIME_KEY] = None

    positive_vertex_rows = numpy.array(
        dataset_object.variables[POSITIVE_VERTEX_ROWS_KEY][:], dtype=float
    )
    positive_vertex_columns = numpy.array(
        dataset_object.variables[POSITIVE_VERTEX_COLUMNS_KEY][:], dtype=float
    )

    if positive_vertex_rows[0] <= SENTINEL_VALUE:
        positive_objects_grid_coords = []
        positive_panel_row_by_polygon = numpy.array([], dtype=int)
        positive_panel_column_by_polygon = numpy.array([], dtype=int)
    else:
        positive_objects_grid_coords, these_poly_to_first_vertex_indices = (
            _vertex_list_to_polygon_list(
                vertex_rows=positive_vertex_rows,
                vertex_columns=positive_vertex_columns)
        )

        positive_panel_row_by_vertex = numpy.array(
            dataset_object.variables[POSITIVE_PANEL_ROW_BY_VERTEX_KEY][:],
            dtype=int
        )

        positive_panel_row_by_polygon = positive_panel_row_by_vertex[
            these_poly_to_first_vertex_indices]

        positive_panel_column_by_vertex = numpy.array(
            dataset_object.variables[POSITIVE_PANEL_COLUMN_BY_VERTEX_KEY][:],
            dtype=int
        )

        positive_panel_column_by_polygon = positive_panel_column_by_vertex[
            these_poly_to_first_vertex_indices]

    polygon_dict[POSITIVE_PANEL_ROW_BY_POLY_KEY] = positive_panel_row_by_polygon
    polygon_dict[
        POSITIVE_PANEL_COLUMN_BY_POLY_KEY] = positive_panel_column_by_polygon

    (negative_objects_grid_coords, these_poly_to_first_vertex_indices
    ) = _vertex_list_to_polygon_list(
        vertex_rows=numpy.array(
            dataset_object.variables[NEGATIVE_VERTEX_ROWS_KEY][:], dtype=float
        ),
        vertex_columns=numpy.array(
            dataset_object.variables[NEGATIVE_VERTEX_COLUMNS_KEY][:],
            dtype=float
        )
    )

    negative_panel_row_by_vertex = numpy.array(
        dataset_object.variables[NEGATIVE_PANEL_ROW_BY_VERTEX_KEY][:], dtype=int
    )

    negative_panel_row_by_polygon = negative_panel_row_by_vertex[
        these_poly_to_first_vertex_indices]

    negative_panel_column_by_vertex = numpy.array(
        dataset_object.variables[NEGATIVE_PANEL_COLUMN_BY_VERTEX_KEY][:],
        dtype=int
    )

    negative_panel_column_by_polygon = negative_panel_column_by_vertex[
        these_poly_to_first_vertex_indices]

    polygon_dict[NEGATIVE_PANEL_ROW_BY_POLY_KEY] = negative_panel_row_by_polygon
    polygon_dict[
        NEGATIVE_PANEL_COLUMN_BY_POLY_KEY] = negative_panel_column_by_polygon

    dataset_object.close()

    polygon_dict[POSITIVE_POLYGON_OBJECTS_KEY] = positive_objects_grid_coords
    polygon_dict[NEGATIVE_POLYGON_OBJECTS_KEY] = negative_objects_grid_coords
    return polygon_dict


def write_points(
        output_file_name, grid_row_by_point, grid_column_by_point,
        panel_row_by_point, panel_column_by_point, full_storm_id_string=None,
        storm_time_unix_sec=None):
    """Writes human points of interest for one image to NetCDF file.

    K = number of points of interest

    :param output_file_name: Path to output (NetCDF) file.
    :param grid_row_by_point: length-K numpy array of row indices in data grid
        (floats).
    :param grid_column_by_point: length-K numpy array of column indices in data
        grid (floats).
    :param panel_row_by_point: length-K numpy array of row indices in panel grid
        (integers).
    :param panel_column_by_point: length-K numpy array of column indices in
        panel grid (integers).
    :param full_storm_id_string: See doc for `write_polygons`.
    :param storm_time_unix_sec: Same.
    """

    is_composite = (
        full_storm_id_string is None and storm_time_unix_sec is None
    )

    if is_composite:
        full_storm_id_string = DUMMY_STORM_ID_STRING
        storm_time_unix_sec = -1

    error_checking.assert_is_string(full_storm_id_string)
    error_checking.assert_is_integer(storm_time_unix_sec)

    # error_checking.assert_is_integer(num_grid_rows)
    # error_checking.assert_is_greater(num_grid_rows, 0)
    # error_checking.assert_is_integer(num_grid_columns)
    # error_checking.assert_is_greater(num_grid_columns, 0)

    error_checking.assert_is_numpy_array(grid_row_by_point, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(grid_row_by_point, -0.5)
    # error_checking.assert_is_leq_numpy_array(
    #     grid_row_by_point, num_grid_rows - 0.5)

    num_points = len(grid_row_by_point)
    these_expected_dim = numpy.array([num_points], dtype=int)

    error_checking.assert_is_numpy_array(
        grid_column_by_point, exact_dimensions=these_expected_dim)
    error_checking.assert_is_geq_numpy_array(grid_column_by_point, -0.5)
    # error_checking.assert_is_leq_numpy_array(
    #     grid_column_by_point, num_grid_columns - 0.5)

    error_checking.assert_is_numpy_array(
        panel_row_by_point, exact_dimensions=these_expected_dim)
    error_checking.assert_is_integer_numpy_array(panel_row_by_point)
    error_checking.assert_is_geq_numpy_array(panel_row_by_point, 0)

    error_checking.assert_is_numpy_array(
        panel_column_by_point, exact_dimensions=these_expected_dim)
    error_checking.assert_is_integer_numpy_array(panel_column_by_point)
    error_checking.assert_is_geq_numpy_array(panel_column_by_point, 0)

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)
    dataset_object = netCDF4.Dataset(
        output_file_name, 'w', format='NETCDF3_64BIT_OFFSET')

    dataset_object.setncattr(STORM_ID_KEY, full_storm_id_string)
    dataset_object.setncattr(STORM_TIME_KEY, storm_time_unix_sec)
    dataset_object.createDimension(POINT_DIMENSION_KEY, num_points)

    dataset_object.createVariable(
        GRID_ROW_BY_POINT_KEY, datatype=numpy.float32,
        dimensions=POINT_DIMENSION_KEY
    )
    dataset_object.variables[GRID_ROW_BY_POINT_KEY][:] = grid_row_by_point

    dataset_object.createVariable(
        GRID_COLUMN_BY_POINT_KEY, datatype=numpy.float32,
        dimensions=POINT_DIMENSION_KEY
    )
    dataset_object.variables[GRID_COLUMN_BY_POINT_KEY][:] = grid_column_by_point

    dataset_object.createVariable(
        PANEL_ROW_BY_POINT_KEY, datatype=numpy.int32,
        dimensions=POINT_DIMENSION_KEY
    )
    dataset_object.variables[PANEL_ROW_BY_POINT_KEY][:] = panel_row_by_point

    dataset_object.createVariable(
        PANEL_COLUMN_BY_POINT_KEY, datatype=numpy.int32,
        dimensions=POINT_DIMENSION_KEY
    )
    dataset_object.variables[
        PANEL_COLUMN_BY_POINT_KEY][:] = panel_column_by_point

    dataset_object.close()


def read_points(netcdf_file_name):
    """Reads human points of interest for one image from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: point_dict: Dictionary with the following keys.
    point_dict['full_storm_id_string']: See input doc for `write_points`.
    point_dict['storm_time_unix_sec']: Same.
    point_dict['grid_row_by_point']: Same.
    point_dict['grid_column_by_point']: Same.
    point_dict['panel_row_by_point']: Same.
    point_dict['panel_column_by_point']: Same.
    """

    error_checking.assert_file_exists(netcdf_file_name)
    dataset_object = netCDF4.Dataset(netcdf_file_name)

    point_dict = {
        STORM_ID_KEY: str(getattr(dataset_object, STORM_ID_KEY)),
        STORM_TIME_KEY: int(numpy.round(
            getattr(dataset_object, STORM_TIME_KEY)
        )),
        GRID_ROW_BY_POINT_KEY: numpy.array(
            dataset_object.variables[GRID_ROW_BY_POINT_KEY][:], dtype=float
        ),
        GRID_COLUMN_BY_POINT_KEY: numpy.array(
            dataset_object.variables[GRID_COLUMN_BY_POINT_KEY][:], dtype=float
        ),
        PANEL_ROW_BY_POINT_KEY: numpy.array(
            dataset_object.variables[PANEL_ROW_BY_POINT_KEY][:], dtype=int
        ),
        PANEL_COLUMN_BY_POINT_KEY: numpy.array(
            dataset_object.variables[PANEL_COLUMN_BY_POINT_KEY][:], dtype=int
        )
    }

    dataset_object.close()

    if point_dict[STORM_ID_KEY] == DUMMY_STORM_ID_STRING:
        point_dict[STORM_ID_KEY] = None
        point_dict[STORM_TIME_KEY] = None

    return point_dict
