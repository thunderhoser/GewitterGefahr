"""Handles polygons drawn interactively by a human."""

import numpy
import matplotlib.pyplot as pyplot
import netCDF4
from PIL import Image
from roipoly import MultiRoi
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import general_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

IMAGE_FILE_KEY = 'orig_image_file_name'
POSITIVE_VERTEX_ROWS_KEY = 'positive_vertex_rows'
POSITIVE_VERTEX_COLUMNS_KEY = 'positive_vertex_columns'
NEGATIVE_VERTEX_ROWS_KEY = 'negative_vertex_rows'
NEGATIVE_VERTEX_COLUMNS_KEY = 'negative_vertex_columns'
POSITIVE_MASK_MATRIX_KEY = 'positive_mask_matrix'
POSITIVE_POLYGON_OBJECTS_KEY = 'positive_polygon_objects_rowcol'
NEGATIVE_MASK_MATRIX_KEY = 'negative_mask_matrix'
NEGATIVE_POLYGON_OBJECTS_KEY = 'negative_polygon_objects_rowcol'

ROW_DIMENSION_KEY = 'grid_row'
COLUMN_DIMENSION_KEY = 'grid_column'
POSITIVE_VERTEX_DIM_KEY = 'positive_polygon_vertex'
NEGATIVE_VERTEX_DIM_KEY = 'negative_polygon_vertex'


def _polygon_list_to_vertex_list(polygon_objects_grid_coords):
    """Converts list of polygons to list of vertices.

    V = total number of vertices

    :param polygon_objects_grid_coords: List of polygons created by
        `polygons_from_pixel_to_grid_coords`.
    :return: vertex_rows: length-V numpy array of row coordinates.
    :return: vertex_columns: length-V numpy array of column coordinates.
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

    for i in range(num_polygons):
        vertex_rows += polygon_objects_grid_coords[i].exterior.xy[1]
        vertex_columns += polygon_objects_grid_coords[i].exterior.xy[0]

        if i == num_polygons - 1:
            continue

        vertex_rows += [numpy.nan]
        vertex_columns += [numpy.nan]

    return numpy.array(vertex_rows), numpy.array(vertex_columns)


def _vertex_list_to_polygon_list(vertex_rows, vertex_columns):
    """This method is the inverse of `_polygon_list_to_vertex_list`.

    :param vertex_rows: See doc for `_polygon_list_to_vertex_list`.
    :param vertex_columns: Same.
    :return: polygon_objects_grid_coords: Same.
    """

    if len(vertex_rows) == 0:
        return []

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

    return polygon_objects_grid_coords


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


def capture_polygons(image_file_name, instruction_string=''):
    """This interactiv method allows you to draw polygons and captures vertices.

    N = number of polygons drawn

    :param image_file_name: Path to image file.  This method will display the
        image in a figure window and allow you to draw polygons on top.
    :param instruction_string: String with instructions for the user.
    :return: polygon_objects_pixel_coords: length-N list of polygons (instances of
        `shapely.geometry.Polygon`), each containing vertices in pixel
        coordinates.
    :return: num_pixel_rows: Number of pixel rows in the image.
    :return: num_pixel_columns: Number of pixel columns in the image.
    """

    error_checking.assert_file_exists(image_file_name)

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

    error_checking.assert_is_integer(num_grid_rows)
    error_checking.assert_is_greater(num_grid_rows, 0)
    error_checking.assert_is_integer(num_grid_columns)
    error_checking.assert_is_greater(num_grid_columns, 0)
    error_checking.assert_is_integer(num_pixel_rows)
    error_checking.assert_is_greater(num_pixel_rows, 0)
    error_checking.assert_is_integer(num_pixel_columns)
    error_checking.assert_is_greater(num_pixel_columns, 0)
    error_checking.assert_is_integer(num_panel_rows)
    error_checking.assert_is_greater(num_panel_rows, 0)
    error_checking.assert_is_integer(num_panel_columns)
    error_checking.assert_is_greater(num_panel_columns, 0)
    error_checking.assert_is_list(polygon_objects_pixel_coords)

    num_polygons = len(polygon_objects_pixel_coords)
    if num_polygons == 0:
        empty_array = numpy.array([], dtype=int)
        return polygon_objects_pixel_coords, empty_array, empty_array

    error_checking.assert_is_numpy_array(
        numpy.array(polygon_objects_pixel_coords, dtype=object),
        num_dimensions=1
    )

    panel_row_to_first_px_row = {}
    for i in range(num_panel_rows):
        panel_row_to_first_px_row[i] = (
            i * float(num_pixel_rows) / num_panel_rows
        )

    panel_column_to_first_px_column = {}
    for j in range(num_panel_columns):
        panel_column_to_first_px_column[j] = (
            j * float(num_pixel_columns) / num_panel_columns
        )

    polygon_objects_grid_coords = [None] * num_polygons
    panel_row_by_polygon = [None] * num_polygons
    panel_column_by_polygon = [None] * num_polygons

    for k in range(num_polygons):
        # TODO(thunderhoser): Modularize this shit.
        these_pixel_columns = 0.5 + numpy.array(
            polygon_objects_pixel_coords[k].exterior.xy[0]
        )

        error_checking.assert_is_geq_numpy_array(these_pixel_columns, 0.)
        error_checking.assert_is_leq_numpy_array(
            these_pixel_columns, num_pixel_columns)

        these_panel_columns = numpy.floor(
            these_pixel_columns * float(num_panel_columns) / num_pixel_columns
        ).astype(int)

        these_panel_columns[
            these_panel_columns == num_panel_columns
        ] = num_panel_columns - 1

        if len(numpy.unique(these_panel_columns)) > 1:
            error_string = (
                'The {0:d}th polygon is in multiple panels.  Panel columns '
                'listed below.\n{1:s}'
            ).format(k + 1, str(these_panel_columns))

            raise ValueError(error_string)

        panel_column_by_polygon[k] = these_panel_columns[0]
        these_pixel_columns = (
            these_pixel_columns -
            panel_column_to_first_px_column[panel_column_by_polygon[k]]
        )

        these_grid_columns = -0.5 + (
            these_pixel_columns *
            float(num_grid_columns * num_panel_columns) / num_pixel_columns
        )

        these_pixel_rows = 0.5 + numpy.array(
            polygon_objects_pixel_coords[k].exterior.xy[1]
        )

        error_checking.assert_is_geq_numpy_array(these_pixel_rows, 0.)
        error_checking.assert_is_leq_numpy_array(
            these_pixel_rows, num_pixel_rows)

        these_panel_rows = numpy.floor(
            these_pixel_rows * float(num_panel_rows) / num_pixel_rows
        ).astype(int)

        these_panel_rows[
            these_panel_rows == num_panel_rows
        ] = num_panel_rows - 1

        if len(numpy.unique(these_panel_rows)) > 1:
            error_string = (
                'The {0:d}th polygon is in multiple panels.  Panel rows '
                'listed below.\n{1:s}'
            ).format(k + 1, str(these_panel_rows))

            raise ValueError(error_string)

        panel_row_by_polygon[k] = these_panel_rows[0]
        these_pixel_rows = (
            these_pixel_rows -
            panel_row_to_first_px_row[panel_row_by_polygon[k]]
        )

        these_grid_rows = -0.5 + (
            these_pixel_rows *
            float(num_grid_rows * num_panel_rows) / num_pixel_rows
        )

        these_grid_rows = num_grid_rows - 1 - these_grid_rows

        this_polygon_object = polygons.vertex_arrays_to_polygon_object(
            exterior_x_coords=these_grid_columns,
            exterior_y_coords=these_grid_rows)

        polygon_objects_grid_coords[k] = this_polygon_object

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
    error_checking.assert_is_integer(num_panel_rows)
    error_checking.assert_is_greater(num_panel_rows, 0)
    error_checking.assert_is_integer(num_panel_columns)
    error_checking.assert_is_greater(num_panel_columns, 0)

    mask_matrix = numpy.full(
        (num_panel_rows, num_panel_columns, num_grid_rows, num_grid_columns),
        False, dtype=bool
    )

    num_polygons = len(polygon_objects_grid_coords)
    if num_polygons == 0:
        return mask_matrix

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
        output_file_name, orig_image_file_name, positive_polygon_objects_rowcol,
        positive_mask_matrix, negative_polygon_objects_rowcol,
        negative_mask_matrix):
    """Writes human polygons for one image to NetCDF file.

    :param output_file_name: Path to output (NetCDF) file.
    :param orig_image_file_name: Path to original image file (over which the
        polygons were drawn).
    :param positive_polygon_objects_rowcol: List of polygons created by
        `polygons_from_pixel_to_grid_coords`, containing positive regions of
        interest.
    :param positive_mask_matrix: Mask matrix created by `polygons_to_mask`,
        corresponding to `positive_polygon_objects_rowcol`.
    :param negative_polygon_objects_rowcol: List of polygons created by
        `polygons_from_pixel_to_grid_coords`, containing negative regions of
        interest.
    :param negative_mask_matrix: Mask matrix created by `polygons_to_mask`,
        corresponding to `negative_polygon_objects_rowcol`.
    """

    error_checking.assert_is_string(orig_image_file_name)
    error_checking.assert_is_boolean_numpy_array(positive_mask_matrix)
    error_checking.assert_is_numpy_array(positive_mask_matrix, num_dimensions=2)

    error_checking.assert_is_boolean_numpy_array(negative_mask_matrix)
    error_checking.assert_is_numpy_array(
        negative_mask_matrix,
        exact_dimensions=numpy.array(positive_mask_matrix.shape, dtype=int)
    )

    positive_vertex_rows, positive_vertex_columns = (
        _polygon_list_to_vertex_list(positive_polygon_objects_rowcol)
    )

    negative_vertex_rows, negative_vertex_columns = (
        _polygon_list_to_vertex_list(negative_polygon_objects_rowcol)
    )

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)
    dataset_object = netCDF4.Dataset(
        output_file_name, 'w', format='NETCDF3_64BIT_OFFSET')

    dataset_object.setncattr(IMAGE_FILE_KEY, orig_image_file_name)

    dataset_object.createDimension(
        ROW_DIMENSION_KEY, positive_mask_matrix.shape[0]
    )
    dataset_object.createDimension(
        COLUMN_DIMENSION_KEY, positive_mask_matrix.shape[1]
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
        POSITIVE_MASK_MATRIX_KEY, datatype=numpy.int32,
        dimensions=(ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY)
    )
    dataset_object.variables[POSITIVE_MASK_MATRIX_KEY][:] = (
        positive_mask_matrix.astype(int)
    )

    dataset_object.createVariable(
        NEGATIVE_MASK_MATRIX_KEY, datatype=numpy.int32,
        dimensions=(ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY)
    )
    dataset_object.variables[NEGATIVE_MASK_MATRIX_KEY][:] = (
        negative_mask_matrix.astype(int)
    )

    dataset_object.close()


def read_polygons(netcdf_file_name):
    """Reads human polygons for one image from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: polygon_dict: Dictionary with the following keys.
    polygon_dict['orig_image_file_name']: See input doc for `write_polygons`.
    polygon_dict['positive_polygon_objects_rowcol']: Same.
    polygon_dict['positive_mask_matrix']: Same.
    polygon_dict['negative_polygon_objects_rowcol']: Same.
    polygon_dict['negative_mask_matrix']: Same.
    """

    error_checking.assert_file_exists(netcdf_file_name)
    dataset_object = netCDF4.Dataset(netcdf_file_name)

    polygon_dict = {
        IMAGE_FILE_KEY: str(getattr(dataset_object, IMAGE_FILE_KEY)),
        POSITIVE_MASK_MATRIX_KEY: numpy.array(
            dataset_object.variables[POSITIVE_MASK_MATRIX_KEY][:], dtype=bool
        ),
        NEGATIVE_MASK_MATRIX_KEY: numpy.array(
            dataset_object.variables[NEGATIVE_MASK_MATRIX_KEY][:], dtype=bool
        )
    }

    positive_polygon_objects_rowcol = _vertex_list_to_polygon_list(
        vertex_rows=numpy.array(
            dataset_object.variables[POSITIVE_VERTEX_ROWS_KEY][:], dtype=float
        ),
        vertex_columns=numpy.array(
            dataset_object.variables[POSITIVE_VERTEX_COLUMNS_KEY][:],
            dtype=float
        )
    )

    negative_polygon_objects_rowcol = _vertex_list_to_polygon_list(
        vertex_rows=numpy.array(
            dataset_object.variables[NEGATIVE_VERTEX_ROWS_KEY][:], dtype=float
        ),
        vertex_columns=numpy.array(
            dataset_object.variables[NEGATIVE_VERTEX_COLUMNS_KEY][:],
            dtype=float
        )
    )

    dataset_object.close()

    polygon_dict[POSITIVE_POLYGON_OBJECTS_KEY] = positive_polygon_objects_rowcol
    polygon_dict[NEGATIVE_POLYGON_OBJECTS_KEY] = negative_polygon_objects_rowcol
    return polygon_dict
