"""Thresholds Grad-CAM output to create regions of interest (polygons)."""

import sys
import copy
import argparse
import numpy
from skimage.measure import label as label_image
from shapely.ops import cascaded_union
from gewittergefahr.deep_learning import gradcam
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import error_checking

INPUT_FILE_ARG_NAME = 'input_gradcam_file_name'
PERCENTILE_ARG_NAME = 'percentile_threshold'
MIN_ACTIVATION_ARG_NAME = 'min_class_activation'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing class-activation maps.  Will be read by '
    '`gradcam.read_standard_file` or `gradcam.read_pmm_file`.')

PERCENTILE_HELP_STRING = (
    'Percentile threshold.  For each class-activation map in `{0:s}`, this '
    'script will create regions of interest (polygons).  The polygons will '
    'outline all grid cells where class activation >= [q]th percentile '
    '(q = `{1:s}`) and >= `{2:s}`.  The percentile will be computed separately '
    'for each example (each map).'
).format(INPUT_FILE_ARG_NAME, PERCENTILE_ARG_NAME, MIN_ACTIVATION_ARG_NAME)

MIN_ACTIVATION_HELP_STRING = 'See documentation for `{0:s}`.'.format(
    PERCENTILE_ARG_NAME)

OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Regions of interest (polygons) will be written here '
    'by `gradcam.read_standard_file` or `gradcam.read_pmm_file`.  To make '
    'output file = input file, leave this empty.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PERCENTILE_ARG_NAME, type=float, required=False, default=90.,
    help=PERCENTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_ACTIVATION_ARG_NAME, type=float, required=False, default=0.001,
    help=MIN_ACTIVATION_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=False, default='',
    help=OUTPUT_FILE_HELP_STRING)


def _grid_cell_to_polygon(grid_row, grid_column):
    """Converts grid cell from single point to polygon.

    :param grid_row: Row index.
    :param grid_column: Column index.
    :return: polygon_object: Instance of `shapely.geometry.Polygon`.
    """

    vertex_rows = grid_row + numpy.array([-0.5, -0.5, 0.5, 0.5, -0.5])
    vertex_columns = grid_column + numpy.array([-0.5, 0.5, 0.5, -0.5, -0.5])

    return polygons.vertex_arrays_to_polygon_object(
        exterior_x_coords=vertex_columns, exterior_y_coords=vertex_rows)


def _grid_cells_to_polygon(grid_rows, grid_columns):
    """Converts list of grid cells to polygon.

    G = number of grid cells in polygon

    :param grid_rows: length-G numpy array of rows.
    :param grid_columns: length-G numpy array of columns.
    :return: polygon_object: Instance of `shapely.geometry.Polygon`.
    """

    list_of_polygon_objects = [
        _grid_cell_to_polygon(r, c) for r, c in zip(grid_rows, grid_columns)
    ]

    return cascaded_union(list_of_polygon_objects)


def _mask_to_polygons(mask_matrix):
    """Turns mask into polygons.

    M = number of rows in grid
    N = number of columns in grid

    :param mask_matrix: M-by-N numpy array of Boolean flags.  This method will
        create polygons that encompass all grid cells labeled True.
    :return: list_of_polygon_objects: 1-D list of polygons (instances of
        `shapely.geometry.Polygon`).
    """

    if not numpy.any(mask_matrix):
        return []

    region_id_matrix = label_image(mask_matrix, connectivity=1)
    num_regions = numpy.max(region_id_matrix)
    list_of_polygon_objects = [None] * num_regions

    for i in range(num_regions):
        these_grid_rows, these_grid_columns = numpy.where(
            region_id_matrix == i + 1)

        list_of_polygon_objects[i] = _grid_cells_to_polygon(
            grid_rows=these_grid_rows, grid_columns=these_grid_columns)

        # pyplot.plot(
        #     numpy.array(list_of_polygon_objects[i].exterior.xy[0]),
        #     numpy.array(list_of_polygon_objects[i].exterior.xy[1]),
        #     'k-'
        # )

    # pyplot.show()

    return list_of_polygon_objects


def _run(input_gradcam_file_name, percentile_threshold, min_class_activation,
         output_file_name):
    """Thresholds Grad-CAM output to create regions of interest (polygons).

    This is effectively the main method.

    :param input_gradcam_file_name: See documentation at top of file.
    :param percentile_threshold: Same.
    :param min_class_activation: Same.
    :param output_file_name: Same.
    :raises: TypeError: if any class-activation map contains not-2 spatial
        dimensions.
    """

    error_checking.assert_is_geq(percentile_threshold, 50.)
    error_checking.assert_is_less_than(percentile_threshold, 100.)
    error_checking.assert_is_greater(min_class_activation, 0.)

    print('Reading data from: "{0:s}"...\n'.format(input_gradcam_file_name))
    pmm_flag = False

    try:
        gradcam_dict = gradcam.read_standard_file(input_gradcam_file_name)
        list_of_cam_matrices = gradcam_dict.pop(gradcam.CAM_MATRICES_KEY)
    except ValueError:
        gradcam_dict = gradcam.read_pmm_file(input_gradcam_file_name)
        list_of_cam_matrices = gradcam_dict.pop(gradcam.MEAN_CAM_MATRICES_KEY)

        for j in range(len(list_of_cam_matrices)):
            if list_of_cam_matrices[j] is None:
                continue

            list_of_cam_matrices[j] = numpy.expand_dims(
                list_of_cam_matrices[j], axis=0
            )

        pmm_flag = True

    num_matrices = len(list_of_cam_matrices)
    num_examples = None

    for j in range(num_matrices):
        if list_of_cam_matrices[j] is None:
            continue

        num_examples = list_of_cam_matrices[j].shape[0]
        this_num_spatial_dim = len(list_of_cam_matrices[j].shape) - 1
        if this_num_spatial_dim == 2:
            continue

        error_string = (
            'This script deals with only 2-D class-activation maps.  {0:d}th '
            'input matrix contains {1:d} spatial dimensions.'
        ).format(j + 1, this_num_spatial_dim)

        raise TypeError(error_string)

    list_of_mask_matrices = [None] * num_matrices
    list_of_polygon_objects = [[[] * 0] * num_examples] * num_matrices

    for i in range(num_examples):
        for j in range(num_matrices):
            if list_of_cam_matrices[j] is None:
                continue

            this_min_class_activation = numpy.percentile(
                list_of_cam_matrices[j][i, ...], percentile_threshold
            )

            this_min_class_activation = max([
                this_min_class_activation, min_class_activation
            ])

            print((
                'Creating mask for {0:d}th example and {1:d}th class-activation'
                ' matrix, with threshold = {2:.3e}...'
            ).format(
                i + 1, j + 1, this_min_class_activation
            ))

            this_mask_matrix = (
                list_of_cam_matrices[j][i, ...] >= this_min_class_activation
            )

            numpy.set_printoptions(threshold=sys.maxsize)
            print(this_mask_matrix.astype(int))

            print('{0:d} of {1:d} grid points are inside mask.\n'.format(
                numpy.sum(this_mask_matrix.astype(int)), this_mask_matrix.size
            ))

            list_of_polygon_objects[j][i] = _mask_to_polygons(this_mask_matrix)
            this_mask_matrix = numpy.expand_dims(this_mask_matrix, axis=0)

            if list_of_mask_matrices[j] is None:
                list_of_mask_matrices[j] = copy.deepcopy(this_mask_matrix)
            else:
                list_of_mask_matrices[j] = numpy.concatenate(
                    (list_of_mask_matrices[j], this_mask_matrix), axis=0
                )

    if pmm_flag:
        for j in range(len(list_of_mask_matrices)):
            if list_of_mask_matrices[j] is None:
                continue

            list_of_mask_matrices[j] = list_of_mask_matrices[j][0, ...]

    region_dict = {
        gradcam.MASK_MATRICES_KEY: list_of_mask_matrices,
        gradcam.POLYGON_OBJECTS_KEY: list_of_polygon_objects,
        gradcam.PERCENTILE_THRESHOLD_KEY: percentile_threshold,
        gradcam.MIN_CLASS_ACTIVATION_KEY: min_class_activation
    }

    if output_file_name in ['', 'None']:
        output_file_name = input_gradcam_file_name

    print('Writing regions of interest to: "{0:s}"...'.format(output_file_name))
    gradcam.add_regions_to_file(
        input_file_name=input_gradcam_file_name,
        output_file_name=output_file_name, region_dict=region_dict)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_gradcam_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        percentile_threshold=getattr(INPUT_ARG_OBJECT, PERCENTILE_ARG_NAME),
        min_class_activation=getattr(INPUT_ARG_OBJECT, MIN_ACTIVATION_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
