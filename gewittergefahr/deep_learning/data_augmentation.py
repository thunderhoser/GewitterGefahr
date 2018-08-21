"""Augments image dataset by shifting, rotating, or adding Gaussian noise."""

import numpy
from scipy.ndimage.interpolation import rotate as scipy_rotate
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils

PADDING_VALUE = 0.

# TODO(thunderhoser): This module still needs unit tests.  I am hoping to do
# this in Boulder.


def shift_radar_images(radar_image_matrix, x_offset_pixels, y_offset_pixels):
    """Shifts each radar image in the x- and y-dimensions.

    :param radar_image_matrix: See doc for
        `deep_learning_utils.check_radar_images`.
    :param x_offset_pixels: Each image will be shifted this many pixels in the
        +x-direction.
    :param y_offset_pixels: Each image will be shifted this many pixels in the
        +y-direction.
    :return: shifted_image_matrix: Same as `radar_image_matrix`, but after
        shifting.  The shapes of the two numpy arrays are the same.
    """

    dl_utils.check_radar_images(
        radar_image_matrix=radar_image_matrix, min_num_dimensions=3,
        max_num_dimensions=5)

    num_grid_rows = radar_image_matrix.shape[1]
    num_grid_columns = radar_image_matrix.shape[2]
    half_num_grid_rows = numpy.floor(float(num_grid_rows) / 2)
    half_num_grid_columns = numpy.floor(float(num_grid_columns) / 2)

    error_checking.assert_is_integer(x_offset_pixels)
    error_checking.assert_is_geq(x_offset_pixels, 0)
    error_checking.assert_is_leq(x_offset_pixels, half_num_grid_columns)
    error_checking.assert_is_integer(y_offset_pixels)
    error_checking.assert_is_geq(y_offset_pixels, 0)
    error_checking.assert_is_leq(y_offset_pixels, half_num_grid_rows)

    if x_offset_pixels == y_offset_pixels == 0:
        return radar_image_matrix + 0.

    num_padding_columns_at_left = min([x_offset_pixels, 0])
    num_padding_columns_at_right = min([-x_offset_pixels, 0])
    num_padding_rows_at_top = min([y_offset_pixels, 0])
    num_padding_rows_at_bottom = min([-y_offset_pixels, 0])

    pad_width_input_arg = (
        (0, 0),
        (num_padding_rows_at_top, num_padding_rows_at_bottom),
        (num_padding_columns_at_left, num_padding_columns_at_right)
    )
    num_dimensions = len(radar_image_matrix.shape)
    for _ in range(3, num_dimensions):
        pad_width_input_arg += ((0, 0),)

    return numpy.pad(
        radar_image_matrix, pad_width=pad_width_input_arg, mode='constant',
        constant_values=PADDING_VALUE)


def rotate_radar_images(radar_image_matrix, ccw_rotation_angle_deg):
    """Rotates each radar image in the xy-plane.

    :param radar_image_matrix: See doc for
        `deep_learning_utils.check_radar_images`.
    :param ccw_rotation_angle_deg: Each image will be rotated counterclockwise
        by this angle (degrees).
    :return: rotated_image_matrix: Same as `radar_image_matrix`, but after
        rotation.  The shapes of the two numpy arrays are the same.
    """

    dl_utils.check_radar_images(
        radar_image_matrix=radar_image_matrix, min_num_dimensions=3,
        max_num_dimensions=5)
    error_checking.assert_is_geq(ccw_rotation_angle_deg, -180.)
    error_checking.assert_is_leq(ccw_rotation_angle_deg, 180.)

    num_dimensions = len(radar_image_matrix.shape)
    if num_dimensions > 3:
        num_channels = radar_image_matrix.shape[-1]
    else:
        num_channels = 1

    if num_dimensions == 5:
        num_heights = radar_image_matrix.shape[-2]
    else:
        num_heights = 1

    num_2d_images = radar_image_matrix.shape[0] * num_channels * num_heights
    num_grid_rows = radar_image_matrix.shape[1]
    num_grid_columns = radar_image_matrix.shape[2]

    rotated_image_matrix = numpy.reshape(
        radar_image_matrix, (num_2d_images, num_grid_rows, num_grid_columns))
    for i in range(num_2d_images):
        rotated_image_matrix[i, ...] = scipy_rotate(
            input=rotated_image_matrix[i, ...], angle=-ccw_rotation_angle_deg,
            axes=(1, 2), reshape=False, order=1, mode='constant',
            cval=PADDING_VALUE)

    return rotated_image_matrix


def noise_radar_images(radar_image_matrix, standard_deviation):
    """Adds Gaussian noise to each radar image.

    This method assumes that images are normalized (as by
    `deep_learning_utils.normalize_radar_images`), so the standard deviation is
    unitless and the same standard deviation applies to all channels.

    :param radar_image_matrix: See doc for
        `deep_learning_utils.check_radar_images`.
    :param standard_deviation: Standard deviation of Gaussian noise.
    :return: noised_image_matrix: Same as `radar_image_matrix`, but after
        noising.  The shapes of the two numpy arrays are the same.
    """

    dl_utils.check_radar_images(
        radar_image_matrix=radar_image_matrix, min_num_dimensions=3,
        max_num_dimensions=5)
    error_checking.assert_is_greater(standard_deviation, 0.)

    noise_matrix = numpy.random.normal(
        loc=0., scale=standard_deviation, size=radar_image_matrix.shape)
    return radar_image_matrix + noise_matrix
