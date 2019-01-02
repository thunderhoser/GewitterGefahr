"""Augments image dataset by shifting, rotating, or adding Gaussian noise."""

import numpy
from scipy.ndimage.interpolation import rotate as scipy_rotate
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils

PADDING_VALUE = 0.
MIN_ABSOLUTE_ROTATION_ANGLE_DEG = 1.
MAX_ABSOLUTE_ROTATION_ANGLE_DEG = 90.
MIN_NOISE_STANDARD_DEVIATION = 1e-9
MAX_NOISE_STANDARD_DEVIATION = 0.25


def get_translations(
        num_translations, max_translation_pixels, num_grid_rows,
        num_grid_columns):
    """Creates an array of x- and y-translations.

    These translations ("offsets") are meant for use in `shift_radar_images`.

    N = number of translations

    :param num_translations: Number of translations.  Image will be translated
        in only the x- and y-directions, not the z-direction.
    :param max_translation_pixels: Max translation in either direction.  Must be
        an integer.
    :param num_grid_rows: Number of rows in the image.
    :param num_grid_columns: Number of columns in the image.
    :return: x_offsets_pixels: length-N numpy array of x-translations
        (integers).
    :return: y_offsets_pixels: length-N numpy array of y-translations
        (integers).
    """

    error_checking.assert_is_integer(num_translations)
    if num_translations == 0:
        return numpy.array([], dtype=int), numpy.array([], dtype=int)

    error_checking.assert_is_greater(num_translations, 0)
    error_checking.assert_is_integer(num_grid_rows)
    error_checking.assert_is_geq(num_grid_rows, 2)
    error_checking.assert_is_integer(num_grid_columns)
    error_checking.assert_is_geq(num_grid_columns, 2)
    error_checking.assert_is_integer(max_translation_pixels)
    error_checking.assert_is_greater(max_translation_pixels, 0)

    smallest_horiz_dimension = min([num_grid_rows, num_grid_columns])
    max_max_translation_pixels = int(numpy.floor(
        float(smallest_horiz_dimension) / 2))
    error_checking.assert_is_leq(
        max_translation_pixels, max_max_translation_pixels)

    x_offsets_pixels = numpy.random.random_integers(
        low=-max_translation_pixels, high=max_translation_pixels,
        size=num_translations * 4)
    y_offsets_pixels = numpy.random.random_integers(
        low=-max_translation_pixels, high=max_translation_pixels,
        size=num_translations * 4)

    good_indices = numpy.where(
        numpy.absolute(x_offsets_pixels) + numpy.absolute(y_offsets_pixels) > 0
    )[0]
    good_indices = numpy.random.choice(
        good_indices, size=num_translations, replace=False)

    return x_offsets_pixels[good_indices], y_offsets_pixels[good_indices]


def get_rotations(num_rotations, max_absolute_rotation_angle_deg):
    """Creates an array of rotation angles.

    These angles are meant for use in `rotate_radar_images`.

    N = number of rotations

    :param num_rotations: Number of rotations.  Image will be rotated only in
        the xy-plane (about the z-axis).
    :param max_absolute_rotation_angle_deg: Max absolute rotation angle
        (degrees).  In general, the image will be rotated both clockwise and
        counterclockwise, up to this angle.
    :return: ccw_rotation_angles_deg: length-N numpy array of counterclockwise
        rotation angles (degrees).
    """

    error_checking.assert_is_integer(num_rotations)
    if num_rotations == 0:
        return numpy.array([], dtype=float)

    error_checking.assert_is_greater(num_rotations, 0)
    error_checking.assert_is_geq(
        max_absolute_rotation_angle_deg, MIN_ABSOLUTE_ROTATION_ANGLE_DEG)
    error_checking.assert_is_leq(
        max_absolute_rotation_angle_deg, MAX_ABSOLUTE_ROTATION_ANGLE_DEG)

    absolute_rotation_angles_deg = numpy.random.uniform(
        low=1., high=max_absolute_rotation_angle_deg, size=num_rotations)

    possible_signs = numpy.array([-1, 1], dtype=int)
    return absolute_rotation_angles_deg * numpy.random.choice(
        possible_signs, size=num_rotations, replace=True)


def get_noisings(num_noisings, max_standard_deviation):
    """Creates an array of standard deviations for Gaussian noising.

    These standard deviations are meant for use in `noise_radar_images`.

    N = number of noisings

    :param num_noisings: Number of times to noise the image.
    :param max_standard_deviation: Max standard deviation of Gaussian noise.
    :return: standard_deviations: length-N numpy array of standard deviations.
    """

    error_checking.assert_is_integer(num_noisings)
    if num_noisings == 0:
        return numpy.array([], dtype=float)

    error_checking.assert_is_greater(num_noisings, 0)
    error_checking.assert_is_geq(
        max_standard_deviation, MIN_NOISE_STANDARD_DEVIATION)
    error_checking.assert_is_leq(
        max_standard_deviation, MAX_NOISE_STANDARD_DEVIATION)

    return numpy.random.uniform(
        low=0., high=max_standard_deviation, size=num_noisings)


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
    half_num_grid_rows = numpy.floor(float(num_grid_rows) / 2)
    error_checking.assert_is_integer(y_offset_pixels)
    error_checking.assert_is_geq(y_offset_pixels, -half_num_grid_rows)
    error_checking.assert_is_leq(y_offset_pixels, half_num_grid_rows)

    num_grid_columns = radar_image_matrix.shape[2]
    half_num_grid_columns = numpy.floor(float(num_grid_columns) / 2)
    error_checking.assert_is_integer(x_offset_pixels)
    error_checking.assert_is_geq(x_offset_pixels, -half_num_grid_columns)
    error_checking.assert_is_leq(x_offset_pixels, half_num_grid_columns)

    if x_offset_pixels == y_offset_pixels == 0:
        return radar_image_matrix + 0.

    num_padding_columns_at_left = max([x_offset_pixels, 0])
    num_padding_columns_at_right = max([-x_offset_pixels, 0])
    num_padding_rows_at_top = max([y_offset_pixels, 0])
    num_padding_rows_at_bottom = max([-y_offset_pixels, 0])

    pad_width_input_arg = (
        (0, 0),
        (num_padding_rows_at_top, num_padding_rows_at_bottom),
        (num_padding_columns_at_left, num_padding_columns_at_right)
    )
    num_dimensions = len(radar_image_matrix.shape)
    for _ in range(3, num_dimensions):
        pad_width_input_arg += ((0, 0),)

    shifted_image_matrix = numpy.pad(
        radar_image_matrix, pad_width=pad_width_input_arg, mode='constant',
        constant_values=PADDING_VALUE)

    if x_offset_pixels >= 0:
        shifted_image_matrix = shifted_image_matrix[
            :, :, :num_grid_columns, ...]
    else:
        shifted_image_matrix = shifted_image_matrix[
            :, :, -num_grid_columns:, ...]

    if y_offset_pixels >= 0:
        shifted_image_matrix = shifted_image_matrix[:, :num_grid_rows, ...]
    else:
        shifted_image_matrix = shifted_image_matrix[:, -num_grid_rows:, ...]

    return shifted_image_matrix


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

    return scipy_rotate(
        input=radar_image_matrix, angle=-ccw_rotation_angle_deg, axes=(1, 2),
        reshape=False, order=1, mode='constant', cval=PADDING_VALUE)


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


def flip_radar_images_x(radar_image_matrix):
    """Flips radar images in the x-direction.

    :param radar_image_matrix: See doc for
        `deep_learning_utils.check_radar_images`.
    :return: flipped_image_matrix: Flipped version of input (same dimensions).
    """

    dl_utils.check_radar_images(
        radar_image_matrix=radar_image_matrix, min_num_dimensions=3,
        max_num_dimensions=5)

    return numpy.flip(radar_image_matrix, axis=2)


def flip_radar_images_y(radar_image_matrix):
    """Flips radar images in the y-direction.

    :param radar_image_matrix: See doc for
        `deep_learning_utils.check_radar_images`.
    :return: flipped_image_matrix: Flipped version of input (same dimensions).
    """

    dl_utils.check_radar_images(
        radar_image_matrix=radar_image_matrix, min_num_dimensions=3,
        max_num_dimensions=5)

    return numpy.flip(radar_image_matrix, axis=1)
