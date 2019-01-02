"""Unit tests for data_augmentation.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import data_augmentation

TOLERANCE = 1e-6

THIS_FIRST_MATRIX = numpy.array([[0, 1, 2, 3],
                                 [4, 5, 6, 7],
                                 [8, 9, 10, 11]], dtype=float)
THIS_SECOND_MATRIX = numpy.array([[2, 5, 8, 11],
                                  [3, 6, 9, 12],
                                  [4, 7, 10, 13]], dtype=float)

RADAR_MATRIX_3D_NO_SHIFT = numpy.stack(
    (THIS_FIRST_MATRIX, THIS_SECOND_MATRIX), axis=0)
RADAR_MATRIX_4D_NO_SHIFT = numpy.stack((RADAR_MATRIX_3D_NO_SHIFT,) * 6, axis=-1)
RADAR_MATRIX_5D_NO_SHIFT = numpy.stack((RADAR_MATRIX_4D_NO_SHIFT,) * 5, axis=-2)

# The following constants are used to test get_translations.
NUM_TRANSLATIONS = 8
MAX_TRANSLATION_PIXELS = 3

# The following constants are used to test get_rotations.
NUM_ROTATIONS = 9
MAX_ABSOLUTE_ROTATION_ANGLE_DEG = 15.

# The following constants are used to test get_noisings.
NUM_NOISINGS = 10
MAX_NOISE_STANDARD_DEVIATION = 0.1

# The following constants are used to test shift_radar_images.
POSITIVE_X_OFFSET_PIXELS = 2
POSITIVE_Y_OFFSET_PIXELS = 1
THIS_FIRST_MATRIX = numpy.array([[0, 0, 0, 0],
                                 [0, 0, 0, 1],
                                 [0, 0, 4, 5]], dtype=float)
THIS_SECOND_MATRIX = numpy.array([[0, 0, 0, 0],
                                  [0, 0, 2, 5],
                                  [0, 0, 3, 6]], dtype=float)

RADAR_MATRIX_3D_POSITIVE_SHIFT = numpy.stack(
    (THIS_FIRST_MATRIX, THIS_SECOND_MATRIX), axis=0)
RADAR_MATRIX_4D_POSITIVE_SHIFT = numpy.stack(
    (RADAR_MATRIX_3D_POSITIVE_SHIFT,) * 6, axis=-1)
RADAR_MATRIX_5D_POSITIVE_SHIFT = numpy.stack(
    (RADAR_MATRIX_4D_POSITIVE_SHIFT,) * 5, axis=-2)

NEGATIVE_X_OFFSET_PIXELS = -1
NEGATIVE_Y_OFFSET_PIXELS = -1
THIS_FIRST_MATRIX = numpy.array([[5, 6, 7, 0],
                                 [9, 10, 11, 0],
                                 [0, 0, 0, 0]], dtype=float)
THIS_SECOND_MATRIX = numpy.array([[6, 9, 12, 0],
                                  [7, 10, 13, 0],
                                  [0, 0, 0, 0]], dtype=float)

RADAR_MATRIX_3D_NEGATIVE_SHIFT = numpy.stack(
    (THIS_FIRST_MATRIX, THIS_SECOND_MATRIX), axis=0)
RADAR_MATRIX_4D_NEGATIVE_SHIFT = numpy.stack(
    (RADAR_MATRIX_3D_NEGATIVE_SHIFT,) * 6, axis=-1)
RADAR_MATRIX_5D_NEGATIVE_SHIFT = numpy.stack(
    (RADAR_MATRIX_4D_NEGATIVE_SHIFT,) * 5, axis=-2)

# The following constants are used to test rotate_radar_images.
THIS_FIRST_MATRIX = numpy.array([[0, 1, 2, 3, 4],
                                 [5, 6, 7, 8, 9],
                                 [10, 11, 12, 13, 14],
                                 [15, 16, 17, 18, 19]], dtype=float)
THIS_SECOND_MATRIX = numpy.array([[-45, -25, -5, 15, 35],
                                  [-40, -20, 0, 20, 40],
                                  [-35, -15, 5, 25, 45],
                                  [-30, -10, 10, 30, 50]], dtype=float)

RADAR_MATRIX_3D_NO_ROTATION = numpy.stack(
    (THIS_FIRST_MATRIX, THIS_SECOND_MATRIX), axis=0)
RADAR_MATRIX_4D_NO_ROTATION = numpy.stack(
    (RADAR_MATRIX_3D_NO_ROTATION,) * 6, axis=-1)
RADAR_MATRIX_5D_NO_ROTATION = numpy.stack(
    (RADAR_MATRIX_4D_NO_ROTATION,) * 5, axis=-2)

POSITIVE_CCW_ANGLE_DEG = 90.
THIS_FIRST_MATRIX = numpy.array([[0, 13, 8, 3, 0],
                                 [0, 14, 9, 4, 0],
                                 [0, 15, 10, 5, 0],
                                 [0, 16, 11, 6, 0]], dtype=float)
THIS_SECOND_MATRIX = numpy.array([[0, -22.5, -27.5, -32.5, 0],
                                  [0, -2.5, -7.5, -12.5, 0],
                                  [0, 17.5, 12.5, 7.5, 0],
                                  [0, 37.5, 32.5, 27.5, 0]])

RADAR_MATRIX_3D_POS_ROTATION = numpy.stack(
    (THIS_FIRST_MATRIX, THIS_SECOND_MATRIX), axis=0)
RADAR_MATRIX_4D_POS_ROTATION = numpy.stack(
    (RADAR_MATRIX_3D_POS_ROTATION,) * 6, axis=-1)
RADAR_MATRIX_5D_POS_ROTATION = numpy.stack(
    (RADAR_MATRIX_4D_POS_ROTATION,) * 5, axis=-2)

NEGATIVE_CCW_ANGLE_DEG = -90.
THIS_FIRST_MATRIX = numpy.array([[0, 6, 11, 16, 0],
                                 [0, 5, 10, 15, 0],
                                 [0, 4, 9, 14, 0],
                                 [0, 3, 8, 13, 0]], dtype=float)
THIS_SECOND_MATRIX = numpy.array([[0, 27.5, 32.5, 37.5, 0],
                                  [0, 7.5, 12.5, 17.5, 0],
                                  [0, -12.5, -7.5, -2.5, 0],
                                  [0, -32.5, -27.5, -22.5, 0]])

RADAR_MATRIX_3D_NEG_ROTATION = numpy.stack(
    (THIS_FIRST_MATRIX, THIS_SECOND_MATRIX), axis=0)
RADAR_MATRIX_4D_NEG_ROTATION = numpy.stack(
    (RADAR_MATRIX_3D_NEG_ROTATION,) * 6, axis=-1)
RADAR_MATRIX_5D_NEG_ROTATION = numpy.stack(
    (RADAR_MATRIX_4D_NEG_ROTATION,) * 5, axis=-2)

# The following constants are used to test flip_radar_images_x and
# flip_radar_images_y.
THIS_FIRST_MATRIX = numpy.array([[0, 1, 2, 3],
                                 [4, 5, 6, 7],
                                 [8, 9, 10, 11]], dtype=float)
THIS_SECOND_MATRIX = numpy.array([[2, 5, 8, 11],
                                  [3, 6, 9, 12],
                                  [4, 7, 10, 13]], dtype=float)

RADAR_MATRIX_3D_NO_FLIP = numpy.stack(
    (THIS_FIRST_MATRIX, THIS_SECOND_MATRIX), axis=0)
RADAR_MATRIX_4D_NO_FLIP = numpy.stack((RADAR_MATRIX_3D_NO_FLIP,) * 6, axis=-1)
RADAR_MATRIX_5D_NO_FLIP = numpy.stack((RADAR_MATRIX_4D_NO_FLIP,) * 5, axis=-2)

THIS_FIRST_MATRIX = numpy.array([[3, 2, 1, 0],
                                 [7, 6, 5, 4],
                                 [11, 10, 9, 8]], dtype=float)
THIS_SECOND_MATRIX = numpy.array([[11, 8, 5, 2],
                                  [12, 9, 6, 3],
                                  [13, 10, 7, 4]], dtype=float)

RADAR_MATRIX_3D_X_FLIP = numpy.stack(
    (THIS_FIRST_MATRIX, THIS_SECOND_MATRIX), axis=0)
RADAR_MATRIX_4D_X_FLIP = numpy.stack((RADAR_MATRIX_3D_X_FLIP,) * 6, axis=-1)
RADAR_MATRIX_5D_X_FLIP = numpy.stack((RADAR_MATRIX_4D_X_FLIP,) * 5, axis=-2)

THIS_FIRST_MATRIX = numpy.array([[8, 9, 10, 11],
                                 [4, 5, 6, 7],
                                 [0, 1, 2, 3]], dtype=float)
THIS_SECOND_MATRIX = numpy.array([[4, 7, 10, 13],
                                  [3, 6, 9, 12],
                                  [2, 5, 8, 11]], dtype=float)

RADAR_MATRIX_3D_Y_FLIP = numpy.stack(
    (THIS_FIRST_MATRIX, THIS_SECOND_MATRIX), axis=0)
RADAR_MATRIX_4D_Y_FLIP = numpy.stack((RADAR_MATRIX_3D_Y_FLIP,) * 6, axis=-1)
RADAR_MATRIX_5D_Y_FLIP = numpy.stack((RADAR_MATRIX_4D_Y_FLIP,) * 5, axis=-2)


class DataAugmentationTests(unittest.TestCase):
    """Each method is a unit test for data_augmentation.py."""

    def test_get_translations(self):
        """Ensures correct output from get_translations."""

        (these_x_offsets_pixels, these_y_offsets_pixels
        ) = data_augmentation.get_translations(
            num_translations=NUM_TRANSLATIONS,
            max_translation_pixels=MAX_TRANSLATION_PIXELS,
            num_grid_rows=2 * MAX_TRANSLATION_PIXELS,
            num_grid_columns=2 * MAX_TRANSLATION_PIXELS)

        self.assertTrue(len(these_x_offsets_pixels) == NUM_TRANSLATIONS)
        error_checking.assert_is_geq_numpy_array(
            these_x_offsets_pixels, -MAX_TRANSLATION_PIXELS)
        error_checking.assert_is_leq_numpy_array(
            these_x_offsets_pixels, MAX_TRANSLATION_PIXELS)

        self.assertTrue(len(these_y_offsets_pixels) == NUM_TRANSLATIONS)
        error_checking.assert_is_geq_numpy_array(
            these_y_offsets_pixels, -MAX_TRANSLATION_PIXELS)
        error_checking.assert_is_leq_numpy_array(
            these_y_offsets_pixels, MAX_TRANSLATION_PIXELS)

        error_checking.assert_is_greater_numpy_array(
            numpy.absolute(these_x_offsets_pixels) +
            numpy.absolute(these_y_offsets_pixels), 0)

    def test_get_rotations(self):
        """Ensures correct output from get_rotations."""

        these_ccw_rotation_angles_deg = data_augmentation.get_rotations(
            num_rotations=NUM_ROTATIONS,
            max_absolute_rotation_angle_deg=MAX_ABSOLUTE_ROTATION_ANGLE_DEG)

        self.assertTrue(len(these_ccw_rotation_angles_deg) == NUM_ROTATIONS)
        error_checking.assert_is_geq_numpy_array(
            numpy.absolute(these_ccw_rotation_angles_deg),
            data_augmentation.MIN_ABSOLUTE_ROTATION_ANGLE_DEG)
        error_checking.assert_is_leq_numpy_array(
            numpy.absolute(these_ccw_rotation_angles_deg),
            data_augmentation.MAX_ABSOLUTE_ROTATION_ANGLE_DEG)

    def test_get_noisings(self):
        """Ensures correct output get_noisings."""

        these_standard_deviations = data_augmentation.get_noisings(
            num_noisings=NUM_NOISINGS,
            max_standard_deviation=MAX_NOISE_STANDARD_DEVIATION)

        self.assertTrue(len(these_standard_deviations) == NUM_NOISINGS)
        error_checking.assert_is_geq_numpy_array(
            these_standard_deviations,
            data_augmentation.MIN_NOISE_STANDARD_DEVIATION)
        error_checking.assert_is_leq_numpy_array(
            these_standard_deviations,
            data_augmentation.MAX_NOISE_STANDARD_DEVIATION)

    def test_shift_radar_images_3d_positive(self):
        """Ensures correct output from shift_radar_images.

        In this case, the input matrix is 3-D and shifted in the positive x- and
        y-directions.
        """

        this_radar_matrix = data_augmentation.shift_radar_images(
            radar_image_matrix=RADAR_MATRIX_3D_NO_SHIFT,
            x_offset_pixels=POSITIVE_X_OFFSET_PIXELS,
            y_offset_pixels=POSITIVE_Y_OFFSET_PIXELS)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_3D_POSITIVE_SHIFT, atol=TOLERANCE))

    def test_shift_radar_images_4d_positive(self):
        """Ensures correct output from shift_radar_images.

        In this case, the input matrix is 4-D and shifted in the positive x- and
        y-directions.
        """

        this_radar_matrix = data_augmentation.shift_radar_images(
            radar_image_matrix=RADAR_MATRIX_4D_NO_SHIFT,
            x_offset_pixels=POSITIVE_X_OFFSET_PIXELS,
            y_offset_pixels=POSITIVE_Y_OFFSET_PIXELS)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_4D_POSITIVE_SHIFT, atol=TOLERANCE))

    def test_shift_radar_images_5d_positive(self):
        """Ensures correct output from shift_radar_images.

        In this case, the input matrix is 5-D and shifted in the positive x- and
        y-directions.
        """

        this_radar_matrix = data_augmentation.shift_radar_images(
            radar_image_matrix=RADAR_MATRIX_5D_NO_SHIFT,
            x_offset_pixels=POSITIVE_X_OFFSET_PIXELS,
            y_offset_pixels=POSITIVE_Y_OFFSET_PIXELS)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_5D_POSITIVE_SHIFT, atol=TOLERANCE))

    def test_shift_radar_images_3d_negative(self):
        """Ensures correct output from shift_radar_images.

        In this case, the input matrix is 3-D and shifted in the negative x- and
        y-directions.
        """

        this_radar_matrix = data_augmentation.shift_radar_images(
            radar_image_matrix=RADAR_MATRIX_3D_NO_SHIFT,
            x_offset_pixels=NEGATIVE_X_OFFSET_PIXELS,
            y_offset_pixels=NEGATIVE_Y_OFFSET_PIXELS)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_3D_NEGATIVE_SHIFT, atol=TOLERANCE))

    def test_shift_radar_images_4d_negative(self):
        """Ensures correct output from shift_radar_images.

        In this case, the input matrix is 4-D and shifted in the negative x- and
        y-directions.
        """

        this_radar_matrix = data_augmentation.shift_radar_images(
            radar_image_matrix=RADAR_MATRIX_4D_NO_SHIFT,
            x_offset_pixels=NEGATIVE_X_OFFSET_PIXELS,
            y_offset_pixels=NEGATIVE_Y_OFFSET_PIXELS)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_4D_NEGATIVE_SHIFT, atol=TOLERANCE))

    def test_shift_radar_images_5d_negative(self):
        """Ensures correct output from shift_radar_images.

        In this case, the input matrix is 5-D and shifted in the negative x- and
        y-directions.
        """

        this_radar_matrix = data_augmentation.shift_radar_images(
            radar_image_matrix=RADAR_MATRIX_5D_NO_SHIFT,
            x_offset_pixels=NEGATIVE_X_OFFSET_PIXELS,
            y_offset_pixels=NEGATIVE_Y_OFFSET_PIXELS)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_5D_NEGATIVE_SHIFT, atol=TOLERANCE))

    def test_shift_radar_images_3d_no_shift(self):
        """Ensures correct output from shift_radar_images.

        In this case, the input matrix is 3-D and not shifted at all.
        """

        this_radar_matrix = data_augmentation.shift_radar_images(
            radar_image_matrix=RADAR_MATRIX_3D_NO_SHIFT,
            x_offset_pixels=0, y_offset_pixels=0)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_3D_NO_SHIFT, atol=TOLERANCE))

    def test_shift_radar_images_4d_no_shift(self):
        """Ensures correct output from shift_radar_images.

        In this case, the input matrix is 4-D and not shifted at all.
        """

        this_radar_matrix = data_augmentation.shift_radar_images(
            radar_image_matrix=RADAR_MATRIX_4D_NO_SHIFT,
            x_offset_pixels=0, y_offset_pixels=0)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_4D_NO_SHIFT, atol=TOLERANCE))

    def test_shift_radar_images_5d_no_shift(self):
        """Ensures correct output from shift_radar_images.

        In this case, the input matrix is 5-D and not shifted at all.
        """

        this_radar_matrix = data_augmentation.shift_radar_images(
            radar_image_matrix=RADAR_MATRIX_5D_NO_SHIFT,
            x_offset_pixels=0, y_offset_pixels=0)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_5D_NO_SHIFT, atol=TOLERANCE))

    def test_rotate_radar_images_3d_positive(self):
        """Ensures correct output from rotate_radar_images.

        In this case, the input matrix is 3-D and undergoes a positive
        (counterclockwise) rotation.
        """

        this_radar_matrix = data_augmentation.rotate_radar_images(
            radar_image_matrix=RADAR_MATRIX_3D_NO_ROTATION,
            ccw_rotation_angle_deg=POSITIVE_CCW_ANGLE_DEG)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_3D_POS_ROTATION, atol=TOLERANCE))

    def test_rotate_radar_images_4d_positive(self):
        """Ensures correct output from rotate_radar_images.

        In this case, the input matrix is 4-D and undergoes a positive
        (counterclockwise) rotation.
        """

        this_radar_matrix = data_augmentation.rotate_radar_images(
            radar_image_matrix=RADAR_MATRIX_4D_NO_ROTATION,
            ccw_rotation_angle_deg=POSITIVE_CCW_ANGLE_DEG)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_4D_POS_ROTATION, atol=TOLERANCE))

    def test_rotate_radar_images_5d_positive(self):
        """Ensures correct output from rotate_radar_images.

        In this case, the input matrix is 5-D and undergoes a positive
        (counterclockwise) rotation.
        """

        this_radar_matrix = data_augmentation.rotate_radar_images(
            radar_image_matrix=RADAR_MATRIX_5D_NO_ROTATION,
            ccw_rotation_angle_deg=POSITIVE_CCW_ANGLE_DEG)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_5D_POS_ROTATION, atol=TOLERANCE))

    def test_rotate_radar_images_3d_negative(self):
        """Ensures correct output from rotate_radar_images.

        In this case, the input matrix is 3-D and undergoes a negative
        (clockwise) rotation.
        """

        this_radar_matrix = data_augmentation.rotate_radar_images(
            radar_image_matrix=RADAR_MATRIX_3D_NO_ROTATION,
            ccw_rotation_angle_deg=NEGATIVE_CCW_ANGLE_DEG)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_3D_NEG_ROTATION, atol=TOLERANCE))

    def test_rotate_radar_images_4d_negative(self):
        """Ensures correct output from rotate_radar_images.

        In this case, the input matrix is 4-D and undergoes a negative
        (clockwise) rotation.
        """

        this_radar_matrix = data_augmentation.rotate_radar_images(
            radar_image_matrix=RADAR_MATRIX_4D_NO_ROTATION,
            ccw_rotation_angle_deg=NEGATIVE_CCW_ANGLE_DEG)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_4D_NEG_ROTATION, atol=TOLERANCE))

    def test_rotate_radar_images_5d_negative(self):
        """Ensures correct output from rotate_radar_images.

        In this case, the input matrix is 5-D and undergoes a negative
        (clockwise) rotation.
        """

        this_radar_matrix = data_augmentation.rotate_radar_images(
            radar_image_matrix=RADAR_MATRIX_5D_NO_ROTATION,
            ccw_rotation_angle_deg=NEGATIVE_CCW_ANGLE_DEG)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_5D_NEG_ROTATION, atol=TOLERANCE))

    def test_rotate_radar_images_3d_no_rotation(self):
        """Ensures correct output from rotate_radar_images.

        In this case, the input matrix is 3-D and not rotated at all.
        """

        this_radar_matrix = data_augmentation.rotate_radar_images(
            radar_image_matrix=RADAR_MATRIX_3D_NO_ROTATION,
            ccw_rotation_angle_deg=0.)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_3D_NO_ROTATION, atol=TOLERANCE))

    def test_rotate_radar_images_4d_no_rotation(self):
        """Ensures correct output from rotate_radar_images.

        In this case, the input matrix is 4-D and not rotated at all.
        """

        this_radar_matrix = data_augmentation.rotate_radar_images(
            radar_image_matrix=RADAR_MATRIX_4D_NO_ROTATION,
            ccw_rotation_angle_deg=0.)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_4D_NO_ROTATION, atol=TOLERANCE))

    def test_rotate_radar_images_5d_no_rotation(self):
        """Ensures correct output from rotate_radar_images.

        In this case, the input matrix is 5-D and not rotated at all.
        """

        this_radar_matrix = data_augmentation.rotate_radar_images(
            radar_image_matrix=RADAR_MATRIX_5D_NO_ROTATION,
            ccw_rotation_angle_deg=0.)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_5D_NO_ROTATION, atol=TOLERANCE))

    def test_flip_radar_images_x_3d(self):
        """Ensures correct output from flip_radar_images_x.

        In this case the input matrix is 3-D.
        """

        this_radar_matrix = data_augmentation.flip_radar_images_x(
            RADAR_MATRIX_3D_NO_FLIP)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_3D_X_FLIP, atol=TOLERANCE))

    def test_flip_radar_images_x_4d(self):
        """Ensures correct output from flip_radar_images_x.

        In this case the input matrix is 4-D.
        """

        this_radar_matrix = data_augmentation.flip_radar_images_x(
            RADAR_MATRIX_4D_NO_FLIP)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_4D_X_FLIP, atol=TOLERANCE))

    def test_flip_radar_images_x_5d(self):
        """Ensures correct output from flip_radar_images_x.

        In this case the input matrix is 5-D.
        """

        this_radar_matrix = data_augmentation.flip_radar_images_x(
            RADAR_MATRIX_5D_NO_FLIP)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_5D_X_FLIP, atol=TOLERANCE))

    def test_flip_radar_images_y_3d(self):
        """Ensures correct output from flip_radar_images_y.

        In this case the input matrix is 3-D.
        """

        this_radar_matrix = data_augmentation.flip_radar_images_y(
            RADAR_MATRIX_3D_NO_FLIP)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_3D_Y_FLIP, atol=TOLERANCE))

    def test_flip_radar_images_y_4d(self):
        """Ensures correct output from flip_radar_images_y.

        In this case the input matrix is 4-D.
        """

        this_radar_matrix = data_augmentation.flip_radar_images_y(
            RADAR_MATRIX_4D_NO_FLIP)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_4D_Y_FLIP, atol=TOLERANCE))

    def test_flip_radar_images_y_5d(self):
        """Ensures correct output from flip_radar_images_y.

        In this case the input matrix is 5-D.
        """

        this_radar_matrix = data_augmentation.flip_radar_images_y(
            RADAR_MATRIX_5D_NO_FLIP)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_5D_Y_FLIP, atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
