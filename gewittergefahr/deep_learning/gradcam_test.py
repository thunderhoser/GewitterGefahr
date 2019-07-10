"""Unit tests for gradcam.py."""

import unittest
import numpy
from keras import backend as K
from gewittergefahr.deep_learning import gradcam

TOLERANCE = 1e-6

# The following constants are used to test _upsample_cam.
CLASS_ACTIV_MATRIX_1D = numpy.linspace(1, 49, num=49, dtype=float)
CLASS_ACTIV_MATRIX_COARSE_1D = CLASS_ACTIV_MATRIX_1D[::2]

THESE_ROW_INDICES = numpy.linspace(1, 64, num=64, dtype=float)
THESE_COLUMN_INDICES = numpy.linspace(1, 64, num=64, dtype=float)
THIS_ROW_INDEX_MATRIX, THIS_COLUMN_INDEX_MATRIX = numpy.meshgrid(
    THESE_ROW_INDICES, THESE_COLUMN_INDICES)

CLASS_ACTIV_MATRIX_2D = THIS_ROW_INDEX_MATRIX + THIS_COLUMN_INDEX_MATRIX
CLASS_ACTIV_MATRIX_COARSE_2D = CLASS_ACTIV_MATRIX_2D[::3, ::3]

THESE_ROW_INDICES = numpy.linspace(1, 33, num=33, dtype=float)
THESE_COLUMN_INDICES = numpy.linspace(1, 33, num=33, dtype=float)
THESE_HEIGHT_INDICES = numpy.linspace(1, 13, num=13, dtype=float)
THIS_ROW_INDEX_MATRIX, THIS_COLUMN_INDEX_MATRIX, THIS_HEIGHT_INDEX_MATRIX = (
    numpy.meshgrid(THESE_ROW_INDICES, THESE_COLUMN_INDICES,
                   THESE_HEIGHT_INDICES)
)

CLASS_ACTIV_MATRIX_3D = (
    THIS_ROW_INDEX_MATRIX + THIS_COLUMN_INDEX_MATRIX + THIS_HEIGHT_INDEX_MATRIX
)
CLASS_ACTIV_MATRIX_COARSE_3D = CLASS_ACTIV_MATRIX_3D[::4, ::4, ::4]

# The following constants are used to test _normalize_guided_gradcam_output.
GRADIENT_MATRIX_DENORM = numpy.array([
    [0, 2, 4, 2, 0],
    [1, 4, 7, 4, 1],
    [-5, -5, -5, -5, -5]
], dtype=float)

GRADIENT_MATRIX_DENORM = numpy.expand_dims(GRADIENT_MATRIX_DENORM, axis=0)

THIS_STANDARD_DEVIATION = (
    numpy.sqrt(numpy.mean(GRADIENT_MATRIX_DENORM ** 2)) + K.epsilon()
)

GRADIENT_MATRIX_NORM = GRADIENT_MATRIX_DENORM / THIS_STANDARD_DEVIATION
GRADIENT_MATRIX_NORM = 0.5 + GRADIENT_MATRIX_NORM * 0.1
GRADIENT_MATRIX_NORM[GRADIENT_MATRIX_NORM < 0.] = 0.
GRADIENT_MATRIX_NORM[GRADIENT_MATRIX_NORM > 1.] = 1.


class GradcamTests(unittest.TestCase):
    """Each method is a unit test for gradcam.py."""

    def test_upsample_cam_1d(self):
        """Ensures correct output from _upsample_cam.

        In this case the CAM (class-activation map) is 1-D.
        """

        this_matrix = gradcam._upsample_cam(
            class_activation_matrix=CLASS_ACTIV_MATRIX_COARSE_1D + 0.,
            new_dimensions=numpy.array(CLASS_ACTIV_MATRIX_1D.shape, dtype=int)
        )

        self.assertTrue(numpy.allclose(
            this_matrix, CLASS_ACTIV_MATRIX_1D, atol=TOLERANCE
        ))

    def test_upsample_cam_2d(self):
        """Ensures correct output from _upsample_cam.

        In this case the CAM (class-activation map) is 2-D.
        """

        this_matrix = gradcam._upsample_cam(
            class_activation_matrix=CLASS_ACTIV_MATRIX_COARSE_2D + 0.,
            new_dimensions=numpy.array(CLASS_ACTIV_MATRIX_2D.shape, dtype=int)
        )

        self.assertTrue(numpy.allclose(
            this_matrix, CLASS_ACTIV_MATRIX_2D, atol=TOLERANCE
        ))

    def test_upsample_cam_3d(self):
        """Ensures correct output from _upsample_cam.

        In this case the CAM (class-activation map) is 3-D.
        """

        this_matrix = gradcam._upsample_cam(
            class_activation_matrix=CLASS_ACTIV_MATRIX_COARSE_3D + 0.,
            new_dimensions=numpy.array(CLASS_ACTIV_MATRIX_3D.shape, dtype=int)
        )

        self.assertTrue(numpy.allclose(
            this_matrix, CLASS_ACTIV_MATRIX_3D, atol=TOLERANCE
        ))

    def test_normalize_guided_gradcam_output(self):
        """Ensures correct output from _normalize_guided_gradcam_output."""

        this_matrix = gradcam._normalize_guided_gradcam_output(
            GRADIENT_MATRIX_DENORM + 0.)

        self.assertTrue(numpy.allclose(
            this_matrix, GRADIENT_MATRIX_NORM, atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
