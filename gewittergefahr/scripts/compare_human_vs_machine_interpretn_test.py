"""Unit tests for compare_human_vs_machine_interpretn.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.scripts import \
    compare_human_vs_machine_interpretn as human_vs_machine

TOLERANCE = 1e-6

# The following constants are used to test _reshape_human_maps.
THIS_FIRST_MATRIX_2D = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
], dtype=bool)

THIS_SECOND_MATRIX_2D = numpy.array([
    [1, 0, 0, 0, 0, 0, 0, 1],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1]
], dtype=bool)

THIS_THIRD_MATRIX_2D = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
], dtype=bool)

THIS_FOURTH_MATRIX_2D = numpy.array([
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]
], dtype=bool)

THIS_FIRST_MATRIX_3D = numpy.stack(
    (THIS_FIRST_MATRIX_2D, THIS_THIRD_MATRIX_2D), axis=0
)
THIS_SECOND_MATRIX_3D = numpy.stack(
    (THIS_SECOND_MATRIX_2D, THIS_FOURTH_MATRIX_2D), axis=0
)
HUMAN_POSITIVE_MASK_MATRIX_4D = numpy.stack(
    (THIS_FIRST_MATRIX_3D, THIS_SECOND_MATRIX_3D), axis=0
)

HUMAN_POSITIVE_MASK_MATRIX_3D = numpy.stack(
    (THIS_FIRST_MATRIX_2D, THIS_SECOND_MATRIX_2D, THIS_THIRD_MATRIX_2D,
     THIS_FOURTH_MATRIX_2D),
    axis=-1
)

HUMAN_NEGATIVE_MASK_MATRIX_4D = numpy.invert(HUMAN_POSITIVE_MASK_MATRIX_4D)
HUMAN_NEGATIVE_MASK_MATRIX_3D = numpy.invert(HUMAN_POSITIVE_MASK_MATRIX_3D)

RADAR_FIELD_NAMES = [
    radar_utils.REFL_NAME, radar_utils.SPECTRUM_WIDTH_NAME,
    radar_utils.VORTICITY_NAME, radar_utils.DIVERGENCE_NAME
]

MODEL_METADATA_DICT = {
    cnn.CONV_2D3D_KEY: False,
    cnn.LAYER_OPERATIONS_KEY: None,
    cnn.TRAINING_OPTION_DICT_KEY: {
        trainval_io.RADAR_FIELDS_KEY: RADAR_FIELD_NAMES
    }
}

# The following constants are used to test _do_comparison_one_channel.
THIS_FIRST_MATRIX = numpy.array([
    [-5, 0, 0, 0, -5, 0, 0, 0],
    [-5, 0, 0, 5, 5, 0, 0, 5],
    [0, 0, 5, 5, 5, 5, 0, 0],
    [-5, -5, 5, 5, 5, -5, -5, -5],
    [-5, 5, 5, 0, 0, 0, 0, 0],
    [-5, -5, -5, -5, 0, 0, 0, 0]
], dtype=float)

THIS_SECOND_MATRIX = numpy.array([
    [6, 0, 0, -6, -6, 0, 0, 6],
    [-6, -6, -6, -6, -6, -6, -6, -6],
    [-6, -6, -6, -6, -6, -6, -6, -6],
    [0, 0, 0, 6, 6, 0, 0, 0],
    [0, 0, 0, 6, 6, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
], dtype=float)

THIS_THIRD_MATRIX = numpy.array([
    [7, 7, 7, 7, 7, 7, 7, 7],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [-7, -7, -7, -7, -7, -7, -7, -7],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [-7, -7, -7, -7, -7, -7, -7, -7],
    [0, 0, 0, 0, 0, 0, 0, 0]
], dtype=float)

THIS_FOURTH_MATRIX = numpy.array([
    [-8, -8, -8, -8, -8, -8, -8, -8],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 0, 0, 0, 0]
], dtype=float)

MACHINE_INTERPRETATION_MATRIX_3D = numpy.stack(
    (THIS_FIRST_MATRIX, THIS_SECOND_MATRIX, THIS_THIRD_MATRIX,
     THIS_FOURTH_MATRIX),
    axis=-1
)

ABS_PERCENTILE_THRESHOLD = 90.
POSITIVE_IOU_BY_CHANNEL = numpy.array([11. / 13, 6. / 14, 0, 16. / 48])
NEGATIVE_IOU_BY_CHANNEL = numpy.array([13. / 36, 14. / 38, 16. / 48, 0])


class HumanVsMachineTests(unittest.TestCase):
    """Each method is a unit test for compare_human_vs_machine_interpretn.py."""

    def test_reshape_human_maps(self):
        """Ensures correct output from _reshape_human_maps."""

        this_positive_matrix_3d, this_negative_matrix_3d = (
            human_vs_machine._reshape_human_maps(
                model_metadata_dict=MODEL_METADATA_DICT,
                positive_mask_matrix_4d=HUMAN_POSITIVE_MASK_MATRIX_4D,
                negative_mask_matrix_4d=HUMAN_NEGATIVE_MASK_MATRIX_4D)
        )

        self.assertTrue(numpy.array_equal(
            this_positive_matrix_3d, HUMAN_POSITIVE_MASK_MATRIX_3D
        ))

        self.assertTrue(numpy.array_equal(
            this_negative_matrix_3d, HUMAN_NEGATIVE_MASK_MATRIX_3D
        ))

    def test_do_comparison_one_channel(self):
        """Ensures correct output from _do_comparison_one_channel."""

        num_channels = MACHINE_INTERPRETATION_MATRIX_3D.shape[-1]
        these_positive_iou = numpy.full(num_channels, numpy.nan)
        these_negative_iou = numpy.full(num_channels, numpy.nan)

        for k in range(num_channels):
            this_comparison_dict = human_vs_machine._do_comparison_one_channel(
                machine_interpretation_matrix_2d=
                MACHINE_INTERPRETATION_MATRIX_3D[..., k],
                abs_percentile_threshold=ABS_PERCENTILE_THRESHOLD,
                human_positive_mask_matrix_2d=
                HUMAN_POSITIVE_MASK_MATRIX_3D[..., k],
                human_negative_mask_matrix_2d=
                HUMAN_NEGATIVE_MASK_MATRIX_3D[..., k]
            )

            these_positive_iou[k] = this_comparison_dict[
                human_vs_machine.POSITIVE_IOU_KEY]
            these_negative_iou[k] = this_comparison_dict[
                human_vs_machine.NEGATIVE_IOU_KEY]

        self.assertTrue(numpy.allclose(
            these_positive_iou, POSITIVE_IOU_BY_CHANNEL, atol=TOLERANCE
        ))

        self.assertTrue(numpy.allclose(
            these_negative_iou, NEGATIVE_IOU_BY_CHANNEL, atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
