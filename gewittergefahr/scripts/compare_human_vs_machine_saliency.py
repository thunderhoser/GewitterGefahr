"""Compares human-generated vs. machine-generated saliency map."""

import os.path
import argparse
import numpy
from gewittergefahr.gg_utils import human_polygons
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import saliency_maps
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.scripts import plot_input_examples

TOLERANCE = 1e-6
TIME_FORMAT = '%Y-%m-%d-%H%M%S'

HUMAN_FILE_ARG_NAME = 'input_human_file_name'
MACHINE_FILE_ARG_NAME = 'input_machine_file_name'
THRESHOLD_ARG_NAME = 'abs_percentile_threshold'

HUMAN_FILE_HELP_STRING = (
    'Path to file with human-generated polygons.  Will be read by '
    '`human_polygons.read_polygons`.')

MACHINE_FILE_HELP_STRING = (
    'Path to file with machine-generated saliency maps.  Will be read by '
    '`saliency_maps.read_file`.')

THRESHOLD_HELP_STRING = (
    'Saliency threshold.  The human polygons will be turned into saliency maps '
    'by assuming that (1) all grid points inside a positive polygon have '
    'saliency >= p and (2) all grid points inside a negative polygon have '
    'saliency <= q, where p is the `{0:s}`th percentile of all positive values '
    'in the machine-generated saliency map and q is the (100 - `{0:s}`th) '
    'percentile of negative values in the machine-generated map.'
).format(THRESHOLD_ARG_NAME)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + HUMAN_FILE_ARG_NAME, type=str, required=True,
    help=HUMAN_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MACHINE_FILE_ARG_NAME, type=str, required=True,
    help=MACHINE_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + THRESHOLD_ARG_NAME, type=float, required=False, default=90.,
    help=THRESHOLD_HELP_STRING)


def _run(input_human_file_name, input_machine_file_name,
         abs_percentile_threshold):
    """Compares human-generated vs. machine-generated saliency map.

    This is effectively the main method.

    :param input_human_file_name: See documentation at top of file.
    :param input_machine_file_name: Same.
    :param abs_percentile_threshold: Same.
    """

    error_checking.assert_is_geq(abs_percentile_threshold, 0.)
    error_checking.assert_is_leq(abs_percentile_threshold, 100.)

    human_polygon_dict = human_polygons.read_polygons(input_human_file_name)
    human_negative_mask_matrix = human_polygon_dict[
        human_polygons.NEGATIVE_MASK_MATRIX_KEY]
    human_positive_mask_matrix = human_polygon_dict[
        human_polygons.POSITIVE_MASK_MATRIX_KEY]

    metadata_dict = plot_input_examples.radar_fig_file_name_to_metadata(
        human_polygon_dict[human_polygons.IMAGE_FILE_KEY]
    )

    full_storm_id_string = metadata_dict[plot_input_examples.FULL_STORM_ID_KEY]
    storm_time_unix_sec = metadata_dict[plot_input_examples.STORM_TIME_KEY]
    radar_field_name = metadata_dict[plot_input_examples.RADAR_FIELD_KEY]
    radar_height_m_asl = metadata_dict[plot_input_examples.RADAR_HEIGHT_KEY]
    layer_operation_dict = metadata_dict[
        plot_input_examples.LAYER_OPERATION_KEY]

    print('Reading data from: "{0:s}"...'.format(input_machine_file_name))

    if full_storm_id_string is None:
        saliency_dict = saliency_maps.read_pmm_file(input_machine_file_name)

        list_of_input_matrices = saliency_dict[
            saliency_maps.MEAN_INPUT_MATRICES_KEY]
        list_of_saliency_matrices = saliency_dict[
            saliency_maps.MEAN_SALIENCY_MATRICES_KEY]
    else:
        saliency_dict = saliency_maps.read_standard_file(
            input_machine_file_name)

        storm_object_index = tracking_utils.find_storm_objects(
            all_id_strings=saliency_dict[saliency_maps.FULL_IDS_KEY],
            all_times_unix_sec=saliency_dict[saliency_maps.STORM_TIMES_KEY],
            id_strings_to_keep=[full_storm_id_string],
            times_to_keep_unix_sec=numpy.array(
                [storm_time_unix_sec], dtype=int
            ),
            allow_missing=False
        )[0]

        list_of_input_matrices = [
            a[storm_object_index, ...]
            for a in saliency_dict[saliency_maps.INPUT_MATRICES_KEY]
        ]

        list_of_saliency_matrices = [
            a[storm_object_index, ...]
            for a in saliency_dict[saliency_maps.SALIENCY_MATRICES_KEY]
        ]

    model_file_name = saliency_dict[saliency_maps.MODEL_FILE_NAME_KEY]
    model_metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0]
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_model_metadata(model_metafile_name)
    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]

    # TODO(thunderhoser): The following code should go in a separate method (and
    # probably a separate file).
    conv_2d3d = model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]
    if conv_2d3d:
        num_radar_dimensions = None
    else:
        num_radar_dimensions = len(list_of_input_matrices[0].shape) - 1

    if num_radar_dimensions is None:
        if radar_field_name == radar_utils.REFL_NAME:
            matrix_index = 0
            field_index = 0
            height_index = numpy.where(
                training_option_dict[trainval_io.RADAR_HEIGHTS_KEY] ==
                radar_height_m_asl
            )[0][0]

            input_matrix = list_of_input_matrices[
                matrix_index
            ][..., height_index, field_index]

            machine_saliency_matrix = list_of_saliency_matrices[
                matrix_index
            ][..., height_index, field_index]
        else:
            matrix_index = 1
            field_index = training_option_dict[
                trainval_io.RADAR_FIELDS_KEY
            ].index(radar_field_name)

            input_matrix = list_of_input_matrices[
                matrix_index
            ][..., field_index]

            machine_saliency_matrix = list_of_saliency_matrices[
                matrix_index
            ][..., field_index]

    elif num_radar_dimensions == 2:
        if layer_operation_dict is None:
            these_flags = numpy.array([
                f == radar_field_name
                for f in training_option_dict[trainval_io.RADAR_FIELDS_KEY]
            ], dtype=bool)

            field_index = numpy.where(numpy.logical_and(
                these_flags,
                training_option_dict[trainval_io.RADAR_HEIGHTS_KEY] ==
                radar_height_m_asl
            ))[0][0]
        else:
            these_flags = numpy.array([
                d == layer_operation_dict
                for d in model_metadata_dict[cnn.LAYER_OPERATIONS_KEY]
            ], dtype=bool)

            field_index = numpy.where(these_flags)[0][0]

        input_matrix = list_of_input_matrices[0][..., field_index]
        machine_saliency_matrix = list_of_saliency_matrices[0][..., field_index]
    else:
        field_index = training_option_dict[trainval_io.RADAR_FIELDS_KEY].index(
            radar_field_name)

        height_index = numpy.where(
            training_option_dict[trainval_io.RADAR_HEIGHTS_KEY] ==
            radar_height_m_asl
        )[0][0]

        input_matrix = list_of_input_matrices[0][..., height_index, field_index]
        machine_saliency_matrix = list_of_saliency_matrices[0][
            ..., height_index, field_index]

    if numpy.any(machine_saliency_matrix > 0):
        positive_saliency_threshold = numpy.percentile(
            machine_saliency_matrix[machine_saliency_matrix > 0],
            abs_percentile_threshold
        )
    else:
        positive_saliency_threshold = TOLERANCE + 0.

    if numpy.any(machine_saliency_matrix < 0):
        negative_saliency_threshold = numpy.percentile(
            machine_saliency_matrix[machine_saliency_matrix < 0],
            100. - abs_percentile_threshold
        )
    else:
        negative_saliency_threshold = -1 * TOLERANCE

    machine_negative_mask_matrix = (
        machine_saliency_matrix <= negative_saliency_threshold
    )

    machine_positive_mask_matrix = (
        machine_saliency_matrix >= positive_saliency_threshold
    )

    print(human_positive_mask_matrix.astype(int))
    print('\n\n')
    print(machine_positive_mask_matrix.astype(int))


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_human_file_name=getattr(INPUT_ARG_OBJECT, HUMAN_FILE_ARG_NAME),
        input_machine_file_name=getattr(
            INPUT_ARG_OBJECT, MACHINE_FILE_ARG_NAME),
        abs_percentile_threshold=getattr(INPUT_ARG_OBJECT, THRESHOLD_ARG_NAME)
    )
