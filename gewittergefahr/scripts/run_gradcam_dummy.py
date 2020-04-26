"""Creates dummy class-activation map for each storm object.

In this case, the dummy "class-activation map" is actually just the filter
produced by an edge-detector with no learned weights.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import copy
import argparse
import numpy
from keras import backend as K
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import testing_io
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.deep_learning import model_interpretation
from gewittergefahr.deep_learning import gradcam
from gewittergefahr.deep_learning import standalone_utils
from gewittergefahr.scripts import make_dummy_saliency_maps as dummy_saliency

# TODO(thunderhoser): Make this script deal with input tensors other than first.

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
    allow_soft_placement=False
)))

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
EDGE_DETECTOR_MATRIX_2D = dummy_saliency.EDGE_DETECTOR_MATRIX_2D
EDGE_DETECTOR_MATRIX_3D = dummy_saliency.EDGE_DETECTOR_MATRIX_3D

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + dummy_saliency.MODEL_FILE_ARG_NAME, type=str, required=True,
    help=dummy_saliency.MODEL_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + dummy_saliency.EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=dummy_saliency.EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + dummy_saliency.STORM_METAFILE_ARG_NAME, type=str, required=True,
    help=dummy_saliency.STORM_METAFILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + dummy_saliency.NUM_EXAMPLES_ARG_NAME, type=int, required=False,
    default=-1, help=dummy_saliency.NUM_EXAMPLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + dummy_saliency.OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=dummy_saliency.OUTPUT_FILE_HELP_STRING
)


def _run(model_file_name, top_example_dir_name, storm_metafile_name,
         num_examples, output_file_name):
    """Creates dummy class-activation map for each storm object.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param top_example_dir_name: Same.
    :param storm_metafile_name: Same.
    :param num_examples: Same.
    :param output_file_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    model_metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0]
    )

    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_model_metadata(model_metafile_name)
    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    training_option_dict[trainval_io.REFLECTIVITY_MASK_KEY] = None

    print('Reading storm metadata from: "{0:s}"...'.format(storm_metafile_name))
    full_storm_id_strings, storm_times_unix_sec = (
        tracking_io.read_ids_and_times(storm_metafile_name)
    )

    print(SEPARATOR_STRING)

    if 0 < num_examples < len(full_storm_id_strings):
        full_storm_id_strings = full_storm_id_strings[:num_examples]
        storm_times_unix_sec = storm_times_unix_sec[:num_examples]

    example_dict = testing_io.read_predictors_specific_examples(
        top_example_dir_name=top_example_dir_name,
        desired_full_id_strings=full_storm_id_strings,
        desired_times_unix_sec=storm_times_unix_sec,
        option_dict=training_option_dict,
        layer_operation_dicts=model_metadata_dict[cnn.LAYER_OPERATIONS_KEY]
    )
    print(SEPARATOR_STRING)

    predictor_matrices = example_dict[testing_io.INPUT_MATRICES_KEY]
    sounding_pressure_matrix_pa = (
        example_dict[testing_io.SOUNDING_PRESSURES_KEY]
    )

    radar_matrix = predictor_matrices[0]
    num_examples = radar_matrix.shape[0]
    num_channels = radar_matrix.shape[-1]
    num_spatial_dim = len(radar_matrix.shape) - 2

    if num_spatial_dim == 2:
        this_kernel_matrix = numpy.expand_dims(EDGE_DETECTOR_MATRIX_2D, axis=-1)
    else:
        this_kernel_matrix = numpy.expand_dims(EDGE_DETECTOR_MATRIX_3D, axis=-1)

    this_kernel_matrix = numpy.repeat(this_kernel_matrix, num_channels, axis=-1)
    kernel_matrix_unguided = numpy.expand_dims(this_kernel_matrix, axis=-1)
    kernel_matrix_guided = numpy.repeat(
        kernel_matrix_unguided, num_channels, axis=-1
    )

    radar_cam_matrix = numpy.full(radar_matrix.shape[:-1], numpy.nan)
    radar_guided_cam_matrix = numpy.full(radar_matrix.shape, numpy.nan)

    for i in range(num_examples):
        if numpy.mod(i, 10) == 0:
            print((
                'Have created dummy CAM for {0:d} of {1:d} examples...'
            ).format(
                i, num_examples
            ))

        if num_spatial_dim == 2:
            this_cam_matrix = standalone_utils.do_2d_convolution(
                feature_matrix=radar_matrix[i, ...],
                kernel_matrix=kernel_matrix_unguided,
                pad_edges=True, stride_length_px=1
            )

            this_guided_cam_matrix = standalone_utils.do_2d_convolution(
                feature_matrix=radar_matrix[i, ...],
                kernel_matrix=kernel_matrix_guided,
                pad_edges=True, stride_length_px=1
            )
        else:
            this_cam_matrix = standalone_utils.do_3d_convolution(
                feature_matrix=radar_matrix[i, ...],
                kernel_matrix=kernel_matrix_unguided,
                pad_edges=True, stride_length_px=1
            )

            this_guided_cam_matrix = standalone_utils.do_3d_convolution(
                feature_matrix=radar_matrix[i, ...],
                kernel_matrix=kernel_matrix_guided,
                pad_edges=True, stride_length_px=1
            )

        radar_cam_matrix[i, ...] = this_cam_matrix[0, ..., 0]
        radar_guided_cam_matrix[i, ...] = this_guided_cam_matrix[0, ...]

    print('Have created dummy CAM for all {0:d} examples!'.format(
        num_examples
    ))
    print(SEPARATOR_STRING)

    radar_cam_matrix = numpy.absolute(radar_cam_matrix)

    cam_matrices = [
        radar_cam_matrix if k == 0 else None
        for k in range(len(predictor_matrices))
    ]
    guided_cam_matrices = [
        radar_guided_cam_matrix if k == 0 else None
        for k in range(len(predictor_matrices))
    ]
    guided_cam_matrices = trainval_io.separate_shear_and_reflectivity(
        list_of_input_matrices=guided_cam_matrices,
        training_option_dict=training_option_dict
    )

    upsample_refl = training_option_dict[trainval_io.UPSAMPLE_REFLECTIVITY_KEY]

    if upsample_refl:
        cam_matrices[0] = numpy.expand_dims(cam_matrices[0], axis=-1)

        num_channels = predictor_matrices[0].shape[-1]
        cam_matrices[0] = numpy.repeat(
            a=cam_matrices[0], repeats=num_channels, axis=-1
        )

        cam_matrices = trainval_io.separate_shear_and_reflectivity(
            list_of_input_matrices=cam_matrices,
            training_option_dict=training_option_dict
        )

        cam_matrices[0] = cam_matrices[0][..., 0]
        cam_matrices[1] = cam_matrices[1][..., 0]

    print('Denormalizing model inputs...')
    denorm_predictor_matrices = trainval_io.separate_shear_and_reflectivity(
        list_of_input_matrices=copy.deepcopy(predictor_matrices),
        training_option_dict=training_option_dict
    )
    denorm_predictor_matrices = model_interpretation.denormalize_data(
        list_of_input_matrices=denorm_predictor_matrices,
        model_metadata_dict=model_metadata_dict
    )

    print('Writing class-activation maps to file: "{0:s}"...'.format(
        output_file_name
    ))

    gradcam.write_standard_file(
        pickle_file_name=output_file_name,
        denorm_predictor_matrices=denorm_predictor_matrices,
        cam_matrices=cam_matrices, guided_cam_matrices=guided_cam_matrices,
        full_storm_id_strings=full_storm_id_strings,
        storm_times_unix_sec=storm_times_unix_sec,
        model_file_name=model_file_name,
        target_class=0, target_layer_name='none',
        sounding_pressure_matrix_pa=sounding_pressure_matrix_pa
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(
            INPUT_ARG_OBJECT, dummy_saliency.MODEL_FILE_ARG_NAME
        ),
        top_example_dir_name=getattr(
            INPUT_ARG_OBJECT, dummy_saliency.EXAMPLE_DIR_ARG_NAME
        ),
        storm_metafile_name=getattr(
            INPUT_ARG_OBJECT, dummy_saliency.STORM_METAFILE_ARG_NAME
        ),
        num_examples=getattr(
            INPUT_ARG_OBJECT, dummy_saliency.NUM_EXAMPLES_ARG_NAME
        ),
        output_file_name=getattr(
            INPUT_ARG_OBJECT, dummy_saliency.OUTPUT_FILE_ARG_NAME
        )
    )
