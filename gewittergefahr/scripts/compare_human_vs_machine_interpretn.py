"""Compares human-generated vs. machine-generated interpretation map.

This script handles 3 types of interpretation maps:

- saliency
- Grad-CAM
- guided Grad-CAM
"""

import os.path
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import human_polygons
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import saliency_maps
from gewittergefahr.deep_learning import gradcam
from gewittergefahr.deep_learning import input_examples
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.scripts import plot_input_examples
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import radar_plotting
from gewittergefahr.plotting import imagemagick_utils

# TODO(thunderhoser): Allow this script to deal with soundings at some point?

TOLERANCE = 1e-6
METRES_TO_KM = 0.001
TIME_FORMAT = '%Y-%m-%d-%H%M%S'

MARKER_TYPE = 'o'
MARKER_SIZE = 15
MARKER_EDGE_WIDTH = 1
MARKER_COLOUR = numpy.full(3, 0.)

HUMAN_STRING = 'H'
MACHINE_STRING = 'M'
OVERLAY_FONT_SIZE = 20
OVERLAY_FONT_COLOUR = numpy.full(3, 0.)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

HUMAN_FILE_ARG_NAME = 'input_human_file_name'
MACHINE_FILE_ARG_NAME = 'input_machine_file_name'
GUIDED_GRADCAM_ARG_NAME = 'guided_gradcam_flag'
THRESHOLD_ARG_NAME = 'abs_percentile_threshold'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

HUMAN_FILE_HELP_STRING = (
    'Path to file with human-generated polygons.  Will be read by '
    '`human_polygons.read_polygons`.')

MACHINE_FILE_HELP_STRING = (
    'Path to file with machine-generated interpretation map.  Will be read by '
    '`saliency_maps.read_standard_file`, `saliency_maps.read_pmm_file`, '
    '`gradcam.read_pmm_file`, or `gradcam.read_pmm_file`.')

GUIDED_GRADCAM_HELP_STRING = (
    '[used only if `{0:s}` contains Grad-CAM output] Boolean flag.  If 1, will '
    'compare human polygons with guided Grad-CAM.  If 0, will compare with '
    'simple Grad-CAM.'
).format(MACHINE_FILE_ARG_NAME)

THRESHOLD_HELP_STRING = (
    'Threshold for interpretation quantity (I).  Human polygons will be turned '
    'into interpretation maps by assuming that (1) all grid points in a '
    'positive polygon have I >= p, where p is the `{0:s}`th percentile of '
    'positive values in the machine-generated map; and (2) all grid points '
    'inside a negative polygon have I <= q, where q is the (100 - `{0:s}`)th '
    'percentile of negative values in the machine-generated map.'
).format(THRESHOLD_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + HUMAN_FILE_ARG_NAME, type=str, required=True,
    help=HUMAN_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MACHINE_FILE_ARG_NAME, type=str, required=True,
    help=MACHINE_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + GUIDED_GRADCAM_ARG_NAME, type=int, required=False, default=0,
    help=GUIDED_GRADCAM_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + THRESHOLD_ARG_NAME, type=float, required=False, default=90.,
    help=THRESHOLD_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _compute_iou(machine_mask_matrix, human_mask_matrix):
    """Computes IoU (intersection over union) between human and machine masks.

    :param machine_mask_matrix: Boolean numpy array, representing areas of
        extreme positive or negative interpretation values.
    :param human_mask_matrix: Same but for human.  The two numpy arrays must
        have the same shape.
    :return: iou: Intersection over union between the two masks.
    """

    union_matrix = numpy.logical_or(machine_mask_matrix, human_mask_matrix)
    intersection_matrix = numpy.logical_and(
        machine_mask_matrix, human_mask_matrix)

    return float(numpy.sum(intersection_matrix)) / numpy.sum(union_matrix)


def _plot_comparison(input_matrix, input_metadata_dict, machine_mask_matrix,
                     human_mask_matrix, title_string, output_file_name):
    """Plots comparison between human and machine masks.

    M = number of rows in grid
    N = number of columns in grid

    This method compares areas of extreme positive *or* negative interpretation
    values, not both.

    :param input_matrix: M-by-N numpy array with input data (radar image) over
        which interpretation map was computed.
    :param input_metadata_dict: Dictionary created by
        `plot_input_examples.radar_fig_file_name_to_metadata`.
    :param machine_mask_matrix: M-by-N Boolean numpy array, representing areas
        of extreme interpretation values selon la machine.
    :param human_mask_matrix: M-by-N Boolean numpy array, representing areas of
        extreme interpretation values selon l'humain.
    :param title_string: Title.
    :param output_file_name: Path to output file (figure will be saved here).
    """

    radar_field_name = input_metadata_dict[plot_input_examples.RADAR_FIELD_KEY]
    radar_height_m_asl = input_metadata_dict[
        plot_input_examples.RADAR_HEIGHT_KEY]
    layer_operation_dict = input_metadata_dict[
        plot_input_examples.LAYER_OPERATION_KEY]

    if radar_field_name is None:
        radar_field_name = layer_operation_dict[input_examples.RADAR_FIELD_KEY]

    _, axes_object = pyplot.subplots(
        nrows=1, ncols=1,
        figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    radar_plotting.plot_2d_grid_without_coords(
        field_matrix=input_matrix, field_name=radar_field_name,
        axes_object=axes_object)

    colour_map_object, colour_norm_object = (
        radar_plotting.get_default_colour_scheme(radar_field_name)
    )

    colour_bar_object = plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=input_matrix,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='horizontal', extend_min=True, extend_max=True,
        fraction_of_axis_length=0.8)

    if layer_operation_dict is None:
        label_string = radar_plotting.FIELD_NAME_TO_VERBOSE_DICT[
            radar_field_name]

        if radar_height_m_asl is not None:
            label_string += ' at {0:.2f} km AGL'.format(
                radar_height_m_asl * METRES_TO_KM)
    else:
        label_string = radar_plotting.layer_ops_to_field_and_panel_names(
            list_of_layer_operation_dicts=[layer_operation_dict]
        )[-1][0]

        label_string = label_string.replace('\n', ', ')

    colour_bar_object.set_label(label_string)

    these_rows, these_columns = numpy.where(numpy.logical_and(
        machine_mask_matrix, human_mask_matrix
    ))

    these_rows = these_rows + 0.5
    these_columns = these_columns + 0.5

    if len(these_rows) > 0:
        marker_colour_as_tuple = plotting_utils.colour_from_numpy_to_tuple(
            MARKER_COLOUR)

        axes_object.plot(
            these_columns, these_rows, linestyle='None', marker=MARKER_TYPE,
            markersize=MARKER_SIZE, markeredgewidth=MARKER_EDGE_WIDTH,
            markerfacecolor=marker_colour_as_tuple,
            markeredgecolor=marker_colour_as_tuple)

    these_rows, these_columns = numpy.where(numpy.logical_and(
        machine_mask_matrix, numpy.invert(human_mask_matrix)
    ))

    these_rows = these_rows + 0.5
    these_columns = these_columns + 0.5

    for k in range(len(these_rows)):
        axes_object.text(
            these_columns[k], these_rows[k], MACHINE_STRING,
            fontsize=OVERLAY_FONT_SIZE, color=OVERLAY_FONT_COLOUR,
            fontweight='bold', horizontalalignment='center',
            verticalalignment='center')

    these_rows, these_columns = numpy.where(numpy.logical_and(
        numpy.invert(machine_mask_matrix), human_mask_matrix
    ))

    these_rows = these_rows + 0.5
    these_columns = these_columns + 0.5

    for k in range(len(these_rows)):
        axes_object.text(
            these_columns[k], these_rows[k], HUMAN_STRING,
            fontsize=OVERLAY_FONT_SIZE, color=OVERLAY_FONT_COLOUR,
            fontweight='bold', horizontalalignment='center',
            verticalalignment='center')

    pyplot.title(title_string)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(input_file_name=output_file_name,
                                      output_file_name=output_file_name)


def _get_machine_maps(
        list_of_input_matrices, list_of_interpretation_matrices,
        model_metadata_dict, human_metadata_dict, saliency_flag,
        guided_gradcam_flag):
    """Returns input and interpretation maps for machine.

    :param list_of_input_matrices: 1-D list of input matrices (numpy arrays) for
        model.  These matrices contain predictors.
    :param list_of_interpretation_matrices: 1-D list of interpretation maps
        (numpy arrays) from model.
    :param model_metadata_dict: Dictionary with model metadata (returned by
        `cnn.read_model_metadata`).
    :param human_metadata_dict: Dictionary with metadata for human polygons
        (returned by `plot_input_examples.radar_fig_file_name_to_metadata`).
    :param saliency_flag: Boolean flag.  If True, interpretation type is
        saliency.
    :param guided_gradcam_flag: Boolean flag.  If True, interpretation type is
        guided Grad-CAM..
    :return: input_matrix: Single numpy array with predictors.
    :return: interpretation_matrix: Single numpy array with interpretation
        values.
    """

    radar_field_name = human_metadata_dict[plot_input_examples.RADAR_FIELD_KEY]
    radar_height_m_asl = human_metadata_dict[
        plot_input_examples.RADAR_HEIGHT_KEY]
    layer_operation_dict = human_metadata_dict[
        plot_input_examples.LAYER_OPERATION_KEY]

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]

    if model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]:
        num_radar_dimensions = None
    else:
        num_radar_dimensions = len(list_of_input_matrices[0].shape) - 1

    if num_radar_dimensions is None:
        if radar_field_name == radar_utils.REFL_NAME:
            height_index = numpy.where(
                training_option_dict[trainval_io.RADAR_HEIGHTS_KEY] ==
                radar_height_m_asl
            )[0][0]

            input_matrix = list_of_input_matrices[0][..., height_index, 0]

            if saliency_flag or guided_gradcam_flag:
                interpretation_matrix = (
                    list_of_interpretation_matrices[0][..., height_index, 0]
                )
            else:
                interpretation_matrix = (
                    list_of_interpretation_matrices[0][..., height_index]
                )
        else:
            field_index = training_option_dict[
                trainval_io.RADAR_FIELDS_KEY
            ].index(radar_field_name)

            input_matrix = list_of_input_matrices[1][..., field_index]

            if saliency_flag or guided_gradcam_flag:
                interpretation_matrix = (
                    list_of_interpretation_matrices[1][..., field_index]
                )
            else:
                interpretation_matrix = list_of_interpretation_matrices[1]

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

        if saliency_flag or guided_gradcam_flag:
            interpretation_matrix = (
                list_of_interpretation_matrices[0][..., field_index]
            )
        else:
            interpretation_matrix = list_of_interpretation_matrices[0]

    else:
        field_index = training_option_dict[trainval_io.RADAR_FIELDS_KEY].index(
            radar_field_name)

        height_index = numpy.where(
            training_option_dict[trainval_io.RADAR_HEIGHTS_KEY] ==
            radar_height_m_asl
        )[0][0]

        input_matrix = list_of_input_matrices[0][..., height_index, field_index]

        if saliency_flag or guided_gradcam_flag:
            interpretation_matrix = (
                list_of_interpretation_matrices[0][
                    ..., height_index, field_index]
            )
        else:
            interpretation_matrix = (
                list_of_interpretation_matrices[0][..., height_index]
            )

    return input_matrix, interpretation_matrix


def _run(input_human_file_name, input_machine_file_name, guided_gradcam_flag,
         abs_percentile_threshold, output_dir_name):
    """Compares human-generated vs. machine-generated interpretation map.

    This is effectively the main method.

    :param input_human_file_name: See documentation at top of file.
    :param input_machine_file_name: Same.
    :param guided_gradcam_flag: Same.
    :param abs_percentile_threshold: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    error_checking.assert_is_geq(abs_percentile_threshold, 0.)
    error_checking.assert_is_leq(abs_percentile_threshold, 100.)

    print('Reading data from: "{0:s}"...'.format(input_human_file_name))
    human_polygon_dict = human_polygons.read_polygons(input_human_file_name)

    human_positive_mask_matrix = human_polygon_dict[
        human_polygons.POSITIVE_MASK_MATRIX_KEY]
    human_negative_mask_matrix = human_polygon_dict[
        human_polygons.NEGATIVE_MASK_MATRIX_KEY]

    human_metadata_dict = plot_input_examples.radar_fig_file_name_to_metadata(
        human_polygon_dict[human_polygons.IMAGE_FILE_KEY]
    )

    pmm_flag = human_metadata_dict[plot_input_examples.PMM_FLAG_KEY]
    full_storm_id_string = human_metadata_dict[
        plot_input_examples.FULL_STORM_ID_KEY]
    storm_time_unix_sec = human_metadata_dict[
        plot_input_examples.STORM_TIME_KEY]

    print('Reading data from: "{0:s}"...'.format(input_machine_file_name))

    if pmm_flag:
        try:
            saliency_dict = saliency_maps.read_pmm_file(input_machine_file_name)

            list_of_input_matrices = saliency_dict.pop(
                saliency_maps.MEAN_INPUT_MATRICES_KEY)
            list_of_interpretation_matrices = saliency_dict.pop(
                saliency_maps.MEAN_SALIENCY_MATRICES_KEY)

            saliency_flag = True
            model_file_name = saliency_dict[saliency_maps.MODEL_FILE_KEY]
        except ValueError:
            gradcam_dict = gradcam.read_pmm_file(input_machine_file_name)

            list_of_input_matrices = gradcam_dict.pop(
                gradcam.MEAN_INPUT_MATRICES_KEY)

            if guided_gradcam_flag:
                list_of_interpretation_matrices = [
                    gradcam_dict.pop(gradcam.MEAN_GUIDED_GRADCAM_KEY)
                ]
            else:
                list_of_interpretation_matrices = [
                    gradcam_dict.pop(gradcam.MEAN_CLASS_ACTIVATIONS_KEY)
                ]

            saliency_flag = False
            model_file_name = gradcam_dict[gradcam.MODEL_FILE_KEY]
    else:
        try:
            saliency_dict = saliency_maps.read_standard_file(
                input_machine_file_name)

            list_of_input_matrices = saliency_dict.pop(
                saliency_maps.INPUT_MATRICES_KEY)
            list_of_interpretation_matrices = saliency_dict.pop(
                saliency_maps.SALIENCY_MATRICES_KEY)

            saliency_flag = True
            all_full_id_strings = saliency_dict[saliency_maps.FULL_IDS_KEY]
            all_times_unix_sec = saliency_dict[saliency_maps.STORM_TIMES_KEY]
            model_file_name = saliency_dict[saliency_maps.MODEL_FILE_KEY]
        except ValueError:
            gradcam_dict = gradcam.read_standard_file(input_machine_file_name)

            list_of_input_matrices = gradcam_dict.pop(
                gradcam.INPUT_MATRICES_KEY)

            if guided_gradcam_flag:
                list_of_interpretation_matrices = [
                    gradcam_dict.pop(gradcam.GUIDED_GRADCAM_KEY)
                ]
            else:
                list_of_interpretation_matrices = [
                    gradcam_dict.pop(gradcam.CLASS_ACTIVATIONS_KEY)
                ]

            saliency_flag = False
            all_full_id_strings = gradcam_dict[gradcam.FULL_IDS_KEY]
            all_times_unix_sec = gradcam_dict[gradcam.STORM_TIMES_KEY]
            model_file_name = gradcam_dict[gradcam.MODEL_FILE_KEY]

        storm_object_index = tracking_utils.find_storm_objects(
            all_id_strings=all_full_id_strings,
            all_times_unix_sec=all_times_unix_sec,
            id_strings_to_keep=[full_storm_id_string],
            times_to_keep_unix_sec=numpy.array(
                [storm_time_unix_sec], dtype=int
            ),
            allow_missing=False
        )[0]

        list_of_input_matrices = [
            a[storm_object_index, ...] for a in list_of_input_matrices
        ]
        list_of_interpretation_matrices = [
            a[storm_object_index, ...] for a in list_of_interpretation_matrices
        ]

    model_metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0]
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_model_metadata(model_metafile_name)

    input_matrix, machine_interpretation_matrix = _get_machine_maps(
        list_of_input_matrices=list_of_input_matrices,
        list_of_interpretation_matrices=list_of_interpretation_matrices,
        model_metadata_dict=model_metadata_dict,
        human_metadata_dict=human_metadata_dict,
        saliency_flag=saliency_flag, guided_gradcam_flag=guided_gradcam_flag)

    input_matrix = numpy.flip(input_matrix, axis=0)
    machine_interpretation_matrix = numpy.flip(
        machine_interpretation_matrix, axis=0)

    if numpy.any(machine_interpretation_matrix > 0):
        positive_threshold = numpy.percentile(
            machine_interpretation_matrix[machine_interpretation_matrix > 0],
            abs_percentile_threshold
        )
    else:
        positive_threshold = TOLERANCE + 0.

    machine_positive_mask_matrix = (
        machine_interpretation_matrix >= positive_threshold
    )

    positive_iou = _compute_iou(
        machine_mask_matrix=machine_positive_mask_matrix,
        human_mask_matrix=human_positive_mask_matrix)

    print('IoU for positive values = {0:.3f}'.format(positive_iou))

    this_title_string = (
        'Positive values ... threshold = {0:.2e} ... IoU = {1:.3f}'
    ).format(positive_threshold, positive_iou)
    this_file_name = '{0:s}/positive_comparison.jpg'.format(output_dir_name)

    _plot_comparison(
        input_matrix=input_matrix, input_metadata_dict=human_metadata_dict,
        machine_mask_matrix=machine_positive_mask_matrix,
        human_mask_matrix=human_positive_mask_matrix,
        title_string=this_title_string, output_file_name=this_file_name)

    if saliency_flag or guided_gradcam_flag:
        if numpy.any(machine_interpretation_matrix < 0):
            negative_threshold = numpy.percentile(
                machine_interpretation_matrix[machine_interpretation_matrix < 0],
                100. - abs_percentile_threshold
            )
        else:
            negative_threshold = -1 * TOLERANCE

        machine_negative_mask_matrix = (
            machine_interpretation_matrix <= negative_threshold
        )

        negative_iou = _compute_iou(
            machine_mask_matrix=machine_negative_mask_matrix,
            human_mask_matrix=human_negative_mask_matrix)

        print('IoU for negative values = {0:.3f}'.format(negative_iou))

        this_title_string = (
            'Negative values ... threshold = {0:.2e} ... IoU = {1:.3f}'
        ).format(negative_threshold, negative_iou)
        this_file_name = '{0:s}/negative_comparison.jpg'.format(output_dir_name)

        _plot_comparison(
            input_matrix=input_matrix, input_metadata_dict=human_metadata_dict,
            machine_mask_matrix=machine_negative_mask_matrix,
            human_mask_matrix=human_negative_mask_matrix,
            title_string=this_title_string, output_file_name=this_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_human_file_name=getattr(INPUT_ARG_OBJECT, HUMAN_FILE_ARG_NAME),
        input_machine_file_name=getattr(
            INPUT_ARG_OBJECT, MACHINE_FILE_ARG_NAME),
        guided_gradcam_flag=bool(getattr(
            INPUT_ARG_OBJECT, GUIDED_GRADCAM_ARG_NAME)),
        abs_percentile_threshold=getattr(INPUT_ARG_OBJECT, THRESHOLD_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
