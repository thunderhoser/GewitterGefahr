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
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import saliency_maps
from gewittergefahr.deep_learning import gradcam
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.scripts import plot_input_examples
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import radar_plotting

# TODO(thunderhoser): Allow this script to deal with soundings at some point?
# TODO(thunderhoser): Add unit tests!

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

MACHINE_POSITIVE_MASK_KEY = 'machine_positive_mask_matrix'
POSITIVE_IOU_KEY = 'positive_iou'
MACHINE_NEGATIVE_MASK_KEY = 'machine_negative_mask_matrix'
NEGATIVE_IOU_KEY = 'negative_iou'

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


def _plot_comparison(
        input_matrix, model_metadata_dict, machine_mask_matrix,
        human_mask_matrix, iou_by_channel, positive_flag, output_file_name):
    """Plots comparison between human and machine interpretation maps.

    M = number of rows in grid (physical space)
    N = number of columns in grid (physical space)
    C = number of channels

    :param input_matrix: M-by-N-by-C numpy array of input values (predictors).
    :param model_metadata_dict: Dictionary returned by
        `cnn.read_model_metadata`.
    :param machine_mask_matrix: M-by-N-by-C numpy array of Boolean flags,
        indicating where machine interpretation value is strongly positive or
        negative.
    :param human_mask_matrix: Same but for human.
    :param iou_by_channel: length-C numpy array of IoU values (intersection over
        union) between human and machine masks.
    :param positive_flag: Boolean flag.  If True (False), masks indicate where
        interpretation value is strongly positive (negative).
    :param output_file_name: Path to output file (figure will be saved here).
    """

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    list_of_layer_operation_dicts = model_metadata_dict[
        cnn.LAYER_OPERATIONS_KEY]

    if list_of_layer_operation_dicts is None:
        field_name_by_panel = training_option_dict[trainval_io.RADAR_FIELDS_KEY]

        panel_names = radar_plotting.radar_fields_and_heights_to_panel_names(
            field_names=field_name_by_panel,
            heights_m_agl=training_option_dict[trainval_io.RADAR_HEIGHTS_KEY]
        )

        plot_colour_bar_by_panel = numpy.full(
            len(field_name_by_panel), True, dtype=bool
        )
    else:
        field_name_by_panel, panel_names = (
            radar_plotting.layer_ops_to_field_and_panel_names(
                list_of_layer_operation_dicts=list_of_layer_operation_dicts
            )
        )

        plot_colour_bar_by_panel = numpy.full(
            len(field_name_by_panel), False, dtype=bool
        )

        plot_colour_bar_by_panel[2::3] = True

    num_panels = len(field_name_by_panel)
    num_panel_rows = int(numpy.floor(
        numpy.sqrt(num_panels)
    ))
    num_panel_columns = int(numpy.ceil(
        float(num_panels) / num_panel_rows
    ))

    for k in range(num_panels):
        panel_names[k] += '\n{0:s} IoU = {1:.3f}'.format(
            'Positive' if positive_flag else 'Negative',
            iou_by_channel[k]
        )

    _, axes_object_matrix = radar_plotting.plot_many_2d_grids_without_coords(
        field_matrix=numpy.flip(input_matrix, axis=0),
        field_name_by_panel=field_name_by_panel, panel_names=panel_names,
        num_panel_rows=num_panel_rows,
        plot_colour_bar_by_panel=plot_colour_bar_by_panel, font_size=14,
        row_major=False)

    for k in range(num_panels):

        # TODO(thunderhoser): Modularize this shit.
        i, j = numpy.unravel_index(
            k, (num_panel_rows, num_panel_columns), order='F'
        )

        these_grid_rows, these_grid_columns = numpy.where(numpy.logical_and(
            numpy.flip(machine_mask_matrix[..., k], axis=0),
            numpy.flip(human_mask_matrix[..., k], axis=0)
        ))

        these_grid_rows = these_grid_rows + 0.5
        these_grid_columns = these_grid_columns + 0.5

        if len(these_grid_rows) > 0:
            marker_colour_as_tuple = plotting_utils.colour_from_numpy_to_tuple(
                MARKER_COLOUR)

            axes_object_matrix[i, j].plot(
                these_grid_columns, these_grid_rows, linestyle='None',
                marker=MARKER_TYPE, markersize=MARKER_SIZE,
                markeredgewidth=MARKER_EDGE_WIDTH,
                markerfacecolor=marker_colour_as_tuple,
                markeredgecolor=marker_colour_as_tuple)

        these_grid_rows, these_grid_columns = numpy.where(numpy.logical_and(
            numpy.flip(machine_mask_matrix[..., k], axis=0),
            numpy.invert(numpy.flip(human_mask_matrix[..., k], axis=0))
        ))

        these_grid_rows = these_grid_rows + 0.5
        these_grid_columns = these_grid_columns + 0.5

        for m in range(len(these_grid_rows)):
            axes_object_matrix[i, j].text(
                these_grid_columns[m], these_grid_rows[m], MACHINE_STRING,
                fontsize=OVERLAY_FONT_SIZE, color=OVERLAY_FONT_COLOUR,
                fontweight='bold', horizontalalignment='center',
                verticalalignment='center')

        print(machine_mask_matrix.shape)
        print(machine_mask_matrix.dtype)
        print(machine_mask_matrix[..., k].dtype)

        these_grid_rows, these_grid_columns = numpy.where(numpy.logical_and(
            numpy.invert(numpy.flip(machine_mask_matrix[..., k], axis=0)),
            numpy.flip(human_mask_matrix[..., k], axis=0)
        ))

        these_grid_rows = these_grid_rows + 0.5
        these_grid_columns = these_grid_columns + 0.5

        for m in range(len(these_grid_rows)):
            axes_object_matrix[i, j].text(
                these_grid_columns[m], these_grid_rows[m], HUMAN_STRING,
                fontsize=OVERLAY_FONT_SIZE, color=OVERLAY_FONT_COLOUR,
                fontweight='bold', horizontalalignment='center',
                verticalalignment='center')

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI,
                   pad_inches=0., bbox_inches='tight')
    pyplot.close()


def _reshape_human_maps(model_metadata_dict, positive_mask_matrix_4d,
                        negative_mask_matrix_4d):
    """Reshapes human interpretation maps to match machine interpretation maps.

    M = number of rows in grid (physical space)
    N = number of columns in grid (physical space)
    J = number of panel rows in image
    K = number of panel columns in image
    C = J * K = number of channels

    :param model_metadata_dict: Dictionary returned by
        `cnn.read_model_metadata`.
    :param positive_mask_matrix_4d: J-by-K-by-M-by-N numpy array of Boolean
        flags.
    :param negative_mask_matrix_4d: Same, except this may be None.
    :return: positive_mask_matrix_3d: M-by-N-by-C numpy array of Boolean flags.
    :return: negative_mask_matrix_3d: Same, except this may be None.
    :raises: TypeError: if model performs 2-D and 3-D convolution.
    :raises: ValueError: if number of channels in masks != number of input
        channels to the model.
    """

    if model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]:
        raise TypeError(
            'This script cannot handle models that perform 2-D and 3-D'
            'convolution.'
        )

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    list_of_layer_operation_dicts = model_metadata_dict[
        cnn.LAYER_OPERATIONS_KEY]

    if list_of_layer_operation_dicts is None:
        num_machine_channels = len(
            training_option_dict[trainval_io.RADAR_FIELDS_KEY]
        )
    else:
        num_machine_channels = len(list_of_layer_operation_dicts)

    num_human_channels = int(numpy.round(
        positive_mask_matrix_4d.shape[0] * positive_mask_matrix_4d.shape[1]
    ))

    if num_machine_channels != num_human_channels:
        error_string = (
            'Number of channels in human masks ({0:d}) != number of input '
            'channels to model ({1:d}).'
        ).format(num_human_channels, num_machine_channels)

        raise ValueError(error_string)

    this_shape = (num_human_channels,) + positive_mask_matrix_4d.shape[2:]

    positive_mask_matrix_3d = numpy.reshape(
        a=positive_mask_matrix_4d, newshape=this_shape, order='F'
    )
    positive_mask_matrix_3d = numpy.swapaxes(positive_mask_matrix_3d, 0, 2)

    if negative_mask_matrix_4d is None:
        negative_mask_matrix_3d = None
    else:
        negative_mask_matrix_3d = numpy.reshape(
            a=negative_mask_matrix_4d, newshape=this_shape, order='F'
        )
        negative_mask_matrix_3d = numpy.swapaxes(negative_mask_matrix_3d, 0, 2)

    return positive_mask_matrix_3d, negative_mask_matrix_3d


def _do_comparison_one_channel(
        machine_interpretation_matrix, abs_percentile_threshold,
        human_positive_mask_matrix, human_negative_mask_matrix=None):
    """Compares human and machine masks for one channel.

    M = number of rows in grid (physical space)
    N = number of columns in grid (physical space)

    :param machine_interpretation_matrix: M-by-N numpy array of
        interpretation values.
    :param abs_percentile_threshold: See documentation at top of file.  This
        threshold will be applied to `machine_interpretation_matrix` to turn
        into one or two masks.
    :param human_positive_mask_matrix: M-by-N numpy array of Boolean flags,
        indicating where the human thinks the interpretation value is strongly
        positive.
    :param human_negative_mask_matrix: Same but for strongly negative.  If
        this is None, will compare human vs. machine only for strongly positive
        values.
    :return: comparison_dict: Dictionary with the following keys.
    comparison_dict['machine_positive_mask_matrix']: Same as
        `human_positive_mask_matrix` but for the machine.
    comparison_dict['positive_iou']: IoU (intersection over union) between
        positive masks for human and machine.
    comparison_dict['machine_negative_mask_matrix']: Same as
        `human_negative_mask_matrix` but for the machine.  May be None.
    comparison_dict['negative_iou']: IoU (intersection over union) between
        negative masks for human and machine.  May be None.
    """

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

    comparison_dict = {
        MACHINE_POSITIVE_MASK_KEY: machine_positive_mask_matrix,
        POSITIVE_IOU_KEY: positive_iou,
        MACHINE_NEGATIVE_MASK_KEY: None,
        NEGATIVE_IOU_KEY: None
    }

    if human_negative_mask_matrix is None:
        return comparison_dict

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

    comparison_dict[MACHINE_NEGATIVE_MASK_KEY] = machine_negative_mask_matrix
    comparison_dict[NEGATIVE_IOU_KEY] = negative_iou
    return comparison_dict


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

    # TODO(thunderhoser): Put this metadata in the file itself!
    # human_metadata_dict = plot_input_examples.radar_fig_file_name_to_metadata(
    #     human_polygon_dict[human_polygons.IMAGE_FILE_KEY]
    # )
    #
    # pmm_flag = human_metadata_dict[plot_input_examples.PMM_FLAG_KEY]
    # full_storm_id_string = human_metadata_dict[
    #     plot_input_examples.FULL_STORM_ID_KEY]
    # storm_time_unix_sec = human_metadata_dict[
    #     plot_input_examples.STORM_TIME_KEY]

    pmm_flag = True
    full_storm_id_string = None
    storm_time_unix_sec = None

    print('Reading data from: "{0:s}"...'.format(input_machine_file_name))

    machine_channel_indices = numpy.array([2, 8], dtype=int)

    if pmm_flag:
        try:
            saliency_dict = saliency_maps.read_pmm_file(input_machine_file_name)
            saliency_flag = True
            model_file_name = saliency_dict[saliency_maps.MODEL_FILE_KEY]

            input_matrix = saliency_dict.pop(
                saliency_maps.MEAN_INPUT_MATRICES_KEY
            )[0][..., machine_channel_indices]

            machine_interpretation_matrix = saliency_dict.pop(
                saliency_maps.MEAN_SALIENCY_MATRICES_KEY
            )[0][..., machine_channel_indices]
        except ValueError:
            gradcam_dict = gradcam.read_pmm_file(input_machine_file_name)
            saliency_flag = False
            model_file_name = gradcam_dict[gradcam.MODEL_FILE_KEY]

            input_matrix = gradcam_dict.pop(
                gradcam.MEAN_INPUT_MATRICES_KEY
            )[0][..., machine_channel_indices]

            if guided_gradcam_flag:
                machine_interpretation_matrix = gradcam_dict.pop(
                    gradcam.MEAN_GUIDED_GRADCAM_KEY
                )[..., machine_channel_indices]
            else:
                machine_interpretation_matrix = gradcam_dict.pop(
                    gradcam.MEAN_CLASS_ACTIVATIONS_KEY
                )[..., machine_channel_indices]
    else:
        try:
            saliency_dict = saliency_maps.read_standard_file(
                input_machine_file_name)

            saliency_flag = True
            all_full_id_strings = saliency_dict[saliency_maps.FULL_IDS_KEY]
            all_times_unix_sec = saliency_dict[saliency_maps.STORM_TIMES_KEY]
            model_file_name = saliency_dict[saliency_maps.MODEL_FILE_KEY]

            input_matrix = saliency_dict.pop(
                saliency_maps.INPUT_MATRICES_KEY
            )[0][..., machine_channel_indices]

            machine_interpretation_matrix = saliency_dict.pop(
                saliency_maps.SALIENCY_MATRICES_KEY
            )[0][..., machine_channel_indices]
        except ValueError:
            gradcam_dict = gradcam.read_standard_file(input_machine_file_name)

            saliency_flag = False
            all_full_id_strings = gradcam_dict[gradcam.FULL_IDS_KEY]
            all_times_unix_sec = gradcam_dict[gradcam.STORM_TIMES_KEY]
            model_file_name = gradcam_dict[gradcam.MODEL_FILE_KEY]

            input_matrix = gradcam_dict.pop(
                gradcam.INPUT_MATRICES_KEY
            )[0][..., machine_channel_indices]

            if guided_gradcam_flag:
                machine_interpretation_matrix = gradcam_dict.pop(
                    gradcam.GUIDED_GRADCAM_KEY
                )[..., machine_channel_indices]
            else:
                machine_interpretation_matrix = gradcam_dict.pop(
                    gradcam.CLASS_ACTIVATIONS_KEY
                )[..., machine_channel_indices]

        storm_object_index = tracking_utils.find_storm_objects(
            all_id_strings=all_full_id_strings,
            all_times_unix_sec=all_times_unix_sec,
            id_strings_to_keep=[full_storm_id_string],
            times_to_keep_unix_sec=numpy.array(
                [storm_time_unix_sec], dtype=int
            ),
            allow_missing=False
        )[0]

        input_matrix = input_matrix[storm_object_index, ...]
        machine_interpretation_matrix = machine_interpretation_matrix[
            storm_object_index, ...]

    if not (saliency_flag or guided_gradcam_flag):
        human_negative_mask_matrix = None

    model_metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0]
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_model_metadata(model_metafile_name)
    model_metadata_dict[cnn.LAYER_OPERATIONS_KEY] = [
        model_metadata_dict[cnn.LAYER_OPERATIONS_KEY][k]
        for k in machine_channel_indices
    ]

    human_positive_mask_matrix, human_negative_mask_matrix = (
        _reshape_human_maps(
            model_metadata_dict=model_metadata_dict,
            positive_mask_matrix_4d=human_positive_mask_matrix,
            negative_mask_matrix_4d=human_negative_mask_matrix)
    )

    num_channels = human_positive_mask_matrix.shape[-1]
    machine_positive_mask_matrix = numpy.full(
        human_positive_mask_matrix.shape, numpy.nan)
    positive_iou_by_channel = numpy.full(num_channels, numpy.nan)

    if human_negative_mask_matrix is not None:
        machine_negative_mask_matrix = numpy.full(
            human_negative_mask_matrix.shape, numpy.nan)
        negative_iou_by_channel = numpy.full(num_channels, numpy.nan)

    for k in range(num_channels):
        this_negative_matrix = (
            None if human_negative_mask_matrix is None
            else human_negative_mask_matrix[..., k]
        )

        this_comparison_dict = _do_comparison_one_channel(
            machine_interpretation_matrix=machine_interpretation_matrix[..., k],
            abs_percentile_threshold=abs_percentile_threshold,
            human_positive_mask_matrix=human_positive_mask_matrix[..., k],
            human_negative_mask_matrix=this_negative_matrix)

        machine_positive_mask_matrix[..., k] = this_comparison_dict[
            MACHINE_POSITIVE_MASK_KEY]
        positive_iou_by_channel[k] = this_comparison_dict[POSITIVE_IOU_KEY]

        if human_negative_mask_matrix is None:
            continue

        machine_negative_mask_matrix[..., k] = this_comparison_dict[
            MACHINE_NEGATIVE_MASK_KEY]
        negative_iou_by_channel[k] = this_comparison_dict[NEGATIVE_IOU_KEY]

    this_file_name = '{0:s}/positive_comparison.jpg'.format(output_dir_name)
    _plot_comparison(
        input_matrix=input_matrix, model_metadata_dict=model_metadata_dict,
        machine_mask_matrix=machine_positive_mask_matrix,
        human_mask_matrix=human_positive_mask_matrix,
        iou_by_channel=positive_iou_by_channel,
        positive_flag=True, output_file_name=this_file_name)

    if human_negative_mask_matrix is None:
        return

    this_file_name = '{0:s}/negative_comparison.jpg'.format(output_dir_name)
    _plot_comparison(
        input_matrix=input_matrix, model_metadata_dict=model_metadata_dict,
        machine_mask_matrix=machine_negative_mask_matrix,
        human_mask_matrix=human_negative_mask_matrix,
        iou_by_channel=negative_iou_by_channel,
        positive_flag=False, output_file_name=this_file_name)


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
