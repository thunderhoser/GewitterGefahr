"""Makes figure with class-activation maps for MYRORSS model."""

import os
import argparse
import numpy
from PIL import Image
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import general_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import gradcam
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import cam_plotting
from gewittergefahr.plotting import imagemagick_utils
from gewittergefahr.scripts import plot_input_examples as plot_examples

REFL_HEIGHTS_M_AGL = numpy.array([2000, 6000, 10000], dtype=int)

MODEL_FILE_KEY = gradcam.MODEL_FILE_KEY
MEAN_PREDICTOR_MATRICES_KEY = gradcam.MEAN_PREDICTOR_MATRICES_KEY
MEAN_CAM_MATRICES_KEY = gradcam.MEAN_CAM_MATRICES_KEY

MIN_COLOUR_VALUE_LOG10 = -2.

COLOUR_BAR_LENGTH = 0.25
PANEL_NAME_FONT_SIZE = 30
COLOUR_BAR_FONT_SIZE = 25

CONVERT_EXE_NAME = '/usr/bin/convert'
TITLE_FONT_SIZE = 150
TITLE_FONT_NAME = 'DejaVu-Sans-Bold'

FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

INPUT_FILES_ARG_NAME = 'input_gradcam_file_names'
COMPOSITE_NAMES_ARG_NAME = 'composite_names'
COLOUR_MAP_ARG_NAME = 'colour_map_name'
MAX_VALUE_ARG_NAME = 'max_colour_value'
NUM_CONTOURS_ARG_NAME = 'num_contours'
SMOOTHING_RADIUS_ARG_NAME = 'smoothing_radius_grid_cells'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILES_HELP_STRING = (
    'List of Grad-CAM files (each will be read by `gradcam.read_file`).'
)
COMPOSITE_NAMES_HELP_STRING = (
    'List of composite names (one for each Grad-CAM file).  This list must be '
    'space-separated, but after reading the list, underscores within each item '
    'will be replaced by spaces.'
)
COLOUR_MAP_HELP_STRING = (
    'Colour scheme for class activation.  Must be accepted by '
    '`matplotlib.pyplot.get_cmap`.'
)

MAX_VALUE_HELP_STRING = 'Max class activation in colour scheme.'
NUM_CONTOURS_HELP_STRING = 'Number of class-activation contours.'

SMOOTHING_RADIUS_HELP_STRING = (
    'e-folding radius for Gaussian smoother (num grid cells).  If you do not '
    'want to smooth class-activation maps, make this negative.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + COMPOSITE_NAMES_ARG_NAME, type=str, nargs='+', required=True,
    help=COMPOSITE_NAMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + COLOUR_MAP_ARG_NAME, type=str, required=False, default='binary',
    help=COLOUR_MAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_VALUE_ARG_NAME, type=float, required=False, default=10 ** 1.5,
    help=MAX_VALUE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_CONTOURS_ARG_NAME, type=int, required=False,
    default=15, help=NUM_CONTOURS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SMOOTHING_RADIUS_ARG_NAME, type=float, required=False,
    default=3., help=SMOOTHING_RADIUS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _smooth_maps(cam_matrices, smoothing_radius_grid_cells):
    """Smooths class-activation maps via Gaussian filter.

    T = number of input tensors to the model

    :param cam_matrices: length-T list of numpy arrays.
    :param smoothing_radius_grid_cells: e-folding radius (number of grid cells).
    :return: cam_matrices: Smoothed version of input.
    """

    print((
        'Smoothing class-activation maps with Gaussian filter (e-folding radius'
        ' of {0:.1f} grid cells)...'
    ).format(
        smoothing_radius_grid_cells
    ))

    num_matrices = len(cam_matrices)

    for j in range(num_matrices):
        if cam_matrices[j] is None:
            continue

        num_examples = cam_matrices[j].shape[0]

        for i in range(num_examples):
            cam_matrices[j][i, ...] = general_utils.apply_gaussian_filter(
                input_matrix=cam_matrices[j][i, ...],
                e_folding_radius_grid_cells=smoothing_radius_grid_cells
            )

    return cam_matrices


def _read_one_composite(gradcam_file_name, smoothing_radius_grid_cells):
    """Reads class-activation map for one composite.

    T = number of input tensors for model

    :param gradcam_file_name: Path to input file (will be read by
        `gradcam.read_file`).
    :param smoothing_radius_grid_cells: Radius for Gaussian smoother, used only
        for class-activation map.
    :return: mean_predictor_matrices: length-T list of numpy arrays, where the
        [i]th item has dimensions of the [i]th input tensor for the model.
    :return: mean_cam_matrices: length-T list of numpy arrays with mean class
        activations.  The [i]th item should have dimensions of the [i]th input
        tensor to the model, except missing the last axis, which is the channel
        axis.
    :return: model_metadata_dict: Dictionary returned by
        `cnn.read_model_metadata`.
    """

    print('Reading data from: "{0:s}"...'.format(gradcam_file_name))
    gradcam_dict = gradcam.read_file(gradcam_file_name)[0]

    mean_predictor_matrices = gradcam_dict[MEAN_PREDICTOR_MATRICES_KEY]
    mean_cam_matrices = gradcam_dict[MEAN_CAM_MATRICES_KEY]
    num_matrices = len(mean_predictor_matrices)

    for i in range(num_matrices):
        mean_predictor_matrices[i] = numpy.expand_dims(
            mean_predictor_matrices[i], axis=0
        )

        if mean_cam_matrices[i] is None:
            continue

        mean_cam_matrices[i] = numpy.expand_dims(mean_cam_matrices[i], axis=0)

    if smoothing_radius_grid_cells is not None:
        mean_cam_matrices = _smooth_maps(
            cam_matrices=mean_cam_matrices,
            smoothing_radius_grid_cells=smoothing_radius_grid_cells
        )

    model_file_name = gradcam_dict[MODEL_FILE_KEY]
    model_metafile_name = cnn.find_metafile(model_file_name)

    print('Reading CNN metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_model_metadata(model_metafile_name)
    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    training_option_dict[trainval_io.UPSAMPLE_REFLECTIVITY_KEY] = False

    all_refl_heights_m_agl = training_option_dict[trainval_io.RADAR_HEIGHTS_KEY]
    good_flags = numpy.array(
        [h in REFL_HEIGHTS_M_AGL for h in all_refl_heights_m_agl], dtype=bool
    )
    good_indices = numpy.where(good_flags)[0]

    mean_predictor_matrices[0] = (
        mean_predictor_matrices[0][..., good_indices, :]
    )
    mean_cam_matrices[0] = mean_cam_matrices[0][..., good_indices]

    training_option_dict[trainval_io.RADAR_HEIGHTS_KEY] = REFL_HEIGHTS_M_AGL
    model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY] = training_option_dict

    return mean_predictor_matrices, mean_cam_matrices, model_metadata_dict


def _overlay_text(
        image_file_name, x_offset_from_center_px, y_offset_from_top_px,
        text_string):
    """Overlays text on image.

    :param image_file_name: Path to image file.
    :param x_offset_from_center_px: Center-relative x-coordinate (pixels).
    :param y_offset_from_top_px: Top-relative y-coordinate (pixels).
    :param text_string: String to overlay.
    :raises: ValueError: if ImageMagick command (which is ultimately a Unix
        command) fails.
    """

    command_string = (
        '"{0:s}" "{1:s}" -gravity north -pointsize {2:d} -font "{3:s}" '
        '-fill "rgb(0, 0, 0)" -annotate {4:+d}{5:+d} "{6:s}" "{1:s}"'
    ).format(
        CONVERT_EXE_NAME, image_file_name, TITLE_FONT_SIZE, TITLE_FONT_NAME,
        x_offset_from_center_px, y_offset_from_top_px, text_string
    )

    exit_code = os.system(command_string)
    if exit_code == 0:
        return

    raise ValueError(imagemagick_utils.ERROR_STRING)


def _plot_one_composite(
        gradcam_file_name, composite_name_abbrev, composite_name_verbose,
        colour_map_object, max_colour_value, num_contours,
        smoothing_radius_grid_cells, output_dir_name):
    """Plots class-activation map for one composite.

    :param gradcam_file_name: Path to input file (will be read by
        `gradcam.read_file`).
    :param composite_name_abbrev: Abbrev composite name (will be used in file
        names).
    :param composite_name_verbose: Verbose composite name (will be used in
        figure title).
    :param colour_map_object: See documentation at top of file.
    :param max_colour_value: Same.
    :param num_contours: Same.
    :param smoothing_radius_grid_cells: Same.
    :param output_dir_name: Name of output directory (figures will be saved
        here).
    :return: main_figure_file_name: Path to main image file created by this
        method.
    """

    mean_predictor_matrices, mean_cam_matrices, model_metadata_dict = (
        _read_one_composite(
            gradcam_file_name=gradcam_file_name,
            smoothing_radius_grid_cells=smoothing_radius_grid_cells)
    )

    refl_heights_m_agl = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY][
        trainval_io.RADAR_HEIGHTS_KEY
    ]
    num_refl_heights = len(refl_heights_m_agl)

    handle_dict = plot_examples.plot_one_example(
        list_of_predictor_matrices=mean_predictor_matrices,
        model_metadata_dict=model_metadata_dict,
        pmm_flag=True, plot_sounding=False, allow_whitespace=True,
        plot_panel_names=True, panel_name_font_size=PANEL_NAME_FONT_SIZE,
        add_titles=False, label_colour_bars=True,
        colour_bar_length=COLOUR_BAR_LENGTH,
        colour_bar_font_size=COLOUR_BAR_FONT_SIZE,
        num_panel_rows=num_refl_heights
    )

    axes_object_matrices = handle_dict[plot_examples.RADAR_AXES_KEY]

    max_colour_value_log10 = numpy.log10(max_colour_value)
    contour_interval_log10 = (
        (max_colour_value_log10 - MIN_COLOUR_VALUE_LOG10) /
        (num_contours - 1)
    )

    this_matrix = numpy.flip(mean_cam_matrices[0][0, ...], axis=0)
    this_matrix = numpy.log10(this_matrix)

    cam_plotting.plot_many_2d_grids(
        class_activation_matrix_3d=this_matrix,
        axes_object_matrix=axes_object_matrices[0],
        colour_map_object=colour_map_object,
        min_contour_level=MIN_COLOUR_VALUE_LOG10,
        max_contour_level=max_colour_value_log10,
        contour_interval=contour_interval_log10,
        row_major=True
    )

    this_matrix = numpy.flip(mean_cam_matrices[1][0, ...], axis=0)
    this_num_channels = mean_predictor_matrices[1].shape[-1]
    this_matrix = numpy.repeat(
        a=numpy.expand_dims(this_matrix, axis=-1),
        axis=-1, repeats=this_num_channels
    )
    this_matrix = numpy.log10(this_matrix)

    cam_plotting.plot_many_2d_grids(
        class_activation_matrix_3d=this_matrix,
        axes_object_matrix=axes_object_matrices[1],
        colour_map_object=colour_map_object,
        min_contour_level=MIN_COLOUR_VALUE_LOG10,
        max_contour_level=max_colour_value_log10,
        contour_interval=contour_interval_log10,
        row_major=False
    )

    refl_figure_object = handle_dict[plot_examples.RADAR_FIGURES_KEY][0]
    refl_figure_file_name = '{0:s}/{1:s}_reflectivity.jpg'.format(
        output_dir_name, composite_name_abbrev
    )

    print('Saving figure to: "{0:s}"...'.format(refl_figure_file_name))
    refl_figure_object.savefig(
        refl_figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(refl_figure_object)

    shear_figure_object = handle_dict[plot_examples.RADAR_FIGURES_KEY][1]
    shear_figure_file_name = '{0:s}/{1:s}_shear.jpg'.format(
        output_dir_name, composite_name_abbrev
    )

    print('Saving figure to: "{0:s}"...'.format(shear_figure_file_name))
    shear_figure_object.savefig(
        shear_figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(shear_figure_object)

    main_figure_file_name = '{0:s}/{1:s}_radar.jpg'.format(
        output_dir_name, composite_name_abbrev
    )
    print('Concatenating panels to: "{0:s}"...'.format(main_figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=[refl_figure_file_name, shear_figure_file_name],
        output_file_name=main_figure_file_name,
        num_panel_rows=1, num_panel_columns=2, border_width_pixels=50,
        extra_args_string='-gravity south'
    )
    imagemagick_utils.resize_image(
        input_file_name=main_figure_file_name,
        output_file_name=main_figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=main_figure_file_name,
        output_file_name=main_figure_file_name,
        border_width_pixels=TITLE_FONT_SIZE + 25
    )
    _overlay_text(
        image_file_name=main_figure_file_name,
        x_offset_from_center_px=0, y_offset_from_top_px=0,
        text_string=composite_name_verbose
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=main_figure_file_name,
        output_file_name=main_figure_file_name,
        border_width_pixels=10
    )

    return main_figure_file_name


def _add_colour_bar(figure_file_name, colour_map_object, max_colour_value,
                    temporary_dir_name):
    """Adds colour bar to saved image file.

    :param figure_file_name: Path to saved image file.  Colour bar will be added
        to this image.
    :param colour_map_object: Colour scheme (instance of `matplotlib.pyplot.cm`
        or similar).
    :param max_colour_value: Max value in colour scheme.
    :param temporary_dir_name: Name of temporary output directory.
    """

    this_image_matrix = Image.open(figure_file_name)
    figure_width_px, figure_height_px = this_image_matrix.size
    figure_width_inches = float(figure_width_px) / FIGURE_RESOLUTION_DPI
    figure_height_inches = float(figure_height_px) / FIGURE_RESOLUTION_DPI

    extra_figure_object, extra_axes_object = pyplot.subplots(
        1, 1, figsize=(figure_width_inches, figure_height_inches)
    )
    extra_axes_object.axis('off')

    dummy_values = numpy.array([0., max_colour_value])

    colour_bar_object = plotting_utils.plot_linear_colour_bar(
        axes_object_or_matrix=extra_axes_object, data_matrix=dummy_values,
        colour_map_object=colour_map_object,
        min_value=MIN_COLOUR_VALUE_LOG10,
        max_value=numpy.log10(max_colour_value),
        orientation_string='vertical', fraction_of_axis_length=1.25,
        extend_min=False, extend_max=True, font_size=COLOUR_BAR_FONT_SIZE
    )

    colour_bar_object.set_label(
        'Class activation', fontsize=COLOUR_BAR_FONT_SIZE
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = [
        '{0:.2f}'.format(10 ** v) for v in tick_values
    ]

    for i in range(len(tick_strings)):
        if '.' in tick_strings[i][:3]:
            tick_strings[i] = tick_strings[i][:4]
        else:
            tick_strings[i] = tick_strings[i].split('.')[0]

    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    extra_file_name = '{0:s}/gradcam_colour-bar.jpg'.format(temporary_dir_name)
    print('Saving colour bar to: "{0:s}"...'.format(extra_file_name))

    extra_figure_object.savefig(
        extra_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(extra_figure_object)

    print('Concatenating colour bar to: "{0:s}"...'.format(figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=[figure_file_name, extra_file_name],
        output_file_name=figure_file_name,
        num_panel_rows=1, num_panel_columns=2,
        extra_args_string='-gravity Center'
    )

    os.remove(extra_file_name)
    imagemagick_utils.trim_whitespace(
        input_file_name=figure_file_name, output_file_name=figure_file_name
    )


def _run(gradcam_file_names, composite_names, colour_map_name,
         max_colour_value, num_contours, smoothing_radius_grid_cells,
         output_dir_name):
    """Makes figure with class-activation maps for MYRORSS model.

    This is effectively the main method.

    :param gradcam_file_names: See documentation at top of file.
    :param composite_names: Same.
    :param colour_map_name: Same.
    :param max_colour_value: Same.
    :param num_contours: Same.
    :param smoothing_radius_grid_cells: Same.
    :param output_dir_name: Same.
    """

    # Process input args.
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    if smoothing_radius_grid_cells <= 0:
        smoothing_radius_grid_cells = None

    colour_map_object = pyplot.cm.get_cmap(colour_map_name)
    error_checking.assert_is_geq(num_contours, 10)
    error_checking.assert_is_greater(
        max_colour_value, 10 ** MIN_COLOUR_VALUE_LOG10
    )

    num_composites = len(gradcam_file_names)
    expected_dim = numpy.array([num_composites], dtype=int)
    error_checking.assert_is_numpy_array(
        numpy.array(composite_names), exact_dimensions=expected_dim
    )

    composite_names_abbrev = [
        n.replace('_', '-').lower() for n in composite_names
    ]
    composite_names_verbose = [
        '({0:s}) {1:s}'.format(
            chr(ord('a') + i), composite_names[i].replace('_', ' ')
        )
        for i in range(num_composites)
    ]

    panel_file_names = [None] * num_composites

    for i in range(num_composites):
        panel_file_names[i] = _plot_one_composite(
            gradcam_file_name=gradcam_file_names[i],
            composite_name_abbrev=composite_names_abbrev[i],
            composite_name_verbose=composite_names_verbose[i],
            colour_map_object=colour_map_object,
            max_colour_value=max_colour_value, num_contours=num_contours,
            smoothing_radius_grid_cells=smoothing_radius_grid_cells,
            output_dir_name=output_dir_name
        )

        print('\n')

    figure_file_name = '{0:s}/gradcam_concat.jpg'.format(output_dir_name)
    print('Concatenating panels to: "{0:s}"...'.format(figure_file_name))

    num_panel_rows = int(numpy.floor(
        numpy.sqrt(num_composites)
    ))
    num_panel_columns = int(numpy.ceil(
        float(num_composites) / num_panel_rows
    ))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=figure_file_name, border_width_pixels=100,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=figure_file_name, output_file_name=figure_file_name,
        border_width_pixels=10
    )

    _add_colour_bar(
        figure_file_name=figure_file_name, colour_map_object=colour_map_object,
        max_colour_value=max_colour_value, temporary_dir_name=output_dir_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        gradcam_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        composite_names=getattr(INPUT_ARG_OBJECT, COMPOSITE_NAMES_ARG_NAME),
        colour_map_name=getattr(INPUT_ARG_OBJECT, COLOUR_MAP_ARG_NAME),
        max_colour_value=getattr(INPUT_ARG_OBJECT, MAX_VALUE_ARG_NAME),
        num_contours=getattr(INPUT_ARG_OBJECT, NUM_CONTOURS_ARG_NAME),
        smoothing_radius_grid_cells=getattr(
            INPUT_ARG_OBJECT, SMOOTHING_RADIUS_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
