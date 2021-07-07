"""Makes figure with gradient-weighted class-activation maps (Grad-CAM)."""

import os
import pickle
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from PIL import Image
from gewittergefahr.gg_utils import general_utils
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import monte_carlo
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import gradcam
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import cam_plotting
from gewittergefahr.plotting import significance_plotting
from gewittergefahr.plotting import imagemagick_utils
from gewittergefahr.scripts import plot_input_examples as plot_examples

NONE_STRINGS = ['None', 'none']

RADAR_HEIGHTS_M_AGL = numpy.array([2000, 6000, 10000], dtype=int)
RADAR_FIELD_NAMES = [radar_utils.REFL_NAME, radar_utils.VORTICITY_NAME]

COLOUR_BAR_LENGTH = 0.25
PANEL_NAME_FONT_SIZE = 30
COLOUR_BAR_FONT_SIZE = 25

CONVERT_EXE_NAME = '/usr/bin/convert'
TITLE_FONT_SIZE = 150
TITLE_FONT_NAME = 'DejaVu-Sans-Bold'

FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

GRADCAM_FILES_ARG_NAME = 'input_gradcam_file_names'
MC_FILES_ARG_NAME = 'input_monte_carlo_file_names'
COMPOSITE_NAMES_ARG_NAME = 'composite_names'
COLOUR_MAP_ARG_NAME = 'colour_map_name'
MIN_VALUES_ARG_NAME = 'min_colour_values'
MAX_VALUES_ARG_NAME = 'max_colour_values'
NUM_CONTOURS_ARG_NAME = 'num_contours'
SMOOTHING_RADIUS_ARG_NAME = 'smoothing_radius_grid_cells'
MAX_FDR_ARG_NAME = 'monte_carlo_max_fdr'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

GRADCAM_FILES_HELP_STRING = (
    'List of Grad-CAM files (each will be read by `gradcam.read_file`).'
)
MC_FILES_HELP_STRING = (
    'List of files with Monte Carlo significance (one per saliency file).  Each'
    ' will be read by `_read_monte_carlo_test`.  If you do not want to plot '
    'significance for the [i]th composite, make the [i]th list element "None".'
)
COMPOSITE_NAMES_HELP_STRING = (
    'List of composite names (one for each Grad-CAM file).  This list must be '
    'space-separated, but after reading the list, underscores within each item '
    'will be replaced by spaces.'
)
COLOUR_MAP_HELP_STRING = (
    'Name of colour map.  Class activation for each predictor will be plotted '
    'with the same colour map.  For example, if name is "Greys", the colour map'
    ' used will be `pyplot.cm.Greys`.  This argument supports only pyplot '
    'colour maps.'
)
MIN_VALUES_HELP_STRING = (
    'Minimum class activation in each colour scheme (one per file).  Use '
    'negative values to let these be determined automatically.'
)
MAX_VALUES_HELP_STRING = 'Same as `{0:s}` but for max values.'.format(
    MIN_VALUES_ARG_NAME
)
NUM_CONTOURS_HELP_STRING = 'Number of contours for class activation.'
SMOOTHING_RADIUS_HELP_STRING = (
    'e-folding radius for Gaussian smoother (num grid cells).  If you do not '
    'want to smooth CAMs, make this non-positive.'
)
MAX_FDR_HELP_STRING = (
    'Max FDR (false-discovery rate) for field-based version of Monte Carlo '
    'significance test.  If you do not want to use field-based version, leave '
    'this argument alone.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + GRADCAM_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=GRADCAM_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MC_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=MC_FILES_HELP_STRING
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
    '--' + MIN_VALUES_ARG_NAME, type=float, nargs='+', required=True,
    help=MIN_VALUES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_VALUES_ARG_NAME, type=float, nargs='+', required=True,
    help=MAX_VALUES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_CONTOURS_ARG_NAME, type=int, required=False,
    default=15, help=NUM_CONTOURS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SMOOTHING_RADIUS_ARG_NAME, type=float, required=False,
    default=1., help=SMOOTHING_RADIUS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_FDR_ARG_NAME, type=float, required=False, default=-1.,
    help=MAX_FDR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _read_one_composite(gradcam_file_name, smoothing_radius_grid_cells,
                        monte_carlo_file_name, monte_carlo_max_fdr):
    """Reads class-activation map for one composite.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid
    H = number of heights in grid
    F = number of radar fields

    :param gradcam_file_name: Path to input file (will be read by
        `gradcam.read_file`).
    :param monte_carlo_file_name: Path to Monte Carlo file (will be read by
        `_read_monte_carlo_file`).
    :param smoothing_radius_grid_cells: Radius for Gaussian smoother, used only
        for class-activation map.
    :param monte_carlo_max_fdr: See documentation at top of file.
    :return: mean_radar_matrix: E-by-M-by-N-by-H-by-F numpy array with mean
        radar fields.
    :return: mean_class_activn_matrix: E-by-M-by-N-by-H numpy array with mean
        class-activation fields.
    :return: significance_matrix: E-by-M-by-N-by-H numpy array of Boolean
        flags.
    :return: model_metadata_dict: Dictionary returned by
        `cnn.read_model_metadata`.
    """

    print('Reading CAMs from: "{0:s}"...'.format(gradcam_file_name))
    gradcam_dict = gradcam.read_file(gradcam_file_name)[0]

    mean_radar_matrix = numpy.expand_dims(
        gradcam_dict[gradcam.MEAN_PREDICTOR_MATRICES_KEY][0], axis=0
    )
    mean_class_activn_matrix = numpy.expand_dims(
        gradcam_dict[gradcam.MEAN_CAM_MATRICES_KEY][0], axis=0
    )

    if smoothing_radius_grid_cells is not None:
        print((
            'Smoothing class-activation maps with Gaussian filter (e-folding '
            'radius of {0:.1f} grid cells)...'
        ).format(
            smoothing_radius_grid_cells
        ))

        mean_class_activn_matrix[0, ...] = general_utils.apply_gaussian_filter(
            input_matrix=mean_class_activn_matrix[0, ...],
            e_folding_radius_grid_cells=smoothing_radius_grid_cells
        )

    model_file_name = gradcam_dict[gradcam.MODEL_FILE_KEY]
    model_metafile_name = cnn.find_metafile(model_file_name)

    if monte_carlo_file_name is None:
        significance_matrix = numpy.full(
            mean_class_activn_matrix.shape, False, dtype=bool
        )
    else:
        print('Reading Monte Carlo test from: "{0:s}"...'.format(
            monte_carlo_file_name
        ))

        this_file_handle = open(monte_carlo_file_name, 'rb')
        monte_carlo_dict = pickle.load(this_file_handle)
        this_file_handle.close()

        p_value_matrix = monte_carlo_dict[monte_carlo.P_VALUE_MATRICES_KEY][0]

        if monte_carlo_max_fdr is None:
            significance_matrix = p_value_matrix <= 0.05
        else:
            significance_matrix = monte_carlo.find_sig_grid_points(
                p_value_matrix=p_value_matrix,
                max_false_discovery_rate=monte_carlo_max_fdr
            )

        significance_matrix = numpy.expand_dims(significance_matrix, axis=0)

    print('Fraction of significant differences: {0:.4f}'.format(
        numpy.mean(significance_matrix.astype(float))
    ))

    print('Reading CNN metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_model_metadata(model_metafile_name)
    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]

    good_indices = numpy.array([
        numpy.where(
            training_option_dict[trainval_io.RADAR_HEIGHTS_KEY] == h
        )[0][0]
        for h in RADAR_HEIGHTS_M_AGL
    ], dtype=int)

    mean_radar_matrix = mean_radar_matrix[..., good_indices, :]
    mean_class_activn_matrix = mean_class_activn_matrix[..., good_indices]
    significance_matrix = significance_matrix[..., good_indices]

    good_indices = numpy.array([
        training_option_dict[trainval_io.RADAR_FIELDS_KEY].index(f)
        for f in RADAR_FIELD_NAMES
    ], dtype=int)

    mean_radar_matrix = mean_radar_matrix[..., good_indices]

    training_option_dict[trainval_io.RADAR_HEIGHTS_KEY] = RADAR_HEIGHTS_M_AGL
    training_option_dict[trainval_io.RADAR_FIELDS_KEY] = RADAR_FIELD_NAMES
    training_option_dict[trainval_io.SOUNDING_FIELDS_KEY] = None
    model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY] = training_option_dict

    return (
        mean_radar_matrix, mean_class_activn_matrix, significance_matrix,
        model_metadata_dict
    )


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

    # TODO(thunderhoser): Put this method somewhere more general.

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
        gradcam_file_name, monte_carlo_file_name, composite_name_abbrev,
        composite_name_verbose, colour_map_object, min_colour_value,
        max_colour_value, num_contours, smoothing_radius_grid_cells,
        monte_carlo_max_fdr, output_dir_name):
    """Plots class-activation map for one composite.

    :param gradcam_file_name: Path to input file (will be read by
        `gradcam.read_file`).
    :param monte_carlo_file_name: Path to Monte Carlo file (will be read by
        `_read_monte_carlo_file`).
    :param composite_name_abbrev: Abbrev composite name (will be used in file
        names).
    :param composite_name_verbose: Verbose composite name (will be used in
        figure title).
    :param colour_map_object: See documentation at top of file.
    :param min_colour_value: Minimum value in colour bar (may be NaN).
    :param max_colour_value: Max value in colour bar (may be NaN).
    :param num_contours: See documentation at top of file.
    :param smoothing_radius_grid_cells: Same.
    :param monte_carlo_max_fdr: Same.
    :param output_dir_name: Name of output directory (figures will be saved
        here).
    :return: main_figure_file_name: Path to main image file created by this
        method.
    :return: min_colour_value: Same as input but cannot be None.
    :return: max_colour_value: Same as input but cannot be None.
    """

    (
        mean_radar_matrix, mean_class_activn_matrix, significance_matrix,
        model_metadata_dict
    ) = _read_one_composite(
        gradcam_file_name=gradcam_file_name,
        smoothing_radius_grid_cells=smoothing_radius_grid_cells,
        monte_carlo_file_name=monte_carlo_file_name,
        monte_carlo_max_fdr=monte_carlo_max_fdr
    )

    if numpy.isnan(min_colour_value) or numpy.isnan(max_colour_value):
        min_colour_value_log10 = numpy.log10(
            numpy.percentile(mean_class_activn_matrix, 1.)
        )
        max_colour_value_log10 = numpy.log10(
            numpy.percentile(mean_class_activn_matrix, 99.)
        )

        min_colour_value_log10 = max([min_colour_value_log10, -2.])
        max_colour_value_log10 = max([max_colour_value_log10, -1.])

        min_colour_value_log10 = min([min_colour_value_log10, 1.])
        max_colour_value_log10 = min([max_colour_value_log10, 2.])

        min_colour_value = 10 ** min_colour_value_log10
        max_colour_value = 10 ** max_colour_value_log10
    else:
        min_colour_value_log10 = numpy.log10(min_colour_value)
        max_colour_value_log10 = numpy.log10(max_colour_value)

    contour_interval_log10 = (
        (max_colour_value_log10 - min_colour_value_log10) /
        (num_contours - 1)
    )
    mean_activn_matrix_log10 = numpy.log10(mean_class_activn_matrix)

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    field_names = training_option_dict[trainval_io.RADAR_FIELDS_KEY]

    num_fields = mean_radar_matrix.shape[-1]
    num_heights = mean_radar_matrix.shape[-2]

    handle_dict = plot_examples.plot_one_example(
        list_of_predictor_matrices=[mean_radar_matrix],
        model_metadata_dict=model_metadata_dict, pmm_flag=True,
        allow_whitespace=True, plot_panel_names=True,
        panel_name_font_size=PANEL_NAME_FONT_SIZE,
        add_titles=False, label_colour_bars=True,
        colour_bar_length=COLOUR_BAR_LENGTH,
        colour_bar_font_size=COLOUR_BAR_FONT_SIZE,
        num_panel_rows=num_heights
    )

    figure_objects = handle_dict[plot_examples.RADAR_FIGURES_KEY]
    axes_object_matrices = handle_dict[plot_examples.RADAR_AXES_KEY]

    for k in range(num_fields):
        cam_plotting.plot_many_2d_grids(
            class_activation_matrix_3d=numpy.flip(
                mean_activn_matrix_log10[0, ...], axis=0
            ),
            axes_object_matrix=axes_object_matrices[k],
            colour_map_object=colour_map_object,
            min_contour_level=min_colour_value_log10,
            max_contour_level=max_colour_value_log10,
            contour_interval=contour_interval_log10
        )

        significance_plotting.plot_many_2d_grids_without_coords(
            significance_matrix=numpy.flip(
                significance_matrix[0, ...], axis=0
            ),
            axes_object_matrix=axes_object_matrices[k]
        )

    panel_file_names = [None] * num_fields

    for k in range(num_fields):
        panel_file_names[k] = '{0:s}/{1:s}_{2:s}.jpg'.format(
            output_dir_name, composite_name_abbrev,
            field_names[k].replace('_', '-')
        )

        print('Saving figure to: "{0:s}"...'.format(panel_file_names[k]))

        figure_objects[k].savefig(
            panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_objects[k])

    main_figure_file_name = '{0:s}/{1:s}_gradcam.jpg'.format(
        output_dir_name, composite_name_abbrev
    )

    print('Concatenating panels to: "{0:s}"...'.format(main_figure_file_name))
    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=main_figure_file_name,
        num_panel_rows=1, num_panel_columns=num_fields, border_width_pixels=50
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

    return main_figure_file_name, min_colour_value, max_colour_value


def _add_colour_bar(figure_file_name, colour_map_object, min_colour_value,
                    max_colour_value, temporary_dir_name):
    """Adds colour bar to saved image file.

    :param figure_file_name: Path to saved image file.  Colour bar will be added
        to this image.
    :param colour_map_object: Colour scheme (instance of `matplotlib.pyplot.cm`
        or similar).
    :param min_colour_value: Minimum value in colour scheme.
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

    dummy_values = numpy.array([min_colour_value, max_colour_value])

    colour_bar_object = plotting_utils.plot_linear_colour_bar(
        axes_object_or_matrix=extra_axes_object, data_matrix=dummy_values,
        colour_map_object=colour_map_object,
        min_value=numpy.log10(min_colour_value),
        max_value=numpy.log10(max_colour_value),
        orientation_string='vertical', fraction_of_axis_length=1.25,
        extend_min=False, extend_max=True, font_size=COLOUR_BAR_FONT_SIZE,
        aspect_ratio=50.
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


def _run(gradcam_file_names, monte_carlo_file_names, composite_names,
         colour_map_name, min_colour_values, max_colour_values, num_contours,
         smoothing_radius_grid_cells, monte_carlo_max_fdr, output_dir_name):
    """Makes figure with gradient-weighted class-activation maps (Grad-CAM).

    This is effectively the main method.

    :param gradcam_file_names: See documentation at top of file.
    :param monte_carlo_file_names: Same.
    :param composite_names: Same.
    :param colour_map_name: Same.
    :param min_colour_values: Same.
    :param max_colour_values: Same.
    :param num_contours: Same.
    :param smoothing_radius_grid_cells: Same.
    :param output_dir_name: Same.
    """

    if smoothing_radius_grid_cells <= 0:
        smoothing_radius_grid_cells = None
    if monte_carlo_max_fdr <= 0:
        monte_carlo_max_fdr = None

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    colour_map_object = pyplot.cm.get_cmap(colour_map_name)
    error_checking.assert_is_geq(num_contours, 10)

    num_composites = len(gradcam_file_names)
    expected_dim = numpy.array([num_composites], dtype=int)
    error_checking.assert_is_numpy_array(
        numpy.array(composite_names), exact_dimensions=expected_dim
    )
    error_checking.assert_is_numpy_array(
        numpy.array(monte_carlo_file_names), exact_dimensions=expected_dim
    )

    monte_carlo_file_names = [
        None if f in NONE_STRINGS else f for f in monte_carlo_file_names
    ]

    nan_indices = numpy.where(numpy.logical_or(
        max_colour_values < 0, min_colour_values < 0
    ))[0]
    min_colour_values[nan_indices] = numpy.nan
    max_colour_values[nan_indices] = numpy.nan

    error_checking.assert_is_numpy_array(
        min_colour_values, exact_dimensions=expected_dim
    )
    error_checking.assert_is_numpy_array(
        max_colour_values, exact_dimensions=expected_dim
    )
    assert not numpy.any(max_colour_values <= min_colour_values)

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
        (
            panel_file_names[i], min_colour_values[i], max_colour_values[i]
        ) = _plot_one_composite(
            gradcam_file_name=gradcam_file_names[i],
            monte_carlo_file_name=monte_carlo_file_names[i],
            composite_name_abbrev=composite_names_abbrev[i],
            composite_name_verbose=composite_names_verbose[i],
            colour_map_object=colour_map_object,
            min_colour_value=min_colour_values[i],
            max_colour_value=max_colour_values[i],
            num_contours=num_contours,
            smoothing_radius_grid_cells=smoothing_radius_grid_cells,
            monte_carlo_max_fdr=monte_carlo_max_fdr,
            output_dir_name=output_dir_name
        )

        _add_colour_bar(
            figure_file_name=panel_file_names[i],
            colour_map_object=colour_map_object,
            min_colour_value=min_colour_values[i],
            max_colour_value=max_colour_values[i],
            temporary_dir_name=output_dir_name
        )

        print('\n')

    figure_file_name = '{0:s}/gradcam_concat.jpg'.format(output_dir_name)
    print('Concatenating panels to: "{0:s}"...'.format(figure_file_name))

    num_panel_rows = int(numpy.ceil(
        numpy.sqrt(num_composites)
    ))
    num_panel_columns = int(numpy.floor(
        float(num_composites) / num_panel_rows
    ))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=figure_file_name, border_width_pixels=25,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=figure_file_name, output_file_name=figure_file_name,
        border_width_pixels=10
    )
    imagemagick_utils.resize_image(
        input_file_name=figure_file_name, output_file_name=figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        gradcam_file_names=getattr(INPUT_ARG_OBJECT, GRADCAM_FILES_ARG_NAME),
        monte_carlo_file_names=getattr(INPUT_ARG_OBJECT, MC_FILES_ARG_NAME),
        composite_names=getattr(INPUT_ARG_OBJECT, COMPOSITE_NAMES_ARG_NAME),
        colour_map_name=getattr(INPUT_ARG_OBJECT, COLOUR_MAP_ARG_NAME),
        min_colour_values=numpy.array(
            getattr(INPUT_ARG_OBJECT, MIN_VALUES_ARG_NAME), dtype=float
        ),
        max_colour_values=numpy.array(
            getattr(INPUT_ARG_OBJECT, MAX_VALUES_ARG_NAME), dtype=float
        ),
        num_contours=getattr(INPUT_ARG_OBJECT, NUM_CONTOURS_ARG_NAME),
        smoothing_radius_grid_cells=getattr(
            INPUT_ARG_OBJECT, SMOOTHING_RADIUS_ARG_NAME
        ),
        monte_carlo_max_fdr=getattr(INPUT_ARG_OBJECT, MAX_FDR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
