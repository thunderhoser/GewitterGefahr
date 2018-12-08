"""Plots saliency maps for a CNN (convolutional neural network)."""

import os
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import saliency_maps
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import radar_plotting
from gewittergefahr.plotting import sounding_plotting
from gewittergefahr.plotting import saliency_plotting
from gewittergefahr.plotting import imagemagick_utils

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TITLE_FONT_SIZE = 20
SALIENCY_COLOUR_MAP_OBJECT = pyplot.cm.Greys

FIGURE_RESOLUTION_DPI = 300
SOUNDING_IMAGE_SIZE_PX = int(5e6)

INPUT_FILE_ARG_NAME = 'input_file_name'
MAX_COLOUR_VALUE_ARG_NAME = 'max_colour_value'
MAX_COLOUR_PRCTILE_ARG_NAME = 'max_colour_percentile'
NUM_PANEL_ROWS_ARG_NAME = 'num_panel_rows'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `saliency_maps.read_file`.')

MAX_COLOUR_VALUE_HELP_STRING = (
    'Max saliency value in colour scheme.  Minimum saliency in colour scheme '
    'will be -1 * `{0:s}`.  To use `{1:s}` instead, leave this argument alone.'
).format(MAX_COLOUR_VALUE_ARG_NAME, MAX_COLOUR_PRCTILE_ARG_NAME)

MAX_COLOUR_PRCTILE_HELP_STRING = (
    'Max saliency value in colour scheme will be the `{0:s}`th percentile of '
    'absolute values in `{1:s}` (over all storm objects, radar field/height '
    'pairs, and sounding field/height pairs).  Minimum saliency in colour '
    'scheme will be -1 * max value.  To use `{2:s}` instead, leave this '
    'argument alone.'
).format(MAX_COLOUR_PRCTILE_ARG_NAME, INPUT_FILE_ARG_NAME,
         MAX_COLOUR_VALUE_ARG_NAME)

NUM_PANEL_ROWS_HELP_STRING = (
    'Number of panel rows in each radar figure.  If radar images are 3-D, there'
    ' will be one figure per storm object and field, containing all heights.  '
    'If radar images are 2-D, there will be one figure per storm object, '
    'containing all field/height pairs.')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_COLOUR_VALUE_ARG_NAME, type=float, required=False, default=-1,
    help=MAX_COLOUR_VALUE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_COLOUR_PRCTILE_ARG_NAME, type=float, required=False, default=-1,
    help=MAX_COLOUR_PRCTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_PANEL_ROWS_ARG_NAME, type=int, required=False, default=3,
    help=NUM_PANEL_ROWS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _plot_2d_radar_saliency(
        radar_matrix, radar_saliency_matrix, saliency_metadata_dict,
        training_option_dict, max_absolute_colour_value, num_panel_rows,
        output_dir_name):
    """Plots 2-D radar images along with their saliency maps.

    E = number of examples
    M = number of rows in spatial grid
    N = number of columns in spatial grid
    C = number of channels (field-height pairs)

    :param radar_matrix: E-by-M-by-N-by-C numpy array of radar values.
    :param radar_saliency_matrix: E-by-M-by-N-by-C numpy array of
        corresponding saliency values.
    :param saliency_metadata_dict: See doc for `_plot_2d3d_radar_saliency`.
    :param training_option_dict: Same.
    :param max_absolute_colour_value: Same.
    :param num_panel_rows: Number of rows in each paneled figure.  Each panel
        corresponds to one field-height pair.  Each figure corresponds to one
        example.
    :param output_dir_name: See doc for `_plot_2d3d_radar_saliency`.
    """

    num_examples = radar_matrix.shape[0]

    for i in range(num_examples):
        this_storm_id = saliency_metadata_dict[
            saliency_maps.STORM_IDS_KEY][i]
        this_storm_time_string = time_conversion.unix_sec_to_string(
            saliency_metadata_dict[saliency_maps.STORM_TIMES_KEY][i],
            TIME_FORMAT)

        _, these_axes_objects = (
            radar_plotting.plot_many_2d_grids_without_coords(
                field_matrix=numpy.flip(radar_matrix[i, ...], axis=-1),
                field_name_by_pair=training_option_dict[
                    trainval_io.RADAR_FIELDS_KEY],
                height_by_pair_metres=training_option_dict[
                    trainval_io.RADAR_HEIGHTS_KEY],
                ground_relative=True, num_panel_rows=num_panel_rows)
        )

        saliency_plotting.plot_many_2d_grids(
            saliency_matrix_3d=numpy.flip(
                radar_saliency_matrix[i, ...], axis=0),
            axes_objects_2d_list=these_axes_objects,
            colour_map_object=SALIENCY_COLOUR_MAP_OBJECT,
            max_absolute_colour_value=max_absolute_colour_value)

        this_title_string = (
            'Radar + saliency for storm "{0:s}" at {1:s}'
        ).format(this_storm_id, this_storm_time_string)
        pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)

        this_figure_file_name = (
            '{0:s}/saliency_{1:s}_{2:s}_radar.jpg'
        ).format(
            output_dir_name, this_storm_id.replace('_', '-'),
            this_storm_time_string
        )

        print 'Saving figure to file: "{0:s}"...'.format(
            this_figure_file_name)
        pyplot.savefig(this_figure_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()


def _plot_3d_radar_saliency(
        radar_matrix, radar_saliency_matrix, saliency_metadata_dict,
        training_option_dict, max_absolute_colour_value, num_panel_rows,
        output_dir_name):
    """Plots 3-D radar images along with their saliency maps.

    E = number of examples
    M = number of rows in spatial grid
    N = number of columns in spatial grid
    H = number of heights in spatial grid
    F = number of fields

    :param radar_matrix: E-by-M-by-N-by-H-by-F numpy array of radar values.
    :param radar_saliency_matrix: E-by-M-by-N-by-H-by-F numpy array of
        corresponding saliency values.
    :param saliency_metadata_dict: See doc for `_plot_2d3d_radar_saliency`.
    :param training_option_dict: Same.
    :param max_absolute_colour_value: Same.
    :param num_panel_rows: Number of rows in each paneled figure.  Each panel
        corresponds to one height in the grid.  Each figure corresponds to one
        field for one example.
    :param output_dir_name: See doc for `_plot_2d3d_radar_saliency`.
    """

    num_examples = radar_matrix.shape[0]
    num_fields = radar_matrix.shape[-1]

    for i in range(num_examples):
        this_storm_id = saliency_metadata_dict[
            saliency_maps.STORM_IDS_KEY][i]
        this_storm_time_string = time_conversion.unix_sec_to_string(
            saliency_metadata_dict[saliency_maps.STORM_TIMES_KEY][i],
            TIME_FORMAT)

        for k in range(num_fields):
            this_field_name = training_option_dict[
                trainval_io.RADAR_FIELDS_KEY][k]

            _, these_axes_objects = (
                radar_plotting.plot_3d_grid_without_coords(
                    field_matrix=numpy.flip(
                        radar_matrix[i, ..., k], axis=0),
                    field_name=this_field_name,
                    grid_point_heights_metres=training_option_dict[
                        trainval_io.RADAR_HEIGHTS_KEY],
                    ground_relative=True, num_panel_rows=num_panel_rows)
            )

            saliency_plotting.plot_many_2d_grids(
                saliency_matrix_3d=numpy.flip(
                    radar_saliency_matrix[i, ..., k], axis=0),
                axes_objects_2d_list=these_axes_objects,
                colour_map_object=SALIENCY_COLOUR_MAP_OBJECT,
                max_absolute_colour_value=max_absolute_colour_value)

            this_colour_map_object, this_colour_norm_object, _ = (
                radar_plotting.get_default_colour_scheme(this_field_name)
            )

            plotting_utils.add_colour_bar(
                axes_object_or_list=these_axes_objects,
                values_to_colour=radar_matrix[i, ..., k],
                colour_map=this_colour_map_object,
                colour_norm_object=this_colour_norm_object,
                orientation='horizontal', extend_min=True, extend_max=True)

            this_title_string = (
                '{0:s} + saliency for storm "{1:s}" at {2:s}'
            ).format(this_field_name, this_storm_id, this_storm_time_string)
            pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)

            this_figure_file_name = (
                '{0:s}/saliency_{1:s}_{2:s}_{3:s}.jpg'
            ).format(
                output_dir_name, this_storm_id.replace('_', '-'),
                this_storm_time_string, this_field_name.replace('_', '-')
            )

            print 'Saving figure to file: "{0:s}"...'.format(
                this_figure_file_name)
            pyplot.savefig(this_figure_file_name, dpi=FIGURE_RESOLUTION_DPI)
            pyplot.close()


def _plot_2d3d_radar_saliency(
        list_of_input_matrices, list_of_saliency_matrices,
        saliency_metadata_dict, training_option_dict, max_absolute_colour_value,
        num_panel_rows, output_dir_name):
    """Plots 2-D and 3-D radar images along with their saliency maps.

    2-D images contain azimuthal shear, and 3-D images contain reflectivity.

    :param list_of_input_matrices: See doc for `saliency_maps.read_saliency`.
    :param list_of_saliency_matrices: Same.
    :param saliency_metadata_dict: Same.
    :param training_option_dict: Dictionary returned by
        `cnn.read_model_metadata`.
    :param max_absolute_colour_value: Max absolute value in saliency colour
        scheme.
    :param num_panel_rows: Number of rows in each paneled figure (used only for
        reflectivity; all figures with azimuthal shear have 1 row and 2
        columns).
    :param output_dir_name: Name of output directory (figures will be saved
        here).
    """

    reflectivity_matrix_dbz = list_of_input_matrices[0]
    refl_saliency_matrix = list_of_saliency_matrices[0]
    azimuthal_shear_matrix_s01 = list_of_input_matrices[1]
    az_shear_saliency_matrix = list_of_saliency_matrices[1]

    az_shear_heights_m_agl = numpy.full(
        len(training_option_dict[trainval_io.RADAR_FIELDS_KEY]),
        radar_utils.SHEAR_HEIGHT_M_ASL)

    num_examples = reflectivity_matrix_dbz.shape[0]

    for i in range(num_examples):
        this_storm_id = saliency_metadata_dict[
            saliency_maps.STORM_IDS_KEY][i]
        this_storm_time_string = time_conversion.unix_sec_to_string(
            saliency_metadata_dict[saliency_maps.STORM_TIMES_KEY][i],
            TIME_FORMAT)

        _, these_axes_objects = radar_plotting.plot_3d_grid_without_coords(
            field_matrix=numpy.flip(
                reflectivity_matrix_dbz[i, ..., 0], axis=0),
            field_name=radar_utils.REFL_NAME,
            grid_point_heights_metres=training_option_dict[
                trainval_io.RADAR_HEIGHTS_KEY],
            ground_relative=True, num_panel_rows=num_panel_rows)

        saliency_plotting.plot_many_2d_grids(
            saliency_matrix_3d=numpy.flip(
                refl_saliency_matrix[i, ..., 0], axis=0),
            axes_objects_2d_list=these_axes_objects,
            colour_map_object=SALIENCY_COLOUR_MAP_OBJECT,
            max_absolute_colour_value=max_absolute_colour_value)

        this_colour_map_object, this_colour_norm_object, _ = (
            radar_plotting.get_default_colour_scheme(radar_utils.REFL_NAME)
        )

        plotting_utils.add_colour_bar(
            axes_object_or_list=these_axes_objects,
            values_to_colour=reflectivity_matrix_dbz[i, ..., 0],
            colour_map=this_colour_map_object,
            colour_norm_object=this_colour_norm_object,
            orientation='horizontal', extend_min=True, extend_max=True)

        this_title_string = (
            'Reflectivity + saliency for storm "{0:s}" at {1:s}'
        ).format(this_storm_id, this_storm_time_string)
        pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)

        this_figure_file_name = (
            '{0:s}/saliency_{1:s}_{2:s}_reflectivity.jpg'
        ).format(
            output_dir_name, this_storm_id.replace('_', '-'),
            this_storm_time_string
        )

        print 'Saving figure to file: "{0:s}"...'.format(
            this_figure_file_name)
        pyplot.savefig(this_figure_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()

        _, these_axes_objects = (
            radar_plotting.plot_many_2d_grids_without_coords(
                field_matrix=numpy.flip(
                    azimuthal_shear_matrix_s01[i, ...], axis=-1),
                field_name_by_pair=training_option_dict[
                    trainval_io.RADAR_FIELDS_KEY],
                height_by_pair_metres=az_shear_heights_m_agl,
                ground_relative=True, num_panel_rows=num_panel_rows)
        )

        saliency_plotting.plot_many_2d_grids(
            saliency_matrix_3d=numpy.flip(
                az_shear_saliency_matrix[i, ...], axis=0),
            axes_objects_2d_list=these_axes_objects,
            colour_map_object=SALIENCY_COLOUR_MAP_OBJECT,
            max_absolute_colour_value=max_absolute_colour_value)

        this_colour_map_object, this_colour_norm_object, _ = (
            radar_plotting.get_default_colour_scheme(
                radar_utils.LOW_LEVEL_SHEAR_NAME)
        )

        plotting_utils.add_colour_bar(
            axes_object_or_list=these_axes_objects,
            values_to_colour=az_shear_saliency_matrix[i, ...],
            colour_map=this_colour_map_object,
            colour_norm_object=this_colour_norm_object,
            orientation='horizontal', extend_min=True, extend_max=True)

        this_title_string = (
            'Azimuthal shear + saliency for storm "{0:s}" at {1:s}'
        ).format(this_storm_id, this_storm_time_string)
        pyplot.suptitle(this_title_string, fontsize=TITLE_FONT_SIZE)

        this_figure_file_name = (
            '{0:s}/saliency_{1:s}_{2:s}_azimuthal-shear.jpg'
        ).format(
            output_dir_name, this_storm_id.replace('_', '-'),
            this_storm_time_string
        )

        print 'Saving figure to file: "{0:s}"...'.format(
            this_figure_file_name)
        pyplot.savefig(this_figure_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()


def _plot_sounding_saliency(
        sounding_matrix, sounding_saliency_matrix, sounding_field_names,
        saliency_metadata_dict, max_absolute_colour_value, output_dir_name):
    """Plots soundings along with their saliency maps.

    E = number of examples
    H = number of sounding heights
    F = number of sounding fields

    :param sounding_matrix: E-by-H-by-F numpy array of sounding values.
    :param sounding_saliency_matrix: E-by-H-by-F numpy array of corresponding
        saliency values.
    :param sounding_field_names: length-F list of field names.
    :param saliency_metadata_dict: Dictionary returned by
        `saliency_maps.read_file`.
    :param max_absolute_colour_value: Max absolute value in saliency colour
        scheme.
    :param output_dir_name: Name of output directory (figures will be saved
        here).
    """

    num_examples = sounding_matrix.shape[0]

    try:
        metpy_dict_by_example = dl_utils.soundings_to_metpy_dictionaries(
            sounding_matrix=sounding_matrix, field_names=sounding_field_names)
    except:
        sounding_pressure_matrix_pa = numpy.expand_dims(
            saliency_metadata_dict[saliency_maps.SOUNDING_PRESSURES_KEY],
            axis=-1)

        this_sounding_matrix = numpy.concatenate(
            (sounding_matrix, sounding_pressure_matrix_pa), axis=-1)

        metpy_dict_by_example = dl_utils.soundings_to_metpy_dictionaries(
            sounding_matrix=this_sounding_matrix,
            field_names=sounding_field_names + [soundings.PRESSURE_NAME])

    for i in range(num_examples):
        this_storm_id = saliency_metadata_dict[
            saliency_maps.STORM_IDS_KEY][i]
        this_storm_time_string = time_conversion.unix_sec_to_string(
            saliency_metadata_dict[saliency_maps.STORM_TIMES_KEY][i],
            TIME_FORMAT)

        this_title_string = 'Storm "{0:s}" at {1:s}'.format(
            this_storm_id, this_storm_time_string)
        sounding_plotting.plot_sounding(
            sounding_dict_for_metpy=metpy_dict_by_example[i],
            title_string=this_title_string)

        this_left_file_name = (
            '{0:s}/{1:s}_{2:s}_sounding-actual.jpg'
        ).format(
            output_dir_name, this_storm_id.replace('_', '-'),
            this_storm_time_string
        )

        print 'Saving figure to file: "{0:s}"...'.format(
            this_left_file_name)
        pyplot.savefig(this_left_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()

        imagemagick_utils.trim_whitespace(
            input_file_name=this_left_file_name,
            output_file_name=this_left_file_name)

        saliency_plotting.plot_saliency_for_sounding(
            saliency_matrix=sounding_saliency_matrix[i, ...],
            sounding_field_names=sounding_field_names,
            pressure_levels_mb=metpy_dict_by_example[i][
                soundings.PRESSURE_COLUMN_METPY],
            colour_map_object=SALIENCY_COLOUR_MAP_OBJECT,
            max_absolute_colour_value=max_absolute_colour_value)

        this_right_file_name = (
            '{0:s}/{1:s}_{2:s}_sounding-saliency.jpg'
        ).format(
            output_dir_name, this_storm_id.replace('_', '-'),
            this_storm_time_string
        )

        print 'Saving figure to file: "{0:s}"...'.format(
            this_right_file_name)
        pyplot.savefig(this_right_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()

        imagemagick_utils.trim_whitespace(
            input_file_name=this_right_file_name,
            output_file_name=this_right_file_name)

        this_figure_file_name = (
            '{0:s}/saliency_{1:s}_{2:s}_sounding.jpg'
        ).format(
            output_dir_name, this_storm_id.replace('_', '-'),
            this_storm_time_string
        )

        print 'Concatenating panels into file: "{0:s}"...\n'.format(
            this_figure_file_name)
        imagemagick_utils.concatenate_images(
            input_file_names=[this_left_file_name, this_right_file_name],
            output_file_name=this_figure_file_name, num_panel_rows=1,
            num_panel_columns=2, output_size_pixels=SOUNDING_IMAGE_SIZE_PX)

        os.remove(this_left_file_name)
        os.remove(this_right_file_name)


def _run(input_file_name, max_colour_value, max_colour_percentile,
         num_panel_rows, output_dir_name):
    """Plots saliency maps for a CNN (convolutional neural network).

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param max_colour_value: Same.
    :param max_colour_percentile: Same.
    :param num_panel_rows: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if both `max_colour_value` and `max_colour_percentile`
        are non-positive.
    :raises: TypeError: if saliency maps come from a model that does 2-D and 3-D
        convolution.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    # Check input args.
    if max_colour_value <= 0:
        max_colour_value = None
    if max_colour_percentile <= 0:
        max_colour_percentile = None

    if max_colour_value is None and max_colour_percentile is None:
        raise ValueError(
            'max_colour_value and max_colour_percentile cannot both be None.')

    # Read saliency maps.
    print 'Reading saliency maps from: "{0:s}"...'.format(input_file_name)
    (list_of_input_matrices, list_of_saliency_matrices, saliency_metadata_dict
     ) = saliency_maps.read_file(input_file_name)

    if max_colour_value is None:
        all_saliency_values = numpy.array([])
        for this_matrix in list_of_saliency_matrices:
            all_saliency_values = numpy.concatenate(
                (all_saliency_values, numpy.ravel(this_matrix)))

        max_colour_value = numpy.percentile(
            numpy.absolute(all_saliency_values), max_colour_percentile)
        del all_saliency_values

    print 'Max saliency value in colour scheme = {0:.3e}\n'.format(
        max_colour_value)

    # Read metadata for the CNN that generated the saliency maps.
    model_file_name = saliency_metadata_dict[saliency_maps.MODEL_FILE_NAME_KEY]
    model_metadata_file_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0])

    print 'Reading metadata from: "{0:s}"...'.format(model_metadata_file_name)
    model_metadata_dict = cnn.read_model_metadata(model_metadata_file_name)
    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]

    sounding_field_names = training_option_dict[trainval_io.SOUNDING_FIELDS_KEY]
    # if sounding_field_names is not None:
        # _plot_sounding_saliency(
        #     sounding_matrix=list_of_input_matrices[-1],
        #     sounding_saliency_matrix=list_of_saliency_matrices[-1],
        #     sounding_field_names=sounding_field_names,
        #     saliency_metadata_dict=saliency_metadata_dict,
        #     max_absolute_colour_value=max_colour_value,
        #     output_dir_name=output_dir_name)
        # print SEPARATOR_STRING

    if model_metadata_dict[cnn.USE_2D3D_CONVOLUTION_KEY]:
        _plot_2d3d_radar_saliency(
            list_of_input_matrices=list_of_input_matrices,
            list_of_saliency_matrices=list_of_saliency_matrices,
            saliency_metadata_dict=saliency_metadata_dict,
            training_option_dict=training_option_dict,
            max_absolute_colour_value=max_colour_value,
            num_panel_rows=num_panel_rows, output_dir_name=output_dir_name)
        return

    num_radar_dimensions = len(list_of_input_matrices[0].shape) - 2
    if num_radar_dimensions == 3:
        _plot_3d_radar_saliency(
            radar_matrix=list_of_input_matrices[0],
            radar_saliency_matrix=list_of_saliency_matrices[0],
            saliency_metadata_dict=saliency_metadata_dict,
            training_option_dict=training_option_dict,
            max_absolute_colour_value=max_colour_value,
            num_panel_rows=num_panel_rows, output_dir_name=output_dir_name)
        return

    _plot_2d_radar_saliency(
        radar_matrix=list_of_input_matrices[0],
        radar_saliency_matrix=list_of_saliency_matrices[0],
        saliency_metadata_dict=saliency_metadata_dict,
        training_option_dict=training_option_dict,
        max_absolute_colour_value=max_colour_value,
        num_panel_rows=num_panel_rows, output_dir_name=output_dir_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        max_colour_value=getattr(INPUT_ARG_OBJECT, MAX_COLOUR_VALUE_ARG_NAME),
        max_colour_percentile=getattr(
            INPUT_ARG_OBJECT, MAX_COLOUR_PRCTILE_ARG_NAME),
        num_panel_rows=getattr(INPUT_ARG_OBJECT, NUM_PANEL_ROWS_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
