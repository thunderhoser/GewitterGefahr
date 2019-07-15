"""Plots saliency maps for a CNN (convolutional neural network)."""

import os
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import monte_carlo
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import saliency_maps
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.plotting import sounding_plotting
from gewittergefahr.plotting import saliency_plotting
from gewittergefahr.plotting import significance_plotting
from gewittergefahr.plotting import imagemagick_utils
from gewittergefahr.scripts import plot_input_examples

# TODO(thunderhoser): Use threshold counts at some point.

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

HALF_NUM_CONTOURS = 10
FIGURE_RESOLUTION_DPI = 300
SOUNDING_IMAGE_SIZE_PX = int(5e6)

INPUT_FILE_ARG_NAME = 'input_file_name'
PLOT_SIGNIFICANCE_ARG_NAME = 'plot_significance'
COLOUR_MAP_ARG_NAME = 'saliency_colour_map_name'
MAX_PERCENTILE_ARG_NAME = 'max_colour_prctile_for_saliency'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `saliency_maps.read_standard_file` or'
    ' `saliency_maps.read_pmm_file`.')

PLOT_SIGNIFICANCE_HELP_STRING = (
    'Boolean flag.  If 1, will plot stippling for significance.  This applies '
    'only if the saliency map contains PMM (proability-matched means) and '
    'results of a Monte Carlo comparison.')

COLOUR_MAP_HELP_STRING = (
    'Name of colour map.  Saliency for each predictor will be plotted with the '
    'same colour map.  For example, if name is "Greys", the colour map used '
    'will be `pyplot.cm.Greys`.  This argument supports only pyplot colour '
    'maps.')

MAX_PERCENTILE_HELP_STRING = (
    'Used to set max absolute value for each saliency map.  The max absolute '
    'value for example e and predictor p will be the [q]th percentile of all '
    'saliency values for example e, where q = `{0:s}`.'
).format(MAX_PERCENTILE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Figures will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_SIGNIFICANCE_ARG_NAME, type=int, required=False, default=0,
    help=PLOT_SIGNIFICANCE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + COLOUR_MAP_ARG_NAME, type=str, required=False, default='Greys',
    help=COLOUR_MAP_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False,
    default=99., help=MAX_PERCENTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _plot_3d_radar_saliency(
        saliency_matrix, colour_map_object, max_colour_value, figure_objects,
        axes_object_matrices, model_metadata_dict, output_dir_name,
        significance_matrix=None, full_storm_id_string=None,
        storm_time_string=None):
    """Plots saliency map for 3-D radar data.

    M = number of rows in spatial grid
    N = number of columns in spatial grid
    H = number of heights in spatial grid
    F = number of radar fields

    If this method is plotting a composite rather than single example (storm
    object), `full_storm_id_string` and `storm_time_string` can be None.

    :param saliency_matrix: M-by-N-by-H-by-F numpy array of saliency values.
    :param colour_map_object: See documentation at top of file.
    :param max_colour_value: Max value in colour scheme for saliency.
    :param figure_objects: See doc for `plot_input_examples._plot_3d_example`.
    :param axes_object_matrices: Same.
    :param model_metadata_dict: Dictionary returned by
        `cnn.read_model_metadata`.
    :param output_dir_name: Path to output directory.  Figure(s) will be saved
        here.
    :param significance_matrix: M-by-N-by-H-by-F numpy array of Boolean flags,
        indicating where differences with some other saliency map are
        significant.
    :param full_storm_id_string: Full storm ID.
    :param storm_time_string: Storm time (format "yyyy-mm-dd-HHMMSS").
    """

    pmm_flag = full_storm_id_string is None and storm_time_string is None
    conv_2d3d = model_metadata_dict[cnn.CONV_2D3D_KEY]
    upsample_refl = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY][
        trainval_io.UPSAMPLE_REFLECTIVITY_KEY]

    if conv_2d3d and not upsample_refl:
        loop_max = 1
        radar_field_names = ['reflectivity']
    else:
        loop_max = len(figure_objects)
        training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
        radar_field_names = training_option_dict[trainval_io.RADAR_FIELDS_KEY]

    for j in range(loop_max):
        saliency_plotting.plot_many_2d_grids_with_pm_signs(
            saliency_matrix_3d=numpy.flip(saliency_matrix[..., j], axis=0),
            axes_object_matrix=axes_object_matrices[j],
            colour_map_object=colour_map_object,
            max_absolute_colour_value=max_colour_value)

        if significance_matrix is not None:
            this_matrix = numpy.flip(significance_matrix[..., j], axis=0)

            significance_plotting.plot_many_2d_grids_without_coords(
                significance_matrix=this_matrix,
                axes_object_matrix=axes_object_matrices[j]
            )

        allow_whitespace = figure_objects[j]._suptitle is not None

        if allow_whitespace:
            this_title_string = '{0:s}; (max abs saliency = {1:.2e})'.format(
                figure_objects[j]._suptitle.get_text(), max_colour_value
            )

            figure_objects[j].suptitle(this_title_string)

        this_file_name = plot_input_examples.metadata_to_radar_fig_file_name(
            output_dir_name=output_dir_name, pmm_flag=pmm_flag,
            full_storm_id_string=full_storm_id_string,
            storm_time_string=storm_time_string,
            radar_field_name=radar_field_names[j]
        )

        print('Saving figure to: "{0:s}"...'.format(this_file_name))
        figure_objects[j].savefig(
            this_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
            bbox_inches='tight'
        )
        pyplot.close(figure_objects[j])


def _plot_2d_radar_saliency(
        saliency_matrix, colour_map_object, max_colour_value, figure_objects,
        axes_object_matrices, model_metadata_dict, output_dir_name,
        significance_matrix=None, full_storm_id_string=None,
        storm_time_string=None):
    """Plots saliency map for 2-D radar data.

    M = number of rows in spatial grid
    N = number of columns in spatial grid
    C = number of radar channels

    If this method is plotting a composite rather than single example (storm
    object), `full_storm_id_string` and `storm_time_string` can be None.

    :param saliency_matrix: M-by-N-by-C numpy array of saliency values.
    :param colour_map_object: See documentation at top of file.
    :param max_colour_value: Max value in colour scheme for saliency.
    :param figure_objects: See doc for `plot_input_examples._plot_3d_example`.
    :param axes_object_matrices: Same.
    :param model_metadata_dict: Dictionary returned by
        `cnn.read_model_metadata`.
    :param output_dir_name: Path to output directory.  Figure(s) will be saved
        here.
    :param significance_matrix: M-by-N-by-H numpy array of Boolean flags,
        indicating where differences with some other saliency map are
        significant.
    :param full_storm_id_string: Full storm ID.
    :param storm_time_string: Storm time (format "yyyy-mm-dd-HHMMSS").
    """

    pmm_flag = full_storm_id_string is None and storm_time_string is None
    conv_2d3d = model_metadata_dict[cnn.CONV_2D3D_KEY]
    upsample_refl = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY][
        trainval_io.UPSAMPLE_REFLECTIVITY_KEY]

    if conv_2d3d and not upsample_refl:
        figure_index = 1
        radar_field_name = 'shear'
    else:
        figure_index = 0
        radar_field_name = None

    saliency_plotting.plot_many_2d_grids_with_contours(
        saliency_matrix_3d=numpy.flip(saliency_matrix, axis=0),
        axes_object_matrix=axes_object_matrices[figure_index],
        colour_map_object=colour_map_object,
        max_absolute_contour_level=max_colour_value,
        contour_interval=max_colour_value / HALF_NUM_CONTOURS,
        row_major=False)

    if significance_matrix is not None:
        significance_plotting.plot_many_2d_grids_without_coords(
            significance_matrix=numpy.flip(significance_matrix, axis=0),
            axes_object_matrix=axes_object_matrices[figure_index],
            row_major=False)

    allow_whitespace = figure_objects[figure_index]._suptitle is not None

    if allow_whitespace:
        this_title_string = '{0:s}; (max abs saliency = {1:.2e})'.format(
            figure_objects[figure_index]._suptitle.get_text(), max_colour_value
        )

        figure_objects[figure_index].suptitle(this_title_string)

    output_file_name = plot_input_examples.metadata_to_radar_fig_file_name(
        output_dir_name=output_dir_name, pmm_flag=pmm_flag,
        full_storm_id_string=full_storm_id_string,
        storm_time_string=storm_time_string, radar_field_name=radar_field_name)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_objects[figure_index].savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_objects[figure_index])


def _plot_sounding_saliency(
        sounding_matrix, saliency_matrix, model_metadata_dict,
        saliency_dict, colour_map_object, max_colour_value_by_example,
        output_dir_name):
    """Plots soundings along with their saliency maps.

    E = number of examples
    H = number of sounding heights
    F = number of sounding fields

    :param sounding_matrix: E-by-H-by-F numpy array of sounding values.
    :param saliency_matrix: E-by-H-by-F numpy array of saliency values.
    :param model_metadata_dict: See doc for `cnn.read_model_metadata`.
    :param saliency_dict: Dictionary returned by
        `saliency_maps.read_standard_file`.
    :param colour_map_object: See documentation at top of file.
    :param max_colour_value_by_example: length-E numpy array with max value in
        saliency colour scheme for each example.
    :param output_dir_name: Path to output directory.  Figure(s) will be saved
        here.
    """

    num_examples = sounding_matrix.shape[0]

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    sounding_field_names = training_option_dict[trainval_io.SOUNDING_FIELDS_KEY]
    sounding_heights_m_agl = training_option_dict[
        trainval_io.SOUNDING_HEIGHTS_KEY]

    if saliency_maps.SOUNDING_PRESSURES_KEY in saliency_dict:
        sounding_pressure_matrix_pa = numpy.expand_dims(
            saliency_dict[saliency_maps.SOUNDING_PRESSURES_KEY], axis=-1
        )

        this_sounding_matrix = numpy.concatenate(
            (sounding_matrix, sounding_pressure_matrix_pa), axis=-1
        )

        metpy_dict_by_example = dl_utils.soundings_to_metpy_dictionaries(
            sounding_matrix=this_sounding_matrix,
            field_names=sounding_field_names + [soundings.PRESSURE_NAME]
        )
    else:
        metpy_dict_by_example = dl_utils.soundings_to_metpy_dictionaries(
            sounding_matrix=sounding_matrix, field_names=sounding_field_names,
            height_levels_m_agl=sounding_heights_m_agl,
            storm_elevations_m_asl=numpy.full(num_examples, 0.)
        )

    full_storm_id_strings = None
    storm_times_unix_sec = None

    if saliency_maps.FULL_IDS_KEY in saliency_dict:
        full_storm_id_strings = saliency_dict[saliency_maps.FULL_IDS_KEY]

    if saliency_maps.STORM_TIMES_KEY in saliency_dict:
        storm_times_unix_sec = saliency_dict[saliency_maps.STORM_TIMES_KEY]

    pmm_flag = full_storm_id_strings is None and storm_times_unix_sec is None

    for i in range(num_examples):
        if pmm_flag:
            this_title_string = 'Probability-matched mean'
            this_base_file_name = '{0:s}/saliency_pmm'.format(output_dir_name)
        else:
            this_storm_time_string = time_conversion.unix_sec_to_string(
                storm_times_unix_sec[i], TIME_FORMAT)

            this_title_string = 'Storm "{0:s}" at {1:s}'.format(
                full_storm_id_strings[i], this_storm_time_string)

            this_base_file_name = '{0:s}/saliency_{1:s}_{2:s}'.format(
                output_dir_name, full_storm_id_strings[i].replace('_', '-'),
                this_storm_time_string
            )

        sounding_plotting.plot_sounding(
            sounding_dict_for_metpy=metpy_dict_by_example[i],
            title_string=this_title_string)

        this_left_file_name = '{0:s}_sounding-actual.jpg'.format(
            this_base_file_name)

        print('Saving figure to file: "{0:s}"...'.format(
            this_left_file_name
        ))
        pyplot.savefig(this_left_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()

        imagemagick_utils.trim_whitespace(
            input_file_name=this_left_file_name,
            output_file_name=this_left_file_name)

        saliency_plotting.plot_saliency_for_sounding(
            saliency_matrix=saliency_matrix[i, ...],
            sounding_field_names=sounding_field_names,
            pressure_levels_mb=metpy_dict_by_example[i][
                soundings.PRESSURE_COLUMN_METPY],
            colour_map_object=colour_map_object,
            max_absolute_colour_value=max_colour_value_by_example[i])

        this_right_file_name = '{0:s}_sounding-saliency.jpg'.format(
            this_base_file_name)

        print('Saving figure to file: "{0:s}"...'.format(
            this_right_file_name
        ))
        pyplot.savefig(this_right_file_name, dpi=FIGURE_RESOLUTION_DPI)
        pyplot.close()

        imagemagick_utils.trim_whitespace(
            input_file_name=this_right_file_name,
            output_file_name=this_right_file_name)

        this_file_name = '{0:s}_sounding.jpg'.format(this_base_file_name)
        print('Concatenating panels into file: "{0:s}"...\n'.format(
            this_file_name))

        imagemagick_utils.concatenate_images(
            input_file_names=[this_left_file_name, this_right_file_name],
            output_file_name=this_file_name, num_panel_rows=1,
            num_panel_columns=2)

        imagemagick_utils.resize_image(
            input_file_name=this_file_name,
            output_file_name=this_file_name,
            output_size_pixels=SOUNDING_IMAGE_SIZE_PX)

        os.remove(this_left_file_name)
        os.remove(this_right_file_name)


def _run(input_file_name, plot_significance, saliency_colour_map_name,
         max_colour_prctile_for_saliency, output_dir_name):
    """Plots saliency maps for a CNN (convolutional neural network).

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param plot_significance: Same.
    :param saliency_colour_map_name: Same.
    :param max_colour_prctile_for_saliency: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    error_checking.assert_is_geq(max_colour_prctile_for_saliency, 0.)
    error_checking.assert_is_leq(max_colour_prctile_for_saliency, 100.)
    saliency_colour_map_object = pyplot.cm.get_cmap(saliency_colour_map_name)

    print('Reading data from: "{0:s}"...'.format(input_file_name))

    try:
        saliency_dict = saliency_maps.read_standard_file(input_file_name)
        list_of_input_matrices = saliency_dict.pop(
            saliency_maps.INPUT_MATRICES_KEY)
        list_of_saliency_matrices = saliency_dict.pop(
            saliency_maps.SALIENCY_MATRICES_KEY)

        full_storm_id_strings = saliency_dict[saliency_maps.FULL_IDS_KEY]
        storm_times_unix_sec = saliency_dict[saliency_maps.STORM_TIMES_KEY]

        storm_time_strings = [
            time_conversion.unix_sec_to_string(
                t, plot_input_examples.TIME_FORMAT)
            for t in storm_times_unix_sec
        ]
    except ValueError:
        saliency_dict = saliency_maps.read_pmm_file(input_file_name)
        list_of_input_matrices = saliency_dict.pop(
            saliency_maps.MEAN_INPUT_MATRICES_KEY)
        list_of_saliency_matrices = saliency_dict.pop(
            saliency_maps.MEAN_SALIENCY_MATRICES_KEY)

        for i in range(len(list_of_input_matrices)):
            list_of_input_matrices[i] = numpy.expand_dims(
                list_of_input_matrices[i], axis=0
            )
            list_of_saliency_matrices[i] = numpy.expand_dims(
                list_of_saliency_matrices[i], axis=0
            )

        full_storm_id_strings = [None]
        storm_times_unix_sec = [None]
        storm_time_strings = [None]

    pmm_flag = (
        full_storm_id_strings[0] is None and storm_time_strings[0] is None
    )

    num_examples = list_of_input_matrices[0].shape[0]
    max_colour_value_by_example = numpy.full(num_examples, numpy.nan)

    for i in range(num_examples):
        these_saliency_values = numpy.concatenate(
            [numpy.ravel(s[i, ...]) for s in list_of_saliency_matrices]
        )
        max_colour_value_by_example[i] = numpy.percentile(
            numpy.absolute(these_saliency_values),
            max_colour_prctile_for_saliency
        )

    model_file_name = saliency_dict[saliency_maps.MODEL_FILE_KEY]
    model_metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0]
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_model_metadata(model_metafile_name)
    print(SEPARATOR_STRING)

    training_option_dict = model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY]
    include_soundings = (
        training_option_dict[trainval_io.SOUNDING_FIELDS_KEY] is not None
    )
    num_radar_matrices = len(list_of_input_matrices) - int(include_soundings)

    if include_soundings:

        # TODO(thunderhoser): Write version of this that plots only one example.
        _plot_sounding_saliency(
            sounding_matrix=list_of_input_matrices[-1],
            saliency_matrix=list_of_saliency_matrices[-1],
            model_metadata_dict=model_metadata_dict,
            saliency_dict=saliency_dict,
            colour_map_object=saliency_colour_map_object,
            max_colour_value_by_example=max_colour_value_by_example,
            output_dir_name=output_dir_name)

        print(SEPARATOR_STRING)

    monte_carlo_dict = (
        saliency_dict[saliency_maps.MONTE_CARLO_DICT_KEY]
        if plot_significance and
        saliency_maps.MONTE_CARLO_DICT_KEY in saliency_dict
        else None
    )

    for i in range(num_examples):

        # TODO(thunderhoser): Make sure to not plot soundings here.
        these_figure_objects, these_axes_object_matrices = (
            plot_input_examples.plot_one_example(
                list_of_predictor_matrices=list_of_input_matrices,
                model_metadata_dict=model_metadata_dict, example_index=i,
                allow_whitespace=True, pmm_flag=pmm_flag,
                full_storm_id_string=full_storm_id_strings[i],
                storm_time_unix_sec=storm_times_unix_sec[i]
            )
        )

        for j in range(num_radar_matrices):
            if monte_carlo_dict is None:
                this_significance_matrix = None
            else:
                this_significance_matrix = numpy.logical_or(
                    monte_carlo_dict[
                        monte_carlo.TRIAL_PMM_MATRICES_KEY][j][i, ...] <
                    monte_carlo_dict[monte_carlo.MIN_MATRICES_KEY][j][i, ...],
                    monte_carlo_dict[
                        monte_carlo.TRIAL_PMM_MATRICES_KEY][j][i, ...] >
                    monte_carlo_dict[monte_carlo.MAX_MATRICES_KEY][j][i, ...]
                )

            this_num_spatial_dim = len(list_of_input_matrices[j].shape) - 2

            if this_num_spatial_dim == 3:
                _plot_3d_radar_saliency(
                    saliency_matrix=list_of_saliency_matrices[0],
                    colour_map_object=saliency_colour_map_object,
                    max_colour_value=max_colour_value_by_example[i],
                    figure_objects=these_figure_objects,
                    axes_object_matrices=these_axes_object_matrices,
                    model_metadata_dict=model_metadata_dict,
                    output_dir_name=output_dir_name,
                    significance_matrix=this_significance_matrix,
                    full_storm_id_string=full_storm_id_strings[i],
                    storm_time_string=storm_time_strings[i]
                )
            else:
                _plot_2d_radar_saliency(
                    saliency_matrix=list_of_saliency_matrices[0],
                    colour_map_object=saliency_colour_map_object,
                    max_colour_value=max_colour_value_by_example[i],
                    figure_objects=these_figure_objects,
                    axes_object_matrices=these_axes_object_matrices,
                    model_metadata_dict=model_metadata_dict,
                    output_dir_name=output_dir_name,
                    significance_matrix=this_significance_matrix,
                    full_storm_id_string=full_storm_id_strings[i],
                    storm_time_string=storm_time_strings[i]
                )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        plot_significance=bool(getattr(
            INPUT_ARG_OBJECT, PLOT_SIGNIFICANCE_ARG_NAME)),
        saliency_colour_map_name=getattr(
            INPUT_ARG_OBJECT, COLOUR_MAP_ARG_NAME),
        max_colour_prctile_for_saliency=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
