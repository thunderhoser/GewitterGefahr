"""Plotting methods for feature optimization.

--- NOTATION ---

The following letters will be used throughout this module.

E = number of examples (storm objects)
M = number of grid rows per image
N = number of grid columns per image
H = number of grid heights per image (only for 3-D images)
F = number of radar fields per image (only for 3-D images)
C = number of radar channels (field/height pairs) per image (only for 2-D)
"""

import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import feature_optimization
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.plotting import radar_plotting

METRES_TO_KM = 1e-3

DEFAULT_FIG_WIDTH_INCHES = 15.
DEFAULT_FIG_HEIGHT_INCHES = 15.
DOTS_PER_INCH = 600


def _check_optimization_metadata(
        optimization_type_string, num_examples, target_class, layer_name,
        channel_index_by_example, neuron_index_matrix):
    """Error-checks metadata for feature optimization.

    E = number of examples to plot

    :param optimization_type_string: Optimization type used to create synthetic
        radar data (must be accepted by
        `feature_optimization.check_optimization_type`).
    :param num_examples: E in the above discussion.
    :param target_class:
        [used only if optimization_type_string = "class"]
        Class for which radar data were optimized.  For details, see
        `feature_optimization.optimize_input_for_class_activation`.
    :param layer_name:
        [used only if optimization_type_string = "neuron" or "channel"]
        Name of layer with neuron or channel whose activation was maximized.
        For details, see
        `feature_optimization.optimize_input_for_neuron_activation` or
        `feature_optimization.optimize_input_for_channel_activation`.
    :param channel_index_by_example:
        [used only if optimization_type_string = "channel"]
        length-E numpy array with indices of channels whose activations were
        maximized.  For details, see
        `feature_optimization.optimize_input_for_channel_activation`.
    :param neuron_index_matrix:
        [used only if optimization_type_string = "neuron"]
        E-by-? numpy array, where neuron_index_matrix[i, :] is the set of
        indices for the neuron whose activation is maximized by the [i]th
        example.  For more details, see doc for
        `feature_optimization.optimize_input_for_neuron_activation`.
    """

    feature_optimization.check_optimization_type(optimization_type_string)
    if (optimization_type_string ==
            feature_optimization.CLASS_OPTIMIZATION_TYPE_STRING):
        error_checking.assert_is_integer(target_class)
        error_checking.assert_is_geq(target_class, 0)
    else:
        error_checking.assert_is_string(layer_name)

    if (optimization_type_string ==
            feature_optimization.CHANNEL_OPTIMIZATION_TYPE_STRING):
        error_checking.assert_is_integer_numpy_array(channel_index_by_example)
        error_checking.assert_is_geq_numpy_array(channel_index_by_example, 0)
        error_checking.assert_is_numpy_array(
            channel_index_by_example,
            exact_dimensions=numpy.array([num_examples]))

    if (optimization_type_string ==
            feature_optimization.NEURON_OPTIMIZATION_TYPE_STRING):
        error_checking.assert_is_integer_numpy_array(neuron_index_matrix)
        error_checking.assert_is_geq_numpy_array(neuron_index_matrix, 0)
        num_indices_per_example = neuron_index_matrix.shape[-1]
        error_checking.assert_is_numpy_array(
            neuron_index_matrix,
            exact_dimensions=numpy.array(
                [num_examples, num_indices_per_example]))


def _optimization_metadata_to_strings(
        optimization_type_string, example_index, target_class, layer_name,
        channel_index_by_example, neuron_index_matrix):
    """Converts optimization metadata to strings.

    Specifically, this method creates two strings:

    - verbose string (to use in figure legends)
    - abbreviation (to use in file names)

    :param optimization_type_string: See doc for `_check_optimization_metadata`.
    :param example_index: This method will create strings for the [i]th example,
        where i = `example_index`.
    :param target_class: See doc for `_check_optimization_metadata`.
    :param layer_name: Same.
    :param channel_index_by_example: Same.
    :param neuron_index_matrix: Same.
    :return: verbose_string: See general discussion above.
    :return: abbrev_string: See general discussion above.
    """

    feature_optimization.check_optimization_type(optimization_type_string)
    if (optimization_type_string ==
            feature_optimization.CLASS_OPTIMIZATION_TYPE_STRING):
        verbose_string = 'class {0:d}'.format(target_class)
        abbrev_string = 'class{0:d}'.format(target_class)
    else:
        verbose_string = 'layer "{0:s}"'.format(layer_name)
        abbrev_string = 'layer={0:s}'.format(layer_name.replace('_', '-'))

    if (optimization_type_string ==
            feature_optimization.CHANNEL_OPTIMIZATION_TYPE_STRING):
        this_channel_index = channel_index_by_example[example_index]
        verbose_string += ', channel {0:d}'.format(this_channel_index)
        abbrev_string += '_channel{0:d}'.format(this_channel_index)

    if (optimization_type_string ==
            feature_optimization.NEURON_OPTIMIZATION_TYPE_STRING):
        these_neuron_indices = neuron_index_matrix[example_index, :]
        this_neuron_string = ', '.join(
            ['{0:d}'.format(i) for i in these_neuron_indices])
        verbose_string += '; neuron ({0:s})'.format(this_neuron_string)

        this_neuron_string = ','.join(
            ['{0:d}'.format(i) for i in these_neuron_indices])
        abbrev_string += '_neuron{0:s}'.format(this_neuron_string)

    return verbose_string, abbrev_string


def _init_panels(num_panel_rows, num_panel_columns, figure_width_inches,
                 figure_height_inches):
    """Initializes paneled figure.

    :param num_panel_rows: Number of panel rows.
    :param num_panel_columns: Number of panel columns.
    :param figure_width_inches: Figure width.
    :param figure_height_inches: Figure height.
    :return: figure_object: Instance of `matplotlib.figure.Figure`.
    :return: axes_objects_2d_list: 2-D list, where axes_objects_2d_list[j][k] is
        a `matplotlib.axes._subplots.AxesSubplot` object for the [j]th row and
        [k]th column.
    """

    figure_object, axes_objects_2d_list = pyplot.subplots(
        num_panel_rows, num_panel_columns,
        figsize=(figure_width_inches, figure_height_inches),
        sharex=True, sharey=True)
    pyplot.subplots_adjust(
        left=0.01, bottom=0.01, right=0.95, top=0.95, hspace=0, wspace=0)

    return figure_object, axes_objects_2d_list


def plot_optimized_field_2d(
        radar_field_matrix, radar_field_name, axes_object,
        annotation_string=None, colour_map_object=None,
        colour_norm_object=None):
    """Plots optimized 2-D radar field.

    :param radar_field_matrix: M-by-N numpy array with values of radar field.
    :param radar_field_name: See doc for
        `radar_plotting.plot_2d_grid_without_coords`.
    :param axes_object: Same.
    :param annotation_string: Same.
    :param colour_map_object: Same.
    :param colour_norm_object: Same.
    """

    radar_plotting.plot_2d_grid_without_coords(
        field_matrix=radar_field_matrix, field_name=radar_field_name,
        axes_object=axes_object, annotation_string=annotation_string,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object)


def plot_many_optimized_fields_3d(
        radar_field_matrix, radar_field_names, radar_heights_m_asl,
        one_figure_per_example, num_panel_rows, optimization_type_string,
        output_dir_name, figure_width_inches=DEFAULT_FIG_WIDTH_INCHES,
        figure_height_inches=DEFAULT_FIG_HEIGHT_INCHES, layer_name=None,
        target_class=None, channel_index_by_example=None,
        neuron_index_matrix=None):
    """Plots many optimized 3-D radar fields.

    :param radar_field_matrix: E-by-M-by-N-by-H-by-F numpy array of radar data.
    :param radar_field_names: length-F list of field names (each must be
        accepted by `radar_utils.check_field_name`).
    :param radar_heights_m_asl: length-H integer numpy array of radar heights
        (metres above sea level).
    :param one_figure_per_example: Boolean flag.  If True, this method will
        create one paneled figure for each example (storm object), where each
        panel contains a different radar field/height pair.  If False, will
        create one paneled figure for each field/height pair, where each panel
        contains a different example.
    :param num_panel_rows: [used only if `one_figure_per_example = False`]
        Number of rows in each paneled figure.
    :param optimization_type_string: See doc for `_check_optimization_metadata`.
    :param output_dir_name: See doc for `plot_many_optimized_fields_2d`.
    :param figure_width_inches: Same.
    :param figure_height_inches: Same.
    :param layer_name: See doc for `_check_optimization_metadata`.
    :param target_class: Same.
    :param channel_index_by_example: Same.
    :param neuron_index_matrix: Same.
    """

    dl_utils.check_radar_images(
        radar_image_matrix=radar_field_matrix, min_num_dimensions=5,
        max_num_dimensions=5)
    num_examples = radar_field_matrix.shape[0]
    num_fields = radar_field_matrix.shape[-1]
    num_heights = radar_field_matrix.shape[-2]

    _check_optimization_metadata(
        optimization_type_string=optimization_type_string,
        num_examples=num_examples, target_class=target_class,
        layer_name=layer_name,
        channel_index_by_example=channel_index_by_example,
        neuron_index_matrix=neuron_index_matrix)

    error_checking.assert_is_numpy_array(
        numpy.array(radar_field_names),
        exact_dimensions=numpy.array([num_fields]))

    error_checking.assert_is_integer_numpy_array(radar_heights_m_asl)
    error_checking.assert_is_geq_numpy_array(radar_heights_m_asl, 0)
    error_checking.assert_is_numpy_array(
        radar_heights_m_asl, exact_dimensions=numpy.array([num_heights]))

    error_checking.assert_is_boolean(one_figure_per_example)
    if one_figure_per_example:
        num_panel_rows = num_fields + 0
        num_panel_columns = num_heights + 0
    else:
        error_checking.assert_is_integer(num_panel_rows)
        error_checking.assert_is_geq(num_panel_rows, 1)
        error_checking.assert_is_leq(num_panel_rows, num_examples)

        num_panel_columns = int(
            numpy.ceil(float(num_examples) / num_panel_rows))

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    if one_figure_per_example:
        for i in range(num_examples):
            _, axes_objects_2d_list = _init_panels(
                num_panel_rows=num_panel_rows,
                num_panel_columns=num_panel_columns,
                figure_width_inches=figure_width_inches,
                figure_height_inches=figure_height_inches)

            for j in range(num_panel_rows):
                for k in range(num_panel_columns):
                    this_annotation_string = '{0:s} at {1:.1f} km'.format(
                        radar_field_names[j],
                        radar_heights_m_asl[k] * METRES_TO_KM)

                    radar_plotting.plot_2d_grid_without_coords(
                        field_matrix=radar_field_matrix[i, ..., k, j],
                        field_name=radar_field_names[j],
                        axes_object=axes_objects_2d_list[j][k],
                        annotation_string=this_annotation_string)

            _, this_metadata_string = _optimization_metadata_to_strings(
                optimization_type_string=optimization_type_string,
                example_index=i, target_class=target_class,
                layer_name=layer_name,
                channel_index_by_example=channel_index_by_example,
                neuron_index_matrix=neuron_index_matrix)
            this_figure_file_name = '{0:s}/optimized-radar_{1:s}.jpg'.format(
                output_dir_name, this_metadata_string)

            print 'Saving figure to file: "{0:s}"...'.format(
                this_figure_file_name)
            pyplot.savefig(this_figure_file_name, dpi=DOTS_PER_INCH)
            pyplot.close()

    else:
        for i in range(num_fields):
            for m in range(num_heights):
                _, axes_objects_2d_list = _init_panels(
                    num_panel_rows=num_panel_rows,
                    num_panel_columns=num_panel_columns,
                    figure_width_inches=figure_width_inches,
                    figure_height_inches=figure_height_inches)

                for j in range(num_panel_rows):
                    for k in range(num_panel_columns):
                        this_example_index = j * num_panel_columns + k
                        (this_annotation_string, _
                        ) = _optimization_metadata_to_strings(
                            optimization_type_string=optimization_type_string,
                            example_index=this_example_index,
                            target_class=target_class, layer_name=layer_name,
                            channel_index_by_example=channel_index_by_example,
                            neuron_index_matrix=neuron_index_matrix)

                        radar_plotting.plot_2d_grid_without_coords(
                            field_matrix=radar_field_matrix[
                                this_example_index, ..., m, i],
                            field_name=radar_field_names[i],
                            axes_object=axes_objects_2d_list[j][k],
                            annotation_string=this_annotation_string)

                this_figure_file_name = (
                    '{0:s}/optimized-radar_{1:s}_{2:05d}metres.jpg'
                ).format(output_dir_name, radar_field_names[i],
                         radar_heights_m_asl[m])
                pyplot.savefig(this_figure_file_name, dpi=DOTS_PER_INCH)
                pyplot.close()


def plot_many_optimized_fields_2d(
        radar_field_matrix, field_name_by_pair, height_by_pair_m_asl,
        one_figure_per_example, num_panel_rows, optimization_type_string,
        output_dir_name, figure_width_inches=DEFAULT_FIG_WIDTH_INCHES,
        figure_height_inches=DEFAULT_FIG_HEIGHT_INCHES, layer_name=None,
        target_class=None, channel_index_by_example=None,
        neuron_index_matrix=None):
    """Plots many optimized 2-D radar fields.

    :param radar_field_matrix: E-by-M-by-N-by-C numpy array of radar data.
    :param field_name_by_pair: length-C list of field names (each must be
        accepted by `radar_utils.check_field_name`).
    :param height_by_pair_m_asl: length-C integer numpy array of radar heights
        (metres above sea level).
    :param one_figure_per_example: Boolean flag.  If True, this method will
        create one paneled figure for each example (storm object), where each
        panel contains a different radar field/height pair.  If False, will
        create one paneled figure for each field/height pair, where each panel
        contains a different example.
    :param num_panel_rows: Number of rows in each paneled figure.
    :param optimization_type_string: See doc for `_check_optimization_metadata`.
    :param output_dir_name: Name of output directory (figures will be saved
        here).
    :param figure_width_inches: Width of each figure.
    :param figure_height_inches: Height of each figure.
    :param layer_name: See doc for `_check_optimization_metadata`.
    :param target_class: Same.
    :param channel_index_by_example: Same.
    :param neuron_index_matrix: Same.
    """

    dl_utils.check_radar_images(
        radar_image_matrix=radar_field_matrix, min_num_dimensions=4,
        max_num_dimensions=4)
    num_examples = radar_field_matrix.shape[0]
    num_field_height_pairs = radar_field_matrix.shape[-1]

    _check_optimization_metadata(
        optimization_type_string=optimization_type_string,
        num_examples=num_examples, target_class=target_class,
        layer_name=layer_name,
        channel_index_by_example=channel_index_by_example,
        neuron_index_matrix=neuron_index_matrix)

    error_checking.assert_is_numpy_array(
        numpy.array(field_name_by_pair),
        exact_dimensions=numpy.array([num_field_height_pairs]))

    error_checking.assert_is_integer_numpy_array(height_by_pair_m_asl)
    error_checking.assert_is_geq_numpy_array(height_by_pair_m_asl, 0)
    error_checking.assert_is_numpy_array(
        height_by_pair_m_asl,
        exact_dimensions=numpy.array([num_field_height_pairs]))

    error_checking.assert_is_boolean(one_figure_per_example)
    error_checking.assert_is_integer(num_panel_rows)
    error_checking.assert_is_geq(num_panel_rows, 1)

    if one_figure_per_example:
        error_checking.assert_is_leq(num_panel_rows, num_field_height_pairs)
        num_panel_columns = int(
            numpy.ceil(float(num_field_height_pairs) / num_panel_rows))
    else:
        error_checking.assert_is_leq(num_panel_rows, num_examples)
        num_panel_columns = int(
            numpy.ceil(float(num_examples) / num_panel_rows))

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    if one_figure_per_example:
        for i in range(num_examples):
            _, axes_objects_2d_list = _init_panels(
                num_panel_rows=num_panel_rows,
                num_panel_columns=num_panel_columns,
                figure_width_inches=figure_width_inches,
                figure_height_inches=figure_height_inches)

            for j in range(num_panel_rows):
                for k in range(num_panel_columns):
                    this_fh_pair_index = j * num_panel_columns + k
                    this_annotation_string = '{0:s}'.format(
                        field_name_by_pair[this_fh_pair_index])

                    if (field_name_by_pair[this_fh_pair_index] ==
                            radar_utils.REFL_NAME):
                        this_annotation_string += ' at {0:.1f} km'.format(
                            height_by_pair_m_asl[this_fh_pair_index] *
                            METRES_TO_KM)

                    radar_plotting.plot_2d_grid_without_coords(
                        field_matrix=radar_field_matrix[
                            i, ..., this_fh_pair_index],
                        field_name=field_name_by_pair[this_fh_pair_index],
                        axes_object=axes_objects_2d_list[j][k],
                        annotation_string=this_annotation_string)

            _, this_metadata_string = _optimization_metadata_to_strings(
                optimization_type_string=optimization_type_string,
                example_index=i, target_class=target_class,
                layer_name=layer_name,
                channel_index_by_example=channel_index_by_example,
                neuron_index_matrix=neuron_index_matrix)
            this_figure_file_name = '{0:s}/optimized-radar_{1:s}.jpg'.format(
                output_dir_name, this_metadata_string)

            print 'Saving figure to file: "{0:s}"...'.format(
                this_figure_file_name)
            pyplot.savefig(this_figure_file_name, dpi=DOTS_PER_INCH)
            pyplot.close()

    else:
        for i in range(num_field_height_pairs):
            _, axes_objects_2d_list = _init_panels(
                num_panel_rows=num_panel_rows,
                num_panel_columns=num_panel_columns,
                figure_width_inches=figure_width_inches,
                figure_height_inches=figure_height_inches)

            for j in range(num_panel_rows):
                for k in range(num_panel_columns):
                    this_example_index = j * num_panel_columns + k
                    radar_plotting.plot_2d_grid_without_coords(
                        field_matrix=radar_field_matrix[
                            this_example_index, ..., i],
                        field_name=field_name_by_pair[i],
                        axes_object=axes_objects_2d_list[j][k],
                        annotation_string='Example {0:d}'.format(
                            this_example_index))

            this_figure_file_name = (
                '{0:s}/optimized-radar_{1:s}_{2:05d}metres.jpg'
            ).format(output_dir_name, field_name_by_pair[i],
                     height_by_pair_m_asl[i])
            pyplot.savefig(this_figure_file_name, dpi=DOTS_PER_INCH)
            pyplot.close()
