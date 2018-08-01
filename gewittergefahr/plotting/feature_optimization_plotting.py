"""Plotting methods for feature optimization.

--- NOTATION ---

The following letters will be used throughout this module.

E = number of model components for which input data were optimized
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
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import radar_plotting

# TODO(thunderhoser): Allow this module to plot optimized soundings, not just
# optimized radar fields.

matplotlib.rcParams['axes.linewidth'] = 2

METRES_TO_KM = 1e-3
DEFAULT_FIG_WIDTH_INCHES = 15.
DEFAULT_FIG_HEIGHT_INCHES = 15.
DOTS_PER_INCH = 600


def _model_component_to_string(
        component_index, component_type_string, target_class=None,
        layer_name=None, neuron_index_matrix=None, channel_indices=None):
    """Returns string descriptions for model component (class/neuron/channel).

    C = number of model components

    :param component_index: Will return descriptions for the [j]th component,
        where j = `component_index`.
    :param component_type_string: See doc for
        `feature_optimization.model_component_to_string`.
    :param target_class: Same.
    :param layer_name: Same.
    :param neuron_index_matrix: C-by-? numpy array, where
        neuron_index_matrix[j, :] contains array indices for the [j]th neuron.
    :param channel_indices: length-C numpy array, where channel_indices[j] is
        the index of the [j]th channel.
    :return: verbose_string: Verbose string (to use in figure legends).
    :return: abbrev_string: Abbreviation (to use in file names).
    """

    if (component_type_string ==
            feature_optimization.CLASS_COMPONENT_TYPE_STRING):
        return feature_optimization.model_component_to_string(
            component_type_string=component_type_string,
            target_class=target_class)

    if (component_type_string ==
            feature_optimization.NEURON_COMPONENT_TYPE_STRING):
        return feature_optimization.model_component_to_string(
            component_type_string=component_type_string, layer_name=layer_name,
            neuron_indices=neuron_index_matrix[component_index, :])

    return feature_optimization.model_component_to_string(
        component_type_string=component_type_string, layer_name=layer_name,
        channel_index=channel_indices[component_index])


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


def plot_many_optimized_fields_2d(
        radar_field_matrix, field_name_by_pair, height_by_pair_m_asl,
        one_figure_per_component, num_panel_rows, component_type_string,
        output_dir_name, figure_width_inches=DEFAULT_FIG_WIDTH_INCHES,
        figure_height_inches=DEFAULT_FIG_HEIGHT_INCHES, target_class=None,
        layer_name=None, neuron_index_matrix=None, channel_indices=None):
    """Plots many optimized 2-D radar fields.

    :param radar_field_matrix: E-by-M-by-N-by-C numpy array of radar data.
    :param field_name_by_pair: length-C list of field names (each must be
        accepted by `radar_utils.check_field_name`).
    :param height_by_pair_m_asl: length-C integer numpy array of radar heights
        (metres above sea level).
    :param one_figure_per_component: Boolean flag.  If True, this method will
        created one paneled figure for each model component, where each panel
        contains a different radar field/height.  If False, will create one
        paneled figure for each radar field/height, where each panel contains
        the optimized field from a different model component.
    :param num_panel_rows: Number of panel rows in each figure.
    :param component_type_string: See doc for
        `feature_optimization.check_optimization_metadata`.
    :param output_dir_name: Name of output directory (figures will be saved
        here).
    :param figure_width_inches: Width of each figure.
    :param figure_height_inches: Height of each figure.
    :param target_class: See doc for
        `feature_optimization.check_optimization_metadata`.
    :param layer_name: Same.
    :param neuron_index_matrix: Same.
    :param channel_indices: Same.
    """

    feature_optimization.check_optimization_metadata(
        num_iterations=feature_optimization.DEFAULT_NUM_ITERATIONS,
        learning_rate=feature_optimization.DEFAULT_LEARNING_RATE,
        component_type_string=component_type_string,
        target_class=target_class, optimize_for_probability=True,
        layer_name=layer_name, ideal_activation=1.,
        neuron_index_matrix=neuron_index_matrix,
        channel_indices=channel_indices)

    error_checking.assert_is_numpy_array(radar_field_matrix, num_dimensions=4)
    num_components = radar_field_matrix.shape[0]
    num_field_height_pairs = radar_field_matrix.shape[-1]

    error_checking.assert_is_numpy_array(
        numpy.array(field_name_by_pair),
        exact_dimensions=numpy.array([num_field_height_pairs]))

    error_checking.assert_is_integer_numpy_array(height_by_pair_m_asl)
    error_checking.assert_is_geq_numpy_array(height_by_pair_m_asl, 0)
    error_checking.assert_is_numpy_array(
        height_by_pair_m_asl,
        exact_dimensions=numpy.array([num_field_height_pairs]))

    error_checking.assert_is_boolean(one_figure_per_component)
    error_checking.assert_is_integer(num_panel_rows)
    error_checking.assert_is_geq(num_panel_rows, 1)

    if one_figure_per_component:
        error_checking.assert_is_leq(num_panel_rows, num_field_height_pairs)
        num_panel_columns = int(
            numpy.ceil(float(num_field_height_pairs) / num_panel_rows))
    else:
        error_checking.assert_is_leq(num_panel_rows, num_components)
        num_panel_columns = int(
            numpy.ceil(float(num_components) / num_panel_rows))

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    if one_figure_per_component:
        for i in range(num_components):
            _, axes_objects_2d_list = plotting_utils.init_panels(
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
                        this_annotation_string += '\nat {0:.1f} km'.format(
                            height_by_pair_m_asl[this_fh_pair_index] *
                            METRES_TO_KM)

                    radar_plotting.plot_2d_grid_without_coords(
                        field_matrix=radar_field_matrix[
                            i, ..., this_fh_pair_index],
                        field_name=field_name_by_pair[this_fh_pair_index],
                        axes_object=axes_objects_2d_list[j][k],
                        annotation_string=this_annotation_string)

            _, this_metadata_string = _model_component_to_string(
                component_index=i, component_type_string=component_type_string,
                target_class=target_class, layer_name=layer_name,
                neuron_index_matrix=neuron_index_matrix,
                channel_indices=channel_indices)

            this_figure_file_name = '{0:s}/optimized-radar_{1:s}.jpg'.format(
                output_dir_name, this_metadata_string)

            print 'Saving figure to file: "{0:s}"...'.format(
                this_figure_file_name)
            pyplot.savefig(this_figure_file_name, dpi=DOTS_PER_INCH)
            pyplot.close()
    else:
        for i in range(num_field_height_pairs):
            _, axes_objects_2d_list = plotting_utils.init_panels(
                num_panel_rows=num_panel_rows,
                num_panel_columns=num_panel_columns,
                figure_width_inches=figure_width_inches,
                figure_height_inches=figure_height_inches)

            for j in range(num_panel_rows):
                for k in range(num_panel_columns):
                    this_component_index = j * num_panel_columns + k
                    this_annotation_string, _ = _model_component_to_string(
                        component_index=this_component_index,
                        component_type_string=component_type_string,
                        target_class=target_class, layer_name=layer_name,
                        neuron_index_matrix=neuron_index_matrix,
                        channel_indices=channel_indices)

                    radar_plotting.plot_2d_grid_without_coords(
                        field_matrix=radar_field_matrix[
                            this_component_index, ..., i],
                        field_name=field_name_by_pair[i],
                        axes_object=axes_objects_2d_list[j][k],
                        annotation_string=this_annotation_string)

            this_figure_file_name = (
                '{0:s}/optimized-radar_{1:s}_{2:05d}metres.jpg'
            ).format(output_dir_name, field_name_by_pair[i].replace('_', '-'),
                     height_by_pair_m_asl[i])

            print 'Saving figure to file: "{0:s}"...'.format(
                this_figure_file_name)
            pyplot.savefig(this_figure_file_name, dpi=DOTS_PER_INCH)
            pyplot.close()


def plot_many_optimized_fields_3d(
        radar_field_matrix, radar_field_names, radar_heights_m_asl,
        one_figure_per_component, component_type_string, output_dir_name,
        num_panel_rows=None, figure_width_inches=DEFAULT_FIG_WIDTH_INCHES,
        figure_height_inches=DEFAULT_FIG_HEIGHT_INCHES, target_class=None,
        layer_name=None, neuron_index_matrix=None, channel_indices=None):
    """Plots many optimized 3-D radar fields.

    :param radar_field_matrix: E-by-M-by-N-by-H-by-F numpy array of radar data.
    :param radar_field_names: length-F list of field names (each must be
        accepted by `radar_utils.check_field_name`).
    :param radar_heights_m_asl: length-H integer numpy array of radar heights
        (metres above sea level).
    :param one_figure_per_component: See doc for
        `plot_many_optimized_fields_2d`.
    :param component_type_string: See doc for
        `feature_optimization.check_optimization_metadata`.
    :param output_dir_name: Name of output directory (figures will be saved
        here).
    :param num_panel_rows: [used only if `one_figure_per_component = False`]
        Number of panel rows in each figure.
    :param figure_width_inches: Width of each figure.
    :param figure_height_inches: Height of each figure.
    :param target_class: See doc for
        `feature_optimization.check_optimization_metadata`.
    :param layer_name: Same.
    :param neuron_index_matrix: Same.
    :param channel_indices: Same.
    """

    feature_optimization.check_optimization_metadata(
        num_iterations=feature_optimization.DEFAULT_NUM_ITERATIONS,
        learning_rate=feature_optimization.DEFAULT_LEARNING_RATE,
        component_type_string=component_type_string,
        target_class=target_class, optimize_for_probability=True,
        layer_name=layer_name, ideal_activation=1.,
        neuron_index_matrix=neuron_index_matrix,
        channel_indices=channel_indices)

    error_checking.assert_is_numpy_array(radar_field_matrix, num_dimensions=5)
    num_components = radar_field_matrix.shape[0]
    num_fields = radar_field_matrix.shape[-1]
    num_heights = radar_field_matrix.shape[-2]

    error_checking.assert_is_numpy_array(
        numpy.array(radar_field_names),
        exact_dimensions=numpy.array([num_fields]))

    error_checking.assert_is_integer_numpy_array(radar_heights_m_asl)
    error_checking.assert_is_geq_numpy_array(radar_heights_m_asl, 0)
    error_checking.assert_is_numpy_array(
        radar_heights_m_asl, exact_dimensions=numpy.array([num_heights]))

    error_checking.assert_is_boolean(one_figure_per_component)
    if one_figure_per_component:
        num_panel_rows = num_fields + 0
        num_panel_columns = num_heights + 0
    else:
        error_checking.assert_is_integer(num_panel_rows)
        error_checking.assert_is_geq(num_panel_rows, 1)
        error_checking.assert_is_leq(num_panel_rows, num_components)

        num_panel_columns = int(
            numpy.ceil(float(num_components) / num_panel_rows))

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    if one_figure_per_component:
        for i in range(num_components):
            _, axes_objects_2d_list = plotting_utils.init_panels(
                num_panel_rows=num_panel_rows,
                num_panel_columns=num_panel_columns,
                figure_width_inches=figure_width_inches,
                figure_height_inches=figure_height_inches)

            for j in range(num_panel_rows):
                for k in range(num_panel_columns):
                    this_annotation_string = '{0:s}\nat {1:.1f} km'.format(
                        radar_field_names[j],
                        radar_heights_m_asl[k] * METRES_TO_KM)

                    radar_plotting.plot_2d_grid_without_coords(
                        field_matrix=radar_field_matrix[i, ..., k, j],
                        field_name=radar_field_names[j],
                        axes_object=axes_objects_2d_list[j][k],
                        annotation_string=this_annotation_string)

            _, this_metadata_string = _model_component_to_string(
                component_index=i, component_type_string=component_type_string,
                target_class=target_class, layer_name=layer_name,
                neuron_index_matrix=neuron_index_matrix,
                channel_indices=channel_indices)
            this_figure_file_name = '{0:s}/optimized-radar_{1:s}.jpg'.format(
                output_dir_name, this_metadata_string)

            print 'Saving figure to file: "{0:s}"...'.format(
                this_figure_file_name)
            pyplot.savefig(this_figure_file_name, dpi=DOTS_PER_INCH)
            pyplot.close()

    else:
        for i in range(num_fields):
            for m in range(num_heights):
                _, axes_objects_2d_list = plotting_utils.init_panels(
                    num_panel_rows=num_panel_rows,
                    num_panel_columns=num_panel_columns,
                    figure_width_inches=figure_width_inches,
                    figure_height_inches=figure_height_inches)

                for j in range(num_panel_rows):
                    for k in range(num_panel_columns):
                        this_component_index = j * num_panel_columns + k
                        this_annotation_string, _ = _model_component_to_string(
                            component_index=this_component_index,
                            component_type_string=component_type_string,
                            target_class=target_class, layer_name=layer_name,
                            neuron_index_matrix=neuron_index_matrix,
                            channel_indices=channel_indices)

                        radar_plotting.plot_2d_grid_without_coords(
                            field_matrix=radar_field_matrix[
                                this_component_index, ..., m, i],
                            field_name=radar_field_names[i],
                            axes_object=axes_objects_2d_list[j][k],
                            annotation_string=this_annotation_string)

                this_figure_file_name = (
                    '{0:s}/optimized-radar_{1:s}_{2:05d}metres.jpg'
                ).format(output_dir_name,
                         radar_field_names[i].replace('_', '-'),
                         radar_heights_m_asl[m])

                print 'Saving figure to file: "{0:s}"...'.format(
                    this_figure_file_name)
                pyplot.savefig(this_figure_file_name, dpi=DOTS_PER_INCH)
                pyplot.close()
