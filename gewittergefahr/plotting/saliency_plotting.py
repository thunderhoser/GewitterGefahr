"""Plots saliency maps."""

import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils

DEFAULT_MIN_FONT_SIZE = 10.
DEFAULT_MAX_FONT_SIZE = 25.
DEFAULT_MIN_SOUNDING_FONT_SIZE = 24.
DEFAULT_MAX_SOUNDING_FONT_SIZE = 60.

WIND_NAME = 'wind_m_s01'
WIND_COMPONENT_NAMES = [soundings.U_WIND_NAME, soundings.V_WIND_NAME]

WIND_BARB_LENGTH = 10.
EMPTY_WIND_BARB_RADIUS = 0.2
WIND_SALIENCY_MULTIPLIER = 52.5

SOUNDING_FIELD_NAME_TO_ABBREV_DICT = {
    soundings.SPECIFIC_HUMIDITY_NAME: r'$q_{v}$',
    soundings.VIRTUAL_POTENTIAL_TEMPERATURE_NAME: r'$\theta_{v}$',
    soundings.TEMPERATURE_NAME: r'$T$',
    soundings.RELATIVE_HUMIDITY_NAME: 'RH',
    soundings.U_WIND_NAME: r'$u$',
    soundings.V_WIND_NAME: r'$v$',
    soundings.PRESSURE_NAME: r'$p$',
    WIND_NAME: 'Wind'
}


def _saliency_to_colour_and_size(
        saliency_matrix, colour_map_object, max_colour_value,
        min_font_size_points, max_font_size_points):
    """Returns font size and colour for each saliency value.

    :param saliency_matrix: M-by-N numpy array of saliency values.
    :param colour_map_object: See doc for `plot_saliency_for_radar`.
    :param max_colour_value: Same.
    :param min_font_size_points: Same.
    :param max_font_size_points: Same.
    :return: rgb_matrix: M-by-N-by-3 numpy array of colours.
    :return: font_size_matrix_points: M-by-N numpy array of font sizes.
    """

    error_checking.assert_is_greater(max_colour_value, 0.)
    error_checking.assert_is_greater(min_font_size_points, 0.)
    error_checking.assert_is_greater(max_font_size_points, min_font_size_points)

    colour_norm_object = pyplot.Normalize(vmin=0., vmax=max_colour_value)
    rgba_matrix = colour_map_object(colour_norm_object(
        numpy.absolute(saliency_matrix)))
    rgb_matrix = rgba_matrix[..., :-1]

    norm_abs_saliency_matrix = (
        numpy.absolute(saliency_matrix) / max_colour_value)
    norm_abs_saliency_matrix[norm_abs_saliency_matrix > 1.] = 1.
    font_size_matrix_points = (
        min_font_size_points + norm_abs_saliency_matrix *
        (max_font_size_points - min_font_size_points)
    )

    return rgb_matrix, font_size_matrix_points


def plot_saliency_for_sounding(
        saliency_matrix, sounding_field_names, pressure_levels_mb, axes_object,
        colour_map_object, max_absolute_colour_value,
        min_font_size=DEFAULT_MIN_SOUNDING_FONT_SIZE,
        max_font_size=DEFAULT_MAX_SOUNDING_FONT_SIZE):
    """Plots saliency for one sounding.

    P = number of pressure levels
    F = number of fields

    :param saliency_matrix: P-by-F numpy array of saliency values.
    :param sounding_field_names: length-F list of field names.
    :param pressure_levels_mb: length-P list of pressure levels (millibars).
    :param axes_object: See doc for `plot_2d_grid`.
    :param colour_map_object: Same.
    :param max_absolute_colour_value: Same.
    :param min_font_size: Same.
    :param max_font_size: Same.
    """

    error_checking.assert_is_greater_numpy_array(pressure_levels_mb, 0.)
    error_checking.assert_is_numpy_array(pressure_levels_mb, num_dimensions=1)

    error_checking.assert_is_list(sounding_field_names)
    error_checking.assert_is_numpy_array(
        numpy.array(sounding_field_names), num_dimensions=1)

    num_pressure_levels = len(pressure_levels_mb)
    num_sounding_fields = len(sounding_field_names)

    error_checking.assert_is_numpy_array_without_nan(saliency_matrix)
    error_checking.assert_is_numpy_array(
        saliency_matrix,
        exact_dimensions=numpy.array([num_pressure_levels, num_sounding_fields])
    )

    try:
        u_wind_index = sounding_field_names.index(soundings.U_WIND_NAME)
        v_wind_index = sounding_field_names.index(soundings.V_WIND_NAME)
        plot_wind_barbs = True
    except ValueError:
        plot_wind_barbs = False

    if plot_wind_barbs:
        u_wind_saliency_values = saliency_matrix[:, u_wind_index]
        v_wind_saliency_values = saliency_matrix[:, v_wind_index]
        wind_saliency_magnitudes = numpy.sqrt(
            u_wind_saliency_values ** 2 + v_wind_saliency_values ** 2)

        colour_norm_object = pyplot.Normalize(
            vmin=0., vmax=max_absolute_colour_value)
        rgb_matrix_for_wind = colour_map_object(colour_norm_object(
            wind_saliency_magnitudes
        ))[..., :-1]

        non_wind_flags = numpy.array(
            [f not in WIND_COMPONENT_NAMES for f in sounding_field_names],
            dtype=bool)
        non_wind_indices = numpy.where(non_wind_flags)[0]

        saliency_matrix = saliency_matrix[:, non_wind_indices]
        sounding_field_names = [
            sounding_field_names[k] for k in non_wind_indices
        ]

        sounding_field_names.append(WIND_NAME)
        num_sounding_fields = len(sounding_field_names)

    rgb_matrix, font_size_matrix = _saliency_to_colour_and_size(
        saliency_matrix=saliency_matrix, colour_map_object=colour_map_object,
        max_colour_value=max_absolute_colour_value,
        min_font_size_points=min_font_size, max_font_size_points=max_font_size)

    for k in range(num_sounding_fields):
        if sounding_field_names[k] == WIND_NAME:
            for j in range(num_pressure_levels):
                this_vector = numpy.array([
                    u_wind_saliency_values[j], v_wind_saliency_values[j]
                ])
                this_vector = (
                    WIND_SALIENCY_MULTIPLIER * this_vector
                    / numpy.linalg.norm(this_vector, ord=2)
                )

                axes_object.barbs(
                    k, pressure_levels_mb[j], this_vector[0], this_vector[1],
                    length=WIND_BARB_LENGTH, fill_empty=True, rounding=False,
                    sizes={'emptybarb': EMPTY_WIND_BARB_RADIUS},
                    color=rgb_matrix_for_wind[j, ...])

            continue

        for j in range(num_pressure_levels):
            if saliency_matrix[j, k] >= 0:
                axes_object.text(
                    k, pressure_levels_mb[j], '+',
                    fontsize=font_size_matrix[j, k],
                    color=rgb_matrix[j, k, ...], horizontalalignment='center',
                    verticalalignment='center')
            else:
                axes_object.text(
                    j, pressure_levels_mb[j], '_',
                    fontsize=font_size_matrix[j, k],
                    color=rgb_matrix[j, k, ...], horizontalalignment='center',
                    verticalalignment='bottom')

    axes_object.set_xlim(-0.5, num_sounding_fields - 0.5)
    axes_object.set_ylim(100, 1000)
    axes_object.invert_yaxis()
    pyplot.yscale('log')
    pyplot.minorticks_off()

    y_tick_locations = numpy.linspace(100, 1000, num=10, dtype=int)
    y_tick_labels = ['{0:d}'.format(p) for p in y_tick_locations]
    pyplot.yticks(y_tick_locations, y_tick_labels)

    x_tick_locations = numpy.linspace(
        0, num_sounding_fields - 1, num=num_sounding_fields, dtype=float)
    x_tick_labels = [
        SOUNDING_FIELD_NAME_TO_ABBREV_DICT[f] for f in sounding_field_names
    ]
    pyplot.xticks(x_tick_locations, x_tick_labels)

    colour_bar_object = plotting_utils.add_linear_colour_bar(
        axes_object_or_list=axes_object, values_to_colour=saliency_matrix,
        colour_map=colour_map_object, colour_min=0.,
        colour_max=max_absolute_colour_value,
        orientation='vertical', extend_min=True, extend_max=True)

    colour_bar_object.set_label('Saliency (absolute value)')


def plot_2d_grid(saliency_matrix_2d, axes_object, colour_map_object,
                 max_absolute_colour_value, min_font_size=DEFAULT_MIN_FONT_SIZE,
                 max_font_size=DEFAULT_MAX_FONT_SIZE):
    """Plots 2-D saliency map with "+" and "-" signs.

    M = number of rows in grid
    N = number of columns in grid

    :param saliency_matrix_2d: M-by-N numpy array of saliency values.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param colour_map_object: Instance of `matplotlib.pyplot.cm`.  This colour
        map will be applied to absolute values, rather than signed values.  In
        other words, this colour map will be duplicated and flipped, to create a
        diverging colour map.  Thus, `colour_map_object` itself should be a
        sequential colour map, not a diverging one.  However, this is not
        enforced by the code, so do whatever you want.
    :param max_absolute_colour_value: Max absolute saliency in colour scheme.
        The min and max values, respectively, will be
        `-1 * max_absolute_colour_value` and `max_absolute_colour_value`.
    :param min_font_size: Minimum font size (for zero saliency).
    :param max_font_size: Max font size (for `max_absolute_colour_value`).
    """

    error_checking.assert_is_numpy_array_without_nan(saliency_matrix_2d)
    error_checking.assert_is_numpy_array(saliency_matrix_2d, num_dimensions=2)
    # error_checking.assert_is_greater(max_absolute_colour_value, 0.)
    # error_checking.assert_is_greater(min_font_size, 0.)
    # error_checking.assert_is_greater(max_font_size, min_font_size)

    rgb_matrix, font_size_matrix = _saliency_to_colour_and_size(
        saliency_matrix=saliency_matrix_2d, colour_map_object=colour_map_object,
        max_colour_value=max_absolute_colour_value,
        min_font_size_points=min_font_size, max_font_size_points=max_font_size)

    num_grid_rows = saliency_matrix_2d.shape[0]
    num_grid_columns = saliency_matrix_2d.shape[1]
    x_coord_spacing = num_grid_columns ** -1
    y_coord_spacing = num_grid_rows ** -1

    x_coords, y_coords = grids.get_xy_grid_points(
        x_min_metres=x_coord_spacing / 2, y_min_metres=y_coord_spacing / 2,
        x_spacing_metres=x_coord_spacing, y_spacing_metres=y_coord_spacing,
        num_rows=num_grid_rows, num_columns=num_grid_columns)

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            if saliency_matrix_2d[i, j] >= 0:
                axes_object.text(
                    x_coords[i], y_coords[j], '+',
                    fontsize=font_size_matrix[i, j],
                    color=rgb_matrix[i, j, ...], horizontalalignment='center',
                    verticalalignment='center', transform=axes_object.transAxes)
            else:
                axes_object.text(
                    x_coords[i], y_coords[j], '_',
                    fontsize=font_size_matrix[i, j],
                    color=rgb_matrix[i, j, ...], horizontalalignment='center',
                    verticalalignment='bottom', transform=axes_object.transAxes)


def plot_many_2d_grids(
        saliency_matrix_3d, axes_objects_2d_list, colour_map_object,
        max_absolute_colour_value, min_font_size=DEFAULT_MIN_FONT_SIZE,
        max_font_size=DEFAULT_MAX_FONT_SIZE):
    """Plots many 2-D saliency maps with "+" and "-" signs.

    The saliency map for each field will be one panel in a paneled figure.

    M = number of spatial rows
    N = number of spatial columns
    C = number of channels (predictors)

    :param saliency_matrix_3d: M-by-N-by-P numpy array of saliency values.
    :param axes_objects_2d_list: 2-D list, where axes_objects_2d_list[i][j] is
        the handle (instance of `matplotlib.axes._subplots.AxesSubplot`) for the
        [i]th row and [j]th column.
    :param colour_map_object: See doc for `plot_2d_grid`.
    :param max_absolute_colour_value: Same.
    :param min_font_size: Same.
    :param max_font_size: Same.
    """

    error_checking.assert_is_numpy_array_without_nan(saliency_matrix_3d)
    error_checking.assert_is_numpy_array(saliency_matrix_3d, num_dimensions=3)
    num_predictors = saliency_matrix_3d.shape[-1]

    num_panel_rows = len(axes_objects_2d_list)
    num_panel_columns = len(axes_objects_2d_list[0])

    for k in range(num_predictors):
        this_panel_row, this_panel_column = numpy.unravel_index(
            k, (num_panel_rows, num_panel_columns)
        )

        plot_2d_grid(
            saliency_matrix_2d=saliency_matrix_3d[..., k],
            axes_object=axes_objects_2d_list[this_panel_row][this_panel_column],
            colour_map_object=colour_map_object,
            max_absolute_colour_value=max_absolute_colour_value,
            min_font_size=min_font_size, max_font_size=max_font_size)
