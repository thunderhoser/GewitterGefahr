"""Plots saliency maps."""

import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils

DEFAULT_CONTOUR_WIDTH = 2

WIND_NAME = 'wind_m_s01'
WIND_COMPONENT_NAMES = [soundings.U_WIND_NAME, soundings.V_WIND_NAME]

WIND_BARB_LENGTH = 10.
EMPTY_WIND_BARB_RADIUS = 0.2
WIND_SALIENCY_MULTIPLIER = 52.5

FIELD_NAME_TO_LATEX_DICT = {
    soundings.SPECIFIC_HUMIDITY_NAME: r'$q_{v}$',
    soundings.VIRTUAL_POTENTIAL_TEMPERATURE_NAME: r'$\theta_{v}$',
    soundings.TEMPERATURE_NAME: r'$T$',
    soundings.RELATIVE_HUMIDITY_NAME: 'RH',
    soundings.U_WIND_NAME: r'$u$',
    soundings.V_WIND_NAME: r'$v$',
    soundings.PRESSURE_NAME: r'$p$',
    WIND_NAME: 'Wind'
}

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
SOUNDING_SALIENCY_BACKGROUND_COLOUR = numpy.array(
    [166, 206, 227], dtype=float
) / 255

DEFAULT_MIN_FONT_SIZE = 10.
DEFAULT_MAX_FONT_SIZE = 25.
DEFAULT_MIN_SOUNDING_FONT_SIZE = 24.
DEFAULT_MAX_SOUNDING_FONT_SIZE = 60.


def _saliency_to_colour_and_size(
        saliency_matrix, colour_map_object, max_absolute_colour_value,
        min_font_size, max_font_size):
    """Returns colour and font size for each saliency value.

    :param saliency_matrix: numpy array (any shape) of saliency values.
    :param colour_map_object: See doc for `plot_2d_grid`.
    :param max_absolute_colour_value: Same.
    :param min_font_size: Same.
    :param max_font_size: Same.
    :return: rgb_matrix: numpy array of colours.  If dimensions of
        `saliency_matrix` are M x N, this will be M x N x 3.  In general, number
        of dimensions will increase by 1 and length of last axis will be 3
        (corresponding to R, G, and B values).
    :return: font_size_matrix: numpy array of font sizes (same shape as
        `saliency_matrix`).
    """

    error_checking.assert_is_geq(max_absolute_colour_value, 0.)
    max_absolute_colour_value = max([max_absolute_colour_value, 0.001])

    error_checking.assert_is_greater(min_font_size, 0.)
    error_checking.assert_is_greater(max_font_size, min_font_size)

    colour_norm_object = pyplot.Normalize(
        vmin=0., vmax=max_absolute_colour_value)

    rgb_matrix = colour_map_object(colour_norm_object(
        numpy.absolute(saliency_matrix)
    ))[..., :-1]

    normalized_saliency_matrix = (
        numpy.absolute(saliency_matrix) / max_absolute_colour_value
    )
    normalized_saliency_matrix[normalized_saliency_matrix > 1.] = 1.

    font_size_matrix = (
        min_font_size + normalized_saliency_matrix *
        (max_font_size - min_font_size)
    )

    return rgb_matrix, font_size_matrix


def plot_saliency_for_sounding(
        saliency_matrix, sounding_field_names, pressure_levels_mb,
        colour_map_object, max_absolute_colour_value,
        min_font_size=DEFAULT_MIN_SOUNDING_FONT_SIZE,
        max_font_size=DEFAULT_MAX_SOUNDING_FONT_SIZE):
    """Plots saliency for one sounding.

    P = number of pressure levels
    F = number of fields

    :param saliency_matrix: P-by-F numpy array of saliency values.
    :param sounding_field_names: length-F list of field names.
    :param pressure_levels_mb: length-P list of pressure levels (millibars).
    :param colour_map_object: See doc for `plot_2d_grid`.
    :param max_absolute_colour_value: Same.
    :param min_font_size: Same.
    :param max_font_size: Same.
    """

    error_checking.assert_is_geq(max_absolute_colour_value, 0.)
    max_absolute_colour_value = max([max_absolute_colour_value, 0.001])

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
            dtype=bool
        )

        non_wind_indices = numpy.where(non_wind_flags)[0]
        saliency_matrix = saliency_matrix[:, non_wind_indices]
        sounding_field_names = [
            sounding_field_names[k] for k in non_wind_indices
        ]

        sounding_field_names.append(WIND_NAME)
        num_sounding_fields = len(sounding_field_names)

    rgb_matrix, font_size_matrix = _saliency_to_colour_and_size(
        saliency_matrix=saliency_matrix, colour_map_object=colour_map_object,
        max_absolute_colour_value=max_absolute_colour_value,
        min_font_size=min_font_size, max_font_size=max_font_size)

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    axes_object.set_facecolor(
        plotting_utils.colour_from_numpy_to_tuple(
            SOUNDING_SALIENCY_BACKGROUND_COLOUR)
    )

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

                this_colour_tuple = plotting_utils.colour_from_numpy_to_tuple(
                    rgb_matrix_for_wind[j, ...]
                )

                axes_object.barbs(
                    k, pressure_levels_mb[j], this_vector[0], this_vector[1],
                    length=WIND_BARB_LENGTH, fill_empty=True, rounding=False,
                    sizes={'emptybarb': EMPTY_WIND_BARB_RADIUS},
                    color=this_colour_tuple)

            continue

        for j in range(num_pressure_levels):
            this_colour_tuple = plotting_utils.colour_from_numpy_to_tuple(
                rgb_matrix[j, k, ...]
            )

            if saliency_matrix[j, k] >= 0:
                axes_object.text(
                    k, pressure_levels_mb[j], '+',
                    fontsize=font_size_matrix[j, k],
                    color=this_colour_tuple, horizontalalignment='center',
                    verticalalignment='center')
            else:
                axes_object.text(
                    k, pressure_levels_mb[j], '_',
                    fontsize=font_size_matrix[j, k],
                    color=this_colour_tuple, horizontalalignment='center',
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
        FIELD_NAME_TO_LATEX_DICT[f] for f in sounding_field_names
    ]
    pyplot.xticks(x_tick_locations, x_tick_labels)

    colour_bar_object = plotting_utils.plot_linear_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=saliency_matrix,
        colour_map_object=colour_map_object, min_value=0.,
        max_value=max_absolute_colour_value, orientation_string='vertical',
        extend_min=False, extend_max=True)

    colour_bar_object.set_label('Absolute saliency')


def plot_2d_grid_with_contours(
        saliency_matrix_2d, axes_object, colour_map_object,
        max_absolute_contour_level, contour_interval,
        line_width=DEFAULT_CONTOUR_WIDTH):
    """Plots 2-D saliency map with line contours.

    M = number of rows in spatial grid
    N = number of columns in spatial grid

    :param saliency_matrix_2d: M-by-N numpy array of saliency values.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
        Will plot on these axes.
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.pyplot.cm`).
    :param max_absolute_contour_level: Max absolute value to plot.  Minimum
        value will be `-1 * max_absolute_contour_level`.
    :param contour_interval: Interval (in saliency units) between successive
        contours.
    :param line_width: Width of contour lines.
    """

    error_checking.assert_is_geq(max_absolute_contour_level, 0.)
    max_absolute_contour_level = max([max_absolute_contour_level, 0.001])

    error_checking.assert_is_geq(contour_interval, 0.)
    contour_interval = max([contour_interval, 0.0001])

    error_checking.assert_is_numpy_array_without_nan(saliency_matrix_2d)
    error_checking.assert_is_numpy_array(saliency_matrix_2d, num_dimensions=2)
    error_checking.assert_is_less_than(
        contour_interval, max_absolute_contour_level)

    num_grid_rows = saliency_matrix_2d.shape[0]
    num_grid_columns = saliency_matrix_2d.shape[1]

    x_coords_unique = numpy.linspace(
        0, num_grid_columns, num=num_grid_columns + 1, dtype=float)
    x_coords_unique = x_coords_unique[:-1]
    x_coords_unique = x_coords_unique + numpy.diff(x_coords_unique[:2]) / 2

    y_coords_unique = numpy.linspace(
        0, num_grid_rows, num=num_grid_rows + 1, dtype=float)
    y_coords_unique = y_coords_unique[:-1]
    y_coords_unique = y_coords_unique + numpy.diff(y_coords_unique[:2]) / 2

    x_coord_matrix, y_coord_matrix = numpy.meshgrid(x_coords_unique,
                                                    y_coords_unique)

    half_num_contours = int(numpy.round(
        1 + max_absolute_contour_level / contour_interval
    ))

    # Plot positive values.
    these_contour_levels = numpy.linspace(
        0., max_absolute_contour_level, num=half_num_contours)

    axes_object.contour(
        x_coord_matrix, y_coord_matrix, saliency_matrix_2d,
        these_contour_levels, cmap=colour_map_object,
        vmin=numpy.min(these_contour_levels),
        vmax=numpy.max(these_contour_levels), linewidths=line_width,
        linestyles='solid', zorder=1e6)

    # Plot negative values.
    these_contour_levels = these_contour_levels[1:]

    axes_object.contour(
        x_coord_matrix, y_coord_matrix, -saliency_matrix_2d,
        these_contour_levels, cmap=colour_map_object,
        vmin=numpy.min(these_contour_levels),
        vmax=numpy.max(these_contour_levels), linewidths=line_width,
        linestyles='dashed', zorder=1e6)


def plot_many_2d_grids_with_contours(
        saliency_matrix_3d, axes_object_matrix, colour_map_object,
        max_absolute_contour_level, contour_interval,
        line_width=DEFAULT_CONTOUR_WIDTH, row_major=True):
    """Plots 2-D saliency map with line contours for each predictor.

    M = number of rows in spatial grid
    N = number of columns in spatial grid
    P = number of predictors

    :param saliency_matrix_3d: M-by-N-by-P numpy array of saliency values.
    :param axes_object_matrix: See doc for
        `plotting_utils.create_paneled_figure`.
    :param colour_map_object: See doc for `plot_2d_grid_with_contours`.
    :param max_absolute_contour_level: Same.
    :param contour_interval: Same.
    :param line_width: Same.
    :param row_major: Boolean flag.  If True, panels will be filled along rows
        first, then down columns.  If False, down columns first, then along
        rows.
    """

    error_checking.assert_is_numpy_array_without_nan(saliency_matrix_3d)
    error_checking.assert_is_numpy_array(saliency_matrix_3d, num_dimensions=3)
    error_checking.assert_is_boolean(row_major)

    if row_major:
        order_string = 'C'
    else:
        order_string = 'F'

    num_predictors = saliency_matrix_3d.shape[-1]
    num_panel_rows = axes_object_matrix.shape[0]
    num_panel_columns = axes_object_matrix.shape[1]

    for k in range(num_predictors):
        this_panel_row, this_panel_column = numpy.unravel_index(
            k, (num_panel_rows, num_panel_columns), order=order_string
        )

        plot_2d_grid_with_contours(
            saliency_matrix_2d=saliency_matrix_3d[..., k],
            axes_object=axes_object_matrix[this_panel_row, this_panel_column],
            colour_map_object=colour_map_object,
            max_absolute_contour_level=max_absolute_contour_level,
            contour_interval=contour_interval, line_width=line_width)


def plot_2d_grid_with_pm_signs(
        saliency_matrix_2d, axes_object, colour_map_object,
        max_absolute_colour_value, min_font_size=DEFAULT_MIN_FONT_SIZE,
        max_font_size=DEFAULT_MAX_FONT_SIZE):
    """Plots 2-D saliency map with plus and minus signs ("+" and "-").

    M = number of rows in spatial grid
    N = number of columns in spatial grid

    :param saliency_matrix_2d: See doc for `plot_2d_grid_with_contours`.
    :param axes_object: Same.
    :param colour_map_object: Same.
    :param max_absolute_colour_value: Same.
    :param min_font_size: Minimum font size (used for zero saliency).
    :param max_font_size: Max font size (used for max absolute value).
    """

    error_checking.assert_is_geq(max_absolute_colour_value, 0.)
    max_absolute_colour_value = max([max_absolute_colour_value, 0.001])

    error_checking.assert_is_numpy_array_without_nan(saliency_matrix_2d)
    error_checking.assert_is_numpy_array(saliency_matrix_2d, num_dimensions=2)

    rgb_matrix, font_size_matrix = _saliency_to_colour_and_size(
        saliency_matrix=saliency_matrix_2d, colour_map_object=colour_map_object,
        max_absolute_colour_value=max_absolute_colour_value,
        min_font_size=min_font_size, max_font_size=max_font_size)

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
            this_colour_tuple = plotting_utils.colour_from_numpy_to_tuple(
                rgb_matrix[i, j, ...]
            )

            if saliency_matrix_2d[i, j] >= 0:
                axes_object.text(
                    x_coords[i], y_coords[j], '+',
                    fontsize=font_size_matrix[i, j],
                    color=this_colour_tuple, horizontalalignment='center',
                    verticalalignment='center', transform=axes_object.transAxes)
            else:
                axes_object.text(
                    x_coords[i], y_coords[j], '_',
                    fontsize=font_size_matrix[i, j],
                    color=this_colour_tuple, horizontalalignment='center',
                    verticalalignment='bottom', transform=axes_object.transAxes)


def plot_many_2d_grids_with_pm_signs(
        saliency_matrix_3d, axes_object_matrix, colour_map_object,
        max_absolute_colour_value, min_font_size=DEFAULT_MIN_FONT_SIZE,
        max_font_size=DEFAULT_MAX_FONT_SIZE, row_major=True):
    """Plots many 2-D saliency map with plus and minus signs ("+" and "-").

    :param saliency_matrix_3d: See doc for `plot_many_2d_grids_with_contours`.
    :param axes_object_matrix: Same.
    :param colour_map_object: See doc for `plot_2d_grid_with_pm_signs`.
    :param max_absolute_colour_value: Same.
    :param min_font_size: Same.
    :param max_font_size: Same.
    :param row_major: See doc for `plot_many_2d_grids_with_contours`.
    """

    error_checking.assert_is_numpy_array_without_nan(saliency_matrix_3d)
    error_checking.assert_is_numpy_array(saliency_matrix_3d, num_dimensions=3)
    error_checking.assert_is_boolean(row_major)

    if row_major:
        order_string = 'C'
    else:
        order_string = 'F'

    num_predictors = saliency_matrix_3d.shape[-1]
    num_panel_rows = axes_object_matrix.shape[0]
    num_panel_columns = axes_object_matrix.shape[1]

    for k in range(num_predictors):
        this_panel_row, this_panel_column = numpy.unravel_index(
            k, (num_panel_rows, num_panel_columns), order=order_string
        )

        plot_2d_grid_with_pm_signs(
            saliency_matrix_2d=saliency_matrix_3d[..., k],
            axes_object=axes_object_matrix[this_panel_row, this_panel_column],
            colour_map_object=colour_map_object,
            max_absolute_colour_value=max_absolute_colour_value,
            min_font_size=min_font_size, max_font_size=max_font_size)
