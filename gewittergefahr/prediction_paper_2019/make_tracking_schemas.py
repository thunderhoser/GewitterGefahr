"""Makes storm-tracking schematics for 2019 prediction paper."""

import numpy
import pandas
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import imagemagick_utils

TRACK_COLOUR_COLUMN = 'track_colour'
MAIN_TRACK_COLOUR = numpy.full(3, 0.)
ALMOST_WHITE_COLOUR = numpy.full(3, 0.99)

FIRST_TRACK_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
SECOND_TRACK_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
THIRD_TRACK_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255

TRACK_WIDTH = 2
DEFAULT_MARKER_TYPE = 'o'
DEFAULT_MARKER_SIZE = 16
DEFAULT_MARKER_EDGE_WIDTH = 2
END_MARKER_TYPE = 'x'
END_MARKER_SIZE = 24
END_MARKER_EDGE_WIDTH = 4

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

# OUTPUT_DIR_NAME = (
#     '/condo/swatwork/ralager/prediction_paper_2019/tracking_schemas'
# )
OUTPUT_DIR_NAME = (
    '/localdata/ryan.lagerquist/eager/prediction_paper_2019/tracking_schemas'
)


def _get_data_for_linkage_schema(extrapolate):
    """Creates synthetic data for linkage schematic.

    T = number of time steps in given track

    :param extrapolate: Boolean flag.  If True, will extrapolate storm objects
        from early time to late time.
    :return: early_track_table: pandas DataFrame with the following columns.
        Each row is one track.
    early_track_table.full_id_string: Full storm ID.
    early_track_table.centroid_x_coords_metres: length-T numpy array of
        x-coords.
    early_track_table.centroid_y_coords_metres: length-T numpy array of
        y-coords.
    early_track_table.track_colour: Colour (length-3 numpy array) for plotting.

    :return: late_storm_object_table: pandas DataFrame with the following
        columns (and only one row).
    late_storm_object_table.centroid_x_metres: x-coordinate.
    late_storm_object_table.centroid_y_metres: y-coordinate.
    """

    early_track_table = pandas.DataFrame.from_dict({
        tracking_utils.FULL_ID_COLUMN: ['A', 'B', 'C']
    })

    nested_array = early_track_table[[
        tracking_utils.FULL_ID_COLUMN, tracking_utils.FULL_ID_COLUMN
    ]].values.tolist()

    early_track_table = early_track_table.assign(**{
        tracking_utils.TRACK_X_COORDS_COLUMN: nested_array,
        tracking_utils.TRACK_Y_COORDS_COLUMN: nested_array,
        TRACK_COLOUR_COLUMN: nested_array
    })

    early_track_table[tracking_utils.TRACK_X_COORDS_COLUMN].values[0] = (
        numpy.array([1, 8, 13, 18, 22, 25], dtype=float)
    )
    early_track_table[tracking_utils.TRACK_X_COORDS_COLUMN].values[1] = (
        numpy.linspace(0, 25, num=4, dtype=float)
    )
    early_track_table[tracking_utils.TRACK_X_COORDS_COLUMN].values[2] = (
        numpy.linspace(0, 25, num=4, dtype=float)
    )

    early_track_table[tracking_utils.TRACK_Y_COORDS_COLUMN].values[0] = (
        numpy.array([25, 30, 26, 24, 24.5, 23])
    )
    early_track_table[tracking_utils.TRACK_Y_COORDS_COLUMN].values[1] = (
        numpy.full(4, 15.)
    )
    early_track_table[tracking_utils.TRACK_Y_COORDS_COLUMN].values[2] = (
        numpy.array([1, 4, 5.5, 7.5])
    )

    early_track_table[TRACK_COLOUR_COLUMN].values[0] = FIRST_TRACK_COLOUR
    early_track_table[TRACK_COLOUR_COLUMN].values[1] = SECOND_TRACK_COLOUR
    early_track_table[TRACK_COLOUR_COLUMN].values[2] = THIRD_TRACK_COLOUR

    if extrapolate:
        late_storm_object_table = pandas.DataFrame.from_dict({
            tracking_utils.CENTROID_X_COLUMN: numpy.full(1, 33.),
            tracking_utils.CENTROID_Y_COLUMN: numpy.full(1, 11.)
        })
    else:
        late_storm_object_table = pandas.DataFrame.from_dict({
            tracking_utils.CENTROID_X_COLUMN: numpy.full(1, 23.),
            tracking_utils.CENTROID_Y_COLUMN: numpy.full(1, 4.5)
        })

    return early_track_table, late_storm_object_table


def _get_data_for_3way_schema():
    """Creates synthetic data for schematic with 3-way split.

    :return: early_track_table: See doc for `_get_data_for_linkage_schema`.
    :return: late_track_table: Same but for later period.
    """

    early_track_table = pandas.DataFrame.from_dict({
        tracking_utils.FULL_ID_COLUMN: ['A']
    })

    nested_array = early_track_table[[
        tracking_utils.FULL_ID_COLUMN, tracking_utils.FULL_ID_COLUMN
    ]].values.tolist()

    early_track_table = early_track_table.assign(**{
        tracking_utils.TRACK_X_COORDS_COLUMN: nested_array,
        tracking_utils.TRACK_Y_COORDS_COLUMN: nested_array,
        TRACK_COLOUR_COLUMN: nested_array
    })

    early_track_table[tracking_utils.TRACK_X_COORDS_COLUMN].values[0] = (
        numpy.array([2, 10, 15, 17], dtype=float)
    )
    early_track_table[tracking_utils.TRACK_Y_COORDS_COLUMN].values[0] = (
        numpy.full(4, 20.)
    )
    early_track_table[TRACK_COLOUR_COLUMN].values[0] = MAIN_TRACK_COLOUR

    late_track_table = pandas.DataFrame.from_dict({
        tracking_utils.FULL_ID_COLUMN: ['B', 'C', 'D']
    })

    nested_array = late_track_table[[
        tracking_utils.FULL_ID_COLUMN, tracking_utils.FULL_ID_COLUMN
    ]].values.tolist()

    late_track_table = late_track_table.assign(**{
        tracking_utils.TRACK_X_COORDS_COLUMN: nested_array,
        tracking_utils.TRACK_Y_COORDS_COLUMN: nested_array,
        TRACK_COLOUR_COLUMN: nested_array
    })

    late_track_table[tracking_utils.TRACK_X_COORDS_COLUMN].values[0] = (
        numpy.array([22, 27, 32, 37], dtype=float)
    )
    late_track_table[tracking_utils.TRACK_X_COORDS_COLUMN].values[1] = (
        numpy.array([22, 27, 32, 37], dtype=float)
    )
    late_track_table[tracking_utils.TRACK_X_COORDS_COLUMN].values[2] = (
        numpy.array([22, 27, 32, 37], dtype=float)
    )

    late_track_table[tracking_utils.TRACK_Y_COORDS_COLUMN].values[0] = (
        numpy.array([26, 32, 38, 44], dtype=float)
    )
    late_track_table[tracking_utils.TRACK_Y_COORDS_COLUMN].values[1] = (
        numpy.array([20, 24, 28, 32], dtype=float)
    )
    late_track_table[tracking_utils.TRACK_Y_COORDS_COLUMN].values[2] = (
        numpy.array([16, 12, 8, 4], dtype=float)
    )

    late_track_table[TRACK_COLOUR_COLUMN].values[0] = FIRST_TRACK_COLOUR
    late_track_table[TRACK_COLOUR_COLUMN].values[1] = SECOND_TRACK_COLOUR
    late_track_table[TRACK_COLOUR_COLUMN].values[2] = THIRD_TRACK_COLOUR

    return early_track_table, late_track_table


def _get_data_for_splitmerge_schema():
    """Creates synthetic data for schematic with split and merger.

    :return: early_track_table: See doc for `_get_data_for_linkage_schema`.
    :return: late_storm_object_table: Same.
    """

    early_track_table = pandas.DataFrame.from_dict({
        tracking_utils.FULL_ID_COLUMN: ['A', 'B']
    })

    nested_array = early_track_table[[
        tracking_utils.FULL_ID_COLUMN, tracking_utils.FULL_ID_COLUMN
    ]].values.tolist()

    early_track_table = early_track_table.assign(**{
        tracking_utils.TRACK_X_COORDS_COLUMN: nested_array,
        tracking_utils.TRACK_Y_COORDS_COLUMN: nested_array,
        TRACK_COLOUR_COLUMN: nested_array
    })

    early_track_table[tracking_utils.TRACK_X_COORDS_COLUMN].values[0] = (
        numpy.array([5, 12.5, 20])
    )
    early_track_table[tracking_utils.TRACK_X_COORDS_COLUMN].values[1] = (
        numpy.array([5, 12.5, 20])
    )

    early_track_table[tracking_utils.TRACK_Y_COORDS_COLUMN].values[0] = (
        numpy.full(3, 5.)
    )
    early_track_table[tracking_utils.TRACK_Y_COORDS_COLUMN].values[1] = (
        numpy.full(3, 20.)
    )

    early_track_table[TRACK_COLOUR_COLUMN].values[0] = FIRST_TRACK_COLOUR
    early_track_table[TRACK_COLOUR_COLUMN].values[1] = SECOND_TRACK_COLOUR

    late_storm_object_table = pandas.DataFrame.from_dict({
        tracking_utils.CENTROID_X_COLUMN: numpy.full(2, 25.),
        tracking_utils.CENTROID_Y_COLUMN: numpy.array([12.5, 27.5]),
        tracking_utils.FULL_ID_COLUMN: ['C', 'D']
    })

    return early_track_table, late_storm_object_table


def _make_linkage_schema(extrapolate):
    """Makes linkage schematic.

    :param extrapolate: Boolean flag.  If True, will extrapolate storm objects
        from early time to late time.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    early_track_table, late_storm_object_table = (
        _get_data_for_linkage_schema(extrapolate)
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    num_storm_tracks = len(early_track_table.index)

    for i in range(num_storm_tracks):
        these_x_coords = early_track_table[
            tracking_utils.TRACK_X_COORDS_COLUMN].values[i]
        these_y_coords = early_track_table[
            tracking_utils.TRACK_Y_COORDS_COLUMN].values[i]

        this_colour_tuple = plotting_utils.colour_from_numpy_to_tuple(
            early_track_table[TRACK_COLOUR_COLUMN].values[i]
        )

        axes_object.plot(
            these_x_coords, these_y_coords, color=this_colour_tuple,
            linestyle='-', linewidth=TRACK_WIDTH)

        axes_object.plot(
            these_x_coords[:-1], these_y_coords[:-1], linestyle='None',
            marker=DEFAULT_MARKER_TYPE, markersize=DEFAULT_MARKER_SIZE,
            markerfacecolor=this_colour_tuple,
            markeredgecolor=this_colour_tuple,
            markeredgewidth=DEFAULT_MARKER_EDGE_WIDTH)

        these_extrap_x_coords = these_x_coords[-1] + (
            these_x_coords[-1] - these_x_coords[-2:]
        )
        these_extrap_y_coords = these_y_coords[-1] + (
            these_y_coords[-1] - these_y_coords[-2:]
        )

        this_full_id_string = early_track_table[
            tracking_utils.FULL_ID_COLUMN].values[i]
        this_label_string = r'$t_1$'
        this_label_string = '{0:s} at {1:s}'.format(
            this_full_id_string, this_label_string)

        axes_object.text(
            these_extrap_x_coords[1], these_extrap_y_coords[1] - 0.5,
            this_label_string, color=this_colour_tuple,
            horizontalalignment='center', verticalalignment='top')

        if extrapolate:
            axes_object.plot(
                these_extrap_x_coords, these_extrap_y_coords,
                color=this_colour_tuple, linestyle='--', linewidth=TRACK_WIDTH,
                marker=END_MARKER_TYPE, markersize=END_MARKER_SIZE,
                markerfacecolor=this_colour_tuple,
                markeredgecolor=this_colour_tuple,
                markeredgewidth=END_MARKER_EDGE_WIDTH)

            this_label_string = r'$t_2$'
            this_label_string = '{0:s} at {1:s}'.format(
                this_full_id_string, this_label_string)

            axes_object.text(
                these_extrap_x_coords[0], these_extrap_y_coords[0] - 0.5,
                this_label_string, color=this_colour_tuple,
                horizontalalignment='center', verticalalignment='top')
        else:
            axes_object.plot(
                these_extrap_x_coords[1], these_extrap_y_coords[1],
                linestyle='None', marker=END_MARKER_TYPE,
                markersize=END_MARKER_SIZE, markerfacecolor=this_colour_tuple,
                markeredgecolor=this_colour_tuple,
                markeredgewidth=END_MARKER_EDGE_WIDTH)

    this_late_x_coord = late_storm_object_table[
        tracking_utils.CENTROID_X_COLUMN].values[0]
    this_late_y_coord = late_storm_object_table[
        tracking_utils.CENTROID_Y_COLUMN].values[0]

    this_early_x_coord = early_track_table[
        tracking_utils.TRACK_X_COORDS_COLUMN].values[2][-1]
    this_early_y_coord = early_track_table[
        tracking_utils.TRACK_Y_COORDS_COLUMN].values[2][-1]

    axes_object.plot(
        [this_early_x_coord, this_late_x_coord],
        [this_early_y_coord, this_late_y_coord],
        color=MAIN_TRACK_COLOUR, linestyle='-', linewidth=TRACK_WIDTH)

    if extrapolate:
        this_early_x_coord = early_track_table[
            tracking_utils.TRACK_X_COORDS_COLUMN].values[1][-1]
        this_early_y_coord = early_track_table[
            tracking_utils.TRACK_Y_COORDS_COLUMN].values[1][-1]

        axes_object.plot(
            [this_early_x_coord, this_late_x_coord],
            [this_early_y_coord, this_late_y_coord],
            color=MAIN_TRACK_COLOUR, linestyle='-', linewidth=TRACK_WIDTH)

    axes_object.plot(
        this_late_x_coord, this_late_y_coord, linestyle='None',
        marker=DEFAULT_MARKER_TYPE, markersize=DEFAULT_MARKER_SIZE,
        markerfacecolor=MAIN_TRACK_COLOUR,
        markeredgecolor=MAIN_TRACK_COLOUR,
        markeredgewidth=DEFAULT_MARKER_EDGE_WIDTH)

    axes_object.text(
        this_late_x_coord + 0.5, this_late_y_coord, r'D at $t_2$',
        color=MAIN_TRACK_COLOUR, horizontalalignment='left',
        verticalalignment='center')

    axes_object.grid(
        b=True, which='major', axis='both', linestyle='-', linewidth=0.5)

    axes_object.set_xlabel(r'$x$-distance (km)')
    axes_object.set_ylabel(r'$y$-distance (km)')
    return figure_object, axes_object


def _make_3way_split_schema():
    """Makes schematic for getting rid of 3-way split."""

    early_track_table, late_track_table = _get_data_for_3way_schema()

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    num_early_tracks = len(early_track_table.index)

    for i in range(num_early_tracks):
        these_x_coords = early_track_table[
            tracking_utils.TRACK_X_COORDS_COLUMN].values[i]
        these_y_coords = early_track_table[
            tracking_utils.TRACK_Y_COORDS_COLUMN].values[i]

        this_colour_tuple = plotting_utils.colour_from_numpy_to_tuple(
            early_track_table[TRACK_COLOUR_COLUMN].values[i]
        )

        axes_object.plot(
            these_x_coords, these_y_coords, color=this_colour_tuple,
            linestyle='-', linewidth=TRACK_WIDTH)

        axes_object.plot(
            these_x_coords[:-1], these_y_coords[:-1], linestyle='None',
            marker=DEFAULT_MARKER_TYPE, markersize=DEFAULT_MARKER_SIZE,
            markerfacecolor=this_colour_tuple,
            markeredgecolor=this_colour_tuple,
            markeredgewidth=DEFAULT_MARKER_EDGE_WIDTH)

        axes_object.plot(
            these_x_coords[-1], these_y_coords[-1], linestyle='None',
            marker=END_MARKER_TYPE, markersize=END_MARKER_SIZE,
            markerfacecolor=this_colour_tuple,
            markeredgecolor=this_colour_tuple,
            markeredgewidth=END_MARKER_EDGE_WIDTH)

        this_full_id_string = early_track_table[
            tracking_utils.FULL_ID_COLUMN].values[i]
        this_label_string = r'$t_1$'
        this_label_string = '{0:s} at {1:s}'.format(
            this_full_id_string, this_label_string)

        axes_object.text(
            these_x_coords[-1], these_y_coords[-1] - 0.5, this_label_string,
            color=this_colour_tuple, horizontalalignment='center',
            verticalalignment='top')

    num_late_tracks = len(late_track_table.index)

    last_early_x_coord = early_track_table[
        tracking_utils.TRACK_X_COORDS_COLUMN].values[0][-1]
    last_early_y_coord = early_track_table[
        tracking_utils.TRACK_Y_COORDS_COLUMN].values[0][-1]

    for i in range(num_late_tracks):
        these_x_coords = late_track_table[
            tracking_utils.TRACK_X_COORDS_COLUMN].values[i]
        these_y_coords = late_track_table[
            tracking_utils.TRACK_Y_COORDS_COLUMN].values[i]

        axes_object.plot(
            [last_early_x_coord, these_x_coords[0]],
            [last_early_y_coord, these_y_coords[0]],
            color=MAIN_TRACK_COLOUR, linestyle='--' if i == 0 else '-',
            linewidth=TRACK_WIDTH
        )

        this_colour_tuple = plotting_utils.colour_from_numpy_to_tuple(
            late_track_table[TRACK_COLOUR_COLUMN].values[i]
        )

        axes_object.plot(
            these_x_coords, these_y_coords, linestyle='None',
            marker=DEFAULT_MARKER_TYPE, markersize=DEFAULT_MARKER_SIZE,
            markerfacecolor=this_colour_tuple,
            markeredgecolor=this_colour_tuple,
            markeredgewidth=DEFAULT_MARKER_EDGE_WIDTH)

        this_full_id_string = late_track_table[
            tracking_utils.FULL_ID_COLUMN].values[i]
        this_label_string = r'$t_2$'
        this_label_string = '{0:s} at {1:s}'.format(
            this_full_id_string, this_label_string)

        axes_object.text(
            these_x_coords[0], these_y_coords[0] - 0.5, this_label_string,
            color=this_colour_tuple, horizontalalignment='center',
            verticalalignment='top')

    axes_object.set_xlabel(r'$x$-distance (km)', color=ALMOST_WHITE_COLOUR)
    axes_object.set_ylabel(r'$y$-distance (km)', color=ALMOST_WHITE_COLOUR)
    axes_object.set_xticklabels(
        numpy.round(axes_object.get_xticks()).astype(int),
        color=ALMOST_WHITE_COLOUR
    )
    axes_object.set_yticklabels(
        numpy.round(axes_object.get_yticks()).astype(int),
        color=ALMOST_WHITE_COLOUR
    )

    return figure_object, axes_object


def _make_splitmerge_schema():
    """Makes schematic for getting rid of hybrid split-merger."""

    early_track_table, late_storm_object_table = (
        _get_data_for_splitmerge_schema()
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    num_early_tracks = len(early_track_table.index)

    for i in range(num_early_tracks):
        these_x_coords = early_track_table[
            tracking_utils.TRACK_X_COORDS_COLUMN].values[i]
        these_y_coords = early_track_table[
            tracking_utils.TRACK_Y_COORDS_COLUMN].values[i]

        this_colour_tuple = plotting_utils.colour_from_numpy_to_tuple(
            early_track_table[TRACK_COLOUR_COLUMN].values[i]
        )

        axes_object.plot(
            these_x_coords, these_y_coords, color=this_colour_tuple,
            linestyle='-', linewidth=TRACK_WIDTH)

        axes_object.plot(
            these_x_coords[:-1], these_y_coords[:-1], linestyle='None',
            marker=DEFAULT_MARKER_TYPE, markersize=DEFAULT_MARKER_SIZE,
            markerfacecolor=this_colour_tuple,
            markeredgecolor=this_colour_tuple,
            markeredgewidth=DEFAULT_MARKER_EDGE_WIDTH)

        axes_object.plot(
            these_x_coords[-1], these_y_coords[-1], linestyle='None',
            marker=END_MARKER_TYPE, markersize=END_MARKER_SIZE,
            markerfacecolor=this_colour_tuple,
            markeredgecolor=this_colour_tuple,
            markeredgewidth=END_MARKER_EDGE_WIDTH)

        this_full_id_string = early_track_table[
            tracking_utils.FULL_ID_COLUMN].values[i]
        this_label_string = r'$t_1$'
        this_label_string = '{0:s} at {1:s}'.format(
            this_full_id_string, this_label_string)

        axes_object.text(
            these_x_coords[-1], these_y_coords[-1] - 0.5, this_label_string,
            color=this_colour_tuple, horizontalalignment='center',
            verticalalignment='top')

    num_late_objects = len(late_storm_object_table.index)

    for j in range(num_late_objects):
        this_x_coord = late_storm_object_table[
            tracking_utils.CENTROID_X_COLUMN].values[j]
        this_y_coord = late_storm_object_table[
            tracking_utils.CENTROID_Y_COLUMN].values[j]

        axes_object.plot(
            this_x_coord, this_y_coord, linestyle='None',
            marker=DEFAULT_MARKER_TYPE, markersize=DEFAULT_MARKER_SIZE,
            markerfacecolor=MAIN_TRACK_COLOUR,
            markeredgecolor=MAIN_TRACK_COLOUR,
            markeredgewidth=DEFAULT_MARKER_EDGE_WIDTH)

        this_full_id_string = late_storm_object_table[
            tracking_utils.FULL_ID_COLUMN].values[j]
        this_label_string = r'$t_2$'
        this_label_string = '{0:s} at {1:s}'.format(
            this_full_id_string, this_label_string)

        axes_object.text(
            this_x_coord, this_y_coord - 0.5, this_label_string,
            color=MAIN_TRACK_COLOUR,
            horizontalalignment='center', verticalalignment='top')

        if j == 0:
            this_last_x_coord = early_track_table[
                tracking_utils.TRACK_X_COORDS_COLUMN].values[0][-1]
            this_last_y_coord = early_track_table[
                tracking_utils.TRACK_Y_COORDS_COLUMN].values[0][-1]

            axes_object.plot(
                [this_last_x_coord, this_x_coord],
                [this_last_y_coord, this_y_coord],
                color=MAIN_TRACK_COLOUR, linestyle='-', linewidth=TRACK_WIDTH
            )

            this_last_x_coord = early_track_table[
                tracking_utils.TRACK_X_COORDS_COLUMN].values[1][-1]
            this_last_y_coord = early_track_table[
                tracking_utils.TRACK_Y_COORDS_COLUMN].values[1][-1]

            axes_object.plot(
                [this_last_x_coord, this_x_coord],
                [this_last_y_coord, this_y_coord],
                color=MAIN_TRACK_COLOUR, linestyle='--', linewidth=TRACK_WIDTH
            )
        else:
            this_last_x_coord = early_track_table[
                tracking_utils.TRACK_X_COORDS_COLUMN].values[1][-1]
            this_last_y_coord = early_track_table[
                tracking_utils.TRACK_Y_COORDS_COLUMN].values[1][-1]

            axes_object.plot(
                [this_last_x_coord, this_x_coord],
                [this_last_y_coord, this_y_coord],
                color=MAIN_TRACK_COLOUR, linestyle='-', linewidth=TRACK_WIDTH
            )

    axes_object.set_xlabel(r'$x$-distance (km)', color=ALMOST_WHITE_COLOUR)
    axes_object.set_ylabel(r'$y$-distance (km)', color=ALMOST_WHITE_COLOUR)
    axes_object.set_xticklabels(
        numpy.round(axes_object.get_xticks()).astype(int),
        color=ALMOST_WHITE_COLOUR
    )
    axes_object.set_yticklabels(
        numpy.round(axes_object.get_yticks()).astype(int),
        color=ALMOST_WHITE_COLOUR
    )

    return figure_object, axes_object


def _run():
    """Makes storm-tracking schematics for 2019 prediction paper.

    This is effectively the main method.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=OUTPUT_DIR_NAME)

    figure_object, axes_object = _make_linkage_schema(True)
    plotting_utils.label_axes(axes_object=axes_object, label_string='(a)')
    axes_object.set_title('Linkage with extrapolation')

    panel_file_names = ['{0:s}/linkage_with_extrap.jpg'.format(OUTPUT_DIR_NAME)]

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object, axes_object = _make_linkage_schema(False)
    plotting_utils.label_axes(axes_object=axes_object, label_string='(b)')
    axes_object.set_title('Linkage without extrapolation')

    panel_file_names.append(
        '{0:s}/linkage_sans_extrap.jpg'.format(OUTPUT_DIR_NAME)
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object, axes_object = _make_3way_split_schema()
    plotting_utils.label_axes(axes_object=axes_object, label_string='(c)')
    axes_object.set_title('Pruning with 3-way split')

    panel_file_names.append(
        '{0:s}/pruning_3way_split.jpg'.format(OUTPUT_DIR_NAME)
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object, axes_object = _make_splitmerge_schema()
    plotting_utils.label_axes(axes_object=axes_object, label_string='(d)')
    axes_object.set_title('Pruning with hybrid split and merger')

    panel_file_names.append(
        '{0:s}/pruning_splitmerge.jpg'.format(OUTPUT_DIR_NAME)
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    concat_file_name = '{0:s}/tracking_schemas.jpg'.format(OUTPUT_DIR_NAME)
    print('Concatenating panels to: "{0:s}"...'.format(concat_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names, output_file_name=concat_file_name,
        num_panel_rows=2, num_panel_columns=2)

    imagemagick_utils.resize_image(
        input_file_name=concat_file_name, output_file_name=concat_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX)


if __name__ == '__main__':
    _run()
