"""Makes event-attribution schematics for 2019 tornado-prediction paper."""

import numpy
import pandas
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from descartes import PolygonPatch
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import storm_plotting
from gewittergefahr.plotting import imagemagick_utils

TORNADIC_FLAG_COLUMN = 'is_tornadic'
SPECIAL_FLAG_COLUMN = 'is_main_tornadic_link'
POLYGON_COLUMN = 'polygon_object_xy_metres'

TRIANGLE_X_COORDS_RELATIVE = numpy.array([-1, 0, 1, -1], dtype=float)
TRIANGLE_Y_COORDS_RELATIVE = numpy.array([-1, 1, -1, -1], dtype=float)
SQUARE_X_COORDS_RELATIVE = numpy.array([-1, -1, 1, 1, -1], dtype=float)
SQUARE_Y_COORDS_RELATIVE = numpy.array([-1, 1, 1, -1, -1], dtype=float)
DIAMOND_X_COORDS_RELATIVE = numpy.array([0, -1, 0, 1, 0], dtype=float)
DIAMOND_Y_COORDS_RELATIVE = numpy.array([-1, 0, 1, 0, -1], dtype=float)

INTERP_COLOUR = numpy.full(3, 0.)
MIDPOINT_COLOUR = numpy.full(3, 152. / 255)
TRACK_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
STORM_OBJECT_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
TORNADIC_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
NON_TORNADIC_COLOUR = TRACK_COLOUR

FONT_SIZE = 30
TEXT_OFFSET = 0.25

TRACK_WIDTH = 4
POLYGON_OPACITY = 0.5
DEFAULT_MARKER_TYPE = 'o'
DEFAULT_MARKER_SIZE = 24
DEFAULT_MARKER_EDGE_WIDTH = 4
TORNADIC_MARKER_TYPE = '*'
TORNADIC_MARKER_SIZE = 48
TORNADIC_MARKER_EDGE_WIDTH = 0

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

OUTPUT_DIR_NAME = (
    '/localdata/ryan.lagerquist/eager/prediction_paper_2019/attribution_schemas'
)


def _get_interp_data_for_merger():
    """Creates synthetic data for interpolation with storm merger.

    :return: early_storm_object_table: pandas DataFrame with the following
        columns.
    early_storm_object_table.centroid_x_metres: x-coordinate.
    early_storm_object_table.centroid_y_metres: y-coordinate.
    early_storm_object_table.polygon_object_xy_metres: Storm outline (instance
        of `shapely.geometry.Polygon`).

    :return: late_storm_object_table: Same but for late period.
    """

    first_polygon_object = polygons.vertex_arrays_to_polygon_object(
        exterior_x_coords=5 + TRIANGLE_X_COORDS_RELATIVE,
        exterior_y_coords=10 + TRIANGLE_Y_COORDS_RELATIVE)

    second_polygon_object = polygons.vertex_arrays_to_polygon_object(
        exterior_x_coords=7 + SQUARE_X_COORDS_RELATIVE,
        exterior_y_coords=4 + SQUARE_Y_COORDS_RELATIVE)

    early_storm_object_table = pandas.DataFrame.from_dict({
        tracking_utils.CENTROID_X_COLUMN: numpy.array([5, 7], dtype=float),
        tracking_utils.CENTROID_Y_COLUMN: numpy.array([10, 4], dtype=float),
        POLYGON_COLUMN: [first_polygon_object, second_polygon_object]
    })

    this_polygon_object = polygons.vertex_arrays_to_polygon_object(
        exterior_x_coords=15 + DIAMOND_X_COORDS_RELATIVE,
        exterior_y_coords=6 + DIAMOND_Y_COORDS_RELATIVE)

    late_storm_object_table = pandas.DataFrame.from_dict({
        tracking_utils.CENTROID_X_COLUMN: numpy.full(1, 15.),
        tracking_utils.CENTROID_Y_COLUMN: numpy.full(1, 6.),
        POLYGON_COLUMN: [this_polygon_object]
    })

    return early_storm_object_table, late_storm_object_table


def _get_interp_data_for_split():
    """Creates synthetic data for interpolation with storm split.

    :return: early_storm_object_table: See doc for `_get_interp_data_for_merger`.
    :return: late_storm_object_table: Same.
    """

    this_polygon_object = polygons.vertex_arrays_to_polygon_object(
        exterior_x_coords=5 + DIAMOND_X_COORDS_RELATIVE,
        exterior_y_coords=5 + DIAMOND_Y_COORDS_RELATIVE)

    early_storm_object_table = pandas.DataFrame.from_dict({
        tracking_utils.CENTROID_X_COLUMN: numpy.full(1, 5.),
        tracking_utils.CENTROID_Y_COLUMN: numpy.full(1, 5.),
        POLYGON_COLUMN: [this_polygon_object]
    })

    first_polygon_object = polygons.vertex_arrays_to_polygon_object(
        exterior_x_coords=17 + TRIANGLE_X_COORDS_RELATIVE,
        exterior_y_coords=8 + TRIANGLE_Y_COORDS_RELATIVE)

    second_polygon_object = polygons.vertex_arrays_to_polygon_object(
        exterior_x_coords=16 + SQUARE_X_COORDS_RELATIVE,
        exterior_y_coords=3 + SQUARE_Y_COORDS_RELATIVE)

    late_storm_object_table = pandas.DataFrame.from_dict({
        tracking_utils.CENTROID_X_COLUMN: numpy.array([17, 16], dtype=float),
        tracking_utils.CENTROID_Y_COLUMN: numpy.array([8, 3], dtype=float),
        POLYGON_COLUMN: [first_polygon_object, second_polygon_object]
    })

    return early_storm_object_table, late_storm_object_table


def _get_track1_for_simple_pred():
    """Creates synthetic data for simple predecessors.

    :return: storm_object_table: See input doc for
        `storm_plotting.plot_storm_tracks`.
    """

    primary_id_strings = ['foo'] * 10
    secondary_id_strings = ['X', 'Y', 'X', 'Y', 'X', 'Y', 'Z', 'Z', 'Z', 'Z']

    valid_times_unix_sec = numpy.array(
        [0, 0, 1, 1, 2, 2, 3, 4, 5, 6], dtype=int
    )
    centroid_x_coords = numpy.array(
        [2, 2, 7, 7, 12, 12, 17, 22, 27, 32], dtype=float
    )
    centroid_y_coords = numpy.array(
        [8, 2, 8, 2, 8, 2, 5, 5, 5, 5], dtype=float
    )
    tornadic_flags = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0], dtype=bool)
    main_tornadic_flags = numpy.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=bool
    )

    first_prev_sec_id_strings = ['', '', 'X', 'Y', 'X', 'Y', 'X', 'Z', 'Z', 'Z']
    second_prev_sec_id_strings = ['', '', '', '', '', '', 'Y', '', '', '']
    first_next_sec_id_strings = [
        'X', 'Y', 'X', 'Y', 'Z', 'Z', 'Z', 'Z', 'Z', ''
    ]
    second_next_sec_id_strings = ['', '', '', '', '', '', '', '', '', '']

    return pandas.DataFrame.from_dict({
        tracking_utils.PRIMARY_ID_COLUMN: primary_id_strings,
        tracking_utils.SECONDARY_ID_COLUMN: secondary_id_strings,
        tracking_utils.VALID_TIME_COLUMN: valid_times_unix_sec,
        tracking_utils.CENTROID_X_COLUMN: centroid_x_coords,
        tracking_utils.CENTROID_Y_COLUMN: centroid_y_coords,
        TORNADIC_FLAG_COLUMN: tornadic_flags,
        SPECIAL_FLAG_COLUMN: main_tornadic_flags,
        tracking_utils.FIRST_PREV_SECONDARY_ID_COLUMN:
            first_prev_sec_id_strings,
        tracking_utils.SECOND_PREV_SECONDARY_ID_COLUMN:
            second_prev_sec_id_strings,
        tracking_utils.FIRST_NEXT_SECONDARY_ID_COLUMN:
            first_next_sec_id_strings,
        tracking_utils.SECOND_NEXT_SECONDARY_ID_COLUMN:
            second_next_sec_id_strings
    })


def _get_track2_for_simple_pred():
    """Creates synthetic data for simple predecessors.

    :return: storm_object_table: See doc for `_get_track1_for_simple_pred`.
    """

    primary_id_strings = ['bar'] * 17
    secondary_id_strings = [
        'A', 'A', 'A', 'B', 'C', 'B', 'C', 'B', 'C',
        'D', 'E', 'D', 'E', 'D', 'E', 'D', 'E'
    ]

    valid_times_unix_sec = numpy.array(
        [0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9], dtype=int
    )
    centroid_x_coords = numpy.array(
        [2, 6, 10, 14, 14, 18, 18, 22, 22, 26, 26, 30, 30, 34, 34, 38, 38],
        dtype=float
    )
    centroid_y_coords = numpy.array(
        [10, 10, 10, 13, 7, 13, 7, 13, 7, 10, 4, 10, 4, 10, 4, 10, 4],
        dtype=float
    )
    tornadic_flags = numpy.array(
        [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0], dtype=bool
    )
    main_tornadic_flags = numpy.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=bool
    )

    first_prev_sec_id_strings = [
        '', 'A', 'A', 'A', 'A', 'B', 'C', 'B', 'C',
        'C', 'C', 'D', 'E', 'D', 'E', 'D', 'E'
    ]
    second_prev_sec_id_strings = [
        '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''
    ]
    first_next_sec_id_strings = [
        'A', 'A', 'B', 'B', 'C', 'B', 'C', '', 'D',
        'D', 'E', 'D', 'E', 'D', 'E', '', ''
    ]
    second_next_sec_id_strings = [
        '', '', 'C', '', '', '', '', '', 'E',
        '', '', '', '', '', '', '', ''
    ]

    return pandas.DataFrame.from_dict({
        tracking_utils.PRIMARY_ID_COLUMN: primary_id_strings,
        tracking_utils.SECONDARY_ID_COLUMN: secondary_id_strings,
        tracking_utils.VALID_TIME_COLUMN: valid_times_unix_sec,
        tracking_utils.CENTROID_X_COLUMN: centroid_x_coords,
        tracking_utils.CENTROID_Y_COLUMN: centroid_y_coords,
        TORNADIC_FLAG_COLUMN: tornadic_flags,
        SPECIAL_FLAG_COLUMN: main_tornadic_flags,
        tracking_utils.FIRST_PREV_SECONDARY_ID_COLUMN:
            first_prev_sec_id_strings,
        tracking_utils.SECOND_PREV_SECONDARY_ID_COLUMN:
            second_prev_sec_id_strings,
        tracking_utils.FIRST_NEXT_SECONDARY_ID_COLUMN:
            first_next_sec_id_strings,
        tracking_utils.SECOND_NEXT_SECONDARY_ID_COLUMN:
            second_next_sec_id_strings
    })


def _get_track_for_simple_succ():
    """Creates synthetic data for simple successors.

    :return: storm_object_table: See doc for `_get_track1_for_simple_pred`.
    """

    primary_id_strings = ['moo'] * 21
    secondary_id_strings = [
        'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'C', 'D', 'C', 'D', 'C', 'D',
        'E', 'E', 'E', 'F', 'G', 'F', 'G'
    ]

    valid_times_unix_sec = numpy.array(
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 10, 11, 11],
        dtype=int
    )
    centroid_x_coords = numpy.array([
        5, 5, 10, 10, 15, 15, 20, 20, 25, 25, 30, 30, 35, 35, 40, 45, 50, 55,
        55, 60, 60
    ], dtype=float)
    centroid_y_coords = numpy.array(
        [8, 2, 8, 2, 8, 2, 8, 2, 11, 5, 11, 5, 11, 5, 8, 8, 8, 11, 5, 11, 5],
        dtype=float
    )
    tornadic_flags = numpy.array(
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        dtype=bool
    )
    main_tornadic_flags = numpy.array(
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        dtype=bool
    )

    first_prev_sec_id_strings = [
        '', '', 'A', 'B', 'A', 'B', 'A', 'B',
        '', 'A', 'C', 'D', 'C', 'D',
        'C', 'E', 'E',
        'E', 'E', 'F', 'G'
    ]
    second_prev_sec_id_strings = [
        '', '', '', '', '', '', '', '',
        '', 'B', '', '', '', '',
        'D', '', '',
        '', '', '', ''
    ]
    first_next_sec_id_strings = [
        'A', 'B', 'A', 'B', 'A', 'B', 'D', 'D',
        'C', 'D', 'C', 'D', 'E', 'E',
        'E', 'E', 'F',
        'F', 'G', '', ''
    ]
    second_next_sec_id_strings = [
        '', '', '', '', '', '', '', '',
        '', '', '', '', '', '',
        '', '', 'G',
        '', '', '', ''
    ]

    return pandas.DataFrame.from_dict({
        tracking_utils.PRIMARY_ID_COLUMN: primary_id_strings,
        tracking_utils.SECONDARY_ID_COLUMN: secondary_id_strings,
        tracking_utils.VALID_TIME_COLUMN: valid_times_unix_sec,
        tracking_utils.CENTROID_X_COLUMN: centroid_x_coords,
        tracking_utils.CENTROID_Y_COLUMN: centroid_y_coords,
        TORNADIC_FLAG_COLUMN: tornadic_flags,
        SPECIAL_FLAG_COLUMN: main_tornadic_flags,
        tracking_utils.FIRST_PREV_SECONDARY_ID_COLUMN:
            first_prev_sec_id_strings,
        tracking_utils.SECOND_PREV_SECONDARY_ID_COLUMN:
            second_prev_sec_id_strings,
        tracking_utils.FIRST_NEXT_SECONDARY_ID_COLUMN:
            first_next_sec_id_strings,
        tracking_utils.SECOND_NEXT_SECONDARY_ID_COLUMN:
            second_next_sec_id_strings
    })


def _plot_interp_two_times(early_storm_object_table, late_storm_object_table,
                           plot_legend):
    """Plots interpolation for one pair of times.

    :param early_storm_object_table: See doc for `_get_interp_data_for_merger`.
    :param late_storm_object_table: Same.
    :param plot_legend: Boolean flag.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    num_early_objects = len(early_storm_object_table.index)
    num_late_objects = len(late_storm_object_table.index)

    legend_handles = []
    legend_strings = []

    for i in range(num_early_objects):
        this_patch_object = PolygonPatch(
            early_storm_object_table[POLYGON_COLUMN].values[i],
            lw=0, ec=STORM_OBJECT_COLOUR, fc=STORM_OBJECT_COLOUR,
            alpha=POLYGON_OPACITY)
        axes_object.add_patch(this_patch_object)

        if i == 0:
            legend_handles.append(this_patch_object)
            legend_strings.append('Storm outline')

        this_early_x_coord = early_storm_object_table[
            tracking_utils.CENTROID_X_COLUMN].values[i]
        this_early_y_coord = early_storm_object_table[
            tracking_utils.CENTROID_Y_COLUMN].values[i]

        axes_object.text(
            this_early_x_coord, this_early_y_coord - 0.2, r'$t_1$',
            color=INTERP_COLOUR, horizontalalignment='center',
            verticalalignment='top')

        for j in range(num_late_objects):
            this_late_x_coord = late_storm_object_table[
                tracking_utils.CENTROID_X_COLUMN].values[j]
            this_late_y_coord = late_storm_object_table[
                tracking_utils.CENTROID_Y_COLUMN].values[j]

            if i == 0:
                this_patch_object = PolygonPatch(
                    late_storm_object_table[POLYGON_COLUMN].values[j],
                    lw=0, ec=STORM_OBJECT_COLOUR, fc=STORM_OBJECT_COLOUR,
                    alpha=POLYGON_OPACITY)
                axes_object.add_patch(this_patch_object)

                axes_object.text(
                    this_late_x_coord, this_late_y_coord - 0.2, r'$t_2$',
                    color=INTERP_COLOUR, horizontalalignment='center',
                    verticalalignment='top')

            this_handle = axes_object.plot(
                [this_early_x_coord, this_late_x_coord],
                [this_early_y_coord, this_late_y_coord],
                color=TRACK_COLOUR, linestyle='-', linewidth=TRACK_WIDTH
            )[0]

            if i == 0 and j == 0:
                legend_handles.append(this_handle)
                legend_strings.append('Storm track')

    this_handle = axes_object.plot(
        early_storm_object_table[tracking_utils.CENTROID_X_COLUMN].values,
        early_storm_object_table[tracking_utils.CENTROID_Y_COLUMN].values,
        linestyle='None', marker=DEFAULT_MARKER_TYPE,
        markersize=DEFAULT_MARKER_SIZE, markerfacecolor=STORM_OBJECT_COLOUR,
        markeredgecolor=STORM_OBJECT_COLOUR,
        markeredgewidth=DEFAULT_MARKER_EDGE_WIDTH
    )[0]

    legend_handles.insert(0, this_handle)
    legend_strings.insert(0, 'Storm center')

    axes_object.plot(
        late_storm_object_table[tracking_utils.CENTROID_X_COLUMN].values,
        late_storm_object_table[tracking_utils.CENTROID_Y_COLUMN].values,
        linestyle='None', marker=DEFAULT_MARKER_TYPE,
        markersize=DEFAULT_MARKER_SIZE, markerfacecolor=STORM_OBJECT_COLOUR,
        markeredgecolor=STORM_OBJECT_COLOUR,
        markeredgewidth=DEFAULT_MARKER_EDGE_WIDTH)

    if num_early_objects > 1:
        original_x_coords = early_storm_object_table[
            tracking_utils.CENTROID_X_COLUMN].values
        original_y_coords = early_storm_object_table[
            tracking_utils.CENTROID_Y_COLUMN].values

        simple_x_coord = late_storm_object_table[
            tracking_utils.CENTROID_X_COLUMN].values[0]
        simple_y_coord = late_storm_object_table[
            tracking_utils.CENTROID_Y_COLUMN].values[0]
        interp_polygon_object_xy = late_storm_object_table[
            POLYGON_COLUMN].values[0]
    else:
        simple_x_coord = early_storm_object_table[
            tracking_utils.CENTROID_X_COLUMN].values[0]
        simple_y_coord = early_storm_object_table[
            tracking_utils.CENTROID_Y_COLUMN].values[0]
        interp_polygon_object_xy = early_storm_object_table[
            POLYGON_COLUMN].values[0]

        original_x_coords = late_storm_object_table[
            tracking_utils.CENTROID_X_COLUMN].values
        original_y_coords = late_storm_object_table[
            tracking_utils.CENTROID_Y_COLUMN].values

    midpoint_x_coord = numpy.mean(original_x_coords)
    midpoint_y_coord = numpy.mean(original_y_coords)

    these_x_coords = numpy.array([
        original_x_coords[0], midpoint_x_coord, original_x_coords[1]
    ])
    these_y_coords = numpy.array([
        original_y_coords[0], midpoint_y_coord, original_y_coords[1]
    ])

    axes_object.plot(
        these_x_coords, these_y_coords, color=MIDPOINT_COLOUR,
        linestyle='--', linewidth=TRACK_WIDTH
    )

    this_handle = axes_object.plot(
        midpoint_x_coord, midpoint_y_coord, linestyle='None',
        marker=DEFAULT_MARKER_TYPE, markersize=DEFAULT_MARKER_SIZE,
        markerfacecolor=MIDPOINT_COLOUR, markeredgecolor=MIDPOINT_COLOUR,
        markeredgewidth=DEFAULT_MARKER_EDGE_WIDTH
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append('Virtual storm center')

    interp_x_coord = 0.5 * (simple_x_coord + midpoint_x_coord)
    interp_y_coord = 0.5 * (simple_y_coord + midpoint_y_coord)

    these_x_coords = numpy.array([
        midpoint_x_coord, interp_x_coord, simple_x_coord
    ])
    these_y_coords = numpy.array([
        midpoint_y_coord, interp_y_coord, simple_y_coord
    ])

    axes_object.plot(
        these_x_coords, these_y_coords, color=INTERP_COLOUR,
        linestyle='--', linewidth=TRACK_WIDTH
    )

    this_handle = axes_object.plot(
        interp_x_coord, interp_y_coord, linestyle='None',
        marker=DEFAULT_MARKER_TYPE, markersize=DEFAULT_MARKER_SIZE,
        markerfacecolor=INTERP_COLOUR, markeredgecolor=INTERP_COLOUR,
        markeredgewidth=DEFAULT_MARKER_EDGE_WIDTH
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append('Interp storm center')

    interp_polygon_object_xy = polygons.vertex_arrays_to_polygon_object(
        exterior_x_coords=(
            interp_x_coord - simple_x_coord +
            numpy.array(interp_polygon_object_xy.exterior.xy[0])
        ),
        exterior_y_coords=(
            interp_y_coord - simple_y_coord +
            numpy.array(interp_polygon_object_xy.exterior.xy[1])
        ),
    )

    this_patch_object = PolygonPatch(
        interp_polygon_object_xy, lw=0, ec=STORM_OBJECT_COLOUR,
        fc=STORM_OBJECT_COLOUR, alpha=POLYGON_OPACITY)
    axes_object.add_patch(this_patch_object)

    axes_object.xaxis.set_ticklabels([])
    axes_object.yaxis.set_ticklabels([])
    axes_object.xaxis.set_ticks_position('none')
    axes_object.yaxis.set_ticks_position('none')

    if plot_legend:
        axes_object.legend(legend_handles, legend_strings, loc='upper right')

    return figure_object, axes_object


def _plot_attribution_one_track(storm_object_table, plot_legend):
    """Plots tornado attribution for one storm track.

    :param storm_object_table: pandas DataFrame created by
        `_get_track1_for_simple_pred`, `_get_track2_for_simple_pred`, or
        `_get_track_for_simple_succ`.
    :param plot_legend: Boolean flag.
    :return: figure_object: See doc for `_plot_interp_two_times`.
    :return: axes_object: Same.
    """

    centroid_x_coords = storm_object_table[
        tracking_utils.CENTROID_X_COLUMN].values
    centroid_y_coords = storm_object_table[
        tracking_utils.CENTROID_Y_COLUMN].values
    secondary_id_strings = storm_object_table[
        tracking_utils.SECONDARY_ID_COLUMN].values

    storm_object_table = storm_object_table.assign(**{
        tracking_utils.CENTROID_LONGITUDE_COLUMN: centroid_x_coords,
        tracking_utils.CENTROID_LATITUDE_COLUMN: centroid_y_coords
    })

    figure_object, axes_object, basemap_object = (
        plotting_utils.create_equidist_cylindrical_map(
            min_latitude_deg=numpy.min(centroid_y_coords),
            max_latitude_deg=numpy.max(centroid_y_coords),
            min_longitude_deg=numpy.min(centroid_x_coords),
            max_longitude_deg=numpy.max(centroid_x_coords)
        )
    )

    storm_plotting.plot_storm_tracks(
        storm_object_table=storm_object_table, axes_object=axes_object,
        basemap_object=basemap_object, colour_map_object=None,
        line_colour=INTERP_COLOUR, line_width=TRACK_WIDTH,
        start_marker_type=None, end_marker_type=None)

    tornadic_flags = storm_object_table[TORNADIC_FLAG_COLUMN].values
    main_tornadic_flags = storm_object_table[SPECIAL_FLAG_COLUMN].values

    legend_handles = [None] * 3
    legend_strings = [None] * 3

    for i in range(len(centroid_x_coords)):
        if main_tornadic_flags[i]:
            this_handle = axes_object.plot(
                centroid_x_coords[i], centroid_y_coords[i], linestyle='None',
                marker=TORNADIC_MARKER_TYPE, markersize=TORNADIC_MARKER_SIZE,
                markerfacecolor=TORNADIC_COLOUR,
                markeredgecolor=TORNADIC_COLOUR,
                markeredgewidth=TORNADIC_MARKER_EDGE_WIDTH
            )[0]

            legend_handles[0] = this_handle
            legend_strings[0] = 'Object initially linked\nto tornado'

            axes_object.text(
                centroid_x_coords[i], centroid_y_coords[i] - TEXT_OFFSET,
                secondary_id_strings[i], color=TORNADIC_COLOUR,
                fontsize=FONT_SIZE, fontweight='bold',
                horizontalalignment='center', verticalalignment='top')
        else:
            if tornadic_flags[i]:
                this_edge_colour = TORNADIC_COLOUR
                this_face_colour = TORNADIC_COLOUR
            else:
                this_edge_colour = NON_TORNADIC_COLOUR
                this_face_colour = 'white'

            this_handle = axes_object.plot(
                centroid_x_coords[i], centroid_y_coords[i], linestyle='None',
                marker=DEFAULT_MARKER_TYPE, markersize=DEFAULT_MARKER_SIZE,
                markerfacecolor=this_face_colour,
                markeredgecolor=this_edge_colour,
                markeredgewidth=DEFAULT_MARKER_EDGE_WIDTH
            )[0]

            if tornadic_flags[i] and legend_handles[1] is None:
                legend_handles[1] = this_handle
                legend_strings[1] = 'Also linked to tornado'

            if not tornadic_flags[i] and legend_handles[2] is None:
                legend_handles[2] = this_handle
                legend_strings[2] = 'Not linked to tornado'

            axes_object.text(
                centroid_x_coords[i], centroid_y_coords[i] - TEXT_OFFSET,
                secondary_id_strings[i], color=this_edge_colour,
                fontsize=FONT_SIZE, fontweight='bold',
                horizontalalignment='center', verticalalignment='top')

    axes_object.xaxis.set_ticklabels([])
    axes_object.yaxis.set_ticklabels([])
    axes_object.xaxis.set_ticks_position('none')
    axes_object.yaxis.set_ticks_position('none')
    axes_object.set_xlabel(r'Time $\longrightarrow$')

    if plot_legend:
        axes_object.legend(legend_handles, legend_strings, loc='lower right')

    return figure_object, axes_object


def _run():
    """Makes event-attribution schematics for 2019 tornado-prediction paper.

    This is effectively the main method.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=OUTPUT_DIR_NAME)

    # Interpolation with merger.
    figure_object, axes_object = _plot_interp_two_times(
        early_storm_object_table=_get_interp_data_for_merger()[0],
        late_storm_object_table=_get_interp_data_for_merger()[1],
        plot_legend=True
    )

    this_file_name = '{0:s}/interp_with_merger_standalone.jpg'.format(
        OUTPUT_DIR_NAME)

    print('Saving figure to: "{0:s}"...'.format(this_file_name))
    figure_object.savefig(
        this_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )

    plotting_utils.label_axes(axes_object=axes_object, label_string='(a)')
    axes_object.set_title('Interpolation with merger')
    panel_file_names = ['{0:s}/interp_with_merger.jpg'.format(OUTPUT_DIR_NAME)]

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Interpolation with split.
    figure_object, axes_object = _plot_interp_two_times(
        early_storm_object_table=_get_interp_data_for_split()[0],
        late_storm_object_table=_get_interp_data_for_split()[1],
        plot_legend=False
    )

    this_file_name = '{0:s}/interp_with_split_standalone.jpg'.format(
        OUTPUT_DIR_NAME)

    print('Saving figure to: "{0:s}"...'.format(this_file_name))
    figure_object.savefig(
        this_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )

    plotting_utils.label_axes(axes_object=axes_object, label_string='(b)')
    axes_object.set_title('Interpolation with split')
    panel_file_names.append(
        '{0:s}/interp_with_split.jpg'.format(OUTPUT_DIR_NAME)
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Simple successors.
    figure_object, axes_object = _plot_attribution_one_track(
        storm_object_table=_get_track_for_simple_succ(), plot_legend=True
    )

    this_file_name = '{0:s}/simple_successors_standalone.jpg'.format(
        OUTPUT_DIR_NAME)

    print('Saving figure to: "{0:s}"...'.format(this_file_name))
    figure_object.savefig(
        this_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )

    plotting_utils.label_axes(axes_object=axes_object, label_string='(c)')
    axes_object.set_title('Linking to simple successors')
    panel_file_names.append(
        '{0:s}/simple_successors.jpg'.format(OUTPUT_DIR_NAME)
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Simple predecessors, example 1.
    figure_object, axes_object = _plot_attribution_one_track(
        storm_object_table=_get_track1_for_simple_pred(), plot_legend=False
    )

    axes_object.set_title('Simple predecessors, example 1')
    this_file_name = '{0:s}/simple_predecessors_track1_standalone.jpg'.format(
        OUTPUT_DIR_NAME)

    print('Saving figure to: "{0:s}"...'.format(this_file_name))
    figure_object.savefig(
        this_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )

    plotting_utils.label_axes(axes_object=axes_object, label_string='(d)')
    axes_object.set_title('Linking to simple predecessors, example 1')
    panel_file_names.append(
        '{0:s}/simple_predecessors_track1.jpg'.format(OUTPUT_DIR_NAME)
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Simple predecessors, example 2.
    figure_object, axes_object = _plot_attribution_one_track(
        storm_object_table=_get_track2_for_simple_pred(), plot_legend=False
    )

    axes_object.set_title('Simple predecessors, example 2')
    this_file_name = '{0:s}/simple_predecessors_track2_standalone.jpg'.format(
        OUTPUT_DIR_NAME)

    print('Saving figure to: "{0:s}"...'.format(this_file_name))
    figure_object.savefig(
        this_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )

    plotting_utils.label_axes(axes_object=axes_object, label_string='(e)')
    axes_object.set_title('Linking to simple predecessors, example 2')
    panel_file_names.append(
        '{0:s}/simple_predecessors_track2.jpg'.format(OUTPUT_DIR_NAME)
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Concatenate all panels into one figure.
    concat_file_name = '{0:s}/attribution_schemas.jpg'.format(OUTPUT_DIR_NAME)
    print('Concatenating panels to: "{0:s}"...'.format(concat_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names, output_file_name=concat_file_name,
        num_panel_rows=2, num_panel_columns=3)

    imagemagick_utils.resize_image(
        input_file_name=concat_file_name, output_file_name=concat_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX)


if __name__ == '__main__':
    _run()
