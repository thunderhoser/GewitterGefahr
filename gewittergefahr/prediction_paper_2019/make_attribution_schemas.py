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

TORNADO_TIME_COLUMN = 'valid_time_unix_sec'
TORNADO_X_COLUMN = 'x_coord_metres'
TORNADO_Y_COLUMN = 'y_coord_metres'

SQUARE_X_COORDS = 2 * numpy.array([-1, -1, 1, 1, -1], dtype=float)
SQUARE_Y_COORDS = numpy.array([-1, 1, 1, -1, -1], dtype=float)

THIS_NUM = numpy.sqrt(3) / 2
HEXAGON_X_COORDS = 2 * numpy.array([1, 0.5, -0.5, -1, -0.5, 0.5, 1])
HEXAGON_Y_COORDS = numpy.array([
    0, -THIS_NUM, -THIS_NUM, 0, THIS_NUM, THIS_NUM, 0
])

THIS_NUM = numpy.sqrt(2) / 2
OCTAGON_X_COORDS = 2 * numpy.array([
    1, THIS_NUM, 0, -THIS_NUM, -1, -THIS_NUM, 0, THIS_NUM, 1
])
OCTAGON_Y_COORDS = numpy.array([
    0, THIS_NUM, 1, THIS_NUM, 0, -THIS_NUM, -1, -THIS_NUM, 0
])

TRACK_COLOUR = numpy.full(3, 0.)
MIDPOINT_COLOUR = numpy.full(3, 152. / 255)
TORNADIC_STORM_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
NON_TORNADIC_STORM_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
NON_INTERP_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
INTERP_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255

DEFAULT_FONT_SIZE = 40
SMALL_LEGEND_FONT_SIZE = 30
TEXT_OFFSET = 0.25

pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

TRACK_WIDTH = 4
POLYGON_OPACITY = 0.5
DEFAULT_MARKER_TYPE = 'o'
DEFAULT_MARKER_SIZE = 24
DEFAULT_MARKER_EDGE_WIDTH = 4
TORNADIC_STORM_MARKER_TYPE = '*'
TORNADIC_STORM_MARKER_SIZE = 48
TORNADIC_STORM_MARKER_EDGE_WIDTH = 0
TORNADO_MARKER_TYPE = 'v'
TORNADO_MARKER_SIZE = 48
TORNADO_MARKER_EDGE_WIDTH = 0

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

OUTPUT_DIR_NAME = (
    '/localdata/ryan.lagerquist/eager/prediction_paper_2019/attribution_schemas'
)


def _get_data_for_interp_with_split():
    """Creates synthetic data for interpolation with storm split.

    :return: storm_object_table: pandas DataFrame with the following columns.
        Each row is one storm object.
    storm_object_table.primary_id_string: Primary storm ID.
    storm_object_table.secondary_id_string: Secondary storm ID.
    storm_object_table.valid_time_unix_sec: Valid time.
    storm_object_table.centroid_x_metres: x-coordinate of centroid.
    storm_object_table.centroid_y_metres: y-coordinate of centroid.
    storm_object_table.polygon_object_xy_metres: Storm outline (instance of
        `shapely.geometry.Polygon`).
    storm_object_table.first_prev_secondary_id_string: Secondary ID of first
        predecessor ("" if no predecessors).
    storm_object_table.second_prev_secondary_id_string: Secondary ID of second
        predecessor ("" if only one predecessor).
    storm_object_table.first_next_secondary_id_string: Secondary ID of first
        successor ("" if no successors).
    storm_object_table.second_next_secondary_id_string: Secondary ID of second
        successor ("" if no successors).

    :return: tornado_table: pandas DataFrame with the following columns.
    tornado_table.valid_time_unix_sec: Valid time.
    tornado_table.x_coord_metres: x-coordinate.
    tornado_table.y_coord_metres: y-coordinate.
    """

    primary_id_strings = ['foo'] * 5
    secondary_id_strings = ['A', 'A', 'A', 'B', 'C']

    valid_times_unix_sec = numpy.array([5, 10, 15, 20, 20], dtype=int)
    centroid_x_coords = numpy.array([2, 7, 12, 17, 17], dtype=float)
    centroid_y_coords = numpy.array([5, 5, 5, 8, 2], dtype=float)

    first_prev_sec_id_strings = ['', 'A', 'A', 'A', 'A']
    second_prev_sec_id_strings = ['', '', '', '', '']
    first_next_sec_id_strings = ['A', 'A', 'B', '', '']
    second_next_sec_id_strings = ['', '', 'C', '', '']

    num_storm_objects = len(secondary_id_strings)
    polygon_objects_xy = [None] * num_storm_objects

    for i in range(num_storm_objects):
        if secondary_id_strings[i] == 'B':
            these_x_coords = OCTAGON_X_COORDS
            these_y_coords = OCTAGON_Y_COORDS
        elif secondary_id_strings[i] == 'C':
            these_x_coords = HEXAGON_X_COORDS
            these_y_coords = HEXAGON_Y_COORDS
        else:
            these_x_coords = SQUARE_X_COORDS
            these_y_coords = SQUARE_Y_COORDS

        polygon_objects_xy[i] = polygons.vertex_arrays_to_polygon_object(
            exterior_x_coords=centroid_x_coords[i] + these_x_coords / 2,
            exterior_y_coords=centroid_y_coords[i] + these_y_coords / 2
        )

    storm_object_table = pandas.DataFrame.from_dict({
        tracking_utils.PRIMARY_ID_COLUMN: primary_id_strings,
        tracking_utils.SECONDARY_ID_COLUMN: secondary_id_strings,
        tracking_utils.VALID_TIME_COLUMN: valid_times_unix_sec,
        tracking_utils.CENTROID_X_COLUMN: centroid_x_coords,
        tracking_utils.CENTROID_Y_COLUMN: centroid_y_coords,
        tracking_utils.FIRST_PREV_SECONDARY_ID_COLUMN:
            first_prev_sec_id_strings,
        tracking_utils.SECOND_PREV_SECONDARY_ID_COLUMN:
            second_prev_sec_id_strings,
        tracking_utils.FIRST_NEXT_SECONDARY_ID_COLUMN:
            first_next_sec_id_strings,
        tracking_utils.SECOND_NEXT_SECONDARY_ID_COLUMN:
            second_next_sec_id_strings,
        POLYGON_COLUMN: polygon_objects_xy
    })

    tornado_table = pandas.DataFrame.from_dict({
        TORNADO_TIME_COLUMN: numpy.array([18], dtype=int),
        TORNADO_X_COLUMN: numpy.array([15.]),
        TORNADO_Y_COLUMN: numpy.array([3.])
    })

    return storm_object_table, tornado_table


def _get_data_for_interp_with_merger():
    """Creates synthetic data for interpolation with storm merger.

    :return: storm_object_table: See doc for `_get_data_for_interp_with_split`.
    :return: tornado_table: Same.
    """

    primary_id_strings = ['foo'] * 6
    secondary_id_strings = ['A', 'B', 'A', 'B', 'C', 'C']

    valid_times_unix_sec = numpy.array([5, 5, 10, 10, 15, 20], dtype=int)
    centroid_x_coords = numpy.array([2, 2, 7, 7, 12, 17], dtype=float)
    centroid_y_coords = numpy.array([8, 2, 8, 2, 5, 5], dtype=float)

    first_prev_sec_id_strings = ['', '', 'A', 'B', 'A', 'C']
    second_prev_sec_id_strings = ['', '', '', '', 'B', '']
    first_next_sec_id_strings = ['A', 'B', 'C', 'C', 'C', '']
    second_next_sec_id_strings = ['', '', '', '', '', '']

    num_storm_objects = len(secondary_id_strings)
    polygon_objects_xy = [None] * num_storm_objects

    for i in range(num_storm_objects):
        if secondary_id_strings[i] == 'A':
            these_x_coords = OCTAGON_X_COORDS
            these_y_coords = OCTAGON_Y_COORDS
        elif secondary_id_strings[i] == 'B':
            these_x_coords = HEXAGON_X_COORDS
            these_y_coords = HEXAGON_Y_COORDS
        else:
            these_x_coords = SQUARE_X_COORDS
            these_y_coords = SQUARE_Y_COORDS

        polygon_objects_xy[i] = polygons.vertex_arrays_to_polygon_object(
            exterior_x_coords=centroid_x_coords[i] + these_x_coords / 2,
            exterior_y_coords=centroid_y_coords[i] + these_y_coords / 2
        )

    storm_object_table = pandas.DataFrame.from_dict({
        tracking_utils.PRIMARY_ID_COLUMN: primary_id_strings,
        tracking_utils.SECONDARY_ID_COLUMN: secondary_id_strings,
        tracking_utils.VALID_TIME_COLUMN: valid_times_unix_sec,
        tracking_utils.CENTROID_X_COLUMN: centroid_x_coords,
        tracking_utils.CENTROID_Y_COLUMN: centroid_y_coords,
        tracking_utils.FIRST_PREV_SECONDARY_ID_COLUMN:
            first_prev_sec_id_strings,
        tracking_utils.SECOND_PREV_SECONDARY_ID_COLUMN:
            second_prev_sec_id_strings,
        tracking_utils.FIRST_NEXT_SECONDARY_ID_COLUMN:
            first_next_sec_id_strings,
        tracking_utils.SECOND_NEXT_SECONDARY_ID_COLUMN:
            second_next_sec_id_strings,
        POLYGON_COLUMN: polygon_objects_xy
    })

    tornado_table = pandas.DataFrame.from_dict({
        TORNADO_TIME_COLUMN: numpy.array([12], dtype=int),
        TORNADO_X_COLUMN: numpy.array([9.]),
        TORNADO_Y_COLUMN: numpy.array([3.])
    })

    return storm_object_table, tornado_table


def _get_track1_for_simple_pred():
    """Creates synthetic data for simple predecessors.

    :return: storm_object_table: Same as table produced by
        `_get_data_for_interp_with_split`, except without column
        "polygon_object_xy_metres" and with the following extra columns.
    storm_object_table.is_tornadic: Boolean flag (True if storm object is
        linked to a tornado).
    storm_object_table.is_main_tornadic_link: Boolean flag (True if storm object
        is the main one linked to a tornado, rather than being linked to tornado
        as a predecessor or successor).
    """

    primary_id_strings = ['foo'] * 10
    secondary_id_strings = ['X', 'Y', 'X', 'Y', 'X', 'Y', 'Z', 'Z', 'Z', 'Z']

    valid_times_unix_sec = numpy.array(
        [5, 5, 10, 10, 15, 15, 20, 25, 30, 35], dtype=int
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
        [5, 10, 15, 20, 20, 25, 25, 30, 30, 35, 35, 40, 40, 45, 45, 50, 50],
        dtype=int
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

    valid_times_unix_sec = numpy.array([
        5, 5, 10, 10, 15, 15, 20, 20, 25, 25, 30, 30, 35, 35, 40, 45, 50, 55,
        55, 60, 60
    ], dtype=int)
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


def _plot_interp_two_times(storm_object_table, tornado_table, legend_font_size,
                           legend_position_string):
    """Plots interpolation for one pair of times.

    :param storm_object_table: See doc for `_get_interp_data_for_split`.
    :param tornado_table: Same.
    :param legend_font_size: Font size in legend.
    :param legend_position_string: Legend position.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
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
        line_colour=TRACK_COLOUR, line_width=TRACK_WIDTH,
        start_marker_type=None, end_marker_type=None)

    num_storm_objects = len(storm_object_table.index)
    legend_handles = []
    legend_strings = []

    for i in range(num_storm_objects):
        this_patch_object = PolygonPatch(
            storm_object_table[POLYGON_COLUMN].values[i],
            lw=0, ec=NON_INTERP_COLOUR, fc=NON_INTERP_COLOUR,
            alpha=POLYGON_OPACITY)

        axes_object.add_patch(this_patch_object)

    this_handle = axes_object.plot(
        storm_object_table[tracking_utils.CENTROID_X_COLUMN].values,
        storm_object_table[tracking_utils.CENTROID_Y_COLUMN].values,
        linestyle='None', marker=DEFAULT_MARKER_TYPE,
        markersize=DEFAULT_MARKER_SIZE, markerfacecolor=NON_INTERP_COLOUR,
        markeredgecolor=NON_INTERP_COLOUR,
        markeredgewidth=DEFAULT_MARKER_EDGE_WIDTH
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append('Actual storm')

    for i in range(num_storm_objects):
        axes_object.text(
            centroid_x_coords[i], centroid_y_coords[i] - TEXT_OFFSET,
            secondary_id_strings[i], color=TRACK_COLOUR,
            fontsize=DEFAULT_FONT_SIZE, fontweight='bold',
            horizontalalignment='center', verticalalignment='top')

    storm_times_minutes = storm_object_table[
        tracking_utils.VALID_TIME_COLUMN].values
    tornado_time_minutes = tornado_table[TORNADO_TIME_COLUMN].values[0]

    previous_time_minutes = numpy.max(
        storm_times_minutes[storm_times_minutes < tornado_time_minutes]
    )
    next_time_minutes = numpy.min(
        storm_times_minutes[storm_times_minutes > tornado_time_minutes]
    )

    previous_object_indices = numpy.where(
        storm_times_minutes == previous_time_minutes
    )[0]
    next_object_indices = numpy.where(
        storm_times_minutes == next_time_minutes
    )[0]

    previous_x_coord = numpy.mean(centroid_x_coords[previous_object_indices])
    previous_y_coord = numpy.mean(centroid_y_coords[previous_object_indices])
    next_x_coord = numpy.mean(centroid_x_coords[next_object_indices])
    next_y_coord = numpy.mean(centroid_y_coords[next_object_indices])

    if len(next_object_indices) == 1:
        midpoint_x_coord = previous_x_coord
        midpoint_y_coord = previous_y_coord
        midpoint_label_string = (
            'Midpoint of {0:s} and {1:s}\n(at {2:d} minutes)'
        ).format(
            secondary_id_strings[previous_object_indices[0]],
            secondary_id_strings[previous_object_indices[1]],
            storm_times_minutes[previous_object_indices[0]]
        )
    else:
        midpoint_x_coord = next_x_coord
        midpoint_y_coord = next_y_coord
        midpoint_label_string = (
            'Midpoint of {0:s} and {1:s}\n(at {2:d} minutes)'
        ).format(
            secondary_id_strings[next_object_indices[0]],
            secondary_id_strings[next_object_indices[1]],
            storm_times_minutes[next_object_indices[0]]
        )

    this_handle = axes_object.plot(
        midpoint_x_coord, midpoint_y_coord, linestyle='None',
        marker=DEFAULT_MARKER_TYPE, markersize=DEFAULT_MARKER_SIZE,
        markerfacecolor='white', markeredgecolor=MIDPOINT_COLOUR,
        markeredgewidth=DEFAULT_MARKER_EDGE_WIDTH
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append(midpoint_label_string)

    interp_x_coord = 0.5 * (previous_x_coord + next_x_coord)
    interp_y_coord = 0.5 * (previous_y_coord + next_y_coord)

    if len(next_object_indices) == 1:
        x_offset = interp_x_coord - next_x_coord
        y_offset = interp_y_coord - next_y_coord
        interp_polygon_object_xy = storm_object_table[POLYGON_COLUMN].values[
            next_object_indices[0]
        ]
    else:
        x_offset = interp_x_coord - previous_x_coord
        y_offset = interp_y_coord - previous_y_coord
        interp_polygon_object_xy = storm_object_table[POLYGON_COLUMN].values[
            previous_object_indices[0]
        ]

    interp_polygon_object_xy = polygons.vertex_arrays_to_polygon_object(
        exterior_x_coords=(
            x_offset + numpy.array(interp_polygon_object_xy.exterior.xy[0])
        ),
        exterior_y_coords=(
            y_offset + numpy.array(interp_polygon_object_xy.exterior.xy[1])
        )
    )

    this_patch_object = PolygonPatch(
        interp_polygon_object_xy, lw=0, ec=INTERP_COLOUR, fc=INTERP_COLOUR,
        alpha=POLYGON_OPACITY)
    axes_object.add_patch(this_patch_object)

    this_handle = axes_object.plot(
        interp_x_coord, interp_y_coord, linestyle='None',
        marker=DEFAULT_MARKER_TYPE, markersize=DEFAULT_MARKER_SIZE,
        markerfacecolor=INTERP_COLOUR, markeredgecolor=INTERP_COLOUR,
        markeredgewidth=DEFAULT_MARKER_EDGE_WIDTH
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append('Interpolated storm')

    this_handle = axes_object.plot(
        tornado_table[TORNADO_X_COLUMN].values[0],
        tornado_table[TORNADO_Y_COLUMN].values[0], linestyle='None',
        marker=TORNADO_MARKER_TYPE, markersize=TORNADO_MARKER_SIZE,
        markerfacecolor=INTERP_COLOUR, markeredgecolor=INTERP_COLOUR,
        markeredgewidth=TORNADO_MARKER_EDGE_WIDTH
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append(
        'Tornado\n(at {0:d} minutes)'.format(tornado_time_minutes)
    )

    x_tick_values, unique_indices = numpy.unique(
        centroid_x_coords, return_index=True)
    x_tick_labels = [
        '{0:d}'.format(storm_times_minutes[i]) for i in unique_indices
    ]

    axes_object.set_xticks(x_tick_values)
    axes_object.set_xticklabels(x_tick_labels)
    axes_object.set_xlabel('Storm time (minutes)')

    axes_object.set_yticks([], [])
    axes_object.legend(
        legend_handles, legend_strings, fontsize=legend_font_size,
        loc=legend_position_string)

    return figure_object, axes_object


def _plot_attribution_one_track(storm_object_table, plot_legend, plot_x_ticks,
                                legend_font_size=None, legend_location=None):
    """Plots tornado attribution for one storm track.

    :param storm_object_table: pandas DataFrame created by
        `_get_track1_for_simple_pred`, `_get_track2_for_simple_pred`, or
        `_get_track_for_simple_succ`.
    :param plot_legend: Boolean flag.
    :param plot_x_ticks: Boolean flag.
    :param legend_font_size: Font size in legend (used only if
        `plot_legend == True`).
    :param legend_location: Legend location (used only if
        `plot_legend == True`).
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
        line_colour=TRACK_COLOUR, line_width=TRACK_WIDTH,
        start_marker_type=None, end_marker_type=None)

    tornadic_flags = storm_object_table[TORNADIC_FLAG_COLUMN].values
    main_tornadic_flags = storm_object_table[SPECIAL_FLAG_COLUMN].values

    legend_handles = [None] * 3
    legend_strings = [None] * 3

    for i in range(len(centroid_x_coords)):
        if main_tornadic_flags[i]:
            this_handle = axes_object.plot(
                centroid_x_coords[i], centroid_y_coords[i], linestyle='None',
                marker=TORNADIC_STORM_MARKER_TYPE,
                markersize=TORNADIC_STORM_MARKER_SIZE,
                markerfacecolor=TORNADIC_STORM_COLOUR,
                markeredgecolor=TORNADIC_STORM_COLOUR,
                markeredgewidth=TORNADIC_STORM_MARKER_EDGE_WIDTH
            )[0]

            legend_handles[0] = this_handle
            legend_strings[0] = 'Object initially linked\nto tornado'

            axes_object.text(
                centroid_x_coords[i], centroid_y_coords[i] - TEXT_OFFSET,
                secondary_id_strings[i], color=TORNADIC_STORM_COLOUR,
                fontsize=DEFAULT_FONT_SIZE, fontweight='bold',
                horizontalalignment='center', verticalalignment='top')
        else:
            if tornadic_flags[i]:
                this_edge_colour = TORNADIC_STORM_COLOUR
                this_face_colour = TORNADIC_STORM_COLOUR
            else:
                this_edge_colour = NON_TORNADIC_STORM_COLOUR
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
                fontsize=DEFAULT_FONT_SIZE, fontweight='bold',
                horizontalalignment='center', verticalalignment='top')

    if plot_x_ticks:
        storm_times_minutes = storm_object_table[
            tracking_utils.VALID_TIME_COLUMN].values

        x_tick_values, unique_indices = numpy.unique(
            centroid_x_coords, return_index=True)
        x_tick_labels = [
            '{0:d}'.format(storm_times_minutes[i]) for i in unique_indices
        ]

        axes_object.set_xticks(x_tick_values)
        axes_object.set_xticklabels(x_tick_labels)
        axes_object.set_xlabel('Storm time (minutes)')
    else:
        axes_object.set_xticks([], [])
        axes_object.set_xlabel(r'Time $\longrightarrow$')

    axes_object.set_yticks([], [])

    if plot_legend:
        axes_object.legend(
            legend_handles, legend_strings, fontsize=legend_font_size,
            loc=legend_location)

    return figure_object, axes_object


def _run():
    """Makes event-attribution schematics for 2019 tornado-prediction paper.

    This is effectively the main method.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=OUTPUT_DIR_NAME)

    # Interpolation with merger.
    figure_object, axes_object = _plot_interp_two_times(
        storm_object_table=_get_data_for_interp_with_merger()[0],
        tornado_table=_get_data_for_interp_with_merger()[1],
        legend_font_size=SMALL_LEGEND_FONT_SIZE, legend_position_string='upper right'
    )

    axes_object.set_title('Interpolation with merger')
    this_file_name = '{0:s}/interp_with_merger_standalone.jpg'.format(
        OUTPUT_DIR_NAME)

    print('Saving figure to: "{0:s}"...'.format(this_file_name))
    figure_object.savefig(
        this_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )

    plotting_utils.label_axes(axes_object=axes_object, label_string='(a)')
    panel_file_names = ['{0:s}/interp_with_merger.jpg'.format(OUTPUT_DIR_NAME)]

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Interpolation with split.
    figure_object, axes_object = _plot_interp_two_times(
        storm_object_table=_get_data_for_interp_with_split()[0],
        tornado_table=_get_data_for_interp_with_split()[1],
        legend_font_size=DEFAULT_FONT_SIZE,
        legend_position_string='upper left'
    )

    axes_object.set_title('Interpolation with split')
    this_file_name = '{0:s}/interp_with_split_standalone.jpg'.format(
        OUTPUT_DIR_NAME)

    print('Saving figure to: "{0:s}"...'.format(this_file_name))
    figure_object.savefig(
        this_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )

    plotting_utils.label_axes(axes_object=axes_object, label_string='(b)')
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
        storm_object_table=_get_track_for_simple_succ(),
        plot_legend=True, plot_x_ticks=True,
        legend_font_size=SMALL_LEGEND_FONT_SIZE, legend_location='lower right'
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
        storm_object_table=_get_track1_for_simple_pred(),
        plot_legend=True, plot_x_ticks=False,
        legend_font_size=DEFAULT_FONT_SIZE, legend_location=(0.28, 0.1)
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
        storm_object_table=_get_track2_for_simple_pred(),
        plot_legend=False, plot_x_ticks=False
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
