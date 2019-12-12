"""Makes schema for storm-velocity estimation."""

import numpy
import pandas
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import temporal_tracking
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import storm_plotting

TRACK_WIDTH = 4
TRACK_COLOUR = numpy.full(3, 0.)

SPECIAL_STORM_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
DEFAULT_STORM_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255

MARKER_TYPE = 'o'
MARKER_SIZE = 24
MARKER_EDGE_WIDTH = 4

FONT_SIZE = 40
TEXT_OFFSET = 0.25
FIGURE_RESOLUTION_DPI = 300

pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

OUTPUT_FILE_NAME = (
    '/localdata/ryan.lagerquist/eager/dissertation_figures/'
    'storm_velocity_schema.jpg'
)


def _create_tracking_data():
    """Creates fictitious storm-tracking data.

    :return: storm_object_table: pandas DataFrame with the following columns.
        Each row is one storm object.
    storm_object_table.primary_id_string: Primary storm ID.
    storm_object_table.secondary_id_string: Secondary storm ID.
    storm_object_table.valid_time_unix_sec: Valid time.
    storm_object_table.centroid_x_metres: x-coordinate of centroid.
    storm_object_table.centroid_y_metres: y-coordinate of centroid.
    storm_object_table.first_prev_secondary_id_string: Secondary ID of first
        predecessor ("" if no predecessors).
    storm_object_table.second_prev_secondary_id_string: Secondary ID of second
        predecessor ("" if only one predecessor).
    storm_object_table.first_next_secondary_id_string: Secondary ID of first
        successor ("" if no successors).
    storm_object_table.second_next_secondary_id_string: Secondary ID of second
        successor ("" if no successors).
    """

    primary_id_strings = ['foo'] * 8
    secondary_id_strings = ['A', 'B', 'C', 'A', 'B', 'C', 'D', 'E']

    valid_times_unix_sec = numpy.array([0, 0, 5, 5, 5, 10, 10, 15], dtype=int)
    centroid_x_coords = numpy.array([0, 0, 5, 5, 5, 10, 10, 15], dtype=float)
    centroid_y_coords = numpy.array([8, 2, 15, 8, 2, 15, 5, 10], dtype=float)

    first_prev_sec_id_strings = ['', '', '', 'A', 'B', 'C', 'A', 'C']
    second_prev_sec_id_strings = ['', '', '', '', '', '', 'B', 'D']
    first_next_sec_id_strings = ['A', 'B', 'C', 'D', 'D', 'E', 'E', '']
    second_next_sec_id_strings = ['', '', '', '', '', '', '', '']

    return pandas.DataFrame.from_dict({
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
            second_next_sec_id_strings
    })


def _plot_schema(storm_object_table, output_file_name):
    """Plots schema for storm-velocity estimation.

    :param storm_object_table: pandas DataFrame created by
        `_create_tracking_data`.
    :param output_file_name: Path to output file (figure will be saved here).
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

    predecessor_rows = temporal_tracking.find_predecessors(
        storm_object_table=storm_object_table, target_row=num_storm_objects - 1,
        num_seconds_back=100, return_all_on_path=False)

    legend_handles = [None] * 2
    legend_strings = [None] * 2

    for i in range(num_storm_objects):
        if i in predecessor_rows or i == num_storm_objects - 1:
            this_colour = SPECIAL_STORM_COLOUR
        else:
            this_colour = DEFAULT_STORM_COLOUR

        this_handle = axes_object.plot(
            centroid_x_coords[i], centroid_y_coords[i], linestyle='None',
            marker=MARKER_TYPE, markersize=MARKER_SIZE,
            markerfacecolor=this_colour, markeredgecolor=this_colour,
            markeredgewidth=MARKER_EDGE_WIDTH
        )[0]

        if i in predecessor_rows or i == num_storm_objects - 1:
            legend_handles[0] = this_handle
            legend_strings[0] = 'Object used in\nvelocity estimate'
        else:
            legend_handles[1] = this_handle
            legend_strings[1] = 'Object not used'

        axes_object.text(
            centroid_x_coords[i], centroid_y_coords[i] - TEXT_OFFSET,
            secondary_id_strings[i], color=this_colour,
            fontsize=FONT_SIZE, fontweight='bold',
            horizontalalignment='center', verticalalignment='top')

    axes_object.set_yticks([], [])

    storm_times_minutes = storm_object_table[
        tracking_utils.VALID_TIME_COLUMN].values

    x_tick_values, unique_indices = numpy.unique(
        centroid_x_coords, return_index=True)
    x_tick_labels = [
        '{0:d}'.format(storm_times_minutes[i]) for i in unique_indices
    ]

    axes_object.set_xticks(x_tick_values)
    axes_object.set_xticklabels(x_tick_labels)
    axes_object.set_xlabel('Time (minutes)')

    axes_object.legend(
        legend_handles, legend_strings, fontsize=FONT_SIZE, loc=(0.02, 0.6)
    )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run():
    """Makes schema for storm-velocity estimation.

    This is effectively the main method.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=OUTPUT_FILE_NAME)

    storm_object_table = _create_tracking_data()

    _plot_schema(storm_object_table=storm_object_table,
                 output_file_name=OUTPUT_FILE_NAME)


if __name__ == '__main__':
    _run()
