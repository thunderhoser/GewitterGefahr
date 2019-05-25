"""Methods for plotting storm outlines and storm tracks."""

import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
import matplotlib.pyplot as pyplot
from matplotlib.collections import LineCollection
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import temporal_tracking
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils

COLOUR_BAR_FONT_SIZE = 25
COLOUR_BAR_TIME_FORMAT = '%H%M %-d %b'

DEFAULT_TRACK_COLOUR = numpy.full(3, 0.)
DEFAULT_TRACK_WIDTH = 2
DEFAULT_TRACK_STYLE = 'solid'

DEFAULT_START_MARKER_TYPE = 'o'
DEFAULT_END_MARKER_TYPE = 'x'
DEFAULT_CENTROID_MARKER_TYPE = 'o'
DEFAULT_START_MARKER_SIZE = 8
DEFAULT_END_MARKER_SIZE = 12
DEFAULT_CENTROID_MARKER_SIZE = 6

DEFAULT_POLYGON_WIDTH = 2

DEFAULT_FONT_SIZE = 12
DEFAULT_FONT_COLOUR = numpy.full(3, 0.)
DEFAULT_CENTROID_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255
DEFAULT_POLYGON_COLOUR = matplotlib.colors.to_rgba(DEFAULT_CENTROID_COLOUR, 0.5)


def get_storm_track_colours():
    """Returns list of colours to use in plotting storm tracks.

    :return: rgb_matrix: 10-by-3 numpy array.  rgb_matrix[i, 0] is the red
        component of the [i]th colour; rgb_matrix[i, 1] is the green component
        of the [i]th colour; rgb_matrix[i, 2] is the blue component of the [i]th
        colour.
    """

    return numpy.array([
        [187, 255, 153],
        [129, 243, 144],
        [108, 232, 181],
        [88, 213, 221],
        [69, 137, 209],
        [52, 55, 198],
        [103, 37, 187],
        [161, 23, 175],
        [164, 10, 107],
        [153, 0, 25]
    ], dtype=float) / 255


def plot_storm_track(
        basemap_object, axes_object, centroid_latitudes_deg,
        centroid_longitudes_deg, line_colour=DEFAULT_TRACK_COLOUR,
        line_width=DEFAULT_TRACK_WIDTH, line_style=DEFAULT_TRACK_STYLE,
        start_marker=DEFAULT_START_MARKER_TYPE,
        end_marker=DEFAULT_END_MARKER_TYPE,
        start_marker_size=DEFAULT_START_MARKER_SIZE,
        end_marker_size=DEFAULT_END_MARKER_SIZE):
    """Plots storm track (path of centroid).

    P = number of points in track

    :param basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param centroid_latitudes_deg: length-P numpy array with latitudes (deg N)
        of storm centroid.
    :param centroid_longitudes_deg: length-P numpy array with longitudes (deg E)
        of storm centroid.
    :param line_colour: Colour (in any format accepted by `matplotlib.colors`).
    :param line_width: Line width (real positive number).
    :param line_style: Line style (in any format accepted by
        `matplotlib.lines`).
    :param start_marker: Marker type for beginning of track (in any format
        accepted by `matplotlib.lines`).  This may also be None.
    :param end_marker: Marker type for end of track (in any format accepted by
        `matplotlib.lines`).  This may also be None.
    :param start_marker_size: Size of marker at beginning of track.
    :param end_marker_size: Size of marker at end of track.
    """

    # TODO(thunderhoser): Get rid of this method (or clean it up a lot).

    error_checking.assert_is_valid_lat_numpy_array(centroid_latitudes_deg)
    error_checking.assert_is_numpy_array(
        centroid_latitudes_deg, num_dimensions=1)
    num_points = len(centroid_latitudes_deg)

    centroid_longitudes_deg = lng_conversion.convert_lng_positive_in_west(
        centroid_longitudes_deg)
    error_checking.assert_is_numpy_array(
        centroid_longitudes_deg, exact_dimensions=numpy.array([num_points]))

    centroid_x_coords_metres, centroid_y_coords_metres = basemap_object(
        centroid_longitudes_deg, centroid_latitudes_deg)
    axes_object.plot(
        centroid_x_coords_metres, centroid_y_coords_metres, color=line_colour,
        linestyle=line_style, linewidth=line_width)

    if start_marker is not None:
        if start_marker == 'x':
            this_edge_width = 2
        else:
            this_edge_width = 1

        axes_object.plot(
            centroid_x_coords_metres[0], centroid_y_coords_metres[0],
            linestyle='None', marker=start_marker, markerfacecolor=line_colour,
            markeredgecolor=line_colour, markersize=start_marker_size,
            markeredgewidth=this_edge_width)

    if end_marker is not None:
        if end_marker == 'x':
            this_edge_width = 2
        else:
            this_edge_width = 1

        axes_object.plot(
            centroid_x_coords_metres[-1], centroid_y_coords_metres[-1],
            linestyle='None', marker=end_marker, markerfacecolor=line_colour,
            markeredgecolor=line_colour, markersize=end_marker_size,
            markeredgewidth=this_edge_width)


def plot_storm_outlines(
        storm_object_table, axes_object, basemap_object,
        line_width=DEFAULT_POLYGON_WIDTH,
        line_colour=DEFAULT_POLYGON_COLOUR):
    """Plots all storm objects in the table (as unfilled polygons).

    :param storm_object_table: See doc for `storm_tracking_io.write_file`.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param basemap_object: Will use this object (instance of
        `mpl_toolkits.basemap.Basemap`) to convert between x-y and lat-long
        coords.
    :param line_width: Width of each polygon.
    :param line_colour: Colour of each polygon.
    """

    num_storm_objects = len(storm_object_table.index)

    for i in range(num_storm_objects):
        this_vertex_dict_latlng = polygons.polygon_object_to_vertex_arrays(
            storm_object_table[tracking_utils.LATLNG_POLYGON_COLUMN].values[i]
        )

        these_x_coords_metres, these_y_coords_metres = basemap_object(
            this_vertex_dict_latlng[polygons.EXTERIOR_X_COLUMN],
            this_vertex_dict_latlng[polygons.EXTERIOR_Y_COLUMN]
        )

        axes_object.plot(
            these_x_coords_metres, these_y_coords_metres, color=line_colour,
            linestyle='solid', linewidth=line_width)


def plot_storm_centroids(
        storm_object_table, axes_object, basemap_object,
        marker_type=DEFAULT_CENTROID_MARKER_TYPE,
        marker_colour=DEFAULT_CENTROID_COLOUR,
        marker_size=DEFAULT_CENTROID_MARKER_SIZE):
    """Plots all storm centroids in the table (as markers).

    :param storm_object_table: See doc for `plot_storm_outlines`.
    :param axes_object: Same.
    :param basemap_object: Same.
    :param marker_type: Marker type for storm centroids (in any format accepted
        by `matplotlib.lines`).
    :param marker_colour: Colour for storm centroids.
    :param marker_size: Marker size for storm centroids.
    """

    x_coords_metres, y_coords_metres = basemap_object(
        storm_object_table[tracking_utils.CENTROID_LONGITUDE_COLUMN].values,
        storm_object_table[tracking_utils.CENTROID_LATITUDE_COLUMN].values
    )

    axes_object.plot(
        x_coords_metres, y_coords_metres, linestyle='None', marker=marker_type,
        markerfacecolor=marker_colour, markeredgecolor=marker_colour,
        markersize=marker_size, markeredgewidth=0)


def plot_storm_ids(
        storm_object_table, axes_object, basemap_object,
        plot_near_centroids=False, include_secondary_ids=False,
        font_colour=DEFAULT_FONT_COLOUR, font_size=DEFAULT_FONT_SIZE):
    """Plots storm IDs as text.

    :param storm_object_table: See doc for `plot_storm_outlines`.
    :param axes_object: Same.
    :param basemap_object: Same.
    :param plot_near_centroids: Boolean flag.  If True, will plot each ID near
        the storm centroid.  If False, will plot each ID near southeasternmost
        point in storm outline.
    :param include_secondary_ids: Boolean flag.  If True, will plot full IDs
        (primary_secondary).  If False, will plot only primary IDs.
    :param font_colour: Font colour.
    :param font_size: Font size.
    """

    error_checking.assert_is_boolean(plot_near_centroids)
    error_checking.assert_is_boolean(include_secondary_ids)

    num_storm_objects = len(storm_object_table.index)

    if plot_near_centroids:
        text_x_coords_metres, text_y_coords_metres = basemap_object(
            storm_object_table[tracking_utils.CENTROID_LONGITUDE_COLUMN].values,
            storm_object_table[tracking_utils.CENTROID_LATITUDE_COLUMN].values
        )
    else:
        text_x_coords_metres = numpy.full(num_storm_objects, numpy.nan)
        text_y_coords_metres = numpy.full(num_storm_objects, numpy.nan)

        for i in range(num_storm_objects):
            this_vertex_dict_latlng = polygons.polygon_object_to_vertex_arrays(
                storm_object_table[
                    tracking_utils.LATLNG_POLYGON_COLUMN].values[i]
            )

            these_x_metres, these_y_metres = basemap_object(
                this_vertex_dict_latlng[polygons.EXTERIOR_X_COLUMN],
                this_vertex_dict_latlng[polygons.EXTERIOR_Y_COLUMN]
            )

            this_index = numpy.argmax(these_x_metres - these_y_metres)
            text_x_coords_metres[i] = these_x_metres[this_index]
            text_y_coords_metres[i] = these_y_metres[this_index]

    for i in range(num_storm_objects):
        this_primary_id_string = storm_object_table[
            tracking_utils.PRIMARY_ID_COLUMN].values[i]

        try:
            this_primary_id_string = this_primary_id_string[-4:]
        except ValueError:
            pass

        if include_secondary_ids:
            this_secondary_id_string = storm_object_table[
                tracking_utils.SECONDARY_ID_COLUMN].values[i]

            try:
                this_secondary_id_string = this_secondary_id_string[-4:]
            except ValueError:
                pass

            this_label_string = '{0:s}_{1:s}'.format(
                this_primary_id_string, this_secondary_id_string)
        else:
            this_label_string = this_primary_id_string

        axes_object.text(
            text_x_coords_metres[i], text_y_coords_metres[i], this_label_string,
            fontsize=font_size, fontweight='bold', color=font_colour,
            horizontalalignment='left', verticalalignment='top')


def plot_storm_tracks(
        storm_object_table, axes_object, basemap_object,
        colour_map_object='random', line_colour=DEFAULT_TRACK_COLOUR,
        line_width=DEFAULT_TRACK_WIDTH,
        start_marker_type=DEFAULT_START_MARKER_TYPE,
        end_marker_type=DEFAULT_END_MARKER_TYPE,
        start_marker_size=DEFAULT_START_MARKER_SIZE,
        end_marker_size=DEFAULT_END_MARKER_SIZE):
    """Plots one or more storm tracks on the same map.

    :param storm_object_table: See doc for `plot_storm_outlines`.
    :param axes_object: Same.
    :param basemap_object: Same.
    :param colour_map_object: There are 3 cases.

    If "random", each track will be plotted in a random colour from
    `get_storm_track_colours`.

    If None, each track will be plotted in `line_colour` (the next input arg).

    If real colour map (instance of `matplotlib.pyplot.cm`), track segments will
    be coloured by time, according to this colour map.

    :param line_colour: [used only if `colour_map_object is None`]
        length-3 numpy array with (R, G, B).  Will be used for all tracks.
    :param line_width: Width of each storm track.
    :param start_marker_type: Marker type for beginning of track (in any format
        accepted by `matplotlib.lines`).  If `start_marker_type is None`,
        markers will not be used to show beginning of each track.
    :param end_marker_type: Same but for end of track.
    :param start_marker_size: Size of each start-point marker.
    :param end_marker_size: Size of each end-point marker.
    """

    plot_start_markers = start_marker_type is not None
    plot_end_markers = end_marker_type is not None

    if start_marker_type is None:
        start_marker_type = DEFAULT_START_MARKER_TYPE
        start_marker_size = DEFAULT_START_MARKER_SIZE

    if end_marker_type is None:
        end_marker_type = DEFAULT_END_MARKER_TYPE
        end_marker_size = DEFAULT_END_MARKER_SIZE

    x_coords_metres, y_coords_metres = basemap_object(
        storm_object_table[tracking_utils.CENTROID_LONGITUDE_COLUMN].values,
        storm_object_table[tracking_utils.CENTROID_LATITUDE_COLUMN].values
    )

    storm_object_table = storm_object_table.assign(**{
        tracking_utils.CENTROID_X_COLUMN: x_coords_metres,
        tracking_utils.CENTROID_Y_COLUMN: y_coords_metres
    })

    rgb_matrix = None
    num_colours = None
    colour_norm_object = None

    if colour_map_object is None:
        error_checking.assert_is_numpy_array(
            line_colour, exact_dimensions=numpy.array([3], dtype=int)
        )

        rgb_matrix = numpy.reshape(line_colour, (1, 3))
        num_colours = rgb_matrix.shape[0]

    elif colour_map_object == 'random':
        rgb_matrix = get_storm_track_colours()
        num_colours = rgb_matrix.shape[0]

        colour_map_object = None

    else:
        first_time_unix_sec = numpy.min(
            storm_object_table[tracking_utils.VALID_TIME_COLUMN].values
        )
        last_time_unix_sec = numpy.max(
            storm_object_table[tracking_utils.VALID_TIME_COLUMN].values
        )

        colour_norm_object = pyplot.Normalize(
            first_time_unix_sec, last_time_unix_sec)

    track_primary_id_strings, object_to_track_indices = numpy.unique(
        storm_object_table[tracking_utils.PRIMARY_ID_COLUMN].values,
        return_inverse=True)

    num_tracks = len(track_primary_id_strings)

    for k in range(num_tracks):
        if colour_map_object is None:
            this_colour = rgb_matrix[numpy.mod(k, num_colours), :]
        else:
            this_colour = None

        these_object_indices = numpy.where(object_to_track_indices == k)[0]

        for i in these_object_indices:
            these_next_indices = temporal_tracking.find_immediate_successors(
                storm_object_table=storm_object_table, target_row=i)

            # if len(these_next_indices) > 1:
            #     axes_object.text(
            #         storm_object_table[
            #             tracking_utils.CENTROID_X_COLUMN].values[i],
            #         storm_object_table[
            #             tracking_utils.CENTROID_Y_COLUMN].values[i],
            #         '{0:d}-WAY SPLIT'.format(len(these_next_indices)),
            #         fontsize=12, color='k',
            #         horizontalalignment='left', verticalalignment='top')

            for j in these_next_indices:
                these_x_coords_metres = storm_object_table[
                    tracking_utils.CENTROID_X_COLUMN
                ].values[[i, j]]

                these_y_coords_metres = storm_object_table[
                    tracking_utils.CENTROID_Y_COLUMN
                ].values[[i, j]]

                if colour_map_object is None:
                    axes_object.plot(
                        these_x_coords_metres, these_y_coords_metres,
                        color=this_colour, linestyle='solid',
                        linewidth=line_width)
                else:
                    this_point_matrix = numpy.array(
                        [these_x_coords_metres, these_y_coords_metres]
                    ).T.reshape(-1, 1, 2)

                    this_segment_matrix = numpy.concatenate(
                        [this_point_matrix[:-1], this_point_matrix[1:]],
                        axis=1
                    )

                    this_time_unix_sec = numpy.mean(
                        storm_object_table[
                            tracking_utils.VALID_TIME_COLUMN].values[[i, j]]
                    )

                    this_line_collection_object = LineCollection(
                        this_segment_matrix, cmap=colour_map_object,
                        norm=colour_norm_object)

                    this_line_collection_object.set_array(
                        numpy.array([this_time_unix_sec])
                    )
                    this_line_collection_object.set_linewidth(line_width)
                    axes_object.add_collection(this_line_collection_object)

            these_prev_indices = temporal_tracking.find_immediate_predecessors(
                storm_object_table=storm_object_table, target_row=i)

            # if len(these_prev_indices) > 1:
            #     axes_object.text(
            #         storm_object_table[
            #             tracking_utils.CENTROID_X_COLUMN].values[i],
            #         storm_object_table[
            #             tracking_utils.CENTROID_Y_COLUMN].values[i],
            #         '{0:d}-WAY MERGER'.format(len(these_prev_indices)),
            #         fontsize=12, color='k',
            #         horizontalalignment='left', verticalalignment='top')

            plot_this_start_marker = (
                (plot_start_markers and len(these_prev_indices) == 0)
                or len(these_object_indices) == 1
            )

            if plot_this_start_marker:
                if colour_map_object is not None:
                    this_colour = colour_map_object(colour_norm_object(
                        storm_object_table[
                            tracking_utils.VALID_TIME_COLUMN].values[i]
                    ))

                print(this_colour)

                if start_marker_type == 'x':
                    this_edge_width = 2
                else:
                    this_edge_width = 1

                print(storm_object_table[tracking_utils.CENTROID_X_COLUMN].values[i])
                print(storm_object_table[tracking_utils.CENTROID_Y_COLUMN].values[i])
                print(start_marker_type)
                print(start_marker_size)
                print(this_edge_width)

                axes_object.plot(
                    storm_object_table[
                        tracking_utils.CENTROID_X_COLUMN].values[i],
                    storm_object_table[
                        tracking_utils.CENTROID_Y_COLUMN].values[i],
                    linestyle='None', marker=start_marker_type,
                    markerfacecolor=this_colour, markeredgecolor=this_colour,
                    markersize=start_marker_size,
                    markeredgewidth=this_edge_width
                )

            plot_this_end_marker = (
                (plot_end_markers and len(these_next_indices) == 0)
                or len(these_object_indices) == 1
            )

            if plot_this_end_marker:
                if colour_map_object is not None:
                    this_colour = colour_map_object(colour_norm_object(
                        storm_object_table[
                            tracking_utils.VALID_TIME_COLUMN].values[i]
                    ))

                print(this_colour)

                if end_marker_type == 'x':
                    this_edge_width = 2
                else:
                    this_edge_width = 1

                axes_object.plot(
                    storm_object_table[
                        tracking_utils.CENTROID_X_COLUMN].values[i],
                    storm_object_table[
                        tracking_utils.CENTROID_Y_COLUMN].values[i],
                    linestyle='None', marker=end_marker_type,
                    markerfacecolor=this_colour, markeredgecolor=this_colour,
                    markersize=end_marker_size,
                    markeredgewidth=this_edge_width
                )

    if colour_map_object is None:
        return

    min_plot_latitude_deg = basemap_object.llcrnrlat
    max_plot_latitude_deg = basemap_object.urcrnrlat
    min_plot_longitude_deg = basemap_object.llcrnrlon
    max_plot_longitude_deg = basemap_object.urcrnrlon

    latitude_range_deg = max_plot_latitude_deg - min_plot_latitude_deg
    longitude_range_deg = max_plot_longitude_deg - min_plot_longitude_deg

    if latitude_range_deg > longitude_range_deg:
        orientation_string = 'vertical'
    else:
        orientation_string = 'horizontal'

    colour_bar_object = plotting_utils.add_linear_colour_bar(
        axes_object_or_list=axes_object,
        values_to_colour=storm_object_table[
            tracking_utils.VALID_TIME_COLUMN].values,
        colour_map=colour_map_object, colour_min=colour_norm_object.vmin,
        colour_max=colour_norm_object.vmax, orientation=orientation_string,
        extend_min=False, extend_max=False, fraction_of_axis_length=0.9,
        font_size=COLOUR_BAR_FONT_SIZE)

    for t in colour_bar_object.ax.get_yticklabels():
        print(t.get_text())

    tick_times_unix_sec = numpy.round(colour_bar_object.get_ticks()).astype(int)
    tick_time_strings = [
        time_conversion.unix_sec_to_string(t, COLOUR_BAR_TIME_FORMAT)
        for t in tick_times_unix_sec
    ]

    colour_bar_object.set_ticks(tick_times_unix_sec)
    colour_bar_object.set_ticklabels(tick_time_strings)
