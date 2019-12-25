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
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils

TOLERANCE = 1e-6
MIN_TRACK_LENGTH_DEG = numpy.sqrt(2) * 0.05

COLOUR_BAR_FONT_SIZE = 25
# COLOUR_BAR_TIME_FORMAT = '%H%M %-d %b'
COLOUR_BAR_TIME_FORMAT = '%H%M UTC'

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


def _lengthen_segment(latitudes_deg, longitudes_deg, basemap_object):
    """Lengthens line segment if necessary.

    If segment does not need to be lengthened, this method returns None for both
    output variables.

    :param latitudes_deg: length-2 numpy array of latitudes (deg N).
    :param longitudes_deg: length-2 numpy array of longitudes (deg E).
    :param basemap_object: See doc for `plot_storm_outlines`.
    :return: x_coords: length-2 numpy array of x-coordinates under basemap
        projection.
    :return: y_coords: Same but y-coordinates.
    """

    # TODO(thunderhoser): Needs unit test.

    latitude_diff_deg = numpy.absolute(numpy.diff(latitudes_deg))[0]
    longitude_diff_deg = numpy.absolute(numpy.diff(longitudes_deg))[0]
    same_points = (
        latitude_diff_deg < TOLERANCE and longitude_diff_deg < TOLERANCE
    )

    if same_points:
        latitudes_deg[1] = (
            latitudes_deg[0] + numpy.sqrt(0.5) * MIN_TRACK_LENGTH_DEG
        )
        longitudes_deg[1] = (
            longitudes_deg[0] + numpy.sqrt(0.5) * MIN_TRACK_LENGTH_DEG
        )

        return basemap_object(longitudes_deg, latitudes_deg)

    arc_length_deg = numpy.sum(numpy.sqrt(
        latitude_diff_deg ** 2 + longitude_diff_deg ** 2
    ))
    similar_points = arc_length_deg < MIN_TRACK_LENGTH_DEG

    if similar_points:
        bearing_radians = numpy.arctan2(longitude_diff_deg, latitude_diff_deg)

        latitudes_deg[1] = latitudes_deg[0] + (
            MIN_TRACK_LENGTH_DEG * numpy.cos(bearing_radians)
        )
        longitudes_deg[1] = longitudes_deg[0] + (
            MIN_TRACK_LENGTH_DEG * numpy.sin(bearing_radians)
        )

        return basemap_object(longitudes_deg, latitudes_deg)

    return None, None


def _plot_one_track_segment(
        storm_object_table_one_segment, axes_object, basemap_object, line_width,
        line_colour=None, colour_map_object=None, colour_norm_object=None):
    """Plots one track segment.

    :param storm_object_table_one_segment: Same as input for `plot_storm_tracks`,
        except that this table contains only two objects.
    :param axes_object: See doc for `plot_storm_outlines`.
    :param basemap_object: Same.
    :param line_width: Track width.
    :param line_colour: Track colour.  This may be None.
    :param colour_map_object: [used only if `line_colour is None`]:
        Colour scheme (instance of `matplotlib.pyplot.cm` or similar).
    :param colour_norm_object: Normalizer for colour scheme.
    """

    latitudes_deg = 0. + storm_object_table_one_segment[
        tracking_utils.CENTROID_LATITUDE_COLUMN
    ].values

    longitudes_deg = 0. + storm_object_table_one_segment[
        tracking_utils.CENTROID_LONGITUDE_COLUMN
    ].values

    x_coords, y_coords = _lengthen_segment(
        latitudes_deg=latitudes_deg, longitudes_deg=longitudes_deg,
        basemap_object=basemap_object
    )

    if x_coords is None:
        x_coords = storm_object_table_one_segment[
            tracking_utils.CENTROID_X_COLUMN
        ].values

        y_coords = storm_object_table_one_segment[
            tracking_utils.CENTROID_Y_COLUMN
        ].values

    if line_colour is None:
        point_matrix = numpy.array(
            [x_coords, y_coords]
        ).T.reshape(-1, 1, 2)

        segment_matrix = numpy.concatenate(
            [point_matrix[:-1], point_matrix[1:]], axis=1
        )

        mean_time_unix_sec = numpy.mean(
            storm_object_table_one_segment[
                tracking_utils.VALID_TIME_COLUMN
            ].values
        )

        this_line_collection_object = LineCollection(
            segment_matrix, cmap=colour_map_object, norm=colour_norm_object
        )
        this_line_collection_object.set_array(numpy.array([mean_time_unix_sec]))

        this_line_collection_object.set_linewidth(line_width)
        axes_object.add_collection(this_line_collection_object)
    else:
        axes_object.plot(
            x_coords, y_coords,
            color=line_colour, linestyle='solid', linewidth=line_width
        )


def _plot_one_track(
        storm_object_table_one_track, axes_object, basemap_object, line_width,
        line_colour=None, colour_map_object=None, colour_norm_object=None):
    """Plots one storm track.

    :param storm_object_table_one_track: Same as input for `plot_storm_tracks`,
        except that this table contains only one track (primary storm ID).
    :param axes_object: See doc for `plot_storm_outlines`.
    :param basemap_object: Same.
    :param line_width: Track width.
    :param line_colour: Track colour.  This may be None.
    :param colour_map_object: [used only if `line_colour is None`]:
        Colour scheme (instance of `matplotlib.pyplot.cm` or similar).
    :param colour_norm_object: Normalizer for colour scheme.
    """

    num_storm_objects = len(storm_object_table_one_track.index)

    if num_storm_objects == 1:
        this_storm_object_table = storm_object_table_one_track.iloc[[0, 0]]

        _plot_one_track_segment(
            storm_object_table_one_segment=this_storm_object_table,
            axes_object=axes_object, basemap_object=basemap_object,
            line_width=line_width, line_colour=line_colour,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object)

        return

    latitudes_deg = 0. + storm_object_table_one_track[
        tracking_utils.CENTROID_LATITUDE_COLUMN
    ].values[[0, -1]]

    longitudes_deg = 0. + storm_object_table_one_track[
        tracking_utils.CENTROID_LONGITUDE_COLUMN
    ].values[[0, -1]]

    x_coords = _lengthen_segment(
        latitudes_deg=latitudes_deg, longitudes_deg=longitudes_deg,
        basemap_object=basemap_object
    )[0]

    if x_coords is not None:
        this_storm_object_table = storm_object_table_one_track.iloc[[0, -1]]

        _plot_one_track_segment(
            storm_object_table_one_segment=this_storm_object_table,
            axes_object=axes_object, basemap_object=basemap_object,
            line_width=line_width, line_colour=line_colour,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object)

        return

    for i in range(num_storm_objects):
        successor_rows = temporal_tracking.find_immediate_successors(
            storm_object_table=storm_object_table_one_track, target_row=i
        )

        for j in successor_rows:
            this_storm_object_table = storm_object_table_one_track.iloc[[i, j]]

            _plot_one_track_segment(
                storm_object_table_one_segment=this_storm_object_table,
                axes_object=axes_object, basemap_object=basemap_object,
                line_width=line_width, line_colour=line_colour,
                colour_map_object=colour_map_object,
                colour_norm_object=colour_norm_object)


def _plot_start_or_end_markers(
        storm_object_table_one_track, axes_object, plot_at_start, marker_type,
        marker_size, track_colour=None, colour_map_object=None,
        colour_norm_object=None):
    """Plots marker at beginning or end of each storm track.

    :param storm_object_table_one_track: Same as input for `plot_storm_tracks`,
        except that this table contains only one track (primary storm ID).
    :param axes_object: See doc for `plot_storm_outlines`.
    :param plot_at_start: Boolean flag.  If True, will plot at beginning of each
        track.  If False, at end of each track.
    :param marker_type: Marker type.
    :param marker_size: Marker size.
    :param track_colour: Track colour.  This may be None.
    :param colour_map_object: [used only if `track_colour is None`]:
        Colour scheme (instance of `matplotlib.pyplot.cm` or similar).
    :param colour_norm_object: Normalizer for colour scheme.
    """

    num_storm_objects = len(storm_object_table_one_track.index)

    for i in range(num_storm_objects):
        if plot_at_start:
            these_indices = temporal_tracking.find_immediate_predecessors(
                storm_object_table=storm_object_table_one_track, target_row=i
            )
        else:
            these_indices = temporal_tracking.find_immediate_successors(
                storm_object_table=storm_object_table_one_track, target_row=i
            )

        if len(these_indices) > 0:
            continue

        if colour_map_object is None:
            this_colour = track_colour
        else:
            this_time_unix_sec = storm_object_table_one_track[
                tracking_utils.VALID_TIME_COLUMN
            ].values[i]

            this_colour = colour_map_object(colour_norm_object(
                this_time_unix_sec
            ))

        this_x_coord = storm_object_table_one_track[
            tracking_utils.CENTROID_X_COLUMN
        ].values[i]

        this_y_coord = storm_object_table_one_track[
            tracking_utils.CENTROID_Y_COLUMN
        ].values[i]

        axes_object.plot(
            this_x_coord, this_y_coord, linestyle='None',
            marker=marker_type, markersize=marker_size,
            markerfacecolor=this_colour, markeredgecolor=this_colour,
            markeredgewidth=2 if marker_type == 'x' else 1
        )


def _process_colour_args(
        colour_map_object, colour_min_unix_sec, colour_max_unix_sec,
        constant_colour, storm_object_table):
    """Processes colour arguments.

    If colour_map_object is None, all points will be the same colour, specified
    by constant_colour.

    If colour_map_object == "random", points will be coloured randomly.

    If colour_map_object is an actual colour map, points will be coloured by
    time.

    :param colour_map_object: See general discussion above.
    :param colour_min_unix_sec: [used only if `colour_map_object is not None`]
        First time in colour scheme.
    :param colour_max_unix_sec: [used only if `colour_map_object is not None`]
        Last time in colour scheme.
    :param constant_colour: [used only if `colour_map_object is None`]
        Constant colour (length-3 numpy array).
    :param storm_object_table: pandas DataFrame with columns documented in
        `storm_tracking_io.write_file`.  Used only if
        `colour_map_object is not None` but either `colour_min_unix_sec is None`
        or `colour_max_unix_sec is None`, to determine min and max times in
        colour scheme.
    :return: rgb_matrix: C-by-3 numpy array of RGB values, where each row is one
        colour.  If colour_map_object != "random", this is None.
    :return: colour_map_object: Colour scheme.  If colours are random or
        constant, this is None.  Otherwise, this is an instance of
        `matplotlib.pyplot.cm`.
    :return: colour_norm_object: If `colour_map_object is None`, this is None.
        Otherwise, this is an instance of `matplotlib.colors.Normalize`.
    """

    rgb_matrix = None
    colour_norm_object = None

    if colour_map_object is None:
        expected_dim = numpy.array([3], dtype=int)
        error_checking.assert_is_numpy_array(
            constant_colour, exact_dimensions=expected_dim
        )

        rgb_matrix = numpy.reshape(constant_colour, (1, 3))
    elif colour_map_object == 'random':
        rgb_matrix = get_storm_track_colours()
        colour_map_object = None
    else:
        if colour_min_unix_sec is None or colour_max_unix_sec is None:
            colour_min_unix_sec = numpy.min(
                storm_object_table[tracking_utils.VALID_TIME_COLUMN].values
            )
            colour_max_unix_sec = numpy.max(
                storm_object_table[tracking_utils.VALID_TIME_COLUMN].values
            )

        colour_norm_object = pyplot.Normalize(
            colour_min_unix_sec, colour_max_unix_sec
        )

    return rgb_matrix, colour_map_object, colour_norm_object


def _add_colour_bar(axes_object, basemap_object, colour_map_object,
                    colour_norm_object):
    """Adds colour bar to figure.

    :param axes_object: See input doc for `plot_storm_tracks`.
    :param basemap_object: Same.
    :param colour_map_object: See output doc for `_process_colour_args`.
    :param colour_norm_object: Same.
    :return: colour_bar_object: Handle for colour bar.
    """

    latitude_range_deg = basemap_object.urcrnrlat - basemap_object.llcrnrlat
    longitude_range_deg = basemap_object.urcrnrlon - basemap_object.llcrnrlon

    if latitude_range_deg > longitude_range_deg:
        orientation_string = 'vertical'
        padding = None
    else:
        orientation_string = 'horizontal'
        padding = 0.05

    dummy_values = numpy.array([0, 1e12], dtype=int)

    colour_bar_object = plotting_utils.plot_linear_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=dummy_values,
        colour_map_object=colour_map_object,
        min_value=colour_norm_object.vmin, max_value=colour_norm_object.vmax,
        orientation_string=orientation_string, padding=padding,
        extend_min=False, extend_max=False, fraction_of_axis_length=0.9,
        font_size=COLOUR_BAR_FONT_SIZE
    )

    tick_times_unix_sec = numpy.round(
        colour_bar_object.get_ticks()
    ).astype(int)

    tick_time_strings = [
        time_conversion.unix_sec_to_string(t, COLOUR_BAR_TIME_FORMAT)
        for t in tick_times_unix_sec
    ]

    colour_bar_object.set_ticks(tick_times_unix_sec)
    colour_bar_object.set_ticklabels(tick_time_strings)
    return colour_bar_object


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


def plot_storm_outlines(
        storm_object_table, axes_object, basemap_object,
        line_width=DEFAULT_POLYGON_WIDTH, line_colour=DEFAULT_POLYGON_COLOUR,
        line_style='solid'):
    """Plots all storm objects in the table (as unfilled polygons).

    :param storm_object_table: See doc for `storm_tracking_io.write_file`.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param basemap_object: Will use this object (instance of
        `mpl_toolkits.basemap.Basemap`) to convert between x-y and lat-long
        coords.
    :param line_width: Width of each polygon.
    :param line_colour: Colour of each polygon.
    :param line_style: Line style for each polygon.
    """

    line_colour_tuple = plotting_utils.colour_from_numpy_to_tuple(line_colour)
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
            these_x_coords_metres, these_y_coords_metres,
            color=line_colour_tuple, linestyle=line_style, linewidth=line_width
        )


def plot_storm_centroids(
        storm_object_table, axes_object, basemap_object,
        colour_map_object='random',
        colour_min_unix_sec=None, colour_max_unix_sec=None,
        constant_colour=DEFAULT_CENTROID_COLOUR,
        marker_type=DEFAULT_CENTROID_MARKER_TYPE,
        marker_size=DEFAULT_CENTROID_MARKER_SIZE):
    """Plots storm centroids.

    :param storm_object_table: See doc for `plot_storm_tracks`.
    :param axes_object: Same.
    :param basemap_object: Same.
    :param colour_map_object: See doc for `_process_colour_args`.
    :param colour_min_unix_sec: Same.
    :param colour_max_unix_sec: Same.
    :param constant_colour: Same.
    :param marker_type: Marker type.
    :param marker_size: Marker size.
    :return: colour_bar_object: Handle for colour bar.  If
        `colour_map_object is None`, this will also be None.
    """

    x_coords_metres, y_coords_metres = basemap_object(
        storm_object_table[tracking_utils.CENTROID_LONGITUDE_COLUMN].values,
        storm_object_table[tracking_utils.CENTROID_LATITUDE_COLUMN].values
    )

    storm_object_table = storm_object_table.assign(**{
        tracking_utils.CENTROID_X_COLUMN: x_coords_metres,
        tracking_utils.CENTROID_Y_COLUMN: y_coords_metres
    })

    rgb_matrix, colour_map_object, colour_norm_object = _process_colour_args(
        colour_map_object=colour_map_object,
        colour_min_unix_sec=colour_min_unix_sec,
        colour_max_unix_sec=colour_max_unix_sec,
        constant_colour=constant_colour,
        storm_object_table=storm_object_table
    )

    num_colours = None if rgb_matrix is None else rgb_matrix.shape[0]

    track_primary_id_strings, object_to_track_indices = numpy.unique(
        storm_object_table[tracking_utils.PRIMARY_ID_COLUMN].values,
        return_inverse=True
    )

    num_tracks = len(track_primary_id_strings)

    for j in range(num_tracks):
        these_object_indices = numpy.where(object_to_track_indices == j)[0]

        if colour_map_object is None:
            this_colour = rgb_matrix[numpy.mod(j, num_colours), :]
            this_colour = plotting_utils.colour_from_numpy_to_tuple(this_colour)

            axes_object.plot(
                x_coords_metres[these_object_indices],
                y_coords_metres[these_object_indices],
                linestyle='None', marker=marker_type,
                markersize=marker_size, markeredgewidth=0,
                markerfacecolor=this_colour, markeredgecolor=this_colour
            )

            continue

        for i in these_object_indices:
            this_colour = colour_map_object(colour_norm_object(
                storm_object_table[tracking_utils.VALID_TIME_COLUMN].values[i]
            ))
            this_colour = plotting_utils.colour_from_numpy_to_tuple(this_colour)

            axes_object.plot(
                x_coords_metres[i], y_coords_metres[i], linestyle='None',
                marker=marker_type, markersize=marker_size, markeredgewidth=0,
                markerfacecolor=this_colour, markeredgecolor=this_colour
            )

    if colour_map_object is None:
        return None

    return _add_colour_bar(
        axes_object=axes_object,
        basemap_object=basemap_object,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object)


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

    font_colour_tuple = plotting_utils.colour_from_numpy_to_tuple(font_colour)
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
            fontsize=font_size, fontweight='bold', color=font_colour_tuple,
            horizontalalignment='left', verticalalignment='top')


def plot_storm_tracks(
        storm_object_table, axes_object, basemap_object,
        colour_map_object='random',
        colour_min_unix_sec=None, colour_max_unix_sec=None,
        constant_colour=DEFAULT_TRACK_COLOUR, line_width=DEFAULT_TRACK_WIDTH,
        start_marker_type=DEFAULT_START_MARKER_TYPE,
        end_marker_type=DEFAULT_END_MARKER_TYPE,
        start_marker_size=DEFAULT_START_MARKER_SIZE,
        end_marker_size=DEFAULT_END_MARKER_SIZE):
    """Plots storm tracks.

    :param storm_object_table: See doc for `plot_storm_outlines`.
    :param axes_object: Same.
    :param basemap_object: Same.
    :param colour_map_object: See doc for `_process_colour_args`.
    :param colour_min_unix_sec: Same.
    :param colour_max_unix_sec: Same.
    :param constant_colour: Same.
    :param line_width: Track width.
    :param start_marker_type: Marker for beginning of each track.  If None,
        markers will not be plotted for beginning of track.
    :param end_marker_type: Same but for end of track.
    :param start_marker_size: Size of start-point marker.
    :param end_marker_size: Size of end-point marker.
    :return: colour_bar_object: Handle for colour bar.  If
        `colour_map_object is None`, this will also be None.
    """

    x_coords_metres, y_coords_metres = basemap_object(
        storm_object_table[tracking_utils.CENTROID_LONGITUDE_COLUMN].values,
        storm_object_table[tracking_utils.CENTROID_LATITUDE_COLUMN].values
    )

    storm_object_table = storm_object_table.assign(**{
        tracking_utils.CENTROID_X_COLUMN: x_coords_metres,
        tracking_utils.CENTROID_Y_COLUMN: y_coords_metres
    })

    rgb_matrix, colour_map_object, colour_norm_object = _process_colour_args(
        colour_map_object=colour_map_object,
        colour_min_unix_sec=colour_min_unix_sec,
        colour_max_unix_sec=colour_max_unix_sec,
        constant_colour=constant_colour,
        storm_object_table=storm_object_table
    )

    num_colours = None if rgb_matrix is None else rgb_matrix.shape[0]

    track_primary_id_strings, object_to_track_indices = numpy.unique(
        storm_object_table[tracking_utils.PRIMARY_ID_COLUMN].values,
        return_inverse=True
    )

    num_tracks = len(track_primary_id_strings)

    for j in range(num_tracks):
        if colour_map_object is None:
            this_colour = rgb_matrix[numpy.mod(j, num_colours), :]
            this_colour = plotting_utils.colour_from_numpy_to_tuple(this_colour)
        else:
            this_colour = None

        these_object_indices = numpy.where(object_to_track_indices == j)[0]
        storm_object_table_one_track = (
            storm_object_table.iloc[these_object_indices]
        )

        _plot_one_track(
            storm_object_table_one_track=storm_object_table_one_track,
            axes_object=axes_object, basemap_object=basemap_object,
            line_width=line_width, line_colour=this_colour,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object)

        if start_marker_type is not None:
            _plot_start_or_end_markers(
                storm_object_table_one_track=storm_object_table_one_track,
                axes_object=axes_object, plot_at_start=True,
                marker_type=start_marker_type, marker_size=start_marker_size,
                track_colour=this_colour,
                colour_map_object=colour_map_object,
                colour_norm_object=colour_norm_object)

        if end_marker_type is not None:
            _plot_start_or_end_markers(
                storm_object_table_one_track=storm_object_table_one_track,
                axes_object=axes_object, plot_at_start=False,
                marker_type=end_marker_type, marker_size=end_marker_size,
                track_colour=this_colour,
                colour_map_object=colour_map_object,
                colour_norm_object=colour_norm_object)

    if colour_map_object is None:
        return None

    return _add_colour_bar(
        axes_object=axes_object,
        basemap_object=basemap_object,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object)
