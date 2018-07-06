"""Methods for plotting storm outlines and storm tracks."""

import numpy
from descartes import PolygonPatch
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking

DEFAULT_TRACK_COLOUR = numpy.array([0., 0., 0.])
DEFAULT_TRACK_WIDTH = 2.
DEFAULT_TRACK_STYLE = 'solid'
DEFAULT_TRACK_START_MARKER = 'o'
DEFAULT_TRACK_END_MARKER = 'x'
DEFAULT_TRACK_START_MARKER_SIZE = 8
DEFAULT_TRACK_END_MARKER_SIZE = 12

DEFAULT_POLYGON_LINE_COLOUR = numpy.array([0., 0., 0.])
DEFAULT_POLYGON_LINE_WIDTH = 2
DEFAULT_POLYGON_HOLE_LINE_COLOUR = numpy.array([152., 152., 152.]) / 255
DEFAULT_POLYGON_HOLE_LINE_WIDTH = 1
DEFAULT_POLYGON_FILL_COLOUR = numpy.array([152., 152., 152.]) / 255
DEFAULT_POLYGON_FILL_OPACITY = 1.


def get_storm_track_colours():
    """Returns list of colours to use in plotting storm tracks.

    :return: rgb_matrix: 10-by-3 numpy array.  rgb_matrix[i, 0] is the red
        component of the [i]th colour; rgb_matrix[i, 1] is the green component
        of the [i]th colour; rgb_matrix[i, 2] is the blue component of the [i]th
        colour.
    """

    return numpy.array([[187, 255, 153],
                        [129, 243, 144],
                        [108, 232, 181],
                        [88, 213, 221],
                        [69, 137, 209],
                        [52, 55, 198],
                        [103, 37, 187],
                        [161, 23, 175],
                        [164, 10, 107],
                        [153, 0, 25]], dtype=float) / 255


def plot_storm_track(
        basemap_object, axes_object, centroid_latitudes_deg,
        centroid_longitudes_deg, line_colour=DEFAULT_TRACK_COLOUR,
        line_width=DEFAULT_TRACK_WIDTH, line_style=DEFAULT_TRACK_STYLE,
        start_marker=DEFAULT_TRACK_START_MARKER,
        end_marker=DEFAULT_TRACK_END_MARKER,
        start_marker_size=DEFAULT_TRACK_START_MARKER_SIZE,
        end_marker_size=DEFAULT_TRACK_END_MARKER_SIZE):
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


def plot_storm_outline_unfilled(
        basemap_object, axes_object, polygon_object_latlng,
        exterior_colour=DEFAULT_POLYGON_LINE_COLOUR,
        exterior_line_width=DEFAULT_POLYGON_LINE_WIDTH,
        hole_colour=DEFAULT_POLYGON_HOLE_LINE_COLOUR,
        hole_line_width=DEFAULT_POLYGON_HOLE_LINE_WIDTH):
    """Plots storm outline (or buffer around storm outline) as unfilled polygon.

    :param basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param polygon_object_latlng: `shapely.geometry.Polygon` object with
        vertices in lat-long coordinates.
    :param exterior_colour: Colour for exterior of polygon (in any format
        accepted by `matplotlib.colors`).
    :param exterior_line_width: Line width for exterior of polygon (real
        positive number).
    :param hole_colour: Colour for holes in polygon (in any format accepted by
        `matplotlib.colors`).
    :param hole_line_width: Line width for holes in polygon (real positive
        number).
    """

    vertex_dict = polygons.polygon_object_to_vertex_arrays(
        polygon_object_latlng)
    exterior_x_coords_metres, exterior_y_coords_metres = basemap_object(
        vertex_dict[polygons.EXTERIOR_X_COLUMN],
        vertex_dict[polygons.EXTERIOR_Y_COLUMN])

    axes_object.plot(
        exterior_x_coords_metres, exterior_y_coords_metres,
        color=exterior_colour, linestyle='solid', linewidth=exterior_line_width)

    num_holes = len(vertex_dict[polygons.HOLE_X_COLUMN])
    for i in range(num_holes):
        these_x_coords_metres, these_y_coords_metres = basemap_object(
            vertex_dict[polygons.HOLE_X_COLUMN][i],
            vertex_dict[polygons.HOLE_Y_COLUMN][i])

        axes_object.plot(
            these_x_coords_metres, these_y_coords_metres, color=hole_colour,
            linestyle='solid', linewidth=hole_line_width)


def plot_storm_outline_filled(
        basemap_object, axes_object, polygon_object_latlng,
        line_colour=DEFAULT_POLYGON_LINE_COLOUR,
        line_width=DEFAULT_POLYGON_LINE_WIDTH,
        fill_colour=DEFAULT_POLYGON_FILL_COLOUR,
        opacity=DEFAULT_POLYGON_FILL_OPACITY):
    """Plots storm outline (or buffer around storm outline) as filled polygon.

    :param basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param polygon_object_latlng: `shapely.geometry.Polygon` object with
        vertices in lat-long coordinates.
    :param line_colour: Colour of polygon edge (in any format accepted by
        `matplotlib.colors`).
    :param line_width: Width of polygon edge.
    :param fill_colour: Colour of polygon interior.
    :param opacity: Opacity of polygon fill (in range 0...1).
    """

    vertex_dict = polygons.polygon_object_to_vertex_arrays(
        polygon_object_latlng)
    exterior_x_coords_metres, exterior_y_coords_metres = basemap_object(
        vertex_dict[polygons.EXTERIOR_X_COLUMN],
        vertex_dict[polygons.EXTERIOR_Y_COLUMN])

    num_holes = len(vertex_dict[polygons.HOLE_X_COLUMN])
    x_coords_by_hole_metres = [None] * num_holes
    y_coords_by_hole_metres = [None] * num_holes

    for i in range(num_holes):
        x_coords_by_hole_metres[i], y_coords_by_hole_metres[i] = basemap_object(
            vertex_dict[polygons.HOLE_X_COLUMN][i],
            vertex_dict[polygons.HOLE_Y_COLUMN][i])

    polygon_object_xy = polygons.vertex_arrays_to_polygon_object(
        exterior_x_coords=exterior_x_coords_metres,
        exterior_y_coords=exterior_y_coords_metres,
        hole_x_coords_list=x_coords_by_hole_metres,
        hole_y_coords_list=y_coords_by_hole_metres)

    polygon_patch = PolygonPatch(
        polygon_object_xy, lw=line_width, ec=line_colour, fc=fill_colour,
        alpha=opacity)
    axes_object.add_patch(polygon_patch)
