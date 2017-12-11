"""Plotting methods for storm polygons and storm tracks."""

import numpy
from descartes import PolygonPatch
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking

# TODO(thunderhoser): Allow multiples of each item (storm tracks, unfilled
# polygons, filled polygons) to be plotted at once.

DEFAULT_TRACK_COLOUR = numpy.array([0., 0., 0.])
DEFAULT_TRACK_WIDTH = 2.
DEFAULT_TRACK_STYLE = 'solid'
DEFAULT_TRACK_START_MARKER = 'o'
DEFAULT_TRACK_END_MARKER = 'x'
DEFAULT_TRACK_START_MARKER_SIZE = 8
DEFAULT_TRACK_END_MARKER_SIZE = 12

DEFAULT_POLY_LINE_COLOUR = numpy.array([0., 0., 0.])
DEFAULT_POLY_LINE_WIDTH = 2
DEFAULT_POLY_HOLE_LINE_COLOUR = numpy.array([152., 152., 152.]) / 255
DEFAULT_POLY_HOLE_LINE_WIDTH = 1
DEFAULT_POLY_FILL_COLOUR = numpy.array([152., 152., 152.]) / 255


def plot_storm_track(basemap_object=None, axes_object=None, latitudes_deg=None,
                     longitudes_deg=None, line_colour=DEFAULT_TRACK_COLOUR,
                     line_width=DEFAULT_TRACK_WIDTH,
                     line_style=DEFAULT_TRACK_STYLE,
                     start_marker=DEFAULT_TRACK_START_MARKER,
                     end_marker=DEFAULT_TRACK_END_MARKER,
                     start_marker_size=DEFAULT_TRACK_START_MARKER_SIZE,
                     end_marker_size=DEFAULT_TRACK_END_MARKER_SIZE):
    """Plots storm track (path of storm centroid).

    P = number of points in track

    :param basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param latitudes_deg: length-P numpy array of latitudes (deg N).
    :param longitudes_deg: length-P numpy array of longitudes (deg E).
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

    error_checking.assert_is_valid_lat_numpy_array(latitudes_deg)
    error_checking.assert_is_numpy_array(latitudes_deg, num_dimensions=1)
    num_points = len(latitudes_deg)

    longitudes_deg = lng_conversion.convert_lng_positive_in_west(longitudes_deg)
    error_checking.assert_is_numpy_array(
        longitudes_deg, exact_dimensions=numpy.array([num_points]))

    x_coords_metres, y_coords_metres = basemap_object(
        longitudes_deg, latitudes_deg)
    axes_object.plot(
        x_coords_metres, y_coords_metres, color=line_colour,
        linestyle=line_style, linewidth=line_width)

    if start_marker is not None:
        if start_marker == 'x':
            start_marker_edge_width = 2
        else:
            start_marker_edge_width = 1

        axes_object.plot(
            x_coords_metres[0], y_coords_metres[0], linestyle='None',
            marker=start_marker, markerfacecolor=line_colour,
            markeredgecolor=line_colour, markersize=start_marker_size,
            markeredgewidth=start_marker_edge_width)

    if end_marker is not None:
        if end_marker == 'x':
            end_marker_edge_width = 2
        else:
            end_marker_edge_width = 1

        axes_object.plot(
            x_coords_metres[-1], y_coords_metres[-1], linestyle='None',
            marker=end_marker, markerfacecolor=line_colour,
            markeredgecolor=line_colour, markersize=end_marker_size,
            markeredgewidth=end_marker_edge_width)


def plot_unfilled_polygon(basemap_object=None, axes_object=None,
                          vertex_latitudes_deg=None, vertex_longitudes_deg=None,
                          exterior_colour=DEFAULT_POLY_LINE_COLOUR,
                          exterior_line_width=DEFAULT_POLY_LINE_WIDTH,
                          hole_colour=DEFAULT_POLY_HOLE_LINE_COLOUR,
                          hole_line_width=DEFAULT_POLY_HOLE_LINE_WIDTH):
    """Plots unfilled polygon (storm object or buffer around storm object).

    V = number of vertices in polygon

    :param basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param vertex_latitudes_deg: length-V numpy array of latitudes (deg N).
    :param vertex_longitudes_deg: length-V numpy array of longitudes (deg E).
    :param exterior_colour: Colour for exterior of polygon (in any format
        accepted by `matplotlib.colors`).
    :param exterior_line_width: Line width for exterior of polygon (real
        positive number).
    :param hole_colour: Colour for holes in polygon (in any format accepted by
        `matplotlib.colors`).
    :param hole_line_width: Line width for holes in polygon (real positive
        number).
    """

    # TODO(thunderhoser): input should be a `shapely.geometry.Polygon` object.

    vertex_dict = polygons.separate_exterior_and_holes(
        vertex_longitudes_deg, vertex_latitudes_deg)

    exterior_latitudes_deg = vertex_dict[polygons.EXTERIOR_Y_COLUMN]
    exterior_longitudes_deg = vertex_dict[polygons.EXTERIOR_X_COLUMN]
    hole_list_latitudes_deg = vertex_dict[polygons.HOLE_Y_COLUMN]
    hole_list_longitudes_deg = vertex_dict[polygons.HOLE_X_COLUMN]

    exterior_x_metres, exterior_y_metres = basemap_object(
        exterior_longitudes_deg, exterior_latitudes_deg)
    axes_object.plot(
        exterior_x_metres, exterior_y_metres, color=exterior_colour,
        linestyle='solid', linewidth=exterior_line_width)

    num_holes = len(hole_list_latitudes_deg)
    for i in range(num_holes):
        these_x_metres, these_y_metres = basemap_object(
            hole_list_longitudes_deg[i], hole_list_latitudes_deg[i])

        axes_object.plot(
            these_x_metres, these_y_metres, color=hole_colour,
            linestyle='solid', linewidth=hole_line_width)


def plot_filled_polygon(basemap_object=None, axes_object=None,
                        vertex_latitudes_deg=None, vertex_longitudes_deg=None,
                        line_colour=DEFAULT_POLY_LINE_COLOUR,
                        line_width=DEFAULT_POLY_LINE_WIDTH,
                        fill_colour=DEFAULT_POLY_FILL_COLOUR):
    """Plots filled polygon (either storm object or buffer around storm object).

    :param basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param vertex_latitudes_deg: length-V numpy array of latitudes (deg N).
    :param vertex_longitudes_deg: length-V numpy array of longitudes (deg E).
    :param line_colour: Colour of polygon edge (in any format accepted by
        `matplotlib.colors`).
    :param line_width: Width of polygon edge.
    :param fill_colour: Colour of polygon interior.
    """

    # TODO(thunderhoser): input should be a `shapely.geometry.Polygon` object.

    vertex_dict = polygons.separate_exterior_and_holes(
        vertex_longitudes_deg, vertex_latitudes_deg)
    (exterior_x_metres, exterior_y_metres) = basemap_object(
        vertex_dict[polygons.EXTERIOR_X_COLUMN],
        vertex_dict[polygons.EXTERIOR_Y_COLUMN])

    num_holes = len(vertex_dict[polygons.HOLE_X_COLUMN])
    hole_list_x_metres = [None] * num_holes
    hole_list_y_metres = [None] * num_holes

    for i in range(num_holes):
        (hole_list_x_metres[i], hole_list_y_metres[i]) = basemap_object(
            numpy.flipud(vertex_dict[polygons.HOLE_X_COLUMN][i]),
            numpy.flipud(vertex_dict[polygons.HOLE_Y_COLUMN][i]))

    polygon_object = polygons.vertex_arrays_to_polygon_object(
        exterior_x_metres, exterior_y_metres,
        hole_x_coords_list=hole_list_x_metres,
        hole_y_coords_list=hole_list_y_metres)

    polygon_patch = PolygonPatch(
        polygon_object, lw=line_width, ec=line_colour, fc=fill_colour)
    axes_object.add_patch(polygon_patch)
