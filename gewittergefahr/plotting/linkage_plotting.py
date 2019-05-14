"""Plots linkages between hazardous events and storms."""

import numpy
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import linkage
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import storm_plotting
from gewittergefahr.plotting import wind_plotting

TORNADO_TIME_FORMAT = '%H%M %b %d'
TORNADO_TIME_FONT_SIZE = 20
LATLNG_TOLERANCE_DEG = 1e-4

DEFAULT_TORNADO_MARKER_TYPE = 'o'
DEFAULT_TORNADO_MARKER_SIZE = 8
DEFAULT_TORNADO_MARKER_COLOUR = numpy.array([228., 26., 28.]) / 255


def _find_unique_events(
        event_latitudes_deg, event_longitudes_deg, event_times_unix_sec):
    """Finds unique events.

    N = original number of events
    n = number of unique events

    :param event_latitudes_deg: length-N numpy array of latitudes (deg N).
    :param event_longitudes_deg: length-N numpy array of longitudes (deg E).
    :param event_times_unix_sec: length-N numpy array of times.
    :return: unique_latitudes_deg: length-n numpy array of latitudes (deg N).
    :return: unique_longitudes_deg: length-n numpy array of longitudes (deg E).
    :return: unique_times_unix_sec: length-n numpy array of times.
    """

    event_latitudes_deg = number_rounding.round_to_nearest(
        event_latitudes_deg, LATLNG_TOLERANCE_DEG)
    event_longitudes_deg = number_rounding.round_to_nearest(
        event_longitudes_deg, LATLNG_TOLERANCE_DEG)

    coord_matrix = numpy.transpose(numpy.vstack((
        event_latitudes_deg, event_longitudes_deg, event_times_unix_sec
    )))

    _, unique_indices = numpy.unique(coord_matrix, return_index=True, axis=0)

    return (event_latitudes_deg[unique_indices],
            event_longitudes_deg[unique_indices],
            event_times_unix_sec[unique_indices])


def plot_wind_linkages(
        storm_to_winds_table, basemap_object, axes_object,
        storm_colour=storm_plotting.DEFAULT_TRACK_COLOUR,
        storm_line_width=storm_plotting.DEFAULT_TRACK_WIDTH,
        wind_barb_length=wind_plotting.DEFAULT_BARB_LENGTH,
        empty_wind_barb_radius=wind_plotting.DEFAULT_EMPTY_BARB_RADIUS,
        fill_empty_wind_barb=wind_plotting.FILL_EMPTY_BARB_DEFAULT,
        wind_colour_map=wind_plotting.DEFAULT_COLOUR_MAP,
        colour_minimum_kt=wind_plotting.DEFAULT_COLOUR_MINIMUM_KT,
        colour_maximum_kt=wind_plotting.DEFAULT_COLOUR_MAXIMUM_KT):
    """Plots wind linkages.

    :param storm_to_winds_table: pandas DataFrame with columns listed in
        `link_events_to_storms.write_storm_to_winds_table`.  This method will
        plot linkages for all storm cells in the table.
    :param basemap_object: Instance of `mpl_toolkits.basemap.SBasemap`.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param storm_colour: Colour for storm tracks and outlines (in any format
        accepted by `matplotlib.colors`).
    :param storm_line_width: Line width for storm tracks and outlines.
    :param wind_barb_length: See doc for `wind_plotting.plot_wind_barbs`.
    :param empty_wind_barb_radius: Same.
    :param fill_empty_wind_barb: Same.
    :param wind_colour_map: Same.
    :param colour_minimum_kt: Same.
    :param colour_maximum_kt: Same.
    """

    unique_primary_id_strings, object_to_cell_indices = numpy.unique(
        storm_to_winds_table[tracking_utils.PRIMARY_ID_COLUMN].values,
        return_inverse=True)

    num_storm_cells = len(unique_primary_id_strings)

    for i in range(num_storm_cells):
        these_object_indices = numpy.where(object_to_cell_indices == i)[0]

        storm_plotting.plot_storm_track(
            basemap_object=basemap_object, axes_object=axes_object,
            centroid_latitudes_deg=storm_to_winds_table[
                tracking_utils.CENTROID_LATITUDE_COLUMN
            ].values[these_object_indices],
            centroid_longitudes_deg=storm_to_winds_table[
                tracking_utils.CENTROID_LONGITUDE_COLUMN
            ].values[these_object_indices],
            line_colour=storm_colour, line_width=storm_line_width
        )

        these_wind_latitudes_deg = numpy.concatenate(tuple([
            storm_to_winds_table[linkage.EVENT_LATITUDES_COLUMN].values[j]
            for j in these_object_indices
        ]))

        these_wind_longitudes_deg = numpy.concatenate(tuple([
            storm_to_winds_table[linkage.EVENT_LONGITUDES_COLUMN].values[j]
            for j in these_object_indices
        ]))

        these_u_winds_m_s01 = numpy.concatenate(tuple([
            storm_to_winds_table[linkage.U_WINDS_COLUMN].values[j]
            for j in these_object_indices
        ]))

        these_v_winds_m_s01 = numpy.concatenate(tuple([
            storm_to_winds_table[linkage.V_WINDS_COLUMN].values[j]
            for j in these_object_indices
        ]))

        wind_plotting.plot_wind_barbs(
            basemap_object=basemap_object, axes_object=axes_object,
            latitudes_deg=these_wind_latitudes_deg,
            longitudes_deg=these_wind_longitudes_deg,
            u_winds_m_s01=these_u_winds_m_s01,
            v_winds_m_s01=these_v_winds_m_s01, barb_length=wind_barb_length,
            empty_barb_radius=empty_wind_barb_radius,
            fill_empty_barb=fill_empty_wind_barb, colour_map=wind_colour_map,
            colour_minimum_kt=colour_minimum_kt,
            colour_maximum_kt=colour_maximum_kt)


def plot_tornado_linkages(
        storm_to_tornadoes_table, basemap_object, axes_object,
        plot_times=False, storm_colour=storm_plotting.DEFAULT_TRACK_COLOUR,
        storm_line_width=storm_plotting.DEFAULT_TRACK_WIDTH,
        tornado_marker_type=DEFAULT_TORNADO_MARKER_TYPE,
        tornado_marker_size=DEFAULT_TORNADO_MARKER_SIZE,
        tornado_marker_colour=DEFAULT_TORNADO_MARKER_COLOUR):
    """Plots tornado linkages.

    :param storm_to_tornadoes_table: pandas DataFrame with columns listed in
        `link_events_to_storms.write_storm_to_tornadoes_table`.  This method
        will plot linkages for all storm cells in the table.
    :param basemap_object: See doc for `plot_wind_linkages`.
    :param axes_object: Same.
    :param plot_times: Boolean flag.  If True, will plot the time for each
        tornado (as a text string in the map).
    :param storm_colour: See doc for `plot_wind_linkages`.
    :param storm_line_width: Same.
    :param tornado_marker_type: Marker type for tornadoes (in any format
        accepted by `matplotlib.lines`).
    :param tornado_marker_size: Marker size for tornadoes.
    :param tornado_marker_colour: Marker colour for tornadoes (in any format
        accepted by `matplotlib.colors`).
    """

    error_checking.assert_is_boolean(plot_times)

    unique_primary_id_strings, object_to_cell_indices = numpy.unique(
        storm_to_tornadoes_table[tracking_utils.FULL_ID_COLUMN].values,
        return_inverse=True)

    num_storm_cells = len(unique_primary_id_strings)

    if tornado_marker_type == 'x':
        tornado_marker_edge_width = 2
    else:
        tornado_marker_edge_width = 1

    for i in range(num_storm_cells):
        these_object_indices = numpy.where(object_to_cell_indices == i)[0]
        storm_plotting.plot_storm_track(
            basemap_object=basemap_object, axes_object=axes_object,
            centroid_latitudes_deg=storm_to_tornadoes_table[
                tracking_utils.CENTROID_LATITUDE_COLUMN
            ].values[these_object_indices],
            centroid_longitudes_deg=storm_to_tornadoes_table[
                tracking_utils.CENTROID_LONGITUDE_COLUMN
            ].values[these_object_indices],
            line_colour=storm_colour, line_width=storm_line_width
        )

        these_tornado_latitudes_deg = numpy.concatenate(tuple(
            [storm_to_tornadoes_table[
                linkage.EVENT_LATITUDES_COLUMN
            ].values[j] for j in these_object_indices]
        ))

        this_num_tornadoes = len(these_tornado_latitudes_deg)
        if this_num_tornadoes == 0:
            continue

        these_tornado_longitudes_deg = numpy.concatenate(tuple([
            storm_to_tornadoes_table[linkage.EVENT_LONGITUDES_COLUMN].values[j]
            for j in these_object_indices
        ]))

        these_tornado_times_unix_sec = numpy.concatenate(tuple([
            storm_to_tornadoes_table[tracking_utils.VALID_TIME_COLUMN].values[j]
            +
            storm_to_tornadoes_table[
                linkage.RELATIVE_EVENT_TIMES_COLUMN
            ].values[j]
            for j in these_object_indices
        ]))

        (these_tornado_latitudes_deg, these_tornado_longitudes_deg,
         these_tornado_times_unix_sec
        ) = _find_unique_events(
            event_latitudes_deg=these_tornado_latitudes_deg,
            event_longitudes_deg=these_tornado_longitudes_deg,
            event_times_unix_sec=these_tornado_times_unix_sec)

        these_tornado_x_metres, these_tornado_y_metres = basemap_object(
            these_tornado_longitudes_deg, these_tornado_latitudes_deg)

        axes_object.plot(
            these_tornado_x_metres, these_tornado_y_metres, linestyle='None',
            marker=tornado_marker_type, markerfacecolor=tornado_marker_colour,
            markeredgecolor=tornado_marker_colour,
            markersize=tornado_marker_size,
            markeredgewidth=tornado_marker_edge_width)

        if not plot_times:
            return

        this_num_tornadoes = len(these_tornado_x_metres)

        for j in range(this_num_tornadoes):
            axes_object.text(
                these_tornado_x_metres[j], these_tornado_y_metres[j],
                time_conversion.unix_sec_to_string(
                    these_tornado_times_unix_sec[j], TORNADO_TIME_FORMAT),
                fontsize=TORNADO_TIME_FONT_SIZE, color=tornado_marker_colour,
                horizontalalignment='center', verticalalignment='top')
