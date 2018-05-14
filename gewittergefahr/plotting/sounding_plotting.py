"""Plotting methods for atmospheric soundings."""

import numpy
from skewt import SkewT
from gewittergefahr.gg_io import raw_wind_io
from gewittergefahr.gg_utils import soundings

# Column names in skewt package.
PRESSURE_COLUMN_SKEWT = 'pres'
TEMPERATURE_COLUMN_SKEWT = 'temp'
DEWPOINT_COLUMN_SKEWT = 'dwpt'
WIND_SPEED_COLUMN_SKEWT = 'sknt'
WIND_DIRECTION_COLUMN_SKEWT = 'drct'

# Column names in GewitterGefahr.
PRESSURE_COLUMN = soundings.PRESSURE_COLUMN_IN_SHARPPY_SOUNDING
TEMPERATURE_COLUMN = soundings.TEMPERATURE_COLUMN_IN_SHARPPY_SOUNDING
DEWPOINT_COLUMN = soundings.DEWPOINT_COLUMN_IN_SHARPPY_SOUNDING
U_WIND_COLUMN = soundings.U_WIND_COLUMN_IN_SHARPPY_SOUNDING
V_WIND_COLUMN = soundings.V_WIND_COLUMN_IN_SHARPPY_SOUNDING

DEFAULT_LINE_COLOUR = numpy.array([228., 26., 28.]) / 255
DEFAULT_LINE_WIDTH = 2


def plot_sounding(sounding_dict, line_colour=DEFAULT_LINE_COLOUR,
                  line_width=DEFAULT_LINE_WIDTH):
    """Plots atmospheric sounding.

    :param sounding_dict: See documentation for output of
        sounding_table_to_skewt.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param line_colour: Colour for dewpoint and temperature traces (in any
        format accepted by `matplotlib.colors`).
    :param line_width: Line width for dewpoint and temperature traces (real
        positive number).
    """

    sounding_object = SkewT.Sounding(data=sounding_dict)
    sounding_object.plot_skewt(color=line_colour, lw=line_width, title='')


def sounding_table_to_skewt(sounding_table):
    """Converts sounding from pandas DataFrame to format used in skewt package.

    P = number of pressure levels

    :param sounding_table: pandas DataFrame with P rows and the following
        columns.
    sounding_table.pressure_mb: Pressure (millibars).
    sounding_table.temperature_deg_c: Temperature (Celsius).
    sounding_table.dewpoint_deg_c: Dewpoint (Celsius).
    sounding_table.u_wind_kt: u-wind (knots).
    sounding_table.v_wind_kt: v-wind (knots).

    :return: sounding_dict_for_skewt: Dictionary with the following keys.
    sounding_dict_for_skewt.PRES: length-P numpy array of pressures (millibars).
    sounding_dict_for_skewt.TEMP: length-P numpy array of temperatures
        (Celsius).
    sounding_dict_for_skewt.DWPT: length-P numpy array of dewpoints (Celsius).
    sounding_dict_for_skewt.SKNT: length-P numpy array of wind speeds (knots).
    sounding_dict_for_skewt.WDIR: length-P numpy array of wind directions
        (direction of origin, as per meteorological convention) (degrees).
    """

    wind_speeds_kt, wind_directions_deg = raw_wind_io.uv_to_speed_and_direction(
        sounding_table[U_WIND_COLUMN].values,
        sounding_table[V_WIND_COLUMN].values)

    return {
        PRESSURE_COLUMN_SKEWT: sounding_table[PRESSURE_COLUMN].values,
        TEMPERATURE_COLUMN_SKEWT: sounding_table[TEMPERATURE_COLUMN].values,
        DEWPOINT_COLUMN_SKEWT: sounding_table[DEWPOINT_COLUMN].values,
        WIND_SPEED_COLUMN_SKEWT: wind_speeds_kt,
        WIND_DIRECTION_COLUMN_SKEWT: wind_directions_deg
    }
