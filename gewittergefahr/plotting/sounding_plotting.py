"""Plotting methods for atmospheric soundings."""

import numpy
from skewt import SkewT

DEFAULT_LINE_COLOUR = numpy.array([228., 26., 28.]) / 255
DEFAULT_LINE_WIDTH = 2


def plot_sounding(sounding_dict_for_skewt, line_colour=DEFAULT_LINE_COLOUR,
                  line_width=DEFAULT_LINE_WIDTH):
    """Plots atmospheric sounding.

    :param sounding_dict_for_skewt: Dictionary with the following keys.
    sounding_dict_for_skewt.PRES: length-P numpy array of pressures (millibars).
    sounding_dict_for_skewt.TEMP: length-P numpy array of temperatures
        (Celsius).
    sounding_dict_for_skewt.DWPT: length-P numpy array of dewpoints (Celsius).
    sounding_dict_for_skewt.SKNT: length-P numpy array of wind speeds (knots).
    sounding_dict_for_skewt.WDIR: length-P numpy array of wind directions
        (direction of origin, as per meteorological convention) (degrees).

    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param line_colour: Colour for dewpoint and temperature traces (in any
        format accepted by `matplotlib.colors`).
    :param line_width: Line width for dewpoint and temperature traces (real
        positive number).
    """

    sounding_object = SkewT.Sounding(data=sounding_dict_for_skewt)
    sounding_object.plot_skewt(color=line_colour, lw=line_width, title='')
