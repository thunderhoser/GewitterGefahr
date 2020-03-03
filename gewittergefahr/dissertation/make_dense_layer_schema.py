"""Makes schema for dense layers."""

import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.plotting import imagemagick_utils

RADIANS_TO_DEGREES = 180. / numpy.pi

NUM_NEURONS_BY_LAYER = numpy.array([4, 3, 2], dtype=int)
MIN_X_COORD = 0.
MAX_X_COORD = 10.
MIN_Y_COORD = 0.
MAX_Y_COORD = 5.

NEURON_MARKER_TYPE = 'o'
NEURON_MARKER_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
NEURON_MARKER_SIZE = 32
NEURON_MARKER_EDGE_WIDTH = 4

LINE_WIDTH = 2
LINE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255

FIGURE_WIDTH_INCHES = FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 600
OUTPUT_FILE_NAME = (
    '/localdata/ryan.lagerquist/eager/dissertation_figures/'
    'dense_layer_schema.jpg'
)

FONT_SIZE = 30
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)


def _run():
    """Makes schema for dense layers.

    This is effectively the main method.
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    axes_object.axis('off')
    axes_object.set_aspect('equal')

    num_layers = len(NUM_NEURONS_BY_LAYER)
    x_coord_by_layer = numpy.linspace(MIN_X_COORD, MAX_X_COORD, num=num_layers)
    y_coords_by_layer = [None] * num_layers

    for k in range(num_layers):
        these_x_coords = numpy.full(
            NUM_NEURONS_BY_LAYER[k], x_coord_by_layer[k]
        )

        if NUM_NEURONS_BY_LAYER[k] == 1:
            y_coords_by_layer[k] = numpy.array([
                0.5 * (MIN_Y_COORD + MAX_Y_COORD)
            ])
        else:
            y_coords_by_layer[k] = numpy.linspace(
                MIN_Y_COORD, MAX_Y_COORD, num=NUM_NEURONS_BY_LAYER[k]
            )

        y_coords_by_layer[k] = y_coords_by_layer[k][::-1]

        axes_object.plot(
            these_x_coords, y_coords_by_layer[k], linestyle='None',
            marker=NEURON_MARKER_TYPE, markersize=NEURON_MARKER_SIZE,
            markerfacecolor=numpy.full(3, 1.),
            markeredgecolor=NEURON_MARKER_COLOUR,
            markeredgewidth=NEURON_MARKER_EDGE_WIDTH
        )

        if k == 0:
            this_label_string = 'Inputs'
        else:
            this_label_string = 'DL{0:d} outputs'.format(k)

        axes_object.text(
            x_coord_by_layer[k], numpy.min(y_coords_by_layer[k]) - 0.4,
            this_label_string, color=NEURON_MARKER_COLOUR, fontsize=FONT_SIZE,
            horizontalalignment='center', verticalalignment='top', rotation=90.
        )

    for k in range(num_layers - 1):
        this_first_num_neurons = NUM_NEURONS_BY_LAYER[k]
        this_second_num_neurons = NUM_NEURONS_BY_LAYER[k + 1]
        these_x_coords = x_coord_by_layer[k:(k + 2)]

        for i in range(this_first_num_neurons):
            if k == num_layers - 2:
                this_label_string = r'$x_{0:d}$'.format(i + 1)

                axes_object.text(
                    x_coord_by_layer[k] + 0.2, y_coords_by_layer[k][i],
                    this_label_string, color=NEURON_MARKER_COLOUR,
                    fontsize=FONT_SIZE, horizontalalignment='left',
                    verticalalignment='top'
                )

            for j in range(this_second_num_neurons):
                these_y_coords = numpy.array([
                    y_coords_by_layer[k][i], y_coords_by_layer[k + 1][j]
                ])

                axes_object.plot(
                    these_x_coords, these_y_coords, linestyle='solid',
                    linewidth=LINE_WIDTH, color=LINE_COLOUR)

                if not (j == 0 and k == num_layers - 2):
                    continue

                this_label_string = r'$y_{0:d}$'.format(j + 1)

                axes_object.text(
                    x_coord_by_layer[k + 1] + 0.2, y_coords_by_layer[k + 1][j],
                    this_label_string, color=NEURON_MARKER_COLOUR,
                    fontsize=FONT_SIZE, horizontalalignment='left',
                    verticalalignment='top'
                )

                this_label_string = r'$w_{{{0:d}1}} x_{0:d}$'.format(i + 1)
                this_rotation_deg = RADIANS_TO_DEGREES * numpy.arctan2(
                    numpy.diff(these_y_coords)[0], numpy.diff(these_x_coords)[0]
                )
                this_y_offset = 0.1 if this_rotation_deg == 0 else -0.1

                axes_object.text(
                    numpy.mean(these_x_coords),
                    numpy.mean(these_y_coords) + this_y_offset,
                    this_label_string, color=LINE_COLOUR, fontsize=FONT_SIZE,
                    horizontalalignment='center', verticalalignment='bottom',
                    rotation=this_rotation_deg
                )

    print('Saving figure to: "{0:s}"...'.format(OUTPUT_FILE_NAME))
    figure_object.savefig(
        OUTPUT_FILE_NAME, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    imagemagick_utils.trim_whitespace(input_file_name=OUTPUT_FILE_NAME,
                                      output_file_name=OUTPUT_FILE_NAME)


if __name__ == '__main__':
    _run()
