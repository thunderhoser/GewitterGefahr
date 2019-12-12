"""Plots common activation functions."""

import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.deep_learning import standalone_utils

FUNCTION_NAMES_KERAS = ['sigmoid', 'relu', 'relu']
FUNCTION_NAMES_NICE = ['Sigmoid', 'ReLU', 'Leaky ReLU']
RELU_SLOPES = numpy.array([0, 0, 0.2])
INPUT_VALUES = numpy.linspace(-5, 5, num=10001, dtype=float)

COLOUR_MATRIX = 255 ** -1 * numpy.array([
    [27, 158, 119],
    [217, 95, 2],
    [117, 112, 179]
], dtype=float)

MAIN_LINE_WIDTH = 4
REFERENCE_LINE_WIDTH = 2
REFERENCE_LINE_COLOUR = numpy.full(3, 152. / 255)

FIGURE_WIDTH_INCHES = FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 600
OUTPUT_FILE_NAME = (
    '/localdata/ryan.lagerquist/eager/dissertation_figures/activation_graph.jpg'
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
    """Plots common activation functions.

    This is effectively the main method.
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    axes_object.plot(
        INPUT_VALUES, numpy.zeros(INPUT_VALUES.shape), linestyle='dashed',
        linewidth=REFERENCE_LINE_WIDTH, color=REFERENCE_LINE_COLOUR
    )

    for i in range(len(FUNCTION_NAMES_KERAS)):
        these_output_values = standalone_utils.do_activation(
            input_values=INPUT_VALUES, function_name=FUNCTION_NAMES_KERAS[i],
            alpha=RELU_SLOPES[i]
        )

        axes_object.plot(
            INPUT_VALUES, these_output_values, linestyle='solid',
            linewidth=MAIN_LINE_WIDTH, color=COLOUR_MATRIX[i, :],
            label=FUNCTION_NAMES_NICE[i]
        )

    axes_object.set_xlabel('Input')
    axes_object.set_ylabel('Output')
    axes_object.legend(loc='upper left')

    print('Saving figure to: "{0:s}"...'.format(OUTPUT_FILE_NAME))
    figure_object.savefig(
        OUTPUT_FILE_NAME, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    _run()
