"""Plots ReLU with different alpha-values (negative slopes)."""

import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.deep_learning import standalone_utils

ALPHA_VALUES = numpy.array([0, 0.1, 0.2, 0.5])
INPUT_VALUES = numpy.linspace(-5, 5, num=10001, dtype=float)

COLOUR_MATRIX = 255 ** -1 * numpy.array([
    [166, 206, 227],
    [31, 120, 180],
    [178, 223, 138],
    [51, 160, 44]
], dtype=float)

MAIN_LINE_WIDTH = 4
REFERENCE_LINE_WIDTH = 2
REFERENCE_LINE_COLOUR = numpy.full(3, 152. / 255)

FIGURE_WIDTH_INCHES = FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 600
OUTPUT_FILE_NAME = (
    '/localdata/ryan.lagerquist/eager/dissertation_figures/relu_graph.jpg'
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

    for i in range(len(ALPHA_VALUES)):
        these_output_values = standalone_utils.do_activation(
            input_values=INPUT_VALUES, function_name='relu',
            alpha=ALPHA_VALUES[i]
        )

        axes_object.plot(
            INPUT_VALUES, these_output_values, linestyle='solid',
            linewidth=MAIN_LINE_WIDTH, color=COLOUR_MATRIX[i, :],
            label=r'$\beta$ = {0:.1f}'.format(ALPHA_VALUES[i])
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
