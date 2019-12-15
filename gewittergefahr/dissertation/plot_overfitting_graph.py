"""Plots graph to explain overfitting."""

import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

ORIG_LOSS = 2.
TRAINING_DECAY_RATE = 1. / 25
FIRST_VALIDN_DECAY_RATE = 1. / 35
SECOND_VALIDN_DECAY_RATE = 1. / 140
SECOND_VALIDN_START_EPOCH = 250

CUTOFF_EPOCH = 50
MAX_EPOCH_TO_PLOT = 75

MAIN_LINE_WIDTH = 4
REFERENCE_LINE_WIDTH = 2
REFERENCE_LINE_COLOUR = numpy.full(3, 152. / 255)
TRAINING_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
VALIDATION_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255

FIGURE_WIDTH_INCHES = FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 600
OUTPUT_FILE_NAME = (
    '/localdata/ryan.lagerquist/eager/dissertation_figures/'
    'overfitting_graph.jpg'
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
    """Plots schema to explain overfitting.
    
    This is effectively the main method.
    """

    epoch_indices = numpy.linspace(
        0, MAX_EPOCH_TO_PLOT, num=MAX_EPOCH_TO_PLOT + 1, dtype=int
    )
    training_losses = ORIG_LOSS * numpy.exp(
        -epoch_indices * TRAINING_DECAY_RATE
    )
    validation_losses = ORIG_LOSS * numpy.exp(
        -epoch_indices * FIRST_VALIDN_DECAY_RATE
    )
    second_validation_losses = ORIG_LOSS * numpy.exp(
        -(SECOND_VALIDN_START_EPOCH - epoch_indices) * SECOND_VALIDN_DECAY_RATE
    )

    validation_losses[CUTOFF_EPOCH:] = second_validation_losses[CUTOFF_EPOCH:]

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    axes_object.plot(
        epoch_indices, training_losses, linewidth=MAIN_LINE_WIDTH,
        color=TRAINING_COLOUR, label='Training')

    axes_object.plot(
        epoch_indices, validation_losses, linewidth=MAIN_LINE_WIDTH,
        color=VALIDATION_COLOUR, label='Validation')

    y_limits = numpy.array([0, ORIG_LOSS])
    axes_object.set_ylim(y_limits)
    axes_object.set_xlim([0, MAX_EPOCH_TO_PLOT])

    axes_object.plot(
        numpy.full(2, CUTOFF_EPOCH), y_limits, linestyle='dashed',
        linewidth=REFERENCE_LINE_WIDTH, color=REFERENCE_LINE_COLOUR,
        label='Overfitting starts here'
    )

    axes_object.set_xlabel('Epoch')
    axes_object.set_ylabel('Loss')
    axes_object.legend(loc='lower left')

    print('Saving figure to: "{0:s}"...'.format(OUTPUT_FILE_NAME))
    figure_object.savefig(
        OUTPUT_FILE_NAME, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    _run()
