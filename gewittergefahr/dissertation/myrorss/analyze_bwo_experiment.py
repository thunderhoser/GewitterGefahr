"""Analyzes backwards-optimization experiment."""

import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from generalexam.machine_learning import evaluation_utils
from gewittergefahr.deep_learning import backwards_optimization as backwards_opt
from gewittergefahr.plotting import plotting_utils

L2_WEIGHTS = numpy.logspace(-4, 1, num=11)
MINMAX_WEIGHTS = numpy.logspace(-4, 1, num=11)

TOP_EXPERIMENT_DIR_NAME = (
    '/glade/p/univ/uokl0026/myrorss_experiment/conus_only/'
    'dropout=0.750_l2=0.003162_num-dense-layers=2_data-aug=1/testing/'
    'extreme_examples/unique_storm_cells/bwo_experiment_best_hits'
)

COLOUR_MAP_OBJECT = pyplot.get_cmap('viridis')
FIGURE_RESOLUTION_DPI = 300

FONT_SIZE = 30
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)


def _run():
    """Analyzes backwards-optimization experiment.

    This is effectively the main method.
    """

    num_l2_weights = len(L2_WEIGHTS)
    num_minmax_weights = len(MINMAX_WEIGHTS)

    mean_final_activation_matrix = numpy.full(
        (num_l2_weights, num_minmax_weights), numpy.nan
    )

    for i in range(num_l2_weights):
        for j in range(num_minmax_weights):
            this_file_name = (
                '{0:s}/bwo_pmm_l2-weight={1:.10f}_minmax-weight={2:.10f}.p'
            ).format(
                TOP_EXPERIMENT_DIR_NAME, L2_WEIGHTS[i], MINMAX_WEIGHTS[j]
            )

            print('Reading data from: "{0:s}"...'.format(this_file_name))
            this_bwo_dict = backwards_opt.read_file(this_file_name)[0]
            mean_final_activation_matrix[i, j] = this_bwo_dict[
                backwards_opt.MEAN_FINAL_ACTIVATION_KEY
            ]

    x_tick_labels = ['{0:.1f}'.format(r) for r in numpy.log10(MINMAX_WEIGHTS)]
    y_tick_labels = ['{0:.1f}'.format(w) for w in numpy.log10(L2_WEIGHTS)]

    axes_object = evaluation_utils.plot_scores_2d(
        score_matrix=mean_final_activation_matrix,
        x_tick_label_strings=x_tick_labels, y_tick_label_strings=y_tick_labels,
        colour_map_object=COLOUR_MAP_OBJECT,
        min_colour_value=0., max_colour_value=1.
    )

    axes_object.set_xlabel(r'Min-max weight (log$_{10}$)')
    axes_object.set_ylabel(r'L$_2$ weight (log$_{10}$)')

    plotting_utils.plot_linear_colour_bar(
        axes_object_or_matrix=axes_object,
        data_matrix=mean_final_activation_matrix,
        colour_map_object=COLOUR_MAP_OBJECT, min_value=0., max_value=1.,
        orientation_string='vertical', extend_min=False, extend_max=False,
        font_size=FONT_SIZE
    )

    output_file_name = '{0:s}/mean_final_activations.jpg'.format(
        TOP_EXPERIMENT_DIR_NAME
    )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    pyplot.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close()


if __name__ == '__main__':
    _run()
