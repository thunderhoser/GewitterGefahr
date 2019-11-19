"""Analyzes part 2 of backwards-optimization experiment."""

import numpy
from gewittergefahr.deep_learning import backwards_optimization as backwards_opt

MINMAX_WEIGHTS = numpy.logspace(-4, 1, num=26)

TOP_EXPERIMENT_DIR_NAME = (
    '/glade/work/ryanlage/prediction_paper_2019/gridrad_experiment/'
    'dropout=0.500_l2=0.010000_num-dense-layers=2_data-aug=1/testing/'
    'extreme_examples/unique_storm_cells/bwo_experiment_part2')


def _run():
    """Analyzes backwards-optimization experiment.

    This is effectively the main method.
    """

    num_weights = len(MINMAX_WEIGHTS)

    for i in range(num_weights):
        this_file_name = '{0:s}/bwo_minmax-weight={1:014.10f}_pmm.p'.format(
            TOP_EXPERIMENT_DIR_NAME, MINMAX_WEIGHTS[i]
        )

        this_bwo_dict = backwards_opt.read_file(this_file_name)[0]
        this_mean_final_activation = this_bwo_dict[
            backwards_opt.MEAN_FINAL_ACTIVATION_KEY
        ]

        print((
            'Min-max weight = 10^{0:.1f} ... mean final activation = {1:.4f}'
        ).format(
            numpy.log10(MINMAX_WEIGHTS[i]), this_mean_final_activation
        ))


if __name__ == '__main__':
    _run()
