"""Makes figure with GridRad and MYRORSS predictors."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import testing_io
from gewittergefahr.deep_learning import training_validation_io as trainval_io
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import radar_plotting
from gewittergefahr.plotting import imagemagick_utils
from gewittergefahr.scripts import plot_input_examples as plot_examples

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
DUMMY_TARGET_NAME = 'tornado_lead-time=0000-3600sec_distance=00000-10000m'

SOUNDING_FIELD_NAMES = [
    soundings.U_WIND_NAME, soundings.V_WIND_NAME,
    soundings.TEMPERATURE_NAME, soundings.SPECIFIC_HUMIDITY_NAME,
    soundings.PRESSURE_NAME
]
SOUNDING_HEIGHTS_M_AGL = soundings.DEFAULT_HEIGHT_LEVELS_M_AGL

NUM_GRIDRAD_ROWS = 32
NUM_GRIDRAD_COLUMNS = 32
RADAR_HEIGHTS_M_AGL = numpy.array([3000], dtype=int)
GRIDRAD_FIELD_NAMES = [
    radar_utils.REFL_NAME, radar_utils.SPECTRUM_WIDTH_NAME,
    radar_utils.VORTICITY_NAME, radar_utils.DIVERGENCE_NAME
]

NUM_MYRORSS_ROWS = 64
NUM_MYRORSS_COLUMNS = 64
MYRORSS_SHEAR_FIELD_NAMES = [
    radar_utils.LOW_LEVEL_SHEAR_NAME, radar_utils.MID_LEVEL_SHEAR_NAME
]

COLOUR_BAR_LENGTH = 0.8
DEFAULT_FONT_SIZE = 45
TITLE_FONT_SIZE = 45
COLOUR_BAR_FONT_SIZE = 45
SOUNDING_FONT_SIZE = 45
PANEL_LETTER_FONT_SIZE = 75

pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

X_LABEL_COORD_NORMALIZED = -0.02
Y_LABEL_COORD_NORMALIZED = 0.85
FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

GRIDRAD_DIR_ARG_NAME = 'gridrad_example_dir_name'
GRIDRAD_ID_ARG_NAME = 'gridrad_full_id_string'
GRIDRAD_TIME_ARG_NAME = 'gridrad_time_string'
MYRORSS_DIR_ARG_NAME = 'myrorss_example_dir_name'
MYRORSS_ID_ARG_NAME = 'myrorss_full_id_string'
MYRORSS_TIME_ARG_NAME = 'myrorss_time_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

GRIDRAD_DIR_HELP_STRING = (
    'Name of top-level directory with GridRad examples.  Files therein will be '
    'found by `input_examples.find_example_file` and read by '
    '`input_examples.read_example_file`.')

GRIDRAD_ID_HELP_STRING = 'Full ID of GridRad storm object.'

GRIDRAD_TIME_HELP_STRING = (
    'Valid time (format "yyyy-mm-dd-HHMMSS") of GridRad storm object.')

MYRORSS_DIR_HELP_STRING = 'Same as `{0:s}` but for MYRORSS.'.format(
    GRIDRAD_DIR_ARG_NAME)

MYRORSS_ID_HELP_STRING = 'Same as `{0:s}` but for MYRORSS.'.format(
    GRIDRAD_ID_ARG_NAME)

MYRORSS_TIME_HELP_STRING = 'Same as `{0:s}` but for MYRORSS.'.format(
    GRIDRAD_TIME_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + GRIDRAD_DIR_ARG_NAME, type=str, required=True,
    help=GRIDRAD_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + GRIDRAD_ID_ARG_NAME, type=str, required=True,
    help=GRIDRAD_ID_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + GRIDRAD_TIME_ARG_NAME, type=str, required=True,
    help=GRIDRAD_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MYRORSS_DIR_ARG_NAME, type=str, required=True,
    help=MYRORSS_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MYRORSS_ID_ARG_NAME, type=str, required=True,
    help=MYRORSS_ID_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MYRORSS_TIME_ARG_NAME, type=str, required=True,
    help=MYRORSS_TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _read_one_example(
        top_example_dir_name, full_storm_id_string, storm_time_unix_sec,
        source_name, radar_field_name, include_sounding):
    """Reads one example (storm object).

    T = number of input tensors to model
    H_s = number of heights in sounding

    :param top_example_dir_name: See documentation at top of file.
    :param full_storm_id_string: Full storm ID.
    :param storm_time_unix_sec: Valid time of storm.
    :param source_name: Radar source (must be accepted by
        `radar_utils.check_data_source`).
    :param radar_field_name: Name of radar field (must be accepted by
        `radar_utils.check_field_name`).
    :param include_sounding: Boolean flag.
    :return: predictor_matrices: length-T list of numpy arrays, where
        the [i]th array is the [i]th input tensor to the model.  The first axis
        of each array has length = 1.
    :return: model_metadata_dict: See doc for `cnn.write_model_metadata`.
    :return: sounding_pressures_pa: length-H numpy array of pressures.  If
        soundings were not read, this is None.
    """

    if source_name == radar_utils.GRIDRAD_SOURCE_ID:
        num_radar_rows = NUM_GRIDRAD_ROWS
        num_radar_columns = NUM_GRIDRAD_COLUMNS
    else:
        num_radar_rows = NUM_MYRORSS_ROWS
        num_radar_columns = NUM_MYRORSS_COLUMNS

    training_option_dict = dict()
    training_option_dict[trainval_io.RADAR_FIELDS_KEY] = [radar_field_name]
    training_option_dict[trainval_io.RADAR_HEIGHTS_KEY] = RADAR_HEIGHTS_M_AGL
    training_option_dict[trainval_io.SOUNDING_FIELDS_KEY] = (
        SOUNDING_FIELD_NAMES if include_sounding else None
    )
    training_option_dict[trainval_io.SOUNDING_HEIGHTS_KEY] = (
        SOUNDING_HEIGHTS_M_AGL
    )

    training_option_dict[trainval_io.NUM_ROWS_KEY] = num_radar_rows
    training_option_dict[trainval_io.NUM_COLUMNS_KEY] = num_radar_columns
    training_option_dict[trainval_io.NORMALIZATION_TYPE_KEY] = None
    training_option_dict[trainval_io.TARGET_NAME_KEY] = DUMMY_TARGET_NAME
    training_option_dict[trainval_io.BINARIZE_TARGET_KEY] = False
    training_option_dict[trainval_io.SAMPLING_FRACTIONS_KEY] = None
    training_option_dict[trainval_io.REFLECTIVITY_MASK_KEY] = None
    training_option_dict[trainval_io.UPSAMPLE_REFLECTIVITY_KEY] = False

    model_metadata_dict = {
        cnn.TRAINING_OPTION_DICT_KEY: training_option_dict,
        cnn.LAYER_OPERATIONS_KEY: None,
    }

    print(MINOR_SEPARATOR_STRING)

    example_dict = testing_io.read_predictors_specific_examples(
        top_example_dir_name=top_example_dir_name,
        desired_full_id_strings=[full_storm_id_string],
        desired_times_unix_sec=numpy.array([storm_time_unix_sec], dtype=int),
        option_dict=model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY],
        layer_operation_dicts=None
    )

    predictor_matrices = example_dict[testing_io.INPUT_MATRICES_KEY]
    sounding_pressure_matrix_pa = example_dict[
        testing_io.SOUNDING_PRESSURES_KEY]

    if sounding_pressure_matrix_pa is None:
        sounding_pressures_pa = None
    else:
        sounding_pressures_pa = sounding_pressure_matrix_pa[0, ...]

    return predictor_matrices, model_metadata_dict, sounding_pressures_pa


def _run(gridrad_example_dir_name, gridrad_full_id_string, gridrad_time_string,
         myrorss_example_dir_name, myrorss_full_id_string, myrorss_time_string,
         output_dir_name):
    """Makes figure with GridRad and MYRORSS predictors.

    This is effectively the main method.

    :param gridrad_example_dir_name: See documentation at top of file.
    :param gridrad_full_id_string: Same.
    :param gridrad_time_string: Same.
    :param myrorss_example_dir_name: Same.
    :param myrorss_full_id_string: Same.
    :param myrorss_time_string: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    gridrad_time_unix_sec = time_conversion.string_to_unix_sec(
        gridrad_time_string, TIME_FORMAT)
    myrorss_time_unix_sec = time_conversion.string_to_unix_sec(
        myrorss_time_string, TIME_FORMAT)

    letter_label = None
    num_gridrad_fields = len(GRIDRAD_FIELD_NAMES)
    panel_file_names = [None] * num_gridrad_fields * 2

    for j in range(num_gridrad_fields):
        these_predictor_matrices, this_metadata_dict = _read_one_example(
            top_example_dir_name=gridrad_example_dir_name,
            full_storm_id_string=gridrad_full_id_string,
            storm_time_unix_sec=gridrad_time_unix_sec,
            source_name=radar_utils.GRIDRAD_SOURCE_ID,
            radar_field_name=GRIDRAD_FIELD_NAMES[j], include_sounding=False
        )[:2]

        print(MINOR_SEPARATOR_STRING)

        this_handle_dict = plot_examples.plot_one_example(
            list_of_predictor_matrices=these_predictor_matrices,
            model_metadata_dict=this_metadata_dict, pmm_flag=False,
            example_index=0, plot_sounding=False, allow_whitespace=True,
            plot_panel_names=False, add_titles=False, label_colour_bars=False,
            colour_bar_font_size=COLOUR_BAR_FONT_SIZE,
            colour_bar_length=COLOUR_BAR_LENGTH)

        this_title_string = radar_plotting.fields_and_heights_to_names(
            field_names=[GRIDRAD_FIELD_NAMES[j]],
            heights_m_agl=RADAR_HEIGHTS_M_AGL[[0]], include_units=True
        )[0]

        this_title_string = this_title_string.replace('\n', ' ').replace(
            ' km AGL', ' km')
        this_title_string = 'GridRad {0:s}{1:s}'.format(
            this_title_string[0].lower(), this_title_string[1:]
        )

        this_figure_object = this_handle_dict[
            plot_examples.RADAR_FIGURES_KEY][0]
        this_axes_object = this_handle_dict[
            plot_examples.RADAR_AXES_KEY][0][0, 0]

        this_figure_object.suptitle('')
        this_axes_object.set_title(
            this_title_string, fontsize=TITLE_FONT_SIZE)

        # this_axes_object.set_yticklabels(
        #     this_axes_object.get_yticks(), color=ALMOST_WHITE_COLOUR
        # )

        if letter_label is None:
            letter_label = 'a'
        else:
            letter_label = chr(ord(letter_label) + 1)

        plotting_utils.label_axes(
            axes_object=this_axes_object,
            label_string='({0:s})'.format(letter_label),
            font_size=PANEL_LETTER_FONT_SIZE,
            x_coord_normalized=X_LABEL_COORD_NORMALIZED,
            y_coord_normalized=Y_LABEL_COORD_NORMALIZED
        )

        panel_file_names[j * 2] = '{0:s}/gridrad_{1:s}.jpg'.format(
            output_dir_name, GRIDRAD_FIELD_NAMES[j].replace('_', '-')
        )

        print('Saving figure to: "{0:s}"...'.format(panel_file_names[j * 2]))
        this_figure_object.savefig(
            panel_file_names[j * 2], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(this_figure_object)

        print(SEPARATOR_STRING)

    num_myrorss_shear_fields = len(MYRORSS_SHEAR_FIELD_NAMES)

    for j in range(num_myrorss_shear_fields):
        (these_predictor_matrices, this_metadata_dict, these_pressures_pascals
        ) = _read_one_example(
            top_example_dir_name=myrorss_example_dir_name,
            full_storm_id_string=myrorss_full_id_string,
            storm_time_unix_sec=myrorss_time_unix_sec,
            source_name=radar_utils.MYRORSS_SOURCE_ID,
            radar_field_name=MYRORSS_SHEAR_FIELD_NAMES[j],
            include_sounding=j == 0)

        print(MINOR_SEPARATOR_STRING)

        this_handle_dict = plot_examples.plot_one_example(
            list_of_predictor_matrices=these_predictor_matrices,
            model_metadata_dict=this_metadata_dict, pmm_flag=False,
            example_index=0, plot_sounding=j == 0,
            sounding_pressures_pascals=these_pressures_pascals,
            allow_whitespace=True, plot_panel_names=False, add_titles=False,
            label_colour_bars=False, colour_bar_font_size=COLOUR_BAR_FONT_SIZE,
            colour_bar_length=COLOUR_BAR_LENGTH,
            sounding_font_size=SOUNDING_FONT_SIZE)

        if j == 0:
            this_axes_object = this_handle_dict[plot_examples.SOUNDING_AXES_KEY]
            this_axes_object.set_title('Proximity sounding')

            letter_label = chr(ord(letter_label) + 1)
            plotting_utils.label_axes(
                axes_object=this_axes_object,
                label_string='({0:s})'.format(letter_label),
                font_size=PANEL_LETTER_FONT_SIZE,
                x_coord_normalized=X_LABEL_COORD_NORMALIZED,
                y_coord_normalized=Y_LABEL_COORD_NORMALIZED
            )

            this_figure_object = this_handle_dict[
                plot_examples.SOUNDING_FIGURE_KEY]
            panel_file_names[1] = '{0:s}/sounding.jpg'.format(output_dir_name)

            print('Saving figure to: "{0:s}"...'.format(panel_file_names[1]))
            this_figure_object.savefig(
                panel_file_names[1], dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(this_figure_object)

            this_title_string = radar_plotting.fields_and_heights_to_names(
                field_names=[radar_utils.REFL_NAME],
                heights_m_agl=RADAR_HEIGHTS_M_AGL[[0]], include_units=True
            )[0]

            this_title_string = this_title_string.replace('\n', ' ').replace(
                ' km AGL', ' km')
            this_title_string = 'MYRORSS {0:s}{1:s}'.format(
                this_title_string[0].lower(), this_title_string[1:]
            )

            this_figure_object = this_handle_dict[
                plot_examples.RADAR_FIGURES_KEY][0]
            this_axes_object = this_handle_dict[
                plot_examples.RADAR_AXES_KEY][0][0, 0]

            this_figure_object.suptitle('')
            this_axes_object.set_title(
                this_title_string, fontsize=TITLE_FONT_SIZE)

            letter_label = chr(ord(letter_label) + 1)
            plotting_utils.label_axes(
                axes_object=this_axes_object,
                label_string='({0:s})'.format(letter_label),
                font_size=PANEL_LETTER_FONT_SIZE,
                x_coord_normalized=X_LABEL_COORD_NORMALIZED,
                y_coord_normalized=Y_LABEL_COORD_NORMALIZED
            )

            panel_file_names[3] = '{0:s}/myrorss_{1:s}.jpg'.format(
                output_dir_name, radar_utils.REFL_NAME.replace('_', '-')
            )

            print('Saving figure to: "{0:s}"...'.format(panel_file_names[3]))
            this_figure_object.savefig(
                panel_file_names[3], dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(this_figure_object)

        this_title_string = radar_plotting.fields_and_heights_to_names(
            field_names=[MYRORSS_SHEAR_FIELD_NAMES[j]],
            heights_m_agl=RADAR_HEIGHTS_M_AGL[[0]], include_units=True
        )[0]

        this_title_string = this_title_string.split('\n')[0]
        this_title_string = 'MYRORSS {0:s}{1:s}'.format(
            this_title_string[0].lower(), this_title_string[1:]
        )

        this_figure_object = this_handle_dict[
            plot_examples.RADAR_FIGURES_KEY][1]
        this_axes_object = this_handle_dict[
            plot_examples.RADAR_AXES_KEY][1][0, 0]

        this_figure_object.suptitle('')
        this_axes_object.set_title(
            this_title_string, fontsize=TITLE_FONT_SIZE)

        letter_label = chr(ord(letter_label) + 1)
        plotting_utils.label_axes(
            axes_object=this_axes_object,
            label_string='({0:s})'.format(letter_label),
            font_size=PANEL_LETTER_FONT_SIZE,
            x_coord_normalized=X_LABEL_COORD_NORMALIZED,
            y_coord_normalized=Y_LABEL_COORD_NORMALIZED
        )

        panel_file_names[5 + j * 2] = '{0:s}/myrorss_{1:s}.jpg'.format(
            output_dir_name, MYRORSS_SHEAR_FIELD_NAMES[j].replace('_', '-')
        )

        print('Saving figure to: "{0:s}"...'.format(
            panel_file_names[5 + j * 2]
        ))
        this_figure_object.savefig(
            panel_file_names[5 + j * 2], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(this_figure_object)

        if j != num_myrorss_shear_fields:
            print(SEPARATOR_STRING)

    concat_file_name = '{0:s}/predictors.jpg'.format(output_dir_name)
    print('Concatenating panels to: "{0:s}"...'.format(concat_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names, output_file_name=concat_file_name,
        num_panel_rows=4, num_panel_columns=2)

    imagemagick_utils.resize_image(
        input_file_name=concat_file_name, output_file_name=concat_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        gridrad_example_dir_name=getattr(
            INPUT_ARG_OBJECT, GRIDRAD_DIR_ARG_NAME),
        gridrad_full_id_string=getattr(INPUT_ARG_OBJECT, GRIDRAD_ID_ARG_NAME),
        gridrad_time_string=getattr(INPUT_ARG_OBJECT, GRIDRAD_TIME_ARG_NAME),
        myrorss_example_dir_name=getattr(
            INPUT_ARG_OBJECT, MYRORSS_DIR_ARG_NAME),
        myrorss_full_id_string=getattr(INPUT_ARG_OBJECT, MYRORSS_ID_ARG_NAME),
        myrorss_time_string=getattr(INPUT_ARG_OBJECT, MYRORSS_TIME_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
