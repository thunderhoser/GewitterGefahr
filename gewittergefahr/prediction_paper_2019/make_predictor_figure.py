"""Makes figure with GridRad and MYRORSS predictors."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_io import storm_tracking_io as tracking_io
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import radar_utils
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
DUMMY_TARGET_NAME = 'tornado_lead-time=0000-3600sec_distance=00000-10000m'

SOUNDING_FIELD_NAMES = [
    soundings.U_WIND_NAME, soundings.V_WIND_NAME,
    soundings.TEMPERATURE_NAME, soundings.SPECIFIC_HUMIDITY_NAME,
    soundings.PRESSURE_NAME
]
SOUNDING_HEIGHTS_M_AGL = soundings.DEFAULT_HEIGHT_LEVELS_M_AGL

RADAR_HEIGHTS_M_AGL = numpy.array([3000], dtype=int)

NUM_GRIDRAD_ROWS = 32
NUM_GRIDRAD_COLUMNS = 32
GRIDRAD_FIELD_NAMES = [
    radar_utils.REFL_NAME, radar_utils.SPECTRUM_WIDTH_NAME,
    radar_utils.VORTICITY_NAME, radar_utils.DIVERGENCE_NAME
]

NUM_MYRORSS_ROWS = 64
NUM_MYRORSS_COLUMNS = 64
MYRORSS_SHEAR_FIELD_NAMES = [
    radar_utils.LOW_LEVEL_SHEAR_NAME, radar_utils.MID_LEVEL_SHEAR_NAME
]

X_LABEL_COORD_NORMALIZED = -0.075
Y_LABEL_COORD_NORMALIZED = 1.
FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

GRIDRAD_DIR_ARG_NAME = 'gridrad_example_dir_name'
GRIDRAD_METAFILE_ARG_NAME = 'gridrad_storm_metafile_name'
GRIDRAD_INDEX_ARG_NAME = 'gridrad_example_index'
MYRORSS_DIR_ARG_NAME = 'myrorss_example_dir_name'
MYRORSS_METAFILE_ARG_NAME = 'myrorss_storm_metafile_name'
MYRORSS_INDEX_ARG_NAME = 'myrorss_example_index'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

GRIDRAD_DIR_HELP_STRING = (
    'Name of top-level directory with GridRad examples.  Files therein will be '
    'found by `input_examples.find_example_file` and read by '
    '`input_examples.read_example_file`.')

GRIDRAD_METAFILE_HELP_STRING = (
    'Name of file with storm metadata (IDs and times) for GridRad examples.  '
    'Will be read by `storm_tracking_io.read_ids_and_times`.')

GRIDRAD_INDEX_HELP_STRING = (
    'Index of GridRad example.  Will use only the [k]th example in `{0:s}`, '
    'where k = `{1:s}`.'
).format(GRIDRAD_METAFILE_ARG_NAME, GRIDRAD_INDEX_ARG_NAME)

MYRORSS_DIR_HELP_STRING = 'Same as `{0:s}` but for MYRORSS.'.format(
    GRIDRAD_DIR_ARG_NAME)

MYRORSS_METAFILE_HELP_STRING = 'Same as `{0:s}` but for MYRORSS.'.format(
    GRIDRAD_METAFILE_ARG_NAME)

MYRORSS_INDEX_HELP_STRING = 'Same as `{0:s}` but for MYRORSS.'.format(
    GRIDRAD_INDEX_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + GRIDRAD_DIR_ARG_NAME, type=str, required=True,
    help=GRIDRAD_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + GRIDRAD_METAFILE_ARG_NAME, type=str, required=True,
    help=GRIDRAD_METAFILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + GRIDRAD_INDEX_ARG_NAME, type=int, required=True,
    help=GRIDRAD_INDEX_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MYRORSS_DIR_ARG_NAME, type=str, required=True,
    help=MYRORSS_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MYRORSS_METAFILE_ARG_NAME, type=str, required=True,
    help=MYRORSS_METAFILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MYRORSS_INDEX_ARG_NAME, type=int, required=True,
    help=MYRORSS_INDEX_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _read_one_example(top_example_dir_name, storm_metafile_name, example_index,
                      source_name, radar_field_name, include_sounding):
    """Reads one example (storm object).

    T = number of input tensors to model

    :param top_example_dir_name: See documentation at top of file.
    :param storm_metafile_name: Same.
    :param example_index: Same.
    :param source_name: Radar source (must be accepted by
        `radar_utils.check_data_source`).
    :param radar_field_name: Name of radar field (must be accepted by
        `radar_utils.check_field_name`).
    :param include_sounding: Boolean flag.  Determines whether or not sounding
        will be read.
    :return: list_of_predictor_matrices: length-T list of numpy arrays, where
        the [i]th array is the [i]th input tensor to the model.  The first axis
        of each array has length = 1.
    :return: model_metadata_dict: See doc for `cnn.write_model_metadata`.
    """

    if source_name == radar_utils.GRIDRAD_SOURCE_ID:
        num_radar_rows = NUM_GRIDRAD_ROWS
        num_radar_columns = NUM_GRIDRAD_COLUMNS
    else:
        num_radar_rows = NUM_MYRORSS_ROWS
        num_radar_columns = NUM_MYRORSS_COLUMNS

    print('Reading data from: "{0:s}"...'.format(storm_metafile_name))
    all_id_strings, all_times_unix_sec = tracking_io.read_ids_and_times(
        storm_metafile_name)

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

    list_of_predictor_matrices = testing_io.read_specific_examples(
        desired_full_id_strings=[all_id_strings[example_index]],
        desired_times_unix_sec=all_times_unix_sec[[example_index]],
        option_dict=model_metadata_dict[cnn.TRAINING_OPTION_DICT_KEY],
        top_example_dir_name=top_example_dir_name,
        list_of_layer_operation_dicts=None
    )[0]

    return list_of_predictor_matrices, model_metadata_dict


def _run(gridrad_example_dir_name, gridrad_storm_metafile_name,
         gridrad_example_index, myrorss_example_dir_name,
         myrorss_storm_metafile_name, myrorss_example_index, output_dir_name):
    """Makes figure with GridRad and MYRORSS predictors.

    This is effectively the main method.

    :param gridrad_example_dir_name: See documentation at top of file.
    :param gridrad_storm_metafile_name: Same.
    :param gridrad_example_index: Same.
    :param myrorss_example_dir_name: Same.
    :param myrorss_storm_metafile_name: Same.
    :param myrorss_example_index: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name)

    letter_label = None
    num_gridrad_fields = len(GRIDRAD_FIELD_NAMES)
    panel_file_names = [None] * num_gridrad_fields * 2

    for j in range(num_gridrad_fields):
        these_predictor_matrices, this_metadata_dict = _read_one_example(
            top_example_dir_name=gridrad_example_dir_name,
            storm_metafile_name=gridrad_storm_metafile_name,
            example_index=gridrad_example_index,
            source_name=radar_utils.GRIDRAD_SOURCE_ID,
            radar_field_name=GRIDRAD_FIELD_NAMES[j], include_sounding=False)
        print(MINOR_SEPARATOR_STRING)

        this_handle_dict = plot_examples.plot_one_example(
            list_of_predictor_matrices=these_predictor_matrices,
            model_metadata_dict=this_metadata_dict, plot_sounding=False,
            allow_whitespace=True, pmm_flag=False, example_index=0,
            full_storm_id_string='A', storm_time_unix_sec=0)

        this_title_string = (
            radar_plotting.radar_fields_and_heights_to_panel_names(
                field_names=[GRIDRAD_FIELD_NAMES[j]],
                heights_m_agl=RADAR_HEIGHTS_M_AGL[[0]], include_units=True
            )[0]
        )

        this_title_string = this_title_string.replace('\n', ' ')
        this_title_string = 'GridRad {0:s}{1:s}'.format(
            this_title_string[0].lower(), this_title_string[1:]
        )

        this_figure_object = this_handle_dict[
            plot_examples.RADAR_FIGURES_KEY][0]
        this_axes_object = this_handle_dict[
            plot_examples.RADAR_AXES_KEY][0][0, 0]
        this_axes_object.set_title(this_title_string)

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
        these_predictor_matrices, this_metadata_dict = _read_one_example(
            top_example_dir_name=myrorss_example_dir_name,
            storm_metafile_name=myrorss_storm_metafile_name,
            example_index=myrorss_example_index,
            source_name=radar_utils.MYRORSS_SOURCE_ID,
            radar_field_name=MYRORSS_SHEAR_FIELD_NAMES[j],
            include_sounding=j == 0)
        print(MINOR_SEPARATOR_STRING)

        this_handle_dict = plot_examples.plot_one_example(
            list_of_predictor_matrices=these_predictor_matrices,
            model_metadata_dict=this_metadata_dict, plot_sounding=j == 0,
            allow_whitespace=True, pmm_flag=False, example_index=0,
            full_storm_id_string='A', storm_time_unix_sec=0)

        if j == 0:
            this_axes_object = this_handle_dict[plot_examples.SOUNDING_AXES_KEY]
            this_axes_object.set_title('Proximity sounding')

            letter_label = chr(ord(letter_label) + 1)
            plotting_utils.label_axes(
                axes_object=this_axes_object,
                label_string='({0:s})'.format(letter_label),
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

            this_title_string = (
                radar_plotting.radar_fields_and_heights_to_panel_names(
                    field_names=[radar_utils.REFL_NAME],
                    heights_m_agl=RADAR_HEIGHTS_M_AGL[[0]], include_units=True
                )[0]
            )

            this_title_string = this_title_string.replace('\n', ' ')
            this_title_string = 'MYRORSS {0:s}{1:s}'.format(
                this_title_string[0].lower(), this_title_string[1:]
            )

            this_figure_object = this_handle_dict[
                plot_examples.RADAR_FIGURES_KEY][0]
            this_axes_object = this_handle_dict[
                plot_examples.RADAR_AXES_KEY][0][0, 0]
            this_axes_object.set_title(this_title_string)

            letter_label = chr(ord(letter_label) + 1)
            plotting_utils.label_axes(
                axes_object=this_axes_object,
                label_string='({0:s})'.format(letter_label),
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

        this_title_string = (
            radar_plotting.radar_fields_and_heights_to_panel_names(
                field_names=[MYRORSS_SHEAR_FIELD_NAMES[j]],
                heights_m_agl=RADAR_HEIGHTS_M_AGL[[0]], include_units=True
            )[0]
        )

        this_title_string = this_title_string.split('\n')[0]
        this_title_string = 'MYRORSS {0:s}{1:s}'.format(
            this_title_string[0].lower(), this_title_string[1:]
        )

        this_figure_object = this_handle_dict[
            plot_examples.RADAR_FIGURES_KEY][1]
        this_axes_object = this_handle_dict[
            plot_examples.RADAR_AXES_KEY][1][0, 0]
        this_axes_object.set_title(this_title_string)

        letter_label = chr(ord(letter_label) + 1)
        plotting_utils.label_axes(
            axes_object=this_axes_object,
            label_string='({0:s})'.format(letter_label),
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
        gridrad_storm_metafile_name=getattr(
            INPUT_ARG_OBJECT, GRIDRAD_METAFILE_ARG_NAME),
        gridrad_example_index=getattr(INPUT_ARG_OBJECT, GRIDRAD_INDEX_ARG_NAME),
        myrorss_example_dir_name=getattr(
            INPUT_ARG_OBJECT, MYRORSS_DIR_ARG_NAME),
        myrorss_storm_metafile_name=getattr(
            INPUT_ARG_OBJECT, MYRORSS_METAFILE_ARG_NAME),
        myrorss_example_index=getattr(INPUT_ARG_OBJECT, MYRORSS_INDEX_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
