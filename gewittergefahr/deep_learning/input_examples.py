"""Deals with input examples for deep learning.

One "input example" is one storm object.

--- NOTATION ---

The following letters will be used throughout this module.

E = number of examples (storm objects)
M = number of rows in each radar image
N = number of columns in each radar image
H_r = number of radar heights
F_r = number of radar fields (or "variables" or "channels")
H_s = number of sounding heights
F_s = number of sounding fields (or "variables" or "channels")
C = number of radar field/height pairs
"""

import copy
import glob
import os.path
import numpy
import netCDF4
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import target_val_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import temperature_conversions as temp_conversion
from gewittergefahr.gg_utils import storm_tracking_utils as tracking_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import storm_images
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

BATCH_NUMBER_REGEX = '[0-9][0-9][0-9][0-9][0-9][0-9][0-9]'
TIME_FORMAT_IN_FILE_NAMES = '%Y-%m-%d-%H%M%S'

DEFAULT_NUM_EXAMPLES_PER_OUT_CHUNK = 8
DEFAULT_NUM_EXAMPLES_PER_OUT_FILE = 128
NUM_BATCHES_PER_DIRECTORY = 1000

AZIMUTHAL_SHEAR_FIELD_NAMES = [
    radar_utils.LOW_LEVEL_SHEAR_NAME, radar_utils.MID_LEVEL_SHEAR_NAME
]

TARGET_NAMES_KEY = 'target_names'
ROTATED_GRIDS_KEY = 'rotated_grids'
ROTATED_GRID_SPACING_KEY = 'rotated_grid_spacing_metres'

FULL_IDS_KEY = 'full_storm_id_strings'
STORM_TIMES_KEY = 'storm_times_unix_sec'
TARGET_MATRIX_KEY = 'target_matrix'
RADAR_IMAGE_MATRIX_KEY = 'radar_image_matrix'
RADAR_FIELDS_KEY = 'radar_field_names'
RADAR_HEIGHTS_KEY = 'radar_heights_m_agl'
SOUNDING_FIELDS_KEY = 'sounding_field_names'
SOUNDING_MATRIX_KEY = 'sounding_matrix'
SOUNDING_HEIGHTS_KEY = 'sounding_heights_m_agl'
REFL_IMAGE_MATRIX_KEY = 'reflectivity_image_matrix_dbz'
AZ_SHEAR_IMAGE_MATRIX_KEY = 'az_shear_image_matrix_s01'

MAIN_KEYS = [
    FULL_IDS_KEY, STORM_TIMES_KEY, RADAR_IMAGE_MATRIX_KEY,
    REFL_IMAGE_MATRIX_KEY, AZ_SHEAR_IMAGE_MATRIX_KEY, TARGET_MATRIX_KEY,
    SOUNDING_MATRIX_KEY
]

REQUIRED_MAIN_KEYS = [
    FULL_IDS_KEY, STORM_TIMES_KEY, TARGET_MATRIX_KEY
]

METADATA_KEYS = [
    TARGET_NAMES_KEY, ROTATED_GRIDS_KEY, ROTATED_GRID_SPACING_KEY,
    RADAR_FIELDS_KEY, RADAR_HEIGHTS_KEY, SOUNDING_FIELDS_KEY,
    SOUNDING_HEIGHTS_KEY
]

TARGET_NAME_KEY = 'target_name'
TARGET_VALUES_KEY = 'target_values'

EXAMPLE_DIMENSION_KEY = 'storm_object'
ROW_DIMENSION_KEY = 'grid_row'
COLUMN_DIMENSION_KEY = 'grid_column'
REFL_ROW_DIMENSION_KEY = 'reflectivity_grid_row'
REFL_COLUMN_DIMENSION_KEY = 'reflectivity_grid_column'
AZ_SHEAR_ROW_DIMENSION_KEY = 'az_shear_grid_row'
AZ_SHEAR_COLUMN_DIMENSION_KEY = 'az_shear_grid_column'
RADAR_FIELD_DIM_KEY = 'radar_field'
RADAR_HEIGHT_DIM_KEY = 'radar_height'
RADAR_CHANNEL_DIM_KEY = 'radar_channel'
SOUNDING_FIELD_DIM_KEY = 'sounding_field'
SOUNDING_HEIGHT_DIM_KEY = 'sounding_height'
TARGET_VARIABLE_DIM_KEY = 'target_variable'
STORM_ID_CHAR_DIM_KEY = 'storm_id_character'
RADAR_FIELD_CHAR_DIM_KEY = 'radar_field_name_character'
SOUNDING_FIELD_CHAR_DIM_KEY = 'sounding_field_name_character'
TARGET_NAME_CHAR_DIM_KEY = 'target_name_character'

RADAR_FIELD_KEY = 'radar_field_name'
OPERATION_NAME_KEY = 'operation_name'
MIN_HEIGHT_KEY = 'min_height_m_agl'
MAX_HEIGHT_KEY = 'max_height_m_agl'

MIN_OPERATION_NAME = 'min'
MAX_OPERATION_NAME = 'max'
MEAN_OPERATION_NAME = 'mean'
VALID_LAYER_OPERATION_NAMES = [
    MIN_OPERATION_NAME, MAX_OPERATION_NAME, MEAN_OPERATION_NAME
]

OPERATION_NAME_TO_FUNCTION_DICT = {
    MIN_OPERATION_NAME: numpy.min,
    MAX_OPERATION_NAME: numpy.max,
    MEAN_OPERATION_NAME: numpy.mean
}

MIN_RADAR_HEIGHTS_KEY = 'min_radar_heights_m_agl'
MAX_RADAR_HEIGHTS_KEY = 'max_radar_heights_m_agl'
RADAR_LAYER_OPERATION_NAMES_KEY = 'radar_layer_operation_names'


def _read_soundings(sounding_file_name, sounding_field_names, radar_image_dict):
    """Reads storm-centered soundings and matches w storm-centered radar imgs.

    :param sounding_file_name: Path to input file (will be read by
        `soundings.read_soundings`).
    :param sounding_field_names: See doc for `soundings.read_soundings`.
    :param radar_image_dict: Dictionary created by
        `storm_images.read_storm_images`.
    :return: sounding_dict: Dictionary created by `soundings.read_soundings`.
    :return: radar_image_dict: Same as input, but excluding storm objects with
        no sounding.
    """

    print('Reading data from: "{0:s}"...'.format(sounding_file_name))
    sounding_dict, _ = soundings.read_soundings(
        netcdf_file_name=sounding_file_name,
        field_names_to_keep=sounding_field_names,
        full_id_strings_to_keep=radar_image_dict[storm_images.FULL_IDS_KEY],
        init_times_to_keep_unix_sec=radar_image_dict[
            storm_images.VALID_TIMES_KEY]
    )

    num_examples_with_soundings = len(sounding_dict[soundings.FULL_IDS_KEY])
    if num_examples_with_soundings == 0:
        return None, None

    radar_full_id_strings = numpy.array(
        radar_image_dict[storm_images.FULL_IDS_KEY]
    )
    orig_storm_times_unix_sec = (
        radar_image_dict[storm_images.VALID_TIMES_KEY] + 0
    )

    indices_to_keep = []

    for i in range(num_examples_with_soundings):
        this_index = numpy.where(numpy.logical_and(
            radar_full_id_strings == sounding_dict[soundings.FULL_IDS_KEY][i],
            orig_storm_times_unix_sec ==
            sounding_dict[soundings.INITIAL_TIMES_KEY][i]
        ))[0][0]

        indices_to_keep.append(this_index)

    indices_to_keep = numpy.array(indices_to_keep, dtype=int)
    radar_image_dict[storm_images.STORM_IMAGE_MATRIX_KEY] = radar_image_dict[
        storm_images.STORM_IMAGE_MATRIX_KEY
    ][indices_to_keep, ...]

    radar_image_dict[storm_images.FULL_IDS_KEY] = sounding_dict[
        soundings.FULL_IDS_KEY
    ]
    radar_image_dict[storm_images.VALID_TIMES_KEY] = sounding_dict[
        soundings.INITIAL_TIMES_KEY
    ]

    return sounding_dict, radar_image_dict


def _create_2d_examples(
        radar_file_names, full_id_strings, storm_times_unix_sec,
        target_matrix, sounding_file_name=None, sounding_field_names=None):
    """Creates 2-D examples for one file time.

    E = number of desired examples (storm objects)
    e = number of examples returned
    T = number of target variables

    :param radar_file_names: length-C list of paths to storm-centered radar
        images.  Files will be read by `storm_images.read_storm_images`.
    :param full_id_strings: length-E list with full IDs of storm objects to
        return.
    :param storm_times_unix_sec: length-E numpy array with valid times of storm
        objects to return.
    :param target_matrix: E-by-T numpy array of target values (integer class
        labels).
    :param sounding_file_name: Path to sounding file (will be read by
        `soundings.read_soundings`).  If `sounding_file_name is None`, examples
        will not include soundings.
    :param sounding_field_names: See doc for `soundings.read_soundings`.
    :return: example_dict: Same as input for `write_example_file`, but without
        key "target_names".
    """

    orig_full_id_strings = copy.deepcopy(full_id_strings)
    orig_storm_times_unix_sec = storm_times_unix_sec + 0

    print('Reading data from: "{0:s}"...'.format(radar_file_names[0]))
    this_radar_image_dict = storm_images.read_storm_images(
        netcdf_file_name=radar_file_names[0],
        full_id_strings_to_keep=full_id_strings,
        valid_times_to_keep_unix_sec=storm_times_unix_sec)

    if this_radar_image_dict is None:
        return None

    if sounding_file_name is None:
        sounding_matrix = None
        sounding_field_names = None
        sounding_heights_m_agl = None
    else:
        sounding_dict, this_radar_image_dict = _read_soundings(
            sounding_file_name=sounding_file_name,
            sounding_field_names=sounding_field_names,
            radar_image_dict=this_radar_image_dict)

        if this_radar_image_dict is None:
            return None
        if len(this_radar_image_dict[storm_images.FULL_IDS_KEY]) == 0:
            return None

        sounding_matrix = sounding_dict[soundings.SOUNDING_MATRIX_KEY]
        sounding_field_names = sounding_dict[soundings.FIELD_NAMES_KEY]
        sounding_heights_m_agl = sounding_dict[soundings.HEIGHT_LEVELS_KEY]

    full_id_strings = this_radar_image_dict[storm_images.FULL_IDS_KEY]
    storm_times_unix_sec = this_radar_image_dict[storm_images.VALID_TIMES_KEY]

    these_indices = tracking_utils.find_storm_objects(
        all_id_strings=orig_full_id_strings,
        all_times_unix_sec=orig_storm_times_unix_sec,
        id_strings_to_keep=full_id_strings,
        times_to_keep_unix_sec=storm_times_unix_sec, allow_missing=False)

    target_matrix = target_matrix[these_indices, :]

    num_channels = len(radar_file_names)
    tuple_of_image_matrices = ()

    for j in range(num_channels):
        if j != 0:
            print('Reading data from: "{0:s}"...'.format(radar_file_names[j]))
            this_radar_image_dict = storm_images.read_storm_images(
                netcdf_file_name=radar_file_names[j],
                full_id_strings_to_keep=full_id_strings,
                valid_times_to_keep_unix_sec=storm_times_unix_sec)

        tuple_of_image_matrices += (
            this_radar_image_dict[storm_images.STORM_IMAGE_MATRIX_KEY],
        )

    radar_field_names = [
        storm_images.image_file_name_to_field(f) for f in radar_file_names
    ]
    radar_heights_m_agl = numpy.array(
        [storm_images.image_file_name_to_height(f) for f in radar_file_names],
        dtype=int
    )

    example_dict = {
        FULL_IDS_KEY: full_id_strings,
        STORM_TIMES_KEY: storm_times_unix_sec,
        RADAR_FIELDS_KEY: radar_field_names,
        RADAR_HEIGHTS_KEY: radar_heights_m_agl,
        ROTATED_GRIDS_KEY:
            this_radar_image_dict[storm_images.ROTATED_GRIDS_KEY],
        ROTATED_GRID_SPACING_KEY:
            this_radar_image_dict[storm_images.ROTATED_GRID_SPACING_KEY],
        RADAR_IMAGE_MATRIX_KEY: dl_utils.stack_radar_fields(
            tuple_of_image_matrices),
        TARGET_MATRIX_KEY: target_matrix
    }

    if sounding_file_name is not None:
        example_dict.update({
            SOUNDING_FIELDS_KEY: sounding_field_names,
            SOUNDING_HEIGHTS_KEY: sounding_heights_m_agl,
            SOUNDING_MATRIX_KEY: sounding_matrix
        })

    return example_dict


def _create_3d_examples(
        radar_file_name_matrix, full_id_strings, storm_times_unix_sec,
        target_matrix, sounding_file_name=None, sounding_field_names=None):
    """Creates 3-D examples for one file time.

    :param radar_file_name_matrix: numpy array (F_r x H_r) of paths to storm-
        centered radar images.  Files will be read by
        `storm_images.read_storm_images`.
    :param full_id_strings: See doc for `_create_2d_examples`.
    :param storm_times_unix_sec: Same.
    :param target_matrix: Same.
    :param sounding_file_name: Same.
    :param sounding_field_names: Same.
    :return: example_dict: Same.
    """

    orig_full_id_strings = copy.deepcopy(full_id_strings)
    orig_storm_times_unix_sec = storm_times_unix_sec + 0

    print('Reading data from: "{0:s}"...'.format(radar_file_name_matrix[0, 0]))
    this_radar_image_dict = storm_images.read_storm_images(
        netcdf_file_name=radar_file_name_matrix[0, 0],
        full_id_strings_to_keep=full_id_strings,
        valid_times_to_keep_unix_sec=storm_times_unix_sec)

    if this_radar_image_dict is None:
        return None

    if sounding_file_name is None:
        sounding_matrix = None
        sounding_field_names = None
        sounding_heights_m_agl = None
    else:
        sounding_dict, this_radar_image_dict = _read_soundings(
            sounding_file_name=sounding_file_name,
            sounding_field_names=sounding_field_names,
            radar_image_dict=this_radar_image_dict)

        if this_radar_image_dict is None:
            return None
        if len(this_radar_image_dict[storm_images.FULL_IDS_KEY]) == 0:
            return None

        sounding_matrix = sounding_dict[soundings.SOUNDING_MATRIX_KEY]
        sounding_field_names = sounding_dict[soundings.FIELD_NAMES_KEY]
        sounding_heights_m_agl = sounding_dict[soundings.HEIGHT_LEVELS_KEY]

    full_id_strings = this_radar_image_dict[storm_images.FULL_IDS_KEY]
    storm_times_unix_sec = this_radar_image_dict[storm_images.VALID_TIMES_KEY]

    these_indices = tracking_utils.find_storm_objects(
        all_id_strings=orig_full_id_strings,
        all_times_unix_sec=orig_storm_times_unix_sec,
        id_strings_to_keep=full_id_strings,
        times_to_keep_unix_sec=storm_times_unix_sec, allow_missing=False)

    target_matrix = target_matrix[these_indices, :]

    num_radar_fields = radar_file_name_matrix.shape[0]
    num_radar_heights = radar_file_name_matrix.shape[1]
    tuple_of_4d_image_matrices = ()

    for k in range(num_radar_heights):
        tuple_of_3d_image_matrices = ()

        for j in range(num_radar_fields):
            if not j == k == 0:
                print('Reading data from: "{0:s}"...'.format(
                    radar_file_name_matrix[j, k]
                ))

                this_radar_image_dict = storm_images.read_storm_images(
                    netcdf_file_name=radar_file_name_matrix[j, k],
                    full_id_strings_to_keep=full_id_strings,
                    valid_times_to_keep_unix_sec=storm_times_unix_sec)

            tuple_of_3d_image_matrices += (
                this_radar_image_dict[storm_images.STORM_IMAGE_MATRIX_KEY],
            )

        tuple_of_4d_image_matrices += (
            dl_utils.stack_radar_fields(tuple_of_3d_image_matrices),
        )

    radar_field_names = [
        storm_images.image_file_name_to_field(f)
        for f in radar_file_name_matrix[:, 0]
    ]
    radar_heights_m_agl = numpy.array([
        storm_images.image_file_name_to_height(f)
        for f in radar_file_name_matrix[0, :]
    ], dtype=int)

    example_dict = {
        FULL_IDS_KEY: full_id_strings,
        STORM_TIMES_KEY: storm_times_unix_sec,
        RADAR_FIELDS_KEY: radar_field_names,
        RADAR_HEIGHTS_KEY: radar_heights_m_agl,
        ROTATED_GRIDS_KEY:
            this_radar_image_dict[storm_images.ROTATED_GRIDS_KEY],
        ROTATED_GRID_SPACING_KEY:
            this_radar_image_dict[storm_images.ROTATED_GRID_SPACING_KEY],
        RADAR_IMAGE_MATRIX_KEY: dl_utils.stack_radar_heights(
            tuple_of_4d_image_matrices),
        TARGET_MATRIX_KEY: target_matrix
    }

    if sounding_file_name is not None:
        example_dict.update({
            SOUNDING_FIELDS_KEY: sounding_field_names,
            SOUNDING_HEIGHTS_KEY: sounding_heights_m_agl,
            SOUNDING_MATRIX_KEY: sounding_matrix
        })

    return example_dict


def _create_2d3d_examples_myrorss(
        azimuthal_shear_file_names, reflectivity_file_names,
        full_id_strings, storm_times_unix_sec, target_matrix,
        sounding_file_name=None, sounding_field_names=None):
    """Creates hybrid 2D-3D examples for one file time.

    Fields in 2-D images: low-level and mid-level azimuthal shear
    Field in 3-D images: reflectivity

    :param azimuthal_shear_file_names: length-2 list of paths to storm-centered
        azimuthal-shear images.  The first (second) file should be (low)
        mid-level azimuthal shear.  Files will be read by
        `storm_images.read_storm_images`.
    :param reflectivity_file_names: length-H list of paths to storm-centered
        reflectivity images, where H = number of reflectivity heights.  Files
        will be read by `storm_images.read_storm_images`.
    :param full_id_strings: See doc for `_create_2d_examples`.
    :param storm_times_unix_sec: Same.
    :param target_matrix: Same.
    :param sounding_file_name: Same.
    :param sounding_field_names: Same.
    :return: example_dict: Same.
    """

    orig_full_id_strings = copy.deepcopy(full_id_strings)
    orig_storm_times_unix_sec = storm_times_unix_sec + 0

    print('Reading data from: "{0:s}"...'.format(reflectivity_file_names[0]))
    this_radar_image_dict = storm_images.read_storm_images(
        netcdf_file_name=reflectivity_file_names[0],
        full_id_strings_to_keep=full_id_strings,
        valid_times_to_keep_unix_sec=storm_times_unix_sec)

    if this_radar_image_dict is None:
        return None

    if sounding_file_name is None:
        sounding_matrix = None
        sounding_field_names = None
        sounding_heights_m_agl = None
    else:
        sounding_dict, this_radar_image_dict = _read_soundings(
            sounding_file_name=sounding_file_name,
            sounding_field_names=sounding_field_names,
            radar_image_dict=this_radar_image_dict)

        if this_radar_image_dict is None:
            return None
        if len(this_radar_image_dict[storm_images.FULL_IDS_KEY]) == 0:
            return None

        sounding_matrix = sounding_dict[soundings.SOUNDING_MATRIX_KEY]
        sounding_field_names = sounding_dict[soundings.FIELD_NAMES_KEY]
        sounding_heights_m_agl = sounding_dict[soundings.HEIGHT_LEVELS_KEY]

    full_id_strings = this_radar_image_dict[storm_images.FULL_IDS_KEY]
    storm_times_unix_sec = this_radar_image_dict[storm_images.VALID_TIMES_KEY]

    these_indices = tracking_utils.find_storm_objects(
        all_id_strings=orig_full_id_strings,
        all_times_unix_sec=orig_storm_times_unix_sec,
        id_strings_to_keep=full_id_strings,
        times_to_keep_unix_sec=storm_times_unix_sec, allow_missing=False)

    target_matrix = target_matrix[these_indices, :]

    azimuthal_shear_field_names = [
        storm_images.image_file_name_to_field(f)
        for f in azimuthal_shear_file_names
    ]
    reflectivity_heights_m_agl = numpy.array([
        storm_images.image_file_name_to_height(f)
        for f in reflectivity_file_names
    ], dtype=int)

    num_reflectivity_heights = len(reflectivity_file_names)
    tuple_of_image_matrices = ()

    for j in range(num_reflectivity_heights):
        if j != 0:
            print('Reading data from: "{0:s}"...'.format(
                reflectivity_file_names[j]
            ))

            this_radar_image_dict = storm_images.read_storm_images(
                netcdf_file_name=reflectivity_file_names[j],
                full_id_strings_to_keep=full_id_strings,
                valid_times_to_keep_unix_sec=storm_times_unix_sec)

        this_matrix = numpy.expand_dims(
            this_radar_image_dict[storm_images.STORM_IMAGE_MATRIX_KEY], axis=-1
        )
        tuple_of_image_matrices += (this_matrix,)

    example_dict = {
        FULL_IDS_KEY: full_id_strings,
        STORM_TIMES_KEY: storm_times_unix_sec,
        RADAR_FIELDS_KEY: azimuthal_shear_field_names,
        RADAR_HEIGHTS_KEY: reflectivity_heights_m_agl,
        ROTATED_GRIDS_KEY:
            this_radar_image_dict[storm_images.ROTATED_GRIDS_KEY],
        ROTATED_GRID_SPACING_KEY:
            this_radar_image_dict[storm_images.ROTATED_GRID_SPACING_KEY],
        REFL_IMAGE_MATRIX_KEY: dl_utils.stack_radar_heights(
            tuple_of_image_matrices),
        TARGET_MATRIX_KEY: target_matrix
    }

    if sounding_file_name is not None:
        example_dict.update({
            SOUNDING_FIELDS_KEY: sounding_field_names,
            SOUNDING_HEIGHTS_KEY: sounding_heights_m_agl,
            SOUNDING_MATRIX_KEY: sounding_matrix
        })

    num_az_shear_fields = len(azimuthal_shear_file_names)
    tuple_of_image_matrices = ()

    for j in range(num_az_shear_fields):
        print('Reading data from: "{0:s}"...'.format(
            azimuthal_shear_file_names[j]
        ))

        this_radar_image_dict = storm_images.read_storm_images(
            netcdf_file_name=azimuthal_shear_file_names[j],
            full_id_strings_to_keep=full_id_strings,
            valid_times_to_keep_unix_sec=storm_times_unix_sec)

        tuple_of_image_matrices += (
            this_radar_image_dict[storm_images.STORM_IMAGE_MATRIX_KEY],
        )

    example_dict.update({
        AZ_SHEAR_IMAGE_MATRIX_KEY: dl_utils.stack_radar_fields(
            tuple_of_image_matrices)
    })

    return example_dict


def _read_metadata_from_example_file(netcdf_file_name, include_soundings):
    """Reads metadata from file with input examples.

    :param netcdf_file_name: Path to input file.
    :param include_soundings: Boolean flag.  If True and file contains
        soundings, this method will return keys "sounding_field_names" and
        "sounding_heights_m_agl".  Otherwise, will not return said keys.

    :return: example_dict: Dictionary with the following keys (explained in doc
        to `write_example_file`).
    example_dict['full_id_strings']
    example_dict['storm_times_unix_sec']
    example_dict['radar_field_names']
    example_dict['radar_heights_m_agl']
    example_dict['rotated_grids']
    example_dict['rotated_grid_spacing_metres']
    example_dict['target_names']
    example_dict['sounding_field_names']
    example_dict['sounding_heights_m_agl']

    :return: netcdf_dataset: Instance of `netCDF4.Dataset`, which can be used to
        keep reading file.
    """

    netcdf_dataset = netCDF4.Dataset(netcdf_file_name)
    include_soundings = (
        include_soundings and
        SOUNDING_FIELDS_KEY in netcdf_dataset.variables
    )

    example_dict = {
        ROTATED_GRIDS_KEY: bool(getattr(netcdf_dataset, ROTATED_GRIDS_KEY)),
        TARGET_NAMES_KEY: [
            str(s) for s in
            netCDF4.chartostring(netcdf_dataset.variables[TARGET_NAMES_KEY][:])
        ],
        FULL_IDS_KEY: [
            str(s) for s in
            netCDF4.chartostring(netcdf_dataset.variables[FULL_IDS_KEY][:])
        ],
        STORM_TIMES_KEY: numpy.array(
            netcdf_dataset.variables[STORM_TIMES_KEY][:], dtype=int
        ),
        RADAR_FIELDS_KEY: [
            str(s) for s in
            netCDF4.chartostring(netcdf_dataset.variables[RADAR_FIELDS_KEY][:])
        ],
        RADAR_HEIGHTS_KEY: numpy.array(
            netcdf_dataset.variables[RADAR_HEIGHTS_KEY][:], dtype=int
        )
    }

    # TODO(thunderhoser): This is a HACK to deal with bad files.
    example_dict[TARGET_NAMES_KEY] = [
        n for n in example_dict[TARGET_NAMES_KEY] if n != ''
    ]

    if example_dict[ROTATED_GRIDS_KEY]:
        example_dict[ROTATED_GRID_SPACING_KEY] = getattr(
            netcdf_dataset, ROTATED_GRID_SPACING_KEY)
    else:
        example_dict[ROTATED_GRID_SPACING_KEY] = None

    if not include_soundings:
        return example_dict, netcdf_dataset

    example_dict.update({
        SOUNDING_FIELDS_KEY: [
            str(s) for s in netCDF4.chartostring(
                netcdf_dataset.variables[SOUNDING_FIELDS_KEY][:])
        ],
        SOUNDING_HEIGHTS_KEY:
            numpy.array(netcdf_dataset.variables[SOUNDING_HEIGHTS_KEY][:],
                        dtype=int)
    })

    return example_dict, netcdf_dataset


def _compare_metadata(netcdf_dataset, example_dict):
    """Compares metadata between existing NetCDF file and new batch of examples.

    This method contains a large number of `assert` statements.  If any of the
    `assert` statements fails, this method will error out.

    :param netcdf_dataset: Instance of `netCDF4.Dataset`.
    :param example_dict: See doc for `write_examples_with_3d_radar`.
    :raises: ValueError: if the two sets have different metadata.
    """

    include_soundings = SOUNDING_MATRIX_KEY in example_dict

    orig_example_dict = {
        TARGET_NAMES_KEY: [
            str(s) for s in
            netCDF4.chartostring(netcdf_dataset.variables[TARGET_NAMES_KEY][:])
        ],
        ROTATED_GRIDS_KEY: bool(getattr(netcdf_dataset, ROTATED_GRIDS_KEY)),
        RADAR_FIELDS_KEY: [
            str(s) for s in netCDF4.chartostring(
                netcdf_dataset.variables[RADAR_FIELDS_KEY][:])
        ],
        RADAR_HEIGHTS_KEY: numpy.array(
            netcdf_dataset.variables[RADAR_HEIGHTS_KEY][:], dtype=int
        )
    }

    if example_dict[ROTATED_GRIDS_KEY]:
        orig_example_dict[ROTATED_GRID_SPACING_KEY] = int(
            getattr(netcdf_dataset, ROTATED_GRID_SPACING_KEY)
        )

    if include_soundings:
        orig_example_dict[SOUNDING_FIELDS_KEY] = [
            str(s) for s in netCDF4.chartostring(
                netcdf_dataset.variables[SOUNDING_FIELDS_KEY][:])
        ]
        orig_example_dict[SOUNDING_HEIGHTS_KEY] = numpy.array(
            netcdf_dataset.variables[SOUNDING_HEIGHTS_KEY][:], dtype=int
        )

    for this_key in orig_example_dict:
        if isinstance(example_dict[this_key], numpy.ndarray):
            if numpy.array_equal(example_dict[this_key],
                                 orig_example_dict[this_key]):
                continue
        else:
            if example_dict[this_key] == orig_example_dict[this_key]:
                continue

        error_string = (
            '\n"{0:s}" in existing NetCDF file:\n{1:s}\n\n"{0:s}" in new batch '
            'of examples:\n{2:s}\n\n'
        ).format(
            this_key, str(orig_example_dict[this_key]),
            str(example_dict[this_key])
        )

        raise ValueError(error_string)


def _filter_examples_by_class(target_values, downsampling_dict,
                              test_mode=False):
    """Filters examples by target value.

    E = number of examples

    :param target_values: length-E numpy array of target values (integer class
        labels).
    :param downsampling_dict: Dictionary, where each key is the integer
        ID for a target class (-2 for "dead storm") and the corresponding value
        is the number of examples desired from said class.  If
        `downsampling_dict is None`, `example_dict` will be returned
        without modification.
    :param test_mode: Never mind.  Just leave this alone.
    :return: indices_to_keep: 1-D numpy array with indices of examples to keep.
        These are all integers in [0, E - 1].
    """

    num_examples = len(target_values)

    if downsampling_dict is None:
        return numpy.linspace(0, num_examples - 1, num=num_examples, dtype=int)

    indices_to_keep = numpy.array([], dtype=int)
    class_keys = list(downsampling_dict.keys())

    for this_class in class_keys:
        this_num_storm_objects = downsampling_dict[this_class]
        these_indices = numpy.where(target_values == this_class)[0]

        this_num_storm_objects = min(
            [this_num_storm_objects, len(these_indices)]
        )
        if this_num_storm_objects == 0:
            continue

        if test_mode:
            these_indices = these_indices[:this_num_storm_objects]
        else:
            these_indices = numpy.random.choice(
                these_indices, size=this_num_storm_objects, replace=False)

        indices_to_keep = numpy.concatenate((indices_to_keep, these_indices))

    return indices_to_keep


def _file_name_to_batch_number(example_file_name):
    """Parses batch number from file.

    :param example_file_name: See doc for `find_example_file`.
    :return: batch_number: Integer.
    :raises: ValueError: if batch number cannot be parsed from file name.
    """

    pathless_file_name = os.path.split(example_file_name)[-1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]
    return int(extensionless_file_name.split('input_examples_batch')[-1])


def _check_target_vars(target_names):
    """Error-checks list of target variables.

    Target variables must all have the same mean lead time (average of min and
    max lead times) and event type (tornado or wind).

    :param target_names: 1-D list with names of target variables.  Each must be
        accepted by `target_val_utils.target_name_to_params`.
    :return: mean_lead_time_seconds: Mean lead time (shared by all target
        variables).
    :return: event_type_string: Event type.
    :raises: ValueError: if target variables do not all have the same mean lead
        time or event type.
    """

    error_checking.assert_is_string_list(target_names)
    error_checking.assert_is_numpy_array(
        numpy.array(target_names), num_dimensions=1
    )

    num_target_vars = len(target_names)
    mean_lead_times = numpy.full(num_target_vars, -1, dtype=int)
    event_type_strings = numpy.full(num_target_vars, '', dtype=object)

    for k in range(num_target_vars):
        this_param_dict = target_val_utils.target_name_to_params(
            target_names[k]
        )

        event_type_strings[k] = this_param_dict[target_val_utils.EVENT_TYPE_KEY]

        mean_lead_times[k] = int(numpy.round(
            (this_param_dict[target_val_utils.MAX_LEAD_TIME_KEY] +
             this_param_dict[target_val_utils.MIN_LEAD_TIME_KEY])
            / 2
        ))

    if len(numpy.unique(mean_lead_times)) != 1:
        error_string = (
            'Target variables (listed below) have different mean lead times.'
            '\n{0:s}'
        ).format(str(target_names))

        raise ValueError(error_string)

    if len(numpy.unique(event_type_strings)) != 1:
        error_string = (
            'Target variables (listed below) have different event types.\n{0:s}'
        ).format(str(target_names))

        raise ValueError(error_string)

    return mean_lead_times[0], event_type_strings[0]


def _check_layer_operation(example_dict, operation_dict):
    """Error-checks layer operation.

    Such operations are used for dimensionality reduction (to convert radar data
    from 3-D to 2-D).

    :param example_dict: See doc for `reduce_examples_3d_to_2d`.
    :param operation_dict: Dictionary with the following keys.
    operation_dict["radar_field_name"]: Field to which operation will be
        applied.
    operation_dict["operation_name"]: Name of operation (must be in list
        `VALID_LAYER_OPERATION_NAMES`).
    operation_dict["min_height_m_agl"]: Minimum height of layer over which
        operation will be applied.
    operation_dict["max_height_m_agl"]: Max height of layer over which operation
        will be applied.

    :raises: ValueError: if something is wrong with the operation params.
    """

    if operation_dict[RADAR_FIELD_KEY] in AZIMUTHAL_SHEAR_FIELD_NAMES:
        error_string = (
            'Layer operations cannot be applied to azimuthal-shear fields '
            '(such as "{0:s}").'
        ).format(operation_dict[RADAR_FIELD_KEY])

        raise ValueError(error_string)

    if (operation_dict[RADAR_FIELD_KEY] == radar_utils.REFL_NAME
            and REFL_IMAGE_MATRIX_KEY in example_dict):
        pass

    else:
        if (operation_dict[RADAR_FIELD_KEY]
                not in example_dict[RADAR_FIELDS_KEY]):
            error_string = (
                '\n{0:s}\nExamples contain only radar fields listed above, '
                'which do not include "{1:s}".'
            ).format(
                str(example_dict[RADAR_FIELDS_KEY]),
                operation_dict[RADAR_FIELD_KEY]
            )

            raise ValueError(error_string)

    if operation_dict[OPERATION_NAME_KEY] not in VALID_LAYER_OPERATION_NAMES:
        error_string = (
            '\n{0:s}\nValid operations (listed above) do not include '
            '"{1:s}".'
        ).format(
            str(VALID_LAYER_OPERATION_NAMES), operation_dict[OPERATION_NAME_KEY]
        )

        raise ValueError(error_string)

    min_height_m_agl = operation_dict[MIN_HEIGHT_KEY]
    max_height_m_agl = operation_dict[MAX_HEIGHT_KEY]

    error_checking.assert_is_geq(
        min_height_m_agl, numpy.min(example_dict[RADAR_HEIGHTS_KEY])
    )
    error_checking.assert_is_leq(
        max_height_m_agl, numpy.max(example_dict[RADAR_HEIGHTS_KEY])
    )
    error_checking.assert_is_greater(max_height_m_agl, min_height_m_agl)


def _apply_layer_operation(example_dict, operation_dict):
    """Applies layer operation to radar data.

    :param example_dict: See doc for `reduce_examples_3d_to_2d`.
    :param operation_dict: See doc for `_check_layer_operation`.
    :return: new_radar_matrix: E-by-M-by-N numpy array resulting from layer
        operation.
    """

    _check_layer_operation(example_dict=example_dict,
                           operation_dict=operation_dict)

    height_diffs_metres = (
        example_dict[RADAR_HEIGHTS_KEY] - operation_dict[MIN_HEIGHT_KEY]
    ).astype(float)
    height_diffs_metres[height_diffs_metres > 0] = -numpy.inf
    min_height_index = numpy.argmax(height_diffs_metres)

    height_diffs_metres = (
        operation_dict[MAX_HEIGHT_KEY] - example_dict[RADAR_HEIGHTS_KEY]
    ).astype(float)
    height_diffs_metres[height_diffs_metres > 0] = -numpy.inf
    max_height_index = numpy.argmax(height_diffs_metres)

    operation_dict[MIN_HEIGHT_KEY] = example_dict[
        RADAR_HEIGHTS_KEY][min_height_index]
    operation_dict[MAX_HEIGHT_KEY] = example_dict[
        RADAR_HEIGHTS_KEY][max_height_index]

    operation_name = operation_dict[OPERATION_NAME_KEY]
    operation_function = OPERATION_NAME_TO_FUNCTION_DICT[operation_name]

    if REFL_IMAGE_MATRIX_KEY in example_dict:
        orig_matrix = example_dict[REFL_IMAGE_MATRIX_KEY][
            ..., min_height_index:(max_height_index + 1), 0]
    else:
        field_index = example_dict[RADAR_FIELDS_KEY].index(
            operation_dict[RADAR_FIELD_KEY])

        orig_matrix = example_dict[RADAR_IMAGE_MATRIX_KEY][
            ..., min_height_index:(max_height_index + 1), field_index]

    return operation_function(orig_matrix, axis=-1), operation_dict


def _subset_radar_data(
        example_dict, netcdf_dataset_object, example_indices_to_keep,
        field_names_to_keep, heights_to_keep_m_agl, num_rows_to_keep,
        num_columns_to_keep):
    """Subsets radar data by field, height, and horizontal extent.

    If the file contains both 2-D shear images and 3-D reflectivity images (like
    MYRORSS data):

    - `field_names_to_keep` will be interpreted as a list of shear fields to
      keep.  If None, all shear fields will be kept.
    - `heights_to_keep_m_agl` will be interpreted as a list of reflectivity
      heights to keep.  If None, all reflectivity heights will be kept.

    If the file contains only 2-D images, `field_names_to_keep` and
    `heights_to_keep_m_agl` will be considered together, as a list of
    field/height pairs to keep.  If either argument is None, then all
    field-height pairs will be kept.

    If the file contains only 3-D images, `field_names_to_keep` and
    `heights_to_keep_m_agl` will be considered separately:

    - `field_names_to_keep` will be interpreted as a list of fields to keep.  If
        None, all fields will be kept.
    - `heights_to_keep_m_agl` will be interpreted as a list of heights to keep.
      If None, all heights will be kept.

    :param example_dict: See output doc for `_read_metadata_from_example_file`.
    :param netcdf_dataset_object: Same.
    :param example_indices_to_keep: 1-D numpy array with indices of examples
        (storm objects) to keep.  These are examples in `netcdf_dataset_object`
        for which radar data will be added to `example_dict`.
    :param field_names_to_keep: See discussion above.
    :param heights_to_keep_m_agl: See discussion above.
    :param num_rows_to_keep: Number of grid rows to keep.  Images will be
        center-cropped (i.e., rows will be removed from the edges) to meet the
        desired number of rows.  If None, all rows will be kept.
    :param num_columns_to_keep: Same as above but for columns.
    :return: example_dict: Same as input but with the following exceptions.

    [1] Keys "radar_field_names" and "radar_heights_m_agl" may have different
        values.
    [2] If file contains both 2-D and 3-D images, dictionary now contains keys
        "reflectivity_image_matrix_dbz" and "az_shear_image_matrix_s01".
    [3] If file contains only 2-D or only 3-D images, dictionary now contains
        key "radar_image_matrix".
    """

    if field_names_to_keep is None:
        field_names_to_keep = copy.deepcopy(example_dict[RADAR_FIELDS_KEY])
    if heights_to_keep_m_agl is None:
        heights_to_keep_m_agl = example_dict[RADAR_HEIGHTS_KEY] + 0

    error_checking.assert_is_numpy_array(
        numpy.array(field_names_to_keep), num_dimensions=1
    )

    heights_to_keep_m_agl = numpy.round(heights_to_keep_m_agl).astype(int)
    error_checking.assert_is_numpy_array(
        heights_to_keep_m_agl, num_dimensions=1)

    if RADAR_IMAGE_MATRIX_KEY in netcdf_dataset_object.variables:
        radar_matrix = numpy.array(
            netcdf_dataset_object.variables[RADAR_IMAGE_MATRIX_KEY][
                example_indices_to_keep, ...
            ],
            dtype=float
        )

        num_radar_dimensions = len(radar_matrix.shape) - 2

        if num_radar_dimensions == 2:
            these_indices = [
                numpy.where(numpy.logical_and(
                    example_dict[RADAR_FIELDS_KEY] == f,
                    example_dict[RADAR_HEIGHTS_KEY] == h
                ))[0][0]
                for f, h in zip(field_names_to_keep, heights_to_keep_m_agl)
            ]

            these_indices = numpy.array(these_indices, dtype=int)
            radar_matrix = radar_matrix[..., these_indices]
        else:
            these_field_indices = numpy.array([
                example_dict[RADAR_FIELDS_KEY].index(f)
                for f in field_names_to_keep
            ], dtype=int)

            radar_matrix = radar_matrix[..., these_field_indices]

            these_height_indices = numpy.array([
                numpy.where(example_dict[RADAR_HEIGHTS_KEY] == h)[0][0]
                for h in heights_to_keep_m_agl
            ], dtype=int)

            radar_matrix = radar_matrix[..., these_height_indices, :]

        radar_matrix = storm_images.downsize_storm_images(
            storm_image_matrix=radar_matrix,
            radar_field_name=field_names_to_keep[0],
            num_rows_to_keep=num_rows_to_keep,
            num_columns_to_keep=num_columns_to_keep)

        example_dict[RADAR_IMAGE_MATRIX_KEY] = radar_matrix

    else:
        reflectivity_matrix_dbz = numpy.array(
            netcdf_dataset_object.variables[REFL_IMAGE_MATRIX_KEY][
                example_indices_to_keep, ...
            ],
            dtype=float
        )

        reflectivity_matrix_dbz = numpy.expand_dims(
            reflectivity_matrix_dbz, axis=-1
        )

        azimuthal_shear_matrix_s01 = numpy.array(
            netcdf_dataset_object.variables[AZ_SHEAR_IMAGE_MATRIX_KEY][
                example_indices_to_keep, ...
            ],
            dtype=float
        )

        these_height_indices = numpy.array([
            numpy.where(example_dict[RADAR_HEIGHTS_KEY] == h)[0][0]
            for h in heights_to_keep_m_agl
        ], dtype=int)

        reflectivity_matrix_dbz = reflectivity_matrix_dbz[
            ..., these_height_indices, :]

        these_field_indices = numpy.array([
            example_dict[RADAR_FIELDS_KEY].index(f)
            for f in field_names_to_keep
        ], dtype=int)

        azimuthal_shear_matrix_s01 = azimuthal_shear_matrix_s01[
            ..., these_field_indices]

        reflectivity_matrix_dbz = storm_images.downsize_storm_images(
            storm_image_matrix=reflectivity_matrix_dbz,
            radar_field_name=radar_utils.REFL_NAME,
            num_rows_to_keep=num_rows_to_keep,
            num_columns_to_keep=num_columns_to_keep)

        azimuthal_shear_matrix_s01 = storm_images.downsize_storm_images(
            storm_image_matrix=azimuthal_shear_matrix_s01,
            radar_field_name=field_names_to_keep[0],
            num_rows_to_keep=num_rows_to_keep,
            num_columns_to_keep=num_columns_to_keep)

        example_dict[REFL_IMAGE_MATRIX_KEY] = reflectivity_matrix_dbz
        example_dict[AZ_SHEAR_IMAGE_MATRIX_KEY] = azimuthal_shear_matrix_s01

    example_dict[RADAR_FIELDS_KEY] = field_names_to_keep
    example_dict[RADAR_HEIGHTS_KEY] = heights_to_keep_m_agl
    return example_dict


def _subset_sounding_data(
        example_dict, netcdf_dataset_object, example_indices_to_keep,
        field_names_to_keep, heights_to_keep_m_agl):
    """Subsets sounding data by field and height.

    :param example_dict: See doc for `_subset_radar_data`.
    :param netcdf_dataset_object: Same.
    :param example_indices_to_keep: Same.
    :param field_names_to_keep: 1-D list of field names to keep.  If None, will
        keep all fields.
    :param heights_to_keep_m_agl: 1-D numpy array of heights to keep.  If None,
        will keep all heights.
    :return: example_dict: Same as input but with the following exceptions.

    [1] Keys "sounding_field_names" and "sounding_heights_m_agl" may have
        different values.
    [2] Key "sounding_matrix" has been added.
    """

    if field_names_to_keep is None:
        field_names_to_keep = copy.deepcopy(example_dict[SOUNDING_FIELDS_KEY])
    if heights_to_keep_m_agl is None:
        heights_to_keep_m_agl = example_dict[SOUNDING_HEIGHTS_KEY] + 0

    error_checking.assert_is_numpy_array(
        numpy.array(field_names_to_keep), num_dimensions=1
    )

    heights_to_keep_m_agl = numpy.round(heights_to_keep_m_agl).astype(int)
    error_checking.assert_is_numpy_array(
        heights_to_keep_m_agl, num_dimensions=1)

    sounding_matrix = numpy.array(
        netcdf_dataset_object.variables[SOUNDING_MATRIX_KEY][
            example_indices_to_keep, ...
        ],
        dtype=float
    )

    # TODO(thunderhoser): This is a HACK.
    spfh_index = example_dict[SOUNDING_FIELDS_KEY].index(
        soundings.SPECIFIC_HUMIDITY_NAME)
    temp_index = example_dict[SOUNDING_FIELDS_KEY].index(
        soundings.TEMPERATURE_NAME)
    pressure_index = example_dict[SOUNDING_FIELDS_KEY].index(
        soundings.PRESSURE_NAME)
    theta_v_index = example_dict[SOUNDING_FIELDS_KEY].index(
        soundings.VIRTUAL_POTENTIAL_TEMPERATURE_NAME)

    sounding_matrix[..., spfh_index][
        numpy.isnan(sounding_matrix[..., spfh_index])
    ] = 0.

    nan_example_indices, nan_height_indices = numpy.where(numpy.isnan(
        sounding_matrix[..., theta_v_index]
    ))

    if len(nan_example_indices) > 0:
        this_temp_matrix_kelvins = sounding_matrix[..., temp_index][
            nan_example_indices, nan_height_indices]

        this_pressure_matrix_pa = sounding_matrix[..., pressure_index][
            nan_example_indices, nan_height_indices]

        this_thetav_matrix_kelvins = (
            temp_conversion.temperatures_to_potential_temperatures(
                temperatures_kelvins=this_temp_matrix_kelvins,
                total_pressures_pascals=this_pressure_matrix_pa)
        )

        sounding_matrix[..., theta_v_index][
            nan_example_indices, nan_height_indices
        ] = this_thetav_matrix_kelvins

    these_indices = numpy.array([
        example_dict[SOUNDING_FIELDS_KEY].index(f)
        for f in field_names_to_keep
    ], dtype=int)

    sounding_matrix = sounding_matrix[..., these_indices]

    these_indices = numpy.array([
        numpy.where(example_dict[SOUNDING_HEIGHTS_KEY] == h)[0][0]
        for h in heights_to_keep_m_agl
    ], dtype=int)

    sounding_matrix = sounding_matrix[..., these_indices, :]

    example_dict[SOUNDING_FIELDS_KEY] = field_names_to_keep
    example_dict[SOUNDING_HEIGHTS_KEY] = heights_to_keep_m_agl
    example_dict[SOUNDING_MATRIX_KEY] = sounding_matrix

    return example_dict


def find_storm_images_2d(
        top_directory_name, radar_source, radar_field_names,
        first_spc_date_string, last_spc_date_string, radar_heights_m_agl=None,
        reflectivity_heights_m_agl=None):
    """Locates files with 2-D storm-centered radar images.

    D = number of SPC dates in time period (`first_spc_date_string`...
        `last_spc_date_string`)

    :param top_directory_name: Name of top-level directory.  Files therein will
        be found by `storm_images.find_storm_image_file`.
    :param radar_source: Data source (must be accepted by
        `radar_utils.check_data_source`).
    :param radar_field_names: 1-D list of radar fields.  Each item must be
        accepted by `radar_utils.check_field_name`.
    :param first_spc_date_string: First SPC date (format "yyyymmdd").  This
        method will locate files from `first_spc_date_string`...
        `last_spc_date_string`.
    :param last_spc_date_string: Same.
    :param radar_heights_m_agl: [used only if radar_source = "gridrad"]
        1-D numpy array of radar heights (metres above ground level).  These
        heights apply to all radar fields.
    :param reflectivity_heights_m_agl: [used only if radar_source != "gridrad"]
        1-D numpy array of reflectivity heights (metres above ground level).
        These heights do not apply to other radar fields.
    :return: radar_file_name_matrix: D-by-C numpy array of file paths.
    """

    radar_utils.check_data_source(radar_source)
    first_spc_date_unix_sec = time_conversion.spc_date_string_to_unix_sec(
        first_spc_date_string)
    last_spc_date_unix_sec = time_conversion.spc_date_string_to_unix_sec(
        last_spc_date_string)

    if radar_source == radar_utils.GRIDRAD_SOURCE_ID:
        storm_image_file_dict = storm_images.find_many_files_gridrad(
            top_directory_name=top_directory_name,
            radar_field_names=radar_field_names,
            radar_heights_m_agl=radar_heights_m_agl,
            start_time_unix_sec=first_spc_date_unix_sec,
            end_time_unix_sec=last_spc_date_unix_sec,
            one_file_per_time_step=False, raise_error_if_all_missing=True)
    else:
        storm_image_file_dict = storm_images.find_many_files_myrorss_or_mrms(
            top_directory_name=top_directory_name, radar_source=radar_source,
            radar_field_names=radar_field_names,
            reflectivity_heights_m_agl=reflectivity_heights_m_agl,
            start_time_unix_sec=first_spc_date_unix_sec,
            end_time_unix_sec=last_spc_date_unix_sec,
            one_file_per_time_step=False,
            raise_error_if_all_missing=True, raise_error_if_any_missing=False)

    radar_file_name_matrix = storm_image_file_dict[
        storm_images.IMAGE_FILE_NAMES_KEY]
    num_file_times = radar_file_name_matrix.shape[0]

    if radar_source == radar_utils.GRIDRAD_SOURCE_ID:
        num_field_height_pairs = (
            radar_file_name_matrix.shape[1] * radar_file_name_matrix.shape[2]
        )
        radar_file_name_matrix = numpy.reshape(
            radar_file_name_matrix, (num_file_times, num_field_height_pairs)
        )

    time_missing_indices = numpy.unique(
        numpy.where(radar_file_name_matrix == '')[0]
    )
    return numpy.delete(
        radar_file_name_matrix, time_missing_indices, axis=0)


def find_storm_images_3d(
        top_directory_name, radar_source, radar_field_names,
        radar_heights_m_agl, first_spc_date_string, last_spc_date_string):
    """Locates files with 3-D storm-centered radar images.

    D = number of SPC dates in time period (`first_spc_date_string`...
        `last_spc_date_string`)

    :param top_directory_name: See doc for `find_storm_images_2d`.
    :param radar_source: Same.
    :param radar_field_names: List (length F_r) of radar fields.  Each item must
        be accepted by `radar_utils.check_field_name`.
    :param radar_heights_m_agl: numpy array (length H_r) of radar heights
        (metres above ground level).
    :param first_spc_date_string: First SPC date (format "yyyymmdd").  This
        method will locate files from `first_spc_date_string`...
        `last_spc_date_string`.
    :param last_spc_date_string: Same.
    :return: radar_file_name_matrix: numpy array (D x F_r x H_r) of file paths.
    """

    radar_utils.check_data_source(radar_source)
    first_spc_date_unix_sec = time_conversion.spc_date_string_to_unix_sec(
        first_spc_date_string)
    last_spc_date_unix_sec = time_conversion.spc_date_string_to_unix_sec(
        last_spc_date_string)

    if radar_source == radar_utils.GRIDRAD_SOURCE_ID:
        file_dict = storm_images.find_many_files_gridrad(
            top_directory_name=top_directory_name,
            radar_field_names=radar_field_names,
            radar_heights_m_agl=radar_heights_m_agl,
            start_time_unix_sec=first_spc_date_unix_sec,
            end_time_unix_sec=last_spc_date_unix_sec,
            one_file_per_time_step=False, raise_error_if_all_missing=True)
    else:
        file_dict = storm_images.find_many_files_myrorss_or_mrms(
            top_directory_name=top_directory_name, radar_source=radar_source,
            radar_field_names=[radar_utils.REFL_NAME],
            reflectivity_heights_m_agl=radar_heights_m_agl,
            start_time_unix_sec=first_spc_date_unix_sec,
            end_time_unix_sec=last_spc_date_unix_sec,
            one_file_per_time_step=False,
            raise_error_if_all_missing=True, raise_error_if_any_missing=False)

    radar_file_name_matrix = file_dict[storm_images.IMAGE_FILE_NAMES_KEY]
    num_file_times = radar_file_name_matrix.shape[0]

    if radar_source != radar_utils.GRIDRAD_SOURCE_ID:
        radar_file_name_matrix = numpy.reshape(
            radar_file_name_matrix,
            (num_file_times, 1, len(radar_heights_m_agl))
        )

    time_missing_indices = numpy.unique(
        numpy.where(radar_file_name_matrix == '')[0]
    )
    return numpy.delete(
        radar_file_name_matrix, time_missing_indices, axis=0)


def find_storm_images_2d3d_myrorss(
        top_directory_name, first_spc_date_string, last_spc_date_string,
        reflectivity_heights_m_agl):
    """Locates files with 2-D and 3-D storm-centered radar images.

    Fields in 2-D images: low-level and mid-level azimuthal shear
    Field in 3-D images: reflectivity

    D = number of SPC dates in time period (`first_spc_date_string`...
        `last_spc_date_string`)

    :param top_directory_name: See doc for `find_storm_images_2d`.
    :param first_spc_date_string: Same.
    :param last_spc_date_string: Same.
    :param reflectivity_heights_m_agl: Same.
    :return: az_shear_file_name_matrix: D-by-2 numpy array of file paths.  Files
        in column 0 are low-level az shear; files in column 1 are mid-level az
        shear.
    :return: reflectivity_file_name_matrix: D-by-H numpy array of file paths,
        where H = number of reflectivity heights.
    """

    first_spc_date_unix_sec = time_conversion.spc_date_string_to_unix_sec(
        first_spc_date_string)
    last_spc_date_unix_sec = time_conversion.spc_date_string_to_unix_sec(
        last_spc_date_string)

    field_names = AZIMUTHAL_SHEAR_FIELD_NAMES + [radar_utils.REFL_NAME]

    storm_image_file_dict = storm_images.find_many_files_myrorss_or_mrms(
        top_directory_name=top_directory_name,
        radar_source=radar_utils.MYRORSS_SOURCE_ID,
        radar_field_names=field_names,
        reflectivity_heights_m_agl=reflectivity_heights_m_agl,
        start_time_unix_sec=first_spc_date_unix_sec,
        end_time_unix_sec=last_spc_date_unix_sec,
        one_file_per_time_step=False,
        raise_error_if_all_missing=True, raise_error_if_any_missing=False)

    radar_file_name_matrix = storm_image_file_dict[
        storm_images.IMAGE_FILE_NAMES_KEY]
    time_missing_indices = numpy.unique(
        numpy.where(radar_file_name_matrix == '')[0]
    )
    radar_file_name_matrix = numpy.delete(
        radar_file_name_matrix, time_missing_indices, axis=0)

    return radar_file_name_matrix[:, :2], radar_file_name_matrix[:, 2:]


def find_sounding_files(
        top_sounding_dir_name, radar_file_name_matrix, target_names,
        lag_time_for_convective_contamination_sec):
    """Locates files with storm-centered soundings.

    D = number of SPC dates in time period

    :param top_sounding_dir_name: Name of top-level directory.  Files therein
        will be found by `soundings.find_sounding_file`.
    :param radar_file_name_matrix: numpy array created by either
        `find_storm_images_2d` or `find_storm_images_3d`.  Length of the first
        axis is D.
    :param target_names: See doc for `_check_target_vars`.
    :param lag_time_for_convective_contamination_sec: See doc for
        `soundings.read_soundings`.
    :return: sounding_file_names: length-D list of file paths.
    """

    error_checking.assert_is_numpy_array(radar_file_name_matrix)
    num_file_dimensions = len(radar_file_name_matrix.shape)
    error_checking.assert_is_geq(num_file_dimensions, 2)
    error_checking.assert_is_leq(num_file_dimensions, 3)

    mean_lead_time_seconds = _check_target_vars(target_names)[0]

    num_file_times = radar_file_name_matrix.shape[0]
    sounding_file_names = [''] * num_file_times

    for i in range(num_file_times):
        if num_file_dimensions == 2:
            this_file_name = radar_file_name_matrix[i, 0]
        else:
            this_file_name = radar_file_name_matrix[i, 0, 0]

        this_time_unix_sec, this_spc_date_string = (
            storm_images.image_file_name_to_time(this_file_name)
        )

        sounding_file_names[i] = soundings.find_sounding_file(
            top_directory_name=top_sounding_dir_name,
            spc_date_string=this_spc_date_string,
            lead_time_seconds=mean_lead_time_seconds,
            lag_time_for_convective_contamination_sec=
            lag_time_for_convective_contamination_sec,
            init_time_unix_sec=this_time_unix_sec, raise_error_if_missing=True)

    return sounding_file_names


def find_target_files(top_target_dir_name, radar_file_name_matrix,
                      target_names):
    """Locates files with target values (storm-hazard indicators).

    D = number of SPC dates in time period

    :param top_target_dir_name: Name of top-level directory.  Files therein
        will be found by `target_val_utils.find_target_file`.
    :param radar_file_name_matrix: numpy array created by either
        `find_storm_images_2d` or `find_storm_images_3d`.  Length of the first
        axis is D.
    :param target_names: See doc for `_check_target_vars`.
    :return: target_file_names: length-D list of file paths.
    """

    error_checking.assert_is_numpy_array(radar_file_name_matrix)
    num_file_dimensions = len(radar_file_name_matrix.shape)
    error_checking.assert_is_geq(num_file_dimensions, 2)
    error_checking.assert_is_leq(num_file_dimensions, 3)

    event_type_string = _check_target_vars(target_names)[-1]

    num_file_times = radar_file_name_matrix.shape[0]
    target_file_names = [''] * num_file_times

    for i in range(num_file_times):
        if num_file_dimensions == 2:
            this_file_name = radar_file_name_matrix[i, 0]
        else:
            this_file_name = radar_file_name_matrix[i, 0, 0]

        _, this_spc_date_string = storm_images.image_file_name_to_time(
            this_file_name)

        target_file_names[i] = target_val_utils.find_target_file(
            top_directory_name=top_target_dir_name,
            event_type_string=event_type_string,
            spc_date_string=this_spc_date_string, raise_error_if_missing=False)

        if os.path.isfile(target_file_names[i]):
            continue

        target_file_names[i] = None

    return target_file_names


def subset_examples(example_dict, indices_to_keep, create_new_dict=False):
    """Subsets examples in dictionary.

    :param example_dict: See doc for `write_example_file`.
    :param indices_to_keep: 1-D numpy array with indices of examples to keep.
    :param create_new_dict: Boolean flag.  If True, this method will create a
        new dictionary, leaving the input dictionary untouched.
    :return: example_dict: Same as input, but possibly with fewer examples.
    """

    error_checking.assert_is_integer_numpy_array(indices_to_keep)
    error_checking.assert_is_numpy_array(indices_to_keep, num_dimensions=1)
    error_checking.assert_is_boolean(create_new_dict)

    if not create_new_dict:
        for this_key in MAIN_KEYS:
            optional_key_missing = (
                this_key not in REQUIRED_MAIN_KEYS
                and this_key not in example_dict
            )

            if optional_key_missing:
                continue

            if this_key == TARGET_MATRIX_KEY:
                if this_key in example_dict:
                    example_dict[this_key] = (
                        example_dict[this_key][indices_to_keep, ...]
                    )
                else:
                    example_dict[TARGET_VALUES_KEY] = (
                        example_dict[TARGET_VALUES_KEY][indices_to_keep]
                    )

                continue

            if this_key == FULL_IDS_KEY:
                example_dict[this_key] = [
                    example_dict[this_key][k] for k in indices_to_keep
                ]
            else:
                example_dict[this_key] = example_dict[this_key][
                    indices_to_keep, ...]

        return example_dict

    new_example_dict = {}

    for this_key in METADATA_KEYS:
        sounding_key_missing = (
            this_key in [SOUNDING_FIELDS_KEY, SOUNDING_HEIGHTS_KEY]
            and this_key not in example_dict
        )

        if sounding_key_missing:
            continue

        if this_key == TARGET_NAMES_KEY:
            if this_key in example_dict:
                new_example_dict[this_key] = example_dict[this_key]
            else:
                new_example_dict[TARGET_NAME_KEY] = example_dict[
                    TARGET_NAME_KEY]

            continue

        new_example_dict[this_key] = example_dict[this_key]

    for this_key in MAIN_KEYS:
        optional_key_missing = (
            this_key not in REQUIRED_MAIN_KEYS
            and this_key not in example_dict
        )

        if optional_key_missing:
            continue

        if this_key == TARGET_MATRIX_KEY:
            if this_key in example_dict:
                new_example_dict[this_key] = (
                    example_dict[this_key][indices_to_keep, ...]
                )
            else:
                new_example_dict[TARGET_VALUES_KEY] = (
                    example_dict[TARGET_VALUES_KEY][indices_to_keep]
                )

            continue

        if this_key == FULL_IDS_KEY:
            new_example_dict[this_key] = [
                example_dict[this_key][k] for k in indices_to_keep
            ]
        else:
            new_example_dict[this_key] = example_dict[this_key][
                indices_to_keep, ...]

    return new_example_dict


def find_example_file(
        top_directory_name, shuffled=True, spc_date_string=None,
        batch_number=None, raise_error_if_missing=True):
    """Looks for file with input examples.

    If `shuffled = True`, this method looks for a file with shuffled examples
    (from many different times).  If `shuffled = False`, this method looks for a
    file with examples from one SPC date.

    :param top_directory_name: Name of top-level directory with input examples.
    :param shuffled: Boolean flag.  The role of this flag is explained in the
        general discussion above.
    :param spc_date_string: [used only if `shuffled = False`]
        SPC date (format "yyyymmdd").
    :param batch_number: [used only if `shuffled = True`]
        Batch number (integer).
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing = True`, this method will error out.
    :return: example_file_name: Path to file with input examples.  If file is
        missing and `raise_error_if_missing = False`, this is the *expected*
        path.
    :raises: ValueError: if file is missing and `raise_error_if_missing = True`.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_boolean(shuffled)
    error_checking.assert_is_boolean(raise_error_if_missing)

    if shuffled:
        error_checking.assert_is_integer(batch_number)
        error_checking.assert_is_geq(batch_number, 0)

        first_batch_number = int(number_rounding.floor_to_nearest(
            batch_number, NUM_BATCHES_PER_DIRECTORY))
        last_batch_number = first_batch_number + NUM_BATCHES_PER_DIRECTORY - 1

        example_file_name = (
            '{0:s}/batches{1:07d}-{2:07d}/input_examples_batch{3:07d}.nc'
        ).format(top_directory_name, first_batch_number, last_batch_number,
                 batch_number)
    else:
        time_conversion.spc_date_string_to_unix_sec(spc_date_string)

        example_file_name = (
            '{0:s}/{1:s}/input_examples_{2:s}.nc'
        ).format(top_directory_name, spc_date_string[:4], spc_date_string)

    if raise_error_if_missing and not os.path.isfile(example_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            example_file_name)
        raise ValueError(error_string)

    return example_file_name


def find_many_example_files(
        top_directory_name, shuffled=True, first_spc_date_string=None,
        last_spc_date_string=None, first_batch_number=None,
        last_batch_number=None, raise_error_if_any_missing=True):
    """Looks for many files with input examples.

    :param top_directory_name: See doc for `find_example_file`.
    :param shuffled: Same.
    :param first_spc_date_string: [used only if `shuffled = False`]
        First SPC date (format "yyyymmdd").  This method will look for all SPC
        dates from `first_spc_date_string`...`last_spc_date_string`.
    :param last_spc_date_string: See above.
    :param first_batch_number: [used only if `shuffled = True`]
        First batch number (integer).  This method will look for all batches
        from `first_batch_number`...`last_batch_number`.
    :param last_batch_number: See above.
    :param raise_error_if_any_missing: Boolean flag.  If *any* desired file is
        not found and `raise_error_if_any_missing = True`, this method will
        error out.
    :return: example_file_names: 1-D list of paths to example files.
    :raises: ValueError: if no files are found.
    """

    error_checking.assert_is_boolean(shuffled)

    if shuffled:
        error_checking.assert_is_integer(first_batch_number)
        error_checking.assert_is_integer(last_batch_number)
        error_checking.assert_is_geq(first_batch_number, 0)
        error_checking.assert_is_geq(last_batch_number, first_batch_number)

        example_file_pattern = (
            '{0:s}/batches{1:s}-{1:s}/input_examples_batch{1:s}.nc'
        ).format(top_directory_name, BATCH_NUMBER_REGEX)

        example_file_names = glob.glob(example_file_pattern)

        if len(example_file_names) > 0:
            batch_numbers = numpy.array(
                [_file_name_to_batch_number(f) for f in example_file_names],
                dtype=int)
            good_indices = numpy.where(numpy.logical_and(
                batch_numbers >= first_batch_number,
                batch_numbers <= last_batch_number
            ))[0]

            example_file_names = [example_file_names[k] for k in good_indices]

        if len(example_file_names) == 0:
            error_string = (
                'Cannot find any files with batch number from {0:d}...{1:d}.'
            ).format(first_batch_number, last_batch_number)
            raise ValueError(error_string)

        return example_file_names

    spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string)

    example_file_names = []
    for this_spc_date_string in spc_date_strings:
        this_file_name = find_example_file(
            top_directory_name=top_directory_name, shuffled=False,
            spc_date_string=this_spc_date_string,
            raise_error_if_missing=raise_error_if_any_missing)

        if not os.path.isfile(this_file_name):
            continue
        example_file_names.append(this_file_name)

    if len(example_file_names) == 0:
        error_string = (
            'Cannot find any file with SPC date from {0:s} to {1:s}.'
        ).format(first_spc_date_string, last_spc_date_string)

        raise ValueError(error_string)

    return example_file_names


def write_example_file(netcdf_file_name, example_dict, append_to_file=False):
    """Writes input examples to NetCDF file.

    The following keys are required in `example_dict` only if the examples
    include soundings:

    - "sounding_field_names"
    - "sounding_heights_m_agl"
    - "sounding_matrix"

    If the examples contain both 2-D azimuthal-shear images and 3-D
    reflectivity images:

    - Keys "reflectivity_image_matrix_dbz" and "az_shear_image_matrix_s01" are
      required.
    - "radar_heights_m_agl" should contain only reflectivity heights.
    - "radar_field_names" should contain only the names of azimuthal-shear
      fields.

    If the examples contain 2-D radar images and no 3-D images:

    - Key "radar_image_matrix" is required.
    - The [j]th element of "radar_field_names" should be the name of the [j]th
      radar field.
    - The [j]th element of "radar_heights_m_agl" should be the corresponding
      height.
    - Thus, there are C elements in "radar_field_names", C elements in
      "radar_heights_m_agl", and C field-height pairs.

    If the examples contain 3-D radar images and no 2-D images:

    - Key "radar_image_matrix" is required.
    - Each field in "radar_field_names" appears at each height in
      "radar_heights_m_agl".
    - Thus, there are F_r elements in "radar_field_names", H_r elements in
      "radar_heights_m_agl", and F_r * H_r field-height pairs.

    :param netcdf_file_name: Path to output file.
    :param example_dict: Dictionary with the following keys.
    example_dict['full_id_strings']: length-E list of full storm IDs.
    example_dict['storm_times_unix_sec']: length-E list of valid times.
    example_dict['radar_field_names']: List of radar fields (see general
        discussion above).
    example_dict['radar_heights_m_agl']: numpy array of radar heights (see
        general discussion above).
    example_dict['rotated_grids']: Boolean flag.  If True, storm-centered radar
        grids are rotated so that storm motion is in the +x-direction.
    example_dict['rotated_grid_spacing_metres']: Spacing of rotated grids.  If
        grids are not rotated, this should be None.
    example_dict['radar_image_matrix']: See general discussion above.  For 2-D
        images, this should be a numpy array with dimensions E x M x N x C.
        For 3-D images, this should be a numpy array with dimensions
        E x M x N x H_r x F_r.
    example_dict['reflectivity_image_matrix_dbz']: See general discussion above.
        Dimensions should be E x M x N x H_refl x 1, where H_refl = number of
        reflectivity heights.
    example_dict['az_shear_image_matrix_s01']: See general discussion above.
        Dimensions should be E x M x N x F_as, where F_as = number of
        azimuthal-shear fields.
    example_dict['target_names']: 1-D list with names of target variables.  Each
        must be accepted by `target_val_utils.target_name_to_params`.
    example_dict['target_matrix']: E-by-T numpy array of target values (integer
        class labels), where T = number of target variables.
    example_dict['sounding_field_names']: list (length F_s) of sounding fields.
        Each item must be accepted by `soundings.check_field_name`.
    example_dict['sounding_heights_m_agl']: numpy array (length H_s) of sounding
        heights (metres above ground level).
    example_dict['sounding_matrix']: numpy array (E x H_s x F_s) of storm-
        centered soundings.

    :param append_to_file: Boolean flag.  If True, this method will append to an
        existing file.  If False, will create a new file, overwriting the
        existing file if necessary.
    """

    error_checking.assert_is_boolean(append_to_file)
    include_soundings = SOUNDING_MATRIX_KEY in example_dict

    if append_to_file:
        netcdf_dataset = netCDF4.Dataset(
            netcdf_file_name, 'a', format='NETCDF3_64BIT_OFFSET'
        )
        _compare_metadata(
            netcdf_dataset=netcdf_dataset, example_dict=example_dict
        )

        num_examples_orig = len(numpy.array(
            netcdf_dataset.variables[STORM_TIMES_KEY][:]
        ))
        num_examples_to_add = len(example_dict[STORM_TIMES_KEY])

        this_string_type = 'S{0:d}'.format(
            netcdf_dataset.dimensions[STORM_ID_CHAR_DIM_KEY].size
        )
        example_dict[FULL_IDS_KEY] = netCDF4.stringtochar(numpy.array(
            example_dict[FULL_IDS_KEY], dtype=this_string_type
        ))

        for this_key in MAIN_KEYS:
            if (this_key not in REQUIRED_MAIN_KEYS and
                    this_key not in netcdf_dataset.variables):
                continue

            netcdf_dataset.variables[this_key][
                num_examples_orig:(num_examples_orig + num_examples_to_add),
                ...
            ] = example_dict[this_key]

        netcdf_dataset.close()
        return

    # Open file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    netcdf_dataset = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET')

    # Set global attributes.
    netcdf_dataset.setncattr(
        ROTATED_GRIDS_KEY, int(example_dict[ROTATED_GRIDS_KEY])
    )

    if example_dict[ROTATED_GRIDS_KEY]:
        netcdf_dataset.setncattr(
            ROTATED_GRID_SPACING_KEY,
            numpy.round(int(example_dict[ROTATED_GRID_SPACING_KEY]))
        )

    # Set dimensions.
    num_storm_id_chars = 10 + numpy.max(
        numpy.array([len(s) for s in example_dict[FULL_IDS_KEY]])
    )
    num_radar_field_chars = numpy.max(
        numpy.array([len(f) for f in example_dict[RADAR_FIELDS_KEY]])
    )
    num_target_name_chars = numpy.max(
        numpy.array([len(t) for t in example_dict[TARGET_NAMES_KEY]])
    )

    num_target_vars = len(example_dict[TARGET_NAMES_KEY])
    netcdf_dataset.createDimension(EXAMPLE_DIMENSION_KEY, None)
    netcdf_dataset.createDimension(TARGET_VARIABLE_DIM_KEY, num_target_vars)

    netcdf_dataset.createDimension(STORM_ID_CHAR_DIM_KEY, num_storm_id_chars)
    netcdf_dataset.createDimension(
        RADAR_FIELD_CHAR_DIM_KEY, num_radar_field_chars
    )
    netcdf_dataset.createDimension(
        TARGET_NAME_CHAR_DIM_KEY, num_target_name_chars
    )

    if RADAR_IMAGE_MATRIX_KEY in example_dict:
        num_grid_rows = example_dict[RADAR_IMAGE_MATRIX_KEY].shape[1]
        num_grid_columns = example_dict[RADAR_IMAGE_MATRIX_KEY].shape[2]
        num_radar_dimensions = len(
            example_dict[RADAR_IMAGE_MATRIX_KEY].shape) - 2

        if num_radar_dimensions == 3:
            num_radar_heights = example_dict[RADAR_IMAGE_MATRIX_KEY].shape[3]
            num_radar_fields = example_dict[RADAR_IMAGE_MATRIX_KEY].shape[4]

            netcdf_dataset.createDimension(
                RADAR_FIELD_DIM_KEY, num_radar_fields)
            netcdf_dataset.createDimension(
                RADAR_HEIGHT_DIM_KEY, num_radar_heights)
        else:
            num_radar_channels = example_dict[RADAR_IMAGE_MATRIX_KEY].shape[3]
            netcdf_dataset.createDimension(
                RADAR_CHANNEL_DIM_KEY, num_radar_channels)

        netcdf_dataset.createDimension(ROW_DIMENSION_KEY, num_grid_rows)
        netcdf_dataset.createDimension(COLUMN_DIMENSION_KEY, num_grid_columns)

    else:
        num_reflectivity_rows = example_dict[REFL_IMAGE_MATRIX_KEY].shape[1]
        num_reflectivity_columns = example_dict[REFL_IMAGE_MATRIX_KEY].shape[2]
        num_reflectivity_heights = example_dict[REFL_IMAGE_MATRIX_KEY].shape[3]
        num_az_shear_rows = example_dict[AZ_SHEAR_IMAGE_MATRIX_KEY].shape[1]
        num_az_shear_columns = example_dict[AZ_SHEAR_IMAGE_MATRIX_KEY].shape[2]
        num_az_shear_fields = example_dict[AZ_SHEAR_IMAGE_MATRIX_KEY].shape[3]

        netcdf_dataset.createDimension(
            REFL_ROW_DIMENSION_KEY, num_reflectivity_rows)
        netcdf_dataset.createDimension(
            REFL_COLUMN_DIMENSION_KEY, num_reflectivity_columns)
        netcdf_dataset.createDimension(
            RADAR_HEIGHT_DIM_KEY, num_reflectivity_heights)

        netcdf_dataset.createDimension(
            AZ_SHEAR_ROW_DIMENSION_KEY, num_az_shear_rows)
        netcdf_dataset.createDimension(
            AZ_SHEAR_COLUMN_DIMENSION_KEY, num_az_shear_columns)
        netcdf_dataset.createDimension(RADAR_FIELD_DIM_KEY, num_az_shear_fields)

        num_radar_dimensions = -1

    # Add storm IDs.
    this_string_type = 'S{0:d}'.format(num_storm_id_chars)
    full_ids_char_array = netCDF4.stringtochar(numpy.array(
        example_dict[FULL_IDS_KEY], dtype=this_string_type
    ))

    netcdf_dataset.createVariable(
        FULL_IDS_KEY, datatype='S1',
        dimensions=(EXAMPLE_DIMENSION_KEY, STORM_ID_CHAR_DIM_KEY)
    )
    netcdf_dataset.variables[FULL_IDS_KEY][:] = numpy.array(full_ids_char_array)

    # Add names of radar fields.
    this_string_type = 'S{0:d}'.format(num_radar_field_chars)
    radar_field_names_char_array = netCDF4.stringtochar(numpy.array(
        example_dict[RADAR_FIELDS_KEY], dtype=this_string_type
    ))

    if num_radar_dimensions == 2:
        this_first_dim_key = RADAR_CHANNEL_DIM_KEY + ''
    else:
        this_first_dim_key = RADAR_FIELD_DIM_KEY + ''

    netcdf_dataset.createVariable(
        RADAR_FIELDS_KEY, datatype='S1',
        dimensions=(this_first_dim_key, RADAR_FIELD_CHAR_DIM_KEY)
    )
    netcdf_dataset.variables[RADAR_FIELDS_KEY][:] = numpy.array(
        radar_field_names_char_array)

    # Add names of target variables.
    this_string_type = 'S{0:d}'.format(num_target_name_chars)
    target_names_char_array = netCDF4.stringtochar(numpy.array(
        example_dict[TARGET_NAMES_KEY], dtype=this_string_type
    ))

    netcdf_dataset.createVariable(
        TARGET_NAMES_KEY, datatype='S1',
        dimensions=(TARGET_VARIABLE_DIM_KEY, TARGET_NAME_CHAR_DIM_KEY)
    )
    netcdf_dataset.variables[TARGET_NAMES_KEY][:] = numpy.array(
        target_names_char_array)

    # Add storm times.
    netcdf_dataset.createVariable(
        STORM_TIMES_KEY, datatype=numpy.int32, dimensions=EXAMPLE_DIMENSION_KEY
    )
    netcdf_dataset.variables[STORM_TIMES_KEY][:] = example_dict[
        STORM_TIMES_KEY]

    # Add target values.
    netcdf_dataset.createVariable(
        TARGET_MATRIX_KEY, datatype=numpy.int32,
        dimensions=(EXAMPLE_DIMENSION_KEY, TARGET_VARIABLE_DIM_KEY)
    )
    netcdf_dataset.variables[TARGET_MATRIX_KEY][:] = example_dict[
        TARGET_MATRIX_KEY]

    # Add radar heights.
    if num_radar_dimensions == 2:
        this_dimension_key = RADAR_CHANNEL_DIM_KEY + ''
    else:
        this_dimension_key = RADAR_HEIGHT_DIM_KEY + ''

    netcdf_dataset.createVariable(
        RADAR_HEIGHTS_KEY, datatype=numpy.int32, dimensions=this_dimension_key
    )
    netcdf_dataset.variables[RADAR_HEIGHTS_KEY][:] = example_dict[
        RADAR_HEIGHTS_KEY]

    # Add storm-centered radar images.
    if RADAR_IMAGE_MATRIX_KEY in example_dict:
        if num_radar_dimensions == 3:
            these_dimensions = (
                EXAMPLE_DIMENSION_KEY, ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY,
                RADAR_HEIGHT_DIM_KEY, RADAR_FIELD_DIM_KEY
            )
        else:
            these_dimensions = (
                EXAMPLE_DIMENSION_KEY, ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY,
                RADAR_CHANNEL_DIM_KEY
            )

        netcdf_dataset.createVariable(
            RADAR_IMAGE_MATRIX_KEY, datatype=numpy.float32,
            dimensions=these_dimensions
        )
        netcdf_dataset.variables[RADAR_IMAGE_MATRIX_KEY][:] = example_dict[
            RADAR_IMAGE_MATRIX_KEY]

    else:
        netcdf_dataset.createVariable(
            REFL_IMAGE_MATRIX_KEY, datatype=numpy.float32,
            dimensions=(EXAMPLE_DIMENSION_KEY, REFL_ROW_DIMENSION_KEY,
                        REFL_COLUMN_DIMENSION_KEY, RADAR_HEIGHT_DIM_KEY)
        )
        netcdf_dataset.variables[REFL_IMAGE_MATRIX_KEY][:] = example_dict[
            REFL_IMAGE_MATRIX_KEY][..., 0]

        netcdf_dataset.createVariable(
            AZ_SHEAR_IMAGE_MATRIX_KEY, datatype=numpy.float32,
            dimensions=(EXAMPLE_DIMENSION_KEY, AZ_SHEAR_ROW_DIMENSION_KEY,
                        AZ_SHEAR_COLUMN_DIMENSION_KEY, RADAR_FIELD_DIM_KEY)
        )
        netcdf_dataset.variables[AZ_SHEAR_IMAGE_MATRIX_KEY][:] = example_dict[
            AZ_SHEAR_IMAGE_MATRIX_KEY]

    if not include_soundings:
        netcdf_dataset.close()
        return

    num_sounding_heights = example_dict[SOUNDING_MATRIX_KEY].shape[1]
    num_sounding_fields = example_dict[SOUNDING_MATRIX_KEY].shape[2]

    num_sounding_field_chars = 1
    for j in range(num_sounding_fields):
        num_sounding_field_chars = max([
            num_sounding_field_chars,
            len(example_dict[SOUNDING_FIELDS_KEY][j])
        ])

    netcdf_dataset.createDimension(
        SOUNDING_FIELD_DIM_KEY, num_sounding_fields)
    netcdf_dataset.createDimension(
        SOUNDING_HEIGHT_DIM_KEY, num_sounding_heights)
    netcdf_dataset.createDimension(
        SOUNDING_FIELD_CHAR_DIM_KEY, num_sounding_field_chars)

    this_string_type = 'S{0:d}'.format(num_sounding_field_chars)
    sounding_field_names_char_array = netCDF4.stringtochar(numpy.array(
        example_dict[SOUNDING_FIELDS_KEY], dtype=this_string_type
    ))

    netcdf_dataset.createVariable(
        SOUNDING_FIELDS_KEY, datatype='S1',
        dimensions=(SOUNDING_FIELD_DIM_KEY, SOUNDING_FIELD_CHAR_DIM_KEY)
    )
    netcdf_dataset.variables[SOUNDING_FIELDS_KEY][:] = numpy.array(
        sounding_field_names_char_array)

    netcdf_dataset.createVariable(
        SOUNDING_HEIGHTS_KEY, datatype=numpy.int32,
        dimensions=SOUNDING_HEIGHT_DIM_KEY
    )
    netcdf_dataset.variables[SOUNDING_HEIGHTS_KEY][:] = example_dict[
        SOUNDING_HEIGHTS_KEY]

    netcdf_dataset.createVariable(
        SOUNDING_MATRIX_KEY, datatype=numpy.float32,
        dimensions=(EXAMPLE_DIMENSION_KEY, SOUNDING_HEIGHT_DIM_KEY,
                    SOUNDING_FIELD_DIM_KEY)
    )
    netcdf_dataset.variables[SOUNDING_MATRIX_KEY][:] = example_dict[
        SOUNDING_MATRIX_KEY]

    netcdf_dataset.close()
    return


def read_example_file(
        netcdf_file_name, read_all_target_vars, target_name=None,
        metadata_only=False, targets_only=False, include_soundings=True,
        radar_field_names_to_keep=None, radar_heights_to_keep_m_agl=None,
        sounding_field_names_to_keep=None, sounding_heights_to_keep_m_agl=None,
        first_time_to_keep_unix_sec=None, last_time_to_keep_unix_sec=None,
        num_rows_to_keep=None, num_columns_to_keep=None,
        downsampling_dict=None):
    """Reads examples from NetCDF file.

    If `metadata_only == True`, later input args are ignored.
    If `targets_only == True`, later input args are ignored.

    :param netcdf_file_name: Path to input file.
    :param read_all_target_vars: Boolean flag.  If True, will read all target
        variables.  If False, will read only `target_name`.  Either way, if
        downsampling is done, it will be based only on `target_name`.
    :param target_name: Will read this target variable.  If
        `read_all_target_vars == True` and `downsampling_dict is None`, you can
        leave this alone.
    :param metadata_only: Boolean flag.  If False, this method will read
        everything.  If True, will read everything except predictor and target
        variables.
    :param targets_only: Boolean flag.  If False, this method will read
        everything.  If True, will read everything except predictors.
    :param include_soundings: Boolean flag.  If True and the file contains
        soundings, this method will return soundings.  Otherwise, no soundings.
    :param radar_field_names_to_keep: See doc for `_subset_radar_data`.
    :param radar_heights_to_keep_m_agl: Same.
    :param sounding_field_names_to_keep: See doc for `_subset_sounding_data`.
    :param sounding_heights_to_keep_m_agl: Same.
    :param first_time_to_keep_unix_sec: First time to keep.  If
        `first_time_to_keep_unix_sec is None`, all storm objects will be kept.
    :param last_time_to_keep_unix_sec: Last time to keep.  If
        `last_time_to_keep_unix_sec is None`, all storm objects will be kept.
    :param num_rows_to_keep: See doc for `_subset_radar_data`.
    :param num_columns_to_keep: Same.
    :param downsampling_dict: See doc for `_filter_examples_by_class`.
    :return: example_dict: If `read_all_target_vars == True`, dictionary will
        have all keys listed in doc for `write_example_file`.  If
        `read_all_target_vars == False`, key "target_names" will be replaced by
        "target_name" and "target_matrix" will be replaced by "target_values".

    example_dict['target_name']: Name of target variable.
    example_dict['target_values']: length-E list of target values (integer
        class labels), where E = number of examples.
    """

    # TODO(thunderhoser): Allow this method to read only soundings without radar
    # data.

    if (
            target_name ==
            'tornado_lead-time=0000-3600sec_distance=00000-10000m'
    ):
        target_name = (
            'tornado_lead-time=0000-3600sec_distance=00000-30000m_min-fujita=0'
        )

    error_checking.assert_is_boolean(read_all_target_vars)
    error_checking.assert_is_boolean(include_soundings)
    error_checking.assert_is_boolean(metadata_only)
    error_checking.assert_is_boolean(targets_only)

    example_dict, netcdf_dataset = _read_metadata_from_example_file(
        netcdf_file_name=netcdf_file_name, include_soundings=include_soundings)

    need_main_target_values = (
        not read_all_target_vars
        or downsampling_dict is not None
    )

    if need_main_target_values:
        target_index = example_dict[TARGET_NAMES_KEY].index(target_name)
    else:
        target_index = -1

    if not read_all_target_vars:
        example_dict[TARGET_NAME_KEY] = target_name
        example_dict.pop(TARGET_NAMES_KEY)

    if metadata_only:
        netcdf_dataset.close()
        return example_dict

    if need_main_target_values:
        main_target_values = numpy.array(
            netcdf_dataset.variables[TARGET_MATRIX_KEY][:, target_index],
            dtype=int
        )
    else:
        main_target_values = None

    if read_all_target_vars:
        example_dict[TARGET_MATRIX_KEY] = numpy.array(
            netcdf_dataset.variables[TARGET_MATRIX_KEY][:], dtype=int
        )
    else:
        example_dict[TARGET_VALUES_KEY] = main_target_values

    # Subset by time.
    if first_time_to_keep_unix_sec is None:
        first_time_to_keep_unix_sec = 0
    if last_time_to_keep_unix_sec is None:
        last_time_to_keep_unix_sec = int(1e12)

    error_checking.assert_is_integer(first_time_to_keep_unix_sec)
    error_checking.assert_is_integer(last_time_to_keep_unix_sec)
    error_checking.assert_is_geq(
        last_time_to_keep_unix_sec, first_time_to_keep_unix_sec)

    example_indices_to_keep = numpy.where(numpy.logical_and(
        example_dict[STORM_TIMES_KEY] >= first_time_to_keep_unix_sec,
        example_dict[STORM_TIMES_KEY] <= last_time_to_keep_unix_sec
    ))[0]

    if downsampling_dict is not None:
        subindices_to_keep = _filter_examples_by_class(
            target_values=main_target_values[example_indices_to_keep],
            downsampling_dict=downsampling_dict
        )
    elif not read_all_target_vars:
        subindices_to_keep = numpy.where(
            main_target_values[example_indices_to_keep] !=
            target_val_utils.INVALID_STORM_INTEGER
        )[0]
    else:
        subindices_to_keep = numpy.linspace(
            0, len(example_indices_to_keep) - 1,
            num=len(example_indices_to_keep), dtype=int
        )

    example_indices_to_keep = example_indices_to_keep[subindices_to_keep]
    if len(example_indices_to_keep) == 0:
        return None

    example_dict[FULL_IDS_KEY] = [
        example_dict[FULL_IDS_KEY][k] for k in example_indices_to_keep
    ]
    example_dict[STORM_TIMES_KEY] = (
        example_dict[STORM_TIMES_KEY][example_indices_to_keep]
    )

    if read_all_target_vars:
        example_dict[TARGET_MATRIX_KEY] = (
            example_dict[TARGET_MATRIX_KEY][example_indices_to_keep, :]
        )
    else:
        example_dict[TARGET_VALUES_KEY] = (
            example_dict[TARGET_VALUES_KEY][example_indices_to_keep]
        )

    if targets_only:
        netcdf_dataset.close()
        return example_dict

    example_dict = _subset_radar_data(
        example_dict=example_dict, netcdf_dataset_object=netcdf_dataset,
        example_indices_to_keep=example_indices_to_keep,
        field_names_to_keep=radar_field_names_to_keep,
        heights_to_keep_m_agl=radar_heights_to_keep_m_agl,
        num_rows_to_keep=num_rows_to_keep,
        num_columns_to_keep=num_columns_to_keep)

    if not include_soundings:
        netcdf_dataset.close()
        return example_dict

    example_dict = _subset_sounding_data(
        example_dict=example_dict, netcdf_dataset_object=netcdf_dataset,
        example_indices_to_keep=example_indices_to_keep,
        field_names_to_keep=sounding_field_names_to_keep,
        heights_to_keep_m_agl=sounding_heights_to_keep_m_agl)

    netcdf_dataset.close()
    return example_dict


def read_specific_examples(
        netcdf_file_name, read_all_target_vars, full_storm_id_strings,
        storm_times_unix_sec, target_name=None, include_soundings=True,
        radar_field_names_to_keep=None, radar_heights_to_keep_m_agl=None,
        sounding_field_names_to_keep=None, sounding_heights_to_keep_m_agl=None,
        num_rows_to_keep=None, num_columns_to_keep=None):
    """Reads specific examples (with specific ID-time pairs) from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :param read_all_target_vars: See doc for `read_example_file`.
    :param full_storm_id_strings: length-E list of storm IDs.
    :param storm_times_unix_sec: length-E numpy array of valid times.
    :param target_name: See doc for `read_example_file`.
    :param metadata_only: Same.
    :param include_soundings: Same.
    :param radar_field_names_to_keep: Same.
    :param radar_heights_to_keep_m_agl: Same.
    :param sounding_field_names_to_keep: Same.
    :param sounding_heights_to_keep_m_agl: Same.
    :param num_rows_to_keep: Same.
    :param num_columns_to_keep: Same.
    :return: example_dict: See doc for `write_example_file`.
    """

    if (
            target_name ==
            'tornado_lead-time=0000-3600sec_distance=00000-10000m'
    ):
        target_name = (
            'tornado_lead-time=0000-3600sec_distance=00000-30000m_min-fujita=0'
        )

    error_checking.assert_is_boolean(read_all_target_vars)
    error_checking.assert_is_boolean(include_soundings)

    example_dict, dataset_object = _read_metadata_from_example_file(
        netcdf_file_name=netcdf_file_name, include_soundings=include_soundings)

    example_indices_to_keep = tracking_utils.find_storm_objects(
        all_id_strings=example_dict[FULL_IDS_KEY],
        all_times_unix_sec=example_dict[STORM_TIMES_KEY],
        id_strings_to_keep=full_storm_id_strings,
        times_to_keep_unix_sec=storm_times_unix_sec, allow_missing=False
    )

    example_dict[FULL_IDS_KEY] = [
        example_dict[FULL_IDS_KEY][k] for k in example_indices_to_keep
    ]
    example_dict[STORM_TIMES_KEY] = example_dict[STORM_TIMES_KEY][
        example_indices_to_keep]

    if read_all_target_vars:
        example_dict[TARGET_MATRIX_KEY] = numpy.array(
            dataset_object.variables[TARGET_MATRIX_KEY][
                example_indices_to_keep, :],
            dtype=int
        )
    else:
        target_index = example_dict[TARGET_NAMES_KEY].index(target_name)
        example_dict[TARGET_NAME_KEY] = target_name
        example_dict.pop(TARGET_NAMES_KEY)

        example_dict[TARGET_VALUES_KEY] = numpy.array(
            dataset_object.variables[TARGET_MATRIX_KEY][
                example_indices_to_keep, target_index],
            dtype=int
        )

    example_dict = _subset_radar_data(
        example_dict=example_dict, netcdf_dataset_object=dataset_object,
        example_indices_to_keep=example_indices_to_keep,
        field_names_to_keep=radar_field_names_to_keep,
        heights_to_keep_m_agl=radar_heights_to_keep_m_agl,
        num_rows_to_keep=num_rows_to_keep,
        num_columns_to_keep=num_columns_to_keep)

    if not include_soundings:
        dataset_object.close()
        return example_dict

    example_dict = _subset_sounding_data(
        example_dict=example_dict, netcdf_dataset_object=dataset_object,
        example_indices_to_keep=example_indices_to_keep,
        field_names_to_keep=sounding_field_names_to_keep,
        heights_to_keep_m_agl=sounding_heights_to_keep_m_agl)

    dataset_object.close()
    return example_dict


def reduce_examples_3d_to_2d(example_dict, list_of_operation_dicts):
    """Reduces examples from 3-D to 2-D.

    If the examples contain both 2-D azimuthal-shear images and 3-D
    reflectivity images:

    - Keys "reflectivity_image_matrix_dbz" and "az_shear_image_matrix_s01" are
      required.
    - "radar_heights_m_agl" should contain only reflectivity heights.
    - "radar_field_names" should contain only the names of azimuthal-shear
      fields.

    If the examples contain 3-D radar images and no 2-D images:

    - Key "radar_image_matrix" is required.
    - Each field in "radar_field_names" appears at each height in
      "radar_heights_m_agl".
    - Thus, there are F_r elements in "radar_field_names", H_r elements in
      "radar_heights_m_agl", and F_r * H_r field-height pairs.

    After dimensionality reduction (from 3-D to 2-D):

    - Keys "reflectivity_image_matrix_dbz", "az_shear_image_matrix_s01", and
      "radar_heights_m_agl" will be absent.
    - Key "radar_image_matrix" will be present.  The dimensions will be
      E x M x N x C.
    - Key "radar_field_names" will be a length-C list, where the [j]th item is
      the field name for the [j]th channel of radar_image_matrix
      (radar_image_matrix[..., j]).
    - Key "min_radar_heights_m_agl" will be a length-C numpy array, where the
      [j]th item is the MINIMUM height for the [j]th channel of
      radar_image_matrix.
    - Key "max_radar_heights_m_agl" will be a length-C numpy array, where the
      [j]th item is the MAX height for the [j]th channel of radar_image_matrix.
    - Key "radar_layer_operation_names" will be a length-C list, where the [j]th
      item is the name of the operation used to create the [j]th channel of
      radar_image_matrix.

    :param example_dict: See doc for `write_example_file`.
    :param list_of_operation_dicts: See doc for `_check_layer_operation`.
    :return: example_dict: See general discussion above, for how the input
        `example_dict` is changed to the output `example_dict`.
    """

    if RADAR_IMAGE_MATRIX_KEY in example_dict:
        num_radar_dimensions = len(
            example_dict[RADAR_IMAGE_MATRIX_KEY].shape
        ) - 2

        assert num_radar_dimensions == 3

    new_radar_image_matrix = None
    new_field_names = []
    new_min_heights_m_agl = []
    new_max_heights_m_agl = []
    new_operation_names = []

    if AZ_SHEAR_IMAGE_MATRIX_KEY in example_dict:
        new_radar_image_matrix = example_dict[AZ_SHEAR_IMAGE_MATRIX_KEY] + 0.

        for this_field_name in example_dict[RADAR_FIELDS_KEY]:
            new_field_names.append(this_field_name)
            new_operation_names.append(MAX_OPERATION_NAME)

            if this_field_name == radar_utils.LOW_LEVEL_SHEAR_NAME:
                new_min_heights_m_agl.append(0)
                new_max_heights_m_agl.append(2000)
            else:
                new_min_heights_m_agl.append(3000)
                new_max_heights_m_agl.append(6000)

    for this_operation_dict in list_of_operation_dicts:
        this_new_matrix, this_operation_dict = _apply_layer_operation(
            example_dict=example_dict, operation_dict=this_operation_dict)

        this_new_matrix = numpy.expand_dims(this_new_matrix, axis=-1)

        if new_radar_image_matrix is None:
            new_radar_image_matrix = this_new_matrix + 0.
        else:
            new_radar_image_matrix = numpy.concatenate(
                (new_radar_image_matrix, this_new_matrix), axis=-1
            )

        new_field_names.append(this_operation_dict[RADAR_FIELD_KEY])
        new_min_heights_m_agl.append(this_operation_dict[MIN_HEIGHT_KEY])
        new_max_heights_m_agl.append(this_operation_dict[MAX_HEIGHT_KEY])
        new_operation_names.append(this_operation_dict[OPERATION_NAME_KEY])

    example_dict.pop(REFL_IMAGE_MATRIX_KEY, None)
    example_dict.pop(AZ_SHEAR_IMAGE_MATRIX_KEY, None)
    example_dict.pop(RADAR_HEIGHTS_KEY, None)

    example_dict[RADAR_IMAGE_MATRIX_KEY] = new_radar_image_matrix
    example_dict[RADAR_FIELDS_KEY] = new_field_names
    example_dict[MIN_RADAR_HEIGHTS_KEY] = numpy.array(
        new_min_heights_m_agl, dtype=int)
    example_dict[MAX_RADAR_HEIGHTS_KEY] = numpy.array(
        new_max_heights_m_agl, dtype=int)
    example_dict[RADAR_LAYER_OPERATION_NAMES_KEY] = new_operation_names

    return example_dict


def create_examples(
        target_file_names, target_names, num_examples_per_in_file,
        top_output_dir_name, radar_file_name_matrix=None,
        reflectivity_file_name_matrix=None, az_shear_file_name_matrix=None,
        downsampling_dict=None, target_name_for_downsampling=None,
        sounding_file_names=None):
    """Creates many input examples.

    If `radar_file_name_matrix is None`, both `reflectivity_file_name_matrix`
    and `az_shear_file_name_matrix` must be specified.

    D = number of SPC dates in time period

    :param target_file_names: length-D list of paths to target files (will be
        read by `read_labels_from_netcdf`).
    :param target_names: See doc for `_check_target_vars`.
    :param num_examples_per_in_file: Number of examples to read from each input
        file.
    :param top_output_dir_name: Name of top-level directory.  Files will be
        written here by `write_example_file`, to locations determined by
        `find_example_file`.
    :param radar_file_name_matrix: numpy array created by either
        `find_storm_images_2d` or `find_storm_images_3d`.  Length of the first
        axis is D.
    :param reflectivity_file_name_matrix: numpy array created by
        `find_storm_images_2d3d_myrorss`.  Length of the first axis is D.
    :param az_shear_file_name_matrix: Same.
    :param downsampling_dict: See doc for `deep_learning_utils.sample_by_class`.
        If None, there will be no downsampling.
    :param target_name_for_downsampling:
        [used only if `downsampling_dict is not None`]
        Name of target variable to use for downsampling.
    :param sounding_file_names: length-D list of paths to sounding files (will
        be read by `soundings.read_soundings`).  If None, will not include
        soundings.
    """

    _check_target_vars(target_names)
    num_target_vars = len(target_names)

    if radar_file_name_matrix is None:
        error_checking.assert_is_numpy_array(
            reflectivity_file_name_matrix, num_dimensions=2)

        num_file_times = reflectivity_file_name_matrix.shape[0]
        these_expected_dim = numpy.array([num_file_times, 2], dtype=int)

        error_checking.assert_is_numpy_array(
            az_shear_file_name_matrix, exact_dimensions=these_expected_dim)
    else:
        error_checking.assert_is_numpy_array(radar_file_name_matrix)
        num_file_dimensions = len(radar_file_name_matrix.shape)
        num_file_times = radar_file_name_matrix.shape[0]

        error_checking.assert_is_geq(num_file_dimensions, 2)
        error_checking.assert_is_leq(num_file_dimensions, 3)

    these_expected_dim = numpy.array([num_file_times], dtype=int)
    error_checking.assert_is_numpy_array(
        numpy.array(target_file_names), exact_dimensions=these_expected_dim
    )

    if sounding_file_names is not None:
        error_checking.assert_is_numpy_array(
            numpy.array(sounding_file_names),
            exact_dimensions=these_expected_dim
        )

    error_checking.assert_is_integer(num_examples_per_in_file)
    error_checking.assert_is_geq(num_examples_per_in_file, 1)

    full_id_strings = []
    storm_times_unix_sec = numpy.array([], dtype=int)
    target_matrix = None

    for i in range(num_file_times):
        print('Reading data from: "{0:s}"...'.format(target_file_names[i]))
        this_target_dict = target_val_utils.read_target_values(
            netcdf_file_name=target_file_names[i], target_names=target_names)

        full_id_strings += this_target_dict[target_val_utils.FULL_IDS_KEY]

        storm_times_unix_sec = numpy.concatenate((
            storm_times_unix_sec,
            this_target_dict[target_val_utils.VALID_TIMES_KEY]
        ))

        if target_matrix is None:
            target_matrix = (
                this_target_dict[target_val_utils.TARGET_MATRIX_KEY] + 0
            )
        else:
            target_matrix = numpy.concatenate(
                (target_matrix,
                 this_target_dict[target_val_utils.TARGET_MATRIX_KEY]),
                axis=0
            )

    print('\n')
    num_examples_found = len(full_id_strings)
    num_examples_to_use = num_examples_per_in_file * num_file_times

    if downsampling_dict is None:
        indices_to_keep = numpy.linspace(
            0, num_examples_found - 1, num=num_examples_found, dtype=int)

        if num_examples_found > num_examples_to_use:
            indices_to_keep = numpy.random.choice(
                indices_to_keep, size=num_examples_to_use, replace=False)
    else:
        downsampling_index = target_names.index(target_name_for_downsampling)

        indices_to_keep = dl_utils.sample_by_class(
            sampling_fraction_by_class_dict=downsampling_dict,
            target_name=target_name_for_downsampling,
            target_values=target_matrix[:, downsampling_index],
            num_examples_total=num_examples_to_use)

    full_id_strings = [full_id_strings[k] for k in indices_to_keep]
    storm_times_unix_sec = storm_times_unix_sec[indices_to_keep]
    target_matrix = target_matrix[indices_to_keep, :]

    for j in range(num_target_vars):
        these_unique_classes, these_unique_counts = numpy.unique(
            target_matrix[:, j], return_counts=True
        )

        for k in range(len(these_unique_classes)):
            print((
                'Number of examples with "{0:s}" in class {1:d} = {2:d}'
            ).format(
                target_names[j], these_unique_classes[k], these_unique_counts[k]
            ))

        print('\n')

    first_spc_date_string = time_conversion.time_to_spc_date_string(
        numpy.min(storm_times_unix_sec)
    )
    last_spc_date_string = time_conversion.time_to_spc_date_string(
        numpy.max(storm_times_unix_sec)
    )
    spc_date_strings = time_conversion.get_spc_dates_in_range(
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string)

    spc_date_to_out_file_dict = {}

    for this_spc_date_string in spc_date_strings:
        this_file_name = find_example_file(
            top_directory_name=top_output_dir_name, shuffled=False,
            spc_date_string=this_spc_date_string,
            raise_error_if_missing=False)

        if os.path.isfile(this_file_name):
            os.remove(this_file_name)

        spc_date_to_out_file_dict[this_spc_date_string] = this_file_name

    for i in range(num_file_times):
        if radar_file_name_matrix is None:
            this_file_name = reflectivity_file_name_matrix[i, 0]
        else:
            this_file_name = numpy.ravel(radar_file_name_matrix[i, ...])[0]

        this_time_unix_sec, this_spc_date_string = (
            storm_images.image_file_name_to_time(this_file_name)
        )

        if this_time_unix_sec is None:
            this_first_time_unix_sec = (
                time_conversion.get_start_of_spc_date(this_spc_date_string)
            )
            this_last_time_unix_sec = (
                time_conversion.get_end_of_spc_date(this_spc_date_string)
            )
        else:
            this_first_time_unix_sec = this_time_unix_sec + 0
            this_last_time_unix_sec = this_time_unix_sec + 0

        these_indices = numpy.where(
            numpy.logical_and(
                storm_times_unix_sec >= this_first_time_unix_sec,
                storm_times_unix_sec <= this_last_time_unix_sec)
        )[0]

        if len(these_indices) == 0:
            continue

        these_full_id_strings = [full_id_strings[m] for m in these_indices]
        these_storm_times_unix_sec = storm_times_unix_sec[these_indices]
        this_target_matrix = target_matrix[these_indices, :]

        if sounding_file_names is None:
            this_sounding_file_name = None
        else:
            this_sounding_file_name = sounding_file_names[i]

        if radar_file_name_matrix is None:
            this_example_dict = _create_2d3d_examples_myrorss(
                azimuthal_shear_file_names=az_shear_file_name_matrix[
                    i, ...].tolist(),
                reflectivity_file_names=reflectivity_file_name_matrix[
                    i, ...].tolist(),
                full_id_strings=these_full_id_strings,
                storm_times_unix_sec=these_storm_times_unix_sec,
                target_matrix=this_target_matrix,
                sounding_file_name=this_sounding_file_name,
                sounding_field_names=None)

        elif num_file_dimensions == 3:
            this_example_dict = _create_3d_examples(
                radar_file_name_matrix=radar_file_name_matrix[i, ...],
                full_id_strings=these_full_id_strings,
                storm_times_unix_sec=these_storm_times_unix_sec,
                target_matrix=this_target_matrix,
                sounding_file_name=this_sounding_file_name,
                sounding_field_names=None)

        else:
            this_example_dict = _create_2d_examples(
                radar_file_names=radar_file_name_matrix[i, ...].tolist(),
                full_id_strings=these_full_id_strings,
                storm_times_unix_sec=these_storm_times_unix_sec,
                target_matrix=this_target_matrix,
                sounding_file_name=this_sounding_file_name,
                sounding_field_names=None)

        print('\n')
        if this_example_dict is None:
            continue

        this_example_dict.update({TARGET_NAMES_KEY: target_names})
        this_output_file_name = spc_date_to_out_file_dict[this_spc_date_string]

        print('Writing examples to: "{0:s}"...'.format(this_output_file_name))
        write_example_file(
            netcdf_file_name=this_output_file_name,
            example_dict=this_example_dict,
            append_to_file=os.path.isfile(this_output_file_name)
        )
