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

import random
import os.path
import numpy
import netCDF4
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import labels
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import storm_images
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

DEFAULT_NUM_EXAMPLES_PER_OUT_CHUNK = 8
DEFAULT_NUM_EXAMPLES_PER_OUT_FILE = 128
NUM_BATCHES_PER_DIRECTORY = 1000

AZIMUTHAL_SHEAR_FIELD_NAMES = [
    radar_utils.LOW_LEVEL_SHEAR_NAME, radar_utils.MID_LEVEL_SHEAR_NAME
]

TARGET_NAME_KEY = 'target_name'
ROTATED_GRIDS_KEY = 'rotated_grids'
ROTATED_GRID_SPACING_KEY = 'rotated_grid_spacing_metres'

STORM_IDS_KEY = 'storm_ids'
STORM_TIMES_KEY = 'storm_times_unix_sec'
TARGET_VALUES_KEY = 'target_values'
RADAR_IMAGE_MATRIX_KEY = 'radar_image_matrix'
RADAR_FIELDS_KEY = 'radar_field_names'
RADAR_HEIGHTS_KEY = 'radar_heights_m_agl'
SOUNDING_FIELDS_KEY = 'sounding_field_names'
SOUNDING_MATRIX_KEY = 'sounding_matrix'
SOUNDING_HEIGHTS_KEY = 'sounding_heights_m_agl'
REFL_IMAGE_MATRIX_KEY = 'reflectivity_image_matrix_dbz'
AZ_SHEAR_IMAGE_MATRIX_KEY = 'az_shear_image_matrix_s01'

MAIN_KEYS = [
    STORM_IDS_KEY, STORM_TIMES_KEY, RADAR_IMAGE_MATRIX_KEY,
    REFL_IMAGE_MATRIX_KEY, AZ_SHEAR_IMAGE_MATRIX_KEY, TARGET_VALUES_KEY,
    SOUNDING_MATRIX_KEY
]
REQUIRED_MAIN_KEYS = [
    STORM_IDS_KEY, STORM_TIMES_KEY, TARGET_VALUES_KEY
]
METADATA_KEYS = [
    TARGET_NAME_KEY, ROTATED_GRIDS_KEY, ROTATED_GRID_SPACING_KEY,
    RADAR_FIELDS_KEY, RADAR_HEIGHTS_KEY, SOUNDING_FIELDS_KEY,
    SOUNDING_HEIGHTS_KEY
]

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
STORM_ID_CHAR_DIM_KEY = 'storm_id_character'
RADAR_FIELD_CHAR_DIM_KEY = 'radar_field_name_character'
SOUNDING_FIELD_CHAR_DIM_KEY = 'sounding_field_name_character'


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

    print 'Reading data from: "{0:s}"...'.format(sounding_file_name)
    sounding_dict, _ = soundings.read_soundings(
        netcdf_file_name=sounding_file_name,
        field_names_to_keep=sounding_field_names,
        storm_ids_to_keep=radar_image_dict[storm_images.STORM_IDS_KEY],
        init_times_to_keep_unix_sec=radar_image_dict[
            storm_images.VALID_TIMES_KEY])

    num_examples_with_soundings = len(sounding_dict[soundings.STORM_IDS_KEY])
    if num_examples_with_soundings == 0:
        return None, None

    orig_storm_ids_numpy = numpy.array(
        radar_image_dict[storm_images.STORM_IDS_KEY]
    )
    orig_storm_times_unix_sec = (
        radar_image_dict[storm_images.VALID_TIMES_KEY] + 0
    )

    indices_to_keep = []
    for i in range(num_examples_with_soundings):
        this_index = numpy.where(numpy.logical_and(
            orig_storm_ids_numpy == sounding_dict[soundings.STORM_IDS_KEY][i],
            orig_storm_times_unix_sec ==
            sounding_dict[soundings.INITIAL_TIMES_KEY][i]
        ))[0][0]

        indices_to_keep.append(this_index)

    indices_to_keep = numpy.array(indices_to_keep, dtype=int)
    radar_image_dict[storm_images.STORM_IMAGE_MATRIX_KEY] = radar_image_dict[
        storm_images.STORM_IMAGE_MATRIX_KEY
    ][indices_to_keep, ...]
    radar_image_dict[storm_images.LABEL_VALUES_KEY] = radar_image_dict[
        storm_images.LABEL_VALUES_KEY
    ][indices_to_keep]

    radar_image_dict[storm_images.STORM_IDS_KEY] = sounding_dict[
        soundings.STORM_IDS_KEY
    ]
    radar_image_dict[storm_images.VALID_TIMES_KEY] = sounding_dict[
        soundings.INITIAL_TIMES_KEY
    ]

    return sounding_dict, radar_image_dict


def _create_2d_examples(
        radar_file_names, storm_ids, storm_times_unix_sec, target_values,
        sounding_file_name=None, sounding_field_names=None):
    """Creates 2-D examples for one file time.

    E = number of desired examples (storm objects)
    e = number of examples returned

    :param radar_file_names: length-C list of paths to storm-centered radar
        images.  Files will be read by `storm_images.read_storm_images`.
    :param storm_ids: length-E list with storm IDs of objects to return.
    :param storm_times_unix_sec: length-E numpy array with valid times of storm
        objects to return.
    :param target_values: length-E numpy array of target values (integer class
        labels).
    :param sounding_file_name: Path to sounding file (will be read by
        `soundings.read_soundings`).  If `sounding_file_name is None`, examples
        will not include soundings.
    :param sounding_field_names: See doc for `soundings.read_soundings`.
    :return: example_dict: Same as input for `write_example_file`, but without
        key "target_name".
    """

    print 'Reading data from: "{0:s}"...'.format(radar_file_names[0])
    this_radar_image_dict = storm_images.read_storm_images(
        netcdf_file_name=radar_file_names[0], storm_ids_to_keep=storm_ids,
        valid_times_to_keep_unix_sec=storm_times_unix_sec)

    if this_radar_image_dict is None:
        return None

    this_radar_image_dict.update({storm_images.LABEL_VALUES_KEY: target_values})

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
        if len(this_radar_image_dict[storm_images.STORM_IDS_KEY]) == 0:
            return None

        sounding_matrix = sounding_dict[soundings.SOUNDING_MATRIX_KEY]
        sounding_field_names = sounding_dict[soundings.FIELD_NAMES_KEY]
        sounding_heights_m_agl = sounding_dict[soundings.HEIGHT_LEVELS_KEY]

    storm_ids = this_radar_image_dict[storm_images.STORM_IDS_KEY]
    storm_times_unix_sec = this_radar_image_dict[storm_images.VALID_TIMES_KEY]
    target_values = this_radar_image_dict[storm_images.LABEL_VALUES_KEY]

    num_channels = len(radar_file_names)
    tuple_of_image_matrices = ()

    for j in range(num_channels):
        if j != 0:
            print 'Reading data from: "{0:s}"...'.format(radar_file_names[j])
            this_radar_image_dict = storm_images.read_storm_images(
                netcdf_file_name=radar_file_names[j],
                storm_ids_to_keep=storm_ids,
                valid_times_to_keep_unix_sec=storm_times_unix_sec)

        tuple_of_image_matrices += (
            this_radar_image_dict[storm_images.STORM_IMAGE_MATRIX_KEY],
        )

    radar_field_names = [
        storm_images.image_file_name_to_field(f) for f in radar_file_names
    ]
    radar_heights_m_agl = numpy.array(
        [storm_images.image_file_name_to_height(f) for f in radar_file_names],
        dtype=int)

    example_dict = {
        STORM_IDS_KEY: storm_ids,
        STORM_TIMES_KEY: storm_times_unix_sec,
        RADAR_FIELDS_KEY: radar_field_names,
        RADAR_HEIGHTS_KEY: radar_heights_m_agl,
        ROTATED_GRIDS_KEY:
            this_radar_image_dict[storm_images.ROTATED_GRIDS_KEY],
        ROTATED_GRID_SPACING_KEY:
            this_radar_image_dict[storm_images.ROTATED_GRID_SPACING_KEY],
        RADAR_IMAGE_MATRIX_KEY: dl_utils.stack_radar_fields(
            tuple_of_image_matrices),
        TARGET_VALUES_KEY: target_values
    }

    if sounding_file_name is not None:
        example_dict.update({
            SOUNDING_FIELDS_KEY: sounding_field_names,
            SOUNDING_HEIGHTS_KEY: sounding_heights_m_agl,
            SOUNDING_MATRIX_KEY: sounding_matrix
        })

    return example_dict


def _create_3d_examples(
        radar_file_name_matrix, storm_ids, storm_times_unix_sec, target_values,
        sounding_file_name=None, sounding_field_names=None):
    """Creates 3-D examples for one file time.

    E = number of desired examples (storm objects)
    e = number of examples returned

    :param radar_file_name_matrix: numpy array (F_r x H_r) of paths to storm-
        centered radar images.  Files will be read by
        `storm_images.read_storm_images`.
    :param storm_ids: See doc for `_create_2d_examples`.
    :param storm_times_unix_sec: Same.
    :param target_values: Same.
    :param sounding_file_name: Same.
    :param sounding_field_names: Same.
    :return: example_dict: Same.
    """

    print 'Reading data from: "{0:s}"...'.format(radar_file_name_matrix[0, 0])
    this_radar_image_dict = storm_images.read_storm_images(
        netcdf_file_name=radar_file_name_matrix[0, 0],
        storm_ids_to_keep=storm_ids,
        valid_times_to_keep_unix_sec=storm_times_unix_sec)

    if this_radar_image_dict is None:
        return None

    this_radar_image_dict.update({storm_images.LABEL_VALUES_KEY: target_values})

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
        if len(this_radar_image_dict[storm_images.STORM_IDS_KEY]) == 0:
            return None

        sounding_matrix = sounding_dict[soundings.SOUNDING_MATRIX_KEY]
        sounding_field_names = sounding_dict[soundings.FIELD_NAMES_KEY]
        sounding_heights_m_agl = sounding_dict[soundings.HEIGHT_LEVELS_KEY]

    storm_ids = this_radar_image_dict[storm_images.STORM_IDS_KEY]
    storm_times_unix_sec = this_radar_image_dict[storm_images.VALID_TIMES_KEY]
    target_values = this_radar_image_dict[storm_images.LABEL_VALUES_KEY]

    num_radar_fields = radar_file_name_matrix.shape[0]
    num_radar_heights = radar_file_name_matrix.shape[1]
    tuple_of_4d_image_matrices = ()

    for k in range(num_radar_heights):
        tuple_of_3d_image_matrices = ()

        for j in range(num_radar_fields):
            if not j == k == 0:
                print 'Reading data from: "{0:s}"...'.format(
                    radar_file_name_matrix[j, k])
                this_radar_image_dict = storm_images.read_storm_images(
                    netcdf_file_name=radar_file_name_matrix[j, k],
                    storm_ids_to_keep=storm_ids,
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
        STORM_IDS_KEY: storm_ids,
        STORM_TIMES_KEY: storm_times_unix_sec,
        RADAR_FIELDS_KEY: radar_field_names,
        RADAR_HEIGHTS_KEY: radar_heights_m_agl,
        ROTATED_GRIDS_KEY:
            this_radar_image_dict[storm_images.ROTATED_GRIDS_KEY],
        ROTATED_GRID_SPACING_KEY:
            this_radar_image_dict[storm_images.ROTATED_GRID_SPACING_KEY],
        RADAR_IMAGE_MATRIX_KEY: dl_utils.stack_radar_heights(
            tuple_of_4d_image_matrices),
        TARGET_VALUES_KEY: target_values
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
        storm_ids, storm_times_unix_sec, target_values,
        sounding_file_name=None, sounding_field_names=None):
    """Creates hybrid 2D-3D examples for one file time.

    Fields in 2-D images: low-level and mid-level azimuthal shear
    Field in 3-D images: reflectivity

    E = number of desired examples (storm objects)
    e = number of examples returned

    :param azimuthal_shear_file_names: length-2 list of paths to storm-centered
        azimuthal-shear images.  The first (second) file should be (low)
        mid-level azimuthal shear.  Files will be read by
        `storm_images.read_storm_images`.
    :param reflectivity_file_names: length-H list of paths to storm-centered
        reflectivity images, where H = number of reflectivity heights.  Files
        will be read by `storm_images.read_storm_images`.
    :param storm_ids: See doc for `_create_2d_examples`.
    :param storm_times_unix_sec: Same.
    :param target_values: Same.
    :param sounding_file_name: Same.
    :param sounding_field_names: Same.
    :return: example_dict: Same.
    """

    print 'Reading data from: "{0:s}"...'.format(reflectivity_file_names[0])
    this_radar_image_dict = storm_images.read_storm_images(
        netcdf_file_name=reflectivity_file_names[0],
        storm_ids_to_keep=storm_ids,
        valid_times_to_keep_unix_sec=storm_times_unix_sec)

    if this_radar_image_dict is None:
        return None

    this_radar_image_dict.update({storm_images.LABEL_VALUES_KEY: target_values})

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
        if len(this_radar_image_dict[storm_images.STORM_IDS_KEY]) == 0:
            return None

        sounding_matrix = sounding_dict[soundings.SOUNDING_MATRIX_KEY]
        sounding_field_names = sounding_dict[soundings.FIELD_NAMES_KEY]
        sounding_heights_m_agl = sounding_dict[soundings.HEIGHT_LEVELS_KEY]

    storm_ids = this_radar_image_dict[storm_images.STORM_IDS_KEY]
    storm_times_unix_sec = this_radar_image_dict[storm_images.VALID_TIMES_KEY]
    target_values = this_radar_image_dict[storm_images.LABEL_VALUES_KEY]

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
            print 'Reading data from: "{0:s}"...'.format(
                reflectivity_file_names[j])
            this_radar_image_dict = storm_images.read_storm_images(
                netcdf_file_name=reflectivity_file_names[j],
                storm_ids_to_keep=storm_ids,
                valid_times_to_keep_unix_sec=storm_times_unix_sec)

        this_matrix = numpy.expand_dims(
            this_radar_image_dict[storm_images.STORM_IMAGE_MATRIX_KEY], axis=-1)
        tuple_of_image_matrices += (this_matrix,)

    example_dict = {
        STORM_IDS_KEY: storm_ids,
        STORM_TIMES_KEY: storm_times_unix_sec,
        RADAR_FIELDS_KEY: azimuthal_shear_field_names,
        RADAR_HEIGHTS_KEY: reflectivity_heights_m_agl,
        ROTATED_GRIDS_KEY:
            this_radar_image_dict[storm_images.ROTATED_GRIDS_KEY],
        ROTATED_GRID_SPACING_KEY:
            this_radar_image_dict[storm_images.ROTATED_GRID_SPACING_KEY],
        REFL_IMAGE_MATRIX_KEY: dl_utils.stack_radar_heights(
            tuple_of_image_matrices),
        TARGET_VALUES_KEY: target_values
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
        print 'Reading data from: "{0:s}"...'.format(
            azimuthal_shear_file_names[j])
        this_radar_image_dict = storm_images.read_storm_images(
            netcdf_file_name=azimuthal_shear_file_names[j],
            storm_ids_to_keep=storm_ids,
            valid_times_to_keep_unix_sec=storm_times_unix_sec)

        tuple_of_image_matrices += (
            this_radar_image_dict[storm_images.STORM_IMAGE_MATRIX_KEY],
        )

    example_dict.update({
        AZ_SHEAR_IMAGE_MATRIX_KEY: dl_utils.stack_radar_fields(
            tuple_of_image_matrices)
    })

    return example_dict


def _write_examples_to_many_files(
        example_dict, output_file_names, num_examples_per_out_chunk):
    """Writes examples to random output files.

    :param example_dict: See doc for `write_example_file`.
    :param output_file_names: 1-D list of paths to output (NetCDF) files.
    :param num_examples_per_out_chunk: See doc for `shuffle_and_write_examples`.
    """

    num_examples = len(example_dict[STORM_IDS_KEY])

    for j in xrange(0, num_examples, num_examples_per_out_chunk):
        this_first_index = j
        this_last_index = min(
            [j + num_examples_per_out_chunk - 1, num_examples - 1]
        )
        these_example_indices = numpy.linspace(
            this_first_index, this_last_index,
            num=this_last_index - this_first_index + 1, dtype=int)

        this_example_dict = dict()
        for this_key in METADATA_KEYS:
            if (this_key in [SOUNDING_FIELDS_KEY, SOUNDING_HEIGHTS_KEY]
                    and this_key not in example_dict):
                continue

            this_example_dict[this_key] = example_dict[this_key]

        for this_key in MAIN_KEYS:
            if (this_key not in REQUIRED_MAIN_KEYS
                    and this_key not in example_dict):
                continue

            if this_key == STORM_IDS_KEY:
                this_example_dict[this_key] = [
                    example_dict[this_key][k] for k in these_example_indices
                ]
            else:
                this_example_dict[this_key] = example_dict[this_key][
                    these_example_indices, ...]

        this_output_file_name = random.choice(output_file_names)
        print 'Writing shuffled examples to: "{0:s}"...'.format(
            this_output_file_name)

        write_example_file(
            netcdf_file_name=this_output_file_name,
            example_dict=this_example_dict,
            append_to_file=os.path.isfile(this_output_file_name)
        )


def _compare_metadata(netcdf_dataset, example_dict):
    """Compares metadata between existing NetCDF file and new batch of examples.

    This method contains a large number of `assert` statements.  If any of the
    `assert` statements fails, this method will error out.

    :param netcdf_dataset: Instance of `netCDF4.Dataset`.
    :param example_dict: See doc for `write_examples_with_3d_radar`.
    :raises: ValueError: if the two sets have different metadata.
    """

    include_soundings = SOUNDING_MATRIX_KEY in example_dict

    orig_example_dict = dict()
    orig_example_dict[TARGET_NAME_KEY] = str(
        getattr(netcdf_dataset, TARGET_NAME_KEY))
    orig_example_dict[ROTATED_GRIDS_KEY] = bool(
        getattr(netcdf_dataset, ROTATED_GRIDS_KEY))
    orig_example_dict[RADAR_FIELDS_KEY] = [
        str(s) for s in netCDF4.chartostring(
            netcdf_dataset.variables[RADAR_FIELDS_KEY][:])
    ]
    orig_example_dict[RADAR_HEIGHTS_KEY] = numpy.array(
        netcdf_dataset.variables[RADAR_HEIGHTS_KEY][:], dtype=int)

    if example_dict[ROTATED_GRIDS_KEY]:
        orig_example_dict[ROTATED_GRID_SPACING_KEY] = int(
            getattr(netcdf_dataset, ROTATED_GRID_SPACING_KEY))

    if include_soundings:
        orig_example_dict[SOUNDING_FIELDS_KEY] = [
            str(s) for s in netCDF4.chartostring(
                netcdf_dataset.variables[SOUNDING_FIELDS_KEY][:])
        ]
        orig_example_dict[SOUNDING_HEIGHTS_KEY] = numpy.array(
            netcdf_dataset.variables[SOUNDING_HEIGHTS_KEY][:], dtype=int)

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


def remove_storms_with_undefined_target(radar_image_dict):
    """Removes storm objects with undefined target value.

    :param radar_image_dict: Dictionary created by
        `storm_images.read_storm_images`.
    :return: radar_image_dict: Same as input, maybe with fewer storm objects.
    """

    valid_indices = numpy.where(
        radar_image_dict[storm_images.LABEL_VALUES_KEY] !=
        labels.INVALID_STORM_INTEGER
    )[0]

    keys_to_change = [
        storm_images.STORM_IMAGE_MATRIX_KEY, storm_images.STORM_IDS_KEY,
        storm_images.VALID_TIMES_KEY, storm_images.LABEL_VALUES_KEY,
    ]

    for this_key in keys_to_change:
        if this_key == storm_images.STORM_IDS_KEY:
            radar_image_dict[this_key] = [
                radar_image_dict[this_key][i] for i in valid_indices
            ]
        else:
            radar_image_dict[this_key] = radar_image_dict[this_key][
                valid_indices, ...]

    return radar_image_dict


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
        top_sounding_dir_name, radar_file_name_matrix, target_name,
        lag_time_for_convective_contamination_sec):
    """Locates files with storm-centered soundings.

    D = number of SPC dates in time period

    :param top_sounding_dir_name: Name of top-level directory.  Files therein
        will be found by `soundings.find_sounding_file`.
    :param radar_file_name_matrix: numpy array created by either
        `find_storm_images_2d` or `find_storm_images_3d`.  Length of the first
        axis is D.
    :param target_name: Name of target variable (must be accepted by
        `labels.check_label_name`).
    :param lag_time_for_convective_contamination_sec: See doc for
        `soundings.read_soundings`.
    :return: sounding_file_names: length-D list of file paths.
    """

    error_checking.assert_is_numpy_array(radar_file_name_matrix)
    num_file_dimensions = len(radar_file_name_matrix.shape)
    error_checking.assert_is_geq(num_file_dimensions, 2)
    error_checking.assert_is_leq(num_file_dimensions, 3)

    target_param_dict = labels.column_name_to_label_params(target_name)
    min_lead_time_sec = target_param_dict[labels.MIN_LEAD_TIME_KEY]
    max_lead_time_sec = target_param_dict[labels.MAX_LEAD_TIME_KEY]

    mean_lead_time_sec = numpy.mean(
        numpy.array([min_lead_time_sec, max_lead_time_sec], dtype=float)
    )
    mean_lead_time_sec = int(numpy.round(mean_lead_time_sec))

    num_file_times = radar_file_name_matrix.shape[0]
    sounding_file_names = [''] * num_file_times

    for i in range(num_file_times):
        if num_file_dimensions == 2:
            this_file_name = radar_file_name_matrix[i, 0]
        else:
            this_file_name = radar_file_name_matrix[i, 0, 0]

        (this_time_unix_sec, this_spc_date_string
        ) = storm_images.image_file_name_to_time(this_file_name)

        sounding_file_names[i] = soundings.find_sounding_file(
            top_directory_name=top_sounding_dir_name,
            spc_date_string=this_spc_date_string,
            lead_time_seconds=mean_lead_time_sec,
            lag_time_for_convective_contamination_sec=
            lag_time_for_convective_contamination_sec,
            init_time_unix_sec=this_time_unix_sec, raise_error_if_missing=True)

    return sounding_file_names


def find_target_files(top_target_dir_name, radar_file_name_matrix, target_name):
    """Locates files with target values (storm-hazard indicators).

    D = number of SPC dates in time period

    :param top_target_dir_name: Name of top-level directory.  Files therein
        will be found by `storm_images.find_storm_label_file`.
    :param radar_file_name_matrix: numpy array created by either
        `find_storm_images_2d` or `find_storm_images_3d`.  Length of the first
        axis is D.
    :param target_name: Name of target variable (must be accepted by
        `labels.check_label_name`).
    :return: target_file_names: length-D list of file paths.
    """

    error_checking.assert_is_numpy_array(radar_file_name_matrix)
    num_file_dimensions = len(radar_file_name_matrix.shape)
    error_checking.assert_is_geq(num_file_dimensions, 2)
    error_checking.assert_is_leq(num_file_dimensions, 3)

    num_file_times = radar_file_name_matrix.shape[0]
    target_file_names = [''] * num_file_times

    for i in range(num_file_times):
        if num_file_dimensions == 2:
            this_file_name = radar_file_name_matrix[i, 0]
        else:
            this_file_name = radar_file_name_matrix[i, 0, 0]

        target_file_names[i] = storm_images.find_storm_label_file(
            storm_image_file_name=this_file_name,
            top_label_dir_name=top_target_dir_name, label_name=target_name,
            raise_error_if_missing=True)

    return target_file_names


def find_example_file(
        top_directory_name, batch_number=None, raise_error_if_missing=True):
    """Locates file with input examples.

    :param top_directory_name: Name of top-level directory with input examples.
    :param batch_number: Batch number (integer).
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing = True`, this method will error out.
    :return: example_file_name: Path to file with input examples.  If file is
        missing and `raise_error_if_missing = False`, this is the *expected*
        path.
    :raises: ValueError: if file is missing and `raise_error_if_missing = True`.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_integer(batch_number)
    error_checking.assert_is_geq(batch_number, 0)
    error_checking.assert_is_boolean(raise_error_if_missing)

    first_batch_number = int(number_rounding.floor_to_nearest(
        batch_number, NUM_BATCHES_PER_DIRECTORY))
    last_batch_number = first_batch_number + NUM_BATCHES_PER_DIRECTORY - 1

    example_file_name = (
        '{0:s}/batches{1:07d}-{2:07d}/input_examples_batch{3:07d}.nc'
    ).format(top_directory_name, first_batch_number, last_batch_number,
             batch_number)

    if raise_error_if_missing and not os.path.isfile(example_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            example_file_name)
        raise ValueError(error_string)

    return example_file_name


def write_example_file(netcdf_file_name, example_dict, append_to_file=False):
    """Writes input examples to NetCDF file.

    If examples do not include soundings, the following keys are not required in
    `example_dict`.

    - "sounding_field_names"
    - "sounding_heights_m_agl"
    - "sounding_matrix"

    If examples contain both 2-D and 3-D radar images, the following keys are
    required in `example_dict`, while "radar_image_matrix" is not required.

    - "reflectivity_image_matrix_dbz"
    - "az_shear_image_matrix_s01"

    In this case, "radar_heights_m_agl" should contain only reflectivity heights
    and "radar_field_names" should contain only the names of azimuthal-shear
    fields.

    :param netcdf_file_name: Path to output file.
    :param example_dict: Dictionary with the following keys.
    example_dict['storm_ids']: length-E list of storm IDs (strings).
    example_dict['storm_times_unix_sec']: length-E list of valid times.
    example_dict['radar_field_names']: List of radar fields (length C if radar
        images are 2-D, length F_r if 3-D).  Each item must be accepted by
        `radar_utils.check_field_name`.
    example_dict['radar_heights_m_agl']: numpy array of radar heights
        (metres above ground level) (length C if radar images are 2-D, length
        H_r if 3-D).
    example_dict['rotated_grids']: Boolean flag.  If True, storm-centered radar
        grids are rotated so that storm motion is in the +x-direction.
    example_dict['rotated_grid_spacing_metres']: Spacing of rotated grids.  If
        grids are not rotated, this should be None.
    example_dict['radar_image_matrix']: numpy array
        (E x M x N x C or E x M x N x H_r x F_r) of storm-centered radar images.
    example_dict['reflectivity_image_matrix_dbz']: numpy array
        (E x M x N x H_r x 1) of storm-centered reflectivity images.
    example_dict['az_shear_image_matrix_s01']: numpy array (E x M x N x 2) of
        storm-centered azimuthal-shear images.
    example_dict['target_name']: Name of target variable.  Must be accepted by
        `labels.check_label_name`.
    example_dict['target_values']: length-E numpy array of target values
        (integer class labels).
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
            netcdf_file_name, 'a', format='NETCDF3_64BIT_OFFSET')
        _compare_metadata(
            netcdf_dataset=netcdf_dataset, example_dict=example_dict)

        num_examples_orig = len(
            numpy.array(netcdf_dataset.variables[STORM_TIMES_KEY][:])
        )
        num_examples_to_add = len(example_dict[STORM_TIMES_KEY])

        this_string_type = 'S{0:d}'.format(
            netcdf_dataset.dimensions[STORM_ID_CHAR_DIM_KEY].size)
        example_dict[STORM_IDS_KEY] = netCDF4.stringtochar(numpy.array(
            example_dict[STORM_IDS_KEY], dtype=this_string_type))

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
    netcdf_dataset.setncattr(TARGET_NAME_KEY, example_dict[TARGET_NAME_KEY])
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
        numpy.array([len(s) for s in example_dict[STORM_IDS_KEY]])
    )
    num_radar_field_chars = numpy.max(
        numpy.array([len(f) for f in example_dict[RADAR_FIELDS_KEY]])
    )

    netcdf_dataset.createDimension(EXAMPLE_DIMENSION_KEY, None)
    netcdf_dataset.createDimension(STORM_ID_CHAR_DIM_KEY, num_storm_id_chars)
    netcdf_dataset.createDimension(
        RADAR_FIELD_CHAR_DIM_KEY, num_radar_field_chars)

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
    storm_ids_char_array = netCDF4.stringtochar(numpy.array(
        example_dict[STORM_IDS_KEY], dtype=this_string_type))

    netcdf_dataset.createVariable(
        STORM_IDS_KEY, datatype='S1',
        dimensions=(EXAMPLE_DIMENSION_KEY, STORM_ID_CHAR_DIM_KEY))
    netcdf_dataset.variables[STORM_IDS_KEY][:] = numpy.array(
        storm_ids_char_array)

    # Add names of radar fields.
    this_string_type = 'S{0:d}'.format(num_radar_field_chars)
    radar_field_names_char_array = netCDF4.stringtochar(numpy.array(
        example_dict[RADAR_FIELDS_KEY], dtype=this_string_type))

    if num_radar_dimensions == 2:
        this_first_dim_key = RADAR_CHANNEL_DIM_KEY + ''
    else:
        this_first_dim_key = RADAR_FIELD_DIM_KEY + ''

    netcdf_dataset.createVariable(
        RADAR_FIELDS_KEY, datatype='S1',
        dimensions=(this_first_dim_key, RADAR_FIELD_CHAR_DIM_KEY))
    netcdf_dataset.variables[RADAR_FIELDS_KEY][:] = numpy.array(
        radar_field_names_char_array)

    # Add storm times.
    netcdf_dataset.createVariable(
        STORM_TIMES_KEY, datatype=numpy.int32, dimensions=EXAMPLE_DIMENSION_KEY)
    netcdf_dataset.variables[STORM_TIMES_KEY][:] = example_dict[
        STORM_TIMES_KEY]

    # Add target values.
    netcdf_dataset.createVariable(
        TARGET_VALUES_KEY, datatype=numpy.int32,
        dimensions=EXAMPLE_DIMENSION_KEY)
    netcdf_dataset.variables[TARGET_VALUES_KEY][:] = example_dict[
        TARGET_VALUES_KEY]

    # Add radar heights.
    if num_radar_dimensions == 2:
        this_dimension_key = RADAR_CHANNEL_DIM_KEY + ''
    else:
        this_dimension_key = RADAR_HEIGHT_DIM_KEY + ''

    netcdf_dataset.createVariable(
        RADAR_HEIGHTS_KEY, datatype=numpy.int32, dimensions=this_dimension_key)
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
            dimensions=these_dimensions)
        netcdf_dataset.variables[RADAR_IMAGE_MATRIX_KEY][:] = example_dict[
            RADAR_IMAGE_MATRIX_KEY]

    else:
        netcdf_dataset.createVariable(
            REFL_IMAGE_MATRIX_KEY, datatype=numpy.float32,
            dimensions=(
                EXAMPLE_DIMENSION_KEY, REFL_ROW_DIMENSION_KEY,
                REFL_COLUMN_DIMENSION_KEY, RADAR_HEIGHT_DIM_KEY
            ))
        netcdf_dataset.variables[REFL_IMAGE_MATRIX_KEY][:] = example_dict[
            RADAR_IMAGE_MATRIX_KEY][..., 0]

        netcdf_dataset.createVariable(
            AZ_SHEAR_IMAGE_MATRIX_KEY, datatype=numpy.float32,
            dimensions=(
                EXAMPLE_DIMENSION_KEY, AZ_SHEAR_ROW_DIMENSION_KEY,
                AZ_SHEAR_COLUMN_DIMENSION_KEY, RADAR_HEIGHT_DIM_KEY
            ))
        netcdf_dataset.variables[AZ_SHEAR_IMAGE_MATRIX_KEY][:] = example_dict[
            AZ_SHEAR_IMAGE_MATRIX_KEY][..., 0]

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
        example_dict[SOUNDING_FIELDS_KEY], dtype=this_string_type))

    netcdf_dataset.createVariable(
        SOUNDING_FIELDS_KEY, datatype='S1',
        dimensions=(SOUNDING_FIELD_DIM_KEY, SOUNDING_FIELD_CHAR_DIM_KEY))
    netcdf_dataset.variables[SOUNDING_FIELDS_KEY][:] = numpy.array(
        sounding_field_names_char_array)

    netcdf_dataset.createVariable(
        SOUNDING_HEIGHTS_KEY, datatype=numpy.int32,
        dimensions=SOUNDING_HEIGHT_DIM_KEY)
    netcdf_dataset.variables[SOUNDING_HEIGHTS_KEY][:] = example_dict[
        SOUNDING_HEIGHTS_KEY]

    netcdf_dataset.createVariable(
        SOUNDING_MATRIX_KEY, datatype=numpy.float32,
        dimensions=(
            EXAMPLE_DIMENSION_KEY, SOUNDING_HEIGHT_DIM_KEY,
            SOUNDING_FIELD_DIM_KEY
        )
    )
    netcdf_dataset.variables[SOUNDING_MATRIX_KEY][:] = example_dict[
        SOUNDING_MATRIX_KEY]

    netcdf_dataset.close()
    return


def read_example_file(
        netcdf_file_name, radar_field_names_to_keep=None,
        radar_heights_to_keep_m_agl=None, sounding_field_names_to_keep=None,
        sounding_heights_to_keep_m_agl=None, first_time_to_keep_unix_sec=None,
        last_time_to_keep_unix_sec=None, num_rows_to_keep=None,
        num_columns_to_keep=None):
    """Reads input examples from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :param radar_field_names_to_keep: 1-D list of radar fields to keep.  If
        None, all radar fields will be kept.
    :param radar_heights_to_keep_m_agl: 1-D numpy array of radar heights to keep
        (metres above ground level).  If None, all radar heights will be kept.
    :param sounding_field_names_to_keep: 1-D list of sounding fields to keep.
        If None, all sounding fields will be kept.
    :param sounding_heights_to_keep_m_agl: 1-D numpy array of sounding heights
        to keep (metres above ground level).  If None, all sounding heights will
        be kept.
    :param first_time_to_keep_unix_sec: First time to keep.  If
        `first_time_to_keep_unix_sec is None`, all storm objects will be kept.
    :param last_time_to_keep_unix_sec: Last time to keep.  If
        `last_time_to_keep_unix_sec is None`, all storm objects will be kept.
    :param num_rows_to_keep: Number of rows to keep from each storm-centered
        radar image.  If `num_rows_to_keep is None`, all rows will be kept.  If
        `num_rows_to_keep is not None`, radar images will be center-cropped, so
        the image center will always be the storm center.
    :param num_columns_to_keep: Same but for columns.
    :return: example_dict: See doc for `write_example_file`.
    """

    # Check times.
    if first_time_to_keep_unix_sec is None:
        first_time_to_keep_unix_sec = 0
    if last_time_to_keep_unix_sec is None:
        last_time_to_keep_unix_sec = int(1e12)

    error_checking.assert_is_integer(first_time_to_keep_unix_sec)
    error_checking.assert_is_integer(last_time_to_keep_unix_sec)
    error_checking.assert_is_geq(
        last_time_to_keep_unix_sec, first_time_to_keep_unix_sec)

    # Open file and read metadata.
    netcdf_dataset = netCDF4.Dataset(netcdf_file_name)
    target_name = str(getattr(netcdf_dataset, TARGET_NAME_KEY))
    rotated_grids = bool(getattr(netcdf_dataset, ROTATED_GRIDS_KEY))

    if rotated_grids:
        rotated_grid_spacing_metres = getattr(
            netcdf_dataset, ROTATED_GRID_SPACING_KEY)
    else:
        rotated_grid_spacing_metres = None

    storm_ids = [
        str(s) for s in netCDF4.chartostring(
            netcdf_dataset.variables[STORM_IDS_KEY][:])
    ]
    storm_times_unix_sec = numpy.array(
        netcdf_dataset.variables[STORM_TIMES_KEY][:], dtype=int)
    target_values = numpy.array(
        netcdf_dataset.variables[TARGET_VALUES_KEY][:], dtype=int)

    radar_field_names = [
        str(s) for s in netCDF4.chartostring(
            netcdf_dataset.variables[RADAR_FIELDS_KEY][:])
    ]
    radar_heights_m_agl = numpy.array(
        netcdf_dataset.variables[RADAR_HEIGHTS_KEY][:], dtype=int)

    if RADAR_IMAGE_MATRIX_KEY in netcdf_dataset.variables:

        # Error-check desired radar fields and heights.
        if radar_field_names_to_keep is None:
            radar_field_names_to_keep = radar_field_names + []
        if radar_heights_to_keep_m_agl is None:
            radar_heights_to_keep_m_agl = radar_heights_m_agl + 0

        error_checking.assert_is_numpy_array(
            numpy.array(radar_field_names_to_keep), num_dimensions=1)
        radar_heights_to_keep_m_agl = numpy.round(
            radar_heights_to_keep_m_agl).astype(int)
        error_checking.assert_is_numpy_array(
            radar_heights_to_keep_m_agl, num_dimensions=1)

        # Subset radar fields and heights.
        radar_image_matrix = netcdf_dataset.variables[RADAR_IMAGE_MATRIX_KEY][:]
        num_radar_dimensions = len(radar_image_matrix.shape) - 2

        if num_radar_dimensions == 2:
            these_indices = [
                numpy.where(numpy.logical_and(
                    radar_field_names == f, radar_heights_m_agl == h
                ))[0][0]
                for f, h in
                zip(radar_field_names_to_keep, radar_heights_to_keep_m_agl)
            ]

            these_indices = numpy.array(these_indices, dtype=int)
            radar_image_matrix = radar_image_matrix[..., these_indices]
        else:
            these_field_indices = numpy.array(
                [radar_field_names.index(f) for f in radar_field_names_to_keep],
                dtype=int)
            radar_image_matrix = radar_image_matrix[..., these_field_indices]

            these_height_indices = numpy.array([
                numpy.where(radar_heights_m_agl == h)[0][0]
                for h in radar_heights_to_keep_m_agl
            ], dtype=int)
            radar_image_matrix = radar_image_matrix[
                ..., these_height_indices, :]

        radar_field_names = radar_field_names_to_keep + []
        radar_heights_m_agl = radar_heights_to_keep_m_agl + 0

        for k in range(len(radar_field_names)):
            radar_image_matrix[..., k] = storm_images.downsize_storm_images(
                storm_image_matrix=radar_image_matrix[..., k],
                radar_field_name=radar_field_names[k],
                num_rows_to_keep=num_rows_to_keep,
                num_columns_to_keep=num_columns_to_keep)
    else:
        reflectivity_image_matrix_dbz = numpy.expand_dims(
            netcdf_dataset.variables[REFL_IMAGE_MATRIX_KEY][:], axis=-1)
        az_shear_image_matrix_s01 = netcdf_dataset.variables[
            AZ_SHEAR_IMAGE_MATRIX_KEY][:]

        reflectivity_image_matrix_dbz = storm_images.downsize_storm_images(
            storm_image_matrix=reflectivity_image_matrix_dbz,
            radar_field_name=radar_utils.REFL_NAME,
            num_rows_to_keep=num_rows_to_keep,
            num_columns_to_keep=num_columns_to_keep)

        for k in range(len(radar_field_names)):
            az_shear_image_matrix_s01[..., k] = (
                storm_images.downsize_storm_images(
                    storm_image_matrix=az_shear_image_matrix_s01[..., k],
                    radar_field_name=radar_field_names[k],
                    num_rows_to_keep=num_rows_to_keep,
                    num_columns_to_keep=num_columns_to_keep)
            )

    example_dict = {
        ROTATED_GRIDS_KEY: rotated_grids,
        ROTATED_GRID_SPACING_KEY: rotated_grid_spacing_metres,
        TARGET_NAME_KEY: target_name,
        STORM_IDS_KEY: storm_ids,
        STORM_TIMES_KEY: storm_times_unix_sec,
        RADAR_FIELDS_KEY: radar_field_names,
        RADAR_HEIGHTS_KEY: radar_heights_m_agl,
        TARGET_VALUES_KEY: target_values
    }

    if RADAR_IMAGE_MATRIX_KEY in netcdf_dataset.variables:
        example_dict.update({RADAR_IMAGE_MATRIX_KEY: radar_image_matrix})
    else:
        example_dict.update({
            REFL_IMAGE_MATRIX_KEY: reflectivity_image_matrix_dbz,
            AZ_SHEAR_IMAGE_MATRIX_KEY: az_shear_image_matrix_s01
        })

    include_soundings = SOUNDING_FIELDS_KEY in netcdf_dataset.variables
    if include_soundings:

        # Error-check desired sounding fields and heights.
        sounding_field_names = [
            str(s) for s in netCDF4.chartostring(
                netcdf_dataset.variables[SOUNDING_FIELDS_KEY][:])
        ]
        sounding_heights_m_agl = numpy.array(
            netcdf_dataset.variables[SOUNDING_HEIGHTS_KEY][:], dtype=int)

        if sounding_field_names_to_keep is None:
            sounding_field_names_to_keep = sounding_field_names + []
        if sounding_heights_to_keep_m_agl is None:
            sounding_heights_to_keep_m_agl = sounding_heights_m_agl + 0

        error_checking.assert_is_numpy_array(
            numpy.array(sounding_field_names_to_keep), num_dimensions=1)
        sounding_heights_to_keep_m_agl = numpy.round(
            sounding_heights_to_keep_m_agl).astype(int)
        error_checking.assert_is_numpy_array(
            sounding_heights_to_keep_m_agl, num_dimensions=1)

        # Subset sounding fields and heights.
        sounding_matrix = netcdf_dataset.variables[SOUNDING_MATRIX_KEY][:]

        these_field_indices = numpy.array([
            sounding_field_names.index(f) for f in sounding_field_names_to_keep
        ], dtype=int)
        sounding_matrix = sounding_matrix[..., these_field_indices]

        these_height_indices = numpy.array([
            numpy.where(sounding_heights_m_agl == h)[0][0]
            for h in sounding_heights_to_keep_m_agl
        ], dtype=int)
        sounding_matrix = sounding_matrix[..., these_height_indices, :]

        example_dict.update({
            SOUNDING_FIELDS_KEY: sounding_field_names_to_keep,
            SOUNDING_HEIGHTS_KEY: sounding_heights_to_keep_m_agl,
            SOUNDING_MATRIX_KEY: sounding_matrix
        })

    netcdf_dataset.close()

    # Subset times.
    time_indices = numpy.where(numpy.logical_and(
        example_dict[STORM_TIMES_KEY] >= first_time_to_keep_unix_sec,
        example_dict[STORM_TIMES_KEY] <= last_time_to_keep_unix_sec
    ))[0]

    for this_key in MAIN_KEYS:
        if this_key not in REQUIRED_MAIN_KEYS and this_key not in example_dict:
            continue

        if this_key == STORM_IDS_KEY:
            example_dict[this_key] = [
                example_dict[this_key][k] for k in time_indices
            ]
        else:
            example_dict[this_key] = example_dict[this_key][time_indices, ...]

    return example_dict


def shuffle_and_write_examples(
        target_file_names, target_name, num_examples_per_in_file,
        top_output_dir_name, first_output_batch_number,
        radar_file_name_matrix=None, reflectivity_file_name_matrix=None,
        az_shear_file_name_matrix=None,
        num_examples_per_out_chunk=DEFAULT_NUM_EXAMPLES_PER_OUT_CHUNK,
        num_examples_per_out_file=DEFAULT_NUM_EXAMPLES_PER_OUT_FILE,
        class_to_sampling_fraction_dict=None, sounding_file_names=None):
    """Writes many example files.

    If `radar_file_name_matrix is None`, both `reflectivity_file_name_matrix`
    and `az_shear_file_name_matrix` must be specified.

    D = number of SPC dates in time period

    :param target_file_names: length-D list of paths to target files (will be
        read by `read_labels_from_netcdf`).
    :param target_name: Name of target variable (must be accepted by
        `labels.check_label_name`).
    :param num_examples_per_in_file: Number of examples to read from each input
        file.
    :param top_output_dir_name: Name of top-level directory.  Files will be
        written here by `write_example_file`, to locations determined by
        `find_example_file`.
    :param first_output_batch_number: First batch number (integer).  Used to
        determine locations of output files.
    :param radar_file_name_matrix: numpy array created by either
        `find_storm_images_2d` or `find_storm_images_3d`.  Length of the first
        axis is D.
    :param reflectivity_file_name_matrix: numpy array created by
        `find_storm_images_2d3d_myrorss`.  Length of the first axis is D.
    :param az_shear_file_name_matrix: Same.
    :param num_examples_per_out_chunk: Number of examples per output chunk (all
        written to the same file).  Smaller `num_examples_per_out_chunk` =>
        fewer examples from the same or nearby time steps written to the same
        file => more temporal shuffling.
    :param num_examples_per_out_file: Number of examples to write to each output
        file.
    :param class_to_sampling_fraction_dict: Dictionary, where each key is the
        integer ID for a target class (-2 for "dead storm") and each value is
        the sampling fraction.  This allows for class-conditional sampling.
        If `class_to_sampling_fraction_dict is None`, there will be no class-
        conditional sampling.
    :param sounding_file_names: length-D list of paths to sounding files (will
        be read by `soundings.read_soundings`).  If
        `sounding_file_names is None`, examples will not include soundings.
    """

    if radar_file_name_matrix is None:
        error_checking.assert_is_numpy_array(
            reflectivity_file_name_matrix, num_dimensions=2)

        num_file_times = reflectivity_file_name_matrix.shape[0]
        these_dimensions = numpy.array([num_file_times, 2], dtype=int)
        error_checking.assert_is_numpy_array(
            az_shear_file_name_matrix, exact_dimensions=these_dimensions)
    else:
        error_checking.assert_is_numpy_array(radar_file_name_matrix)
        num_file_dimensions = len(radar_file_name_matrix.shape)
        num_file_times = radar_file_name_matrix.shape[0]

        error_checking.assert_is_geq(num_file_dimensions, 2)
        error_checking.assert_is_leq(num_file_dimensions, 3)

    error_checking.assert_is_numpy_array(
        numpy.array(target_file_names),
        exact_dimensions=numpy.array([num_file_times])
    )

    if sounding_file_names is not None:
        error_checking.assert_is_numpy_array(
            numpy.array(sounding_file_names),
            exact_dimensions=numpy.array([num_file_times])
        )

    error_checking.assert_is_integer(num_examples_per_in_file)
    error_checking.assert_is_geq(num_examples_per_in_file, 1)
    error_checking.assert_is_integer(first_output_batch_number)
    error_checking.assert_is_geq(first_output_batch_number, 0)
    error_checking.assert_is_integer(num_examples_per_out_chunk)
    error_checking.assert_is_geq(num_examples_per_out_chunk, 1)
    error_checking.assert_is_integer(num_examples_per_out_file)
    error_checking.assert_is_geq(
        num_examples_per_out_file, num_examples_per_out_chunk)

    storm_ids = []
    storm_times_unix_sec = numpy.array([], dtype=int)
    target_values = numpy.array([], dtype=int)

    for i in range(num_file_times):
        print 'Reading "{0:s}" from: "{1:s}"...'.format(
            target_name, target_file_names[i])
        this_target_dict = labels.read_labels_from_netcdf(
            netcdf_file_name=target_file_names[i], label_name=target_name)

        storm_ids += this_target_dict[labels.STORM_IDS_KEY]
        storm_times_unix_sec = numpy.concatenate((
            storm_times_unix_sec, this_target_dict[labels.VALID_TIMES_KEY]
        ))
        target_values = numpy.concatenate((
            target_values, this_target_dict[labels.LABEL_VALUES_KEY]
        ))

    good_indices = numpy.where(target_values != labels.INVALID_STORM_INTEGER)[0]
    storm_ids = [storm_ids[k] for k in good_indices]
    storm_times_unix_sec = storm_times_unix_sec[good_indices]
    target_values = target_values[good_indices]

    print '\n'
    num_examples_found = len(storm_ids)
    num_examples_to_use = num_examples_per_in_file * num_file_times

    if class_to_sampling_fraction_dict is None:
        indices_to_keep = numpy.linspace(
            0, num_examples_found - 1, num=num_examples_found, dtype=int)

        if num_examples_found > num_examples_to_use:
            indices_to_keep = numpy.random.choice(
                indices_to_keep, size=num_examples_to_use, replace=False)
    else:
        indices_to_keep = dl_utils.sample_by_class(
            sampling_fraction_by_class_dict=class_to_sampling_fraction_dict,
            target_name=target_name, target_values=target_values,
            num_examples_total=num_examples_to_use)

    num_examples_to_use = len(indices_to_keep)
    storm_ids = [storm_ids[k] for k in indices_to_keep]
    storm_times_unix_sec = storm_times_unix_sec[indices_to_keep]
    target_values = target_values[indices_to_keep]

    unique_target_values, unique_counts = numpy.unique(
        target_values, return_counts=True)
    for k in range(len(unique_target_values)):
        print '{0:d} examples with target class = {1:d}'.format(
            unique_counts[k], unique_target_values[k])
    print '\n'

    num_output_files = int(numpy.ceil(
        float(num_examples_to_use) / num_examples_per_out_file
    ))
    batch_numbers = numpy.linspace(
        first_output_batch_number,
        first_output_batch_number + num_output_files - 1,
        num=num_output_files, dtype=int)

    output_file_names = []
    for k in batch_numbers:
        this_file_name = find_example_file(
            top_directory_name=top_output_dir_name, batch_number=k,
            raise_error_if_missing=False)

        if os.path.isfile(this_file_name):
            os.remove(this_file_name)
        output_file_names.append(this_file_name)

    for i in range(num_file_times):
        if radar_file_name_matrix is None:
            this_file_name = reflectivity_file_name_matrix[0, 0]
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

        these_storm_ids = [storm_ids[m] for m in these_indices]
        these_storm_times_unix_sec = storm_times_unix_sec[these_indices]
        these_target_values = target_values[these_indices]

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
                storm_ids=these_storm_ids,
                storm_times_unix_sec=these_storm_times_unix_sec,
                target_values=these_target_values,
                sounding_file_name=this_sounding_file_name,
                sounding_field_names=None)
        elif num_file_dimensions == 3:
            this_example_dict = _create_3d_examples(
                radar_file_name_matrix=radar_file_name_matrix[i, ...],
                storm_ids=these_storm_ids,
                storm_times_unix_sec=these_storm_times_unix_sec,
                target_values=these_target_values,
                sounding_file_name=this_sounding_file_name,
                sounding_field_names=None)
        else:
            this_example_dict = _create_2d_examples(
                radar_file_names=radar_file_name_matrix[i, ...].tolist(),
                storm_ids=these_storm_ids,
                storm_times_unix_sec=these_storm_times_unix_sec,
                target_values=these_target_values,
                sounding_file_name=this_sounding_file_name,
                sounding_field_names=None)

        print '\n'
        if this_example_dict is None:
            continue

        this_example_dict.update({TARGET_NAME_KEY: target_name})
        _write_examples_to_many_files(
            example_dict=this_example_dict, output_file_names=output_file_names,
            num_examples_per_out_chunk=num_examples_per_out_chunk)
        print SEPARATOR_STRING
