"""Unit tests for deep_learning_utils.py"""

import copy
import unittest
import numpy
import pandas
import keras
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import soundings
from gewittergefahr.gg_utils import temperature_conversions
from gewittergefahr.gg_utils import moisture_conversions
from gewittergefahr.gg_utils import standard_atmosphere as standard_atmo
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils

TOLERANCE = 1e-6
TOLERANCE_FOR_CLASS_WEIGHT = 1e-3
TOLERANCE_FOR_METPY_DICTIONARIES = 1e-3

PASCALS_TO_MB = 0.01
METRES_PER_SECOND_TO_KT = 3.6 / 1.852

# The following constants are used to test class_fractions_to_num_examples.
SAMPLING_FRACTION_BY_TORNADO_CLASS_DICT = {0: 0.1, 1: 0.9}
SAMPLING_FRACTION_BY_WIND_3CLASS_DICT = {-2: 0.1, 0: 0.2, 1: 0.7}
WIND_TARGET_NAME_3CLASSES = (
    'wind-speed_percentile=100.0_lead-time=1800-3600sec_distance=00000-10000m'
    '_cutoffs=50kt')

NUM_EXAMPLES_LARGE = 17
NUM_EXAMPLES_LARGE_BY_TORNADO_CLASS_DICT = {0: 2, 1: 15}
NUM_EXAMPLES_LARGE_BY_WIND_3CLASS_DICT = {-2: 2, 0: 3, 1: 12}

NUM_EXAMPLES_MEDIUM = 8
NUM_EXAMPLES_MEDIUM_BY_TORNADO_CLASS_DICT = {0: 1, 1: 7}
NUM_EXAMPLES_MEDIUM_BY_WIND_3CLASS_DICT = {-2: 1, 0: 2, 1: 5}

NUM_EXAMPLES_SMALL = 4
NUM_EXAMPLES_SMALL_BY_TORNADO_CLASS_DICT = {0: 1, 1: 3}
NUM_EXAMPLES_SMALL_BY_WIND_3CLASS_DICT = {-2: 1, 0: 1, 1: 2}

NUM_EXAMPLES_XSMALL = 3
NUM_EXAMPLES_XSMALL_BY_TORNADO_CLASS_DICT = {0: 1, 1: 2}
NUM_EXAMPLES_XSMALL_BY_WIND_3CLASS_DICT = {-2: 1, 0: 1, 1: 1}

# The following constants are used to test class_fractions_to_weights.
LF_WEIGHT_BY_TORNADO_CLASS_DICT = {0: 0.9, 1: 0.1}
LF_WEIGHT_BY_WIND_3CLASS_DICT = {0: 0.7, 1: 0.3}

SAMPLING_FRACTION_BY_WIND_7CLASS_DICT = {
    -2: 0.1, 0: 0.4, 1: 0.2, 2: 0.1, 3: 0.05, 4: 0.05, 5: 0.1}
WIND_TARGET_NAME_7CLASSES = (
    'wind-speed_percentile=100.0_lead-time=1800-3600sec_distance=00000-10000m'
    '_cutoffs=10-20-30-40-50kt')

LF_WEIGHT_BY_WIND_7CLASS_DICT = {
    0: 2. / 67, 1: 5. / 67, 2: 10. / 67, 3: 20. / 67, 4: 20. / 67, 5: 10. / 67}
LF_WEIGHT_BY_WIND_7CLASS_DICT_BINARIZED = {0: 0.1, 1: 0.9}

# The following constants are used to test check_radar_images,
# stack_radar_fields, and stack_radar_heights.
RADAR_IMAGE_MATRIX_1D = numpy.array([1, 2, 3, 4], dtype=numpy.float32)
RADAR_IMAGE_MATRIX_2D = numpy.array([[1, 2, 3, 4],
                                     [5, 6, 7, 8],
                                     [9, 10, 11, 12]], dtype=numpy.float32)

RADAR_IMAGE_MATRIX_3D = numpy.stack(
    (RADAR_IMAGE_MATRIX_2D, RADAR_IMAGE_MATRIX_2D), axis=0)

TUPLE_OF_3D_RADAR_MATRICES = (
    RADAR_IMAGE_MATRIX_3D, RADAR_IMAGE_MATRIX_3D, RADAR_IMAGE_MATRIX_3D,
    RADAR_IMAGE_MATRIX_3D, RADAR_IMAGE_MATRIX_3D, RADAR_IMAGE_MATRIX_3D)
RADAR_IMAGE_MATRIX_4D = numpy.stack(TUPLE_OF_3D_RADAR_MATRICES, axis=-1)

TUPLE_OF_4D_RADAR_MATRICES = (
    RADAR_IMAGE_MATRIX_4D, RADAR_IMAGE_MATRIX_4D, RADAR_IMAGE_MATRIX_4D,
    RADAR_IMAGE_MATRIX_4D, RADAR_IMAGE_MATRIX_4D)
RADAR_IMAGE_MATRIX_5D = numpy.stack(TUPLE_OF_4D_RADAR_MATRICES, axis=-2)

# The following constants are used to test check_target_array.
TORNADO_CLASSES_1D = numpy.array(
    [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1], dtype=int)
WIND_CLASSES_1D = numpy.array(
    [1, 2, 3, 4, 0, 2, 0, 1, 1, 5, 2], dtype=int)

TORNADO_CLASS_MATRIX = keras.utils.to_categorical(TORNADO_CLASSES_1D, 2)
WIND_CLASS_MATRIX = keras.utils.to_categorical(
    WIND_CLASSES_1D, numpy.max(WIND_CLASSES_1D) + 1)

# The following constants are used to test normalize_radar_images and
# denormalize_radar_images.
MIN_NORMALIZED_VALUE = -1.
MAX_NORMALIZED_VALUE = 1.
RADAR_FIELD_NAMES = [radar_utils.REFL_NAME, radar_utils.DIFFERENTIAL_REFL_NAME]

RADAR_NORMALIZATION_DICT = {
    radar_utils.DIFFERENTIAL_REFL_NAME: numpy.array([0, 4, -8, 8], dtype=float),
    radar_utils.REFL_NAME: numpy.array([5, 2, 1, 10], dtype=float)
}
RADAR_NORMALIZATION_TABLE = pandas.DataFrame.from_dict(
    RADAR_NORMALIZATION_DICT, orient='index')

COLUMN_DICT_OLD_TO_NEW = {
    0: dl_utils.MEAN_VALUE_COLUMN,
    1: dl_utils.STANDARD_DEVIATION_COLUMN,
    2: dl_utils.MIN_VALUE_COLUMN,
    3: dl_utils.MAX_VALUE_COLUMN
}
RADAR_NORMALIZATION_TABLE.rename(columns=COLUMN_DICT_OLD_TO_NEW, inplace=True)

REFL_MATRIX_EXAMPLE1_HEIGHT1 = numpy.array(
    [[0, 1, 2, 3],
     [4, 5, 6, 7]], dtype=numpy.float32)
REFL_MATRIX_EXAMPLE1_HEIGHT2 = numpy.array(
    [[2, 4, 6, 8],
     [10, 12, 14, 16]], dtype=numpy.float32)
REFL_MATRIX_EXAMPLE1_HEIGHT3 = numpy.array(
    [[-5, 2, 0, 5],
     [3, -3, 11, 10]], dtype=numpy.float32)

REFL_MATRIX_EXAMPLE2_HEIGHT1 = numpy.array(
    [[3, 2, 1, 2],
     [6, 10, 16, -6]], dtype=numpy.float32)
REFL_MATRIX_EXAMPLE2_HEIGHT2 = numpy.array(
    [[0, 0, 0, 0],
     [0, 1, 1, 2]], dtype=numpy.float32)
REFL_MATRIX_EXAMPLE2_HEIGHT3 = numpy.array(
    [[17, 7, 0, 3],
     [6, 7, 8, 4]], dtype=numpy.float32)

DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT1 = -1 * REFL_MATRIX_EXAMPLE1_HEIGHT1
DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT2 = -1 * REFL_MATRIX_EXAMPLE1_HEIGHT2
DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT3 = -1 * REFL_MATRIX_EXAMPLE1_HEIGHT3
DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT1 = -1 * REFL_MATRIX_EXAMPLE2_HEIGHT1
DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT2 = -1 * REFL_MATRIX_EXAMPLE2_HEIGHT2
DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT3 = -1 * REFL_MATRIX_EXAMPLE2_HEIGHT3

THIS_MATRIX_EXAMPLE1_HEIGHT1 = numpy.stack(
    (REFL_MATRIX_EXAMPLE1_HEIGHT1, DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT1), axis=-1)
THIS_MATRIX_EXAMPLE2_HEIGHT1 = numpy.stack(
    (REFL_MATRIX_EXAMPLE2_HEIGHT1, DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT1), axis=-1)
RADAR_MATRIX_4D_UNNORMALIZED = numpy.stack(
    (THIS_MATRIX_EXAMPLE1_HEIGHT1, THIS_MATRIX_EXAMPLE2_HEIGHT1), axis=0)

THIS_MATRIX_EXAMPLE1_HEIGHT1 = numpy.stack((
    (REFL_MATRIX_EXAMPLE1_HEIGHT1 - 5) / 2,
    (DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT1 - 0) / 4
), axis=-1)
THIS_MATRIX_EXAMPLE2_HEIGHT1 = numpy.stack((
    (REFL_MATRIX_EXAMPLE2_HEIGHT1 - 5) / 2,
    (DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT1 - 0) / 4
), axis=-1)
RADAR_MATRIX_4D_Z_SCORES = numpy.stack(
    (THIS_MATRIX_EXAMPLE1_HEIGHT1, THIS_MATRIX_EXAMPLE2_HEIGHT1), axis=0)

THIS_MATRIX_EXAMPLE1_HEIGHT1 = numpy.stack((
    (REFL_MATRIX_EXAMPLE1_HEIGHT1 - 1) / 9,
    (DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT1 + 8) / 16
), axis=-1)
THIS_MATRIX_EXAMPLE2_HEIGHT1 = numpy.stack((
    (REFL_MATRIX_EXAMPLE2_HEIGHT1 - 1) / 9,
    (DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT1 + 8) / 16
), axis=-1)
RADAR_MATRIX_4D_MINMAX = -1 + 2 * numpy.stack(
    (THIS_MATRIX_EXAMPLE1_HEIGHT1, THIS_MATRIX_EXAMPLE2_HEIGHT1), axis=0)

THIS_MATRIX_EXAMPLE1_HEIGHT2 = numpy.stack(
    (REFL_MATRIX_EXAMPLE1_HEIGHT2, DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT2), axis=-1)
THIS_MATRIX_EXAMPLE2_HEIGHT2 = numpy.stack(
    (REFL_MATRIX_EXAMPLE2_HEIGHT2, DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT2), axis=-1)
THIS_MATRIX_HEIGHT2 = numpy.stack(
    (THIS_MATRIX_EXAMPLE1_HEIGHT2, THIS_MATRIX_EXAMPLE2_HEIGHT2), axis=0)

THIS_MATRIX_EXAMPLE1_HEIGHT3 = numpy.stack(
    (REFL_MATRIX_EXAMPLE1_HEIGHT3, DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT3), axis=-1)
THIS_MATRIX_EXAMPLE2_HEIGHT3 = numpy.stack(
    (REFL_MATRIX_EXAMPLE2_HEIGHT3, DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT3), axis=-1)
THIS_MATRIX_HEIGHT3 = numpy.stack(
    (THIS_MATRIX_EXAMPLE1_HEIGHT3, THIS_MATRIX_EXAMPLE2_HEIGHT3), axis=0)

RADAR_MATRIX_5D_UNNORMALIZED = numpy.stack(
    (RADAR_MATRIX_4D_UNNORMALIZED, THIS_MATRIX_HEIGHT2, THIS_MATRIX_HEIGHT3),
    axis=-2)

THIS_MATRIX_EXAMPLE1_HEIGHT2 = numpy.stack((
    (REFL_MATRIX_EXAMPLE1_HEIGHT2 - 5) / 2,
    (DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT2 - 0) / 4
), axis=-1)
THIS_MATRIX_EXAMPLE2_HEIGHT2 = numpy.stack((
    (REFL_MATRIX_EXAMPLE2_HEIGHT2 - 5) / 2,
    (DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT2 - 0) / 4
), axis=-1)
THIS_MATRIX_HEIGHT2 = numpy.stack(
    (THIS_MATRIX_EXAMPLE1_HEIGHT2, THIS_MATRIX_EXAMPLE2_HEIGHT2), axis=0)

THIS_MATRIX_EXAMPLE1_HEIGHT3 = numpy.stack((
    (REFL_MATRIX_EXAMPLE1_HEIGHT3 - 5) / 2,
    (DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT3 - 0) / 4
), axis=-1)
THIS_MATRIX_EXAMPLE2_HEIGHT3 = numpy.stack((
    (REFL_MATRIX_EXAMPLE2_HEIGHT3 - 5) / 2,
    (DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT3 - 0) / 4
), axis=-1)
THIS_MATRIX_HEIGHT3 = numpy.stack(
    (THIS_MATRIX_EXAMPLE1_HEIGHT3, THIS_MATRIX_EXAMPLE2_HEIGHT3), axis=0)

RADAR_MATRIX_5D_Z_SCORES = numpy.stack(
    (RADAR_MATRIX_4D_Z_SCORES, THIS_MATRIX_HEIGHT2, THIS_MATRIX_HEIGHT3),
    axis=-2)

THIS_MATRIX_EXAMPLE1_HEIGHT2 = numpy.stack((
    (REFL_MATRIX_EXAMPLE1_HEIGHT2 - 1) / 9,
    (DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT2 + 8) / 16
), axis=-1)
THIS_MATRIX_EXAMPLE2_HEIGHT2 = numpy.stack((
    (REFL_MATRIX_EXAMPLE2_HEIGHT2 - 1) / 9,
    (DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT2 + 8) / 16
), axis=-1)
THIS_MATRIX_HEIGHT2 = -1 + 2 * numpy.stack(
    (THIS_MATRIX_EXAMPLE1_HEIGHT2, THIS_MATRIX_EXAMPLE2_HEIGHT2), axis=0)

THIS_MATRIX_EXAMPLE1_HEIGHT3 = numpy.stack((
    (REFL_MATRIX_EXAMPLE1_HEIGHT3 - 1) / 9,
    (DIFF_REFL_MATRIX_EXAMPLE1_HEIGHT3 + 8) / 16
), axis=-1)
THIS_MATRIX_EXAMPLE2_HEIGHT3 = numpy.stack((
    (REFL_MATRIX_EXAMPLE2_HEIGHT3 - 1) / 9,
    (DIFF_REFL_MATRIX_EXAMPLE2_HEIGHT3 + 8) / 16
), axis=-1)
THIS_MATRIX_HEIGHT3 = -1 + 2 * numpy.stack(
    (THIS_MATRIX_EXAMPLE1_HEIGHT3, THIS_MATRIX_EXAMPLE2_HEIGHT3), axis=0)

RADAR_MATRIX_5D_MINMAX = numpy.stack(
    (RADAR_MATRIX_4D_MINMAX, THIS_MATRIX_HEIGHT2, THIS_MATRIX_HEIGHT3),
    axis=-2)

# The following constants are used to test mask_low_reflectivity_pixels.
FIELD_NAMES_FOR_MASKING = [radar_utils.REFL_NAME, radar_utils.DIVERGENCE_NAME]
MASKING_THRESHOLD_DBZ = 15.

THIS_REFL_MATRIX_HEIGHT2 = numpy.array([[17, 24, 1, 8],
                                        [23, 5, 7, 14],
                                        [4, 6, 13, 20],
                                        [10, 12, 19, 21],
                                        [11, 18, 25, 2]], dtype=float)
THIS_REFL_MATRIX_HEIGHT1 = 2 * THIS_REFL_MATRIX_HEIGHT2
THIS_REFL_MATRIX_HEIGHT0 = 3 * THIS_REFL_MATRIX_HEIGHT2
THIS_REFL_MATRIX_EXAMPLE0 = numpy.stack(
    (THIS_REFL_MATRIX_HEIGHT0, THIS_REFL_MATRIX_HEIGHT1,
     THIS_REFL_MATRIX_HEIGHT2), axis=-1)

THIS_REFL_MATRIX_HEIGHT2_MASKED = numpy.array([[17, 24, 0, 0],
                                               [23, 0, 0, 0],
                                               [0, 0, 0, 20],
                                               [0, 0, 19, 21],
                                               [0, 18, 25, 0]], dtype=float)
THIS_REFL_MATRIX_HEIGHT1_MASKED = numpy.array([[34, 48, 0, 16],
                                               [46, 0, 0, 28],
                                               [0, 0, 26, 40],
                                               [20, 24, 38, 42],
                                               [22, 36, 50, 0]], dtype=float)
THIS_REFL_MATRIX_HEIGHT0_MASKED = numpy.array([[51, 72, 0, 24],
                                               [69, 15, 21, 42],
                                               [0, 18, 39, 60],
                                               [30, 36, 57, 63],
                                               [33, 54, 75, 0]], dtype=float)
THIS_REFL_MATRIX_EXAMPLE0_MASKED = numpy.stack(
    (THIS_REFL_MATRIX_HEIGHT0_MASKED, THIS_REFL_MATRIX_HEIGHT1_MASKED,
     THIS_REFL_MATRIX_HEIGHT2_MASKED), axis=-1)

THIS_REFL_MATRIX_HEIGHT2 = numpy.array([[17, 19, 18, 21],
                                        [0, 19, 0, 18],
                                        [22, 10, 7, 8],
                                        [24, 17, 1, 24],
                                        [17, 4, 2, 0]], dtype=float)
THIS_REFL_MATRIX_HEIGHT1 = 2 * THIS_REFL_MATRIX_HEIGHT2
THIS_REFL_MATRIX_HEIGHT0 = 3 * THIS_REFL_MATRIX_HEIGHT2
THIS_REFL_MATRIX_EXAMPLE1 = numpy.stack(
    (THIS_REFL_MATRIX_HEIGHT0, THIS_REFL_MATRIX_HEIGHT1,
     THIS_REFL_MATRIX_HEIGHT2), axis=-1)

THIS_REFL_MATRIX_HEIGHT2_MASKED = numpy.array([[17, 19, 18, 21],
                                               [0, 19, 0, 18],
                                               [22, 0, 0, 0],
                                               [24, 17, 0, 24],
                                               [17, 0, 0, 0]], dtype=float)
THIS_REFL_MATRIX_HEIGHT1_MASKED = numpy.array([[34, 38, 36, 42],
                                               [0, 38, 0, 36],
                                               [44, 20, 0, 16],
                                               [48, 34, 0, 48],
                                               [34, 0, 0, 0]], dtype=float)
THIS_REFL_MATRIX_HEIGHT0_MASKED = numpy.array([[51, 57, 54, 63],
                                               [0, 57, 0, 54],
                                               [66, 30, 21, 24],
                                               [72, 51, 0, 72],
                                               [51, 0, 0, 0]], dtype=float)
THIS_REFL_MATRIX_EXAMPLE1_MASKED = numpy.stack(
    (THIS_REFL_MATRIX_HEIGHT0_MASKED, THIS_REFL_MATRIX_HEIGHT1_MASKED,
     THIS_REFL_MATRIX_HEIGHT2_MASKED), axis=-1)

THIS_DIVERGENCE_MATRIX_EXAMPLE0 = 0.001 * THIS_REFL_MATRIX_EXAMPLE0
THIS_DIVERGENCE_MATRIX_EXAMPLE0_MASKED = (
    0.001 * THIS_REFL_MATRIX_EXAMPLE0_MASKED)
THIS_DIVERGENCE_MATRIX_EXAMPLE1 = 0.001 * THIS_REFL_MATRIX_EXAMPLE1
THIS_DIVERGENCE_MATRIX_EXAMPLE1_MASKED = (
    0.001 * THIS_REFL_MATRIX_EXAMPLE1_MASKED)

THIS_EXAMPLE0_MATRIX = numpy.stack(
    (THIS_REFL_MATRIX_EXAMPLE0, THIS_DIVERGENCE_MATRIX_EXAMPLE0), axis=-1)
THIS_EXAMPLE1_MATRIX = numpy.stack(
    (THIS_REFL_MATRIX_EXAMPLE1, THIS_DIVERGENCE_MATRIX_EXAMPLE1), axis=-1)
RADAR_MATRIX_3D_UNMASKED = numpy.stack(
    (THIS_EXAMPLE0_MATRIX, THIS_EXAMPLE1_MATRIX), axis=0)

THIS_EXAMPLE0_MATRIX_MASKED = numpy.stack(
    (THIS_REFL_MATRIX_EXAMPLE0_MASKED, THIS_DIVERGENCE_MATRIX_EXAMPLE0_MASKED),
    axis=-1)
THIS_EXAMPLE1_MATRIX_MASKED = numpy.stack(
    (THIS_REFL_MATRIX_EXAMPLE1_MASKED, THIS_DIVERGENCE_MATRIX_EXAMPLE1_MASKED),
    axis=-1)
RADAR_MATRIX_3D_MASKED = numpy.stack(
    (THIS_EXAMPLE0_MATRIX_MASKED, THIS_EXAMPLE1_MATRIX_MASKED), axis=0)

# The following constants are used to test soundings_to_metpy_dictionaries.
SOUNDING_FIELD_NAMES_NO_PRESSURE = [
    soundings.RELATIVE_HUMIDITY_NAME, soundings.TEMPERATURE_NAME,
    soundings.U_WIND_NAME, soundings.V_WIND_NAME,
    soundings.SPECIFIC_HUMIDITY_NAME,
    soundings.VIRTUAL_POTENTIAL_TEMPERATURE_NAME
]

SOUNDING_FIELD_NAMES = (
    [soundings.PRESSURE_NAME] + SOUNDING_FIELD_NAMES_NO_PRESSURE)

THIS_FIRST_MATRIX = numpy.array([[0.9, 300, -10, 5, 0.02, 310],
                                 [0.7, 285, 0, 15, 0.015, 310],
                                 [0.95, 270, 15, 20, 0.01, 325],
                                 [0.93, 260, 30, 30, 0.007, 341]])
THIS_SECOND_MATRIX = numpy.array([[0.7, 305, 0, 0, 0.015, 312.5],
                                  [0.5, 290, 15, 12.5, 0.015, 310],
                                  [0.8, 273, 25, 15, 0.012, 333],
                                  [0.9, 262, 40, 20, 0.008, 345]])
SOUNDING_MATRIX_NO_PRESSURE = numpy.stack(
    (THIS_FIRST_MATRIX, THIS_SECOND_MATRIX), axis=0)

HEIGHT_LEVELS_M_AGL = numpy.array([0, 1500, 3000, 5000], dtype=int)
STORM_ELEVATIONS_M_ASL = numpy.array([1045, 645], dtype=float)

FIRST_PRESSURES_PASCALS = standard_atmo.height_to_pressure(
    STORM_ELEVATIONS_M_ASL[0] + HEIGHT_LEVELS_M_AGL)
FIRST_PRESSURES_PASCALS = numpy.reshape(
    FIRST_PRESSURES_PASCALS, (len(FIRST_PRESSURES_PASCALS), 1))
THIS_FIRST_MATRIX = numpy.hstack((FIRST_PRESSURES_PASCALS, THIS_FIRST_MATRIX))

SECOND_PRESSURES_PASCALS = standard_atmo.height_to_pressure(
    STORM_ELEVATIONS_M_ASL[1] + HEIGHT_LEVELS_M_AGL)
SECOND_PRESSURES_PASCALS = numpy.reshape(
    SECOND_PRESSURES_PASCALS, (len(SECOND_PRESSURES_PASCALS), 1))
THIS_SECOND_MATRIX = numpy.hstack((
    SECOND_PRESSURES_PASCALS, THIS_SECOND_MATRIX))

SOUNDING_MATRIX_UNNORMALIZED = numpy.stack(
    (THIS_FIRST_MATRIX, THIS_SECOND_MATRIX), axis=0)

FIRST_DEWPOINTS_KELVINS = moisture_conversions.specific_humidity_to_dewpoint(
    specific_humidities_kg_kg01=THIS_FIRST_MATRIX[:, 5],
    total_pressures_pascals=THIS_FIRST_MATRIX[:, 0])

FIRST_METPY_DICT = {
    soundings.TEMPERATURE_COLUMN_METPY:
        temperature_conversions.kelvins_to_celsius(THIS_FIRST_MATRIX[:, 2]),
    soundings.U_WIND_COLUMN_METPY:
        METRES_PER_SECOND_TO_KT * THIS_FIRST_MATRIX[:, 3],
    soundings.V_WIND_COLUMN_METPY:
        METRES_PER_SECOND_TO_KT * THIS_FIRST_MATRIX[:, 4],
    soundings.DEWPOINT_COLUMN_METPY:
        temperature_conversions.kelvins_to_celsius(FIRST_DEWPOINTS_KELVINS),
    soundings.PRESSURE_COLUMN_METPY: THIS_FIRST_MATRIX[:, 0] * PASCALS_TO_MB
}

SECOND_DEWPOINTS_KELVINS = moisture_conversions.specific_humidity_to_dewpoint(
    specific_humidities_kg_kg01=THIS_SECOND_MATRIX[:, 5],
    total_pressures_pascals=THIS_SECOND_MATRIX[:, 0])

SECOND_METPY_DICT = {
    soundings.TEMPERATURE_COLUMN_METPY:
        temperature_conversions.kelvins_to_celsius(THIS_SECOND_MATRIX[:, 2]),
    soundings.U_WIND_COLUMN_METPY:
        METRES_PER_SECOND_TO_KT * THIS_SECOND_MATRIX[:, 3],
    soundings.V_WIND_COLUMN_METPY:
        METRES_PER_SECOND_TO_KT * THIS_SECOND_MATRIX[:, 4],
    soundings.DEWPOINT_COLUMN_METPY:
        temperature_conversions.kelvins_to_celsius(SECOND_DEWPOINTS_KELVINS),
    soundings.PRESSURE_COLUMN_METPY: THIS_SECOND_MATRIX[:, 0] * PASCALS_TO_MB
}

LIST_OF_METPY_DICTIONARIES = [FIRST_METPY_DICT, SECOND_METPY_DICT]

# The following constants are used to test normalize_soundings.
SOUNDING_NORMALIZATION_DICT = {
    soundings.PRESSURE_NAME:
        numpy.array([50000, 20000, 10000, 100000], dtype=float),
    soundings.RELATIVE_HUMIDITY_NAME: numpy.array([0.5, 0.2, 0, 1]),
    soundings.TEMPERATURE_NAME: numpy.array([280, 10, 250, 300], dtype=float),
    soundings.U_WIND_NAME: numpy.array([5, 10, -10, 40], dtype=float),
    soundings.V_WIND_NAME: numpy.array([10, 5, -20, 30], dtype=float),
    soundings.SPECIFIC_HUMIDITY_NAME: numpy.array([0.005, 0.005, 0, 0.02]),
    soundings.VIRTUAL_POTENTIAL_TEMPERATURE_NAME:
        numpy.array([310, 10, 300, 350], dtype=float)
}
SOUNDING_NORMALIZATION_TABLE = pandas.DataFrame.from_dict(
    SOUNDING_NORMALIZATION_DICT, orient='index')
SOUNDING_NORMALIZATION_TABLE.rename(
    columns=COLUMN_DICT_OLD_TO_NEW, inplace=True)

THIS_FIRST_MATRIX = numpy.array([[2, 2, -1.5, -1, 3, 0],
                                 [1, 0.5, -0.5, 1, 2, 0],
                                 [2.25, -1, 1, 2, 1, 1.5],
                                 [2.15, -2, 2.5, 4, 0.4, 3.1]])
THIS_SECOND_MATRIX = numpy.array([[1, 2.5, -0.5, -2, 2, 0.25],
                                  [0, 1, 1, 0.5, 2, 0],
                                  [1.5, -0.7, 2, 1, 1.4, 2.3],
                                  [2, -1.8, 3.5, 2, 0.6, 3.5]])

THESE_FIRST_PRESSURES = (FIRST_PRESSURES_PASCALS - 50000) / 20000
THESE_SECOND_PRESSURES = (SECOND_PRESSURES_PASCALS - 50000) / 20000
THIS_FIRST_MATRIX = numpy.hstack((THESE_FIRST_PRESSURES, THIS_FIRST_MATRIX))
THIS_SECOND_MATRIX = numpy.hstack((THESE_SECOND_PRESSURES, THIS_SECOND_MATRIX))
SOUNDING_MATRIX_Z_SCORES = numpy.stack(
    (THIS_FIRST_MATRIX, THIS_SECOND_MATRIX), axis=0)

THIS_FIRST_MATRIX = numpy.array([[0.9, 1, 0, 0.5, 1, 0.2],
                                 [0.7, 0.7, 0.2, 0.7, 0.75, 0.2],
                                 [0.95, 0.4, 0.5, 0.8, 0.5, 0.5],
                                 [0.93, 0.2, 0.8, 1, 0.35, 0.82]])
THIS_SECOND_MATRIX = numpy.array([[0.7, 1.1, 0.2, 0.4, 0.75, 0.25],
                                  [0.5, 0.8, 0.5, 0.65, 0.75, 0.2],
                                  [0.8, 0.46, 0.7, 0.7, 0.6, 0.66],
                                  [0.9, 0.24, 1, 0.8, 0.4, 0.9]])

THESE_FIRST_PRESSURES = (FIRST_PRESSURES_PASCALS - 10000) / 90000
THESE_SECOND_PRESSURES = (SECOND_PRESSURES_PASCALS - 10000) / 90000
THIS_FIRST_MATRIX = numpy.hstack((THESE_FIRST_PRESSURES, THIS_FIRST_MATRIX))
THIS_SECOND_MATRIX = numpy.hstack((THESE_SECOND_PRESSURES, THIS_SECOND_MATRIX))
SOUNDING_MATRIX_MINMAX = -1 + 2 * numpy.stack(
    (THIS_FIRST_MATRIX, THIS_SECOND_MATRIX), axis=0)

# The following constants are used to test sample_by_class.
TORNADO_LABELS_TO_SAMPLE = numpy.array(
    [0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0,
     1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    dtype=int)
TORNADO_TARGET_NAME = 'tornado_lead-time=0000-3600sec_distance=00000-10000m'

NUM_EXAMPLES_TOTAL = 30
BALANCED_FRACTION_BY_TORNADO_CLASS_DICT = {0: 0.5, 1: 0.5}
TORNADO_INDICES_TO_KEEP = numpy.array(
    [0, 1, 2, 4, 7, 8, 9, 10, 11, 13, 3, 5, 6, 12, 15, 16, 19, 25, 45, 48],
    dtype=int)

WIND_LABELS_TO_SAMPLE = numpy.array(
    [2, 1, 1, 2, 0, -2, -2, 1, -2, 2, 2, 1, 2, 1, 0, 0, 2, 1, 1, 0, 0, -2, 0, 1,
     2, 2, 2, 0, -2, 0, 0, 2, -2, 2, -2, 0, -2, 0, 1, -2, 2, -2, 2, 1, 1, 1, 0,
     0, 0, 1], dtype=int)
WIND_TARGET_NAME_4CLASSES = (
    'wind-speed_percentile=100.0_lead-time=1800-3600sec_distance=00000-10000m'
    '_cutoffs=30-50kt')

SAMPLING_FRACTION_BY_WIND_4CLASS_DICT = {-2: 0.2, 0: 0.1, 1: 0.4, 2: 0.3}
WIND_INDICES_TO_KEEP = numpy.array(
    [4, 14, 15, 1, 2, 7, 11, 13, 17, 18, 23, 38, 43, 44, 45, 0, 3, 9, 10, 12,
     16, 24, 25, 26, 5, 6, 8, 21, 28, 32], dtype=int)


def _compare_lists_of_metpy_dicts(first_list_of_dicts, second_list_of_dicts):
    """Compares two lists of MetPy dictionaries.

    :param first_list_of_dicts: First list.
    :param second_list_of_dicts: Second list.
    :return: are_lists_equal: Boolean flag, indicating whether or not the two
        lists are equal.
    """

    num_first_dicts = len(first_list_of_dicts)
    num_second_dicts = len(second_list_of_dicts)
    if num_first_dicts != num_second_dicts:
        return False

    for i in range(num_first_dicts):
        these_first_keys = list(first_list_of_dicts[i].keys())
        these_second_keys = list(second_list_of_dicts[i].keys())
        if set(these_first_keys) != set(these_second_keys):
            return False

        for this_key in these_first_keys:
            if not numpy.allclose(
                    first_list_of_dicts[i][this_key],
                    second_list_of_dicts[i][this_key],
                    atol=TOLERANCE_FOR_METPY_DICTIONARIES):
                return False

    return True


class DeepLearningUtilsTests(unittest.TestCase):
    """Each method is a unit test for deep_learning_utils.py."""

    def test_class_fractions_to_num_examples_tornado_large(self):
        """Ensures correct output from class_fractions_to_num_examples.

        In this case, target variable = tornado occurrence and number of
        examples to draw is large.
        """

        this_dict = dl_utils.class_fractions_to_num_examples(
            sampling_fraction_by_class_dict=
            SAMPLING_FRACTION_BY_TORNADO_CLASS_DICT,
            target_name=TORNADO_TARGET_NAME,
            num_examples_total=NUM_EXAMPLES_LARGE)

        self.assertTrue(this_dict == NUM_EXAMPLES_LARGE_BY_TORNADO_CLASS_DICT)

    def test_class_fractions_to_num_examples_wind_large(self):
        """Ensures correct output from class_fractions_to_num_examples.

        In this case, target variable = wind-speed class and number of examples
        to draw is large.
        """

        this_dict = dl_utils.class_fractions_to_num_examples(
            sampling_fraction_by_class_dict=
            SAMPLING_FRACTION_BY_WIND_3CLASS_DICT,
            target_name=WIND_TARGET_NAME_3CLASSES,
            num_examples_total=NUM_EXAMPLES_LARGE)

        self.assertTrue(this_dict == NUM_EXAMPLES_LARGE_BY_WIND_3CLASS_DICT)

    def test_class_fractions_to_num_examples_tornado_medium(self):
        """Ensures correct output from class_fractions_to_num_examples.

        In this case, target variable = tornado occurrence and number of
        examples to draw is medium.
        """

        this_dict = dl_utils.class_fractions_to_num_examples(
            sampling_fraction_by_class_dict=
            SAMPLING_FRACTION_BY_TORNADO_CLASS_DICT,
            target_name=TORNADO_TARGET_NAME,
            num_examples_total=NUM_EXAMPLES_MEDIUM)

        self.assertTrue(this_dict == NUM_EXAMPLES_MEDIUM_BY_TORNADO_CLASS_DICT)

    def test_class_fractions_to_num_examples_wind_medium(self):
        """Ensures correct output from class_fractions_to_num_examples.

        In this case, target variable = wind-speed class and number of examples
        to draw is medium.
        """

        this_dict = dl_utils.class_fractions_to_num_examples(
            sampling_fraction_by_class_dict=
            SAMPLING_FRACTION_BY_WIND_3CLASS_DICT,
            target_name=WIND_TARGET_NAME_3CLASSES,
            num_examples_total=NUM_EXAMPLES_MEDIUM)

        self.assertTrue(this_dict == NUM_EXAMPLES_MEDIUM_BY_WIND_3CLASS_DICT)

    def test_class_fractions_to_num_examples_tornado_small(self):
        """Ensures correct output from class_fractions_to_num_examples.

        In this case, target variable = tornado occurrence and number of
        examples to draw is small.
        """

        this_dict = dl_utils.class_fractions_to_num_examples(
            sampling_fraction_by_class_dict=
            SAMPLING_FRACTION_BY_TORNADO_CLASS_DICT,
            target_name=TORNADO_TARGET_NAME,
            num_examples_total=NUM_EXAMPLES_SMALL)

        self.assertTrue(this_dict == NUM_EXAMPLES_SMALL_BY_TORNADO_CLASS_DICT)

    def test_class_fractions_to_num_examples_wind_small(self):
        """Ensures correct output from class_fractions_to_num_examples.

        In this case, target variable = wind-speed class and number of examples
        to draw is small.
        """

        this_dict = dl_utils.class_fractions_to_num_examples(
            sampling_fraction_by_class_dict=
            SAMPLING_FRACTION_BY_WIND_3CLASS_DICT,
            target_name=WIND_TARGET_NAME_3CLASSES,
            num_examples_total=NUM_EXAMPLES_SMALL)

        self.assertTrue(this_dict == NUM_EXAMPLES_SMALL_BY_WIND_3CLASS_DICT)

    def test_class_fractions_to_num_examples_tornado_xsmall(self):
        """Ensures correct output from class_fractions_to_num_examples.

        In this case, target variable = tornado occurrence and number of
        examples to draw is extra small.
        """

        this_dict = dl_utils.class_fractions_to_num_examples(
            sampling_fraction_by_class_dict=
            SAMPLING_FRACTION_BY_TORNADO_CLASS_DICT,
            target_name=TORNADO_TARGET_NAME,
            num_examples_total=NUM_EXAMPLES_XSMALL)

        self.assertTrue(this_dict == NUM_EXAMPLES_XSMALL_BY_TORNADO_CLASS_DICT)

    def test_class_fractions_to_num_examples_wind_xsmall(self):
        """Ensures correct output from class_fractions_to_num_examples.

        In this case, target variable = wind-speed class and number of examples
        to draw is extra small.
        """

        this_dict = dl_utils.class_fractions_to_num_examples(
            sampling_fraction_by_class_dict=
            SAMPLING_FRACTION_BY_WIND_3CLASS_DICT,
            target_name=WIND_TARGET_NAME_3CLASSES,
            num_examples_total=NUM_EXAMPLES_XSMALL)

        self.assertTrue(this_dict == NUM_EXAMPLES_XSMALL_BY_WIND_3CLASS_DICT)

    def test_class_fractions_to_weights_tornado_nonbinarized(self):
        """Ensures correct output from class_fractions_to_weights.

        In this case, target variable = tornado occurrence and binarize_target =
        False.
        """

        this_dict = dl_utils.class_fractions_to_weights(
            sampling_fraction_by_class_dict=
            SAMPLING_FRACTION_BY_TORNADO_CLASS_DICT,
            target_name=TORNADO_TARGET_NAME, binarize_target=False)

        self.assertTrue(this_dict == LF_WEIGHT_BY_TORNADO_CLASS_DICT)

    def test_class_fractions_to_weights_tornado_binarized(self):
        """Ensures correct output from class_fractions_to_weights.

        In this case, target variable = tornado occurrence and binarize_target =
        True.
        """

        this_dict = dl_utils.class_fractions_to_weights(
            sampling_fraction_by_class_dict=
            SAMPLING_FRACTION_BY_TORNADO_CLASS_DICT,
            target_name=TORNADO_TARGET_NAME, binarize_target=True)

        expected_keys = list(LF_WEIGHT_BY_TORNADO_CLASS_DICT.keys())
        actual_keys = list(this_dict.keys())
        self.assertTrue(set(expected_keys) == set(actual_keys))

        for this_key in expected_keys:
            self.assertTrue(numpy.isclose(
                LF_WEIGHT_BY_TORNADO_CLASS_DICT[this_key], this_dict[this_key],
                atol=TOLERANCE_FOR_CLASS_WEIGHT))

    def test_class_fractions_to_weights_wind_3class_nonbinarized(self):
        """Ensures correct output from class_fractions_to_weights.

        In this case, target variable = wind-speed class; there are 3 classes;
        and binarize_target = False.
        """

        this_dict = dl_utils.class_fractions_to_weights(
            sampling_fraction_by_class_dict=
            SAMPLING_FRACTION_BY_WIND_3CLASS_DICT,
            target_name=WIND_TARGET_NAME_3CLASSES, binarize_target=False)

        self.assertTrue(this_dict == LF_WEIGHT_BY_WIND_3CLASS_DICT)

    def test_class_fractions_to_weights_wind_3class_binarized(self):
        """Ensures correct output from class_fractions_to_weights.

        In this case, target variable = wind-speed class; there are 3 classes;
        and binarize_target = True.
        """

        this_dict = dl_utils.class_fractions_to_weights(
            sampling_fraction_by_class_dict=
            SAMPLING_FRACTION_BY_WIND_3CLASS_DICT,
            target_name=WIND_TARGET_NAME_3CLASSES, binarize_target=True)

        self.assertTrue(this_dict == LF_WEIGHT_BY_WIND_3CLASS_DICT)

    def test_class_fractions_to_weights_wind_7class_nonbinarized(self):
        """Ensures correct output from class_fractions_to_weights.

        In this case, target variable = wind-speed class; there are 7 classes;
        and binarize_target = False.
        """

        this_dict = dl_utils.class_fractions_to_weights(
            sampling_fraction_by_class_dict=
            SAMPLING_FRACTION_BY_WIND_7CLASS_DICT,
            target_name=WIND_TARGET_NAME_7CLASSES, binarize_target=False)

        self.assertTrue(this_dict == LF_WEIGHT_BY_WIND_7CLASS_DICT)

    def test_class_fractions_to_weights_wind_7class_binarized(self):
        """Ensures correct output from class_fractions_to_weights.

        In this case, target variable = wind-speed class; there are 7 classes;
        and binarize_target = True.
        """

        this_dict = dl_utils.class_fractions_to_weights(
            sampling_fraction_by_class_dict=
            SAMPLING_FRACTION_BY_WIND_7CLASS_DICT,
            target_name=WIND_TARGET_NAME_7CLASSES, binarize_target=True)

        self.assertTrue(this_dict == LF_WEIGHT_BY_WIND_7CLASS_DICT_BINARIZED)

    def test_check_radar_images_1d(self):
        """Ensures correct output from check_radar_images.

        In this case the input matrix is 1-D (bad).
        """

        with self.assertRaises(ValueError):
            dl_utils.check_radar_images(
                radar_image_matrix=RADAR_IMAGE_MATRIX_1D)

    def test_check_radar_images_2d(self):
        """Ensures correct output from check_radar_images.

        In this case the input matrix is 2-D (bad).
        """

        with self.assertRaises(ValueError):
            dl_utils.check_radar_images(
                radar_image_matrix=RADAR_IMAGE_MATRIX_2D)

    def test_check_radar_images_3d_good(self):
        """Ensures correct output from check_radar_images.

        In this case the input matrix is 3-D (good).
        """

        dl_utils.check_radar_images(radar_image_matrix=RADAR_IMAGE_MATRIX_3D)

    def test_check_radar_images_3d_bad(self):
        """Ensures correct output from check_radar_images.

        In this case, the input matrix is 3-D but a 4-D or 5-D matrix is
        expected.
        """

        with self.assertRaises(ValueError):
            dl_utils.check_radar_images(
                radar_image_matrix=RADAR_IMAGE_MATRIX_3D, min_num_dimensions=4)

    def test_check_radar_images_4d(self):
        """Ensures correct output from check_radar_images.

        In this case the input matrix is 4-D (good).
        """

        dl_utils.check_radar_images(radar_image_matrix=RADAR_IMAGE_MATRIX_4D)

    def test_check_radar_images_5d_good(self):
        """Ensures correct output from check_radar_images.

        In this case the input matrix is 5-D (good).
        """

        dl_utils.check_radar_images(radar_image_matrix=RADAR_IMAGE_MATRIX_5D)

    def test_check_radar_images_5d_bad(self):
        """Ensures correct output from check_radar_images.

        In this case, the input matrix is 5-D but a 3-D or 4-D matrix is
        expected.
        """

        with self.assertRaises(ValueError):
            dl_utils.check_radar_images(
                radar_image_matrix=RADAR_IMAGE_MATRIX_5D, max_num_dimensions=4)

    def test_check_target_array_1d_binary_good(self):
        """Ensures correct output from check_target_array.

        In this case the input array is 1-D and contains 2 classes, as expected.
        """

        dl_utils.check_target_array(
            target_array=TORNADO_CLASSES_1D, num_dimensions=1, num_classes=2)

    def test_check_target_array_1d_binary_bad_dim(self):
        """Ensures correct output from check_target_array.

        In this case, the input array is 1-D but a 2-D array is expected.
        """

        with self.assertRaises(TypeError):
            dl_utils.check_target_array(
                target_array=TORNADO_CLASSES_1D, num_dimensions=2,
                num_classes=2)

    def test_check_target_array_1d_binary_bad_class_num(self):
        """Ensures correct output from check_target_array.

        In this case, 6 classes are expected and the input array contains only 2
        classes.  However, there is no way to ascertain that the 2-class array
        is wrong (maybe higher classes just did not occur in the sample).
        """

        dl_utils.check_target_array(
            target_array=TORNADO_CLASSES_1D, num_dimensions=1,
            num_classes=WIND_CLASS_MATRIX.shape[1])

    def test_check_target_array_1d_multiclass_good(self):
        """Ensures correct output from check_target_array.

        In this case the input array is 1-D and multiclass, as expected.
        """

        dl_utils.check_target_array(
            target_array=WIND_CLASSES_1D, num_dimensions=1,
            num_classes=WIND_CLASS_MATRIX.shape[1])

    def test_check_target_array_1d_multiclass_bad_class_num(self):
        """Ensures correct output from check_target_array.

        In this case, the input array contains 6 classes but only 2 classes are
        expected.
        """

        with self.assertRaises(ValueError):
            dl_utils.check_target_array(
                target_array=WIND_CLASSES_1D, num_dimensions=1, num_classes=2)

    def test_check_target_array_2d_binary_good(self):
        """Ensures correct output from check_target_array.

        In this case the input array is 2-D and contains 2 classes, as expected.
        """

        dl_utils.check_target_array(
            target_array=TORNADO_CLASS_MATRIX, num_dimensions=2, num_classes=2)

    def test_check_target_array_2d_bad_dim(self):
        """Ensures correct output from check_target_array.

        In this case, the input array is 2-D but a 1-D array is expected.
        """

        with self.assertRaises(TypeError):
            dl_utils.check_target_array(
                target_array=TORNADO_CLASS_MATRIX, num_dimensions=1,
                num_classes=2)

    def test_check_target_array_2d_binary_bad_class_num(self):
        """Ensures correct output from check_target_array.

        In this case, the input array contains 2 classes but 6 classes are
        expected.
        """

        with self.assertRaises(TypeError):
            dl_utils.check_target_array(
                target_array=TORNADO_CLASS_MATRIX, num_dimensions=2,
                num_classes=WIND_CLASS_MATRIX.shape[1])

    def test_check_target_array_2d_multiclass_good(self):
        """Ensures correct output from check_target_array.

        In this case the input array is 2-D and multiclass, as expected.
        """

        dl_utils.check_target_array(
            target_array=WIND_CLASS_MATRIX, num_dimensions=2,
            num_classes=WIND_CLASS_MATRIX.shape[1])

    def test_check_target_array_2d_multiclass_bad_class_num(self):
        """Ensures correct output from check_target_array.

        In this case, the input array contains 6 classes but 2 classes are
        expected.
        """

        with self.assertRaises(TypeError):
            dl_utils.check_target_array(
                target_array=WIND_CLASS_MATRIX, num_dimensions=2,
                num_classes=2)

    def test_stack_radar_fields(self):
        """Ensures correct output from stack_radar_fields."""

        this_matrix = dl_utils.stack_radar_fields(TUPLE_OF_3D_RADAR_MATRICES)
        self.assertTrue(numpy.allclose(
            this_matrix, RADAR_IMAGE_MATRIX_4D, atol=TOLERANCE, equal_nan=True))

    def test_stack_radar_heights(self):
        """Ensures correct output from stack_radar_heights."""

        this_matrix = dl_utils.stack_radar_heights(TUPLE_OF_4D_RADAR_MATRICES)
        self.assertTrue(numpy.allclose(
            this_matrix, RADAR_IMAGE_MATRIX_5D, atol=TOLERANCE, equal_nan=True))

    def test_normalize_radar_images_4d_z(self):
        """Ensures correct output from normalize_radar_images.

        In this case, the input matrix is 4-D and normalization type is z-score.
        """

        this_radar_matrix = dl_utils.normalize_radar_images(
            radar_image_matrix=copy.deepcopy(RADAR_MATRIX_4D_UNNORMALIZED),
            field_names=RADAR_FIELD_NAMES,
            normalization_type_string=dl_utils.Z_NORMALIZATION_TYPE_STRING,
            normalization_param_file_name=None, test_mode=True,
            normalization_table=RADAR_NORMALIZATION_TABLE)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_4D_Z_SCORES,
            atol=TOLERANCE, equal_nan=True))

    def test_normalize_radar_images_4d_minmax(self):
        """Ensures correct output from normalize_radar_images.

        In this case, the input matrix is 4-D and normalization type is minmax.
        """

        this_radar_matrix = dl_utils.normalize_radar_images(
            radar_image_matrix=copy.deepcopy(RADAR_MATRIX_4D_UNNORMALIZED),
            field_names=RADAR_FIELD_NAMES,
            normalization_type_string=dl_utils.MINMAX_NORMALIZATION_TYPE_STRING,
            normalization_param_file_name=None, test_mode=True,
            min_normalized_value=MIN_NORMALIZED_VALUE,
            max_normalized_value=MAX_NORMALIZED_VALUE,
            normalization_table=RADAR_NORMALIZATION_TABLE)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_4D_MINMAX,
            atol=TOLERANCE, equal_nan=True))

    def test_normalize_radar_images_5d_z(self):
        """Ensures correct output from normalize_radar_images.

        In this case, the input matrix is 5-D and normalization type is z-score.
        """

        this_radar_matrix = dl_utils.normalize_radar_images(
            radar_image_matrix=copy.deepcopy(RADAR_MATRIX_5D_UNNORMALIZED),
            field_names=RADAR_FIELD_NAMES,
            normalization_type_string=dl_utils.Z_NORMALIZATION_TYPE_STRING,
            normalization_param_file_name=None, test_mode=True,
            normalization_table=RADAR_NORMALIZATION_TABLE)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_5D_Z_SCORES,
            atol=TOLERANCE, equal_nan=True))

    def test_normalize_radar_images_5d_minmax(self):
        """Ensures correct output from normalize_radar_images.

        In this case, the input matrix is 5-D and normalization type is minmax.
        """

        this_radar_matrix = dl_utils.normalize_radar_images(
            radar_image_matrix=copy.deepcopy(RADAR_MATRIX_5D_UNNORMALIZED),
            field_names=RADAR_FIELD_NAMES,
            normalization_type_string=dl_utils.MINMAX_NORMALIZATION_TYPE_STRING,
            normalization_param_file_name=None, test_mode=True,
            min_normalized_value=MIN_NORMALIZED_VALUE,
            max_normalized_value=MAX_NORMALIZED_VALUE,
            normalization_table=RADAR_NORMALIZATION_TABLE)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_5D_MINMAX,
            atol=TOLERANCE, equal_nan=True))

    def test_denormalize_radar_images_4d_z(self):
        """Ensures correct output from denormalize_radar_images.

        In this case, the input matrix is 4-D and normalization type is z-score.
        """

        this_radar_matrix = dl_utils.denormalize_radar_images(
            radar_image_matrix=copy.deepcopy(RADAR_MATRIX_4D_Z_SCORES),
            field_names=RADAR_FIELD_NAMES,
            normalization_type_string=dl_utils.Z_NORMALIZATION_TYPE_STRING,
            normalization_param_file_name=None, test_mode=True,
            normalization_table=RADAR_NORMALIZATION_TABLE)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_4D_UNNORMALIZED,
            atol=TOLERANCE, equal_nan=True))

    def test_denormalize_radar_images_4d_minmax(self):
        """Ensures correct output from denormalize_radar_images.

        In this case, the input matrix is 4-D and normalization type is minmax.
        """

        this_radar_matrix = dl_utils.denormalize_radar_images(
            radar_image_matrix=copy.deepcopy(RADAR_MATRIX_4D_MINMAX),
            field_names=RADAR_FIELD_NAMES,
            normalization_type_string=dl_utils.MINMAX_NORMALIZATION_TYPE_STRING,
            normalization_param_file_name=None, test_mode=True,
            min_normalized_value=MIN_NORMALIZED_VALUE,
            max_normalized_value=MAX_NORMALIZED_VALUE,
            normalization_table=RADAR_NORMALIZATION_TABLE)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_4D_UNNORMALIZED,
            atol=TOLERANCE, equal_nan=True))

    def test_denormalize_radar_images_5d_z(self):
        """Ensures correct output from denormalize_radar_images.

        In this case, the input matrix is 5-D and normalization type is z-score.
        """

        this_radar_matrix = dl_utils.denormalize_radar_images(
            radar_image_matrix=copy.deepcopy(RADAR_MATRIX_5D_Z_SCORES),
            field_names=RADAR_FIELD_NAMES,
            normalization_type_string=dl_utils.Z_NORMALIZATION_TYPE_STRING,
            normalization_param_file_name=None, test_mode=True,
            normalization_table=RADAR_NORMALIZATION_TABLE)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_5D_UNNORMALIZED,
            atol=TOLERANCE, equal_nan=True))

    def test_denormalize_radar_images_5d_minmax(self):
        """Ensures correct output from denormalize_radar_images.

        In this case, the input matrix is 5-D and normalization type is minmax.
        """

        this_radar_matrix = dl_utils.denormalize_radar_images(
            radar_image_matrix=copy.deepcopy(RADAR_MATRIX_5D_MINMAX),
            field_names=RADAR_FIELD_NAMES,
            normalization_type_string=dl_utils.MINMAX_NORMALIZATION_TYPE_STRING,
            normalization_param_file_name=None, test_mode=True,
            min_normalized_value=MIN_NORMALIZED_VALUE,
            max_normalized_value=MAX_NORMALIZED_VALUE,
            normalization_table=RADAR_NORMALIZATION_TABLE)

        self.assertTrue(numpy.allclose(
            this_radar_matrix, RADAR_MATRIX_5D_UNNORMALIZED,
            atol=TOLERANCE, equal_nan=True))

    def test_mask_low_reflectivity_pixels(self):
        """Ensures correct output from mask_low_reflectivity_pixels."""

        this_matrix = dl_utils.mask_low_reflectivity_pixels(
            radar_image_matrix_3d=copy.deepcopy(RADAR_MATRIX_3D_UNMASKED),
            field_names=FIELD_NAMES_FOR_MASKING,
            reflectivity_threshold_dbz=MASKING_THRESHOLD_DBZ)

        self.assertTrue(numpy.allclose(
            this_matrix, RADAR_MATRIX_3D_MASKED, atol=TOLERANCE))

    def test_normalize_soundings_z(self):
        """Ensures correct output from normalize_soundings; z-score method."""

        this_sounding_matrix = dl_utils.normalize_soundings(
            sounding_matrix=copy.deepcopy(SOUNDING_MATRIX_UNNORMALIZED),
            field_names=SOUNDING_FIELD_NAMES,
            normalization_type_string=dl_utils.Z_NORMALIZATION_TYPE_STRING,
            normalization_param_file_name=None, test_mode=True,
            normalization_table=SOUNDING_NORMALIZATION_TABLE)

        self.assertTrue(numpy.allclose(
            this_sounding_matrix, SOUNDING_MATRIX_Z_SCORES, atol=TOLERANCE))

    def test_normalize_soundings_minmax(self):
        """Ensures correct output from normalize_soundings; z-score method."""

        this_sounding_matrix = dl_utils.normalize_soundings(
            sounding_matrix=copy.deepcopy(SOUNDING_MATRIX_UNNORMALIZED),
            field_names=SOUNDING_FIELD_NAMES,
            normalization_type_string=dl_utils.MINMAX_NORMALIZATION_TYPE_STRING,
            normalization_param_file_name=None, test_mode=True,
            min_normalized_value=MIN_NORMALIZED_VALUE,
            max_normalized_value=MAX_NORMALIZED_VALUE,
            normalization_table=SOUNDING_NORMALIZATION_TABLE)

        self.assertTrue(numpy.allclose(
            this_sounding_matrix, SOUNDING_MATRIX_MINMAX, atol=TOLERANCE))

    def test_denormalize_soundings_z(self):
        """Ensures correct output from denormalize_soundings; z-score method."""

        this_sounding_matrix = dl_utils.denormalize_soundings(
            sounding_matrix=copy.deepcopy(SOUNDING_MATRIX_Z_SCORES),
            field_names=SOUNDING_FIELD_NAMES,
            normalization_type_string=dl_utils.Z_NORMALIZATION_TYPE_STRING,
            normalization_param_file_name=None, test_mode=True,
            normalization_table=SOUNDING_NORMALIZATION_TABLE)

        self.assertTrue(numpy.allclose(
            this_sounding_matrix, SOUNDING_MATRIX_UNNORMALIZED, atol=TOLERANCE))

    def test_denormalize_soundings_minmax(self):
        """Ensures correct output from denormalize_soundings; minmax method."""

        this_sounding_matrix = dl_utils.denormalize_soundings(
            sounding_matrix=copy.deepcopy(SOUNDING_MATRIX_MINMAX),
            field_names=SOUNDING_FIELD_NAMES,
            normalization_type_string=dl_utils.MINMAX_NORMALIZATION_TYPE_STRING,
            normalization_param_file_name=None, test_mode=True,
            min_normalized_value=MIN_NORMALIZED_VALUE,
            max_normalized_value=MAX_NORMALIZED_VALUE,
            normalization_table=SOUNDING_NORMALIZATION_TABLE)

        self.assertTrue(numpy.allclose(
            this_sounding_matrix, SOUNDING_MATRIX_UNNORMALIZED, atol=TOLERANCE))

    def test_soundings_to_metpy_dictionaries_no_pressure(self):
        """Ensures correct output from soundings_to_metpy_dictionaries.

        In this case the soundings do not contain pressure.
        """

        these_metpy_dictionaries = dl_utils.soundings_to_metpy_dictionaries(
            sounding_matrix=SOUNDING_MATRIX_NO_PRESSURE,
            field_names=SOUNDING_FIELD_NAMES_NO_PRESSURE,
            height_levels_m_agl=HEIGHT_LEVELS_M_AGL,
            storm_elevations_m_asl=STORM_ELEVATIONS_M_ASL)

        self.assertTrue(_compare_lists_of_metpy_dicts(
            these_metpy_dictionaries, LIST_OF_METPY_DICTIONARIES))

    def test_soundings_to_metpy_dictionaries_with_pressure(self):
        """Ensures correct output from soundings_to_metpy_dictionaries.

        In this case the soundings contain pressure.
        """

        these_metpy_dictionaries = dl_utils.soundings_to_metpy_dictionaries(
            sounding_matrix=SOUNDING_MATRIX_UNNORMALIZED,
            field_names=SOUNDING_FIELD_NAMES)

        self.assertTrue(_compare_lists_of_metpy_dicts(
            these_metpy_dictionaries, LIST_OF_METPY_DICTIONARIES))

    def test_sample_by_class_tornado(self):
        """Ensures correct output from sample_by_class.

        In this case, target variable = tornado occurrence.
        """

        these_indices = dl_utils.sample_by_class(
            sampling_fraction_by_class_dict=
            BALANCED_FRACTION_BY_TORNADO_CLASS_DICT,
            target_name=TORNADO_TARGET_NAME,
            target_values=TORNADO_LABELS_TO_SAMPLE,
            num_examples_total=NUM_EXAMPLES_TOTAL, test_mode=True)

        self.assertTrue(numpy.array_equal(
            these_indices, TORNADO_INDICES_TO_KEEP))

    def test_sample_by_class_wind(self):
        """Ensures correct output from sample_by_class.

        In this case, target variable = wind-speed category.
        """

        these_indices = dl_utils.sample_by_class(
            sampling_fraction_by_class_dict=
            SAMPLING_FRACTION_BY_WIND_4CLASS_DICT,
            target_name=WIND_TARGET_NAME_4CLASSES,
            target_values=WIND_LABELS_TO_SAMPLE,
            num_examples_total=NUM_EXAMPLES_TOTAL, test_mode=True)

        self.assertTrue(numpy.array_equal(these_indices, WIND_INDICES_TO_KEEP))


if __name__ == '__main__':
    unittest.main()
