"""Trains small 2-D convolutional neural net (CNN).

In other words, the CNN performs 2-D convolution.
"""

import os.path
import numpy
from gewittergefahr.deep_learning import cnn
from gewittergefahr.deep_learning import deep_learning_utils as dl_utils
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import time_conversion

NUM_CLASSES = 2
RADAR_FIELD_NAMES = [
    radar_utils.REFL_NAME, radar_utils.VORTICITY_NAME,
    radar_utils.DIVERGENCE_NAME]
RADAR_HEIGHTS_M_ASL = numpy.array([5000], dtype=int)
TARGET_NAME = 'tornado_lead-time=0000-7200sec_distance=00000-30000m'

NUM_EPOCHS = 100
NUM_EXAMPLES_PER_BATCH = 512
NUM_EXAMPLES_PER_TIME = 100
NUM_TRAINING_BATCHES_PER_EPOCH = 32
CLASS_FRACTIONS_TO_SAMPLE = numpy.array([0.9, 0.1])

FIRST_TRAIN_TIME_STRING = '2011-01-25-180000'
LAST_TRAIN_TIME_STRING = '2016-12-31-235500'
TIME_FORMAT = '%Y-%m-%d-%H%M%S'
TOP_STORM_IMAGE_DIR_NAME = (
    '/condo/swatcommon/common/gridrad_final/myrorss_format/tracks/reanalyzed/'
    'storm_images')

MODEL_FILE_NAME = (
    '/condo/swatcommon/common/gridrad_final/small_2d_cnn_2011-2016.h5')

if __name__ == '__main__':
    model_object = cnn.get_2d_swirlnet_architecture(
        num_classes=NUM_CLASSES, num_input_channels=len(RADAR_FIELD_NAMES))

    model_directory_name, _ = os.path.split(MODEL_FILE_NAME)
    metadata_file_name = '{0:s}/model_metadata.p'.format(model_directory_name)
    print 'Writing metadata to: "{0:s}"...'.format(metadata_file_name)

    first_train_time_unix_sec = time_conversion.string_to_unix_sec(
        FIRST_TRAIN_TIME_STRING, TIME_FORMAT)
    last_train_time_unix_sec = time_conversion.string_to_unix_sec(
        LAST_TRAIN_TIME_STRING, TIME_FORMAT)

    cnn.write_model_metadata(
        num_epochs=NUM_EPOCHS, num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
        num_examples_per_time=NUM_EXAMPLES_PER_TIME,
        num_training_batches_per_epoch=NUM_TRAINING_BATCHES_PER_EPOCH,
        first_train_time_unix_sec=first_train_time_unix_sec,
        last_train_time_unix_sec=last_train_time_unix_sec,
        num_validation_batches_per_epoch=None, first_validn_time_unix_sec=None,
        last_validn_time_unix_sec=None,
        radar_source=radar_utils.GRIDRAD_SOURCE_ID,
        radar_field_names=RADAR_FIELD_NAMES,
        radar_heights_m_asl=RADAR_HEIGHTS_M_ASL,
        reflectivity_heights_m_asl=None,
        target_name=TARGET_NAME, normalize_by_batch=False,
        normalization_dict=dl_utils.DEFAULT_NORMALIZATION_DICT,
        percentile_offset_for_normalization=None,
        class_fractions=CLASS_FRACTIONS_TO_SAMPLE,
        pickle_file_name=metadata_file_name)

    cnn.train_2d_cnn(
        model_object=model_object, output_file_name=MODEL_FILE_NAME,
        num_epochs=NUM_EPOCHS,
        num_training_batches_per_epoch=NUM_TRAINING_BATCHES_PER_EPOCH,
        top_input_dir_name=TOP_STORM_IMAGE_DIR_NAME,
        radar_source=radar_utils.GRIDRAD_SOURCE_ID,
        radar_field_names=RADAR_FIELD_NAMES,
        num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
        num_examples_per_time=NUM_EXAMPLES_PER_TIME,
        first_train_time_unix_sec=first_train_time_unix_sec,
        last_train_time_unix_sec=last_train_time_unix_sec,
        target_name=TARGET_NAME, radar_heights_m_asl=RADAR_HEIGHTS_M_ASL,
        normalize_by_batch=False,
        normalization_dict=dl_utils.DEFAULT_NORMALIZATION_DICT,
        class_fractions_to_sample=CLASS_FRACTIONS_TO_SAMPLE)
