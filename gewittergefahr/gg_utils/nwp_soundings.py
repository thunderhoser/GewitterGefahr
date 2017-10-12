"""Methods for creating soundings from an NWP model.

--- DEFINITIONS ---

NWP = numerical weather prediction
"""

from gewittergefahr.gg_utils import narr_utils
from gewittergefahr.gg_utils import rap_model_utils
from gewittergefahr.gg_utils import error_checking

RAP_MODEL_NAME = 'rap'
NARR_MODEL_NAME = 'narr'
MODEL_NAMES = [RAP_MODEL_NAME, NARR_MODEL_NAME]


def _check_model_name(model_name):
    """Ensures that model name is valid.

    :param model_name: Name of model (either "rap" or "narr" -- this list will
        be expanded if/when more NWP models are included in GewitterGefahr).
    :raises: ValueError: model_name is not in list of valid names.
    """

    error_checking.assert_is_string(model_name)
    if model_name not in MODEL_NAMES:
        error_string = (
            '\n\n' + str(MODEL_NAMES) +
            '\n\nValid model names (listed above) do not include "' +
            model_name + '".')
        raise ValueError(error_string)


def _get_sounding_columns(model_name):
    """Returns list of sounding variables in NWP model.

    N = number of sounding variables

    :param model_name: Name of model.
    :return: sounding_columns: length-N list with names of sounding variables.
    :return: sounding_columns_orig: length-N list with original names of
        sounding variables (names used in grib/grib2 files).
    """

    _check_model_name(model_name)

    if model_name == RAP_MODEL_NAME:
        pressure_levels_mb = rap_model_utils.PRESSURE_LEVELS_MB
        main_sounding_columns = rap_model_utils.MAIN_SOUNDING_COLUMNS
        main_sounding_columns_orig = rap_model_utils.MAIN_SOUNDING_COLUMNS_ORIG
    elif model_name == NARR_MODEL_NAME:
        pressure_levels_mb = narr_utils.PRESSURE_LEVELS_MB
        main_sounding_columns = narr_utils.MAIN_SOUNDING_COLUMNS
        main_sounding_columns_orig = narr_utils.MAIN_SOUNDING_COLUMNS_ORIG

    num_pressure_levels = len(pressure_levels_mb)
    num_main_sounding_columns = len(main_sounding_columns_orig)
    sounding_columns = []
    sounding_columns_orig = []

    for j in range(num_main_sounding_columns):
        for k in range(num_pressure_levels):
            sounding_columns.append('{0:s}_{1:d}mb'.format(
                main_sounding_columns[j], int(pressure_levels_mb[k])))
            sounding_columns_orig.append('{0:s}:{1:d} mb'.format(
                main_sounding_columns_orig[j], int(pressure_levels_mb[k])))

    return sounding_columns, sounding_columns_orig
