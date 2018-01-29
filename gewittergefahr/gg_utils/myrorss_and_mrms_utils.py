"""Methods for handling radar data from MYRORSS and MRMS.

--- DEFINITIONS ---

MYRORSS = Multi-year Reanalysis of Remotely Sensed Storms (Ortega et al. 2012)

MRMS = Multi-radar Multi-sensor network (Smith et al. 2016)

--- REFERENCES ---

Ortega, K., and Coauthors, 2012: "The multi-year reanalysis of remotely sensed
    storms (MYRORSS) project". Conference on Severe Local Storms, Nashville, TN,
    American Meteorological Society.

Smith, T., and Coauthors, 2016: "Multi-radar Multi-sensor (MRMS) severe weather
    and aviation products: Initial operating capabilities". Bulletin of the
    American Meteorological Society, 97 (9), 1617-1630.
"""

import copy
import numpy
from gewittergefahr.gg_utils import radar_utils
from gewittergefahr.gg_utils import error_checking

DATA_SOURCE_IDS = [radar_utils.MYRORSS_SOURCE_ID, radar_utils.MRMS_SOURCE_ID]


def check_data_source(data_source):
    """Ensures that data source is recognized.

    :param data_source: Data source (string).
    :raises: ValueError: if `data_source not in DATA_SOURCE_IDS`.
    """

    error_checking.assert_is_string(data_source)
    if data_source not in DATA_SOURCE_IDS:
        error_string = (
            '\n\n' + str(DATA_SOURCE_IDS) +
            '\n\nValid data sources (listed above) do not include "' +
            data_source + '".')
        raise ValueError(error_string)


def fields_and_refl_heights_to_dict(
        field_names, data_source, refl_heights_m_asl=None):
    """Converts two arrays (field names and reflectivity heights) to dictionary.

    :param field_names: 1-D list with names of radar fields in GewitterGefahr
        format.
    :param data_source: Data source (string).
    :param refl_heights_m_asl: 1-D numpy array of reflectivity heights (metres
        above sea level).
    :return: field_to_heights_dict_m_asl: Dictionary, where each key is a field
        name and each value is a 1-D numpy array of heights (metres above sea
        level).
    """

    check_data_source(data_source)
    error_checking.assert_is_string_list(field_names)
    error_checking.assert_is_numpy_array(
        numpy.array(field_names), num_dimensions=1)

    field_to_heights_dict_m_asl = {}

    for this_field_name in field_names:
        if this_field_name == radar_utils.REFL_NAME:
            radar_utils.check_heights(
                data_source=data_source, heights_m_asl=refl_heights_m_asl,
                field_name=this_field_name)

            field_to_heights_dict_m_asl.update(
                {this_field_name: refl_heights_m_asl})

        else:
            field_to_heights_dict_m_asl.update({
                this_field_name: radar_utils.get_valid_heights(
                    data_source=data_source, field_name=this_field_name)})

    return field_to_heights_dict_m_asl


def fields_and_refl_heights_to_pairs(
        field_names, data_source, refl_heights_m_asl=None):
    """Converts unique arrays (field names and refl heights) to non-unique ones.

    F = number of fields
    N = number of field-height pairs

    :param field_names: length-F list with names of radar fields in
        GewitterGefahr format.
    :param data_source: Data source (string).
    :param refl_heights_m_asl: 1-D numpy array of reflectivity heights (metres
        above sea level).
    :return: field_name_by_pair: length-N list of field names.
    :return: height_by_pair_m_asl: length-N numpy array of corresponding heights
        (metres above sea level).
    """

    check_data_source(data_source)
    error_checking.assert_is_string_list(field_names)
    error_checking.assert_is_numpy_array(
        numpy.array(field_names), num_dimensions=1)

    field_name_by_pair = []
    height_by_pair_m_asl = numpy.array([])

    for this_field_name in field_names:
        if this_field_name == radar_utils.REFL_NAME:
            radar_utils.check_heights(
                data_source=data_source, heights_m_asl=refl_heights_m_asl,
                field_name=this_field_name)

            these_heights_m_asl = copy.deepcopy(refl_heights_m_asl)

        else:
            these_heights_m_asl = radar_utils.get_valid_heights(
                data_source=data_source, field_name=this_field_name)

        field_name_by_pair += [this_field_name] * len(these_heights_m_asl)
        height_by_pair_m_asl = numpy.concatenate((
            height_by_pair_m_asl, these_heights_m_asl))

    return field_name_by_pair, height_by_pair_m_asl
