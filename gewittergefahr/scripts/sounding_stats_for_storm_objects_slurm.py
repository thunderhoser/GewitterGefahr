"""Writes Slurm file to run sounding_stats_for_storm_objects.py on sprcmptr."""

import argparse
import numpy
from gewittergefahr.gg_io import slurm_io
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.scripts import \
    sounding_stats_for_storm_objects as sounding_stats

PYTHON_EXE_NAME = '/home/ralager/anaconda2/bin/python2.7'
PYTHON_SCRIPT_NAME = (
    '/condo/swatwork/ralager/gewittergefahr_master/gewittergefahr/scripts/'
    'sounding_stats_for_storm_objects.py')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER = slurm_io.add_input_arguments(
    argument_parser_object=INPUT_ARG_PARSER, use_array=True, use_spc_dates=True)
INPUT_ARG_PARSER = sounding_stats.add_input_arguments(INPUT_ARG_PARSER)


def _write_slurm_file(
        first_spc_date_string, last_spc_date_string, max_num_simultaneous_tasks,
        email_address, partition_name, slurm_file_name, lead_times_seconds,
        tracking_scale_metres2, top_ruc_directory_name, top_tracking_dir_name,
        output_dir_name):
    """Writes Slurm file to run sounding_stats_for_storm_objects.py on sprcmptr.

    :param first_spc_date_string: SPC (Storm Prediction Center) date in format
        "yyyymmdd".  Sounding stats will be computed independently for each date
        from `first_spc_date_string`...`last_spc_date_string`.  In other words,
        each date will be one task.
    :param last_spc_date_string: See above.
    :param max_num_simultaneous_tasks: Max number of tasks (SPC dates) running
        at once.
    :param email_address: Slurm notifications will be sent to this e-mail
        address.
    :param partition_name: Job will be run on this partition of the
        supercomputer.
    :param slurm_file_name: Path to output file.
    :param lead_times_seconds: See documentation for
        `sounding_stats_for_storm_objects.py`.
    :param tracking_scale_metres2: See doc for
        `sounding_stats_for_storm_objects.py`.
    :param top_ruc_directory_name: See doc for
        `sounding_stats_for_storm_objects.py`.
    :param top_tracking_dir_name: See doc for
        `sounding_stats_for_storm_objects.py`.
    :param output_dir_name: See doc for `sounding_stats_for_storm_objects.py`.
    """

    num_spc_dates = len(time_conversion.get_spc_dates_in_range(
        first_spc_date_string, last_spc_date_string))

    slurm_file_handle = slurm_io.write_slurm_file_header(
        slurm_file_name=slurm_file_name, email_address=email_address,
        partition_name=partition_name, use_array=True,
        num_array_tasks=num_spc_dates,
        max_num_simultaneous_tasks=max_num_simultaneous_tasks)

    slurm_io.write_spc_date_list_to_slurm_file(
        slurm_file_handle=slurm_file_handle,
        first_spc_date_string=first_spc_date_string,
        last_spc_date_string=last_spc_date_string)

    # The following statement calls sounding_stats_for_storm_objects.py for the
    # given task (SPC date).
    lead_times_seconds = numpy.array(lead_times_seconds, dtype=int)

    slurm_file_handle.write(
        '"{0:s}" -u "{1:s}" --{2:s}='.format(
            PYTHON_EXE_NAME, PYTHON_SCRIPT_NAME,
            sounding_stats.SPC_DATE_INPUT_ARG))
    slurm_file_handle.write('"${this_spc_date_string}"')

    slurm_file_handle.write(' --{0:s}'.format(
        sounding_stats.LEAD_TIMES_INPUT_ARG))
    for this_lead_time_sec in lead_times_seconds:
        slurm_file_handle.write(' {0:d}'.format(this_lead_time_sec))

    slurm_file_handle.write(
        ' --{0:s}={1:d} --{2:s}="{3:s}" --{4:s}="{5:s}" --{6:s}="{7:s}"'.format(
            sounding_stats.TRACKING_SCALE_INPUT_ARG, tracking_scale_metres2,
            sounding_stats.RUC_DIRECTORY_INPUT_ARG, top_ruc_directory_name,
            sounding_stats.TRACKING_DIR_INPUT_ARG, top_tracking_dir_name,
            sounding_stats.OUTPUT_DIR_INPUT_ARG, output_dir_name))
    slurm_file_handle.close()


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _write_slurm_file(
        first_spc_date_string=getattr(
            INPUT_ARG_OBJECT, slurm_io.FIRST_SPC_DATE_INPUT_ARG),
        last_spc_date_string=getattr(
            INPUT_ARG_OBJECT, slurm_io.LAST_SPC_DATE_INPUT_ARG),
        max_num_simultaneous_tasks=getattr(
            INPUT_ARG_OBJECT, slurm_io.MAX_SIMULTANEOUS_TASKS_INPUT_ARG),
        email_address=getattr(
            INPUT_ARG_OBJECT, slurm_io.EMAIL_ADDRESS_INPUT_ARG),
        partition_name=getattr(
            INPUT_ARG_OBJECT, slurm_io.PARTITION_NAME_INPUT_ARG),
        slurm_file_name=getattr(
            INPUT_ARG_OBJECT, slurm_io.SLURM_FILE_INPUT_ARG),
        lead_times_seconds=getattr(
            INPUT_ARG_OBJECT, sounding_stats.LEAD_TIMES_INPUT_ARG),
        tracking_scale_metres2=getattr(
            INPUT_ARG_OBJECT, sounding_stats.TRACKING_SCALE_INPUT_ARG),
        top_ruc_directory_name=getattr(
            INPUT_ARG_OBJECT, sounding_stats.RUC_DIRECTORY_INPUT_ARG),
        top_tracking_dir_name=getattr(
            INPUT_ARG_OBJECT, sounding_stats.TRACKING_DIR_INPUT_ARG),
        output_dir_name=getattr(
            INPUT_ARG_OBJECT, sounding_stats.OUTPUT_DIR_INPUT_ARG))
