"""Writes Slurm file to run best-track algorithm on supercomputer."""

import argparse
from gewittergefahr.gg_io import slurm_io
from gewittergefahr.scripts import run_best_track

PYTHON_EXE_NAME = '/home/ralager/anaconda2/bin/python2.7'
PYTHON_SCRIPT_NAME = (
    '/condo/swatwork/ralager/gewittergefahr_master/gewittergefahr/scripts/'
    'run_best_track.py')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER = slurm_io.add_input_arguments(
    argument_parser_object=INPUT_ARG_PARSER, use_array=False,
    use_spc_dates=True)
INPUT_ARG_PARSER = run_best_track.add_input_arguments(INPUT_ARG_PARSER)


def _write_slurm_file(
        first_spc_date_string, last_spc_date_string, top_segmotion_dir_name,
        top_best_track_dir_name, tracking_scale_metres2, email_address,
        partition_name, slurm_file_name):
    """Writes Slurm file to run best-track algorithm on supercomputer.

    :param first_spc_date_string: See documentation for `run_best_track.py`.
    :param last_spc_date_string: See doc for `run_best_track.py`.
    :param top_segmotion_dir_name: See doc for `run_best_track.py`.
    :param top_best_track_dir_name: See doc for `run_best_track.py`.
    :param tracking_scale_metres2: See doc for `run_best_track.py`.
    :param email_address: Slurm notifications will be sent to this e-mail
        address.
    :param partition_name: Job will be run on this partition of the
        supercomputer.
    :param slurm_file_name: Path to output file.
    """

    slurm_file_handle = slurm_io.write_slurm_file_header(
        slurm_file_name=slurm_file_name, email_address=email_address,
        partition_name=partition_name, use_array=False)

    # The following statement calls run_best_track.py for the given range of SPC
    # dates.
    slurm_file_handle.write(
        ('"{0:s}" -u "{1:s}" --{2:s}="{3:s}" --{4:s}="{5:s}" --{6:s}="{7:s}" '
         '--{8:s}="{9:s}" --{10:s}={11:d}').format(
             PYTHON_EXE_NAME, PYTHON_SCRIPT_NAME,
             run_best_track.FIRST_SPC_DATE_INPUT_ARG, first_spc_date_string,
             run_best_track.LAST_SPC_DATE_INPUT_ARG, last_spc_date_string,
             run_best_track.SEGMOTION_DIR_INPUT_ARG, top_segmotion_dir_name,
             run_best_track.BEST_TRACK_DIR_INPUT_ARG, top_best_track_dir_name,
             run_best_track.TRACKING_SCALE_INPUT_ARG, tracking_scale_metres2))
    slurm_file_handle.close()


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _write_slurm_file(
        first_spc_date_string=getattr(
            INPUT_ARG_OBJECT, run_best_track.FIRST_SPC_DATE_INPUT_ARG),
        last_spc_date_string=getattr(
            INPUT_ARG_OBJECT, run_best_track.LAST_SPC_DATE_INPUT_ARG),
        top_segmotion_dir_name=getattr(
            INPUT_ARG_OBJECT, run_best_track.SEGMOTION_DIR_INPUT_ARG),
        top_best_track_dir_name=getattr(
            INPUT_ARG_OBJECT, run_best_track.BEST_TRACK_DIR_INPUT_ARG),
        tracking_scale_metres2=getattr(
            INPUT_ARG_OBJECT, run_best_track.TRACKING_SCALE_INPUT_ARG),
        email_address=getattr(
            INPUT_ARG_OBJECT, slurm_io.EMAIL_ADDRESS_INPUT_ARG),
        partition_name=getattr(
            INPUT_ARG_OBJECT, slurm_io.PARTITION_NAME_INPUT_ARG),
        slurm_file_name=getattr(
            INPUT_ARG_OBJECT, slurm_io.SLURM_FILE_INPUT_ARG))
