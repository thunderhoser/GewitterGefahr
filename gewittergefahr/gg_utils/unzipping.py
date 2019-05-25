"""Methods for unzipping files."""

import os
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking


def unzip_tar(tar_file_name, target_directory_name=None,
              file_and_dir_names_to_unzip=None):
    """Unzips tar file.

    :param tar_file_name: Path to input file.
    :param target_directory_name: Path to output directory.
    :param file_and_dir_names_to_unzip: List of files and directories to extract
        from the tar file.  Each list element should be a relative path inside
        the tar file.  After unzipping, the same relative path will exist inside
        `target_directory_name`.
    :raises: ValueError: if the Unix command fails.
    """

    error_checking.assert_is_string(tar_file_name)
    error_checking.assert_is_string_list(file_and_dir_names_to_unzip)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=target_directory_name)

    unix_command_string = 'tar -C "{0:s}" -xvf "{1:s}"'.format(
        target_directory_name, tar_file_name)

    for this_relative_path in file_and_dir_names_to_unzip:
        unix_command_string += ' "' + this_relative_path + '"'

    exit_code = os.system(unix_command_string)
    if exit_code != 0:
        raise ValueError('\nUnix command failed (log messages shown above '
                         'should explain why).')


def unzip_gzip(gzip_file_name, extracted_file_name):
    """Unzips gzip archive.

    Keep in mind that all gzip archive contain only one file.

    :param gzip_file_name: Path to gzip archive.
    :param extracted_file_name: The one file in the gzip archive will be saved
        here.
    :raises: ValueError: if the Unix command fails.
    """

    error_checking.assert_is_string(gzip_file_name)
    file_system_utils.mkdir_recursive_if_necessary(
        file_name=extracted_file_name)

    unix_command_string = 'gunzip -v -c "{0:s}" > "{1:s}"'.format(
        gzip_file_name, extracted_file_name)

    exit_code = os.system(unix_command_string)

    if exit_code != 0:
        raise ValueError('\nUnix command failed (log messages shown above '
                         'should explain why).')


def gzip_file(input_file_name, output_file_name=None, delete_input_file=True):
    """Creates gzip archive with one file.

    :param input_file_name: Path to input file (will be gzipped).
    :param output_file_name: Path to output file (extension must be ".gz").  If
        `output_file_name is None`, will simply append ".gz" to name of input
        file.
    :param delete_input_file: Boolean flag.  If True, will delete input file
        after gzipping.
    :raises: ValueError: if `output_file_name` does not end with ".gz".
    :raises: ValueError: if the Unix command fails.
    """

    error_checking.assert_file_exists(input_file_name)
    error_checking.assert_is_boolean(delete_input_file)
    if output_file_name is None:
        output_file_name = '{0:s}.gz'.format(input_file_name)

    if not output_file_name.endswith('.gz'):
        error_string = (
            'Output file ("{0:s}") should have extension ".gz".'
        ).format(output_file_name)

        raise ValueError(error_string)

    unix_command_string = 'gzip -v -c "{0:s}" > "{1:s}"'.format(
        input_file_name, output_file_name)
    exit_code = os.system(unix_command_string)

    if exit_code != 0:
        raise ValueError('\nUnix command failed (log messages shown above '
                         'should explain why).')

    if delete_input_file:
        os.remove(input_file_name)
