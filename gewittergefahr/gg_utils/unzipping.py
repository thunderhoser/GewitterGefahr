"""Methods for unzipping files."""

import subprocess
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
    """

    error_checking.assert_is_string(tar_file_name)
    error_checking.assert_is_string_list(file_and_dir_names_to_unzip)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=target_directory_name)

    unix_command_string = 'tar -C "{0:s}" -xvf "{1:s}"'.format(
        target_directory_name, tar_file_name)
    for this_relative_path in file_and_dir_names_to_unzip:
        unix_command_string += ' "' + this_relative_path + '"'

    subprocess.call(unix_command_string)


def unzip_gzip(gzip_file_name, extracted_file_name):
    """Unzips gzip archive.

    Keep in mind that all gzip archive contain only one file.

    :param gzip_file_name: Path to gzip archive.
    :param extracted_file_name: The one file in the gzip archive will be saved
        here.
    """

    error_checking.assert_is_string(gzip_file_name)
    file_system_utils.mkdir_recursive_if_necessary(
        file_name=extracted_file_name)

    unix_command_string = 'gunzip -v -c "{0:s}" > "{1:s}"'.format(
        gzip_file_name, extracted_file_name)
    subprocess.call(unix_command_string)
