"""Methods for downloading files."""

import os
import warnings
import subprocess
import ftplib
import urllib2
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

NUM_BYTES_PER_DOWNLOAD_CHUNK = 16384
URL_NOT_FOUND_ERROR_CODE = 404
SERVICE_TEMP_UNAVAILABLE_ERROR_CODE = 503
ACCEPTABLE_HTTP_ERROR_CODES = [
    URL_NOT_FOUND_ERROR_CODE, SERVICE_TEMP_UNAVAILABLE_ERROR_CODE]

FTP_NOT_FOUND_ERROR_CODE = 550
SSH_ARG_STRING = (
    'ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=5')


def download_file_via_passwordless_ssh(host_name=None, user_name=None,
                                       remote_file_name=None,
                                       local_file_name=None,
                                       raise_error_if_fails=True):
    """Downloads file via passwordless SSH.

    For this to work, the remote machine (from which you are downloading) must
    have the RSA key of the local machine.  See the following page for
    instructions on sharing RSA keys: http://www.linuxproblem.org/art_9.html

    :param host_name: Name of remote machine (example: "thunderhoser.ou.edu").
    :param user_name: User name on remote machine (example: "thunderhoser").
    :param remote_file_name: File path on remote machine (where the file will be
        downloaded from).
    :param local_file_name: File path on local machine (where the file will be
        stored).
    :param raise_error_if_fails: Boolean flag.  If raise_error_if_fails = True
        and download fails, will raise an error.
    :return: local_file_name: If raise_error_if_fails = False and download
        failed, this will be None.  Otherwise, this will be the same as input.
    :raises: ValueError: if download failed and raise_error_if_fails = True.
    """

    # TODO(thunderhoser): Handle exceptions more intelligently.  Currently, if
    # the download fails, this method does not know why it failed.  If the
    # download failed because the file does not exist, this is less severe than
    # if it failed because we can't login to the remote machine.

    error_checking.assert_is_string(host_name)
    error_checking.assert_is_string(user_name)
    error_checking.assert_is_string(remote_file_name)
    error_checking.assert_is_string(local_file_name)
    error_checking.assert_is_boolean(raise_error_if_fails)

    file_system_utils.mkdir_recursive_if_necessary(file_name=local_file_name)

    unix_command_string = (
        'LD_LIBRARY_PATH= rsync -rv -e "{0:s}"'
        ' {1:s}@{2:s}:"{3:s}" "{4:s}"').format(
            SSH_ARG_STRING, user_name, host_name, remote_file_name,
            local_file_name)

    devnull_handle = open(os.devnull, 'w')
    subprocess.call(unix_command_string, shell=True, stdout=devnull_handle,
                    stderr=devnull_handle)

    if not os.path.isfile(local_file_name):
        info_string = (
            'Download failed.  Local file expected at: ' + local_file_name)
        if raise_error_if_fails:
            raise ValueError(info_string)
        else:
            warnings.warn(info_string)
            local_file_name = None

    return local_file_name


def download_file_via_http(file_url, local_file_name,
                           raise_error_if_fails=True):
    """Downloads file via HTTP.

    :param file_url: URL (where the file is located online).  Example:
        "https://nomads.ncdc.noaa.gov/data/narr/201212/20121212/
        narr-a_221_20121212_1200_000.grb".
    :param local_file_name: File path on local machine (where the file will be
        stored).
    :param raise_error_if_fails: Boolean flag.  If raise_error_if_fails = True
        and download fails, will raise an error.
    :return: local_file_name: If raise_error_if_fails = False and download
        failed, this will be None.  Otherwise, this will be the same as input.
    :raises: ValueError: if download failed and raise_error_if_fails = True.
    :raises: urllib2.HTTPError: if download failed for any reason other than URL
        not found.
    """

    error_checking.assert_is_string(file_url)
    error_checking.assert_is_string(local_file_name)
    error_checking.assert_is_boolean(raise_error_if_fails)

    try:
        response_object = urllib2.urlopen(file_url)
    except urllib2.HTTPError as this_error:
        if (raise_error_if_fails or
                this_error.code not in ACCEPTABLE_HTTP_ERROR_CODES):
            raise

        warnings.warn('Cannot find URL.  Expected at: ' + file_url)
        return None

    file_system_utils.mkdir_recursive_if_necessary(file_name=local_file_name)
    with open(local_file_name, 'wb') as local_file_handle:
        while True:
            this_chunk = response_object.read(NUM_BYTES_PER_DOWNLOAD_CHUNK)
            if not this_chunk:
                break
            local_file_handle.write(this_chunk)

    if not os.path.isfile(local_file_name):
        info_string = (
            'Download failed.  Local file expected at: ' + local_file_name)
        if raise_error_if_fails:
            raise ValueError(info_string)
        else:
            warnings.warn(info_string)
            local_file_name = None

    return local_file_name


def download_file_via_ftp(server_name=None, user_name=None, password=None,
                          ftp_file_name=None, local_file_name=None,
                          raise_error_if_fails=True):
    """Downloads file from FTP server.

    :param server_name: Name of FTP server (for example,
        "madis-data.ncep.noaa.gov").
    :param user_name: User name on FTP server (example: "thunderhoser").  If you
        want to login anonymously, leave this as None.
    :param password: Password on FTP server.  If you want to login anonymously,
        leave this as None.
    :param ftp_file_name: File path on FTP server.
    :param local_file_name: File path on local machine (where the file will be
        stored).
    :param raise_error_if_fails: Boolean flag.  If raise_error_if_fails = True
        and download fails, will raise an error.
    :return: local_file_name: If raise_error_if_fails = False and download
        failed, this will be None.  Otherwise, this will be the same as input.
    :raises: ValueError: if download failed and raise_error_if_fails = True.
    :raises: ftplib.error_perm: if download failed for any reason other than
        file not found.
    """

    error_checking.assert_is_string(server_name)
    error_checking.assert_is_string(ftp_file_name)
    error_checking.assert_is_string(local_file_name)
    error_checking.assert_is_boolean(raise_error_if_fails)

    if user_name is None or password is None:
        ftp_object = ftplib.FTP(server_name)
        ftp_object.login()
    else:
        error_checking.assert_is_string(user_name)
        error_checking.assert_is_string(password)
        ftp_object = ftplib.FTP(server_name, user_name, password)

    file_system_utils.mkdir_recursive_if_necessary(file_name=local_file_name)
    local_file_handle = open(local_file_name, 'wb')

    try:
        ftp_object.retrbinary('RETR ' + ftp_file_name, local_file_handle.write,
                              blocksize=NUM_BYTES_PER_DOWNLOAD_CHUNK)
    except ftplib.error_perm as this_error:
        if (raise_error_if_fails or not this_error.message.startswith(
                str(FTP_NOT_FOUND_ERROR_CODE))):
            raise

        warnings.warn(
            'Cannot find file on FTP server.  Expected at: ' + ftp_file_name)
        return None

    if not os.path.isfile(local_file_name):
        info_string = (
            'Download failed.  Local file expected at: ' + local_file_name)
        if raise_error_if_fails:
            raise ValueError(info_string)
        else:
            warnings.warn(info_string)
            local_file_name = None

    return local_file_name
