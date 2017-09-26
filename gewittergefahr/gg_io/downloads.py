"""Methods for downloading files."""

import os.path
import warnings
import ftplib
import urllib2
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

NUM_BYTES_PER_DOWNLOAD_CHUNK = 16384
URL_NOT_FOUND_ERROR_CODE = 404
FTP_NOT_FOUND_ERROR_CODE = 550


def download_file_from_url(file_url, local_file_name,
                           raise_error_if_fails=True):
    """Downloads file from URL.

    :param file_url: [string] File will be downloaded from here.
    :param local_file_name: [string] File will be saved to this path on local
        machine.
    :param raise_error_if_fails: [Boolean] If True and download fails, this
        method will raise an error.
    :return: local_file_name: Path to file on local machine.  If download failed
        but raise_error_if_fails = False, this will be None.
    :raises: ValueError: if download failed and raise_error_if_fails = True; if
        download failed for any reason other than URL not found.
    """

    error_checking.assert_is_string(file_url)
    error_checking.assert_is_string(local_file_name)
    error_checking.assert_is_boolean(raise_error_if_fails)

    try:
        response_object = urllib2.urlopen(file_url)
    except urllib2.HTTPError as this_error:
        if raise_error_if_fails or this_error.code != URL_NOT_FOUND_ERROR_CODE:
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


def download_file_from_ftp(server_name=None, user_name=None, password=None,
                           ftp_file_name=None, local_file_name=None,
                           raise_error_if_fails=True):
    """Downloads file from FTP server.

    :param server_name: [string] Name of FTP server.
    :param user_name: [string] Username on FTP server.  If you want to login
        anonymously, leave this as None.
    :param password: [string] Password on FTP server.  If you want to login
        anonymously, leave this as None.
    :param ftp_file_name: [string] File path on FTP server.
    :param local_file_name: [string] File will be saved to this path on local
        machine.
    :param raise_error_if_fails: [Boolean] If True and download fails, this
        method will raise an error.
    :return: local_file_name: Path to file on local machine.  If download failed
        but raise_error_if_fails = False, this will be None.
    :raises: ValueError: if download failed and raise_error_if_fails = True; if
        download failed for any reason other than file not found.
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
