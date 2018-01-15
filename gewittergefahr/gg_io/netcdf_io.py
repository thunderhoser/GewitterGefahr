"""IO methods for NetCDF files."""

import os
import gzip
import shutil
import tempfile
from netCDF4 import Dataset
from gewittergefahr.gg_utils import error_checking

GZIP_FILE_EXTENSION = '.gz'


def open_netcdf(netcdf_file_name, raise_error_if_fails=False):
    """Attempts to open NetCDF file.

    Code for handling gzip files comes from jochen at the following
    StackOverflow page: https://stackoverflow.com/posts/45356133/revisions

    :param netcdf_file_name: Path to input file.
    :param raise_error_if_fails: Boolean flag.  If raise_error_if_fails = True
        and file cannot be opened, this method will throw an error.
    :return: netcdf_dataset: Instance of `NetCDF4.Dataset`, containing all data
        from the file.  If raise_error_if_fails = False and file could not be
        opened, this will be None.
    :raises: IOError: if file could not be opened and raise_error_if_fails =
        True.
    """

    error_checking.assert_file_exists(netcdf_file_name)
    error_checking.assert_is_boolean(raise_error_if_fails)
    gzip_as_input = netcdf_file_name.endswith(GZIP_FILE_EXTENSION)

    if gzip_as_input:
        gzip_file_object = gzip.open(netcdf_file_name, 'rb')
        netcdf_temporary_file_object = tempfile.NamedTemporaryFile(delete=False)
        netcdf_file_name = netcdf_temporary_file_object.name

        success = False
        try:
            shutil.copyfileobj(gzip_file_object, netcdf_temporary_file_object)
            success = True
        except:
            if raise_error_if_fails:
                raise

        gzip_file_object.close()
        netcdf_temporary_file_object.close()
        if not success:
            os.remove(netcdf_file_name)
            return None

    try:
        netcdf_dataset = Dataset(netcdf_file_name)
    except IOError:
        if raise_error_if_fails:
            if gzip_as_input:
                os.remove(netcdf_file_name)
            raise

        netcdf_dataset = None

    if gzip_as_input:
        os.remove(netcdf_file_name)
    return netcdf_dataset
