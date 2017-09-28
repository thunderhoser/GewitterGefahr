grib_io.py uses wgrib and wgrib2, which are command-line tools for parsing grib and grib2 files, respectively.

--- INSTALLATION INSTRUCTIONS FOR wgrib ---

The following instructions are for Linux only.  See the following page for Mac instructions: https://rda.ucar.edu/datasets/ds083.2/software/wgrib_install_guide.txt

[1] Download the tar file from here: ftp://ftp.cpc.ncep.noaa.gov/wd51we/wgrib/wgrib.tar

    If the above link is dead, search for "wgrib.tar" on this page: http://www.cpc.ncep.noaa.gov/products/wesley/wgrib.html

[2] Unzip the tar file to your chosen directory.  Henceforth, I will assume that this "/home/user/wgrib".  If you unzip the tar file somewhere else, replace "/home/user/wgrib" with the relevant directory.

[3] Open a terminal and type the following:

    cd /home/user/wgrib  # Navigate to directory with tar contents.
    make  # Install wgrib (the makefile should do all the work for you).  This assumes that you have a C-compiler, which should come with every Linux system.

    The wgrib executable should now be at the following location: /home/user/wgrib/wgrib

[4] To ensure that the install worked, open a terminal and type the following:

    /home/user/wgrib/wgrib

    This should print a help menu to the terminal.  If not, see the following page for more detailed Linux instructions: https://rda.ucar.edu/datasets/ds083.2/software/wgrib_install_guide.txt

--- INSTALLATION INSTRUCTIONS FOR wgrib2 ---

The following instructions are for Linux only.  See the following page for Mac instructions: https://rda.ucar.edu/datasets/ds083.2/software/wgrib2_install+.txt

[1] Download the tar file from here: ftp://ftp.cpc.ncep.noaa.gov/wd51we/wgrib2/wgrib2.tgz

[2] Unzip the tar file to your chosen directory.  Henceforth, I will assume that this "/home/user/wgrib2".  If you unzip the tar file somewhere else, replace "/home/user/wgrib2" with the relevant directory.

[3] Open a terminal and type the following:

    cd /home/user/wgrib2  # Navigate to directory with tar contents.
    export CC=gcc  # CC is now a variable pointing to the C-compiler, which should come with every Linux system.
    export FC=gfortran  # FC is now a variable pointing to the FORTRAN compiler, which should come with every Linux system.
    make  # Install wgrib2 (the makefile should do all the work for you).

[4] To ensure that the install worked, open a terminal and type the following:

    /home/user/wgrib2/wgrib2/wgrib2

    This should print a help menu to the terminal.  If not, see the following page for more detailed Linux instructions: https://rda.ucar.edu/datasets/ds083.2/software/wgrib_install_guide.txt