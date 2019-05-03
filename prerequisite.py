import sys
import os
import shutil
import tarfile
import subprocess
import json

packages = {
    "spg"  : {
        "url"     : "https://github.com/atztogo/spglib/archive/v1.12.0.tar.gz",
        "options" : []
    },
    "fftw" : {
        "url"     : "http://www.fftw.org/fftw-3.3.8.tar.gz",
        "options" : []
    },
    "gsl" : {
        "url"     : "ftp://ftp.gnu.org/gnu/gsl/gsl-2.5.tar.gz",
        "options" : ["--disable-shared"]
    },
    "hdf5" : {
        "url"     : "https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.5/src/hdf5-1.10.5.tar.gz",
        "options" : ["--enable-fortran",
                     "--disable-shared",
                     "--enable-static=yes",
                     "--disable-deprecated-symbols",
                     "--disable-filters",
                     "--disable-parallel",
                     "--with-zlib=no",
                     "--with-szlib=no"]
    },
    "xc"   : {
        "url"     : "http://www.tddft.org/programs/octopus/down.php?file=libxc/4.3.4/libxc-4.3.4.tar.gz",
        "options" : []
    }
}

def configure_package(package_name, prefix):

    package = packages[package_name]

    file_url = package["url"]

    local_file_name = os.path.split(file_url)[1]

    package_dir = os.path.splitext(os.path.splitext(local_file_name)[0])[0]

    # spglib requires a special care
    if package_name == 'spg':
        package_dir = "spglib-" + package_dir[1:]

    cwdlibs = os.getcwd() + "/libs/"

    if not os.path.exists(cwdlibs):
        os.mkdir(cwdlibs)

    if (not os.path.exists("./libs/" + local_file_name)):
        try:
            if sys.version_info < (3, 0):
                import urllib
                print("Downloading %s"%file_url)
                urllib.urlretrieve(file_url, "./libs/" + local_file_name)
            else:
                import urllib.request
                print("Downloading %s"%file_url)
                req = urllib.request.Request(file_url)
                furl = urllib.request.urlopen(req)

                local_file = open("./libs/" + local_file_name, "wb")
                local_file.write(furl.read())
                local_file.close()

        except Exception as e:
            print("{0}".format(e));
            sys.exit(1)

    tf = tarfile.open("./libs/" + local_file_name)
    tf.extractall("./libs/")

    new_env = os.environ.copy()
    if 'FC' in new_env:
        new_env['F77'] = new_env['FC']
    # python modules needs position independent code
    if not 'CCFLAGS' in new_env:
        new_env['CCFLAGS'] = '-fPIC'
    else:
        new_env['CCFLAGS'] += ' -fPIC'
    if not 'CFLAGS' in new_env:
        new_env['CFLAGS'] = '-fPIC'
    else:
        new_env['CFLAGS'] += ' -fPIC'

    # spglib requires a special care
    if package_name == 'spg':
        os.mkdir('./libs/_build')
        p = subprocess.Popen(["cmake", "../" + package_dir, "-DCMAKE_INSTALL_PREFIX=" + prefix], cwd = './libs/_build', env = new_env)
        p.wait()

        #p = subprocess.Popen(["aclocal"], cwd = "./libs/" + package_dir, env = new_env)
        #p.wait()
        #p = subprocess.Popen(["autoheader"], cwd = "./libs/" + package_dir, env = new_env)
        #p.wait()
        #if sys.platform == 'darwin':
        #    p = subprocess.Popen(["glibtoolize"], cwd = "./libs/" + package_dir, env = new_env)
        #else:
        #    p = subprocess.Popen(["libtoolize"], cwd = "./libs/" + package_dir, env = new_env)
        #p.wait()
        #p = subprocess.Popen(["touch", "INSTALL", "NEWS", "README", "AUTHORS"], cwd = "./libs/" + package_dir, env = new_env)
        #p.wait()
        #p = subprocess.Popen(["automake", "-acf"], cwd = "./libs/" + package_dir, env = new_env)
        #p.wait()
        #p = subprocess.Popen(["autoconf"], cwd = "./libs/" + package_dir, env = new_env)
        #p.wait()
        p = subprocess.Popen(["make", "install"], cwd = './libs/_build', env = new_env)
        p.wait()
        shutil.rmtree('./libs/_build')
        os.makedirs(prefix + '/include/spglib', exist_ok=True)
        shutil.copyfile(prefix + '/include/spglib.h', prefix + '/include/spglib/spglib.h')
    else:
        cmd = ["./configure"] + package["options"] + ["--prefix=%s"%prefix]
        p = subprocess.Popen(cmd, cwd = "./libs/" + package_dir, env = new_env)
        p.wait()
        p = subprocess.Popen(["make"], cwd = "./libs/" + package_dir, env = new_env)
        p.wait()
        p = subprocess.Popen(["make", "install"], cwd = "./libs/" + package_dir, env = new_env)
        p.wait()


def main():
    if len(sys.argv) < 2:
        print("\n" \
              "Install dependencies for the SIRIUS library.\n\n" \
              "Usage:\n\n" \
              "> python prerequisite.py install_prefix [packages]\n\n" \
              "The \"packages\" string contains the list of packages which are currently\n" \
              "missing on your system and which will be downloaded and configured by the script.\n\n" \
              "The following packages can be specified:\n\n" \
              "  \"fftw\" - FFTW library\n" \
              "  \"gsl\"  - GNU scientific library\n" \
              "  \"hdf5\" - HDF5 library\n" \
              "  \"xc\"   - XC library\n" \
              "  \"spg\"  - Spglib\n\n"\
              "The following standard environment variables are expected:\n\n" \
              "  CC    - C compiler\n" \
              "  CXX   - C++ compiler\n" \
              "  FC    - Fortran compiler\n" \
              "  FCCPP - Fortran preprocessor\n\n" \
              "For example:\n\n" \
              "> CC=gcc CXX=gcc FC=gfortran FCCPP=cpp python prerequisite.py $HOME/local fftw gsl hdf5 xc spg\n\n" \
              "will download, configure and install all the packages into $HOME/local\n")

        sys.exit(0)

    prefix = sys.argv[1]
    print("Installation prefix: %s"%prefix)
    for pkg in sys.argv[2:]:
        configure_package(pkg, prefix)

if __name__ == "__main__":
    main()
