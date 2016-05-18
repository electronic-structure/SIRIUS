import sys
import os
import urllib.request
import tarfile
import subprocess
import json

packages = {
    "fftw" : {
        "url"     : "http://www.fftw.org/fftw-3.3.4.tar.gz",
        "options" : []
    },
    "gsl" : {
        "url"     : "ftp://ftp.gnu.org/gnu/gsl/gsl-2.1.tar.gz",
        "options" : ["--disable-shared"]
    },
    "hdf5" : {
        "url"     : "http://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.16.tar.gz",
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
        "url"     : "http://www.tddft.org/programs/octopus/down.php?file=libxc/libxc-3.0.0.tar.gz",
        "options" : []
    },
    "spg"  : {
        "url"     : "http://downloads.sourceforge.net/project/spglib/spglib/spglib-1.9/spglib-1.9.2.tar.gz",
        "options" : []
    }
}

def configure_package(package_name, platform):

    package = packages[package_name]

    file_url = package["url"]

    local_file_name = os.path.split(file_url)[1]

    package_dir = os.path.splitext(os.path.splitext(local_file_name)[0])[0]

    cwdlibs = os.getcwd() + "/libs/"

    if (not os.path.exists("./libs/" + local_file_name)):
        try:
            print("Downloading %s"%file_url)
            req = urllib.request.Request(file_url)
            furl = urllib.request.urlopen(req)

            local_file = open("./libs/" + local_file_name, "wb")
            local_file.write(furl.read())
            local_file.close()

        except urllib2.HTTPError as err:
            print("HTTP Error: %i %s"%(e.code, url))

        except urllib2.URLError as err:
            print("URL Error: %i %s"%(err.reason, url))

    tf = tarfile.open("./libs/" + local_file_name)
    tf.extractall("./libs/")

    new_env = os.environ.copy()

    new_env["CC"] = platform["CC"]
    new_env["CXX"] = platform["CXX"]
    new_env["FC"] = platform["FC"]
    new_env["F77"] = platform["FC"]
    new_env["FCCPP"] = platform["FCCPP"]

    p = subprocess.Popen(["./configure"] + package["options"], cwd = "./libs/" + package_dir, env = new_env)
    p.wait()

    retval = []

    if (package_name == "xc"):
        retval = ["-I" + cwdlibs + package_dir + "/src" + " -I" + cwdlibs + package_dir,
                  cwdlibs + package_dir + "/src/.libs/libxc.a",
                  "\tcd ./libs/" + package_dir + "/src; make; ar -r ./.libs/libxc.a *.o\n",
                  "\tcd ./libs/" + package_dir + "; make clean\n"]

    if (package_name == "spg"):
        retval = ["-I" + cwdlibs + package_dir + "/src",
                  cwdlibs + package_dir + "/src/.libs/libsymspg.a",
                  "\tcd ./libs/" + package_dir + "; make\n",
                  "\tcd ./libs/" + package_dir + "; make clean\n"]

    if (package_name == "gsl"):
        retval = ["-I" + cwdlibs + package_dir,
                  cwdlibs + package_dir + "/.libs/libgsl.a",
                  "\tcd ./libs/" + package_dir + "; make\n",
                  "\tcd ./libs/" + package_dir + "; make clean\n"]

    if (package_name == "fftw"):
        retval = ["-I" + cwdlibs + package_dir + "/api",
                  cwdlibs + package_dir + "/.libs/libfftw3.a",
                  "\tcd ./libs/" + package_dir + "; make\n",
                  "\tcd ./libs/" + package_dir + "; make clean\n"]

    if (package_name == "hdf5"):
        retval = ["-I" + cwdlibs + package_dir + "/src",
                  cwdlibs + package_dir + "/src/.libs/libhdf5.a",
                  "\tcd ./libs/" + package_dir + "; make\n",
                  "\tcd ./libs/" + package_dir + "; make clean\n"]

    return retval



def main():

    if "--help" in sys.argv:
        print("\n" \
              "SIRIUS configuration script\n\n" \
              "First, edit 'platform.json' and specify your system compilers, system libraries and\n" \
              "libraries that you want to install. Then run:\n\n" \
              "  python configure.py\n\n" \
              "The \"install\" element in 'platform.json' contains the list of packages which are currently\n" \
              "missing on your system and which will be downloaded and configured by the script.\n" \
              "The following package can be specified:\n\n" \
              "  \"fftw\" - FFTW library\n" \
              "  \"gsl\"  - GNU scientific library\n" \
              "  \"hdf5\" - HDF5 library\n" \
              "  \"xc\"   - XC library\n" \
              "  \"spg\"  - Spglib\n")
        sys.exit(0)

    fin = open("platform.json", "r");
    platform = json.load(fin)
    fin.close()

    makeinc = open("make.inc", "w")
    if "MPI_CXX" in platform:
        makeinc.write("CXX = " + platform["MPI_CXX"] + "\n")
    else:
        makeinc.write("CXX = " + platform["CXX"] + "\n")

    if "MPI_CXX_OPT" in platform:
        makeinc.write("CXX_OPT = " + platform["MPI_CXX_OPT"] + "\n")
    else:
        makeinc.write("CXX_OPT = " + platform["CXX_OPT"] + "\n")



    if "NVCC" in platform: makeinc.write("NVCC = " + platform["NVCC"] + "\n")
    if "NVCC_OPT" in platform: makeinc.write("NVCC_OPT = " + platform["NVCC_OPT"] + "\n")

    if "MPI_FC" in platform: makeinc.write("MPI_FC = " + platform["MPI_FC"] + "\n")
    if "MPI_FC_OPT" in platform: makeinc.write("MPI_FC_OPT = " + platform["MPI_FC_OPT"] + "\n")

    make_packages = []
    clean_packages = []

    makeinc.write("CXX_OPT := $(CXX_OPT) -I" + os.getcwd() + "/src\n")

    if "install" in platform:
        for name in platform["install"]:
            opts = configure_package(name, platform)
            makeinc.write("CXX_OPT := $(CXX_OPT) " + opts[0] + "\n")
            makeinc.write("LIBS := $(LIBS) " + opts[1] + "\n")
            make_packages.append(opts[2])
            clean_packages.append(opts[3])

    build_elpa = False
    if "-D__ELPA" in platform["MPI_CXX_OPT"]:
        build_elpa = True
        makeinc.write("LIBS := $(LIBS) " + os.getcwd() + "/libs/elpa/latest/libelpa.a\n")

    makeinc.write("CXX_OPT := $(CXX_OPT) -I" + os.getcwd() + "/libs/libjson\n")
    makeinc.write("LIBS := $(LIBS) " + os.getcwd() + "/libs/libjson/libjson.a\n")
    makeinc.write("LIBS := $(LIBS) " + platform["SYSTEM_LIBS"] + "\n")


### TEST DEBUG CONF ####
    if "MPI_CXX_OPT_DBG" in platform:
        makeinc.write("CXX_OPT_DBG = " + platform["MPI_CXX_OPT_DBG"] + "\n")
    else:
        makeinc.write("CXX_OPT_DBG = " + platform["CXX_OPT_DBG"] + "\n")

    makeinc.write("CXX_OPT_DBG := $(CXX_OPT_DBG) -I" + os.getcwd() + "/src\n")

    if "install" in platform:
        for name in platform["install"]:
            opts = configure_package(name, platform)
            makeinc.write("CXX_OPT_DBG := $(CXX_OPT_DBG) " + opts[0] + "\n")

    makeinc.write("CXX_OPT_DBG := $(CXX_OPT_DBG) -I" + os.getcwd() + "/libs/libjson\n")
############

    makeinc.close()

    makef = open("Makefile", "w")
    makef.write("include ./make.inc \n")
    makef.write("\n")
    makef.write(".PHONY: apps\n")
    makef.write("\n")
    makef.write("all: packages sirius apps\n")
    makef.write("\n")
    makef.write("sirius:\n")
    makef.write("\tcd src; make\n")
    makef.write("\n")
    makef.write("apps:\n")
    makef.write("\tcd apps; make\n")
    makef.write("\n")
    makef.write("packages:\n")
    for i in range(len(make_packages)):
        makef.write(make_packages[i])
    makef.write("\tcd ./libs/libjson; make\n")
    if build_elpa: makef.write("\tcd ./libs/elpa/latest; make\n")

    makef.write("cleanall:\n")
    for i in range(len(clean_packages)):
        makef.write(clean_packages[i])
    makef.write("\tcd ./libs/libjson; make clean\n")
    if build_elpa: makef.write("\tcd ./libs/elpa/latest; make clean\n")

    makef.write("\n")
    makef.write("clean:\n")
    makef.write("\tcd src; make clean\n")
    makef.write("\tcd apps; make clean\n")

    makef.close()

if __name__ == "__main__":
    main()
