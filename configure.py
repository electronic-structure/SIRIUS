import sys
import os
import urllib2
import tarfile
import subprocess
import json

packages = {
    "fftw" : ["http://www.fftw.org/fftw-3.3.4.tar.gz", 
              ["--disable-fortran", "--disable-mpi", "--disable-openmp", "--disable-threads"]
             ],
    "gsl"  : ["ftp://ftp.gnu.org/gnu/gsl/gsl-1.16.tar.gz", 
              ["--disable-shared"]
             ],
    "hdf5" : ["http://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.13.tar.gz",
              ["--enable-fortran", "--disable-shared", "--enable-static=yes", 
               "--disable-deprecated-symbols", "--disable-filters","--disable-parallel", "--with-zlib=no","--with-szlib=no"]
             ],
    "xc"   : ["http://www.tddft.org/programs/octopus/down.php?file=libxc/libxc-2.2.0.tar.gz",
              []
             ],
    "spg"  : ["http://downloads.sourceforge.net/project/spglib/spglib/spglib-1.6/spglib-1.6.0.tar.gz",
              []
             ]
}

def configure_package(package_name, platform):

    package = packages[package_name]
   
    file_url = package[0]
    
    local_file_name = os.path.split(file_url)[1]

    package_dir = os.path.splitext(os.path.splitext(local_file_name)[0])[0]

    cwdlibs = os.getcwd() + "/libs/"
    
    if (not os.path.exists("./libs/" + local_file_name)):
        try:
            print "Downloading " + file_url
            req = urllib2.Request(file_url)
            furl = urllib2.urlopen(req)
            
            local_file = open("./libs/" + local_file_name, "wb")
            local_file.write(furl.read())
            local_file.close()
        
        except urllib2.HTTPError, e:
            print "HTTP Error: ", e.code, url

        except urllib2.URLError, e:
            print "URL Error: ", e.reason, url

    tf = tarfile.open("./libs/" + local_file_name)
    tf.extractall("./libs/")

    new_env = os.environ.copy()
    new_env["CC"] = platform["CC"]
    new_env["CXX"] = platform["CXX"]
    new_env["FC"] = platform["FC"]
    new_env["FCCPP"] = platform["FCCPP"]

    p = subprocess.Popen(["./configure"] + package[1], cwd="./libs/"+package_dir, env=new_env)
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
        print ""
        print "SIRIUS configuration script"
        print ""
        print "First, edit 'platform.json' and specify your system compilers, system libraries and"
        print "libraries that you want to install. Then run:"
        print ""
        print "  python configure.py"
        print ""
        print "The \"install\" element in 'platform.json' contains the list of packages which are currently"
        print "missing on your system and which will be downloaded and configured by the script."
        print "The following package can be specified:"
        print ""
        print "  \"fftw\" - FFTW library"
        print "  \"gsl\"  - GNU scientific library"
        print "  \"hdf5\" - HDF5 library"
        print "  \"xc\"   - XC library"
        print "  \"spg\"  - Spglib"
        print ""
        sys.exit(0)
    
    fin = open("platform.json", "r");
    platform = json.load(fin)
    fin.close()

    makeinc = open("make.inc", "w")
    if ("MPI_CXX") in platform: 
        makeinc.write("CXX = " + platform["MPI_CXX"] + "\n")
    else:
        makeinc.write("CXX = " + platform["CXX"] + "\n")

    if ("MPI_CXX_OPT") in platform:
        makeinc.write("CXX_OPT = " + platform["MPI_CXX_OPT"] + "\n")
    else:
        makeinc.write("CXX_OPT = " + platform["CXX_OPT"] + "\n")

    if "NVCC" in platform: makeinc.write("NVCC = " + platform["NVCC"] + "\n")
    if "NVCC_OPT" in platform: makeinc.write("NVCC_OPT = " + platform["NVCC_OPT"] + "\n")
    
    make_packages = []
    clean_packages = []

    if "install" in platform:
        for name in platform["install"]:
            opts = configure_package(name, platform)
            makeinc.write("CXX_OPT := $(CXX_OPT) " + opts[0] + "\n")
            makeinc.write("LIBS := $(LIBS) " + opts[1] + "\n")
            make_packages.append(opts[2])
            clean_packages.append(opts[3])

    makeinc.write("CXX_OPT := $(CXX_OPT) -I" + os.getcwd() + "/libs/libjson\n")
    makeinc.write("LIBS := $(LIBS) " + os.getcwd() + "/libs/libjson/libjson.a\n")
    makeinc.write("LIBS := $(LIBS) " + platform["SYSTEM_LIBS"] + "\n")

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
    
    makef.write("clean_packages:\n")
    for i in range(len(clean_packages)):
        makef.write(clean_packages[i])
    makef.write("\tcd ./libs/libjson; make clean\n")

    makef.write("\n")
    makef.write("clean:\n")
    makef.write("\tcd src; make clean\n")
    makef.write("\tcd apps; make clean\n")

    makef.close()

if __name__ == "__main__":
    main()


