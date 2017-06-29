import sys
import os
import tarfile
import subprocess
import json

packages = {
    "spg"  : {
        "url"     : "https://github.com/atztogo/spglib/archive/v1.9.9.tar.gz",
        "options" : []
    },
    "fftw" : {
        "url"     : "http://www.fftw.org/fftw-3.3.5.tar.gz",
        "options" : []
    },
    "gsl" : {
        "url"     : "ftp://ftp.gnu.org/gnu/gsl/gsl-2.3.tar.gz",
        "options" : ["--disable-shared"]
    },
    "hdf5" : {
        "url"     : "https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.1/src/hdf5-1.10.1.tar.gz",
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
    }
}

def configure_package(package_name, platform):

    package = packages[package_name]

    file_url = package["url"]

    local_file_name = os.path.split(file_url)[1]

    package_dir = os.path.splitext(os.path.splitext(local_file_name)[0])[0]

    # spglib requires a special care
    if package_name == 'spg':
        package_dir = "spglib-" + package_dir[1:]

    cwdlibs = os.getcwd() + "/libs/"

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

    new_env["CC"] = platform["CC"]
    new_env["CXX"] = platform["CXX"]
    new_env["FC"] = platform["FC"]
    new_env["F77"] = platform["FC"]
    new_env["FCCPP"] = platform["FCCPP"]

    # spglib requires a special care
    if package_name == 'spg':
        p = subprocess.Popen(["aclocal"], cwd = "./libs/" + package_dir, env = new_env)
        p.wait()
        p = subprocess.Popen(["autoheader"], cwd = "./libs/" + package_dir, env = new_env)
        p.wait()
        if sys.platform == 'darwin':
            p = subprocess.Popen(["glibtoolize"], cwd = "./libs/" + package_dir, env = new_env)
        else:
            p = subprocess.Popen(["libtoolize"], cwd = "./libs/" + package_dir, env = new_env)
        p.wait()
        p = subprocess.Popen(["touch", "INSTALL", "NEWS", "README", "AUTHORS"], cwd = "./libs/" + package_dir, env = new_env)
        p.wait()
        p = subprocess.Popen(["automake", "-acf"], cwd = "./libs/" + package_dir, env = new_env)
        p.wait()
        p = subprocess.Popen(["autoconf"], cwd = "./libs/" + package_dir, env = new_env)
        p.wait()
            

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
    if len(sys.argv) < 2:
        print("\n" \
              "SIRIUS configuration script\n\n" \
              "First, edit 'platform.json' and specify your system compilers, system libraries and\n" \
              "libraries that you want to install. Then run:\n\n" \
              "> python configure.py platform.json\n\n" \
              "The \"install\" element in 'platform.json' contains the list of packages which are currently\n" \
              "missing on your system and which will be downloaded and configured by the script.\n" \
              "The following packages can be specified:\n\n" \
              "  \"fftw\" - FFTW library\n" \
              "  \"gsl\"  - GNU scientific library\n" \
              "  \"hdf5\" - HDF5 library\n" \
              "  \"xc\"   - XC library\n" \
              "  \"spg\"  - Spglib\n")
        sys.exit(0)

    fin = open(sys.argv[1], "r");
    platform = json.load(fin)
    fin.close()

    makeinc = open('make.inc', "w")

    makeinc.write('''debug = false
BASIC_CXX_OPT = -O3 -DNDEBUG
ifeq ($(debug), true)
  BASIC_CXX_OPT = -O1 -g -ggdb
endif'''+"\n")

    if 'MPI_CXX' in platform:
        makeinc.write("CXX = %s\n"%platform['MPI_CXX'])
    else:
        makeinc.write("CXX = %s\n"%platform['CXX'])

    if 'MPI_CXX_OPT' in platform:
        makeinc.write("CXX_OPT = $(BASIC_CXX_OPT) %s\n"%platform['MPI_CXX_OPT'])
    else:
        makeinc.write("CXX_OPT = $(BASIC_CXX_OPT) %s\n"%platform['CXX_OPT'])
    
    makeinc.write("CXX_OPT := $(CXX_OPT) -I%s/src\n"%os.getcwd())
    makeinc.write("CXX_OPT := $(CXX_OPT) -I%s/src/SDDK\n"%os.getcwd())
    if 'CUDA_ROOT' in platform:
        makeinc.write("CXX_OPT := $(CXX_OPT) -I%s/include\n"%platform['CUDA_ROOT']);

    if 'MAGMA_ROOT' in platform:
        makeinc.write("CXX_OPT := $(CXX_OPT) -I%s/include\n"%platform['MAGMA_ROOT']);
        makeinc.write("CXX_OPT := $(CXX_OPT) -I%s/control\n"%platform['MAGMA_ROOT']);


    if 'CUDA_ROOT' in platform:
        if 'NVCC' in platform: 
            makeinc.write("NVCC = %s/bin/%s\n"%(platform['CUDA_ROOT'], platform['NVCC']))
        if 'NVCC_OPT' in platform: 
            makeinc.write("NVCC_OPT = %s\n"%platform['NVCC_OPT'])

    if 'MPI_FC' in platform:
        makeinc.write("MPI_FC = %s\n"%platform['MPI_FC'])
    if 'MPI_FC_OPT' in platform:
        makeinc.write("MPI_FC_OPT = %s\n"%platform['MPI_FC_OPT'])

    make_packages = []
    clean_packages = []

    if 'install' in platform:
        for name in platform['install']:
            opts = configure_package(name, platform)
            makeinc.write("CXX_OPT := $(CXX_OPT) %s\n"%opts[0])
            makeinc.write("LIBS := $(LIBS) %s\n"%opts[1])
            make_packages.append(opts[2])
            clean_packages.append(opts[3])

    build_elpa = False
    if "-D__ELPA" in platform['MPI_CXX_OPT']:
        build_elpa = True
        makeinc.write("LIBS := $(LIBS) %s/libs/elpa/latest/libelpa.a\n"%os.getcwd())

    if 'CUDA_ROOT' in platform:
        makeinc.write("LIBS := $(LIBS) -L%s/lib -lcublas -lcudart -lcufft -lcusparse -lnvToolsExt -Wl,-rpath,%s/lib  \n"%(platform['CUDA_ROOT'], platform['CUDA_ROOT']))
    if 'MAGMA_ROOT' in platform:
        makeinc.write("LIBS := $(LIBS) %s/lib/libmagma_sparse.a\n"%platform['MAGMA_ROOT']) 
        makeinc.write("LIBS := $(LIBS) %s/lib/libmagma.a\n"%platform['MAGMA_ROOT']) 
        
    makeinc.write("LIBS := $(LIBS) " + platform["SYSTEM_LIBS"] + "\n")

    dbg_conf = False
    if "MPI_CXX_OPT_DBG" in platform:
        makeinc.write("CXX_OPT_DBG = " + platform["MPI_CXX_OPT_DBG"] + "\n")
        dbg_conf = True
    if "CXX_OPT_DBG" in platform:
        makeinc.write("CXX_OPT_DBG = " + platform["CXX_OPT_DBG"] + "\n")
        dbg_conf = True
    
    if dbg_conf:
        makeinc.write("CXX_OPT_DBG := $(CXX_OPT_DBG) -I" + os.getcwd() + "/src\n")

        if "install" in platform:
            for name in platform["install"]:
                opts = configure_package(name, platform)
                makeinc.write("CXX_OPT_DBG := $(CXX_OPT_DBG) " + opts[0] + "\n")

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
    if build_elpa: makef.write("\tcd ./libs/elpa/latest; make\n")

    makef.write("cleanall:\n")
    for i in range(len(clean_packages)):
        makef.write(clean_packages[i])
    if build_elpa: makef.write("\tcd ./libs/elpa/latest; make clean\n")

    makef.write("\n")
    makef.write("clean:\n")
    makef.write("\tcd src; make clean\n")
    makef.write("\tcd apps; make clean\n")

    makef.close()

if __name__ == "__main__":
    main()
