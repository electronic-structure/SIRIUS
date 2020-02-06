cmake_minimum_required(VERSION 3.12)

project(SIRIUS VERSION 6.4.4)

# set language and standard
enable_language(CXX Fortran)
set(CMAKE_CXX_STANDARD 14)

# user variables
set(CREATE_PYTHON_MODULE OFF CACHE BOOL "create sirius python module")
set(CREATE_FORTRAN_BINDINGS ON CACHE BOOL "build Fortran bindings")
set(BUILD_DOCS OFF CACHE BOOL "build doxygen doc")
set(USE_ELPA OFF CACHE BOOL "use scalapack")
set(USE_MAGMA OFF CACHE BOOL "use MAGMA")
set(USE_CUDA OFF CACHE BOOL "use CUDA")
set(USE_ROCM OFF CACHE BOOL "use ROCM AMD GPU code")
set(USE_NVTX OFF CACHE BOOL "use Nvidia profiling tools library")
set(USE_VDWXC OFF CACHE BOOL "use libvdwxc for van der Walls corrections")
set(USE_MKL OFF CACHE BOOL "use Intel MKL")
set(USE_CRAY_LIBSCI OFF CACHE BOOL "use LAPACK/SCALAPACK from Cray LIBSCI")
set(USE_SCALAPACK OFF CACHE BOOL "use scalapack")
set(BUILD_TESTS OFF CACHE BOOL "build tests")
set(GPU_MODEL "none" CACHE STRING "The target GPU architecture; one of {none,P100,V100,G10x0}")
set(DEBUG_MEMORY_POOL OFF CACHE BOOL "explicit debugging of memory pool")
set(USE_OPENMP ON CACHE BOOL "use OpenMP")
set(PYTHON2 OFF CACHE STRING "Use Python 2.7")
set(USE_PROFILER ON CACHE BOOL "measure execution of functions with timer")
set(USE_MEMORY_POOL ON CACHE BOOL "use memory pool")

if(USE_MAGMA AND NOT USE_CUDA)
  message(FATAL_ERROR "MAGMA depends on Cuda, must enable Cuda or disable MAGMA")
endif()

if(USE_CUDA AND USE_ROCM)
  message(FATAL_ERROR "USE_CUDA and USE_ROCM must not be enabled at the same time!")
endif()

if(USE_MKL AND NOT (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel"))
  message(FATAL_ERROR "Unsupported compiler")
endif()

set_property(CACHE GPU_MODEL PROPERTY STRINGS "none" "P100" "V100" "G10x0")

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")
list(APPEND CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH})


include(GitSubmodule)    # include custom defined FindPackage modules
include(GNUInstallDirs)  # required to get a proper LIBDIR variable

if(PYTHON2)
  # force cmake to use python2
  set(PYBIND11_PYTHON_VERSION 2.7)
  find_package(Python2 REQUIRED)
  set(PYTHON_EXECUTABLE ${Python2_EXECUTABLE})
else()
  find_package(Python3 REQUIRED)
  set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
endif()

# Set release as the default build type.
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "debug" "release" "relwithdebinfo")
endif()

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -ggdb -DDEBUG")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -ggdb -O2")
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O2")
endif()

# Makes the build directory relocatable: requires CMake 3.14.7
#set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)

# preserve rpaths when installing and make the install folder relocatable
# use `CMAKE_SKIP_INSTALL_RPATH` to skip this
# https://spack.readthedocs.io/en/latest/workflows.html#write-the-cmake-build
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES 
          "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" isSystemDir)
# skip RPATH if SIRIUS is installed to system directories
if(isSystemDir STREQUAL "-1")
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  if(APPLE)
    set(basePoint @loader_path)
  else()
    set(basePoint $ORIGIN)
  endif()
  file(RELATIVE_PATH relDir ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}
                            ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
  set(CMAKE_INSTALL_RPATH ${basePoint} ${basePoint}/${relDir})
endif()

# build shared libraries by default
option(BUILD_SHARED_LIBS "Build shared libraries." ON)

# generate compile_commands.json with compile commands for each target
set(CMAKE_EXPORT_COMPILE_COMMANDS "YES")

# dependencies
find_package(MPI REQUIRED)
find_package(GSL REQUIRED)
find_package(LibXC 3.0.0 REQUIRED)
find_package(LibSPG REQUIRED)
find_package(HDF5 REQUIRED C)
include(cmake/hdf5_target.cmake)
find_package(SpFFT 0.9.6 CONFIG REQUIRED)

if (USE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

set(LINALG_LIB "")
if(USE_MKL)
  set(USE_MKL_SHARED_LIBS On) # link against shared MKL libraries
  find_package(MKL REQUIRED)
  set(LINALG_LIB "sirius::mkl")
elseif(USE_CRAY_LIBSCI)
  find_package(CRAY_LIBSCI REQUIRED)
  set(LINALG_LIB "${CRAY_LIBSCI_LIBRARIES}")
else()
  find_package(LAPACK REQUIRED)
  set(LINALG_LIB "${LAPACK_LIBRARIES}")
  if(USE_SCALAPACK)
    find_package(SCALAPACK REQUIRED) # just sets scalapack_DIR
    set(LINALG_LIB "${LINALG_LIB};${SCALAPACK_LIBRARIES}")
  endif()
endif()

if(USE_ELPA)
  find_package(Elpa REQUIRED)
endif(USE_ELPA)

if(USE_MAGMA)
  find_package(MAGMA REQUIRED)
endif(USE_MAGMA)

if(USE_VDWXC)
  find_package(LibVDWXC 0.3.0 REQUIRED)
endif(USE_VDWXC)

if(USE_CUDA)
  enable_language(CUDA)
  find_package(CUDA)
  include(cmake/cudalibs_target.cmake)

  if(GPU_MODEL MATCHES "P100")
    message("set nvcc flags for P100")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")
  elseif(GPU_MODEL MATCHES "V100")
    message("set nvcc flags for V100")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70")
  elseif(GPU_MODEL MATCHES "G10x0")
    message("set nvcc flags for geforce 1000")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_61")
  endif()

endif(USE_CUDA)

if(USE_ROCM)
  message(STATUS "WARNING: ROCM enabled, prototype feature! Only limited functionality available.")
  find_package(ROCM COMPONENTS rocfft hipblas)
  if(NOT ${ROCM_HIP_PLATFORM} STREQUAL hcc)
    message(FATAL_ERROR "Compilation on Nvidia platform not supported with ROCM enabled!")
  endif()
  add_definitions(${ROCM_DEFINITIONS})
  include_directories(${ROCM_INCLUDE_DIRS})
  add_subdirectory(src/SDDK/GPU/hipblas_port)
  include_directories(src/SDDK/GPU/hipblas_port)
endif()

# check if git command exists
find_program(GIT_EXE NAMES git)

# generate version header
string(TIMESTAMP SIRIUS_TIMESTAMP "%Y-%m-%d %H:%M:%S")
if(DEFINED GIT_EXE AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  execute_process(COMMAND git rev-parse HEAD
                  OUTPUT_VARIABLE SIRIUS_SHA
                  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
                  ERROR_QUIET
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND git describe --all
                  OUTPUT_VARIABLE SIRIUS_GIT_BRANCH
                  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
                  ERROR_QUIET
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
else()
  set(SIRIUS_GIT_BRANCH "release v${SIRIUS_VERSION}")
  set(SIRIUS_SHA "https://api.github.com/repos/electronic-structure/SIRIUS/git/ref/tags/v${SIRIUS_VERSION}")
endif()
configure_file("${PROJECT_SOURCE_DIR}/src/version.hpp.in"
               "${PROJECT_BINARY_DIR}/src/version.hpp"
               @ONLY)

# generate runtime options
file(STRINGS "${PROJECT_SOURCE_DIR}/src/options.json"
              SIRIUS_RUNTIME_OPTIONS NEWLINE_CONSUME)
configure_file("${PROJECT_SOURCE_DIR}/src/runtime_options_json.hpp.in"
               "${PROJECT_BINARY_DIR}/src/runtime_options_json.hpp"
               @ONLY)

# install targets and modules
include(cmake/env_vars_map.cmake)
configure_file("${PROJECT_SOURCE_DIR}/cmake/siriusConfig.cmake.in"
               "${PROJECT_BINARY_DIR}/siriusConfig.cmake"
               @ONLY)
install(FILES "${PROJECT_BINARY_DIR}/siriusConfig.cmake"
              "${PROJECT_SOURCE_DIR}/cmake/cudalibs_target.cmake"
              "${PROJECT_SOURCE_DIR}/cmake/hdf5_target.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/sirius")
install(DIRECTORY "${PROJECT_SOURCE_DIR}/cmake/modules"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/sirius"
        FILES_MATCHING PATTERN "*.cmake")

# sirius library
add_subdirectory(src)

# applications
if(BUILD_TESTS)
  add_subdirectory(apps/tests)
  add_subdirectory(apps/unit_tests)
endif(BUILD_TESTS)

add_subdirectory(apps/atoms)
add_subdirectory(apps/dft_loop)
add_subdirectory(apps/upf)
add_subdirectory(apps/utils)
add_subdirectory(python_module)
add_subdirectory(doc)
