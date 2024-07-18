cmake_minimum_required(VERSION 3.23)

include(CMakePackageConfigHelpers)

project(SIRIUS
  DESCRIPTION "Domain specific library for electronic structure calculations"
  HOMEPAGE_URL "https://github.com/electronic-structure/SIRIUS"
  VERSION 7.6.0
  LANGUAGES CXX C)

# for CUDA_ARCHITECTURES
if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()

# user variables
option(SIRIUS_CREATE_PYTHON_MODULE    "create sirius Python module" OFF)
option(SIRIUS_CREATE_FORTRAN_BINDINGS "build Fortran bindings" ON)
option(SIRIUS_BUILD_DOCS              "build doxygen doc" OFF)
option(SIRIUS_USE_ELPA                "use elpa" OFF)
option(SIRIUS_USE_MAGMA               "use MAGMA" OFF)
option(SIRIUS_USE_NLCGLIB             "enable nlcglib" OFF)
option(SIRIUS_USE_CUDA                "use CUDA" OFF)
option(SIRIUS_USE_ROCM                "use ROCM AMD GPU code" OFF)
option(SIRIUS_USE_NVTX                "use Nvidia profiling tools library" OFF)
option(SIRIUS_USE_VDWXC               "use libvdwxc for van der Walls corrections" OFF)
option(SIRIUS_USE_MKL                 "use Intel MKL" OFF)
option(SIRIUS_USE_CRAY_LIBSCI         "use LAPACK/SCALAPACK from Cray LIBSCI" OFF)
option(SIRIUS_USE_SCALAPACK           "use scalapack" OFF)
option(SIRIUS_USE_DLAF                "use DLA-Future" OFF)
option(SIRIUS_BUILD_APPS              "build apps" ON)
option(SIRIUS_USE_OPENMP              "use OpenMP" ON)
option(SIRIUS_USE_PROFILER            "measure execution of functions with timer" ON)
option(SIRIUS_USE_MEMORY_POOL         "use memory pool" ON)
option(SIRIUS_USE_PUGIXML             "enable reading of UPF v2 files with pugixml" OFF)
option(SIRIUS_USE_POWER_COUNTER       "measure energy consumption with power counters" OFF)
option(BUILD_TESTING                  "build test executables" OFF) # override default setting in CTest module
option(SIRIUS_USE_VCSQNM              "use variable cell stabilized quasi Newton method" OFF)

set(SIRIUS_GPU_MEMORY_ALIGMENT "512" CACHE STRING "memory alignment of the GPU")
set(SIRIUS_USE_FP32 "AUTO" CACHE STRING "Enable single precision support.")

set_property(CACHE SIRIUS_USE_FP32 PROPERTY STRINGS "AUTO" "ON" "OFF")

# set language and standard

if(SIRIUS_CREATE_FORTRAN_BINDINGS)
  enable_language(Fortran)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# This ensures that -std=c++17 is passed to nvcc. See
# https://github.com/electronic-structure/SIRIUS/issues/716 for details.
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
endif()

if(SIRIUS_USE_CUDA AND SIRIUS_USE_ROCM)
  message(FATAL_ERROR "SIRIUS_USE_CUDA and SIRIUS_USE_ROCM must not be enabled at the same time!")
endif()

add_compile_definitions(SIRIUS_GPU_MEMORY_ALIGMENT=${GPU_MEMORY_ALIGMENT})

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")
list(APPEND CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH})

include(GitSubmodule)    # include custom defined FindPackage modules
include(GNUInstallDirs)  # required to get a proper LIBDIR variable

if (SIRIUS_CREATE_PYTHON_MODULE)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
endif()

# Set release as the default build type.
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "debug" "release" "relwithdebinfo")
endif()

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
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
set(MPI_DETERMINE_LIBRARY_VERSION TRUE)

# build shared libraries by default
option(BUILD_SHARED_LIBS "Build shared libraries." ON)

# generate compile_commands.json with compile commands for each target
set(CMAKE_EXPORT_COMPILE_COMMANDS "YES")

if(SIRIUS_USE_NLCGLIB)
  find_package(nlcglib REQUIRED)
  find_package(Kokkos REQUIRED)
endif()

find_package(MPI REQUIRED)
find_package(GSL REQUIRED)
find_package(LibXC 4.0.0 REQUIRED)
find_package(LibSPG REQUIRED)
find_package(HDF5 REQUIRED C)
include(cmake/hdf5_target.cmake)
find_package(SpFFT 1.0.2 CONFIG REQUIRED)
find_package(SPLA 1.4.0 CONFIG REQUIRED)

if (SIRIUS_USE_PUGIXML)
   find_package(pugixml REQUIRED)
endif()

# Check if SpFFT support single precision if required
if(NOT SIRIUS_USE_FP32 STREQUAL "OFF")
  if(SPFFT_SINGLE_PRECISION)
    set(SIRIUS_USE_FP32_BOOL "ON")
  elseif(SIRIUS_USE_FP32 STREQUAL "ON")
    message(FATAL_ERROR "Single precision option enabled, but SpFFT not compiled with single precision support.")
  else()
    set(SIRIUS_USE_FP32_BOOL "OFF")
  endif()
endif()

if (SIRIUS_USE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

set(SIRIUS_LINALG_LIB "")
if(SIRIUS_USE_MKL)
  set(MKL_INTERFACE "lp64" CACHE STRING "")
  set(MKL_THREADING "sequential" CACHE STRING "")
  set(MKL_MPI "mpich" CACHE STRING "")
  find_package(MKL CONFIG REQUIRED)
  set(SIRIUS_LINALG_LIB "MKL::MKL")
elseif(SIRIUS_USE_CRAY_LIBSCI)
  find_package(CRAY_LIBSCI REQUIRED)
  set(SIRIUS_LINALG_LIB "${SIRIUS_CRAY_LIBSCI_LIBRARIES}")
else()
  find_package(LAPACK REQUIRED)
  set(SIRIUS_LINALG_LIB "${LAPACK_LIBRARIES}")
  if(SIRIUS_USE_SCALAPACK)
    find_package(SCALAPACK REQUIRED) # just sets scalapack_DIR
    set(SIRIUS_LINALG_LIB "${SIRIUS_LINALG_LIB};${SIRIUS_SCALAPACK_LIBRARIES}")
  endif()
endif()

if(SIRIUS_USE_DLAF)
  find_package(DLAF REQUIRED)
endif(SIRIUS_USE_DLAF)

if(SIRIUS_USE_ELPA)
  find_package(Elpa REQUIRED)
endif(SIRIUS_USE_ELPA)

if(SIRIUS_USE_MAGMA)
  find_package(MAGMA REQUIRED)
endif(SIRIUS_USE_MAGMA)

if(SIRIUS_USE_VDWXC)
  find_package(LibVDWXC 0.3.0 REQUIRED)
endif(SIRIUS_USE_VDWXC)

find_package(costa CONFIG REQUIRED)

if(SIRIUS_USE_CUDA)
  enable_language(CUDA)
  # note find cudatoolkit is called inside the include file. the
  # sirius::cuda_libs target is also defined there
  include(cmake/cudalibs_target.cmake)
  message(STATUS "CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}")
endif(SIRIUS_USE_CUDA)

if(SIRIUS_USE_ROCM)
  # safegaurd.
  if (NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES gfx801 gfx900 gfx90a)
  endif()

  enable_language(HIP)
  find_package(rocblas CONFIG REQUIRED)
  find_package(rocsolver CONFIG REQUIRED)
  set(CMAKE_HIP_STANDARD 14)
endif()

if(SIRIUS_USE_MEMORY_POOL)
  find_package(umpire REQUIRED)
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
  message(STATUS "git hash ${SIRIUS_SHA}")
else()
  set(SIRIUS_GIT_BRANCH "release v${SIRIUS_VERSION}")
  set(SIRIUS_SHA "https://api.github.com/repos/electronic-structure/SIRIUS/git/ref/tags/v${SIRIUS_VERSION}")
endif()
configure_file("${PROJECT_SOURCE_DIR}/src/core/version.hpp.in"
               "${PROJECT_BINARY_DIR}/src/core/version.hpp"
               @ONLY)

# generate schema
file(READ "${PROJECT_SOURCE_DIR}/src/context/input_schema.json" SIRIUS_INPUT_SCHEMA NEWLINE_CONSUME)
configure_file("${PROJECT_SOURCE_DIR}/src/context/input_schema.hpp.in"
               "${PROJECT_BINARY_DIR}/src/context/input_schema.hpp"
               @ONLY)

# generate atomic configurations
file(READ "${PROJECT_SOURCE_DIR}/apps/atoms/atoms.json" SIRIUS_ATOMIC_CONF NEWLINE_CONSUME)
configure_file("${PROJECT_SOURCE_DIR}/apps/atoms/atomic_conf.hpp.in"
               "${PROJECT_BINARY_DIR}/src/unit_cell/atomic_conf.hpp"
               @ONLY)

if(SIRIUS_CREATE_FORTRAN_BINDINGS)
  write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/siriusConfigVersion.cmake"
    VERSION "${SIRIUS_VERSION}"
    COMPATIBILITY SameMajorVersion)
  # write siriusConfig.cmake for F90 API
  configure_package_config_file("${PROJECT_SOURCE_DIR}/cmake/siriusConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/siriusConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sirius)
  install(FILES "${PROJECT_BINARY_DIR}/siriusConfig.cmake"
                "${PROJECT_BINARY_DIR}/siriusConfigVersion.cmake"
                DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/sirius")
endif()
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/sirius_cxxConfigVersion.cmake"
  VERSION "${SIRIUS_VERSION}"
  COMPATIBILITY SameMajorVersion)

# install targets and modules
include(cmake/env_vars_map.cmake)
# write sirius_cxxConfig for C++ API
configure_file("${PROJECT_SOURCE_DIR}/cmake/sirius_cxxConfig.cmake.in"
              "${PROJECT_BINARY_DIR}/sirius_cxxConfig.cmake"
               @ONLY)
install(FILES "${PROJECT_BINARY_DIR}/sirius_cxxConfig.cmake"
              "${PROJECT_BINARY_DIR}/sirius_cxxConfigVersion.cmake"
              "${PROJECT_SOURCE_DIR}/cmake/cudalibs_target.cmake"
              "${PROJECT_SOURCE_DIR}/cmake/hdf5_target.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/sirius_cxx")
install(DIRECTORY "${PROJECT_SOURCE_DIR}/cmake/modules"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/sirius_cxx"
        FILES_MATCHING PATTERN "*.cmake")

# sirius library
add_subdirectory(src)

if(SIRIUS_CREATE_PYTHON_MODULE)
  # Python module
  add_subdirectory(python_module)
endif(SIRIUS_CREATE_PYTHON_MODULE)

# applications
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
  add_subdirectory(apps/tests)
  add_subdirectory(apps/unit_tests)
endif(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)

if(SIRIUS_BUILD_APPS)
  add_subdirectory(apps/atoms)
  add_subdirectory(apps/hydrogen)
  add_subdirectory(apps/mini_app)
  if(SIRIUS_USE_NLCGLIB)
    add_subdirectory(apps/nlcg)
  endif()
  add_subdirectory(apps/upf)
  add_subdirectory(apps/utils)
endif(SIRIUS_BUILD_APPS)
add_subdirectory(doc)
