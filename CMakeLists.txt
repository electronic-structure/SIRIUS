cmake_minimum_required(VERSION 3.21)

include(CMakePackageConfigHelpers)

project(SIRIUS
  DESCRIPTION "Domain specific library for electronic structure calculations"
  HOMEPAGE_URL "https://github.com/electronic-structure/SIRIUS"
  VERSION 7.5.0
  LANGUAGES CXX C)

# for CUDA_ARCHITECTURES
if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()

# user variables
option(CREATE_PYTHON_MODULE    "create sirius Python module" OFF)
option(CREATE_FORTRAN_BINDINGS "build Fortran bindings" ON)
option(BUILD_DOCS              "build doxygen doc" OFF)
option(USE_ELPA                "use elpa" OFF)
option(USE_MAGMA               "use MAGMA" OFF)
option(USE_NLCGLIB             "enable nlcglib" OFF)
option(USE_CUDA                "use CUDA" OFF)
option(USE_ROCM                "use ROCM AMD GPU code" OFF)
option(USE_NVTX                "use Nvidia profiling tools library" OFF)
option(USE_VDWXC               "use libvdwxc for van der Walls corrections" OFF)
option(USE_MKL                 "use Intel MKL" OFF)
option(USE_CRAY_LIBSCI         "use LAPACK/SCALAPACK from Cray LIBSCI" OFF)
option(USE_SCALAPACK           "use scalapack" OFF)
option(BUILD_APPS              "build apps" ON)
option(USE_OPENMP              "use OpenMP" ON)
option(USE_PROFILER            "measure execution of functions with timer" ON)
option(USE_MEMORY_POOL         "use memory pool" ON)
option(USE_POWER_COUNTER       "measure energy consumption with power counters" OFF)
option(BUILD_TESTING           "build test executables" OFF) # override default setting in CTest module
option(USE_VCSQNM              "use variable cell stabilized quasi Newton method" OFF)

set(GPU_MEMORY_ALIGMENT "512" CACHE STRING "memory alignment of the GPU")
set(USE_FP32 "AUTO" CACHE STRING "Enable single precision support.")

set_property(CACHE USE_FP32 PROPERTY STRINGS "AUTO" "ON" "OFF")

# set language and standard

if(CREATE_FORTRAN_BINDINGS)
  enable_language(Fortran)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "60;70;80" CACHE STRING "CUDA architectures to generate code for")
endif()

# This ensures that -std=c++14 is passed to nvcc. See
# https://github.com/electronic-structure/SIRIUS/issues/716 for details.
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
endif()

if(USE_CUDA AND USE_ROCM)
  message(FATAL_ERROR "USE_CUDA and USE_ROCM must not be enabled at the same time!")
endif()

add_compile_definitions(SIRIUS_GPU_MEMORY_ALIGMENT=${GPU_MEMORY_ALIGMENT})

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")
list(APPEND CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH})


include(GitSubmodule)    # include custom defined FindPackage modules
include(GNUInstallDirs)  # required to get a proper LIBDIR variable

if (CREATE_PYTHON_MODULE)
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

if(USE_NLCGLIB)
  find_package(nlcglib REQUIRED)
  find_package(Kokkos REQUIRED)
endif()

if(USE_MEMORY_POOL)
  find_package(umpire)
endif()

find_package(MPI REQUIRED)
find_package(GSL REQUIRED)
find_package(LibXC 4.0.0 REQUIRED)
find_package(LibSPG REQUIRED)
find_package(HDF5 REQUIRED C)
include(cmake/hdf5_target.cmake)
find_package(SpFFT 1.0.2 CONFIG REQUIRED)
find_package(SPLA 1.4.0 CONFIG REQUIRED)

# Check if SpFFT support single precision if required
set(USE_FP32_BOOL OFF)
if(NOT USE_FP32 STREQUAL "OFF")
  if(SPFFT_SINGLE_PRECISION)
    set(USE_FP32_BOOL ON)
  elseif(USE_FP32 STREQUAL "ON")
    message(FATAL_ERROR "Single precision option enabled, but SpFFT not compiled with single precision support.")
  endif()
endif()

# Either find std::filesystem or boost::filesystem
find_package(Filesystem COMPONENTS Final Experimental)

add_library(sirius::filesystem INTERFACE IMPORTED)
if(TARGET std::filesystem)
  target_link_libraries(sirius::filesystem INTERFACE std::filesystem)
  target_compile_definitions(sirius::filesystem INTERFACE
    $<$<BOOL:${Filesystem_FOUND}>:SIRIUS_STD_FILESYSTEM>
    $<$<BOOL:${CXX_FILESYSTEM_IS_EXPERIMENTAL}>:SIRIUS_STD_FILESYSTEM_EXPERIMENTAL>
  )
else()
  find_package(Boost 1.65 REQUIRED COMPONENTS filesystem)
  target_link_libraries(sirius::filesystem INTERFACE Boost::filesystem)
  target_compile_definitions(sirius::filesystem INTERFACE SIRIUS_BOOST_FILESYSTEM)
endif()

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


if(USE_WANNIER90)
  #target_link_libraries(sirius::filesystem INTERFACE ifcore)
  find_package(Wannier90 REQUIRED)
endif(USE_WANNIER90)


if(USE_ELPA)
  find_package(Elpa REQUIRED)
endif(USE_ELPA)

if(USE_MAGMA)
  find_package(MAGMA REQUIRED)
endif(USE_MAGMA)

if(USE_VDWXC)
  find_package(LibVDWXC 0.3.0 REQUIRED)
endif(USE_VDWXC)

find_package(costa CONFIG REQUIRED)

if(USE_CUDA)
  enable_language(CUDA)
  # note find cudatoolkit is called inside the include file. the
  # sirius::cuda_libs target is also defined there
  include(cmake/cudalibs_target.cmake)
  message(STATUS "CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}")
endif(USE_CUDA)

if(USE_ROCM)
  # safegaurd.
  if (NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES gfx801 gfx900 gfx90a)
  endif()

  enable_language(HIP)
  find_package(rocblas CONFIG REQUIRED)
  set(CMAKE_HIP_STANDARD 14)
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
configure_file("${PROJECT_SOURCE_DIR}/src/version.hpp.in"
               "${PROJECT_BINARY_DIR}/src/version.hpp"
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

write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/siriusConfigVersion.cmake"
  VERSION "${SIRIUS_VERSION}"
  COMPATIBILITY SameMajorVersion)

# install targets and modules
include(cmake/env_vars_map.cmake)
configure_file("${PROJECT_SOURCE_DIR}/cmake/siriusConfig.cmake.in"
               "${PROJECT_BINARY_DIR}/siriusConfig.cmake"
               @ONLY)
install(FILES "${PROJECT_BINARY_DIR}/siriusConfig.cmake"
              "${PROJECT_BINARY_DIR}/siriusConfigVersion.cmake"
              "${PROJECT_SOURCE_DIR}/cmake/cudalibs_target.cmake"
              "${PROJECT_SOURCE_DIR}/cmake/hdf5_target.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/sirius")
install(DIRECTORY "${PROJECT_SOURCE_DIR}/cmake/modules"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/sirius"
        FILES_MATCHING PATTERN "*.cmake")

# sirius library
add_subdirectory(src)

if(CREATE_PYTHON_MODULE)
  # Python module
  add_subdirectory(python_module)
endif(CREATE_PYTHON_MODULE)

# applications
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
  add_subdirectory(apps/tests)
  add_subdirectory(apps/unit_tests)
endif(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)

if(BUILD_APPS)
  add_subdirectory(apps/atoms)
  add_subdirectory(apps/hydrogen)
  add_subdirectory(apps/dft_loop)
  if(USE_NLCGLIB)
    add_subdirectory(apps/nlcg)
  endif()
  add_subdirectory(apps/upf)
  add_subdirectory(apps/utils)
endif(BUILD_APPS)
add_subdirectory(doc)
