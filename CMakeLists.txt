cmake_minimum_required(VERSION 3.18)

project(SIRIUS VERSION 7.4.4)

# user variables
set(CREATE_PYTHON_MODULE    OFF CACHE BOOL "create sirius Python module")
set(CREATE_FORTRAN_BINDINGS ON  CACHE BOOL "build Fortran bindings")
set(BUILD_DOCS              OFF CACHE BOOL "build doxygen doc")
set(USE_ELPA                OFF CACHE BOOL "use scalapack")
set(USE_MAGMA               OFF CACHE BOOL "use MAGMA")
set(USE_NLCGLIB             OFF CACHE BOOL "enable nlcglib")
set(USE_CUDA                OFF CACHE BOOL "use CUDA")
set(USE_ROCM                OFF CACHE BOOL "use ROCM AMD GPU code")
set(USE_NVTX                OFF CACHE BOOL "use Nvidia profiling tools library")
set(USE_VDWXC               OFF CACHE BOOL "use libvdwxc for van der Walls corrections")
set(USE_MKL                 OFF CACHE BOOL "use Intel MKL")
set(USE_CRAY_LIBSCI         OFF CACHE BOOL "use LAPACK/SCALAPACK from Cray LIBSCI")
set(USE_SCALAPACK           OFF CACHE BOOL "use scalapack")
set(BUILD_APPS              ON  CACHE BOOL "build apps")
set(DEBUG_MEMORY_POOL       OFF CACHE BOOL "explicit debugging of memory pool")
set(USE_OPENMP              ON  CACHE BOOL "use OpenMP")
set(USE_PROFILER            ON  CACHE BOOL "measure execution of functions with timer")
set(USE_MEMORY_POOL         ON  CACHE BOOL "use memory pool")
set(USE_POWER_COUNTER       OFF CACHE BOOL "measure energy consumption with power counters")
set(BUILD_TESTING           OFF CACHE BOOL "build test executables") # override default setting in CTest module
set(GPU_MEMORY_ALIGMENT "512" CACHE STRING "memory alignment of the GPU")
set(USE_FP32 "AUTO" CACHE STRING "Enable single precision support.")
set(USE_VCSQNM              OFF CACHE BOOL "use variable cell stabilized quasi Newton method")

set_property(CACHE USE_FP32 PROPERTY STRINGS "AUTO" "ON" "OFF")

# set language and standard

if(CREATE_FORTRAN_BINDINGS)
  enable_language(CXX Fortran)
else()
  enable_language(CXX)
endif()

set(CMAKE_CXX_STANDARD 14)

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
  set(CMAKE_CUDA_STANDARD 14)
  find_package(CUDA)
  include(cmake/cudalibs_target.cmake)
endif(USE_CUDA)

if(USE_ROCM)
  find_package(hip CONFIG REQUIRED)
  find_package(rocblas CONFIG REQUIRED)

  # FindHIP module provides compilation command for GPU code
  if(NOT HIP_HCC_FLAGS)
    message(STATUS "Using default AMD gpu targets: gfx803, gfx900, gfx906. Set HIP_HCC_FLAGS to override.")
    set(HIP_HCC_FLAGS ${HIP_HCC_FLAGS} --amdgpu-target=gfx803 --amdgpu-target=gfx900 --amdgpu-target=gfx906)
  endif()
  find_package(HIP MODULE REQUIRED)
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
