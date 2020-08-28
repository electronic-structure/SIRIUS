cmake_minimum_required(VERSION 3.12)

project(SIRIUS VERSION 7.0.0)

# user variables
set(CREATE_PYTHON_MODULE OFF CACHE BOOL "create sirius Python module")
set(CREATE_FORTRAN_BINDINGS ON CACHE BOOL "build Fortran bindings")
set(BUILD_DOCS OFF CACHE BOOL "build doxygen doc")
set(USE_ELPA OFF CACHE BOOL "use scalapack")
set(USE_MAGMA OFF CACHE BOOL "use MAGMA")
set(USE_NLCGLIB OFF CACHE BOOL "enable nlcglib")
set(USE_CUDA OFF CACHE BOOL "use CUDA")
set(USE_ROCM OFF CACHE BOOL "use ROCM AMD GPU code")
set(USE_NVTX OFF CACHE BOOL "use Nvidia profiling tools library")
set(USE_VDWXC OFF CACHE BOOL "use libvdwxc for van der Walls corrections")
set(USE_MKL OFF CACHE BOOL "use Intel MKL")
set(USE_CRAY_LIBSCI OFF CACHE BOOL "use LAPACK/SCALAPACK from Cray LIBSCI")
set(USE_SCALAPACK OFF CACHE BOOL "use scalapack")
set(CUDA_ARCH "none" CACHE STRING "The target GPU architectures; 60 (Pascal), 70 (Volta), etc. Multiple archs are supported too.")
set(BUILD_APPS ON CACHE BOOL "build apps")
set(DEBUG_MEMORY_POOL OFF CACHE BOOL "explicit debugging of memory pool")
set(USE_OPENMP ON CACHE BOOL "use OpenMP")
set(PYTHON2 OFF CACHE STRING "Use Python 2.7")
set(USE_PROFILER ON CACHE BOOL "measure execution of functions with timer")
set(USE_MEMORY_POOL ON CACHE BOOL "use memory pool")
set(GPU_MEMORY_ALIGMENT "512" CACHE STRING "memory alignment of the GPU")
set(BUILD_TESTING OFF CACHE BOOL "Build test executables") # override default setting in CTest module

# set language and standard

if(CREATE_FORTRAN_BINDINGS)
  enable_language(CXX Fortran)
else()
  enable_language(CXX)
endif()

set(CMAKE_CXX_STANDARD 14)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
endif()

if(USE_CUDA AND USE_ROCM)
  message(FATAL_ERROR "USE_CUDA and USE_ROCM must not be enabled at the same time!")
endif()

add_compile_definitions(__GPU_MEMORY_ALIGMENT=${GPU_MEMORY_ALIGMENT})

# if(USE_MKL AND NOT (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel"))
#   message(FATAL_ERROR "Unsupported compiler")
# endif()

set_property(CACHE CUDA_ARCH PROPERTY STRINGS "none" "30" "32" "35" "37"
  "50" "52" "53" "60" "61" "62" "70" "72" "75" "80")

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")
list(APPEND CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH})


include(GitSubmodule)    # include custom defined FindPackage modules
include(GNUInstallDirs)  # required to get a proper LIBDIR variable

if (CREATE_PYTHON_MODULE)
  if(PYTHON2)
    # force cmake to use python2
    set(PYBIND11_PYTHON_VERSION 2.7)
    find_package(Python2 REQUIRED)
    set(PYTHON_EXECUTABLE ${Python2_EXECUTABLE})
  else()
    find_package(Python3 REQUIRED)
    set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
  endif()
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
find_package(LibXC 3.0.0 REQUIRED)
find_package(LibSPG REQUIRED)
find_package(HDF5 REQUIRED C)
include(cmake/hdf5_target.cmake)
find_package(SpFFT 0.9.6 CONFIG REQUIRED)

# Either find std::filesystem or boost::filesystem
find_package(Filesystem COMPONENTS Final Experimental)

add_library(sirius::filesystem INTERFACE IMPORTED)
if(TARGET std::filesystem)
  target_link_libraries(sirius::filesystem INTERFACE std::filesystem)
  target_compile_definitions(sirius::filesystem INTERFACE
    $<$<BOOL:${Filesystem_FOUND}>:__STD_FILESYSTEM>
    $<$<BOOL:${CXX_FILESYSTEM_IS_EXPERIMENTAL}>:__STD_FILESYSTEM_EXPERIMENTAL>
  )
else()
  find_package(Boost 1.65 REQUIRED COMPONENTS filesystem)
  target_link_libraries(sirius::filesystem INTERFACE Boost::filesystem)
  target_compile_definitions(sirius::filesystem INTERFACE __BOOST_FILESYSTEM)
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

if(USE_CUDA)
  enable_language(CUDA)
  find_package(CUDA)
  include(cmake/cudalibs_target.cmake)
  # In CMake 3.18+ we should be using the CUDA_ARCHITECTURES target property
  # instead of setting compiler by hand
  if(${CMAKE_VERSION} VERSION_LESS "3.18.0")
    foreach(arch ${CUDA_ARCH})
      if (NOT ${arch} STREQUAL "none")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${arch},code=compute_${arch}")
      endif()
    endforeach()
  endif()
endif(USE_CUDA)

if(USE_ROCM)
  if(NOT HIP_HCC_FLAGS)
    message(STATUS "Using default AMD gpu targets: gfx803, gfx900, gfx906. Set HIP_HCC_FLAGS to override.")
    set(HIP_HCC_FLAGS ${HIP_HCC_FLAGS} --amdgpu-target=gfx803 --amdgpu-target=gfx900 --amdgpu-target=gfx906)
  endif()
  find_package(HIP REQUIRED)
  # rocblas and hip have cmake config files, but add incompatible flags for mixed compiler usage, so we use custom find modules
  find_package(ROCBLAS REQUIRED)
  find_package(HIPLIBS REQUIRED)
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

# generate atomic configurations
file(STRINGS "${PROJECT_SOURCE_DIR}/apps/atoms/atoms.json"
              SIRIUS_ATOMIC_CONF NEWLINE_CONSUME)
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

# Python module
add_subdirectory(python_module)

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
