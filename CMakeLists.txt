cmake_minimum_required(VERSION 3.12)

project(SIRIUS)

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
set(USE_TIMER ON CACHE BOOL "measure execution of functions with timer")
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

# preserve rpaths when installing
# https://spack.readthedocs.io/en/latest/workflows.html#write-the-cmake-build
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(SIRIUS_RPATH_DIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES ${SIRIUS_RPATH_DIR} is_sys_dir)
if("${is_sys_dir}" STREQUAL "-1")
  # set only if installed in outside of a system directory
  set(CMAKE_INSTALL_RPATH ${SIRIUS_RPATH_DIR})
endif()

# build shared libraries by default
option(BUILD_SHARED_LIBS "Build shared libraries (.dll/.so) instead of static ones (.lib/.a)" ON)

# generate compile_commands.json with compile commands for each target
set(CMAKE_EXPORT_COMPILE_COMMANDS "YES")

# dependencies
find_package(MPI REQUIRED)
find_package(GSL REQUIRED)
find_package(LibXC 3.0.0 REQUIRED)
find_package(LibSPG REQUIRED)
find_package(HDF5 REQUIRED C)
find_package(SpFFT 0.9.6 CONFIG REQUIRED)

if (USE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

set(LINALG_LIB "")
if(USE_MKL)
  set(USE_MKL_SHARED_LIBS On) # link against shared MKL libraries
  find_package(MKL REQUIRED)
  set(LINALG_LIB "mkl::mkl")
elseif(USE_CRAY_LIBSCI)
  find_package(CRAY_LIBSCI REQUIRED)
  set(LINALG_TGT "${CRAY_LIBSCI_LIBRARIES}")
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

# external library targets
if(NOT TARGET hdf5::hdf5)
  add_library(hdf5::hdf5 INTERFACE IMPORTED)
  set_target_properties(hdf5::hdf5 PROPERTIES 
                                   INTERFACE_INCLUDE_DIRECTORIES "${HDF5_INCLUDE_DIR}"
                                   INTERFACE_LINK_LIBRARIES "${HDF5_C_LIBRARIES}")
endif()

if (USE_CUDA AND NOT TARGET cuda::cudalibs)
  add_library(cuda::cudalibs INTERFACE IMPORTED)
  set_target_properties(cuda::cudalibs PROPERTIES 
                                       INTERFACE_INCLUDE_DIRECTORIES "${CUDA_INCLUDE_DIRS}"
                                       INTERFACE_LINK_LIBRARIES "${CUDA_LIBRARIES};${CUDA_CUBLAS_LIBRARIES};${CUDA_CUFFT_LIBRARIES};${CUDA_cusolver_LIBRARY}")
endif()

# configure generation of the version header
add_custom_command(
  OUTPUT _always_rebuild
  COMMAND true
  )

# handle the generation of the version.hpp file
add_custom_command(
  OUTPUT version.hpp-test
  DEPENDS _always_rebuild
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/make_version_hpp.py ${CMAKE_SOURCE_DIR}/VERSION > version.hpp-test
  )
set(version_hpp_path src/version.hpp)
add_custom_command(
  OUTPUT ${version_hpp_path}
  DEPENDS version.hpp-test
  COMMAND ${CMAKE_COMMAND} -E copy_if_different version.hpp-test ${version_hpp_path}
  )

add_custom_target(generate_version_hpp DEPENDS ${version_hpp_path})

# handle the generation of runtime_options_json.hpp file
add_custom_command(
  OUTPUT runtime_options_json.hpp-test
  DEPENDS _always_rebuild
  COMMAND ${CMAKE_SOURCE_DIR}/generate_options_header_file.sh $(CMAKE_SOURCE_DIR)/src/options.json > runtime_options_json.hpp-test
  )

set(runtime_options_json_path src/runtime_options_json.hpp)
add_custom_command(
  OUTPUT ${runtime_options_json_path}-test
  DEPENDS _always_rebuild
  COMMAND ${CMAKE_SOURCE_DIR}/generate_options_header_file.sh ${CMAKE_SOURCE_DIR}/src/options.json > ${runtime_options_json_path}-test
  )
add_custom_command(
  OUTPUT ${runtime_options_json_path}
  DEPENDS ${runtime_options_json_path}-test
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${runtime_options_json_path}-test ${runtime_options_json_path}
  )

add_custom_target(runtime_options_json_hpp DEPENDS ${runtime_options_json_path})

MACRO(SIRIUS_SETUP_TARGET _target)

  add_dependencies(${_target} generate_version_hpp)
  add_dependencies(${_target} runtime_options_json_hpp)

  target_link_libraries(${_target} PRIVATE sirius)

  if(USE_ROCM)
    target_link_libraries(${_target} PRIVATE sirius_rocm sirius_rocm_interface hipblas_port ${ROCM_LIBRARIES})
  endif()

ENDMACRO()

# sirius library
add_subdirectory(src)
set(sirius_location $<TARGET_FILE:sirius>)

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
