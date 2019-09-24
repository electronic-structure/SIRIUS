cmake_minimum_required(VERSION 3.9) # Intel C++11 support starts from 3.6 or even later version
project(SIRIUS)

if(POLICY CMP0074)
  # allow {module}_ROOT variables to be set
  cmake_policy(SET CMP0074 NEW)
endif()

# set language and standard
enable_language(CXX Fortran)
set(CMAKE_CXX_STANDARD 11)

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

set_property(CACHE GPU_MODEL PROPERTY STRINGS "none" "P100" "V100" "G10x0")

# include custom defined FindPackage modules
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")
include(GitSubmodule)
list(APPEND CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH})

if(PYTHON2)
  # force cmake to use python2
  set(PYBIND11_PYTHON_VERSION 2.7)
  find_package(Python2 REQUIRED)
  set(PYTHON_EXECUTABLE ${Python2_EXECUTABLE})
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

# generate compile_commands.json with compile commands for each target
set(CMAKE_EXPORT_COMPILE_COMMANDS "YES")

if(DEBUG_MEMORY_POOL)
  add_definitions("-D__DEBUG_MEMORY_POOL")
endif()

if(USE_SCALAPACK)
  add_definitions("-D__SCALAPACK")
endif()

if (USE_TIMER)
  add_definitions("-D__USE_TIMER")
  add_definitions("-D__PROFILE")
  add_definitions("-D__PROFILE_TIME")
endif()

find_package(MPI REQUIRED)
find_package(GSL REQUIRED)
set(SYSTEM_LIBRARIES "${SYSTEM_LIBRARIES};${GSL_LIBRARY}")

if(USE_MKL)
  set(USE_MKL_SHARED_LIBS On) # link against shared MKL libraries
  find_package(MKL REQUIRED)
  include_directories(BEFORE ${MKL_INCLUDE_DIR})
elseif(USE_CRAY_LIBSCI)
  if(USE_MKL)
    message(FATAL_ERROR "CANNOT USE_CRAY_LIBSCI _AND_ USE_MKL, use one of them.")
  endif()
  find_package(CRAY_LIBSCI REQUIRED)
  set(SYSTEM_LIBRARIES "${SYSTEM_LIBRARIES};${CRAY_LIBSCI_LIBRARIES}")
else()
  find_package(LAPACK REQUIRED)
  if(USE_SCALAPACK)
    find_package(SCALAPACK REQUIRED) # just sets scalapack_DIR
    #include_directories(BEFORE ${SCALAPACK_INCLUDE_DIR})
    set(SYSTEM_LIBRARIES "${SYSTEM_LIBRARIES};${SCALAPACK_LIBRARIES}")
  endif()
endif()

find_package(FFTW REQUIRED)
if (USE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()
find_package(LibXC 3.0.0 REQUIRED)
find_package(LibSPG REQUIRED)
find_package(HDF5 REQUIRED C HL)


if(USE_ELPA)
  find_package(Elpa REQUIRED)
  include_directories(BEFORE SYSTEM "${ELPA_INCLUDE_DIR}")
  set(SYSTEM_LIBRARIES "${SYSTEM_LIBRARIES};${ELPA_LIBRARIES}")
  add_definitions("-D__ELPA")
endif(USE_ELPA)

if(USE_CUDA)
  enable_language(CUDA)
  find_package(CUDA)
  add_definitions("-D__GPU")
  add_definitions("-D__CUDA")
  include_directories(BEFORE SYSTEM ${CUDA_INCLUDE_DIRS})
  set(SYSTEM_LIBRARIES "${SYSTEM_LIBRARIES};${CUDA_LIBRARIES};${CUDA_CUBLAS_LIBRARIES};${CUDA_CUFFT_LIBRARIES};${CUDA_cusolver_LIBRARY}")

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

  if(USE_NVTX)
    add_definitions("-D__CUDA_NVTX")
    set(SYSTEM_LIBRARIES "${SYSTEM_LIBRARIES};-lnvToolsExt")
  endif()
endif(USE_CUDA)

if(USE_MAGMA)
  if(NOT USE_CUDA)
    message(FATAL_ERROR "MAGMA depends on Cuda, must enable Cuda or disable MAGMA")
  endif(NOT USE_CUDA)
  find_package(MAGMA)
  add_definitions("-D__MAGMA")
  include_directories(BEFORE SYSTEM ${MAGMA_INCLUDE_DIR})
  set(SYSTEM_LIBRARIES "${SYSTEM_LIBRARIES};${MAGMA_LIBRARIES}")
endif(USE_MAGMA)

if(USE_CUDA AND USE_ROCM)
  message(FATAL_ERROR "USE_CUDA and USE_ROCM must not be enabled at the same time!")
endif()

if(USE_ROCM)
    message(STATUS "WARNING: ROCM enabled, prototype feature! Only limited functionality available.")
    find_package(ROCM COMPONENTS rocfft hipblas)
    if(NOT ${ROCM_HIP_PLATFORM} STREQUAL hcc)
  message(FATAL_ERROR "Compilation on Nvidia platform not supported with ROCM enabled!")
    endif()
    add_definitions("-D__GPU")
    add_definitions("-D__ROCM")
    add_definitions(${ROCM_DEFINITIONS})
    include_directories(${ROCM_INCLUDE_DIRS})
    add_subdirectory(src/SDDK/GPU/hipblas_port)
    include_directories(src/SDDK/GPU/hipblas_port)
endif()

# add required libraries
set(SYSTEM_LIBRARIES "${SYSTEM_LIBRARIES};${MPI_CXX_LIBRARIES};${LIBXC_LIBRARIES};${LIBSPG_LIBRARIES}")


# include library headers
include_directories(BEFORE SYSTEM ${FFTW_INCLUDE_DIR})
include_directories(BEFORE SYSTEM ${LIBXC_INCLUDE_DIR})
include_directories(BEFORE SYSTEM ${LIBSPG_INCLUDE_DIR})
include_directories(BEFORE SYSTEM ${HDF5_INCLUDE_DIR})
include_directories(BEFORE SYSTEM ${MPI_C_INCLUDE_DIRS})
include_directories(BEFORE SYSTEM ${GSL_INCLUDE_DIRS})

if(USE_VDWXC)
  find_package(LibVDWXC 0.3.0 REQUIRED)
  include_directories(${LIBVDWXC_INCLUDE_DIR})
  set(SYSTEM_LIBRARIES "${SYSTEM_LIBRARIES};${LIBVDWXC_LIBRARIES}")
  add_definitions("-DUSE_VDWXC")
endif(USE_VDWXC)

# project header locations
include_directories(BEFORE src)
include_directories(BEFORE src/SDDK)

# configure generation of the version header
add_custom_command(
  OUTPUT _always_rebuild
  COMMAND true
  )

if(NOT PYTHON2)
  # attempt to find python3
  find_package(PythonInterp REQUIRED)
endif()

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

include_directories(BEFORE ${PROJECT_BINARY_DIR}/src)

MACRO(SIRIUS_SETUP_TARGET _target)

  add_dependencies(${_target} sirius)
  add_dependencies(${_target} generate_version_hpp)
  add_dependencies(${_target} runtime_options_json_hpp)

  target_link_libraries(${_target} PRIVATE "${sirius_location};${SYSTEM_LIBRARIES};${HDF5_C_LIBRARIES};${HDF5_C_HL_LIBRARIES}")
  if(USE_MKL)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
      target_link_libraries(${_target} PRIVATE "-Wl,--no-as-needed;${MKL_LIBRARIES}")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
      target_link_libraries(${_target} PRIVATE "${MKL_LIBRARIES}")
    else()
      message(FATAL_ERROR "Unsupported compiler")
    endif()
  else()
    target_link_libraries(${_target} PRIVATE "${LAPACK_LIBRARIES};${FFTW_LIBRARIES}")
  endif()

  if(USE_ROCM)
    target_link_libraries(${_target} PRIVATE sirius_rocm sirius_rocm_interface hipblas_port ${ROCM_LIBRARIES})
  endif()

  if(OpenMP_CXX_FOUND)
    target_link_libraries(${_target} PUBLIC OpenMP::OpenMP_CXX)
  endif()

  if(HAVE_LIBVDW_WITH_MPI)
    target_compile_definitions(${_target} PUBLIC -D__HAVE_VDWXC_MPI)
  endif()

  # target_link_libraries(${_target} PRIVATE GSL::gsl)

  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_link_libraries(${_target} PRIVATE "-lgfortran")
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
