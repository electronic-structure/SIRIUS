################################################################################
#
# \file      cmake/FindMKL.cmake
# \author    J. Bakosi
# \copyright 2012-2015, Jozsef Bakosi, 2016, Los Alamos National Security, LLC.
# \brief     Find the Math Kernel Library from Intel
# \date      Thu 26 Jan 2017 02:05:50 PM MST
# downloaded from: https://gist.github.com/scivision/5108cf6ab1515f581a84cd9ad1ef72aa
# modified by: Simon Pintarelli <simon.pintarelli@cscs.ch>
#
################################################################################

# Find the Math Kernel Library from Intel
#
#  MKL_FOUND - System has MKL
#  MKL_INCLUDE_DIRS - MKL include files directories
#  MKL_LIBRARIES - The MKL libraries
#  MKL_INTERFACE_LIBRARY - MKL interface library
#  MKL_SEQUENTIAL_LAYER_LIBRARY - MKL sequential layer library
#  MKL_CORE_LIBRARY - MKL core library
#
#  The environment variables MKLROOT and INTEL are used to find the library.
#  Everything else is ignored. If MKL is found "" is added to
#  CMAKE_C_FLAGS and CMAKE_CXX_FLAGS.
#
#  Example usage:
#
#  find_package(MKL)
#  if(MKL_FOUND)
#    target_link_libraries(TARGET ${MKL_LIBRARIES})
#  endif()


include(FindPackageHandleStandardArgs)

# If already in cache, be silent
if (MKL_INCLUDE_DIRS AND MKL_LIBRARIES AND MKL_INTERFACE_LIBRARY AND
    MKL_SEQUENTIAL_LAYER_LIBRARY AND MKL_CORE_LIBRARY)
  set (MKL_FIND_QUIETLY TRUE)
endif()

if(CMAKE_FIND_LIBRARY_SUFFIXES STREQUAL ".a")
  message(FATAL_ERROR "Attempting to find static MKL libraries. SIRIUS supports only shared linking against MKL. NOTE: On Cray systems you must set `CRAYPE_LINK_TYPE=dynamic`.")
endif()

if(NOT USE_MKL_SHARED_LIBS)
  set(INT_LIB "libmkl_intel_lp64.a")
  set(SEQ_LIB "libmkl_sequential.a")
  if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(THR_LIB "libmkl_intel_thread.a")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(THR_LIB "libmkl_gnu_thread.a")
  else()
    message(FATAL_ERROR "FindMKL: Unknown compiler")
  endif()
  set(COR_LIB "libmkl_core.a")
  set(SCA_LIB "libmkl_scalapack_lp64.a")
  set(BLACS_LIB "libmkl_blacs_intelmpi_lp64.a")
else()
  message("MKL using shared libs!")
  set(INT_LIB "libmkl_intel_lp64.so")
  set(SEQ_LIB "libmkl_sequential.so")
  if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(THR_LIB "libmkl_intel_thread.so")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(THR_LIB "libmkl_gnu_thread.so")
  else()
    message(FATAL_ERROR "FindMKL: Unknown compiler")
  endif()
  set(COR_LIB "libmkl_core.so")
  set(SCA_LIB "libmkl_scalapack_lp64.so")
  set(BLACS_LIB "libmkl_blacs_intelmpi_lp64.so")
  set(DEF_LIB "libmkl_def.so")
endif()

find_path(MKL_INCLUDE_DIR NAMES mkl.h HINTS $ENV{MKLROOT}/include)

find_library(MKL_INTERFACE_LIBRARY
  NAMES ${INT_LIB}
  PATHS $ENV{MKLROOT}/lib
  $ENV{MKLROOT}/lib/intel64
  $ENV{INTEL}/mkl/lib/intel64
  NO_DEFAULT_PATH)

find_library(MKL_SEQUENTIAL_LAYER_LIBRARY
  NAMES ${SEQ_LIB}
  PATHS $ENV{MKLROOT}/lib
  $ENV{MKLROOT}/lib/intel64
  $ENV{INTEL}/mkl/lib/intel64
  NO_DEFAULT_PATH)

find_library(MKL_THREAD_LIBRARY
  NAMES ${THR_LIB}
  PATHS $ENV{MKLROOT}/lib
  $ENV{MKLROOT}/lib/intel64
  $ENV{INTEL}/mkl/lib/intel64
  NO_DEFAULT_PATH)

find_library(MKL_CORE_LIBRARY
  NAMES ${COR_LIB}
  PATHS $ENV{MKLROOT}/lib
  $ENV{MKLROOT}/lib/intel64
  $ENV{INTEL}/mkl/lib/intel64
  NO_DEFAULT_PATH)

find_library(MKL_SCALAPACK_LIBRARY
  NAMES ${SCA_LIB}
  PATHS $ENV{MKLROOT}/lib
  $ENV{MKLROOT}/lib/intel64
  $ENV{INTEL}/mkl/lib/intel64
  NO_DEFAULT_PATH)

find_library(MKL_BLACS_LIBRARY
  NAMES ${BLACS_LIB}
  PATHS $ENV{MKLROOT}/lib
  $ENV{MKLROOT}/lib/intel64
  $ENV{INTEL}/mkl/lib/intel64
  NO_DEFAULT_PATH)

find_library(MKL_DEF_LIBRARY
  NAMES ${DEF_LIB}
  PATHS $ENV{MKLROOT}/lib
  $ENV{MKLROOT}/lib/intel64
  $ENV{INTEL}/mkl/lib/intel64
  NO_DEFAULT_PATH)

set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
# TODO: decide when to use MKL_SEQUENTIAL_LAYER_LIBRARY / MKL_THREAD_LIBRARY
# set(MKL_LIBRARIES "${MKL_INTERFACE_LIBRARY} ${MKL_SEQUENTIAL_LAYER_LIBRARY} ${MKL_CORE_LIBRARY}")
set(MKL_LIBRARIES "${MKL_CORE_LIBRARY};${MKL_BLACS_LIBRARY};${MKL_SCALAPACK_LIBRARY};${MKL_INTERFACE_LIBRARY};${MKL_THREAD_LIBRARY}")



if (MKL_INCLUDE_DIR AND
    MKL_INTERFACE_LIBRARY AND
    MKL_SEQUENTIAL_LAYER_LIBRARY AND
    MKL_CORE_LIBRARY)

  if (NOT DEFINED ENV{CRAY_PRGENVPGI} AND
      NOT DEFINED ENV{CRAY_PRGENVGNU} AND
      NOT DEFINED ENV{CRAY_PRGENVCRAY} AND
      NOT DEFINED ENV{CRAY_PRGENVINTEL})
    set(ABI "-m64")
  endif()
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ABI}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ABI}")
else()
  set(MKL_INCLUDE_DIRS "")
  set(MKL_LIBRARIES "")
  set(MKL_INTERFACE_LIBRARY "")
  set(MKL_SEQUENTIAL_LAYER_LIBRARY "")
  set(MKL_CORE_LIBRARY "")
endif()

# Handle the QUIETLY and REQUIRED arguments and set MKL_FOUND to TRUE if
# all listed variables are TRUE.
FIND_PACKAGE_HANDLE_STANDARD_ARGS(MKL DEFAULT_MSG
  MKL_LIBRARIES MKL_SCALAPACK_LIBRARY MKL_INCLUDE_DIRS MKL_INTERFACE_LIBRARY MKL_SEQUENTIAL_LAYER_LIBRARY MKL_CORE_LIBRARY MKL_DEF_LIBRARY)
MARK_AS_ADVANCED(MKL_INCLUDE_DIRS MKL_LIBRARIES MKL_SCALAPACK_LIBRARY MKL_INTERFACE_LIBRARY MKL_SEQUENTIAL_LAYER_LIBRARY MKL_CORE_LIBRARY MKL_DEF_LIBRARY)

if(MKL_FOUND AND NOT TARGET sirius::mkl)
  add_library(sirius::mkl INTERFACE IMPORTED)
  set_target_properties(sirius::mkl PROPERTIES
                                    INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR}"
                                    INTERFACE_LINK_LIBRARIES "${MKL_LIBRARIES}")
endif()
