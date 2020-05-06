#  Copyright (c) 2019 ETH Zurich
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.


#.rst:
# FindHIPLIBS
# -----------
#
# This module searches for the fftw3 library.
#
# The following variables are set
#
# ::
#
#   HIPLIBS_FOUND           - True if hiplibs is found
#   HIPLIBS_LIBRARIES       - The required libraries
#   HIPLIBS_INCLUDE_DIRS    - The required include directory
#
# The following import target is created
#
# ::
#
#   HIPLIBS::hiplibs

#set paths to look for library from ROOT variables.If new policy is set, find_library() automatically uses them.
if(NOT POLICY CMP0074)
    set(_HIPLIBS_PATHS ${HIPLIBS_ROOT} $ENV{HIPLIBS_ROOT})
endif()

if(NOT _HIPLIBS_PATHS)
    set(_HIPLIBS_PATHS /opt/rocm)
endif()

find_path(
    HIPLIBS_INCLUDE_DIRS
    NAMES "hip/hip_runtime_api.h"
    HINTS ${_HIPLIBS_PATHS}
    PATH_SUFFIXES "hip/include" "include"
)
find_library(
    HIPLIBS_LIBRARIES
    NAMES "hip_hcc"
    HINTS ${_ROCBLAS_PATHS}
    PATH_SUFFIXES "hip/lib" "lib" "lib64" 
)
find_path(
    HSA_INCLUDE_DIRS
    NAMES "hsa/hsa.h"
    HINTS ${_HIPLIBS_PATHS}
    PATH_SUFFIXES "hip/include" "include"
)

# check if found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HIPLIBS REQUIRED_VARS HIPLIBS_INCLUDE_DIRS HSA_INCLUDE_DIRS HIPLIBS_LIBRARIES)

list(APPEND HIPLIBS_INCLUDE_DIRS ${HSA_INCLUDE_DIRS})

# add target to link against
if(HIPLIBS_FOUND)
    if(NOT TARGET HIPLIBS::hiplibs)
        add_library(HIPLIBS::hiplibs INTERFACE IMPORTED)
    endif()
    set_property(TARGET HIPLIBS::hiplibs PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${HIPLIBS_INCLUDE_DIRS})
    set_property(TARGET HIPLIBS::hiplibs PROPERTY INTERFACE_LINK_LIBRARIES ${HIPLIBS_LIBRARIES})
endif()

# prevent clutter in cache
MARK_AS_ADVANCED(HIPLIBS_FOUND HIPLIBS_LIBRARIES HIPLIBS_INCLUDE_DIRS)
