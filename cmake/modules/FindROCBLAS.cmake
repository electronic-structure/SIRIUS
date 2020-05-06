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
# FindROCBLAS
# -----------
#
# This module searches for the fftw3 library.
#
# The following variables are set
#
# ::
#
#   ROCBLAS_FOUND           - True if rocblas is found
#   ROCBLAS_LIBRARIES       - The required libraries
#   ROCBLAS_INCLUDE_DIRS    - The required include directory
#
# The following import target is created
#
# ::
#
#   ROCBLAS::rocblas

#set paths to look for library from ROOT variables.If new policy is set, find_library() automatically uses them.
if(NOT POLICY CMP0074)
    set(_ROCBLAS_PATHS ${ROCBLAS_ROOT} $ENV{ROCBLAS_ROOT})
endif()

if(NOT _ROCBLAS_PATHS)
    set(_ROCBLAS_PATHS /opt/rocm)
endif()

find_library(
    ROCBLAS_LIBRARIES
    NAMES "rocblas"
    HINTS ${_ROCBLAS_PATHS}
    PATH_SUFFIXES "rocblas/lib" "rocblas" 
)
find_path(
    ROCBLAS_INCLUDE_DIRS
    NAMES "rocblas.h"
    HINTS ${_ROCBLAS_PATHS}
    PATH_SUFFIXES "rocblas/include" "include"
)

# check if found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ROCBLAS REQUIRED_VARS ROCBLAS_INCLUDE_DIRS ROCBLAS_LIBRARIES )

# add target to link against
if(ROCBLAS_FOUND)
    if(NOT TARGET ROCBLAS::rocblas)
        add_library(ROCBLAS::rocblas INTERFACE IMPORTED)
    endif()
    set_property(TARGET ROCBLAS::rocblas PROPERTY INTERFACE_LINK_LIBRARIES ${ROCBLAS_LIBRARIES})
    set_property(TARGET ROCBLAS::rocblas PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${ROCBLAS_INCLUDE_DIRS})
endif()

# prevent clutter in cache
MARK_AS_ADVANCED(ROCBLAS_FOUND ROCBLAS_LIBRARIES ROCBLAS_INCLUDE_DIRS)
