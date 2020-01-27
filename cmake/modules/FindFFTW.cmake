include(FindPackageHandleStandardArgs)

find_path(FFTW_INCLUDE_DIR
  NAMES fftw3.h
  PATH_SUFFIXES include include/fftw
  HINTS ENV MKLROOT
  HINTS ENV FFTWROOT
  HINTS ENV FFTW_INC
  )

find_library(FFTW_LIBRARIES
  NAMES fftw3
  PATH_SUFFIXES lib
  HINTS ENV MKLROOT
  HINTS ENV FFTW_DIR
  HINTS ENV FFTWROOT
  )

set(FFTW_INCLUDE_DIRS ${FFTW_INCLUDE_DIR})

if(FFTW_LIBRARIES MATCHES "NOTFOUND")
  # ok, fftw libraries not found.
  # MKL contains fftw, lets assume we use MKL
  # TODO: handle this properly
  set(FFTW_LIBRARIES "")
  find_package_handle_standard_args(FFTW
    REQUIRED_VARS FFTW_INCLUDE_DIR )
  mark_as_advanced(FFTW_FOUND FFTW_INCLUDE_DIR)
else()
  find_package_handle_standard_args(FFTW
    REQUIRED_VARS FFTW_INCLUDE_DIR FFTW_LIBRARIES)
  mark_as_advanced(FFTW_FOUND FFTW_INCLUDE_DIR FFTW_LIBRARIES)
endif()
