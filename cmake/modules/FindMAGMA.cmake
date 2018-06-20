include(FindPackageHandleStandardArgs)

find_path(MAGMA_INCLUDE_DIR magmablas.h
  HINTS
  ENV EBROOTMAGMA
  ENV MAGMA_DIR
  ENV MAGMAROOT
  PATH_SUFFIXES include magma/include
  )

find_library(MAGMA_LIBRARIES NAMES magma magma_sparse
  HINTS
  ENV EBROOTMAGMA
  ENV MAGMA_DIR
  ENV MAGMAROOT
  PATH_SUFFIXES lib magma/lib
  )


find_package_handle_standard_args(MAGMA DEFAULT_MSG MAGMA_INCLUDE_DIR MAGMA_LIBRARIES)
mark_as_advanced(MAGMA_FOUND MAGMA_INCLUDE_DIR MAGMA_LIBRARIES)
