include(FindPackageHandleStandardArgs)

find_path(SIRIUS_MAGMA_INCLUDE_DIR magmablas.h
  HINTS
  ENV EBROOTMAGMA
  ENV MAGMA_DIR
  ENV MAGMAROOT
  PATH_SUFFIXES include magma/include
  )

find_library(SIRIUS_MAGMA_LIBRARIES NAMES magma magma_sparse
  HINTS
  ENV EBROOTMAGMA
  ENV MAGMA_DIR
  ENV MAGMAROOT
  PATH_SUFFIXES lib magma/lib
  )

find_package_handle_standard_args(MAGMA DEFAULT_MSG SIRIUS_MAGMA_INCLUDE_DIR SIRIUS_MAGMA_LIBRARIES)
mark_as_advanced(MAGMA_FOUND SIRIUS_MAGMA_INCLUDE_DIR SIRIUS_MAGMA_LIBRARIES)

if(MAGMA_FOUND AND NOT TARGET sirius::magma)
  add_library(sirius::magma INTERFACE IMPORTED)
  set_target_properties(sirius::magma PROPERTIES
                                     INTERFACE_INCLUDE_DIRECTORIES "${SIRIUS_MAGMA_INCLUDE_DIR}"
                                     INTERFACE_LINK_LIBRARIES "${SIRIUS_MAGMA_LIBRARIES}")
endif()
