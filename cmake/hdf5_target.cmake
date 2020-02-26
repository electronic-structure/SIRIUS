if(NOT TARGET sirius::hdf5)
  add_library(sirius::hdf5 INTERFACE IMPORTED)
  set_target_properties(sirius::hdf5 PROPERTIES
                                     INTERFACE_INCLUDE_DIRECTORIES "${HDF5_INCLUDE_DIR}"
                                     INTERFACE_LINK_LIBRARIES "${HDF5_C_LIBRARIES}")
endif()
