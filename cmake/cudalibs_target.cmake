if (NOT TARGET sirius::cudalibs)
  add_library(sirius::cudalibs INTERFACE IMPORTED)
  set_target_properties(sirius::cudalibs PROPERTIES
                                         INTERFACE_INCLUDE_DIRECTORIES "${CUDA_INCLUDE_DIRS}"
                                         INTERFACE_LINK_LIBRARIES "${CUDA_LIBRARIES};${CUDA_CUBLAS_LIBRARIES};${CUDA_CUFFT_LIBRARIES};${CUDA_cusolver_LIBRARY}")
endif()
