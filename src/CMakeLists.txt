# workaround for set_target_properties(... POSITION_INDEPENDENT_CODE ON) not
# working correctly
# list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")

if(USE_CUDA)
  file(GLOB_RECURSE CUFILES_SDDK "SDDK/*.cu")
  file(GLOB_RECURSE CUFILES_KERNELS "Kernels/*.cu")
  add_library(sirius "sirius_api.cpp;${CUFILES_KERNELS};${CUFILES_SDDK}")
else()
  add_library(sirius "sirius_api.cpp")
endif()

set_target_properties(sirius PROPERTIES POSITION_INDEPENDENT_CODE ON
  #CUDA_SEPARABLE_COMPILATION ON
  )
