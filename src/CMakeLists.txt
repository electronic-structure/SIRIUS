# workaround for set_target_properties(... POSITION_INDEPENDENT_CODE ON) not
# working correctly
# list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")

if(USE_CUDA)
  file(GLOB_RECURSE CUFILES_SDDK "SDDK/*.cu")
  file(GLOB_RECURSE CUFILES_KERNELS "Kernels/*.cu")
  add_library(sirius "sirius_api.cpp;${CUFILES_KERNELS};${CUFILES_SDDK}")
  add_library(sirius_f "sirius.f90;sirius_api.cpp;${CUFILES_KERNELS};${CUFILES_SDDK}")
else()
  add_library(sirius "sirius_api.cpp")
  add_library(sirius_f "sirius.f90;sirius_api.cpp")
endif()

set_target_properties(sirius PROPERTIES POSITION_INDEPENDENT_CODE ON
  #CUDA_SEPARABLE_COMPILATION ON
  )
 
install(TARGETS sirius LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")
install(TARGETS sirius_f LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")
install(FILES "sirius.mod" DESTINATION "${CMAKE_INSTALL_PREFIX}/include")

# TODO: install headers with subdirectories
#file(GLOB_RECURSE headers "*.h" "*.hpp")
#install(FILES ${headers} DESTINATION "${CMAKE_INSTALL_PREFIX}/include" COMPONENT Headers)
