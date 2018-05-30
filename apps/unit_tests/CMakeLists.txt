FILE(GLOB _tests RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.cpp")

foreach(_test ${_tests})
  string(REGEX REPLACE ".cpp" "" target ${_test})
  add_executable(${target} "${_test}")
  add_dependencies(${target} sirius)
  target_link_libraries(${target} "${sirius_location};${SYSTEM_LIBRARIES};${HDF5_C_LIBRARIES};${HDF5_C_HL_LIBRARIES}")
  install(TARGETS ${target} RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")


endforeach()
