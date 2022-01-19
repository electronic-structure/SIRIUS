add_executable(linear_response linear_response.cpp)
target_link_libraries(linear_response PRIVATE sirius sirius::filesystem)
install(TARGETS linear_response RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
set_property(TARGET linear_response PROPERTY POSITION_INDEPENDENT_CODE OFF)
