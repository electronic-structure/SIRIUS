add_executable(hydrogen hydrogen.cpp)
target_link_libraries(hydrogen PRIVATE sirius_cxx)
install(TARGETS hydrogen RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
