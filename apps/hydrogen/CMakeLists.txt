add_executable(hydrogen hydrogen.cpp)
target_link_libraries(hydrogen PRIVATE sirius)
install(TARGETS hydrogen RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
