add_executable(atom atom.cpp)
target_link_libraries(atom PRIVATE sirius)
install(TARGETS atom RUNTIME
    DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
