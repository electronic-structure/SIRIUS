add_executable(sirius2wannier sirius2wannier.cpp)
target_link_libraries(sirius2wannier PRIVATE sirius_cxx)
install(TARGETS sirius2wannier RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
