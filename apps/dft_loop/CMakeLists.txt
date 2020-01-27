add_executable(sirius.scf sirius.scf.cpp)
target_link_libraries(sirius.scf PRIVATE sirius)
install(TARGETS sirius.scf RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
set_property(TARGET sirius.scf PROPERTY POSITION_INDEPENDENT_CODE OFF)
