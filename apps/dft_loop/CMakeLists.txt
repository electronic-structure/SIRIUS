add_executable(sirius.scf sirius.scf.cpp)
SIRIUS_SETUP_TARGET(sirius.scf)
install(TARGETS sirius.scf RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
set_property(TARGET sirius.scf PROPERTY POSITION_INDEPENDENT_CODE OFF)
