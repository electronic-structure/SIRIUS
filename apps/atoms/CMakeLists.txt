add_executable(atom atom.cpp)
SIRIUS_SETUP_TARGET(atom)
install(TARGETS atom RUNTIME
    DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")

# atom2.cpp is missing develop branch
# add_executable(atom2 atom2.cpp)
# add_dependencies(atom2 sirius)
# target_link_libraries(atom2 "${sirius_location};${SYSTEM_LIBRARIES};${HDF5_C_LIBRARIES}")
# install(TARGETS atom2 RUNTIME
#     DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
