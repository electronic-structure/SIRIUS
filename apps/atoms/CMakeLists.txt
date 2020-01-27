add_executable(atom atom.cpp)
target_link_libraries(atom PRIVATE sirius)
install(TARGETS atom RUNTIME
    DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")

# atom2.cpp is missing develop branch
# add_executable(atom2 atom2.cpp)
# target_link_libraries(atom2 PRIVATE sirius)
# install(TARGETS atom2 RUNTIME
#     DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
