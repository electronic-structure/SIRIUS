add_executable(unit_cell_tools unit_cell_tools.cpp)
target_link_libraries(unit_cell_tools PRIVATE sirius)
install(TARGETS unit_cell_tools RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
set_property(TARGET unit_cell_tools PROPERTY POSITION_INDEPENDENT_CODE OFF)
