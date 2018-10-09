add_executable(unit_cell_tools unit_cell_tools.cpp)
add_dependencies(unit_cell_tools sirius)
target_link_libraries(unit_cell_tools "${sirius_location};${LIBXC_LIBRARIES};${MPI_CXX_LIBRARIES};${LAPACK_LIBRARIES};${GSL_LIBRARIES};${FFTW_LIBRARIES}")
