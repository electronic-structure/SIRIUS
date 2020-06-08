include(FindPackageHandleStandardArgs)

exec_program(${PYTHON_EXECUTABLE} ARGS
  -c
  "'import mpi4py; import os; print(os.path.dirname(mpi4py.__file__))'"
  OUTPUT_VARIABLE MPI4PY_PATH
  RETURN_VALUE exit_code)

find_path(MPI4PY_INCLUDE_DIR
  mpi4py/mpi4py.h
  HINTS ${MPI4PY_PATH}
  PATH_SUFFIXES include inc
  )

if(NOT exit_code EQUAL "0")
    message( FATAL_ERROR "Not able to import mpi4py in python interpreter: ${pyinterp}. The mpi4py package can be installed using pip")
endif()


find_package_handle_standard_args(mpi4py DEFAULT_MSG MPI4PY_INCLUDE_DIR)
mark_as_advanced(mpi4py_FOUND MPI4PY_INCLUDE_DIR)

if(mpi4py_FOUND AND NOT TARGET mpi4py::mpi4py)
  add_library(mpi4py::mpi4py INTERFACE IMPORTED)
  set_target_properties(mpi4py::mpi4py PROPERTIES
                                       INTERFACE_INCLUDE_DIRECTORIES "${MPI4PY_INCLUDE_DIR}")
endif()
