
include(FindPackageHandleStandardArgs)

find_package(Python${PYTHON_VERSION_MAJOR} COMPONENTS Interpreter)

set(pyinterp "${Python${PYTHON_VERSION_MAJOR}_EXECUTABLE}")
message("py interp: ${pyinterp}")
exec_program(${pyinterp} ARGS
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


find_package_handle_standard_args(MPI4PY DEFAULT_MSG MPI4PY_INCLUDE_DIR)
mark_as_advanced(MPI4PY_FOUND MPI4PY_INCLUDE_DIR)
