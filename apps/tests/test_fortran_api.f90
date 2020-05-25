program test_fortran_api
use sirius
type(C_PTR) :: handler

call sirius_initialize(call_mpi_init=.true.)

call sirius_create_context(MPI_COMM_WORLD, handler)

call sirius_free_handler(handler)

call sirius_finalize


end program
