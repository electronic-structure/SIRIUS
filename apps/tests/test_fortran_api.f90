program test_fortran_api
use sirius
type(C_PTR) :: handler
logical :: stat
integer i,j,k
character(100) key

call sirius_initialize(call_mpi_init=.true.)

i = MPI_COMM_WORLD
call sirius_create_context(i, handler)

stat = .true.
call sirius_context_initialized(handler, stat)
if (stat) then
    stop 'error'
endif

call sirius_option_get_length('control', i)
write(*,*)'length of control:', i

do j=1,i
  call sirius_option_get_name_and_type('control', j, key, k)
  write(*,*)j,trim(adjustl(key)),k
enddo

call sirius_initialize_context(handler, i)
write(*,*)'err.code of sirius_initialize_context: ', i

call sirius_initialize_context(handler)

call sirius_free_handler(handler)

call sirius_finalize


end program
