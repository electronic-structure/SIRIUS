program test_sddk
use sddk
use mpi
implicit none
integer :: fft_grid_id, gvec_id, fft_id, ierr, num_ranks, rank, num_ranks_fft, rank_fft
integer :: num_gvec_loc, gvec_offset
integer :: gvec_count_fft, gvec_offset_fft
real(8) :: vk(3)
real(8) :: recip_lat(3, 3)
real(8) :: gmax
integer :: comm_fft, i
complex(8), allocatable :: psi(:), psi_out(:)

call mpi_init(ierr)
call mpi_comm_size(MPI_COMM_WORLD, num_ranks, ierr)
call mpi_comm_rank(MPI_COMM_WORLD, rank, ierr)

! just for the test: create non-trivial FFT communicator
if (num_ranks.eq.4) then
  call mpi_comm_split(MPI_COMM_WORLD, rank / 2, rank, comm_fft, ierr)
else
  comm_fft = MPI_COMM_WORLD
endif
call mpi_comm_size(comm_fft, num_ranks_fft, ierr)
call mpi_comm_rank(comm_fft, rank_fft, ierr)

write(*,*)'rank: ', rank, 'rank_fft: ', rank_fft

! create FFT grid object
call sddk_create_fft_grid((/100, 100, 100/), fft_grid_id)

recip_lat = 0
recip_lat(1, 1) = 1
recip_lat(2, 2) = 1
recip_lat(3, 3) = 1

gmax = 8.0

! create G-vector object
call sddk_create_gvec((/0.d0, 0.d0, 0.d0/), recip_lat(:, 1), recip_lat(:, 2), recip_lat(:, 3), gmax,&
                      &0, MPI_COMM_WORLD, comm_fft, gvec_id)
! get local number of G-vectors
call sddk_get_gvec_count(gvec_id, rank, num_gvec_loc)
! get offset in global index
call sddk_get_gvec_offset(gvec_id, rank, gvec_offset)
write(*,*)"local number of G-vectors and offset: ", num_gvec_loc, gvec_offset

! get local number of G-vectors for the FFT buffer
call sddk_get_gvec_count_fft(gvec_id, gvec_count_fft)
! get offset in global index
call sddk_get_gvec_offset_fft(gvec_id, gvec_offset_fft)
write(*,*)"local number of G-vectors and offset for FFT: ", gvec_count_fft, gvec_offset_fft

! create FFT driver
call sddk_create_fft(fft_grid_id, comm_fft, fft_id)

allocate(psi(gvec_count_fft))
allocate(psi_out(gvec_count_fft))
psi = 0.d0
psi(1) = 1

call sddk_fft_prepare(fft_id, gvec_id)
call sddk_fft(fft_id, 1, psi(1))
call sddk_fft(fft_id, -1, psi_out(1))
call sddk_fft_dismiss(fft_id)

do i = 1, gvec_count_fft
  if (abs(psi(i) - psi_out(i)) > 1d-12) then
    write(*,*)'wrong FFT result'
    stop
  endif
enddo


deallocate(psi)
deallocate(psi_out)

! destroy FFT driver
call sddk_delete_fft(fft_id)
! destroy G-vecgtors
call sddk_delete_gvec(gvec_id)
! destroy FFT grid
call sddk_delete_fft_grid(fft_grid_id)

call sddk_print_timers()

call mpi_finalize(ierr)

end program
