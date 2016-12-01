program test_sddk
use sddk
use mpi
implicit none
integer :: fft_grid_id, gvec_id, fft_id, ierr, num_ranks, rank
integer :: num_gvec_loc, gvec_offset
real(8) :: vk(3)
real(8) :: recip_lat(3, 3)
real(8) :: gmax

call mpi_init(ierr)
call mpi_comm_size(MPI_COMM_WORLD, num_ranks, ierr)
call mpi_comm_rank(MPI_COMM_WORLD, rank, ierr)

call sddk_create_fft_grid((/100, 100, 100/), fft_grid_id)

recip_lat = 0
recip_lat(1, 1) = 1
recip_lat(2, 2) = 1
recip_lat(3, 3) = 1

gmax = 6.0

call sddk_create_gvec((/0.d0, 0.d0, 0.d0/), recip_lat(:, 1), recip_lat(:, 2), recip_lat(:, 3), gmax,&
                      &fft_grid_id, num_ranks, 0, MPI_COMM_WORLD, gvec_id)

call sddk_get_gvec_count(gvec_id, rank, num_gvec_loc)
call sddk_get_gvec_offset(gvec_id, rank, gvec_offset)
write(*,*)"local number of G-vectors and offset: ", num_gvec_loc, gvec_offset

call sddk_create_fft(fft_grid_id, MPI_COMM_WORLD, fft_id)

write(*,*)fft_grid_id, gvec_id, fft_id

call sddk_delete_fft(fft_id)
call sddk_delete_gvec(gvec_id)
call sddk_delete_fft_grid(fft_grid_id)




call sddk_create_fft_grid((/50, 50, 50/), fft_grid_id)
call sddk_create_gvec((/0.d0, 0.d0, 0.d0/), recip_lat(:, 1), recip_lat(:, 2), recip_lat(:, 3), gmax,&
                      &fft_grid_id, num_ranks, 0, MPI_COMM_WORLD, gvec_id)

write(*,*)fft_grid_id, gvec_id

call sddk_delete_gvec(gvec_id)
call sddk_delete_fft_grid(fft_grid_id)

call mpi_finalize(ierr)

end program
