program test_sddk
use sddk
use mpi
implicit none
integer :: fft_grid_id, gvec_id, fft_id, wf_id, ierr, num_ranks, rank, num_ranks_fft, rank_fft
integer :: num_gvec_loc, gvec_offset
integer :: gvec_count_fft, gvec_offset_fft
integer :: num_gvec
real(8) :: vk(3)
real(8) :: recip_lat(3, 3)
real(8) :: gmax
integer :: mpi_grid(2), fft_grid(3)
integer :: comm_fft, i, nbnd, ig
complex(8), allocatable :: psi(:), psi_out(:)
complex(8), pointer :: wf_ptr(:, :), wf_extra_ptr(:, :)
complex(8), allocatable :: wf(:, :)
type(C_PTR) gvec_h, fft_h

!--------------------------!
! basic MPI initialization !
!--------------------------!
call mpi_init(ierr)
call mpi_comm_size(MPI_COMM_WORLD, num_ranks, ierr)
call mpi_comm_rank(MPI_COMM_WORLD, rank, ierr)

!---------------------------!
! set the initial paramters !
!---------------------------!
! this grid is for the full wave-functions remapping and single-node FFT
mpi_grid = (/1, num_ranks/)
! this grid is an example for 6 MPI ranks: 3 for FFT, 2 for band parallelziation
!mpi_grid = (/3, 2/)

! size of the FFT box
fft_grid = (/200, 200, 200/)

! reciprocal lattice
recip_lat = 0
recip_lat(1, 1) = 1
recip_lat(2, 2) = 1
recip_lat(3, 3) = 1

! cutoff
gmax = 20

! number of bands
nbnd = 100

!-------------------------!
! create FFT communicator !
!-------------------------!
call mpi_comm_split(MPI_COMM_WORLD, rank / mpi_grid(1), rank, comm_fft, ierr)
call mpi_comm_size(comm_fft, num_ranks_fft, ierr)
call mpi_comm_rank(comm_fft, rank_fft, ierr)

!== if (rank.eq.0) then
!==   write(*,*)'FFT communicator size: ', num_ranks_fft
!== endif
!== !write(*,*)'rank: ', rank, 'rank_fft: ', rank_fft
!== 
!== ! create FFT grid object
!== call sddk_create_fft_grid(fft_grid(1), fft_grid_id)
!== 

! create G-vector object
call sddk_create_gkvec((/0.d0, 0.d0, 0.d0/), recip_lat(:, 1), recip_lat(:, 2), recip_lat(:, 3), gmax,&
                      &c_logical(.false.), MPI_COMM_WORLD, gvec_h)

call sddk_delete_object(gvec_h)

!== call sddk_get_num_gvec(gvec_id, num_gvec)
!== if (rank.eq.0) then
!==   write(*,*)'num_gvec: ', num_gvec
!== endif
!== ! get local number of G-vectors
!== call sddk_get_gvec_count(gvec_id, rank, num_gvec_loc)
!== ! get offset in global index
!== call sddk_get_gvec_offset(gvec_id, rank, gvec_offset)
!== !write(*,*)"local number of G-vectors and offset: ", num_gvec_loc, gvec_offset
!== 
!== ! get local number of G-vectors for the FFT buffer
!== call sddk_get_gvec_count_fft(gvec_id, gvec_count_fft)
!== ! get offset in global index
!== call sddk_get_gvec_offset_fft(gvec_id, gvec_offset_fft)
!== !write(*,*)"local number of G-vectors and offset for FFT: ", gvec_count_fft, gvec_offset_fft
!== 
! create FFT driver
call sddk_create_fft(fft_grid(1), comm_fft, fft_h)
call sddk_delete_object(fft_h)

!== ! create wave-functions
!== call sddk_create_wave_functions(gvec_id, nbnd, wf_id)
!== 
!== call sddk_get_wave_functions_prime_ptr(wf_id, wf_ptr)
!== 
!== ! create a reference array of wave-functions
!== allocate(wf(num_gvec_loc, nbnd))
!== do i = 1, nbnd
!==   do ig = 1, num_gvec_loc
!==     ! fill with some data
!==     wf(ig, i) = dcmplx(ig * 0.001d0, i * 0.1d0)
!==     wf_ptr(ig, i) = wf(ig, i)
!==   enddo
!== enddo
!== 
!== ! remap wave-functions to extra storage
!== call sddk_remap_wave_functions_forward(wf_id, nbnd, 1)
!== 
!== ! get pointer to extra storage
!== call sddk_get_wave_functions_extra_ptr(wf_id, wf_extra_ptr)
!== 
!== !write(*,*)'sizes: ', size(wf_extra_ptr, 1), size(wf_extra_ptr, 2)
!== 
!== ! prepare FFT driver
!== call sddk_fft_prepare(fft_id, gvec_id)
!== 
!== ! loop over local number of bands
!== do i = 1, size(wf_extra_ptr, 2)
!==   call sddk_fft(fft_id,  1, wf_extra_ptr(1, i))
!==   call sddk_fft(fft_id, -1, wf_extra_ptr(1, i))
!== enddo
!== 
!== ! free FFT driver
!== call sddk_fft_dismiss(fft_id)
!== 
!== ! remap wave-functions back from extra to the prime storage
!== call sddk_remap_wave_functions_backward(wf_id, nbnd, 1)
!== 
!== ! compare with the reference data
!== do i = 1, nbnd
!==   do ig = 1, num_gvec_loc
!==     if (abs(wf_ptr(ig, i) - wf(ig, i)) > 1d-10) then
!==       write(*,*)'wrong FFT result: ', abs(wf_ptr(ig, i) - wf(ig, i)), ' for ig=', ig, 'and i=', i
!==       stop
!==     endif
!==   enddo
!== enddo
!== 
!== deallocate(wf)

! get pointer of FFT buffer

! loop over bands, transform to r, multiply by V(r), transform back

!allocate(psi(gvec_count_fft))
!allocate(psi_out(gvec_count_fft))
!psi = 0.d0
!psi(1) = 1
!call sddk_fft_prepare(fft_id, gvec_id)
!call sddk_fft(fft_id, 1, psi(1))
!call sddk_fft(fft_id, -1, psi_out(1))
!call sddk_fft_dismiss(fft_id)
!do i = 1, gvec_count_fft
!  if (abs(psi(i) - psi_out(i)) > 1d-12) then
!    write(*,*)'wrong FFT result'
!    stop
!  endif
!enddo
!deallocate(psi)
!deallocate(psi_out)

!! destroy FFT driver
!call sddk_delete_object(fft_id)
!! destroy G-vecgtors
!call sddk_delete_object(gvec_id)
!! destroy FFT grid
!call sddk_delete_object(fft_grid_id)
!! destroy wave functions
!call sddk_delete_object(wf_id)
!
!call sddk_print_timers()

call mpi_finalize(ierr)

end program
