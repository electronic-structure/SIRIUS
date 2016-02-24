!    This file is part of ELPA.
!
!    The ELPA library was originally created by the ELPA consortium,
!    consisting of the following organizations:
!
!    - Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
!    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
!      Informatik,
!    - Technische Universität München, Lehrstuhl für Informatik mit
!      Schwerpunkt Wissenschaftliches Rechnen ,
!    - Fritz-Haber-Institut, Berlin, Abt. Theorie,
!    - Max-Plack-Institut für Mathematik in den Naturwissenschaftrn,
!      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
!      and
!    - IBM Deutschland GmbH
!
!    This particular source code file contains additions, changes and
!    enhancements authored by Intel Corporation which is not part of
!    the ELPA consortium.
!
!    More information can be found here:
!    http://elpa.rzg.mpg.de/
!
!    ELPA is free software: you can redistribute it and/or modify
!    it under the terms of the version 3 of the license of the
!    GNU Lesser General Public License as published by the Free
!    Software Foundation.
!
!    ELPA is distributed in the hope that it will be useful,
!    but WITHOUT ANY WARRANTY; without even the implied warranty of
!    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!    GNU Lesser General Public License for more details.
!
!    You should have received a copy of the GNU Lesser General Public License
!    along with ELPA.  If not, see <http://www.gnu.org/licenses/>
!
!    ELPA reflects a substantial effort on the part of the original
!    ELPA consortium, and we ask you to respect the spirit of the
!    license that we chose: i.e., please contribute any changes you
!    may have back to the original ELPA library distribution, and keep
!    any derivatives of ELPA under the same license that we chose for
!    the original distribution, the GNU Lesser General Public License.
!
!
! ELPA1 -- Faster replacements for ScaLAPACK symmetric eigenvalue routines
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".



! ELPA2 -- 2-stage solver for ELPA
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".


#include "config-f90.h"

module ELPA2

! Version 1.1.2, 2011-02-21

  use elpa_utilities
  USE ELPA1
  use elpa2_utilities
  use elpa_pdgeqrf

  implicit none

  PRIVATE ! By default, all routines contained are private

  ! The following routines are public:

  public :: solve_evp_real_2stage
  public :: solve_evp_complex_2stage

  public :: bandred_real
  public :: tridiag_band_real
  public :: trans_ev_tridi_to_band_real
  public :: trans_ev_band_to_full_real

  public :: bandred_complex
  public :: tridiag_band_complex
  public :: trans_ev_tridi_to_band_complex
  public :: trans_ev_band_to_full_complex

  public :: band_band_real
  public :: divide_band

  integer, public :: which_qr_decomposition = 1     ! defines, which QR-decomposition algorithm will be used
                                                    ! 0 for unblocked
                                                    ! 1 for blocked (maxrank: nblk)
!-------------------------------------------------------------------------------

  ! The following array contains the Householder vectors of the
  ! transformation band -> tridiagonal.
  ! It is allocated and set in tridiag_band_real and used in
  ! trans_ev_tridi_to_band_real.
  ! It must be deallocated by the user after trans_ev_tridi_to_band_real!

  real*8, allocatable :: hh_trans_real(:,:)
  complex*16, allocatable :: hh_trans_complex(:,:)

!-------------------------------------------------------------------------------

  include 'mpif.h'


!******
contains

function solve_evp_real_2stage(na, nev, a, lda, ev, q, ldq, nblk,        &
                               matrixCols,                               &
                                 mpi_comm_rows, mpi_comm_cols,           &
                                 mpi_comm_all, THIS_REAL_ELPA_KERNEL_API,&
                                 useQR) result(success)

!-------------------------------------------------------------------------------
!  solve_evp_real_2stage: Solves the real eigenvalue problem with a 2 stage approach
!
!  Parameters
!
!  na          Order of matrix a
!
!  nev         Number of eigenvalues needed
!
!  a(lda,matrixCols)    Distributed matrix for which eigenvalues are to be computed.
!              Distribution is like in Scalapack.
!              The full matrix must be set (not only one half like in scalapack).
!              Destroyed on exit (upper and lower half).
!
!  lda         Leading dimension of a
!  matrixCols  local columns of matrix a and q
!
!  ev(na)      On output: eigenvalues of a, every processor gets the complete set
!
!  q(ldq,matrixCols)    On output: Eigenvectors of a
!              Distribution is like in Scalapack.
!              Must be always dimensioned to the full size (corresponding to (na,na))
!              even if only a part of the eigenvalues is needed.
!
!  ldq         Leading dimension of q
!
!  nblk        blocksize of cyclic distribution, must be the same in both directions!
!
!  mpi_comm_rows
!  mpi_comm_cols
!              MPI-Communicators for rows/columns
!  mpi_comm_all
!              MPI-Communicator for the total processor set
!
!-------------------------------------------------------------------------------
#ifdef HAVE_DETAILED_TIMINGS
 use timings
#endif
   implicit none
   logical, intent(in), optional :: useQR
   logical                       :: useQRActual, useQREnvironment
   integer, intent(in), optional :: THIS_REAL_ELPA_KERNEL_API
   integer                       :: THIS_REAL_ELPA_KERNEL

   integer, intent(in)           :: na, nev, lda, ldq, matrixCols, mpi_comm_rows, &
                                    mpi_comm_cols, mpi_comm_all
   integer, intent(in)           :: nblk
   real*8, intent(inout)         :: a(lda,matrixCols), ev(na), q(ldq,matrixCols)

   integer                       :: my_pe, n_pes, my_prow, my_pcol, np_rows, np_cols, mpierr
   integer                       :: nbw, num_blocks
   real*8, allocatable           :: tmat(:,:,:), e(:)
   real*8                        :: ttt0, ttt1, ttts
   integer                       :: i
   logical                       :: success
   logical, save                 :: firstCall = .true.
   logical                       :: wantDebug

#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("solve_evp_real_2stage")
#endif
   call mpi_comm_rank(mpi_comm_all,my_pe,mpierr)
   call mpi_comm_size(mpi_comm_all,n_pes,mpierr)

   call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
   call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
   call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
   call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)


   wantDebug = .false.
   if (firstCall) then
     ! are debug messages desired?
     wantDebug = debug_messages_via_environment_variable()
     firstCall = .false.
   endif

   success = .true.

   useQRActual = .false.

   ! set usage of qr decomposition via API call
   if (present(useQR)) then
     if (useQR) useQRActual = .true.
     if (.not.(useQR)) useQRACtual = .false.
   endif

   ! overwrite this with environment variable settings
   if (qr_decomposition_via_environment_variable(useQREnvironment)) then
     useQRActual = useQREnvironment
   endif

   if (useQRActual) then
     if (mod(na,nblk) .ne. 0) then
       if (wantDebug) then
         write(error_unit,*) "solve_evp_real_2stage: QR-decomposition: blocksize does not fit with matrixsize"
       endif
     print *, "Do not use QR-decomposition for this matrix and blocksize."
     success = .false.
     return
     endif
   endif


   if (present(THIS_REAL_ELPA_KERNEL_API)) then
     ! user defined kernel via the optional argument in the API call
     THIS_REAL_ELPA_KERNEL = THIS_REAL_ELPA_KERNEL_API
   else

     ! if kernel is not choosen via api
     ! check whether set by environment variable
     THIS_REAL_ELPA_KERNEL = get_actual_real_kernel()
   endif

   ! check whether choosen kernel is allowed
   if (check_allowed_real_kernels(THIS_REAL_ELPA_KERNEL)) then

     if (my_pe == 0) then
       write(error_unit,*) " "
       write(error_unit,*) "The choosen kernel ",REAL_ELPA_KERNEL_NAMES(THIS_REAL_ELPA_KERNEL)
       write(error_unit,*) "is not in the list of the allowed kernels!"
       write(error_unit,*) " "
       write(error_unit,*) "Allowed kernels are:"
       do i=1,size(REAL_ELPA_KERNEL_NAMES(:))
         if (AVAILABLE_REAL_ELPA_KERNELS(i) .ne. 0) then
           write(error_unit,*) REAL_ELPA_KERNEL_NAMES(i)
         endif
       enddo

       write(error_unit,*) " "
       write(error_unit,*) "The defaul kernel REAL_ELPA_KERNEL_GENERIC will be used !"
     endif
     THIS_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_GENERIC

   endif

   ! Choose bandwidth, must be a multiple of nblk, set to a value >= 32
   ! On older systems (IBM Bluegene/P, Intel Nehalem) a value of 32 was optimal.
   ! For Intel(R) Xeon(R) E5 v2 and v3, better use 64 instead of 32!
   ! For IBM Bluegene/Q this is not clear at the moment. We have to keep an eye
   ! on this and maybe allow a run-time optimization here
   nbw = (63/nblk+1)*nblk

   num_blocks = (na-1)/nbw + 1

   allocate(tmat(nbw,nbw,num_blocks))

   ! Reduction full -> band

   ttt0 = MPI_Wtime()
   ttts = ttt0
   call bandred_real(na, a, lda, nblk, nbw, matrixCols, num_blocks, mpi_comm_rows, mpi_comm_cols, &
                     tmat, wantDebug, success, useQRActual)
   if (.not.(success)) return
   ttt1 = MPI_Wtime()
   if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
      write(error_unit,*) 'Time bandred_real               :',ttt1-ttt0

   ! Reduction band -> tridiagonal

   allocate(e(na))

   ttt0 = MPI_Wtime()
   call tridiag_band_real(na, nbw, nblk, a, lda, ev, e, matrixCols, mpi_comm_rows, &
                          mpi_comm_cols, mpi_comm_all)
   ttt1 = MPI_Wtime()
   if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
      write(error_unit,*) 'Time tridiag_band_real          :',ttt1-ttt0

   call mpi_bcast(ev,na,MPI_REAL8,0,mpi_comm_all,mpierr)
   call mpi_bcast(e,na,MPI_REAL8,0,mpi_comm_all,mpierr)

   ttt1 = MPI_Wtime()
   time_evp_fwd = ttt1-ttts

   ! Solve tridiagonal system

   ttt0 = MPI_Wtime()
   call solve_tridi(na, nev, ev, e, q, ldq, nblk, matrixCols, mpi_comm_rows,  &
                    mpi_comm_cols, wantDebug, success)
   if (.not.(success)) return

   ttt1 = MPI_Wtime()
   if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
     write(error_unit,*) 'Time solve_tridi                :',ttt1-ttt0
   time_evp_solve = ttt1-ttt0
   ttts = ttt1

   deallocate(e)

   ! Backtransform stage 1

   ttt0 = MPI_Wtime()
   call trans_ev_tridi_to_band_real(na, nev, nblk, nbw, q, ldq, matrixCols, mpi_comm_rows, &
                                    mpi_comm_cols, wantDebug, success, THIS_REAL_ELPA_KERNEL)
   if (.not.(success)) return
   ttt1 = MPI_Wtime()
   if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
      write(error_unit,*) 'Time trans_ev_tridi_to_band_real:',ttt1-ttt0

   ! We can now deallocate the stored householder vectors
   deallocate(hh_trans_real)

   ! Backtransform stage 2

   ttt0 = MPI_Wtime()
   call trans_ev_band_to_full_real(na, nev, nblk, nbw, a, lda, tmat, q, ldq, matrixCols, num_blocks, mpi_comm_rows, &
                                   mpi_comm_cols, useQRActual)
   ttt1 = MPI_Wtime()
   if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
      write(error_unit,*) 'Time trans_ev_band_to_full_real :',ttt1-ttt0
   time_evp_back = ttt1-ttts

   deallocate(tmat)
#ifdef HAVE_DETAILED_TIMINGS
   call timer%stop("solve_evp_real_2stage")
#endif
1  format(a,f10.3)

end function solve_evp_real_2stage

!-------------------------------------------------------------------------------

!-------------------------------------------------------------------------------

function solve_evp_complex_2stage(na, nev, a, lda, ev, q, ldq, nblk, &
                                  matrixCols, mpi_comm_rows, mpi_comm_cols,      &
                                    mpi_comm_all, THIS_COMPLEX_ELPA_KERNEL_API) result(success)

!-------------------------------------------------------------------------------
!  solve_evp_complex_2stage: Solves the complex eigenvalue problem with a 2 stage approach
!
!  Parameters
!
!  na          Order of matrix a
!
!  nev         Number of eigenvalues needed
!
!  a(lda,matrixCols)    Distributed matrix for which eigenvalues are to be computed.
!              Distribution is like in Scalapack.
!              The full matrix must be set (not only one half like in scalapack).
!              Destroyed on exit (upper and lower half).
!
!  lda         Leading dimension of a
!  matrixCols  local columns of matrix a and q
!
!  ev(na)      On output: eigenvalues of a, every processor gets the complete set
!
!  q(ldq,matrixCols)    On output: Eigenvectors of a
!              Distribution is like in Scalapack.
!              Must be always dimensioned to the full size (corresponding to (na,na))
!              even if only a part of the eigenvalues is needed.
!
!  ldq         Leading dimension of q
!
!  nblk        blocksize of cyclic distribution, must be the same in both directions!
!
!  mpi_comm_rows
!  mpi_comm_cols
!              MPI-Communicators for rows/columns
!  mpi_comm_all
!              MPI-Communicator for the total processor set
!
!-------------------------------------------------------------------------------
#ifdef HAVE_DETAILED_TIMINGS
 use timings
#endif
   implicit none
   integer, intent(in), optional :: THIS_COMPLEX_ELPA_KERNEL_API
   integer                       :: THIS_COMPLEX_ELPA_KERNEL
   integer, intent(in)           :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all
   complex*16, intent(inout)     :: a(lda,matrixCols), q(ldq,matrixCols)
   real*8, intent(inout)         :: ev(na)

   integer                       :: my_prow, my_pcol, np_rows, np_cols, mpierr, my_pe, n_pes
   integer                       :: l_cols, l_rows, l_cols_nev, nbw, num_blocks
   complex*16, allocatable       :: tmat(:,:,:)
   real*8, allocatable           :: q_real(:,:), e(:)
   real*8                        :: ttt0, ttt1, ttts
   integer                       :: i

   logical                       :: success, wantDebug
   logical, save                 :: firstCall = .true.

#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("solve_evp_complex_2stage")
#endif
   call mpi_comm_rank(mpi_comm_all,my_pe,mpierr)
   call mpi_comm_size(mpi_comm_all,n_pes,mpierr)

   call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
   call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
   call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
   call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)

   wantDebug = .false.
   if (firstCall) then
     ! are debug messages desired?
     wantDebug = debug_messages_via_environment_variable()
     firstCall = .false.
   endif


   success = .true.

   if (present(THIS_COMPLEX_ELPA_KERNEL_API)) then
     ! user defined kernel via the optional argument in the API call
     THIS_COMPLEX_ELPA_KERNEL = THIS_COMPLEX_ELPA_KERNEL_API
   else
     ! if kernel is not choosen via api
     ! check whether set by environment variable
     THIS_COMPLEX_ELPA_KERNEL = get_actual_complex_kernel()
   endif

   ! check whether choosen kernel is allowed
   if (check_allowed_complex_kernels(THIS_COMPLEX_ELPA_KERNEL)) then

     if (my_pe == 0) then
       write(error_unit,*) " "
       write(error_unit,*) "The choosen kernel ",COMPLEX_ELPA_KERNEL_NAMES(THIS_COMPLEX_ELPA_KERNEL)
       write(error_unit,*) "is not in the list of the allowed kernels!"
       write(error_unit,*) " "
       write(error_unit,*) "Allowed kernels are:"
       do i=1,size(COMPLEX_ELPA_KERNEL_NAMES(:))
         if (AVAILABLE_COMPLEX_ELPA_KERNELS(i) .ne. 0) then
           write(error_unit,*) COMPLEX_ELPA_KERNEL_NAMES(i)
         endif
       enddo

       write(error_unit,*) " "
       write(error_unit,*) "The defaul kernel COMPLEX_ELPA_KERNEL_GENERIC will be used !"
     endif
     THIS_COMPLEX_ELPA_KERNEL = COMPLEX_ELPA_KERNEL_GENERIC
   endif
   ! Choose bandwidth, must be a multiple of nblk, set to a value >= 32

   nbw = (31/nblk+1)*nblk

   num_blocks = (na-1)/nbw + 1

   allocate(tmat(nbw,nbw,num_blocks))

   ! Reduction full -> band

   ttt0 = MPI_Wtime()
   ttts = ttt0
   call bandred_complex(na, a, lda, nblk, nbw, matrixCols, num_blocks, mpi_comm_rows, mpi_comm_cols, &
                        tmat, wantDebug, success)
   if (.not.(success)) then
#ifdef HAVE_DETAILED_TIMINGS
     call timer%stop()
#endif
     return
   endif
   ttt1 = MPI_Wtime()
   if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
      write(error_unit,*) 'Time bandred_complex               :',ttt1-ttt0

   ! Reduction band -> tridiagonal

   allocate(e(na))

   ttt0 = MPI_Wtime()
   call tridiag_band_complex(na, nbw, nblk, a, lda, ev, e, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all)
   ttt1 = MPI_Wtime()
   if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
      write(error_unit,*) 'Time tridiag_band_complex          :',ttt1-ttt0

   call mpi_bcast(ev,na,MPI_REAL8,0,mpi_comm_all,mpierr)
   call mpi_bcast(e,na,MPI_REAL8,0,mpi_comm_all,mpierr)

   ttt1 = MPI_Wtime()
   time_evp_fwd = ttt1-ttts

   l_rows = local_index(na, my_prow, np_rows, nblk, -1) ! Local rows of a and q
   l_cols = local_index(na, my_pcol, np_cols, nblk, -1) ! Local columns of q
   l_cols_nev = local_index(nev, my_pcol, np_cols, nblk, -1) ! Local columns corresponding to nev

   allocate(q_real(l_rows,l_cols))

   ! Solve tridiagonal system

   ttt0 = MPI_Wtime()
   call solve_tridi(na, nev, ev, e, q_real, ubound(q_real,dim=1), nblk, matrixCols, &
                    mpi_comm_rows, mpi_comm_cols, wantDebug, success)
   if (.not.(success)) return

   ttt1 = MPI_Wtime()
   if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times)  &
      write(error_unit,*) 'Time solve_tridi                   :',ttt1-ttt0
   time_evp_solve = ttt1-ttt0
   ttts = ttt1

   q(1:l_rows,1:l_cols_nev) = q_real(1:l_rows,1:l_cols_nev)

   deallocate(e, q_real)

   ! Backtransform stage 1

   ttt0 = MPI_Wtime()
   call trans_ev_tridi_to_band_complex(na, nev, nblk, nbw, q, ldq,  &
                                       matrixCols, mpi_comm_rows, mpi_comm_cols,&
                                       wantDebug, success,THIS_COMPLEX_ELPA_KERNEL)
   if (.not.(success)) return
   ttt1 = MPI_Wtime()
   if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
      write(error_unit,*) 'Time trans_ev_tridi_to_band_complex:',ttt1-ttt0

   ! We can now deallocate the stored householder vectors
   deallocate(hh_trans_complex)

   ! Backtransform stage 2

   ttt0 = MPI_Wtime()
   call trans_ev_band_to_full_complex(na, nev, nblk, nbw, a, lda, tmat, q, ldq, matrixCols, num_blocks, &
                                      mpi_comm_rows, mpi_comm_cols)
   ttt1 = MPI_Wtime()
   if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
      write(error_unit,*) 'Time trans_ev_band_to_full_complex :',ttt1-ttt0
   time_evp_back = ttt1-ttts

   deallocate(tmat)
#ifdef HAVE_DETAILED_TIMINGS
   call timer%stop("solve_evp_complex_2stage")
#endif

1  format(a,f10.3)

end function solve_evp_complex_2stage

!-------------------------------------------------------------------------------

subroutine bandred_real(na, a, lda, nblk, nbw, matrixCols, numBlocks, mpi_comm_rows, mpi_comm_cols, &
                        tmat, wantDebug, success, useQR)

!-------------------------------------------------------------------------------
!  bandred_real: Reduces a distributed symmetric matrix to band form
!
!  Parameters
!
!  na          Order of matrix
!
!  a(lda,matrixCols)    Distributed matrix which should be reduced.
!              Distribution is like in Scalapack.
!              Opposed to Scalapack, a(:,:) must be set completely (upper and lower half)
!              a(:,:) is overwritten on exit with the band and the Householder vectors
!              in the upper half.
!
!  lda         Leading dimension of a
!  matrixCols  local columns of matrix a
!
!  nblk        blocksize of cyclic distribution, must be the same in both directions!
!
!  nbw         semi bandwith of output matrix
!
!  mpi_comm_rows
!  mpi_comm_cols
!              MPI-Communicators for rows/columns
!
!  tmat(nbw,nbw,numBlocks)    where numBlocks = (na-1)/nbw + 1
!              Factors for the Householder vectors (returned), needed for back transformation
!
!-------------------------------------------------------------------------------
#ifdef HAVE_DETAILED_TIMINGS
 use timings
#endif
#ifdef WITH_OPENMP
   use omp_lib
#endif
   implicit none

   integer             :: na, lda, nblk, nbw, matrixCols, numBlocks, mpi_comm_rows, mpi_comm_cols
   real*8              :: a(lda,matrixCols), tmat(nbw,nbw,numBlocks)

   integer             :: my_prow, my_pcol, np_rows, np_cols, mpierr
   integer             :: l_cols, l_rows
   integer             :: i, j, lcs, lce, lrs, lre, lc, lr, cur_pcol, n_cols, nrow
   integer             :: istep, ncol, lch, lcx, nlc, mynlc
   integer             :: tile_size, l_rows_tile, l_cols_tile

   real*8              :: vnorm2, xf, aux1(nbw), aux2(nbw), vrl, tau, vav(nbw,nbw)

   real*8, allocatable :: tmp(:,:), vr(:), vmr(:,:), umc(:,:)

   ! needed for blocked QR decomposition
   integer             :: PQRPARAM(11), work_size
   real*8              :: dwork_size(1)
   real*8, allocatable :: work_blocked(:), tauvector(:), blockheuristic(:)

   logical, intent(in) :: wantDebug
   logical, intent(out):: success

   logical, intent(in) :: useQR

   integer :: mystart, myend, m_way, n_way, work_per_thread, m_id, n_id, n_threads, ii, pp, transformChunkSize

#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("bandred_real")
#endif
   call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
   call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
   call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
   call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)
   success = .true.


   ! Semibandwith nbw must be a multiple of blocksize nblk
   if (mod(nbw,nblk)/=0) then
     if (my_prow==0 .and. my_pcol==0) then
       if (wantDebug) then
         write(error_unit,*) 'ELPA2_bandred_real: ERROR: nbw=',nbw,', nblk=',nblk
         write(error_unit,*) 'ELPA2_bandred_real: ELPA2 works only for nbw==n*nblk'
       endif
       success = .false.
       return
     endif
   endif

   ! Matrix is split into tiles; work is done only for tiles on the diagonal or above

   tile_size = nblk*least_common_multiple(np_rows,np_cols) ! minimum global tile size
   tile_size = ((128*max(np_rows,np_cols)-1)/tile_size+1)*tile_size ! make local tiles at least 128 wide

   l_rows_tile = tile_size/np_rows ! local rows of a tile
   l_cols_tile = tile_size/np_cols ! local cols of a tile

   if (useQR) then
     if (which_qr_decomposition == 1) then
       call qr_pqrparam_init(pqrparam,    nblk,'M',0,   nblk,'M',0,   nblk,'M',1,'s')
       allocate(tauvector(na))
       allocate(blockheuristic(nblk))
       l_rows = local_index(na, my_prow, np_rows, nblk, -1)
       allocate(vmr(max(l_rows,1),na))

       call qr_pdgeqrf_2dcomm(a, lda, vmr, max(l_rows,1), tauvector(1), tmat(1,1,1), nbw, dwork_size(1), -1, na, &
                             nbw, nblk, nblk, na, na, 1, 0, PQRPARAM, mpi_comm_rows, mpi_comm_cols, blockheuristic)
       work_size = dwork_size(1)
       allocate(work_blocked(work_size))

       work_blocked = 0.0d0
       deallocate(vmr)
     endif
   endif

   do istep = (na-1)/nbw, 1, -1

     n_cols = MIN(na,(istep+1)*nbw) - istep*nbw ! Number of columns in current step

     ! Number of local columns/rows of remaining matrix
     l_cols = local_index(istep*nbw, my_pcol, np_cols, nblk, -1)
     l_rows = local_index(istep*nbw, my_prow, np_rows, nblk, -1)

     ! Allocate vmr and umc to their exact sizes so that they can be used in bcasts and reduces

     allocate(vmr(max(l_rows,1),2*n_cols))
     allocate(umc(max(l_cols,1),2*n_cols))

     allocate(vr(l_rows+1))

     vmr(1:l_rows,1:n_cols) = 0.
     vr(:) = 0
     tmat(:,:,istep) = 0

     ! Reduce current block to lower triangular form

     if (useQR) then
       if (which_qr_decomposition == 1) then
         call qr_pdgeqrf_2dcomm(a, lda, vmr, max(l_rows,1), tauvector(1), &
                                  tmat(1,1,istep), nbw, work_blocked,       &
                                  work_size, na, n_cols, nblk, nblk,        &
                                  istep*nbw+n_cols-nbw, istep*nbw+n_cols, 1,&
                                  0, PQRPARAM, mpi_comm_rows, mpi_comm_cols,&
                                  blockheuristic)
       endif
     else

       do lc = n_cols, 1, -1

         ncol = istep*nbw + lc ! absolute column number of householder vector
         nrow = ncol - nbw ! Absolute number of pivot row

         lr  = local_index(nrow, my_prow, np_rows, nblk, -1) ! current row length
         lch = local_index(ncol, my_pcol, np_cols, nblk, -1) ! HV local column number

         tau = 0

         if (nrow == 1) exit ! Nothing to do

         cur_pcol = pcol(ncol, nblk, np_cols) ! Processor column owning current block

         if (my_pcol==cur_pcol) then

           ! Get vector to be transformed; distribute last element and norm of
           ! remaining elements to all procs in current column

           vr(1:lr) = a(1:lr,lch) ! vector to be transformed

           if (my_prow==prow(nrow, nblk, np_rows)) then
             aux1(1) = dot_product(vr(1:lr-1),vr(1:lr-1))
             aux1(2) = vr(lr)
           else
             aux1(1) = dot_product(vr(1:lr),vr(1:lr))
             aux1(2) = 0.
           endif

           call mpi_allreduce(aux1,aux2,2,MPI_REAL8,MPI_SUM,mpi_comm_rows,mpierr)

           vnorm2 = aux2(1)
           vrl    = aux2(2)

           ! Householder transformation

           call hh_transform_real(vrl, vnorm2, xf, tau)

           ! Scale vr and store Householder vector for back transformation

           vr(1:lr) = vr(1:lr) * xf
           if (my_prow==prow(nrow, nblk, np_rows)) then
             a(1:lr-1,lch) = vr(1:lr-1)
             a(lr,lch) = vrl
             vr(lr) = 1.
           else
             a(1:lr,lch) = vr(1:lr)
           endif

         endif

         ! Broadcast Householder vector and tau along columns

         vr(lr+1) = tau
         call MPI_Bcast(vr,lr+1,MPI_REAL8,cur_pcol,mpi_comm_cols,mpierr)
         vmr(1:lr,lc) = vr(1:lr)
         tau = vr(lr+1)
         tmat(lc,lc,istep) = tau ! Store tau in diagonal of tmat

         ! Transform remaining columns in current block with Householder vector
         ! Local dot product

         aux1 = 0
#ifdef WITH_OPENMP
         !Open up one omp region to avoid paying openmp overhead.
         !This does not help performance due to the addition of two openmp barriers around the MPI call,
         !But in the future this may be beneficial if these barriers are replaced with a faster implementation

         !$omp parallel private(mynlc, j, lcx, ii, pp ) shared(aux1)
         mynlc = 0 ! number of local columns

         !This loop does not have independent iterations,
         !'mynlc' is incremented each iteration, and it is difficult to remove this dependency 
         !Thus each thread executes every iteration of the loop, except it only does the work if it 'owns' that iteration
         !That is, a thread only executes the work associated with an iteration if its thread id is congruent to 
         !the iteration number modulo the number of threads
         do j=1,lc-1
           lcx = local_index(istep*nbw+j, my_pcol, np_cols, nblk, 0)
           if (lcx>0 ) then
             mynlc = mynlc+1
             if ( mod((j-1), omp_get_num_threads()) .eq. omp_get_thread_num() ) then
                 if (lr>0) aux1(mynlc) = dot_product(vr(1:lr),a(1:lr,lcx))
             endif
           endif
         enddo
         
         ! Get global dot products
         !$omp barrier
         !$omp single 
         if (mynlc>0) call mpi_allreduce(aux1,aux2,mynlc,MPI_REAL8,MPI_SUM,mpi_comm_rows,mpierr)
         !$omp end single 
         !$omp barrier

         ! Transform
         transformChunkSize=32
         mynlc = 0
         do j=1,lc-1
           lcx = local_index(istep*nbw+j, my_pcol, np_cols, nblk, 0)
           if (lcx>0) then
             mynlc = mynlc+1
             !This loop could be parallelized with an openmp pragma with static scheduling and chunk size 32
             !However, for some reason this is slower than doing it manually, so it is parallelized as below.
             do ii=omp_get_thread_num()*transformChunkSize,lr,omp_get_num_threads()*transformChunkSize
                do pp = 1,transformChunkSize
                    if (pp + ii > lr) exit
                        a(ii+pp,lcx) = a(ii+pp,lcx) - tau*aux2(mynlc)*vr(ii+pp)
                enddo
             enddo
           endif
         enddo
         !$omp end parallel
#else
         nlc = 0 ! number of local columns
         do j=1,lc-1
           lcx = local_index(istep*nbw+j, my_pcol, np_cols, nblk, 0)
           if (lcx>0) then
             nlc = nlc+1
             if (lr>0) aux1(nlc) = dot_product(vr(1:lr),a(1:lr,lcx))
           endif
         enddo

         ! Get global dot products
         if (nlc>0) call mpi_allreduce(aux1,aux2,nlc,MPI_REAL8,MPI_SUM,mpi_comm_rows,mpierr)

         ! Transform

         nlc = 0
         do j=1,lc-1
           lcx = local_index(istep*nbw+j, my_pcol, np_cols, nblk, 0)
           if (lcx>0) then
             nlc = nlc+1
             a(1:lr,lcx) = a(1:lr,lcx) - tau*aux2(nlc)*vr(1:lr)
           endif
         enddo
#endif
       enddo

       ! Calculate scalar products of stored Householder vectors.
       ! This can be done in different ways, we use dsyrk

       vav = 0
       if (l_rows>0) &
           call dsyrk('U','T',n_cols,l_rows,1.d0,vmr,ubound(vmr,dim=1),0.d0,vav,ubound(vav,dim=1))
       call symm_matrix_allreduce(n_cols,vav, nbw, nbw,mpi_comm_rows)

       ! Calculate triangular matrix T for block Householder Transformation

       do lc=n_cols,1,-1
         tau = tmat(lc,lc,istep)
         if (lc<n_cols) then
           call dtrmv('U','T','N',n_cols-lc,tmat(lc+1,lc+1,istep),ubound(tmat,dim=1),vav(lc+1,lc),1)
           tmat(lc,lc+1:n_cols,istep) = -tau * vav(lc+1:n_cols,lc)
         endif
       enddo
     endif

    ! Transpose vmr -> vmc (stored in umc, second half)

    call elpa_transpose_vectors_real  (vmr, ubound(vmr,dim=1), mpi_comm_rows, &
                                    umc(1,n_cols+1), ubound(umc,dim=1), mpi_comm_cols, &
                                    1, istep*nbw, n_cols, nblk)

    ! Calculate umc = A**T * vmr
    ! Note that the distributed A has to be transposed
    ! Opposed to direct tridiagonalization there is no need to use the cache locality
    ! of the tiles, so we can use strips of the matrix
    !Code for Algorithm 4

    n_way = 1
#ifdef WITH_OPENMP
    n_way = omp_get_max_threads()
#endif
    !umc(1:l_cols,1:n_cols) = 0.d0
    !vmr(1:l_rows,n_cols+1:2*n_cols) = 0
#ifdef WITH_OPENMP
    !$omp parallel private( i,lcs,lce,lrs,lre)
#endif
    if(n_way > 1) then
        !$omp do
        do i=1,min(l_cols_tile, l_cols)
            umc(i,1:n_cols) = 0.d0
        enddo
        !$omp do
        do i=1,l_rows
            vmr(i,n_cols+1:2*n_cols) = 0.d0
        enddo
        if (l_cols>0 .and. l_rows>0) then

          !SYMM variant 4
          !Partitioned Matrix Expression:
          ! Ct = Atl Bt + Atr Bb
          ! Cb = Atr' Bt + Abl Bb
          !
          !Loop invariant:
          ! Ct = Atl Bt + Atr Bb
          !
          !Update:
          ! C1 = A10'B0 + A11B1 + A21 B2
          !
          !This algorithm chosen because in this algoirhtm, the loop around the dgemm calls
          !is easily parallelized, and regardless of choise of algorithm,
          !the startup cost for parallelizing the dgemms inside the loop is too great

          !$omp do schedule(static,1)
          do i=0,(istep*nbw-1)/tile_size
            lcs = i*l_cols_tile+1                   ! local column start
            lce = min(l_cols, (i+1)*l_cols_tile)    ! local column end

            lrs = i*l_rows_tile+1                   ! local row start
            lre = min(l_rows, (i+1)*l_rows_tile)    ! local row end

            !C1 += [A11 A12] [B1
            !                 B2]
            if( lre > lrs .and. l_cols > lcs ) then
            call DGEMM('N','N', lre-lrs+1, n_cols, l_cols-lcs+1,    &
                       1.d0, a(lrs,lcs), ubound(a,dim=1),           &
                             umc(lcs,n_cols+1), ubound(umc,dim=1),  &
                       0.d0, vmr(lrs,n_cols+1), ubound(vmr,dim=1))
            endif

            ! C1 += A10' B0
            if( lce > lcs .and. i > 0 ) then
            call DGEMM('T','N', lce-lcs+1, n_cols, lrs-1,           &
                       1.d0, a(1,lcs),   ubound(a,dim=1),           &
                             vmr(1,1),   ubound(vmr,dim=1),         &
                       0.d0, umc(lcs,1), ubound(umc,dim=1))
            endif
          enddo
        endif
    else
        umc(1:l_cols,1:n_cols) = 0.d0
        vmr(1:l_rows,n_cols+1:2*n_cols) = 0
        if (l_cols>0 .and. l_rows>0) then
          do i=0,(istep*nbw-1)/tile_size

            lcs = i*l_cols_tile+1
            lce = min(l_cols,(i+1)*l_cols_tile)
            if (lce<lcs) cycle

            lre = min(l_rows,(i+1)*l_rows_tile)
            call DGEMM('T','N',lce-lcs+1,n_cols,lre,1.d0,a(1,lcs),ubound(a,dim=1), &
                         vmr,ubound(vmr,dim=1),1.d0,umc(lcs,1),ubound(umc,dim=1))

            if (i==0) cycle
            lre = min(l_rows,i*l_rows_tile)
            call DGEMM('N','N',lre,n_cols,lce-lcs+1,1.d0,a(1,lcs),lda, &
                         umc(lcs,n_cols+1),ubound(umc,dim=1),1.d0,vmr(1,n_cols+1),ubound(vmr,dim=1))
          enddo
        endif
    endif
#ifdef WITH_OPENMP
    !$omp end parallel
#endif
    ! Sum up all ur(:) parts along rows and add them to the uc(:) parts
    ! on the processors containing the diagonal
    ! This is only necessary if ur has been calculated, i.e. if the
    ! global tile size is smaller than the global remaining matrix
    ! Or if we used the Algorithm 4
    if (tile_size < istep*nbw .or. n_way > 1) then
    call elpa_reduce_add_vectors_real  (vmr(1,n_cols+1),ubound(vmr,dim=1),mpi_comm_rows, &
                                        umc, ubound(umc,dim=1), mpi_comm_cols, &
                                        istep*nbw, n_cols, nblk)
    endif

    if (l_cols>0) then
      allocate(tmp(l_cols,n_cols))
      call mpi_allreduce(umc,tmp,l_cols*n_cols,MPI_REAL8,MPI_SUM,mpi_comm_rows,mpierr)
      umc(1:l_cols,1:n_cols) = tmp(1:l_cols,1:n_cols)
      deallocate(tmp)
    endif

    ! U = U * Tmat**T

    call dtrmm('Right','Upper','Trans','Nonunit',l_cols,n_cols,1.d0,tmat(1,1,istep),ubound(tmat,dim=1),umc,ubound(umc,dim=1))

    ! VAV = Tmat * V**T * A * V * Tmat**T = (U*Tmat**T)**T * V * Tmat**T

    call dgemm('T','N',n_cols,n_cols,l_cols,1.d0,umc,ubound(umc,dim=1),umc(1,n_cols+1), &
               ubound(umc,dim=1),0.d0,vav,ubound(vav,dim=1))
    call dtrmm('Right','Upper','Trans','Nonunit',n_cols,n_cols,1.d0,tmat(1,1,istep),    &
               ubound(tmat,dim=1),vav,ubound(vav,dim=1))

    call symm_matrix_allreduce(n_cols,vav, nbw, nbw ,mpi_comm_cols)

    ! U = U - 0.5 * V * VAV
    call dgemm('N','N',l_cols,n_cols,n_cols,-0.5d0,umc(1,n_cols+1),ubound(umc,dim=1),vav, &
                ubound(vav,dim=1),1.d0,umc,ubound(umc,dim=1))

    ! Transpose umc -> umr (stored in vmr, second half)

    call elpa_transpose_vectors_real  (umc, ubound(umc,dim=1), mpi_comm_cols, &
                                   vmr(1,n_cols+1), ubound(vmr,dim=1), mpi_comm_rows, &
                                   1, istep*nbw, n_cols, nblk)

    ! A = A - V*U**T - U*V**T
#ifdef WITH_OPENMP
    !$omp parallel private( ii, i, lcs, lce, lre, n_way, m_way, m_id, n_id, work_per_thread, mystart, myend  )
    n_threads = omp_get_num_threads()
    if(mod(n_threads, 2) == 0) then
        n_way = 2
    else
        n_way = 1
    endif

    m_way = n_threads / n_way

    m_id = mod(omp_get_thread_num(),  m_way)
    n_id = omp_get_thread_num() / m_way

    do ii=n_id*tile_size,(istep*nbw-1),tile_size*n_way
      i = ii / tile_size
      lcs = i*l_cols_tile+1
      lce = min(l_cols,(i+1)*l_cols_tile)
      lre = min(l_rows,(i+1)*l_rows_tile)
      if (lce<lcs .or. lre<1) cycle

      !Figure out this thread's range
      work_per_thread = lre / m_way
      if (work_per_thread * m_way < lre) work_per_thread = work_per_thread + 1
      mystart = m_id * work_per_thread + 1
      myend   = mystart + work_per_thread - 1
      if( myend > lre ) myend = lre
      if( myend-mystart+1 < 1) cycle

      call dgemm('N','T',myend-mystart+1, lce-lcs+1, 2*n_cols, -1.d0, &
                  vmr(mystart, 1), ubound(vmr,1), umc(lcs,1), ubound(umc,1), &
                  1.d0,a(mystart,lcs),ubound(a,1))
    enddo
    !$omp end parallel

#else /* WITH_OPENMP */
    do i=0,(istep*nbw-1)/tile_size
      lcs = i*l_cols_tile+1
      lce = min(l_cols,(i+1)*l_cols_tile)
      lre = min(l_rows,(i+1)*l_rows_tile)
      if (lce<lcs .or. lre<1) cycle
      call dgemm('N','T',lre,lce-lcs+1,2*n_cols,-1.d0, &
                  vmr,ubound(vmr,dim=1),umc(lcs,1),ubound(umc,dim=1), &
                  1.d0,a(1,lcs),lda)
    enddo
#endif /* WITH_OPENMP */
    deallocate(vmr, umc, vr)

  enddo

  if (useQR) then
    if (which_qr_decomposition == 1) then
      deallocate(work_blocked)
      deallocate(tauvector)
    endif
  endif

#ifdef HAVE_DETAILED_TIMINGS
  call timer%stop("bandred_real")
#endif
end subroutine bandred_real

!-------------------------------------------------------------------------------

subroutine symm_matrix_allreduce(n,a,lda,ldb,comm)

!-------------------------------------------------------------------------------
!  symm_matrix_allreduce: Does an mpi_allreduce for a symmetric matrix A.
!  On entry, only the upper half of A needs to be set
!  On exit, the complete matrix is set
!-------------------------------------------------------------------------------
#ifdef HAVE_DETAILED_TIMINGS
 use timings
#endif
   implicit none
   integer  :: n, lda, ldb, comm
   real*8   :: a(lda,ldb)

   integer  :: i, nc, mpierr
   real*8   :: h1(n*n), h2(n*n)

#ifdef HAVE_DETAILED_TIMINGS
  call timer%start("symm_matrix_allreduce")
#endif

   nc = 0
   do i=1,n
     h1(nc+1:nc+i) = a(1:i,i)
     nc = nc+i
   enddo

   call mpi_allreduce(h1,h2,nc,MPI_REAL8,MPI_SUM,comm,mpierr)

   nc = 0
   do i=1,n
     a(1:i,i) = h2(nc+1:nc+i)
     a(i,1:i-1) = a(1:i-1,i)
     nc = nc+i
   enddo

#ifdef HAVE_DETAILED_TIMINGS
  call timer%stop("symm_matrix_allreduce")
#endif

end subroutine symm_matrix_allreduce

!-------------------------------------------------------------------------------

subroutine trans_ev_band_to_full_real(na, nqc, nblk, nbw, a, lda, tmat, q, ldq, matrixCols, numBlocks, mpi_comm_rows, &
                                      mpi_comm_cols, useQR)


!-------------------------------------------------------------------------------
!  trans_ev_band_to_full_real:
!  Transforms the eigenvectors of a band matrix back to the eigenvectors of the original matrix
!
!  Parameters
!
!  na          Order of matrix a, number of rows of matrix q
!
!  nqc         Number of columns of matrix q
!
!  nblk        blocksize of cyclic distribution, must be the same in both directions!
!
!  nbw         semi bandwith
!
!  a(lda,matrixCols)    Matrix containing the Householder vectors (i.e. matrix a after bandred_real)
!              Distribution is like in Scalapack.
!
!  lda         Leading dimension of a
!  matrixCols  local columns of matrix a and q
!
!  tmat(nbw,nbw,numBlocks) Factors returned by bandred_real
!
!  q           On input: Eigenvectors of band matrix
!              On output: Transformed eigenvectors
!              Distribution is like in Scalapack.
!
!  ldq         Leading dimension of q
!
!  mpi_comm_rows
!  mpi_comm_cols
!              MPI-Communicators for rows/columns
!
!-------------------------------------------------------------------------------
#ifdef HAVE_DETAILED_TIMINGS
 use timings
#endif
   implicit none

   integer              :: na, nqc, lda, ldq, nblk, nbw, matrixCols, numBlocks, mpi_comm_rows, mpi_comm_cols
   real*8               :: a(lda,matrixCols), q(ldq,matrixCols), tmat(nbw, nbw, numBlocks)

   integer              :: my_prow, my_pcol, np_rows, np_cols, mpierr
   integer              :: max_blocks_row, max_blocks_col, max_local_rows, &
                           max_local_cols
   integer              :: l_cols, l_rows, l_colh, n_cols
   integer              :: istep, lc, ncol, nrow, nb, ns

   real*8, allocatable  :: tmp1(:), tmp2(:), hvb(:), hvm(:,:)

   integer              :: i

   real*8, allocatable  :: tmat_complete(:,:), t_tmp(:,:), t_tmp2(:,:)
   integer              :: cwy_blocking, t_blocking, t_cols, t_rows
   logical, intent(in)  :: useQR

#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("trans_ev_band_to_full_real")
#endif

   call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
   call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
   call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
   call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)

   max_blocks_row = ((na -1)/nblk)/np_rows + 1  ! Rows of A
   max_blocks_col = ((nqc-1)/nblk)/np_cols + 1  ! Columns of q!

   max_local_rows = max_blocks_row*nblk
   max_local_cols = max_blocks_col*nblk


! This conditional was introduced due to an merge error. For better performance this code path should
! always be used
!   if (useQR) then

     ! t_blocking was formerly 2; 3 is a better choice
     t_blocking = 3 ! number of matrices T (tmat) which are aggregated into a new (larger) T matrix (tmat_complete) and applied at once
     cwy_blocking = t_blocking * nbw

     allocate(tmp1(max_local_cols*cwy_blocking))
     allocate(tmp2(max_local_cols*cwy_blocking))
     allocate(hvb(max_local_rows*cwy_blocking))
     allocate(hvm(max_local_rows,cwy_blocking))
     allocate(tmat_complete(cwy_blocking,cwy_blocking))
     allocate(t_tmp(cwy_blocking,nbw))
     allocate(t_tmp2(cwy_blocking,nbw))
!   else
!     allocate(tmp1(max_local_cols*nbw))
!     allocate(tmp2(max_local_cols*nbw))
!     allocate(hvb(max_local_rows*nbw))
!     allocate(hvm(max_local_rows,nbw))
!   endif

   hvm = 0   ! Must be set to 0 !!!
   hvb = 0   ! Safety only

   l_cols = local_index(nqc, my_pcol, np_cols, nblk, -1) ! Local columns of q

! This conditional has been introduced by the same merge error. Execute always this code path
!   if (useQR) then

     do istep=1,((na-1)/nbw-1)/t_blocking + 1
       n_cols = MIN(na,istep*cwy_blocking+nbw) - (istep-1)*cwy_blocking - nbw ! Number of columns in current step

       ! Broadcast all Householder vectors for current step compressed in hvb

       nb = 0
       ns = 0

       do lc = 1, n_cols
         ncol = (istep-1)*cwy_blocking + nbw + lc ! absolute column number of householder vector
         nrow = ncol - nbw ! absolute number of pivot row

         l_rows = local_index(nrow-1, my_prow, np_rows, nblk, -1) ! row length for bcast
         l_colh = local_index(ncol  , my_pcol, np_cols, nblk, -1) ! HV local column number

         if (my_pcol==pcol(ncol, nblk, np_cols)) hvb(nb+1:nb+l_rows) = a(1:l_rows,l_colh)

         nb = nb+l_rows

         if (lc==n_cols .or. mod(ncol,nblk)==0) then
           call MPI_Bcast(hvb(ns+1),nb-ns,MPI_REAL8,pcol(ncol, nblk, np_cols),mpi_comm_cols,mpierr)
           ns = nb
         endif
       enddo

       ! Expand compressed Householder vectors into matrix hvm

       nb = 0
       do lc = 1, n_cols
         nrow = (istep-1)*cwy_blocking + lc ! absolute number of pivot row
         l_rows = local_index(nrow-1, my_prow, np_rows, nblk, -1) ! row length for bcast

         hvm(1:l_rows,lc) = hvb(nb+1:nb+l_rows)
         if (my_prow==prow(nrow, nblk, np_rows)) hvm(l_rows+1,lc) = 1.

         nb = nb+l_rows
       enddo

       l_rows = local_index(MIN(na,(istep+1)*cwy_blocking), my_prow, np_rows, nblk, -1)

       ! compute tmat2 out of tmat(:,:,)
       tmat_complete = 0
       do i = 1, t_blocking
         t_cols = MIN(nbw, n_cols - (i-1)*nbw)
         if (t_cols <= 0) exit
         t_rows = (i - 1) * nbw
         tmat_complete(t_rows+1:t_rows+t_cols,t_rows+1:t_rows+t_cols) = tmat(1:t_cols,1:t_cols,(istep-1)*t_blocking + i)
         if (i > 1) then
           call dgemm('T', 'N', t_rows, t_cols, l_rows, 1.d0, hvm(1,1), max_local_rows, hvm(1,(i-1)*nbw+1), &
                     max_local_rows, 0.d0, t_tmp, cwy_blocking)
           call mpi_allreduce(t_tmp,t_tmp2,cwy_blocking*nbw,MPI_REAL8,MPI_SUM,mpi_comm_rows,mpierr)
           call dtrmm('L','U','N','N',t_rows,t_cols,1.0d0,tmat_complete,cwy_blocking,t_tmp2,cwy_blocking)
           call dtrmm('R','U','N','N',t_rows,t_cols,-1.0d0,tmat_complete(t_rows+1,t_rows+1),cwy_blocking,t_tmp2,cwy_blocking)
           tmat_complete(1:t_rows,t_rows+1:t_rows+t_cols) = t_tmp2(1:t_rows,1:t_cols)
         endif
       enddo

       ! Q = Q - V * T**T * V**T * Q

       if (l_rows>0) then
         call dgemm('T','N',n_cols,l_cols,l_rows,1.d0,hvm,ubound(hvm,dim=1), &
                    q,ldq,0.d0,tmp1,n_cols)
       else
         tmp1(1:l_cols*n_cols) = 0
       endif
       call mpi_allreduce(tmp1,tmp2,n_cols*l_cols,MPI_REAL8,MPI_SUM,mpi_comm_rows,mpierr)


       if (l_rows>0) then
         call dtrmm('L','U','T','N',n_cols,l_cols,1.0d0,tmat_complete,cwy_blocking,tmp2,n_cols)
         call dgemm('N','N',l_rows,l_cols,n_cols,-1.d0,hvm,ubound(hvm,dim=1), tmp2,n_cols,1.d0,q,ldq)
       endif
     enddo

!   else !  do not useQR
!
!     do istep=1,(na-1)/nbw
!
!       n_cols = MIN(na,(istep+1)*nbw) - istep*nbw ! Number of columns in current step
!
!       ! Broadcast all Householder vectors for current step compressed in hvb
!
!       nb = 0
!       ns = 0
!
!       do lc = 1, n_cols
!         ncol = istep*nbw + lc ! absolute column number of householder vector
!         nrow = ncol - nbw ! absolute number of pivot row
!
!         l_rows = local_index(nrow-1, my_prow, np_rows, nblk, -1) ! row length for bcast
!         l_colh = local_index(ncol  , my_pcol, np_cols, nblk, -1) ! HV local column number
!
!         if (my_pcol==pcol(ncol, nblk, np_cols)) hvb(nb+1:nb+l_rows) = a(1:l_rows,l_colh)
!
!         nb = nb+l_rows
!
!         if (lc==n_cols .or. mod(ncol,nblk)==0) then
!           call MPI_Bcast(hvb(ns+1),nb-ns,MPI_REAL8,pcol(ncol, nblk, np_cols),mpi_comm_cols,mpierr)
!           ns = nb
!         endif
!       enddo
!
!       ! Expand compressed Householder vectors into matrix hvm
!
!       nb = 0
!       do lc = 1, n_cols
!         nrow = (istep-1)*nbw+lc ! absolute number of pivot row
!         l_rows = local_index(nrow-1, my_prow, np_rows, nblk, -1) ! row length for bcast
!
!         hvm(1:l_rows,lc) = hvb(nb+1:nb+l_rows)
!         if (my_prow==prow(nrow, nblk, np_rows)) hvm(l_rows+1,lc) = 1.
!
!         nb = nb+l_rows
!       enddo
!
!       l_rows = local_index(MIN(na,(istep+1)*nbw), my_prow, np_rows, nblk, -1)
!
!       ! Q = Q - V * T**T * V**T * Q
!
!       if (l_rows>0) then
!         call dgemm('T','N',n_cols,l_cols,l_rows,1.d0,hvm,ubound(hvm,dim=1), &
!                    q,ldq,0.d0,tmp1,n_cols)
!       else
!         tmp1(1:l_cols*n_cols) = 0
!       endif
!
!       call mpi_allreduce(tmp1,tmp2,n_cols*l_cols,MPI_REAL8,MPI_SUM,mpi_comm_rows,mpierr)
!
!       if (l_rows>0) then
!         call dtrmm('L','U','T','N',n_cols,l_cols,1.0d0,tmat(1,1,istep),ubound(tmat,dim=1),tmp2,n_cols)
!         call dgemm('N','N',l_rows,l_cols,n_cols,-1.d0,hvm,ubound(hvm,dim=1), &
!                    tmp2,n_cols,1.d0,q,ldq)
!       endif
!     enddo
!   endif ! endQR

   deallocate(tmp1, tmp2, hvb, hvm)
!   if (useQr) then
     deallocate(tmat_complete, t_tmp, t_tmp2)
!   endif

#ifdef HAVE_DETAILED_TIMINGS
   call timer%stop("trans_ev_band_to_full_real")
#endif
end subroutine trans_ev_band_to_full_real

! --------------------------------------------------------------------------------------------------

subroutine tridiag_band_real(na, nb, nblk, a, lda, d, e, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm)

!-------------------------------------------------------------------------------
! tridiag_band_real:
! Reduces a real symmetric band matrix to tridiagonal form
!
!  na          Order of matrix a
!
!  nb          Semi bandwith
!
!  nblk        blocksize of cyclic distribution, must be the same in both directions!
!
!  a(lda,matrixCols)    Distributed system matrix reduced to banded form in the upper diagonal
!
!  lda         Leading dimension of a
!  matrixCols  local columns of matrix a
!
!  d(na)       Diagonal of tridiagonal matrix, set only on PE 0 (output)
!
!  e(na)       Subdiagonal of tridiagonal matrix, set only on PE 0 (output)
!
!  mpi_comm_rows
!  mpi_comm_cols
!              MPI-Communicators for rows/columns
!  mpi_comm
!              MPI-Communicator for the total processor set
!-------------------------------------------------------------------------------
#ifdef HAVE_DETAILED_TIMINGS
 use timings
#endif
   implicit none

   integer, intent(in) ::  na, nb, nblk, lda, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm
   real*8, intent(in)  :: a(lda,matrixCols)
   real*8, intent(out) :: d(na), e(na) ! set only on PE 0


   real*8 vnorm2, hv(nb), tau, x, h(nb), ab_s(1+nb), hv_s(nb), hv_new(nb), tau_new, hf
   real*8 hd(nb), hs(nb)

   integer i, j, n, nc, nr, ns, ne, istep, iblk, nblocks_total, nblocks, nt
   integer my_pe, n_pes, mpierr
   integer my_prow, np_rows, my_pcol, np_cols
   integer ireq_ab, ireq_hv
   integer na_s, nx, num_hh_vecs, num_chunks, local_size, max_blk_size, n_off
#ifdef WITH_OPENMP
   integer max_threads, my_thread, my_block_s, my_block_e, iter
   integer mpi_status(MPI_STATUS_SIZE)
   integer, allocatable :: mpi_statuses(:,:), global_id_tmp(:,:)
   integer, allocatable :: omp_block_limits(:)
   real*8, allocatable :: hv_t(:,:), tau_t(:)
#endif
   integer, allocatable :: ireq_hhr(:), ireq_hhs(:), global_id(:,:), hh_cnt(:), hh_dst(:)
   integer, allocatable :: limits(:), snd_limits(:,:)
   integer, allocatable :: block_limits(:)
   real*8, allocatable :: ab(:,:), hh_gath(:,:,:), hh_send(:,:,:)
!   ! dummies for calling redist_band
!   complex*16 :: c_a(1,1), c_ab(1,1)

#ifdef WITH_OPENMP
   integer :: omp_get_max_threads
#endif

#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("tridiag_band_real")
#endif

   call mpi_comm_rank(mpi_comm,my_pe,mpierr)
   call mpi_comm_size(mpi_comm,n_pes,mpierr)

   call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
   call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
   call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
   call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)

   ! Get global_id mapping 2D procssor coordinates to global id

   allocate(global_id(0:np_rows-1,0:np_cols-1))
   global_id(:,:) = 0
   global_id(my_prow, my_pcol) = my_pe
#ifdef WITH_OPENMP
   allocate(global_id_tmp(0:np_rows-1,0:np_cols-1))
#endif

#ifndef WITH_OPENMP
   call mpi_allreduce(mpi_in_place, global_id, np_rows*np_cols, mpi_integer, mpi_sum, mpi_comm, mpierr)
#else
    global_id_tmp(:,:) = global_id(:,:)
    call mpi_allreduce(global_id_tmp, global_id, np_rows*np_cols, mpi_integer, mpi_sum, mpi_comm, mpierr)
    deallocate(global_id_tmp)
#endif

   ! Total number of blocks in the band:

   nblocks_total = (na-1)/nb + 1

   ! Set work distribution

   allocate(block_limits(0:n_pes))
   call divide_band(nblocks_total, n_pes, block_limits)

   ! nblocks: the number of blocks for my task
   nblocks = block_limits(my_pe+1) - block_limits(my_pe)

   ! allocate the part of the band matrix which is needed by this PE
   ! The size is 1 block larger than needed to avoid extensive shifts
   allocate(ab(2*nb,(nblocks+1)*nb))
   ab = 0 ! needed for lower half, the extra block should also be set to 0 for safety

   ! n_off: Offset of ab within band
   n_off = block_limits(my_pe)*nb

   ! Redistribute band in a to ab
   call redist_band_real(a, lda, na, nblk, nb, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm, ab)

   ! Calculate the workload for each sweep in the back transformation
   ! and the space requirements to hold the HH vectors

   allocate(limits(0:np_rows))
   call determine_workload(na, nb, np_rows, limits)
   max_blk_size = maxval(limits(1:np_rows) - limits(0:np_rows-1))

   num_hh_vecs = 0
   num_chunks  = 0
   nx = na
   do n = 1, nblocks_total
     call determine_workload(nx, nb, np_rows, limits)
     local_size = limits(my_prow+1) - limits(my_prow)
     ! add to number of householder vectors
     ! please note: for nx==1 the one and only HH vector is 0 and is neither calculated nor send below!
     if (mod(n-1,np_cols) == my_pcol .and. local_size>0 .and. nx>1) then
       num_hh_vecs = num_hh_vecs + local_size
       num_chunks  = num_chunks+1
     endif
     nx = nx - nb
   enddo

   ! Allocate space for HH vectors

   allocate(hh_trans_real(nb,num_hh_vecs))

   ! Allocate and init MPI requests

   allocate(ireq_hhr(num_chunks)) ! Recv requests
   allocate(ireq_hhs(nblocks))    ! Send requests

   num_hh_vecs = 0
   num_chunks  = 0
   nx = na
   nt = 0
   do n = 1, nblocks_total
     call determine_workload(nx, nb, np_rows, limits)
     local_size = limits(my_prow+1) - limits(my_prow)
     if (mod(n-1,np_cols) == my_pcol .and. local_size>0 .and. nx>1) then
       num_chunks  = num_chunks+1
       call mpi_irecv(hh_trans_real(1,num_hh_vecs+1), nb*local_size, mpi_real8, nt, &
                        10+n-block_limits(nt), mpi_comm, ireq_hhr(num_chunks), mpierr)
       num_hh_vecs = num_hh_vecs + local_size
     endif
     nx = nx - nb
     if (n == block_limits(nt+1)) then
       nt = nt + 1
     endif
   enddo

   ireq_hhs(:) = MPI_REQUEST_NULL

   ! Buffers for gathering/sending the HH vectors

   allocate(hh_gath(nb,max_blk_size,nblocks)) ! gathers HH vectors
   allocate(hh_send(nb,max_blk_size,nblocks)) ! send buffer for HH vectors
   hh_gath(:,:,:) = 0
   hh_send(:,:,:) = 0

   ! Some counters

   allocate(hh_cnt(nblocks))
   allocate(hh_dst(nblocks))

   hh_cnt(:) = 1 ! The first transfomation vector is always 0 and not calculated at all
   hh_dst(:) = 0 ! PE number for receive

   ireq_ab = MPI_REQUEST_NULL
   ireq_hv = MPI_REQUEST_NULL

   ! Limits for sending

   allocate(snd_limits(0:np_rows,nblocks))

   do iblk=1,nblocks
     call determine_workload(na-(iblk+block_limits(my_pe)-1)*nb, nb, np_rows, snd_limits(:,iblk))
   enddo

#ifdef WITH_OPENMP
   ! OpenMP work distribution:

   max_threads = 1
   max_threads = omp_get_max_threads()

   ! For OpenMP we need at least 2 blocks for every thread
   max_threads = MIN(max_threads, nblocks/2)
   if (max_threads==0) max_threads = 1

   allocate(omp_block_limits(0:max_threads))

   ! Get the OpenMP block limits
   call divide_band(nblocks, max_threads, omp_block_limits)

   allocate(hv_t(nb,max_threads), tau_t(max_threads))
   hv_t = 0
   tau_t = 0
#endif

   ! ---------------------------------------------------------------------------
   ! Start of calculations

   na_s = block_limits(my_pe)*nb + 1

   if (my_pe>0 .and. na_s<=na) then
     ! send first column to previous PE
     ! Only the PE owning the diagonal does that (sending 1 element of the subdiagonal block also)
     ab_s(1:nb+1) = ab(1:nb+1,na_s-n_off)
     call mpi_isend(ab_s,nb+1,mpi_real8,my_pe-1,1,mpi_comm,ireq_ab,mpierr)
   endif

#ifdef WITH_OPENMP
   do istep=1,na-1-block_limits(my_pe)*nb
#else
   do istep=1,na-1
#endif

     if (my_pe==0) then
       n = MIN(na-na_s,nb) ! number of rows to be reduced
       hv(:) = 0
       tau = 0
       ! The last step (istep=na-1) is only needed for sending the last HH vectors.
       ! We don't want the sign of the last element flipped (analogous to the other sweeps)
       if (istep < na-1) then
         ! Transform first column of remaining matrix
         vnorm2 = sum(ab(3:n+1,na_s-n_off)**2)
         call hh_transform_real(ab(2,na_s-n_off),vnorm2,hf,tau)
         hv(1) = 1
         hv(2:n) = ab(3:n+1,na_s-n_off)*hf
       endif
       d(istep) = ab(1,na_s-n_off)
       e(istep) = ab(2,na_s-n_off)
       if (istep == na-1) then
         d(na) = ab(1,na_s+1-n_off)
         e(na) = 0
       endif
     else
       if (na>na_s) then
         ! Receive Householder vector from previous task, from PE owning subdiagonal
#ifdef WITH_OPENMP
         call mpi_recv(hv,nb,mpi_real8,my_pe-1,2,mpi_comm,MPI_STATUS,mpierr)
#else
         call mpi_recv(hv,nb,mpi_real8,my_pe-1,2,mpi_comm,MPI_STATUS_IGNORE,mpierr)
#endif
         tau = hv(1)
         hv(1) = 1.
       endif
     endif

     na_s = na_s+1
     if (na_s-n_off > nb) then
       ab(:,1:nblocks*nb) = ab(:,nb+1:(nblocks+1)*nb)
       ab(:,nblocks*nb+1:(nblocks+1)*nb) = 0
       n_off = n_off + nb
     endif


#ifdef WITH_OPENMP
     if (max_threads > 1) then

       ! Codepath for OpenMP

       ! Please note that in this case it is absolutely necessary to have at least 2 blocks per thread!
       ! Every thread is one reduction cycle behind its predecessor and thus starts one step later.
       ! This simulates the behaviour of the MPI tasks which also work after each other.
       ! The code would be considerably easier, if the MPI communication would be made within
       ! the parallel region - this is avoided here since this would require
       ! MPI_Init_thread(MPI_THREAD_MULTIPLE) at the start of the program.

       hv_t(:,1) = hv
       tau_t(1) = tau

       do iter = 1, 2

         ! iter=1 : work on first block
         ! iter=2 : work on remaining blocks
         ! This is done in 2 iterations so that we have a barrier in between:
         ! After the first iteration, it is guaranteed that the last row of the last block
         ! is completed by the next thread.
         ! After the first iteration it is also the place to exchange the last row
         ! with MPI calls
#ifdef HAVE_DETAILED_TIMINGS
         call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread, my_block_s, my_block_e, iblk, ns, ne, hv, tau, &
!$omp&                    nc, nr, hs, hd, vnorm2, hf, x, h, i), schedule(static,1), num_threads(max_threads)
         do my_thread = 1, max_threads

           if (iter == 1) then
             my_block_s = omp_block_limits(my_thread-1) + 1
             my_block_e = my_block_s
           else
             my_block_s = omp_block_limits(my_thread-1) + 2
             my_block_e = omp_block_limits(my_thread)
           endif

           do iblk = my_block_s, my_block_e

             ns = na_s + (iblk-1)*nb - n_off - my_thread + 1 ! first column in block
             ne = ns+nb-1                    ! last column in block

             if (istep<my_thread .or. ns+n_off>na) exit

             hv = hv_t(:,my_thread)
             tau = tau_t(my_thread)

             ! Store Householder vector for back transformation

             hh_cnt(iblk) = hh_cnt(iblk) + 1

             hh_gath(1   ,hh_cnt(iblk),iblk) = tau
             hh_gath(2:nb,hh_cnt(iblk),iblk) = hv(2:nb)

             nc = MIN(na-ns-n_off+1,nb) ! number of columns in diagonal block
             nr = MIN(na-nb-ns-n_off+1,nb) ! rows in subdiagonal block (may be < 0!!!)
                                       ! Note that nr>=0 implies that diagonal block is full (nc==nb)!

             ! Transform diagonal block

             call DSYMV('L',nc,tau,ab(1,ns),2*nb-1,hv,1,0.d0,hd,1)

             x = dot_product(hv(1:nc),hd(1:nc))*tau
             hd(1:nc) = hd(1:nc) - 0.5*x*hv(1:nc)

             call DSYR2('L',nc,-1.d0,hd,1,hv,1,ab(1,ns),2*nb-1)

             hv_t(:,my_thread) = 0
             tau_t(my_thread)  = 0

             if (nr<=0) cycle ! No subdiagonal block present any more

             ! Transform subdiagonal block

             call DGEMV('N',nr,nb,tau,ab(nb+1,ns),2*nb-1,hv,1,0.d0,hs,1)

             if (nr>1) then

               ! complete (old) Householder transformation for first column

               ab(nb+1:nb+nr,ns) = ab(nb+1:nb+nr,ns) - hs(1:nr) ! Note: hv(1) == 1

               ! calculate new Householder transformation for first column
               ! (stored in hv_t(:,my_thread) and tau_t(my_thread))

               vnorm2 = sum(ab(nb+2:nb+nr,ns)**2)
               call hh_transform_real(ab(nb+1,ns),vnorm2,hf,tau_t(my_thread))
               hv_t(1   ,my_thread) = 1.
               hv_t(2:nr,my_thread) = ab(nb+2:nb+nr,ns)*hf
               ab(nb+2:,ns) = 0

               ! update subdiagonal block for old and new Householder transformation
               ! This way we can use a nonsymmetric rank 2 update which is (hopefully) faster

               call DGEMV('T',nr,nb-1,tau_t(my_thread),ab(nb,ns+1),2*nb-1,hv_t(1,my_thread),1,0.d0,h(2),1)
               x = dot_product(hs(1:nr),hv_t(1:nr,my_thread))*tau_t(my_thread)
               h(2:nb) = h(2:nb) - x*hv(2:nb)
               ! Unfortunately there is no BLAS routine like DSYR2 for a nonsymmetric rank 2 update ("DGER2")
               do i=2,nb
                 ab(2+nb-i:1+nb+nr-i,i+ns-1) = ab(2+nb-i:1+nb+nr-i,i+ns-1) - hv_t(1:nr,my_thread)*h(i) - hs(1:nr)*hv(i)
               enddo

             else

               ! No new Householder transformation for nr=1, just complete the old one
               ab(nb+1,ns) = ab(nb+1,ns) - hs(1) ! Note: hv(1) == 1
               do i=2,nb
                 ab(2+nb-i,i+ns-1) = ab(2+nb-i,i+ns-1) - hs(1)*hv(i)
               enddo
               ! For safety: there is one remaining dummy transformation (but tau is 0 anyways)
               hv_t(1,my_thread) = 1.

             endif

           enddo

         enddo ! my_thread
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
         call timer%stop("OpenMP parallel")
#endif

         if (iter==1) then
           ! We are at the end of the first block

           ! Send our first column to previous PE
           if (my_pe>0 .and. na_s <= na) then
             call mpi_wait(ireq_ab,mpi_status,mpierr)
             ab_s(1:nb+1) = ab(1:nb+1,na_s-n_off)
             call mpi_isend(ab_s,nb+1,mpi_real8,my_pe-1,1,mpi_comm,ireq_ab,mpierr)
           endif

           ! Request last column from next PE
           ne = na_s + nblocks*nb - (max_threads-1) - 1
           if (istep>=max_threads .and. ne <= na) then
             call mpi_recv(ab(1,ne-n_off),nb+1,mpi_real8,my_pe+1,1,mpi_comm,mpi_status,mpierr)
           endif

         else
           ! We are at the end of all blocks

           ! Send last HH vector and TAU to next PE if it has been calculated above
           ne = na_s + nblocks*nb - (max_threads-1) - 1
           if (istep>=max_threads .and. ne < na) then
             call mpi_wait(ireq_hv,mpi_status,mpierr)
             hv_s(1) = tau_t(max_threads)
             hv_s(2:) = hv_t(2:,max_threads)
             call mpi_isend(hv_s,nb,mpi_real8,my_pe+1,2,mpi_comm,ireq_hv,mpierr)
           endif

           ! "Send" HH vector and TAU to next OpenMP thread
           do my_thread = max_threads, 2, -1
             hv_t(:,my_thread) = hv_t(:,my_thread-1)
             tau_t(my_thread)  = tau_t(my_thread-1)
           enddo

         endif
       enddo ! iter

     else

       ! Codepath for 1 thread without OpenMP

       ! The following code is structured in a way to keep waiting times for
       ! other PEs at a minimum, especially if there is only one block.
       ! For this reason, it requests the last column as late as possible
       ! and sends the Householder vector and the first column as early
       ! as possible.

#endif /* WITH_OPENMP */

       do iblk=1,nblocks

         ns = na_s + (iblk-1)*nb - n_off ! first column in block
         ne = ns+nb-1                    ! last column in block

         if (ns+n_off>na) exit

         ! Store Householder vector for back transformation

         hh_cnt(iblk) = hh_cnt(iblk) + 1

         hh_gath(1   ,hh_cnt(iblk),iblk) = tau
         hh_gath(2:nb,hh_cnt(iblk),iblk) = hv(2:nb)

#ifndef WITH_OPENMP
         if (hh_cnt(iblk) == snd_limits(hh_dst(iblk)+1,iblk)-snd_limits(hh_dst(iblk),iblk)) then
           ! Wait for last transfer to finish

           call mpi_wait(ireq_hhs(iblk), MPI_STATUS_IGNORE, mpierr)

           ! Copy vectors into send buffer
           hh_send(:,1:hh_cnt(iblk),iblk) = hh_gath(:,1:hh_cnt(iblk),iblk)
           ! Send to destination
           call mpi_isend(hh_send(1,1,iblk), nb*hh_cnt(iblk), mpi_real8, &
                        global_id(hh_dst(iblk),mod(iblk+block_limits(my_pe)-1,np_cols)), &
                        10+iblk, mpi_comm, ireq_hhs(iblk), mpierr)
         ! Reset counter and increase destination row
           hh_cnt(iblk) = 0
           hh_dst(iblk) = hh_dst(iblk)+1
         endif

         ! The following code is structured in a way to keep waiting times for
         ! other PEs at a minimum, especially if there is only one block.
         ! For this reason, it requests the last column as late as possible
         ! and sends the Householder vector and the first column as early
         ! as possible.
#endif
         nc = MIN(na-ns-n_off+1,nb) ! number of columns in diagonal block
         nr = MIN(na-nb-ns-n_off+1,nb) ! rows in subdiagonal block (may be < 0!!!)
                                       ! Note that nr>=0 implies that diagonal block is full (nc==nb)!

         ! Multiply diagonal block and subdiagonal block with Householder vector

         if (iblk==nblocks .and. nc==nb) then

           ! We need the last column from the next PE.
           ! First do the matrix multiplications without last column ...

           ! Diagonal block, the contribution of the last element is added below!
           ab(1,ne) = 0
           call DSYMV('L',nc,tau,ab(1,ns),2*nb-1,hv,1,0.d0,hd,1)

           ! Subdiagonal block
           if (nr>0) call DGEMV('N',nr,nb-1,tau,ab(nb+1,ns),2*nb-1,hv,1,0.d0,hs,1)

           ! ... then request last column ...
#ifdef WITH_OPENMP
           call mpi_recv(ab(1,ne),nb+1,mpi_real8,my_pe+1,1,mpi_comm,MPI_STATUS,mpierr)
#else
           call mpi_recv(ab(1,ne),nb+1,mpi_real8,my_pe+1,1,mpi_comm,MPI_STATUS_IGNORE,mpierr)
#endif

           ! ... and complete the result
           hs(1:nr) = hs(1:nr) + ab(2:nr+1,ne)*tau*hv(nb)
           hd(nb) = hd(nb) + ab(1,ne)*hv(nb)*tau

         else

           ! Normal matrix multiply
           call DSYMV('L',nc,tau,ab(1,ns),2*nb-1,hv,1,0.d0,hd,1)
           if (nr>0) call DGEMV('N',nr,nb,tau,ab(nb+1,ns),2*nb-1,hv,1,0.d0,hs,1)

         endif

         ! Calculate first column of subdiagonal block and calculate new
         ! Householder transformation for this column

         hv_new(:) = 0 ! Needed, last rows must be 0 for nr < nb
         tau_new = 0

         if (nr>0) then

           ! complete (old) Householder transformation for first column

           ab(nb+1:nb+nr,ns) = ab(nb+1:nb+nr,ns) - hs(1:nr) ! Note: hv(1) == 1

           ! calculate new Householder transformation ...
           if (nr>1) then
             vnorm2 = sum(ab(nb+2:nb+nr,ns)**2)
             call hh_transform_real(ab(nb+1,ns),vnorm2,hf,tau_new)
             hv_new(1) = 1.
             hv_new(2:nr) = ab(nb+2:nb+nr,ns)*hf
             ab(nb+2:,ns) = 0
           endif

           ! ... and send it away immediatly if this is the last block

           if (iblk==nblocks) then
#ifdef WITH_OPENMP
             call mpi_wait(ireq_hv,MPI_STATUS,mpierr)
#else
             call mpi_wait(ireq_hv,MPI_STATUS_IGNORE,mpierr)
#endif
             hv_s(1) = tau_new
             hv_s(2:) = hv_new(2:)
             call mpi_isend(hv_s,nb,mpi_real8,my_pe+1,2,mpi_comm,ireq_hv,mpierr)
           endif

         endif

         ! Transform diagonal block
         x = dot_product(hv(1:nc),hd(1:nc))*tau
         hd(1:nc) = hd(1:nc) - 0.5*x*hv(1:nc)

         if (my_pe>0 .and. iblk==1) then

           ! The first column of the diagonal block has to be send to the previous PE
           ! Calculate first column only ...

           ab(1:nc,ns) = ab(1:nc,ns) - hd(1:nc)*hv(1) - hv(1:nc)*hd(1)

           ! ... send it away ...

#ifdef WITH_OPENMP
           call mpi_wait(ireq_ab,MPI_STATUS,mpierr)
#else
           call mpi_wait(ireq_ab,MPI_STATUS_IGNORE,mpierr)
#endif
           ab_s(1:nb+1) = ab(1:nb+1,ns)
           call mpi_isend(ab_s,nb+1,mpi_real8,my_pe-1,1,mpi_comm,ireq_ab,mpierr)

           ! ... and calculate remaining columns with rank-2 update
           if (nc>1) call DSYR2('L',nc-1,-1.d0,hd(2),1,hv(2),1,ab(1,ns+1),2*nb-1)
         else
           ! No need to  send, just a rank-2 update
           call DSYR2('L',nc,-1.d0,hd,1,hv,1,ab(1,ns),2*nb-1)
         endif

         ! Do the remaining double Householder transformation on the subdiagonal block cols 2 ... nb

         if (nr>0) then
           if (nr>1) then
             call DGEMV('T',nr,nb-1,tau_new,ab(nb,ns+1),2*nb-1,hv_new,1,0.d0,h(2),1)
             x = dot_product(hs(1:nr),hv_new(1:nr))*tau_new
             h(2:nb) = h(2:nb) - x*hv(2:nb)
             ! Unfortunately there is no BLAS routine like DSYR2 for a nonsymmetric rank 2 update
             do i=2,nb
               ab(2+nb-i:1+nb+nr-i,i+ns-1) = ab(2+nb-i:1+nb+nr-i,i+ns-1) - hv_new(1:nr)*h(i) - hs(1:nr)*hv(i)
             enddo
           else
             ! No double Householder transformation for nr=1, just complete the row
             do i=2,nb
               ab(2+nb-i,i+ns-1) = ab(2+nb-i,i+ns-1) - hs(1)*hv(i)
             enddo
           endif
         endif

         ! Use new HH vector for the next block
         hv(:) = hv_new(:)
         tau = tau_new

       enddo

#ifdef WITH_OPENMP
     endif


     do iblk = 1, nblocks

      if (hh_dst(iblk) >= np_rows) exit
      if (snd_limits(hh_dst(iblk)+1,iblk) == snd_limits(hh_dst(iblk),iblk)) exit

      if (hh_cnt(iblk) == snd_limits(hh_dst(iblk)+1,iblk)-snd_limits(hh_dst(iblk),iblk)) then
        ! Wait for last transfer to finish
        call mpi_wait(ireq_hhs(iblk), mpi_status, mpierr)
        ! Copy vectors into send buffer
        hh_send(:,1:hh_cnt(iblk),iblk) = hh_gath(:,1:hh_cnt(iblk),iblk)
        ! Send to destination
        call mpi_isend(hh_send(1,1,iblk), nb*hh_cnt(iblk), mpi_real8, &
              global_id(hh_dst(iblk),mod(iblk+block_limits(my_pe)-1,np_cols)), &
              10+iblk, mpi_comm, ireq_hhs(iblk), mpierr)
        ! Reset counter and increase destination row
        hh_cnt(iblk) = 0
        hh_dst(iblk) = hh_dst(iblk)+1
      endif

    enddo
#endif
  enddo

  ! Finish the last outstanding requests
#ifdef WITH_OPENMP
  call mpi_wait(ireq_ab,MPI_STATUS,mpierr)
  call mpi_wait(ireq_hv,MPI_STATUS,mpierr)

  allocate(mpi_statuses(MPI_STATUS_SIZE,max(nblocks,num_chunks)))
  call mpi_waitall(nblocks, ireq_hhs, MPI_STATUSES, mpierr)
  call mpi_waitall(num_chunks, ireq_hhr, MPI_STATUSES, mpierr)
  deallocate(mpi_statuses)
#else
  call mpi_wait(ireq_ab,MPI_STATUS_IGNORE,mpierr)
  call mpi_wait(ireq_hv,MPI_STATUS_IGNORE,mpierr)

  call mpi_waitall(nblocks, ireq_hhs, MPI_STATUSES_IGNORE, mpierr)
  call mpi_waitall(num_chunks, ireq_hhr, MPI_STATUSES_IGNORE, mpierr)
#endif

  call mpi_barrier(mpi_comm,mpierr)

  deallocate(ab)
  deallocate(ireq_hhr, ireq_hhs)
  deallocate(hh_cnt, hh_dst)
  deallocate(hh_gath, hh_send)
  deallocate(limits, snd_limits)
  deallocate(block_limits)
  deallocate(global_id)

#ifdef HAVE_DETAILED_TIMINGS
  call timer%stop("tridiag_band_real")
#endif

 end subroutine tridiag_band_real

! --------------------------------------------------------------------------------------------------


subroutine trans_ev_tridi_to_band_real(na, nev, nblk, nbw, q, ldq, matrixCols, &
                                       mpi_comm_rows, mpi_comm_cols, wantDebug, success, &
                                       THIS_REAL_ELPA_KERNEL)
!-------------------------------------------------------------------------------
!  trans_ev_tridi_to_band_real:
!  Transforms the eigenvectors of a tridiagonal matrix back to the eigenvectors of the band matrix
!
!  Parameters
!
!  na          Order of matrix a, number of rows of matrix q
!
!  nev         Number eigenvectors to compute (= columns of matrix q)
!
!  nblk        blocksize of cyclic distribution, must be the same in both directions!
!
!  nb          semi bandwith
!
!  q           On input: Eigenvectors of tridiagonal matrix
!              On output: Transformed eigenvectors
!              Distribution is like in Scalapack.
!
!  ldq         Leading dimension of q
!  matrixCols  local columns of matrix q
!
!  mpi_comm_rows
!  mpi_comm_cols
!              MPI-Communicators for rows/columns/both
!
!-------------------------------------------------------------------------------
#ifdef HAVE_DETAILED_TIMINGS
 use timings
#endif
    implicit none

    integer, intent(in) :: THIS_REAL_ELPA_KERNEL
    integer, intent(in) :: na, nev, nblk, nbw, ldq, matrixCols, mpi_comm_rows, mpi_comm_cols
    real*8              :: q(ldq,matrixCols)

    integer np_rows, my_prow, np_cols, my_pcol

    integer i, j, ip, sweep, nbuf, l_nev, a_dim2
    integer current_n, current_local_n, current_n_start, current_n_end
    integer next_n, next_local_n, next_n_start, next_n_end
    integer bottom_msg_length, top_msg_length, next_top_msg_length
    integer stripe_width, last_stripe_width, stripe_count
#ifdef WITH_OPENMP
    integer thread_width, csw, b_off, b_len
#endif
    integer num_result_blocks, num_result_buffers, num_bufs_recvd
    integer a_off, current_tv_off, max_blk_size
    integer mpierr, src, src_offset, dst, offset, nfact, num_blk
#ifdef WITH_OPENMP
    integer mpi_status(MPI_STATUS_SIZE)
#endif
    logical flag

#ifdef WITH_OPENMP
    real*8, allocatable :: a(:,:,:,:), row(:)
#else
    real*8, allocatable :: a(:,:,:), row(:)
#endif

#ifdef WITH_OPENMP
    real*8, allocatable :: top_border_send_buffer(:,:), top_border_recv_buffer(:,:)
    real*8, allocatable :: bottom_border_send_buffer(:,:), bottom_border_recv_buffer(:,:)
#else
    real*8, allocatable :: top_border_send_buffer(:,:,:), top_border_recv_buffer(:,:,:)
    real*8, allocatable :: bottom_border_send_buffer(:,:,:), bottom_border_recv_buffer(:,:,:)
#endif
    real*8, allocatable :: result_buffer(:,:,:)
    real*8, allocatable :: bcast_buffer(:,:)

    integer n_off
    integer, allocatable :: result_send_request(:), result_recv_request(:), limits(:)
    integer, allocatable :: top_send_request(:), bottom_send_request(:)
    integer, allocatable :: top_recv_request(:), bottom_recv_request(:)
#ifdef WITH_OPENMP
    integer, allocatable :: mpi_statuses(:,:)
#endif
    ! MPI send/recv tags, arbitrary

    integer, parameter  :: bottom_recv_tag = 111
    integer, parameter  :: top_recv_tag    = 222
    integer, parameter  :: result_recv_tag = 333

    ! Just for measuring the kernel performance
    real*8              :: kernel_time
    integer*8           :: kernel_flops

#ifdef WITH_OPENMP
    integer             :: max_threads, my_thread
    integer             :: omp_get_max_threads
#endif

    logical, intent(in) :: wantDebug
    logical             :: success

#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("trans_ev_tridi_to_band_real")
#endif
    success = .true.
    kernel_time = 1.d-100
    kernel_flops = 0

#ifdef WITH_OPENMP
    max_threads = 1
    max_threads = omp_get_max_threads()
#endif

    call MPI_Comm_rank(mpi_comm_rows, my_prow, mpierr)
    call MPI_Comm_size(mpi_comm_rows, np_rows, mpierr)
    call MPI_Comm_rank(mpi_comm_cols, my_pcol, mpierr)
    call MPI_Comm_size(mpi_comm_cols, np_cols, mpierr)

    if (mod(nbw,nblk)/=0) then
      if (my_prow==0 .and. my_pcol==0) then
        if (wantDebug) then
          write(error_unit,*) 'ELPA2_trans_ev_tridi_to_band_real: ERROR: nbw=',nbw,', nblk=',nblk
          write(error_unit,*) 'ELPA2_trans_ev_tridi_to_band_real: band backtransform works only for nbw==n*nblk'
        endif
        success = .false.
        return
      endif
    endif

    nfact = nbw / nblk


    ! local number of eigenvectors
    l_nev = local_index(nev, my_pcol, np_cols, nblk, -1)

    if (l_nev==0) then
#ifdef WITH_OPENMP
      thread_width = 0
#endif
      stripe_width = 0
      stripe_count = 0
      last_stripe_width = 0
    else
      ! Suggested stripe width is 48 since 48*64 real*8 numbers should fit into
      ! every primary cache
#ifdef WITH_OPENMP
      thread_width = (l_nev-1)/max_threads + 1 ! number of eigenvectors per OMP thread
#endif
      stripe_width = 48 ! Must be a multiple of 4
#ifdef WITH_OPENMP
      stripe_count = (thread_width-1)/stripe_width + 1
#else
      stripe_count = (l_nev-1)/stripe_width + 1
#endif
      ! Adapt stripe width so that last one doesn't get too small
#ifdef WITH_OPENMP
      stripe_width = (thread_width-1)/stripe_count + 1
#else
      stripe_width = (l_nev-1)/stripe_count + 1
#endif
      stripe_width = ((stripe_width+3)/4)*4 ! Must be a multiple of 4 !!!
      last_stripe_width = l_nev - (stripe_count-1)*stripe_width
    endif

    ! Determine the matrix distribution at the beginning

    allocate(limits(0:np_rows))

    call determine_workload(na, nbw, np_rows, limits)

    max_blk_size = maxval(limits(1:np_rows) - limits(0:np_rows-1))

    a_dim2 = max_blk_size + nbw

!DEC$ ATTRIBUTES ALIGN: 64:: a
#ifdef WITH_OPENMP
    allocate(a(stripe_width,a_dim2,stripe_count,max_threads))
    ! a(:,:,:,:) should be set to 0 in a parallel region, not here!
#else
    allocate(a(stripe_width,a_dim2,stripe_count))
    a(:,:,:) = 0
#endif

    allocate(row(l_nev))
    row(:) = 0

    ! Copy q from a block cyclic distribution into a distribution with contiguous rows,
    ! and transpose the matrix using stripes of given stripe_width for cache blocking.

    ! The peculiar way it is done below is due to the fact that the last row should be
    ! ready first since it is the first one to start below

#ifdef WITH_OPENMP
    ! Please note about the OMP usage below:
    ! This is not for speed, but because we want the matrix a in the memory and
    ! in the cache of the correct thread (if possible)
#ifdef HAVE_DETAILED_TIMINGS
    call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread), schedule(static, 1)
    do my_thread = 1, max_threads
      a(:,:,:,my_thread) = 0 ! if possible, do first touch allocation!
    enddo
    !$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
    call timer%stop("OpenMP parallel")
#endif
#endif

   do ip = np_rows-1, 0, -1
     if (my_prow == ip) then
       ! Receive my rows which have not yet been received
       src_offset = local_index(limits(ip), my_prow, np_rows, nblk, -1)
       do i=limits(ip)+1,limits(ip+1)
         src = mod((i-1)/nblk, np_rows)
         if (src < my_prow) then
#ifdef WITH_OPENMP
           call MPI_Recv(row, l_nev, MPI_REAL8, src, 0, mpi_comm_rows, MPI_STATUS, mpierr)
#ifdef HAVE_DETAILED_TIMINGS
           call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread), schedule(static, 1)
           do my_thread = 1, max_threads
             call unpack_row(row,i-limits(ip),my_thread)
           enddo
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
           call timer%stop("OpenMP parallel")
#endif

#else
           call MPI_Recv(row, l_nev, MPI_REAL8, src, 0, mpi_comm_rows, MPI_STATUS_IGNORE, mpierr)
           call unpack_row(row,i-limits(ip))
#endif
         elseif (src==my_prow) then
           src_offset = src_offset+1
           row(:) = q(src_offset, 1:l_nev)
#ifdef WITH_OPENMP
#ifdef HAVE_DETAILED_TIMINGS
           call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread), schedule(static, 1)
           do my_thread = 1, max_threads
             call unpack_row(row,i-limits(ip),my_thread)
           enddo
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
           call timer%stop("OpenMP parallel")
#endif

#else
           call unpack_row(row,i-limits(ip))
#endif
         endif
       enddo
       ! Send all rows which have not yet been send
       src_offset = 0
       do dst = 0, ip-1
         do i=limits(dst)+1,limits(dst+1)
           if (mod((i-1)/nblk, np_rows) == my_prow) then
             src_offset = src_offset+1
             row(:) = q(src_offset, 1:l_nev)
             call MPI_Send(row, l_nev, MPI_REAL8, dst, 0, mpi_comm_rows, mpierr)
           endif
         enddo
       enddo
     else if (my_prow < ip) then
       ! Send all rows going to PE ip
       src_offset = local_index(limits(ip), my_prow, np_rows, nblk, -1)
       do i=limits(ip)+1,limits(ip+1)
         src = mod((i-1)/nblk, np_rows)
         if (src == my_prow) then
           src_offset = src_offset+1
           row(:) = q(src_offset, 1:l_nev)
           call MPI_Send(row, l_nev, MPI_REAL8, ip, 0, mpi_comm_rows, mpierr)
         endif
       enddo
       ! Receive all rows from PE ip
       do i=limits(my_prow)+1,limits(my_prow+1)
         src = mod((i-1)/nblk, np_rows)
         if (src == ip) then
#ifdef WITH_OPENMP
           call MPI_Recv(row, l_nev, MPI_REAL8, src, 0, mpi_comm_rows, MPI_STATUS, mpierr)
#ifdef HAVE_DETAILED_TIMINGS
           call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread), schedule(static, 1)
           do my_thread = 1, max_threads
             call unpack_row(row,i-limits(my_prow),my_thread)
           enddo
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
           call timer%stop("OpenMP parallel")
#endif

#else
           call MPI_Recv(row, l_nev, MPI_REAL8, src, 0, mpi_comm_rows, MPI_STATUS_IGNORE, mpierr)
           call unpack_row(row,i-limits(my_prow))
#endif
         endif
       enddo
     endif
   enddo


   ! Set up result buffer queue

   num_result_blocks = ((na-1)/nblk + np_rows - my_prow) / np_rows

   num_result_buffers = 4*nfact
   allocate(result_buffer(l_nev,nblk,num_result_buffers))

   allocate(result_send_request(num_result_buffers))
   allocate(result_recv_request(num_result_buffers))
   result_send_request(:) = MPI_REQUEST_NULL
   result_recv_request(:) = MPI_REQUEST_NULL

   ! Queue up buffers

   if (my_prow > 0 .and. l_nev>0) then ! note: row 0 always sends
     do j = 1, min(num_result_buffers, num_result_blocks)
       call MPI_Irecv(result_buffer(1,1,j), l_nev*nblk, MPI_REAL8, 0, result_recv_tag, &
                           mpi_comm_rows, result_recv_request(j), mpierr)
     enddo
   endif

   num_bufs_recvd = 0 ! No buffers received yet

   ! Initialize top/bottom requests

   allocate(top_send_request(stripe_count))
   allocate(top_recv_request(stripe_count))
   allocate(bottom_send_request(stripe_count))
   allocate(bottom_recv_request(stripe_count))

   top_send_request(:) = MPI_REQUEST_NULL
   top_recv_request(:) = MPI_REQUEST_NULL
   bottom_send_request(:) = MPI_REQUEST_NULL
   bottom_recv_request(:) = MPI_REQUEST_NULL

#ifdef WITH_OPENMP
   allocate(top_border_send_buffer(stripe_width*nbw*max_threads, stripe_count))
   allocate(top_border_recv_buffer(stripe_width*nbw*max_threads, stripe_count))
   allocate(bottom_border_send_buffer(stripe_width*nbw*max_threads, stripe_count))
   allocate(bottom_border_recv_buffer(stripe_width*nbw*max_threads, stripe_count))

   top_border_send_buffer(:,:) = 0
   top_border_recv_buffer(:,:) = 0
   bottom_border_send_buffer(:,:) = 0
   bottom_border_recv_buffer(:,:) = 0

   ! Initialize broadcast buffer
#else
   allocate(top_border_send_buffer(stripe_width, nbw, stripe_count))
   allocate(top_border_recv_buffer(stripe_width, nbw, stripe_count))
   allocate(bottom_border_send_buffer(stripe_width, nbw, stripe_count))
   allocate(bottom_border_recv_buffer(stripe_width, nbw, stripe_count))

   top_border_send_buffer(:,:,:) = 0
   top_border_recv_buffer(:,:,:) = 0
   bottom_border_send_buffer(:,:,:) = 0
   bottom_border_recv_buffer(:,:,:) = 0
#endif

   allocate(bcast_buffer(nbw, max_blk_size))
   bcast_buffer = 0

   current_tv_off = 0 ! Offset of next row to be broadcast


    ! ------------------- start of work loop -------------------

   a_off = 0 ! offset in A (to avoid unnecessary shifts)

   top_msg_length = 0
   bottom_msg_length = 0

   do sweep = 0, (na-1)/nbw

     current_n = na - sweep*nbw
     call determine_workload(current_n, nbw, np_rows, limits)
     current_n_start = limits(my_prow)
     current_n_end   = limits(my_prow+1)
     current_local_n = current_n_end - current_n_start

     next_n = max(current_n - nbw, 0)
     call determine_workload(next_n, nbw, np_rows, limits)
     next_n_start = limits(my_prow)
     next_n_end   = limits(my_prow+1)
     next_local_n = next_n_end - next_n_start

     if (next_n_end < next_n) then
       bottom_msg_length = current_n_end - next_n_end
     else
       bottom_msg_length = 0
     endif

     if (next_local_n > 0) then
       next_top_msg_length = current_n_start - next_n_start
     else
       next_top_msg_length = 0
     endif

     if (sweep==0 .and. current_n_end < current_n .and. l_nev > 0) then
       do i = 1, stripe_count
#ifdef WITH_OPENMP
         csw = min(stripe_width, thread_width-(i-1)*stripe_width) ! "current_stripe_width"
         b_len = csw*nbw*max_threads
         call MPI_Irecv(bottom_border_recv_buffer(1,i), b_len, MPI_REAL8, my_prow+1, bottom_recv_tag, &
                           mpi_comm_rows, bottom_recv_request(i), mpierr)
#else
         call MPI_Irecv(bottom_border_recv_buffer(1,1,i), nbw*stripe_width, MPI_REAL8, my_prow+1, bottom_recv_tag, &
                        mpi_comm_rows, bottom_recv_request(i), mpierr)
#endif
       enddo
     endif

     if (current_local_n > 1) then
       if (my_pcol == mod(sweep,np_cols)) then
         bcast_buffer(:,1:current_local_n) = hh_trans_real(:,current_tv_off+1:current_tv_off+current_local_n)
         current_tv_off = current_tv_off + current_local_n
       endif
       call mpi_bcast(bcast_buffer, nbw*current_local_n, MPI_REAL8, mod(sweep,np_cols), mpi_comm_cols, mpierr)
     else
       ! for current_local_n == 1 the one and only HH vector is 0 and not stored in hh_trans_real
       bcast_buffer(:,1) = 0
     endif

     if (l_nev == 0) cycle

     if (current_local_n > 0) then

       do i = 1, stripe_count
#ifdef WITH_OPENMP
         ! Get real stripe width for strip i;
         ! The last OpenMP tasks may have an even smaller stripe with,
         ! but we don't care about this, i.e. we send/recv a bit too much in this case.
         ! csw: current_stripe_width

         csw = min(stripe_width, thread_width-(i-1)*stripe_width)
#endif
         !wait_b
         if (current_n_end < current_n) then
#ifdef WITH_OPENMP
           call MPI_Wait(bottom_recv_request(i), MPI_STATUS, mpierr)
#ifdef HAVE_DETAILED_TIMINGS
           call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread, n_off, b_len, b_off), schedule(static, 1)
           do my_thread = 1, max_threads
             n_off = current_local_n+a_off
             b_len = csw*nbw
             b_off = (my_thread-1)*b_len
             a(1:csw,n_off+1:n_off+nbw,i,my_thread) = &
               reshape(bottom_border_recv_buffer(b_off+1:b_off+b_len,i), (/ csw, nbw /))
           enddo
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
           call timer%stop("OpenMP parallel")
#endif

#else
           call MPI_Wait(bottom_recv_request(i), MPI_STATUS_IGNORE, mpierr)
           n_off = current_local_n+a_off
           a(:,n_off+1:n_off+nbw,i) = bottom_border_recv_buffer(:,1:nbw,i)

#endif
           if (next_n_end < next_n) then
#ifdef WITH_OPENMP
             call MPI_Irecv(bottom_border_recv_buffer(1,i), csw*nbw*max_threads, &
                                   MPI_REAL8, my_prow+1, bottom_recv_tag, &
                                   mpi_comm_rows, bottom_recv_request(i), mpierr)
#else
             call MPI_Irecv(bottom_border_recv_buffer(1,1,i), nbw*stripe_width, MPI_REAL8, my_prow+1, bottom_recv_tag, &

                                   mpi_comm_rows, bottom_recv_request(i), mpierr)
#endif
           endif
         endif

         if (current_local_n <= bottom_msg_length + top_msg_length) then

           !wait_t
           if (top_msg_length>0) then
#ifdef WITH_OPENMP
             call MPI_Wait(top_recv_request(i), MPI_STATUS, mpierr)
#else
             call MPI_Wait(top_recv_request(i), MPI_STATUS_IGNORE, mpierr)
             a(:,a_off+1:a_off+top_msg_length,i) = top_border_recv_buffer(:,1:top_msg_length,i)
#endif
           endif

           !compute
#ifdef WITH_OPENMP
#ifdef HAVE_DETAILED_TIMINGS
           call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread, n_off, b_len, b_off), schedule(static, 1)
           do my_thread = 1, max_threads
             if (top_msg_length>0) then
               b_len = csw*top_msg_length
               b_off = (my_thread-1)*b_len
               a(1:csw,a_off+1:a_off+top_msg_length,i,my_thread) = &
                          reshape(top_border_recv_buffer(b_off+1:b_off+b_len,i), (/ csw, top_msg_length /))
             endif
             call compute_hh_trafo(0, current_local_n, i, my_thread, &
                                      THIS_REAL_ELPA_KERNEL)
           enddo
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
           call timer%stop("OpenMP parallel")
#endif

#else
           call compute_hh_trafo(0, current_local_n, i, &
                                      THIS_REAL_ELPA_KERNEL)
#endif
           !send_b
#ifdef WITH_OPENMP
           call MPI_Wait(bottom_send_request(i), mpi_status, mpierr)
           if (bottom_msg_length>0) then
             n_off = current_local_n+nbw-bottom_msg_length+a_off
             b_len = csw*bottom_msg_length*max_threads
             bottom_border_send_buffer(1:b_len,i) = &
                 reshape(a(1:csw,n_off+1:n_off+bottom_msg_length,i,:), (/ b_len /))
             call MPI_Isend(bottom_border_send_buffer(1,i), b_len, MPI_REAL8, my_prow+1, &
                            top_recv_tag, mpi_comm_rows, bottom_send_request(i), mpierr)
           endif
#else
           call MPI_Wait(bottom_send_request(i), MPI_STATUS_IGNORE, mpierr)
           if (bottom_msg_length>0) then
             n_off = current_local_n+nbw-bottom_msg_length+a_off
             bottom_border_send_buffer(:,1:bottom_msg_length,i) = a(:,n_off+1:n_off+bottom_msg_length,i)
             call MPI_Isend(bottom_border_send_buffer(1,1,i), bottom_msg_length*stripe_width, MPI_REAL8, my_prow+1, &


                            top_recv_tag, mpi_comm_rows, bottom_send_request(i), mpierr)
           endif
#endif
         else

         !compute
#ifdef WITH_OPENMP
#ifdef HAVE_DETAILED_TIMINGS
         call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread, b_len, b_off), schedule(static, 1)
        do my_thread = 1, max_threads
          call compute_hh_trafo(current_local_n - bottom_msg_length, bottom_msg_length, i, my_thread, &
                              THIS_REAL_ELPA_KERNEL)
        enddo
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
        call timer%stop("OpenMP parallel")
#endif

        !send_b
        call MPI_Wait(bottom_send_request(i), mpi_status, mpierr)
        if (bottom_msg_length > 0) then
          n_off = current_local_n+nbw-bottom_msg_length+a_off
          b_len = csw*bottom_msg_length*max_threads
          bottom_border_send_buffer(1:b_len,i) = &
              reshape(a(1:csw,n_off+1:n_off+bottom_msg_length,i,:), (/ b_len /))
          call MPI_Isend(bottom_border_send_buffer(1,i), b_len, MPI_REAL8, my_prow+1, &
                           top_recv_tag, mpi_comm_rows, bottom_send_request(i), mpierr)
        endif
#else
        call compute_hh_trafo(current_local_n - bottom_msg_length, bottom_msg_length, i, &
                                      THIS_REAL_ELPA_KERNEL)

        !send_b
        call MPI_Wait(bottom_send_request(i), MPI_STATUS_IGNORE, mpierr)
        if (bottom_msg_length > 0) then
          n_off = current_local_n+nbw-bottom_msg_length+a_off
          bottom_border_send_buffer(:,1:bottom_msg_length,i) = a(:,n_off+1:n_off+bottom_msg_length,i)
          call MPI_Isend(bottom_border_send_buffer(1,1,i), bottom_msg_length*stripe_width, MPI_REAL8, my_prow+1, &
                         top_recv_tag, mpi_comm_rows, bottom_send_request(i), mpierr)
        endif
#endif

        !compute
#ifdef WITH_OPENMP
#ifdef HAVE_DETAILED_TIMINGS
        call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread), schedule(static, 1)
        do my_thread = 1, max_threads
          call compute_hh_trafo(top_msg_length, current_local_n-top_msg_length-bottom_msg_length, i, my_thread, &
                                THIS_REAL_ELPA_KERNEL)
        enddo
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
        call timer%stop("OpenMP parallel")
#endif

#else
        call compute_hh_trafo(top_msg_length, current_local_n-top_msg_length-bottom_msg_length, i, &
                              THIS_REAL_ELPA_KERNEL)

#endif
        !wait_t
        if (top_msg_length>0) then
#ifdef WITH_OPENMP
          call MPI_Wait(top_recv_request(i), mpi_status, mpierr)
#else
          call MPI_Wait(top_recv_request(i), MPI_STATUS_IGNORE, mpierr)
          a(:,a_off+1:a_off+top_msg_length,i) = top_border_recv_buffer(:,1:top_msg_length,i)
#endif
        endif

        !compute
#ifdef WITH_OPENMP
#ifdef HAVE_DETAILED_TIMINGS
        call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread, b_len, b_off), schedule(static, 1)
        do my_thread = 1, max_threads
          if (top_msg_length>0) then
            b_len = csw*top_msg_length
            b_off = (my_thread-1)*b_len
            a(1:csw,a_off+1:a_off+top_msg_length,i,my_thread) = &
              reshape(top_border_recv_buffer(b_off+1:b_off+b_len,i), (/ csw, top_msg_length /))
          endif
          call compute_hh_trafo(0, top_msg_length, i, my_thread, THIS_REAL_ELPA_KERNEL)
        enddo
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
        call timer%stop("OpenMP parallel")
#endif

#else
        call compute_hh_trafo(0, top_msg_length, i, THIS_REAL_ELPA_KERNEL)
#endif
      endif

      if (next_top_msg_length > 0) then
        !request top_border data
#ifdef WITH_OPENMP
        b_len = csw*next_top_msg_length*max_threads
        call MPI_Irecv(top_border_recv_buffer(1,i), b_len, MPI_REAL8, my_prow-1, &
                       top_recv_tag, mpi_comm_rows, top_recv_request(i), mpierr)
#else
        call MPI_Irecv(top_border_recv_buffer(1,1,i), next_top_msg_length*stripe_width, MPI_REAL8, my_prow-1, &
                       top_recv_tag, mpi_comm_rows, top_recv_request(i), mpierr)
#endif
      endif

      !send_t
      if (my_prow > 0) then
#ifdef WITH_OPENMP
        call MPI_Wait(top_send_request(i), mpi_status, mpierr)
        b_len = csw*nbw*max_threads
        top_border_send_buffer(1:b_len,i) = reshape(a(1:csw,a_off+1:a_off+nbw,i,:), (/ b_len /))
        call MPI_Isend(top_border_send_buffer(1,i), b_len, MPI_REAL8, &
                       my_prow-1, bottom_recv_tag, &
                       mpi_comm_rows, top_send_request(i), mpierr)
#else
        call MPI_Wait(top_send_request(i), MPI_STATUS_IGNORE, mpierr)
        top_border_send_buffer(:,1:nbw,i) = a(:,a_off+1:a_off+nbw,i)
        call MPI_Isend(top_border_send_buffer(1,1,i), nbw*stripe_width, MPI_REAL8, my_prow-1, bottom_recv_tag, &
                       mpi_comm_rows, top_send_request(i), mpierr)

#endif
      endif

      ! Care that there are not too many outstanding top_recv_request's
      if (stripe_count > 1) then
        if (i>1) then
#ifdef WITH_OPENMP
          call MPI_Wait(top_recv_request(i-1), MPI_STATUS, mpierr)
#else
          call MPI_Wait(top_recv_request(i-1), MPI_STATUS_IGNORE, mpierr)
#endif
        else
#ifdef WITH_OPENMP
          call MPI_Wait(top_recv_request(stripe_count), MPI_STATUS, mpierr)
#else
          call MPI_Wait(top_recv_request(stripe_count), MPI_STATUS_IGNORE, mpierr)
#endif
        endif
      endif

    enddo

    top_msg_length = next_top_msg_length

  else
    ! wait for last top_send_request
    do i = 1, stripe_count
#ifdef WITH_OPENMP
      call MPI_Wait(top_send_request(i), MPI_STATUS, mpierr)
#else
      call MPI_Wait(top_send_request(i), MPI_STATUS_IGNORE, mpierr)
#endif
      enddo
    endif

    ! Care about the result

    if (my_prow == 0) then

      ! topmost process sends nbw rows to destination processes

      do j=0,nfact-1
        num_blk = sweep*nfact+j ! global number of destination block, 0 based
        if (num_blk*nblk >= na) exit

          nbuf = mod(num_blk, num_result_buffers) + 1 ! buffer number to get this block

#ifdef WITH_OPENMP
          call MPI_Wait(result_send_request(nbuf), MPI_STATUS, mpierr)
#else
          call MPI_Wait(result_send_request(nbuf), MPI_STATUS_IGNORE, mpierr)
#endif
          dst = mod(num_blk, np_rows)

          if (dst == 0) then
            do i = 1, min(na - num_blk*nblk, nblk)
              call pack_row(row, j*nblk+i+a_off)
              q((num_blk/np_rows)*nblk+i,1:l_nev) = row(:)
            enddo
          else
            do i = 1, nblk
              call pack_row(result_buffer(:,i,nbuf),j*nblk+i+a_off)
            enddo
            call MPI_Isend(result_buffer(1,1,nbuf), l_nev*nblk, MPI_REAL8, dst, &
                                    result_recv_tag, mpi_comm_rows, result_send_request(nbuf), mpierr)
          endif
        enddo

      else

        ! receive and store final result

        do j = num_bufs_recvd, num_result_blocks-1

          nbuf = mod(j, num_result_buffers) + 1 ! buffer number to get this block

          ! If there is still work to do, just test for the next result request
          ! and leave the loop if it is not ready, otherwise wait for all
          ! outstanding requests

          if (next_local_n > 0) then
#ifdef WITH_OPENMP
            call MPI_Test(result_recv_request(nbuf), flag, MPI_STATUS, mpierr)
#else
            call MPI_Test(result_recv_request(nbuf), flag, MPI_STATUS_IGNORE, mpierr)

#endif
            if (.not.flag) exit
          else
#ifdef WITH_OPENMP
            call MPI_Wait(result_recv_request(nbuf), MPI_STATUS, mpierr)
#else
            call MPI_Wait(result_recv_request(nbuf), MPI_STATUS_IGNORE, mpierr)
#endif
          endif

          ! Fill result buffer into q
           num_blk = j*np_rows + my_prow ! global number of current block, 0 based
           do i = 1, min(na - num_blk*nblk, nblk)
             q(j*nblk+i, 1:l_nev) = result_buffer(1:l_nev, i, nbuf)
           enddo

           ! Queue result buffer again if there are outstanding blocks left
           if (j+num_result_buffers < num_result_blocks) &
                     call MPI_Irecv(result_buffer(1,1,nbuf), l_nev*nblk, MPI_REAL8, 0, result_recv_tag, &
                                    mpi_comm_rows, result_recv_request(nbuf), mpierr)

         enddo
         num_bufs_recvd = j

       endif

       ! Shift the remaining rows to the front of A (if necessary)

       offset = nbw - top_msg_length
       if (offset<0) then
         if (wantDebug) write(error_unit,*) 'ELPA2_trans_ev_tridi_to_band_real: internal error, offset for shifting = ',offset
         success = .false.
         return
       endif

       a_off = a_off + offset
       if (a_off + next_local_n + nbw > a_dim2) then
#ifdef WITH_OPENMP
#ifdef HAVE_DETAILED_TIMINGS
         call timer%start("OpenMP parallel")
#endif

 !$omp parallel do private(my_thread, i, j), schedule(static, 1)
         do my_thread = 1, max_threads
           do i = 1, stripe_count
             do j = top_msg_length+1, top_msg_length+next_local_n
               A(:,j,i,my_thread) = A(:,j+a_off,i,my_thread)
             enddo
#else
         do i = 1, stripe_count
           do j = top_msg_length+1, top_msg_length+next_local_n
             A(:,j,i) = A(:,j+a_off,i)
#endif
           enddo
         enddo
#ifdef WITH_OPENMP
#ifdef HAVE_DETAILED_TIMINGS
         call timer%stop("OpenMP parallel")
#endif
#endif
         a_off = 0
       endif

     enddo

     ! Just for safety:
     if (ANY(top_send_request    /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR top_send_request ***',my_prow,my_pcol
     if (ANY(bottom_send_request /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR bottom_send_request ***',my_prow,my_pcol
     if (ANY(top_recv_request    /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR top_recv_request ***',my_prow,my_pcol
     if (ANY(bottom_recv_request /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR bottom_recv_request ***',my_prow,my_pcol

     if (my_prow == 0) then
#ifdef WITH_OPENMP
       allocate(mpi_statuses(MPI_STATUS_SIZE,num_result_buffers))
       call MPI_Waitall(num_result_buffers, result_send_request, mpi_statuses, mpierr)
       deallocate(mpi_statuses)
#else
       call MPI_Waitall(num_result_buffers, result_send_request, MPI_STATUSES_IGNORE, mpierr)
#endif
     endif

     if (ANY(result_send_request /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR result_send_request ***',my_prow,my_pcol
     if (ANY(result_recv_request /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR result_recv_request ***',my_prow,my_pcol

     if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
         write(error_unit,'(" Kernel time:",f10.3," MFlops: ",f10.3)')  kernel_time, kernel_flops/kernel_time*1.d-6

     ! deallocate all working space

     deallocate(a)
     deallocate(row)
     deallocate(limits)
     deallocate(result_send_request)
     deallocate(result_recv_request)
     deallocate(top_border_send_buffer)
     deallocate(top_border_recv_buffer)
     deallocate(bottom_border_send_buffer)
     deallocate(bottom_border_recv_buffer)
     deallocate(result_buffer)
     deallocate(bcast_buffer)
     deallocate(top_send_request)
     deallocate(top_recv_request)
     deallocate(bottom_send_request)
     deallocate(bottom_recv_request)

#ifdef HAVE_DETAILED_TIMINGS
     call timer%stop("trans_ev_tridi_to_band_real")
#endif
   return

 contains

   subroutine pack_row(row, n)
#ifdef HAVE_DETAILED_TIMINGS
     use timings
#endif
     implicit none
     real*8  :: row(:)
     integer :: n, i, noff, nl
#ifdef WITH_OPENMP
     integer :: nt
#endif

#ifdef HAVE_DETAILED_TIMINGS
     call timer%start("pack_row")
#endif

#ifdef WITH_OPENMP
     do nt = 1, max_threads
       do i = 1, stripe_count
         noff = (nt-1)*thread_width + (i-1)*stripe_width
         nl   = min(stripe_width, nt*thread_width-noff, l_nev-noff)
         if (nl<=0) exit
         row(noff+1:noff+nl) = a(1:nl,n,i,nt)
       enddo
     enddo
#else
     do i=1,stripe_count
       nl = merge(stripe_width, last_stripe_width, i<stripe_count)
       noff = (i-1)*stripe_width
       row(noff+1:noff+nl) = a(1:nl,n,i)
     enddo
#endif

#ifdef HAVE_DETAILED_TIMINGS
     call timer%stop("pack_row")
#endif

   end subroutine pack_row

#ifdef WITH_OPENMP
   subroutine unpack_row(row, n, my_thread)
#ifdef HAVE_DETAILED_TIMINGS
     use timings
#endif
     implicit none

     ! Private variables in OMP regions (my_thread) should better be in the argument list!
     integer, intent(in) :: n, my_thread
     real*8, intent(in)  :: row(:)
     integer             :: i, noff, nl

#ifdef HAVE_DETAILED_TIMINGS
     call timer%start("unpack_row")
#endif
     do i=1,stripe_count
       noff = (my_thread-1)*thread_width + (i-1)*stripe_width
       nl   = min(stripe_width, my_thread*thread_width-noff, l_nev-noff)
       if(nl<=0) exit
       a(1:nl,n,i,my_thread) = row(noff+1:noff+nl)
     enddo

#ifdef HAVE_DETAILED_TIMINGS
     call timer%stop("unpack_row")
#endif

   end subroutine unpack_row
#else
   subroutine unpack_row(row, n)
#ifdef HAVE_DETAILED_TIMINGS
     use timings
#endif
     implicit none

     real*8  :: row(:)
     integer :: n, i, noff, nl

#ifdef HAVE_DETAILED_TIMINGS
     call timer%start("unpack_row")
#endif

     do i=1,stripe_count
       nl = merge(stripe_width, last_stripe_width, i<stripe_count)
       noff = (i-1)*stripe_width
       a(1:nl,n,i) = row(noff+1:noff+nl)
     enddo

#ifdef HAVE_DETAILED_TIMINGS
     call timer%stop("unpack_row")
#endif
   end subroutine unpack_row
#endif

#ifdef WITH_OPENMP
   subroutine compute_hh_trafo(off, ncols, istripe, my_thread, THIS_REAL_ELPA_KERNEL)
#else
   subroutine compute_hh_trafo(off, ncols, istripe, THIS_REAL_ELPA_KERNEL)
#endif

#if defined(WITH_REAL_GENERIC_SIMPLE_KERNEL)
      use real_generic_simple_kernel, only : double_hh_trafo_generic_simple
#endif

!#if defined(WITH_REAL_GENERIC_KERNEL)
!      use real_generic_kernel, only : double_hh_trafo_generic
!#endif

#if defined(WITH_REAL_BGP_KERNEL)
      use real_bgp_kernel, only : double_hh_trafo_bgp
#endif

#if defined(WITH_REAL_BGQ_KERNEL)
      use real_bgq_kernel, only : double_hh_trafo_bgq
#endif
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#endif
      implicit none

      integer, intent(in) :: THIS_REAL_ELPA_KERNEL

      ! Private variables in OMP regions (my_thread) should better be in the argument list!
      integer             :: off, ncols, istripe
#ifdef WITH_OPENMP
      integer             :: my_thread, noff
#endif
      integer             :: j, nl, jj, jjj
      real*8              :: w(nbw,6), ttt

#ifdef HAVE_DETAILED_TIMINGS
      call timer%start("compute_hh_trafo")
#endif

      ttt = mpi_wtime()

#ifndef WITH_OPENMP
      nl = merge(stripe_width, last_stripe_width, istripe<stripe_count)
#else

      if (istripe<stripe_count) then
        nl = stripe_width
      else
        noff = (my_thread-1)*thread_width + (istripe-1)*stripe_width
        nl = min(my_thread*thread_width-noff, l_nev-noff)
        if (nl<=0) return
      endif
#endif

#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
      if (THIS_REAL_ELPA_KERNEL .eq. REAL_ELPA_KERNEL_AVX_BLOCK2 .or. &
          THIS_REAL_ELPA_KERNEL .eq. REAL_ELPA_KERNEL_GENERIC    .or. &
          THIS_REAL_ELPA_KERNEL .eq. REAL_ELPA_KERNEL_GENERIC_SIMPLE .or. &
          THIS_REAL_ELPA_KERNEL .eq. REAL_ELPA_KERNEL_SSE .or.        &
          THIS_REAL_ELPA_KERNEL .eq. REAL_ELPA_KERNEL_BGP .or.        &
          THIS_REAL_ELPA_KERNEL .eq. REAL_ELPA_KERNEL_BGQ) then
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */

        !FORTRAN CODE / X86 INRINISIC CODE / BG ASSEMBLER USING 2 HOUSEHOLDER VECTORS
#if defined(WITH_REAL_GENERIC_KERNEL)
#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
        if (THIS_REAL_ELPA_KERNEL .eq. REAL_ELPA_KERNEL_GENERIC) then
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */
          do j = ncols, 2, -2
            w(:,1) = bcast_buffer(1:nbw,j+off)
            w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP
            call double_hh_trafo_generic(a(1,j+off+a_off-1,istripe,my_thread), w, &
                                      nbw, nl, stripe_width, nbw)
#else
            call double_hh_trafo_generic(a(1,j+off+a_off-1,istripe),           w, &
                                      nbw, nl, stripe_width, nbw)
#endif
          enddo
#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
        endif
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */
#endif /* WITH_REAL_GENERIC_KERNEL */


#if defined(WITH_REAL_GENERIC_SIMPLE_KERNEL)
#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
        if (THIS_REAL_ELPA_KERNEL .eq. REAL_ELPA_KERNEL_GENERIC_SIMPLE) then
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */
          do j = ncols, 2, -2
            w(:,1) = bcast_buffer(1:nbw,j+off)
            w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP
            call double_hh_trafo_generic_simple(a(1,j+off+a_off-1,istripe,my_thread), &
                                                     w, nbw, nl, stripe_width, nbw)
#else
            call double_hh_trafo_generic_simple(a(1,j+off+a_off-1,istripe), &
                                                     w, nbw, nl, stripe_width, nbw)
#endif
          enddo
#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
        endif
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */
#endif /* WITH_REAL_GENERIC_SIMPLE_KERNEL */


#if defined(WITH_REAL_SSE_KERNEL)
#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
        if (THIS_REAL_ELPA_KERNEL .eq. REAL_ELPA_KERNEL_SSE) then
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */
          do j = ncols, 2, -2
            w(:,1) = bcast_buffer(1:nbw,j+off)
            w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP
            call double_hh_trafo(a(1,j+off+a_off-1,istripe,my_thread), w, nbw, nl, &
                                      stripe_width, nbw)
#else
            call double_hh_trafo(a(1,j+off+a_off-1,istripe), w, nbw, nl, &
                                      stripe_width, nbw)
#endif
          enddo
#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
        endif
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */
#endif /* WITH_REAL_SSE_KERNEL */


#if defined(WITH_REAL_AVX_BLOCK2_KERNEL)
#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
        if (THIS_REAL_ELPA_KERNEL .eq. REAL_ELPA_KERNEL_AVX_BLOCK2) then
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */
          do j = ncols, 2, -2
            w(:,1) = bcast_buffer(1:nbw,j+off)
            w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP
            call double_hh_trafo_real_sse_avx_2hv(a(1,j+off+a_off-1,istripe,my_thread), &
                                                       w, nbw, nl, stripe_width, nbw)
#else
            call double_hh_trafo_real_sse_avx_2hv(a(1,j+off+a_off-1,istripe), &
                                                       w, nbw, nl, stripe_width, nbw)
#endif
          enddo
#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
        endif
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */
#endif /* WITH_REAL_AVX_BLOCK2_KERNEL */

#if defined(WITH_REAL_BGP_KERNEL)
#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
        if (THIS_REAL_ELPA_KERNEL .eq. REAL_ELPA_KERNEL_BGP) then
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */
          do j = ncols, 2, -2
            w(:,1) = bcast_buffer(1:nbw,j+off)
            w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP
            call double_hh_trafo_bgp(a(1,j+off+a_off-1,istripe,my_thread), w, nbw, nl, &
                                          stripe_width, nbw)
#else
            call double_hh_trafo_bgp(a(1,j+off+a_off-1,istripe), w, nbw, nl, &
                                          stripe_width, nbw)
#endif
          enddo
#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
        endif
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */
#endif /* WITH_REAL_BGP_KERNEL */


#if defined(WITH_REAL_BGQ_KERNEL)
#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
        if (THIS_REAL_ELPA_KERNEL .eq. REAL_ELPA_KERNEL_BGQ) then
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */
          do j = ncols, 2, -2
            w(:,1) = bcast_buffer(1:nbw,j+off)
            w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP
            call double_hh_trafo_bgq(a(1,j+off+a_off-1,istripe,my_thread), w, nbw, nl, &
                                          stripe_width, nbw)
#else
            call double_hh_trafo_bgq(a(1,j+off+a_off-1,istripe), w, nbw, nl, &
                                          stripe_width, nbw)
#endif
          enddo
#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
        endif
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */
#endif /* WITH_REAL_BGQ_KERNEL */


!#if defined(WITH_AVX_SANDYBRIDGE)
!              call double_hh_trafo_real_sse_avx_2hv(a(1,j+off+a_off-1,istripe), w, nbw, nl, stripe_width, nbw)
!#endif

#ifdef WITH_OPENMP
        if(j==1) call single_hh_trafo(a(1,1+off+a_off,istripe,my_thread), &
                                      bcast_buffer(1,off+1), nbw, nl,     &
                                      stripe_width)
#else
        if(j==1) call single_hh_trafo(a(1,1+off+a_off,istripe),           &
                                      bcast_buffer(1,off+1), nbw, nl,     &
                                      stripe_width)
#endif


#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
      endif !
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */



#if defined(WITH_REAL_AVX_BLOCK4_KERNEL)
#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
      if (THIS_REAL_ELPA_KERNEL .eq. REAL_ELPA_KERNEL_AVX_BLOCK4) then
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */
        ! X86 INTRINSIC CODE, USING 4 HOUSEHOLDER VECTORS
        do j = ncols, 4, -4
          w(:,1) = bcast_buffer(1:nbw,j+off)
          w(:,2) = bcast_buffer(1:nbw,j+off-1)
          w(:,3) = bcast_buffer(1:nbw,j+off-2)
          w(:,4) = bcast_buffer(1:nbw,j+off-3)
#ifdef WITH_OPENMP
          call quad_hh_trafo_real_sse_avx_4hv(a(1,j+off+a_off-3,istripe,my_thread), w, &
                                                  nbw, nl, stripe_width, nbw)
#else
          call quad_hh_trafo_real_sse_avx_4hv(a(1,j+off+a_off-3,istripe), w, &
                                                  nbw, nl, stripe_width, nbw)
#endif
        enddo
        do jj = j, 2, -2
          w(:,1) = bcast_buffer(1:nbw,jj+off)
          w(:,2) = bcast_buffer(1:nbw,jj+off-1)
#ifdef WITH_OPENMP
          call double_hh_trafo_real_sse_avx_2hv(a(1,jj+off+a_off-1,istripe,my_thread), &
                                                    w, nbw, nl, stripe_width, nbw)
#else
          call double_hh_trafo_real_sse_avx_2hv(a(1,jj+off+a_off-1,istripe), &
                                                    w, nbw, nl, stripe_width, nbw)
#endif
        enddo
#ifdef WITH_OPENMP
        if (jj==1) call single_hh_trafo(a(1,1+off+a_off,istripe,my_thread), &
                                          bcast_buffer(1,off+1), nbw, nl, stripe_width)
#else
        if (jj==1) call single_hh_trafo(a(1,1+off+a_off,istripe), &
                                          bcast_buffer(1,off+1), nbw, nl, stripe_width)
#endif
#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
      endif
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */
#endif /* WITH_REAL_AVX_BLOCK4_KERNEL */


#if defined(WITH_REAL_AVX_BLOCK6_KERNEL)
#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
      if (THIS_REAL_ELPA_KERNEL .eq. REAL_ELPA_KERNEL_AVX_BLOCK6) then
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */
        ! X86 INTRINSIC CODE, USING 6 HOUSEHOLDER VECTORS
        do j = ncols, 6, -6
          w(:,1) = bcast_buffer(1:nbw,j+off)
          w(:,2) = bcast_buffer(1:nbw,j+off-1)
          w(:,3) = bcast_buffer(1:nbw,j+off-2)
          w(:,4) = bcast_buffer(1:nbw,j+off-3)
          w(:,5) = bcast_buffer(1:nbw,j+off-4)
          w(:,6) = bcast_buffer(1:nbw,j+off-5)
#ifdef WITH_OPENMP
          call hexa_hh_trafo_real_sse_avx_6hv(a(1,j+off+a_off-5,istripe,my_thread), w, &
                                                  nbw, nl, stripe_width, nbw)
#else
          call hexa_hh_trafo_real_sse_avx_6hv(a(1,j+off+a_off-5,istripe), w, &
                                                  nbw, nl, stripe_width, nbw)
#endif
        enddo
        do jj = j, 4, -4
          w(:,1) = bcast_buffer(1:nbw,jj+off)
          w(:,2) = bcast_buffer(1:nbw,jj+off-1)
          w(:,3) = bcast_buffer(1:nbw,jj+off-2)
          w(:,4) = bcast_buffer(1:nbw,jj+off-3)
#ifdef WITH_OPENMP
          call quad_hh_trafo_real_sse_avx_4hv(a(1,jj+off+a_off-3,istripe,my_thread), w, &
                                                  nbw, nl, stripe_width, nbw)
#else
          call quad_hh_trafo_real_sse_avx_4hv(a(1,jj+off+a_off-3,istripe), w, &
                                                  nbw, nl, stripe_width, nbw)
#endif
        enddo
        do jjj = jj, 2, -2
          w(:,1) = bcast_buffer(1:nbw,jjj+off)
          w(:,2) = bcast_buffer(1:nbw,jjj+off-1)
#ifdef WITH_OPENMP
        call double_hh_trafo_real_sse_avx_2hv(a(1,jjj+off+a_off-1,istripe,my_thread), &
                                                    w, nbw, nl, stripe_width, nbw)
#else
        call double_hh_trafo_real_sse_avx_2hv(a(1,jjj+off+a_off-1,istripe), &
                                                    w, nbw, nl, stripe_width, nbw)
#endif
      enddo
#ifdef WITH_OPENMP
      if (jjj==1) call single_hh_trafo(a(1,1+off+a_off,istripe,my_thread), &
                                           bcast_buffer(1,off+1), nbw, nl, stripe_width)
#else
      if (jjj==1) call single_hh_trafo(a(1,1+off+a_off,istripe), &
                                           bcast_buffer(1,off+1), nbw, nl, stripe_width)
#endif
#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
    endif
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */
#endif /* WITH_REAL_AVX_BLOCK4_KERNEL */

#ifdef WITH_OPENMP
    if (my_thread==1) then
#endif
      kernel_flops = kernel_flops + 4*int(nl,8)*int(ncols,8)*int(nbw,8)
      kernel_time = kernel_time + mpi_wtime()-ttt
#ifdef WITH_OPENMP
    endif
#endif
#ifdef HAVE_DETAILED_TIMINGS
    call timer%stop("compute_hh_trafo")
#endif

  end subroutine compute_hh_trafo

 end subroutine  trans_ev_tridi_to_band_real

!-------------------------------------------------------------------------------

subroutine single_hh_trafo(q, hh, nb, nq, ldq)
#ifdef HAVE_DETAILED_TIMINGS
    use timings
#endif

    ! Perform single real Householder transformation.
    ! This routine is not performance critical and thus it is coded here in Fortran

    implicit none
    integer  :: nb, nq, ldq
    real*8   :: q(ldq, *), hh(*)

    integer  :: i
    real*8   :: v(nq)

#ifdef HAVE_DETAILED_TIMINGS
    call timer%start("single_hh_trafo")
#endif

    ! v = q * hh
    v(:) = q(1:nq,1)
    do i=2,nb
      v(:) = v(:) + q(1:nq,i) * hh(i)
    enddo

    ! v = v * tau
    v(:) = v(:) * hh(1)

    ! q = q - v * hh**T
    q(1:nq,1) = q(1:nq,1) - v(:)
    do i=2,nb
      q(1:nq,i) = q(1:nq,i) - v(:) * hh(i)
    enddo

#ifdef HAVE_DETAILED_TIMINGS
    call timer%stop("single_hh_trafo")
#endif


end subroutine

!-------------------------------------------------------------------------------

subroutine determine_workload(na, nb, nprocs, limits)
#ifdef HAVE_DETAILED_TIMINGS
    use timings
#endif
    implicit none

    integer, intent(in)  :: na, nb, nprocs
    integer, intent(out) :: limits(0:nprocs)

    integer              :: i

#ifdef HAVE_DETAILED_TIMINGS
    call timer%start("determine_workload")
#endif

    if (na <= 0) then
      limits(:) = 0
      return
    endif

    if (nb*nprocs > na) then
        ! there is not enough work for all
      do i = 0, nprocs
        limits(i) = min(na, i*nb)
      enddo
    else
       do i = 0, nprocs
         limits(i) = (i*na)/nprocs
       enddo
    endif

#ifdef HAVE_DETAILED_TIMINGS
    call timer%stop("determine_workload")
#endif
end subroutine

!-------------------------------------------------------------------------------

subroutine bandred_complex(na, a, lda, nblk, nbw, matrixCols, numBlocks, mpi_comm_rows, mpi_comm_cols, tmat, wantDebug, success)

!-------------------------------------------------------------------------------
!  bandred_complex: Reduces a distributed hermitian matrix to band form
!
!  Parameters
!
!  na          Order of matrix
!
!  a(lda,matrixCols)    Distributed matrix which should be reduced.
!              Distribution is like in Scalapack.
!              Opposed to Scalapack, a(:,:) must be set completely (upper and lower half)
!              a(:,:) is overwritten on exit with the band and the Householder vectors
!              in the upper half.
!
!  lda         Leading dimension of a
!  matrixCols  local columns of matrix a
!
!  nblk        blocksize of cyclic distribution, must be the same in both directions!
!
!  nbw         semi bandwith of output matrix
!
!  mpi_comm_rows
!  mpi_comm_cols
!              MPI-Communicators for rows/columns
!
!  tmat(nbw,nbw,numBlocks)    where numBlocks = (na-1)/nbw + 1
!              Factors for the Householder vectors (returned), needed for back transformation
!
!-------------------------------------------------------------------------------
#ifdef HAVE_DETAILED_TIMINGS
   use timings
#endif
   implicit none

   integer                 :: na, lda, nblk, nbw, matrixCols, numBlocks, mpi_comm_rows, mpi_comm_cols
   complex*16              :: a(lda,matrixCols), tmat(nbw,nbw,numBlocks)

   complex*16, parameter   :: CZERO = (0.d0,0.d0), CONE = (1.d0,0.d0)

   integer                 :: my_prow, my_pcol, np_rows, np_cols, mpierr
   integer                 :: l_cols, l_rows
   integer                 :: i, j, lcs, lce, lre, lc, lr, cur_pcol, n_cols, nrow
   integer                 :: istep, ncol, lch, lcx, nlc
   integer                 :: tile_size, l_rows_tile, l_cols_tile

   real*8                  :: vnorm2
   complex*16              :: xf, aux1(nbw), aux2(nbw), vrl, tau, vav(nbw,nbw)

   complex*16, allocatable :: tmp(:,:), vr(:), vmr(:,:), umc(:,:)

   logical, intent(in)  :: wantDebug
   logical, intent(out) :: success
#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("bandred_complex")
#endif
   call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
   call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
   call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
   call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)

   success = .true.

   ! Semibandwith nbw must be a multiple of blocksize nblk

   if (mod(nbw,nblk)/=0) then
     if (my_prow==0 .and. my_pcol==0) then
       if (wantDebug) then
         write(error_unit,*) 'ELPA2_bandred_complex: ERROR: nbw=',nbw,', nblk=',nblk
         write(error_unit,*) 'ELPA2_bandred_complex: ELPA2 works only for nbw==n*nblk'
       endif
       success = .false.
       return
     endif
   endif

   ! Matrix is split into tiles; work is done only for tiles on the diagonal or above

   tile_size = nblk*least_common_multiple(np_rows,np_cols) ! minimum global tile size
   tile_size = ((128*max(np_rows,np_cols)-1)/tile_size+1)*tile_size ! make local tiles at least 128 wide

   l_rows_tile = tile_size/np_rows ! local rows of a tile
   l_cols_tile = tile_size/np_cols ! local cols of a tile

   do istep = (na-1)/nbw, 1, -1

     n_cols = MIN(na,(istep+1)*nbw) - istep*nbw ! Number of columns in current step

     ! Number of local columns/rows of remaining matrix
     l_cols = local_index(istep*nbw, my_pcol, np_cols, nblk, -1)
     l_rows = local_index(istep*nbw, my_prow, np_rows, nblk, -1)

     ! Allocate vmr and umc to their exact sizes so that they can be used in bcasts and reduces

     allocate(vmr(max(l_rows,1),2*n_cols))
     allocate(umc(max(l_cols,1),2*n_cols))

     allocate(vr(l_rows+1))

     vmr(1:l_rows,1:n_cols) = 0.
     vr(:) = 0
     tmat(:,:,istep) = 0

     ! Reduce current block to lower triangular form

     do lc = n_cols, 1, -1

       ncol = istep*nbw + lc ! absolute column number of householder vector
       nrow = ncol - nbw ! Absolute number of pivot row

       lr  = local_index(nrow, my_prow, np_rows, nblk, -1) ! current row length
       lch = local_index(ncol, my_pcol, np_cols, nblk, -1) ! HV local column number

       tau = 0

       if(nrow == 1) exit ! Nothing to do

       cur_pcol = pcol(ncol, nblk, np_cols) ! Processor column owning current block

       if (my_pcol==cur_pcol) then

         ! Get vector to be transformed; distribute last element and norm of
         ! remaining elements to all procs in current column

         vr(1:lr) = a(1:lr,lch) ! vector to be transformed

         if (my_prow==prow(nrow, nblk, np_rows)) then
           aux1(1) = dot_product(vr(1:lr-1),vr(1:lr-1))
           aux1(2) = vr(lr)
         else
           aux1(1) = dot_product(vr(1:lr),vr(1:lr))
           aux1(2) = 0.
         endif

         call mpi_allreduce(aux1,aux2,2,MPI_DOUBLE_COMPLEX,MPI_SUM,mpi_comm_rows,mpierr)

         vnorm2 = aux2(1)
         vrl    = aux2(2)

         ! Householder transformation

         call hh_transform_complex(vrl, vnorm2, xf, tau)

         ! Scale vr and store Householder vector for back transformation

         vr(1:lr) = vr(1:lr) * xf
         if (my_prow==prow(nrow, nblk, np_rows)) then
           a(1:lr-1,lch) = vr(1:lr-1)
           a(lr,lch) = vrl
           vr(lr) = 1.
         else
           a(1:lr,lch) = vr(1:lr)
         endif

       endif

       ! Broadcast Householder vector and tau along columns

       vr(lr+1) = tau
       call MPI_Bcast(vr,lr+1,MPI_DOUBLE_COMPLEX,cur_pcol,mpi_comm_cols,mpierr)
       vmr(1:lr,lc) = vr(1:lr)
       tau = vr(lr+1)
       tmat(lc,lc,istep) = conjg(tau) ! Store tau in diagonal of tmat

       ! Transform remaining columns in current block with Householder vector

       ! Local dot product

       aux1 = 0

       nlc = 0 ! number of local columns
       do j=1,lc-1
         lcx = local_index(istep*nbw+j, my_pcol, np_cols, nblk, 0)
         if (lcx>0) then
           nlc = nlc+1
           aux1(nlc) = dot_product(vr(1:lr),a(1:lr,lcx))
         endif
       enddo

       ! Get global dot products
       if (nlc>0) call mpi_allreduce(aux1,aux2,nlc,MPI_DOUBLE_COMPLEX,MPI_SUM,mpi_comm_rows,mpierr)

       ! Transform

       nlc = 0
       do j=1,lc-1
         lcx = local_index(istep*nbw+j, my_pcol, np_cols, nblk, 0)
         if (lcx>0) then
           nlc = nlc+1
           a(1:lr,lcx) = a(1:lr,lcx) - conjg(tau)*aux2(nlc)*vr(1:lr)
         endif
       enddo

     enddo

     ! Calculate scalar products of stored Householder vectors.
     ! This can be done in different ways, we use zherk

     vav = 0
     if (l_rows>0) &
        call zherk('U','C',n_cols,l_rows,CONE,vmr,ubound(vmr,dim=1),CZERO,vav,ubound(vav,dim=1))
     call herm_matrix_allreduce(n_cols,vav, nbw,nbw,mpi_comm_rows)

     ! Calculate triangular matrix T for block Householder Transformation

     do lc=n_cols,1,-1
       tau = tmat(lc,lc,istep)
       if (lc<n_cols) then
         call ztrmv('U','C','N',n_cols-lc,tmat(lc+1,lc+1,istep),ubound(tmat,dim=1),vav(lc+1,lc),1)
         tmat(lc,lc+1:n_cols,istep) = -tau * conjg(vav(lc+1:n_cols,lc))
       endif
     enddo

     ! Transpose vmr -> vmc (stored in umc, second half)

     call elpa_transpose_vectors_complex  (vmr, ubound(vmr,dim=1), mpi_comm_rows, &
                                   umc(1,n_cols+1), ubound(umc,dim=1), mpi_comm_cols, &
                                   1, istep*nbw, n_cols, nblk)

     ! Calculate umc = A**T * vmr
     ! Note that the distributed A has to be transposed
     ! Opposed to direct tridiagonalization there is no need to use the cache locality
     ! of the tiles, so we can use strips of the matrix

     umc(1:l_cols,1:n_cols) = 0.d0
     vmr(1:l_rows,n_cols+1:2*n_cols) = 0
     if (l_cols>0 .and. l_rows>0) then
       do i=0,(istep*nbw-1)/tile_size

         lcs = i*l_cols_tile+1
         lce = min(l_cols,(i+1)*l_cols_tile)
         if (lce<lcs) cycle

         lre = min(l_rows,(i+1)*l_rows_tile)
         call ZGEMM('C','N',lce-lcs+1,n_cols,lre,CONE,a(1,lcs),ubound(a,dim=1), &
                      vmr,ubound(vmr,dim=1),CONE,umc(lcs,1),ubound(umc,dim=1))

         if (i==0) cycle
         lre = min(l_rows,i*l_rows_tile)
         call ZGEMM('N','N',lre,n_cols,lce-lcs+1,CONE,a(1,lcs),lda, &
                      umc(lcs,n_cols+1),ubound(umc,dim=1),CONE,vmr(1,n_cols+1),ubound(vmr,dim=1))
       enddo
     endif

     ! Sum up all ur(:) parts along rows and add them to the uc(:) parts
     ! on the processors containing the diagonal
     ! This is only necessary if ur has been calculated, i.e. if the
     ! global tile size is smaller than the global remaining matrix

     if (tile_size < istep*nbw) then
       call elpa_reduce_add_vectors_complex  (vmr(1,n_cols+1),ubound(vmr,dim=1),mpi_comm_rows, &
                                       umc, ubound(umc,dim=1), mpi_comm_cols, &
                                       istep*nbw, n_cols, nblk)
     endif

     if (l_cols>0) then
       allocate(tmp(l_cols,n_cols))
       call mpi_allreduce(umc,tmp,l_cols*n_cols,MPI_DOUBLE_COMPLEX,MPI_SUM,mpi_comm_rows,mpierr)
       umc(1:l_cols,1:n_cols) = tmp(1:l_cols,1:n_cols)
       deallocate(tmp)
     endif

     ! U = U * Tmat**T

     call ztrmm('Right','Upper','C','Nonunit',l_cols,n_cols,CONE,tmat(1,1,istep),ubound(tmat,dim=1),umc,ubound(umc,dim=1))

     ! VAV = Tmat * V**T * A * V * Tmat**T = (U*Tmat**T)**T * V * Tmat**T

     call zgemm('C','N',n_cols,n_cols,l_cols,CONE,umc,ubound(umc,dim=1),umc(1,n_cols+1), &
         ubound(umc,dim=1),CZERO,vav,ubound(vav,dim=1))
     call ztrmm('Right','Upper','C','Nonunit',n_cols,n_cols,CONE,tmat(1,1,istep),ubound(tmat,dim=1),vav,ubound(vav,dim=1))

     call herm_matrix_allreduce(n_cols,vav, nbw,nbw,mpi_comm_cols)

     ! U = U - 0.5 * V * VAV
     call zgemm('N','N',l_cols,n_cols,n_cols,(-0.5d0,0.d0),umc(1,n_cols+1),ubound(umc,dim=1),vav,ubound(vav,dim=1), &
         CONE,umc,ubound(umc,dim=1))

     ! Transpose umc -> umr (stored in vmr, second half)

     call elpa_transpose_vectors_complex  (umc, ubound(umc,dim=1), mpi_comm_cols, &
                                    vmr(1,n_cols+1), ubound(vmr,dim=1), mpi_comm_rows, &
                                    1, istep*nbw, n_cols, nblk)

     ! A = A - V*U**T - U*V**T

     do i=0,(istep*nbw-1)/tile_size
       lcs = i*l_cols_tile+1
       lce = min(l_cols,(i+1)*l_cols_tile)
       lre = min(l_rows,(i+1)*l_rows_tile)
       if (lce<lcs .or. lre<1) cycle
       call zgemm('N','C',lre,lce-lcs+1,2*n_cols,-CONE, &
                   vmr,ubound(vmr,dim=1),umc(lcs,1),ubound(umc,dim=1), &
                   CONE,a(1,lcs),lda)
     enddo

     deallocate(vmr, umc, vr)

   enddo
#ifdef HAVE_DETAILED_TIMINGS
   call timer%stop("bandred_complex")
#endif

end subroutine bandred_complex

!-------------------------------------------------------------------------------

subroutine herm_matrix_allreduce(n,a,lda,ldb,comm)

!-------------------------------------------------------------------------------
!  herm_matrix_allreduce: Does an mpi_allreduce for a hermitian matrix A.
!  On entry, only the upper half of A needs to be set
!  On exit, the complete matrix is set
!-------------------------------------------------------------------------------
#ifdef HAVE_DETAILED_TIMINGS
   use timings
#endif
   implicit none
   integer    :: n, lda, ldb, comm
   complex*16 :: a(lda,ldb)

   integer    :: i, nc, mpierr
   complex*16 :: h1(n*n), h2(n*n)

#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("herm_matrix_allreduce")
#endif

   nc = 0
   do i=1,n
     h1(nc+1:nc+i) = a(1:i,i)
     nc = nc+i
   enddo

   call mpi_allreduce(h1,h2,nc,MPI_DOUBLE_COMPLEX,MPI_SUM,comm,mpierr)

   nc = 0
   do i=1,n
     a(1:i,i) = h2(nc+1:nc+i)
     a(i,1:i-1) = conjg(a(1:i-1,i))
     nc = nc+i
   enddo

#ifdef HAVE_DETAILED_TIMINGS
   call timer%stop("herm_matrix_allreduce")
#endif

end subroutine herm_matrix_allreduce

!-------------------------------------------------------------------------------

subroutine trans_ev_band_to_full_complex(na, nqc, nblk, nbw, a, lda, tmat, q, ldq, matrixCols,  &
                                         numBlocks, mpi_comm_rows, mpi_comm_cols)

!-------------------------------------------------------------------------------
!  trans_ev_band_to_full_complex:
!  Transforms the eigenvectors of a band matrix back to the eigenvectors of the original matrix
!
!  Parameters
!
!  na          Order of matrix a, number of rows of matrix q
!
!  nqc         Number of columns of matrix q
!
!  nblk        blocksize of cyclic distribution, must be the same in both directions!
!
!  nbw         semi bandwith
!
!  a(lda,matrixCols)    Matrix containing the Householder vectors (i.e. matrix a after bandred_complex)
!              Distribution is like in Scalapack.
!
!  lda         Leading dimension of a
!  matrixCols  local columns of matrix a and q
!
!  tmat(nbw,nbw,numBlocks) Factors returned by bandred_complex
!
!  q           On input: Eigenvectors of band matrix
!              On output: Transformed eigenvectors
!              Distribution is like in Scalapack.
!
!  ldq         Leading dimension of q
!
!  mpi_comm_rows
!  mpi_comm_cols
!              MPI-Communicators for rows/columns
!
!-------------------------------------------------------------------------------
#ifdef HAVE_DETAILED_TIMINGS
 use timings
#endif
   implicit none

   integer                 :: na, nqc, lda, ldq, nblk, nbw, matrixCols, numBlocks, mpi_comm_rows, mpi_comm_cols
   complex*16              :: a(lda,matrixCols), q(ldq,matrixCols), tmat(nbw, nbw, numBlocks)

   complex*16, parameter   :: CZERO = (0.d0,0.d0), CONE = (1.d0,0.d0)

   integer                 :: my_prow, my_pcol, np_rows, np_cols, mpierr
   integer                 :: max_blocks_row, max_blocks_col, max_local_rows, max_local_cols
   integer                 :: l_cols, l_rows, l_colh, n_cols
   integer                 :: istep, lc, ncol, nrow, nb, ns

   complex*16, allocatable :: tmp1(:), tmp2(:), hvb(:), hvm(:,:)

   integer                 :: i

#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("trans_ev_band_to_full_complex")
#endif
   call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
   call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
   call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
   call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)

   max_blocks_row = ((na -1)/nblk)/np_rows + 1  ! Rows of A
   max_blocks_col = ((nqc-1)/nblk)/np_cols + 1  ! Columns of q!

   max_local_rows = max_blocks_row*nblk
   max_local_cols = max_blocks_col*nblk

   allocate(tmp1(max_local_cols*nbw))
   allocate(tmp2(max_local_cols*nbw))
   allocate(hvb(max_local_rows*nbw))
   allocate(hvm(max_local_rows,nbw))

   hvm = 0   ! Must be set to 0 !!!
   hvb = 0   ! Safety only

   l_cols = local_index(nqc, my_pcol, np_cols, nblk, -1) ! Local columns of q

   do istep=1,(na-1)/nbw

     n_cols = MIN(na,(istep+1)*nbw) - istep*nbw ! Number of columns in current step

     ! Broadcast all Householder vectors for current step compressed in hvb

     nb = 0
     ns = 0

     do lc = 1, n_cols
       ncol = istep*nbw + lc ! absolute column number of householder vector
       nrow = ncol - nbw ! absolute number of pivot row

       l_rows = local_index(nrow-1, my_prow, np_rows, nblk, -1) ! row length for bcast
       l_colh = local_index(ncol  , my_pcol, np_cols, nblk, -1) ! HV local column number

       if (my_pcol==pcol(ncol, nblk, np_cols)) hvb(nb+1:nb+l_rows) = a(1:l_rows,l_colh)

       nb = nb+l_rows

       if (lc==n_cols .or. mod(ncol,nblk)==0) then
         call MPI_Bcast(hvb(ns+1),nb-ns,MPI_DOUBLE_COMPLEX,pcol(ncol, nblk, np_cols),mpi_comm_cols,mpierr)
         ns = nb
       endif
     enddo

     ! Expand compressed Householder vectors into matrix hvm

     nb = 0
     do lc = 1, n_cols
       nrow = (istep-1)*nbw+lc ! absolute number of pivot row
       l_rows = local_index(nrow-1, my_prow, np_rows, nblk, -1) ! row length for bcast

       hvm(1:l_rows,lc) = hvb(nb+1:nb+l_rows)
       if (my_prow==prow(nrow, nblk, np_rows)) hvm(l_rows+1,lc) = 1.

       nb = nb+l_rows
     enddo

     l_rows = local_index(MIN(na,(istep+1)*nbw), my_prow, np_rows, nblk, -1)

     ! Q = Q - V * T**T * V**T * Q

     if (l_rows>0) then
       call zgemm('C','N',n_cols,l_cols,l_rows,CONE,hvm,ubound(hvm,dim=1), &
                   q,ldq,CZERO,tmp1,n_cols)
     else
       tmp1(1:l_cols*n_cols) = 0
     endif
     call mpi_allreduce(tmp1,tmp2,n_cols*l_cols,MPI_DOUBLE_COMPLEX,MPI_SUM,mpi_comm_rows,mpierr)
     if (l_rows>0) then
       call ztrmm('L','U','C','N',n_cols,l_cols,CONE,tmat(1,1,istep),ubound(tmat,dim=1),tmp2,n_cols)
       call zgemm('N','N',l_rows,l_cols,n_cols,-CONE,hvm,ubound(hvm,dim=1), &
                   tmp2,n_cols,CONE,q,ldq)
     endif

   enddo

   deallocate(tmp1, tmp2, hvb, hvm)

#ifdef HAVE_DETAILED_TIMINGS
   call timer%stop("trans_ev_band_to_full_complex")
#endif

 end subroutine trans_ev_band_to_full_complex

!---------------------------------------------------------------------------------------------------

subroutine tridiag_band_complex(na, nb, nblk, a, lda, d, e, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm)

!-------------------------------------------------------------------------------
! tridiag_band_complex:
! Reduces a complex hermitian symmetric band matrix to tridiagonal form
!
!  na          Order of matrix a
!
!  nb          Semi bandwith
!
!  nblk        blocksize of cyclic distribution, must be the same in both directions!
!
!  a(lda,matrixCols)    Distributed system matrix reduced to banded form in the upper diagonal
!
!  lda         Leading dimension of a
!  matrixCols  local columns of matrix a
!
!  d(na)       Diagonal of tridiagonal matrix, set only on PE 0 (output)
!
!  e(na)       Subdiagonal of tridiagonal matrix, set only on PE 0 (output)
!
!  mpi_comm_rows
!  mpi_comm_cols
!              MPI-Communicators for rows/columns
!  mpi_comm
!              MPI-Communicator for the total processor set
!-------------------------------------------------------------------------------
#ifdef HAVE_DETAILED_TIMINGS
 use timings
#endif
   implicit none

   integer, intent(in)      ::  na, nb, nblk, lda, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm
   complex*16, intent(in)   :: a(lda,matrixCols)
   real*8, intent(out)      :: d(na), e(na) ! set only on PE 0


   real*8                   :: vnorm2
   complex*16               :: hv(nb), tau, x, h(nb), ab_s(1+nb), hv_s(nb), hv_new(nb), tau_new, hf
   complex*16               :: hd(nb), hs(nb)

   integer                  :: i, j, n, nc, nr, ns, ne, istep, iblk, nblocks_total, nblocks, nt
   integer                  :: my_pe, n_pes, mpierr
   integer                  :: my_prow, np_rows, my_pcol, np_cols
   integer                  :: ireq_ab, ireq_hv
   integer                  :: na_s, nx, num_hh_vecs, num_chunks, local_size, max_blk_size, n_off
#ifdef WITH_OPENMP
    integer, allocatable    :: mpi_statuses(:,:)
    integer, allocatable    :: omp_block_limits(:)
    integer                 :: max_threads, my_thread, my_block_s, my_block_e, iter
    integer                 :: omp_get_max_threads
    integer                 :: mpi_status(MPI_STATUS_SIZE)
    complex*16, allocatable :: hv_t(:,:), tau_t(:)
#endif
   integer, allocatable     :: ireq_hhr(:), ireq_hhs(:), global_id(:,:), hh_cnt(:), hh_dst(:)
   integer, allocatable     :: limits(:), snd_limits(:,:)
   integer, allocatable     :: block_limits(:)
   complex*16, allocatable  :: ab(:,:), hh_gath(:,:,:), hh_send(:,:,:)
!   ! dummies for calling redist_band
!   real*8                   :: r_a(1,1), r_ab(1,1)

#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("tridiag_band_complex")
#endif
   call mpi_comm_rank(mpi_comm,my_pe,mpierr)
   call mpi_comm_size(mpi_comm,n_pes,mpierr)

   call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
   call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
   call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
   call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)

   ! Get global_id mapping 2D procssor coordinates to global id

   allocate(global_id(0:np_rows-1,0:np_cols-1))
   global_id(:,:) = 0
   global_id(my_prow, my_pcol) = my_pe

   call mpi_allreduce(mpi_in_place, global_id, np_rows*np_cols, mpi_integer, mpi_sum, mpi_comm, mpierr)


   ! Total number of blocks in the band:

   nblocks_total = (na-1)/nb + 1

   ! Set work distribution

   allocate(block_limits(0:n_pes))
   call divide_band(nblocks_total, n_pes, block_limits)

   ! nblocks: the number of blocks for my task
   nblocks = block_limits(my_pe+1) - block_limits(my_pe)

   ! allocate the part of the band matrix which is needed by this PE
   ! The size is 1 block larger than needed to avoid extensive shifts
   allocate(ab(2*nb,(nblocks+1)*nb))
   ab = 0 ! needed for lower half, the extra block should also be set to 0 for safety

   ! n_off: Offset of ab within band
   n_off = block_limits(my_pe)*nb

   ! Redistribute band in a to ab
   call redist_band_complex(a, lda, na, nblk, nb, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm, ab)

   ! Calculate the workload for each sweep in the back transformation
   ! and the space requirements to hold the HH vectors

   allocate(limits(0:np_rows))
   call determine_workload(na, nb, np_rows, limits)
   max_blk_size = maxval(limits(1:np_rows) - limits(0:np_rows-1))

   num_hh_vecs = 0
   num_chunks  = 0
   nx = na
   do n = 1, nblocks_total
     call determine_workload(nx, nb, np_rows, limits)
     local_size = limits(my_prow+1) - limits(my_prow)
     ! add to number of householder vectors
     ! please note: for nx==1 the one and only HH vector is 0 and is neither calculated nor send below!
     if (mod(n-1,np_cols) == my_pcol .and. local_size>0 .and. nx>1) then
       num_hh_vecs = num_hh_vecs + local_size
       num_chunks  = num_chunks+1
     endif
     nx = nx - nb
   enddo

   ! Allocate space for HH vectors

   allocate(hh_trans_complex(nb,num_hh_vecs))

   ! Allocate and init MPI requests

   allocate(ireq_hhr(num_chunks)) ! Recv requests
   allocate(ireq_hhs(nblocks))    ! Send requests

   num_hh_vecs = 0
   num_chunks  = 0
   nx = na
   nt = 0
   do n = 1, nblocks_total
     call determine_workload(nx, nb, np_rows, limits)
     local_size = limits(my_prow+1) - limits(my_prow)
     if (mod(n-1,np_cols) == my_pcol .and. local_size>0 .and. nx>1) then
       num_chunks  = num_chunks+1
       call mpi_irecv(hh_trans_complex(1,num_hh_vecs+1), nb*local_size, MPI_COMPLEX16, nt, &
                        10+n-block_limits(nt), mpi_comm, ireq_hhr(num_chunks), mpierr)
       num_hh_vecs = num_hh_vecs + local_size
     endif
     nx = nx - nb
     if (n == block_limits(nt+1)) then
       nt = nt + 1
     endif
   enddo

   ireq_hhs(:) = MPI_REQUEST_NULL

   ! Buffers for gathering/sending the HH vectors

   allocate(hh_gath(nb,max_blk_size,nblocks)) ! gathers HH vectors
   allocate(hh_send(nb,max_blk_size,nblocks)) ! send buffer for HH vectors
   hh_gath(:,:,:) = 0
   hh_send(:,:,:) = 0

   ! Some counters

   allocate(hh_cnt(nblocks))
   allocate(hh_dst(nblocks))

   hh_cnt(:) = 1 ! The first transfomation vector is always 0 and not calculated at all
   hh_dst(:) = 0 ! PE number for receive

   ireq_ab = MPI_REQUEST_NULL
   ireq_hv = MPI_REQUEST_NULL

   ! Limits for sending

   allocate(snd_limits(0:np_rows,nblocks))

   do iblk=1,nblocks
     call determine_workload(na-(iblk+block_limits(my_pe)-1)*nb, nb, np_rows, snd_limits(:,iblk))
   enddo

#ifdef WITH_OPENMP
    ! OpenMP work distribution:

    max_threads = 1
!$ max_threads = omp_get_max_threads()

    ! For OpenMP we need at least 2 blocks for every thread
    max_threads = MIN(max_threads, nblocks/2)
    if (max_threads==0) max_threads = 1

    allocate(omp_block_limits(0:max_threads))

    ! Get the OpenMP block limits
    call divide_band(nblocks, max_threads, omp_block_limits)

    allocate(hv_t(nb,max_threads), tau_t(max_threads))
    hv_t = 0
    tau_t = 0
#endif


   ! ---------------------------------------------------------------------------
   ! Start of calculations

   na_s = block_limits(my_pe)*nb + 1

   if (my_pe>0 .and. na_s<=na) then
     ! send first column to previous PE
     ! Only the PE owning the diagonal does that (sending 1 element of the subdiagonal block also)
     ab_s(1:nb+1) = ab(1:nb+1,na_s-n_off)
     call mpi_isend(ab_s,nb+1,MPI_COMPLEX16,my_pe-1,1,mpi_comm,ireq_ab,mpierr)
   endif

#ifdef WITH_OPENMP
   do istep=1,na-1-block_limits(my_pe)*nb
#else
   do istep=1,na-1
#endif
     if (my_pe==0) then
       n = MIN(na-na_s,nb) ! number of rows to be reduced
       hv(:) = 0
       tau = 0
       ! Transform first column of remaining matrix
       ! Opposed to the real case, the last step (istep=na-1) is needed here for making
       ! the last subdiagonal element a real number
       vnorm2 = sum(dble(ab(3:n+1,na_s-n_off))**2+dimag(ab(3:n+1,na_s-n_off))**2)
       if (n<2) vnorm2 = 0. ! Safety only
       call hh_transform_complex(ab(2,na_s-n_off),vnorm2,hf,tau)

       hv(1) = 1
       hv(2:n) = ab(3:n+1,na_s-n_off)*hf

       d(istep) = ab(1,na_s-n_off)
       e(istep) = ab(2,na_s-n_off)
       if (istep == na-1) then
         d(na) = ab(1,na_s+1-n_off)
         e(na) = 0
       endif
     else
       if (na>na_s) then
         ! Receive Householder vector from previous task, from PE owning subdiagonal
#ifdef WITH_OPENMP
         call mpi_recv(hv,nb,MPI_COMPLEX16,my_pe-1,2,mpi_comm,mpi_status,mpierr)
#else
         call mpi_recv(hv,nb,MPI_COMPLEX16,my_pe-1,2,mpi_comm,MPI_STATUS_IGNORE,mpierr)
#endif
         tau = hv(1)
         hv(1) = 1.
       endif
     endif

     na_s = na_s+1
     if (na_s-n_off > nb) then
       ab(:,1:nblocks*nb) = ab(:,nb+1:(nblocks+1)*nb)
       ab(:,nblocks*nb+1:(nblocks+1)*nb) = 0
       n_off = n_off + nb
     endif
#ifdef WITH_OPENMP
     if (max_threads > 1) then

       ! Codepath for OpenMP

       ! Please note that in this case it is absolutely necessary to have at least 2 blocks per thread!
       ! Every thread is one reduction cycle behind its predecessor and thus starts one step later.
       ! This simulates the behaviour of the MPI tasks which also work after each other.
       ! The code would be considerably easier, if the MPI communication would be made within
       ! the parallel region - this is avoided here since this would require
       ! MPI_Init_thread(MPI_THREAD_MULTIPLE) at the start of the program.

       hv_t(:,1) = hv
       tau_t(1) = tau

       do iter = 1, 2

         ! iter=1 : work on first block
         ! iter=2 : work on remaining blocks
         ! This is done in 2 iterations so that we have a barrier in between:
         ! After the first iteration, it is guaranteed that the last row of the last block
         ! is completed by the next thread.
         ! After the first iteration it is also the place to exchange the last row
         ! with MPI calls
#ifdef HAVE_DETAILED_TIMINGS
         call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread, my_block_s, my_block_e, iblk, ns, ne, hv, tau, &
!$omp&                    nc, nr, hs, hd, vnorm2, hf, x, h, i), schedule(static,1), num_threads(max_threads)
         do my_thread = 1, max_threads

           if (iter == 1) then
             my_block_s = omp_block_limits(my_thread-1) + 1
             my_block_e = my_block_s
           else
             my_block_s = omp_block_limits(my_thread-1) + 2
             my_block_e = omp_block_limits(my_thread)
           endif

           do iblk = my_block_s, my_block_e

             ns = na_s + (iblk-1)*nb - n_off - my_thread + 1 ! first column in block
             ne = ns+nb-1                    ! last column in block

             if (istep<my_thread .or. ns+n_off>na) exit

             hv = hv_t(:,my_thread)
             tau = tau_t(my_thread)

             ! Store Householder vector for back transformation

             hh_cnt(iblk) = hh_cnt(iblk) + 1

             hh_gath(1   ,hh_cnt(iblk),iblk) = tau
             hh_gath(2:nb,hh_cnt(iblk),iblk) = hv(2:nb)

             nc = MIN(na-ns-n_off+1,nb) ! number of columns in diagonal block
             nr = MIN(na-nb-ns-n_off+1,nb) ! rows in subdiagonal block (may be < 0!!!)
                                            ! Note that nr>=0 implies that diagonal block is full (nc==nb)!

             ! Transform diagonal block

             call ZHEMV('L',nc,tau,ab(1,ns),2*nb-1,hv,1,(0.d0,0.d0),hd,1)

             x = dot_product(hv(1:nc),hd(1:nc))*conjg(tau)
             hd(1:nc) = hd(1:nc) - 0.5*x*hv(1:nc)

             call ZHER2('L',nc,(-1.d0,0.d0),hd,1,hv,1,ab(1,ns),2*nb-1)

             hv_t(:,my_thread) = 0
             tau_t(my_thread)  = 0

             if (nr<=0) cycle ! No subdiagonal block present any more

             ! Transform subdiagonal block

             call ZGEMV('N',nr,nb,tau,ab(nb+1,ns),2*nb-1,hv,1,(0.d0,0.d0),hs,1)

             if (nr>1) then

               ! complete (old) Householder transformation for first column

               ab(nb+1:nb+nr,ns) = ab(nb+1:nb+nr,ns) - hs(1:nr) ! Note: hv(1) == 1

               ! calculate new Householder transformation for first column
               ! (stored in hv_t(:,my_thread) and tau_t(my_thread))

               vnorm2 = sum(dble(ab(nb+2:nb+nr,ns))**2+dimag(ab(nb+2:nb+nr,ns))**2)
               call hh_transform_complex(ab(nb+1,ns),vnorm2,hf,tau_t(my_thread))
               hv_t(1   ,my_thread) = 1.
               hv_t(2:nr,my_thread) = ab(nb+2:nb+nr,ns)*hf
               ab(nb+2:,ns) = 0

               ! update subdiagonal block for old and new Householder transformation
               ! This way we can use a nonsymmetric rank 2 update which is (hopefully) faster

               call ZGEMV('C',nr,nb-1,tau_t(my_thread),ab(nb,ns+1),2*nb-1,hv_t(1,my_thread),1,(0.d0,0.d0),h(2),1)
               x = dot_product(hs(1:nr),hv_t(1:nr,my_thread))*tau_t(my_thread)
               h(2:nb) = h(2:nb) - x*hv(2:nb)
               ! Unfortunately there is no BLAS routine like DSYR2 for a nonsymmetric rank 2 update ("DGER2")
               do i=2,nb
                 ab(2+nb-i:1+nb+nr-i,i+ns-1) = ab(2+nb-i:1+nb+nr-i,i+ns-1) &
                                                - hv_t(1:nr,my_thread)*conjg(h(i)) - hs(1:nr)*conjg(hv(i))
               enddo

             else

               ! No new Householder transformation for nr=1, just complete the old one
               ab(nb+1,ns) = ab(nb+1,ns) - hs(1) ! Note: hv(1) == 1
               do i=2,nb
                 ab(2+nb-i,i+ns-1) = ab(2+nb-i,i+ns-1) - hs(1)*conjg(hv(i))
               enddo
               ! For safety: there is one remaining dummy transformation (but tau is 0 anyways)
               hv_t(1,my_thread) = 1.

             endif

           enddo

         enddo ! my_thread
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
         call timer%stop("OpenMP parallel")
#endif



         if (iter==1) then
           ! We are at the end of the first block

           ! Send our first column to previous PE
           if (my_pe>0 .and. na_s <= na) then
             call mpi_wait(ireq_ab,mpi_status,mpierr)
             ab_s(1:nb+1) = ab(1:nb+1,na_s-n_off)
             call mpi_isend(ab_s,nb+1,MPI_COMPLEX16,my_pe-1,1,mpi_comm,ireq_ab,mpierr)
           endif

           ! Request last column from next PE
           ne = na_s + nblocks*nb - (max_threads-1) - 1
           if (istep>=max_threads .and. ne <= na) then
             call mpi_recv(ab(1,ne-n_off),nb+1,MPI_COMPLEX16,my_pe+1,1,mpi_comm,mpi_status,mpierr)
           endif

         else
           ! We are at the end of all blocks

           ! Send last HH vector and TAU to next PE if it has been calculated above
           ne = na_s + nblocks*nb - (max_threads-1) - 1
           if (istep>=max_threads .and. ne < na) then
             call mpi_wait(ireq_hv,mpi_status,mpierr)
             hv_s(1) = tau_t(max_threads)
             hv_s(2:) = hv_t(2:,max_threads)
             call mpi_isend(hv_s,nb,MPI_COMPLEX16,my_pe+1,2,mpi_comm,ireq_hv,mpierr)
           endif

           ! "Send" HH vector and TAU to next OpenMP thread
           do my_thread = max_threads, 2, -1
             hv_t(:,my_thread) = hv_t(:,my_thread-1)
             tau_t(my_thread)  = tau_t(my_thread-1)
           enddo

         endif
       enddo ! iter

     else

       ! Codepath for 1 thread without OpenMP

       ! The following code is structured in a way to keep waiting times for
       ! other PEs at a minimum, especially if there is only one block.
       ! For this reason, it requests the last column as late as possible
       ! and sends the Householder vector and the first column as early
       ! as possible.

#endif

       do iblk=1,nblocks

         ns = na_s + (iblk-1)*nb - n_off ! first column in block
         ne = ns+nb-1                    ! last column in block

         if (ns+n_off>na) exit

         ! Store Householder vector for back transformation

         hh_cnt(iblk) = hh_cnt(iblk) + 1

         hh_gath(1   ,hh_cnt(iblk),iblk) = tau
         hh_gath(2:nb,hh_cnt(iblk),iblk) = hv(2:nb)


#ifndef WITH_OPENMP
         if (hh_cnt(iblk) == snd_limits(hh_dst(iblk)+1,iblk)-snd_limits(hh_dst(iblk),iblk)) then
           ! Wait for last transfer to finish
           call mpi_wait(ireq_hhs(iblk), MPI_STATUS_IGNORE, mpierr)
           ! Copy vectors into send buffer
           hh_send(:,1:hh_cnt(iblk),iblk) = hh_gath(:,1:hh_cnt(iblk),iblk)
           ! Send to destination
           call mpi_isend(hh_send(1,1,iblk), nb*hh_cnt(iblk), MPI_COMPLEX16, &
                           global_id(hh_dst(iblk),mod(iblk+block_limits(my_pe)-1,np_cols)), &
                           10+iblk, mpi_comm, ireq_hhs(iblk), mpierr)
           ! Reset counter and increase destination row
           hh_cnt(iblk) = 0
           hh_dst(iblk) = hh_dst(iblk)+1
         endif


         ! The following code is structured in a way to keep waiting times for
         ! other PEs at a minimum, especially if there is only one block.
         ! For this reason, it requests the last column as late as possible
         ! and sends the Householder vector and the first column as early
         ! as possible.
#endif

         nc = MIN(na-ns-n_off+1,nb) ! number of columns in diagonal block
         nr = MIN(na-nb-ns-n_off+1,nb) ! rows in subdiagonal block (may be < 0!!!)
                                       ! Note that nr>=0 implies that diagonal block is full (nc==nb)!


         ! Multiply diagonal block and subdiagonal block with Householder vector

         if (iblk==nblocks .and. nc==nb) then

           ! We need the last column from the next PE.
           ! First do the matrix multiplications without last column ...

           ! Diagonal block, the contribution of the last element is added below!
           ab(1,ne) = 0
           call ZHEMV('L',nc,tau,ab(1,ns),2*nb-1,hv,1,(0.d0,0.d0),hd,1)

           ! Subdiagonal block
           if (nr>0) call ZGEMV('N',nr,nb-1,tau,ab(nb+1,ns),2*nb-1,hv,1,(0.d0,0.d0),hs,1)

           ! ... then request last column ...
#ifdef WITH_OPENMP
           call mpi_recv(ab(1,ne),nb+1,MPI_COMPLEX16,my_pe+1,1,mpi_comm,mpi_status,mpierr)

#else
           call mpi_recv(ab(1,ne),nb+1,MPI_COMPLEX16,my_pe+1,1,mpi_comm,MPI_STATUS_IGNORE,mpierr)
#endif
           ! ... and complete the result
           hs(1:nr) = hs(1:nr) + ab(2:nr+1,ne)*tau*hv(nb)
           hd(nb) = hd(nb) + ab(1,ne)*hv(nb)*tau

         else
           ! Normal matrix multiply
           call ZHEMV('L',nc,tau,ab(1,ns),2*nb-1,hv,1,(0.d0,0.d0),hd,1)
           if (nr>0) call ZGEMV('N',nr,nb,tau,ab(nb+1,ns),2*nb-1,hv,1,(0.d0,0.d0),hs,1)

         endif

           ! Calculate first column of subdiagonal block and calculate new
           ! Householder transformation for this column

           hv_new(:) = 0 ! Needed, last rows must be 0 for nr < nb
           tau_new = 0

           if (nr>0) then

             ! complete (old) Householder transformation for first column

             ab(nb+1:nb+nr,ns) = ab(nb+1:nb+nr,ns) - hs(1:nr) ! Note: hv(1) == 1

             ! calculate new Householder transformation ...
             if (nr>1) then
               vnorm2 = sum(dble(ab(nb+2:nb+nr,ns))**2+dimag(ab(nb+2:nb+nr,ns))**2)
               call hh_transform_complex(ab(nb+1,ns),vnorm2,hf,tau_new)
               hv_new(1) = 1.
               hv_new(2:nr) = ab(nb+2:nb+nr,ns)*hf
               ab(nb+2:,ns) = 0
             endif

             ! ... and send it away immediatly if this is the last block

             if (iblk==nblocks) then
#ifdef WITH_OPENMP
               call mpi_wait(ireq_hv,mpi_status,mpierr)
#else
               call mpi_wait(ireq_hv,MPI_STATUS_IGNORE,mpierr)
#endif
               hv_s(1) = tau_new
               hv_s(2:) = hv_new(2:)
               call mpi_isend(hv_s,nb,MPI_COMPLEX16,my_pe+1,2,mpi_comm,ireq_hv,mpierr)
             endif

           endif


          ! Transform diagonal block
          x = dot_product(hv(1:nc),hd(1:nc))*conjg(tau)
          hd(1:nc) = hd(1:nc) - 0.5*x*hv(1:nc)

          if (my_pe>0 .and. iblk==1) then

            ! The first column of the diagonal block has to be send to the previous PE
            ! Calculate first column only ...

            ab(1:nc,ns) = ab(1:nc,ns) - hd(1:nc)*conjg(hv(1)) - hv(1:nc)*conjg(hd(1))

            ! ... send it away ...
#ifdef WITH_OPENMP
            call mpi_wait(ireq_ab,mpi_status,mpierr)
#else
            call mpi_wait(ireq_ab,MPI_STATUS_IGNORE,mpierr)
#endif
            ab_s(1:nb+1) = ab(1:nb+1,ns)
            call mpi_isend(ab_s,nb+1,MPI_COMPLEX16,my_pe-1,1,mpi_comm,ireq_ab,mpierr)

            ! ... and calculate remaining columns with rank-2 update
            if (nc>1) call ZHER2('L',nc-1,(-1.d0,0.d0),hd(2),1,hv(2),1,ab(1,ns+1),2*nb-1)
          else
            ! No need to  send, just a rank-2 update
            call ZHER2('L',nc,(-1.d0,0.d0),hd,1,hv,1,ab(1,ns),2*nb-1)
          endif

          ! Do the remaining double Householder transformation on the subdiagonal block cols 2 ... nb

          if (nr>0) then
            if (nr>1) then
              call ZGEMV('C',nr,nb-1,tau_new,ab(nb,ns+1),2*nb-1,hv_new,1,(0.d0,0.d0),h(2),1)
              x = dot_product(hs(1:nr),hv_new(1:nr))*tau_new
               h(2:nb) = h(2:nb) - x*hv(2:nb)
               ! Unfortunately there is no BLAS routine like DSYR2 for a nonsymmetric rank 2 update
               do i=2,nb
                 ab(2+nb-i:1+nb+nr-i,i+ns-1) = ab(2+nb-i:1+nb+nr-i,i+ns-1) - hv_new(1:nr)*conjg(h(i)) - hs(1:nr)*conjg(hv(i))
               enddo
             else
               ! No double Householder transformation for nr=1, just complete the row
               do i=2,nb
                 ab(2+nb-i,i+ns-1) = ab(2+nb-i,i+ns-1) - hs(1)*conjg(hv(i))
               enddo
             endif
           endif

           ! Use new HH vector for the next block
           hv(:) = hv_new(:)
           tau = tau_new

         enddo
#ifdef WITH_OPENMP
       endif
#endif

#ifdef WITH_OPENMP
       do iblk = 1, nblocks

         if (hh_dst(iblk) >= np_rows) exit
         if (snd_limits(hh_dst(iblk)+1,iblk) == snd_limits(hh_dst(iblk),iblk)) exit

         if (hh_cnt(iblk) == snd_limits(hh_dst(iblk)+1,iblk)-snd_limits(hh_dst(iblk),iblk)) then
           ! Wait for last transfer to finish
           call mpi_wait(ireq_hhs(iblk), mpi_status, mpierr)
           ! Copy vectors into send buffer
           hh_send(:,1:hh_cnt(iblk),iblk) = hh_gath(:,1:hh_cnt(iblk),iblk)
           ! Send to destination
           call mpi_isend(hh_send(1,1,iblk), nb*hh_cnt(iblk), mpi_complex16, &
                         global_id(hh_dst(iblk),mod(iblk+block_limits(my_pe)-1,np_cols)), &
                         10+iblk, mpi_comm, ireq_hhs(iblk), mpierr)
           ! Reset counter and increase destination row
           hh_cnt(iblk) = 0
           hh_dst(iblk) = hh_dst(iblk)+1
         endif
       enddo
#endif
     enddo

     ! Finish the last outstanding requests
#ifdef WITH_OPENMP
     call mpi_wait(ireq_ab,mpi_status,mpierr)
     call mpi_wait(ireq_hv,mpi_status,mpierr)

     allocate(mpi_statuses(MPI_STATUS_SIZE,max(nblocks,num_chunks)))
     call mpi_waitall(nblocks, ireq_hhs, mpi_statuses, mpierr)
     call mpi_waitall(num_chunks, ireq_hhr, mpi_statuses, mpierr)
     deallocate(mpi_statuses)
#else
     call mpi_wait(ireq_ab,MPI_STATUS_IGNORE,mpierr)
     call mpi_wait(ireq_hv,MPI_STATUS_IGNORE,mpierr)

     call mpi_waitall(nblocks, ireq_hhs, MPI_STATUSES_IGNORE, mpierr)
     call mpi_waitall(num_chunks, ireq_hhr, MPI_STATUSES_IGNORE, mpierr)

#endif
     call mpi_barrier(mpi_comm,mpierr)

     deallocate(ab)
     deallocate(ireq_hhr, ireq_hhs)
     deallocate(hh_cnt, hh_dst)
     deallocate(hh_gath, hh_send)
     deallocate(limits, snd_limits)
     deallocate(block_limits)
     deallocate(global_id)
#ifdef HAVE_DETAILED_TIMINGS
     call timer%stop("tridiag_band_complex")
#endif

 end subroutine tridiag_band_complex

!---------------------------------------------------------------------------------------------------


subroutine trans_ev_tridi_to_band_complex(na, nev, nblk, nbw, q, ldq, matrixCols,  &
                                          mpi_comm_rows, mpi_comm_cols, &
                                          wantDebug, success, THIS_COMPLEX_ELPA_KERNEL)

!-------------------------------------------------------------------------------
!  trans_ev_tridi_to_band_complex:
!  Transforms the eigenvectors of a tridiagonal matrix back to the eigenvectors of the band matrix
!
!  Parameters
!
!  na          Order of matrix a, number of rows of matrix q
!
!  nev         Number eigenvectors to compute (= columns of matrix q)
!
!  nblk        blocksize of cyclic distribution, must be the same in both directions!
!
!  nb          semi bandwith
!
!  q           On input: Eigenvectors of tridiagonal matrix
!              On output: Transformed eigenvectors
!              Distribution is like in Scalapack.
!
!  ldq         Leading dimension of q
! matrixCols   local columns of matrix q
!
!  mpi_comm_rows
!  mpi_comm_cols
!              MPI-Communicators for rows/columns/both
!
!-------------------------------------------------------------------------------
#ifdef HAVE_DETAILED_TIMINGS
    use timings
#endif
    implicit none

    integer, intent(in)     :: THIS_COMPLEX_ELPA_KERNEL
    integer, intent(in)     :: na, nev, nblk, nbw, ldq, matrixCols, mpi_comm_rows, mpi_comm_cols
    complex*16              :: q(ldq,matrixCols)

    integer                 :: np_rows, my_prow, np_cols, my_pcol

    integer                 :: i, j, ip, sweep, nbuf, l_nev, a_dim2
    integer                 :: current_n, current_local_n, current_n_start, current_n_end
    integer                 :: next_n, next_local_n, next_n_start, next_n_end
    integer                 :: bottom_msg_length, top_msg_length, next_top_msg_length
    integer                 :: stripe_width, last_stripe_width, stripe_count
#ifdef WITH_OPENMP
    integer                 :: thread_width, csw, b_off, b_len
#endif
    integer                 :: num_result_blocks, num_result_buffers, num_bufs_recvd
    integer                 :: a_off, current_tv_off, max_blk_size
    integer                 :: mpierr, src, src_offset, dst, offset, nfact, num_blk
    logical                 :: flag

#ifdef WITH_OPENMP
    complex*16, allocatable :: a(:,:,:,:), row(:)
#else
    complex*16, allocatable :: a(:,:,:), row(:)
#endif
#ifdef WITH_OPENMP
    complex*16, allocatable :: top_border_send_buffer(:,:), top_border_recv_buffer(:,:)
    complex*16, allocatable :: bottom_border_send_buffer(:,:), bottom_border_recv_buffer(:,:)
#else
    complex*16, allocatable :: top_border_send_buffer(:,:,:), top_border_recv_buffer(:,:,:)
    complex*16, allocatable :: bottom_border_send_buffer(:,:,:), bottom_border_recv_buffer(:,:,:)
#endif
    complex*16, allocatable :: result_buffer(:,:,:)
    complex*16, allocatable :: bcast_buffer(:,:)

    integer                 :: n_off
    integer, allocatable    :: result_send_request(:), result_recv_request(:), limits(:)
    integer, allocatable    :: top_send_request(:), bottom_send_request(:)
    integer, allocatable    :: top_recv_request(:), bottom_recv_request(:)
#ifdef WITH_OPENMP
    integer, allocatable    :: mpi_statuses(:,:)
    integer                 :: mpi_status(MPI_STATUS_SIZE)
#endif

    ! MPI send/recv tags, arbitrary

    integer, parameter      :: bottom_recv_tag = 111
    integer, parameter      :: top_recv_tag    = 222
    integer, parameter      :: result_recv_tag = 333

#ifdef WITH_OPENMP
    integer                 :: max_threads, my_thread
    integer                 :: omp_get_max_threads
#endif

    ! Just for measuring the kernel performance
    real*8                  :: kernel_time
    integer*8               :: kernel_flops

    logical, intent(in)     :: wantDebug
    logical                 :: success

#ifdef HAVE_DETAILED_TIMINGS
    call timer%start("trans_ev_tridi_to_band_complex")
#endif

    kernel_time = 1.d-100
    kernel_flops = 0

#ifdef WITH_OPENMP
    max_threads = 1
    max_threads = omp_get_max_threads()
#endif

    call MPI_Comm_rank(mpi_comm_rows, my_prow, mpierr)
    call MPI_Comm_size(mpi_comm_rows, np_rows, mpierr)
    call MPI_Comm_rank(mpi_comm_cols, my_pcol, mpierr)
    call MPI_Comm_size(mpi_comm_cols, np_cols, mpierr)

    success = .true.

    if (mod(nbw,nblk)/=0) then
      if (my_prow==0 .and. my_pcol==0) then
        if (wantDebug) then
          write(error_unit,*) 'ELPA2_trans_ev_tridi_to_band_complex: ERROR: nbw=',nbw,', nblk=',nblk
          write(error_unit,*) 'ELPA2_trans_ev_tridi_to_band_complex: band backtransform works only for nbw==n*nblk'
        endif

        success = .false.
        return
      endif
    endif

    nfact = nbw / nblk


    ! local number of eigenvectors
    l_nev = local_index(nev, my_pcol, np_cols, nblk, -1)

    if (l_nev==0) then
#ifdef WITH_OPENMP
      thread_width = 0
#endif
      stripe_width = 0
      stripe_count = 0
      last_stripe_width = 0
    else
      ! Suggested stripe width is 48 - should this be reduced for the complex case ???
#ifdef WITH_OPENMP
      thread_width = (l_nev-1)/max_threads + 1 ! number of eigenvectors per OMP thread
#endif

      stripe_width = 48 ! Must be a multiple of 4
#ifdef WITH_OPENMP
      stripe_count = (thread_width-1)/stripe_width + 1
#else
      stripe_count = (l_nev-1)/stripe_width + 1
#endif
      ! Adapt stripe width so that last one doesn't get too small
#ifdef WITH_OPENMP
      stripe_width = (thread_width-1)/stripe_count + 1
#else
      stripe_width = (l_nev-1)/stripe_count + 1
#endif
      stripe_width = ((stripe_width+3)/4)*4 ! Must be a multiple of 4 !!!
#ifndef WITH_OPENMP
      last_stripe_width = l_nev - (stripe_count-1)*stripe_width
#endif
    endif

    ! Determine the matrix distribution at the beginning

    allocate(limits(0:np_rows))

    call determine_workload(na, nbw, np_rows, limits)

    max_blk_size = maxval(limits(1:np_rows) - limits(0:np_rows-1))

    a_dim2 = max_blk_size + nbw

!DEC$ ATTRIBUTES ALIGN: 64:: a
#ifdef WITH_OPENMP
    allocate(a(stripe_width,a_dim2,stripe_count,max_threads))
    ! a(:,:,:,:) should be set to 0 in a parallel region, not here!
#else
    allocate(a(stripe_width,a_dim2,stripe_count))
    a(:,:,:) = 0
#endif

    allocate(row(l_nev))
    row(:) = 0

    ! Copy q from a block cyclic distribution into a distribution with contiguous rows,
    ! and transpose the matrix using stripes of given stripe_width for cache blocking.

    ! The peculiar way it is done below is due to the fact that the last row should be
    ! ready first since it is the first one to start below

#ifdef WITH_OPENMP
    ! Please note about the OMP usage below:
    ! This is not for speed, but because we want the matrix a in the memory and
    ! in the cache of the correct thread (if possible)
#ifdef HAVE_DETAILED_TIMINGS
    call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread), schedule(static, 1)
    do my_thread = 1, max_threads
      a(:,:,:,my_thread) = 0 ! if possible, do first touch allocation!
    enddo
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
    call timer%stop("OpenMP parallel")
#endif

#endif
    do ip = np_rows-1, 0, -1
      if (my_prow == ip) then
        ! Receive my rows which have not yet been received
        src_offset = local_index(limits(ip), my_prow, np_rows, nblk, -1)
        do i=limits(ip)+1,limits(ip+1)
          src = mod((i-1)/nblk, np_rows)
          if (src < my_prow) then
#ifdef WITH_OPENMP
            call MPI_Recv(row, l_nev, MPI_COMPLEX16, src, 0, mpi_comm_rows, mpi_status, mpierr)

#else
            call MPI_Recv(row, l_nev, MPI_COMPLEX16, src, 0, mpi_comm_rows, MPI_STATUS_IGNORE, mpierr)
#endif

#ifdef WITH_OPENMP
#ifdef HAVE_DETAILED_TIMINGS
            call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread), schedule(static, 1)
            do my_thread = 1, max_threads
              call unpack_row(row,i-limits(ip),my_thread)
            enddo
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
            call timer%stop("OpenMP parallel")
#endif

#else
            call unpack_row(row,i-limits(ip))
#endif
          elseif (src==my_prow) then
            src_offset = src_offset+1
            row(:) = q(src_offset, 1:l_nev)
#ifdef WITH_OPENMP
#ifdef HAVE_DETAILED_TIMINGS
            call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread), schedule(static, 1)
            do my_thread = 1, max_threads
              call unpack_row(row,i-limits(ip),my_thread)
            enddo
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
            call timer%stop("OpenMP parallel")
#endif

#else
            call unpack_row(row,i-limits(ip))
#endif
          endif
        enddo
        ! Send all rows which have not yet been send
        src_offset = 0
        do dst = 0, ip-1
          do i=limits(dst)+1,limits(dst+1)
            if(mod((i-1)/nblk, np_rows) == my_prow) then
                src_offset = src_offset+1
                row(:) = q(src_offset, 1:l_nev)
                call MPI_Send(row, l_nev, MPI_COMPLEX16, dst, 0, mpi_comm_rows, mpierr)
            endif
          enddo
        enddo
      else if(my_prow < ip) then
        ! Send all rows going to PE ip
        src_offset = local_index(limits(ip), my_prow, np_rows, nblk, -1)
        do i=limits(ip)+1,limits(ip+1)
          src = mod((i-1)/nblk, np_rows)
          if (src == my_prow) then
            src_offset = src_offset+1
            row(:) = q(src_offset, 1:l_nev)
            call MPI_Send(row, l_nev, MPI_COMPLEX16, ip, 0, mpi_comm_rows, mpierr)
          endif
        enddo
        ! Receive all rows from PE ip
        do i=limits(my_prow)+1,limits(my_prow+1)
          src = mod((i-1)/nblk, np_rows)
          if (src == ip) then
#ifdef WITH_OPENMP
            call MPI_Recv(row, l_nev, MPI_COMPLEX16, src, 0, mpi_comm_rows, mpi_status, mpierr)
#else
            call MPI_Recv(row, l_nev, MPI_COMPLEX16, src, 0, mpi_comm_rows, MPI_STATUS_IGNORE, mpierr)
#endif

#ifdef WITH_OPENMP
#ifdef HAVE_DETAILED_TIMINGS
            call timer%start("OpenMP parallel")
#endif
!$omp parallel do private(my_thread), schedule(static, 1)
            do my_thread = 1, max_threads
              call unpack_row(row,i-limits(my_prow),my_thread)
            enddo
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
            call timer%stop("OpenMP parallel")
#endif

#else
            call unpack_row(row,i-limits(my_prow))
#endif
          endif
        enddo
      endif
    enddo


    ! Set up result buffer queue

    num_result_blocks = ((na-1)/nblk + np_rows - my_prow) / np_rows

    num_result_buffers = 4*nfact
    allocate(result_buffer(l_nev,nblk,num_result_buffers))

    allocate(result_send_request(num_result_buffers))
    allocate(result_recv_request(num_result_buffers))
    result_send_request(:) = MPI_REQUEST_NULL
    result_recv_request(:) = MPI_REQUEST_NULL

    ! Queue up buffers

    if (my_prow > 0 .and. l_nev>0) then ! note: row 0 always sends
      do j = 1, min(num_result_buffers, num_result_blocks)
        call MPI_Irecv(result_buffer(1,1,j), l_nev*nblk, MPI_COMPLEX16, 0, result_recv_tag, &
                           mpi_comm_rows, result_recv_request(j), mpierr)
      enddo
    endif

    num_bufs_recvd = 0 ! No buffers received yet

    ! Initialize top/bottom requests

    allocate(top_send_request(stripe_count))
    allocate(top_recv_request(stripe_count))
    allocate(bottom_send_request(stripe_count))
    allocate(bottom_recv_request(stripe_count))

    top_send_request(:) = MPI_REQUEST_NULL
    top_recv_request(:) = MPI_REQUEST_NULL
    bottom_send_request(:) = MPI_REQUEST_NULL
    bottom_recv_request(:) = MPI_REQUEST_NULL

#ifdef WITH_OPENMP
    allocate(top_border_send_buffer(stripe_width*nbw*max_threads, stripe_count))
    allocate(top_border_recv_buffer(stripe_width*nbw*max_threads, stripe_count))
    allocate(bottom_border_send_buffer(stripe_width*nbw*max_threads, stripe_count))
    allocate(bottom_border_recv_buffer(stripe_width*nbw*max_threads, stripe_count))

    top_border_send_buffer(:,:) = 0
    top_border_recv_buffer(:,:) = 0
    bottom_border_send_buffer(:,:) = 0
    bottom_border_recv_buffer(:,:) = 0
#else
    allocate(top_border_send_buffer(stripe_width, nbw, stripe_count))
    allocate(top_border_recv_buffer(stripe_width, nbw, stripe_count))
    allocate(bottom_border_send_buffer(stripe_width, nbw, stripe_count))
    allocate(bottom_border_recv_buffer(stripe_width, nbw, stripe_count))

    top_border_send_buffer(:,:,:) = 0
    top_border_recv_buffer(:,:,:) = 0
    bottom_border_send_buffer(:,:,:) = 0
    bottom_border_recv_buffer(:,:,:) = 0
#endif

    ! Initialize broadcast buffer

    allocate(bcast_buffer(nbw, max_blk_size))
    bcast_buffer = 0

    current_tv_off = 0 ! Offset of next row to be broadcast


    ! ------------------- start of work loop -------------------

    a_off = 0 ! offset in A (to avoid unnecessary shifts)

    top_msg_length = 0
    bottom_msg_length = 0

    do sweep = 0, (na-1)/nbw

      current_n = na - sweep*nbw
      call determine_workload(current_n, nbw, np_rows, limits)
      current_n_start = limits(my_prow)
      current_n_end   = limits(my_prow+1)
      current_local_n = current_n_end - current_n_start

      next_n = max(current_n - nbw, 0)
      call determine_workload(next_n, nbw, np_rows, limits)
      next_n_start = limits(my_prow)
      next_n_end   = limits(my_prow+1)
      next_local_n = next_n_end - next_n_start

      if (next_n_end < next_n) then
        bottom_msg_length = current_n_end - next_n_end
      else
        bottom_msg_length = 0
      endif

      if (next_local_n > 0) then
        next_top_msg_length = current_n_start - next_n_start
      else
        next_top_msg_length = 0
      endif

      if (sweep==0 .and. current_n_end < current_n .and. l_nev > 0) then
        do i = 1, stripe_count
#ifdef WITH_OPENMP
          csw = min(stripe_width, thread_width-(i-1)*stripe_width) ! "current_stripe_width"
          b_len = csw*nbw*max_threads
          call MPI_Irecv(bottom_border_recv_buffer(1,i), b_len, MPI_COMPLEX16, my_prow+1, bottom_recv_tag, &
                     mpi_comm_rows, bottom_recv_request(i), mpierr)
#else
          call MPI_Irecv(bottom_border_recv_buffer(1,1,i), nbw*stripe_width, MPI_COMPLEX16, my_prow+1, bottom_recv_tag, &
                         mpi_comm_rows, bottom_recv_request(i), mpierr)
#endif
        enddo
      endif

      if (current_local_n > 1) then
        if (my_pcol == mod(sweep,np_cols)) then
          bcast_buffer(:,1:current_local_n) = hh_trans_complex(:,current_tv_off+1:current_tv_off+current_local_n)
          current_tv_off = current_tv_off + current_local_n
        endif
        call mpi_bcast(bcast_buffer, nbw*current_local_n, MPI_COMPLEX16, mod(sweep,np_cols), mpi_comm_cols, mpierr)
       else
         ! for current_local_n == 1 the one and only HH vector is 0 and not stored in hh_trans_complex
         bcast_buffer(:,1) = 0
       endif

       if (l_nev == 0) cycle

       if (current_local_n > 0) then

         do i = 1, stripe_count

#ifdef WITH_OPENMP
           ! Get real stripe width for strip i;
           ! The last OpenMP tasks may have an even smaller stripe with,
           ! but we don't care about this, i.e. we send/recv a bit too much in this case.
           ! csw: current_stripe_width

           csw = min(stripe_width, thread_width-(i-1)*stripe_width)
#endif

           !wait_b
           if (current_n_end < current_n) then
#ifdef WITH_OPENMP
             call MPI_Wait(bottom_recv_request(i), mpi_status, mpierr)
#else
             call MPI_Wait(bottom_recv_request(i), MPI_STATUS_IGNORE, mpierr)
#endif
#ifdef WITH_OPENMP
#ifdef HAVE_DETAILED_TIMINGS
             call timer%start("OpenMP parallel")
#endif
!$omp parallel do private(my_thread, n_off, b_len, b_off), schedule(static, 1)
             do my_thread = 1, max_threads
               n_off = current_local_n+a_off
               b_len = csw*nbw
               b_off = (my_thread-1)*b_len
               a(1:csw,n_off+1:n_off+nbw,i,my_thread) = &
                  reshape(bottom_border_recv_buffer(b_off+1:b_off+b_len,i), (/ csw, nbw /))
             enddo
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
             call timer%stop("OpenMP parallel")
#endif

#else
             n_off = current_local_n+a_off
             a(:,n_off+1:n_off+nbw,i) = bottom_border_recv_buffer(:,1:nbw,i)
#endif
             if (next_n_end < next_n) then
#ifdef WITH_OPENMP
               call MPI_Irecv(bottom_border_recv_buffer(1,i), csw*nbw*max_threads, &
                                   MPI_COMPLEX16, my_prow+1, bottom_recv_tag, &
                                   mpi_comm_rows, bottom_recv_request(i), mpierr)
#else
               call MPI_Irecv(bottom_border_recv_buffer(1,1,i), nbw*stripe_width, MPI_COMPLEX16, my_prow+1, bottom_recv_tag, &

                                   mpi_comm_rows, bottom_recv_request(i), mpierr)
#endif
             endif
           endif

           if (current_local_n <= bottom_msg_length + top_msg_length) then

             !wait_t
             if (top_msg_length>0) then
#ifdef WITH_OPENMP
               call MPI_Wait(top_recv_request(i), mpi_status, mpierr)
#else
               call MPI_Wait(top_recv_request(i), MPI_STATUS_IGNORE, mpierr)
#endif
#ifndef WITH_OPENMP
               a(:,a_off+1:a_off+top_msg_length,i) = top_border_recv_buffer(:,1:top_msg_length,i)
#endif
             endif

             !compute
#ifdef WITH_OPENMP
#ifdef HAVE_DETAILED_TIMINGS
             call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread, n_off, b_len, b_off), schedule(static, 1)
             do my_thread = 1, max_threads
               if (top_msg_length>0) then
                 b_len = csw*top_msg_length
                 b_off = (my_thread-1)*b_len
                 a(1:csw,a_off+1:a_off+top_msg_length,i,my_thread) = &
                          reshape(top_border_recv_buffer(b_off+1:b_off+b_len,i), (/ csw, top_msg_length /))
               endif
               call compute_hh_trafo_complex(0, current_local_n, i, my_thread, &
                                      THIS_COMPLEX_ELPA_KERNEL)
             enddo
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
             call timer%stop("OpenMP parallel")
#endif

#else
             call compute_hh_trafo_complex(0, current_local_n, i, &
                                      THIS_COMPLEX_ELPA_KERNEL)
#endif
             !send_b
#ifdef WITH_OPENMP
             call MPI_Wait(bottom_send_request(i), mpi_status, mpierr)
#else
             call MPI_Wait(bottom_send_request(i), MPI_STATUS_IGNORE, mpierr)
#endif
             if (bottom_msg_length>0) then
               n_off = current_local_n+nbw-bottom_msg_length+a_off
#ifdef WITH_OPENMP
               b_len = csw*bottom_msg_length*max_threads
               bottom_border_send_buffer(1:b_len,i) = &
                        reshape(a(1:csw,n_off+1:n_off+bottom_msg_length,i,:), (/ b_len /))
               call MPI_Isend(bottom_border_send_buffer(1,i), b_len, MPI_COMPLEX16, my_prow+1, &
                                   top_recv_tag, mpi_comm_rows, bottom_send_request(i), mpierr)
#else
               bottom_border_send_buffer(:,1:bottom_msg_length,i) = a(:,n_off+1:n_off+bottom_msg_length,i)
               call MPI_Isend(bottom_border_send_buffer(1,1,i), bottom_msg_length*stripe_width, MPI_COMPLEX16, my_prow+1, &
                              top_recv_tag, mpi_comm_rows, bottom_send_request(i), mpierr)
#endif
             endif

           else

             !compute
#ifdef WITH_OPENMP
#ifdef HAVE_DETAILED_TIMINGS
             call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread, b_len, b_off), schedule(static, 1)
             do my_thread = 1, max_threads
               call compute_hh_trafo_complex(current_local_n - bottom_msg_length, bottom_msg_length, i, my_thread, &
                                      THIS_COMPLEX_ELPA_KERNEL)
             enddo
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
             call timer%stop("OpenMP parallel")
#endif

#else
             call compute_hh_trafo_complex(current_local_n - bottom_msg_length, bottom_msg_length, i, &
                                      THIS_COMPLEX_ELPA_KERNEL)


#endif
             !send_b
#ifdef WITH_OPENMP
             call MPI_Wait(bottom_send_request(i), mpi_status, mpierr)
#else

             call MPI_Wait(bottom_send_request(i), MPI_STATUS_IGNORE, mpierr)
#endif
             if (bottom_msg_length > 0) then
               n_off = current_local_n+nbw-bottom_msg_length+a_off
#ifdef WITH_OPENMP
               b_len = csw*bottom_msg_length*max_threads
               bottom_border_send_buffer(1:b_len,i) = &
                      reshape(a(1:csw,n_off+1:n_off+bottom_msg_length,i,:), (/ b_len /))
               call MPI_Isend(bottom_border_send_buffer(1,i), b_len, MPI_COMPLEX16, my_prow+1, &
                                   top_recv_tag, mpi_comm_rows, bottom_send_request(i), mpierr)
#else
               bottom_border_send_buffer(:,1:bottom_msg_length,i) = a(:,n_off+1:n_off+bottom_msg_length,i)
               call MPI_Isend(bottom_border_send_buffer(1,1,i), bottom_msg_length*stripe_width, MPI_COMPLEX16, my_prow+1, &
                              top_recv_tag, mpi_comm_rows, bottom_send_request(i), mpierr)
#endif
             endif

             !compute
#ifdef WITH_OPENMP
#ifdef HAVE_DETAILED_TIMINGS
             call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread), schedule(static, 1)
             do my_thread = 1, max_threads
               call compute_hh_trafo_complex(top_msg_length, current_local_n-top_msg_length-bottom_msg_length, i, my_thread, &
                                      THIS_COMPLEX_ELPA_KERNEL)
             enddo
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
             call timer%stop("OpenMP parallel")
#endif

#else
             call compute_hh_trafo_complex(top_msg_length, current_local_n-top_msg_length-bottom_msg_length, i, &
                                      THIS_COMPLEX_ELPA_KERNEL)

#endif
             !wait_t
             if (top_msg_length>0) then
#ifdef WITH_OPENMP
               call MPI_Wait(top_recv_request(i), mpi_status, mpierr)
#else
               call MPI_Wait(top_recv_request(i), MPI_STATUS_IGNORE, mpierr)
#endif
#ifndef WITH_OPENMP
               a(:,a_off+1:a_off+top_msg_length,i) = top_border_recv_buffer(:,1:top_msg_length,i)

#endif
             endif

             !compute
#ifdef WITH_OPENMP
#ifdef HAVE_DETAILED_TIMINGS
             call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread, b_len, b_off), schedule(static, 1)
             do my_thread = 1, max_threads
               if (top_msg_length>0) then
                 b_len = csw*top_msg_length
                 b_off = (my_thread-1)*b_len
                 a(1:csw,a_off+1:a_off+top_msg_length,i,my_thread) = &
                          reshape(top_border_recv_buffer(b_off+1:b_off+b_len,i), (/ csw, top_msg_length /))
               endif
               call compute_hh_trafo_complex(0, top_msg_length, i, my_thread, &
                                      THIS_COMPLEX_ELPA_KERNEL)
             enddo
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
             call timer%stop("OpenMP parallel")
#endif

#else
             call compute_hh_trafo_complex(0, top_msg_length, i, &
                                      THIS_COMPLEX_ELPA_KERNEL)
#endif
           endif

           if (next_top_msg_length > 0) then
             !request top_border data
#ifdef WITH_OPENMP
             b_len = csw*next_top_msg_length*max_threads
             call MPI_Irecv(top_border_recv_buffer(1,i), b_len, MPI_COMPLEX16, my_prow-1, &
                               top_recv_tag, mpi_comm_rows, top_recv_request(i), mpierr)
#else
             call MPI_Irecv(top_border_recv_buffer(1,1,i), next_top_msg_length*stripe_width, MPI_COMPLEX16, my_prow-1, &
                               top_recv_tag, mpi_comm_rows, top_recv_request(i), mpierr)
#endif
           endif

           !send_t
           if (my_prow > 0) then
#ifdef WITH_OPENMP
             call MPI_Wait(top_send_request(i), mpi_status, mpierr)
#else
             call MPI_Wait(top_send_request(i), MPI_STATUS_IGNORE, mpierr)
#endif

#ifdef WITH_OPENMP
             b_len = csw*nbw*max_threads
             top_border_send_buffer(1:b_len,i) = reshape(a(1:csw,a_off+1:a_off+nbw,i,:), (/ b_len /))
             call MPI_Isend(top_border_send_buffer(1,i), b_len, MPI_COMPLEX16, &
                               my_prow-1, bottom_recv_tag, &
                               mpi_comm_rows, top_send_request(i), mpierr)
#else
             top_border_send_buffer(:,1:nbw,i) = a(:,a_off+1:a_off+nbw,i)
             call MPI_Isend(top_border_send_buffer(1,1,i), nbw*stripe_width, MPI_COMPLEX16, my_prow-1, bottom_recv_tag, &
                               mpi_comm_rows, top_send_request(i), mpierr)

#endif
           endif

           ! Care that there are not too many outstanding top_recv_request's
           if (stripe_count > 1) then
             if (i>1) then
#ifdef WITH_OPENMP
               call MPI_Wait(top_recv_request(i-1), mpi_status, mpierr)
#else
               call MPI_Wait(top_recv_request(i-1), MPI_STATUS_IGNORE, mpierr)
#endif
             else
#ifdef WITH_OPENMP
               call MPI_Wait(top_recv_request(stripe_count), mpi_status, mpierr)
#else
               call MPI_Wait(top_recv_request(stripe_count), MPI_STATUS_IGNORE, mpierr)
#endif
             endif
           endif

         enddo

         top_msg_length = next_top_msg_length

       else
         ! wait for last top_send_request
         do i = 1, stripe_count
#ifdef WITH_OPENMP
           call MPI_Wait(top_send_request(i), mpi_status, mpierr)
#else
           call MPI_Wait(top_send_request(i), MPI_STATUS_IGNORE, mpierr)
#endif
         enddo
       endif

       ! Care about the result

       if (my_prow == 0) then

         ! topmost process sends nbw rows to destination processes

         do j=0,nfact-1

           num_blk = sweep*nfact+j ! global number of destination block, 0 based
           if (num_blk*nblk >= na) exit

           nbuf = mod(num_blk, num_result_buffers) + 1 ! buffer number to get this block

#ifdef WITH_OPENMP
           call MPI_Wait(result_send_request(nbuf), mpi_status, mpierr)
#else
           call MPI_Wait(result_send_request(nbuf), MPI_STATUS_IGNORE, mpierr)
#endif

           dst = mod(num_blk, np_rows)

           if (dst == 0) then
             do i = 1, min(na - num_blk*nblk, nblk)
               call pack_row(row, j*nblk+i+a_off)
               q((num_blk/np_rows)*nblk+i,1:l_nev) = row(:)
             enddo
           else
             do i = 1, nblk
               call pack_row(result_buffer(:,i,nbuf),j*nblk+i+a_off)
             enddo
             call MPI_Isend(result_buffer(1,1,nbuf), l_nev*nblk, MPI_COMPLEX16, dst, &
                                   result_recv_tag, mpi_comm_rows, result_send_request(nbuf), mpierr)
           endif
         enddo

       else

         ! receive and store final result

         do j = num_bufs_recvd, num_result_blocks-1

           nbuf = mod(j, num_result_buffers) + 1 ! buffer number to get this block

           ! If there is still work to do, just test for the next result request
           ! and leave the loop if it is not ready, otherwise wait for all
           ! outstanding requests

           if (next_local_n > 0) then
#ifdef WITH_OPENMP
             call MPI_Test(result_recv_request(nbuf), flag, mpi_status, mpierr)

#else
             call MPI_Test(result_recv_request(nbuf), flag, MPI_STATUS_IGNORE, mpierr)
#endif
             if (.not.flag) exit
           else
#ifdef WITH_OPENMP
             call MPI_Wait(result_recv_request(nbuf), mpi_status, mpierr)

#else

             call MPI_Wait(result_recv_request(nbuf), MPI_STATUS_IGNORE, mpierr)
#endif
           endif

             ! Fill result buffer into q
             num_blk = j*np_rows + my_prow ! global number of current block, 0 based
             do i = 1, min(na - num_blk*nblk, nblk)
               q(j*nblk+i, 1:l_nev) = result_buffer(1:l_nev, i, nbuf)
             enddo

             ! Queue result buffer again if there are outstanding blocks left
             if (j+num_result_buffers < num_result_blocks) &
                    call MPI_Irecv(result_buffer(1,1,nbuf), l_nev*nblk, MPI_COMPLEX16, 0, result_recv_tag, &
                                   mpi_comm_rows, result_recv_request(nbuf), mpierr)

           enddo
           num_bufs_recvd = j

         endif

         ! Shift the remaining rows to the front of A (if necessary)

         offset = nbw - top_msg_length

         if (offset<0) then
           if (wantDebug) then
             write(error_unit,*) 'ELPA2_trans_ev_tridi_to_band_complex: internal error, offset for shifting = ',offset
           endif
           success = .false.
           return
         endif

         a_off = a_off + offset
         if (a_off + next_local_n + nbw > a_dim2) then
#ifdef WITH_OPENMP
#ifdef HAVE_DETAILED_TIMINGS
           call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread, i, j), schedule(static, 1)
           do my_thread = 1, max_threads
             do i = 1, stripe_count
               do j = top_msg_length+1, top_msg_length+next_local_n
                 A(:,j,i,my_thread) = A(:,j+a_off,i,my_thread)
               enddo
#else
           do i = 1, stripe_count
             do j = top_msg_length+1, top_msg_length+next_local_n
               A(:,j,i) = A(:,j+a_off,i)
#endif
             enddo
           enddo
#ifdef WITH_OPENMP
#ifdef HAVE_DETAILED_TIMINGS
           call timer%stop("OpenMP parallel")
#endif
#endif

           a_off = 0
        endif
      enddo

     ! Just for safety:
     if (ANY(top_send_request    /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR top_send_request ***',my_prow,my_pcol
     if (ANY(bottom_send_request /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR bottom_send_request ***',my_prow,my_pcol
     if (ANY(top_recv_request    /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR top_recv_request ***',my_prow,my_pcol
     if (ANY(bottom_recv_request /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR bottom_recv_request ***',my_prow,my_pcol

     if (my_prow == 0) then
#ifdef WITH_OPENMP
       allocate(mpi_statuses(MPI_STATUS_SIZE,num_result_buffers))
       call MPI_Waitall(num_result_buffers, result_send_request, mpi_statuses, mpierr)
       deallocate(mpi_statuses)
#else
       call MPI_Waitall(num_result_buffers, result_send_request, MPI_STATUSES_IGNORE, mpierr)
#endif
     endif

     if (ANY(result_send_request /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR result_send_request ***',my_prow,my_pcol
     if (ANY(result_recv_request /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR result_recv_request ***',my_prow,my_pcol

     if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
       write(error_unit,'(" Kernel time:",f10.3," MFlops: ",f10.3)') kernel_time, kernel_flops/kernel_time*1.d-6

     ! deallocate all working space

     deallocate(a)
     deallocate(row)
     deallocate(limits)
     deallocate(result_send_request)
     deallocate(result_recv_request)
     deallocate(top_border_send_buffer)
     deallocate(top_border_recv_buffer)
     deallocate(bottom_border_send_buffer)
     deallocate(bottom_border_recv_buffer)
     deallocate(result_buffer)
     deallocate(bcast_buffer)
     deallocate(top_send_request)
     deallocate(top_recv_request)
     deallocate(bottom_send_request)
     deallocate(bottom_recv_request)
#ifdef HAVE_DETAILED_TIMINGS
     call timer%stop("trans_ev_tridi_to_band_complex")
#endif
     return
contains

#ifdef WITH_OPENMP
  subroutine pack_row(row, n)
#ifdef HAVE_DETAILED_TIMINGS
    use timings
#endif
    implicit none
    complex*16 :: row(:)
    integer    :: n, i, noff, nl, nt

#ifdef HAVE_DETAILED_TIMINGS
    call timer%start("pack_row")
#endif
    do nt = 1, max_threads
      do i = 1, stripe_count
        noff = (nt-1)*thread_width + (i-1)*stripe_width
        nl   = min(stripe_width, nt*thread_width-noff, l_nev-noff)
        if (nl<=0) exit
        row(noff+1:noff+nl) = a(1:nl,n,i,nt)
      enddo
    enddo

#ifdef HAVE_DETAILED_TIMINGS
    call timer%stop("pack_row")
#endif

  end subroutine pack_row
#else
  subroutine pack_row(row, n)

#ifdef HAVE_DETAILED_TIMINGS
    use timings
#endif
    implicit none
    complex*16 :: row(:)
    integer    :: n, i, noff, nl

#ifdef HAVE_DETAILED_TIMINGS
    call timer%start("unpack_row")
#endif

    do i=1,stripe_count
      nl = merge(stripe_width, last_stripe_width, i<stripe_count)
      noff = (i-1)*stripe_width
      row(noff+1:noff+nl) = a(1:nl,n,i)
    enddo

#ifdef HAVE_DETAILED_TIMINGS
    call timer%stop("unpack_row")
#endif


  end subroutine pack_row
#endif

#ifdef WITH_OPENMP
  subroutine unpack_row(row, n, my_thread)

#ifdef HAVE_DETAILED_TIMINGS
    use timings
#endif

    implicit none

    ! Private variables in OMP regions (my_thread) should better be in the argument list!
    integer, intent(in)     :: n, my_thread
    complex*16, intent(in)  :: row(:)
    integer                 :: i, noff, nl

#ifdef HAVE_DETAILED_TIMINGS
    call timer%start("unpack_row")
#endif

    do i=1,stripe_count
      noff = (my_thread-1)*thread_width + (i-1)*stripe_width
      nl   = min(stripe_width, my_thread*thread_width-noff, l_nev-noff)
      if (nl<=0) exit
      a(1:nl,n,i,my_thread) = row(noff+1:noff+nl)
    enddo

#ifdef HAVE_DETAILED_TIMINGS
    call timer%stop("unpack_row")
#endif
  end subroutine unpack_row
#else
  subroutine unpack_row(row, n)
#ifdef HAVE_DETAILED_TIMINGS
    use timings
#endif

    implicit none

    complex*16 :: row(:)
    integer    :: n, i, noff, nl

#ifdef HAVE_DETAILED_TIMINGS
    call timer%start("unpack_row")
#endif


    do i=1,stripe_count
      nl = merge(stripe_width, last_stripe_width, i<stripe_count)
      noff = (i-1)*stripe_width
      a(1:nl,n,i) = row(noff+1:noff+nl)
    enddo

#ifdef HAVE_DETAILED_TIMINGS
    call timer%stop("unpack_row")
#endif

  end  subroutine unpack_row
#endif

#ifdef WITH_OPENMP
  subroutine compute_hh_trafo_complex(off, ncols, istripe, my_thread, THIS_COMPLEX_ELPA_KERNEL)
#else
  subroutine compute_hh_trafo_complex(off, ncols, istripe, THIS_COMPLEX_ELPA_KERNEL)
#endif

#if defined(WITH_COMPLEX_GENERIC_SIMPLE_KERNEL)
      use complex_generic_simple_kernel, only : single_hh_trafo_complex_generic_simple
#endif
#if defined(WITH_COMPLEX_GENERIC_KERNEL)
      use complex_generic_kernel, only : single_hh_trafo_complex_generic
#endif
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#endif
      implicit none
      integer, intent(in) :: THIS_COMPLEX_ELPA_KERNEL

        ! Private variables in OMP regions (my_thread) should better be in the argument list!

        integer           :: off, ncols, istripe, j, nl, jj
#ifdef WITH_OPENMP
        integer           :: my_thread, noff
#endif
        real*8            :: ttt

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!        Currently (on Sandy Bridge), single is faster than double
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        complex*16        :: w(nbw,2)

#ifdef HAVE_DETAILED_TIMINGS
        call timer%start("compute_hh_trafo_complex")
#endif

#ifdef WITH_OPENMP
        if (istripe<stripe_count) then
          nl = stripe_width
        else
          noff = (my_thread-1)*thread_width + (istripe-1)*stripe_width
          nl = min(my_thread*thread_width-noff, l_nev-noff)
          if(nl<=0) return
        endif
#else
        nl = merge(stripe_width, last_stripe_width, istripe<stripe_count)
#endif


#if defined(WITH_COMPLEX_AVX_BLOCK2_KERNEL)
#if defined(WITH_NO_SPECIFIC_COMPLEX_KERNEL)
        if (THIS_COMPLEX_ELPA_KERNEL .eq. COMPLEX_ELPA_KERNEL_AVX_BLOCK2) then
#endif  /* WITH_NO_SPECIFIC_COMPLEX_KERNEL */
          ttt = mpi_wtime()
          do j = ncols, 2, -2
            w(:,1) = bcast_buffer(1:nbw,j+off)
            w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP
            call double_hh_trafo_complex_sse_avx_2hv(a(1,j+off+a_off-1,istripe,my_thread), &
                                                       w, nbw, nl, stripe_width, nbw)
#else
            call double_hh_trafo_complex_sse_avx_2hv(a(1,j+off+a_off-1,istripe), &
                                                       w, nbw, nl, stripe_width, nbw)
#endif
          enddo
#ifdef WITH_OPENMP
          if (j==1) call single_hh_trafo_complex_sse_avx_1hv(a(1,1+off+a_off,istripe,my_thread), &
                                                             bcast_buffer(1,off+1), nbw, nl, stripe_width)
#else
          if (j==1) call single_hh_trafo_complex_sse_avx_1hv(a(1,1+off+a_off,istripe), &
                                                             bcast_buffer(1,off+1), nbw, nl, stripe_width)
#endif
#if defined(WITH_NO_SPECIFIC_COMPLEX_KERNEL)
        endif
#endif  /* WITH_NO_SPECIFIC_COMPLEX_KERNEL */
#endif /* WITH_COMPLEX_AVX_BLOCK2_KERNEL */


#if defined(WITH_COMPLEX_GENERIC_SIMPLE_KERNEL)
#if defined(WITH_NO_SPECIFIC_COMPLEX_KERNEL)
        if (THIS_COMPLEX_ELPA_KERNEL .eq. COMPLEX_ELPA_KERNEL_GENERIC_SIMPLE) then
#endif /* WITH_NO_SPECIFIC_COMPLEX_KERNEL */
          ttt = mpi_wtime()
          do j = ncols, 1, -1
#ifdef WITH_OPENMP
            call single_hh_trafo_complex_generic_simple(a(1,j+off+a_off,istripe,my_thread), &
                                                          bcast_buffer(1,j+off),nbw,nl,stripe_width)
#else
            call single_hh_trafo_complex_generic_simple(a(1,j+off+a_off,istripe), &
                                                          bcast_buffer(1,j+off),nbw,nl,stripe_width)
#endif
          enddo
#if defined(WITH_NO_SPECIFIC_COMPLEX_KERNEL)
        endif
#endif /* WITH_NO_SPECIFIC_COMPLEX_KERNEL */
#endif /* WITH_COMPLEX_GENERIC_SIMPLE_KERNEL */


#if defined(WITH_COMPLEX_GENERIC_KERNEL)
#if defined(WITH_NO_SPECIFIC_COMPLEX_KERNEL)
        if (THIS_COMPLEX_ELPA_KERNEL .eq. COMPLEX_ELPA_KERNEL_GENERIC .or. &
            THIS_COMPLEX_ELPA_KERNEL .eq. COMPLEX_ELPA_KERNEL_BGP .or. &
            THIS_COMPLEX_ELPA_KERNEL .eq. COMPLEX_ELPA_KERNEL_BGQ ) then
#endif /* WITH_NO_SPECIFIC_COMPLEX_KERNEL */
          ttt = mpi_wtime()
          do j = ncols, 1, -1
#ifdef WITH_OPENMP
            call single_hh_trafo_complex_generic(a(1,j+off+a_off,istripe,my_thread), &
                                                   bcast_buffer(1,j+off),nbw,nl,stripe_width)
#else
            call single_hh_trafo_complex_generic(a(1,j+off+a_off,istripe), &
                                                   bcast_buffer(1,j+off),nbw,nl,stripe_width)
#endif
          enddo
#if defined(WITH_NO_SPECIFIC_COMPLEX_KERNEL)
        endif
#endif /* WITH_NO_SPECIFIC_COMPLEX_KERNEL */
#endif /* WITH_COMPLEX_GENERIC_KERNEL */

#if defined(WITH_COMPLEX_SSE_KERNEL)
#if defined(WITH_NO_SPECIFIC_COMPLEX_KERNEL)
        if (THIS_COMPLEX_ELPA_KERNEL .eq. COMPLEX_ELPA_KERNEL_SSE) then
#endif /* WITH_NO_SPECIFIC_COMPLEX_KERNEL */
          ttt = mpi_wtime()
          do j = ncols, 1, -1
#ifdef WITH_OPENMP
            call single_hh_trafo_complex(a(1,j+off+a_off,istripe,my_thread), &
                                           bcast_buffer(1,j+off),nbw,nl,stripe_width)
#else
            call single_hh_trafo_complex(a(1,j+off+a_off,istripe), &
                                           bcast_buffer(1,j+off),nbw,nl,stripe_width)
#endif
          enddo
#if defined(WITH_NO_SPECIFIC_COMPLEX_KERNEL)
        endif
#endif /* WITH_NO_SPECIFIC_COMPLEX_KERNEL */
#endif /* WITH_COMPLEX_SSE_KERNEL */


!#if defined(WITH_AVX_SANDYBRIDGE)
!              call single_hh_trafo_complex_sse_avx_1hv(a(1,j+off+a_off,istripe),bcast_buffer(1,j+off),nbw,nl,stripe_width)
!#endif

!#if defined(WITH_AMD_BULLDOZER)
!              call single_hh_trafo_complex_sse_avx_1hv(a(1,j+off+a_off,istripe),bcast_buffer(1,j+off),nbw,nl,stripe_width)
!#endif

#if defined(WITH_COMPLEX_AVX_BLOCK1_KERNEL)
#if defined(WITH_NO_SPECIFIC_COMPLEX_KERNEL)
        if (THIS_COMPLEX_ELPA_KERNEL .eq. COMPLEX_ELPA_KERNEL_AVX_BLOCK1) then
#endif /* WITH_NO_SPECIFIC_COMPLEX_KERNEL */
          ttt = mpi_wtime()
          do j = ncols, 1, -1
#ifdef WITH_OPENMP
            call single_hh_trafo_complex_sse_avx_1hv(a(1,j+off+a_off,istripe,my_thread), &
                                                       bcast_buffer(1,j+off),nbw,nl,stripe_width)
#else
            call single_hh_trafo_complex_sse_avx_1hv(a(1,j+off+a_off,istripe), &
                                                       bcast_buffer(1,j+off),nbw,nl,stripe_width)
#endif
          enddo
#if defined(WITH_NO_SPECIFIC_COMPLEX_KERNEL)
        endif
#endif /* WITH_NO_SPECIFIC_COMPLEX_KERNEL */
#endif /* WITH_COMPLEX_AVX_BLOCK1_KERNE */

#ifdef WITH_OPENMP
        if (my_thread==1) then
#endif
          kernel_flops = kernel_flops + 4*4*int(nl,8)*int(ncols,8)*int(nbw,8)
          kernel_time  = kernel_time + mpi_wtime()-ttt
#ifdef WITH_OPENMP
        endif
#endif
#ifdef HAVE_DETAILED_TIMINGS
        call timer%stop("compute_hh_trafo_complex")
#endif


    end subroutine

end subroutine

#define DATATYPE REAL
#define BYTESIZE 8
#define REALCASE 1
#include "redist_band.X90"
#undef DATATYPE
#undef BYTESIZE
#undef REALCASE

#define DATATYPE COMPLEX
#define BYTESIZE 16
#define COMPLEXCASE 1
#include "redist_band.X90"
#undef DATATYPE
#undef BYTESIZE
#undef COMPLEXCASE

!---------------------------------------------------------------------------------------------------
! divide_band: sets the work distribution in band
! Proc n works on blocks block_limits(n)+1 .. block_limits(n+1)

subroutine divide_band(nblocks_total, n_pes, block_limits)
#ifdef HAVE_DETAILED_TIMINGS
   use timings
#endif
   implicit none
   integer, intent(in)  :: nblocks_total ! total number of blocks in band
   integer, intent(in)  :: n_pes         ! number of PEs for division
   integer, intent(out) :: block_limits(0:n_pes)

   integer              :: n, nblocks, nblocks_left

#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("divide_band")
#endif

   block_limits(0) = 0
   if (nblocks_total < n_pes) then
     ! Not enough work for all: The first tasks get exactly 1 block
     do n=1,n_pes
       block_limits(n) = min(nblocks_total,n)
     enddo
   else
     ! Enough work for all. If there is no exact loadbalance,
     ! the LAST tasks get more work since they are finishing earlier!
     nblocks = nblocks_total/n_pes
     nblocks_left = nblocks_total - n_pes*nblocks
     do n=1,n_pes
       if (n<=n_pes-nblocks_left) then
         block_limits(n) = block_limits(n-1) + nblocks
       else
         block_limits(n) = block_limits(n-1) + nblocks + 1
       endif
     enddo
   endif

#ifdef HAVE_DETAILED_TIMINGS
   call timer%stop("divide_band")
#endif

end subroutine

subroutine band_band_real(na, nb, nb2, ab, ab2, d, e, mpi_comm)

!-------------------------------------------------------------------------------
! band_band_real:
! Reduces a real symmetric banded matrix to a real symmetric matrix with smaller bandwidth. Householder transformations are not stored.
! Matrix size na and original bandwidth nb have to be a multiple of the target bandwidth nb2. (Hint: expand your matrix with zero entries, if this
! requirement doesn't hold)
!
!  na          Order of matrix
!
!  nb          Semi bandwidth of original matrix
!
!  nb2         Semi bandwidth of target matrix
!
!  ab          Input matrix with bandwidth nb. The leading dimension of the banded matrix has to be 2*nb. The parallel data layout
!              has to be accordant to divide_band(), i.e. the matrix columns block_limits(n)*nb+1 to min(na, block_limits(n+1)*nb)
!              are located on rank n.
!
!  ab2         Output matrix with bandwidth nb2. The leading dimension of the banded matrix is 2*nb2. The parallel data layout is
!              accordant to divide_band(), i.e. the matrix columns block_limits(n)*nb2+1 to min(na, block_limits(n+1)*nb2) are located
!              on rank n.
!
!  d(na)       Diagonal of tridiagonal matrix, set only on PE 0, set only if ab2 = 1 (output)
!
!  e(na)       Subdiagonal of tridiagonal matrix, set only on PE 0, set only if ab2 = 1 (output)
!
!  mpi_comm
!              MPI-Communicator for the total processor set
!-------------------------------------------------------------------------------
#ifdef HAVE_DETAILED_TIMINGS
   use timings
#endif
   implicit none

   integer, intent(in)    ::  na, nb, nb2, mpi_comm
   real*8, intent(inout)  :: ab(2*nb,*)
   real*8, intent(inout)  :: ab2(2*nb2,*)
   real*8, intent(out)    :: d(na), e(na) ! set only on PE 0

!----------------

   real*8                 :: hv(nb,nb2), w(nb,nb2), w_new(nb,nb2), tau(nb2), hv_new(nb,nb2), &
                             tau_new(nb2), ab_s(1+nb,nb2), ab_r(1+nb,nb2), ab_s2(2*nb2,nb2), hv_s(nb,nb2)

   real*8                 :: work(nb*nb2), work2(nb2*nb2)
   integer                :: lwork, info

   integer                :: istep, i, n, dest
   integer                :: n_off, na_s
   integer                :: my_pe, n_pes, mpierr
   integer                :: nblocks_total, nblocks
   integer                :: nblocks_total2, nblocks2
   integer                :: ireq_ab, ireq_hv
   integer                :: mpi_status(MPI_STATUS_SIZE)
   integer, allocatable   :: mpi_statuses(:,:)
   integer, allocatable   :: block_limits(:), block_limits2(:), ireq_ab2(:)

!----------------

   integer                :: j, nc, nr, ns, ne, iblk

#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("band_band_real")
#endif
   call mpi_comm_rank(mpi_comm,my_pe,mpierr)
   call mpi_comm_size(mpi_comm,n_pes,mpierr)

   ! Total number of blocks in the band:
   nblocks_total = (na-1)/nb + 1
   nblocks_total2 = (na-1)/nb2 + 1

   ! Set work distribution
   allocate(block_limits(0:n_pes))
   call divide_band(nblocks_total, n_pes, block_limits)

   allocate(block_limits2(0:n_pes))
   call divide_band(nblocks_total2, n_pes, block_limits2)

   ! nblocks: the number of blocks for my task
   nblocks = block_limits(my_pe+1) - block_limits(my_pe)
   nblocks2 = block_limits2(my_pe+1) - block_limits2(my_pe)

   allocate(ireq_ab2(1:nblocks2))
   ireq_ab2 = MPI_REQUEST_NULL
   if (nb2>1) then
     do i=0,nblocks2-1
       call mpi_irecv(ab2(1,i*nb2+1),2*nb2*nb2,mpi_real8,0,3,mpi_comm,ireq_ab2(i+1),mpierr)
     enddo
   endif

   ! n_off: Offset of ab within band
   n_off = block_limits(my_pe)*nb
   lwork = nb*nb2
   dest = 0

   ireq_ab = MPI_REQUEST_NULL
   ireq_hv = MPI_REQUEST_NULL

   ! ---------------------------------------------------------------------------
   ! Start of calculations

   na_s = block_limits(my_pe)*nb + 1

   if (my_pe>0 .and. na_s<=na) then
     ! send first nb2 columns to previous PE
     ! Only the PE owning the diagonal does that (sending 1 element of the subdiagonal block also)
     do i=1,nb2
       ab_s(1:nb+1,i) = ab(1:nb+1,na_s-n_off+i-1)
     enddo
     call mpi_isend(ab_s,(nb+1)*nb2,mpi_real8,my_pe-1,1,mpi_comm,ireq_ab,mpierr)
   endif

   do istep=1,na/nb2

     if (my_pe==0) then

       n = MIN(na-na_s-nb2+1,nb) ! number of rows to be reduced
       hv(:,:) = 0
       tau(:) = 0

       ! The last step (istep=na-1) is only needed for sending the last HH vectors.
       ! We don't want the sign of the last element flipped (analogous to the other sweeps)
       if (istep < na/nb2) then

         ! Transform first block column of remaining matrix
         call dgeqrf(n, nb2, ab(1+nb2,na_s-n_off), 2*nb-1, tau, work, lwork, info);

         do i=1,nb2
           hv(i,i) = 1.0
           hv(i+1:n,i) = ab(1+nb2+1:1+nb2+n-i,na_s-n_off+i-1)
           ab(1+nb2+1:2*nb,na_s-n_off+i-1) = 0
         enddo

       endif

       if (nb2==1) then
         d(istep) = ab(1,na_s-n_off)
	 e(istep) = ab(2,na_s-n_off)
	 if (istep == na) then
	   e(na) = 0
         endif
       else
         ab_s2 = 0
         ab_s2(:,:) = ab(1:nb2+1,na_s-n_off:na_s-n_off+nb2-1)
         if (block_limits2(dest+1)<istep) then
           dest = dest+1
         endif
         call mpi_send(ab_s2,2*nb2*nb2,mpi_real8,dest,3,mpi_comm,mpierr)
       endif

     else
       if (na>na_s+nb2-1) then
         ! Receive Householder vectors from previous task, from PE owning subdiagonal
         call mpi_recv(hv,nb*nb2,mpi_real8,my_pe-1,2,mpi_comm,mpi_status,mpierr)
         do i=1,nb2
	   tau(i) = hv(i,i)
	   hv(i,i) = 1.
         enddo
       endif
     endif

     na_s = na_s+nb2
     if (na_s-n_off > nb) then
       ab(:,1:nblocks*nb) = ab(:,nb+1:(nblocks+1)*nb)
       ab(:,nblocks*nb+1:(nblocks+1)*nb) = 0
       n_off = n_off + nb
     endif

     do iblk=1,nblocks
       ns = na_s + (iblk-1)*nb - n_off ! first column in block
       ne = ns+nb-nb2                    ! last column in block

       if (ns+n_off>na) exit

         nc = MIN(na-ns-n_off+1,nb) ! number of columns in diagonal block
         nr = MIN(na-nb-ns-n_off+1,nb) ! rows in subdiagonal block (may be < 0!!!)
                                       ! Note that nr>=0 implies that diagonal block is full (nc==nb)!

         call wy_gen(nc,nb2,w,hv,tau,work,nb)

         if (iblk==nblocks .and. nc==nb) then
           !request last nb2 columns
           call mpi_recv(ab_r,(nb+1)*nb2,mpi_real8,my_pe+1,1,mpi_comm,mpi_status,mpierr)
           do i=1,nb2
	     ab(1:nb+1,ne+i-1) = ab_r(:,i)
           enddo
         endif

         hv_new(:,:) = 0 ! Needed, last rows must be 0 for nr < nb
         tau_new(:) = 0

         if (nr>0) then
           call wy_right(nr,nb,nb2,ab(nb+1,ns),2*nb-1,w,hv,work,nb)

           call dgeqrf(nr,nb2,ab(nb+1,ns),2*nb-1,tau_new,work,lwork,info);

           do i=1,nb2
	     hv_new(i,i) = 1.0
	     hv_new(i+1:,i) = ab(nb+2:2*nb-i+1,ns+i-1)
	     ab(nb+2:,ns+i-1) = 0
	   enddo

	   !send hh-vector
	   if (iblk==nblocks) then
	     call mpi_wait(ireq_hv,mpi_status,mpierr)
	     hv_s = hv_new
	     do i=1,nb2
	       hv_s(i,i) = tau_new(i)
             enddo
	     call mpi_isend(hv_s,nb*nb2,mpi_real8,my_pe+1,2,mpi_comm,ireq_hv,mpierr)
           endif

         endif

	 call wy_symm(nc,nb2,ab(1,ns),2*nb-1,w,hv,work,work2,nb)

         if (my_pe>0 .and. iblk==1) then
	   !send first nb2 columns to previous PE
	   call mpi_wait(ireq_ab,mpi_status,mpierr)
	   do i=1,nb2
	     ab_s(1:nb+1,i) = ab(1:nb+1,ns+i-1)
	   enddo
	   call mpi_isend(ab_s,(nb+1)*nb2,mpi_real8,my_pe-1,1,mpi_comm,ireq_ab,mpierr)
         endif

         if (nr>0) then
           call wy_gen(nr,nb2,w_new,hv_new,tau_new,work,nb)
	   call wy_left(nb-nb2,nr,nb2,ab(nb+1-nb2,ns+nb2),2*nb-1,w_new,hv_new,work,nb)
         endif

         ! Use new HH vector for the next block
	 hv(:,:) = hv_new(:,:)
         tau = tau_new
       enddo
     enddo

     ! Finish the last outstanding requests
     call mpi_wait(ireq_ab,mpi_status,mpierr)
     call mpi_wait(ireq_hv,mpi_status,mpierr)
     allocate(mpi_statuses(MPI_STATUS_SIZE,nblocks2))
     call mpi_waitall(nblocks2,ireq_ab2,mpi_statuses,mpierr)
     deallocate(mpi_statuses)

     call mpi_barrier(mpi_comm,mpierr)

     deallocate(block_limits)
     deallocate(block_limits2)
     deallocate(ireq_ab2)
#ifdef HAVE_DETAILED_TIMINGS
   call timer%stop("band_band_real")
#endif

end subroutine

subroutine wy_gen(n, nb, W, Y, tau, mem, lda)
#ifdef HAVE_DETAILED_TIMINGS
   use timings
#endif
   implicit none
   integer, intent(in) :: n		!length of householder-vectors
   integer, intent(in) :: nb		!number of householder-vectors
   integer, intent(in) :: lda		!leading dimension of Y and W
   real*8, intent(in)  :: Y(lda,nb)	!matrix containing nb householder-vectors of length b
   real*8, intent(in)  :: tau(nb)	!tau values
   real*8, intent(out) :: W(lda,nb)	!output matrix W
   real*8, intent(in)  :: mem(nb)	!memory for a temporary matrix of size nb

   integer             :: i

#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("wy_gen")
#endif

   W(1:n,1) = tau(1)*Y(1:n,1)
   do i=2,nb
     W(1:n,i) = tau(i)*Y(1:n,i)
     call DGEMV('T',n,i-1,1.d0,Y,lda,W(1,i),1,0.d0,mem,1)
     call DGEMV('N',n,i-1,-1.d0,W,lda,mem,1,1.d0,W(1,i),1)
   enddo

#ifdef HAVE_DETAILED_TIMINGS
   call timer%stop("wy_gen")
#endif

end subroutine

subroutine wy_left(n, m, nb, A, lda, W, Y, mem, lda2)
#ifdef HAVE_DETAILED_TIMINGS
   use timings
#endif
   implicit none
   integer, intent(in)   :: n		!width of the matrix A
   integer, intent(in)   :: m		!length of matrix W and Y
   integer, intent(in)   :: nb		!width of matrix W and Y
   integer, intent(in)   :: lda		!leading dimension of A
   integer, intent(in)   :: lda2		!leading dimension of W and Y
   real*8, intent(inout) :: A(lda,*)	!matrix to be transformed
   real*8, intent(in)    :: W(m,nb)	!blocked transformation matrix W
   real*8, intent(in)    :: Y(m,nb)	!blocked transformation matrix Y
   real*8, intent(inout) :: mem(n,nb)	!memory for a temporary matrix of size n x nb

#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("wy_left")
#endif

   call DGEMM('T', 'N', nb, n, m, 1.d0, W, lda2, A, lda, 0.d0, mem, nb)
   call DGEMM('N', 'N', m, n, nb, -1.d0, Y, lda2, mem, nb, 1.d0, A, lda)

#ifdef HAVE_DETAILED_TIMINGS
   call timer%stop("wy_left")
#endif

end subroutine

! --------------------------------------------------------------------------------------------------

subroutine wy_right(n, m, nb, A, lda, W, Y, mem, lda2)
#ifdef HAVE_DETAILED_TIMINGS
   use timings
#endif
   implicit none
   integer, intent(in)   :: n		!height of the matrix A
   integer, intent(in)   :: m		!length of matrix W and Y
   integer, intent(in)   :: nb		!width of matrix W and Y
   integer, intent(in)   :: lda		!leading dimension of A
   integer, intent(in)   :: lda2		!leading dimension of W and Y
   real*8, intent(inout) :: A(lda,*)	!matrix to be transformed
   real*8, intent(in)    :: W(m,nb)	!blocked transformation matrix W
   real*8, intent(in)    :: Y(m,nb)	!blocked transformation matrix Y
   real*8, intent(inout) :: mem(n,nb)	!memory for a temporary matrix of size n x nb

#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("wy_right")
#endif

   call DGEMM('N', 'N', n, nb, m, 1.d0, A, lda, W, lda2, 0.d0, mem, n)
   call DGEMM('N', 'T', n, m, nb, -1.d0, mem, n, Y, lda2, 1.d0, A, lda)

#ifdef HAVE_DETAILED_TIMINGS
   call timer%stop("wy_right")
#endif

end subroutine

! --------------------------------------------------------------------------------------------------

subroutine wy_symm(n, nb, A, lda, W, Y, mem, mem2, lda2)
#ifdef HAVE_DETAILED_TIMINGS
   use timings
#endif
   implicit none
   integer, intent(in)   :: n		!width/heigth of the matrix A; length of matrix W and Y
   integer, intent(in)   :: nb		!width of matrix W and Y
   integer, intent(in)   :: lda		!leading dimension of A
   integer, intent(in)   :: lda2		!leading dimension of W and Y
   real*8, intent(inout) :: A(lda,*)	!matrix to be transformed
   real*8, intent(in)    :: W(n,nb)	!blocked transformation matrix W
   real*8, intent(in)    :: Y(n,nb)	!blocked transformation matrix Y
   real*8                :: mem(n,nb)	!memory for a temporary matrix of size n x nb
   real*8                :: mem2(nb,nb)	!memory for a temporary matrix of size nb x nb

#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("wy_symm")
#endif

   call DSYMM('L', 'L', n, nb, 1.d0, A, lda, W, lda2, 0.d0, mem, n)
   call DGEMM('T', 'N', nb, nb, n, 1.d0, mem, n, W, lda2, 0.d0, mem2, nb)
   call DGEMM('N', 'N', n, nb, nb, -0.5d0, Y, lda2, mem2, nb, 1.d0, mem, n)
   call DSYR2K('L', 'N', n, nb, -1.d0, Y, lda2, mem, n, 1.d0, A, lda)

#ifdef HAVE_DETAILED_TIMINGS
   call timer%stop("wy_symm")
#endif
end subroutine

end module ELPA2
