!    This file is part of ELPA.
!
!    The ELPA library was originally created by the ELPA consortium,
!    consisting of the following organizations:
!
!    - Max Planck Computing and Data Facility (MPCDF), fomerly known as
!      Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
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
!    http://elpa.mpcdf.mpg.de/
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
!> \brief Fortran module which provides the routines to use the two-stage ELPA solver
module ELPA2

! Version 1.1.2, 2011-02-21

  use elpa_utilities
  use elpa1, only : elpa_print_times, time_evp_back, time_evp_fwd, time_evp_solve
  use elpa2_utilities

  implicit none

  PRIVATE ! By default, all routines contained are private

  ! The following routines are public:

  public :: solve_evp_real_2stage
  public :: solve_evp_complex_2stage


!******
contains
!-------------------------------------------------------------------------------
!>  \brief solve_evp_real_2stage: Fortran function to solve the real eigenvalue problem with a 2 stage approach
!>
!>  Parameters
!>
!>  \param na                                   Order of matrix a
!>
!>  \param nev                                  Number of eigenvalues needed
!>
!>  \param a(lda,matrixCols)                    Distributed matrix for which eigenvalues are to be computed.
!>                                              Distribution is like in Scalapack.
!>                                              The full matrix must be set (not only one half like in scalapack).
!>                                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                                  Leading dimension of a
!>
!>  \param ev(na)                               On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)                    On output: Eigenvectors of a
!>                                              Distribution is like in Scalapack.
!>                                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                                  Leading dimension of q
!>
!>  \param nblk                                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols                           local columns of matrix a and q
!>
!>  \param mpi_comm_rows                        MPI communicator for rows
!>  \param mpi_comm_cols                        MPI communicator for columns
!>  \param mpi_comm_all                         MPI communicator for the total processor set
!>
!>  \param THIS_REAL_ELPA_KERNEL_API (optional) specify used ELPA2 kernel via API
!>
!>  \param use_qr (optional)                    use QR decomposition
!>
!>  \result success                             logical, false if error occured
!-------------------------------------------------------------------------------

function solve_evp_real_2stage(na, nev, a, lda, ev, q, ldq, nblk,        &
                               matrixCols,                               &
                                 mpi_comm_rows, mpi_comm_cols,           &
                                 mpi_comm_all, THIS_REAL_ELPA_KERNEL_API,&
                                 useQR) result(success)

#ifdef HAVE_DETAILED_TIMINGS
   use timings
#endif
   use elpa1_compute
   use elpa2_compute
   use elpa_mpi
   use precision
   implicit none
   logical, intent(in), optional          :: useQR
   logical                                :: useQRActual, useQREnvironment
   integer(kind=ik), intent(in), optional :: THIS_REAL_ELPA_KERNEL_API
   integer(kind=ik)                       :: THIS_REAL_ELPA_KERNEL

   integer(kind=ik), intent(in)           :: na, nev, lda, ldq, matrixCols, mpi_comm_rows, &
                                             mpi_comm_cols, mpi_comm_all
   integer(kind=ik), intent(in)           :: nblk
   real(kind=rk), intent(inout)           :: a(lda,matrixCols), ev(na), q(ldq,matrixCols)
   ! was
   ! real a(lda,*), q(ldq,*)
   real(kind=rk), allocatable             :: hh_trans_real(:,:)

   integer(kind=ik)                       :: my_pe, n_pes, my_prow, my_pcol, np_rows, np_cols, mpierr
   integer(kind=ik)                       :: nbw, num_blocks
   real(kind=rk), allocatable             :: tmat(:,:,:), e(:)
   real(kind=rk)                          :: ttt0, ttt1, ttts
   integer(kind=ik)                       :: i
   logical                                :: success
   logical, save                          :: firstCall = .true.
   logical                                :: wantDebug

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
     if (mod(na,2) .ne. 0) then
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

   ! check whether choosen kernel is allowed: function returns true if NOT allowed! change this
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
       ! check whether generic kernel is defined
       if (AVAILABLE_REAL_ELPA_KERNELS(REAL_ELPA_KERNEL_GENERIC) .eq. 1) then
         write(error_unit,*) "The default kernel REAL_ELPA_KERNEL_GENERIC will be used !"
       else
         write(error_unit,*) "As default kernel ",REAL_ELPA_KERNEL_NAMES(DEFAULT_REAL_ELPA_KERNEL)," will be used"
       endif
     endif  ! my_pe == 0
     if (AVAILABLE_REAL_ELPA_KERNELS(REAL_ELPA_KERNEL_GENERIC) .eq. 1) then
       THIS_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_GENERIC
     else
       THIS_REAL_ELPA_KERNEL = DEFAULT_REAL_ELPA_KERNEL
     endif
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
   call tridiag_band_real(na, nbw, nblk, a, lda, ev, e, matrixCols, hh_trans_real, &
                          mpi_comm_rows, mpi_comm_cols, mpi_comm_all)
   ttt1 = MPI_Wtime()
   if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
      write(error_unit,*) 'Time tridiag_band_real          :',ttt1-ttt0
#ifdef WITH_MPI
   call mpi_bcast(ev,na,MPI_REAL8,0,mpi_comm_all,mpierr)
   call mpi_bcast(e,na,MPI_REAL8,0,mpi_comm_all,mpierr)
#endif
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
   call trans_ev_tridi_to_band_real(na, nev, nblk, nbw, q, ldq, matrixCols, hh_trans_real, &
                                    mpi_comm_rows, mpi_comm_cols, wantDebug, success,      &
                                    THIS_REAL_ELPA_KERNEL)
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
!>  \brief solve_evp_complex_2stage: Fortran function to solve the complex eigenvalue problem with a 2 stage approach
!>
!>  Parameters
!>
!>  \param na                                   Order of matrix a
!>
!>  \param nev                                  Number of eigenvalues needed
!>
!>  \param a(lda,matrixCols)                    Distributed matrix for which eigenvalues are to be computed.
!>                                              Distribution is like in Scalapack.
!>                                              The full matrix must be set (not only one half like in scalapack).
!>                                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                                  Leading dimension of a
!>
!>  \param ev(na)                               On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)                    On output: Eigenvectors of a
!>                                              Distribution is like in Scalapack.
!>                                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                                  Leading dimension of q
!>
!>  \param nblk                                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols                           local columns of matrix a and q
!>
!>  \param mpi_comm_rows                        MPI communicator for rows
!>  \param mpi_comm_cols                        MPI communicator for columns
!>  \param mpi_comm_all                         MPI communicator for the total processor set
!>
!>  \param THIS_REAL_ELPA_KERNEL_API (optional) specify used ELPA2 kernel via API
!>
!>  \result success                             logical, false if error occured
!-------------------------------------------------------------------------------
function solve_evp_complex_2stage(na, nev, a, lda, ev, q, ldq, nblk, &
                                  matrixCols, mpi_comm_rows, mpi_comm_cols,      &
                                    mpi_comm_all, THIS_COMPLEX_ELPA_KERNEL_API) result(success)

#ifdef HAVE_DETAILED_TIMINGS
   use timings
#endif
   use elpa1_compute
   use elpa2_compute
   use elpa_mpi
   use precision
   implicit none
   integer(kind=ik), intent(in), optional :: THIS_COMPLEX_ELPA_KERNEL_API
   integer(kind=ik)                       :: THIS_COMPLEX_ELPA_KERNEL
   integer(kind=ik), intent(in)           :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all
   complex(kind=ck), intent(inout)        :: a(lda,matrixCols), q(ldq,matrixCols)
   ! was
   ! complex a(lda,*), q(ldq,*)
   real(kind=rk), intent(inout)           :: ev(na)
   complex(kind=ck), allocatable          :: hh_trans_complex(:,:)

   integer(kind=ik)                       :: my_prow, my_pcol, np_rows, np_cols, mpierr, my_pe, n_pes
   integer(kind=ik)                       :: l_cols, l_rows, l_cols_nev, nbw, num_blocks
   complex(kind=ck), allocatable          :: tmat(:,:,:)
   real(kind=rk), allocatable             :: q_real(:,:), e(:)
   real(kind=rk)                          :: ttt0, ttt1, ttts
   integer(kind=ik)                       :: i

   logical                                :: success, wantDebug
   logical, save                          :: firstCall = .true.

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
       ! check whether generic kernel is defined
       if (AVAILABLE_COMPLEX_ELPA_KERNELS(COMPLEX_ELPA_KERNEL_GENERIC) .eq. 1) then
         write(error_unit,*) "The default kernel COMPLEX_ELPA_KERNEL_GENERIC will be used !"
       else
         write(error_unit,*) "As default kernel ",COMPLEX_ELPA_KERNEL_NAMES(DEFAULT_COMPLEX_ELPA_KERNEL)," will be used"
       endif
     endif  ! my_pe == 0
     if (AVAILABLE_COMPLEX_ELPA_KERNELS(COMPLEX_ELPA_KERNEL_GENERIC) .eq. 1) then
       THIS_COMPLEX_ELPA_KERNEL = COMPLEX_ELPA_KERNEL_GENERIC
     else
       THIS_COMPLEX_ELPA_KERNEL = DEFAULT_COMPLEX_ELPA_KERNEL
     endif
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
   call tridiag_band_complex(na, nbw, nblk, a, lda, ev, e, matrixCols, hh_trans_complex, &
                             mpi_comm_rows, mpi_comm_cols, mpi_comm_all)
   ttt1 = MPI_Wtime()
   if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
      write(error_unit,*) 'Time tridiag_band_complex          :',ttt1-ttt0
#ifdef WITH_MPI
   call mpi_bcast(ev,na,MPI_REAL8,0,mpi_comm_all,mpierr)
   call mpi_bcast(e,na,MPI_REAL8,0,mpi_comm_all,mpierr)
#endif
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
   call trans_ev_tridi_to_band_complex(na, nev, nblk, nbw, q, ldq,   &
                                       matrixCols, hh_trans_complex, &
                                       mpi_comm_rows, mpi_comm_cols, &
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

end module ELPA2
