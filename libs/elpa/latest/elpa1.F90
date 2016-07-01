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

!> \mainpage
!> Eigenvalue SoLvers for Petaflop-Applications (ELPA)
!> \par
!> http://elpa.mpcdf.mpg.de
!>
!> \par
!>    The ELPA library was originally created by the ELPA consortium,
!>    consisting of the following organizations:
!>
!>    - Max Planck Computing and Data Facility (MPCDF) formerly known as
!>      Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
!>    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
!>      Informatik,
!>    - Technische Universität München, Lehrstuhl für Informatik mit
!>      Schwerpunkt Wissenschaftliches Rechnen ,
!>    - Fritz-Haber-Institut, Berlin, Abt. Theorie,
!>    - Max-Plack-Institut für Mathematik in den Naturwissenschaftrn,
!>      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
!>      and
!>    - IBM Deutschland GmbH
!>
!>   Some parts and enhancements of ELPA have been contributed and authored
!>   by the Intel Corporation which is not part of the ELPA consortium.
!>
!>   Contributions to the ELPA source have been authored by (in alphabetical order):
!>
!> \author T. Auckenthaler, Volker Blum, A. Heinecke, L. Huedepohl, R. Johanni, Werner Jürgens, and A. Marek


#include "config-f90.h"
!> \brief Fortran module which provides the routines to use the one-stage ELPA solver
module ELPA1
  use precision
  use elpa_utilities
  use elpa1_auxiliary

  implicit none

  ! The following routines are public:
  private

  public :: get_elpa_row_col_comms     !< old, deprecated interface: Sets MPI row/col communicators
  public :: get_elpa_communicators     !< Sets MPI row/col communicators

  public :: solve_evp_real             !< old, deprecated interface: Driver routine for real eigenvalue problem
  public :: solve_evp_real_1stage      !< Driver routine for real eigenvalue problem
  public :: solve_evp_complex          !< old, deprecated interface:  Driver routine for complex eigenvalue problem
  public :: solve_evp_complex_1stage   !< Driver routine for complex eigenvalue problem

  ! Timing results, set by every call to solve_evp_xxx

  real(kind=rk), public :: time_evp_fwd    !< time for forward transformations (to tridiagonal form)
  real(kind=rk), public :: time_evp_solve  !< time for solving the tridiagonal system
  real(kind=rk), public :: time_evp_back   !< time for back transformations of eigenvectors

  logical, public :: elpa_print_times = .false. !< Set elpa_print_times to .true. for explicit timing outputs


!> \brief get_elpa_row_col_comms:  old, deprecated Fortran function to create the MPI communicators for ELPA. Better use "elpa_get_communicators"
!> \details
!> The interface and variable definition is the same as in "elpa_get_communicators"
!> \param  mpi_comm_global   Global communicator for the calculations (in)
!>
!> \param  my_prow           Row coordinate of the calling process in the process grid (in)
!>
!> \param  my_pcol           Column coordinate of the calling process in the process grid (in)
!>
!> \param  mpi_comm_rows     Communicator for communicating within rows of processes (out)
!>
!> \param  mpi_comm_cols     Communicator for communicating within columns of processes (out)
!> \result mpierr            integer error value of mpi_comm_split function
  interface get_elpa_row_col_comms
    module procedure get_elpa_communicators
  end interface

!> \brief solve_evp_real: old, deprecated Fortran function to solve the real eigenvalue problem with 1-stage solver. Better use "solve_evp_real_1stage"
!>
!> \details
!>  The interface and variable definition is the same as in "elpa_solve_evp_real_1stage"
!  Parameters
!
!> \param  na                   Order of matrix a
!>
!> \param  nev                  Number of eigenvalues needed.
!>                              The smallest nev eigenvalues/eigenvectors are calculated.
!>
!> \param  a(lda,matrixCols)    Distributed matrix for which eigenvalues are to be computed.
!>                              Distribution is like in Scalapack.
!>                              The full matrix must be set (not only one half like in scalapack).
!>                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                  Leading dimension of a
!>
!>  \param ev(na)               On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)    On output: Eigenvectors of a
!>                              Distribution is like in Scalapack.
!>                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                  Leading dimension of q
!>
!>  \param nblk                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols           distributed number of matrix columns
!>
!>  \param mpi_comm_rows        MPI-Communicator for rows
!>  \param mpi_comm_cols        MPI-Communicator for columns
!>
!>  \result                     success


  interface solve_evp_real
    module procedure solve_evp_real_1stage
  end interface

!> \brief solve_evp_complex: old, deprecated Fortran function to solve the complex eigenvalue problem with 1-stage solver. Better use "solve_evp_complex_1stage"
!>
!> \details
!> The interface and variable definition is the same as in "elpa_solve_evp_complex_1stage"
!  Parameters
!
!> \param  na                   Order of matrix a
!>
!> \param  nev                  Number of eigenvalues needed.
!>                              The smallest nev eigenvalues/eigenvectors are calculated.
!>
!> \param  a(lda,matrixCols)    Distributed matrix for which eigenvalues are to be computed.
!>                              Distribution is like in Scalapack.
!>                              The full matrix must be set (not only one half like in scalapack).
!>                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                  Leading dimension of a
!>
!>  \param ev(na)               On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)    On output: Eigenvectors of a
!>                              Distribution is like in Scalapack.
!>                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                  Leading dimension of q
!>
!>  \param nblk                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols           distributed number of matrix columns
!>
!>  \param mpi_comm_rows        MPI-Communicator for rows
!>  \param mpi_comm_cols        MPI-Communicator for columns
!>
!>  \result                     success


  interface solve_evp_complex
    module procedure solve_evp_complex_1stage
  end interface

contains

!-------------------------------------------------------------------------------

!> \brief Fortran function to create the MPI communicators for ELPA.
! All ELPA routines need MPI communicators for communicating within
! rows or columns of processes, these are set here.
! mpi_comm_rows/mpi_comm_cols can be free'd with MPI_Comm_free if not used any more.
!
!  Parameters
!
!> \param  mpi_comm_global   Global communicator for the calculations (in)
!>
!> \param  my_prow           Row coordinate of the calling process in the process grid (in)
!>
!> \param  my_pcol           Column coordinate of the calling process in the process grid (in)
!>
!> \param  mpi_comm_rows     Communicator for communicating within rows of processes (out)
!>
!> \param  mpi_comm_cols     Communicator for communicating within columns of processes (out)
!> \result mpierr            integer error value of mpi_comm_split function


function get_elpa_communicators(mpi_comm_global, my_prow, my_pcol, mpi_comm_rows, mpi_comm_cols) result(mpierr)
   use precision
   use elpa_mpi
   implicit none

   integer(kind=ik), intent(in)  :: mpi_comm_global, my_prow, my_pcol
   integer(kind=ik), intent(out) :: mpi_comm_rows, mpi_comm_cols

   integer(kind=ik)              :: mpierr

   ! mpi_comm_rows is used for communicating WITHIN rows, i.e. all processes
   ! having the same column coordinate share one mpi_comm_rows.
   ! So the "color" for splitting is my_pcol and the "key" is my row coordinate.
   ! Analogous for mpi_comm_cols

   call mpi_comm_split(mpi_comm_global,my_pcol,my_prow,mpi_comm_rows,mpierr)
   call mpi_comm_split(mpi_comm_global,my_prow,my_pcol,mpi_comm_cols,mpierr)

end function get_elpa_communicators


!> \brief solve_evp_real_1stage: Fortran function to solve the real eigenvalue problem with 1-stage solver
!>
!  Parameters
!
!> \param  na                   Order of matrix a
!>
!> \param  nev                  Number of eigenvalues needed.
!>                              The smallest nev eigenvalues/eigenvectors are calculated.
!>
!> \param  a(lda,matrixCols)    Distributed matrix for which eigenvalues are to be computed.
!>                              Distribution is like in Scalapack.
!>                              The full matrix must be set (not only one half like in scalapack).
!>                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                  Leading dimension of a
!>
!>  \param ev(na)               On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)    On output: Eigenvectors of a
!>                              Distribution is like in Scalapack.
!>                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                  Leading dimension of q
!>
!>  \param nblk                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols           distributed number of matrix columns
!>
!>  \param mpi_comm_rows        MPI-Communicator for rows
!>  \param mpi_comm_cols        MPI-Communicator for columns
!>
!>  \result                     success


function solve_evp_real_1stage(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols) result(success)
   use precision
#ifdef HAVE_DETAILED_TIMINGS
   use timings
#endif
   use elpa_mpi
   use elpa1_compute
   implicit none

   integer(kind=ik), intent(in)  :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
   real(kind=rk)                 :: a(lda,matrixCols), ev(na), q(ldq,matrixCols)
   ! was
   ! real a(lda,*), q(ldq,*)

   integer(kind=ik)              :: my_prow, my_pcol, mpierr
   real(kind=rk), allocatable    :: e(:), tau(:)
   real(kind=rk)                 :: ttt0, ttt1
   logical                       :: success
   logical, save                 :: firstCall = .true.
   logical                       :: wantDebug

#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("solve_evp_real_1stage")
#endif

   call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
   call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)

   success = .true.

   wantDebug = .false.
   if (firstCall) then
     ! are debug messages desired?
     wantDebug = debug_messages_via_environment_variable()
     firstCall = .false.
   endif

   allocate(e(na), tau(na))

   ttt0 = MPI_Wtime()
   call tridiag_real(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, ev, e, tau)

   ttt1 = MPI_Wtime()
   if(my_prow==0 .and. my_pcol==0 .and. elpa_print_times) write(error_unit,*) 'Time tridiag_real :',ttt1-ttt0
   time_evp_fwd = ttt1-ttt0

   ttt0 = MPI_Wtime()
   call solve_tridi(na, nev, ev, e, q, ldq, nblk, matrixCols, mpi_comm_rows, &
                    mpi_comm_cols, wantDebug, success)
   if (.not.(success)) return

   ttt1 = MPI_Wtime()
   if(my_prow==0 .and. my_pcol==0 .and. elpa_print_times) write(error_unit,*) 'Time solve_tridi  :',ttt1-ttt0
   time_evp_solve = ttt1-ttt0

   ttt0 = MPI_Wtime()
   call trans_ev_real(na, nev, a, lda, tau, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols)
   ttt1 = MPI_Wtime()
   if(my_prow==0 .and. my_pcol==0 .and. elpa_print_times) write(error_unit,*) 'Time trans_ev_real:',ttt1-ttt0
   time_evp_back = ttt1-ttt0

   deallocate(e, tau)

#ifdef HAVE_DETAILED_TIMINGS
   call timer%stop("solve_evp_real_1stage")
#endif

end function solve_evp_real_1stage


!> \brief solve_evp_complex_1stage: Fortran function to solve the complex eigenvalue problem with 1-stage solver
!>
!  Parameters
!
!> \param  na                   Order of matrix a
!>
!> \param  nev                  Number of eigenvalues needed.
!>                              The smallest nev eigenvalues/eigenvectors are calculated.
!>
!> \param  a(lda,matrixCols)    Distributed matrix for which eigenvalues are to be computed.
!>                              Distribution is like in Scalapack.
!>                              The full matrix must be set (not only one half like in scalapack).
!>                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                  Leading dimension of a
!>
!>  \param ev(na)               On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)    On output: Eigenvectors of a
!>                              Distribution is like in Scalapack.
!>                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                  Leading dimension of q
!>
!>  \param nblk                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols           distributed number of matrix columns
!>
!>  \param mpi_comm_rows        MPI-Communicator for rows
!>  \param mpi_comm_cols        MPI-Communicator for columns
!>
!>  \result                     success

function solve_evp_complex_1stage(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols) result(success)
#ifdef HAVE_DETAILED_TIMINGS
   use timings
#endif
   use precision
   use elpa_mpi
   use elpa1_compute
   implicit none

   integer(kind=ik), intent(in)     :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
   complex(kind=ck)                 :: a(lda,matrixCols), q(ldq,matrixCols)
   ! was
   ! complex a(lda,*), q(ldq,*)
   real(kind=rk)                    :: ev(na)

   integer(kind=ik)                 :: my_prow, my_pcol, np_rows, np_cols, mpierr
   integer(kind=ik)                 :: l_rows, l_cols, l_cols_nev
   real(kind=rk), allocatable       :: q_real(:,:), e(:)
   complex(kind=ck), allocatable    :: tau(:)
   real(kind=rk)                    :: ttt0, ttt1

   logical                          :: success
   logical, save                    :: firstCall = .true.
   logical                          :: wantDebug

#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("solve_evp_complex_1stage")
#endif

   call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
   call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
   call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
   call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)

   success = .true.

   wantDebug = .false.
   if (firstCall) then
     ! are debug messages desired?
     wantDebug = debug_messages_via_environment_variable()
     firstCall = .false.
   endif


   l_rows = local_index(na, my_prow, np_rows, nblk, -1) ! Local rows of a and q
   l_cols = local_index(na, my_pcol, np_cols, nblk, -1) ! Local columns of q

   l_cols_nev = local_index(nev, my_pcol, np_cols, nblk, -1) ! Local columns corresponding to nev

   allocate(e(na), tau(na))
   allocate(q_real(l_rows,l_cols))

   ttt0 = MPI_Wtime()
   call tridiag_complex(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, ev, e, tau)
   ttt1 = MPI_Wtime()
   if(my_prow==0 .and. my_pcol==0 .and. elpa_print_times) write(error_unit,*) 'Time tridiag_complex :',ttt1-ttt0
   time_evp_fwd = ttt1-ttt0

   ttt0 = MPI_Wtime()
   call solve_tridi(na, nev, ev, e, q_real, l_rows, nblk, matrixCols, mpi_comm_rows, &
                    mpi_comm_cols, wantDebug, success)
   if (.not.(success)) return

   ttt1 = MPI_Wtime()
   if(my_prow==0 .and. my_pcol==0 .and. elpa_print_times) write(error_unit,*) 'Time solve_tridi     :',ttt1-ttt0
   time_evp_solve = ttt1-ttt0

   ttt0 = MPI_Wtime()
   q(1:l_rows,1:l_cols_nev) = q_real(1:l_rows,1:l_cols_nev)

   call trans_ev_complex(na, nev, a, lda, tau, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols)
   ttt1 = MPI_Wtime()
   if(my_prow==0 .and. my_pcol==0 .and. elpa_print_times) write(error_unit,*) 'Time trans_ev_complex:',ttt1-ttt0
   time_evp_back = ttt1-ttt0

   deallocate(q_real)
   deallocate(e, tau)
#ifdef HAVE_DETAILED_TIMINGS
   call timer%stop("solve_evp_complex_1stage")
#endif

end function solve_evp_complex_1stage



end module ELPA1
