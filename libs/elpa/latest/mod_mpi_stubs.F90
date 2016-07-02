!    This file is part of ELPA.
!
!    The ELPA library was originally created by the ELPA consortium,
!    consisting of the following organizations:
!
!    - Max Planck Computing and Data Facility (MPCDF), formerly known as
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
! Author Andreas Marek, MPCDF

#include "config-f90.h"

module elpa_mpi_stubs
  use precision
  implicit none

  public

  integer(kind=ik), parameter :: MPI_COMM_SELF=1, MPI_COMM_WORLD=1

  contains
    function MPI_WTIME() result(time)
      use iso_c_binding
#ifndef WITH_MPI
      use time_c
#endif
      implicit none

      real(kind=c_double) :: time
#ifndef WITH_MPI
      time = seconds()
#endif
    end function

    subroutine mpi_comm_size(mpi_comm_world, ntasks, mpierr)

      use precision

      implicit none

      integer(kind=ik), intent(in)    :: mpi_comm_world
      integer(kind=ik), intent(inout) :: ntasks
      integer(kind=ik), intent(inout) :: mpierr

      ntasks = 1
      mpierr = 0

      return

    end subroutine mpi_comm_size

    subroutine mpi_comm_rank(mpi_comm_world, myid, mpierr)
      use precision
      implicit none
      integer(kind=ik), intent(in)    :: mpi_comm_world
      integer(kind=ik), intent(inout) :: mpierr
      integer(kind=ik), intent(inout) :: myid

      myid = 0
      mpierr = 0

      return
    end subroutine mpi_comm_rank

    subroutine mpi_comm_split(mpi_communicator, color, key, new_comm, mpierr)
      use precision
      implicit none
      integer(kind=ik), intent(in)    :: mpi_communicator, color, key
      integer(kind=ik), intent(inout) :: new_comm, mpierr

      new_comm = mpi_communicator
      mpierr = 0
      return
    end subroutine mpi_comm_split

end module
