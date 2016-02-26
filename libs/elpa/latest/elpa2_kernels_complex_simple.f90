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
!
! --------------------------------------------------------------------------------------------------
!
! This file contains the compute intensive kernels for the Householder transformations.
!
! This is the small and simple version (no hand unrolling of loops etc.) but for some
! compilers this performs better than a sophisticated version with transformed and unrolled loops.
!
! It should be compiled with the highest possible optimization level.
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".
!
! --------------------------------------------------------------------------------------------------

#include "config-f90.h"

module complex_generic_simple_kernel

  private
  public single_hh_trafo_complex_generic_simple
contains
  subroutine single_hh_trafo_complex_generic_simple(q, hh, nb, nq, ldq)
    use precision
#ifdef HAVE_DETAILED_TIMINGS
    use timings
#endif
    implicit none

    integer(kind=ik), intent(in)    :: nb, nq, ldq
#ifdef DESPERATELY_WANT_ASSUMED_SIZE
    complex(kind=ck), intent(inout) :: q(ldq,*)
    complex(kind=ck), intent(in)    :: hh(*)
#else
    complex(kind=ck), intent(inout) :: q(1:ldq,1:nb)
    complex(kind=ck), intent(in)    :: hh(1:nb)
#endif
    integer(kind=ik)                :: i
    complex(kind=ck)                :: h1, tau1, x(nq)

#ifdef HAVE_DETAILED_TIMINGS
    call timer%start("kernel complex generic simple: single_hh_trafo_complex_generic_simple")
#endif
    ! Just one Householder transformation

    x(1:nq) = q(1:nq,1)

    do i=2,nb
       x(1:nq) = x(1:nq) + q(1:nq,i)*conjg(hh(i))
    enddo

    tau1 = hh(1)
    x(1:nq) = x(1:nq)*(-tau1)

    q(1:nq,1) = q(1:nq,1) + x(1:nq)

    do i=2,nb
       q(1:nq,i) = q(1:nq,i) + x(1:nq)*hh(i)
    enddo
#ifdef HAVE_DETAILED_TIMINGS
    call timer%stop("kernel complex generic simple: single_hh_trafo_complex_generic_simple")
#endif
  end subroutine single_hh_trafo_complex_generic_simple

  ! --------------------------------------------------------------------------------------------------
  subroutine double_hh_trafo_complex_generic_simple(q, hh, nb, nq, ldq, ldh)
    use precision
#ifdef HAVE_DETAILED_TIMINGS
    use timings
#endif
    implicit none

    integer(kind=ik), intent(in)    :: nb, nq, ldq, ldh
#ifdef DESPERATELY_WANT_ASSUMED_SIZE
    complex(kind=ck), intent(inout) :: q(ldq,*)
    complex(kind=ck), intent(in)    :: hh(ldh,*)
#else
    complex(kind=ck), intent(inout) :: q(1:ldq,1:nb+1)
    complex(kind=ck), intent(in)    :: hh(1:ldh,1:2)
#endif
    complex(kind=ck)                :: s, h1, h2, tau1, tau2, x(nq), y(nq)
    integer(kind=ik)                :: i

#ifdef HAVE_DETAILED_TIMINGS
    call timer%start("kernel complex generic simple: double_hh_trafo_complex_generic_simple")
#endif
    ! Calculate dot product of the two Householder vectors

    s = conjg(hh(2,2))*1
    do i=3,nb
       s = s+(conjg(hh(i,2))*hh(i-1,1))
    enddo

    ! Do the Householder transformations

    x(1:nq) = q(1:nq,2)

    y(1:nq) = q(1:nq,1) + q(1:nq,2)*conjg(hh(2,2))

    do i=3,nb
       h1 = conjg(hh(i-1,1))
       h2 = conjg(hh(i,2))
       x(1:nq) = x(1:nq) + q(1:nq,i)*h1
       y(1:nq) = y(1:nq) + q(1:nq,i)*h2
    enddo

    x(1:nq) = x(1:nq) + q(1:nq,nb+1)*conjg(hh(nb,1))

    tau1 = hh(1,1)
    tau2 = hh(1,2)

    h1 = -tau1
    x(1:nq) = x(1:nq)*h1
    h1 = -tau2
    h2 = -tau2*s
    y(1:nq) = y(1:nq)*h1 + x(1:nq)*h2

    q(1:nq,1) = q(1:nq,1) + y(1:nq)
    q(1:nq,2) = q(1:nq,2) + x(1:nq) + y(1:nq)*hh(2,2)

    do i=3,nb
       h1 = hh(i-1,1)
       h2 = hh(i,2)
       q(1:nq,i) = q(1:nq,i) + x(1:nq)*h1 + y(1:nq)*h2
    enddo

    q(1:nq,nb+1) = q(1:nq,nb+1) + x(1:nq)*hh(nb,1)

#ifdef HAVE_DETAILED_TIMINGS
    call timer%stop("kernel complex generic simple: double_hh_trafo_complex_generic_simple")
#endif
  end subroutine double_hh_trafo_complex_generic_simple
end module complex_generic_simple_kernel
! --------------------------------------------------------------------------------------------------
