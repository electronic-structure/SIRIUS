module single_hh_trafo_real
  implicit none
#include "config-f90.h"

#ifdef WITH_OPENMP
  public single_hh_trafo_real_cpu_openmp
#else
  public single_hh_trafo_real_cpu
#endif
  contains

#ifdef WITH_OPENMP
    subroutine single_hh_trafo_real_cpu_openmp(q, hh, nb, nq, ldq)
#else
    subroutine single_hh_trafo_real_cpu(q, hh, nb, nq, ldq)
#endif

#ifdef HAVE_DETAILED_TIMINGS
      use timings
#endif
      use precision
      ! Perform single real Householder transformation.
      ! This routine is not performance critical and thus it is coded here in Fortran

      implicit none
      integer(kind=ik), intent(in)   :: nb, nq, ldq
!      real(kind=rk), intent(inout)   :: q(ldq, *)
!      real(kind=rk), intent(in)      :: hh(*)
      real(kind=rk), intent(inout)   :: q(1:ldq, 1:nb)
      real(kind=rk), intent(in)      :: hh(1:nb)
      integer(kind=ik)               :: i
      real(kind=rk)                  :: v(nq)

#ifdef HAVE_DETAILED_TIMINGS
#ifdef WITH_OPENMP
      call timer%start("single_hh_trafo_real_cpu_openmp")
#else
      call timer%start("single_hh_trafo_real_cpu")
#endif
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
#ifdef WITH_OPENMP
      call timer%stop("single_hh_trafo_real_cpu_openmp")
#else
      call timer%stop("single_hh_trafo_real_cpu")
#endif
#endif
    end subroutine


end module
