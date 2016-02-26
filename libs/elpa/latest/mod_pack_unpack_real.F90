module pack_unpack_real
#include "config-f90.h"
  implicit none

#ifdef WITH_OPENMP
  public pack_row_real_cpu_openmp, unpack_row_real_cpu_openmp
#else
  public pack_row_real_cpu, unpack_row_real_cpu
#endif
  contains

#ifdef WITH_OPENMP
        subroutine pack_row_real_cpu_openmp(a, row, n, stripe_width, stripe_count, max_threads, thread_width, l_nev)
#else
        subroutine pack_row_real_cpu(a, row, n, stripe_width, last_stripe_width, stripe_count)
#endif

#ifdef HAVE_DETAILED_TIMINGS
          use timings
#endif
          use precision
          implicit none
          integer(kind=ik), intent(in) :: n, stripe_count, stripe_width
#ifdef WITH_OPENMP
          integer(kind=ik), intent(in) :: max_threads, thread_width, l_nev
          real(kind=rk), intent(in)    :: a(:,:,:,:)
#else
          integer(kind=ik), intent(in) :: last_stripe_width
          real(kind=rk), intent(in)    :: a(:,:,:)
#endif
          real(kind=rk)                :: row(:)

          integer(kind=ik)             :: i, noff, nl
#ifdef WITH_OPENMP
          integer(kind=ik)             :: nt
#endif

#ifdef HAVE_DETAILED_TIMINGS
#ifdef WITH_OPENMP
          call timer%start("pack_row_real_cpu_openmp")

#else
          call timer%start("pack_row_real_cpu")
#endif
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
#ifdef WITH_OPENMP
          call timer%stop("pack_row_real_cpu_openmp")

#else
          call timer%stop("pack_row_real_cpu")
#endif
#endif

#ifdef WITH_OPENMP
        end subroutine pack_row_real_cpu_openmp
#else
        end subroutine pack_row_real_cpu
#endif

#ifdef WITH_OPENMP
        subroutine unpack_row_real_cpu_openmp(a, row, n, my_thread, stripe_count, thread_width, stripe_width, l_nev)
#ifdef HAVE_DETAILED_TIMINGS
          use timings
#endif
          use precision
          implicit none

          ! Private variables in OMP regions (my_thread) should better be in the argument list!
          integer(kind=ik), intent(in) :: stripe_count, thread_width, stripe_width, l_nev
          real(kind=rk)                :: a(:,:,:,:)
          integer(kind=ik), intent(in) :: n, my_thread
          real(kind=rk), intent(in)    :: row(:)
          integer(kind=ik)             :: i, noff, nl

#ifdef HAVE_DETAILED_TIMINGS
          call timer%start("unpack_row_real_cpu_openmp")
#endif
          do i=1,stripe_count
            noff = (my_thread-1)*thread_width + (i-1)*stripe_width
            nl   = min(stripe_width, my_thread*thread_width-noff, l_nev-noff)
            if(nl<=0) exit
            a(1:nl,n,i,my_thread) = row(noff+1:noff+nl)
          enddo

#ifdef HAVE_DETAILED_TIMINGS
          call timer%stop("unpack_row_real_cpu_openmp")
#endif

        end subroutine unpack_row_real_cpu_openmp

#else /* WITH_OPENMP */
        subroutine unpack_row_real_cpu(a, row, n, stripe_count, stripe_width, last_stripe_width)
#ifdef HAVE_DETAILED_TIMINGS
          use timings
#endif
         use precision
         implicit none

         integer(kind=ik), intent(in) :: n, stripe_count, stripe_width, last_stripe_width
         real(kind=rk)                :: row(:)
         real(kind=rk)                :: a(:,:,:)
         integer(kind=ik)             :: i, noff, nl

#ifdef HAVE_DETAILED_TIMINGS
          call timer%start("unpack_row_real_cpu")
#endif

          do i=1,stripe_count
            nl = merge(stripe_width, last_stripe_width, i<stripe_count)
            noff = (i-1)*stripe_width
            a(1:nl,n,i) = row(noff+1:noff+nl)
          enddo

#ifdef HAVE_DETAILED_TIMINGS
          call timer%stop("unpack_row_real_cpu")
#endif
        end subroutine unpack_row_real_cpu
#endif /* WITH_OPENMP */

end module
