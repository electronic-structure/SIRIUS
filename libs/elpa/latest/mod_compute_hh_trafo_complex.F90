module compute_hh_trafo_complex
#include "config-f90.h"
  use elpa_mpi
  implicit none

#ifdef WITH_OPENMP
  public compute_hh_trafo_complex_cpu_openmp
#else
  public compute_hh_trafo_complex_cpu
#endif


  contains

#ifdef WITH_OPENMP
         subroutine compute_hh_trafo_complex_cpu_openmp(a, stripe_width, a_dim2, stripe_count, max_threads, l_nev,         &
                                                        a_off, nbw, max_blk_size, bcast_buffer, kernel_flops, kernel_time, &
                                                        off, ncols, istripe,                                               &
                                                        my_thread, thread_width, THIS_COMPLEX_ELPA_KERNEL)
#else
         subroutine compute_hh_trafo_complex_cpu       (a, stripe_width, a_dim2, stripe_count,                             &
                                                        a_off, nbw, max_blk_size, bcast_buffer, kernel_flops, kernel_time, &
                                                        off, ncols, istripe, last_stripe_width,                            &
                                                        THIS_COMPLEX_ELPA_KERNEL)
#endif
           use precision
           use elpa2_utilities
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
           real(kind=rk), intent(inout) :: kernel_time
           integer(kind=lik)            :: kernel_flops
           integer(kind=ik), intent(in) :: nbw, max_blk_size
           complex(kind=ck)             :: bcast_buffer(nbw,max_blk_size)
           integer(kind=ik), intent(in) :: a_off

           integer(kind=ik), intent(in) :: stripe_width, a_dim2, stripe_count
#ifndef WITH_OPENMP
           integer(kind=ik), intent(in) :: last_stripe_width
           complex(kind=ck)             :: a(stripe_width,a_dim2,stripe_count)
#else
           integer(kind=ik), intent(in) :: max_threads, l_nev, thread_width
           complex(kind=ck)             :: a(stripe_width,a_dim2,stripe_count,max_threads)
#endif
           integer(kind=ik), intent(in) :: THIS_COMPLEX_ELPA_KERNEL

           ! Private variables in OMP regions (my_thread) should better be in the argument list!

           integer(kind=ik)             :: off, ncols, istripe, j, nl, jj
#ifdef WITH_OPENMP
           integer(kind=ik)             :: my_thread, noff
#endif
           real(kind=rk)                :: ttt

           !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
           !        Currently (on Sandy Bridge), single is faster than double
           !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

           complex(kind=ck)             :: w(nbw,2)

#ifdef HAVE_DETAILED_TIMINGS
#ifdef WITH_OPENMP
          call timer%stop("compute_hh_trafo_complex_cpu_openmp")
#else
          call timer%stop("compute_hh_trafo_complex_cpu")
#endif
#endif

#ifdef WITH_OPENMP
           if (istripe<stripe_count) then
             nl = stripe_width
           else
             noff = (my_thread-1)*thread_width + (istripe-1)*stripe_width
             nl = min(my_thread*thread_width-noff, l_nev-noff)
             if(nl<=0) then
#ifdef HAVE_DETAILED_TIMINGS
               call timer%stop("compute_hh_trafo_complex_cpu_openmp")
#endif
               return
             endif
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
#ifdef DESPERATELY_WANT_ASSUMED_SIZE
               call single_hh_trafo_complex_generic_simple(a(1,j+off+a_off,istripe,my_thread), &
                                                          bcast_buffer(1,j+off),nbw,nl,stripe_width)
#else
               call single_hh_trafo_complex_generic_simple(a(1:stripe_width,j+off+a_off:j+off+a_off+nbw-1,istripe,my_thread), &
                                                           bcast_buffer(1:nbw,j+off),nbw,nl,stripe_width)
#endif

#else /* WITH_OPENMP */
#ifdef DESPERATELY_WANT_ASSUMED_SIZE
               call single_hh_trafo_complex_generic_simple(a(1,j+off+a_off,istripe), &
                                                          bcast_buffer(1,j+off),nbw,nl,stripe_width)
#else
               call single_hh_trafo_complex_generic_simple(a(1:stripe_width,j+off+a_off:j+off+a_off+nbw-1,istripe), &
                                                          bcast_buffer(1:nbw,j+off),nbw,nl,stripe_width)
#endif

#endif /* WITH_OPENMP */
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
#ifdef DESPERATELY_WANT_ASSUMED_SIZE

              call single_hh_trafo_complex_generic(a(1,j+off+a_off,istripe,my_thread), &
                                                   bcast_buffer(1,j+off),nbw,nl,stripe_width)
#else
              call single_hh_trafo_complex_generic(a(1:stripe_width,j+off+a_off:j+off+a_off+nbw-1,istripe,my_thread), &
                                                   bcast_buffer(1:nbw,j+off),nbw,nl,stripe_width)
#endif

#else /* WITH_OPENMP */
#ifdef DESPERATELY_WANT_ASSUMED_SIZE
              call single_hh_trafo_complex_generic(a(1,j+off+a_off,istripe), &
                                                   bcast_buffer(1,j+off),nbw,nl,stripe_width)
#else
              call single_hh_trafo_complex_generic(a(1:stripe_width,j+off+a_off:j+off+a_off+nbw-1,istripe), &
                                                   bcast_buffer(1:nbw,j+off),nbw,nl,stripe_width)
#endif
#endif /* WITH_OPENMP */

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
#ifdef WITH_OPENMP
          call timer%stop("compute_hh_trafo_complex_cpu_openmp")
#else
          call timer%stop("compute_hh_trafo_complex_cpu")
#endif
#endif

#ifdef WITH_OPENMP
        end subroutine compute_hh_trafo_complex_cpu_openmp
#else
        end subroutine compute_hh_trafo_complex_cpu

#endif

end module
