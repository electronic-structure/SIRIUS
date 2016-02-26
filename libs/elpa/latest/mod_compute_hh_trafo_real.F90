module compute_hh_trafo_real
#include "config-f90.h"
  use elpa_mpi
  implicit none

#ifdef WITH_OPENMP
  public compute_hh_trafo_real_cpu_openmp
#else
  public compute_hh_trafo_real_cpu
#endif

  contains

#ifdef WITH_OPENMP
       subroutine compute_hh_trafo_real_cpu_openmp(a, stripe_width, a_dim2, stripe_count, max_threads, l_nev,         &
                                                   a_off, nbw, max_blk_size, bcast_buffer, kernel_flops, kernel_time, &
                                                   off, ncols, istripe,                                               &
                                                   my_thread, thread_width,  THIS_REAL_ELPA_KERNEL)
#else
       subroutine compute_hh_trafo_real_cpu       (a, stripe_width, a_dim2, stripe_count,                              &
                                                   a_off, nbw, max_blk_size, bcast_buffer,  kernel_flops, kernel_time, &
                                                   off, ncols, istripe, last_stripe_width,                             &
                                                   THIS_REAL_ELPA_KERNEL)
#endif


         use precision
         use elpa2_utilities
         use single_hh_trafo_real
#if defined(WITH_REAL_GENERIC_SIMPLE_KERNEL)
         use real_generic_simple_kernel, only : double_hh_trafo_generic_simple
#endif

#if defined(WITH_REAL_GENERIC_KERNEL) && !(defined(DESPERATELY_WANT_ASSUMED_SIZE))
        use real_generic_kernel, only : double_hh_trafo_generic
#endif

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
         real(kind=rk), intent(inout) :: kernel_time
         integer(kind=lik)            :: kernel_flops
         integer(kind=ik), intent(in) :: nbw, max_blk_size
         real(kind=rk)                :: bcast_buffer(nbw,max_blk_size)
         integer(kind=ik), intent(in) :: a_off

         integer(kind=ik), intent(in) :: stripe_width,a_dim2,stripe_count

#ifndef WITH_OPENMP
         integer(kind=ik), intent(in) :: last_stripe_width
         real(kind=rk)                :: a(stripe_width,a_dim2,stripe_count)
#else
         integer(kind=ik), intent(in) :: max_threads, l_nev, thread_width
         real(kind=rk)                :: a(stripe_width,a_dim2,stripe_count,max_threads)
#endif
         integer(kind=ik), intent(in) :: THIS_REAL_ELPA_KERNEL

         ! Private variables in OMP regions (my_thread) should better be in the argument list!
         integer(kind=ik)             :: off, ncols, istripe
#ifdef WITH_OPENMP
         integer(kind=ik)             :: my_thread, noff
#endif
         integer(kind=ik)             :: j, nl, jj, jjj
         real(kind=rk)                :: w(nbw,6), ttt

#ifdef HAVE_DETAILED_TIMINGS
#ifdef WITH_OPENMP
         call timer%start("compute_hh_trafo_real_cpu_openmp")
#else
         call timer%start("compute_hh_trafo_real_cpu")
#endif
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
           if (nl<=0) then
#ifdef HAVE_DETAILED_TIMINGS
#ifdef WITH_OPENMP
             call timer%stop("compute_hh_trafo_real_cpu_openmp")
#else
             call timer%stop("compute_hh_trafo_real_cpu")
#endif
#endif
             return
           endif
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
#ifdef DESPERATELY_WANT_ASSUMED_SIZE
               call double_hh_trafo_generic(a(1,j+off+a_off-1,istripe,my_thread), w, &
                                            nbw, nl, stripe_width, nbw)

#else
               call double_hh_trafo_generic(a(1:stripe_width,j+off+a_off-1:j+off+a_off+nbw-1, &
                                              istripe,my_thread), w(1:nbw,1:6), &
                                              nbw, nl, stripe_width, nbw)
#endif

#else /* WITH_OPENMP */

#ifdef DESPERATELY_WANT_ASSUMED_SIZE
               call double_hh_trafo_generic(a(1,j+off+a_off-1,istripe),w, &
                                            nbw, nl, stripe_width, nbw)

#else
               call double_hh_trafo_generic(a(1:stripe_width,j+off+a_off-1:j+off+a_off+nbw-1,istripe),w(1:nbw,1:6), &
                                            nbw, nl, stripe_width, nbw)
#endif
#endif /* WITH_OPENMP */

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
#ifdef DESPERATELY_WANT_ASSUMED_SIZE
               call double_hh_trafo_generic_simple(a(1,j+off+a_off-1,istripe,my_thread), &
                                                     w, nbw, nl, stripe_width, nbw)
#else
               call double_hh_trafo_generic_simple(a(1:stripe_width,j+off+a_off-1:j+off+a_off-1+nbw,istripe,my_thread), &
                                                     w, nbw, nl, stripe_width, nbw)

#endif

#else /* WITH_OPENMP */
#ifdef DESPERATELY_WANT_ASSUMED_SIZE
               call double_hh_trafo_generic_simple(a(1,j+off+a_off-1,istripe), &
                                                     w, nbw, nl, stripe_width, nbw)
#else
               call double_hh_trafo_generic_simple(a(1:stripe_width,j+off+a_off-1:j+off+a_off-1+nbw,istripe), &
                                                     w, nbw, nl, stripe_width, nbw)

#endif

#endif /* WITH_OPENMP */

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
           if (j==1) call single_hh_trafo_real_cpu_openmp(a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1,istripe,my_thread), &
                                      bcast_buffer(1:nbw,off+1), nbw, nl,     &
                                      stripe_width)
#else
           if (j==1) call single_hh_trafo_real_cpu(a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1,istripe),           &
                                      bcast_buffer(1:nbw,off+1), nbw, nl,     &
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
           if (jj==1) call single_hh_trafo_real_cpu_openmp(a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1,istripe,my_thread), &
                                          bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#else
           if (jj==1) call single_hh_trafo_real_cpu(a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1,istripe), &
                                          bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
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
           if (jjj==1) call single_hh_trafo_real_cpu_openmp(a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1,istripe,my_thread), &
                                           bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#else
           if (jjj==1) call single_hh_trafo_real_cpu(a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1,istripe), &
                                           bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
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
#ifdef WITH_OPENMP
         call timer%stop("compute_hh_trafo_real_cpu_openmp")
#else
         call timer%stop("compute_hh_trafo_real_cpu")
#endif
#endif

#ifdef WITH_OPENMP
       end subroutine compute_hh_trafo_real_cpu_openmp
#else
       end subroutine compute_hh_trafo_real_cpu
#endif

end module
