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
#include "elpa_kernel_constants.h"

module ELPA2_utilities
  use ELPA_utilities
  implicit none

  PRIVATE ! By default, all routines contained are private

  ! The following routines are public:

  public :: get_actual_real_kernel_name, get_actual_complex_kernel_name
  public :: REAL_ELPA_KERNEL_GENERIC, REAL_ELPA_KERNEL_GENERIC_SIMPLE, &
            REAL_ELPA_KERNEL_BGP, REAL_ELPA_KERNEL_BGQ,                &
            REAL_ELPA_KERNEL_SSE, REAL_ELPA_KERNEL_SSE_BLOCK2,         &
            REAL_ELPA_KERNEL_SSE_BLOCK4, REAL_ELPA_KERNEL_SSE_BLOCK6,  &
            REAL_ELPA_KERNEL_AVX_BLOCK2,                               &
            REAL_ELPA_KERNEL_AVX_BLOCK4, REAL_ELPA_KERNEL_AVX_BLOCK6,  &
            REAL_ELPA_KERNEL_AVX2_BLOCK2,                              &
            REAL_ELPA_KERNEL_AVX2_BLOCK4, REAL_ELPA_KERNEL_AVX2_BLOCK6,&
            DEFAULT_REAL_ELPA_KERNEL

  public :: COMPLEX_ELPA_KERNEL_GENERIC, COMPLEX_ELPA_KERNEL_GENERIC_SIMPLE, &
            COMPLEX_ELPA_KERNEL_BGP, COMPLEX_ELPA_KERNEL_BGQ,                &
            COMPLEX_ELPA_KERNEL_SSE, COMPLEX_ELPA_KERNEL_SSE_BLOCK1,         &
            COMPLEX_ELPA_KERNEL_SSE_BLOCK2,                                  &
            COMPLEX_ELPA_KERNEL_AVX_BLOCK1,COMPLEX_ELPA_KERNEL_AVX_BLOCK2,   &
            COMPLEX_ELPA_KERNEL_AVX2_BLOCK1,COMPLEX_ELPA_KERNEL_AVX2_BLOCK2, &
            DEFAULT_COMPLEX_ELPA_KERNEL

  public :: REAL_ELPA_KERNEL_NAMES, COMPLEX_ELPA_KERNEL_NAMES

  public :: get_actual_complex_kernel, get_actual_real_kernel

  public :: check_allowed_complex_kernels, check_allowed_real_kernels

  public :: AVAILABLE_COMPLEX_ELPA_KERNELS, AVAILABLE_REAL_ELPA_KERNELS

  public :: print_available_real_kernels, print_available_complex_kernels
  public :: query_available_real_kernels, query_available_complex_kernels

  public :: qr_decomposition_via_environment_variable

  integer, parameter :: number_of_real_kernels           = ELPA2_NUMBER_OF_REAL_KERNELS
  integer, parameter :: REAL_ELPA_KERNEL_GENERIC         = ELPA2_REAL_KERNEL_GENERIC
  integer, parameter :: REAL_ELPA_KERNEL_GENERIC_SIMPLE  = ELPA2_REAL_KERNEL_GENERIC_SIMPLE
  integer, parameter :: REAL_ELPA_KERNEL_BGP             = ELPA2_REAL_KERNEL_BGP
  integer, parameter :: REAL_ELPA_KERNEL_BGQ             = ELPA2_REAL_KERNEL_BGQ
  integer, parameter :: REAL_ELPA_KERNEL_SSE             = ELPA2_REAL_KERNEL_SSE
  integer, parameter :: REAL_ELPA_KERNEL_SSE_BLOCK2      = ELPA2_REAL_KERNEL_SSE_BLOCK2
  integer, parameter :: REAL_ELPA_KERNEL_SSE_BLOCK4      = ELPA2_REAL_KERNEL_SSE_BLOCK4
  integer, parameter :: REAL_ELPA_KERNEL_SSE_BLOCK6      = ELPA2_REAL_KERNEL_SSE_BLOCK6
  integer, parameter :: REAL_ELPA_KERNEL_AVX_BLOCK2      = ELPA2_REAL_KERNEL_AVX_BLOCK2
  integer, parameter :: REAL_ELPA_KERNEL_AVX_BLOCK4      = ELPA2_REAL_KERNEL_AVX_BLOCK4
  integer, parameter :: REAL_ELPA_KERNEL_AVX_BLOCK6      = ELPA2_REAL_KERNEL_AVX_BLOCK6
  integer, parameter :: REAL_ELPA_KERNEL_AVX2_BLOCK2     = ELPA2_REAL_KERNEL_AVX2_BLOCK2
  integer, parameter :: REAL_ELPA_KERNEL_AVX2_BLOCK4     = ELPA2_REAL_KERNEL_AVX2_BLOCK4
  integer, parameter :: REAL_ELPA_KERNEL_AVX2_BLOCK6     = ELPA2_REAL_KERNEL_AVX2_BLOCK6

#if defined(WITH_REAL_AVX_BLOCK2_KERNEL)

#ifndef WITH_ONE_SPECIFIC_REAL_KERNEL
  integer, parameter :: DEFAULT_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_GENERIC
#else /* WITH_ONE_SPECIFIC_REAL_KERNEL */

#ifdef WITH_REAL_GENERIC_KERNEL
  integer, parameter :: DEFAULT_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_GENERIC
#endif
#ifdef WITH_REAL_GENERIC_SIMPLE_KERNEL
  integer, parameter :: DEFAULT_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_GENERIC_SIMPLE
#endif
#ifdef WITH_REAL_SSE_ASSEMBLY_KERNEL
  integer, parameter :: DEFAULT_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_SSE
#endif
#if defined(WITH_REAL_SSE_BLOCK2_KERNEL) || defined(WITH_REAL_SSE_BLOCK4_KERNEL) || defined(WITH_REAL_SSE_BLOCK6_KERNEL)

#ifdef WITH_REAL_SSE_BLOCK6_KERNEL
  integer, parameter :: DEFAULT_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_SSE_BLOCK6
#else

#ifdef WITH_REAL_SSE_BLOCK4_KERNEL
  integer, parameter :: DEFAULT_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_SSE_BLOCK4
#else
#ifdef WITH_REAL_SSE_BLOCK2_KERNEL
  integer, parameter :: DEFAULT_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_SSE_BLOCK2
#endif
#endif
#endif
#endif /*  #if defined(WITH_REAL_SSE_BLOCK2_KERNEL) || defined(WITH_REAL_SSE_BLOCK4_KERNEL) || defined(WITH_REAL_SSE_BLOCK6_KERNEL) */

#if defined(WITH_REAL_AVX_BLOCK2_KERNEL) || defined(WITH_REAL_AVX_BLOCK4_KERNEL) || defined(WITH_REAL_AVX_BLOCK6_KERNEL)
#ifdef WITH_REAL_AVX_BLOCK6_KERNEL
  integer, parameter :: DEFAULT_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_AVX_BLOCK6
#else
#ifdef WITH_REAL_AVX_BLOCK4_KERNEL
  integer, parameter :: DEFAULT_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_AVX_BLOCK4
#else
#ifdef WITH_REAL_AVX_BLOCK2_KERNEL
  integer, parameter :: DEFAULT_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_AVX_BLOCK2
#endif
#endif
#endif
#endif /*  #if defined(WITH_REAL_AVX_BLOCK2_KERNEL) || defined(WITH_REAL_AVX_BLOCK4_KERNEL) || defined(WITH_REAL_AVX_BLOCK6_KERNEL) */

#if defined(WITH_REAL_AVX2_BLOCK2_KERNEL) || defined(WITH_REAL_AVX2_BLOCK4_KERNEL) || defined(WITH_REAL_AVX2_BLOCK6_KERNEL)
#ifdef WITH_REAL_AVX2_BLOCK6_KERNEL
  integer, parameter :: DEFAULT_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_AVX2_BLOCK6
#else
#ifdef WITH_REAL_AVX2_BLOCK4_KERNEL
  integer, parameter :: DEFAULT_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_AVX2_BLOCK4
#else
#ifdef WITH_REAL_AVX2_BLOCK2_KERNEL
  integer, parameter :: DEFAULT_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_AVX2_BLOCK2
#endif
#endif
#endif
#endif /*  #if defined(WITH_REAL_AVX2_BLOCK2_KERNEL) || defined(WITH_REAL_AVX2_BLOCK4_KERNEL) || defined(WITH_REAL_AVX2_BLOCK6_KERNEL) */

#ifdef WITH_REAL_BGP_KERNEL
  integer, parameter :: DEFAULT_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_AVX_BGP
#endif
#ifdef WITH_REAL_BGQ_KERNEL
  integer, parameter :: DEFAULT_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_AVX_BGQ
#endif

#endif /* WITH_ONE_SPECIFIC_REAL_KERNEL */

#else /* WITH_REAL_AVX_BLOCK2_KERNEL */

#ifndef WITH_ONE_SPECIFIC_REAL_KERNEL
  integer, parameter :: DEFAULT_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_GENERIC
#else /* WITH_ONE_SPECIFIC_REAL_KERNEL */

#ifdef WITH_REAL_GENERIC_KERNEL
  integer, parameter :: DEFAULT_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_GENERIC
#endif
#ifdef WITH_REAL_GENERIC_SIMPLE_KERNEL
  integer, parameter :: DEFAULT_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_GENERIC_SIMPLE
#endif
#ifdef WITH_REAL_SSE_ASSEMBLY_KERNEL
  integer, parameter :: DEFAULT_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_SSE
#endif

#if defined(WITH_REAL_SSE_BLOCK2_KERNEL) || defined(WITH_REAL_SSE_BLOCK4_KERNEL) || defined(WITH_REAL_SSE_BLOCK6_KERNEL)
#ifdef WITH_REAL_SSE_BLOCK6_KERNEL
  integer, parameter :: DEFAULT_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_SSE_BLOCK6
#else
#ifdef WITH_REAL_SSE_BLOCK4_KERNEL
  integer, parameter :: DEFAULT_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_SSE_BLOCK4
#else
#ifdef WITH_REAL_SSE_BLOCK2_KERNEL
  integer, parameter :: DEFAULT_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_SSE_BLOCK2
#endif
#endif
#endif
#endif /*  #if defined(WITH_REAL_SSE_BLOCK2_KERNEL) || defined(WITH_REAL_SSE_BLOCK4_KERNEL) || defined(WITH_REAL_SSE_BLOCK6_KERNEL) */

#if defined(WITH_REAL_AVX_BLOCK2_KERNEL) || defined(WITH_REAL_AVX_BLOCK4_KERNEL) || defined(WITH_REAL_AVX_BLOCK6_KERNEL)
#ifdef WITH_REAL_AVX_BLOCK6_KERNEL
  integer, parameter :: DEFAULT_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_AVX_BLOCK6
#else
#ifdef WITH_REAL_AVX_BLOCK4_KERNEL
  integer, parameter :: DEFAULT_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_AVX_BLOCK4
#else
#ifdef WITH_REAL_AVX_BLOCK2_KERNEL
  integer, parameter :: DEFAULT_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_AVX_BLOCK2
#endif
#endif
#endif
#endif /*  #if defined(WITH_REAL_AVX_BLOCK2_KERNEL) || defined(WITH_REAL_AVX_BLOCK4_KERNEL) || defined(WITH_REAL_AVX_BLOCK6_KERNEL) */

#if defined(WITH_REAL_AVX2_BLOCK2_KERNEL) || defined(WITH_REAL_AVX2_BLOCK4_KERNEL) || defined(WITH_REAL_AVX2_BLOCK6_KERNEL)
#ifdef WITH_REAL_AVX2_BLOCK6_KERNEL
  integer, parameter :: DEFAULT_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_AVX2_BLOCK6
#else
#ifdef WITH_REAL_AVX2_BLOCK4_KERNEL
  integer, parameter :: DEFAULT_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_AVX2_BLOCK4
#else
#ifdef WITH_REAL_AVX2_BLOCK2_KERNEL
  integer, parameter :: DEFAULT_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_AVX2_BLOCK2
#endif
#endif
#endif
#endif /*  #if defined(WITH_REAL_AVX2_BLOCK2_KERNEL) || defined(WITH_REAL_AVX2_BLOCK4_KERNEL) || defined(WITH_REAL_AVX2_BLOCK6_KERNEL) */


#ifdef WITH_REAL_BGP_KERNEL
  integer, parameter :: DEFAULT_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_AVX_BGP
#endif
#ifdef WITH_REAL_BGQ_KERNEL
  integer, parameter :: DEFAULT_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_AVX_BGQ
#endif

#endif  /* WITH_ONE_SPECIFIC_REAL_KERNEL */

#endif /* WITH_REAL_AVX_BLOCK2_KERNEL */

  character(35), parameter, dimension(number_of_real_kernels) :: &
  REAL_ELPA_KERNEL_NAMES =    (/"REAL_ELPA_KERNEL_GENERIC         ", &
                                "REAL_ELPA_KERNEL_GENERIC_SIMPLE  ", &
                                "REAL_ELPA_KERNEL_BGP             ", &
                                "REAL_ELPA_KERNEL_BGQ             ", &
                                "REAL_ELPA_KERNEL_SSE             ", &
                                "REAL_ELPA_KERNEL_SSE_BLOCK2      ", &
                                "REAL_ELPA_KERNEL_SSE_BLOCK4      ", &
                                "REAL_ELPA_KERNEL_SSE_BLOCK6      ", &
                                "REAL_ELPA_KERNEL_AVX_BLOCK2      ", &
                                "REAL_ELPA_KERNEL_AVX_BLOCK4      ", &
                                "REAL_ELPA_KERNEL_AVX_BLOCK6      ", &
                                "REAL_ELPA_KERNEL_AVX2_BLOCK2     ", &
                                "REAL_ELPA_KERNEL_AVX2_BLOCK4     ", &
                                "REAL_ELPA_KERNEL_AVX2_BLOCK6     "/)

  integer, parameter :: number_of_complex_kernels           = ELPA2_NUMBER_OF_COMPLEX_KERNELS
  integer, parameter :: COMPLEX_ELPA_KERNEL_GENERIC         = ELPA2_COMPLEX_KERNEL_GENERIC
  integer, parameter :: COMPLEX_ELPA_KERNEL_GENERIC_SIMPLE  = ELPA2_COMPLEX_KERNEL_GENERIC_SIMPLE
  integer, parameter :: COMPLEX_ELPA_KERNEL_BGP             = ELPA2_COMPLEX_KERNEL_BGP
  integer, parameter :: COMPLEX_ELPA_KERNEL_BGQ             = ELPA2_COMPLEX_KERNEL_BGQ
  integer, parameter :: COMPLEX_ELPA_KERNEL_SSE             = ELPA2_COMPLEX_KERNEL_SSE
  integer, parameter :: COMPLEX_ELPA_KERNEL_SSE_BLOCK1      = ELPA2_COMPLEX_KERNEL_SSE_BLOCK1
  integer, parameter :: COMPLEX_ELPA_KERNEL_SSE_BLOCK2      = ELPA2_COMPLEX_KERNEL_SSE_BLOCK2
  integer, parameter :: COMPLEX_ELPA_KERNEL_AVX_BLOCK1      = ELPA2_COMPLEX_KERNEL_AVX_BLOCK1
  integer, parameter :: COMPLEX_ELPA_KERNEL_AVX_BLOCK2      = ELPA2_COMPLEX_KERNEL_AVX_BLOCK2
  integer, parameter :: COMPLEX_ELPA_KERNEL_AVX2_BLOCK1     = ELPA2_COMPLEX_KERNEL_AVX2_BLOCK1
  integer, parameter :: COMPLEX_ELPA_KERNEL_AVX2_BLOCK2     = ELPA2_COMPLEX_KERNEL_AVX2_BLOCK2

#if defined(WITH_COMPLEX_AVX_BLOCK1_KERNEL)

#ifndef WITH_ONE_SPECIFIC_COMPLEX_KERNEL
  integer, parameter :: DEFAULT_COMPLEX_ELPA_KERNEL = COMPLEX_ELPA_KERNEL_GENERIC
#else /* WITH_ONE_SPECIFIC_COMPLEX_KERNEL */

! go through all kernels and set them
#ifdef WITH_COMPLEX_GENERIC_KERNEL
  integer, parameter :: DEFAULT_COMPLEX_ELPA_KERNEL = COMPLEX_ELPA_KERNEL_GENERIC
#endif
#ifdef WITH_COMPLEX_GENERIC_SIMPLE_KERNEL
  integer, parameter :: DEFAULT_COMPLEX_ELPA_KERNEL = COMPLEX_ELPA_KERNEL_GENERIC_SIMPLE
#endif
#ifdef WITH_COMPLEX_SSE_ASSEMBLY_KERNEL
  integer, parameter :: DEFAULT_COMPLEX_ELPA_KERNEL = COMPLEX_ELPA_KERNEL_SSE
#endif

#if defined(WITH_COMPLEX_SSE_BLOCK1_KERNEL) || defined(WITH_COMPLEX_SSE_BLOCK2_KERNEL)
#ifdef WITH_COMPLEX_SSE_BLOCK2_KERNEL
  integer, parameter :: DEFAULT_COMPLEX_ELPA_KERNEL = COMPLEX_ELPA_KERNEL_SSE_BLOCK2
#else
#ifdef WITH_COMPLEX_SSE_BLOCK1_KERNEL
  integer, parameter :: DEFAULT_COMPLEX_ELPA_KERNEL = COMPLEX_ELPA_KERNEL_SSE_BLOCK1
#endif
#endif
#endif /* defined(WITH_COMPLEXL_SSE_BLOCK1_KERNEL) || defined(WITH_COMPLEX_SSE_BLOCK2_KERNEL) */

#if defined(WITH_COMPLEX_AVX_BLOCK1_KERNEL) || defined(WITH_COMPLEX_AVX_BLOCK2_KERNEL)
#ifdef WITH_COMPLEX_AVX_BLOCK2_KERNEL
  integer, parameter :: DEFAULT_COMPLEX_ELPA_KERNEL = COMPLEX_ELPA_KERNEL_AVX_BLOCK2
#else
#ifdef WITH_COMPLEX_AVX_BLOCK1_KERNEL
  integer, parameter :: DEFAULT_COMPLEX_ELPA_KERNEL = COMPLEX_ELPA_KERNEL_AVX_BLOCK1
#endif
#endif
#endif /* defined(WITH_COMPLEX_AVX_BLOCK1_KERNEL) || defined(WITH_COMPLEX_AVX_BLOCK2_KERNEL) */

#if defined(WITH_COMPLEX_AVX2_BLOCK1_KERNEL) || defined(WITH_COMPLEX_AVX2_BLOCK2_KERNEL)
#ifdef WITH_COMPLEX_AVX2_BLOCK2_KERNEL
  integer, parameter :: DEFAULT_COMPLEX_ELPA_KERNEL = COMPLEX_ELPA_KERNEL_AVX2_BLOCK2
#else
#ifdef WITH_COMPLEX_AVX2_BLOCK1_KERNEL
  integer, parameter :: DEFAULT_COMPLEX_ELPA_KERNEL = COMPLEX_ELPA_KERNEL_AVX2_BLOCK1
#endif
#endif
#endif /* defined(WITH_COMPLEX_AVX2_BLOCK1_KERNEL) || defined(WITH_COMPLEX_AVX2_BLOCK2_KERNEL) */
#endif /* WITH_ONE_SPECIFIC_COMPLEX_KERNEL */

#else /* WITH_COMPLEX_AVX_BLOCK1_KERNEL */

#ifndef WITH_ONE_SPECIFIC_COMPLEX_KERNEL
  integer, parameter :: DEFAULT_COMPLEX_ELPA_KERNEL = COMPLEX_ELPA_KERNEL_GENERIC

#else /* WITH_ONE_SPECIFIC_COMPLEX_KERNEL */

! go through all kernels and set them
#ifdef WITH_COMPLEX_GENERIC_KERNEL
  integer, parameter :: DEFAULT_COMPLEX_ELPA_KERNEL = COMPLEX_ELPA_KERNEL_GENERIC
#endif
#ifdef WITH_COMPLEX_GENERIC_SIMPLE_KERNEL
  integer, parameter :: DEFAULT_COMPLEX_ELPA_KERNEL = COMPLEX_ELPA_KERNEL_GENERIC_SIMPLE
#endif
#ifdef WITH_COMPLEX_SSE_ASSEMBLY_KERNEL
  integer, parameter :: DEFAULT_COMPLEX_ELPA_KERNEL = COMPLEX_ELPA_KERNEL_SSE
#endif

#if defined(WITH_COMPLEX_SSE_BLOCK1_KERNEL) || defined(WITH_COMPLEX_SSE_BLOCK2_KERNEL)
#ifdef WITH_COMPLEX_SSE_BLOCK2_KERNEL
  integer, parameter :: DEFAULT_COMPLEX_ELPA_KERNEL = COMPLEX_ELPA_KERNEL_SSE_BLOCK2
#else
#ifdef WITH_COMPLEX_SSE_BLOCK1_KERNEL
  integer, parameter :: DEFAULT_COMPLEX_ELPA_KERNEL = COMPLEX_ELPA_KERNEL_SSE_BLOCK1
#endif
#endif
#endif /* defined(WITH_COMPLEXL_SSE_BLOCK1_KERNEL) || defined(WITH_COMPLEX_SSE_BLOCK2_KERNEL) */

#if defined(WITH_COMPLEX_AVX_BLOCK1_KERNEL) || defined(WITH_COMPLEX_AVX_BLOCK2_KERNEL)
#ifdef WITH_COMPLEX_AVX_BLOCK2_KERNEL
  integer, parameter :: DEFAULT_COMPLEX_ELPA_KERNEL = COMPLEX_ELPA_KERNEL_AVX_BLOCK2
#else
#ifdef WITH_COMPLEX_AVX_BLOCK1_KERNEL
  integer, parameter :: DEFAULT_COMPLEX_ELPA_KERNEL = COMPLEX_ELPA_KERNEL_AVX_BLOCK1
#endif
#endif
#endif /* defined(WITH_COMPLEX_AVX_BLOCK1_KERNEL) || defined(WITH_COMPLEX_AVX_BLOCK2_KERNEL) */

#if defined(WITH_COMPLEX_AVX2_BLOCK1_KERNEL) || defined(WITH_COMPLEX_AVX2_BLOCK2_KERNEL)
#ifdef WITH_COMPLEX_AVX2_BLOCK2_KERNEL
  integer, parameter :: DEFAULT_COMPLEX_ELPA_KERNEL = COMPLEX_ELPA_KERNEL_AVX2_BLOCK2
#else
#ifdef WITH_COMPLEX_AVX2_BLOCK1_KERNEL
  integer, parameter :: DEFAULT_COMPLEX_ELPA_KERNEL = COMPLEX_ELPA_KERNEL_AVX2_BLOCK1
#endif
#endif
#endif /* defined(WITH_COMPLEX_AVX2_BLOCK1_KERNEL) || defined(WITH_COMPLEX_AVX2_BLOCK2_KERNEL) */


#endif /* WITH_ONE_SPECIFIC_COMPLEX_KERNEL */

#endif /* WITH_COMPLEX_AVX_BLOCK1_KERNEL */

  character(35), parameter, dimension(number_of_complex_kernels) :: &
  COMPLEX_ELPA_KERNEL_NAMES = (/"COMPLEX_ELPA_KERNEL_GENERIC         ", &
                                "COMPLEX_ELPA_KERNEL_GENERIC_SIMPLE  ", &
                                "COMPLEX_ELPA_KERNEL_BGP             ", &
                                "COMPLEX_ELPA_KERNEL_BGQ             ", &
                                "COMPLEX_ELPA_KERNEL_SSE             ", &
                                "COMPLEX_ELPA_KERNEL_SSE_BLOCK1      ", &
                                "COMPLEX_ELPA_KERNEL_SSE_BLOCK2      ", &
                                "COMPLEX_ELPA_KERNEL_AVX_BLOCK1      ", &
                                "COMPLEX_ELPA_KERNEL_AVX_BLOCK2      ", &
                                "COMPLEX_ELPA_KERNEL_AVX2_BLOCK1     ", &
                                "COMPLEX_ELPA_KERNEL_AVX2_BLOCK2     "/)

  integer, parameter                                    ::             &
           AVAILABLE_REAL_ELPA_KERNELS(number_of_real_kernels) =       &
                                      (/                               &
#if WITH_REAL_GENERIC_KERNEL
                                        1                              &
#else
                                        0                              &
#endif
#if WITH_REAL_GENERIC_SIMPLE_KERNEL
                                          ,1                           &
#else
                                          ,0                           &
#endif
#if WITH_REAL_BGP_KERNEL
                                            ,1                         &
#else
                                            ,0                         &
#endif
#if WITH_REAL_BGQ_KERNEL
                                              ,1                       &
#else
                                              ,0                       &
#endif
#if WITH_REAL_SSE_ASSEMBLY_KERNEL
                                                ,1                     &
#else
                                                ,0                     &
#endif
#if WITH_REAL_SSE_BLOCK2_KERNEL
                                                  ,1                   &
#else
                                                  ,0                   &
#endif
#if WITH_REAL_SSE_BLOCK4_KERNEL
                                                    ,1                 &
#else
                                                    ,0                 &
#endif
#if WITH_REAL_SSE_BLOCK6_KERNEL
                                                      ,1               &
#else
                                                      ,0               &

#endif
#if WITH_REAL_AVX_BLOCK2_KERNEL
                                                        ,1             &
#else
                                                        ,0             &
#endif
#if WITH_REAL_AVX_BLOCK4_KERNEL
                                                          ,1           &
#else
                                                          ,0           &
#endif
#if WITH_REAL_AVX_BLOCK6_KERNEL
                                                            ,1         &
#else
                                                            ,0         &
#endif
#if WITH_REAL_AVX2_BLOCK2_KERNEL
                                                              ,1       &
#else
                                                              ,0       &
#endif
#if WITH_REAL_AVX2_BLOCK4_KERNEL
                                                               ,1      &
#else
                                                               ,0      &
#endif
#if WITH_REAL_AVX2_BLOCK6_KERNEL
                                                               ,1      &
#else
                                                               ,0      &
#endif


                                                       /)

  integer, parameter ::                                                   &
           AVAILABLE_COMPLEX_ELPA_KERNELS(number_of_complex_kernels) =    &
                                      (/                                  &
#if WITH_COMPLEX_GENERIC_KERNEL
                                        1                                 &
#else
                                        0                                 &
#endif
#if WITH_COMPLEX_GENERIC_SIMPLE_KERNEL
                                          ,1                              &
#else
                                          ,0                              &
#endif
#if WITH_COMPLEX_BGP_KERNEL
                                            ,1                            &
#else
                                            ,0                            &
#endif
#if WITH_COMPLEX_BGQ_KERNEL
                                              ,1                          &
#else
                                              ,0                          &
#endif
#if WITH_COMPLEX_SSE_ASSEMBLY_KERNEL
                                                ,1                        &
#else
                                                ,0                        &
#endif
#if WITH_COMPLEX_SSE_BLOCK1_KERNEL
                                                  ,1                      &
#else
                                                  ,0                      &
#endif
#if WITH_COMPLEX_SSE_BLOCK2_KERNEL
                                                    ,1                    &
#else
                                                    ,0                    &
#endif

#if WITH_COMPLEX_AVX_BLOCK1_KERNEL
                                                      ,1                  &
#else
                                                      ,0                  &
#endif
#if WITH_COMPLEX_AVX_BLOCK2_KERNEL
                                                        ,1                &
#else
                                                        ,0                &
#endif
#if WITH_COMPLEX_AVX2_BLOCK1_KERNEL
                                                         ,1               &
#else
                                                         ,0               &
#endif
#if WITH_COMPLEX_AVX2_BLOCK2_KERNEL
                                                           ,1             &
#else
                                                           ,0             &
#endif

                                                   /)

!******
  contains
    subroutine print_available_real_kernels
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#endif
      implicit none

      integer :: i

#ifdef HAVE_DETAILED_TIMINGS
      call timer%start("print_available_real_kernels")
#endif

      do i=1, number_of_real_kernels
        if (AVAILABLE_REAL_ELPA_KERNELS(i) .eq. 1) then
          write(*,*) REAL_ELPA_KERNEL_NAMES(i)
        endif
      enddo
      write(*,*) " "
      write(*,*) " At the moment the following kernel would be choosen:"
      write(*,*) get_actual_real_kernel_name()

#ifdef HAVE_DETAILED_TIMINGS
      call timer%stop("print_available_real_kernels")
#endif

    end subroutine print_available_real_kernels

    subroutine query_available_real_kernels
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#endif
      implicit none

      integer :: i

#ifdef HAVE_DETAILED_TIMINGS
      call timer%start("query_available_real_kernels")
#endif

      do i=1, number_of_real_kernels
        if (AVAILABLE_REAL_ELPA_KERNELS(i) .eq. 1) then
          write(error_unit,*) REAL_ELPA_KERNEL_NAMES(i)
        endif
      enddo
      write(error_unit,*) " "
      write(error_unit,*) " At the moment the following kernel would be choosen:"
      write(error_unit,*) get_actual_real_kernel_name()

#ifdef HAVE_DETAILED_TIMINGS
      call timer%stop("query_available_real_kernels")
#endif

    end subroutine query_available_real_kernels

    subroutine print_available_complex_kernels
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#endif

      implicit none

      integer :: i
#ifdef HAVE_DETAILED_TIMINGS
      call timer%start("print_available_complex_kernels")
#endif

      do i=1, number_of_complex_kernels
        if (AVAILABLE_COMPLEX_ELPA_KERNELS(i) .eq. 1) then
           write(*,*) COMPLEX_ELPA_KERNEL_NAMES(i)
        endif
      enddo
      write(*,*) " "
      write(*,*) " At the moment the following kernel would be choosen:"
      write(*,*) get_actual_complex_kernel_name()

#ifdef HAVE_DETAILED_TIMINGS
      call timer%stop("print_available_complex_kernels")
#endif

    end subroutine print_available_complex_kernels

    subroutine query_available_complex_kernels
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#endif

      implicit none

      integer :: i
#ifdef HAVE_DETAILED_TIMINGS
      call timer%start("query_available_complex_kernels")
#endif

      do i=1, number_of_complex_kernels
        if (AVAILABLE_COMPLEX_ELPA_KERNELS(i) .eq. 1) then
           write(error_unit,*) COMPLEX_ELPA_KERNEL_NAMES(i)
        endif
      enddo
      write(error_unit,*) " "
      write(error_unit,*) " At the moment the following kernel would be choosen:"
      write(error_unit,*) get_actual_complex_kernel_name()

#ifdef HAVE_DETAILED_TIMINGS
      call timer%stop("query_available_complex_kernels")
#endif

    end subroutine query_available_complex_kernels

    function get_actual_real_kernel() result(actual_kernel)
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#endif
      implicit none

      integer :: actual_kernel

#ifdef HAVE_DETAILED_TIMINGS
      call timer%start("get_actual_real_kernel")
#endif


      ! if kernel is not choosen via api
      ! check whether set by environment variable
      actual_kernel = real_kernel_via_environment_variable()

      if (actual_kernel .eq. 0) then
        ! if not then set default kernel
        actual_kernel = DEFAULT_REAL_ELPA_KERNEL
      endif

#ifdef HAVE_DETAILED_TIMINGS
      call timer%stop("get_actual_real_kernel")
#endif

    end function get_actual_real_kernel

    function get_actual_real_kernel_name() result(actual_kernel_name)
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#endif
      implicit none

      character(35) :: actual_kernel_name
      integer       :: actual_kernel

#ifdef HAVE_DETAILED_TIMINGS
      call timer%start("get_actual_real_kernel_name")
#endif

      actual_kernel = get_actual_real_kernel()
      actual_kernel_name = REAL_ELPA_KERNEL_NAMES(actual_kernel)

#ifdef HAVE_DETAILED_TIMINGS
      call timer%stop("get_actual_real_kernel_name")
#endif

    end function get_actual_real_kernel_name

    function get_actual_complex_kernel() result(actual_kernel)
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#endif
      implicit none
      integer :: actual_kernel

#ifdef HAVE_DETAILED_TIMINGS
      call timer%start("get_actual_complex_kernel")
#endif


     ! if kernel is not choosen via api
     ! check whether set by environment variable
     actual_kernel = complex_kernel_via_environment_variable()

     if (actual_kernel .eq. 0) then
       ! if not then set default kernel
       actual_kernel = DEFAULT_COMPLEX_ELPA_KERNEL
     endif

#ifdef HAVE_DETAILED_TIMINGS
     call timer%stop("get_actual_complex_kernel")
#endif

   end function get_actual_complex_kernel

   function get_actual_complex_kernel_name() result(actual_kernel_name)
#ifdef HAVE_DETAILED_TIMINGS
     use timings
#endif
     implicit none
     character(35) :: actual_kernel_name
     integer       :: actual_kernel

#ifdef HAVE_DETAILED_TIMINGS
     call timer%start("get_actual_complex_kernel_name")
#endif

     actual_kernel = get_actual_complex_kernel()
     actual_kernel_name = COMPLEX_ELPA_KERNEL_NAMES(actual_kernel)

#ifdef HAVE_DETAILED_TIMINGS
     call timer%stop("get_actual_complex_kernel_name")
#endif

   end function get_actual_complex_kernel_name

   function check_allowed_real_kernels(THIS_REAL_ELPA_KERNEL) result(err)
#ifdef HAVE_DETAILED_TIMINGS
     use timings
#endif
     implicit none
     integer, intent(in) :: THIS_REAL_ELPA_KERNEL

     logical             :: err

#ifdef HAVE_DETAILED_TIMINGS
     call timer%start("check_allowed_real_kernels")
#endif
     err = .false.

     if (AVAILABLE_REAL_ELPA_KERNELS(THIS_REAL_ELPA_KERNEL) .ne. 1) err=.true.

#ifdef HAVE_DETAILED_TIMINGS
     call timer%stop("check_allowed_real_kernels")
#endif

   end function check_allowed_real_kernels

   function check_allowed_complex_kernels(THIS_COMPLEX_ELPA_KERNEL) result(err)
#ifdef HAVE_DETAILED_TIMINGS
     use timings
#endif
     implicit none
     integer, intent(in) :: THIS_COMPLEX_ELPA_KERNEL

     logical             :: err
#ifdef HAVE_DETAILED_TIMINGS
     call timer%start("check_allowed_complex_kernels")
#endif
     err = .false.

     if (AVAILABLE_COMPLEX_ELPA_KERNELS(THIS_COMPLEX_ELPA_KERNEL) .ne. 1) err=.true.

#ifdef HAVE_DETAILED_TIMINGS
     call timer%stop("check_allowed_complex_kernels")
#endif

   end function check_allowed_complex_kernels

   function qr_decomposition_via_environment_variable(useQR) result(isSet)
#ifdef HAVE_DETAILED_TIMINGS
     use timings
#endif
     implicit none
     logical, intent(out) :: useQR
     logical              :: isSet
     CHARACTER(len=255)   :: ELPA_QR_DECOMPOSITION

#ifdef HAVE_DETAILED_TIMINGS
     call timer%start("qr_decomposition_via_environment_variable")
#endif

     isSet = .false.

#if defined(HAVE_ENVIRONMENT_CHECKING)
     call get_environment_variable("ELPA_QR_DECOMPOSITION",ELPA_QR_DECOMPOSITION)
#endif
     if (trim(ELPA_QR_DECOMPOSITION) .eq. "yes") then
       useQR = .true.
       isSet = .true.
     endif
     if (trim(ELPA_QR_DECOMPOSITION) .eq. "no") then
       useQR = .false.
       isSet = .true.
     endif

#ifdef HAVE_DETAILED_TIMINGS
     call timer%stop("qr_decomposition_via_environment_variable")
#endif

   end function qr_decomposition_via_environment_variable


   function real_kernel_via_environment_variable() result(kernel)
#ifdef HAVE_DETAILED_TIMINGS
     use timings
#endif
     implicit none
     integer :: kernel
     CHARACTER(len=255) :: REAL_KERNEL_ENVIRONMENT
     integer :: i

#ifdef HAVE_DETAILED_TIMINGS
     call timer%start("real_kernel_via_environment_variable")
#endif

#if defined(HAVE_ENVIRONMENT_CHECKING)
     call get_environment_variable("REAL_ELPA_KERNEL",REAL_KERNEL_ENVIRONMENT)
#endif
     do i=1,size(REAL_ELPA_KERNEL_NAMES(:))
       !     if (trim(dummy_char) .eq. trim(REAL_ELPA_KERNEL_NAMES(i))) then
       if (trim(REAL_KERNEL_ENVIRONMENT) .eq. trim(REAL_ELPA_KERNEL_NAMES(i))) then
         kernel = i
         exit
       else
         kernel = 0
       endif
     enddo

#ifdef HAVE_DETAILED_TIMINGS
     call timer%stop("real_kernel_via_environment_variable")
#endif

   end function real_kernel_via_environment_variable

   function complex_kernel_via_environment_variable() result(kernel)
#ifdef HAVE_DETAILED_TIMINGS
     use timings
#endif
     implicit none
     integer :: kernel

     CHARACTER(len=255) :: COMPLEX_KERNEL_ENVIRONMENT
     integer :: i

#ifdef HAVE_DETAILED_TIMINGS
     call timer%start("complex_kernel_via_environment_variable")
#endif

#if defined(HAVE_ENVIRONMENT_CHECKING)
     call get_environment_variable("COMPLEX_ELPA_KERNEL",COMPLEX_KERNEL_ENVIRONMENT)
#endif

     do i=1,size(COMPLEX_ELPA_KERNEL_NAMES(:))
       if (trim(COMPLEX_ELPA_KERNEL_NAMES(i)) .eq. trim(COMPLEX_KERNEL_ENVIRONMENT)) then
         kernel = i
         exit
       else
         kernel = 0
       endif
     enddo

#ifdef HAVE_DETAILED_TIMINGS
     call timer%stop("complex_kernel_via_environment_variable")
#endif

   end function
!-------------------------------------------------------------------------------

end module ELPA2_utilities
