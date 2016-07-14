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

!> \file print_available_elpa2_kernels.F90
!> \par
!> \brief Provide information which ELPA2 kernels are available on this system
!>
!> \details
!> It is possible to configure ELPA2 such, that different compute intensive
!> "ELPA2 kernels" can be choosen at runtime.
!> The service binary print_available_elpa2_kernels will query the library and tell
!> whether ELPA2 has been configured in this way, and if this is the case which kernels can be
!> choosen at runtime.
!> It will furthermore detail whether ELPA has been configured with OpenMP support
!>
!> Synopsis: print_available_elpa2_kernels
!>
!> \author A. Marek (MPCDF)
program print_available_elpa2_kernels

   use precision
   use ELPA1
   use ELPA2

   use elpa2_utilities

   implicit none

   integer(kind=ik) :: i

   print *, "This program will give information on the ELPA2 kernels, "
   print *, "which are available with this library and it will give "
   print *, "information if (and how) the kernels can be choosen at "
   print *, "runtime"
   print *
   print *
#ifdef WITH_OPENMP
   print *, " ELPA supports threads: yes"
#else
   print *, " ELPA supports threads: no"
#endif

   print *, "Information on ELPA2 real case: "
   print *, "=============================== "
#ifdef HAVE_ENVIRONMENT_CHECKING
   print *, " choice via environment variable: yes"
   print *, " environment variable name      : REAL_ELPA_KERNEL"
#else
   print *, " choice via environment variable: no"
#endif
   print *
   print *, " Available real kernels are: "
#ifdef HAVE_AVX2
   print *, " AVX kernels are optimized for FMA (AVX2)"
#endif
   call print_available_real_kernels()

   print *
   print *
   print *, "Information on ELPA2 complex case: "
   print *, "=============================== "
#ifdef HAVE_ENVIRONMENT_CHECKING
   print *, " choice via environment variable: yes"
   print *, " environment variable name      : COMPLEX_ELPA_KERNEL"
#else
   print *,  " choice via environment variable: no"
#endif
   print *
   print *, " Available complex kernels are: "
#ifdef HAVE_AVX2
   print *, " AVX kernels are optimized for FMA (AVX2)"
#endif
   call print_available_complex_kernels()

end program print_available_elpa2_kernels
