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
!
!    More information can be found here:
!    http://elpa.rzg.mpg.de/
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

module ELPA_utilities

#ifdef HAVE_ISO_FORTRAN_ENV
  use iso_fortran_env, only : error_unit
#endif

  implicit none

  private ! By default, all routines contained are private

  public :: debug_messages_via_environment_variable, pcol, prow, error_unit
#ifndef HAVE_ISO_FORTRAN_ENV
  integer, parameter :: error_unit = 0
#endif


  !******
  contains

   function debug_messages_via_environment_variable() result(isSet)
#ifdef HAVE_DETAILED_TIMINGS
     use timings
#endif
     implicit none
     logical              :: isSet
     CHARACTER(len=255)   :: ELPA_DEBUG_MESSAGES

#ifdef HAVE_DETAILED_TIMINGS
     call timer%start("debug_messages_via_environment_variable")
#endif

     isSet = .false.

#if defined(HAVE_ENVIRONMENT_CHECKING)
     call get_environment_variable("ELPA_DEBUG_MESSAGES",ELPA_DEBUG_MESSAGES)
#endif
     if (trim(ELPA_DEBUG_MESSAGES) .eq. "yes") then
       isSet = .true.
     endif
     if (trim(ELPA_DEBUG_MESSAGES) .eq. "no") then
       isSet = .true.
     endif

#ifdef HAVE_DETAILED_TIMINGS
     call timer%stop("debug_messages_via_environment_variable")
#endif

   end function debug_messages_via_environment_variable

!-------------------------------------------------------------------------------

  !Processor col for global col number
  pure function pcol(i, nblk, np_cols) result(col)
    integer, intent(in) :: i, nblk, np_cols
    integer :: col
    col = MOD((i-1)/nblk,np_cols)
  end function

!-------------------------------------------------------------------------------

  !Processor row for global row number
  pure function prow(i, nblk, np_rows) result(row)
    integer, intent(in) :: i, nblk, np_rows
    integer :: row
    row = MOD((i-1)/nblk,np_rows)
  end function

!-------------------------------------------------------------------------------

end module ELPA_utilities
