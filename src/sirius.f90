!> @file sirius.f90
!! @brierf SIRIUS interface to Fortran
module sirius

use, intrinsic :: ISO_C_BINDING

interface sirius_access_hubbard_occupancies
  module procedure sirius_access_hubbard_occupancies_double, sirius_access_hubbard_occupancies_complex
end interface

interface sirius_access_hubbard_potential
  module procedure sirius_access_hubbard_potential_double, sirius_access_hubbard_potential_complex
end interface

contains

function string(f_string) result(res)
    implicit none
    character(len=*), intent(in)  :: f_string
    character(len=1, kind=C_CHAR) :: res(len_trim(f_string) + 1)
    integer i
    do i = 1, len_trim(f_string)
        res(i) = f_string(i:i)
    end do
    res(len_trim(f_string) + 1) = C_NULL_CHAR
end function string

function bool(val) result(res)
    implicit none
    logical, intent(in) :: val
    logical(C_BOOL)     :: res
    res = val
end function bool

include 'generated.f90'

end module
