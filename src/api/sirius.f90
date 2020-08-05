!> @file sirius.f90
!! @brief SIRIUS interface to Fortran
module sirius

use, intrinsic :: ISO_C_BINDING

contains

function string_f2c(f_string) result(res)
    implicit none
    character(kind=C_CHAR,len=*), intent(in)  :: f_string
    character(kind=C_CHAR,len=1) :: res(len_trim(f_string) + 1)
    integer i
    do i = 1, len_trim(f_string)
        res(i) = f_string(i:i)
    end do
    res(len_trim(f_string) + 1) = C_NULL_CHAR
end function string_f2c

function string_c2f(c_string) result(res)
    implicit none
    character(kind=C_CHAR,len=1), intent(in) :: c_string(:)
    character(kind=C_CHAR,len=size(c_string) - 1) :: res
    character(C_CHAR) c
    integer i
    do i = 1, size(c_string)
        c = c_string(i)
        if (c == C_NULL_CHAR) then
            res(i:) = ' '
            exit
        endif
        res(i:i) = c
    end do
end function string_c2f

include 'generated.f90'

end module
