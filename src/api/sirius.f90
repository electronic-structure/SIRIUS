!> @file sirius.f90
!! @brief SIRIUS interface to Fortran
module sirius

use, intrinsic :: ISO_C_BINDING

!> @brief Opaque wrapper for simulation context handler.
type sirius_context_handler
    type(C_PTR) :: handler_ptr_
end type

!> @brief Opaque wrapper for DFT ground statee handler.
type sirius_ground_state_handler
    type(C_PTR) :: handler_ptr_
end type

!> @brief Opaque wrapper for K-point set handler.
type sirius_kpoint_set_handler
    type(C_PTR) :: handler_ptr_
end type

interface sirius_free_handler
    module procedure sirius_free_handler_ctx, sirius_free_handler_ks, sirius_free_handler_dft
end interface

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

subroutine sirius_free_handler_ctx(handler, error_code)
    implicit none
    type(sirius_context_handler), intent(inout) :: handler
    integer, optional, target, intent(out) :: error_code
    call sirius_free_object_handler(handler%handler_ptr_, error_code)
end subroutine sirius_free_handler_ctx

subroutine sirius_free_handler_ks(handler, error_code)
    implicit none
    type(sirius_kpoint_set_handler), intent(inout) :: handler
    integer, optional, target, intent(out) :: error_code
    call sirius_free_object_handler(handler%handler_ptr_, error_code)
end subroutine sirius_free_handler_ks

subroutine sirius_free_handler_dft(handler, error_code)
    implicit none
    type(sirius_ground_state_handler), intent(inout) :: handler
    integer, optional, target, intent(out) :: error_code
    call sirius_free_object_handler(handler%handler_ptr_, error_code)
end subroutine sirius_free_handler_dft

!subroutine sirius_free_handler_ptr(handler, error_code)
!    implicit none
!    type(C_PTR), intent(inout) :: handler
!    integer, optional, target, intent(out) :: error_code
!    call sirius_free_object_handler(handler, error_code)
!end subroutine sirius_free_handler_ptr

end module
