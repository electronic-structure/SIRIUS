module sirius
use, intrinsic :: ISO_C_BINDING

interface

    subroutine sirius_add_atom_type(label, fname)&
       &bind(C, name="sirius_add_atom_type")
        character, dimension(*), intent(in) :: label
        character, dimension(*), intent(in) :: fname
    end subroutine

    subroutine sirius_set_atom_type_properties(label, symbol, zn, mass, mt_radius, num_mt_points)&
       &bind(C, name="sirius_set_atom_type_properties")
        character, dimension(*), intent(in) :: label
        character, dimension(*), intent(in) :: symbol
        integer,                 intent(in) :: zn
        real(8),                 intent(in) :: mass
        real(8),                 intent(in) :: mt_radius
        integer,                 intent(in) :: num_mt_points
    end subroutine

    subroutine sirius_set_atom_type_radial_grid(label, num_radial_points, radial_points)&
       &bind(C, name="sirius_set_atom_type_radial_grid")
        character, dimension(*), intent(in) :: label
        integer,                 intent(in) :: num_radial_points
        real(8),                 intent(in) :: radial_points
    end subroutine
        
    subroutine c_sirius_density_initialize(rhoit, rhomt, magit, magmt)&
       &bind(C, name="sirius_density_initialize")
        use, intrinsic :: ISO_C_BINDING
        type(C_PTR), value, intent(in) :: rhoit
        type(C_PTR), value, intent(in) :: rhomt
        type(C_PTR), value, intent(in) :: magit
        type(C_PTR), value, intent(in) :: magmt
    end subroutine

    subroutine sirius_set_atom_type_rho_core(label, num_points, rho_core)&
       &bind(C, name="sirius_set_atom_type_rho_core")
        character, dimension(*), intent(in) :: label
        integer,                 intent(in) :: num_points
        real(8),                 intent(in) :: rho_core
    end subroutine

end interface

contains

    function c_str(f_string) result(c_string)
        implicit none
        character(len=*), intent(in)  :: f_string
        character(len=1, kind=C_CHAR) :: c_string(len_trim(f_string) + 1)
        integer i
        do i = 1, len_trim(f_string)
          c_string(i) = f_string(i:i)
        end do
        c_string(len_trim(f_string) + 1) = C_NULL_CHAR
    end function c_str

    subroutine sirius_density_initialize(rhoit, rhomt, magit, magmt)
        implicit none
        real(8),           target, intent(in) :: rhoit
        real(8), optional, target, intent(in) :: rhomt
        real(8), optional, target, intent(in) :: magit
        real(8), optional, target, intent(in) :: magmt
        type(C_PTR) rhoit_ptr, rhomt_ptr, magit_ptr, magmt_ptr

        rhoit_ptr = C_LOC(rhoit)
        rhomt_ptr = C_NULL_PTR
        magit_ptr = C_NULL_PTR
        magmt_ptr = C_NULL_PTR

        call c_sirius_density_initialize(rhoit_ptr, rhomt_ptr, magit_ptr, magmt_ptr)

    end subroutine

end module
