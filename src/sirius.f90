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

    subroutine sirius_set_atom_type_beta_rf(label, num_beta, beta_l, num_mesh_points, beta_rf, ld)&
       &bind(C, name="sirius_set_atom_type_beta_rf")
        character, dimension(*), intent(in) :: label
        integer,                 intent(in) :: num_beta
        integer,                 intent(in) :: beta_l
        integer,                 intent(in) :: num_mesh_points
        real(8),                 intent(in) :: beta_rf
        integer,                 intent(in) :: ld
    end subroutine

    subroutine sirius_set_atom_type_dion(label, num_beta, dion)&
       &bind(C, name="sirius_set_atom_type_dion")
        character, dimension(*), intent(in) :: label
        integer,                 intent(in) :: num_beta
        real(8),                 intent(in) :: dion
    end subroutine

    subroutine sirius_set_atom_type_q_rf(label, num_q_coefs, lmax_q, q_coefs, rinner, q_rf)&
       &bind(C, name="sirius_set_atom_type_q_rf")
        character, dimension(*), intent(in) :: label
        integer,                 intent(in) :: num_q_coefs
        integer,                 intent(in) :: lmax_q
        real(8),                 intent(in) :: q_coefs
        real(8),                 intent(in) :: rinner
        real(8),                 intent(in) :: q_rf
    end subroutine

    subroutine sirius_add_atom(label, pos, vfield)&
       &bind(C, name="sirius_add_atom")
        character, dimension(*), intent(in) :: label
        real(8),                 intent(in) :: pos
        real(8),                 intent(in) :: vfield
    end subroutine
        
    subroutine sirius_set_atom_type_rho_core(label, num_points, rho_core)&
       &bind(C, name="sirius_set_atom_type_rho_core")
        character, dimension(*), intent(in) :: label
        integer,                 intent(in) :: num_points
        real(8),                 intent(in) :: rho_core
    end subroutine

    subroutine sirius_set_atom_type_rho_tot(label, num_points, rho_tot)&
       &bind(C, name="sirius_set_atom_type_rho_tot")
        character, dimension(*), intent(in) :: label
        integer,                 intent(in) :: num_points
        real(8),                 intent(in) :: rho_tot
    end subroutine

    subroutine sirius_set_atom_type_vloc(label, num_points, vloc)&
       &bind(C, name="sirius_set_atom_type_vloc")
        character, dimension(*), intent(in) :: label
        integer,                 intent(in) :: num_points
        real(8),                 intent(in) :: vloc
    end subroutine

    subroutine sirius_ground_state_initialize(kset_id)&
       &bind(C, name="sirius_ground_state_initialize")
        integer,                 intent(in) :: kset_id
    end subroutine

    subroutine sirius_find_eigen_states(kset_id, precompute)&
       &bind(C, name="sirius_find_eigen_states")
        integer,                 intent(in) :: kset_id
        integer,                 intent(in) :: precompute
    end subroutine

    subroutine sirius_generate_effective_potential()&
       &bind(C, name="sirius_generate_effective_potential")
    end subroutine

    subroutine sirius_find_band_occupancies(kset_id)&
       &bind(C, name="sirius_find_band_occupancies")
        integer,                 intent(in) :: kset_id
    end subroutine

    subroutine sirius_generate_density(kset_id)&
       &bind(C, name="sirius_generate_density")
        integer,                 intent(in) :: kset_id
    end subroutine

    subroutine sirius_density_initialize_aux(rhoit, rhomt, magit, magmt)&
       &bind(C, name="sirius_density_initialize")
        use, intrinsic :: ISO_C_BINDING
        type(C_PTR), value, intent(in) :: rhoit
        type(C_PTR), value, intent(in) :: rhomt
        type(C_PTR), value, intent(in) :: magit
        type(C_PTR), value, intent(in) :: magmt
    end subroutine

    subroutine sirius_potential_initialize_aux(veffit, veffmt, beffit, beffmt)&
       &bind(C, name="sirius_potential_initialize")
        use, intrinsic :: ISO_C_BINDING
        type(C_PTR), value, intent(in) :: veffit
        type(C_PTR), value, intent(in) :: veffmt
        type(C_PTR), value, intent(in) :: beffit
        type(C_PTR), value, intent(in) :: beffmt
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

        call sirius_density_initialize_aux(rhoit_ptr, rhomt_ptr, magit_ptr, magmt_ptr)

    end subroutine

    subroutine sirius_potential_initialize(veffit, veffmt, beffit, beffmt)
        implicit none
        real(8),           target, intent(in) :: veffit
        real(8), optional, target, intent(in) :: veffmt
        real(8), optional, target, intent(in) :: beffit
        real(8), optional, target, intent(in) :: beffmt
        type(C_PTR) veffit_ptr, veffmt_ptr, beffit_ptr, beffmt_ptr

        veffit_ptr = C_LOC(veffit)
        beffmt_ptr = C_NULL_PTR
        beffit_ptr = C_NULL_PTR
        beffmt_ptr = C_NULL_PTR

        call sirius_potential_initialize_aux(veffit_ptr, veffmt_ptr, beffit_ptr, beffmt_ptr)

    end subroutine

end module
