module sirius
use, intrinsic :: ISO_C_BINDING

interface

    subroutine sirius_platform_initialize(call_mpi_init)&
       &bind(C, name="sirius_platform_initialize")
        integer,                 intent(in) :: call_mpi_init
    end subroutine

    subroutine sirius_create_global_parameters()&
       &bind(C, name="sirius_create_global_parameters")
    end subroutine

    subroutine sirius_set_lattice_vectors(a1, a2, a3)&
       &bind(C, name="sirius_set_lattice_vectors")
        real(8),                 intent(in) :: a1
        real(8),                 intent(in) :: a2
        real(8),                 intent(in) :: a3
    end subroutine

    subroutine sirius_set_pw_cutoff(pw_cutoff)&
       &bind(C, name="sirius_set_pw_cutoff")
        real(8),                 intent(in) :: pw_cutoff
    end subroutine

    subroutine sirius_set_gk_cutoff(gk_cutoff)&
       &bind(C, name="sirius_set_gk_cutoff")
        real(8),                 intent(in) :: gk_cutoff
    end subroutine

    subroutine sirius_set_num_fv_states(num_fv_states)&
       &bind(C, name="sirius_set_num_fv_states")
        integer,                 intent(in) :: num_fv_states
    end subroutine

    subroutine sirius_set_auto_rmt(auto_rmt)&
       &bind(C, name="sirius_set_auto_rmt")
        integer,                 intent(in) :: auto_rmt
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

    subroutine sirius_get_num_gvec(num_gvec)&
       &bind(C, name="sirius_get_num_gvec")
        integer,                 intent(out) :: num_gvec
    end subroutine

    subroutine sirius_get_num_fft_grid_points(num_grid_points)&
       &bind(C, name="sirius_get_num_fft_grid_points")
        integer,                 intent(out) :: num_grid_points
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

    subroutine sirius_generate_valence_density(kset_id)&
       &bind(C, name="sirius_generate_valence_density")
        integer,                 intent(in) :: kset_id
    end subroutine

    subroutine sirius_augment_density(kset_id)&
       &bind(C, name="sirius_augment_density")
    end subroutine

    subroutine sirius_density_mixer_initialize()&
       &bind(C, name="sirius_density_mixer_initialize")
    end subroutine

    subroutine sirius_mix_density(rms)&
       &bind(C, name="sirius_mix_density")
        real(8),                 intent(out) :: rms
    end subroutine

    subroutine sirius_generate_initial_density()&
       &bind(C, name="sirius_generate_initial_density")
    end subroutine

    subroutine sirius_create_kset(num_kpoints, kpoints, kpoint_weights, init_kset, kset_id)&
       &bind(C, name="sirius_create_kset")
        integer,                 intent(in) :: num_kpoints
        real(8),                 intent(in) :: kpoints
        real(8),                 intent(in) :: kpoint_weights
        integer,                 intent(in) :: init_kset
        integer,                 intent(out) :: kset_id
    end subroutine

    subroutine sirius_get_band_energies(kset_id, ik, band_energies)&
       &bind(C, name="sirius_get_band_energies")
        integer,                 intent(in) :: kset_id
        integer,                 intent(in) :: ik
        real(8),                 intent(in) :: band_energies
    end subroutine

    subroutine sirius_set_band_occupancies(kset_id, ik, band_occupancies)&
       &bind(C, name="sirius_set_band_occupancies")
        integer,                 intent(in) :: kset_id
        integer,                 intent(in) :: ik
        real(8),                 intent(in) :: band_occupancies
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

    subroutine sirius_add_atom_type_aux(label, fname)&
       &bind(C, name="sirius_add_atom_type")
        use, intrinsic :: ISO_C_BINDING
        type(C_PTR), value, intent(in) :: label
        type(C_PTR), value, intent(in) :: fname
    end subroutine

    subroutine sirius_add_atom_aux(label, pos, vfield)&
       &bind(C, name="sirius_add_atom")
        use, intrinsic :: ISO_C_BINDING
        type(C_PTR), value, intent(in) :: label
        type(C_PTR), value, intent(in) :: pos
        type(C_PTR), value, intent(in) :: vfield
    end subroutine

    subroutine sirius_global_initialize_aux(num_mag_dims_ptr, lmax_apw_ptr, lmax_rho_ptr, lmax_pot_ptr)&
       &bind(C, name="sirius_global_initialize")
        use, intrinsic :: ISO_C_BINDING
        type(C_PTR), value, intent(in) :: num_mag_dims_ptr
        type(C_PTR), value, intent(in) :: lmax_apw_ptr
        type(C_PTR), value, intent(in) :: lmax_rho_ptr
        type(C_PTR), value, intent(in) :: lmax_pot_ptr
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

    subroutine sirius_add_atom_type(label, fname)
        implicit none
        character,           target, dimension(*), intent(in) :: label
        character, optional, target, dimension(*), intent(in) :: fname
        type(C_PTR) label_ptr, fname_ptr

        label_ptr = C_LOC(label(1))
        fname_ptr = C_NULL_PTR

        call sirius_add_atom_type_aux(label_ptr, fname_ptr)

    end subroutine

    subroutine sirius_add_atom(label, pos, vfield)
        implicit none
        character,         target, dimension(*), intent(in) :: label
        real(8),           target,               intent(in) :: pos
        real(8), optional, target,               intent(in) :: vfield
        type(C_PTR) label_ptr, pos_ptr, vfield_ptr
        
        label_ptr = C_LOC(label(1))
        pos_ptr = C_LOC(pos)
        vfield_ptr = C_NULL_PTR
        if (present(vfield)) vfield_ptr = C_LOC(vfield)

        call sirius_add_atom_aux(label_ptr, pos_ptr, vfield_ptr)

    end subroutine

    subroutine sirius_global_initialize(num_mag_dims, lmax_apw, lmax_rho, lmax_pot)
        implicit none
        integer, optional, target, intent(in) :: num_mag_dims
        integer, optional, target, intent(in) :: lmax_apw
        integer, optional, target, intent(in) :: lmax_rho
        integer, optional, target, intent(in) :: lmax_pot
        type(C_PTR) num_mag_dims_ptr, lmax_apw_ptr, lmax_rho_ptr, lmax_pot_ptr

        num_mag_dims_ptr = C_NULL_PTR
        if (present(num_mag_dims)) num_mag_dims_ptr = C_LOC(num_mag_dims)

        lmax_apw_ptr = C_NULL_PTR
        if (present(lmax_apw)) lmax_apw_ptr = C_LOC(lmax_apw)

        lmax_rho_ptr = C_NULL_PTR
        if (present(lmax_rho)) lmax_rho_ptr = C_LOC(lmax_rho)
        
        lmax_pot_ptr = C_NULL_PTR
        if (present(lmax_pot)) lmax_pot_ptr = C_LOC(lmax_pot)

        call sirius_global_initialize_aux(num_mag_dims_ptr, lmax_apw_ptr, lmax_rho_ptr, lmax_pot_ptr)

    end subroutine





end module
