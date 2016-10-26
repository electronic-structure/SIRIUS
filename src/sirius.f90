module sirius
    use, intrinsic :: ISO_C_BINDING

    interface

        subroutine sirius_initialize(call_mpi_init)&
            &bind(C, name="sirius_initialize")
            integer,                 intent(in) :: call_mpi_init
        end subroutine

        subroutine sirius_clear()&
            &bind(C, name="sirius_clear")
        end subroutine

        subroutine sirius_create_simulation_context(config_file_name)&
            &bind(C, name="sirius_create_simulation_context")
            character, dimension(*), intent(in) :: config_file_name
        end subroutine

        subroutine sirius_global_initialize()&
            &bind(C, name="sirius_global_initialize")
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

        subroutine sirius_set_aw_cutoff(aw_cutoff)&
            &bind(C, name="sirius_set_aw_cutoff")
            real(8),                 intent(in) :: aw_cutoff
        end subroutine

        subroutine sirius_set_num_fv_states(num_fv_states)&
            &bind(C, name="sirius_set_num_fv_states")
            integer,                 intent(in) :: num_fv_states
        end subroutine

        subroutine sirius_set_auto_rmt(auto_rmt)&
            &bind(C, name="sirius_set_auto_rmt")
            integer,                 intent(in) :: auto_rmt
        end subroutine

        subroutine sirius_set_lmax_apw(lmax_apw)&
            &bind(C, name="sirius_set_lmax_apw")
            integer,                 intent(in) :: lmax_apw
        end subroutine

        subroutine sirius_set_lmax_pot(lmax_pot)&
            &bind(C, name="sirius_set_lmax_pot")
            integer,                 intent(in) :: lmax_pot
        end subroutine

        subroutine sirius_set_lmax_rho(lmax_rho)&
            &bind(C, name="sirius_set_lmax_rho")
            integer,                 intent(in) :: lmax_rho
        end subroutine

        subroutine sirius_set_num_mag_dims(num_mag_dims)&
            &bind(C, name="sirius_set_num_mag_dims")
            integer,                 intent(in) :: num_mag_dims
        end subroutine

        subroutine sirius_add_xc_functional(xc_name)&
            &bind(C, name="sirius_add_xc_functional")
            character, dimension(*), intent(in) :: xc_name
        end subroutine

        subroutine sirius_set_esm_type(esm_name)&
            &bind(C, name="sirius_set_esm_type")
            character, dimension(*), intent(in) :: esm_name
        end subroutine

        subroutine sirius_set_gamma_point(gamma_point)&
            &bind(C, name="sirius_set_gamma_point")
            logical,                 intent(in) :: gamma_point
        end subroutine

        subroutine sirius_set_mpi_grid_dims(ndims, dims)&
            &bind(C, name="sirius_set_mpi_grid_dims")
            integer,                 intent(in) :: ndims
            integer,                 intent(in) :: dims
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

        subroutine sirius_set_atom_type_configuration(label, n, l, k, occupancy, core)&
            &bind(C, name="sirius_set_atom_type_configuration")
            character, dimension(*), intent(in) :: label
            integer,                 intent(in) :: n
            integer,                 intent(in) :: l
            integer,                 intent(in) :: k
            real(8),                 intent(in) :: occupancy
            logical,                 intent(in) :: core
        end subroutine

        subroutine sirius_set_atom_type_radial_grid(label, num_radial_points, radial_points)&
            &bind(C, name="sirius_set_atom_type_radial_grid")
            character, dimension(*), intent(in) :: label
            integer,                 intent(in) :: num_radial_points
            real(8),                 intent(in) :: radial_points
        end subroutine

        subroutine sirius_set_free_atom_density(label, num_radial_points, dens)&
            &bind(C, name="sirius_set_free_atom_density")
            character, dimension(*), intent(in) :: label
            integer,                 intent(in) :: num_radial_points
            real(8),                 intent(in) :: dens
        end subroutine

        subroutine sirius_set_equivalent_atoms(eqatoms)&
            &bind(C, name="sirius_set_equivalent_atoms")
            integer,                  intent(in) :: eqatoms
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

        subroutine sirius_set_atom_type_q_rf(label, q_rf, lmax)&
            &bind(C, name="sirius_set_atom_type_q_rf")
            character, dimension(*), intent(in) :: label
            real(8),                 intent(in) :: q_rf
            integer,                 intent(in) :: lmax
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

        subroutine sirius_add_atom_type_aw_descriptor(label, n, l, enu, dme, auto_enu)&
            &bind(C, name="sirius_add_atom_type_aw_descriptor")
            character, dimension(*), intent(in) :: label
            integer,                 intent(in) :: n
            integer,                 intent(in) :: l
            real(8),                 intent(in) :: enu
            integer,                 intent(in) :: dme
            integer,                 intent(in) :: auto_enu
        end subroutine

        subroutine sirius_add_atom_type_lo_descriptor(label, idxlo, n, l, enu, dme, auto_enu)&
            &bind(C, name="sirius_add_atom_type_lo_descriptor")
            character, dimension(*), intent(in) :: label
            integer,                 intent(in) :: idxlo
            integer,                 intent(in) :: n
            integer,                 intent(in) :: l
            real(8),                 intent(in) :: enu
            integer,                 intent(in) :: dme
            integer,                 intent(in) :: auto_enu
        end subroutine

        subroutine sirius_set_aw_enu(ia, l, order, enu)&
            &bind(C, name="sirius_set_aw_enu")
            integer,                 intent(in) :: ia
            integer,                 intent(in) :: l
            integer,                 intent(in) :: order
            real(8),                 intent(in) :: enu
        end subroutine

        subroutine sirius_set_lo_enu(ia, idxlo, order, enu)&
            &bind(C, name="sirius_set_lo_enu")
            integer,                 intent(in) :: ia
            integer,                 intent(in) :: idxlo
            integer,                 intent(in) :: order
            real(8),                 intent(in) :: enu
        end subroutine

        subroutine sirius_set_aw_surface_derivative(ia, l, io, dm, deriv)&
            &bind(C, name="sirius_set_aw_surface_derivative")
            integer,                 intent(in) :: ia
            integer,                 intent(in) :: l
            integer,                 intent(in) :: io
            integer,                 intent(in) :: dm
            real(8),                 intent(in) :: deriv
        end subroutine

        subroutine sirius_set_aw_lo_o_radial_integral(ia, l, io1, ilo2, oalo)&
            &bind(C, name="sirius_set_aw_lo_o_radial_integral")
            integer,                 intent(in) :: ia
            integer,                 intent(in) :: l
            integer,                 intent(in) :: io1
            integer,                 intent(in) :: ilo2
            real(8),                 intent(in) :: oalo
        end subroutine

        subroutine sirius_set_lo_lo_o_radial_integral(ia, l, ilo1, ilo2, ololo)&
            &bind(C, name="sirius_set_lo_lo_o_radial_integral")
            integer,                 intent(in) :: ia
            integer,                 intent(in) :: l
            integer,                 intent(in) :: ilo1
            integer,                 intent(in) :: ilo2
            real(8),                 intent(in) :: ololo
        end subroutine

        subroutine sirius_set_aw_aw_h_radial_integral(ia, l1, io1, l2, io2, lm3, haa)&
            &bind(C, name="sirius_set_aw_aw_h_radial_integral")
            integer,                 intent(in) :: ia
            integer,                 intent(in) :: l1
            integer,                 intent(in) :: io1
            integer,                 intent(in) :: l2
            integer,                 intent(in) :: io2
            integer,                 intent(in) :: lm3
            real(8),                 intent(in) :: haa
        end subroutine

        subroutine sirius_set_lo_aw_h_radial_integral(ia, ilo1, l2, io2, lm3, hloa)&
            &bind(C, name="sirius_set_lo_aw_h_radial_integral")
            integer,                 intent(in) :: ia
            integer,                 intent(in) :: ilo1
            integer,                 intent(in) :: l2
            integer,                 intent(in) :: io2
            integer,                 intent(in) :: lm3
            real(8),                 intent(in) :: hloa
        end subroutine

        subroutine sirius_set_lo_lo_h_radial_integral(ia, ilo1, ilo2, lm3, hlolo)&
            &bind(C, name="sirius_set_lo_lo_h_radial_integral")
            integer,                intent(in) :: ia
            integer,                intent(in) :: ilo1
            integer,                intent(in) :: ilo2
            integer,                intent(in) :: lm3
            real(8),                intent(in) :: hlolo
        end subroutine

        subroutine sirius_set_effective_potential_pw_coeffs(f_pw)&
            &bind(C, name="sirius_set_effective_potential_pw_coeffs")
            complex(8),           intent(in) :: f_pw
        end subroutine

        subroutine sirius_get_num_gvec(num_gvec)&
            &bind(C, name="sirius_get_num_gvec")
            integer,                 intent(out) :: num_gvec
        end subroutine

        subroutine sirius_get_num_fft_grid_points(num_grid_points)&
            &bind(C, name="sirius_get_num_fft_grid_points")
            integer,                 intent(out) :: num_grid_points
        end subroutine

        subroutine sirius_get_num_fv_states(num_fv_states)&
            &bind(C, name="sirius_get_num_fv_states")
            integer,                 intent(out) :: num_fv_states
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
            integer,                 intent(in) :: kset_id
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

        subroutine sirius_symmetrize_density()&
            &bind(C, name="sirius_symmetrize_density")
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
            real(8),                 intent(out) :: band_energies
        end subroutine

        subroutine sirius_get_energy_fermi(kset_id, efermi)&
            &bind(C, name="sirius_get_energy_fermi")
            integer,                 intent(in) :: kset_id
            real(8),                 intent(out) :: efermi
        end subroutine

        subroutine sirius_set_band_occupancies(kset_id, ik, band_occupancies)&
            &bind(C, name="sirius_set_band_occupancies")
            integer,                 intent(in) :: kset_id
            integer,                 intent(in) :: ik
            real(8),                 intent(in) :: band_occupancies
        end subroutine

        subroutine sirius_set_rho_pw(num_gvec, gvec, rho_pw, comm)&
            &bind(C, name="sirius_set_rho_pw")
            integer,                 intent(in) :: num_gvec
            integer,                 intent(in) :: gvec
            complex(8),              intent(in) :: rho_pw
            integer,                 intent(in) :: comm
        end subroutine

        subroutine sirius_get_rho_pw(num_gvec, gvec, rho_pw)&
            &bind(C, name="sirius_get_rho_pw")
            integer,                 intent(in)  :: num_gvec
            integer,                 intent(in)  :: gvec
            complex(8),              intent(out) :: rho_pw
        end subroutine

        subroutine sirius_get_gvec_index(gvec, ig)&
            &bind(C, name="sirius_get_gvec_index")
            integer,                  intent(in)  :: gvec(3)
            integer,                  intent(out) :: ig
        end subroutine

        subroutine sirius_get_max_num_mt_points(max_num_mt_points)&
            &bind(C, name="sirius_get_max_num_mt_points")
            integer,                 intent(out) :: max_num_mt_points
        end subroutine

        subroutine sirius_get_fft_grid_limits(d, lower, upper)&
            &bind(C, name="sirius_get_fft_grid_limits")
            integer,                 intent(in)  :: d
            integer,                 intent(out) :: lower
            integer,                 intent(out) :: upper
        end subroutine

        subroutine sirius_get_fft_grid_size(fft_grid_size)&
            &bind(C, name="sirius_get_fft_grid_size")
            integer,                 intent(out) :: fft_grid_size
        end subroutine

        subroutine sirius_get_fft_index(fft_index)&
            &bind(C, name="sirius_get_fft_index")
            integer,                 intent(out) :: fft_index
        end subroutine

        subroutine sirius_get_gvec(gvec)&
            &bind(C, name="sirius_get_gvec")
            integer,                 intent(out) :: gvec
        end subroutine

        subroutine sirius_get_gvec_cart(gvec)&
            &bind(C, name="sirius_get_gvec_cart")
            real(8),                 intent(out) :: gvec
        end subroutine

        subroutine sirius_get_gvec_len(gvec_len)&
            &bind(C, name="sirius_get_gvec_len")
            real(8),                 intent(out) :: gvec_len
        end subroutine

        subroutine sirius_get_gvec_phase_factors(sfacg)&
            &bind(C, name="sirius_get_gvec_phase_factors")
            complex(8),               intent(out) :: sfacg
        end subroutine

        subroutine sirius_get_gvec_ylm(gvec_ylm, ld, lmax)&
            &bind(C, name="sirius_get_gvec_ylm")
            complex(8),               intent(out) :: gvec_ylm
            integer,                  intent(in)  :: ld
            integer,                  intent(in)  :: lmax
        end subroutine

        subroutine sirius_get_max_num_gkvec(kset_id, max_num_gkvec)&
            &bind(C, name="sirius_get_max_num_gkvec")
            integer,                 intent(in)  :: kset_id
            integer,                 intent(out) :: max_num_gkvec
        end subroutine

        subroutine sirius_get_gkvec_arrays(kset_id, ik, num_gkvec, gvec_index, gkvec, gkvec_cart,&
            &gkvec_len, gkvec_tp)&
            &bind(C, name="sirius_get_gkvec_arrays")
            integer,                 intent(in)  :: kset_id
            integer,                 intent(in)  :: ik
            integer,                 intent(out) :: num_gkvec
            integer,                 intent(out) :: gvec_index
            real(8),                 intent(out) :: gkvec
            real(8),                 intent(out) :: gkvec_cart
            real(8),                 intent(out) :: gkvec_len
            real(8),                 intent(out) :: gkvec_tp
        end subroutine

        subroutine sirius_get_index_by_gvec(index_by_gvec)&
            &bind(C, name="sirius_get_index_by_gvec")
            integer,                 intent(out) :: index_by_gvec
        end subroutine

        subroutine sirius_get_step_function(cfunig, cfunir)&
            &bind(C, name="sirius_get_step_function")
            complex(8),              intent(out) :: cfunig
            real(8),                 intent(out) :: cfunir
        end subroutine

        subroutine sirius_start_timer(timer_name)&
            &bind(C, name="sirius_start_timer")
            character, dimension(*), intent(in) :: timer_name
        end subroutine

        subroutine sirius_stop_timer(timer_name)&
            &bind(C, name="sirius_stop_timer")
            character, dimension(*), intent(in) :: timer_name
        end subroutine

        subroutine sirius_print_timers()&
            &bind(C, name="sirius_print_timers")
        end subroutine

        subroutine sirius_use_internal_mixer(flag)&
            &bind(C, name="sirius_use_internal_mixer")
            integer,                 intent(out) :: flag
        end subroutine

        subroutine sirius_set_iterative_solver_tolerance(tol)&
            &bind(C, name="sirius_set_iterative_solver_tolerance")
            real(8),                 intent(in) :: tol
        end subroutine

        subroutine sirius_get_density_dr2(dr2)&
            &bind(C, name="sirius_get_density_dr2")
            real(8),                 intent(out) :: dr2
        end subroutine

        subroutine sirius_generate_coulomb_potential(vclmt, vclit)&
            &bind(C, name="sirius_generate_coulomb_potential")
            real(8),                 intent(out) :: vclmt
            real(8),                 intent(out) :: vclit
        end subroutine

        subroutine sirius_generate_xc_potential(vxcmt, vxcit, bxcmt, bxcit)&
            &bind(C, name="sirius_generate_xc_potential")
            real(8),                 intent(out) :: vxcmt
            real(8),                 intent(out) :: vxcit
            real(8),                 intent(out) :: bxcmt
            real(8),                 intent(out) :: bxcit
        end subroutine

        subroutine sirius_generate_potential_pw_coefs()&
            &bind(C, name="sirius_generate_potential_pw_coefs")
        end subroutine

        subroutine sirius_generate_radial_functions()&
            &bind(C, name="sirius_generate_radial_functions")
        end subroutine

        subroutine sirius_generate_radial_integrals()&
            &bind(C, name="sirius_generate_radial_integrals")
        end subroutine

        subroutine sirius_get_aw_deriv_radial_function(ia, l, io, dfdr)&
            &bind(C, name="sirius_get_aw_deriv_radial_function")
            integer,                 intent(in)  :: ia
            integer,                 intent(in)  :: l
            integer,                 intent(in)  :: io
            real(8),                 intent(out) :: dfdr
        end subroutine

        subroutine sirius_get_lo_deriv_radial_function(ia, idxlo, dfdr)&
            &bind(C, name="sirius_get_lo_deriv_radial_function")
            integer,                 intent(in)  :: ia
            integer,                 intent(in)  :: idxlo
            real(8),                 intent(out) :: dfdr
        end subroutine

        subroutine sirius_get_aw_radial_function(ia, l, io, f)&
            &bind(C, name="sirius_get_aw_radial_function")
            integer,                 intent(in)  :: ia
            integer,                 intent(in)  :: l
            integer,                 intent(in)  :: io
            real(8),                 intent(out) :: f
        end subroutine

        subroutine sirius_set_aw_radial_function(ia, l, io, f)&
            &bind(C, name="sirius_set_aw_radial_function")
            integer,                 intent(in)  :: ia
            integer,                 intent(in)  :: l
            integer,                 intent(in)  :: io
            real(8),                 intent(in)  :: f
        end subroutine

        subroutine sirius_get_lo_radial_function(ia, idxlo, f)&
            &bind(C, name="sirius_get_lo_radial_function")
            integer,                 intent(in)  :: ia
            integer,                 intent(in)  :: idxlo
            real(8),                 intent(out) :: f
        end subroutine

        subroutine sirius_set_lo_radial_function(ia, idxlo, f)&
            &bind(C, name="sirius_set_lo_radial_function")
            integer,                 intent(in)  :: ia
            integer,                 intent(in)  :: idxlo
            real(8),                 intent(in)  :: f
        end subroutine

        subroutine sirius_get_evalsum(val)&
            &bind(C, name="sirius_get_evalsum")
            real(8),                 intent(out) :: val
        end subroutine

        subroutine sirius_get_energy_exc(val)&
            &bind(C, name="sirius_get_energy_exc")
            real(8),                 intent(out) :: val
        end subroutine

        subroutine sirius_get_energy_vxc(val)&
            &bind(C, name="sirius_get_energy_vxc")
            real(8),                 intent(out) :: val
        end subroutine

        subroutine sirius_get_energy_bxc(val)&
            &bind(C, name="sirius_get_energy_bxc")
            real(8),                 intent(out) :: val
        end subroutine

        subroutine sirius_get_energy_veff(val)&
            &bind(C, name="sirius_get_energy_veff")
            real(8),                 intent(out) :: val
        end subroutine

        subroutine sirius_get_energy_vha(val)&
            &bind(C, name="sirius_get_energy_vha")
            real(8),                 intent(out) :: val
        end subroutine

        subroutine sirius_get_energy_enuc(val)&
            &bind(C, name="sirius_get_energy_enuc")
            real(8),                 intent(out) :: val
        end subroutine

        subroutine sirius_get_energy_kin(val)&
            &bind(C, name="sirius_get_energy_kin")
            real(8),                 intent(out) :: val
        end subroutine

        subroutine sirius_get_energy_tot(val)&
            &bind(C, name="sirius_get_energy_tot")
            real(8),                 intent(out) :: val
        end subroutine

        subroutine sirius_get_energy_ewald(val)&
            &bind(C, name="sirius_get_energy_ewald")
            real(8),                 intent(out) :: val
        end subroutine

        subroutine sirius_get_fv_h_o(kset_id, ik, msize, h, o)&
            &bind(C, name="sirius_get_fv_h_o")
            integer,                 intent(in)  :: kset_id
            integer,                 intent(in)  :: ik
            integer,                 intent(in)  :: msize
            complex(8),              intent(out) :: h
            complex(8),              intent(out) :: o
        end subroutine

        subroutine sirius_solve_fv(kset_id, ik, h, o, eval, evec, ld)&
            &bind(C, name="sirius_solve_fv")
            integer,                 intent(in)  :: kset_id
            integer,                 intent(in)  :: ik
            complex(8),              intent(out) :: h
            complex(8),              intent(out) :: o
            real(8),                 intent(out) :: eval
            complex(8),              intent(out) :: evec
            integer,                 intent(in)  :: ld
        end subroutine

        subroutine sirius_update_atomic_potential()&
            &bind(C, name="sirius_update_atomic_potential")
        end subroutine

        subroutine sirius_get_matching_coefficients(kset_id, ik, apwalm, ngkmax, apwordmax)&
            &bind(C, name="sirius_get_matching_coefficients")
            integer,                 intent(in)  :: kset_id
            integer,                 intent(in)  :: ik
            complex(8),              intent(out) :: apwalm
            integer,                 intent(in)  :: ngkmax
            integer,                 intent(in)  :: apwordmax
        end subroutine

        subroutine sirius_get_fft_comm(fcomm)&
            &bind(C, name="sirius_get_fft_comm")
            integer,                 intent(out) :: fcomm
        end subroutine
        
        subroutine sirius_get_kpoint_inner_comm(fcomm)&
            &bind(C, name="sirius_get_kpoint_inner_comm")
            integer,                 intent(out) :: fcomm
        end subroutine

        subroutine sirius_get_all_kpoints_comm(fcomm)&
            &bind(C, name="sirius_get_all_kpoints_comm")
            integer,                 intent(out) :: fcomm
        end subroutine

        subroutine sirius_radial_solver(solver_type, zn, dme, l, k, enu, nr, r, v, nn, p0, p1, q0, q1)&
            &bind(C, name="sirius_radial_solver")
            character, dimension(*), intent(in)  :: solver_type
            integer,                 intent(in)  :: zn
            integer,                 intent(in)  :: dme
            integer,                 intent(in)  :: l
            integer,                 intent(in)  :: k
            real(8),                 intent(in)  :: enu
            integer,                 intent(in)  :: nr
            real(8),                 intent(in)  :: r
            real(8),                 intent(in)  :: v
            integer,                 intent(in)  :: nn
            real(8),                 intent(in)  :: p0
            real(8),                 intent(in)  :: p1
            real(8),                 intent(in)  :: q0
            real(8),                 intent(in)  :: q1
        end subroutine

        subroutine sirius_write_json_output()&
            &bind(C, name="sirius_write_json_output")
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


        !-------------------------------------
        !---- PAW API ------------------------
        !-------------------------------------
        subroutine sirius_set_atom_type_paw_data(label__, &
                                                ae_wfc_rf__, &
                                                ps_wfc_rf__, &
                                                num_wfc__,&
                                                ld__,&
                                                cutoff_radius_index__,&
                                                core_energy__,&
                                                ae_core_charge__,&
                                                num_ae_core_charge__,&
                                                occupations__,&
                                                num_occ__)&

            &bind(C, name="sirius_set_atom_type_paw_data")

            character, dimension(*),  intent(in) :: label__
            real(8),                  intent(in) :: ae_wfc_rf__
            real(8),                  intent(in) :: ps_wfc_rf__
            integer,                  intent(in) :: num_wfc__
            integer,                  intent(in) :: ld__
            integer,                  intent(in) :: cutoff_radius_index__
            real(8),                  intent(in) :: core_energy__
            real(8),                  intent(in) :: ae_core_charge__
            integer,                  intent(in) :: num_ae_core_charge__
            real(8),                  intent(in) :: occupations__
            integer,                  intent(in) :: num_occ__

        end subroutine



        !-------------------------------------------------------
        !-------------------------------------------------------
        subroutine sirius_get_paw_total_energy(tot_en__)&
            &bind(C, name="sirius_get_paw_total_energy")
            real(8),              intent(out) :: tot_en__
        end subroutine

        !-------------------------------------------------------
        !-------------------------------------------------------
        subroutine sirius_get_paw_one_elec_energy(one_elec_en__)&
            &bind(C, name="sirius_get_paw_one_elec_energy")
            real(8),              intent(out) :: one_elec_en__
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

    subroutine sirius_density_initialize(rhoit, magit, rhomt, magmt)
        implicit none
        real(8),           target, intent(in) :: rhoit
        real(8), optional, target, intent(in) :: magit
        real(8), optional, target, intent(in) :: rhomt
        real(8), optional, target, intent(in) :: magmt
        type(C_PTR) rhoit_ptr, rhomt_ptr, magit_ptr, magmt_ptr

        rhoit_ptr = C_LOC(rhoit)
        
        magit_ptr = C_NULL_PTR
        if (present(magit)) magit_ptr = C_LOC(magit)
        
        rhomt_ptr = C_NULL_PTR
        if (present(rhomt)) rhomt_ptr = C_LOC(rhomt)
        
        magmt_ptr = C_NULL_PTR
        if (present(magmt)) magmt_ptr = C_LOC(magmt)

        call sirius_density_initialize_aux(rhoit_ptr, rhomt_ptr, magit_ptr, magmt_ptr)

    end subroutine

    subroutine sirius_potential_initialize(veffit, beffit, veffmt, beffmt)
        implicit none
        real(8),           target, intent(in) :: veffit
        real(8), optional, target, intent(in) :: beffit
        real(8), optional, target, intent(in) :: veffmt
        real(8), optional, target, intent(in) :: beffmt
        type(C_PTR) veffit_ptr, veffmt_ptr, beffit_ptr, beffmt_ptr

        veffit_ptr = C_LOC(veffit)

        beffit_ptr = C_NULL_PTR
        if (present(beffit)) beffit_ptr = C_LOC(beffit)

        veffmt_ptr = C_NULL_PTR
        if (present(veffmt)) veffmt_ptr = C_LOC(veffmt)

        beffmt_ptr = C_NULL_PTR
        if (present(beffmt)) beffmt_ptr = C_LOC(beffmt)

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

end module
