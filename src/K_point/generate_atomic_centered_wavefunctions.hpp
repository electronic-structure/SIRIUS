// TODO: pass a list of atomic orbitals to generate
//       this list should contain: index of atom, index of wave-function and some flag to indicate if we average
//       wave-functions in case of spin-orbit; this should be sufficient to generate a desired sub-set of atomic wave-functions

inline void K_point::generate_atomic_centered_wavefunctions_aux(const int         num_ao__,
                                                                Wave_functions&   phi,
                                                                std::vector<int>& offset,
                                                                const bool        hubbard)
{
    if (!num_ao__) {
        return;
    }

    int lmax{0};
    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        auto& atom_type = unit_cell_.atom_type(iat);
        lmax = std::max(lmax, atom_type.lmax_ps_atomic_wf());
    }
    lmax = std::max(lmax, unit_cell_.lmax());

    #pragma omp parallel for schedule(static)
    for (int igk_loc = 0; igk_loc < this->num_gkvec_loc(); igk_loc++) {
        /* global index of G+k vector */
        int igk = this->idxgk(igk_loc);
        /* vs = {r, theta, phi} */
        auto vs = SHT::spherical_coordinates(this->gkvec().gkvec_cart<index_domain_t::local>(igk_loc));
        /* compute real spherical harmonics for G+k vector */
        std::vector<double> rlm(utils::lmmax(lmax));
        SHT::spherical_harmonics(lmax, vs[1], vs[2], &rlm[0]);
        /* get values of radial integrals for a given G+k vector length */
        std::vector<mdarray<double, 1>> ri_values(unit_cell_.num_atom_types());
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            ri_values[iat] = ctx_.atomic_wf_ri().values(iat, vs[0]);
        }

        int n{0};
        if (!hubbard) {
            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                auto phase        = twopi * dot(gkvec().gkvec(igk), unit_cell_.atom(ia).position());
                auto phase_factor = std::exp(double_complex(0.0, phase));
                auto& atom_type   = unit_cell_.atom(ia).type();
                for (int i = 0; i < atom_type.num_ps_atomic_wf(); i++) {
                    auto l = std::abs(atom_type.ps_atomic_wf(i).first);
                    auto z = std::pow(double_complex(0, -1), l) * fourpi / std::sqrt(unit_cell_.omega());
                    for (int m = -l; m <= l; m++) {
                        int lm = utils::lm(l, m);
                        phi.pw_coeffs(0).prime(igk_loc, n) = z * std::conj(phase_factor) * rlm[lm] * ri_values[atom_type.id()][i];
                        n++;
                    }
                } // i
            }
        } else {
            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                auto phase        = twopi * dot(gkvec().gkvec(igk), unit_cell_.atom(ia).position());
                auto phase_factor = double_complex(std::cos(phase), std::sin(phase));
                auto& atom_type   = unit_cell_.atom(ia).type();
                if (atom_type.hubbard_correction()) {
                    for (int i = 0; i < atom_type.num_ps_atomic_wf(); i++) {
                        auto l = std::abs(atom_type.ps_atomic_wf(i).first);
                        auto z = std::pow(double_complex(0, -1), l) * fourpi / std::sqrt(unit_cell_.omega());
                        if (l == atom_type.hubbard_l()) {
                            for (int m = -l; m <= l; m++) {
                                int lm = utils::lm(l, m);
                                if (atom_type.spin_orbit_coupling()) {
                                    phi.pw_coeffs(0).prime(igk_loc, offset[ia] + l + m) += 0.5 * z * std::conj(phase_factor) * rlm[lm] * ri_values[atom_type.id()][i];
                                    phi.pw_coeffs(1).prime(igk_loc, offset[ia] + 3 * l + m + 1) += 0.5 * z * std::conj(phase_factor) * rlm[lm] * ri_values[atom_type.id()][i];
                                } else {
                                    phi.pw_coeffs(0).prime(igk_loc, offset[ia] + l + m) = z * std::conj(phase_factor) * rlm[lm] * ri_values[atom_type.id()][i];
                                    if (ctx_.num_mag_dims() == 3) {
                                        phi.pw_coeffs(1).prime(igk_loc, offset[ia] + 3 * l + m + 1) = z * std::conj(phase_factor) * rlm[lm] * ri_values[atom_type.id()][i];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } // igk_loc
}

inline void K_point::generate_atomic_centered_wavefunctions(const int num_ao__, Wave_functions& phi)
{
    std::vector<int> vs(1, 0);
    generate_atomic_centered_wavefunctions_aux(num_ao__, phi, vs, false);
}

inline void K_point::compute_gradient_wavefunctions(Wave_functions& phi,
                                                    const int       starting_position_i,
                                                    const int       num_wf,
                                                    Wave_functions& dphi,
                                                    const int       starting_position_j,
                                                    const int       direction) {
    std::vector<double_complex> qalpha(this->num_gkvec_loc());

    for (int igk_loc = 0; igk_loc < this->num_gkvec_loc(); igk_loc++) {
        auto G = this->gkvec().gkvec_cart<index_domain_t::local>(igk_loc);

        qalpha[igk_loc] = double_complex(0.0, -G[direction]);
    }

    #pragma omp parallel for schedule(static)
    for (int nphi = 0; nphi < num_wf; nphi++) {
        for (int ispn = 0; ispn < phi.num_sc(); ispn++) {
            for (int igk_loc = 0; igk_loc < this->num_gkvec_loc(); igk_loc++) {
                dphi.pw_coeffs(ispn).prime(igk_loc, nphi + starting_position_j) = qalpha[igk_loc] * phi.pw_coeffs(ispn).prime(igk_loc, nphi + starting_position_i);
            }
        }
    }
}
