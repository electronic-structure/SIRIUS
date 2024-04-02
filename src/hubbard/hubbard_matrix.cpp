/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file hubbard_matrix.cpp
 *
 *  \brief Base class for Hubbard occupancy and potential matrices.
 */

#include <iomanip>
#include "hubbard_matrix.hpp"

namespace sirius {

Hubbard_matrix::Hubbard_matrix(Simulation_context& ctx__)
    : ctx_(ctx__)
{
    if (!ctx_.full_potential() && ctx_.hubbard_correction()) {

        /* first compute the number of atomic levels involved in the hubbard correction */
        int num_atomic_level{0};
        atomic_orbitals_.clear();
        for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
            auto& atom_type = ctx_.unit_cell().atom(ia).type();
            if (atom_type.hubbard_correction()) {
                num_atomic_level += atom_type.lo_descriptor_hub().size();

                for (int lo = 0; lo < static_cast<int>(atom_type.lo_descriptor_hub().size()); lo++) {
                    atomic_orbitals_.push_back(std::make_pair(ia, lo));
                }
            }
        }

        local_ = std::vector<mdarray<std::complex<double>, 3>>(num_atomic_level);

        if (ctx_.cfg().hubbard().constrained_calculation() && ctx_.cfg().hubbard().local_constraint().size()) {
            apply_constraints_.clear();
            apply_constraints_.resize(num_atomic_level, false);
            local_constraints_       = std::vector<mdarray<std::complex<double>, 3>>(num_atomic_level);
            multipliers_constraints_ = std::vector<mdarray<std::complex<double>, 3>>(num_atomic_level);
        }
        /* the offsets here match the offsets of the hubbard wave functions but
         * are more fine grained. The offsets of the hubbard wave functions are
         * for atom while here they are for each atomic level. Since all atomic
         * level of a given atom are next to each other, the offset of the first
         * atomic level of a given atom has the same value than the offset
         * giving the position of the first hubbard wave function of this
         * atom. */
        offset_ = std::vector<int>(num_atomic_level, 0);

        int size{0};
        for (int at_lvl = 0; at_lvl < static_cast<int>(local_.size()); at_lvl++) {
            offset_[at_lvl]  = size;
            const int ia     = atomic_orbitals_[at_lvl].first;
            auto& atom_type  = ctx_.unit_cell().atom(ia).type();
            const int lo_ind = atomic_orbitals_[at_lvl].second;
            const int l      = atom_type.lo_descriptor_hub(lo_ind).l();
            const int n      = atom_type.lo_descriptor_hub(lo_ind).n();
            const int mmax   = 2 * l + 1;

            local_[at_lvl] = mdarray<std::complex<double>, 3>({mmax, mmax, 4}, mdarray_label("local_hubbard"));
            local_[at_lvl].zero();

            if (ctx_.cfg().hubbard().constrained_calculation()) {
                local_constraints_[at_lvl] =
                        mdarray<std::complex<double>, 3>({mmax, mmax, 4}, mdarray_label("local_hubbard_constraint"));
                multipliers_constraints_[at_lvl] = mdarray<std::complex<double>, 3>(
                        {mmax, mmax, 4}, mdarray_label("lagrange_multiplier_constraint"));
                multipliers_constraints_[at_lvl].zero();
                local_constraints_[at_lvl].zero();

                // fill the constrained occupation numbers. I need to go through the full list of orbitals
                for (int cts_index = 0; cts_index < ctx_.cfg().hubbard().local_constraint().size(); cts_index++) {
                    const auto& constraint_    = ctx_.cfg().hubbard().local_constraint(cts_index);
                    apply_constraints_[at_lvl] = (constraint_.atom_index() == ia) && (constraint_.l() == l) &&
                                                 ((constraint_.n() == n) || (n < 0) || (constraint_.n() < 0));
                    if (apply_constraints_[at_lvl]) {
                        // fill the constrained occupation numbers. I need to go through the full for each orbital
                        const auto& cts_ = ctx_.cfg().hubbard().local_constraint();
                        for (int cts_index = 0; cts_index < cts_.size(); cts_index++) {
                            const auto& constraint_ = ctx_.cfg().hubbard().local_constraint(cts_index);
                            if ((constraint_.atom_index() == ia) && (constraint_.l() == l) &&
                                (((n == constraint_.n()) || (n < 0)) || (constraint_.n() < 0))) {
                                const auto& occ_matrix__ = constraint_.occupancy();
                                if (constraint_.contains("lm_order")) {
                                    const auto& lm_order_ = constraint_.lm_order();
                                    for (unsigned int sp = 0; sp < occ_matrix__.size();
                                         sp++) { // spin blocks up-up, up-down, down-down
                                        for (int m1 = 0; m1 < mmax; m1++) {
                                            for (int m2 = 0; m2 < mmax; m2++) {
                                                local_constraints_[at_lvl](m2, m1, sp) =
                                                        occ_matrix__[sp][l + lm_order_[m1]][l + lm_order_[m2]];
                                            }
                                        }
                                    }
                                } else {
                                    for (unsigned int sp = 0; sp < occ_matrix__.size();
                                         sp++) { // spin blocks up-up, up-down, down-down
                                        for (int m1 = 0; m1 < mmax; m1++) {
                                            for (int m2 = 0; m2 < mmax; m2++) {
                                                local_constraints_[at_lvl](m2, m1, sp) = occ_matrix__[sp][m1][m2];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            size += mmax;
        }

        nonlocal_.clear();
        nonlocal_ = std::vector<mdarray<std::complex<double>, 3>>(ctx_.cfg().hubbard().nonlocal().size());
        for (int i = 0; i < static_cast<int>(ctx_.cfg().hubbard().nonlocal().size()); i++) {
            auto nl      = ctx_.cfg().hubbard().nonlocal(i);
            int il       = nl.l()[0];
            int jl       = nl.l()[1];
            nonlocal_[i] = mdarray<std::complex<double>, 3>({2 * il + 1, 2 * jl + 1, ctx_.num_spins()},
                                                            mdarray_label("nonlocal_hubbard"));
            nonlocal_[i].zero();
        }
    }
}

void
Hubbard_matrix::access(std::string const& what__, std::complex<double>* occ__, int ld__)
{
    if (!(what__ == "get" || what__ == "set")) {
        std::stringstream s;
        s << "wrong access label: " << what__;
        RTE_THROW(s);
    }

    mdarray<std::complex<double>, 4> occ_mtrx;
    /* in non-collinear case the occupancy matrix is complex */
    if (ctx_.num_mag_dims() == 3) {
        occ_mtrx = mdarray<std::complex<double>, 4>({ld__, ld__, 4, ctx_.unit_cell().num_atoms()}, occ__);
    } else {
        occ_mtrx =
                mdarray<std::complex<double>, 4>({ld__, ld__, ctx_.num_spins(), ctx_.unit_cell().num_atoms()}, occ__);
    }
    if (what__ == "get") {
        occ_mtrx.zero();
    }

    for (int at_lvl = 0; at_lvl < static_cast<int>(local().size()); at_lvl++) {
        const int ia1    = atomic_orbitals(at_lvl).first;
        const auto& atom = ctx_.unit_cell().atom(ia1);
        const int lo     = atomic_orbitals(at_lvl).second;
        if (atom.type().lo_descriptor_hub(lo).use_for_calculation()) {
            const int l      = atom.type().lo_descriptor_hub(lo).l();
            const int offset = offset_[at_lvl];
            for (int m1 = -l; m1 <= l; m1++) {
                for (int m2 = -l; m2 <= l; m2++) {
                    if (what__ == "get") {
                        for (int j = 0; j < ((ctx_.num_mag_dims() == 3) ? 4 : ctx_.num_spins()); j++) {
                            occ_mtrx(offset + l + m1, offset + l + m2, j, ia1) = this->local(at_lvl)(l + m1, l + m2, j);
                        }
                    } else {
                        for (int j = 0; j < ((ctx_.num_mag_dims() == 3) ? 4 : ctx_.num_spins()); j++) {
                            this->local(at_lvl)(l + m1, l + m2, j) = occ_mtrx(offset + l + m1, offset + l + m2, j, ia1);
                        }
                    }
                }
            }
        }
    }
}

void
Hubbard_matrix::print_local(int at_lvl__, std::ostream& out__) const
{
    int const prec{5};
    int const width{10};

    auto print_number = [&](double x) { out__ << std::setw(width) << std::setprecision(prec) << std::fixed << x; };
    auto const& atom  = ctx_.unit_cell().atom(atomic_orbitals_[at_lvl__].first);

    out__ << "level : " << atom.type().lo_descriptor_hub(atomic_orbitals_[at_lvl__].second).n();
    out__ << " l: " << atom.type().lo_descriptor_hub(atomic_orbitals_[at_lvl__].second).l() << std::endl;
    const int l = atom.type().lo_descriptor_hub(atomic_orbitals_[at_lvl__].second).l();
    if (ctx_.num_mag_dims() != 3) {
        int mmax = 2 * l + 1;
        for (int is = 0; is < ctx_.num_spins(); is++) {
            out__ << hbar(width * mmax, '-') << std::endl;
            bool has_imag{false};
            for (int m = 0; m < mmax; m++) {
                for (int mp = 0; mp < mmax; mp++) {
                    if (std::abs(std::imag(this->local(at_lvl__)(m, mp, is))) > 1e-12) {
                        has_imag = true;
                    }
                    print_number(std::real(this->local(at_lvl__)(m, mp, is)));
                }
                out__ << std::endl;
            }
            if (has_imag) {
                out__ << "imaginary part:" << std::endl;
                for (int m = 0; m < mmax; m++) {
                    for (int mp = 0; mp < mmax; mp++) {
                        print_number(std::imag(this->local(at_lvl__)(m, mp, is)));
                    }
                    out__ << std::endl;
                }
            }
        }
        out__ << hbar(width * mmax, '-') << std::endl;
    } else {
        int mmax = 2 * l + 1;
        out__ << hbar(2 * width * mmax + 3, '-') << std::endl;
        for (int m = 0; m < mmax; m++) {
            for (int mp = 0; mp < mmax; mp++) {
                print_number(std::real(this->local(at_lvl__)(m, mp, 0)));
            }
            out__ << " | ";
            for (int mp = 0; mp < mmax; mp++) {
                print_number(std::real(this->local(at_lvl__)(m, mp, 2)));
            }
            out__ << std::endl;
        }
        out__ << hbar(2 * width * mmax + 3, '-') << std::endl;
        for (int m = 0; m < mmax; m++) {
            for (int mp = 0; mp < mmax; mp++) {
                print_number(std::real(this->local(at_lvl__)(m, mp, 3)));
            }
            out__ << " | ";
            for (int mp = 0; mp < mmax; mp++) {
                print_number(std::real(this->local(at_lvl__)(m, mp, 1)));
            }
            out__ << std::endl;
        }
        out__ << hbar(2 * width * mmax + 3, '-') << std::endl;
    }

    if (apply_constraint()) {
        out__ << "Hubbard constraint error (l2-norm): " << constraint_error_ << std::endl;
    }
}

void
Hubbard_matrix::print_nonlocal(int idx__, std::ostream& out__) const
{

    auto nl      = ctx_.cfg().hubbard().nonlocal(idx__);
    int ia       = nl.atom_pair()[0];
    int ja       = nl.atom_pair()[1];
    int il       = nl.l()[0];
    int jl       = nl.l()[1];
    const int jb = 2 * jl + 1;
    const int ib = 2 * il + 1;
    r3::vector<int> T(nl.T());

    r3::vector<double> r = ctx_.unit_cell().atom(ja).position() + T - ctx_.unit_cell().atom(ia).position();
    /* convert to Cartesian coordinates */
    auto rc = dot(ctx_.unit_cell().lattice_vectors(), r);

    out__ << "atom: " << ia << ", l: " << il << " -> atom: " << ja << ", l: " << jl << ", T: " << T << ", r: " << rc
          << std::endl;

    int const prec{5};
    int const width{10};
    auto print_number = [&](double x) { out__ << std::setw(width) << std::setprecision(prec) << std::fixed << x; };

    if (ctx_.num_mag_dims() != 3) {
        for (int is = 0; is < ctx_.num_spins(); is++) {
            out__ << hbar(width * jb, '-') << std::endl;
            bool has_imag{false};
            for (int m = 0; m < ib; m++) {
                for (int mp = 0; mp < jb; mp++) {
                    if (std::abs(std::imag(this->nonlocal(idx__)(m, mp, is))) > 1e-12) {
                        has_imag = true;
                    }
                    print_number(std::real(this->nonlocal(idx__)(m, mp, is)));
                }
                out__ << std::endl;
            }
            if (has_imag) {
                out__ << "imaginary part:" << std::endl;
                for (int m = 0; m < ib; m++) {
                    for (int mp = 0; mp < jb; mp++) {
                        print_number(std::imag(this->nonlocal(idx__)(m, mp, is)));
                    }
                    out__ << std::endl;
                }
            }
        }
        out__ << hbar(width * jb, '-') << std::endl;
    }
    //} else {
    //    int l = atom.type().indexr_hub().am(0).l();
    //    int mmax = 2 *l + 1;
    //    draw_bar(2 * width * mmax + 3);
    //    for (int m = 0; m < mmax; m++) {
    //        for (int mp = 0; mp < mmax; mp++) {
    //            print_number(std::real(this->local(ia__)(m, mp, 0)));
    //        }
    //        out__ << " | ";
    //        for (int mp = 0; mp < mmax; mp++) {
    //            print_number(std::real(this->local(ia__)(m, mp, 2)));
    //        }
    //        out__ << std::endl;
    //    }
    //    draw_bar(2 * width * mmax + 3);
    //    for (int m = 0; m < mmax; m++) {
    //        for (int mp = 0; mp < mmax; mp++) {
    //            print_number(std::real(this->local(ia__)(m, mp, 3)));
    //        }
    //        out__ << " | ";
    //        for (int mp = 0; mp < mmax; mp++) {
    //            print_number(std::real(this->local(ia__)(m, mp, 1)));
    //        }
    //        out__ << std::endl;
    //    }
    //    draw_bar(2 * width * mmax + 3);
    //}
}

void
Hubbard_matrix::zero()
{
    for (int ia = 0; ia < static_cast<int>(local_.size()); ia++) {
        local_[ia].zero();
    }

    for (int i = 0; i < static_cast<int>(ctx_.cfg().hubbard().nonlocal().size()); i++) {
        nonlocal_[i].zero();
    }

    for (int at_lvl = 0; at_lvl < static_cast<int>(local_constraints_.size()); at_lvl++) {
        if (apply_constraints(at_lvl)) {
            multipliers_constraints_[at_lvl].zero();
        }
    }
}

} // namespace sirius
