// Copyright (c) 2013-2021 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

        // first compute the number of atomic levels involved in the hubbard correction
        int num_atomic_level_{0};
        atomic_orbitals_.clear();
        for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
            auto& atom_type = ctx_.unit_cell().atom(ia).type();
            if (atom_type.hubbard_correction()) {
                num_atomic_level_ += atom_type.lo_descriptor_hub().size();

                for (int lo = 0; lo < static_cast<int>(atom_type.lo_descriptor_hub().size()); lo++) {
                    std::pair<int, int> id = std::make_pair(ia, lo);
                    atomic_orbitals_.push_back(id);
                }
            }
        }

        local_ = std::vector<sddk::mdarray<double_complex, 3>>(num_atomic_level_);

        /* the offsets here match the offsets of the hubbard wave functions but
         * are more fine grained. The offsets of the hubbard wave functions are
         * for atom while here they are for each atomic level. Since all atomic
         * level of a given atom are next to each other, the offset of the first
         * atomic level of a given atom has the same value than the offset
         * giving the position of the first hubbard wave function of this
         * atom. */
        offset_ = std::vector<int>(num_atomic_level_, 0);

        int size__ = 0;
        for (int at_lvl = 0; at_lvl < static_cast<int>(local_.size()); at_lvl++) {
            offset_[at_lvl] = size__;
            const int ia    = atomic_orbitals_[at_lvl].first;
            auto& atom_type = ctx_.unit_cell().atom(ia).type();
            int lo_ind      = atomic_orbitals_[at_lvl].second;
            const int l     = atom_type.lo_descriptor_hub(lo_ind).l;
            local_[at_lvl] = sddk::mdarray<double_complex, 3>(2 * l + 1, 2 * l + 1, 4, memory_t::host, "local_hubbard");
            local_[at_lvl].zero();
            size__ += (2 * l + 1);
        }

        nonlocal_.clear();
        nonlocal_ = std::vector<sddk::mdarray<double_complex, 3>>(ctx_.cfg().hubbard().nonlocal().size());
        for (int i = 0; i < static_cast<int>(ctx_.cfg().hubbard().nonlocal().size()); i++) {
            auto nl      = ctx_.cfg().hubbard().nonlocal(i);
            int il       = nl.l()[0];
            int jl       = nl.l()[1];
            nonlocal_[i] = sddk::mdarray<double_complex, 3>(2 * il + 1, 2 * jl + 1, 4);
            nonlocal_[i].zero();
        }
    }
}

void
Hubbard_matrix::access(std::string const& what__, double_complex* occ__, int ld__)
{
    if (!(what__ == "get" || what__ == "set")) {
        std::stringstream s;
        s << "wrong access label: " << what__;
        RTE_THROW(s);
    }

    sddk::mdarray<double_complex, 4> occ_mtrx;
    /* in non-collinear case the occupancy matrix is complex */
    if (ctx_.num_mag_dims() == 3) {
        occ_mtrx = sddk::mdarray<double_complex, 4>(occ__, ld__, ld__, 4, ctx_.unit_cell().num_atoms());
    } else {
        occ_mtrx = sddk::mdarray<double_complex, 4>(occ__, ld__, ld__, ctx_.num_spins(), ctx_.unit_cell().num_atoms());
    }
    if (what__ == "get") {
        occ_mtrx.zero();
    }

    for (int at_lvl = 0; at_lvl < static_cast<int>(local().size()); at_lvl++) {
        const int ia1    = atomic_orbitals(at_lvl).first;
        const auto& atom = ctx_.unit_cell().atom(ia1);
        const int lo     = atomic_orbitals(at_lvl).second;
        if (atom.type().lo_descriptor_hub(lo).use_for_calculation()) {
            const int l      = atom.type().lo_descriptor_hub(lo).l;
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

    auto draw_bar = [&](int w) { out__ << std::setfill('-') << std::setw(w) << '-' << std::setfill(' ') << std::endl; };
    auto print_number = [&](double x) { out__ << std::setw(width) << std::setprecision(prec) << std::fixed << x; };
    auto const& atom  = ctx_.unit_cell().atom(atomic_orbitals_[at_lvl__].first);

    out__ << "level : " << atom.type().lo_descriptor_hub(atomic_orbitals_[at_lvl__].second).n();
    out__ << " l: " << atom.type().lo_descriptor_hub(atomic_orbitals_[at_lvl__].second).l << std::endl;
    const int l = atom.type().lo_descriptor_hub(atomic_orbitals_[at_lvl__].second).l;
    if (ctx_.num_mag_dims() != 3) {
        int mmax = 2 * l + 1;
        for (int is = 0; is < ctx_.num_spins(); is++) {
            draw_bar(width * mmax);
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
        draw_bar(width * mmax);
    } else {
        int mmax = 2 * l + 1;
        draw_bar(2 * width * mmax + 3);
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
        draw_bar(2 * width * mmax + 3);
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
        draw_bar(2 * width * mmax + 3);
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
    vector3d<int> T(nl.T());

    out__ << "atom: " << ia << ", l: " << il << " -> atom: " << ja << ", l: " << jl << ", T: " << T << std::endl;

    // auto const& atom = ctx_.unit_cell().atom(ia__);
    // if (!atom.type().hubbard_correction()) {
    //    return;
    //}
    int const prec{5};
    int const width{10};
    auto draw_bar = [&](int w) { out__ << std::setfill('-') << std::setw(w) << '-' << std::setfill(' ') << std::endl; };
    auto print_number = [&](double x) { out__ << std::setw(width) << std::setprecision(prec) << std::fixed << x; };

    if (ctx_.num_mag_dims() != 3) {
        for (int is = 0; is < ctx_.num_spins(); is++) {
            draw_bar(width * jb);
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
        draw_bar(width * jb);
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
}

} // namespace sirius
