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

        local_ = std::vector<sddk::mdarray<double_complex, 3>>(ctx_.unit_cell().num_atoms());

        for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
            if (ctx_.unit_cell().atom(ia).type().hubbard_correction()) {
                int nb = ctx_.unit_cell().atom(ia).type().indexb_hub().size();
                local_[ia] = sddk::mdarray<double_complex, 3>(nb, nb, 4);
                local_[ia].zero();
            }
        }
    }
}

void Hubbard_matrix::access(std::string const& what__, double_complex* occ__, int ld__) {
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

    for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
        auto& atom = ctx_.unit_cell().atom(ia);
        if (atom.type().hubbard_correction()) {
            const int l = ctx_.unit_cell().atom(ia).type().indexr_hub().am(0).l();
            for (int m1 = -l; m1 <= l; m1++) {
                for (int m2 = -l; m2 <= l; m2++) {
                    if (what__ == "get") {
                        for (int j = 0; j < ((ctx_.num_mag_dims() == 3) ? 4 : ctx_.num_spins()); j++) {
                            occ_mtrx(l + m1, l + m2, j, ia) = this->local(ia)(l + m1, l + m2, j);
                        }
                    } else {
                        for (int j = 0; j < ((ctx_.num_mag_dims() == 3) ? 4 : ctx_.num_spins()); j++) {
                            this->local(ia)(l + m1, l + m2, j) = occ_mtrx(l + m1, l + m2, j, ia);
                        }
                    }
                }
            }
        }
    }
}

void Hubbard_matrix::print_local(int ia__, std::ostream& out__) const
{
    auto const& atom = ctx_.unit_cell().atom(ia__);
    if (!atom.type().hubbard_correction()) {
        return;
    }
    int const prec{5};
    int const width{10};
    auto draw_bar = [&](int w)
    {
        out__ << std::setfill('-') << std::setw(w) << '-' << std::setfill(' ') << std::endl;
    };
    auto print_number = [&](double x)
    {
        out__ << std::setw(width) << std::setprecision(prec) << std::fixed << x;
    };

    if (ctx_.num_mag_dims() != 3) {
        int l = atom.type().indexr_hub().am(0).l();
        int mmax = 2 *l + 1;
        for (int is = 0; is < ctx_.num_spins(); is++) {
            draw_bar(width * mmax);
            bool has_imag{false};
            for (int m = 0; m < mmax; m++) {
                for (int mp = 0; mp < mmax; mp++) {
                    if (std::abs(std::imag(this->local(ia__)(m, mp, is))) > 1e-12) {
                        has_imag = true;
                    }
                    print_number(std::real(this->local(ia__)(m, mp, is)));
                }
                out__ << std::endl;
            }
            if (has_imag) {
                out__ << "imaginary part:" << std::endl;
                for (int m = 0; m < mmax; m++) {
                    for (int mp = 0; mp < mmax; mp++) {
                        print_number(std::imag(this->local(ia__)(m, mp, is)));
                    }
                    out__ << std::endl;
                }
            }
        }
        draw_bar(width * mmax);
    } else {
        int l = atom.type().indexr_hub().am(0).l();
        int mmax = 2 *l + 1;
        draw_bar(2 * width * mmax + 3);
        for (int m = 0; m < mmax; m++) {
            for (int mp = 0; mp < mmax; mp++) {
                print_number(std::real(this->local(ia__)(m, mp, 0)));
            }
            out__ << " | ";
            for (int mp = 0; mp < mmax; mp++) {
                print_number(std::real(this->local(ia__)(m, mp, 2)));
            }
            out__ << std::endl;
        }
        draw_bar(2 * width * mmax + 3);
        for (int m = 0; m < mmax; m++) {
            for (int mp = 0; mp < mmax; mp++) {
                print_number(std::real(this->local(ia__)(m, mp, 3)));
            }
            out__ << " | ";
            for (int mp = 0; mp < mmax; mp++) {
                print_number(std::real(this->local(ia__)(m, mp, 1)));
            }
            out__ << std::endl;
        }
        draw_bar(2 * width * mmax + 3);
    }
}

}

