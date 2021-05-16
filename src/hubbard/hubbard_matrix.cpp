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

#include "hubbard_matrix.hpp"

namespace sirius {

Hubbard_matrix::Hubbard_matrix(Simulation_context& ctx__)
    : ctx_(ctx__)
{
    if (!ctx_.full_potential() && ctx_.hubbard_correction()) {

        int indexb_max = -1;

        // TODO: move detection of indexb_max to unit_cell
        // Don't forget that Hubbard class has the same code
        for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
            if (ctx_.unit_cell().atom(ia).type().hubbard_correction()) {
                indexb_max = std::max(indexb_max, static_cast<int>(ctx_.unit_cell().atom(ia).type().indexb_hub().size()));
            }
        }

        // TODO: work on the general definition of the occupation matrix with offsite terms
        // store it as list of small matrices, thewn indexb_max is not needed
        data_ = sddk::mdarray<double_complex, 4>(indexb_max, indexb_max, 4, ctx_.unit_cell().num_atoms(),
                memory_t::host, "Hubbard_matrix.data_");
        data_.zero();
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
                            occ_mtrx(l + m1, l + m2, j, ia) = data_(l + m1, l + m2, j, ia);
                        }
                    } else {
                        for (int j = 0; j < ((ctx_.num_mag_dims() == 3) ? 4 : ctx_.num_spins()); j++) {
                            data_(l + m1, l + m2, j, ia) = occ_mtrx(l + m1, l + m2, j, ia);
                        }
                    }
                }
            }
        }
    }
}

void Hubbard_matrix::print(int min_verbosity__) const
{
    if (ctx_.verbosity() >= min_verbosity__ && ctx_.comm().rank() == 0 && data_.size()) {
        std::printf("\n");
        for (int ci = 0; ci < 10; ci++) {
            std::printf("--------");
        }
        std::printf("\n");
        //std::printf("hubbard occupancies\n");
        for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
            std::printf("Atom : %d\n", ia);
            std::printf("Mag Dim : %d\n", ctx_.num_mag_dims());
            const auto& atom = ctx_.unit_cell().atom(ia);

            if (atom.type().hubbard_correction()) {
                const int lmax_at = 2 * atom.type().lo_descriptor_hub(0).l + 1;
                for (int m1 = 0; m1 < lmax_at; m1++) {
                    for (int m2 = 0; m2 < lmax_at; m2++) {
                        std::printf("%8.3f ", std::abs(this->data_(m1, m2, 0, ia)));
                    }

                    if (ctx_.num_mag_dims() == 3) {
                        std::printf(" ");
                        for (int m2 = 0; m2 < lmax_at; m2++) {
                            std::printf("%8.3f ", std::abs(this->data_(m1, m2, 2, ia)));
                        }
                    }
                    std::printf("\n");
                }

                if (ctx_.num_spins() == 2) {
                    for (int m1 = 0; m1 < lmax_at; m1++) {
                        if (ctx_.num_mag_dims() == 3) {
                            for (int m2 = 0; m2 < lmax_at; m2++) {
                                std::printf("%8.3f ", std::abs(this->data_(m1, m2, 3, ia)));
                            }
                            std::printf(" ");
                        }
                        for (int m2 = 0; m2 < lmax_at; m2++) {
                            std::printf("%8.3f ", std::abs(this->data_(m1, m2, 1, ia)));
                        }
                        std::printf("\n");
                    }
                }

                //double n_up, n_down, n_total;
                //n_up   = 0.0;
                //n_down = 0.0;
                //for (int m1 = 0; m1 < lmax_at; m1++) {
                //    n_up += this->data_(m1, m1, 0, ia).real();
                //}

                //if (ctx_.num_spins() == 2) {
                //    for (int m1 = 0; m1 < lmax_at; m1++) {
                //        n_down += this->data_(m1, m1, 1, ia).real();
                //    }
                //}
                //std::printf("\n");
                //n_total = n_up + n_down;
                //if (ctx_.num_spins() == 2) {
                //    std::printf("Atom charge (total) %.5lf (n_up) %.5lf (n_down) %.5lf (mz) %.5lf\n", n_total, n_up, n_down, n_up - n_down);
                //} else {
                //    std::printf("Atom charge (total) %.5lf\n", 2.0 * n_total);
                //}

                std::printf("\n");
                for (int ci = 0; ci < 10; ci++) {
                    std::printf("--------");
                }
                std::printf("\n");
            }
        }
    }
}



}

