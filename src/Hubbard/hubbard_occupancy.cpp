// Copyright (c) 2013-2018 Mathieu Taillefumier, Anton Kozhevnikov, Thomas Schulthess
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

/** \file hubbard_occupancy.hpp
 *
 *  \brief Generate occupation matrix for Hubbard orbitals.
 */

/** Compute the occupation numbers associated to the hubbard wavefunctions (locally centered orbitals, wannier
 *  functions, etc) that are relevant for the hubbard correction.
 *
 * These quantities are defined by
 * \f[
 *    n_{m,m'}^I \sigma = \sum_{kv} f(\varepsilon_{kv}) |<\psi_{kv}| phi_I_m>|^2
 * \f]
 * where \f[m=-l\cdot l$ (same for m')\f], I is the atom.
 *
 * Requires symmetrization. */

#include "hubbard.hpp"
#include "Symmetry/symmetrize.hpp"

namespace sirius {
void Hubbard::hubbard_compute_occupation_numbers(K_point_set& kset_)
{
    PROFILE("sirius::Hubbard::compute_occupancy");

    if (!ctx_.hubbard_correction()) {
        return;
    }

    this->occupancy_number_.zero();

    int HowManyBands = -1;
    /* if we are doing calculations for non colinear magnetism or
       simple LDA then do not change the number of bands. the factor
       two is important for colinear magnetism since the up-up and
       down-down blocks are decoupled but the wave-functions are up
       and down are still stored as a spinor to conserve space. */
    for (int ikloc = 0; ikloc < kset_.spl_num_kpoints().local_size(); ikloc++) {
        int  ik = kset_.spl_num_kpoints(ikloc);
        auto kp = kset_[ik];

        HowManyBands = std::max(kp->num_occupied_bands(0), HowManyBands);
        if (ctx_.num_mag_dims() == 1) {
            HowManyBands = std::max(kp->num_occupied_bands(1), HowManyBands);
        }
    }

    /* now for each spin components and each atom we need to calculate
       <psi_{nk}|phi^I_m'><phi^I_m|psi_{nk}> */
    int Ncf = 1;
    if (ctx_.num_mag_dims() == 1) {
        /* !!! colinear magnetism. We need to have 2 times more space. */
        Ncf = 2;
    }

    dmatrix<double_complex> dm(HowManyBands, this->number_of_hubbard_orbitals() * Ncf);
    matrix<double_complex>  dm1(HowManyBands, this->number_of_hubbard_orbitals() * Ncf);
    matrix<double_complex>  Op(this->number_of_hubbard_orbitals() * Ncf, this->number_of_hubbard_orbitals() * Ncf);

    dm.zero();

    if (ctx_.processing_unit() == device_t::GPU) {
        /* the communicator is always of size 1.  I need to allocate memory
           on the device manually */
        dm.allocate(memory_t::device);
    }
    memory_t mem{memory_t::host};
    linalg_t la{linalg_t::blas};
    if (ctx_.processing_unit() == device_t::GPU) {
        mem = memory_t::device;
        la = linalg_t::gpublas;
    }

    for (int ikloc = 0; ikloc < kset_.spl_num_kpoints().local_size(); ikloc++) {
        int  ik = kset_.spl_num_kpoints(ikloc);
        auto kp = kset_[ik];
        if (ctx_.processing_unit() == device_t::GPU) {
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                /* allocate GPU memory */
                kp->spinor_wave_functions().pw_coeffs(ispn).prime().allocate(memory_t::device);
                kp->spinor_wave_functions().pw_coeffs(ispn).copy_to(memory_t::device, 0, kp->num_occupied_bands(ispn));
            }

            for (int ispn = 0; ispn < kp->hubbard_wave_functions().num_sc(); ispn++) {

                if (!kp->hubbard_wave_functions().pw_coeffs(ispn).prime().on_device()) {
                    kp->hubbard_wave_functions().pw_coeffs(ispn).prime().allocate(memory_t::device);
                }

                kp->hubbard_wave_functions().pw_coeffs(ispn).copy_to(memory_t::device, 0, this->number_of_hubbard_orbitals());
            }
        }

        dm.zero();

        if (ctx_.num_mag_dims() == 3) {
            /* full non colinear magnetism */
            inner(mem, la, 2, kp->spinor_wave_functions(), 0, kp->num_occupied_bands(), kp->hubbard_wave_functions(), 0,
                  this->number_of_hubbard_orbitals(), dm, 0, 0);
        } else {
            /* SLDA + U, we need to do the explicit calculation. The hubbard
               orbitals only have one component while the bloch wave functions
               have two. The inner product takes care of this case internally. */

            for (int ispn_ = 0; ispn_ < ctx_.num_spins(); ispn_++) {
                inner(mem, la, ispn_, kp->spinor_wave_functions(), 0, kp->num_occupied_bands(ispn_),
                      kp->hubbard_wave_functions(), 0, this->number_of_hubbard_orbitals(), dm, 0,
                      ispn_ * this->number_of_hubbard_orbitals());
            }
        }

        if (ctx_.processing_unit() == device_t::GPU) {
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                /* deallocate GPU memory */
                kp->spinor_wave_functions().pw_coeffs(ispn).deallocate(memory_t::device);
            }

            for (int ispn = 0; ispn < kp->hubbard_wave_functions().num_sc(); ispn++) {
                kp->hubbard_wave_functions().pw_coeffs(ispn).deallocate(memory_t::device);
            }

            dm.copy_to(memory_t::host);
        }

        // compute O'_{nk,j} = O_{nk,j} * f_{nk}
        // NO summation over band yet

        dm1.zero(); // O'

        if (ctx_.num_mag_dims() == 3) {
#pragma omp parallel for
            for (int m = 0; m < this->number_of_hubbard_orbitals(); m++) {
                for (int nband = 0; nband < kp->num_occupied_bands(0); nband++) {
                    dm1(nband, m) = dm(nband, m) * kp->band_occupancy(nband, 0);
                }
            }
        } else {
#pragma omp parallel for
            for (int m = 0; m < this->number_of_hubbard_orbitals(); m++) {
                for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                    for (int nband = 0; nband < kp->num_occupied_bands(ispn); nband++) {
                        dm1(nband, ispn * this->number_of_hubbard_orbitals() + m) =
                            dm(nband, ispn * this->number_of_hubbard_orbitals() + m) * kp->band_occupancy(nband, ispn);
                    }
                }
            }
        }

        // now compute O_{ij}^{sigma,sigma'} = \sum_{nk} <psi_nk|phi_{i,sigma}><phi_{j,sigma^'}|psi_nk> f_{nk}
        const double scal = (ctx_.num_mag_dims() == 0) ? 0.5 : 1.0;
        auto alpha = double_complex(kp->weight() * scal, 0.0);
        linalg(linalg_t::blas).gemm('C', 'N', this->number_of_hubbard_orbitals() * Ncf,
            this->number_of_hubbard_orbitals() * Ncf, HowManyBands, &alpha, dm.at(memory_t::host), dm.ld(),
            dm1.at(memory_t::host), dm1.ld(), &linalg_const<double_complex>::zero(), Op.at(memory_t::host), Op.ld());

        if (ctx_.num_mag_dims() == 3) {
            // there must be a way to do that with matrix multiplication
            #pragma omp parallel for schedule(static)
            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                const auto& atom = unit_cell_.atom(ia);
                if (atom.type().hubbard_correction()) {

                    /* loop over the different channels */
                    /* note that for atom with SO interactions, we need to jump
                       by 2 instead of 1. This is due to the fact that the
                       relativistic wave functions have different total angular
                       momentum for the same n */

                    for (int orb = 0; orb < atom.type().num_hubbard_orbitals(); orb += (atom.type().spin_orbit_coupling() ? 2 : 1)) {
                        /*
                           I know that the index of the hubbard wave functions (indexb_....) is
                           consistent with the index of the hubbard orbitals
                        */
                        const int lmax_at = 2 * atom.type().hubbard_orbital(0).l + 1;
                        for (int s1 = 0; s1 < ctx_.num_spins(); s1++) {
                            for (int s2 = 0; s2 < ctx_.num_spins(); s2++) {
                                int s = (s1 == s2) * s1 + (s1 != s2) * (1 + 2 * s2 + s1);
                                for (int mp = 0; mp < lmax_at; mp++) {
                                    for (int m = 0; m < lmax_at; m++) {
                                        this->occupancy_number_(m, mp, s, ia) +=
                                            Op(this->offset_[ia] + m + s1 * lmax_at, this->offset_[ia] + mp + s2 * lmax_at);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // Well we need to apply a factor 1/2 (the constant scal
            // above) when we compute the occupancies for the boring LDA
            // + U. It is because the calculations of E and U consider
            // occupancies <= 1.  Sirius for the boring lda+U has a
            // factor 2 in the kp band occupancies. We need to
            // compensate for it because it is taken into account in the
            // calculation of the hubbard potential

            #pragma omp parallel for schedule(static)
            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                const auto& atom = unit_cell_.atom(ia);
                if (atom.type().hubbard_correction()) {
                    for (int orb = 0; orb < atom.type().num_hubbard_orbitals(); orb++) {
                        const int lmax_at = 2 * atom.type().hubbard_orbital(0).l + 1;
                        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                            for (int mp = 0; mp < lmax_at; mp++) {
                                const int mmp = this->offset_[ia] + mp + ispn * this->number_of_hubbard_orbitals();
                                for (int m = 0; m < lmax_at; m++) {
                                    const int mm = this->offset_[ia] + m + ispn * this->number_of_hubbard_orbitals();
                                    this->occupancy_number_(m, mp, ispn, ia) += Op(mm, mmp);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /* global reduction over k points */
    ctx_.comm_k().allreduce<double_complex, mpi_op_t::sum>(this->occupancy_number_.at(memory_t::host),
                                                           static_cast<int>(this->occupancy_number_.size()));

    // Now symmetrization procedure. We need to review that
    //symmetrize_occupancy_matrix();

    print_occupancies();
}

// The initial occupancy is calculated following Hund rules. We first
// fill the d (f) states according to the hund's rules and with majority
// spin first and the remaining electrons distributed among the minority
// states.
void
Hubbard::calculate_initial_occupation_numbers()
{
    this->occupancy_number_.zero();
#pragma omp parallel for schedule(static)
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        const auto& atom = unit_cell_.atom(ia);
        if (atom.type().hubbard_correction()) {
            const int lmax_at = 2 * atom.type().hubbard_orbital(0).l + 1;
            // compute the total charge for the hubbard orbitals
            double charge = atom.type().hubbard_orbital(0).occupancy();
            bool   nm     = true; // true if the atom is non magnetic
            int    majs, mins;
            if (ctx_.num_spins() != 1) {
                if (atom.vector_field()[2] > 0.0) {
                    nm   = false;
                    majs = 0;
                    mins = 1;
                } else if (atom.vector_field()[2] < 0.0) {
                    nm   = false;
                    majs = 1;
                    mins = 0;
                }
            }

            if (!nm) {
                if (ctx_.num_mag_dims() != 3) {
                    // colinear case

                    if (charge > (lmax_at)) {
                        for (int m = 0; m < lmax_at; m++) {
                            this->occupancy_number_(m, m, majs, ia) = 1.0;
                            this->occupancy_number_(m, m, mins, ia) =
                                (charge - static_cast<double>(lmax_at)) / static_cast<double>(lmax_at);
                        }
                    } else {
                        for (int m = 0; m < lmax_at; m++) {
                            this->occupancy_number_(m, m, majs, ia) = charge / static_cast<double>(lmax_at);
                        }
                    }
                } else {
                    // double c1, s1;
                    // sincos(atom.type().starting_magnetization_theta(), &s1, &c1);
                    double         c1 = atom.vector_field()[2];
                    double_complex cs = double_complex(atom.vector_field()[0], atom.vector_field()[1]) / sqrt(1.0 - c1 * c1);
                    double_complex ns[4];

                    if (charge > (lmax_at)) {
                        ns[majs] = 1.0;
                        ns[mins] = (charge - static_cast<double>(lmax_at)) / static_cast<double>(lmax_at);
                    } else {
                        ns[majs] = charge / static_cast<double>(lmax_at);
                        ns[mins] = 0.0;
                    }

                    // charge and moment
                    double nc  = ns[majs].real() + ns[mins].real();
                    double mag = ns[majs].real() - ns[mins].real();

                    // rotate the occ matrix
                    ns[0] = (nc + mag * c1) * 0.5;
                    ns[1] = (nc - mag * c1) * 0.5;
                    ns[2] = mag * std::conj(cs) * 0.5;
                    ns[3] = mag * cs * 0.5;

                    for (int m = 0; m < lmax_at; m++) {
                        this->occupancy_number_(m, m, 0, ia) = ns[0];
                        this->occupancy_number_(m, m, 1, ia) = ns[1];
                        this->occupancy_number_(m, m, 2, ia) = ns[2];
                        this->occupancy_number_(m, m, 3, ia) = ns[3];
                    }
                }
            } else {
                for (int s = 0; s < ctx_.num_spins(); s++) {
                    for (int m = 0; m < lmax_at; m++) {
                        this->occupancy_number_(m, m, s, ia) = charge * 0.5 / static_cast<double>(lmax_at);
                    }
                }
            }
        }
    }

    print_occupancies();
}

void Hubbard::print_occupancies()
{
    if (ctx_.control().verbosity_ > 1 && ctx_.comm().rank() == 0) {
        std::printf("\n");
        for (int ci = 0; ci < 10; ci++) {
            std::printf("--------");
        }
        std::printf("\n");
        std::printf("hubbard occupancies\n");
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            std::printf("Atom : %d\n", ia);
            std::printf("Mag Dim : %d\n", ctx_.num_mag_dims());
            const auto& atom = unit_cell_.atom(ia);

            if (atom.type().hubbard_correction()) {
                const int lmax_at = 2 * atom.type().hubbard_orbital(0).l + 1;
                for (int m1 = 0; m1 < lmax_at; m1++) {
                    for (int m2 = 0; m2 < lmax_at; m2++) {
                        std::printf("%.3lf ", std::abs(this->occupancy_number_(m1, m2, 0, ia)));
                    }

                    if (ctx_.num_mag_dims() == 3) {
                        std::printf(" ");
                        for (int m2 = 0; m2 < lmax_at; m2++) {
                            std::printf("%.3lf ", std::abs(this->occupancy_number_(m1, m2, 2, ia)));
                        }
                    }
                    std::printf("\n");
                }

                if (ctx_.num_spins() == 2) {
                    for (int m1 = 0; m1 < lmax_at; m1++) {
                        if (ctx_.num_mag_dims() == 3) {
                            for (int m2 = 0; m2 < lmax_at; m2++) {
                                std::printf("%.3lf ", std::abs(this->occupancy_number_(m1, m2, 3, ia)));
                            }
                            std::printf(" ");
                        }
                        for (int m2 = 0; m2 < lmax_at; m2++) {
                            std::printf("%.3lf ", std::abs(this->occupancy_number_(m1, m2, 1, ia)));
                        }
                        std::printf("\n");
                    }
                }

                double n_up, n_down, n_total;
                n_up   = 0.0;
                n_down = 0.0;
                for (int m1 = 0; m1 < lmax_at; m1++) {
                    n_up += this->occupancy_number_(m1, m1, 0, ia).real();
                }

                if (ctx_.num_spins() == 2) {
                    for (int m1 = 0; m1 < lmax_at; m1++) {
                        n_down += this->occupancy_number_(m1, m1, 1, ia).real();
                    }
                }
                std::printf("\n");
                n_total = n_up + n_down;
                if (ctx_.num_spins() == 2) {
                    std::printf("Atom charge (total) %.5lf (n_up) %.5lf (n_down) %.5lf (mz) %.5lf\n", n_total, n_up, n_down, n_up - n_down);
                } else {
                    std::printf("Atom charge (total) %.5lf\n", 2.0 * n_total);
                }

                std::printf("\n");
                for (int ci = 0; ci < 10; ci++) {
                    std::printf("--------");
                }
                std::printf("\n");
            }
        }
    }
}

void Hubbard::symmetrize_occupancy_matrix()
{
    auto& sym = unit_cell_.symmetry();

    // check if we have some symmetries
    if (sym.num_mag_sym()) {
        int lmax  = unit_cell_.lmax();
        int lmmax = utils::lmmax(lmax);

        mdarray<double, 2> rotm(lmmax, lmmax);
        mdarray<double_complex, 4> dm_(occupancy_number_.size(0),
                                       occupancy_number_.size(1),
                                       occupancy_number_.size(2),
                                       unit_cell_.num_atoms());
        double alpha = 1.0 / static_cast<double>(sym.num_mag_sym());

        dm_.zero();

        for (int i = 0; i < sym.num_mag_sym(); i++) {
            int  pr   = sym.magnetic_group_symmetry(i).spg_op.proper;
            auto eang = sym.magnetic_group_symmetry(i).spg_op.euler_angles;
            int isym  = sym.magnetic_group_symmetry(i).isym;
            SHT::rotation_matrix(lmax, eang, pr, rotm);
            auto spin_rot_su2 = rotation_matrix_su2(sym.magnetic_group_symmetry(i).spin_rotation);

            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                auto& atom_type = unit_cell_.atom(ia).type();
                int   ja        = sym.sym_table(ia, isym);
                if (atom_type.hubbard_correction()) {
                    sirius::symmetrize(occupancy_number_, unit_cell_.atom(ia).type().hubbard_indexb_wfc(), ia, ja,
                                       ctx_.num_mag_comp(), rotm, spin_rot_su2, dm_, true);
                }
            }
        }

        for (auto d3 = 0u; d3 < dm_.size(3); d3++) {
            for(auto d1 = 0u; d1 < dm_.size(1); d1++) {
                for(auto d0 = 0u; d0 < dm_.size(0); d0++) {
                    dm_(d0, d1, 0, d3) = dm_(d0, d1, 0, d3) * alpha;
                    dm_(d0, d1, 1, d3) = dm_(d0, d1, 1, d3) * alpha;
                    dm_(d0, d1, 2, d3) = std::conj(dm_(d0, d1, 2, d3))  * alpha;
                        dm_(d0, d1, 3, d3) = dm_(d0, d1, 2, d3) * alpha;
                }
            }
        }
        dm_ >> occupancy_number_;
    }
}


/**
 * retrieve or initialize the hubbard occupancies
 *
 * this functions helps retrieving or setting up the hubbard occupancy
 * tensors from an external tensor. Retrieving it is done by specifying
 * "get" in the first argument of the method while setting it is done
 * with the parameter set up to "set". The second parameter is the
 * output pointer and the last parameter is the leading dimension of the
 * tensor.
 *
 * The returned result has the same layout than SIRIUS layout, * i.e.,
 * the harmonic orbitals are stored from m_z = -l..l. The occupancy
 * matrix can also be accessed through the method occupation_matrix()
 *
 *
 * @param what__ string to set to "set" for initializing sirius
 * occupancy tensor and "get" for retrieving it
 * @param pointer to external occupancy tensor
 * @param leading dimension of the outside tensor
 * @return
 * return the occupancy matrix if the first parameter is set to "get"
 */

void
Hubbard::access_hubbard_occupancies(char const*     what__,
                                    double_complex* occ__,
                                    int const*      ld__)
{
    std::string what(what__);

    if (!(what == "get" || what == "set")) {
        std::stringstream s;
        s << "wrong access label: " << what;
        TERMINATE(s);
    }

    mdarray<double_complex, 4> occ_mtrx;
    /* in non-collinear case the occupancy matrix is complex */
    if (ctx_.num_mag_dims() == 3) {
        occ_mtrx = mdarray<double_complex, 4>(reinterpret_cast<double_complex*>(occ__), *ld__, *ld__, 4, ctx_.unit_cell().num_atoms());
    } else {
        occ_mtrx = mdarray<double_complex, 4>(reinterpret_cast<double_complex*>(occ__), *ld__, *ld__, ctx_.num_spins(), ctx_.unit_cell().num_atoms());
    }
    if (what == "get") {
        occ_mtrx.zero();
    }

    auto& occupation_matrix = this->occupation_matrix();

    for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
        auto& atom = ctx_.unit_cell().atom(ia);
        if (atom.type().hubbard_correction()) {
            const int l = ctx_.unit_cell().atom(ia).type().hubbard_orbital(0).l;
            for (int m1 = -l; m1 <= l; m1++) {
                for (int m2 = -l; m2 <= l; m2++) {
                    if (what == "get") {
                        for (int j = 0; j < ((ctx_.num_mag_dims() == 3) ? 4 : ctx_.num_spins()); j++) {
                            occ_mtrx(l + m1, l + m2, j, ia) = occupation_matrix(l + m1, l + m2, j, ia);
                        }
                    } else {
                        for (int j = 0; j < ((ctx_.num_mag_dims() == 3) ? 4 : ctx_.num_spins()); j++) {
                            occupation_matrix(l + m1, l + m2, j, ia) = occ_mtrx(l + m1, l + m2, j, ia);
                        }
                    }
                }
            }
        }
    }
}
}
