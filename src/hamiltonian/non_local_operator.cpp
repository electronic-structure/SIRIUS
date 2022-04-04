// Copyright (c) 2013-2019 Anton Kozhevnikov, Thomas Schulthess
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

/** \file non_local_operator.cpp
 *
 *  \brief Contains implementation of sirius::Non_local_operator class.
 */

#include "SDDK/wf_inner.hpp"
#include "SDDK/wf_trans.hpp"
#include "non_local_operator.hpp"
#include "hubbard/hubbard_matrix.hpp"

namespace sirius {

template <typename T>
Non_local_operator<T>::Non_local_operator(Simulation_context const& ctx__)
    : ctx_(ctx__)
{
    PROFILE("sirius::Non_local_operator");

    pu_                 = this->ctx_.processing_unit();
    auto& uc            = this->ctx_.unit_cell();
    packed_mtrx_offset_ = sddk::mdarray<int, 1>(uc.num_atoms());
    packed_mtrx_size_   = 0;
    for (int ia = 0; ia < uc.num_atoms(); ia++) {
        int nbf                 = uc.atom(ia).mt_basis_size();
        packed_mtrx_offset_(ia) = packed_mtrx_size_;
        packed_mtrx_size_ += nbf * nbf;
    }

    switch (pu_) {
        case device_t::GPU: {
            packed_mtrx_offset_.allocate(memory_t::device).copy_to(memory_t::device);
            break;
        }
        case device_t::CPU: {
            break;
        }
    }
}

template <typename T>
D_operator<T>::D_operator(Simulation_context const& ctx_)
    : Non_local_operator<T>(ctx_)
{
    if (ctx_.gamma_point()) {
        this->op_ = mdarray<T, 3>(1, this->packed_mtrx_size_, ctx_.num_mag_dims() + 1);
    } else {
        this->op_ = mdarray<T, 3>(2, this->packed_mtrx_size_, ctx_.num_mag_dims() + 1);
    }
    this->op_.zero();
    initialize();
}

template <typename T>
void
D_operator<T>::initialize()
{
    PROFILE("sirius::D_operator::initialize");

    auto& uc = this->ctx_.unit_cell();

    const int s_idx[2][2] = {{0, 3}, {2, 1}};

    #pragma omp parallel for
    for (int ia = 0; ia < uc.num_atoms(); ia++) {
        auto& atom = uc.atom(ia);
        int nbf    = atom.mt_basis_size();
        auto& dion = atom.type().d_mtrx_ion();

        /* in case of spin orbit coupling */
        if (uc.atom(ia).type().spin_orbit_coupling()) {
            mdarray<std::complex<T>, 3> d_mtrx_so(nbf, nbf, 4);
            d_mtrx_so.zero();

            /* transform the d_mtrx */
            for (int xi2 = 0; xi2 < nbf; xi2++) {
                for (int xi1 = 0; xi1 < nbf; xi1++) {

                    /* first compute \f[A^_\alpha I^{I,\alpha}_{xi,xi}\f] cf Eq.19 in doi:10.1103/PhysRevB.71.115106  */

                    /* note that the `I` integrals are already calculated and stored in atom.d_mtrx */
                    for (int sigma = 0; sigma < 2; sigma++) {
                        for (int sigmap = 0; sigmap < 2; sigmap++) {
                            std::complex<T> result(0, 0);
                            for (auto xi2p = 0; xi2p < nbf; xi2p++) {
                                if (atom.type().compare_index_beta_functions(xi2, xi2p)) {
                                    /* just sum over m2, all other indices are the same */
                                    for (auto xi1p = 0; xi1p < nbf; xi1p++) {
                                        if (atom.type().compare_index_beta_functions(xi1, xi1p)) {
                                            /* just sum over m1, all other indices are the same */

                                            /* loop over the 0, z,x,y coordinates */
                                            for (int alpha = 0; alpha < 4; alpha++) {
                                                for (int sigma1 = 0; sigma1 < 2; sigma1++) {
                                                    for (int sigma2 = 0; sigma2 < 2; sigma2++) {
                                                        result += atom.d_mtrx(xi1p, xi2p, alpha) *
                                                                  pauli_matrix[alpha][sigma1][sigma2] *
                                                                  atom.type().f_coefficients(xi1, xi1p, sigma, sigma1) *
                                                                  atom.type().f_coefficients(xi2p, xi2, sigma2, sigmap);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            d_mtrx_so(xi1, xi2, s_idx[sigma][sigmap]) = result;
                        }
                    }
                }
            }

            /* add ionic contribution */

            /* spin orbit coupling mixes terms */

            /* keep the order of the indices because it is crucial here;
               permuting the indices makes things wrong */
            for (int xi2 = 0; xi2 < nbf; xi2++) {
                int l2     = atom.type().indexb(xi2).l;
                double j2  = atom.type().indexb(xi2).j;
                int idxrf2 = atom.type().indexb(xi2).idxrf;
                for (int xi1 = 0; xi1 < nbf; xi1++) {
                    int l1     = atom.type().indexb(xi1).l;
                    double j1  = atom.type().indexb(xi1).j;
                    int idxrf1 = atom.type().indexb(xi1).idxrf;
                    if ((l1 == l2) && (std::abs(j1 - j2) < 1e-8)) {
                        /* up-up down-down */
                        d_mtrx_so(xi1, xi2, 0) += dion(idxrf1, idxrf2) * atom.type().f_coefficients(xi1, xi2, 0, 0);
                        d_mtrx_so(xi1, xi2, 1) += dion(idxrf1, idxrf2) * atom.type().f_coefficients(xi1, xi2, 1, 1);

                        /* up-down down-up */
                        d_mtrx_so(xi1, xi2, 2) += dion(idxrf1, idxrf2) * atom.type().f_coefficients(xi1, xi2, 0, 1);
                        d_mtrx_so(xi1, xi2, 3) += dion(idxrf1, idxrf2) * atom.type().f_coefficients(xi1, xi2, 1, 0);
                    }
                }
            }

            /* the pseudo potential contains information about
               spin orbit coupling so we use a different formula
               Eq.19 doi:10.1103/PhysRevB.71.115106 for calculating the D matrix

               Note that the D matrices are stored and
               calculated in the up-down basis already not the (Veff,Bx,By,Bz) one */
            for (int xi2 = 0; xi2 < nbf; xi2++) {
                for (int xi1 = 0; xi1 < nbf; xi1++) {
                    int idx = xi2 * nbf + xi1;
                    for (int s = 0; s < 4; s++) {
                        this->op_(0, this->packed_mtrx_offset_(ia) + idx, s) = d_mtrx_so(xi1, xi2, s).real();
                        this->op_(1, this->packed_mtrx_offset_(ia) + idx, s) = d_mtrx_so(xi1, xi2, s).imag();
                    }
                }
            }
        } else {
            /* No spin orbit coupling for this atom \f[D = D(V_{eff})
               I + D(B_x) \sigma_x + D(B_y) sigma_y + D(B_z)
               sigma_z\f] since the D matrices are calculated that way */
            for (int xi2 = 0; xi2 < nbf; xi2++) {
                int lm2    = atom.type().indexb(xi2).lm;
                int idxrf2 = atom.type().indexb(xi2).idxrf;
                for (int xi1 = 0; xi1 < nbf; xi1++) {
                    int lm1    = atom.type().indexb(xi1).lm;
                    int idxrf1 = atom.type().indexb(xi1).idxrf;

                    int idx = xi2 * nbf + xi1;
                    switch (this->ctx_.num_mag_dims()) {
                        case 3: {
                            T bx = uc.atom(ia).d_mtrx(xi1, xi2, 2);
                            T by = uc.atom(ia).d_mtrx(xi1, xi2, 3);

                            this->op_(0, this->packed_mtrx_offset_(ia) + idx, 2) = bx;
                            this->op_(1, this->packed_mtrx_offset_(ia) + idx, 2) = -by;

                            this->op_(0, this->packed_mtrx_offset_(ia) + idx, 3) = bx;
                            this->op_(1, this->packed_mtrx_offset_(ia) + idx, 3) = by;
                        }
                        case 1: {
                            T v  = uc.atom(ia).d_mtrx(xi1, xi2, 0);
                            T bz = uc.atom(ia).d_mtrx(xi1, xi2, 1);

                            /* add ionic part */
                            if (lm1 == lm2) {
                                v += dion(idxrf1, idxrf2);
                            }

                            this->op_(0, this->packed_mtrx_offset_(ia) + idx, 0) = v + bz;
                            this->op_(0, this->packed_mtrx_offset_(ia) + idx, 1) = v - bz;
                            break;
                        }
                        case 0: {
                            this->op_(0, this->packed_mtrx_offset_(ia) + idx, 0) = uc.atom(ia).d_mtrx(xi1, xi2, 0);
                            /* add ionic part */
                            if (lm1 == lm2) {
                                this->op_(0, this->packed_mtrx_offset_(ia) + idx, 0) += dion(idxrf1, idxrf2);
                            }
                            break;
                        }
                        default: {
                            TERMINATE("wrong number of magnetic dimensions");
                        }
                    }
                }
            }
        }
    }

    if (this->ctx_.print_checksum() && this->ctx_.comm().rank() == 0) {
        auto cs = this->op_.checksum();
        utils::print_checksum("D_operator", cs);
    }

    if (this->pu_ == device_t::GPU && uc.mt_lo_basis_size() != 0) {
        this->op_.allocate(memory_t::device).copy_to(memory_t::device);
    }

    /* D-operator is not diagonal in spin in case of non-collinear magnetism
       (spin-orbit coupling falls into this case) */
    if (this->ctx_.num_mag_dims() == 3) {
        this->is_diag_ = false;
    }
}

template <typename T>
Q_operator<T>::Q_operator(Simulation_context const& ctx__)
    : Non_local_operator<T>(ctx__)
{
    /* Q-operator is independent of spin if there is no spin-orbit; however, it simplifies the apply()
     * method if the Q-operator has a spin index */
    if (this->ctx_.gamma_point()) {
        this->op_ = mdarray<T, 3>(1, this->packed_mtrx_size_, this->ctx_.num_mag_dims() + 1);
    } else {
        this->op_ = mdarray<T, 3>(2, this->packed_mtrx_size_, this->ctx_.num_mag_dims() + 1);
    }
    this->op_.zero();
    initialize();
}

template <typename T>
void
Q_operator<T>::initialize()
{
    PROFILE("sirius::Q_operator::initialize");

    auto& uc = this->ctx_.unit_cell();

    #pragma omp parallel for
    for (int ia = 0; ia < uc.num_atoms(); ia++) {
        int iat = uc.atom(ia).type().id();
        if (!uc.atom_type(iat).augment()) {
            continue;
        }
        int nbf = uc.atom(ia).mt_basis_size();
        for (int xi2 = 0; xi2 < nbf; xi2++) {
            for (int xi1 = 0; xi1 < nbf; xi1++) {
                /* The ultra soft pseudo potential has spin orbit coupling incorporated to it, so we
                   need to rotate the Q matrix */
                if (uc.atom_type(iat).spin_orbit_coupling()) {
                    /* this is nothing else than Eq.18 of doi:10.1103/PhysRevB.71.115106 */
                    for (auto si = 0; si < 2; si++) {
                        for (auto sj = 0; sj < 2; sj++) {

                            std::complex<T> result(0, 0);

                            for (int xi2p = 0; xi2p < nbf; xi2p++) {
                                if (uc.atom(ia).type().compare_index_beta_functions(xi2, xi2p)) {
                                    for (int xi1p = 0; xi1p < nbf; xi1p++) {
                                        /* The F coefficients are already "block diagonal" so we do a full
                                           summation. We actually rotate the q_matrices only */
                                        if (uc.atom(ia).type().compare_index_beta_functions(xi1, xi1p)) {
                                            result += this->ctx_.augmentation_op(iat)->q_mtrx(xi1p, xi2p) *
                                                      (uc.atom(ia).type().f_coefficients(xi1, xi1p, sj, 0) *
                                                           uc.atom(ia).type().f_coefficients(xi2p, xi2, 0, si) +
                                                       uc.atom(ia).type().f_coefficients(xi1, xi1p, sj, 1) *
                                                           uc.atom(ia).type().f_coefficients(xi2p, xi2, 1, si));
                                        }
                                    }
                                }
                            }

                            /* the order of the index is important */
                            const int ind = (si == sj) ? si : sj + 2;
                            /* this gives
                               ind = 0 if si = up and sj = up
                               ind = 1 if si = sj = down
                               ind = 2 if si = down and sj = up
                               ind = 3 if si = up and sj = down */
                            this->op_(0, this->packed_mtrx_offset_(ia) + xi2 * nbf + xi1, ind) = result.real();
                            this->op_(1, this->packed_mtrx_offset_(ia) + xi2 * nbf + xi1, ind) = result.imag();
                        }
                    }
                } else {
                    for (int ispn = 0; ispn < this->ctx_.num_spins(); ispn++) {
                        this->op_(0, this->packed_mtrx_offset_(ia) + xi2 * nbf + xi1, ispn) =
                            this->ctx_.augmentation_op(iat)->q_mtrx(xi1, xi2);
                    }
                }
            }
        }
    }
    if (this->ctx_.print_checksum() && this->ctx_.comm().rank() == 0) {
        auto cs = this->op_.checksum();
        utils::print_checksum("Q_operator", cs);
    }

    if (this->pu_ == device_t::GPU && uc.mt_lo_basis_size() != 0) {
        this->op_.allocate(memory_t::device).copy_to(memory_t::device);
    }

    this->is_null_ = true;
    for (int iat = 0; iat < uc.num_atom_types(); iat++) {
        if (uc.atom_type(iat).augment()) {
            this->is_null_ = false;
        }
        /* Q-operator is not diagonal in spin only in the case of spin-orbit coupling */
        if (uc.atom_type(iat).spin_orbit_coupling()) {
            this->is_diag_ = false;
        }
    }
}

template <typename T>
void
apply_non_local_d_q(spin_range spins__, int N__, int n__, Beta_projectors<real_type<T>>& beta__,
                    Wave_functions<real_type<T>>& phi__, D_operator<real_type<T>>* d_op__,
                    Wave_functions<real_type<T>>* hphi__, Q_operator<real_type<T>>* q_op__,
                    Wave_functions<real_type<T>>* sphi__)
{

    for (int i = 0; i < beta__.num_chunks(); i++) {
        /* generate beta-projectors for a block of atoms */
        beta__.generate(i);

        for (int ispn : spins__) {
            auto beta_phi = beta__.template inner<T>(i, phi__, ispn, N__, n__);

            if (hphi__ && d_op__) {
                /* apply diagonal spin blocks */
                d_op__->apply(i, ispn, *hphi__, N__, n__, beta__, beta_phi);
                if (!d_op__->is_diag() && hphi__->num_sc() == 2) {
                    /* apply non-diagonal spin blocks */
                    /* xor 3 operator will map 0 to 3 and 1 to 2 */
                    d_op__->apply(i, ispn ^ 3, *hphi__, N__, n__, beta__, beta_phi);
                }
            }

            if (sphi__ && q_op__) {
                /* apply Q operator (diagonal in spin) */
                q_op__->apply(i, ispn, *sphi__, N__, n__, beta__, beta_phi);
                if (!q_op__->is_diag() && sphi__->num_sc() == 2) {
                    q_op__->apply(i, ispn ^ 3, *sphi__, N__, n__, beta__, beta_phi);
                }
            }
        }
    }
}

/// Compute |sphi> = (1 + Q)|phi>
template <typename T>
void
apply_S_operator(device_t pu__, spin_range spins__, int N__, int n__, Beta_projectors<real_type<T>>& beta__,
                 Wave_functions<real_type<T>>& phi__, Q_operator<real_type<T>>* q_op__,
                 Wave_functions<real_type<T>>& sphi__)
{
    for (auto s : spins__) {
        sphi__.copy_from(pu__, n__, phi__, s, N__, s, N__);
    }

    if (q_op__) {
        apply_non_local_d_q<T>(spins__, N__, n__, beta__, phi__, nullptr, nullptr, q_op__, &sphi__);
    }
}

template <typename T>
void
apply_U_operator(Simulation_context& ctx__, spin_range spins__, int N__, int n__, Wave_functions<T>& hub_wf__,
                 Wave_functions<T>& phi__, U_operator<T>& um__, Wave_functions<T>& hphi__)
{
    if (!ctx__.hubbard_correction()) {
        return;
    }

    dmatrix<std::complex<T>> dm(hub_wf__.num_wf(), n__);
    if (ctx__.processing_unit() == device_t::GPU) {
        dm.allocate(memory_t::device);
    }

    auto la = linalg_t::none;
    auto mt = memory_t::none;
    switch (ctx__.processing_unit()) {
        case device_t::CPU: {
            la = linalg_t::blas;
            mt = memory_t::host;
            break;
        }
        case device_t::GPU: {
            la = linalg_t::gpublas;
            mt = memory_t::device;
            break;
        }
        default:
            break;
    }
    /* First calculate the local part of the projections
       dm(i, n) = <phi_i| S |psi_{nk}> */
    sddk::inner(ctx__.spla_context(), spins__, hub_wf__, 0, hub_wf__.num_wf(), phi__, N__, n__, dm, 0, 0);

    dmatrix<std::complex<T>> Up(hub_wf__.num_wf(), n__);
    if (ctx__.processing_unit() == device_t::GPU) {
        Up.allocate(memory_t::device);
    }

    if (ctx__.num_mag_dims() == 3) {
        Up.zero();
        #pragma omp parallel for schedule(static)
        for (int at_lvl = 0; at_lvl < (int)um__.atomic_orbitals().size(); at_lvl++) {
            const int ia     = um__.atomic_orbitals(at_lvl).first;
            auto const& atom = ctx__.unit_cell().atom(ia);
            if (atom.type().lo_descriptor_hub(um__.atomic_orbitals(at_lvl).second).use_for_calculation()) {
                const int lmax_at = 2 * atom.type().lo_descriptor_hub(um__.atomic_orbitals(at_lvl).second).l() + 1;
                // we apply the hubbard correction. For now I have no papers
                // giving me the formula for the SO case so I rely on QE for it
                // but I do not like it at all
                for (int s1 = 0; s1 < ctx__.num_spins(); s1++) {
                    for (int s2 = 0; s2 < ctx__.num_spins(); s2++) {
                      // TODO: replace this with matrix matrix multiplication
                        for (int nbd = 0; nbd < n__; nbd++) {
                            for (int m1 = 0; m1 < lmax_at; m1++) {
                                for (int m2 = 0; m2 < lmax_at; m2++) {
                                    const int ind = (s1 == s2) * s1 + (1 + 2 * s2 + s1) * (s1 != s2);
                                    Up(um__.nhwf() * s1 + um__.offset(at_lvl) + m1, nbd) +=
                                      um__(um__.offset(at_lvl) + m2, um__.offset(at_lvl) + m1, ind) *
                                      dm(um__.nhwf() * s2 + um__.offset(at_lvl) + m2, nbd);
                                }
                            }
                        }
                    }
                }
            }
        }
        if (ctx__.processing_unit() == device_t::GPU) {
            Up.copy_to(memory_t::device);
        }
    } else {
        linalg(la).gemm('N', 'N', um__.nhwf(), n__, um__.nhwf(), &linalg_const<double_complex>::one(),
                        um__.at(mt, 0, 0, spins__()), um__.nhwf(), dm.at(mt, 0, 0), dm.ld(),
                        &linalg_const<double_complex>::zero(), Up.at(mt, 0, 0), Up.ld());
        if (ctx__.processing_unit() == device_t::GPU) {
            Up.copy_to(memory_t::host);
        }
    }
    transform<std::complex<T>, std::complex<T>>(ctx__.spla_context(), spins__(), 1.0, {&hub_wf__}, 0, hub_wf__.num_wf(),
        Up, 0, 0, 1.0, {&hphi__}, N__, n__);
}

template class Non_local_operator<double>;

template class D_operator<double>;

template class Q_operator<double>;

template void apply_non_local_d_q<double>(spin_range spins__, int N__, int n__, Beta_projectors<double>& beta__,
                                          Wave_functions<double>& phi__, D_operator<double>* d_op__,
                                          Wave_functions<double>* hphi__, Q_operator<double>* q_op__,
                                          Wave_functions<double>* sphi__);

template void apply_non_local_d_q<double_complex>(spin_range spins__, int N__, int n__, Beta_projectors<double>& beta__,
                                                  Wave_functions<double>& phi__, D_operator<double>* d_op__,
                                                  Wave_functions<double>* hphi__, Q_operator<double>* q_op__,
                                                  Wave_functions<double>* sphi__);

template void apply_S_operator<double>(device_t pu__, spin_range spins__, int N__, int n__,
                                       Beta_projectors<double>& beta__, Wave_functions<double>& phi__,
                                       Q_operator<double>* q_op__, Wave_functions<double>& sphi__);

template void apply_S_operator<double_complex>(device_t pu__, spin_range spins__, int N__, int n__,
                                               Beta_projectors<double>& beta__, Wave_functions<double>& phi__,
                                               Q_operator<double>* q_op__, Wave_functions<double>& sphi__);

template void apply_U_operator<double>(Simulation_context& ctx__, spin_range spins__, int N__, int n__,
                                       Wave_functions<double>& hub_wf__, Wave_functions<double>& phi__,
                                       U_operator<double>& um__, Wave_functions<double>& hphi__);

#if defined(USE_FP32)
template class Non_local_operator<float>;

template class D_operator<float>;

template class Q_operator<float>;

template void apply_non_local_d_q<float>(spin_range spins__, int N__, int n__, Beta_projectors<float>& beta__,
                                         Wave_functions<float>& phi__, D_operator<float>* d_op__,
                                         Wave_functions<float>* hphi__, Q_operator<float>* q_op__,
                                         Wave_functions<float>* sphi__);

template void apply_non_local_d_q<std::complex<float>>(spin_range spins__, int N__, int n__,
                                                       Beta_projectors<float>& beta__, Wave_functions<float>& phi__,
                                                       D_operator<float>* d_op__, Wave_functions<float>* hphi__,
                                                       Q_operator<float>* q_op__, Wave_functions<float>* sphi__);

template void apply_S_operator<float>(device_t pu__, spin_range spins__, int N__, int n__,
                                      Beta_projectors<float>& beta__, Wave_functions<float>& phi__,
                                      Q_operator<float>* q_op__, Wave_functions<float>& sphi__);

template void apply_S_operator<std::complex<float>>(device_t pu__, spin_range spins__, int N__, int n__,
                                                    Beta_projectors<float>& beta__, Wave_functions<float>& phi__,
                                                    Q_operator<float>* q_op__, Wave_functions<float>& sphi__);

template void apply_U_operator<float>(Simulation_context& ctx__, spin_range spins__, int N__, int n__,
                                      Wave_functions<float>& hub_wf__, Wave_functions<float>& phi__,
                                      U_operator<float>& um__, Wave_functions<float>& hphi__);
#endif
} // namespace sirius
