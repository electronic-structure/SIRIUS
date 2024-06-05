/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file non_local_operator.cpp
 *
 *  \brief Contains implementation of sirius::Non_local_operator class.
 */

#include "non_local_operator.hpp"
#include "beta_projectors/beta_projectors_base.hpp"
#include "hubbard/hubbard_matrix.hpp"

namespace sirius {

using namespace wf;

template <typename T>
Non_local_operator<T>::Non_local_operator(Simulation_context const& ctx__)
    : ctx_(ctx__)
{
    PROFILE("sirius::Non_local_operator");

    pu_                 = this->ctx_.processing_unit();
    auto& uc            = this->ctx_.unit_cell();
    packed_mtrx_offset_ = mdarray<int, 1>({uc.num_atoms()});
    packed_mtrx_size_   = 0;
    size_               = 0;
    for (int ia = 0; ia < uc.num_atoms(); ia++) {
        int nbf                 = uc.atom(ia).mt_basis_size();
        packed_mtrx_offset_(ia) = packed_mtrx_size_;
        packed_mtrx_size_ += nbf * nbf;
        size_ += nbf;
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
template <typename F>
matrix<F>
Non_local_operator<T>::get_matrix(int ispn, memory_t mem) const
{
    static_assert(is_complex<F>::value, "not implemented for gamma point");

    using double_complex = std::complex<double>;

    auto& uc = ctx_.unit_cell();
    std::vector<int> offsets(uc.num_atoms() + 1, 0);
    for (int ia = 0; ia < uc.num_atoms(); ++ia) {
        offsets[ia + 1] = offsets[ia] + uc.atom(ia).mt_basis_size();
    }

    matrix<double_complex> O({this->size(0), this->size(1)}, mem);
    O.zero(mem);
    int num_atoms = uc.num_atoms();
    for (int ia = 0; ia < num_atoms; ++ia) {
        int offset = offsets[ia];
        int lsize  = offsets[ia + 1] - offsets[ia];
        if (mem == memory_t::device) {
            double_complex* out_ptr = O.at(memory_t::device, offset, offset);
            const double_complex* op_ptr =
                    reinterpret_cast<const double_complex*>(op_.at(memory_t::device, 0, packed_mtrx_offset_(ia), ispn));
            // copy column by column
            for (int col = 0; col < lsize; ++col) {
                acc::copy(out_ptr + col * O.ld(), op_ptr + col * lsize, lsize);
            }
        } else if (mem == memory_t::host) {
            double_complex* out_ptr = O.at(memory_t::host, offset, offset);
            const double_complex* op_ptr =
                    reinterpret_cast<const double_complex*>(op_.at(memory_t::host, 0, packed_mtrx_offset_(ia), ispn));
            // copy column by column
            for (int col = 0; col < lsize; ++col) {
                std::copy(op_ptr + col * lsize, op_ptr + col * lsize + lsize, out_ptr + col * O.ld());
            }
        } else {
            RTE_THROW("invalid memory type.");
        }
    }
    return O;
}

template <typename T>
D_operator<T>::D_operator(Simulation_context const& ctx_)
    : Non_local_operator<T>(ctx_)
{
    if (ctx_.gamma_point()) {
        this->op_ = mdarray<T, 3>({1, this->packed_mtrx_size_, ctx_.num_mag_dims() + 1});
    } else {
        this->op_ = mdarray<T, 3>({2, this->packed_mtrx_size_, ctx_.num_mag_dims() + 1});
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
            mdarray<std::complex<T>, 3> d_mtrx_so({nbf, nbf, 4});
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
                int idxrf2 = atom.type().indexb(xi2).idxrf;
                for (int xi1 = 0; xi1 < nbf; xi1++) {
                    int idxrf1 = atom.type().indexb(xi1).idxrf;
                    if (atom.type().indexb(xi1).am == atom.type().indexb(xi2).am) {
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
                            RTE_THROW("wrong number of magnetic dimensions");
                        }
                    }
                }
            }
        }
    }

    if (env::print_checksum()) {
        auto cs = this->op_.checksum();
        print_checksum("D_operator", cs, this->ctx_.out());
    }

    if (this->pu_ == device_t::GPU && uc.max_mt_basis_size() != 0) {
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
        this->op_ = mdarray<T, 3>({1, this->packed_mtrx_size_, this->ctx_.num_mag_dims() + 1});
    } else {
        this->op_ = mdarray<T, 3>({2, this->packed_mtrx_size_, this->ctx_.num_mag_dims() + 1});
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
                                            result += this->ctx_.augmentation_op(iat).q_mtrx(xi1p, xi2p) *
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
                                this->ctx_.augmentation_op(iat).q_mtrx(xi1, xi2);
                    }
                }
            }
        }
    }
    if (env::print_checksum()) {
        auto cs = this->op_.checksum();
        print_checksum("Q_operator", cs, this->ctx_.out());
    }

    if (this->pu_ == device_t::GPU && uc.max_mt_basis_size() != 0) {
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
U_operator<T>::U_operator(Simulation_context const& ctx__, Hubbard_matrix const& um1__, std::array<double, 3> vk__)
    : ctx_(ctx__)
    , vk_(vk__)
{
    if (!ctx_.hubbard_correction()) {
        return;
    }
    /* a pair of "total number, offests" for the Hubbard orbitals idexing */
    auto r                 = ctx_.unit_cell().num_hubbard_wf();
    this->nhwf_            = r.first;
    this->offset_          = um1__.offset();
    this->atomic_orbitals_ = um1__.atomic_orbitals();
    for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
        um_[j] = la::dmatrix<std::complex<T>>(r.first, r.first);
        um_[j].zero();
    }

    /* copy local blocks */
    for (int at_lvl = 0; at_lvl < static_cast<int>(um1__.atomic_orbitals().size()); at_lvl++) {
        const int ia    = um1__.atomic_orbitals(at_lvl).first;
        auto& atom_type = ctx_.unit_cell().atom(ia).type();
        int lo_ind      = um1__.atomic_orbitals(at_lvl).second;
        if (atom_type.lo_descriptor_hub(lo_ind).use_for_calculation()) {
            int lmmax_at = 2 * atom_type.lo_descriptor_hub(lo_ind).l() + 1;
            for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
                for (int m2 = 0; m2 < lmmax_at; m2++) {
                    for (int m1 = 0; m1 < lmmax_at; m1++) {
                        um_[j](um1__.offset(at_lvl) + m1, um1__.offset(at_lvl) + m2) = um1__.local(at_lvl)(m1, m2, j);
                    }
                }
            }
        }
    }

    for (int i = 0; i < ctx_.cfg().hubbard().nonlocal().size(); i++) {
        auto nl = ctx_.cfg().hubbard().nonlocal(i);
        int ia  = nl.atom_pair()[0];
        int ja  = nl.atom_pair()[1];
        int il  = nl.l()[0];
        int jl  = nl.l()[1];
        auto Tr = nl.T();

        /* we need to find the index of the radial function corresponding to the atomic level of each atom.  */
        int at1_lvl = um1__.find_orbital_index(ia, nl.n()[0], il);
        int at2_lvl = um1__.find_orbital_index(ja, nl.n()[1], jl);

        auto z1 = std::exp(std::complex<double>(0, twopi * dot(vk_, r3::vector<int>(Tr))));
        for (int is = 0; is < ctx_.num_spins(); is++) {
            for (int m1 = 0; m1 < 2 * il + 1; m1++) {
                for (int m2 = 0; m2 < 2 * jl + 1; m2++) {
                    um_[is](um1__.offset(at1_lvl) + m1, um1__.offset(at2_lvl) + m2) +=
                            z1 * um1__.nonlocal(i)(m1, m2, is);
                }
            }
        }
    }
    for (int is = 0; is < ctx_.num_spins(); is++) {
        auto diff = check_hermitian(um_[is], r.first);
        if (diff > 1e-10) {
            RTE_THROW("um is not Hermitian");
        }
        if (env::print_checksum()) {
            print_checksum("um" + std::to_string(is), um_[is].checksum(r.first, r.first), RTE_OUT(ctx_.out()));
        }
        if (ctx_.processing_unit() == device_t::GPU) {
            um_[is].allocate(get_memory_pool(memory_t::device)).copy_to(memory_t::device);
        }
    }
}

/** Apply Hubbard U correction
 * \tparam T  Precision type of wave-functions (flat or double).
 * \param [in]  hub_wf   Hubbard atomic wave-functions.
 * \param [in]  phi      Set of wave-functions to which Hubbard correction is applied.
 * \param [out] hphi     Output wave-functions to which the result is added.
 */
template <typename T>
void
apply_U_operator(Simulation_context& ctx__, wf::spin_range spins__, wf::band_range br__,
                 wf::Wave_functions<T> const& hub_wf__, wf::Wave_functions<T> const& phi__, U_operator<T> const& um__,
                 wf::Wave_functions<T>& hphi__)
{
    if (!ctx__.hubbard_correction()) {
        return;
    }

    la::dmatrix<std::complex<T>> dm(hub_wf__.num_wf().get(), br__.size());

    auto mt = ctx__.processing_unit_memory_t();
    auto la = la::lib_t::blas;
    if (is_device_memory(mt)) {
        la = la::lib_t::gpublas;
        dm.allocate(mt);
    }

    /* First calculate the local part of the projections
       dm(i, n) = <phi_i| S |psi_{nk}> */
    wf::inner(ctx__.spla_context(), mt, spins__, hub_wf__, wf::band_range(0, hub_wf__.num_wf().get()), phi__, br__, dm,
              0, 0);

    la::dmatrix<std::complex<T>> Up(hub_wf__.num_wf().get(), br__.size());
    if (is_device_memory(mt)) {
        Up.allocate(mt);
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
                        for (int nbd = 0; nbd < br__.size(); nbd++) {
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
    } else {
        la::wrap(la).gemm('N', 'N', um__.nhwf(), br__.size(), um__.nhwf(), &la::constant<std::complex<T>>::one(),
                          um__.at(mt, 0, 0, spins__.begin().get()), um__.nhwf(), dm.at(mt, 0, 0), dm.ld(),
                          &la::constant<std::complex<T>>::zero(), Up.at(mt, 0, 0), Up.ld());
        if (is_device_memory(mt)) {
            Up.copy_to(memory_t::host);
        }
    }
    for (auto s = spins__.begin(); s != spins__.end(); s++) {
        auto sp  = hub_wf__.actual_spin_index(s);
        auto sp1 = hphi__.actual_spin_index(s);
        wf::transform(ctx__.spla_context(), mt, Up, 0, 0, 1.0, hub_wf__, sp, wf::band_range(0, hub_wf__.num_wf().get()),
                      1.0, hphi__, sp1, br__);
    }
}

/// Apply strain derivative of S-operator to all scalar functions.
void
apply_S_operator_strain_deriv(memory_t mem__, int comp__, Beta_projector_generator<double>& bp__,
                              beta_projectors_coeffs_t<double>& bp_coeffs__,
                              Beta_projector_generator<double>& bp_strain_deriv__,
                              beta_projectors_coeffs_t<double>& bp_strain_deriv_coeffs__,
                              wf::Wave_functions<double>& phi__, Q_operator<double>& q_op__,
                              wf::Wave_functions<double>& ds_phi__)
{
    if (is_device_memory(mem__)) {
        RTE_ASSERT((bp__.pu() == device_t::GPU));
    }
    // NOTE: Beta_projectors_generator knows the target memory!
    using complex_t = std::complex<double>;

    RTE_ASSERT(ds_phi__.num_wf() == phi__.num_wf());
    for (int ichunk = 0; ichunk < bp__.num_chunks(); ichunk++) {
        /* generate beta-projectors for a block of atoms */
        bp__.generate(bp_coeffs__, ichunk);
        /* generate derived beta-projectors for a block of atoms */
        bp_strain_deriv__.generate(bp_strain_deriv_coeffs__, ichunk, comp__);

        auto host_mem         = bp__.ctx().host_memory_t();
        auto& spla_ctx        = bp__.ctx().spla_context();
        auto band_range_phi   = wf::band_range(0, phi__.num_wf().get());
        bool result_on_device = bp__.ctx().processing_unit() == device_t::GPU;
        auto dbeta_phi        = inner_prod_beta<complex_t>(spla_ctx, mem__, host_mem, result_on_device,
                                                    bp_strain_deriv_coeffs__, phi__, wf::spin_index(0), band_range_phi);
        auto beta_phi = inner_prod_beta<complex_t>(spla_ctx, mem__, host_mem, result_on_device, bp_coeffs__, phi__,
                                                   wf::spin_index(0), band_range_phi);

        auto band_range = wf::band_range(0, ds_phi__.num_wf().get());
        q_op__.apply(mem__, ichunk, 0, ds_phi__, band_range, bp_coeffs__, dbeta_phi);
        q_op__.apply(mem__, ichunk, 0, ds_phi__, band_range, bp_strain_deriv_coeffs__, beta_phi);
    }
}

template matrix<std::complex<double>>
Non_local_operator<double>::get_matrix(int, memory_t) const;

template class Non_local_operator<double>;

template class D_operator<double>;

template class Q_operator<double>;

template class U_operator<double>;

template void
apply_U_operator<double>(Simulation_context&, wf::spin_range, wf::band_range, const wf::Wave_functions<double>&,
                         const wf::Wave_functions<double>&, U_operator<double> const&, wf::Wave_functions<double>&);

#if defined(SIRIUS_USE_FP32)
template class Non_local_operator<float>;

template class D_operator<float>;

template class Q_operator<float>;

template class U_operator<float>;

template void
apply_U_operator<float>(Simulation_context&, wf::spin_range, wf::band_range, const wf::Wave_functions<float>&,
                        const wf::Wave_functions<float>&, U_operator<float> const&, wf::Wave_functions<float>&);
#endif

} // namespace sirius
