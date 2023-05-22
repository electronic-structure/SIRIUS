// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
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

/** \file non_local_operator.hpp
 *
 *  \brief Contains declaration of sirius::Non_local_operator class.
 */

#ifndef __NON_LOCAL_OPERATOR_HPP__
#define __NON_LOCAL_OPERATOR_HPP__

#include "SDDK/omp.hpp"
#include "SDDK/memory.hpp"
#include "beta_projectors/beta_projectors.hpp"
#include "beta_projectors/beta_projectors_strain_deriv.hpp"
#include "context/simulation_context.hpp"
#include "hubbard/hubbard_matrix.hpp"

namespace sirius {
/* forward declaration */
template <typename T>
class Beta_projectors;
template <typename T>
class Beta_projectors_base;

/// Non-local part of the Hamiltonian and S-operator in the pseudopotential method.
template <typename T>
class Non_local_operator
{
  protected:
    Simulation_context const& ctx_;

    sddk::device_t pu_;

    int packed_mtrx_size_;

    sddk::mdarray<int, 1> packed_mtrx_offset_;

    /// Non-local operator matrix.
    sddk::mdarray<T, 3> op_;

    bool is_null_{false};

    /// True if the operator is diagonal in spin.
    bool is_diag_{true};

    /* copy assignment operrator is forbidden */
    Non_local_operator<T>& operator=(Non_local_operator<T> const& src) = delete;
    /* copy constructor is forbidden */
    Non_local_operator(Non_local_operator<T> const& src) = delete;

  public:
    /// Constructor.
    Non_local_operator(Simulation_context const& ctx__);

    /// Apply chunk of beta-projectors to all wave functions.
    /** \tparam F  Type of the subspace matrix
     */
    template <typename F>
    void
    apply(sddk::memory_t mem__, int chunk__, int ispn_block__, wf::Wave_functions<T>& op_phi__, wf::band_range br__,
          Beta_projectors_base<T> const& beta__, sddk::matrix<F> const& beta_phi__) const
    {
        PROFILE("sirius::Non_local_operator::apply");

        if (is_null_) {
            return;
        }

        auto const& beta_gk = beta__.pw_coeffs_a();
        int num_gkvec_loc   = beta__.num_gkvec_loc();
        int nbeta           = beta__.chunk(chunk__).num_beta_;

        /* setup linear algebra parameters */
        la::lib_t la{la::lib_t::blas};
        sddk::device_t pu{sddk::device_t::CPU};
        if (is_device_memory(mem__)) {
            la = la::lib_t::gpublas;
            pu = sddk::device_t::GPU;
        }

        int size_factor = 1;
        if (std::is_same<F, real_type<F>>::value) {
            size_factor = 2;
        }

        auto work = sddk::mdarray<F, 2>(nbeta, br__.size(), get_memory_pool(mem__));

        /* compute O * <beta|phi> for atoms in a chunk */
        #pragma omp parallel
        {
            acc::set_device_id(mpi::get_device_id(acc::num_devices())); // avoid cuda mth bugs

            #pragma omp for
            for (int i = 0; i < beta__.chunk(chunk__).num_atoms_; i++) {
                /* number of beta functions for a given atom */
                int nbf  = beta__.chunk(chunk__).desc_(beta_desc_idx::nbf, i);
                int offs = beta__.chunk(chunk__).desc_(beta_desc_idx::offset, i);
                int ia   = beta__.chunk(chunk__).desc_(beta_desc_idx::ia, i);

                if (nbf) {
                    la::wrap(la).gemm(
                        'N', 'N', nbf, br__.size(), nbf, &la::constant<F>::one(),
                        reinterpret_cast<F const*>(op_.at(mem__, 0, packed_mtrx_offset_(ia), ispn_block__)), nbf,
                        reinterpret_cast<F const*>(beta_phi__.at(mem__, offs, 0)), beta_phi__.ld(),
                        &la::constant<F>::zero(),
                        reinterpret_cast<F*>(work.at(mem__, offs, 0)), nbeta, stream_id(omp_get_thread_num()));
                }
            }
        }
        switch (pu) { // TODO: check if this is needed. Null stream later should sync the streams.
            case sddk::device_t::GPU: {
                /* wait for previous zgemms */
                #pragma omp parallel
                acc::sync_stream(stream_id(omp_get_thread_num()));
                break;
            }
            case sddk::device_t::CPU: {
                break;
            }
        }

        auto sp = op_phi__.actual_spin_index(wf::spin_index(ispn_block__ & 1));

        /* compute <G+k|beta> * O * <beta|phi> and add to op_phi */
        la::wrap(la)
            .gemm('N', 'N', num_gkvec_loc * size_factor, br__.size(), nbeta, &la::constant<F>::one(),
                reinterpret_cast<F const*>(beta_gk.at(mem__)), num_gkvec_loc * size_factor, 
                work.at(mem__), nbeta, &la::constant<F>::one(),
                reinterpret_cast<F*>(op_phi__.at(mem__, 0, sp, wf::band_index(br__.begin()))),
                op_phi__.ld() * size_factor);

        switch (pu) {
            case sddk::device_t::GPU: {
                acc::sync_stream(stream_id(-1));
                break;
            }
            case sddk::device_t::CPU: {
                break;
            }
        }
    }

    /// Apply beta projectors from one atom in a chunk of beta projectors to all wave-functions.
    template <typename F>
    std::enable_if_t<std::is_same<std::complex<T>, F>::value, void>
    apply(sddk::memory_t mem__, int chunk__, wf::atom_index ia__, int ispn_block__, wf::Wave_functions<T>& op_phi__,
            wf::band_range br__, Beta_projectors_base<T>& beta__, sddk::matrix<F>& beta_phi__)
    {
        if (is_null_) {
            return;
        }

        auto& beta_gk     = beta__.pw_coeffs_a();
        int num_gkvec_loc = beta__.num_gkvec_loc();

        int nbf  = beta__.chunk(chunk__).desc_(beta_desc_idx::nbf, ia__.get());
        int offs = beta__.chunk(chunk__).desc_(beta_desc_idx::offset, ia__.get());
        int ia   = beta__.chunk(chunk__).desc_(beta_desc_idx::ia, ia__.get());

        if (nbf == 0) {
            return;
        }

        la::lib_t la{la::lib_t::blas};
        sddk::device_t pu{sddk::device_t::CPU};
        if (is_device_memory(mem__)) {
            la = la::lib_t::gpublas;
            pu = sddk::device_t::GPU;
        }

        auto work = sddk::mdarray<std::complex<T>, 1>(nbf * br__.size(), get_memory_pool(mem__));

        la::wrap(la).gemm('N', 'N', nbf, br__.size(), nbf, &la::constant<std::complex<T>>::one(),
                        reinterpret_cast<std::complex<T>*>(op_.at(mem__, 0, packed_mtrx_offset_(ia), ispn_block__)), nbf,
                        beta_phi__.at(mem__, offs, 0), beta_phi__.ld(), &la::constant<std::complex<T>>::zero(),
                        work.at(mem__), nbf);

        int jspn = ispn_block__ & 1;

        la::wrap(la)
            .gemm('N', 'N', num_gkvec_loc, br__.size(), nbf, &la::constant<std::complex<T>>::one(),
                    beta_gk.at(mem__, 0, offs), num_gkvec_loc, work.at(mem__), nbf, &la::constant<std::complex<T>>::one(),
                  op_phi__.at(mem__, 0, wf::spin_index(jspn), wf::band_index(br__.begin())),
                  op_phi__.ld());

        switch (pu) {
            case sddk::device_t::CPU: {
                break;
            }
            case sddk::device_t::GPU: {
#ifdef SIRIUS_GPU
                acc::sync_stream(stream_id(-1));
#endif
                break;
            }
        }
    }

    template <typename F, typename = std::enable_if_t<std::is_same<T, real_type<F>>::value>>
    inline F value(int xi1__, int xi2__, int ia__)
    {
        return this->value<F>(xi1__, xi2__, 0, ia__);
    }

    template <typename F, std::enable_if_t<std::is_same<T, F>::value, bool> = true>
    F value(int xi1__, int xi2__, int ispn__, int ia__)
    {
        int nbf = this->ctx_.unit_cell().atom(ia__).mt_basis_size();
        return this->op_(0, packed_mtrx_offset_(ia__) + xi2__ * nbf + xi1__, ispn__);
    }

    template <typename F, std::enable_if_t<std::is_same<std::complex<T>, F>::value, bool> = true>
    F value(int xi1__, int xi2__, int ispn__, int ia__)
    {
        int nbf = this->ctx_.unit_cell().atom(ia__).mt_basis_size();
        return std::complex<T>(this->op_(0, packed_mtrx_offset_(ia__) + xi2__ * nbf + xi1__, ispn__),
                               this->op_(1, packed_mtrx_offset_(ia__) + xi2__ * nbf + xi1__, ispn__));
    }

    inline bool is_diag() const
    {
        return is_diag_;
    }
};

template <typename T>
class D_operator : public Non_local_operator<T>
{
  private:
    void initialize();

  public:
    D_operator(Simulation_context const& ctx_);
};

template <typename T>
class Q_operator : public Non_local_operator<T>
{
  private:
    void initialize();

  public:
    Q_operator(Simulation_context const& ctx__);
};

template <typename T>
class U_operator
{
  private:
    Simulation_context const& ctx_;
    //sddk::mdarray<std::complex<T>, 3> um_;
    std::array<la::dmatrix<std::complex<T>>, 4> um_;
    std::vector<int> offset_;
    std::vector<std::pair<int, int>> atomic_orbitals_;
    int nhwf_;
    r3::vector<double> vk_;

  public:
    U_operator(Simulation_context const& ctx__, Hubbard_matrix const& um1__, std::array<double, 3> vk__)
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
        for (int j = 0; j <  ctx_.num_mag_dims() + 1; j++) {
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
                            um_[j](um1__.offset(at_lvl) + m1, um1__.offset(at_lvl) + m2) =
                                um1__.local(at_lvl)(m1, m2, j);
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
            if (ctx_.print_checksum()) {
                utils::print_checksum("um" + std::to_string(is), um_[is].checksum(r.first, r.first), RTE_OUT(ctx_.out()));
            }
            if (ctx_.processing_unit() == sddk::device_t::GPU) {
                um_[is].allocate(get_memory_pool(sddk::memory_t::device)).copy_to(sddk::memory_t::device);
            }
        }
    }

    ~U_operator()
    {
    }

    inline auto atomic_orbitals() const
    {
        return atomic_orbitals_;
    }

    inline auto atomic_orbitals(const int idx__) const
    {
        return atomic_orbitals_[idx__];
    }
    inline auto nhwf() const
    {
        return nhwf_;
    }

    inline auto offset(int ia__) const
    {
        return offset_[ia__];
    }

    auto operator()(int m1, int m2, int j)
    {
        return um_[j](m1, m2);
    }

    auto* at(sddk::memory_t mem__, const int idx1, const int idx2, const int idx3)
    {
        return um_[idx3].at(mem__, idx1, idx2);
    }

    const int find_orbital_index(const int ia__, const int n__, const int l__) const
    {
        int at_lvl = 0;
        for (at_lvl = 0; at_lvl < static_cast<int>(atomic_orbitals_.size()); at_lvl++) {
            int lo_ind  = atomic_orbitals_[at_lvl].second;
            int atom_id = atomic_orbitals_[at_lvl].first;
            if ((atomic_orbitals_[at_lvl].first == ia__) &&
                (ctx_.unit_cell().atom(atom_id).type().lo_descriptor_hub(lo_ind).n() == n__) &&
                (ctx_.unit_cell().atom(atom_id).type().lo_descriptor_hub(lo_ind).l() == l__))
                break;
        }

        if (at_lvl == static_cast<int>(atomic_orbitals_.size())) {
            std::cout << "atom: " << ia__ << "n: " << n__ << ", l: " << l__ << std::endl;
            RTE_THROW("Found an arbital that is not listed\n");
        }
        return at_lvl;
    }
};

// template <typename T>
// class P_operator : public Non_local_operator<T>
//{
//  public:
//    P_operator(Simulation_context const& ctx_, mdarray<std::complex<double>, 3>& p_mtrx__)
//        : Non_local_operator<T>(ctx_)
//    {
//        /* Q-operator is independent of spin */
//        this->op_ = mdarray<T, 2>(this->packed_mtrx_size_, 1);
//        this->op_.zero();
//
//        auto& uc = ctx_.unit_cell();
//        for (int ia = 0; ia < uc.num_atoms(); ia++) {
//            int iat = uc.atom(ia).type().id();
//            if (!uc.atom_type(iat).augment()) {
//                continue;
//            }
//            int nbf = uc.atom(ia).mt_basis_size();
//            for (int xi2 = 0; xi2 < nbf; xi2++) {
//                for (int xi1 = 0; xi1 < nbf; xi1++) {
//                    this->op_(this->packed_mtrx_offset_(ia) + xi2 * nbf + xi1, 0) = -p_mtrx__(xi1, xi2, iat).real();
//                }
//            }
//        }
//        if (this->pu_ == device_t::GPU) {
//            this->op_.allocate(memory_t::device);
//            this->op_.copy_to(memory_t::device);
//        }
//    }
//};

/// Apply non-local part of the Hamiltonian and S operators.
/** This operations must be combined because of the expensive inner product between wave-functions and beta
 *  projectors, which is computed only once.
 *
 *  \param [in]  spins   Range of the spin index.
 *  \param [in]  N       Starting index of the wave-functions.
 *  \param [in]  n       Number of wave-functions to which D and Q are applied.
 *  \param [in]  beta    Beta-projectors.
 *  \param [in]  phi     Wave-functions.
 *  \param [in]  d_op    Pointer to D-operator.
 *  \param [out] hphi    Resulting |beta>D<beta|phi>
 *  \param [in]  q_op    Pointer to Q-operator.
 *  \param [out] sphi    Resulting |beta>Q<beta|phi>
 */
//template <typename T>
//void apply_non_local_d_q(sddk::spin_range spins__, int N__, int n__, Beta_projectors<real_type<T>>& beta__,
//                         sddk::Wave_functions<real_type<T>>& phi__, D_operator<real_type<T>>* d_op__,
//                         sddk::Wave_functions<real_type<T>>* hphi__, Q_operator<real_type<T>>* q_op__,
//                         sddk::Wave_functions<real_type<T>>* sphi__);

/** \tparam T  Precision of the wave-functions.
 *  \tparam F  Type of the subspace.
 **/
template <typename T, typename F>
void
apply_non_local_D_Q(sddk::memory_t mem__, wf::spin_range spins__, wf::band_range br__, Beta_projectors<T>& beta__,
    wf::Wave_functions<T> const& phi__, D_operator<T> const* d_op__, wf::Wave_functions<T>* hphi__,
    Q_operator<T> const* q_op__, wf::Wave_functions<T>* sphi__)
{
    for (int i = 0; i < beta__.num_chunks(); i++) {
        /* generate beta-projectors for a block of atoms */
        beta__.generate(mem__, i);

        for (auto s = spins__.begin(); s != spins__.end(); s++) {
            auto sp = phi__.actual_spin_index(s);
            auto beta_phi = beta__.template inner<F>(mem__, i, phi__, sp, br__);

            if (hphi__ && d_op__) {
                /* apply diagonal spin blocks */
                d_op__->apply(mem__, i, s.get(), *hphi__, br__, beta__, beta_phi);
                if (!d_op__->is_diag() && hphi__->num_md() == wf::num_mag_dims(3)) {
                    /* apply non-diagonal spin blocks */
                    /* xor 3 operator will map 0 to 3 and 1 to 2 */
                    d_op__->apply(mem__, i, s.get() ^ 3, *hphi__, br__, beta__, beta_phi);
                }
            }

            if (sphi__ && q_op__) {
                /* apply Q operator (diagonal in spin) */
                q_op__->apply(mem__, i, s.get(), *sphi__, br__, beta__, beta_phi);
                if (!q_op__->is_diag() && sphi__->num_md() == wf::num_mag_dims(3)) {
                    q_op__->apply(mem__, i, s.get() ^ 3, *sphi__, br__, beta__, beta_phi);
                }
            }
        }
    }
}

/// Compute |sphi> = (1 + Q)|phi>
template <typename T, typename F>
void apply_S_operator(sddk::memory_t mem__, wf::spin_range spins__, wf::band_range br__,
                      Beta_projectors<T>& beta__, wf::Wave_functions<T> const& phi__,
                      Q_operator<T> const* q_op__, wf::Wave_functions<T>& sphi__)
{
    for (auto s = spins__.begin(); s != spins__.end(); s++) {
        wf::copy(mem__, phi__, s, br__, sphi__, s, br__);
    }

    if (q_op__) {
        apply_non_local_D_Q<T, F>(mem__, spins__, br__, beta__, phi__, nullptr, nullptr, q_op__, &sphi__);
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
        wf::Wave_functions<T> const& hub_wf__, wf::Wave_functions<T> const& phi__, U_operator<T>& um__,
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
    wf::inner(ctx__.spla_context(), mt, spins__, hub_wf__, wf::band_range(0, hub_wf__.num_wf().get()), phi__, br__, dm, 0, 0);

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
        la::wrap(la).gemm('N', 'N', um__.nhwf(), br__.size(), um__.nhwf(),
                &la::constant<std::complex<T>>::one(), um__.at(mt, 0, 0, spins__.begin().get()), um__.nhwf(),
                dm.at(mt, 0, 0), dm.ld(), &la::constant<std::complex<T>>::zero(), Up.at(mt, 0, 0), Up.ld());
        if (is_device_memory(mt)) {
            Up.copy_to(sddk::memory_t::host);
        }
    }
    for (auto s = spins__.begin(); s != spins__.end(); s++) {
        auto sp = hub_wf__.actual_spin_index(s);
        wf::transform(ctx__.spla_context(), mt, Up, 0, 0, 1.0, hub_wf__, sp, wf::band_range(0, hub_wf__.num_wf().get()),
            1.0, hphi__, sp, br__);
    }
}

/// Apply strain derivative of S-operator to all scalar functions.
inline void
apply_S_operator_strain_deriv(sddk::memory_t mem__, int comp__, Beta_projectors<double>& bp__,
                              Beta_projectors_strain_deriv<double>& bp_strain_deriv__, wf::Wave_functions<double>& phi__,
                              Q_operator<double>& q_op__, wf::Wave_functions<double>& ds_phi__)
{
    RTE_ASSERT(ds_phi__.num_wf() == phi__.num_wf());
    for (int ichunk = 0; ichunk < bp__.num_chunks(); ichunk++) {
        /* generate beta-projectors for a block of atoms */
        bp__.generate(mem__, ichunk);
        /* generate derived beta-projectors for a block of atoms */
        bp_strain_deriv__.generate(mem__, ichunk, comp__);
        auto dbeta_phi = bp_strain_deriv__.inner<std::complex<double>>(mem__, ichunk, phi__, wf::spin_index(0),
                wf::band_range(0, phi__.num_wf().get()));
        auto beta_phi = bp__.inner<std::complex<double>>(mem__, ichunk, phi__, wf::spin_index(0), wf::band_range(0, phi__.num_wf().get()));
        q_op__.apply(mem__, ichunk, 0, ds_phi__, wf::band_range(0, ds_phi__.num_wf().get()), bp__, dbeta_phi);
        q_op__.apply(mem__, ichunk, 0, ds_phi__, wf::band_range(0, ds_phi__.num_wf().get()), bp_strain_deriv__, beta_phi);
    }
}

} // namespace sirius

#endif
