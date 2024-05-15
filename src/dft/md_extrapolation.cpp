#include "md_extrapolation.hpp"
#include "context/simulation_context.hpp"
#include "core/env/env.hpp"
#include "core/la/eigenproblem.hpp"
#include "core/la/linalg.hpp"
#include "core/la/linalg_base.hpp"
#include "core/rte/rte.hpp"
#include "core/wf/wave_functions.hpp"
#include "hamiltonian/hamiltonian.hpp"
#include "k_point/k_point.hpp"
#include "k_point/k_point_set.hpp"
#include "core/memory.hpp"
#include "potential/potential.hpp"
#include <stdexcept>
#include "core/env/env.hpp"

namespace sirius {

namespace md {

/** Aligns subspace between two wave-functions.
 *  R = arg min_R' || Ψᵒ R' -  Ψⁱ ||
 *  O = <Ψᵒ|S|Ψⁱ>
 *  R = (O O*)^(-1/2) O = U Vt
 *  where O = U s Vt, is the singular value decomposition of O
 *
 *  Returns: Ψᵒ ← Ψᵒ R
 */
template <class T>
auto
subspace_alignment(Simulation_context& ctx, wf::Wave_functions<T>& wf_out, wf::Wave_functions<T> const& wf_in,
                   K_point<T>& kp, Hamiltonian0<T>& H0) -> std::array<la::dmatrix<std::complex<T>>, 2>
{
    auto Hk               = H0(kp);
    const int num_spinors = (ctx.num_mag_dims() == 1) ? 2 : 1;
    const bool nc_mag     = ctx.num_mag_dims() == 3;
    auto num_wf           = wf::num_bands(wf_in.num_wf());
    auto num_mag_dims     = wf::num_mag_dims(ctx.num_mag_dims());
    int n                 = wf_in.num_wf();

    // psi_tmp <- wf_in
    wf::Wave_functions<T> psi_tmp(kp.gkvec_sptr(), num_mag_dims, num_wf, memory_t::host);
    for (auto ispin_step = 0; ispin_step < num_spinors; ++ispin_step) {
        wf::copy(memory_t::host, wf_out, wf::spin_index(ispin_step), wf::band_range(n), psi_tmp,
                 wf::spin_index(ispin_step), wf::band_range(n));
    }
    // compute <wf_out|S|wf_in>
    auto sphi       = std::make_shared<wf::Wave_functions<T>>(kp.gkvec_sptr(), num_mag_dims, num_wf, memory_t::host);
    auto proc_mem_t = ctx.processing_unit_memory_t();
    std::array<la::dmatrix<std::complex<T>>, 2> ovlp;
    /*  the unitary transformation for each spin index */
    std::array<la::dmatrix<std::complex<T>>, 2> rots;
    {
        auto sphi_guard   = sphi->memory_guard(proc_mem_t, wf::copy_to::host);
        auto wf_in_guard  = wf_in.memory_guard(proc_mem_t, wf::copy_to::device);
        auto wf_out_guard = wf_out.memory_guard(proc_mem_t, wf::copy_to::device);

        for (auto ispin_step = 0; ispin_step < num_spinors; ++ispin_step) {
            ovlp[ispin_step] = la::dmatrix<std::complex<T>>(num_wf, num_wf, memory_t::host);
            auto br          = wf::band_range(0, num_wf);
            auto sr          = nc_mag ? wf::spin_range(0, 2) : wf::spin_range(ispin_step);
            if (ctx.gamma_point()) {
                Hk.template apply_s<double>(sr, br, wf_in, *sphi);
            } else {
                Hk.template apply_s<std::complex<double>>(sr, br, wf_in, *sphi);
            }
            /*   compute overlap <wf_out|S|wf_in>   */
            wf::inner(ctx.spla_context(), proc_mem_t, sr, wf_out, br, *sphi, br, ovlp[ispin_step], 0, 0);
            // compute SVD
            la::dmatrix<std::complex<T>> U(n, n);
            la::dmatrix<std::complex<T>> Vt(n, n);
            mdarray<double, 1> s({n}, "singular values");
            auto la = la::lib_t::lapack;
            la::wrap(la).gesvd('A', 'A', ovlp[ispin_step], s, U, Vt);
            // rotation matrix, to align wf_out
            la::dmatrix<std::complex<T>> Rot(n, n);
            auto ptr_one  = &la::constant<std::complex<T>>::one();
            auto ptr_zero = &la::constant<std::complex<T>>::zero();
            la::wrap(la::lib_t::blas)
                    .gemm('N', 'N', n, n, n, ptr_one, U.at(memory_t::host), U.ld(), Vt.at(memory_t::host), Vt.ld(),
                          ptr_zero, Rot.at(memory_t::host), Rot.ld());
            // call spla to transform wfc
            wf::transform(ctx.spla_context(), memory_t::host, Rot, 0, 0,               // irow0, jcol0
                          1.0, psi_tmp, wf::spin_index(ispin_step), wf::band_range(n), // input
                          0.0,                                                         // beta
                          wf_out, wf::spin_index(ispin_step), wf::band_range(n)        // output
            );
            rots[ispin_step] = std::move(Rot);
        }
    }
    return rots;
}

LinearWfcExtrapolation::LinearWfcExtrapolation(std::shared_ptr<spla::Context> spla_context)
    : spla_context_(spla_context)
{
    if (env::skip_wfct_extrapolation()) {
        this->skip_ = true;
    }
}

void
LinearWfcExtrapolation::push_back_history(const K_point_set& kset__, const Density& density__,
                                          const Potential& potential__)
{
    /*
      copy
      - plane-wave coefficients
      - band energies
      into internal data structures
     */

    if (this->skip_) {
        return;
    }

    int nbnd  = kset__.ctx().num_bands();
    auto& ctx = kset__.ctx();
    kp_map<std::shared_ptr<wf::Wave_functions<double>>> wfc_k;
    kp_map<s_op_vt> s_k;
    decltype(this->band_energies_)::value_type e_k;
    auto num_wf       = wf::num_bands(nbnd);
    auto num_mag_dims = wf::num_mag_dims(ctx.num_mag_dims());

    for (auto it : kset__.spl_num_kpoints()) {

        // wf::Wave_functions<double>
        auto& kp = *kset__.get<double>(it.i);
        auto wf_tmp =
                std::make_shared<wf::Wave_functions<double>>(kp.gkvec_sptr(), num_mag_dims, num_wf, memory_t::host);
        const auto& wfc = kp.spinor_wave_functions();
        int num_sc      = wfc.num_sc();
        for (int i = 0; i < num_sc; ++i) {

            wf::copy(memory_t::host, kp.spinor_wave_functions(), wf::spin_index(i), wf::band_range(nbnd), *wf_tmp,
                     wf::spin_index(i), wf::band_range(nbnd));

            mdarray<std::complex<double>, 1> ek_loc({nbnd});
            for (int ie = 0; ie < nbnd; ++ie) {
                ek_loc(ie) = kp.band_energy(ie, i);
            }
            e_k[it.i][i] = std::move(ek_loc);
        }
        wfc_k[it.i] = wf_tmp;

        // S operator
        auto q_op = std::make_shared<Q_operator<double>>(ctx);
        for (int ispn = 0; ispn < num_sc; ++ispn) {
            s_k[it.i][ispn] = std::make_shared<S_k<std::complex<double>>>(ctx.processing_unit_memory_t(), spla_context_,
                                                                          q_op, kp.beta_projectors_ptr(), ispn);
        }
    }

    this->s_op_.push_front(std::move(s_k));
    this->wfc_.push_front(wfc_k);
    this->band_energies_.push_front(std::move(e_k));

    if (wfc_.size() > 2) {
        this->wfc_.pop_back();
        this->band_energies_.pop_back();
        this->s_op_.pop_back();
    }
}

void
LinearWfcExtrapolation::extrapolate(K_point_set& kset__, Density& density__, Potential& potential__) const
{
    auto& ctx = kset__.ctx();

    if (wfc_.size() < 2 || this->skip_) {
        // skip extrapolation, but regenerate density with updated ionic positions
        density__.generate<double>(kset__, ctx.use_symmetry(), true /* add core */, true /* transform to rg */);
        potential__.generate(density__, ctx.use_symmetry(), true);
        return;
    }

    if (wfc_.size() != 2) {
        throw std::runtime_error("expected size =2");
    }

    std::stringstream ss;
    ss << "extrapolate";
    ctx.message(1, __func__, ss);
    /* H0 */
    auto H0 = Hamiltonian0<double>(potential__, false);

    /* true if this is a non-collinear case */
    const bool nc_mag = ctx.num_mag_dims() == 3;
    if (nc_mag) {
        RTE_THROW("non-collinear case not implemented");
    }

    const int num_spinors = (ctx.num_mag_dims() == 1) ? 2 : 1;

    // Ψ⁽ⁿ⁺¹⁾ = Löwdin(2 Ψ⁽ⁿ⁾ - Ψ⁽ⁿ⁻¹⁾)
    for (auto it : kset__.spl_num_kpoints()) {
        auto& kp          = *kset__.get<double>(it.i);
        auto& wfc         = kp.spinor_wave_functions();
        int num_sc        = wfc.num_sc(); // number of spin components
        auto num_wf       = wf::num_bands(wfc.num_wf());
        auto num_mag_dims = wf::num_mag_dims(ctx.num_mag_dims());

        wf::Wave_functions<double> psi_tilde(kp.gkvec_sptr(), num_mag_dims, num_wf, memory_t::host);

        auto& wfc_prev = wfc_.back().at(it.i);
        auto rot_up_dn = subspace_alignment(ctx, wfc, *wfc_prev, kp, H0);

        for (int ispn = 0; ispn < num_sc; ++ispn) {
            auto sp             = wf::spin_index(ispn);
            auto* psi_tilde_ptr = psi_tilde.pw_coeffs(sp).host_data();
            auto* psi_ptr       = wfc.pw_coeffs(sp).host_data();
            auto* psi_prev_ptr  = wfc_prev->pw_coeffs(sp).host_data();
            #pragma omp parallel for
            for (auto i = 0ul; i < wfc.pw_coeffs(sp).size(); ++i) {
                *(psi_tilde_ptr + i) = 2.0 * (*(psi_ptr + i)) - (*(psi_prev_ptr + i));
            }
        }
        /* Löwdin orthogonalization */
        auto Hk         = H0(kp);
        auto proc_mem_t = ctx.processing_unit_memory_t();
        std::array<la::dmatrix<std::complex<double>>, 2> ovlp_spinor;
        // compute S|psi~>
        {
            auto sphi = std::make_shared<wf::Wave_functions<double>>(kp.gkvec_sptr(), num_mag_dims, num_wf, proc_mem_t);
            auto phi_guard = psi_tilde.memory_guard(proc_mem_t, wf::copy_to::device);

            for (auto ispin_step = 0; ispin_step < num_spinors; ++ispin_step) {
                ovlp_spinor[ispin_step] = la::dmatrix<std::complex<double>>(num_wf, num_wf, memory_t::host);
                auto br                 = wf::band_range(0, num_wf);
                auto sr                 = nc_mag ? wf::spin_range(0, 2) : wf::spin_range(ispin_step);
                if (ctx.gamma_point()) {
                    Hk.apply_s<double>(sr, br, psi_tilde, *sphi);
                } else {
                    Hk.apply_s<std::complex<double>>(sr, br, psi_tilde, *sphi);
                }
                /*   compute overlap <psi~|S|psi~>   */
                wf::inner(kset__.ctx().spla_context(), proc_mem_t, sr, psi_tilde, br, *sphi, br,
                          ovlp_spinor[ispin_step], 0, 0);
            }
        }
        la::lib_t la{la::lib_t::blas};
        for (int ispn = 0; ispn < num_sc; ++ispn) {
            int ovlp_index = nc_mag ? 0 : ispn;
            /*   compute eig(<psi|S|psi>)   */
            la::dmatrix<std::complex<double>> Z(num_wf, num_wf);
            la::Eigensolver_lapack lapack_ev;
            std::vector<double> eval(num_wf);
            lapack_ev.solve(num_wf, ovlp_spinor[ovlp_index], eval.data(), Z);

            mdarray<std::complex<double>, 1> d({num_wf.get()});
            for (int i = 0; i < num_wf; ++i) {
                d[i] = 1 / sqrt(eval[i]);
            }

            /* R = U * diag(1/sqrt(eval)) * U^H */
            auto R_lowedin = unitary_similarity_transform(0, d, Z);
            auto ptr_one   = &la::constant<std::complex<double>>::one();
            auto ptr_zero  = &la::constant<std::complex<double>>::zero();
            const std::complex<double>* psi_tilde_ptr =
                    psi_tilde.at(memory_t::host, 0, wf::spin_index(ispn), wf::band_index(0));

            /* band energies */
            matrix<std::complex<double>> U_e({num_wf.get(), num_wf.get()});
            auto eprev = la::diag(band_energies_.back().at(it.i)[ispn]);
            auto ecur  = la::unitary_similarity_transform(1, band_energies_.back().at(it.i)[ispn], rot_up_dn[ispn]);

            auto eprime = empty_like(ecur);
#pragma omp parallel for
            for (auto ii = 0ul; ii < ecur.size(); ++ii) {
                *(eprime.at(memory_t::host) + ii) =
                        2.0 * (*(ecur.at(memory_t::host) + ii)) - *(eprev.at(memory_t::host) + ii);
            }
            // // diagonalize eprime, U_e
            std::vector<double> enew(num_wf.get());
            lapack_ev.solve_(num_wf.get(), eprime, enew.data(), U_e);
            for (auto ib = 0ul; ib < enew.size(); ++ib) {
                kp.band_energy(ib, ispn, enew[ib]);
            }
            ///  U_final <- R_loweding * U_e
            matrix<std::complex<double>> U_final = empty_like(U_e);
            la::wrap(la).gemm('N', 'N', num_wf, num_wf, num_wf, ptr_one, R_lowedin.at(memory_t::host), R_lowedin.ld(),
                              U_e.at(memory_t::host), U_e.ld(), ptr_zero, U_final.at(memory_t::host), U_final.ld());

            /* transform wfc  with U_final */
            int num_gkvec_loc = kp.num_gkvec_loc();
            auto wf_i_ptr     = wfc.at(memory_t::host, 0, wf::spin_index(ispn), wf::band_index(0));
            la::wrap(la).gemm('N', 'N', num_gkvec_loc, num_wf, num_wf, ptr_one, psi_tilde_ptr, psi_tilde.ld(),
                              U_final.at(memory_t::host), U_final.ld(), ptr_zero, wf_i_ptr, num_gkvec_loc);
        }
    }

    // compute occupation numbers from new band energies
    kset__.sync_band<double, sync_band_t::energy>();
    kset__.find_band_occupancies<double>();

    // generate density
    density__.generate<double>(kset__, ctx.use_symmetry(), true /* add core */, true /* transform to rg */);
    // generate potential
    potential__.generate(density__, ctx.use_symmetry(), true);
}

} // namespace md

} // namespace sirius
