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

/**
 *   Compute C(t_{n-2}) * C(t_{n-2}).H *  S(t_{n-2}) * C(t_{n-1})
 */
template <class T, class Op>
void
transform_dm_extrapolation(Simulation_context& ctx, K_point<T>& kp, wf::Wave_functions<T>& wf_out,
                           wf::Wave_functions<T> const& wf_n1, wf::Wave_functions<T> const& wf_n2, Op s_op)
{
    const int num_spinors = (ctx.num_mag_dims() == 1) ? 2 : 1;
    const bool nc_mag     = ctx.num_mag_dims() == 3;
    auto num_wf           = wf::num_bands(wf_n1.num_wf());
    auto num_mag_dims     = wf::num_mag_dims(ctx.num_mag_dims());
    int n                 = wf_n1.num_wf();

    auto sphi_n1 = wf::Wave_functions<T>(kp.gkvec_sptr(), num_mag_dims, num_wf, memory_t::host);

    auto proc_mem_t = ctx.processing_unit_memory_t();

    for (int ispn_step = 0; ispn_step < num_spinors; ++ispn_step) {

        auto Ol = la::dmatrix<std::complex<T>>(num_wf, num_wf, memory_t::host);
        {
            auto wf_n1_guard   = wf_n1.memory_guard(proc_mem_t, wf::copy_to::device);
            auto wf_n2_guard   = wf_n2.memory_guard(proc_mem_t, wf::copy_to::device);
            auto sphi_n1_guard = sphi_n1.memory_guard(proc_mem_t, wf::copy_to::host);
            s_op[ispn_step]->apply(sphi_n1.pw_coeffs(wf::spin_index(ispn_step)),
                                   wf_n1.pw_coeffs(wf::spin_index(ispn_step)), proc_mem_t);
            auto br = wf::band_range(0, num_wf);
            auto sr = nc_mag ? wf::spin_range(0, 2) : wf::spin_range(ispn_step);
            // spla inner product
            wf::inner(ctx.spla_context(), proc_mem_t, sr, wf_n2, br, sphi_n1, br, Ol, 0, 0);
        }
        // transform wf_n2 with it
        wf::transform(ctx.spla_context(), memory_t::host, Ol, 0, 0,             // irow0, jcol0
                      1.0, wf_n2, wf::spin_index(ispn_step), wf::band_range(n), // input
                      0.0,                                                      // beta
                      wf_out, wf::spin_index(ispn_step), wf::band_range(n)      // output
        );
    }
}

template <class T>
void
loewdin(Simulation_context& ctx, K_point<T>& kp, Hamiltonian0<T>& H0, wf::Wave_functions<T>& wf_out,
        wf::Wave_functions<T> const& wf_in)
{
    auto Hk               = H0(kp);
    const int num_spinors = (ctx.num_mag_dims() == 1) ? 2 : 1;
    const bool nc_mag     = ctx.num_mag_dims() == 3;
    auto num_wf           = wf::num_bands(wf_in.num_wf());
    auto num_mag_dims     = wf::num_mag_dims(ctx.num_mag_dims());
    auto proc_mem_t       = ctx.processing_unit_memory_t();

    std::array<la::dmatrix<std::complex<double>>, 2> ovlp;
    {
        wf::Wave_functions<T> swf(kp.gkvec_sptr(), num_mag_dims, num_wf, proc_mem_t);
        auto wf_in_guard = wf_in.memory_guard(proc_mem_t, wf::copy_to::device);

        for (auto ispin_step = 0; ispin_step < num_spinors; ++ispin_step) {
            ovlp[ispin_step] = la::dmatrix<std::complex<T>>(num_wf, num_wf, memory_t::host);
            auto br          = wf::band_range(0, num_wf);
            auto sr          = nc_mag ? wf::spin_range(0, 2) : wf::spin_range(ispin_step);
            if (ctx.gamma_point()) {
                Hk.template apply_s<double>(sr, br, wf_in, swf);
            } else {
                Hk.template apply_s<std::complex<double>>(sr, br, wf_in, swf);
            }
            /*   compute overlap <wf_out|S|wf_in>   */
            wf::inner(ctx.spla_context(), proc_mem_t, sr, wf_in, br, swf, br, ovlp[ispin_step], 0, 0);
        }
    }
    for (int ispn = 0; ispn < wf_in.num_sc(); ++ispn) {
        la::dmatrix<std::complex<double>> Z(num_wf, num_wf);
        la::Eigensolver_lapack lapack_ev;
        std::vector<double> eval(num_wf);

        int ovlp_index = nc_mag ? 0 : ispn;
        lapack_ev.solve(num_wf, ovlp[ovlp_index], eval.data(), Z);
        mdarray<std::complex<double>, 1> d({num_wf.get()});
        for (int i = 0; i < num_wf; ++i) {
            d[i] = 1 / sqrt(eval[i]);
        }
        /* R = U * diag(1/sqrt(eval)) * U^H */
        mdarray<std::complex<double>, 2> tmp = unitary_similarity_transform(0 /* kind */, d, Z);
        la::dmatrix<std::complex<double>> R_loewdin(num_wf, num_wf);
        auto_copy(R_loewdin, tmp, device_t::CPU);
        int n = wf_in.num_wf();
        // call spla to transform wfc
        wf::transform(ctx.spla_context(), memory_t::host, R_loewdin, 0, 0, // irow0, jcol0
                      1.0, wf_in, wf::spin_index(ispn), wf::band_range(n), // input
                      0.0,                                                 // beta
                      wf_out, wf::spin_index(ispn), wf::band_range(n)      // output
        );
    }
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

    for (auto it : kset__.spl_num_kpoints()) {
        // wf::Wave_functions<double>
        auto& kp        = *kset__.get<double>(it.i);
        const auto& wfc = kp.spinor_wave_functions();

        int num_sc = wfc.num_sc();
        for (int i = 0; i < num_sc; ++i) {
            mdarray<double, 1> ek_loc({nbnd});
            for (int ie = 0; ie < nbnd; ++ie) {
                ek_loc(ie) = kp.band_energy(ie, i);
            }
            e_k[it.i][i] = std::move(ek_loc);
        }
        wfc_k[it.i] = copy(wfc);

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
    auto H0   = Hamiltonian0<double>(potential__, false);

    if (wfc_.size() < 2 || this->skip_) {
        std::stringstream ss;
        ss << "extrapolate skip";
        ctx.message(2, __func__, ss);
        // orthogonalize wfc (overlap matrix depends on ion positions)
        for (auto it : kset__.spl_num_kpoints()) {
            auto& kp  = *kset__.get<double>(it.i);
            auto& wfc = kp.spinor_wave_functions();
            if (wfc_.size() == 1) {
                auto& wfc_prev = *(wfc_.back().at(it.i));
                loewdin(ctx, kp, H0, wfc, wfc_prev);
            } else {
                loewdin(ctx, kp, H0, wfc, *wf::copy(wfc));
            }
        }
        // skip extrapolation, but regenerate density with updated ionic positions
        density__.generate<double>(kset__, ctx.use_symmetry(), true /* add core */, true /* transform to rg */);
        potential__.generate(density__, ctx.use_symmetry(), true);
        return;
    }

    if (wfc_.size() != 2) {
        throw std::runtime_error("expected size == 2");
    }

    std::stringstream ss;
    ss << "extrapolate";
    ctx.message(2, __func__, ss);

    /* true if this is a non-collinear case */
    const bool nc_mag = ctx.num_mag_dims() == 3;
    if (nc_mag) {
        RTE_THROW("non-collinear case not implemented");
    }

    // Ψ⁽ⁿ⁺¹⁾ = Löwdin(2 Ψ⁽ⁿ⁾ - Ψ⁽ⁿ⁻¹⁾)
    for (auto it : kset__.spl_num_kpoints()) {
        auto& kp          = *kset__.get<double>(it.i);
        auto& wfc         = kp.spinor_wave_functions();
        int num_sc        = wfc.num_sc(); // number of spin components
        auto num_wf       = wf::num_bands(wfc.num_wf());
        auto num_mag_dims = wf::num_mag_dims(ctx.num_mag_dims());

        auto& wfc_prev = *(wfc_.back().at(it.i));

        // psi_tilde <- 2*C(t_{n-1})
        wf::Wave_functions<double> psi_tilde(kp.gkvec_sptr(), num_mag_dims, num_wf, memory_t::host);
        auto br = wf::band_range(0, wfc.num_wf());
        auto sr = wf::spin_range(0, num_mag_dims + 1);
        std::vector<double> twos(num_wf, 2);
        std::vector<double> zeros(num_wf, 0);
        wf::axpby(memory_t::host, sr, br, twos.data(), &wfc, zeros.data(), &psi_tilde);

        wf::Wave_functions<double> wf_tmp(kp.gkvec_sptr(), num_mag_dims, num_wf, memory_t::host);
        transform_dm_extrapolation(ctx, kp, wf_tmp /* output */, wfc, wfc_prev, this->s_op_.back().at(it.i));
        // psi_tilde <- psi_tilde - wf_tmp
        std::vector<double> minus_ones(num_wf, -1.0);
        std::vector<double> ones(num_wf, 1.0);
        wf::axpby(memory_t::host, sr, br, minus_ones.data(), &wf_tmp, ones.data(), &psi_tilde);
        // std::cout << "after axpby: " << psi_tilde.checksum(memory_t::host) << "\n";

        // re-orthogonalize and write result back to wfc stored in k-point (host memory)
        loewdin(ctx, kp, H0, wfc, psi_tilde);
        // std::cout << "after loewdin: " << wfc.checksum(memory_t::host) << "\n";
        // extrapolate band energies
        for (int ispn = 0; ispn < num_sc; ++ispn) {
            // extrapolate band energies
            int nbnd = ctx.num_bands();
            std::vector<double> enew(nbnd);
            auto& en_curr = this->band_energies_.front().at(it.i)[ispn];
            auto& en_prev = this->band_energies_.back().at(it.i)[ispn];
            #pragma omp parallel for
            for (int ib = 0; ib < ctx.num_bands(); ++ib) {
                kp.band_energy(ib, ispn, 2 * en_curr[ib] - en_prev[ib]);
            }
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
