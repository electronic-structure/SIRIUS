/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file adaptor.cpp
 *
 *  \brief Contains implementation of the interface to nlcglib.
 */

#include "core/rte/rte.hpp"
#include "core/wf/wave_functions.hpp"
#ifdef SIRIUS_NLCGLIB
#include <stdexcept>

#include "adaptor.hpp"
#include "apply_hamiltonian.hpp"
#include "hamiltonian/local_operator.hpp"
#include "hamiltonian/hamiltonian.hpp"
#include "dft/energy.hpp"

using namespace nlcglib;

using prec_t = double;

namespace sirius {

template <class wfc_ptr_t>
std::shared_ptr<Matrix>
make_vector(std::vector<wfc_ptr_t> const& wfct, Simulation_context const& ctx, K_point_set const& kset,
            nlcglib::memory_type memory = nlcglib::memory_type::none)
{
    std::map<memory_t, nlcglib::memory_type> memtype        = {{memory_t::device, nlcglib::memory_type::device},
                                                               {memory_t::host, nlcglib::memory_type::host},
                                                               {memory_t::host_pinned, nlcglib::memory_type::host}};
    std::map<nlcglib::memory_type, memory_t> memtype_lookup = {{nlcglib::memory_type::none, memory_t::none},
                                                               {nlcglib::memory_type::device, memory_t::device},
                                                               {nlcglib::memory_type::host, memory_t::host},
                                                               {nlcglib::memory_type::host, memory_t::host_pinned}};

    memory_t target_memory = memtype_lookup.at(memory);
    if (target_memory == memory_t::none) {
        target_memory = ctx.processing_unit_memory_t();
    }

    std::vector<Matrix::buffer_t> data;
    std::vector<std::pair<int, int>> kpoint_indices;
    // memory_t preferred_memory = ctx.preferred_memory_t();
    int num_spins = ctx.num_spins();
    int nb        = ctx.num_bands();
    for (auto i = 0u; i < wfct.size(); ++i) {
        auto gidk = kset.spl_num_kpoints().global_index(typename kp_index_t::local(i)); //(i); // global k-point index
        for (int ispn = 0; ispn < num_spins; ++ispn) {
            auto& array   = wfct[i]->pw_coeffs(wf::spin_index(ispn));
            int lda       = array.size(0);
            MPI_Comm comm = wfct[i]->comm().native();
            // check that wfct has been allocated
            if (is_device_memory(target_memory)) {
                // make sure that array is on device
                if (!array.on_device()) {
                    RTE_THROW("Error: expected device storage, but got nullptr");
                }
            }
            kpoint_indices.emplace_back(std::make_pair(gidk, ispn));
            data.emplace_back(std::array<int, 2>{1, lda},  /* stride */
                              std::array<int, 2>{lda, nb}, /* size */
                              array.at(target_memory),     /* pointer */
                              memtype.at(target_memory), comm /* mpi communicator */);
        }
    }
    return std::make_shared<Matrix>(std::move(data), std::move(kpoint_indices), kset.comm().native());
}

Matrix::buffer_t
Matrix::get(int i)
{
    return data[i];
}

const Matrix::buffer_t
Matrix::get(int i) const
{
    return data[i];
}

Energy::Energy(K_point_set& kset, Density& density, Potential& potential)
    : kset_(kset)
    , density_(density)
    , potential_(potential)
{
    int nk    = kset.spl_num_kpoints().local_size();
    auto& ctx = kset.ctx();
    hphis_.resize(nk);
    cphis_.resize(nk);
    for (auto it : kset.spl_num_kpoints()) {
        auto& kp                    = *kset.get<double>(it.i);
        memory_t preferred_memory_t = ctx.processing_unit_memory_t();
        auto num_mag_dims           = wf::num_mag_dims(ctx.num_mag_dims());
        auto num_bands              = wf::num_bands(ctx.num_bands());
        // make a new wf for Hamiltonian apply...
        hphis_[it.li] = std::make_shared<wf::Wave_functions<prec_t>>(kp.gkvec_sptr(), num_mag_dims, num_bands,
                                                                     preferred_memory_t);
        cphis_[it.li] = &kp.spinor_wave_functions();
        hphis_[it.li]->allocate(memory_t::host);
    }
    // need to allocate wavefunctions on GPU
}

void
Energy::compute()
{
    auto& ctx     = kset_.ctx();
    int num_spins = ctx.num_spins();
    int num_bands = ctx.num_bands();

    density_.generate<prec_t>(kset_, ctx.use_symmetry(), true /* add core */, true /* transform to rg */);

    potential_.generate(density_, ctx.use_symmetry(), true);

    /* compute H@X and new band energies */
    auto H0 = Hamiltonian0<double>(potential_, true);

    auto proc_mem_t = ctx.processing_unit_memory_t();

    // apply Hamiltonian
    for (auto it : kset_.spl_num_kpoints()) {
        auto& kp = *kset_.get<prec_t>(it.i);
        std::vector<double> band_energies(num_bands);

        auto mem_guard   = cphis_[it.li]->memory_guard(proc_mem_t, wf::copy_to::device);
        auto mem_guard_h = hphis_[it.li]->memory_guard(proc_mem_t, wf::copy_to::host);

        auto null_ptr_wfc = std::shared_ptr<wf::Wave_functions<prec_t>>();
        apply_hamiltonian(H0, kp, *hphis_[it.li], kp.spinor_wave_functions(), null_ptr_wfc);
        // compute band energies = diag(<psi|H|psi>)
        for (int ispn = 0; ispn < num_spins; ++ispn) {
            for (int jj = 0; jj < num_bands; ++jj) {
                la::dmatrix<std::complex<double>> dmat(1, 1, memory_t::host);
                dmat.allocate(memory_t::device);
                wf::band_range bandr{jj, jj + 1};
                wf::inner(ctx.spla_context(), proc_mem_t, wf::spin_range(ispn),
                          /* bra */ kp.spinor_wave_functions(), bandr,
                          /* ket */ *hphis_[it.li], bandr,
                          /*result*/ dmat, 0, 0);
                kp.band_energy(jj, ispn, dmat(0, 0).real());
            }
        }
    }

    kset_.sync_band<double, sync_band_t::energy>();

    // evaluate total energy
    // double eewald      = ewald_energy(ctx, ctx.gvec(), ctx.unit_cell());
    energy_components_ = total_energy_components(ctx, kset_, density_, potential_);
    etot_              = ks_energy(ctx, this->energy_components_);
}

int
Energy::occupancy()
{
    return kset_.ctx().max_occupancy();
}

int
Energy::nelectrons()
{
    return kset_.unit_cell().num_electrons();
}

std::shared_ptr<nlcglib::MatrixBaseZ>
Energy::get_hphi(nlcglib::memory_type memory = nlcglib::memory_type::none)
{
    return make_vector(this->hphis_, this->kset_.ctx(), this->kset_, memory);
}

std::shared_ptr<nlcglib::MatrixBaseZ>
Energy::get_sphi(nlcglib::memory_type memory = nlcglib::memory_type::none)
{
    return make_vector(this->sphis_, this->kset_.ctx(), this->kset_, memory);
}

std::shared_ptr<nlcglib::MatrixBaseZ>
Energy::get_C(nlcglib::memory_type memory = nlcglib::memory_type::none)
{
    return make_vector(this->cphis_, this->kset_.ctx(), this->kset_, memory);
}

std::shared_ptr<nlcglib::VectorBaseZ>
Energy::get_fn()
{
    const int ns = kset_.ctx().num_spins();
    int nbands   = kset_.ctx().num_bands();
    std::vector<std::vector<double>> fn;
    std::vector<std::pair<int, int>> kindices;
    for (auto it : kset_.spl_num_kpoints()) {
        auto& kp = *kset_.get<prec_t>(it.i);
        for (int ispn = 0; ispn < ns; ++ispn) {
            std::vector<double> fn_local(nbands);
            for (int i = 0; i < nbands; ++i) {
                fn_local[i] = kp.band_occupancy(i, ispn);
            }
            fn.push_back(std::move(fn_local));
            kindices.emplace_back(it.i, ispn);
        }
    }
    return std::make_shared<Array1d>(fn, kindices, kset_.comm().native());
}

void
Energy::set_fn(const std::vector<std::pair<int, int>>& keys, const std::vector<std::vector<double>>& fn)
{
    const int nbands = kset_.ctx().num_bands();
#ifndef NDEBUG
    const int ns         = kset_.ctx().num_spins();
    auto nk              = kset_.spl_num_kpoints().local_size();
    const double max_occ = ns == 1 ? 2.0 : 1.0;
#endif

    assert(static_cast<int>(fn.size()) == nk * ns);
    for (auto iloc = 0u; iloc < fn.size(); ++iloc) {
        // global k-point index
        int gidk           = keys[iloc].first;
        int ispn           = keys[iloc].second;
        auto& kp           = *kset_.get<prec_t>(gidk);
        const auto& fn_loc = fn[iloc];
        assert(static_cast<int>(fn_loc.size()) == nbands);
        for (int i = 0; i < nbands; ++i) {
            assert(fn_loc[i] >= 0);
            kp.band_occupancy(i, ispn, fn_loc[i]);
        }
    }
    kset_.sync_band<double, sync_band_t::occupancy>();
}

std::shared_ptr<nlcglib::VectorBaseZ>
Energy::get_ek()
{
    const int ns = kset_.ctx().num_spins();
    int nbands   = kset_.ctx().num_bands();
    std::vector<std::vector<double>> ek;
    std::vector<std::pair<int, int>> kindices;
    for (auto it : kset_.spl_num_kpoints()) {
        auto& kp = *kset_.get<prec_t>(it.i);
        for (int ispn = 0; ispn < ns; ++ispn) {
            std::vector<double> ek_local(nbands);
            for (int i = 0; i < nbands; ++i) {
                ek_local[i] = kp.band_energy(i, ispn);
            }
            ek.push_back(std::move(ek_local));
            kindices.emplace_back(it.i.get(), ispn);
        }
    }
    return std::make_shared<Array1d>(ek, kindices, kset_.comm().native());
}

std::shared_ptr<nlcglib::VectorBaseZ>
Energy::get_gkvec_ekin()
{
    const int ns = kset_.ctx().num_spins();
    std::vector<std::vector<double>> gkvec_cart;
    std::vector<std::pair<int, int>> kindices;
    for (auto it : kset_.spl_num_kpoints()) {
        auto& kp = *kset_.get<prec_t>(it.i);
        for (int ispn = 0; ispn < ns; ++ispn) {
            int gkvec_count = kp.gkvec().count();
            auto& gkvec     = kp.gkvec();
            std::vector<double> gkvec_local(gkvec_count);
            for (int i = 0; i < gkvec_count; ++i) {
                gkvec_local[i] = gkvec.gkvec_cart(gvec_index_t::global(i)).length();
            }
            gkvec_cart.push_back(std::move(gkvec_local));
            kindices.emplace_back(it.i.get(), ispn);
        }
    }
    return std::make_shared<Array1d>(gkvec_cart, kindices, kset_.comm().native());
}

std::shared_ptr<nlcglib::ScalarBaseZ>
Energy::get_kpoint_weights()
{
    const int ns = kset_.ctx().num_spins();
    std::vector<double> weights;
    std::vector<std::pair<int, int>> kindices;
    for (auto it : kset_.spl_num_kpoints()) {
        auto& kp = *kset_.get<double>(it.i);

        // also return weights for every spin index
        for (int ispn = 0; ispn < ns; ++ispn) {
            weights.push_back(kp.weight());
            kindices.emplace_back(it.i.get(), ispn);
        }
    }
    return std::make_shared<Scalar>(weights, kindices, kset_.comm().native());
}

double
Energy::get_total_energy()
{
    return etot_;
}

std::map<std::string, double>
Energy::get_energy_components()
{
    return energy_components_;
}

void
Energy::print_info() const
{
    auto& ctx       = kset_.ctx();
    auto& unit_cell = kset_.unit_cell();

    auto result_mag = density_.get_magnetisation();

    if (ctx.num_mag_dims() && ctx.comm().rank() == 0) {
        std::printf("atom              moment                |moment|");
        std::printf("\n");
        for (int i = 0; i < 80; i++) {
            std::printf("-");
        }
        std::printf("\n");

        for (int ia = 0; ia < unit_cell.num_atoms(); ia++) {
            r3::vector<double> v({result_mag[0].mt[ia], result_mag[1].mt[ia], result_mag[2].mt[ia]});
            std::printf("%4i  [%8.4f, %8.4f, %8.4f]  %10.6f", ia, v[0], v[1], v[2], v.length());
            std::printf("\n");
        }

        std::printf("\n");
    }
}

void
Energy::set_chemical_potential(double mu)
{
    // set Fermi energy.
    kset_.set_energy_fermi(mu);
}

double
Energy::get_chemical_potential()
{
    return kset_.energy_fermi();
}

Array1d::buffer_t
Array1d::get(int i)
{
    // call 1d constructor
    return buffer_t(data[i].size(), data[i].data(), nlcglib::memory_type::host);
}

const Array1d::buffer_t
Array1d::get(int i) const
{
    // call 1d constructor
    return buffer_t(data[i].size(), const_cast<double*>(data[i].data()), nlcglib::memory_type::host);
}

} // namespace sirius
#endif
