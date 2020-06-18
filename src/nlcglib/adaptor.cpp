#ifdef __NLCGLIB
#include <stdexcept>

#include "adaptor.hpp"
#include "apply_hamiltonian.hpp"
#include "hamiltonian/local_operator.hpp"
#include "hamiltonian/hamiltonian.hpp"
#include "dft/energy.hpp"
#include "SDDK/wf_inner.hpp"

using namespace nlcglib;

namespace sirius {

std::shared_ptr<Matrix> make_vector(const std::vector<std::shared_ptr<sddk::Wave_functions>>& wfct,
                                    const Simulation_context& ctx,
                                    const K_point_set& kset,
                                    nlcglib::memory_type memory = nlcglib::memory_type::none)
{
    std::map<memory_t, nlcglib::memory_type> memtype = {{memory_t::device, nlcglib::memory_type::device},
                                                        {memory_t::host, nlcglib::memory_type::host},
                                                        {memory_t::host_pinned, nlcglib::memory_type::host}};
    std::map<nlcglib::memory_type, memory_t> memtype_lookup = {{nlcglib::memory_type::none, memory_t::none},
                                                               {nlcglib::memory_type::device, memory_t::device},
                                                               {nlcglib::memory_type::host, memory_t::host},
                                                               {nlcglib::memory_type::host, memory_t::host_pinned}};

    memory_t target_memory = memtype_lookup.at(memory);
    if (target_memory == memory_t::none) {
        target_memory = ctx.preferred_memory_t();
    }

    std::vector<Matrix::buffer_t> data;
    std::vector<std::pair<int, int>> kpoint_indices;
    // sddk::memory_t preferred_memory = ctx.preferred_memory_t();
    int num_spins                     = ctx.num_spins();
    int nb                            = ctx.num_bands();
    for (auto i = 0u; i < wfct.size(); ++i) {
        auto gidk = kset.spl_num_kpoints(i); // global k-point index
        for (int ispn = 0; ispn < num_spins; ++ispn) {
            auto& array = wfct[i]->pw_coeffs(ispn).prime();
            int lda              = array.size(0);
            MPI_Comm comm        = wfct[i]->comm().mpi_comm();
            // check that wfct has been allocated
            if (is_device_memory(target_memory)) {
                // make sure that array is on device
                if (! array.on_device()) {
                    throw std::runtime_error("Error: expected device storage, but got nullptr");
                }
            }
            kpoint_indices.emplace_back(std::make_pair(gidk, ispn));
            data.emplace_back(std::array<int, 2>{1, lda},   /* stride */
                              std::array<int, 2>{lda, nb},  /* size */
                              array.at(target_memory), /* pointer */
                              memtype.at(target_memory),
                              comm /* mpi communicator */);
        }
    }
    return std::make_shared<Matrix>(std::move(data), std::move(kpoint_indices), kset.comm().mpi_comm());
} // namespace sirius

Matrix::buffer_t Matrix::get(int i)
{
    return data[i];
}

const Matrix::buffer_t Matrix::get(int i) const
{
    return data[i];
}

Energy::Energy(K_point_set& kset, Density& density, Potential& potential)
    : kset(kset)
    , density(density)
    , potential(potential)
{
    // intialize hphi and sphi and allocate (device) memory
    int nk    = kset.spl_num_kpoints().local_size();
    auto& ctx = kset.ctx();
    // auto& mpd = ctx.mem_pool(ctx.preferred_memory_t());
    hphis.resize(nk);
    sphis.resize(nk);
    cphis.resize(nk);
    for (int i = 0; i < nk; ++i) {
        auto global_kpoint_index = kset.spl_num_kpoints(i);
        auto& kp = *kset[global_kpoint_index];
        int num_wf                        = ctx.num_bands();
        sddk::memory_t preferred_memory_t = ctx.preferred_memory_t();
        int num_spins                     = ctx.num_spins();
        // make a new wf for Hamiltonian apply...
        hphis[i] = std::make_shared<sddk::Wave_functions>(kp.gkvec_partition(), num_wf, preferred_memory_t, num_spins);
        hphis[i]->allocate(sddk::spin_range(num_spins), ctx.preferred_memory_t());
        sphis[i] = std::make_shared<sddk::Wave_functions>(kp.gkvec_partition(), num_wf, preferred_memory_t, num_spins);
        sphis[i]->allocate(sddk::spin_range(num_spins), ctx.preferred_memory_t());
        cphis[i] = kp.spinor_wave_functions_ptr();
        // allocate on device
        if (is_device_memory(ctx.preferred_memory_t())) {
            const int num_sc = (ctx.num_mag_dims() == 3) ? 2 : 1;
            auto& mpd = ctx.mem_pool(memory_t::device);
            for (int ispn = 0; ispn < num_sc; ispn++) {
                hphis[i]->pw_coeffs(ispn).allocate(mpd);
                sphis[i]->pw_coeffs(ispn).allocate(mpd);
            }
        }
    }
    // need to allocate wavefunctions on GPU
}

void Energy::compute()
{
    auto& ctx = kset.ctx();
    int num_spins = ctx.num_spins();
    int num_bands = ctx.num_bands();
    int nk = kset.spl_num_kpoints().local_size();

    // // // transfer from device to host (only if data on GPU is present)
    // if(is_device_memory(ctx.preferred_memory_t())) {
    //     for (int ik = 0; ik < nk; ++ik) {
    //         for (int ispn = 0; ispn < num_spins; ++ispn) {
    //             int num_wf = cphis[ik]->num_wf();
    //             if (cphis[ik]->pw_coeffs(ispn).prime().on_device()) {
    //                 // std::cout << "copying wfc from DEVICE -> HOST" << "\n";
    //                 cphis[ik]->pw_coeffs(ispn).copy_to(memory_t::host, 0, num_wf);
    //             }

    //         }
    //     }
    // }

    density.generate(kset, true /* add core */, false /* transform to rg */);

    if (ctx.use_symmetry()) {
        density.symmetrize();
        density.symmetrize_density_matrix();
    }

    density.fft_transform(1);
    potential.generate(density);

    if (ctx.use_symmetry()) {
        potential.symmetrize();
    }
    potential.fft_transform(1);


    /* compute H@X and new band energies */
    memory_t mem{memory_t::host};
    linalg_t la{linalg_t::blas};
    if (ctx.processing_unit() == device_t::GPU) {
        mem = memory_t::device;
        la  = linalg_t::gpublas;
    }
    auto H0 = Hamiltonian0(potential);
    // apply Hamiltonian
    for (int i = 0; i < nk; ++i) {
        auto& kp = *kset[kset.spl_num_kpoints(i)];
        std::vector<double> band_energies(num_bands);

        if (is_device_memory(ctx.preferred_memory_t())) {
            auto& mpd        = ctx.mem_pool(memory_t::device);
            for (int ispn = 0; ispn < num_spins; ispn++) {
                cphis[i]->pw_coeffs(ispn).allocate(mpd);
                // copy to device
                int num_wf = cphis[i]->num_wf();
                cphis[i]->pw_coeffs(ispn).copy_to(memory_t::device, 0, num_wf);
            }
        }

        assert(cphis[i] == kp.spinor_wave_functions_ptr());
        apply_hamiltonian(H0, kp, *hphis[i], kp.spinor_wave_functions(), sphis[i]);
        // compute band energies
        for (int ispn = 0; ispn < num_spins; ++ispn) {
            for (int jj = 0; jj < num_bands; ++jj) {
                dmatrix<std::complex<double>> dmat(1, 1, memory_t::host);
                dmat.allocate(memory_t::device);
                sddk::inner(mem, la, ispn,
                            /* bra */ kp.spinor_wave_functions(), jj, 1,
                            /* ket */ *hphis[i], jj, 1,
                            /* out */ dmat, 0, 0);
                // deal with memory...
                // assert(std::abs(dmat(0, 0).imag()) < 1e-10);
                kp.band_energy(jj, ispn, dmat(0, 0).real());
            }
        }
    }
    kset.sync_band_energies();

    // evaluate total energy
    double eewald = ewald_energy(ctx, ctx.gvec(), ctx.unit_cell());
    this->etot    = total_energy(ctx, kset, density, potential, eewald);
}


void Energy::set_occupation_numbers(const std::vector<std::vector<double>>& fn)
{
    auto nk      = kset.spl_num_kpoints().local_size();
    const int ns = kset.ctx().num_spins();
    if (nk * ns != int(fn.size())) {
        throw std::runtime_error("set_occupation_numbers: wrong number of k-points");
    }

    for (auto i = 0u; i < fn.size(); ++i) {
        int ik   = i / ns;
        int ispn = i % ns;
        auto& kp = *kset[kset.spl_num_kpoints(ik)];
        // BEWARE: nothing is allocated, it must be done outside.
        for (auto j = 0u; j < fn[i].size(); ++j) {
            kp.band_occupancy(j, ispn, fn[i][j]);
        }
    }
}

int Energy::occupancy()
{
    return kset.ctx().max_occupancy();
}

int Energy::nelectrons()
{
    return kset.unit_cell().num_electrons();
}

std::shared_ptr<nlcglib::MatrixBaseZ> Energy::get_hphi()
{
    return make_vector(this->hphis, this->kset.ctx(), this->kset);
}

std::shared_ptr<nlcglib::MatrixBaseZ> Energy::get_sphi()
{
    return make_vector(this->sphis, this->kset.ctx(), this->kset);
}

std::shared_ptr<nlcglib::MatrixBaseZ> Energy::get_C(nlcglib::memory_type memory = nlcglib::memory_type::none)
{
    return make_vector(this->cphis, this->kset.ctx(), this->kset, memory);
}

std::shared_ptr<nlcglib::VectorBaseZ> Energy::get_fn()
{
    auto nk      = kset.spl_num_kpoints().local_size();
    const int ns = kset.ctx().num_spins();
    int nbands = kset.ctx().num_bands();
    std::vector<std::vector<double>> fn;
    std::vector<std::pair<int, int>> kindices;
    for (int ik = 0; ik < nk; ++ik) {
        // global k-point index
        auto gidk = kset.spl_num_kpoints(ik);
        auto& kp = *kset[gidk];
        for (int ispn = 0; ispn < ns; ++ispn) {
            std::vector<double> fn_local(nbands);
            for (int i = 0; i < nbands; ++i) {
                fn_local[i] = kp.band_occupancy(i, ispn);
            }
            fn.push_back(std::move(fn_local));
            kindices.emplace_back(gidk, ispn);
        }
    }
    return std::make_shared<Array1d>(fn, kindices, kset.comm().mpi_comm());
}

void Energy::set_fn(const std::vector<std::vector<double>>& fn)
{
    auto nk      = kset.spl_num_kpoints().local_size();
    const int ns = kset.ctx().num_spins();
    const int nbands   = kset.ctx().num_bands();
    #ifdef DEBUG
    const double max_occ = ns == 1 ? 2.0 : 1.0;
    #endif

    assert(static_cast<int>(fn.size()) == nk*ns);
    for (int ik = 0; ik < nk; ++ik) {
        // global k-point index
        auto gidk = kset.spl_num_kpoints(ik);
        auto& kp  = *kset[gidk];
        for (int ispn = 0; ispn < ns; ++ispn) {
            const auto& fn_loc = fn[ik * ns + ispn];
            assert(static_cast<int>(fn_loc.size()) == nbands);
            for (int i = 0; i < nbands; ++i)
            {
                assert(fn_loc[i] >= 0 && fn_loc[i] <= max_occ);
                kp.band_occupancy(i, ispn, fn_loc[i]);
            }
        }
    }
    kset.sync_band_occupancies();
}

std::shared_ptr<nlcglib::VectorBaseZ> Energy::get_ek()
{
    auto nk      = kset.spl_num_kpoints().local_size();
    const int ns = kset.ctx().num_spins();
    int nbands   = kset.ctx().num_bands();
    std::vector<std::vector<double>> ek;
    std::vector<std::pair<int, int>> kindices;
    for (int ik = 0; ik < nk; ++ik) {
        // global k-point index
        auto gidk = kset.spl_num_kpoints(ik);
        auto& kp  = *kset[gidk];
        for (int ispn = 0; ispn < ns; ++ispn) {
            std::vector<double> ek_local(nbands);
            for (int i = 0; i < nbands; ++i) {
                ek_local[i] = kp.band_energy(i, ispn);
            }
            ek.push_back(std::move(ek_local));
            kindices.emplace_back(gidk, ispn);
        }
    }
    return std::make_shared<Array1d>(ek, kindices, kset.comm().mpi_comm());
}

std::shared_ptr<nlcglib::VectorBaseZ> Energy::get_gkvec_ekin()
{
    auto nk      = kset.spl_num_kpoints().local_size();
    const int ns = kset.ctx().num_spins();
    std::vector<std::vector<double>> gkvec_cart;
    std::vector<std::pair<int, int>> kindices;
    for (int ik = 0; ik < nk; ++ik) {
        // global k-point index
        auto gidk = kset.spl_num_kpoints(ik);
        auto& kp  = *kset[gidk];
        for (int ispn = 0; ispn < ns; ++ispn) {
            int gkvec_count = kp.gkvec().count();
            auto& gkvec = kp.gkvec();
            std::vector<double> gkvec_local(gkvec_count);
            for (int i = 0; i < gkvec_count; ++i) {
                gkvec_local[i] = gkvec.gkvec_cart<index_domain_t::global>(i).length();
            }
            gkvec_cart.push_back(std::move(gkvec_local));
            kindices.emplace_back(gidk, ispn);
        }
    }
    return std::make_shared<Array1d>(gkvec_cart, kindices, kset.comm().mpi_comm());
}

std::shared_ptr<nlcglib::ScalarBaseZ> Energy::get_kpoint_weights()
{
    auto nk = kset.spl_num_kpoints().local_size();
    const int ns = kset.ctx().num_spins();
    std::vector<double> weights;
    std::vector<std::pair<int, int>> kindices;
    for (int ik = 0; ik < nk; ++ik) {
        // global k-point index
        auto gidk = kset.spl_num_kpoints(ik);
        auto& kp  = *kset[gidk];

        // also return weights for every spin index
        for (int ispn = 0; ispn < ns; ++ispn) {
            weights.push_back(kp.weight());
            kindices.emplace_back(gidk, ispn);
        }
    }
    return std::make_shared<Scalar>(weights, kindices, kset.comm().mpi_comm());
}

double Energy::get_total_energy()
{
    return etot;
}

void Energy::set_wfct(nlcglib::MatrixBaseZ& vector)
{
    throw std::runtime_error("not implemented.");
}

void Energy::print_info() const
{
    auto& ctx = kset.ctx();
    auto& unit_cell = kset.unit_cell();

    auto result_mag = density.get_magnetisation();
    // auto total_mag  = std::get<0>(result_mag);
    // auto it_mag     = std::get<1>(result_mag);
    auto mt_mag     = std::get<2>(result_mag);

    if (ctx.num_mag_dims()) {
        std::printf("atom              moment                |moment|");
        std::printf("\n");
        for (int i = 0; i < 80; i++) {
            std::printf("-");
        }
        std::printf("\n");

        for (int ia = 0; ia < unit_cell.num_atoms(); ia++) {
            vector3d<double> v(mt_mag[ia]);
            std::printf("%4i  [%8.4f, %8.4f, %8.4f]  %10.6f", ia, v[0], v[1], v[2], v.length());
            std::printf("\n");
        }

        std::printf("\n");
    }
}

Array1d::buffer_t Array1d::get(int i)
{
    // call 1d constructor
    return buffer_t(data[i].size(), data[i].data(), nlcglib::memory_type::host);
}

const Array1d::buffer_t Array1d::get(int i) const
{
    // call 1d constructor
    return buffer_t(data[i].size(), const_cast<double*>(data[i].data()), nlcglib::memory_type::host);
}

} // namespace sirius
#endif
