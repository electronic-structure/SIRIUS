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

/** \file simulation_context.cpp
 *
 *  \brief Implementation of Simulation_context class.
 */

#include <gsl/gsl_sf_bessel.h>
#include "sirius_version.hpp"
#include "simulation_context.hpp"
#include "symmetry/lattice.hpp"
#include "symmetry/crystal_symmetry.hpp"
#include "symmetry/check_gvec.hpp"
#include "utils/profiler.hpp"
#include "utils/env.hpp"
#include "SDDK/omp.hpp"
#include "potential/xc_functional.hpp"
#include "linalg/linalg_spla.hpp"

namespace sirius {

double
unit_step_function_form_factors(double R__, double g__)
{
    if (g__ < 1e-12) {
        return std::pow(R__, 3) / 3.0;
    } else {
        return (std::sin(g__ * R__) - g__ * R__ * std::cos(g__ * R__)) / std::pow(g__, 3);
    }
}

template <>
spfft::Transform& Simulation_context::spfft<double>()
{
    return *spfft_transform_;
}

template <>
spfft::Transform const& Simulation_context::spfft<double>() const
{
    return *spfft_transform_;
}

template <>
spfft::Transform& Simulation_context::spfft_coarse<double>()
{
    return *spfft_transform_coarse_;
}

template <>
spfft::Transform const& Simulation_context::spfft_coarse<double>() const
{
    return *spfft_transform_coarse_;
}

template <>
spfft::Grid& Simulation_context::spfft_grid_coarse<double>()
{
    return *spfft_grid_coarse_;
}

#if defined(USE_FP32)
template <>
spfft::TransformFloat& Simulation_context::spfft<float>()
{
    return *spfft_transform_float_;
}

template <>
spfft::TransformFloat const& Simulation_context::spfft<float>() const
{
    return *spfft_transform_float_;
}

template <>
spfft::TransformFloat& Simulation_context::spfft_coarse<float>()
{
    return *spfft_transform_coarse_float_;
}

template <>
spfft::TransformFloat const& Simulation_context::spfft_coarse<float>() const
{
    return *spfft_transform_coarse_float_;
}

template <>
spfft::GridFloat& Simulation_context::spfft_grid_coarse<float>()
{
    return *spfft_grid_coarse_float_;
}
#endif

void
Simulation_context::init_fft_grid()
{
    if (!(cfg().control().fft_mode() == "serial" || cfg().control().fft_mode() == "parallel")) {
        RTE_THROW("wrong FFT mode");
    }

    auto rlv = unit_cell().reciprocal_lattice_vectors();

    /* create FFT driver for dense mesh (density and potential) */
    auto fft_grid = cfg().settings().fft_grid_size();
    if (fft_grid[0] * fft_grid[1] * fft_grid[2] == 0) {
        fft_grid_ = sddk::get_min_fft_grid(pw_cutoff(), rlv);
        cfg().settings().fft_grid_size(fft_grid_);
    } else {
        /* else create a grid with user-specified dimensions */
        fft_grid_ = sddk::FFT3D_grid(fft_grid);
    }

    /* create FFT grid for coarse mesh */
    fft_coarse_grid_ = sddk::get_min_fft_grid(2 * gk_cutoff(), rlv);
}

sddk::mdarray<double, 3>
Simulation_context::generate_sbessel_mt(int lmax__) const
{
    PROFILE("sirius::Simulation_context::generate_sbessel_mt");

    sddk::mdarray<double, 3> sbessel_mt(lmax__ + 1, gvec().count(), unit_cell().num_atom_types());
    for (int iat = 0; iat < unit_cell().num_atom_types(); iat++) {
        #pragma omp parallel for schedule(static)
        for (int igloc = 0; igloc < gvec().count(); igloc++) {
            auto gv = gvec().gvec_cart<sddk::index_domain_t::local>(igloc);
            gsl_sf_bessel_jl_array(lmax__, gv.length() * unit_cell().atom_type(iat).mt_radius(),
                                   &sbessel_mt(0, igloc, iat));
        }
    }
    return sbessel_mt;
}

sddk::matrix<double_complex>
Simulation_context::generate_gvec_ylm(int lmax__)
{
    PROFILE("sirius::Simulation_context::generate_gvec_ylm");

    sddk::matrix<double_complex> gvec_ylm(utils::lmmax(lmax__), gvec().count(), sddk::memory_t::host, "gvec_ylm");
    #pragma omp parallel for schedule(static)
    for (int igloc = 0; igloc < gvec().count(); igloc++) {
        auto rtp = SHT::spherical_coordinates(gvec().gvec_cart<sddk::index_domain_t::local>(igloc));
        sf::spherical_harmonics(lmax__, rtp[1], rtp[2], &gvec_ylm(0, igloc));
    }
    return gvec_ylm;
}

sddk::mdarray<double_complex, 2>
Simulation_context::sum_fg_fl_yg(int lmax__, double_complex const* fpw__, sddk::mdarray<double, 3>& fl__,
                                 sddk::matrix<double_complex>& gvec_ylm__)
{
    PROFILE("sirius::Simulation_context::sum_fg_fl_yg");

    int ngv_loc = gvec().count();

    int na_max{0};
    for (int iat = 0; iat < unit_cell().num_atom_types(); iat++) {
        na_max = std::max(na_max, unit_cell().atom_type(iat).num_atoms());
    }

    int lmmax = utils::lmmax(lmax__);
    /* resuling matrix */
    sddk::mdarray<double_complex, 2> flm(lmmax, unit_cell().num_atoms());

    sddk::matrix<double_complex> phase_factors;
    sddk::matrix<double_complex> zm;
    sddk::matrix<double_complex> tmp;

    switch (processing_unit()) {
        case sddk::device_t::CPU: {
            auto& mp      = get_memory_pool(sddk::memory_t::host);
            phase_factors = sddk::matrix<double_complex>(ngv_loc, na_max, mp);
            zm            = sddk::matrix<double_complex>(lmmax, ngv_loc, mp);
            tmp           = sddk::matrix<double_complex>(lmmax, na_max, mp);
            break;
        }
        case sddk::device_t::GPU: {
            auto& mp      = get_memory_pool(sddk::memory_t::host);
            auto& mpd     = get_memory_pool(sddk::memory_t::device);
            phase_factors = sddk::matrix<double_complex>(nullptr, ngv_loc, na_max);
            phase_factors.allocate(mpd);
            zm = sddk::matrix<double_complex>(lmmax, ngv_loc, mp);
            zm.allocate(mpd);
            tmp = sddk::matrix<double_complex>(lmmax, na_max, mp);
            tmp.allocate(mpd);
            break;
        }
    }

    std::vector<double_complex> zil(lmax__ + 1);
    for (int l = 0; l <= lmax__; l++) {
        zil[l] = std::pow(double_complex(0, 1), l);
    }

    for (int iat = 0; iat < unit_cell().num_atom_types(); iat++) {
        int na = unit_cell().atom_type(iat).num_atoms();
        generate_phase_factors(iat, phase_factors);
        PROFILE_START("sirius::Simulation_context::sum_fg_fl_yg|zm");
        #pragma omp parallel for schedule(static)
        for (int igloc = 0; igloc < ngv_loc; igloc++) {
            for (int l = 0, lm = 0; l <= lmax__; l++) {
                double_complex z = fourpi * fl__(l, igloc, iat) * zil[l] * fpw__[igloc];
                for (int m = -l; m <= l; m++, lm++) {
                    zm(lm, igloc) = z * std::conj(gvec_ylm__(lm, igloc));
                }
            }
        }
        PROFILE_STOP("sirius::Simulation_context::sum_fg_fl_yg|zm");
        PROFILE_START("sirius::Simulation_context::sum_fg_fl_yg|mul");
        switch (processing_unit()) {
            case sddk::device_t::CPU: {
                sddk::linalg(sddk::linalg_t::blas)
                    .gemm('N', 'N', lmmax, na, ngv_loc, &sddk::linalg_const<double_complex>::one(), zm.at(sddk::memory_t::host),
                          zm.ld(), phase_factors.at(sddk::memory_t::host), phase_factors.ld(),
                          &sddk::linalg_const<double_complex>::zero(), tmp.at(sddk::memory_t::host), tmp.ld());
                break;
            }
            case sddk::device_t::GPU: {
                zm.copy_to(sddk::memory_t::device);
                sddk::linalg(sddk::linalg_t::gpublas)
                    .gemm('N', 'N', lmmax, na, ngv_loc, &sddk::linalg_const<double_complex>::one(), zm.at(sddk::memory_t::device),
                          zm.ld(), phase_factors.at(sddk::memory_t::device), phase_factors.ld(),
                          &sddk::linalg_const<double_complex>::zero(), tmp.at(sddk::memory_t::device), tmp.ld());
                tmp.copy_to(sddk::memory_t::host);
                break;
            }
        }
        PROFILE_STOP("sirius::Simulation_context::sum_fg_fl_yg|mul");

        for (int i = 0; i < na; i++) {
            int ia = unit_cell().atom_type(iat).atom_id(i);
            std::copy(&tmp(0, i), &tmp(0, i) + lmmax, &flm(0, ia));
        }
    }

    comm().allreduce(&flm(0, 0), (int)flm.size());

    return flm;
}

double
Simulation_context::ewald_lambda() const
{
    /* alpha = 1 / (2*sigma^2), selecting alpha here for better convergence */
    double lambda{1};
    double gmax = pw_cutoff();
    double upper_bound{0};
    double charge = unit_cell().num_electrons();

    /* iterate to find lambda */
    do {
        lambda += 0.1;
        upper_bound =
            charge * charge * std::sqrt(2.0 * lambda / twopi) * std::erfc(gmax * std::sqrt(1.0 / (4.0 * lambda)));
    } while (upper_bound < 1e-8);

    if (lambda < 1.5 && comm().rank() == 0) {
        std::stringstream s;
        s << "ewald_lambda(): pw_cutoff is too small";
        WARNING(s);
    }
    return lambda;
}

sddk::splindex<sddk::splindex_t::block>
Simulation_context::split_gvec_local() const
{
    /* local number of G-vectors for this MPI rank */
    int ngv_loc = gvec().count();
    /* estimate number of G-vectors in a block */
    int ld{-1};
    for (int iat = 0; iat < unit_cell().num_atom_types(); iat++) {
        int nat = unit_cell().atom_type(iat).num_atoms();
        int nbf = unit_cell().atom_type(iat).mt_basis_size();

        ld = std::max(ld, std::max(nbf * (nbf + 1) / 2, nat));
    }
    /* limit the size of relevant array to ~1Gb */
    int ngv_b = (1 << 30) / sizeof(double_complex) / ld;
    ngv_b     = std::max(1, std::min(ngv_loc, ngv_b));
    /* number of blocks of G-vectors */
    int nb = ngv_loc / ngv_b;
    /* split local number of G-vectors between blocks */
    return sddk::splindex<sddk::splindex_t::block>(ngv_loc, nb, 0);
}

void
Simulation_context::initialize()
{
    PROFILE("sirius::Simulation_context::initialize");

    /* can't initialize twice */
    if (initialized_) {
        RTE_THROW("Simulation parameters are already initialized.");
    }

    /* setup the output stream */
    if (this->comm().rank() == 0) {
        output_stream_ = &std::cout;
    } else {
        output_stream_ = &utils::null_stream__;
    }

    auto verb_lvl = env::get_value_ptr<int>("SIRIUS_VERBOSITY");
    if (verb_lvl) {
        this->verbosity(*verb_lvl);
    }

    electronic_structure_method(cfg().parameters().electronic_structure_method());
    core_relativity(cfg().parameters().core_relativity());
    valence_relativity(cfg().parameters().valence_relativity());

    /* can't run fp-lapw with Gamma point trick */
    if (full_potential()) {
        gamma_point(false);
    }

    /* Gamma-point calculation and non-collinear magnetism are not compatible */
    if (num_mag_dims() == 3) {
        gamma_point(false);
    }

    /* set processing unit type */
    processing_unit(cfg().control().processing_unit());

    /* check if we can use a GPU device */
    if (processing_unit() == sddk::device_t::GPU) {
#if !defined(SIRIUS_GPU)
        RTE_THROW("not compiled with GPU support!");
#endif
    }

    /* initialize MPI communicators */
    init_comm();

    auto print_mpi_layout = env::print_mpi_layout();

    if (verbosity() >= 3 || print_mpi_layout) {
        sddk::pstdout pout(comm());
        if (comm().rank() == 0) {
            pout << "MPI rank placement" << std::endl;
            pout << "------------------" << std::endl;
        }
        pout << "rank: " << comm().rank()
             << ", comm_band_rank: " << comm_band().rank()
             << ", comm_k_rank: " << comm_k().rank()
             << ", hostname: " << utils::hostname()
             << ", mpi processor name: " << sddk::Communicator::processor_name() << std::endl;
        this->out() << pout.flush(0);
    }

    switch (processing_unit()) {
        case sddk::device_t::CPU: {
            host_memory_t_ = sddk::memory_t::host;
            break;
        }
        case sddk::device_t::GPU: {
            host_memory_t_ = sddk::memory_t::host_pinned;
            break;
        }
    }

    if (processing_unit() == sddk::device_t::GPU) {
        spla_ctx_.reset(new spla::Context{SPLA_PU_GPU});
        spla_ctx_->set_tile_size_gpu(1688); // limit GPU memory usage to around 500MB
    }
    // share context for blas operations to reduce memory consumption
    splablas::get_handle_ptr() = spla_ctx_;

    /* can't use reduced G-vectors in LAPW code */
    if (full_potential()) {
        cfg().control().reduce_gvec(false);
    }

    if (!cfg().iterative_solver().type().size() || (cfg().iterative_solver().type() == "auto")) {
        if (full_potential()) {
            cfg().iterative_solver().type("exact");
        } else {
            cfg().iterative_solver().type("davidson");
        }
    }
    /* set default values for the G-vector cutoff */
    if (pw_cutoff() <= 0) {
        pw_cutoff(full_potential() ? 12 : 20);
    }

    /* initialize variables related to the unit cell */
    unit_cell().initialize();
    /* save the volume of the initial unit cell */
    omega0_ = unit_cell().omega();
    /* save initial lattice vectors */
    lattice_vectors0_ = unit_cell().lattice_vectors();

    /* check the lattice symmetries */
    if (use_symmetry()) {
        auto lv = matrix3d<double>(unit_cell().lattice_vectors());

        auto lat_sym = find_lat_sym(lv, cfg().control().spglib_tolerance());

        #pragma omp parallel for
        for (int i = 0; i < unit_cell().symmetry().size(); i++) {
            auto& spgR = unit_cell().symmetry()[i].spg_op.R;
            bool found{false};
            for (size_t i = 0; i < lat_sym.size(); i++) {
                auto latR = lat_sym[i];
                found     = true;
                for (int x : {0, 1, 2}) {
                    for (int y : {0, 1, 2}) {
                        found = found && (spgR(x, y) == latR(x, y));
                    }
                }
                if (found) {
                    break;
                }
            }
            if (!found) {
                RTE_THROW("spglib lattice symmetry was not found in the list of SIRIUS generated symmetries");
            }
        }
    }

    /* force global spin-orbit correction flag if one of the species has spin-orbit */
    for (int iat = 0; iat < unit_cell().num_atom_types(); iat++) {
        if (unit_cell().atom_type(iat).spin_orbit_coupling()) {
            this->so_correction(true);
        }
    }

    /* set default values for the G+k-vector cutoff */
    if (full_potential()) {
        /* find the cutoff for G+k vectors (derived from rgkmax (aw_cutoff here) and minimum MT radius) */
        if (aw_cutoff() > 0) {
            gk_cutoff(aw_cutoff() / unit_cell().min_mt_radius());
        }
        if (gk_cutoff() <= 0) {
            gk_cutoff(3);
        }
    } else {
        /* in pseudopotential case */
        if (gk_cutoff() <= 0) {
            gk_cutoff(7);
        }
    }

    /* check the G+k cutoff */
    if (gk_cutoff() * 2 > pw_cutoff()) {
        std::stringstream s;
        s << "G+k cutoff is too large for a given plane-wave cutoff" << std::endl
          << "  pw cutoff : " << pw_cutoff() << std::endl
          << "  doubled G+k cutoff : " << gk_cutoff() * 2;
        RTE_THROW(s);
    }

    if (!full_potential()) {
        lmax_rho(unit_cell().lmax() * 2);
        lmax_pot(unit_cell().lmax() * 2);
        lmax_apw(-1);
    }

    /* initialize FFT grid dimensions */
    init_fft_grid();

    int nbnd = static_cast<int>(unit_cell().num_valence_electrons() / 2.0) +
               std::max(10, static_cast<int>(0.1 * unit_cell().num_valence_electrons()));
    if (full_potential()) {
        /* take 10% of empty non-magnetic states */
        if (num_fv_states() < 0) {
            num_fv_states(nbnd);
        }
        if (num_fv_states() < static_cast<int>(unit_cell().num_valence_electrons() / 2.0)) {
            std::stringstream s;
            s << "not enough first-variational states : " << num_fv_states();
            RTE_THROW(s);
        }
    } else {
        if (num_mag_dims() == 3) {
            nbnd *= 2;
        }
        /* if number of bands was not set by the host code, set it here */
        if (num_bands() < 0) {
            num_bands(nbnd);
        }
    }

    std::string evsn[] = {std_evp_solver_name(), gen_evp_solver_name()};
#if defined(SIRIUS_CUDA)
    bool is_cuda{true};
#else
    bool is_cuda{false};
#endif
#if defined(SIRIUS_MAGMA)
    bool is_magma{true};
#else
    bool is_magma{false};
#endif
#if defined(SIRIUS_SCALAPACK)
    bool is_scalapack{true};
#else
    bool is_scalapack{false};
#endif
#if defined(SIRIUS_ELPA)
    bool is_elpa{true};
#else
    bool is_elpa{false};
#endif

    if (processing_unit() == sddk::device_t::CPU || acc::num_devices() == 0) {
        is_cuda  = false;
        is_magma = false;
    }

    int npr = cfg().control().mpi_grid_dims()[0];
    int npc = cfg().control().mpi_grid_dims()[1];

    /* deduce the default eigen-value solver */
    for (int i : {0, 1}) {
        if (evsn[i] == "auto") {
            /* conditions for sequential diagonalization */
            if (comm_band().size() == 1 || npc == 1 || npr == 1 || !is_scalapack) {
                if (full_potential()) {
                    if (is_magma) {
                        evsn[i] = "magma";
                    } else if (is_cuda) {
                        evsn[i] = "cusolver";
                    } else {
                        evsn[i] = "lapack";
                    }
                } else {
                    if (is_cuda) {
                        evsn[i] = "cusolver";
                    } else if (is_magma && num_bands() > 200) {
                        evsn[i] = "magma";
                    } else {
                        evsn[i] = "lapack";
                    }
                }
            } else {
                if (is_scalapack) {
                    evsn[i] = "scalapack";
                }
                if (is_elpa) {
                    evsn[i] = "elpa1";
                }
            }
        }
    }

    auto ev_str = env::get_ev_solver();
    if (ev_str.size()) {
        evsn[0] = ev_str;
        evsn[1] = ev_str;
    }

    std_evp_solver_name(evsn[0]);
    gen_evp_solver_name(evsn[1]);

    std_evp_solver_ = Eigensolver_factory(std_evp_solver_name());
    gen_evp_solver_ = Eigensolver_factory(gen_evp_solver_name());

    auto& std_solver = std_evp_solver();
    auto& gen_solver = gen_evp_solver();

    if (std_solver.is_parallel() != gen_solver.is_parallel()) {
        RTE_THROW("both solvers must be sequential or parallel");
    }

    /* setup BLACS grid */
    if (std_solver.is_parallel()) {
        blacs_grid_ = std::unique_ptr<sddk::BLACS_grid>(new sddk::BLACS_grid(comm_band(), npr, npc));
    } else {
        blacs_grid_ = std::unique_ptr<sddk::BLACS_grid>(new sddk::BLACS_grid(sddk::Communicator::self(), 1, 1));
    }

    /* setup the cyclic block size */
    if (cyclic_block_size() < 0) {
        double a = std::min(std::log2(double(num_bands()) / blacs_grid_->num_ranks_col()),
                            std::log2(double(num_bands()) / blacs_grid_->num_ranks_row()));
        if (a < 1) {
            cfg().control().cyclic_block_size(2);
        } else {
            cfg().control().cyclic_block_size(
                static_cast<int>(std::min(128.0, std::pow(2.0, static_cast<int>(a))) + 1e-12));
        }
    }

    /* placeholder for augmentation operator for each atom type */
    augmentation_op_.resize(unit_cell().num_atom_types());

    if (this->hubbard_correction()) {
        /* if spin orbit coupling or non collinear magnetisms are activated, then
           we consider the full spherical hubbard correction */
        if (this->so_correction() || this->num_mag_dims() == 3) {
            this->cfg().hubbard().simplified(false);
        }
    }

    if (cfg().parameters().precision_wf() == "fp32" && cfg().parameters().precision_gs() == "fp64") {
        double t = std::numeric_limits<float>::epsilon() * 10;
        auto tol = std::max(cfg().settings().itsol_tol_min(), t);
        cfg().settings().itsol_tol_min(tol);
    }


    /* set the smearing */
    smearing(cfg().parameters().smearing());

    /* create G-vectors on the first call to update() */
    update();

    ::sirius::print_memory_usage(__FILE__, __LINE__, this->out());

    if (verbosity() >= 1 && comm().rank() == 0) {
        print_info(this->out());
    }

    auto pcs = env::print_checksum();
    if (pcs) {
        this->cfg().control().print_checksum(true);
    }

    initialized_ = true;
    cfg().lock();
}

void
Simulation_context::print_info(std::ostream& out__) const
{
    {
        rte::rte_ostream os(out__, "info");
        tm const* ptm = localtime(&start_time_.tv_sec);
        char buf[100];
        strftime(buf, sizeof(buf), "%a, %e %b %Y %H:%M:%S", ptm);

        os << "SIRIUS version : " << sirius::major_version() << "." << sirius::minor_version()
           << "." << sirius::revision() << std::endl
           << "git hash       : " << sirius::git_hash() << std::endl
           << "git branch     : " << sirius::git_branchname() << std::endl
           << "build time     : " << sirius::build_date() << std::endl
           << "start time     : " << std::string(buf) << std::endl
           << std::endl
           << "number of MPI ranks           : " << this->comm().size() << std::endl;
        if (mpi_grid_) {
            os << "MPI grid                      :";
            for (int i = 0; i < mpi_grid_->num_dimensions(); i++) {
                os << " " << mpi_grid_->communicator(1 << i).size();
            }
            os << std::endl;
        }
        os << "maximum number of OMP threads : " << omp_get_max_threads() << std::endl
           << "number of MPI ranks per node  : " << sddk::num_ranks_per_node() << std::endl
           << "page size (Kb)                : " << (utils::get_page_size() >> 10) << std::endl
           << "number of pages               : " << utils::get_num_pages() << std::endl
           << "available memory (GB)         : " << (utils::get_total_memory() >> 30) << std::endl;
        os << std::endl;
    }
    {
        rte::rte_ostream os(out__, "fft");
        std::string headers[]       = {"FFT context for density and potential", "FFT context for coarse grid"};
        double cutoffs[]            = {pw_cutoff(), 2 * gk_cutoff()};
        sddk::Communicator const* comms[] = {&comm_fft(), &comm_fft_coarse()};
        sddk::FFT3D_grid fft_grids[]      = {this->fft_grid_, this->fft_coarse_grid_};
        sddk::Gvec const* gvecs[]         = {&gvec(), &gvec_coarse()};

        for (int i = 0; i < 2; i++) {
            os << headers[i] << std::endl
               << utils::hbar(37, '=') << std::endl
               << "  comm size                             : " << comms[i]->size() << std::endl
               << "  plane wave cutoff                     : " << cutoffs[i] << std::endl
               << "  grid size                             : " << fft_grids[i][0] << " "
                                                               << fft_grids[i][1] << " "
                                                               << fft_grids[i][2] << "   total : "
                                                               << fft_grids[i].num_points() << std::endl
               << "  grid limits                           : " << fft_grids[i].limits(0).first << " "
                                                               << fft_grids[i].limits(0).second << "   "
                                                               << fft_grids[i].limits(1).first << " "
                                                               << fft_grids[i].limits(1).second << "   "
                                                               << fft_grids[i].limits(2).first << " "
                                                               << fft_grids[i].limits(2).second << std::endl
               << "  number of G-vectors within the cutoff : " << gvecs[i]->num_gvec() << std::endl
               << "  local number of G-vectors             : " << gvecs[i]->count() << std::endl
               << "  number of G-shells                    : " << gvecs[i]->num_shells() << std::endl
               << std::endl;
        }
        os << "number of local G-vector blocks: " << split_gvec_local().num_ranks() << std::endl;
        os << std::endl;
    }
    {
        rte::rte_ostream os(out__, "unit cell");
        unit_cell().print_info(os, verbosity());
    }
    {
        rte::rte_ostream os(out__, "sym");
        unit_cell().symmetry().print_info(os, verbosity());
    }
    {
        rte::rte_ostream os(out__, "atom type");
        for (int i = 0; i < unit_cell().num_atom_types(); i++) {
            unit_cell().atom_type(i).print_info(os);
        }
    }
    if (this->cfg().control().print_neighbors()) {
        rte::rte_ostream os(out__, "nghbr");
        unit_cell().print_nearest_neighbours(os);
    }

    {
        rte::rte_ostream os(out__, "info");
        os << "total nuclear charge               : " << unit_cell().total_nuclear_charge() << std::endl
           << "number of core electrons           : " << unit_cell().num_core_electrons() << std::endl
           << "number of valence electrons        : " << unit_cell().num_valence_electrons() << std::endl
           << "total number of electrons          : " << unit_cell().num_electrons() << std::endl
           << "extra charge                       : " << cfg().parameters().extra_charge() << std::endl
           << "total number of aw basis functions : " << unit_cell().mt_aw_basis_size() << std::endl
           << "total number of lo basis functions : " << unit_cell().mt_lo_basis_size() << std::endl
           << "number of first-variational states : " << num_fv_states() << std::endl
           << "number of bands                    : " << num_bands() << std::endl
           << "number of spins                    : " << num_spins() << std::endl
           << "number of magnetic dimensions      : " << num_mag_dims() << std::endl
           << "number of spinor components        : " << num_spinor_comp() << std::endl
           << "number of spinors per band index   : " << num_spinors() << std::endl
           << "lmax_apw                           : " << unit_cell().lmax_apw() << std::endl
           << "lmax_rho                           : " << lmax_rho() << std::endl
           << "lmax_pot                           : " << lmax_pot() << std::endl
           << "lmax_rf                            : " << unit_cell().lmax() << std::endl
           << "smearing type                      : " << cfg().parameters().smearing().c_str() << std::endl
           << "smearing width                     : " << smearing_width() << std::endl
           << "cyclic block size                  : " << cyclic_block_size() << std::endl
           << "|G+k| cutoff                       : " << gk_cutoff() << std::endl
           << "symmetry                           : " << std::boolalpha << use_symmetry() << std::endl
           << "so_correction                      : " << std::boolalpha << so_correction() << std::endl;

        std::string reln[] = {"valence relativity                 : ", "core relativity                    : "};
        relativity_t relt[] = {valence_relativity_, core_relativity_};
        std::map<relativity_t, std::string> const relm = {
            {relativity_t::none, "none"},
            {relativity_t::koelling_harmon, "Koelling-Harmon"},
            {relativity_t::zora, "zora"},
            {relativity_t::iora, "iora"},
            {relativity_t::dirac, "Dirac"}
        };
        for (int i = 0; i < 2; i++) {
            os << reln[i] << relm.at(relt[i]) << std::endl;
        }

        std::string evsn[] = {"standard eigen-value solver        : ", "generalized eigen-value solver     : "};
        ev_solver_t evst[] = {std_evp_solver().type(), gen_evp_solver().type()};
        std::map<ev_solver_t, std::string> const evsm = {
            {ev_solver_t::lapack, "LAPACK"},
            {ev_solver_t::scalapack, "ScaLAPACK"},
            {ev_solver_t::elpa, "ELPA"},
            {ev_solver_t::magma, "MAGMA"},
            {ev_solver_t::magma_gpu, "MAGMA with GPU pointers"},
            {ev_solver_t::cusolver, "cuSOLVER"}
        };
        for (int i = 0; i < 2; i++) {
            os << evsn[i] << evsm.at(evst[i]) << std::endl;
        }
        os << "processing unit                    : ";
        switch (processing_unit()) {
            case sddk::device_t::CPU: {
                os << "CPU" << std::endl;
                break;
            }
            case sddk::device_t::GPU: {
                os << "GPU" << std::endl;
                os << "number of devices                  : " << acc::num_devices() << std::endl;
                acc::print_device_info(0, os);
                break;
            }
        }
        os << std::endl
           << "iterative solver                   : " << cfg().iterative_solver().type() << std::endl
           << "number of steps                    : " << cfg().iterative_solver().num_steps() << std::endl
           << "subspace size                      : " << cfg().iterative_solver().subspace_size() << std::endl
           << "early restart ratio                : " << cfg().iterative_solver().early_restart() << std::endl
           << "precision_wf                       : " << cfg().parameters().precision_wf() << std::endl
           << "precision_hs                       : " << cfg().parameters().precision_hs() << std::endl
           << "mixer                              : " << cfg().mixer().type() << std::endl
           << "mixing beta                        : " << cfg().mixer().beta() << std::endl
           << "max_history                        : " << cfg().mixer().max_history() << std::endl
           << "use_hartree                        : " << std::boolalpha << cfg().mixer().use_hartree() << std::endl
           << std::endl
           << "spglib version: " << spg_get_major_version() << "." << spg_get_minor_version() << "."
           << spg_get_micro_version() << std::endl;
    }
    {
        rte::rte_ostream os(out__, "info");
        unsigned int vmajor, vminor, vmicro;
        H5get_libversion(&vmajor, &vminor, &vmicro);
        os << "HDF5 version: " << vmajor << "." << vminor << "." << vmicro << std::endl;
    }
    {
        rte::rte_ostream os(out__, "info");
        int vmajor, vminor, vmicro;
        xc_version(&vmajor, &vminor, &vmicro);
        os << "Libxc version: " << vmajor << "." << vminor << "." << vmicro << std::endl;
    }

    {
        rte::rte_ostream os(out__, "info");
        int i{1};
        os << std::endl << "XC functionals" << std::endl
           << utils::hbar(14, '=') << std::endl;
        for (auto& xc_label : xc_functionals()) {
            XC_functional xc(spfft<double>(), unit_cell().lattice_vectors(), xc_label, num_spins());
#if defined(SIRIUS_USE_VDWXC)
            if (xc.is_vdw()) {
                os << "Van der Walls functional" << std::endl
                   << xc.refs() << std::endl;
                continue;
            }
#endif
            os << i << ") " << xc_label << " : " << xc.name() << std::endl
               << xc.refs() << std::endl;
            i++;
        }
    }

    if (!full_potential()) {
        rte::rte_ostream os(out__, "info");
        os << std::endl
           << "memory consumption" << std::endl
           << utils::hbar(18, '=') << std::endl;
        /* volume of the Brillouin zone */
        double v0 = std::pow(twopi, 3) / unit_cell().omega();
        /* volume of the cutoff sphere for wave-functions */
        double v1 = fourpi * std::pow(gk_cutoff(), 3) / 3;
        /* volume of the cutoff sphere for density and potential */
        double v2 = fourpi * std::pow(pw_cutoff(), 3) / 3;
        /* volume of the cutoff sphere for coarse FFT grid */
        double v3 = fourpi * std::pow(2 * gk_cutoff(), 3) / 3;
        /* approximate number of G+k vectors */
        auto ngk = static_cast<size_t>(v1 / v0);
        if (gamma_point()) {
            ngk /= 2;
        }
        /* approximate number of G vectors */
        auto ng = static_cast<size_t>(v2 / v0);
        if (cfg().control().reduce_gvec()) {
            ng /= 2;
        }
        /* approximate number of coarse G vectors */
        auto ngc = static_cast<size_t>(v3 / v0);
        if (cfg().control().reduce_gvec()) {
            ngc /= 2;
        }
        os << "approximate number of G+k vectors        : " << ngk << std::endl
           << "approximate number of G vectors          : " << ng << std::endl
           << "approximate number of coarse G vectors   : " << ngc << std::endl;
        size_t wf_size = ngk * num_bands() * num_spins() * 16;
        os << "approximate size of wave-functions for each k-point: " << static_cast<int>(wf_size >> 20) << " Mb,  "
           << static_cast<int>((wf_size / comm_band().size()) >> 20) << " Mb/rank" << std::endl;

        /* number of simultaneously treated spin components */
        int num_sc = (num_mag_dims() == 3) ? 2 : 1;
        /* number of auxiliary basis functions */
        int num_phi = cfg().iterative_solver().subspace_size() * num_bands();
        /* memory consumption for Davidson:
           - wave-functions psi (num_bands x num_spin)
           - Hpsi and Spsi (num_bands * num_sc)
           - auxiliary basis phi (num_bands x num_sc) and also Hphi and Sphi of the same size
           - residuals (num_bands * num_sc)
           - beta-projectors (estimated as num_bands)

           Each wave-function is of size ngk

           TODO: add estimation of subspace matrix size (H_{ij} and S_{ij})
        */
        size_t tot_size = (num_bands() * num_spins() + 2 * num_bands() * num_sc + 3 * num_phi * num_sc +
                           num_bands() * num_sc + num_bands()) *
                          ngk * sizeof(double_complex);
        os << "approximate memory consumption of Davidson solver: "
           << static_cast<int>((tot_size / comm_band().size()) >> 20) << " Mb/rank" << std::endl;

        if (unit_cell().augment()) {
            /* approximate size of local fraction of G vectors */
            size_t ngloc = std::max(static_cast<size_t>(1), ng / comm().size());
            /* upper limit of packed {xi,xi'} bete-projectors index */
            int nb = unit_cell().max_mt_basis_size() * (unit_cell().max_mt_basis_size() + 1) / 2;
            /* size of augmentation operator;
               factor 2 is needed for the estimation of GPU memory, as augmentation operator for two atom types
               will be stored on GPU and computation will be overlapped with transfer of the  next augmentation
               operator */
            // TODO: optimize generated_rho_aug() for less memory consumption
            size_t size_aug = nb * ngloc * sizeof(double_complex);
            if (unit_cell().num_atom_types() > 1) {
                size_aug *= 2;
            }

            /* and two more arrays will be allocated in generate_rho_aug() with 1Gb maximum size each */
            size_t size1 = nb * ngloc * sizeof(double_complex);
            size1        = std::min(size1, static_cast<size_t>(1 << 30));

            int max_atoms{0};
            for (int iat = 0; iat < unit_cell().num_atom_types(); iat++) {
                max_atoms = std::max(max_atoms, unit_cell().atom_type(iat).num_atoms());
            }
            size_t size2 = max_atoms * ngloc * sizeof(double_complex);
            size2        = std::min(size2, static_cast<size_t>(1 << 30));

            size_aug += (size1 + size2);
            os << "approximate memory consumption of charge density augmentation: "
               <<  static_cast<int>(size_aug >> 20) << " Mb/rank" << std::endl;
        }
        /* FFT buffers of fine and coarse meshes */
        size_t size_fft = spfft<double>().local_slice_size() + spfft_coarse<double>().local_slice_size();
        size_fft *= sizeof(double);
        if (!gamma_point()) {
            size_fft *= 2;
        }
        os << "approximate memory consumption of FFT transforms: "
           << static_cast<int>(size_fft >> 20) << " Mb/rank" << std::endl;
    }
}

/** The update of the lattice vectors or atomic positions has an impact on many quantities which have to be
    recomputed in the correct order. First, the unit cell is updated and the new reciprocal lattice vectors
    are obtained. Then the G-vectors are computed (if this is the first call to update()) or recomputed with a
    new reciprocal lattice, but without rebuilding a new list. On the first call the spfft objects are created
    after G-vectors. */
void
Simulation_context::update()
{
    PROFILE("sirius::Simulation_context::update");

    /* update unit cell (reciprocal lattice, etc.) */
    unit_cell().update();

    /* get new reciprocal vector */
    auto rlv = unit_cell().reciprocal_lattice_vectors();

    auto spfft_pu = this->processing_unit() == sddk::device_t::CPU ? SPFFT_PU_HOST : SPFFT_PU_GPU;

    /* create a list of G-vectors for corase FFT grid; this is done only once,
       the next time only reciprocal lattice of the G-vectors is updated */
    if (!gvec_coarse_) {
        /* create list of coarse G-vectors */
        gvec_coarse_ = std::make_unique<sddk::Gvec>(rlv, 2 * gk_cutoff(), comm(), cfg().control().reduce_gvec());
        /* create FFT friendly partiton */
        gvec_coarse_fft_ = std::make_shared<sddk::Gvec_fft>(*gvec_coarse_, comm_fft_coarse(),
                comm_ortho_fft_coarse());

        auto spl_z = split_fft_z(fft_coarse_grid_[2], comm_fft_coarse());

        /* create spfft buffer for coarse transform */
        spfft_grid_coarse_ = std::unique_ptr<spfft::Grid>(new spfft::Grid(
            fft_coarse_grid_[0], fft_coarse_grid_[1], fft_coarse_grid_[2], gvec_coarse_fft_->zcol_count_fft(),
            spl_z.local_size(), spfft_pu, -1, comm_fft_coarse().mpi_comm(), SPFFT_EXCH_DEFAULT));
#ifdef USE_FP32
        spfft_grid_coarse_float_ = std::unique_ptr<spfft::GridFloat>(new spfft::GridFloat(
            fft_coarse_grid_[0], fft_coarse_grid_[1], fft_coarse_grid_[2], gvec_coarse_fft_->zcol_count_fft(),
            spl_z.local_size(), spfft_pu, -1, comm_fft_coarse().mpi_comm(), SPFFT_EXCH_DEFAULT));
#endif
        /* create spfft transformations */
        const auto fft_type_coarse = gvec_coarse().reduced() ? SPFFT_TRANS_R2C : SPFFT_TRANS_C2C;

        auto const& gv = gvec_coarse_fft_->gvec_array();

        /* create actual transform object */
        spfft_transform_coarse_.reset(new spfft::Transform(spfft_grid_coarse_->create_transform(
            spfft_pu, fft_type_coarse, fft_coarse_grid_[0], fft_coarse_grid_[1], fft_coarse_grid_[2],
            spl_z.local_size(), gvec_coarse_fft_->gvec_count_fft(), SPFFT_INDEX_TRIPLETS,
            gv.at(sddk::memory_t::host))));
#ifdef USE_FP32
        spfft_transform_coarse_float_.reset(new spfft::TransformFloat(spfft_grid_coarse_float_->create_transform(
            spfft_pu, fft_type_coarse, fft_coarse_grid_[0], fft_coarse_grid_[1], fft_coarse_grid_[2],
            spl_z.local_size(), gvec_coarse_fft_->gvec_count_fft(), SPFFT_INDEX_TRIPLETS,
            gv.at(sddk::memory_t::host))));
#endif
    } else {
        gvec_coarse_->lattice_vectors(rlv);
    }

    /* create a list of G-vectors for dense FFT grid; G-vectors are divided between all available MPI ranks.*/
    if (!gvec_) {
        gvec_     = std::make_shared<sddk::Gvec>(pw_cutoff(), *gvec_coarse_);
        gvec_fft_ = std::make_shared<sddk::Gvec_fft>(*gvec_, comm_fft(), comm_ortho_fft());

        auto spl_z = split_fft_z(fft_grid_[2], comm_fft());

        /* create spfft buffer for fine-grained transform */
        spfft_grid_ = std::unique_ptr<spfft::Grid>(
            new spfft::Grid(fft_grid_[0], fft_grid_[1], fft_grid_[2],
                            gvec_fft_->zcol_count_fft(), spl_z.local_size(), spfft_pu, -1,
                            comm_fft().mpi_comm(), SPFFT_EXCH_DEFAULT));
#if defined(USE_FP32)
        spfft_grid_float_ = std::unique_ptr<spfft::GridFloat>(
            new spfft::GridFloat(fft_grid_[0], fft_grid_[1], fft_grid_[2], gvec_fft_->zcol_count_fft(),
                                 spl_z.local_size(), spfft_pu, -1, comm_fft().mpi_comm(), SPFFT_EXCH_DEFAULT));
#endif
        const auto fft_type = gvec().reduced() ? SPFFT_TRANS_R2C : SPFFT_TRANS_C2C;

        auto const& gv = gvec_fft_->gvec_array();

        spfft_transform_.reset(new spfft::Transform(spfft_grid_->create_transform(
            spfft_pu, fft_type, fft_grid_[0], fft_grid_[1], fft_grid_[2],
            spl_z.local_size(), gvec_fft_->gvec_count_fft(), SPFFT_INDEX_TRIPLETS, gv.at(sddk::memory_t::host))));
#if defined(USE_FP32)
        spfft_transform_float_.reset(new spfft::TransformFloat(spfft_grid_float_->create_transform(
            spfft_pu, fft_type, fft_grid_[0], fft_grid_[1], fft_grid_[2], spl_z.local_size(),
            gvec_fft_->gvec_count_fft(), SPFFT_INDEX_TRIPLETS, gv.at(sddk::memory_t::host))));
#endif

        /* copy G-vectors to GPU; this is done once because Miller indices of G-vectors
           do not change during the execution */
        switch (this->processing_unit()) {
            case sddk::device_t::CPU: {
                break;
            }
            case sddk::device_t::GPU: {
                gvec_coord_ = sddk::mdarray<int, 2>(gvec().count(), 3, sddk::memory_t::host, "gvec_coord_");
                #pragma omp parallel for schedule(static)
                for (int igloc = 0; igloc < gvec().count(); igloc++) {
                    auto G = gvec().gvec<sddk::index_domain_t::local>(igloc);
                    for (int x : {0, 1, 2}) {
                        gvec_coord_(igloc, x) = G[x];
                    }
                }
                gvec_coord_.allocate(sddk::memory_t::device).copy_to(sddk::memory_t::device);
                break;
            }
        }
    } else {
        gvec_->lattice_vectors(rlv);
    }

    /* After each update of the lattice vectors we might get a different set of G-vector shells.
     * Always update the mapping between the canonical FFT distribution and "local G-shells"
     * distribution which is used in symmetriezation of lattice periodic functions. */
    remap_gvec_ = std::make_unique<sddk::Gvec_shells>(gvec());

    /* check symmetry of G-vectors */
    if (unit_cell().num_atoms() != 0 && use_symmetry() && cfg().control().verification() >= 1) {
        check_gvec(gvec(), unit_cell().symmetry());
        if (!full_potential()) {
            check_gvec(gvec_coarse(), unit_cell().symmetry());
        }
        check_gvec(*remap_gvec_, unit_cell().symmetry());
    }

    /* check if FFT grid is OK; this check is especially needed if the grid is set as external parameter */
    if (cfg().control().verification() >= 0) {
        #pragma omp parallel for
        for (int igloc = 0; igloc < gvec().count(); igloc++) {
            int ig = gvec().offset() + igloc;

            auto gv = gvec().gvec<sddk::index_domain_t::local>(igloc);
            /* check limits */
            for (int x : {0, 1, 2}) {
                auto limits = fft_grid().limits(x);
                /* check boundaries */
                if (gv[x] < limits.first || gv[x] > limits.second) {
                    std::stringstream s;
                    s << "G-vector is outside of grid limits\n"
                      << "  G: " << gv << ", length: " << gvec().gvec_cart<sddk::index_domain_t::global>(ig).length() << "\n"
                      << "  FFT grid limits: " << fft_grid().limits(0).first << " " << fft_grid().limits(0).second
                      << " " << fft_grid().limits(1).first << " " << fft_grid().limits(1).second << " "
                      << fft_grid().limits(2).first << " " << fft_grid().limits(2).second << "\n"
                      << "  FFT grid is not compatible with G-vector cutoff (" << this->pw_cutoff() << ")";
                    RTE_THROW(s);
                }
            }
        }
    }

    if (unit_cell().num_atoms()) {
        init_atoms_to_grid_idx(cfg().control().rmt_max());
    }

    std::pair<int, int> limits(0, 0);
    for (int x : {0, 1, 2}) {
        limits.first  = std::min(limits.first, fft_grid().limits(x).first);
        limits.second = std::max(limits.second, fft_grid().limits(x).second);
    }

    /* recompute phase factors for atoms */
    phase_factors_ = sddk::mdarray<double_complex, 3>(3, limits, unit_cell().num_atoms(), sddk::memory_t::host, "phase_factors_");
    #pragma omp parallel for
    for (int i = limits.first; i <= limits.second; i++) {
        for (int ia = 0; ia < unit_cell().num_atoms(); ia++) {
            auto pos = unit_cell().atom(ia).position();
            for (int x : {0, 1, 2}) {
                phase_factors_(x, i, ia) = std::exp(double_complex(0.0, twopi * (i * pos[x])));
            }
        }
    }

    /* recompute phase factors for atom types */
    phase_factors_t_ = sddk::mdarray<double_complex, 2>(gvec().count(), unit_cell().num_atom_types());
    #pragma omp parallel for schedule(static)
    for (int igloc = 0; igloc < gvec().count(); igloc++) {
        /* global index of G-vector */
        int ig = gvec().offset() + igloc;
        for (int iat = 0; iat < unit_cell().num_atom_types(); iat++) {
            double_complex z(0, 0);
            for (int ia = 0; ia < unit_cell().atom_type(iat).num_atoms(); ia++) {
                z += gvec_phase_factor(ig, unit_cell().atom_type(iat).atom_id(ia));
            }
            phase_factors_t_(igloc, iat) = z;
        }
    }

    if (use_symmetry()) {
        sym_phase_factors_ = sddk::mdarray<double_complex, 3>(3, limits, unit_cell().symmetry().size());

        #pragma omp parallel for
        for (int i = limits.first; i <= limits.second; i++) {
            for (int isym = 0; isym < unit_cell().symmetry().size(); isym++) {
                auto t = unit_cell().symmetry()[isym].spg_op.t;
                for (int x : {0, 1, 2}) {
                    sym_phase_factors_(x, i, isym) = std::exp(double_complex(0.0, twopi * (i * t[x])));
                }
            }
        }
    }

    /* precompute some G-vector related arrays */
    gvec_tp_ = sddk::mdarray<double, 2>(gvec().count(), 2, sddk::memory_t::host, "gvec_tp_");
    #pragma omp parallel for schedule(static)
    for (int igloc = 0; igloc < gvec().count(); igloc++) {
        auto rtp           = SHT::spherical_coordinates(gvec().gvec_cart<sddk::index_domain_t::local>(igloc));
        gvec_tp_(igloc, 0) = rtp[1];
        gvec_tp_(igloc, 1) = rtp[2];
    }

    switch (this->processing_unit()) {
        case sddk::device_t::CPU: {
            break;
        }
        case sddk::device_t::GPU: {
            gvec_tp_.allocate(sddk::memory_t::device).copy_to(sddk::memory_t::device);
            break;
        }
    }

    /* create or update radial integrals */
    if (!full_potential()) {
        /* find the new maximum length of G-vectors */
        double new_pw_cutoff{this->pw_cutoff()};
        for (int igloc = 0; igloc < gvec().count(); igloc++) {
            new_pw_cutoff = std::max(new_pw_cutoff, gvec().gvec_len<sddk::index_domain_t::local>(igloc));
        }
        gvec().comm().allreduce<double, sddk::mpi_op_t::max>(&new_pw_cutoff, 1);
        /* estimate new G+k-vectors cutoff */
        double new_gk_cutoff = this->gk_cutoff();
        if (new_pw_cutoff > this->pw_cutoff()) {
            new_gk_cutoff += (new_pw_cutoff - this->pw_cutoff());
        }

        /* radial integrals with pw_cutoff */
        if (!aug_ri_ || aug_ri_->qmax() < new_pw_cutoff) {
            aug_ri_ = std::unique_ptr<Radial_integrals_aug<false>>(new Radial_integrals_aug<false>(
                unit_cell(), new_pw_cutoff, cfg().settings().nprii_aug(), aug_ri_callback_));
        }

        if (!aug_ri_djl_ || aug_ri_djl_->qmax() < new_pw_cutoff) {
            aug_ri_djl_ = std::unique_ptr<Radial_integrals_aug<true>>(new Radial_integrals_aug<true>(
                unit_cell(), new_pw_cutoff, cfg().settings().nprii_aug(), aug_ri_djl_callback_));
        }

        if (!ps_core_ri_ || ps_core_ri_->qmax() < new_pw_cutoff) {
            ps_core_ri_ =
                std::unique_ptr<Radial_integrals_rho_core_pseudo<false>>(new Radial_integrals_rho_core_pseudo<false>(
                    unit_cell(), new_pw_cutoff, cfg().settings().nprii_rho_core(), rhoc_ri_callback_));
        }

        if (!ps_core_ri_djl_ || ps_core_ri_djl_->qmax() < new_pw_cutoff) {
            ps_core_ri_djl_ =
                std::unique_ptr<Radial_integrals_rho_core_pseudo<true>>(new Radial_integrals_rho_core_pseudo<true>(
                    unit_cell(), new_pw_cutoff, cfg().settings().nprii_rho_core(), rhoc_ri_djl_callback_));
        }

        if (!ps_rho_ri_ || ps_rho_ri_->qmax() < new_pw_cutoff) {
            ps_rho_ri_ = std::unique_ptr<Radial_integrals_rho_pseudo>(
                new Radial_integrals_rho_pseudo(unit_cell(), new_pw_cutoff, 20, ps_rho_ri_callback_));
        }

        if (!vloc_ri_ || vloc_ri_->qmax() < new_pw_cutoff) {
            vloc_ri_ = std::unique_ptr<Radial_integrals_vloc<false>>(new Radial_integrals_vloc<false>(
                unit_cell(), new_pw_cutoff, cfg().settings().nprii_vloc(), vloc_ri_callback_));
        }

        if (!vloc_ri_djl_ || vloc_ri_djl_->qmax() < new_pw_cutoff) {
            vloc_ri_djl_ = std::unique_ptr<Radial_integrals_vloc<true>>(new Radial_integrals_vloc<true>(
                unit_cell(), new_pw_cutoff, cfg().settings().nprii_vloc(), vloc_ri_djl_callback_));
        }

        /* radial integrals with pw_cutoff */
        if (!beta_ri_ || beta_ri_->qmax() < new_gk_cutoff) {
            beta_ri_ = std::unique_ptr<Radial_integrals_beta<false>>(new Radial_integrals_beta<false>(
                unit_cell(), new_gk_cutoff, cfg().settings().nprii_beta(), beta_ri_callback_));
        }

        if (!beta_ri_djl_ || beta_ri_djl_->qmax() < new_gk_cutoff) {
            beta_ri_djl_ = std::unique_ptr<Radial_integrals_beta<true>>(new Radial_integrals_beta<true>(
                unit_cell(), new_gk_cutoff, cfg().settings().nprii_beta(), beta_ri_djl_callback_));
        }

        auto idxr_wf = [&](int iat) -> sirius::experimental::radial_functions_index const& {
            return unit_cell().atom_type(iat).indexr_wfs();
        };

        auto ps_wf = [&](int iat, int i) -> Spline<double> const& {
            return unit_cell().atom_type(iat).ps_atomic_wf(i).f;
        };

        if (!ps_atomic_wf_ri_ || ps_atomic_wf_ri_->qmax() < new_gk_cutoff) {
            ps_atomic_wf_ri_ = std::unique_ptr<Radial_integrals_atomic_wf<false>>(new Radial_integrals_atomic_wf<false>(
                unit_cell(), new_gk_cutoff, 20, idxr_wf, ps_wf, ps_atomic_wf_ri_callback_));
        }

        if (!ps_atomic_wf_ri_djl_ || ps_atomic_wf_ri_djl_->qmax() < new_gk_cutoff) {
            ps_atomic_wf_ri_djl_ = std::unique_ptr<Radial_integrals_atomic_wf<true>>(
                new Radial_integrals_atomic_wf<true>(unit_cell(), new_gk_cutoff, 20, idxr_wf, ps_wf, ps_atomic_wf_ri_djl_callback_));
        }

        /* update augmentation operator */
        sddk::memory_pool* mp{nullptr};
        sddk::memory_pool* mpd{nullptr};
        switch (this->processing_unit()) {
            case sddk::device_t::CPU: {
                mp = &get_memory_pool(sddk::memory_t::host);
                break;
            }
            case sddk::device_t::GPU: {
                mp  = &get_memory_pool(sddk::memory_t::host_pinned);
                mpd = &get_memory_pool(sddk::memory_t::device);
                break;
            }
        }
        for (int iat = 0; iat < unit_cell().num_atom_types(); iat++) {
            if (unit_cell().atom_type(iat).augment() && unit_cell().atom_type(iat).num_atoms() > 0) {
                augmentation_op_[iat] = std::unique_ptr<Augmentation_operator>(
                    new Augmentation_operator(unit_cell().atom_type(iat), gvec()));
                augmentation_op_[iat]->generate_pw_coeffs(aug_ri(), gvec_tp_, *mp, mpd);
            } else {
                augmentation_op_[iat] = nullptr;
            }
        }
    }

    if (full_potential()) { // TODO: add corresponging radial integarls of Theta
        init_step_function();
    }

    auto save_config = env::save_config();
    if (save_config.size() && this->comm().rank() == 0) {
        std::string name;
        if (save_config == "all") {
            static int count{0};
            std::stringstream s;
            s << "sirius" << std::setfill('0') << std::setw(6) << count << ".json";
            name = s.str();
            count++;
        } else {
            name = save_config;
        }
        std::ofstream fi(name, std::ofstream::out | std::ofstream::trunc);
        auto conf_dict = this->serialize();
        fi << conf_dict.dump(4);
    }
}

void
Simulation_context::create_storage_file() const
{
    if (comm_.rank() == 0) {
        /* create new hdf5 file */
        sddk::HDF5_tree fout(storage_file_name, sddk::hdf5_access_t::truncate);
        fout.create_node("parameters");
        fout.create_node("effective_potential");
        fout.create_node("effective_magnetic_field");
        fout.create_node("density");
        fout.create_node("magnetization");

        for (int j = 0; j < num_mag_dims(); j++) {
            fout["magnetization"].create_node(j);
            fout["effective_magnetic_field"].create_node(j);
        }

        fout["parameters"].write("num_spins", num_spins());
        fout["parameters"].write("num_mag_dims", num_mag_dims());
        fout["parameters"].write("num_bands", num_bands());

        sddk::mdarray<int, 2> gv(3, gvec().num_gvec());
        for (int ig = 0; ig < gvec().num_gvec(); ig++) {
            auto G = gvec().gvec<sddk::index_domain_t::global>(ig);
            for (int x : {0, 1, 2}) {
                gv(x, ig) = G[x];
            }
        }
        fout["parameters"].write("num_gvec", gvec().num_gvec());
        fout["parameters"].write("gvec", gv);

        fout.create_node("unit_cell");
        fout["unit_cell"].create_node("atoms");
        for (int j = 0; j < unit_cell().num_atoms(); j++) {
            fout["unit_cell"]["atoms"].create_node(j);
            fout["unit_cell"]["atoms"][j].write("mt_basis_size", unit_cell().atom(j).mt_basis_size());
        }
    }
    comm_.barrier();
}

void
Simulation_context::generate_phase_factors(int iat__, sddk::mdarray<double_complex, 2>& phase_factors__) const
{
    PROFILE("sirius::Simulation_context::generate_phase_factors");
    int na = unit_cell().atom_type(iat__).num_atoms();
    switch (processing_unit_) {
        case sddk::device_t::CPU: {
            #pragma omp parallel for
            for (int igloc = 0; igloc < gvec().count(); igloc++) {
                int ig = gvec().offset() + igloc;
                for (int i = 0; i < na; i++) {
                    int ia                    = unit_cell().atom_type(iat__).atom_id(i);
                    phase_factors__(igloc, i) = gvec_phase_factor(ig, ia);
                }
            }
            break;
        }
        case sddk::device_t::GPU: {
#if defined(SIRIUS_GPU)
            generate_phase_factors_gpu(gvec().count(), na, gvec_coord().at(sddk::memory_t::device),
                                       unit_cell().atom_coord(iat__).at(sddk::memory_t::device),
                                       phase_factors__.at(sddk::memory_t::device));
#endif
            break;
        }
    }
}

void
Simulation_context::init_atoms_to_grid_idx(double R__)
{
    PROFILE("sirius::Simulation_context::init_atoms_to_grid_idx");

    auto Rmt = unit_cell().find_mt_radii(1, true);

    double R{0};
    for (auto e : Rmt) {
        R = std::max(e, R);
    }

    //double R = R__;

    atoms_to_grid_idx_.resize(unit_cell().num_atoms());

    vector3d<double> delta(1.0 / spfft<double>().dim_x(), 1.0 / spfft<double>().dim_y(), 1.0 / spfft<double>().dim_z());

    int z_off = spfft<double>().local_z_offset();
    vector3d<int> grid_beg(0, 0, z_off);
    vector3d<int> grid_end(spfft<double>().dim_x(), spfft<double>().dim_y(), z_off + spfft<double>().local_z_length());
    std::vector<vector3d<double>> verts_cart{{-R, -R, -R}, {R, -R, -R}, {-R, R, -R}, {R, R, -R},
                                             {-R, -R, R},  {R, -R, R},  {-R, R, R},  {R, R, R}};

    auto bounds_box = [&](vector3d<double> pos) {
        std::vector<vector3d<double>> verts;

        /* pos is a position of atom */
        for (auto v : verts_cart) {
            verts.push_back(pos + unit_cell().get_fractional_coordinates(v));
        }

        std::pair<vector3d<int>, vector3d<int>> bounds_ind;

        for (int x : {0, 1, 2}) {
            std::sort(verts.begin(), verts.end(),
                      [x](vector3d<double>& a, vector3d<double>& b) { return a[x] < b[x]; });
            bounds_ind.first[x]  = std::max(static_cast<int>(verts[0][x] / delta[x]) - 1, grid_beg[x]);
            bounds_ind.second[x] = std::min(static_cast<int>(verts[5][x] / delta[x]) + 1, grid_end[x]);
        }

        return bounds_ind;
    };

    #pragma omp parallel for
    for (int ia = 0; ia < unit_cell().num_atoms(); ia++) {

        std::vector<std::pair<int, double>> atom_to_ind_map;

        for (int t0 = -1; t0 <= 1; t0++) {
            for (int t1 = -1; t1 <= 1; t1++) {
                for (int t2 = -1; t2 <= 1; t2++) {
                    auto pos = unit_cell().atom(ia).position() + vector3d<double>(t0, t1, t2);

                    /* find the small box around this atom */
                    auto box = bounds_box(pos);

                    for (int j0 = box.first[0]; j0 < box.second[0]; j0++) {
                        for (int j1 = box.first[1]; j1 < box.second[1]; j1++) {
                            for (int j2 = box.first[2]; j2 < box.second[2]; j2++) {
                                auto v = pos - vector3d<double>(delta[0] * j0, delta[1] * j1, delta[2] * j2);
                                auto r = unit_cell().get_cartesian_coordinates(v).length();
                                if (r < Rmt[unit_cell().atom(ia).type_id()]) {
                                    auto ir = fft_grid_.index_by_coord(j0, j1, j2 - z_off);
                                    atom_to_ind_map.push_back({ir, r});
                                }
                            }
                        }
                    }
                }
            }
        }

        atoms_to_grid_idx_[ia] = std::move(atom_to_ind_map);
    }
}

void
Simulation_context::init_step_function()
{
    auto v = make_periodic_function<sddk::index_domain_t::global>([&](int iat, double g) {
        auto R = unit_cell().atom_type(iat).mt_radius();
        return unit_step_function_form_factors(R, g);
    });

    theta_    = sddk::mdarray<double, 1>(spfft<double>().local_slice_size());
    theta_pw_ = sddk::mdarray<double_complex, 1>(gvec().num_gvec());

    try {
        for (int ig = 0; ig < gvec().num_gvec(); ig++) {
            theta_pw_[ig] = -v[ig];
        }
        theta_pw_[0] += 1.0;

        std::vector<double_complex> ftmp(gvec_fft().gvec_count_fft());
        this->gvec_fft().scatter_pw_global(&theta_pw_[0], &ftmp[0]);
        spfft<double>().backward(reinterpret_cast<double const*>(ftmp.data()), SPFFT_PU_HOST);
        double* theta_ptr = spfft<double>().local_slice_size() == 0 ? nullptr : &theta_[0];
        spfft_output(spfft<double>(), theta_ptr);
    } catch (...) {
        std::stringstream s;
        s << "fft_grid = " << fft_grid_[0] << " " << fft_grid_[1] << " " << fft_grid_[2] << std::endl
          << "spfft<double>().local_slice_size() = " << spfft<double>().local_slice_size() << std::endl
          << "gvec_fft().gvec_count_fft() = " << gvec_fft().gvec_count_fft();
        RTE_THROW(s);
    }

    double vit{0};
    for (int i = 0; i < spfft<double>().local_slice_size(); i++) {
        vit += theta_[i];
    }
    vit *= (unit_cell().omega() / fft_grid().num_points());
    sddk::Communicator(spfft<double>().communicator()).allreduce(&vit, 1);

    if (std::abs(vit - unit_cell().volume_it()) > 1e-10) {
        std::stringstream s;
        s << "step function gives a wrong volume for IT region" << std::endl
          << "  difference with exact value : " << std::abs(vit - unit_cell().volume_it());
        if (comm().rank() == 0) {
            WARNING(s);
        }
    }
    if (cfg().control().print_checksum()) {
        double_complex z1 = theta_pw_.checksum();
        double d1         = theta_.checksum();
        sddk::Communicator(spfft<double>().communicator()).allreduce(&d1, 1);
        utils::print_checksum("theta", d1, this->out());
        utils::print_checksum("theta_pw", z1, this->out());
    }
}

void
Simulation_context::init_comm()
{
    PROFILE("sirius::Simulation_context::init_comm");

    /* check MPI grid dimensions and set a default grid if needed */
    if (!cfg().control().mpi_grid_dims().size()) {
        mpi_grid_dims({1, 1});
    }
    if (cfg().control().mpi_grid_dims().size() != 2) {
        RTE_THROW("wrong MPI grid");
    }

    int npr = cfg().control().mpi_grid_dims()[0];
    int npc = cfg().control().mpi_grid_dims()[1];
    int npb = npr * npc;
    if (npb <= 0) {
        std::stringstream s;
        s << "wrong mpi grid dimensions : " << npr << " " << npc;
        RTE_THROW(s);
    }
    int npk = comm_.size() / npb;
    if (npk * npb != comm_.size()) {
        std::stringstream s;
        s << "Can't divide " << comm_.size() << " ranks into groups of size " << npb;
        RTE_THROW(s);
    }

    /* create k- and band- communicators */
    if (comm_k_.is_null() && comm_band_.is_null()) {
        comm_band_ = comm_.split(comm_.rank() / npb);
        comm_k_    = comm_.split(comm_.rank() % npb);
    }

    /* setup MPI grid */
    mpi_grid_ = std::make_unique<sddk::MPI_grid>(std::vector<int>({npc, npr}), comm_band_);

    /* here we know the number of ranks for band parallelization */

    /* if we have multiple ranks per node and band parallelization, switch to parallel FFT for coarse mesh */
    if (sddk::num_ranks_per_node() > 1 && comm_band().size() > 1) {
        cfg().control().fft_mode("parallel");
    }

    /* create communicator, orthogonal to comm_fft_coarse */
    comm_ortho_fft_coarse_ = comm().split(comm_fft_coarse().rank());

    /* create communicator, orthogonal to comm_fft_coarse within a band communicator */
    comm_band_ortho_fft_coarse_ = comm_band().split(comm_fft_coarse().rank());
}
} // namespace sirius
