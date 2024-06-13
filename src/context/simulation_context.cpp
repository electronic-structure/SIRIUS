/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file simulation_context.cpp
 *
 *  \brief Implementation of Simulation_context class.
 */

#include <gsl/gsl_sf_bessel.h>
#include "core/profiler.hpp"
#include "core/env/env.hpp"
#include "core/omp.hpp"
#include "core/sirius_version.hpp"
#include "core/ostream_tools.hpp"
#include "simulation_context.hpp"
#include "symmetry/lattice.hpp"
#include "symmetry/crystal_symmetry.hpp"
#include "symmetry/check_gvec.hpp"
#include "potential/xc_functional.hpp"
#include "core/la/linalg_spla.hpp"
#include "lapw/step_function.hpp"

namespace sirius {

template <>
spfft::Transform&
Simulation_context::spfft<double>()
{
    return *spfft_transform_;
}

template <>
spfft::Transform const&
Simulation_context::spfft<double>() const
{
    return *spfft_transform_;
}

template <>
spfft::Transform&
Simulation_context::spfft_coarse<double>()
{
    return *spfft_transform_coarse_;
}

template <>
spfft::Transform const&
Simulation_context::spfft_coarse<double>() const
{
    return *spfft_transform_coarse_;
}

template <>
spfft::Grid&
Simulation_context::spfft_grid_coarse<double>()
{
    return *spfft_grid_coarse_;
}

#if defined(SIRIUS_USE_FP32)
template <>
spfft::TransformFloat&
Simulation_context::spfft<float>()
{
    return *spfft_transform_float_;
}

template <>
spfft::TransformFloat const&
Simulation_context::spfft<float>() const
{
    return *spfft_transform_float_;
}

template <>
spfft::TransformFloat&
Simulation_context::spfft_coarse<float>()
{
    return *spfft_transform_coarse_float_;
}

template <>
spfft::TransformFloat const&
Simulation_context::spfft_coarse<float>() const
{
    return *spfft_transform_coarse_float_;
}

template <>
spfft::GridFloat&
Simulation_context::spfft_grid_coarse<float>()
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
        fft_grid_ = fft::get_min_grid(pw_cutoff(), rlv);
        cfg().settings().fft_grid_size(fft_grid_);
    } else {
        /* else create a grid with user-specified dimensions */
        fft_grid_ = fft::Grid(fft_grid);
    }

    /* create FFT grid for coarse mesh */
    if (cfg().settings().use_coarse_fft_grid()) {
        fft_coarse_grid_ = fft::get_min_grid(2 * gk_cutoff(), rlv);
    } else {
        fft_coarse_grid_ = fft_grid_;
    }
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
        RTE_WARNING(s);
    }
    return lambda;
}

void
Simulation_context::initialize()
{
    PROFILE("sirius::Simulation_context::initialize");

    /* can't initialize twice */
    if (initialized_) {
        RTE_THROW("Simulation parameters are already initialized.");
    }

    auto verb_lvl = env::get_value_ptr<int>("SIRIUS_VERBOSITY");
    if (verb_lvl) {
        this->verbosity(*verb_lvl);
    }

    /* setup the output stream */
    if (this->comm().rank() == 0 && this->verbosity() >= 1) {
        auto out_str = split(cfg().control().output(), ':');
        if (out_str.size() != 2) {
            RTE_THROW("wrong output stream parameter");
        }
        if (out_str[0] == "stdout") {
            output_stream_ = &std::cout;
        } else if (out_str[0] == "file") {
            output_file_stream_ = std::ofstream(out_str[1]);
            output_stream_      = &output_file_stream_;
        } else {
            RTE_THROW("unknown output stream type");
        }
    } else {
        output_stream_ = &null_stream();
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
    if (processing_unit() == device_t::GPU) {
#if !defined(SIRIUS_GPU)
        RTE_THROW("not compiled with GPU support!");
#endif
    }

    /* initialize MPI communicators */
    init_comm();

    switch (processing_unit()) {
        case device_t::CPU: {
            host_memory_t_ = memory_t::host;
            break;
        }
        case device_t::GPU: {
            host_memory_t_ = memory_t::host_pinned;
            break;
        }
    }

    if (processing_unit() == device_t::GPU) {
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
        auto lv = r3::matrix<double>(unit_cell().lattice_vectors());

        auto lat_sym = find_lat_sym(lv, cfg().control().spglib_tolerance());

        #pragma omp parallel for
        for (int i = 0; i < unit_cell().symmetry().size(); i++) {
            auto& spgR = unit_cell().symmetry()[i].spg_op.R;
            bool found{false};
            for (size_t j = 0; j < lat_sym.size(); j++) {
                auto latR = lat_sym[j];
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

    if (full_potential() && (this->gk_cutoff() * this->unit_cell().max_mt_radius() > this->unit_cell().lmax_apw()) &&
        this->comm().rank() == 0 && this->verbosity() >= 0) {
        std::stringstream s;
        s << "G+k cutoff (" << this->gk_cutoff() << ") is too large for a given lmax (" << this->unit_cell().lmax_apw()
          << ") and a maximum MT radius (" << this->unit_cell().max_mt_radius() << ")" << std::endl
          << "suggested minimum value for lmax : " << int(this->gk_cutoff() * this->unit_cell().max_mt_radius()) + 1;
        RTE_WARNING(s);
    }

    if (!full_potential()) {
        lmax_rho(-1);
        lmax_pot(-1);
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
#if defined(SIRIUS_DLAF)
    bool is_dlaf{true};
#else
    bool is_dlaf{false};
#endif

    if (processing_unit() == device_t::CPU || acc::num_devices() == 0) {
        is_cuda  = false;
        is_magma = false;
    }

    int npr = mpi_grid_dims()[0];
    int npc = mpi_grid_dims()[1];

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
                if (is_dlaf) {
                    evsn[i] = "dlaf";
                }
            }
        }
    }

    /* environment variable has a highest priority */
    auto ev_str = env::get_ev_solver();
    if (ev_str.size()) {
        evsn[0] = ev_str;
        evsn[1] = ev_str;
    }

    std_evp_solver_name(evsn[0]);
    gen_evp_solver_name(evsn[1]);

    std_evp_solver_ = la::Eigensolver_factory(std_evp_solver_name());
    gen_evp_solver_ = la::Eigensolver_factory(gen_evp_solver_name());

    auto& std_solver = std_evp_solver();
    auto& gen_solver = gen_evp_solver();

    if (std_solver.is_parallel() != gen_solver.is_parallel()) {
        RTE_THROW("both solvers must be sequential or parallel");
    }

    /* setup BLACS grid */
    if (std_solver.is_parallel()) {
        blacs_grid_ = std::make_unique<la::BLACS_grid>(comm_band(), npr, npc);
    } else {
        blacs_grid_ = std::make_unique<la::BLACS_grid>(mpi::Communicator::self(), 1, 1);
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
        auto tol = std::max(cfg().iterative_solver().min_tolerance(), t);
        cfg().iterative_solver().min_tolerance(tol);
    }

    /* set the smearing */
    smearing(cfg().parameters().smearing());

    /* create auxiliary mpi grid for symmetrization */
    auto make_mpi_grid_mt_sym = [](int na, int np) {
        std::vector<int> result;
        for (int ia = 1; ia <= na; ia++) {
            if (na % ia == 0 && np % ia == 0) {
                result = std::vector<int>({ia, np / ia});
            }
        }
        return result;
    };

    for (int ic = 0; ic < unit_cell().num_atom_symmetry_classes(); ic++) {
        if (this->full_potential() || unit_cell().atom_symmetry_class(ic).atom_type().is_paw()) {
            auto r = make_mpi_grid_mt_sym(unit_cell().atom_symmetry_class(ic).num_atoms(), this->comm().size());
            mpi_grid_mt_sym_.push_back(std::make_unique<mpi::Grid>(r, this->comm()));
        } else {
            mpi_grid_mt_sym_.push_back(nullptr);
        }
    }

    /* create G-vectors on the first call to update() */
    update();

    print_memory_usage(this->out(), FILE_LINE);

    if (verbosity() >= 1 && comm().rank() == 0) {
        print_info(this->out());
    }

    auto print_mpi_layout = env::print_mpi_layout();

    if (verbosity() >= 3 || print_mpi_layout) {
        mpi::pstdout pout(comm());
        if (comm().rank() == 0) {
            pout << "MPI rank placement" << std::endl;
            pout << hbar(136, '-') << std::endl;
            pout << "             |  comm tot, band, k | comm fft, ortho | mpi_grid tot, row, col | blacs tot, row, col"
                 << std::endl;
        }
        pout << std::setw(12) << hostname() << " | " << std::setw(6) << comm().rank() << std::setw(6)
             << comm_band().rank() << std::setw(6) << comm_k().rank() << " | " << std::setw(6)
             << comm_fft_coarse().rank() << std::setw(6) << comm_band_ortho_fft_coarse().rank() << "    |   "
             << std::setw(6) << mpi_grid_->communicator(3).rank() << std::setw(6)
             << mpi_grid_->communicator(1 << 0).rank() << std::setw(6) << mpi_grid_->communicator(1 << 1).rank()
             << "   | " << std::setw(6) << blacs_grid().comm().rank() << std::setw(6) << blacs_grid().comm_row().rank()
             << std::setw(6) << blacs_grid().comm_col().rank() << std::endl;
        rte::ostream(this->out(), "info") << pout.flush(0);
    }

    initialized_ = true;
    cfg().lock();
}

void
Simulation_context::print_info(std::ostream& out__) const
{
    {
        rte::ostream os(out__, "info");
        tm const* ptm = localtime(&start_time_.tv_sec);
        char buf[100];
        strftime(buf, sizeof(buf), "%a, %e %b %Y %H:%M:%S", ptm);

        os << "SIRIUS version : " << major_version() << "." << minor_version() << "." << revision() << std::endl
           << "git hash       : " << git_hash() << std::endl
           << "git branch     : " << git_branchname() << std::endl
           << "build time     : " << build_date() << std::endl
           << "start time     : " << std::string(buf) << std::endl
           << std::endl
           << "number of MPI ranks           : " << this->comm().size() << std::endl;
        if (mpi_grid_) {
            os << "MPI grid                      :";
            for (int i : {0, 1}) {
                os << " " << mpi_grid_->communicator(1 << i).size();
            }
            os << std::endl;
        }
        os << "maximum number of OMP threads : " << omp_get_max_threads() << std::endl
           << "number of MPI ranks per node  : " << mpi::num_ranks_per_node() << std::endl
           << "page size (Kb)                : " << (get_page_size() >> 10) << std::endl
           << "number of pages               : " << get_num_pages() << std::endl
           << "available memory (GB)         : " << (get_total_memory() >> 30) << std::endl;
        os << std::endl;
    }
    {
        rte::ostream os(out__, "fft");
        std::string headers[]            = {"FFT context for density and potential", "FFT context for coarse grid"};
        double cutoffs[]                 = {pw_cutoff(), 2 * gk_cutoff()};
        mpi::Communicator const* comms[] = {&comm_fft(), &comm_fft_coarse()};
        fft::Grid fft_grids[]            = {this->fft_grid_, this->fft_coarse_grid_};
        fft::Gvec const* gvecs[]         = {&gvec(), &gvec_coarse()};

        for (int i = 0; i < 2; i++) {
            os << headers[i] << std::endl
               << hbar(37, '=') << std::endl
               << "  comm size                             : " << comms[i]->size() << std::endl
               << "  plane wave cutoff                     : " << cutoffs[i] << std::endl
               << "  grid size                             : " << fft_grids[i][0] << " " << fft_grids[i][1] << " "
               << fft_grids[i][2] << "   total : " << fft_grids[i].num_points() << std::endl
               << "  grid limits                           : " << fft_grids[i].limits(0).first << " "
               << fft_grids[i].limits(0).second << "   " << fft_grids[i].limits(1).first << " "
               << fft_grids[i].limits(1).second << "   " << fft_grids[i].limits(2).first << " "
               << fft_grids[i].limits(2).second << std::endl
               << "  number of G-vectors within the cutoff : " << gvecs[i]->num_gvec() << std::endl
               << "  local number of G-vectors             : " << gvecs[i]->count() << std::endl
               << "  number of G-shells                    : " << gvecs[i]->num_shells() << std::endl
               << std::endl;
        }
        os << std::endl;
    }
    {
        rte::ostream os(out__, "unit cell");
        unit_cell().print_info(os, verbosity());
    }
    {
        rte::ostream os(out__, "sym");
        unit_cell().symmetry().print_info(os, verbosity());
    }
    {
        rte::ostream os(out__, "atom type");
        for (int i = 0; i < unit_cell().num_atom_types(); i++) {
            unit_cell().atom_type(i).print_info(os);
        }
    }
    if (this->cfg().control().print_neighbors()) {
        rte::ostream os(out__, "nghbr");
        unit_cell().print_nearest_neighbours(os);
    }

    {
        rte::ostream os(out__, "info");
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

        std::string reln[]  = {"valence relativity                 : ", "core relativity                    : "};
        relativity_t relt[] = {valence_relativity_, core_relativity_};
        std::map<relativity_t, std::string> const relm = {{relativity_t::none, "none"},
                                                          {relativity_t::koelling_harmon, "Koelling-Harmon"},
                                                          {relativity_t::zora, "zora"},
                                                          {relativity_t::iora, "iora"},
                                                          {relativity_t::dirac, "Dirac"}};
        for (int i = 0; i < 2; i++) {
            os << reln[i] << relm.at(relt[i]) << std::endl;
        }

        std::string evsn[]     = {"standard eigen-value solver        : ", "generalized eigen-value solver     : "};
        la::ev_solver_t evst[] = {std_evp_solver().type(), gen_evp_solver().type()};
        std::map<la::ev_solver_t, std::string> const evsm = {
                {la::ev_solver_t::lapack, "LAPACK"},    {la::ev_solver_t::scalapack, "ScaLAPACK"},
                {la::ev_solver_t::elpa, "ELPA"},        {la::ev_solver_t::dlaf, "DLA-Future"},
                {la::ev_solver_t::magma, "MAGMA"},      {la::ev_solver_t::magma_gpu, "MAGMA with GPU pointers"},
                {la::ev_solver_t::cusolver, "cuSOLVER"}};
        for (int i = 0; i < 2; i++) {
            os << evsn[i] << evsm.at(evst[i]) << std::endl;
        }
        os << "processing unit                    : ";
        switch (processing_unit()) {
            case device_t::CPU: {
                os << "CPU" << std::endl;
                break;
            }
            case device_t::GPU: {
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
        rte::ostream os(out__, "info");
        unsigned int vmajor, vminor, vmicro;
        H5get_libversion(&vmajor, &vminor, &vmicro);
        os << "HDF5 version: " << vmajor << "." << vminor << "." << vmicro << std::endl;
    }
    {
        rte::ostream os(out__, "info");
        int vmajor, vminor, vmicro;
        xc_version(&vmajor, &vminor, &vmicro);
        os << "Libxc version: " << vmajor << "." << vminor << "." << vmicro << std::endl;
    }
    {
        rte::ostream os(out__, "info");
        int i{1};
        os << std::endl << "XC functionals" << std::endl << hbar(14, '=') << std::endl;
        for (auto& xc_label : xc_functionals()) {
            XC_functional xc(spfft<double>(), unit_cell().lattice_vectors(), xc_label, num_spins());
#if defined(SIRIUS_USE_VDWXC)
            if (xc.is_vdw()) {
                os << "Van der Walls functional" << std::endl << xc.refs() << std::endl;
                continue;
            }
#endif
            os << i << ") " << xc_label << " : " << xc.name() << std::endl << xc.refs() << std::endl;
            i++;
        }
    }

    if (!full_potential()) {
        rte::ostream os(out__, "info");
        os << std::endl << "memory consumption" << std::endl << hbar(18, '=') << std::endl;
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
                          ngk * sizeof(std::complex<double>);
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
            size_t size_aug = nb * ngloc * sizeof(std::complex<double>);
            if (unit_cell().num_atom_types() > 1) {
                size_aug *= 2;
            }

            /* and two more arrays will be allocated in generate_rho_aug() with 1Gb maximum size each */
            size_t size1 = nb * ngloc * sizeof(std::complex<double>);
            size1        = std::min(size1, static_cast<size_t>(1 << 30));

            int max_atoms{0};
            for (int iat = 0; iat < unit_cell().num_atom_types(); iat++) {
                max_atoms = std::max(max_atoms, unit_cell().atom_type(iat).num_atoms());
            }
            size_t size2 = max_atoms * ngloc * sizeof(std::complex<double>);
            size2        = std::min(size2, static_cast<size_t>(1 << 30));

            size_aug += (size1 + size2);
            os << "approximate memory consumption of charge density augmentation: " << static_cast<int>(size_aug >> 20)
               << " Mb/rank" << std::endl;
        }
        /* FFT buffers of fine and coarse meshes */
        size_t size_fft = spfft<double>().local_slice_size() + spfft_coarse<double>().local_slice_size();
        size_fft *= sizeof(double);
        if (!gamma_point()) {
            size_fft *= 2;
        }
        os << "approximate memory consumption of FFT transforms: " << static_cast<int>(size_fft >> 20) << " Mb/rank"
           << std::endl;
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

    /* cache rotation symmetry matrices */
    int lmax  = this->full_potential() ? std::max(this->lmax_pot(), this->lmax_rho()) : 2 * this->unit_cell().lmax();
    int lmmax = sf::lmmax(lmax);
    rotm_.resize(this->unit_cell().symmetry().size());
    /* loop over crystal symmetries */
    #pragma omp parallel for
    for (int i = 0; i < this->unit_cell().symmetry().size(); i++) {
        rotm_[i] = mdarray<double, 2>({lmmax, lmmax});
        /* compute Rlm rotation matrix */
        sht::rotation_matrix(lmax, this->unit_cell().symmetry()[i].spg_op.euler_angles,
                             this->unit_cell().symmetry()[i].spg_op.proper, rotm_[i]);
    }

    /* get new reciprocal vector */
    auto rlv = unit_cell().reciprocal_lattice_vectors();

    auto spfft_pu = this->processing_unit() == device_t::CPU ? SPFFT_PU_HOST : SPFFT_PU_GPU;

    /* create a list of G-vectors for corase FFT grid; this is done only once,
       the next time only reciprocal lattice of the G-vectors is updated */
    if (!gvec_coarse_) {
        /* create list of coarse G-vectors */
        gvec_coarse_ = std::make_unique<fft::Gvec>(rlv, 2 * gk_cutoff(), comm(), cfg().control().reduce_gvec(),
                                                   cfg().control().spglib_tolerance());
        /* create FFT friendly partiton */
        gvec_coarse_fft_ = std::make_shared<fft::Gvec_fft>(*gvec_coarse_, comm_fft_coarse(), comm_ortho_fft_coarse());

        auto spl_z = fft::split_z_dimension(fft_coarse_grid_[2], comm_fft_coarse());

        /* create spfft buffer for coarse transform */
        spfft_grid_coarse_ = std::make_unique<spfft::Grid>(
                fft_coarse_grid_[0], fft_coarse_grid_[1], fft_coarse_grid_[2], gvec_coarse_fft_->zcol_count(),
                spl_z.local_size(), spfft_pu, -1, comm_fft_coarse().native(), SPFFT_EXCH_DEFAULT);
#ifdef SIRIUS_USE_FP32
        spfft_grid_coarse_float_ = std::make_unique<spfft::GridFloat>(
                fft_coarse_grid_[0], fft_coarse_grid_[1], fft_coarse_grid_[2], gvec_coarse_fft_->zcol_count(),
                spl_z.local_size(), spfft_pu, -1, comm_fft_coarse().native(), SPFFT_EXCH_DEFAULT);
#endif
        /* create spfft transformations */
        const auto fft_type_coarse = gvec_coarse().reduced() ? SPFFT_TRANS_R2C : SPFFT_TRANS_C2C;

        auto const& gv = gvec_coarse_fft_->gvec_array();

        /* create actual transform object */
        spfft_transform_coarse_.reset(new spfft::Transform(spfft_grid_coarse_->create_transform(
                spfft_pu, fft_type_coarse, fft_coarse_grid_[0], fft_coarse_grid_[1], fft_coarse_grid_[2],
                spl_z.local_size(), gvec_coarse_fft_->count(), SPFFT_INDEX_TRIPLETS, gv.at(memory_t::host))));
#ifdef SIRIUS_USE_FP32
        spfft_transform_coarse_float_.reset(new spfft::TransformFloat(spfft_grid_coarse_float_->create_transform(
                spfft_pu, fft_type_coarse, fft_coarse_grid_[0], fft_coarse_grid_[1], fft_coarse_grid_[2],
                spl_z.local_size(), gvec_coarse_fft_->count(), SPFFT_INDEX_TRIPLETS, gv.at(memory_t::host))));
#endif
    } else {
        gvec_coarse_->lattice_vectors(rlv);
    }

    /* create a list of G-vectors for dense FFT grid; G-vectors are divided between all available MPI ranks.*/
    if (!gvec_) {
        gvec_     = std::make_shared<fft::Gvec>(pw_cutoff(), *gvec_coarse_);
        gvec_fft_ = std::make_shared<fft::Gvec_fft>(*gvec_, comm_fft(), comm_ortho_fft());

        auto spl_z = fft::split_z_dimension(fft_grid_[2], comm_fft());

        /* create spfft buffer for fine-grained transform */
        spfft_grid_ = std::unique_ptr<spfft::Grid>(
                new spfft::Grid(fft_grid_[0], fft_grid_[1], fft_grid_[2], gvec_fft_->zcol_count(), spl_z.local_size(),
                                spfft_pu, -1, comm_fft().native(), SPFFT_EXCH_DEFAULT));
#if defined(SIRIUS_USE_FP32)
        spfft_grid_float_ = std::unique_ptr<spfft::GridFloat>(
                new spfft::GridFloat(fft_grid_[0], fft_grid_[1], fft_grid_[2], gvec_fft_->zcol_count(),
                                     spl_z.local_size(), spfft_pu, -1, comm_fft().native(), SPFFT_EXCH_DEFAULT));
#endif
        const auto fft_type = gvec().reduced() ? SPFFT_TRANS_R2C : SPFFT_TRANS_C2C;

        auto const& gv = gvec_fft_->gvec_array();

        spfft_transform_.reset(new spfft::Transform(spfft_grid_->create_transform(
                spfft_pu, fft_type, fft_grid_[0], fft_grid_[1], fft_grid_[2], spl_z.local_size(), gvec_fft_->count(),
                SPFFT_INDEX_TRIPLETS, gv.at(memory_t::host))));
#if defined(SIRIUS_USE_FP32)
        spfft_transform_float_.reset(new spfft::TransformFloat(spfft_grid_float_->create_transform(
                spfft_pu, fft_type, fft_grid_[0], fft_grid_[1], fft_grid_[2], spl_z.local_size(), gvec_fft_->count(),
                SPFFT_INDEX_TRIPLETS, gv.at(memory_t::host))));
#endif

        /* copy G-vectors to GPU; this is done once because Miller indices of G-vectors
           do not change during the execution */
        switch (this->processing_unit()) {
            case device_t::CPU: {
                break;
            }
            case device_t::GPU: {
                gvec_coord_ = mdarray<int, 2>({gvec().count(), 3}, mdarray_label("gvec_coord_"));
                #pragma omp parallel for schedule(static)
                for (int igloc = 0; igloc < gvec().count(); igloc++) {
                    auto G = gvec().gvec(gvec_index_t::local(igloc));
                    for (int x : {0, 1, 2}) {
                        gvec_coord_(igloc, x) = G[x];
                    }
                }
                gvec_coord_.allocate(memory_t::device).copy_to(memory_t::device);
                break;
            }
        }
    } else {
        gvec_->lattice_vectors(rlv);
    }

    /* After each update of the lattice vectors we might get a different set of G-vector shells.
     * Always update the mapping between the canonical FFT distribution and "local G-shells"
     * distribution which is used in symmetriezation of lattice periodic functions. */
    remap_gvec_ = std::make_unique<fft::Gvec_shells>(gvec());

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

            auto gv = gvec().gvec(gvec_index_t::local(igloc));
            /* check limits */
            for (int x : {0, 1, 2}) {
                auto limits = fft_grid().limits(x);
                /* check boundaries */
                if (gv[x] < limits.first || gv[x] > limits.second) {
                    std::stringstream s;
                    s << "G-vector is outside of grid limits\n"
                      << "  G: " << gv << ", length: " << gvec().gvec_cart(gvec_index_t::global(ig)).length() << "\n"
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
    phase_factors_ =
            mdarray<std::complex<double>, 3>({3, index_range(limits.first, limits.second + 1), unit_cell().num_atoms()},
                                             mdarray_label("phase_factors_"));
    #pragma omp parallel for
    for (int i = limits.first; i <= limits.second; i++) {
        for (int ia = 0; ia < unit_cell().num_atoms(); ia++) {
            auto pos = unit_cell().atom(ia).position();
            for (int x : {0, 1, 2}) {
                phase_factors_(x, i, ia) = std::exp(std::complex<double>(0.0, twopi * (i * pos[x])));
            }
        }
    }

    /* recompute phase factors for atom types */
    phase_factors_t_ = mdarray<std::complex<double>, 2>({gvec().count(), unit_cell().num_atom_types()});
    #pragma omp parallel for schedule(static)
    for (int igloc = 0; igloc < gvec().count(); igloc++) {
        /* global index of G-vector */
        int ig = gvec().offset() + igloc;
        for (int iat = 0; iat < unit_cell().num_atom_types(); iat++) {
            std::complex<double> z(0, 0);
            for (int ia = 0; ia < unit_cell().atom_type(iat).num_atoms(); ia++) {
                z += gvec_phase_factor(ig, unit_cell().atom_type(iat).atom_id(ia));
            }
            phase_factors_t_(igloc, iat) = z;
        }
    }

    if (use_symmetry()) {
        sym_phase_factors_ = mdarray<std::complex<double>, 3>(
                {3, index_range(limits.first, limits.second + 1), unit_cell().symmetry().size()});

        #pragma omp parallel for
        for (int i = limits.first; i <= limits.second; i++) {
            for (int isym = 0; isym < unit_cell().symmetry().size(); isym++) {
                auto t = unit_cell().symmetry()[isym].spg_op.t;
                for (int x : {0, 1, 2}) {
                    sym_phase_factors_(x, i, isym) = std::exp(std::complex<double>(0.0, twopi * (i * t[x])));
                }
            }
        }
    }

    switch (this->processing_unit()) {
        case device_t::CPU: {
            break;
        }
        case device_t::GPU: {
            gvec_->gvec_tp().allocate(memory_t::device).copy_to(memory_t::device);
            break;
        }
    }

    /* create or update radial integrals */
    if (!full_potential()) {
        /* find the new maximum length of G-vectors */
        double new_pw_cutoff{this->pw_cutoff()};
        for (int igloc = 0; igloc < gvec().count(); igloc++) {
            new_pw_cutoff = std::max(new_pw_cutoff, gvec().gvec_len(gvec_index_t::local(igloc)));
        }
        gvec().comm().allreduce<double, mpi::op_t::max>(&new_pw_cutoff, 1);
        /* estimate new G+k-vectors cutoff */
        double new_gk_cutoff = this->gk_cutoff();
        if (new_pw_cutoff > this->pw_cutoff()) {
            new_gk_cutoff += (new_pw_cutoff - this->pw_cutoff());
        }

        /* radial integrals with pw_cutoff */
        if (!ri_.aug_ || ri_.aug_->qmax() < new_pw_cutoff) {
            ri_.aug_ = std::make_unique<Radial_integrals_aug<false>>(unit_cell(), new_pw_cutoff,
                                                                     cfg().settings().nprii_aug(), cb_.aug_ri_);
        }

        if (!ri_.aug_djl_ || ri_.aug_djl_->qmax() < new_pw_cutoff) {
            ri_.aug_djl_ = std::make_unique<Radial_integrals_aug<true>>(unit_cell(), new_pw_cutoff,
                                                                        cfg().settings().nprii_aug(), cb_.aug_ri_djl_);
        }

        if (!ri_.ps_core_ || ri_.ps_core_->qmax() < new_pw_cutoff) {
            ri_.ps_core_ = std::make_unique<Radial_integrals_rho_core_pseudo<false>>(
                    unit_cell(), new_pw_cutoff, cfg().settings().nprii_rho_core(), cb_.rhoc_ri_);
        }

        if (!ri_.ps_core_djl_ || ri_.ps_core_djl_->qmax() < new_pw_cutoff) {
            ri_.ps_core_djl_ = std::make_unique<Radial_integrals_rho_core_pseudo<true>>(
                    unit_cell(), new_pw_cutoff, cfg().settings().nprii_rho_core(), cb_.rhoc_ri_djl_);
        }

        if (!ri_.ps_rho_ || ri_.ps_rho_->qmax() < new_pw_cutoff) {
            ri_.ps_rho_ = std::make_unique<Radial_integrals_rho_pseudo>(unit_cell(), new_pw_cutoff, 20, cb_.ps_rho_ri_);
        }

        if (!ri_.vloc_ || ri_.vloc_->qmax() < new_pw_cutoff) {
            ri_.vloc_ = std::make_unique<Radial_integrals_vloc<false>>(unit_cell(), new_pw_cutoff,
                                                                       cfg().settings().nprii_vloc(), cb_.vloc_ri_);
        }

        if (!ri_.vloc_djl_ || ri_.vloc_djl_->qmax() < new_pw_cutoff) {
            ri_.vloc_djl_ = std::make_unique<Radial_integrals_vloc<true>>(
                    unit_cell(), new_pw_cutoff, cfg().settings().nprii_vloc(), cb_.vloc_ri_djl_);
        }

        /* radial integrals with pw_cutoff */
        if (!ri_.beta_ || ri_.beta_->qmax() < new_gk_cutoff) {
            ri_.beta_ = std::make_unique<Radial_integrals_beta<false>>(unit_cell(), new_gk_cutoff,
                                                                       cfg().settings().nprii_beta(), cb_.beta_ri_);
        }

        if (!ri_.beta_djl_ || ri_.beta_djl_->qmax() < new_gk_cutoff) {
            ri_.beta_djl_ = std::make_unique<Radial_integrals_beta<true>>(
                    unit_cell(), new_gk_cutoff, cfg().settings().nprii_beta(), cb_.beta_ri_djl_);
        }

        auto idxr_wf = [&](int iat) -> radial_functions_index const& {
            return unit_cell().atom_type(iat).indexr_wfs();
        };

        auto ps_wf = [&](int iat, int i) -> Spline<double> const& {
            return unit_cell().atom_type(iat).ps_atomic_wf(i).f;
        };

        if (!ri_.ps_atomic_wf_ || ri_.ps_atomic_wf_->qmax() < new_gk_cutoff) {
            ri_.ps_atomic_wf_ = std::make_unique<Radial_integrals_atomic_wf<false>>(
                    unit_cell(), new_gk_cutoff, 20, idxr_wf, ps_wf, cb_.ps_atomic_wf_ri_);
        }

        if (!ri_.ps_atomic_wf_djl_ || ri_.ps_atomic_wf_djl_->qmax() < new_gk_cutoff) {
            ri_.ps_atomic_wf_djl_ = std::make_unique<Radial_integrals_atomic_wf<true>>(
                    unit_cell(), new_gk_cutoff, 20, idxr_wf, ps_wf, cb_.ps_atomic_wf_ri_djl_);
        }

        for (int iat = 0; iat < unit_cell().num_atom_types(); iat++) {
            if (unit_cell().atom_type(iat).augment() && unit_cell().atom_type(iat).num_atoms() > 0) {
                augmentation_op_[iat] = std::make_unique<Augmentation_operator>(unit_cell().atom_type(iat), gvec(),
                                                                                *ri_.aug_, *ri_.aug_djl_);
                augmentation_op_[iat]->generate_pw_coeffs();
            } else {
                augmentation_op_[iat] = nullptr;
            }
        }
    }

    if (full_potential()) {
        theta_ = init_step_function(this->unit_cell(), this->gvec(), this->gvec_fft(), this->phase_factors_t(),
                                    *this->spfft_transform_);
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
Simulation_context::create_storage_file(std::string name__) const
{
    if (comm_.rank() == 0) {
        /* create new hdf5 file */
        HDF5_tree fout(name__, hdf5_access_t::truncate);
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

        mdarray<int, 2> gv({3, gvec().num_gvec()});
        for (int ig = 0; ig < gvec().num_gvec(); ig++) {
            auto G = gvec().gvec(gvec_index_t::global(ig));
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
Simulation_context::generate_phase_factors(int iat__, mdarray<std::complex<double>, 2>& phase_factors__) const
{
    PROFILE("sirius::Simulation_context::generate_phase_factors");
    int na = unit_cell().atom_type(iat__).num_atoms();
    switch (processing_unit_) {
        case device_t::CPU: {
            #pragma omp parallel for
            for (int igloc = 0; igloc < gvec().count(); igloc++) {
                const int ig = gvec().offset() + igloc;
                for (int i = 0; i < na; i++) {
                    int ia                    = unit_cell().atom_type(iat__).atom_id(i);
                    phase_factors__(igloc, i) = gvec_phase_factor(ig, ia);
                }
            }
            break;
        }
        case device_t::GPU: {
#if defined(SIRIUS_GPU)
            generate_phase_factors_gpu(gvec().count(), na, gvec_coord().at(memory_t::device),
                                       unit_cell().atom_coord(iat__).at(memory_t::device),
                                       phase_factors__.at(memory_t::device));
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

    // double R = R__;

    atoms_to_grid_idx_.resize(unit_cell().num_atoms());

    r3::vector<double> delta(1.0 / spfft<double>().dim_x(), 1.0 / spfft<double>().dim_y(),
                             1.0 / spfft<double>().dim_z());

    const int z_off = spfft<double>().local_z_offset();
    r3::vector<int> grid_beg(0, 0, z_off);
    r3::vector<int> grid_end(spfft<double>().dim_x(), spfft<double>().dim_y(),
                             z_off + spfft<double>().local_z_length());
    std::vector<r3::vector<double>> verts_cart{{-R, -R, -R}, {R, -R, -R}, {-R, R, -R}, {R, R, -R},
                                               {-R, -R, R},  {R, -R, R},  {-R, R, R},  {R, R, R}};

    auto bounds_box = [&](r3::vector<double> pos) {
        std::vector<r3::vector<double>> verts;

        /* pos is a position of atom */
        for (auto v : verts_cart) {
            verts.push_back(pos + unit_cell().get_fractional_coordinates(v));
        }

        std::pair<r3::vector<int>, r3::vector<int>> bounds_ind;

        for (int x : {0, 1, 2}) {
            std::sort(verts.begin(), verts.end(),
                      [x](r3::vector<double>& a, r3::vector<double>& b) { return a[x] < b[x]; });
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
                    auto pos = unit_cell().atom(ia).position() + r3::vector<double>(t0, t1, t2);

                    /* find the small box around this atom */
                    auto box = bounds_box(pos);

                    for (int j0 = box.first[0]; j0 < box.second[0]; j0++) {
                        for (int j1 = box.first[1]; j1 < box.second[1]; j1++) {
                            for (int j2 = box.first[2]; j2 < box.second[2]; j2++) {
                                auto v = pos - r3::vector<double>(delta[0] * j0, delta[1] * j1, delta[2] * j2);
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
Simulation_context::init_comm()
{
    PROFILE("sirius::Simulation_context::init_comm");

    /* check MPI grid dimensions and set a default grid if needed */
    if (!mpi_grid_dims().size()) {
        mpi_grid_dims({1, 1});
    }
    if (mpi_grid_dims().size() != 2) {
        std::stringstream s;
        auto g = mpi_grid_dims();
        s << "MPI grid for band parallelization " << g << " is not 2D";
        RTE_THROW(s);
    }

    const int npr = mpi_grid_dims()[0];
    const int npc = mpi_grid_dims()[1];
    const int npb = npr * npc;
    if (npb <= 0) {
        std::stringstream s;
        s << "wrong mpi grid dimensions : " << npr << " " << npc;
        RTE_THROW(s);
    }
    const int npk = comm_.size() / npb;
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
    mpi_grid_ = std::make_unique<mpi::Grid>(std::vector<int>({npr, npc}), comm_band_);

    /* here we know the number of ranks for band parallelization */

    /* if we have multiple ranks per node and band parallelization, switch to parallel FFT for coarse mesh */
    if ((npr == npb) || (mpi::num_ranks_per_node() > acc::num_devices() && comm_band().size() > 1)) {
        cfg().control().fft_mode("parallel");
    }

    /* create communicator, orthogonal to comm_fft_coarse */
    comm_ortho_fft_coarse_ = comm().split(comm_fft_coarse().rank());
}

} // namespace sirius
