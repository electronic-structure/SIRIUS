/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file sirius_init.cpp
 *
 *  \brief Initialization and finalization of the library.
 */

#include <fstream>
#include <memory>
#include "sirius_init.hpp"
#if defined(__APEX)
#include <apex_api.hpp>
#endif
#include "core/omp.hpp"
#include "core/acc/acc.hpp"
#if defined(SIRIUS_GPU)
#include "core/acc/acc_blas.hpp"
#endif
#if defined(SIRIUS_GPU) && defined(SIRIUS_CUDA)
#include "core/acc/cusolver.hpp"
#endif
#include "core/env/env.hpp"
#include "core/la/linalg_spla.hpp"
#include "core/la/eigenproblem.hpp"
#include "core/memory.hpp"
#include "core/mpi/communicator.hpp"
#include "core/power.hpp"
#include "core/profiler.hpp"
#include "core/sirius_version.hpp"

namespace sirius {

bool&
is_initialized()
{
    static bool b{false};
    return b;
}

namespace {

struct null_buffer : std::streambuf
{
    int
    overflow(int c) override
    {
        return c;
    }
};

std::unique_ptr<power::Profile> energy_profiler;

} // namespace

void
initialize(bool call_mpi_init__)
{
    PROFILE_START("sirius");
    PROFILE("sirius::initialize");
    if (is_initialized()) {
        RTE_THROW("SIRIUS library is already initialized");
    }
    if (call_mpi_init__) {
        PROFILE("sirius::initialize::mpi");
        mpi::Communicator::initialize(MPI_THREAD_MULTIPLE);
    }
#if defined(__APEX)
    apex::init("sirius", Communicator::world().rank(), Communicator::world().size());
#endif

#if defined(SIRIUS_USE_POWER_COUNTER)
    energy_profiler = std::make_unique<power::Profile>("sirius");
#endif

    if (env::get_verbosity() > 0) {
        if (mpi::Communicator::world().rank() == 0) {
            std::cout << "# SIRIUS " << major_version() << "." << minor_version() << "." << revision() << "."
                      << ", git hash: " << git_hash() << std::endl;
#if !defined(NDEBUG)
            std::cout << "# Warning! Compiled in 'debug' mode with assert statements enabled!" << std::endl;
#endif
        }
    }

    // uncomment this if you want to supress std::cout from all MPI ranks except from rank=0
    //static null_buffer null;
    //if (mpi::Communicator::world().rank() != 0) {
    //    std::cout.rdbuf(&null);
    //}

    /* get number of ranks per node during the global call to sirius::initialize() */
    mpi::num_ranks_per_node();
    if (acc::num_devices() > 0) {
        PROFILE("sirius::initialize::acc");
        int devid = mpi::get_device_id(acc::num_devices());
        acc::set_device_id(devid);
        /* create extensive amount of streams */
        /* some parts of the code rely on the number of streams not related to the
           number of OMP threads */
        acc::create_streams(std::max(omp_get_max_threads(), 6));
#if defined(SIRIUS_GPU)
        acc::blas::create_stream_handles();
#endif
#if defined(SIRIUS_CUDA)
        acc::blas::xt::create_handle();
        acc::cusolver::create_handle();
#endif
    }
    splablas::reset_handle();

#if defined(SIRIUS_MAGMA)
    magma::init();
#endif
#if defined(SIRIUS_ELPA)
    la::Eigensolver_elpa::initialize();
#endif
    /* for the fortran interface to blas/lapack */
    RTE_ASSERT(sizeof(int) == 4);
    RTE_ASSERT(sizeof(double) == 8);

    is_initialized() = true;
}

void
finalize(bool call_mpi_fin__, bool reset_device__)
{
    PROFILE_START("sirius::finalize");
    if (!is_initialized()) {
        RTE_THROW("SIRIUS library was not initialized");
    }
#if defined(SIRIUS_MAGMA)
    magma::finalize();
#endif

    /* must be called before device is reset */
    splablas::reset_handle();

    get_memory_pool(memory_t::host).clear();

    if (acc::num_devices()) {
        get_memory_pool(memory_t::host_pinned).clear();
        get_memory_pool(memory_t::device).clear();
#if defined(SIRIUS_GPU)
        acc::blas::destroy_stream_handles();
#endif
#if defined(SIRIUS_CUDA)
        acc::cusolver::destroy_handle();
        acc::blas::xt::destroy_handle();
#endif
        acc::destroy_streams();
    }

#if defined(__APEX)
    apex::finalize();
#endif

#if defined(SIRIUS_USE_POWER_COUNTER)
    energy_profiler = nullptr;

    auto local_id = env::get_value_ptr<int>("SLURM_LOCALID");
    if (local_id) {
        int color = (*local_id == 0) ? 0 : 1;
        auto comm = mpi::Communicator::world().split(color);
        if (color == 0) {
            /* only first rank of each node participates in report generation */
            power::report(comm);
        }
    }
#endif

#if defined(SIRIUS_DLAF)
    // Finalization frees DLA-Future grids
    // It needs to be called before finalizing MPI
    la::Eigensolver_dlaf::finalize();
#endif
    auto rank = mpi::Communicator::world().rank();
    if (call_mpi_fin__) {
        mpi::Communicator::finalize();
    }
#if defined(SIRIUS_ELPA)
    la::Eigensolver_elpa::finalize();
#endif
    if (acc::num_devices() && reset_device__) {
        acc::reset();
    }

    is_initialized() = false;

    PROFILE_STOP("sirius::finalize");
    PROFILE_STOP("sirius");

    auto pt = env::print_timing();
    if (pt && rank == 0) {
        auto timing_result = global_rtgraph_timer.process();

        if (pt & 1) {
            std::cout << timing_result.print({rt_graph::Stat::Count, rt_graph::Stat::Total, rt_graph::Stat::Percentage,
                                              rt_graph::Stat::SelfPercentage, rt_graph::Stat::Median,
                                              rt_graph::Stat::Min, rt_graph::Stat::Max});
        }
        if (pt & 2) {
            timing_result = timing_result.flatten(1).sort_nodes();
            std::cout << timing_result.print({rt_graph::Stat::Count, rt_graph::Stat::Total, rt_graph::Stat::Percentage,
                                              rt_graph::Stat::SelfPercentage, rt_graph::Stat::Median,
                                              rt_graph::Stat::Min, rt_graph::Stat::Max});
        }
        if (pt & 4) {
            std::ofstream ofs("timers.json", std::ofstream::out | std::ofstream::trunc);
            ofs << timing_result.json();
        }
    }
}

} // namespace sirius
