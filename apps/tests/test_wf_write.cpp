/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>

using namespace sirius;

int
test_wf_write_impl(std::vector<int> mpi_grid_dims__, double cutoff__, int num_bands__, int single_file__)
{
    // MPI_grid mpi_grid(mpi_grid_dims__, mpi_comm_world());

    std::vector<int> wf_mpi_grid = {1, mpi::Communicator::world().size()};

    mpi::Grid mpi_grid(wf_mpi_grid, mpi::Communicator::world());

    auto gvec = fft::gkvec_factory(cutoff__, mpi::Communicator::world());
    printf("num_gvec: %i\n", gvec->num_gvec());

    wf::Wave_functions<double> wf(gvec, wf::num_mag_dims(0), wf::num_bands(num_bands__), memory_t::host);
    for (int i = 0; i < num_bands__; i++) {
        for (auto it : *gvec) {
            wf.pw_coeffs(it.igloc, wf::spin_index(0), wf::band_index(i)) = random<std::complex<double>>();
        }
    }
    auto gvec_full = std::make_shared<fft::Gvec_fft>(*gvec, mpi::Communicator::self(), mpi::Communicator::world());
    if (gvec_full->count() != gvec->num_gvec()) {
        RTE_THROW("wrong number of G-vectors");
    }

    auto wf_full = wf::Wave_functions_fft<double>(gvec_full, wf, wf::spin_index(0), wf::band_range(0, num_bands__),
                                                  wf::shuffle_to::fft_layout);

    auto const& comm = mpi::Communicator::world();

    auto t0 = time_now();

    if (single_file__) {
        if (comm.rank() == 0) {
            sirius::HDF5_tree f("wf.h5", hdf5_access_t::truncate);
            f.create_node("wf");
        }

        for (int r = 0; r < comm.size(); r++) {
            if (r == comm.rank()) {
                sirius::HDF5_tree f("wf.h5", hdf5_access_t::read_write);
                mdarray<std::complex<double>, 1> single_wf({gvec->num_gvec()});
                for (auto it : wf_full.spl_num_wf()) {
                    auto ptr = &wf_full.pw_coeffs(0, wf::band_index(it.li));
                    std::copy(ptr, ptr + gvec_full->count(), &single_wf[0]);
                    f["wf"].write(it.i, single_wf);
                }
            }
            comm.barrier();
        }
    } else {
        std::stringstream fname;
        fname << "wf" << comm.rank() << ".h5";
        sirius::HDF5_tree f(fname.str(), hdf5_access_t::truncate);
        f.create_node("wf");

        mdarray<std::complex<double>, 1> single_wf({gvec->num_gvec()});
        for (auto it : wf_full.spl_num_wf()) {
            auto ptr = &wf_full.pw_coeffs(0, wf::band_index(it.li));
            std::copy(ptr, ptr + gvec_full->count(), &single_wf[0]);
            f["wf"].write(it.i, single_wf);
        }
    }
    double t = time_interval(t0);
    if (comm.rank() == 0) {
        printf("io time  : %f (sec.)\n", t);
        printf("io speed : %f (Gb/sec.)\n",
               (gvec->num_gvec() * num_bands__ * sizeof(std::complex<double>)) / double(1 << 30) / t);
    }
    return 0;
}

int
test_wf_write(cmd_args const& args)
{
    double cutoff   = args.value<double>("cutoff", 10);
    int num_bands   = args.value<int>("num_bands", 50);
    int single_file = args.value<int>("single_file", 1);
    auto mpi_grid   = args.value("mpi_grid", std::vector<int>({1, 1}));
    return test_wf_write_impl(mpi_grid, cutoff, num_bands, single_file);
}

int
main(int argn, char** argv)
{
    cmd_args args(argn, argv,
                  {{"cutoff=", "{double} cutoff radius in G-space"},
                   {"num_bands=", "{int} number of bands"},
                   {"mpi_grid=", "{vector2d<int>} MPI grid"},
                   {"single_file=", "{int} write to a single file"}});

    sirius::initialize(1);
    int result = call_test("test_wf_write", test_wf_write, args);
    sirius::finalize();
    return result;
}
