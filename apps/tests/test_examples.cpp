/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <spla/spla.hpp>
#include "core/wf/wave_functions.hpp"

using namespace sirius;
using namespace la;
using namespace fft;

void
test1()
{
    if (mpi::Communicator::world().rank() == 0) {
        std::cout << "test1" << std::endl;
    }
    /* reciprocal lattice vectors in
       inverse atomic units */
    r3::matrix<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    /* G-vector cutoff radius in
        inverse atomic units */
    double Gmax = 10;
    /* create a list of G-vectors;
       last boolean parameter switches
       off the reduction of G-vectors by
       inversion symmetry */
    Gvec gvec(M, Gmax, mpi::Communicator::world(), false);
    /* loop over local number of G-vectors
       for current MPI rank */
    for (int j = 0; j < gvec.count(); j++) {
        /* get global index of G-vector */
        int ig = gvec.offset() + j;
        /* get lattice coordinates */
        auto G = gvec.gvec(gvec_index_t::global(ig));
        /* get index of G-vector by lattice coordinates */
        int jg = gvec.index_by_gvec(G);
        /* check for correctness */
        if (ig != jg) {
            throw std::runtime_error("wrong index");
        }
    }
}

void
test2()
{
    if (mpi::Communicator::world().rank() == 0) {
        std::cout << "test2" << std::endl;
    }
    /* reciprocal lattice vectors in
        inverse atomic units */
    r3::matrix<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    /* G-vector cutoff radius in
        inverse atomic units */
    double Gmax = 10;
    /* create a list of G-vectors;
       last boolean parameter switches
       off the reduction of G-vectors by
       inversion symmetry */
    Gvec gvec(M, Gmax, mpi::Communicator::world(), false);
    /* FFT-friendly distribution */
    Gvec_fft gvp(gvec, mpi::Communicator::world(), mpi::Communicator::self());
    /* dimensions of the FFT box */
    fft::Grid dims({40, 40, 40});

    /* this is how our code splits the z-dimension
     * of the FFT buffer */
    auto spl_z = split_z_dimension(dims[2], mpi::Communicator::world());

    /* create SpFFT grid object */
    spfft_grid_type<double> spfft_grid(dims[0], dims[1], dims[2], gvp.zcol_count(), spl_z.local_size(), SPFFT_PU_HOST,
                                       -1, mpi::Communicator::world().native(), SPFFT_EXCH_DEFAULT);

    auto const& gv = gvp.gvec_array();
    /* create SpFFT transformation object */
    spfft_transform_type<double> spfft(spfft_grid.create_transform(SPFFT_PU_HOST, SPFFT_TRANS_C2C, dims[0], dims[1],
                                                                   dims[2], spl_z.local_size(), gvp.count(),
                                                                   SPFFT_INDEX_TRIPLETS, gv.at(memory_t::host)));

    /* create data buffer with local number of G-vectors
       and fill with random numbers */
    mdarray<std::complex<double>, 1> f({gvp.count()});
    f = [](int64_t) { return random<std::complex<double>>(); };
    /* transform to real space */
    spfft.backward(reinterpret_cast<double const*>(&f[0]), SPFFT_PU_HOST);
    /* get real space data pointer */
    auto ptr = reinterpret_cast<std::complex<double>*>(spfft.space_domain_data(SPFFT_PU_HOST));
    /* now the fft buffer contains the real space values */
    for (int j0 = 0; j0 < dims[0]; j0++) {
        for (int j1 = 0; j1 < dims[1]; j1++) {
            for (int jz = 0; jz < spfft.local_z_length(); jz++) {
                auto j2 = spfft.local_z_offset() + jz;
                int idx = dims.index_by_coord(j0, j1, j2);
                /* get the value at (j0, j1, j2) point of the grid */
                ptr[idx] += 0.0;
            }
        }
    }
}

void
test3()
{
    if (mpi::Communicator::world().rank() == 0) {
        std::cout << "test3" << std::endl;
    }
    spla::Context spla_ctx(SPLA_PU_HOST);

    /* reciprocal lattice vectors in
       inverse atomic units */
    r3::matrix<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    /* G-vector cutoff radius in
       inverse atomic units */
    double Gmax = 10;
    /* create a list of G-vectors;
       last boolean parameter switches
       off the reduction of G-vectors by
       inversion symmetry */
    auto gvec = std::make_shared<Gvec>(M, Gmax, mpi::Communicator::world(), false);
    /* number of wave-functions */
    int N = 100;
    /* create scalar wave-functions for N bands */
    wf::Wave_functions<double> wf(gvec, wf::num_mag_dims(0), wf::num_bands(N), memory_t::host);
    /* fill with random numbers */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < gvec->count(); j++) {
            wf.pw_coeffs(j, wf::spin_index(0), wf::band_index(i)) = random<std::complex<double>>();
        }
    }
    /* create a 2x2 BLACS grid */
    BLACS_grid grid(mpi::Communicator::world(), 2, 2);
    /* cyclic block size */
    int bs = 16;
    /* create a distributed overlap matrix */
    dmatrix<std::complex<double>> o(N, N, grid, bs, bs);
    /* create temporary wave-functions */
    wf::Wave_functions<double> tmp(gvec, wf::num_mag_dims(0), wf::num_bands(N), memory_t::host);
    /* orthogonalize wave-functions */
    wf::orthogonalize(spla_ctx, memory_t::host, wf::spin_range(0), wf::band_range(0, 0), wf::band_range(0, N), wf, wf,
                      {&wf}, o, tmp, false);
    /* compute overlap */
    wf::inner(spla_ctx, memory_t::host, wf::spin_range(0), wf, wf::band_range(0, N), wf, wf::band_range(0, N), o, 0, 0);
    /* get the diagonal of the matrix */
    auto d = o.get_diag(N);
    /* check diagonal */
    for (int i = 0; i < N; i++) {
        if (std::abs(d[i] - 1.0) > 1e-10) {
            throw std::runtime_error("wrong overlap");
        }
    }
}

int
main(int argn, char** argv)
{
    cmd_args args;

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(1);
    test1();
    test2();
    test3();
    sirius::finalize();
}
