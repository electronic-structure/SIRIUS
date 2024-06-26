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
test1()
{
    int N{400};
    matrix<std::complex<double>> A({N, N});
    matrix<std::complex<double>> B({N, N});
    matrix<std::complex<double>> C({N, N});
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            B(j, i) = random<std::complex<double>>();
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            A(i, j) = B(i, j) + std::conj(B(j, i));
    }
    copy(A, B);

    la::wrap(la::lib_t::lapack).syinv(N, A);
    la::wrap(la::lib_t::blas)
            .hemm('L', 'U', N, N, &la::constant<std::complex<double>>::one(), &A(0, 0), A.ld(), &B(0, 0), B.ld(),
                  &la::constant<std::complex<double>>::zero(), &C(0, 0), C.ld());

    int err{0};
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::complex<double> z = C(i, j);
            if (i == j) {
                z -= 1.0;
            }
            if (std::abs(z) > 1e-10) {
                err++;
            }
        }
    }

    la::wrap(la::lib_t::blas)
            .hemm('L', 'U', N, N, &la::constant<std::complex<double>>::one(), &A(0, 0), A.ld(), &B(0, 0), B.ld(),
                  &la::constant<std::complex<double>>::zero(), &C(0, 0), C.ld());
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::complex<double> z = C(i, j);
            if (i == j) {
                z -= 1.0;
            }
            if (std::abs(z) > 1e-10) {
                err++;
            }
        }
    }
    return err;
}

template <typename T>
int
test2()
{
    int N{400};
    matrix<T> A({N, N});
    matrix<T> B({N, N});
    matrix<T> C({N, N});
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A(j, i) = random<T>();
        }
    }
    copy(A, B);

    la::wrap(la::lib_t::lapack).geinv(N, A);
    la::wrap(la::lib_t::blas)
            .gemm('N', 'N', N, N, N, &la::constant<T>::one(), A.at(memory_t::host), A.ld(), B.at(memory_t::host),
                  B.ld(), &la::constant<T>::zero(), C.at(memory_t::host), C.ld());

    int err{0};
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            T c = C(i, j);
            if (i == j) {
                c -= 1.0;
            }
            if (std::abs(c) > 1e-10) {
                err++;
            }
        }
    }
    return err;
}

/*
#ifdef SIRIUS_SCALAPACK
template <typename T>
int test3()
{
    int bs = 32;

    int num_ranks = mpi::Communicator::world().size();
    int nrc = (int)std::sqrt(0.1 + num_ranks);
    if (nrc * nrc != num_ranks) {
        printf("wrong mpi grid\n");
        exit(-1);
    }

    int N = 400;
    la::BLACS_grid blacs_grid(mpi::Communicator::world(), nrc, nrc);

    la::dmatrix<T> A(N, N, blacs_grid, bs, bs);
    la::dmatrix<T> B(N, N, blacs_grid, bs, bs);
    la::dmatrix<T> C(N, N, blacs_grid, bs, bs);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A.set(j, i, random<T>());
        }
    }
    copy(A, B);

    T alpha = 1.0;
    T beta = 0.0;

    la::wrap(la::lib_t::scalapack).geinv(N, A);

    la::wrap(la::lib_t::scalapack).gemm('N', 'N', N, N, N, &alpha, A, 0, 0, B, 0, 0, &beta, C, 0, 0);

    int err{0};
    for (int i = 0; i < C.num_cols_local(); i++) {
        for (int j = 0; j < C.num_rows_local(); j++) {
            T c = C(j, i);
            if (C.icol(i) == C.irow(j)) {
                c -= 1.0;
            }
            if (std::abs(c) > 1e-10) {
                err++;
            }
        }
    }
    return err;
}
#endif
*/

int
test_linalg()
{
    int err = test1();
    err += test2<double>();
    err += test2<std::complex<double>>();
#ifdef SIRIUS_SCALAPACK
    // err += test3<std::complex<double>>();
#endif
    return err;
}

int
main(int argn, char** argv)
{
    sirius::initialize(1);
    int result = call_test(argv[0], test_linalg);
    sirius::finalize();
    return result;
}
