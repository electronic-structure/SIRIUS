/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file testing.hpp
 *
 *  \brief Common functions for the tests and unit tests.
 */

#ifndef __TESTING_HPP__
#define __TESTING_HPP__

#include <string>
#include <iostream>
#include <vector>
#include <cmath>
#include "core/la/linalg.hpp"
#include "core/la/dmatrix.hpp"
#include "core/wf/wave_functions.hpp"
#include "core/r3/r3.hpp"
#include "core/cmd_args.hpp"
#include "core/profiler.hpp"
#include "core/math_tools.hpp"

namespace sirius {

template <typename F, typename... Args>
inline int
call_test(std::string label__, F&& f__, Args&&... args__)
{
    int err{0};
    std::string msg;
    try {
        err = f__(std::forward<Args>(args__)...);
    } catch (std::exception const& e) {
        err = 1;
        msg = e.what();
    } catch (...) {
        err = 2;
        msg = "unknown exception";
    }
    if (err) {
        std::cout << label__ << " : Failed" << std::endl;
        if (msg.size()) {
            std::cout << "exception occured:" << std::endl;
            std::cout << msg << std::endl;
        }
    } else {
        std::cout << label__ << " : OK" << std::endl;
    }
    return err;
}

class Measurement : public std::vector<double>
{
  public:
    double
    average() const
    {
        double d = 0;
        for (size_t i = 0; i < this->size(); i++) {
            d += (*this)[i];
        }
        d /= static_cast<double>(this->size());
        return d;
    }

    double
    sigma() const
    {
        double avg      = average();
        double variance = 0;
        for (size_t i = 0; i < this->size(); i++) {
            variance += std::pow((*this)[i] - avg, 2);
        }
        variance /= static_cast<double>(this->size());
        return std::sqrt(variance);
    }
};

template <typename T>
inline auto
random_symmetric(int N__, int bs__, la::BLACS_grid const& blacs_grid__)
{
    PROFILE("random_symmetric");

    la::dmatrix<T> A(N__, N__, blacs_grid__, bs__, bs__);
    la::dmatrix<T> B(N__, N__, blacs_grid__, bs__, bs__);
    for (int j = 0; j < A.num_cols_local(); j++) {
        for (int i = 0; i < A.num_rows_local(); i++) {
            A(i, j) = random<T>();
        }
    }

#ifdef SIRIUS_SCALAPACK
    la::wrap(la::lib_t::scalapack).tranc(N__, N__, A, 0, 0, B, 0, 0);
#else
    for (int i = 0; i < N__; i++) {
        for (int j = 0; j < N__; j++) {
            B(i, j) = sirius::conj(A(j, i));
        }
    }
#endif

    for (int j = 0; j < A.num_cols_local(); j++) {
        for (int i = 0; i < A.num_rows_local(); i++) {
            A(i, j) = 0.5 * (A(i, j) + B(i, j));
        }
    }

    for (int i = 0; i < N__; i++) {
        A.set(i, i, 50.0);
    }

    return A;
}

template <typename T>
inline auto
random_positive_definite(int N__, int bs__ = 16, la::BLACS_grid const* blacs_grid__ = nullptr)
{
    PROFILE("random_positive_definite");

    double p = 1.0 / N__;
    auto A   = (blacs_grid__) ? la::dmatrix<T>(N__, N__, *blacs_grid__, bs__, bs__) : la::dmatrix<T>(N__, N__);
    auto B   = (blacs_grid__) ? la::dmatrix<T>(N__, N__, *blacs_grid__, bs__, bs__) : la::dmatrix<T>(N__, N__);
    for (int j = 0; j < A.num_cols_local(); j++) {
        for (int i = 0; i < A.num_rows_local(); i++) {
            A(i, j) = p * random<T>();
        }
    }

    if (blacs_grid__) {
#ifdef SIRIUS_SCALAPACK
        la::wrap(la::lib_t::scalapack)
                .gemm('C', 'N', N__, N__, N__, &la::constant<T>::one(), A, 0, 0, A, 0, 0, &la::constant<T>::zero(), B,
                      0, 0);
#else
        RTE_THROW("not compiled with scalapack");
#endif
    } else {
        la::wrap(la::lib_t::blas)
                .gemm('C', 'N', N__, N__, N__, &la::constant<T>::one(), &A(0, 0), A.ld(), &A(0, 0), A.ld(),
                      &la::constant<T>::zero(), &B(0, 0), B.ld());
    }

    for (int i = 0; i < N__; i++) {
        B.set(i, i, 50.0);
    }

    return B;
}

template <typename T>
inline void
randomize(wf::Wave_functions<T>& wf__)
{
    for (int i = 0; i < wf__.num_wf().get(); i++) {
        for (int s = 0; s < wf__.num_sc().get(); s++) {
            auto ptr = wf__.at(memory_t::host, 0, wf::spin_index(s), wf::band_index(i));
            for (int j = 0; j < wf__.ld(); j++) {
                ptr[j] = random<std::complex<double>>();
            }
        }
    }
}

} // namespace sirius

#endif
