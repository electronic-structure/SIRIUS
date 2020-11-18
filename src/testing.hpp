// Copyright (c) 2013-2020 Anton Kozhevnikov, Thomas Schulthess
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
#include "SDDK/dmatrix.hpp"
#include "utils/profiler.hpp"
#include "linalg/linalg.hpp"

namespace sirius {

template <typename F>
int call_test(std::string label__, F&& f__)
{
    int err{0};
    std::string msg;
    try {
        err = f__();
    }
    catch (std::exception const& e) {
        err = 1;
        msg = e.what();
    }
    catch (...) {
        err = 1;
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

class Measurement: public std::vector<double>
{
  public:

    double average() const
    {
        double d = 0;
        for (size_t i = 0; i < this->size(); i++) {
            d += (*this)[i];
        }
        d /= static_cast<double>(this->size());
        return d;
    }

    double sigma() const
    {
        double avg = average();
        double variance = 0;
        for (size_t i = 0; i < this->size(); i++) {
            variance += std::pow((*this)[i] - avg, 2);
        }
        variance /= static_cast<double>(this->size());
        return std::sqrt(variance);
    }
};

template <typename T>
sddk::dmatrix<T> random_symmetric(int N__, int bs__, sddk::BLACS_grid const& blacs_grid__)
{
    PROFILE("random_symmetric");

    sddk::dmatrix<T> A(N__, N__, blacs_grid__, bs__, bs__);
    sddk::dmatrix<T> B(N__, N__, blacs_grid__, bs__, bs__);
    for (int j = 0; j < A.num_cols_local(); j++) {
        for (int i = 0; i < A.num_rows_local(); i++) {
            A(i, j) = utils::random<T>();
        }
    }

#ifdef SIRIUS_SCALAPACK
    sddk::linalg(sddk::linalg_t::scalapack).tranc(N__, N__, A, 0, 0, B, 0, 0);
#else
    for (int i = 0; i < N__; i++) {
        for (int j = 0; j < N__; j++) {
            B(i, j) = utils::conj(A(j, i));
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
sddk::dmatrix<T> random_positive_definite(int N__, int bs__, sddk::BLACS_grid const& blacs_grid__)
{
    PROFILE("random_positive_definite");

    double p = 1.0 / N__;
    sddk::dmatrix<T> A(N__, N__, blacs_grid__, bs__, bs__);
    sddk::dmatrix<T> B(N__, N__, blacs_grid__, bs__, bs__);
    for (int j = 0; j < A.num_cols_local(); j++) {
        for (int i = 0; i < A.num_rows_local(); i++) {
            A(i, j) = p * utils::random<T>();
        }
    }

#ifdef SIRIUS_SCALAPACK
    sddk::linalg(sddk::linalg_t::scalapack).gemm('C', 'N', N__, N__, N__, &sddk::linalg_const<T>::one(), A, 0, 0, A, 0, 0,
        &sddk::linalg_const<T>::zero(), B, 0, 0);
#else
    sddk::linalg(sddk::linalg_t::blas).gemm('C', 'N', N__, N__, N__, &sddk::linalg_const<T>::one(), &A(0, 0), A.ld(),
            &A(0, 0), A.ld(), &sddk::linalg_const<T>::zero(), &B(0, 0), B.ld());
#endif

    for (int i = 0; i < N__; i++) {
        B.set(i, i, 50.0);
    }

    return B;
}

}

#endif
