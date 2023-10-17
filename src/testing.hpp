// Copyright (c) 2013-2022 Anton Kozhevnikov, Thomas Schulthess
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
#include "core/wf/wave_functions.hpp"
#include "linalg/linalg.hpp"
#include "linalg/dmatrix.hpp"
#include "core/r3/r3.hpp"
#include "utils/cmd_args.hpp"
#include "utils/profiler.hpp"
#include "context/simulation_context.hpp"

namespace sirius {

template <typename F>
inline int call_test(std::string label__, F&& f__)
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

template <typename F>
inline int call_test(std::string label__, F&& f__, cmd_args const& args__)
{
    printf("running %-30s : ", label__.c_str());
    int result = f__(args__);
    if (result) {
        printf("\x1b[31m" "Failed" "\x1b[0m" "\n");
    } else {
        printf("\x1b[32m" "OK" "\x1b[0m" "\n");
    }
    return result;
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
            B(i, j) = conj(A(j, i));
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
random_positive_definite(int N__, int bs__ = 16, la::BLACS_grid const *blacs_grid__ = nullptr)
{
    PROFILE("random_positive_definite");

    double p = 1.0 / N__;
    auto A = (blacs_grid__) ? la::dmatrix<T>(N__, N__, *blacs_grid__, bs__, bs__) : la::dmatrix<T>(N__, N__);
    auto B = (blacs_grid__) ? la::dmatrix<T>(N__, N__, *blacs_grid__, bs__, bs__) : la::dmatrix<T>(N__, N__);
    for (int j = 0; j < A.num_cols_local(); j++) {
        for (int i = 0; i < A.num_rows_local(); i++) {
            A(i, j) = p * random<T>();
        }
    }

    if (blacs_grid__) {
#ifdef SIRIUS_SCALAPACK
        la::wrap(la::lib_t::scalapack).gemm('C', 'N', N__, N__, N__, &la::constant<T>::one(), A, 0, 0, A, 0, 0,
            &la::constant<T>::zero(), B, 0, 0);
#else
        RTE_THROW("not compiled with scalapack");
#endif
    } else {
        la::wrap(la::lib_t::blas).gemm('C', 'N', N__, N__, N__, &la::constant<T>::one(), &A(0, 0), A.ld(),
                &A(0, 0), A.ld(), &la::constant<T>::zero(), &B(0, 0), B.ld());
    }

    for (int i = 0; i < N__; i++) {
        B.set(i, i, 50.0);
    }

    return B;
}

inline auto
create_simulation_context(nlohmann::json const& conf__, r3::matrix<double> L__, int num_atoms__,
std::vector<r3::vector<double>> coord__, bool add_vloc__, bool add_dion__)
{
    auto ctx = std::make_unique<sirius::Simulation_context>(conf__);

    ctx->unit_cell().set_lattice_vectors(L__);
    if (num_atoms__) {
        auto& atype = ctx->unit_cell().add_atom_type("Cu");

        if (ctx->cfg().parameters().electronic_structure_method() == "full_potential_lapwlo") {

        } else {
            /* set pseudo charge */
            atype.zn(11);
            /* set radial grid */
            atype.set_radial_grid(radial_grid_t::lin_exp, 1000, 0.0, 100.0, 6);
            /* cutoff at ~1 a.u. */
            int icut = atype.radial_grid().index_of(1.0);
            double rcut = atype.radial_grid(icut);
            /* create beta radial function */
            std::vector<double> beta(icut + 1);
            std::vector<double> beta1(icut + 1);
            for (int l = 0; l <= 2; l++) {
                for (int i = 0; i <= icut; i++) {
                    double x = atype.radial_grid(i);
                    beta[i] = confined_polynomial(x, rcut, l, l + 1, 0);
                    beta1[i] = confined_polynomial(x, rcut, l, l + 2, 0);
                }
                /* add radial function for l */
                atype.add_beta_radial_function(angular_momentum(l), beta);
                atype.add_beta_radial_function(angular_momentum(l), beta1);
            }

            std::vector<double> ps_wf(atype.radial_grid().num_points());
            for (int l = 0; l <= 2; l++) {
                for (int i = 0; i < atype.radial_grid().num_points(); i++) {
                    double x = atype.radial_grid(i);
                    ps_wf[i] = std::exp(-x) * std::pow(x, l);
                }
                /* add radial function for l */
                atype.add_ps_atomic_wf(3, angular_momentum(l), ps_wf);
            }

            /* set local part of potential */
            std::vector<double> vloc(atype.radial_grid().num_points(), 0);
            if (add_vloc__) {
                for (int i = 0; i < atype.radial_grid().num_points(); i++) {
                    double x = atype.radial_grid(i);
                    vloc[i] = -atype.zn() / (std::exp(-x * (x + 1)) + x);
                }
            }
            atype.local_potential(vloc);
            /* set Dion matrix */
            int nbf = atype.num_beta_radial_functions();
            sddk::matrix<double> dion(nbf, nbf);
            dion.zero();
            if (add_dion__) {
                for (int i = 0; i < nbf; i++) {
                    dion(i, i) = -10.0;
                }
            }
            atype.d_mtrx_ion(dion);
            /* set atomic density */
            std::vector<double> arho(atype.radial_grid().num_points());
            for (int i = 0; i < atype.radial_grid().num_points(); i++) {
                double x = atype.radial_grid(i);
                arho[i] = 2 * atype.zn() * std::exp(-x * x) * x;
            }
            atype.ps_total_charge_density(arho);
        }

        for (auto v: coord__) {
            ctx->unit_cell().add_atom("Cu", v, {0, 0, 1});
        }
    }
    ctx->initialize();
    return ctx;
}

template <typename T>
inline void
randomize(wf::Wave_functions<T>& wf__)
{
    for (int i = 0; i < wf__.num_wf().get(); i++) {
        for (int s = 0; s < wf__.num_sc().get(); s++) {
            auto ptr = wf__.at(sddk::memory_t::host, 0, wf::spin_index(s), wf::band_index(i));
            for (int j = 0; j < wf__.ld(); j++) {
                ptr[j] = random<std::complex<double>>();
            }
        }
    }
}

}

#endif
