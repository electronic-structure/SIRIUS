// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
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

/** \file mixer_functions.cpp
 *
 *   \brief Contains implemenations of functions required for mixing.
 */

#include <cassert>

#include "Mixer/mixer_functions.hpp"

namespace sirius {

namespace mixer {
FunctionProperties<Periodic_function<double>> periodic_function_property()
{
    auto global_size_func = [](const Periodic_function<double>& x) -> std::size_t { 
        auto size =  x.f_pw_local().size(); 
        x.ctx().comm().allreduce(&size, 1);
        return size;
    };

    auto inner_prod_func = [](const Periodic_function<double>& x, const Periodic_function<double>& y) -> double {
        return sirius::inner(static_cast<Smooth_periodic_function<double> const&>(x),
                             static_cast<Smooth_periodic_function<double> const&>(y));
    };

    auto scal_function = [](double alpha, Periodic_function<double>& x) -> void {
        #pragma omp parallel
        {
            #pragma omp for schedule(static) nowait
            for (std::size_t i = 0; i < x.f_rg().size(); ++i) {
                x.f_rg(i) *= alpha;
            }
            if (x.ctx().full_potential()) {
                for (int ialoc = 0; ialoc < x.ctx().unit_cell().spl_num_atoms().local_size(); ialoc++) {
                    auto& x_f_mt = x.f_mt(ialoc);
                    #pragma omp for schedule(static) nowait
                    for (int i = 0; i < static_cast<int>(x.f_mt(ialoc).size()); i++) {
                        x_f_mt[i] *= alpha;
                    }
                }
            }
        }
    };

    auto copy_function = [](const Periodic_function<double>& x, Periodic_function<double>& y) -> void {
        #pragma omp parallel
        {
            #pragma omp for schedule(static) nowait
            for (std::size_t i = 0; i < x.f_rg().size(); ++i) {
                y.f_rg(i) = x.f_rg(i);
            }
            if (x.ctx().full_potential()) {
                for (int ialoc = 0; ialoc < x.ctx().unit_cell().spl_num_atoms().local_size(); ialoc++) {
                    assert(x.f_mt(ialoc).size() == y.f_mt(ialoc).size());
                    const auto& x_f_mt = x.f_mt(ialoc);
                    auto& y_f_mt       = y.f_mt(ialoc);
                    #pragma omp for schedule(static) nowait
                    for (int i = 0; i < static_cast<int>(x.f_mt(ialoc).size()); i++) {
                        y_f_mt[i] = x_f_mt[i];
                    }
                }
            }
        }
    };

    auto axpy_function = [](double alpha, const Periodic_function<double>& x, Periodic_function<double>& y) -> void {
        #pragma omp parallel
        {
            #pragma omp for schedule(static) nowait
            for (std::size_t i = 0; i < x.f_rg().size(); ++i) {
                y.f_rg(i) += alpha * x.f_rg(i);
            }
            if (x.ctx().full_potential()) {
                for (int ialoc = 0; ialoc < x.ctx().unit_cell().spl_num_atoms().local_size(); ialoc++) {
                    assert(x.f_mt(ialoc).size() == y.f_mt(ialoc).size());
                    const auto& x_f_mt = x.f_mt(ialoc);
                    auto& y_f_mt       = y.f_mt(ialoc);
                    #pragma omp for schedule(static) nowait
                    for (int i = 0; i < static_cast<int>(x.f_mt(ialoc).size()); i++) {
                        y_f_mt[i] += alpha * x_f_mt[i];
                    }
                }
            }
        }
    };

    return FunctionProperties<Periodic_function<double>>(global_size_func, inner_prod_func, scal_function, copy_function,
                                                         axpy_function);
}

FunctionProperties<sddk::mdarray<double_complex, 4>> density_function_property()
{
    auto global_size_func = [](const mdarray<double_complex, 4>& x) -> std::size_t { return x.size(); };

    auto inner_prod_func = [](const mdarray<double_complex, 4>& x, const mdarray<double_complex, 4>& y) -> double {
        // do not contribute to mixing
        return 0.0;
    };

    auto scal_function = [](double alpha, mdarray<double_complex, 4>& x) -> void {
        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < x.size(); ++i) {
            x[i] *= alpha;
        }
    };

    auto copy_function = [](const mdarray<double_complex, 4>& x, mdarray<double_complex, 4>& y) -> void {
        assert(x.size() == y.size());
        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < x.size(); ++i) {
            y[i] = x[i];
        }
    };

    auto axpy_function = [](double alpha, const mdarray<double_complex, 4>& x, mdarray<double_complex, 4>& y) -> void {
        assert(x.size() == y.size());
        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < x.size(); ++i) {
            y[i] += alpha * x[i];
        }
    };

    return FunctionProperties<sddk::mdarray<double_complex, 4>>(global_size_func, inner_prod_func, scal_function,
                                                                copy_function, axpy_function);
}

} // namespace mixer

} // namespace sirius

