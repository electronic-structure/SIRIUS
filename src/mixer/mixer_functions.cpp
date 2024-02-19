// Copyright (c) 2013-2019 Simon Frasch, Anton Kozhevnikov, Thomas Schulthess
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
 *  \brief Contains implemenations of functions required for mixing.
 */

#include <cassert>

#include "mixer/mixer_functions.hpp"

namespace sirius {

namespace mixer {

FunctionProperties<Periodic_function<double>>
periodic_function_property()
{
    auto global_size_func = [](const Periodic_function<double>& x) -> double { return x.ctx().unit_cell().omega(); };

    auto inner_prod_func = [](const Periodic_function<double>& x, const Periodic_function<double>& y) -> double {
        return sirius::inner(x, y);
    };

    auto scal_function = [](double alpha, Periodic_function<double>& x) -> void {
        scale(alpha, x.rg());
        if (x.ctx().full_potential()) {
            scale(alpha, x.mt());
        }
    };

    auto copy_function = [](const Periodic_function<double>& x, Periodic_function<double>& y) -> void {
        copy(x.rg(), y.rg());
        if (x.ctx().full_potential()) {
            copy(x.mt(), y.mt());
        }
    };

    auto axpy_function = [](double alpha, const Periodic_function<double>& x, Periodic_function<double>& y) -> void {
        axpy(alpha, x.rg(), y.rg());
        if (x.ctx().full_potential()) {
            axpy(alpha, x.mt(), y.mt());
        }
    };

    auto rotate_function = [](double c, double s, Periodic_function<double>& x, Periodic_function<double>& y) -> void {
        #pragma omp parallel
        {
            #pragma omp for schedule(static) nowait
            for (std::size_t i = 0; i < x.rg().values().size(); ++i) {
                auto xi         = x.rg().value(i);
                auto yi         = y.rg().value(i);
                x.rg().value(i) = xi * c + yi * s;
                y.rg().value(i) = xi * -s + yi * c;
            }
            if (x.ctx().full_potential()) {
                for (auto it : x.ctx().unit_cell().spl_num_atoms()) {
                    int ia       = it.i;
                    auto& x_f_mt = x.mt()[ia];
                    auto& y_f_mt = y.mt()[ia];
                    #pragma omp for schedule(static) nowait
                    for (int i = 0; i < static_cast<int>(x.mt()[ia].size()); i++) {
                        auto xi   = x_f_mt[i];
                        auto yi   = y_f_mt[i];
                        x_f_mt[i] = xi * c + yi * s;
                        y_f_mt[i] = xi * -s + yi * c;
                    }
                }
            }
        }
    };

    return FunctionProperties<Periodic_function<double>>(global_size_func, inner_prod_func, scal_function,
                                                         copy_function, axpy_function, rotate_function);
}

/// Only for the PP-PW case.
FunctionProperties<Periodic_function<double>>
periodic_function_property_modified(bool use_coarse_gvec__)
{
    auto global_size_func = [](Periodic_function<double> const& x) -> double {
        return 1.0 / x.ctx().unit_cell().omega();
    };

    auto inner_prod_func = [use_coarse_gvec__](Periodic_function<double> const& x,
                                               Periodic_function<double> const& y) -> double {
        double result{0};
        if (use_coarse_gvec__) {
            for (int igloc = x.ctx().gvec_coarse().skip_g0(); igloc < x.ctx().gvec_coarse().count(); igloc++) {
                /* local index in fine G-vector list */
                int ig1 = x.ctx().gvec().gvec_base_mapping(igloc);

                result += std::real(std::conj(x.rg().f_pw_local(ig1)) * y.rg().f_pw_local(ig1)) /
                          std::pow(x.ctx().gvec().gvec_len(gvec_index_t::local(ig1)), 2);
            }
        } else {
            for (int igloc = x.ctx().gvec().skip_g0(); igloc < x.ctx().gvec().count(); igloc++) {
                result += std::real(std::conj(x.rg().f_pw_local(igloc)) * y.rg().f_pw_local(igloc)) /
                          std::pow(x.ctx().gvec().gvec_len(gvec_index_t::local(igloc)), 2);
            }
        }
        if (x.ctx().gvec().reduced()) {
            result *= 2;
        }
        result *= fourpi;
        x.ctx().comm().allreduce(&result, 1);
        return result;
    };

    auto scal_function = [](double alpha, Periodic_function<double>& x) -> void { scale(alpha, x.rg()); };

    auto copy_function = [](Periodic_function<double> const& x, Periodic_function<double>& y) -> void {
        copy(x.rg(), y.rg());
    };

    auto axpy_function = [](double alpha, const Periodic_function<double>& x, Periodic_function<double>& y) -> void {
        axpy(alpha, x.rg(), y.rg());
    };

    auto rotate_function = [](double c, double s, Periodic_function<double>& x, Periodic_function<double>& y) -> void {
        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < x.rg().values().size(); ++i) {
            auto xi         = x.rg().value(i);
            auto yi         = y.rg().value(i);
            x.rg().value(i) = xi * c + yi * s;
            y.rg().value(i) = xi * -s + yi * c;
        }
    };

    return FunctionProperties<Periodic_function<double>>(global_size_func, inner_prod_func, scal_function,
                                                         copy_function, axpy_function, rotate_function);
}

FunctionProperties<density_matrix_t>
density_function_property()
{
    auto global_size_func = [](density_matrix_t const& x) -> double {
        size_t result{0};
        for (auto& e : x) {
            result += e.size();
        }
        return result;
    };

    auto inner_prod_func = [](density_matrix_t const& x, density_matrix_t const& y) -> double {
        // do not contribute to mixing
        return 0.0;
    };

    auto scal_function = [](double alpha, density_matrix_t& x) -> void {
        for (std::size_t i = 0; i < x.size(); i++) {
            for (std::size_t j = 0; j < x[i].size(); j++) {
                x[i][j] *= alpha;
            }
        }
    };

    auto copy_function = [](density_matrix_t const& x, density_matrix_t& y) -> void {
        assert(x.size() == y.size());
        for (std::size_t i = 0; i < x.size(); i++) {
            copy(x[i], y[i]);
        }
    };

    auto axpy_function = [](double alpha, density_matrix_t const& x, density_matrix_t& y) -> void {
        assert(x.size() == y.size());
        for (std::size_t i = 0; i < x.size(); i++) {
            for (std::size_t j = 0; j < x[i].size(); j++) {
                y[i][j] += alpha * x[i][j];
            }
        }
    };

    auto rotate_function = [](double c, double s, density_matrix_t& x, density_matrix_t& y) -> void {
        assert(x.size() == y.size());
        for (std::size_t i = 0; i < x.size(); i++) {
            for (std::size_t j = 0; j < x[i].size(); j++) {
                auto xi = x[i][j];
                auto yi = y[i][j];
                x[i][j] = xi * c + yi * s;
                y[i][j] = xi * -s + yi * c;
            }
        }
    };

    return FunctionProperties<density_matrix_t>(global_size_func, inner_prod_func, scal_function, copy_function,
                                                axpy_function, rotate_function);
}

FunctionProperties<PAW_density<double>>
paw_density_function_property()
{
    auto global_size_func = [](PAW_density<double> const& x) -> double { return x.unit_cell().num_paw_atoms(); };

    auto inner_prod_func = [](PAW_density<double> const& x, PAW_density<double> const& y) -> double {
        return inner(x, y);
    };

    auto scale_func = [](double alpha, PAW_density<double>& x) -> void {
        for (auto it : x.unit_cell().spl_num_paw_atoms()) {
            int ia = x.unit_cell().paw_atom_index(it.i);
            for (int j = 0; j < x.unit_cell().parameters().num_mag_dims() + 1; j++) {
                x.ae_density(j, ia) *= alpha;
                x.ps_density(j, ia) *= alpha;
            }
        }
    };

    auto copy_function = [](PAW_density<double> const& x, PAW_density<double>& y) -> void {
        for (auto it : x.unit_cell().spl_num_paw_atoms()) {
            int ia = x.unit_cell().paw_atom_index(it.i);
            for (int j = 0; j < x.unit_cell().parameters().num_mag_dims() + 1; j++) {
                copy(x.ae_density(j, ia), y.ae_density(j, ia));
                copy(x.ps_density(j, ia), y.ps_density(j, ia));
            }
        }
    };

    auto axpy_function = [](double alpha, PAW_density<double> const& x, PAW_density<double>& y) -> void {
        for (auto it : x.unit_cell().spl_num_paw_atoms()) {
            int ia = x.unit_cell().paw_atom_index(it.i);
            for (int j = 0; j < x.unit_cell().parameters().num_mag_dims() + 1; j++) {
                y.ae_density(j, ia) = x.ae_density(j, ia) * alpha + y.ae_density(j, ia);
                y.ps_density(j, ia) = x.ps_density(j, ia) * alpha + y.ps_density(j, ia);
            }
        }
    };

    auto rotate_function = [](double c, double s, PAW_density<double>& x, PAW_density<double>& y) -> void {
        for (auto it : x.unit_cell().spl_num_paw_atoms()) {
            int ia = x.unit_cell().paw_atom_index(it.i);
            for (int j = 0; j < x.unit_cell().parameters().num_mag_dims() + 1; j++) {
                x.ae_density(j, ia) = x.ae_density(j, ia) * c + s * y.ae_density(j, ia);
                y.ae_density(j, ia) = y.ae_density(j, ia) * c - s * x.ae_density(j, ia);

                x.ps_density(j, ia) = x.ps_density(j, ia) * c + s * y.ps_density(j, ia);
                y.ps_density(j, ia) = y.ps_density(j, ia) * c - s * x.ps_density(j, ia);
            }
        }
    };

    return FunctionProperties<PAW_density<double>>(global_size_func, inner_prod_func, scale_func, copy_function,
                                                   axpy_function, rotate_function);
}

FunctionProperties<Hubbard_matrix>
hubbard_matrix_function_property()
{
    auto global_size_func = [](Hubbard_matrix const& x) -> double { return 1.0; };

    auto inner_prod_func = [](Hubbard_matrix const& x, Hubbard_matrix const& y) -> double {
        /* do not contribute to mixing */
        return 0;
    };

    auto scale_func = [](double alpha, Hubbard_matrix& x) -> void {
        for (size_t at_lvl = 0; at_lvl < x.local().size(); at_lvl++) {
            for (size_t i = 0; i < x.local(at_lvl).size(); i++) {
                x.local(at_lvl)[i] *= alpha;
            }
        }

        for (size_t at_lvl = 0; at_lvl < x.nonlocal().size(); at_lvl++) {
            for (size_t i = 0; i < x.nonlocal(at_lvl).size(); i++) {
                x.nonlocal(at_lvl)[i] *= alpha;
            }
        }

        if (x.ctx().cfg().hubbard().constrained_calculation()) {
            for (size_t at_lvl = 0; at_lvl < x.multipliers_constraints().size(); at_lvl++) {
                for (size_t i = 0; i < x.multipliers_constraints(at_lvl).size(); i++) {
                    x.multipliers_constraints(at_lvl)[i] *= alpha;
                }
            }
        }
    };

    auto copy_func = [](Hubbard_matrix const& x, Hubbard_matrix& y) -> void {
        for (size_t at_lvl = 0; at_lvl < x.local().size(); at_lvl++) {
            copy(x.local(at_lvl), y.local(at_lvl));
        }

        for (size_t at_lvl = 0; at_lvl < x.nonlocal().size(); at_lvl++) {
            copy(x.nonlocal(at_lvl), y.nonlocal(at_lvl));
        }

        if (x.ctx().cfg().hubbard().constrained_calculation()) {
            for (size_t at_lvl = 0; at_lvl < x.nonlocal().size(); at_lvl++) {
                copy(x.multipliers_constraints(at_lvl), y.multipliers_constraints(at_lvl));
            }
        }
    };

    auto axpy_func = [](double alpha, Hubbard_matrix const& x, Hubbard_matrix& y) -> void {
        for (size_t at_lvl = 0; at_lvl < x.local().size(); at_lvl++) {
            for (size_t i = 0; i < x.local(at_lvl).size(); i++) {
                y.local(at_lvl)[i] = alpha * x.local(at_lvl)[i] + y.local(at_lvl)[i];
            }
        }
        for (size_t at_lvl = 0; at_lvl < x.nonlocal().size(); at_lvl++) {
            for (size_t i = 0; i < x.nonlocal(at_lvl).size(); i++) {
                y.nonlocal(at_lvl)[i] = alpha * x.nonlocal(at_lvl)[i] + y.nonlocal(at_lvl)[i];
            }
        }

        if (x.ctx().cfg().hubbard().constrained_calculation()) {
            for (size_t at_lvl = 0; at_lvl < x.multipliers_constraints().size(); at_lvl++) {
                for (size_t i = 0; i < x.multipliers_constraints(at_lvl).size(); i++) {
                    y.multipliers_constraints(at_lvl)[i] =
                            alpha * x.multipliers_constraints(at_lvl)[i] + y.multipliers_constraints(at_lvl)[i];
                }
            }
        }
    };

    auto rotate_func = [](double c, double s, Hubbard_matrix& x, Hubbard_matrix& y) -> void {
        for (size_t at_lvl = 0; at_lvl < x.local().size(); at_lvl++) {
            for (size_t i = 0; i < x.local(at_lvl).size(); i++) {
                auto xi            = x.local(at_lvl)[i];
                auto yi            = y.local(at_lvl)[i];
                x.local(at_lvl)[i] = xi * c + yi * s;
                y.local(at_lvl)[i] = yi * c - xi * s;
            }
        }

        for (size_t at_lvl = 0; at_lvl < x.nonlocal().size(); at_lvl++) {
            for (size_t i = 0; i < x.nonlocal(at_lvl).size(); i++) {
                auto xi               = x.nonlocal(at_lvl)[i];
                auto yi               = y.nonlocal(at_lvl)[i];
                x.nonlocal(at_lvl)[i] = xi * c + yi * s;
                y.nonlocal(at_lvl)[i] = yi * c - xi * s;
            }
        }

        if (x.ctx().cfg().hubbard().constrained_calculation()) {
            for (size_t at_lvl = 0; at_lvl < x.multipliers_constraints().size(); at_lvl++) {
                for (size_t i = 0; i < x.multipliers_constraints(at_lvl).size(); i++) {
                    auto xi                              = x.multipliers_constraints(at_lvl)[i];
                    auto yi                              = y.multipliers_constraints(at_lvl)[i];
                    x.multipliers_constraints(at_lvl)[i] = xi * c + yi * s;
                    y.multipliers_constraints(at_lvl)[i] = yi * c - xi * s;
                }
            }
        }
    };

    return FunctionProperties<Hubbard_matrix>(global_size_func, inner_prod_func, scale_func, copy_func, axpy_func,
                                              rotate_func);
}
} // namespace mixer

} // namespace sirius
