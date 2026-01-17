/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

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
    auto inner_prod_func = [](const Periodic_function<double>& x, const Periodic_function<double>& y) -> double {
        return inner(x, y);
    };

    auto scal_function = [](double alpha, Periodic_function<double>& x) -> void { x *= alpha; };

    auto copy_function = [](Periodic_function<double> const& x, Periodic_function<double>& y) -> void { copy(x, y); };

    auto axpy_function = [](double alpha, const Periodic_function<double>& x, Periodic_function<double>& y) -> void {
        axpy(alpha, x, y);
    };

    auto rotate_function = [](double c, double s, Periodic_function<double>& x, Periodic_function<double>& y) -> void {
        rotate(c, s, x, y);
    };

    return FunctionProperties<Periodic_function<double>>(inner_prod_func, scal_function, copy_function, axpy_function,
                                                         rotate_function);
}

/// Only for the PP-PW case.
FunctionProperties<Periodic_function<double>>
periodic_function_property_rho_pw(bool use_coarse_gvec__)
{
    auto inner_prod_func = [use_coarse_gvec__](Periodic_function<double> const& x,
                                               Periodic_function<double> const& y) -> double {
        double result{0};
        if (use_coarse_gvec__) {
            #pragma omp parallel for reduction(+ : result)
            for (int igloc = x.ctx().gvec_coarse().skip_g0(); igloc < x.ctx().gvec_coarse().count(); igloc++) {
                /* local index in fine G-vector list */
                int ig1 = x.ctx().gvec().gvec_base_mapping(igloc);

                result += std::real(std::conj(x.rg().f_pw_local(ig1)) * y.rg().f_pw_local(ig1)) /
                          std::pow(x.ctx().gvec().gvec_len(gvec_index_t::local(ig1)), 2);
            }
        } else {
            #pragma omp parallel for reduction(+ : result)
            for (int igloc = x.ctx().gvec().skip_g0(); igloc < x.ctx().gvec().count(); igloc++) {
                result += std::real(std::conj(x.rg().f_pw_local(igloc)) * y.rg().f_pw_local(igloc)) /
                          std::pow(x.ctx().gvec().gvec_len(gvec_index_t::local(igloc)), 2);
            }
        }
        if (x.ctx().gvec().reduced()) {
            result *= 2;
        }
        result *= (twopi * x.ctx().unit_cell().omega());
        x.ctx().comm().allreduce(&result, 1);
        return result;
    };

    auto scal_function = [](double alpha, Periodic_function<double>& x) -> void { x *= alpha; };

    auto copy_function = [](Periodic_function<double> const& x, Periodic_function<double>& y) -> void { copy(x, y); };

    auto axpy_function = [](double alpha, const Periodic_function<double>& x, Periodic_function<double>& y) -> void {
        axpy(alpha, x, y);
    };

    auto rotate_function = [](double c, double s, Periodic_function<double>& x, Periodic_function<double>& y) -> void {
        rotate(c, s, x, y);
    };

    return FunctionProperties<Periodic_function<double>>(inner_prod_func, scal_function, copy_function, axpy_function,
                                                         rotate_function);
}

/// Only for the PP-PW case.
FunctionProperties<Periodic_function<double>>
periodic_function_property_mag_pw(bool use_coarse_gvec__)
{
    auto inner_prod_func = [use_coarse_gvec__](Periodic_function<double> const& x,
                                               Periodic_function<double> const& y) -> double {
        double result{0};
        if (use_coarse_gvec__) {
            #pragma omp parallel for reduction(+ : result)
            for (int igloc = x.ctx().gvec_coarse().skip_g0(); igloc < x.ctx().gvec_coarse().count(); igloc++) {
                /* local index in fine G-vector list */
                int ig1 = x.ctx().gvec().gvec_base_mapping(igloc);

                result += std::real(std::conj(x.rg().f_pw_local(ig1)) * y.rg().f_pw_local(ig1));
            }
        } else {
            #pragma omp parallel for reduction(+ : result)
            for (int igloc = x.ctx().gvec().skip_g0(); igloc < x.ctx().gvec().count(); igloc++) {
                result += std::real(std::conj(x.rg().f_pw_local(igloc)) * y.rg().f_pw_local(igloc));
            }
        }
        if (x.ctx().gvec().reduced()) {
            result *= 2;
        }
        result *= (0.5 * x.ctx().unit_cell().omega() / pi);
        x.ctx().comm().allreduce(&result, 1);
        return result;
    };

    auto scal_function = [](double alpha, Periodic_function<double>& x) -> void { x *= alpha; };

    auto copy_function = [](Periodic_function<double> const& x, Periodic_function<double>& y) -> void { copy(x, y); };

    auto axpy_function = [](double alpha, const Periodic_function<double>& x, Periodic_function<double>& y) -> void {
        axpy(alpha, x, y);
    };

    auto rotate_function = [](double c, double s, Periodic_function<double>& x, Periodic_function<double>& y) -> void {
        rotate(c, s, x, y);
    };

    return FunctionProperties<Periodic_function<double>>(inner_prod_func, scal_function, copy_function, axpy_function,
                                                         rotate_function);
}

FunctionProperties<density_matrix_t>
density_function_property()
{
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

    return FunctionProperties<density_matrix_t>(inner_prod_func, scal_function, copy_function, axpy_function,
                                                rotate_function);
}

FunctionProperties<PAW_density<double>>
paw_density_function_property()
{
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

    return FunctionProperties<PAW_density<double>>(inner_prod_func, scale_func, copy_function, axpy_function,
                                                   rotate_function);
}

FunctionProperties<Hubbard_matrix>
hubbard_matrix_function_property()
{
    auto inner_prod_func = [](Hubbard_matrix const& x, Hubbard_matrix const& y) -> double {
        /* do not contribute to mixing */
        return 0;
    };

    auto scale_func = [](double alpha, Hubbard_matrix& x) -> void { scale(alpha, x); };

    // TODO: check with Mathieu which copy function is the one; replace
    auto copy_func = [](Hubbard_matrix const& x, Hubbard_matrix& y) -> void { copy(x, y); };

    auto axpy_func = [](double alpha, Hubbard_matrix const& x, Hubbard_matrix& y) -> void { axpy(alpha, x, y); };

    auto rotate_func = [](double c, double s, Hubbard_matrix& x, Hubbard_matrix& y) -> void { rotate(c, s, x, y); };

    return FunctionProperties<Hubbard_matrix>(inner_prod_func, scale_func, copy_func, axpy_func, rotate_func);
}
} // namespace mixer

} // namespace sirius
