/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file py_sirius_kpoints.cpp
 *
 *  \brief Python bindings: k-points, application of the Hamiltonian, subspace initialization.
 */

#include <pybind11/pybind11.h>
#include "k_point/k_point_set.hpp"
#include "hamiltonian/hamiltonian.hpp"
#include "hamiltonian/initialize_subspace.hpp"
#include "py_sirius.hpp"

using namespace pybind11::literals;
namespace py = pybind11;
using namespace sirius;
using complex_double = std::complex<double>;

template <bool gamma_point = false>
void
apply_hamiltonian(Hamiltonian0<double>& H0, K_point<double>& kp, wf::Wave_functions<double>& wf_out,
                  wf::Wave_functions<double>& wf, std::shared_ptr<wf::Wave_functions<double>>& swf)
{
    /////////////////////////////////////////////////////////////
    // // TODO: Hubbard needs manual call to copy to device // //
    /////////////////////////////////////////////////////////////
    int num_wf = wf.num_wf();
    int num_sc = wf.num_sc();
    if (num_wf != wf_out.num_wf() || wf_out.num_sc() != num_sc) {
        RTE_THROW("Hamiltonian::apply_ref (python bindings): num_sc or num_wf do not match");
    }
    auto H         = H0(kp);
    auto& ctx      = H0.ctx();
    auto mg_wf     = wf.memory_guard(ctx.processing_unit_memory_t(), wf::copy_to::device);
    auto mg_wf_out = wf_out.memory_guard(ctx.processing_unit_memory_t(), wf::copy_to::host);

    /* apply H to all wave functions */
    int N = 0;
    int n = num_wf;
    for (int ispn_step = 0; ispn_step < ctx.num_spinors(); ispn_step++) {
        // sping_range: 2 for non-colinear magnetism, otherwise ispn_step
        auto spin_range = wf::spin_range((ctx.num_mag_dims() == 3) ? 2 : ispn_step);
        if constexpr (gamma_point) {
            H.apply_h_s<double>(spin_range, wf::band_range(N, N + n), wf, &wf_out, swf.get());
        } else {
            H.apply_h_s<complex_double>(spin_range, wf::band_range(N, N + n), wf, &wf_out, swf.get());
        }
    }
    if (is_device_memory(ctx.processing_unit_memory_t())) {
        if (swf) {
            swf->copy_to(memory_t::host);
        }
    }
}

void
init_kpoints(py::module& m)
{
    py::class_<K_point<double>>(m, "K_point")
            .def("band_energy", py::overload_cast<int, int>(&K_point<double>::band_energy, py::const_))
            .def_property_readonly("vk", &K_point<double>::vk, py::return_value_policy::copy)
            .def("set_band_energy",
                 [](K_point<double>& kpoint, int j, int ispn, double val) { kpoint.band_energy(j, ispn, val); })
            .def(
                    "band_energies",
                    [](K_point<double> const& kpoint, int ispn) {
                        std::vector<double> energies(kpoint.ctx().num_bands());
                        for (int i = 0; i < kpoint.ctx().num_bands(); ++i) {
                            energies[i] = kpoint.band_energy(i, ispn);
                        }
                        return energies;
                    },
                    py::return_value_policy::copy)
            .def("band_occupancy",
                 [](K_point<double> const& kpoint, int ispn) {
                     std::vector<double> occ(kpoint.ctx().num_bands());
                     for (int i = 0; i < kpoint.ctx().num_bands(); ++i) {
                         occ[i] = kpoint.band_occupancy(i, ispn);
                     }
                     return occ;
                 })
            .def(
                    "set_band_occupancy",
                    [](K_point<double>& kpoint, int ispn, const std::vector<double>& fn) {
                        assert(static_cast<int>(fn.size()) == kpoint.ctx().num_bands());
                        for (size_t i = 0; i < fn.size(); ++i) {
                            kpoint.band_occupancy(i, ispn, fn[i]);
                        }
                    },
                    "ispn"_a, "fn"_a)
            .def("gkvec_partition", &K_point<double>::gkvec_fft_sptr)
            .def("gkvec", &K_point<double>::gkvec_sptr)
            .def("fv_states", &K_point<double>::fv_states, py::return_value_policy::reference_internal)
            .def("ctx", &K_point<double>::ctx, py::return_value_policy::reference_internal)
            .def("weight", &K_point<double>::weight)
            .def("beta_projectors", py::overload_cast<>(&K_point<double>::beta_projectors, py::const_),
                 py::return_value_policy::reference_internal)
            .def("spinor_wave_functions", py::overload_cast<>(&K_point<double>::spinor_wave_functions),
                 py::return_value_policy::reference_internal);

    py::class_<K_point_set>(m, "K_point_set")
            .def(py::init<Simulation_context&>(), py::keep_alive<1, 2>())
            .def(py::init<Simulation_context&, std::vector<std::array<double, 3>>>(), py::keep_alive<1, 2>())
            .def(py::init<Simulation_context&, std::initializer_list<std::array<double, 3>>>(), py::keep_alive<1, 2>())
            .def(py::init<Simulation_context&, r3::vector<int>, r3::vector<int>, bool>(), py::keep_alive<1, 2>())
            .def(py::init<Simulation_context&, std::vector<int>, std::vector<int>, bool>(), py::keep_alive<1, 2>())
            .def("initialize", &K_point_set::initialize, py::arg("counts") = std::vector<int>{})
            .def("ctx", &K_point_set::ctx, py::return_value_policy::reference_internal)
            .def("unit_cell", &K_point_set::unit_cell, py::return_value_policy::reference_internal)
            .def("_num_kpoints", &K_point_set::num_kpoints)
            .def("size", [](K_point_set& ks) -> int { return ks.spl_num_kpoints().local_size(); })
            .def("energy_fermi", &K_point_set::energy_fermi)
            .def("get_band_energies", &K_point_set::get_band_energies<double>)
            .def("find_band_occupancies", &K_point_set::find_band_occupancies<double>)
            .def("band_gap", &K_point_set::band_gap)
            .def("sync_band_energy", &K_point_set::sync_band<double, sync_band_t::energy>)
            .def("sync_band_occupancy", &K_point_set::sync_band<double, sync_band_t::occupancy>)
            .def("valence_eval_sum", static_cast<double (K_point_set::*)() const>(&K_point_set::valence_eval_sum))
            .def("__contains__",
                 [](K_point_set& ks, int i) { return (i >= 0 && i < ks.spl_num_kpoints().local_size()); })
            .def(
                    "__getitem__",
                    [](K_point_set& ks, int i) -> K_point<double>& {
                        if (i >= ks.spl_num_kpoints().local_size())
                            throw pybind11::index_error("out of bounds");
                        return *ks.get<double>(ks.spl_num_kpoints().global_index(typename kp_index_t::local(i)));
                    },
                    py::return_value_policy::reference_internal)
            .def("__len__", [](K_point_set const& ks) { return ks.spl_num_kpoints().local_size(); })
            .def("add_kpoint",
                 [](K_point_set& ks, std::vector<double> v, double weight) { ks.add_kpoint(v.data(), weight); })
            .def("add_kpoint",
                 [](K_point_set& ks, r3::vector<double>& v, double weight) { ks.add_kpoint(&v[0], weight); });

    m.def("apply_hamiltonian", &apply_hamiltonian<>, "Hamiltonian0"_a, "kpoint"_a, "wf_out"_a, "wf_in"_a,
          py::arg("swf_out") = nullptr);
    m.def("apply_hamiltonian_gamma", &apply_hamiltonian<true>, "Hamiltonian0"_a, "kpoint"_a, "wf_out"_a, "wf_in"_a,
          py::arg("swf_out") = nullptr);

    m.def("initialize_subspace", &initialize_subspace<double, std::complex<double>>);
    m.def("initialize_subspace_gamma", &initialize_subspace<double, double>);
}
