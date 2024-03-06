#include "python_module_includes.hpp"
#include <mpi.h>

#include "core/fft/gvec.hpp"
#include "core/wf/wave_functions.hpp"
#include "magnetization.hpp"
#include "unit_cell_accessors.hpp"
#include "make_sirius_comm.hpp"
#include "dft/smearing.hpp"
#include "hamiltonian/initialize_subspace.hpp"
#include <string>
#include <vector>
#include <list>
#include <complex>
#include <iostream>

using namespace pybind11::literals;
namespace py = pybind11;
using namespace sirius;
using json = nlohmann::json;

using nlohmann::basic_json;

void
init_operators(py::module&);
void
init_r3(py::module&);

// inspired by: https://github.com/mdcb/python-jsoncpp11/blob/master/extension.cpp
py::object
pj_convert(json& node)
{
    switch (node.type()) {
        case json::value_t::null: {
            return py::reinterpret_borrow<py::object>(Py_None);
        }
        case json::value_t::boolean: {
            return py::bool_(node.get<bool>());
        }
        case json::value_t::string: {
            return py::str(node.get<std::string>());
        }
        case json::value_t::number_integer: {
            return py::int_(node.get<int>());
        }
        case json::value_t::number_unsigned: {
            return py::int_(node.get<unsigned int>());
        }
        case json::value_t::number_float: {
            return py::float_(node.get<double>());
        }
        case json::value_t::object: {
            py::dict result;
            for (auto it = node.begin(); it != node.end(); ++it) {
                json my_key(it.key());
                result[pj_convert(my_key)] = pj_convert(*it);
            }
            return result;
        }
        case json::value_t::array: {
            py::list result;
            for (auto it = node.begin(); it != node.end(); ++it) {
                result.append(pj_convert(*it));
            }
            return result;
        }
        default: {
            RTE_THROW("undefined json value");
            /* make compiler happy */
            return py::reinterpret_borrow<py::object>(Py_None);
        }
    }
}

/* typedefs */
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
initialize_subspace(DFT_ground_state& dft_gs, Simulation_context& ctx)
{
    auto& kset = dft_gs.k_point_set();
    Hamiltonian0<double> H0(dft_gs.potential(), false);
    initialize_subspace(kset, H0);
}

PYBIND11_MODULE(py_sirius, m)
{
    // this is needed to be able to pass MPI_Comm from Python->C++
    if (import_mpi4py() < 0) {
        return;
    }
    // MPI_Init/Finalize
    int mpi_init_flag;
    MPI_Initialized(&mpi_init_flag);
    if (mpi_init_flag == true) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            std::cout << "loading SIRIUS python module, MPI already initialized\n";
        }
        sirius::initialize(false);
    } else {
        sirius::initialize(true);
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            std::cout << "loading SIRIUS python module, initialize MPI\n";
        }
    }
    auto atexit = py::module::import("atexit");
    atexit.attr("register")(py::cpp_function([]() {
        int mpi_finalized_flag;
        MPI_Finalized(&mpi_finalized_flag);
        if (mpi_finalized_flag == true) {
            sirius::finalize(false);
        } else {
            sirius::finalize(
                    /* call MPI_Finalize */
                    true,
                    /* reset device */
                    false,
                    /* fftw cleanup */
                    false);
        }
    }));

    try {
        py::module::import("numpy");
    } catch (...) {
        return;
    }

    init_operators(m);
    init_r3(m);

    m.def("num_devices", &acc::num_devices);

    py::class_<mpi::Communicator>(m, "Communicator");

    py::class_<Simulation_context>(m, "Simulation_context")
            .def(py::init<std::string const&>())
            .def(py::init<std::string const&, mpi::Communicator const&>(), py::keep_alive<1, 3>())
            .def("initialize", &Simulation_context::initialize)
            .def("num_bands", py::overload_cast<>(&Simulation_context::num_bands, py::const_))
            .def("num_bands", py::overload_cast<int>(&Simulation_context::num_bands))
            .def("max_occupancy", &Simulation_context::max_occupancy)
            .def("num_fv_states", py::overload_cast<>(&Simulation_context::num_fv_states, py::const_))
            .def("num_spins", &Simulation_context::num_spins)
            .def("verbosity", py::overload_cast<>(&Simulation_context::verbosity, py::const_))
            .def("create_storage_file", &Simulation_context::create_storage_file)
            .def("processing_unit", py::overload_cast<>(&Simulation_context::processing_unit, py::const_))
            .def("processing_unit", py::overload_cast<std::string>(&Simulation_context::processing_unit))
            .def("gvec", &Simulation_context::gvec, py::return_value_policy::reference_internal)
            .def("full_potential", &Simulation_context::full_potential)
            .def_property_readonly("hubbard_correction", &Simulation_context::hubbard_correction)
            .def("unit_cell", py::overload_cast<>(&Simulation_context::unit_cell, py::const_),
                 py::return_value_policy::reference)
            .def("pw_cutoff", py::overload_cast<>(&Simulation_context::pw_cutoff, py::const_))
            .def("pw_cutoff", py::overload_cast<double>(&Simulation_context::pw_cutoff))
            .def("gk_cutoff", py::overload_cast<>(&Simulation_context::gk_cutoff, py::const_))
            .def("gk_cutoff", py::overload_cast<double>(&Simulation_context::gk_cutoff))
            .def("aw_cutoff", py::overload_cast<>(&Simulation_context::aw_cutoff, py::const_))
            .def("aw_cutoff", py::overload_cast<double>(&Simulation_context::aw_cutoff))
            .def("num_spinors", &Simulation_context::num_spinors)
            .def("num_mag_dims", &Simulation_context::num_mag_dims)
            .def_property_readonly("gamma_point", py::overload_cast<>(&Simulation_context::gamma_point, py::const_))
            .def("update", &Simulation_context::update)
            .def("use_symmetry", py::overload_cast<>(&Simulation_context::use_symmetry, py::const_))
            .def("processing_unit_memory_t", &Simulation_context::processing_unit_memory_t)
            .def(
                    "comm", [](Simulation_context& obj) { return make_pycomm(obj.comm()); },
                    py::return_value_policy::reference_internal)
            .def(
                    "comm_k", [](Simulation_context& obj) { return make_pycomm(obj.comm_k()); },
                    py::return_value_policy::reference_internal)
            .def(
                    "comm_fft", [](Simulation_context& obj) { return make_pycomm(obj.comm_fft()); },
                    py::return_value_policy::reference_internal);

    py::class_<Atom>(m, "Atom")
            .def("position", &Atom::position)
            .def("type_id", &Atom::type_id)
            .def("type", &Atom::type, py::return_value_policy::reference)
            .def_property_readonly("label", [](const Atom& obj) { return obj.type().label(); })
            .def_property_readonly("mass", [](const Atom& obj) { return obj.type().mass(); })
            .def("set_position", [](Atom& obj, const std::vector<double>& pos) {
                if (pos.size() != 3)
                    RTE_THROW("wrong input");
                obj.set_position({pos[0], pos[1], pos[2]});
            });

    py::class_<Atom_type>(m, "Atom_type")
            .def_property_readonly("augment", [](const Atom_type& atype) { return atype.augment(); })
            .def_property_readonly("mass", &Atom_type::mass)
            .def_property_readonly("num_atoms", [](const Atom_type& atype) { return atype.num_atoms(); });

    py::class_<Unit_cell>(m, "Unit_cell")
            .def("add_atom_type", &Unit_cell::add_atom_type, py::return_value_policy::reference)
            .def("add_atom",
                 [](Unit_cell& obj, std::string& label, std::array<double, 3>& v) { obj.add_atom(label, v); })
            .def("add_atom", py::overload_cast<const std::string, r3::vector<double>>(&Unit_cell::add_atom))
            .def("atom", py::overload_cast<int>(&Unit_cell::atom), py::return_value_policy::reference)
            .def("atom_type", py::overload_cast<int>(&Unit_cell::atom_type), py::return_value_policy::reference)
            .def("lattice_vectors", &Unit_cell::lattice_vectors)
            .def(
                    "set_lattice_vectors",
                    [](Unit_cell& obj, py::buffer l1, py::buffer l2, py::buffer l3) {
                        set_lattice_vectors(obj, l1, l2, l3);
                    },
                    "l1"_a, "l2"_a, "l3"_a)
            .def(
                    "set_lattice_vectors",
                    [](Unit_cell& obj, std::vector<double> l1, std::vector<double> l2, std::vector<double> l3) {
                        obj.set_lattice_vectors(r3::vector<double>(l1), r3::vector<double>(l2), r3::vector<double>(l3));
                    },
                    "l1"_a, "l2"_a, "l3"_a)
            .def("get_symmetry", &Unit_cell::get_symmetry)
            .def_property_readonly("num_electrons", &Unit_cell::num_electrons)
            .def_property_readonly("num_atoms", &Unit_cell::num_atoms)
            .def_property_readonly("num_valence_electrons", &Unit_cell::num_valence_electrons)
            .def_property_readonly("reciprocal_lattice_vectors", &Unit_cell::reciprocal_lattice_vectors)
            .def("generate_radial_functions", &Unit_cell::generate_radial_functions)
            .def_property_readonly("min_mt_radius", &Unit_cell::min_mt_radius)
            .def_property_readonly("max_mt_radius", &Unit_cell::max_mt_radius)
            .def_property_readonly("omega", &Unit_cell::omega)
            .def("print_info", &Unit_cell::print_info);

    py::class_<Field4D>(m, "Field4D")
            .def(
                    "f_pw_local",
                    [](py::object& obj, int i) -> py::array_t<complex_double> {
                        Field4D& field       = obj.cast<Field4D&>();
                        auto& matrix_storage = field.component_raise(i).rg().f_pw_local();
                        int nrows            = matrix_storage.size(0);
                        /* return underlying data as numpy.ndarray view */
                        return py::array_t<complex_double>({nrows}, {1 * sizeof(complex_double)},
                                                           matrix_storage.at(memory_t::host), obj);
                    },
                    py::keep_alive<0, 1>())
            .def("f_rg",
                 [](py::object& obj, int i) -> py::array_t<double> {
                     Field4D& field       = obj.cast<Field4D&>();
                     auto& matrix_storage = field.component_raise(i).rg().values();
                     int nrows            = matrix_storage.size(0);
                     /* return underlying data as numpy.ndarray view */
                     return py::array_t<double>({nrows}, {1 * sizeof(double)}, matrix_storage.at(memory_t::host), obj);
                 })
            .def("component", py::overload_cast<int>(&Field4D::component), py::return_value_policy::reference_internal);

    py::class_<Potential, Field4D>(m, "Potential")
            .def(py::init<Simulation_context&>(), py::keep_alive<1, 2>(), "ctx"_a)
            .def("generate", &Potential::generate, "density"_a, "use_sym"_a, "transform_to_rg"_a)
            .def("fft_transform", &Potential::fft_transform)
            .def("save", &Potential::save)
            .def("load", &Potential::load)
            .def_property("vxc", py::overload_cast<>(&Potential::xc_potential),
                          py::overload_cast<>(&Potential::xc_potential), py::return_value_policy::reference_internal)
            .def_property("exc", py::overload_cast<>(&Potential::xc_energy_density),
                          py::overload_cast<>(&Potential::xc_energy_density),
                          py::return_value_policy::reference_internal)
            .def_property("vha", py::overload_cast<>(&Potential::hartree_potential),
                          py::overload_cast<>(&Potential::hartree_potential),
                          py::return_value_policy::reference_internal)
            .def_property("vloc", py::overload_cast<>(&Potential::local_potential),
                          py::overload_cast<>(&Potential::local_potential), py::return_value_policy::reference_internal)
            .def("energy_vha", &Potential::energy_vha)
            .def("energy_vxc", &Potential::energy_vxc)
            .def("energy_exc", &Potential::energy_exc)
            .def("PAW_total_energy", &Potential::PAW_total_energy)
            .def("PAW_one_elec_energy", &Potential::PAW_one_elec_energy);

    py::class_<Density, Field4D>(m, "Density")
            .def(py::init<Simulation_context&>(), py::keep_alive<1, 2>(), "ctx"_a)
            .def("initial_density", &Density::initial_density)
            .def("mixer_init", &Density::mixer_init)
            .def("check_num_electrons", &Density::check_num_electrons)
            .def("fft_transform", &Density::fft_transform)
            .def("mix", &Density::mix)
            .def("generate", py::overload_cast<K_point_set const&, bool, bool, bool>(&Density::generate<double>),
                 "kpointset"_a, "symmetrize"_a = false, "add_core"_a = true, "transform_to_rg"_a = false)
            .def("compute_atomic_mag_mom", &Density::compute_atomic_mag_mom)
            .def("save", &Density::save)
            .def("check_num_electrons", &Density::check_num_electrons)
            .def("get_magnetisation", &Density::get_magnetisation)
            .def("load", &Density::load);

    py::class_<DFT_ground_state>(m, "DFT_ground_state")
            .def(py::init<K_point_set&>(), py::keep_alive<1, 2>())
            .def("print_info", &DFT_ground_state::print_info)
            .def("initial_state", &DFT_ground_state::initial_state)
            .def("total_energy", &DFT_ground_state::total_energy)
            .def("serialize",
                 [](DFT_ground_state& dft) {
                     auto json = dft.serialize();
                     return pj_convert(json);
                 })
            .def("density", &DFT_ground_state::density, py::return_value_policy::reference)
            .def(
                    "find",
                    [](DFT_ground_state& dft, double density_tol, double energy_tol, double initial_tol,
                       int num_dft_iter, bool write_state) {
                        json js = dft.find(density_tol, energy_tol, initial_tol, num_dft_iter, write_state);
                        return pj_convert(js);
                    },
                    "density_tol"_a, "energy_tol"_a, "initial_tol"_a, "num_dft_iter"_a, "write_state"_a)
            .def("check_scf_density",
                 [](DFT_ground_state& dft) {
                     json js = dft.check_scf_density();
                     return pj_convert(js);
                 })
            .def("k_point_set", &DFT_ground_state::k_point_set, py::return_value_policy::reference_internal)
            .def("potential", &DFT_ground_state::potential, py::return_value_policy::reference_internal)
            .def("forces", &DFT_ground_state::forces, py::return_value_policy::reference_internal)
            .def("stress", &DFT_ground_state::stress, py::return_value_policy::reference_internal)
            .def("update", &DFT_ground_state::update)
            .def("energy_kin_sum_pw", &DFT_ground_state::energy_kin_sum_pw);

    py::class_<K_point<double>>(m, "K_point")
            .def("band_energy", py::overload_cast<int, int>(&K_point<double>::band_energy, py::const_))
            .def_property_readonly("vk", &K_point<double>::vk, py::return_value_policy::copy)
            .def("generate_fv_states", &K_point<double>::generate_fv_states)
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

    py::class_<Stress>(m, "Stress")
            .def(py::init<Simulation_context&, Density&, Potential&, K_point_set&>())
            .def("calc_stress_total", &Stress::calc_stress_total, py::return_value_policy::reference_internal)
            .def("calc_stress_har", &Stress::calc_stress_har, py::return_value_policy::reference_internal)
            .def("calc_stress_ewald", &Stress::calc_stress_ewald, py::return_value_policy::reference_internal)
            .def("calc_stress_xc", &Stress::calc_stress_xc, py::return_value_policy::reference_internal)
            .def("calc_stress_kin", &Stress::calc_stress_kin, py::return_value_policy::reference_internal)
            .def("calc_stress_vloc", &Stress::calc_stress_vloc, py::return_value_policy::reference_internal)
            .def("print_info", &Stress::print_info);

    py::class_<Force>(m, "Force")
            .def(py::init<Simulation_context&, Density&, Potential&, K_point_set&>())
            .def("calc_forces_total", py::overload_cast<bool>(&Force::calc_forces_total), "add_scf_corr"_a,
                 py::return_value_policy::reference_internal)
            .def_property_readonly("ewald", &Force::forces_ewald, py::return_value_policy::reference_internal)
            .def_property_readonly("hubbard", &Force::forces_hubbard, py::return_value_policy::reference_internal)
            .def_property_readonly("vloc", &Force::forces_vloc, py::return_value_policy::reference_internal)
            .def_property_readonly("nonloc", &Force::forces_nonloc, py::return_value_policy::reference_internal)
            .def_property_readonly("core", &Force::forces_core, py::return_value_policy::reference_internal)
            .def_property_readonly("scf_corr", &Force::forces_scf_corr, py::return_value_policy::reference_internal)
            .def_property_readonly("us", &Force::forces_us, py::return_value_policy::reference_internal)
            .def_property_readonly("total", &Force::forces_total, py::return_value_policy::reference_internal)
            .def("print_info", &Force::print_info);

    py::class_<fft::Grid>(m, "FFT3D_grid")
            .def_property_readonly("num_points", py::overload_cast<>(&fft::Grid::num_points, py::const_))
            .def_property_readonly("shape", [](const fft::Grid& obj) -> std::array<int, 3> {
                return {obj[0], obj[1], obj[2]};
            });

    py::class_<mdarray<complex_double, 1>>(m, "mdarray1c")
            .def("on_device", &mdarray<complex_double, 1>::on_device)
            .def("copy_to_host", [](mdarray<complex_double, 1>& mdarray) { mdarray.copy_to(memory_t::host); })
            .def("__array__", [](py::object& obj) {
                mdarray<complex_double, 1>& arr = obj.cast<mdarray<complex_double, 1>&>();
                int nrows                       = arr.size(0);
                return py::array_t<complex_double>({nrows}, {1 * sizeof(complex_double)}, arr.at(memory_t::host), obj);
            });

    py::class_<mdarray<double, 1>>(m, "mdarray1r")
            .def("on_device", &mdarray<double, 1>::on_device)
            .def("copy_to_host", [](mdarray<double, 1>& mdarray) { mdarray.copy_to(memory_t::host); })
            .def("__array__", [](py::object& obj) {
                mdarray<double, 1>& arr = obj.cast<mdarray<double, 1>&>();
                int nrows               = arr.size(0);
                return py::array_t<double>({nrows}, {1 * sizeof(double)}, arr.at(memory_t::host), obj);
            });

    py::class_<mdarray<complex_double, 2>>(m, "mdarray2c")
            .def("on_device", &mdarray<complex_double, 2>::on_device)
            .def("copy_to_host", [](mdarray<complex_double, 2>& mdarray) { mdarray.copy_to(memory_t::host); })
            .def("__array__", [](py::object& obj) {
                mdarray<complex_double, 2>& arr = obj.cast<mdarray<complex_double, 2>&>();
                int nrows                       = arr.size(0);
                int ncols                       = arr.size(1);
                return py::array_t<complex_double>({nrows, ncols},
                                                   {1 * sizeof(complex_double), nrows * sizeof(complex_double)},
                                                   arr.at(memory_t::host), obj);
            });

    py::class_<mdarray<double, 2>>(m, "mdarray2")
            .def("on_device", &mdarray<double, 2>::on_device)
            .def("copy_to_host",
                 [](mdarray<double, 2>& mdarray) { mdarray.copy_to(memory_t::host, 0, mdarray.size(1)); })
            .def("__array__", [](py::object& obj) {
                mdarray<double, 2>& arr = obj.cast<mdarray<double, 2>&>();
                int nrows               = arr.size(0);
                int ncols               = arr.size(1);
                return py::array_t<double>({nrows, ncols}, {1 * sizeof(double), nrows * sizeof(double)},
                                           arr.at(memory_t::host), obj);
            });

    py::enum_<device_t>(m, "DeviceEnum").value("CPU", device_t::CPU).value("GPU", device_t::GPU);

    py::enum_<memory_t>(m, "MemoryEnum").value("device", memory_t::device).value("host", memory_t::host);
    py::enum_<wf::copy_to>(m, "CopyEnum")
            .value("none", wf::copy_to::none)
            .value("device", wf::copy_to::device)
            .value("host", wf::copy_to::host);
    py::class_<wf::num_mag_dims>(m, "num_mag_dims").def(py::init<int>());
    py::class_<wf::num_bands>(m, "num_bands").def(py::init<int>());
    py::class_<fft::Gvec_fft, std::shared_ptr<fft::Gvec_fft>>(m, "Gvec_fft");
    py::class_<fft::Gvec, std::shared_ptr<fft::Gvec>>(m, "Gvec")
            .def("count", py::overload_cast<>(&fft::Gvec::count, py::const_))
            .def("gkvec_cart", py::overload_cast<gvec_index_t::global>(&fft::Gvec::gkvec_cart, py::const_));
    // use std::shared_ptr as holder type, this required by Hamiltonian.apply_ref, apply_ref_inner
    py::class_<wf::device_memory_guard>(m, "device_memory_guard");

    py::class_<wf::Wave_functions_base<double>, std::shared_ptr<wf::Wave_functions_base<double>>>(m,
                                                                                                  "Wave_functions_base")
            .def("copy_to", &wf::Wave_functions_base<double>::copy_to)
            .def("allocate", &wf::Wave_functions_base<double>::allocate)
            .def("memory_guard", &wf::Wave_functions<double>::memory_guard, "mem"_a, "copy_to"_a)
            .def("deallocate", &wf::Wave_functions_base<double>::deallocate);

    py::class_<wf::Wave_functions<double>, wf::Wave_functions_base<double>,
               std::shared_ptr<wf::Wave_functions<double>>>(m, "Wave_functions")
            .def(py::init<std::shared_ptr<fft::Gvec>, wf::num_mag_dims, wf::num_bands, memory_t>(), "gvecp"_a,
                 "num_mag_dims"_a, "mum_bands"_a, "memory_t"_a)
            .def("num_sc", &wf::Wave_functions<double>::num_sc)
            .def("num_wf", &wf::Wave_functions<double>::num_wf)
            .def("zero",
                 [](wf::Wave_functions<double>& wf) {
                     for (int i = 0; i < wf.num_md(); ++i)
                         wf.pw_coeffs(wf::spin_index(i)).zero();
                 })
            .def(
                    "pw_coeffs",
                    [](py::object& obj, int i) -> py::array_t<complex_double> {
                        auto& wf             = obj.cast<wf::Wave_functions<double>&>();
                        auto& matrix_storage = wf.pw_coeffs(wf::spin_index(i));
                        int nrows            = matrix_storage.size(0);
                        int ncols            = matrix_storage.size(1);
                        /* return underlying data as numpy.ndarray view */
                        return py::array_t<complex_double>({nrows, ncols},
                                                           {1 * sizeof(complex_double), nrows * sizeof(complex_double)},
                                                           matrix_storage.at(memory_t::host), obj);
                    },
                    py::keep_alive<0, 1>())

            .def("allocated_on_device", [](wf::Wave_functions<double>& wf) {
                bool is_on_device = true;
                for (int i = 0; i < wf.num_sc(); ++i) {
                    is_on_device = is_on_device && wf.pw_coeffs(wf::spin_index(i)).on_device();
                }
                return is_on_device;
            });

    py::class_<Smooth_periodic_function<double>>(m, "Smooth_periodic_function")
            .def("fft", [](Smooth_periodic_function<double>& obj) { return obj.fft_transform(-1); })
            .def("ifft", [](Smooth_periodic_function<double>& obj) { return obj.fft_transform(1); })
            .def_property("pw", py::overload_cast<>(&Smooth_periodic_function<double>::f_pw_local, py::const_),
                          py::overload_cast<>(&Smooth_periodic_function<double>::f_pw_local),
                          py::return_value_policy::reference_internal)
            .def_property("rg", py::overload_cast<>(&Smooth_periodic_function<double>::values, py::const_),
                          py::overload_cast<>(&Smooth_periodic_function<double>::values),
                          py::return_value_policy::reference_internal);

    py::class_<Periodic_function<double>>(m, "RPeriodic_function");

    m.def("total_energy", &total_energy);
    m.def("ewald_energy", &ewald_energy);
    m.def("set_atom_positions", &set_atom_positions);
    m.def("atom_positions", &atom_positions);
    m.def("energy_bxc", &energy_bxc);
    m.def("omp_set_num_threads", &omp_set_num_threads);
    m.def("omp_get_num_threads", &omp_get_num_threads);
    m.def("make_sirius_comm", &make_sirius_comm);
    m.def("make_pycomm", &make_pycomm);
    m.def("magnetization", &magnetization);
    m.def("sprint_magnetization", &sprint_magnetization);
    m.def("apply_hamiltonian", &apply_hamiltonian<>, "Hamiltonian0"_a, "kpoint"_a, "wf_out"_a, "wf_in"_a,
          py::arg("swf_out") = nullptr);
    m.def("apply_hamiltonian_gamma", &apply_hamiltonian<true>, "Hamiltonian0"_a, "kpoint"_a, "wf_out"_a, "wf_in"_a,
          py::arg("swf_out") = nullptr);

    m.def("initialize_subspace", &initialize_subspace<double, std::complex<double>>);
    m.def("initialize_subspace_gamma", &initialize_subspace<double, double>);

    /* sirius.smearing submodules */
    py::module smearing_module = m.def_submodule("smearing");
    {
        py::module mpm = smearing_module.def_submodule("methfessel_paxton");
        mpm.def("entropy", py::vectorize(&smearing::methfessel_paxton::entropy), "x"_a, "w"_a, "n"_a);
        mpm.def("delta", py::vectorize(&smearing::methfessel_paxton::delta), "x"_a, "w"_a, "n"_a);
        mpm.def("occupancy", py::vectorize(&smearing::methfessel_paxton::occupancy), "x"_a, "w"_a, "n"_a);
        mpm.def("dxdelta", py::vectorize(&smearing::methfessel_paxton::dxdelta), "x"_a, "w"_a, "n"_a);
    }
    {
        py::module mcold = smearing_module.def_submodule("cold");
        mcold.def("entropy", py::vectorize(&smearing::cold::entropy), "x"_a, "w"_a);
        mcold.def("occupancy", py::vectorize(&smearing::cold::occupancy), "x"_a, "w"_a);
        mcold.def("delta", py::vectorize(&smearing::cold::delta), "x"_a, "w"_a);
        mcold.def("dxdelta", py::vectorize(&smearing::cold::dxdelta), "x"_a, "w"_a);
    }
    {
        py::module mfd = smearing_module.def_submodule("fermi_dirac");
        mfd.def("entropy", py::vectorize(&smearing::fermi_dirac::entropy), "x"_a, "w"_a);
        mfd.def("occupancy", py::vectorize(&smearing::fermi_dirac::occupancy), "x"_a, "w"_a);
        mfd.def("delta", py::vectorize(&smearing::fermi_dirac::delta), "x"_a, "w"_a);
        mfd.def("dxdelta", py::vectorize(&smearing::fermi_dirac::dxdelta), "x"_a, "w"_a);
    }
    {
        py::module mgauss = smearing_module.def_submodule("gaussian");
        mgauss.def("delta", py::vectorize(&smearing::gaussian::delta), "x"_a, "w"_a);
        mgauss.def("occupancy", py::vectorize(&smearing::gaussian::occupancy), "x"_a, "w"_a);
    }
    /* sirius.smearing submodules (end) */
}
