#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <sirius.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/complex.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <utility>
#include <memory>
#include <stdexcept>
#include <omp.h>

#include "utils/json.hpp"
#include "Unit_cell/free_atom.hpp"
#include "energy.hpp"


using namespace pybind11::literals;
namespace py = pybind11;
using namespace sirius;
using namespace geometry3d;
using json = nlohmann::json;
using nlohmann::basic_json;

//inspired by: https://github.com/mdcb/python-jsoncpp11/blob/master/extension.cpp
py::object pj_convert(json& node)
{
    switch (node.type()) {
        case json::value_t::null: {
            return py::reinterpret_borrow<py::object>(Py_None);
        }
        case json::value_t::boolean: {
            bool b(node);
            return py::bool_(b);
        }
        case json::value_t::string: {
            std::string s;
            s = static_cast<std::string const&>(node);
            return py::str(s);
        }
        case json::value_t::number_integer: {
            int i(node);
            return py::int_(i);
        }
        case json::value_t::number_unsigned: {
            unsigned int u(node);
            return py::int_(u);
        }
        case json::value_t::number_float: {
            float f(node);
            return py::float_(f);
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
            throw std::runtime_error("undefined json value");
            /* make compiler happy */
            return py::reinterpret_borrow<py::object>(Py_None);
        }
    }
}

std::string show_mat(const matrix3d<double>& mat)
{
    std::string str = "[";
    for (int i = 0; i < 2; ++i) {
        str = str + "[" + std::to_string(mat(i, 0)) + "," + std::to_string(mat(i, 1)) + "," + std::to_string(mat(i, 2)) + "]" + "\n";
    }
    str = str + "[" + std::to_string(mat(2, 0)) + "," + std::to_string(mat(2, 1)) + "," + std::to_string(mat(2, 2)) + "]" + "]";
    return str;
}

template <class T>
std::string show_vec(const vector3d<T>& vec)
{
    std::string str = "[" + std::to_string(vec[0]) + "," + std::to_string(vec[1]) + "," + std::to_string(vec[2]) + "]";
    return str;
}

/* typedefs */
template <typename T>
using matrix_storage_slab = sddk::matrix_storage<T, sddk::matrix_storage_t::slab>;
using complex_double      = std::complex<double>;

PYBIND11_MODULE(py_sirius, m)
{

    // MPI_Init/Finalize
    int mpi_init_flag;
    MPI_Initialized(&mpi_init_flag);
    if (mpi_init_flag == true) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
            std::cout << "loading SIRIUS python module, MPI already initialized"
                      << "\n";
        sirius::initialize(false);
    } else {
        sirius::initialize(true);
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
            std::cout << "loading SIRIUS python module, initialize MPI"
                      << "\n";
    }
    auto atexit = py::module::import("atexit");
    atexit.attr("register")(py::cpp_function([]() {
        int mpi_finalized_flag;
        MPI_Finalized(&mpi_finalized_flag);
        if(mpi_finalized_flag == true) {
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

    m.def("timer_print", &utils::timer::print);
#ifdef __GPU
    m.def("num_devices", &acc::num_devices);
#endif

    py::class_<Parameters_input>(m, "Parameters_input")
        .def(py::init<>())
        .def_readwrite("potential_tol_", &Parameters_input::potential_tol_)
        .def_readwrite("energy_tol_", &Parameters_input::energy_tol_)
        .def_readwrite("num_dft_iter_", &Parameters_input::num_dft_iter_);

    py::class_<Simulation_context>(m, "Simulation_context")
        .def(py::init<>())
        .def(py::init<std::string const&>())
        .def("initialize", &Simulation_context::initialize)
        .def("num_bands", py::overload_cast<>(&Simulation_context::num_bands, py::const_))
        .def("num_bands", py::overload_cast<int>(&Simulation_context::num_bands))
        .def("num_spins", &Simulation_context::num_spins)
        .def("set_verbosity", &Simulation_context::set_verbosity)
        .def("create_storage_file", &Simulation_context::create_storage_file)
        .def("processing_unit", &Simulation_context::processing_unit)
        .def("set_processing_unit", py::overload_cast<device_t>(&Simulation_context::set_processing_unit))
        .def("gvec", &Simulation_context::gvec, py::return_value_policy::reference_internal)
        .def("full_potential", &Simulation_context::full_potential)
        .def("hubbard_correction", &Simulation_context::hubbard_correction)
        .def("fft", &Simulation_context::fft, py::return_value_policy::reference_internal)
        .def("unit_cell", py::overload_cast<>(&Simulation_context::unit_cell, py::const_), py::return_value_policy::reference)
        .def("pw_cutoff", &Simulation_context::pw_cutoff)
        .def("parameters_input", py::overload_cast<>(&Simulation_context::parameters_input, py::const_), py::return_value_policy::reference)
        .def("num_spin_dims", &Simulation_context::num_spin_dims)
        .def("num_mag_dims", &Simulation_context::num_mag_dims)
        .def("set_gamma_point", &Simulation_context::set_gamma_point)
        .def("set_pw_cutoff", &Simulation_context::set_pw_cutoff)
        .def("update", &Simulation_context::update)
        .def("use_symmetry", py::overload_cast<>(&Simulation_context::use_symmetry, py::const_))
        .def("set_iterative_solver_tolerance", &Simulation_context::set_iterative_solver_tolerance);

    py::class_<Unit_cell>(m, "Unit_cell")
        .def("add_atom_type", static_cast<void (Unit_cell::*)(const std::string, const std::string)>(&Unit_cell::add_atom_type))
        .def("add_atom", py::overload_cast<const std::string, std::vector<double>>(&Unit_cell::add_atom))
        .def("atom_type", py::overload_cast<int>(&Unit_cell::atom_type), py::return_value_policy::reference)
        .def("set_lattice_vectors", static_cast<void (Unit_cell::*)(matrix3d<double>)>(&Unit_cell::set_lattice_vectors))
        .def("lattice_vectors", &Unit_cell::lattice_vectors)
        .def("get_symmetry", &Unit_cell::get_symmetry)
        .def("reciprocal_lattice_vectors", &Unit_cell::reciprocal_lattice_vectors)
        .def("generate_radial_functions", &Unit_cell::generate_radial_functions);

    py::class_<z_column_descriptor>(m, "z_column_descriptor")
        .def_readwrite("x", &z_column_descriptor::x)
        .def_readwrite("y", &z_column_descriptor::y)
        .def_readwrite("z", &z_column_descriptor::z)
        .def(py::init<int, int, std::vector<int>>());

    py::class_<Gvec>(m, "Gvec")
        .def(py::init<matrix3d<double>, double, bool>())
        .def("num_gvec", &sddk::Gvec::num_gvec)
        .def("count", &sddk::Gvec::count)
        .def("offset", &sddk::Gvec::offset)
        .def("gvec", &sddk::Gvec::gvec)
        .def("gkvec", &sddk::Gvec::gkvec)
        .def("gkvec_cart", &sddk::Gvec::gkvec_cart<index_domain_t::global>)
        .def("num_zcol", &sddk::Gvec::num_zcol)
        .def("gvec_alt", [](Gvec& obj, int idx) {
            vector3d<int>    vec(obj.gvec(idx));
            std::vector<int> retr = {vec[0], vec[1], vec[2]};
            return retr;
        })
        .def("index_by_gvec", [](Gvec& obj, std::vector<int> vec) {
            vector3d<int> vec3d(vec);
            return obj.index_by_gvec(vec3d);
        })
        .def("zcol", [](Gvec& gvec, int idx) {
            z_column_descriptor obj(gvec.zcol(idx));
            py::dict            dict("x"_a = obj.x, "y"_a = obj.y, "z"_a = obj.z);
            return dict;
        })
        .def("index_by_gvec", &Gvec::index_by_gvec);

    py::class_<Gvec_partition>(m, "Gvec_partition");

    py::class_<vector3d<int>>(m, "vector3d_int")
        .def(py::init<std::vector<int>>())
        .def("__call__", [](const vector3d<int>& obj, int x) {
            return obj[x];
        })
        .def("__repr__", [](const vector3d<int>& vec) {
            return show_vec(vec);
        })
        .def("__len__", &vector3d<int>::length)
        .def("__array__", [](vector3d<int>& v3d) {
            py::array_t<int> x(3);
            auto r = x.mutable_unchecked<1>();
            r(0) = v3d[0];
            r(1) = v3d[1];
            r(2) = v3d[2];
            return x;
        });

    py::class_<vector3d<double>>(m, "vector3d_double")
        .def(py::init<std::vector<double>>())
        .def("__call__", [](const vector3d<double>& obj, int x) {
            return obj[x];
        })
        .def("__repr__", [](const vector3d<double>& vec) {
            return show_vec(vec);
        })
        .def("__array__", [](vector3d<double>& v3d) {
            py::array_t<double> x(3);
            auto r = x.mutable_unchecked<1>();
            r(0) = v3d[0];
            r(1) = v3d[1];
            r(2) = v3d[2];
            return x;
        })
        .def("__len__", &vector3d<double>::length)
        .def(py::self - py::self)
        .def(py::self * float())
        .def(py::self + py::self)
        .def(py::init<vector3d<double>>());

    py::class_<matrix3d<double>>(m, "matrix3d")
        .def(py::init<std::vector<std::vector<double>>>())
        .def(py::init<>())
        .def("__call__", [](const matrix3d<double>& obj, int x, int y) {
            return obj(x, y);
        })
        .def("__array__", [](const matrix3d<double>& mat) {
            return py::array_t<double>({3, 3},
                                       {3 * sizeof(double), sizeof(double)},
                                       &mat(0, 0));
        }, py::return_value_policy::reference_internal)
        .def(py::self * py::self)
        .def("__getitem__", [](const matrix3d<double>& obj, int x, int y) {
            return obj(x, y);
        })
        .def("__mul__", [](const matrix3d<double>& obj, vector3d<double> const& b) {
            vector3d<double> res = obj * b;
            return res;
        })
        .def("__repr__", [](const matrix3d<double>& mat) {
            return show_mat(mat);
        })
        .def(py::init<matrix3d<double>>())
        .def("det", &matrix3d<double>::det);

    py::class_<Potential>(m, "Potential")
        .def(py::init<Simulation_context&>(), py::keep_alive<1, 2>(), "ctx"_a)
        .def("generate", &Potential::generate)
        .def("symmetrize", &Potential::symmetrize)
        .def("fft_transform", &Potential::fft_transform)
        .def("allocate", &Potential::allocate)
        .def("save", &Potential::save)
        .def("load", &Potential::load)
        .def("energy_vha", &Potential::energy_vha)
        .def("energy_vxc", &Potential::energy_vxc)
        .def("energy_exc", &Potential::energy_exc)
        .def("PAW_total_energy", &Potential::PAW_total_energy)
        .def("PAW_one_elec_energy", &Potential::PAW_one_elec_energy);

    py::class_<Density>(m, "Density")
        .def(py::init<Simulation_context&>(), py::keep_alive<1, 2>(), "ctx"_a)
        .def("initial_density", &Density::initial_density)
        .def("allocate", &Density::allocate)
        .def("check_num_electrons", &Density::check_num_electrons)
        .def("fft_transform", &Density::fft_transform)
        .def("mix", &Density::mix)
        .def("symmetrize", &Density::symmetrize)
        .def("symmetrize_density_matrix", &Density::symmetrize_density_matrix)
        .def("generate", &Density::generate)
        .def("generate_paw_loc_density", &Density::generate_paw_loc_density)
        .def("save", &Density::save)
        .def("check_num_electrons", &Density::check_num_electrons)
        .def("load", &Density::load);

    py::class_<Band>(m, "Band")
        .def(py::init<Simulation_context&>())
        .def("initialize_subspace", py::overload_cast<K_point_set&, Hamiltonian&>(&Band::initialize_subspace, py::const_))
        .def("solve", &Band::solve);

    py::class_<DFT_ground_state>(m, "DFT_ground_state")
        .def(py::init<K_point_set&>(), py::keep_alive<1, 2>())
        .def("print_info", &DFT_ground_state::print_info)
        .def("initial_state", &DFT_ground_state::initial_state)
        .def("print_magnetic_moment", &DFT_ground_state::print_magnetic_moment)
        .def("total_energy", &DFT_ground_state::total_energy)
        .def("density", &DFT_ground_state::density, py::return_value_policy::reference)
        .def("find", [](DFT_ground_state& dft, double potential_tol, double energy_tol, int num_dft_iter, bool write_state) {
            json js = dft.find(potential_tol, energy_tol, num_dft_iter, write_state);
            return pj_convert(js);
        }, "potential_tol"_a, "energy_tol"_a, "num_dft_iter"_a, "write_state"_a)
        .def("k_point_set", &DFT_ground_state::k_point_set, py::return_value_policy::reference_internal)
        .def("hamiltonian", &DFT_ground_state::hamiltonian, py::return_value_policy::reference_internal)
        .def("potential", &DFT_ground_state::potential, py::return_value_policy::reference_internal)
        .def("update", &DFT_ground_state::update);

    py::class_<K_point>(m, "K_point")
        .def("band_energy", py::overload_cast<int, int>(&K_point::band_energy))
        .def_property_readonly("vk", &K_point::vk, py::return_value_policy::copy)
        .def("generate_fv_states", &K_point::generate_fv_states)
        .def("set_band_energy", [](K_point& kpoint, int j, int ispn, double val) {
            kpoint.band_energy(j, ispn) = val;
        })
        .def("band_energies", [](K_point const& kpoint, int ispn) {
            std::vector<double> energies(kpoint.num_bands());
            for (int i = 0; i < kpoint.num_bands(); ++i) {
                energies[i] = kpoint.band_energy(i, ispn);
            }
            return energies;
        }, py::return_value_policy::copy)
        .def("band_occupancy", [](K_point const& kpoint, int ispn) {
            std::vector<double> occ(kpoint.num_bands());
            for (int i = 0; i < kpoint.num_bands(); ++i) {
                occ[i] = kpoint.band_occupancy(i, ispn);
            }
            return occ;
        })
        .def("gkvec_partition", &K_point::gkvec_partition, py::return_value_policy::reference_internal)
        .def("gkvec", &K_point::gkvec, py::return_value_policy::reference_internal)
        .def("fv_states", &K_point::fv_states, py::return_value_policy::reference_internal)
        .def("ctx", &K_point::ctx, py::return_value_policy::reference_internal)
        .def("weight", &K_point::weight)
        .def("spinor_wave_functions", &K_point::spinor_wave_functions, py::return_value_policy::reference_internal);

    py::class_<K_point_set>(m, "K_point_set")
        .def(py::init<Simulation_context&>(), py::keep_alive<1, 2>())
        .def(py::init<Simulation_context&, std::vector<vector3d<double>>>())
        .def(py::init<Simulation_context&, std::initializer_list<std::initializer_list<double>>>())
        .def(py::init<Simulation_context&, vector3d<int>, vector3d<int>, bool>())
        .def(py::init<Simulation_context&, std::vector<int>, std::vector<int>, bool>())
        .def("initialize", &K_point_set::initialize, py::arg("counts") = std::vector<int>{})
        .def("ctx", &K_point_set::ctx, py::return_value_policy::reference_internal)
        .def("unit_cell", &K_point_set::unit_cell, py::return_value_policy::reference_internal)
        .def("_num_kpoints", &K_point_set::num_kpoints)
        .def("size", [](K_point_set& ks) -> int {
            return ks.spl_num_kpoints().local_size();
        })
        .def("energy_fermi", &K_point_set::energy_fermi)
        .def("get_band_energies", &K_point_set::get_band_energies)
        .def("find_band_occupancies", &K_point_set::find_band_occupancies)
        .def("sync_band_energies", &K_point_set::sync_band_energies)
        .def("valence_eval_sum", &K_point_set::valence_eval_sum)
        .def("__contains__", [](K_point_set& ks, int i) {
            return (i >= 0 && i < ks.spl_num_kpoints().local_size());
        })
        .def("__getitem__", [](K_point_set& ks, int i) -> K_point& {
            return *ks[ks.spl_num_kpoints(i)];
        }, py::return_value_policy::reference_internal)
        .def("__len__", [](K_point_set const& ks) {
            return ks.spl_num_kpoints().local_size();
        })
        .def("add_kpoint", [](K_point_set& ks, std::vector<double> v, double weight) {
            ks.add_kpoint(v.data(), weight);
        })
        .def("add_kpoint", [](K_point_set& ks, vector3d<double>& v, double weight) {
            ks.add_kpoint(&v[0], weight);
        });

    py::class_<Hamiltonian>(m, "Hamiltonian")
        .def(py::init<Simulation_context&, Potential&>(), py::keep_alive<1, 2>())
        .def("potential", &Hamiltonian::potential, py::return_value_policy::reference_internal)
        .def("ctx", &Hamiltonian::ctx, py::return_value_policy::reference_internal)
        .def("on_gpu", [](Hamiltonian& hamiltonian) -> bool {
            const auto& ctx = hamiltonian.ctx();
            auto        pu  = ctx.processing_unit();
            if (pu == device_t::GPU) {
                return true;
            } else {
                return false;
            }
        })
        .def("apply", [](Hamiltonian& hamiltonian, K_point& kp, Wave_functions& wf) -> Wave_functions {
            auto&          gkvec_partition = wf.gkvec_partition();
            int            num_wf          = wf.num_wf();
            int            num_sc          = wf.num_sc();
            Wave_functions wf_out(gkvec_partition, num_wf, num_sc);
#ifdef __GPU
            if (hamiltonian.ctx().processing_unit() == GPU) {
                for (int ispn = 0; ispn < num_sc; ++ispn) {
                    wf_out.allocate_on_device(ispn);
                    if (!wf.pw_coeffs(0).prime().on_device()) {
                        for (int i = 0; i < num_sc; ++i) {
                            wf.pw_coeffs(i).allocate_on_device();
                            wf.pw_coeffs(i).copy_to_device(0, num_wf);
                        }
                    } else {
                        wf.copy_to_device(ispn, 0, num_wf);
                    }
                }
            }
            #endif
            /* apply H to all wave functions */
            int N = 0;
            int n = num_wf;
            if (n != hamiltonian.ctx().num_bands()) {
                throw std::runtime_error("num_wf != num_bands");
            }
            hamiltonian.local_op().prepare(hamiltonian.potential());
            if (!hamiltonian.ctx().gamma_point()) {
                hamiltonian.prepare<double_complex>();
            } else {
                hamiltonian.prepare<double>();
            }
            hamiltonian.local_op().prepare(kp.gkvec_partition());
            hamiltonian.ctx().fft_coarse().prepare(kp.gkvec_partition());
            kp.beta_projectors().prepare();
            if (!hamiltonian.ctx().gamma_point()) {
                for (int ispn = 0; ispn < num_sc; ++ispn)
                    hamiltonian.apply_h_s<complex_double>(&kp, ispn, N, n, wf, &wf_out, nullptr);
            } else {
                std::cout << "applying Hamiltonian at Gamma-point" << "\n";
                for (int ispn = 0; ispn < num_sc; ++ispn)
                    hamiltonian.apply_h_s<double>(&kp, ispn, N, n, wf, &wf_out, nullptr);
            }
            kp.beta_projectors().dismiss();
            if (!hamiltonian.ctx().full_potential()) {
                hamiltonian.dismiss();
            }
            hamiltonian.ctx().fft_coarse().dismiss();
            #ifdef __GPU
            if (hamiltonian.ctx().processing_unit() == GPU) {
                for (int ispn = 0; ispn < num_sc; ++ispn) {
                    wf_out.copy_to_host(ispn, 0, n);
                }
            }
            #endif // __GPU
            return wf_out;
        }, "kpoint"_a, "wf_in"_a)
        .def("apply_ref", [](Hamiltonian& hamiltonian, K_point& kp, Wave_functions& wf_out, Wave_functions& wf) {
            int num_wf = wf.num_wf();
            int num_sc = wf.num_sc();
            if (num_wf != wf_out.num_wf() || wf_out.num_sc() != num_sc) {
                throw std::runtime_error("Hamiltonian::apply_ref (python bindings): num_sc or num_wf do not match");
            }
            #ifdef __GPU
            if (hamiltonian.ctx().processing_unit() == GPU) {
                for (int ispn = 0; ispn < num_sc; ++ispn) {
                    wf_out.allocate_on_device(ispn);
                    if (!wf.pw_coeffs(ispn).prime().on_device()) {
                        wf.pw_coeffs(ispn).allocate_on_device();
                        wf.pw_coeffs(ispn).copy_to_device(0, num_wf);
                    } else {
                        wf.copy_to_device(ispn, 0, num_wf);
                    }
                }
            }
            #endif
            /* apply H to all wave functions */
            int N = 0;
            int n = num_wf;
            if (n != hamiltonian.ctx().num_bands()) {
                throw std::runtime_error("num_wf != num_bands");
            }
            hamiltonian.local_op().prepare(hamiltonian.potential());
            if (!hamiltonian.ctx().gamma_point()) {
                hamiltonian.prepare<double_complex>();
            } else {
                hamiltonian.prepare<double>();
            }
            hamiltonian.local_op().prepare(kp.gkvec_partition());
            hamiltonian.ctx().fft_coarse().prepare(kp.gkvec_partition());
            kp.beta_projectors().prepare();
            if (!hamiltonian.ctx().gamma_point()) {
                for (int ispn = 0; ispn < num_sc; ++ispn) {
                    hamiltonian.apply_h_s<complex_double>(&kp, ispn, N, n, wf, &wf_out, nullptr);
                }
            } else {
                std::cerr << "python module:: H applied at gamma-point\n";
                for (int ispn = 0; ispn < num_sc; ++ispn) {
                    hamiltonian.apply_h_s<double>(&kp, ispn, N, n, wf, &wf_out, nullptr);
                }
            }
            kp.beta_projectors().dismiss();
            hamiltonian.local_op().dismiss();
            hamiltonian.ctx().fft_coarse().dismiss();
            if (!hamiltonian.ctx().full_potential()) {
                hamiltonian.dismiss();
            }
            #ifdef __GPU
            if (hamiltonian.ctx().processing_unit() == GPU) {
                for (int ispn = 0; ispn < num_sc; ++ispn)
                    wf_out.copy_to_host(ispn, 0, n);
            }
            #endif // __GPU
        }, "kpoint"_a, "wf_out"_a, "wf_in"_a)
        .def("apply_ref_single", [](Hamiltonian& hamiltonian, K_point& kp, int ispn, Wave_functions& wf_out, Wave_functions& wf) {
            int num_wf = wf.num_wf();
            int num_sc = wf.num_sc();
            if (num_wf != wf_out.num_wf() || wf_out.num_sc() != num_sc) {
                throw std::runtime_error("Hamiltonian::apply_ref (python bindings): num_sc or num_wf do not match");
            }
#ifdef __GPU
            if (hamiltonian.ctx().processing_unit() == GPU) {
                std::cerr << "WFCT: copy host -> device\n";
                wf_out.allocate_on_device(ispn);
                if (!wf.pw_coeffs(ispn).prime().on_device()) {
                    wf.pw_coeffs(ispn).allocate_on_device();
                    wf.pw_coeffs(ispn).copy_to_device(0, num_wf);
                } else {
                    wf.copy_to_device(ispn, 0, num_wf);
                }
            }
#endif
            /* apply H to all wave functions */
            int N = 0;
            int n = num_wf;
            if (n != hamiltonian.ctx().num_bands()) {
                throw std::runtime_error("num_wf != num_bands");
            }
            hamiltonian.local_op().prepare(hamiltonian.potential());
            if (!hamiltonian.ctx().gamma_point()) {
                hamiltonian.prepare<double_complex>();
            } else {
                hamiltonian.prepare<double>();
            }
            hamiltonian.local_op().prepare(kp.gkvec_partition());
            hamiltonian.ctx().fft_coarse().prepare(kp.gkvec_partition());
            kp.beta_projectors().prepare();
            if (!hamiltonian.ctx().gamma_point()) {
                hamiltonian.apply_h_s<complex_double>(&kp, ispn, N, n, wf, &wf_out, nullptr);
            } else {
                hamiltonian.apply_h_s<double>(&kp, ispn, N, n, wf, &wf_out, nullptr);
            }
            kp.beta_projectors().dismiss();
            hamiltonian.local_op().dismiss();
            hamiltonian.ctx().fft_coarse().dismiss();
            if (!hamiltonian.ctx().full_potential()) {
                hamiltonian.dismiss();
            }
#ifdef __GPU
            if (hamiltonian.ctx().processing_unit() == GPU) {
                wf_out.copy_to_host(ispn, 0, n);
            }
#endif // __GPU
        }, "kpoint"_a, "ispn"_a, "wf_out"_a, "wf_in"_a);

    py::class_<Stress>(m, "Stress")
        .def(py::init<Simulation_context&, Density&, Potential&, Hamiltonian&, K_point_set&>())
        .def("calc_stress_total", &Stress::calc_stress_total, py::return_value_policy::reference_internal)
        .def("print_info", &Stress::print_info);

py::class_<Free_atom>(m, "Free_atom")
    .def(py::init<std::string>())
    .def(py::init<int>())
    .def("ground_state", [](Free_atom& atom, double energy_tol, double charge_tol, bool rel)
                         {
                             json js = atom.ground_state(energy_tol, charge_tol, rel);
                             return pj_convert(js);
                         })
    .def("radial_grid_points", &Free_atom::radial_grid_points)
    .def("num_atomic_levels", &Free_atom::num_atomic_levels)
    .def("atomic_level", [](Free_atom& atom, int idx)
                         {
                            auto level = atom.atomic_level(idx);
                            json js;
                            js["n"]         = level.n;
                            js["l"]         = level.l;
                            js["k"]         = level.k;
                            js["occupancy"] = level.occupancy;
                            js["energy"]    = atom.atomic_level_energy(idx);
                            return pj_convert(js);
                         })
    .def("free_atom_electronic_potential", [](Free_atom& atom)
                                           {
                                               return atom.free_atom_electronic_potential();
                                           })
    .def("free_atom_wave_function", [](Free_atom& atom, int idx)
                                    {
                                        return atom.free_atom_wave_function(idx);
                                    })
    .def("free_atom_wave_function_x", [](Free_atom& atom, int idx)
                                      {
                                          return atom.free_atom_wave_function_x(idx);
                                      })
    .def("free_atom_wave_function_x_deriv", [](Free_atom& atom, int idx)
                                            {
                                                return atom.free_atom_wave_function_x_deriv(idx);
                                            })
    .def("free_atom_wave_function_residual", [](Free_atom& atom, int idx)
                                             {
                                                 return atom.free_atom_wave_function_residual(idx);
                                             });

    py::class_<Force>(m, "Force")
        .def(py::init<Simulation_context&, Density&, Potential&, Hamiltonian&, K_point_set&>())
        .def("calc_forces_total", &Force::calc_forces_total, py::return_value_policy::reference_internal)
        .def("print_info", &Force::print_info);

    py::class_<matrix_storage_slab<complex_double>>(m, "MatrixStorageSlabC")
        .def("is_remapped", &matrix_storage_slab<complex_double>::is_remapped)
        .def("prime", py::overload_cast<>(&matrix_storage_slab<complex_double>::prime), py::return_value_policy::reference_internal);

    py::class_<mdarray<complex_double, 2>>(m, "mdarray2c")
        .def("on_device", &mdarray<complex_double, 2>::on_device)
        .def("copy_to_host", [](mdarray<complex_double, 2>& mdarray) {
            mdarray.copy<memory_t::device, memory_t::host>(mdarray.size(1));
        })
        .def("__array__", [](py::object& obj) {
            mdarray<complex_double, 2>& arr = obj.cast<mdarray<complex_double, 2>&>();
            int nrows = arr.size(0);
            int ncols = arr.size(1);
            return py::array_t<complex_double>({nrows, ncols},
                                               {1 * sizeof(complex_double), nrows * sizeof(complex_double)},
                                               arr.data<CPU>(), obj);
        });

    py::class_<dmatrix<complex_double>, mdarray<complex_double,2>>(m, "dmatrix");

    py::enum_<sddk::device_t>(m, "DeviceEnum")
        .value("CPU", sddk::device_t::CPU)
        .value("GPU", sddk::device_t::GPU);

    py::class_<Wave_functions>(m, "Wave_functions")
        .def(py::init<Gvec_partition const&, int, int>(), "gvecp"_a, "num_wf"_a, "num_sc"_a)
        .def("num_sc", &Wave_functions::num_sc)
        .def("num_wf", &Wave_functions::num_wf)
        .def("has_mt", &Wave_functions::has_mt)
        .def("zero_pw", &Wave_functions::zero_pw)
        .def("pw_coeffs", [](py::object& obj, int i) -> py::array_t<complex_double> {
            Wave_functions& wf = obj.cast<Wave_functions&>();
            // coefficients are _always_ (i.e. usually ;) ) on GPU; copy to host
            // TODO: delete this, copy to host is done immediately after apply, copy below not needed
            // #ifdef __GPU
            // bool is_on_device = wf.pw_coeffs(0).prime().on_device();
            // if (is_on_device) {
            //     /* on device, assume this is primary storage ... */
            //     for (int ispn = 0; ispn < wf.num_sc(); ispn++) {
            //         wf.copy_to_host(ispn, 0, wf.num_wf());
            //     }
            // }
            // #endif // __GPU
            auto& matrix_storage = wf.pw_coeffs(i);
            int   nrows          = matrix_storage.prime().size(0);
            int   ncols          = matrix_storage.prime().size(1);
            /* return underlying data as numpy.ndarray view, e.g. non-copying */
            /* TODO this might be a distributed array, should/can we use dask? */
            return py::array_t<complex_double>({nrows, ncols},
                                               {1 * sizeof(complex_double), nrows * sizeof(complex_double)},
                                               matrix_storage.prime().data<CPU>(),
                                               obj);
        },
             py::keep_alive<0, 1>())
#ifdef __GPU
        .def("copy_to_gpu", [](Wave_functions& wf) {
            /* is_on_device -> true if all internal storage is allocated on device */
            bool is_on_device = true;
            for (int i = 0; i < wf.num_sc(); ++i) {
                is_on_device = is_on_device && wf.pw_coeffs(i).prime().on_device();
            }
            if (!is_on_device) {
                for (int ispn = 0; ispn < wf.num_sc(); ispn++) {
                    wf.pw_coeffs(ispn).prime().allocate(memory_t::device);
                }
            }
            for (int ispn = 0; ispn < wf.num_sc(); ispn++) {
                wf.copy_to_device(ispn, 0, wf.num_wf());
            }
        })
        .def("copy_to_cpu", [](Wave_functions& wf) {
            /* is_on_device -> true if all internal storage is allocated on device */
            bool is_on_device = true;
            for (int i = 0; i < wf.num_sc(); ++i) {
                is_on_device = is_on_device && wf.pw_coeffs(i).prime().on_device();
            }
            if (!is_on_device) {
            } else {
                for (int ispn = 0; ispn < wf.num_sc(); ispn++) {
                    wf.copy_to_host(ispn, 0, wf.num_wf());
                }
            }
        })
#endif // __GPU
        .def("allocated_on_device", [](Wave_functions& wf) {
            bool is_on_device = true;
            for (int i = 0; i < wf.num_sc(); ++i) {
                is_on_device = is_on_device && wf.pw_coeffs(i).prime().on_device();
            }
            return is_on_device;
        })
        .def("pw_coeffs_obj", py::overload_cast<int>(&Wave_functions::pw_coeffs, py::const_), py::return_value_policy::reference_internal);


    /* TODO: group this kind of functions somewhere */
    m.def("ewald_energy", &ewald_energy);
    m.def("energy_bxc", &energy_bxc);
    m.def("omp_set_num_threads", &omp_set_num_threads);
    m.def("omp_get_num_threads", &omp_get_num_threads);
    // m.def("pseudopotential_hmatrix", &pseudopotential_hmatrix<complex_double>, "kpoint"_a, "ispn"_a, "H"_a);
}
