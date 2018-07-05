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
#include "utils/json.hpp"
#include "Unit_cell/free_atom.hpp"

using namespace pybind11::literals; // to bring in the `_a` literal
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
    sirius::initialize();
    auto atexit = py::module::import("atexit");
    atexit.attr("register")(py::cpp_function([]() { sirius::finalize(); }));

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

    py::class_<Simulation_parameters>(m, "Simulation_parameters")
        .def(py::init<>())
        .def("pw_cutoff", &Simulation_parameters::pw_cutoff)
        .def("parameters_input", py::overload_cast<>(&Simulation_parameters::parameters_input, py::const_), py::return_value_policy::reference)
        .def("num_spin_dims", &Simulation_parameters::num_spin_dims)
        .def("num_mag_dims", &Simulation_parameters::num_mag_dims)
        .def("set_gamma_point", &Simulation_parameters::set_gamma_point)
        .def("set_pw_cutoff", &Simulation_parameters::set_pw_cutoff)
        .def("set_iterative_solver_tolerance", &Simulation_parameters::set_iterative_solver_tolerance);

    py::class_<Simulation_context_base, Simulation_parameters>(m, "Simulation_context_base");

    py::class_<Simulation_context, Simulation_context_base>(m, "Simulation_context")
        .def(py::init<>())
        .def(py::init<std::string const&>())
        .def("initialize", &Simulation_context::initialize)
        .def("num_bands", py::overload_cast<>(&Simulation_context::num_bands, py::const_))
        .def("num_bands", py::overload_cast<int>(&Simulation_context::num_bands))
        .def("set_verbosity", &Simulation_context::set_verbosity)
        .def("create_storage_file", &Simulation_context::create_storage_file)
        .def("processing_unit", &Simulation_context::processing_unit)
        .def("set_processing_unit", py::overload_cast<device_t>(&Simulation_context::set_processing_unit))
        .def("gvec", &Simulation_context::gvec)
        .def("fft", &Simulation_context::fft)
        .def("unit_cell", py::overload_cast<>(&Simulation_context::unit_cell, py::const_), py::return_value_policy::reference);

    py::class_<Unit_cell>(m, "Unit_cell")
        .def("add_atom_type", static_cast<void (Unit_cell::*)(const std::string, const std::string)>(&Unit_cell::add_atom_type))
        .def("add_atom", py::overload_cast<const std::string, std::vector<double>>(&Unit_cell::add_atom))
        .def("atom_type", py::overload_cast<int>(&Unit_cell::atom_type), py::return_value_policy::reference)
        .def("set_lattice_vectors", static_cast<void (Unit_cell::*)(matrix3d<double>)>(&Unit_cell::set_lattice_vectors))
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

    py::class_<vector3d<int>>(m, "vector3d_int")
        .def(py::init<std::vector<int>>())
        .def("__call__", [](const vector3d<int>& obj, int x) {
            return obj[x];
        })
        .def("__repr__", [](const vector3d<int>& vec) {
            return show_vec(vec);
        })
        .def(py::init<vector3d<int>>());

    py::class_<vector3d<double>>(m, "vector3d_double")
        .def(py::init<std::vector<double>>())
        .def("__call__", [](const vector3d<double>& obj, int x) {
            return obj[x];
        })
        .def("__repr__", [](const vector3d<double>& vec) {
            return show_vec(vec);
        })
        .def("length", &vector3d<double>::length)
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
        .def(py::init<Simulation_context&>())
        .def("generate", &Potential::generate)
        .def("allocate", &Potential::allocate)
        .def("save", &Potential::save)
        .def("load", &Potential::load);

    py::class_<Density>(m, "Density")
        .def(py::init<Simulation_context&>())
        .def("initial_density", &Density::initial_density)
        .def("allocate", &Density::allocate)
        .def("save", &Density::save)
        .def("load", &Density::load);

    py::class_<Band>(m, "Band")
        .def(py::init<Simulation_context&>())
        .def("initialize_subspace", py::overload_cast<K_point_set&, Hamiltonian&>(&Band::initialize_subspace, py::const_))
        .def("solve", &Band::solve);

    py::class_<DFT_ground_state>(m, "DFT_ground_state")
        .def(py::init<K_point_set&>())
        .def("print_info", &DFT_ground_state::print_info)
        .def("initial_state", &DFT_ground_state::initial_state)
        .def("print_magnetic_moment", &DFT_ground_state::print_magnetic_moment)
        .def("total_energy", &DFT_ground_state::total_energy)
        .def("density", &DFT_ground_state::density, py::return_value_policy::reference)
        .def("find", [](DFT_ground_state& dft, double potential_tol, double energy_tol, int num_dft_iter, bool write_state)
                    {
                        json js = dft.find(potential_tol, energy_tol, num_dft_iter, write_state);
                        return pj_convert(js);
                    })
        .def("k_point_set", &DFT_ground_state::k_point_set, py::return_value_policy::reference_internal)
        .def("hamiltonian", &DFT_ground_state::hamiltonian, py::return_value_policy::reference)
        .def("potential",   &DFT_ground_state::potential, py::return_value_policy::reference);

    py::class_<K_point>(m, "K_point")
        .def("band_energy", py::overload_cast<int, int>(&K_point::band_energy))
        .def("vk", &K_point::vk, py::return_value_policy::reference)
        .def("generate_fv_states", &K_point::generate_fv_states)
        .def("fv_states", &K_point::fv_states, py::return_value_policy::reference_internal)
        .def("spinor_wave_functions", &K_point::spinor_wave_functions, py::return_value_policy::reference_internal);
    // .def("hubbard_wave_functions", &K_point::hubbard_wave_functions, py::return_value_policy::reference_internal)

    py::class_<K_point_set>(m, "K_point_set")
        .def(py::init<Simulation_context&>())
        .def(py::init<Simulation_context&, std::vector<vector3d<double>>>())
        .def(py::init<Simulation_context&, vector3d<int>, vector3d<int>, bool>())
        .def(py::init<Simulation_context&, std::vector<int>, std::vector<int>, bool>())
        .def("initialize", py::overload_cast<>(&K_point_set::initialize))
        .def("num_kpoints", &K_point_set::num_kpoints)
        .def("energy_fermi", &K_point_set::energy_fermi)
        .def("get_band_energies", &K_point_set::get_band_energies, py::return_value_policy::reference)
        .def("sync_band_energies", &K_point_set::sync_band_energies)
        .def("__getitem__", [](K_point_set& ks, int i) -> K_point& {
            if (ks[i] == nullptr) {
                throw std::runtime_error("oops");
            }
            return *ks[i];
        },
             py::return_value_policy::reference_internal)
        .def("add_kpoint", [](K_point_set& ks, std::vector<double> v, double weight) {
            ks.add_kpoint(v.data(), weight);
        })
        .def("add_kpoint", [](K_point_set& ks, vector3d<double>& v, double weight) {
            ks.add_kpoint(&v[0], weight);
        });

    py::class_<Hamiltonian>(m, "Hamiltonian")
        .def(py::init<Simulation_context&, Potential&>())
        .def("apply", [](Hamiltonian& hamiltonian, K_point& kp, int ispn, Wave_functions& wf) -> Wave_functions {
            auto&          gkvec_partition = wf.gkvec_partition();
            int            num_wf           = wf.num_wf();
            int            num_sc           = wf.num_sc();
            Wave_functions wf_out(gkvec_partition, num_wf, num_sc);
            #ifdef __GPU
            if(hamiltonian.ctx().processing_unit() == GPU) {
                for (int i = 0; i < num_sc; ++i) {
                    wf_out.pw_coeffs(i).allocate_on_device();
                }
                if(!wf.pw_coeffs(0).prime().on_device()) {
                    for (int i = 0; i < num_sc; ++i) {
                        wf.pw_coeffs(i).allocate_on_device();
                        wf.pw_coeffs(i).copy_to_device(0, num_wf);
                    }
                } else {
                    // copy input wf to device (assuming it is located on CPU)
                    for (int i = 0; i < num_sc; ++i) {
                        wf.pw_coeffs(i).copy_to_device(0, num_wf);
                    }
                }
            }
            #endif

            /* apply H to all wave functions */
            int N = 0;
            int n = num_wf;
            auto& ctx = hamiltonian.ctx();
            // hamiltonian.local_op().dismiss();
            hamiltonian.local_op().prepare(hamiltonian.potential());
            hamiltonian.local_op().prepare(kp.gkvec_partition());
            hamiltonian.ctx().fft_coarse().prepare(kp.gkvec_partition());
            hamiltonian.prepare<double_complex>();
            hamiltonian.apply_h_s<complex_double>(&kp, ispn, N, n, wf, &wf_out, nullptr);
            hamiltonian.dismiss();
            return wf_out;
        });

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
    // py::class_<matrix_storage_slab<double>>(m, "MatrixStorageSlabD", py::buffer_protocol())
    //     .def_buffer([](matrix_storage_slab<double>& matrix_storage) -> py::buffer_info {
    //         // Fortran storage order
    //         int nrows = matrix_storage.prime().size(0);
    //         int ncols = matrix_storage.prime().size(1);
    //         return py::buffer_info(
    //             (void*)matrix_storage.prime().data<CPU>(),
    //             sizeof(double),
    //             py::format_descriptor<double>::format(),
    //             2,
    //             {nrows, ncols},
    //             {sizeof(double), sizeof(double) * nrows});
    //     });

    py::class_<matrix_storage_slab<complex_double>>(m, "MatrixStorageSlabC")
        .def("is_remapped", &matrix_storage_slab<complex_double>::is_remapped)
        .def("prime", py::overload_cast<>(&matrix_storage_slab<complex_double>::prime), py::return_value_policy::reference_internal);
        // .def_buffer([](matrix_storage_slab<complex_double>& matrix_storage) -> py::array {
        //     // Fortran storage order
        //     int nrows = matrix_storage.prime().size(0);
        //     int ncols = matrix_storage.prime().size(1);
        //     return py::array(py::buffer_info((void*)matrix_storage.prime().data<CPU>(),
        //                                      sizeof(complex_double),
        //                                      py::format_descriptor<complex_double>::format(),
        //                                      2,
        //                                      {nrows, ncols},
        //                                      {sizeof(complex_double), sizeof(complex_double) * nrows}));
        // });

    py::class_<mdarray<complex_double,2>>(m, "mdarray2c")
        .def("on_device", &mdarray<complex_double,2>::on_device);

    py::enum_<sddk::device_t>(m, "DeviceEnum")
        .value("CPU", sddk::device_t::CPU)
        .value("GPU", sddk::device_t::GPU);

    py::class_<Wave_functions>(m, "Wave_functions")
        .def("num_sc", &Wave_functions::num_sc)
        .def("num_wf", &Wave_functions::num_wf)
        .def("has_mt", &Wave_functions::has_mt)
        .def("pw_coeffs", [](py::object& obj, int i) -> py::array_t<complex_double> {
            Wave_functions& wf = obj.cast<Wave_functions&>();
            // coefficients are _always_ (i.e. usually ;) ) on GPU; copy to host
            bool is_on_device = wf.pw_coeffs(0).prime().on_device();
            if (is_on_device) {
                /* on device, assume this is primary storage ... */
                for (int ispn = 0; ispn < wf.num_sc(); ispn++) {
                    wf.copy_to_host(ispn, 0, wf.num_wf());
                }
            }
            auto& matrix_storage = wf.pw_coeffs(i);
            int   nrows          = matrix_storage.prime().size(0);
            int   ncols          = matrix_storage.prime().size(1);
            /* return underlying data as numpy.ndarray view, e.g. non-copying */
            /* TODO this might be a distributed array, should/can we use dask? */
            return py::array_t<complex_double>({nrows, ncols},
                                               {1 * sizeof(complex_double), nrows * sizeof(complex_double)},
                                               matrix_storage.prime().data<CPU>(),
                                               obj);
        })
        .def("copy_to_gpu", [](Wave_functions& wf) {
            /* is_on_device -> true if all internal storage is allocated on device */
            bool is_on_device = true;
            for (int i = 0; i < wf.num_sc(); ++i) {
                is_on_device = is_on_device && wf.pw_coeffs(i).prime().on_device();
            }
            if (!is_on_device) {
                for (int ispn = 0; ispn < wf.num_sc(); ispn++) {
                    std::cerr << "allocating storage for spin-component " << ispn << "\n";
                    wf.pw_coeffs(ispn).prime().allocate(memory_t::device);
                }
            } else {
                std::cerr << "wave function is already on device, possibly loosing data...\n";
            }

            for (int ispn = 0; ispn < wf.num_sc(); ispn++) {
                std::cerr << "wave function copying sc=" << ispn << " to device...\n";
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
                std::cerr << "not on device" << "\n";
            } else {
                for (int ispn = 0; ispn < wf.num_sc(); ispn++) {
                    std::cerr << "wave function copying sc=" << ispn << " to device...\n";
                    wf.copy_to_host(ispn, 0, wf.num_wf());
                }
            }
        })
        .def("on_device", [](Wave_functions& wf) {
            bool is_on_device = true;
            for (int i = 0; i < wf.num_sc(); ++i) {
                is_on_device = is_on_device && wf.pw_coeffs(i).prime().on_device();
            }
            return is_on_device;
        })
        .def("pw_coeffs_obj", py::overload_cast<int>(&Wave_functions::pw_coeffs, py::const_), py::return_value_policy::reference_internal);

    m.def("wf_inner", [](device_t pu, int ispn, Wave_functions& bra, int i0, int m, Wave_functions& ket, int j0, int n) {
        dmatrix<complex_double> S(m, n, pu == device_t::GPU ? memory_t::device | memory_t::host : memory_t::host);
        /* S holds the result in the CPU pointer */
        inner(pu, ispn, bra, i0, m, ket, j0, n, S, 0, 0);

        return py::array_t<complex_double>({m, n},
                                           {1 * sizeof(complex_double), m * sizeof(complex_double)},
                                           S.data<CPU>());
    });
}
