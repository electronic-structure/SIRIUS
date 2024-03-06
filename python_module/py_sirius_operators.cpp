#include "python_module_includes.hpp"
#include <string>
#include <iomanip>
#include <complex>
#include <sstream>
#include <cstdio>

using namespace pybind11::literals;
namespace py = pybind11;
using namespace sirius;
using json = nlohmann::json;
using nlohmann::basic_json;
using complex_double = std::complex<double>;

using PT = double;

void
init_operators(py::module& m)
{
    py::class_<Non_local_operator<PT>>(m, "Non_local_operator")
            .def("get_matrix", &Non_local_operator<PT>::get_matrix<std::complex<double>>);
    py::class_<D_operator<PT>, Non_local_operator<PT>>(m, "D_operator");
    py::class_<Q_operator<PT>, Non_local_operator<PT>>(m, "Q_operator");

    py::class_<Hamiltonian0<PT>>(m, "Hamiltonian0")
            .def(py::init<Potential&, bool>(), py::keep_alive<1, 2>(), "Potential"_a,
                 py::arg("precompute_lapw") = false)
            .def("Q", &Hamiltonian0<PT>::Q, py::return_value_policy::reference_internal)
            .def("D", &Hamiltonian0<PT>::D, py::return_value_policy::reference_internal)
            .def("Hk", &Hamiltonian0<PT>::operator(), py::keep_alive<0, 1>())
            .def("potential", &Hamiltonian0<PT>::potential, py::return_value_policy::reference_internal);

    py::class_<U_operator<PT>>(m, "U_operator")
            .def("mat", &U_operator<PT>::mat, py::return_value_policy::reference_internal, py::keep_alive<0, 1>());

    py::class_<Hamiltonian_k<PT>>(m, "Hamiltonian_k")
            .def(py::init<Hamiltonian0<PT>&, K_point<PT>&>(), py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
            .def_property_readonly("U", &Hamiltonian_k<PT>::U, py::return_value_policy::reference_internal);

    py::class_<S_k<complex_double>>(m, "S_k")
            .def(py::init<Simulation_context&, const Q_operator<PT>&, const Beta_projectors_base<PT>&, int>(),
                 py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), py::keep_alive<1, 4>())
            .def_property_readonly("size", &S_k<complex_double>::size)
            .def("apply", [](py::object& obj, py::array_t<complex_double>& X) {
                using class_t = S_k<complex_double>;
                class_t& sk   = obj.cast<class_t&>();

                if (X.strides(0) != sizeof(complex_double)) {
                    char msg[256];
                    std::sprintf(msg, "invalid stride: [%ld, %ld] in %s:%d", X.strides(0), X.strides(1), __FILE__,
                                 __LINE__);
                    RTE_THROW(msg);
                }
                if (X.ndim() != 2) {
                    char msg[256];
                    std::sprintf(msg, "wrong dimension: %lu in %s:%d", X.ndim(), __FILE__, __LINE__);
                    RTE_THROW(msg);
                }
                auto ptr = X.request().ptr;
                int rows = X.shape(0);
                int cols = X.shape(1);
                const mdarray<complex_double, 2> array({rows, cols}, static_cast<complex_double*>(ptr));
                return sk.apply(array, memory_t::host);
            });

    py::class_<InverseS_k<complex_double>>(m, "InverseS_k")
            .def(py::init<Simulation_context&, const Q_operator<PT>&, const Beta_projectors_base<PT>&, int>(),
                 py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), py::keep_alive<1, 4>())
            .def_property_readonly("size", &InverseS_k<complex_double>::size)
            .def("apply", [](py::object& obj, py::array_t<complex_double>& X) {
                using class_t       = InverseS_k<complex_double>;
                class_t& inverse_sk = obj.cast<class_t&>();

                if (X.strides(0) != sizeof(complex_double)) {
                    char msg[256];
                    std::sprintf(msg, "invalid stride: [%ld, %ld] in %s:%d", X.strides(0), X.strides(1), __FILE__,
                                 __LINE__);
                    RTE_THROW(msg);
                }
                if (X.ndim() != 2) {
                    char msg[256];
                    std::sprintf(msg, "wrong dimension: %lu in %s:%d", X.ndim(), __FILE__, __LINE__);
                    RTE_THROW(msg);
                }
                auto ptr = X.request().ptr;
                int rows = X.shape(0);
                int cols = X.shape(1);
                const mdarray<complex_double, 2> array({rows, cols}, reinterpret_cast<complex_double*>(ptr));
                return inverse_sk.apply(array, memory_t::host);
            });

    py::class_<Ultrasoft_preconditioner<complex_double>>(m, "Precond_us")
            .def(py::init<Simulation_context&, const Q_operator<PT>&, int, const Beta_projectors_base<PT>&,
                          const fft::Gvec&>(),
                 py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), py::keep_alive<1, 5>(), py::keep_alive<1, 6>())
            .def_property_readonly("size", &Ultrasoft_preconditioner<complex_double>::size)
            .def("apply", [](py::object& obj, py::array_t<complex_double>& X) {
                using class_t    = Ultrasoft_preconditioner<complex_double>;
                class_t& precond = obj.cast<class_t&>();
                if (X.strides(0) != sizeof(complex_double)) {
                    char msg[256];
                    std::sprintf(msg, "invalid stride: [%ld, %ld] in %s:%d", X.strides(0), X.strides(1), __FILE__,
                                 __LINE__);
                    RTE_THROW(msg);
                }
                if (X.ndim() != 2) {
                    char msg[256];
                    std::sprintf(msg, "wrong dimension: %lu in %s:%d", X.ndim(), __FILE__, __LINE__);
                    RTE_THROW(msg);
                }
                auto ptr = X.request().ptr;
                int rows = X.shape(0);
                int cols = X.shape(1);
                const mdarray<complex_double, 2> array({rows, cols}, static_cast<complex_double*>(ptr));
                return precond.apply(array, memory_t::host);
            });

    py::class_<beta_chunk_t>(m, "beta_chunk")
            .def_readonly("num_beta", &beta_chunk_t::num_beta_)
            .def_readonly("num_atoms", &beta_chunk_t::num_atoms_)
            .def_readonly("desc", &beta_chunk_t::desc_, py::return_value_policy::reference_internal)
            .def("__repr__",
                 [](const beta_chunk_t& obj) {
                     std::stringstream buffer;
                     for (int ia = 0; ia < obj.num_atoms_; ++ia) {
                         buffer << "\t atom ia = " << ia << " ";
                         buffer << "\t nbf     : " << std::setw(10)
                                << obj.desc_(static_cast<int>(beta_desc_idx::nbf), ia) << " ";
                         buffer << "\t offset  : " << std::setw(10)
                                << obj.desc_(static_cast<int>(beta_desc_idx::offset), ia) << " ";
                         buffer << "\t offset_t: " << std::setw(10)
                                << obj.desc_(static_cast<int>(beta_desc_idx::offset_t), ia) << " ";
                         buffer << "\t ja      : " << std::setw(10)
                                << obj.desc_(static_cast<int>(beta_desc_idx::ia), ia) << " ";
                         buffer << "\n";
                     }

                     return "num_beta: " + std::to_string(obj.num_beta_) + "\n" +
                            "offset: " + std::to_string(obj.offset_) +
                            "\n"
                            "num_atoms: " +
                            std::to_string(obj.num_atoms_) + "\n" + "desc:\n" + buffer.str();
                 })
            .def_readonly("offset", &beta_chunk_t::offset_);

    py::class_<beta_projectors_coeffs_t<PT>>(m, "beta_projector_coeffs")
            .def_readonly("a", &beta_projectors_coeffs_t<PT>::pw_coeffs_a_, py::return_value_policy::reference_internal)
            .def_readonly("chunk", &beta_projectors_coeffs_t<PT>::beta_chunk_,
                          py::return_value_policy::reference_internal);

    py::class_<Beta_projector_generator<PT>>(m, "Beta_projector_generator")
            .def("prepare", &Beta_projector_generator<PT>::prepare, py::keep_alive<1, 0>())
            .def("generate", py::overload_cast<beta_projectors_coeffs_t<PT>&, int>(
                                     &Beta_projector_generator<PT>::generate, py::const_))
            .def("generate_j", py::overload_cast<beta_projectors_coeffs_t<PT>&, int, int>(
                                       &Beta_projector_generator<PT>::generate, py::const_));

    py::class_<Beta_projectors_base<PT>>(m, "Beta_projectors_base")
            .def_property_readonly("num_chunks", &Beta_projectors_base<PT>::num_chunks)
            .def("make_generator", py::overload_cast<device_t>(&Beta_projectors_base<PT>::make_generator, py::const_),
                 py::keep_alive<1, 0>());

    py::class_<Beta_projectors<PT>, Beta_projectors_base<PT>>(m, "Beta_projectors");
    m.def("apply_U_operator", &apply_U_operator<double>, py::arg("ctx"), py::arg("spin_range"), py::arg("band_range"),
          py::arg("hub_wf"), py::arg("phi"), py::arg("u_op"), py::arg("hphi"));

    py::class_<wf::band_range>(m, "band_range")
            .def(py::init<int, int>())
            .def("__len__", &wf::band_range::size)
            .def("__getitem__", [](const wf::band_range& br, int i) {
                if (i >= br.size()) {
                    throw pybind11::index_error("out of bounds");
                }
                return br.begin() + i;
            });

    py::class_<wf::spin_range>(m, "spin_range")
            .def(py::init<int, int>())
            .def("__getitem__",
                 [](const wf::spin_range& sp, int i) {
                     if (i >= sp.size()) {
                         throw pybind11::index_error("out of bounds");
                     }
                     auto index = sp.begin();
                     for (int k = 0; k < i; ++k) {
                         index++;
                     }
                     return static_cast<int>(index);
                 })
            .def("__len__", &wf::spin_range::size);
}
