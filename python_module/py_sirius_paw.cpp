#include "memory.hpp"
#include "python_module_includes.hpp"
#include "sht/gaunt.hpp"
#include "sht/sht.hpp"
#include "sht/gaunt.hpp"

#include <memory>

using complex_double = std::complex<double>;
using namespace sddk;
using namespace sirius;
using namespace sirius::sht;

using PT = double;

using gaunt_t     = Gaunt_coefficients<double>;
using gauntl3_t   = gaunt_L3<double>;
using gauntl1l2_t = gaunt_L1_L2<double>;

void
init_paw_util(py::module& m)
{
    py::class_<SHT>(m, "SHT").def(py::init<device_t, int, int>());

    py::class_<gaunt_t>(m, "Gaunt_coefficients_rrr")
        .def(py::init([](int lmax1, int lmax2, int lmax3) {
            return std::make_unique<Gaunt_coefficients<double>>(lmax1, lmax2, lmax3, SHT::gaunt_rrr);
        }))
        .def("num_gaunt_lm3", py::overload_cast<int>(&Gaunt_coefficients<double>::num_gaunt, py::const_))
        .def("num_gaunt3", py::overload_cast<int, int>(&Gaunt_coefficients<double>::num_gaunt, py::const_))
        .def("gaunt", [](gaunt_t& obj, int lm1, int lm2) -> std::vector<gauntl3_t> {
            int num = obj.num_gaunt(lm1, lm2);
            std::vector<gauntl3_t> coeffs(num);
            for (int i = 0; i < num; ++i) {
                coeffs[i] = obj.gaunt(lm1, lm2, i);
            }
            return coeffs;
        });
    // .def(py::init<int, int, int, std::function<double(int, int, int, int, int, int)>>, py::arg("lmax1"),
    //      py::arg("lmax2"), py::arg("lmax3"), py::arg("fct") = SHT::gaunt_rrr);

    py::class_<gaunt_L3<double>>(m, "gaunt_L3")
        .def_readonly("lm3", &gaunt_L3<double>::lm3)
        .def_readonly("l3", &gaunt_L3<double>::l3)
        .def_readonly("coeff", &gaunt_L3<double>::coef);

    py::class_<gaunt_L1_L2<double>>(m, "gaunt_L1_L2")
        .def_readonly("lm1", &gaunt_L1_L2<double>::lm1)
        .def_readonly("lm2", &gaunt_L1_L2<double>::lm2)
        .def_readonly("coeff", &gaunt_L1_L2<double>::coef);
}
