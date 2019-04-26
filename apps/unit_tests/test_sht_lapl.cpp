#include <sirius.h>

using namespace sirius;


template <typename T>
double test()
{
    int lmax{9};
    SHT sht(lmax);
    int lmmax = utils::lmmax(lmax);

    auto r = Radial_grid_factory<double>(radial_grid_t::exponential, 1000, 0.01, 2.0, 1.0);

    Spheric_function<function_domain_t::spectral, T> f(lmmax, r);

    for (int ir = 0; ir < r.num_points(); ir++) {
        for (int l = 0; l <= lmax; l++) {
            for (int m = -l; m <= l; m++) {
                int lm = utils::lm(l, m);
                f(lm, ir) = std::exp(-0.1 * (lm + 1) * r[ir]) * std::pow(r[ir], l % 3);
            }
        }
    }

    auto lapl_f = laplacian(f);
    auto grad_f = gradient(f);
    auto div_grad_f = divergence(grad_f);

    Spline<double> s(r);
    for (int ir = 0; ir < r.num_points(); ir++) {
        for (int lm = 0; lm < utils::lmmax(lmax); lm++) {
            s(ir) += std::abs(lapl_f(lm, ir) - div_grad_f(lm, ir));
        }
    }
    return s.interpolate().integrate(0);
}

int main(int argn, char** argv)
{
    sirius::initialize(true);

    if (test<double_complex>() > 1e-9) {
        return 1;
    }

    if (test<double>() > 1e-9) {
        return 2;
    }

    sirius::finalize();

    return 0;
}
