#include "smearing.hpp"
#include "constants.hpp"
#include "specfunc/specfunc.hpp"

namespace smearing {


/**
   These are the coefficients \f$A_n\f$ required to compute the MP-smearing:
   \f[
   \frac{(-1)^n}{n! 4^n \sqrt{\pi}}
   \f]
 */
double
mp_coefficients(int n)
{
    double sqrtpi = std::sqrt(pi);
    int sign      = n % 2 == 0 ? 1 : -1;
    return sign / tgamma(n+1) / std::pow(4, n) / sqrtpi;
}

double
methfessel_paxton::delta(double x__, double width__, int n__)
{
    double result{0};
    double z = -x__/width__;
    for (int i = 1; i <= n__; ++i) {
        result += mp_coefficients(i) * sf::hermiteh(2*i, z) * std::exp(-z*z);
    }
    return result;
}

double
methfessel_paxton::occupancy(double x__, double width__, int n__)
{
    double z = -x__ / width__;
    double result{0};
    result = 0.5*(1-std::erf(z));
    // todo s0 is missing
    for (int i = 1; i <= n__; ++i) {
        double A = mp_coefficients(i);
        result += A * sf::hermiteh(2*i-1, z) * std::exp(-z * z);
    }
    return result;
}

double
methfessel_paxton::occupancy_deriv(double x__, double width__, int n__)
{
    double z = -x__ / width__;
    double result = -std::exp(-z*z) / std::sqrt(pi) / width__ * (-1);
    for (int i = 1; i <= n__; ++i) {
        double A = mp_coefficients(i);
        result -= A * sf::hermiteh(2*i, z) * std::exp(-z * z) * (-1);
    }
    return result;
}

double
methfessel_paxton::occupancy_deriv2(double x__, double width__, int n__)
{
    double z = -x__ / width__;
    double result = 2 * std::exp(-z*z) * z / std::sqrt(pi) / (width__ * width__);
    for (int i = 1; i <= n__; ++i) {
        double A = mp_coefficients(i);
        result += A * sf::hermiteh(2*i + 1, z) * std::exp(-z * z);
    }
    return result;
}

double
methfessel_paxton::entropy(double x__, double width__, int n__) {
    // TODO: find formula for this
    return 0;
}

} // namespace smearing
