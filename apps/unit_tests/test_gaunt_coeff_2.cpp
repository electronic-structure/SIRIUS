#include <sirius.hpp>
#include "testing.hpp"

using namespace sirius;

int test1()
{
    int lmax{20};

    SHT sht(device_t::CPU, lmax);

    mdarray<double_complex, 2> ylm(utils::lmmax(lmax), sht.num_points());
    for (int i = 0; i < sht.num_points(); i++) {
        sf::spherical_harmonics(lmax, sht.theta(i), sht.phi(i), &ylm(0, i));
    }

    double d{0};

    for (int l1 = 0; l1 <= 8; l1++) {
        for (int m1 = -l1; m1 <= l1; m1++) {
            for (int l2 = 0; l2 <= 8; l2++) {
                for (int m2 = -l2; m2 <= l2; m2++) {
                    for (int l3 = 0; l3 <= 8; l3++) {
                        for (int m3 = -l3; m3 <= l3; m3++) {
                            double_complex s{0};
                            for (int i = 0; i < sht.num_points(); i++) {
                                s += std::conj(ylm(utils::lm(l1, m1), i)) * ylm(utils::lm(l2, m2), i) *
                                     ylm(utils::lm(l3, m3), i) * sht.weight(i);
                            }
                            s *= fourpi;
                            d += std::abs(s.real() - SHT::gaunt_yyy(l1, l2, l3, m1, m2, m3));
                        }
                    }
                }
            }
        }
    }
    std::cout << d << std::endl;
    if (d < 1e-10) {
        return 0;
    } else {
        return 1;
    }
}

int main(int argn, char** argv)
{
    int err{0};
    err += call_test("<Ylm|Ylm|Ylm> numerical", test1);
    return std::min(err, 1);
}
