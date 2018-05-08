#include <sirius.h>

using namespace sirius;

void test1()
{
    int lmax{20};

    SHT sht(lmax);

    mdarray<double_complex, 2> ylm(Utils::lmmax(lmax), sht.num_points());
    for (int i = 0; i < sht.num_points(); i++) {
        SHT::spherical_harmonics(lmax, sht.theta(i), sht.phi(i), &ylm(0, i));
    }

    for (int l = 0; l <= 8; l++) {
        for (int m = -l; m <= l; m++) {
            double_complex s{0};
            for (int i = 0; i < sht.num_points(); i++) {
                s += std::conj(ylm(Utils::lm_by_l_m(l, m), i)) * ylm(Utils::lm_by_l_m(l, m), i) *
                     ylm(Utils::lm_by_l_m(l, m), i) * sht.weight(i);
            }
            s *= fourpi;
            if (std::abs(s.real() - SHT::gaunt_ylm(l, l, l, m, m, m)) > 1e-12) {
                std::cout << std::abs(s.real() - SHT::gaunt_ylm(l, l, l, m, m, m)) << "\n";
            }
        }
    }

}

int main(int argn, char** argv)
{
    sirius::initialize(1);
    test1();
    sirius::finalize();
    return 0;
}
