#include <sirius.h>

using namespace sirius;

void test_xc()
{
    XC_functional Ex("XC_LDA_X", 2);
    XC_functional Ec("XC_LDA_C_VWN", 2);

    int npt = 5;
    double rho_up[] = {0, 0.1, 0.2, 0.5, 2.234};
    double rho_dn[] = {0, 0.0, 0.1, 1.5, 2.234};

    std::vector<double> vx_up(npt);
    std::vector<double> vx_dn(npt);
    std::vector<double> vc_up(npt);
    std::vector<double> vc_dn(npt);
    std::vector<double> ex(npt);
    std::vector<double> ec(npt);

    /* compute XC potential and energy */
    Ex.get_lda(npt, rho_up, rho_dn, &vx_up[0], &vx_dn[0], &ex[0]);
    Ec.get_lda(npt, rho_up, rho_dn, &vc_up[0], &vc_dn[0], &ec[0]);
    for (int i = 0; i < npt; i++)
    {
        printf("%12.6f %12.6f %12.6f %12.6f %12.6f\n", rho_up[i], rho_dn[i], vx_up[i] + vc_up[i], vx_dn[i] + vc_dn[i], ex[i] + ec[i]);

    }
}

int main(int argn, char** argv)
{
    sirius::initialize(1);
    test_xc();
    sirius::finalize();
}
