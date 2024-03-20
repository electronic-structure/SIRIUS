/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include "potential/xc_functional.hpp"

using namespace sirius;

void
test_xc()
{
    XC_functional_base Ex("XC_LDA_X", 2);
    XC_functional_base Ec("XC_LDA_C_VWN", 2);

    int npt         = 5;
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
    for (int i = 0; i < npt; i++) {
        printf("%12.6f %12.6f %12.6f %12.6f %12.6f\n", rho_up[i], rho_dn[i], vx_up[i] + vc_up[i], vx_dn[i] + vc_dn[i],
               ex[i] + ec[i]);
    }
}

void
test_xc2()
{
    XC_functional_base Ex("XC_GGA_X_PBE", 1);
    XC_functional_base Ec("XC_GGA_C_PBE", 1);
    XC_functional_base E1("XC_LDA_X", 1);
    XC_functional_base E2("XC_LDA_C_PZ", 1);

    // XC_functional Ec("XC_LDA_C_PZ", 1);
    /// XC_functional Ex("XC_GGA_X_PW91", 1);
    // XC_functional Ec("XC_GGA_C_PW91", 1);

    std::vector<double> sigma(101);
    std::vector<double> vrho(101);
    std::vector<double> vsigma(101);
    std::vector<double> ex(101);
    std::vector<double> ec(101);
    std::vector<double> e1(101);
    std::vector<double> e2(101);

    int Nsigma{1};

    auto fout = fopen("xc_libxc.dat", "w+");
    for (int i = 0; i <= 50; i++) {
        std::vector<double> rho(Nsigma, i);
        for (int j = 0; j < Nsigma; j++) {
            sigma[j] = j;
        }
        std::vector<double> result(Nsigma, 0);

        // Ex.get_gga(Nsigma, rho.data(), sigma.data(), vrho.data(), vsigma.data(), ex.data());
        // Ec.get_gga(Nsigma, rho.data(), sigma.data(), vrho.data(), vsigma.data(), ec.data());
        E1.get_lda(Nsigma, rho.data(), vrho.data(), e1.data());
        E2.get_lda(Nsigma, rho.data(), vrho.data(), e2.data());

        for (int i = 0; i < Nsigma; i++) {
            // result[i] = (ex[i] - e1[i]) * rho[i] + e1[i] + (ec[i] - e2[i]) * rho[i] + e2[i];
            result[i] = (e1[i] + e2[i]) * rho[i];
        }

        /////Ex.get_lda(101, rho.data(), vrho.data(), e.data());
        /// for (int i = 0; i <= 100; i++) {
        ///    //result[i] += e[i];
        ///}
        // Ec.get_gga(101, rho.data(), sigma.data(), vrho.data(), vsigma.data(), e.data());
        ////Ec.get_lda(101, rho.data(), vrho.data(), e.data());
        // for (int i = 0; i <= 100; i++) {
        //     //result[i] += e[i];
        // }

        // E3.get_lda(101, rho.data(), vrho.data(), e.data());
        //////Ec.get_lda(101, rho.data(), vrho.data(), e.data());
        // for (int i = 0; i <= 100; i++) {
        //     result[i] += e[i];
        // }
        // E4.get_lda(101, rho.data(), vrho.data(), e.data());
        // for (int i = 0; i <= 100; i++) {
        //     result[i] += e[i];
        // }

        for (int i = 0; i < Nsigma; i++) {
            fprintf(fout, "%18.10f", result[i]);
        }
        fprintf(fout, "\n");
    }
    fclose(fout);
}

int
main(int argn, char** argv)
{
    sirius::initialize(1);
    test_xc();
    test_xc2();
    sirius::finalize();
}
