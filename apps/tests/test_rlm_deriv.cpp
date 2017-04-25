#include <sirius.h>

using namespace sirius;

void test()
{
    vector3d<double> v(0, 0, 1);
    vector3d<double> rtp = SHT::spherical_coordinates(v);

    std::vector<double> drlmdx = {0,0,0,-0.4886025119029199,0,0,0,-1.092548430592079,0,0,0,0,0,-1.828183197857863,0,0,0,0,0,0,0,-2.676186174229157,0,0,0,0,0,0,0,0,0,-3.623573209565575,0,0,0,0,0,0,0,0,0,0,0,-4.660970900149851,0,0,0,0,0,0,0,0,0,0,0,0,0,-5.781222885281108,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-6.978639737521917,0,0,0,0,0,0,0};
    std::vector<double> drlmdy = {0,-0.4886025119029199,0,0,0,-1.092548430592079,0,0,0,0,0,-1.828183197857863,0,0,0,0,0,0,0,-2.676186174229157,0,0,0,0,0,0,0,0,0,-3.623573209565575,0,0,0,0,0,0,0,0,0,0,0,-4.660970900149851,0,0,0,0,0,0,0,0,0,0,0,0,0,-5.781222885281108,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-6.978639737521917,0,0,0,0,0,0,0,0,0}; 

    int lmax{8};
    int lmmax = Utils::lmmax(lmax);

    mdarray<double, 2> rlm_dg(lmmax, 3);

    double theta = rtp[1];
    double phi   = rtp[2];
    std::vector<double> dRlm_dtheta(lmmax);
    std::vector<double> dRlm_dphi_sin_theta(lmmax);
    
    vector3d<double> dtheta_dq({std::cos(phi) * std::cos(theta), std::cos(theta) * std::sin(phi), -std::sin(theta)});
    vector3d<double> dphi_dq({-std::sin(phi), std::cos(phi), 0.0});
    SHT::dRlm_dtheta(lmax, theta, phi, &dRlm_dtheta[0]);
    SHT::dRlm_dphi_sin_theta(lmax, theta, phi, &dRlm_dphi_sin_theta[0]);

    for (int nu = 0; nu < 3; nu++) {
        for (int lm = 0; lm < lmmax; lm++) {
            rlm_dg(lm, nu) = (dRlm_dtheta[lm] * dtheta_dq[nu] + dRlm_dphi_sin_theta[lm] * dphi_dq[nu]) / rtp[0];
        }
    }

    double dg = 1e-4 * rtp[0];
    mdarray<double, 2> rlm_dg_v2(lmmax, 3);
    for (int x = 0; x < 3; x++) {
        vector3d<double> g1 = v;
        g1[x] += dg;
        vector3d<double> g2 = v;
        g2[x] -= dg;
        
        auto gs1 = SHT::spherical_coordinates(g1);
        auto gs2 = SHT::spherical_coordinates(g2);
        std::vector<double> rlm1(lmmax);
        std::vector<double> rlm2(lmmax);
        
        SHT::spherical_harmonics(lmax, gs1[1], gs1[2], &rlm1[0]);
        SHT::spherical_harmonics(lmax, gs2[1], gs2[2], &rlm2[0]);
        
        for (int lm = 0; lm < lmmax; lm++) {
            rlm_dg_v2(lm, x) = (rlm1[lm] - rlm2[lm]) / 2 / dg;
        }
    }

    for (int x = 0; x < 3; x++) {
        for (int lm = 0; lm < lmmax; lm++) {
            printf("x: %i, lm: %2i, diff: %18.12f, (numerical: %18.12f, analytical: %18.12f)\n", x, lm,
                   std::abs(rlm_dg_v2(lm, x) - rlm_dg(lm, x)),
                   rlm_dg_v2(lm, x),
                   rlm_dg(lm, x));
        }
    }
    
    printf("============\n");
    for (int lm = 0; lm < lmmax; lm++) {
        printf("x: %i, lm: %2i, diff with numerical: %18.12f, diff with analytical: %18.12f\n", 0, lm,
               std::abs(rlm_dg_v2(lm, 0) - drlmdx[lm]), std::abs(rlm_dg(lm, 0) - drlmdx[lm]));
    }
    for (int lm = 0; lm < lmmax; lm++) {
        printf("x: %i, lm: %2i, diff with numerical: %18.12f, diff with analytical: %18.12f\n", 1, lm,
               std::abs(rlm_dg_v2(lm, 1) - drlmdy[lm]), std::abs(rlm_dg(lm, 1) - drlmdy[lm]));
    }
}

int main(int argn, char** argv)
{
    cmd_args args;

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(1);
    test();
    sirius::finalize();
}
