#ifndef __STRESS_H__
#define __STRESS_H__

namespace sirius {

class Stress {
  private:
    Simulation_context& ctx_;
    
    K_point_set& kset_;

    matrix3d<double> stress_kin_;
    
    inline void calc_stress_kin()
    {
        for (int ikloc = 0; ikloc < kset_.spl_num_kpoints().local_size(); ikloc++) {
            int ik = kset_.spl_num_kpoints(ikloc);
            auto kp = kset_[ik];

            for (int igloc = 0; igloc < kp->num_gkvec_loc(); igloc++) {
                int ig = kp->idxgk(igloc);
                auto Gk = kp->gkvec().gkvec_cart(ig);
                
                double d{0};
                for (int i = 0; i < ctx_.num_bands(); i++) {
                    double f = kp->band_occupancy(i);
                    if (f > 1e-12) {
                        auto z = kp->spinor_wave_functions(0).pw_coeffs().prime(igloc, i);
                        d += std::pow(z.real(), 2) + std::pow(z.imag(), 2);
                    }
                }
                d *= kp->weight();
                for (int mu: {0, 1, 2}) {
                    for (int nu: {0, 1, 2}) {
                        stress_kin_(mu, nu) += Gk[mu] * Gk[nu] * d;
                    }
                }
            } // igloc
        } // ikloc

        ctx_.comm().allreduce(&stress_kin_(0, 0), 9 * sizeof(double));

        stress_kin_ *= (-1.0 / ctx_.unit_cell().omega());
    }

  public:
    Stress(Simulation_context& ctx__,
           K_point_set& kset__)
        : ctx_(ctx__)
        , kset_(kset__)
    {
        calc_stress_kin();

        printf("== stress_kin ==\n");
        for (int mu: {0, 1, 2}) {
            printf("%12.6f %12.6f %12.6f\n", stress_kin_(mu, 0), stress_kin_(mu, 1), stress_kin_(mu, 2));
        }
    }



};

}

#endif
