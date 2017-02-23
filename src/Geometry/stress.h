#ifndef __STRESS_H__
#define __STRESS_H__

namespace sirius {

class Stress {
  private:
    Simulation_context& ctx_;
    
    K_point_set& kset_;

    Density& density_;

    matrix3d<double> stress_kin_;

    matrix3d<double> stress_har_;
    
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
                        d += f * (std::pow(z.real(), 2) + std::pow(z.imag(), 2));
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

        symmetrize(stress_kin_);
    }

    inline void calc_stress_har()
    {
        for (int igloc = 0; igloc < ctx_.gvec_count(); igloc++) {
            int ig = ctx_.gvec_offset() + igloc;
            if (!ig) {
                continue;
            }

            auto G = ctx_.gvec().gvec_cart(ig);
            double g2 = std::pow(G.length(), 2);
            auto z = density_.rho()->f_pw(ig);
            double d = (std::pow(z.real(), 2) + std::pow(z.imag(), 2)) / g2;

            for (int mu: {0, 1, 2}) {
                for (int nu: {0, 1, 2}) {
                    stress_har_(mu, nu) += d * 2 * G[mu] * G[nu] / g2;
                }
            }
            for (int mu: {0, 1, 2}) {
                stress_har_(mu, mu) -= d;
            }
        }

        if (ctx_.gvec().reduced()) {
            stress_har_ *= 2;
        } 

        ctx_.comm().allreduce(&stress_har_(0, 0), 9 * sizeof(double));
        stress_har_ *= twopi;

        symmetrize(stress_har_);
    }

    inline void symmetrize(matrix3d<double>& mtrx__)
    {
        if (!ctx_.use_symmetry()) {
            return;
        }

        matrix3d<double> result;

        for (int i = 0; i < ctx_.unit_cell().symmetry().num_mag_sym(); i++) {
            auto R = ctx_.unit_cell().symmetry().magnetic_group_symmetry(i).spg_op.rotation;
            result = result + transpose(R) * mtrx__ * R;
        }

        mtrx__ = result * (1.0 / ctx_.unit_cell().symmetry().num_mag_sym());
    }

  public:
    Stress(Simulation_context& ctx__,
           K_point_set& kset__,
           Density& density__)
        : ctx_(ctx__)
        , kset_(kset__)
        , density_(density__)
    {
        calc_stress_kin();
        calc_stress_har();

        printf("== stress_kin ==\n");
        for (int mu: {0, 1, 2}) {
            printf("%12.6f %12.6f %12.6f\n", stress_kin_(mu, 0), stress_kin_(mu, 1), stress_kin_(mu, 2));
        }
        printf("== stress_har ==\n");
        for (int mu: {0, 1, 2}) {
            printf("%12.6f %12.6f %12.6f\n", stress_har_(mu, 0), stress_har_(mu, 1), stress_har_(mu, 2));
        }
    }

};

}

#endif
