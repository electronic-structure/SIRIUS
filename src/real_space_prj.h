#ifndef __REAL_SPACE_PRJ_H__
#define __REAL_SPACE_PRJ_H__

#include <vector>
#include "typedefs.h"
#include "vector3d.h"
#include "mdarray.hpp"
#include "unit_cell.h"
#include "fft3d.h"

namespace sirius {

struct beta_real_space_prj_descriptor
{
    int num_points_;

    int offset_;

    /* list of real-space point indices */
    std::vector<int> ir_;

    /* list of real-space point fractional coordinates */
    std::vector< vector3d<double> > r_;

    /* list of beta-projector translations (fractional coordinates) */
    std::vector< vector3d<int> > T_;

    /* distance from the atom */
    std::vector<double> dist_;

    std::vector< vector3d<double> > rtp_;

    /* beta projectors on a real-space grid */
    mdarray<double, 2> beta_;
};

class Real_space_prj
{
    private:

        Unit_cell& unit_cell_;

        FFT3D* fft_;

        Gvec gvec_;

        splindex<block> spl_num_gvec_;

        Communicator const& comm_;

        double R_mask_scale_;

        double mask_alpha_;

        Spline<double> mask_spline_;

        std::vector<int> nr_beta_;

        std::vector<double> R_beta_;

        //std::vector<Radial_grid> r_beta_;
        
        mdarray<Spline<double>, 2> beta_rf_filtered_;

        double mask(double x__, double R__)
        {
            double t = x__ / R__;
            //return std::pow(t, 2) - 2 * t + 1;

            //return 1 - t;

            //if (std::abs(x__ - R__) < 0.001) return 0;
            //else return std::exp(-mask_alpha_ * std::pow(x__ / R__, 1.5) / (1 - x__ / R__));

            //return mask_spline_(t);

           return -0.14610149615556967*(t - 1)*(120.48032361607294 + t)*(1.0031058954387562 - 2.00016615867864*t + t*t)*
           (0.9883501022892099 - 1.953812672395167*t + t*t)*(1.0008029587995915 - 1.7991334350405186*t + t*t)*
           (0.26138854134766676 + 0.6554735145842409*t + t*t)*(0.21904658711289443 + 0.9309071316277036*t + t*t);
           //return 1;
        }

        double mask_step_f(double x__, double Rmin__, double Rmax__)
        {
            if (x__ <= Rmin__)
            {
                return 1;
            }
            else
            {
                double a = std::log(std::pow(10, 13)) / std::pow(Rmax__ / Rmin__ - 1, 2);
                return std::exp(-a * std::pow(x__ / Rmin__ - 1, 2));
            }
        }

        mdarray<double, 3> generate_beta_radial_integrals(mdarray<Spline<double>, 2>& beta_rf__, int m__);

        mdarray<double_complex, 2> generate_beta_pw_t(mdarray<double, 3>& beta_radial_integrals__);

        void filter_radial_functions(double pw_cutoff__);
        void filter_radial_functions_v2(double pw_cutoff__);

        void get_beta_R();

        void get_beta_grid();

    public:
        
        std::vector<beta_real_space_prj_descriptor> beta_projectors_;

        int max_num_points_;

        int num_points_;

        Real_space_prj(Unit_cell& unit_cell__,
                       Communicator const& comm__,
                       double R_mask_scale__,
                       double mask_alpha__,
                       double pw_cutoff__,
                       int num_fft_threads__,
                       int num_fft_workers__);

        ~Real_space_prj()
        {
            delete fft_;
        }

        FFT3D* fft() const
        {
            return fft_;
        }
};

};

#endif // __REAL_SPACE_PRJ_H__
