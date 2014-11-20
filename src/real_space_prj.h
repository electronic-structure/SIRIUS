#ifndef __REAL_SPACE_PRJ_H__
#define __REAL_SPACE_PRJ_H__

#include <vector>
#include "typedefs.h"
#include "reciprocal_lattice.h"

namespace sirius {

struct beta_real_space_prj_descriptor
{
    int num_points_;

    /* list of real-space point indices */
    std::vector<int> ir_;

    /* list of real-space point fractional coordinates */
    std::vector< vector3d<double> > r_;

    /* list of beta-projector translations (fractional coordinates) */
    std::vector< vector3d<int> > T_;

    /* distance from the atom */
    std::vector<double> dist_;

    /* beta projectors on a real-space grid */
    mdarray<double_complex, 2> beta_;
};

class Real_space_prj
{
    private:

        Reciprocal_lattice* reciprocal_lattice_;

        FFT3D<CPU>* fft_;

        splindex<block> spl_num_gvec_;

        Communicator const& comm_;

        double mask(double x__, double R__)
        {
            return std::pow(x__ / R__, 2) - 2 * (x__ / R__) + 1;
        }

        mdarray<double, 3> generate_beta_radial_integrals(Unit_cell* uc__,
                                                          std::vector<int>& nmt_beta__,
                                                          std::vector<double>& R_beta__);

        mdarray<double_complex, 2> generate_beta_pw_t(Unit_cell* uc__,
                                                      mdarray<double, 3>& beta_radial_integrals__);

    public:
        
        std::vector<beta_real_space_prj_descriptor> beta_projectors_;

        int max_num_points_;

        Real_space_prj(Reciprocal_lattice* reciprocal_lattice__, double gk_cutoff__, Communicator const& comm__);
};

};

#endif // __REAL_SPACE_PRJ_H__
