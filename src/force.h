#ifndef __FORCE_H__
#define __FORCE_H__

/** \file force.h
    
    \brief Calculation of forces
*/
namespace sirius
{

class Force
{
    private:

        /** In the second-variational approach we need to compute the following expression for the k-dependent 
            contribution to the forces:
            \f[
                {\bf F}_{\rm IBS}^{\alpha}=\sum_{\bf k}w_{\bf k}\sum_{l\sigma}n_{l{\bf k}}
                \sum_{ij}c_{\sigma i}^{l{\bf k}*}c_{\sigma j}^{l{\bf k}}
                {\bf F}_{ij}^{\alpha{\bf k}}
            \f]
            First, we sum over band and spin indices to get the "density matrix":
            \f[
                q_{ij} = \sum_{l\sigma}n_{l{\bf k}} c_{\sigma i}^{l{\bf k}*}c_{\sigma j}^{l{\bf k}}
            \f]
        */
        static void compute_dmat();

        static void ibs_force(Global& parameters_, Band* band, K_point* kp, mdarray<double, 2>& ffac, mdarray<double, 2>& force);

    public:

        static void total_force(Global& parameters_, Potential* potential, Density* density, K_set* ks, mdarray<double, 2>& force);
};

#include "force.hpp"

}

#endif // __FORCE_H__
