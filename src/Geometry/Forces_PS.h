/*
 * force_ps.h
 *
 *  Created on: Sep 20, 2016
 *      Author: isivkov
 */

#ifndef SRC_FORCES_PS_H_
#define SRC_FORCES_PS_H_

#include "../simulation_context.h"
#include "../periodic_function.h"
#include "../augmentation_operator.h"
#include "../Beta_projectors/Beta_projectors.h"
#include "../Beta_projectors/Beta_projectors_gradient.h"
#include "../potential.h"
#include "../density.h"

namespace sirius
{


class Forces_PS
{
private:
    Simulation_context &ctx_;
    Density &density_;
    Potential &potential_;


public:
    Forces_PS(Simulation_context &ctx, Density& density, Potential& potential)
    : ctx_(ctx), density_(density), potential_(potential)
    {}

    mdarray<double,2> calc_local_forces() const;

    mdarray<double,2> calc_ultrasoft_forces() const;

    mdarray<double,2> calc_nonlocal_forces(K_set& kset) const;
    //vector<vector3d> calc_local_forces(mdarray<double, 2> &rho_radial_integrals, mdarray<double, 2> &vloc_radial_integrals);
};

}

#endif /* SRC_FORCES_PS_H_ */
