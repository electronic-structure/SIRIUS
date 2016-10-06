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

namespace sirius
{


class Forces_PS
{
private:
    Simulation_context &sim_ctx_;


public:
    Forces_PS(Simulation_context &sim_ctx)
    : sim_ctx_(sim_ctx)
    {}

    vector<vector3d> calc_local_forces(Periodic_function &valence_rho, mdarray<double, 2> &vloc_radial_integrals);

    //vector<vector3d> calc_local_forces(mdarray<double, 2> &rho_radial_integrals, mdarray<double, 2> &vloc_radial_integrals);
};

}

#endif /* SRC_FORCES_PS_H_ */
