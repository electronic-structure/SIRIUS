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
    Simulation_context &ctx_;


public:
    Forces_PS(Simulation_context &ctx)
    : ctx_(ctx)
    {}

    mdarray<double,2> calc_local_forces(const Periodic_function<double>& valence_rho, const mdarray<double, 2>& vloc_radial_integrals) const;

    //vector<vector3d> calc_local_forces(mdarray<double, 2> &rho_radial_integrals, mdarray<double, 2> &vloc_radial_integrals);
};

}

#endif /* SRC_FORCES_PS_H_ */
