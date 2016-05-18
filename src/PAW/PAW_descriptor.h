/*
 * PAW_descriptor.h
 *
 *  Created on: Mar 30, 2016
 *      Author: isivkov
 */

#ifndef SRC_PAW_PAW_DESCRIPTOR_H_
#define SRC_PAW_PAW_DESCRIPTOR_H_

#include "mdarray.h"
#include "vector3d.h"
#include "utils.h"


class PAW_descriptor
{
public:
	PAW_descriptor(): is_initialized(false){}

    /// augmentation integrals
    std::vector<double> aug_integrals;

    /// multipoles qij
    std::vector<double> aug_multopoles;

    /// all electron basis wave functions, have the same dimensionality as uspp.beta_radial_functions
    mdarray<double, 2> all_elec_wfc;

    /// pseudo basis wave functions, have the same dimensionality as uspp.beta_radial_functions
	mdarray<double, 2> pseudo_wfc;

	// core energy - was ist das?
	double core_energy;

	/// occupaations of basis states, length of vector is the same as
	/// number of beta projectors and all_elec_wfc and pseudo_wfc
	std::vector<double> occupations;

	/// density of core electron contribution to all electron charge density
	std::vector<double> all_elec_core_charge;

	/// electrostatic potential of all electron core charge
	std::vector<double> all_elec_loc_potential;

	int cutoff_radius_index;

	bool is_initialized;
};



#endif /* SRC_PAW_PAW_DESCRIPTOR_H_ */
