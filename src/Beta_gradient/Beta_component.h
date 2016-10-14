/*
 * Beta_component.h
 *
 *  Created on: Oct 14, 2016
 *      Author: isivkov
 */

#ifndef SRC_BETA_GRADIENT_BETA_COMPONENT_H_
#define SRC_BETA_GRADIENT_BETA_COMPONENT_H_


#include "../beta_projectors.h"

namespace sirius
{

class Beta_components: protected Beta_projectors
{
private:

public:
    Beta_components(Beta_projectors& bp, matrix<double_complex> &&components_store)
        :  comm_(bp.comm_),
           unit_cell_(bp.unit_cell_),
           gkvec_(bp.gkvec_),
           lmax_beta_(bp.lmax_beta_),
           pu_(bp.pu_),
           num_beta_t_(bp.num_beta_t_),
           max_num_beta_(bp.max_num_beta_)
    {
        beta_gk_t_ = matrix<double_complex>(&bp.beta_gk_t_(0,0),bp.beta_gk_t_.size(0),bp.beta_gk_t_.size(1));

        if(components_store.size() != bp.beta_gk_a_)
        {
            TERMINATE("Beta_component is initialized with wrong size of gk components atrray");
        }

        beta_gk_a_ = std::move(components_store);

        #ifdef __GPU
        if (pu_ == GPU)
        {
           TERMINATE_NOT_IMPLEMENTED;
        }
        #endif
    }

    Beta_components(Beta_projectors& bp)
        :  Beta_components(bp, matrix<double_complex>(bp.beta_gk_a_.size(0),bp.beta_gk_a_.size(1)))
    { }
};

}


#endif /* SRC_BETA_GRADIENT_BETA_COMPONENT_H_ */
