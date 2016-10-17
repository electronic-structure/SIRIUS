/*
 * Beta_projectors_gradient.h
 *
 *  Created on: Oct 14, 2016
 *      Author: isivkov
 */

#ifndef SRC_BETA_PROJECTORS_BETA_PROJECTORS_GRADIENT_H_
#define SRC_BETA_PROJECTORS_BETA_PROJECTORS_GRADIENT_H_


#include "Beta_projectors.h"

namespace sirius
{

class Beta_projectors_gradient
{
protected:

    // local array of gradient components. dimensions: 0 - gk, 1-orbitals
    matrix<double_complex> components_gk_a_;

    // the same but for one chunk
    std::array<matrix<double_complex>, 3> chunk_comp_gk_a_;

    // inner product store
    std::array<matrix<double_complex>, 3> beta_phi_;

    Beta_projectors *bp_;

    int calc_component_;

public:

    Beta_projectors_gradient(Beta_projectors* bp)
    : bp_(bp)
    {
        components_gk_a_ = matrix<double_complex>( bp_->beta_gk_a().size(0), bp_->beta_gk_a().size(1) );
        calc_component_=0;
    }

    void set_component(int calc_component){ calc_component_ = calc_component; }

    int current_component() { return calc_component_; }

    void calc_gradient(int calc_component)
    {
        calc_component_ = calc_component;

        const Gvec &gkvec = bp_->gk_vectors();

        const matrix<double_complex> &beta_comps = bp_->beta_gk_a();

        #pragma omp parallel for
        for(int ibf=0; ibf< bp_->beta_gk_a().size(1); ibf++)
        {
            for(int igk_loc=0; igk_loc < bp_->num_gkvec_loc(); igk_loc++)
            {
                int igk = gkvec.gvec_offset(bp_->comm().rank()) + igk_loc;

                double gkvec_comp = gkvec.gkvec_cart(igk)[calc_component_];

                components_gk_a_(igk, ibf) = - gkvec_comp * beta_comps(igk,ibf);
            }
        }
    }

    void generate(int chunk__)
    {
        if (bp_->proc_unit() == CPU)
        {
            chunk_comp_gk_a_[calc_component_] = mdarray<double_complex, 2>(&components_gk_a_(0, bp_->beta_chunk(chunk__).offset_),
                                                  bp_->num_gkvec_loc(), bp_->beta_chunk(chunk__).num_beta_);
        }
        #ifdef __GPU
        if (bp_->proc_unit() == GPU)
        {
            TERMINATE_NOT_IMPLEMENTED
        }
        #endif
    }

    template <typename T>
    const matrix<double_complex>& inner(int chunk__, wave_functions& phi__, int idx0__, int n__)
    {
        bp_->inner<T>(chunk__, phi__, idx0__, n__, chunk_comp_gk_a_[calc_component_], beta_phi_[calc_component_]);

        return beta_phi_[calc_component_];
    }
};

}

#endif /* SRC_BETA_PROJECTORS_BETA_PROJECTORS_GRADIENT_H_ */
