/*
 * Beta_projectors_gradient.h
 *
 *  Created on: Oct 14, 2016
 *      Author: isivkov
 */

#ifndef __BETA_PROJECTORS_GRADIENT_H__
#define __BETA_PROJECTORS_GRADIENT_H__

#include "beta_projectors_base.h"
#include "beta_projectors.h"

namespace sirius {

class Beta_projectors_gradient: public Beta_projectors_base<3>
{
  private:
    void generate_pw_coefs_t()
    {
        auto& bchunk = ctx_.beta_projector_chunks();

        if (!bchunk.num_beta_t()) {
            return;
        }
        
        auto& comm = gkvec_.comm();

        for (int i = 0; i < 3; i++) {
            pw_coeffs_t_[i] = matrix<double_complex>(num_gkvec_loc(), bchunk.num_beta_t());
        }

    }

  public:
    Beta_projectors_gradient(Simulation_context& ctx__,
                             Gvec const&         gkvec__,
                             Beta_projectors&    beta__)
        : Beta_projectors_base<3>(ctx__, gkvec__)
    {
        generate_pw_coefs_t();
    }
};

///// Stores gradient components of beta over atomic positions d <G+k | Beta > / d Rn
//class Beta_projectors_gradient
//{
//  protected:
//
//    /// local array of gradient components. dimensions: 0 - gk, 1-orbitals
//    std::array<matrix<double_complex>, 3> components_gk_a_;
//
//    /// the same but for one chunk
//    std::array<matrix<double_complex>, 3> chunk_comp_gk_a_;
//
//    /// the same but for one chunk on gpu
//    std::array<matrix<double_complex>, 3> chunk_comp_gk_a_gpu_;
//
//    /// inner product store
//    std::array<mdarray<double, 1>, 3> beta_phi_;
//
//    Beta_projectors* bp_{nullptr};
//
//  public:
//
//    Beta_projectors_gradient(Beta_projectors* bp__)
//        : bp_(bp__)
//    {
//        for (int comp: {0, 1, 2}) {
//            components_gk_a_[comp] = matrix<double_complex>(bp_->beta_gk_total().size(0), bp_->beta_gk_total().size(1));
//            calc_gradient(comp);
//        }
//
//        // on GPU we create arrays without allocation, it will before use
//        #ifdef __GPU
//        for(int comp: {0, 1, 2}) {
//            chunk_comp_gk_a_gpu_[comp] = matrix<double_complex>(bp_->num_gkvec_loc(), bp_->max_num_beta(), memory_t::none);
//        }
//        #endif
//    }
//
//    void calc_gradient(int calc_component__)
//    {
//        Gvec const& gkvec = bp_->gk_vectors();
//
//        matrix<double_complex> const& beta_comps = bp_->beta_gk_a();
//
//        double_complex Im(0, 1);
//
//        #pragma omp parallel for
//        for (size_t ibf = 0; ibf < bp_->beta_gk_a().size(1); ibf++) {
//            for (int igk_loc = 0; igk_loc < bp_->num_gkvec_loc(); igk_loc++) {
//                int igk = gkvec.gvec_offset(bp_->comm().rank()) + igk_loc;
//
//                double gkvec_comp = gkvec.gkvec_cart(igk)[calc_component__];
//
//                components_gk_a_[calc_component__](igk_loc, ibf) = - Im * gkvec_comp * beta_comps(igk_loc,ibf);
//            }
//        }
//    }
//
//    void generate(int chunk__, int calc_component__)
//    {
//        auto& bp_chunks = bp_->beta_projector_chunks();
//        if (bp_->proc_unit() == CPU)
//        {
//            chunk_comp_gk_a_[calc_component__] = mdarray<double_complex, 2>(&components_gk_a_[calc_component__](0, bp_chunks(chunk__).offset_),
//                                                                            bp_->num_gkvec_loc(),
//                                                                            bp_chunks(chunk__).num_beta_);
//        }
//
//        #ifdef __GPU
//        if (bp_->proc_unit() == GPU)
//        {
//            chunk_comp_gk_a_[calc_component__] = mdarray<double_complex, 2>(&components_gk_a_[calc_component__](0, bp_->beta_chunk(chunk__).offset_),
//                                                                            chunk_comp_gk_a_gpu_[calc_component__].at<GPU>(),
//                                                                            bp_->num_gkvec_loc(),
//                                                                            bp_->beta_chunk(chunk__).num_beta_);
//
//            chunk_comp_gk_a_[calc_component__].copy_to_device();
//        }
//        #endif
//    }
//
//    void generate(int chunk__)
//    {
//        for(int comp: {0, 1, 2}) {
//            generate(chunk__, comp);
//        }
//    }
//
//    /// Calculates inner product <beta_grad | Psi>.
//    template <typename T>
//    void inner(int chunk__, wave_functions& phi__, int idx0__, int n__, int calc_component__)
//    {
//        bp_->inner<T>(chunk__, phi__, idx0__, n__, chunk_comp_gk_a_[calc_component__], beta_phi_[calc_component__]);
//    }
//
//    //void inner(int chunk__,  wave_functions& phi__, int idx0__, int n__, mdarray<double_complex, 2> &beta_gk, mdarray<double, 1> &beta_phi);
//    template <typename T>
//    void inner(int chunk__, wave_functions& phi__, int idx0__, int n__)
//    {
//        for(int comp: {0,1,2}) inner<T>(chunk__, phi__, idx0__, n__, comp);
//    }
//
//    template <typename T>
//    matrix<T> beta_phi(int chunk__, int n__, int calc_component__)
//    {
//        int nbeta = bp_->beta_projector_chunks()(chunk__).num_beta_;
//
//        if (bp_->proc_unit() == GPU) {
//            return std::move(matrix<T>(reinterpret_cast<T*>(beta_phi_[calc_component__].at<CPU>()),
//                                       reinterpret_cast<T*>(beta_phi_[calc_component__].at<GPU>()),
//                                       nbeta, n__));
//        } else {
//            return std::move(matrix<T>(reinterpret_cast<T*>(beta_phi_[calc_component__].at<CPU>()),
//                                       nbeta, n__));
//        }
//    }
//
//    template <typename T>
//    std::array<matrix<T>,3> beta_phi(int chunk__, int n__)
//    {
//        std::array<matrix<T>,3> chunk_beta_phi;
//
//        for(int comp: {0,1,2}) chunk_beta_phi[comp] = beta_phi<T>(chunk__, n__, comp);
//
//        return std::move(chunk_beta_phi);
//    }
//
//    void prepare()
//    {
//        #ifdef __GPU
//        if (bp_->proc_unit() == GPU)
//        {
//            for(int comp: {0,1,2})
//            {
//                chunk_comp_gk_a_gpu_[comp].allocate(memory_t::device);
//                beta_phi_[comp].allocate(memory_t::device);
//            }
//        }
//        #endif
//    }
//
//    void dismiss()
//    {
//        #ifdef __GPU
//        if (bp_->proc_unit() == GPU)
//        {
//            for(int comp: {0,1,2})
//            {
//                chunk_comp_gk_a_gpu_[comp].deallocate_on_device();
//                beta_phi_[comp].deallocate_on_device();
//            }
//        }
//        #endif
//    }
//};

}

#endif // __BETA_PROJECTORS_GRADIENT_H__
