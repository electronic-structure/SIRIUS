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
#include "../Beta_projectors/beta_projectors.h"
#include "../Beta_projectors/beta_projectors_gradient.h"
#include "../potential.h"
#include "../density.h"

namespace sirius {

class Forces_PS
{
  private:
    Simulation_context& ctx_;

    Density& density_;

    Potential& potential_;

    K_point_set& kset_;

    mdarray<double, 2> local_forces_;
    mdarray<double, 2> ultrasoft_forces_;
    mdarray<double, 2> nonlocal_forces_;
    mdarray<double, 2> nlcc_forces_;
    mdarray<double, 2> ewald_forces_;

    template<typename T>
    void add_k_point_contribution_to_nonlocal2(K_point& kpoint, mdarray<double,2>& forces)
    {
        Unit_cell &unit_cell = ctx_.unit_cell();

        Beta_projectors& bp = kpoint.beta_projectors();

        Beta_projectors_gradient bp_grad(ctx_, kpoint.gkvec(), bp);

        auto& bp_chunks = bp.beta_projector_chunks();

        // from formula
        double main_two_factor = -2.0;

        #ifdef __GPU
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            if (ctx_.processing_unit() == GPU) {
                int nbnd = kpoint.num_occupied_bands(ispn);
                kpoint.spinor_wave_functions(ispn).allocate_on_device();
                kpoint.spinor_wave_functions(ispn).copy_to_device(0, nbnd);
            }
        }
        #endif

        bp_grad.prepare();
        bp.prepare();

        for (int icnk = 0; icnk < bp_chunks.num_chunks(); icnk++) {
            /* generate chunk for inner product of beta */
            bp.generate(icnk);

            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                /* total number of occupied bands for this spin */
                int nbnd = kpoint.num_occupied_bands(ispn);

                // inner product of beta and WF
                auto bp_phi_chunk = bp.inner<T>(icnk, kpoint.spinor_wave_functions(ispn), 0, nbnd);

                for (int x = 0; x < 3; x++) {
                    /* generate chunk for inner product of beta gradient */
                    bp_grad.generate(icnk, x);

                    // inner product of beta gradient and WF
                    auto bp_grad_phi_chunk = bp_grad.inner<T>(icnk, kpoint.spinor_wave_functions(ispn), 0, nbnd);

                    splindex<block> spl_nbnd(nbnd, kpoint.comm().size(), kpoint.comm().rank());

                    int nbnd_loc = spl_nbnd.local_size();

                    int bnd_offset = spl_nbnd.global_offset();

                    #pragma omp parallel for
                    for(int ia_chunk = 0; ia_chunk < bp_chunks(icnk).num_atoms_; ia_chunk++) {
                        int ia   = bp_chunks(icnk).desc_(3, ia_chunk);
                        int offs = bp_chunks(icnk).desc_(1, ia_chunk);
                        int nbf  = bp_chunks(icnk).desc_(0, ia_chunk);
                        int iat  = unit_cell.atom(ia).type_id();

//                        linalg<CPU>::gemm(0, 0, nbf, n__, nbf,
//                                          op_.at<CPU>(packed_mtrx_offset_(ia), ispn__), nbf,
//                                          beta_phi.at<CPU>(offs, 0), nbeta,
//                                          work_.at<CPU>(offs), nbeta);

                        // mpi
                        // TODO make in smart way with matrix multiplication
                        for (int ibnd_loc = 0; ibnd_loc < nbnd_loc; ibnd_loc++) {
                            int ibnd = spl_nbnd[ibnd_loc];

                            auto D_aug_mtrx = [&](int i, int j)
                            {
                                if (unit_cell.atom(ia).type().pp_desc().augment) {
                                    return unit_cell.atom(ia).d_mtrx(i, j, ispn) - kpoint.band_energy(ibnd) *
                                           ctx_.augmentation_op(iat).q_mtrx(i, j);
                                } else {
                                    return unit_cell.atom(ia).d_mtrx(i, j, ispn);
                                }
                            };

                            for (int ibf = 0; ibf < unit_cell.atom(ia).type().mt_lo_basis_size(); ibf++) {
                                for (int jbf = 0; jbf < unit_cell.atom(ia).type().mt_lo_basis_size(); jbf++) {
                                    /* calculate scalar part of the forces */
                                    double_complex scalar_part = main_two_factor *
                                        kpoint.band_occupancy(ibnd + ispn * ctx_.num_fv_states()) * kpoint.weight() *
                                        D_aug_mtrx(ibf, jbf) * std::conj(bp_phi_chunk(offs + jbf, ibnd));

                                    /* multiply scalar part by gradient components */
                                    forces(x, ia) += (scalar_part * bp_grad_phi_chunk(offs + ibf, ibnd)).real();
                                }
                            }
                        }
                    }
                }
            }
        }

        bp.dismiss();
        bp_grad.dismiss();

        #ifdef __GPU
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            if (ctx_.processing_unit() == GPU ) {
                kpoint.spinor_wave_functions(ispn).deallocate_on_device();
            }
        }
        #endif
    }

    //---------------------------------------------------------------
    //---------------------------------------------------------------
    //template<typename T>
    //void add_k_point_contribution_to_nonlocal(K_point& kpoint, mdarray<double,2>& forces)
    //{
    //    Unit_cell &unit_cell = ctx_.unit_cell();

    //    Beta_projectors &bp = kpoint.beta_projectors();

    //    Beta_projectors_gradient bp_grad(&bp);

    //    // from formula
    //    double main_two_factor = -2.0;
    //    
    //    auto& bp_chunks = bp.beta_projector_chunks();

    //    for (int icnk = 0; icnk < bp_chunks.num_chunks(); icnk++)
    //    {
    //        // generate chunk for inner product of beta gradient
    //        bp_grad.generate(icnk);

    //        // generate chunk for inner product of beta
    //        bp.generate(icnk);

    //        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
    //        {
    //            /* total number of occupied bands for this spin */
    //            int nbnd = kpoint.num_occupied_bands(ispn);

    //            splindex<block> spl_nbnd(nbnd, kpoint.comm().size(), kpoint.comm().rank());

    //            int nbnd_loc = spl_nbnd.local_size();

    //            int bnd_offset = spl_nbnd.global_offset();

    //            printf("rank: %d   nbnd: %d   nbnd_loc: %d   bnd_offset: %d   wf_size: %d %d    beta_gk_size: %d %d\n",
    //                   ctx_.comm().rank(),
    //                   nbnd,
    //                   nbnd_loc,
    //                   bnd_offset,
    //                   kpoint.spinor_wave_functions(ispn).pw_coeffs().prime().size(0),
    //                   kpoint.spinor_wave_functions(ispn).pw_coeffs().prime().size(1),
    //                   bp.beta_gk().size(0),
    //                   bp.beta_gk().size(1));


    //            printf("kp vec: %f  %f  %f \n", kpoint.vk()[0],kpoint.vk()[1], kpoint.vk()[2]);

    //            printf("nl1\n");
    //            // inner product of beta and WF
    //            auto bp_phi_chunk = bp.inner<T>(icnk, kpoint.spinor_wave_functions(ispn), bnd_offset, nbnd_loc);

    //            printf("nl2\n");
    //            // get inner product
    //            //matrix<T> bp_phi_chunk = bp.beta_phi<T>(icnk, nbnd_loc);

    //            printf("nl3\n");
    //            // inner product of beta gradient and WF
    //            bp_grad.inner<T>(icnk, kpoint.spinor_wave_functions(ispn), bnd_offset, nbnd_loc);

    //            printf("nl0\n");
    //            // get inner product
    //            std::array<matrix<T>, 3> bp_grad_phi_chunk = bp_grad.beta_phi<T>(icnk, nbnd_loc);



    //            #pragma omp parallel for
    //            for(int ia_chunk = 0; ia_chunk < bp_chunks(icnk).num_atoms_; ia_chunk++)
    //            {
    //                int ia   = bp_chunks(icnk).desc_(3, ia_chunk);
    //                int offs = bp_chunks(icnk).desc_(1, ia_chunk);
    //                int iat  = unit_cell.atom(ia).type_id();

    //                // mpi
    //                // TODO make in smart way with matrix multiplication
    //                for (int ibnd_loc = 0; ibnd_loc < nbnd_loc; ibnd_loc++)
    //                {
    //                    int ibnd = spl_nbnd[ibnd_loc];

    //                    auto D_aug_mtrx = [&](int i, int j)
    //                    {
    //                        if (unit_cell.atom(ia).type().pp_desc().augment) {
    //                            return unit_cell.atom(ia).d_mtrx(i, j, ispn) - kpoint.band_energy(ibnd) *
    //                                    ctx_.augmentation_op(iat).q_mtrx(i, j);
    //                        } else {
    //                            return unit_cell.atom(ia).d_mtrx(i, j, ispn);
    //                        }
    //                    };

    //                    for(int ibf = 0; ibf < unit_cell.atom(ia).type().mt_lo_basis_size(); ibf++ )
    //                    {
    //                        for(int jbf = 0; jbf < unit_cell.atom(ia).type().mt_lo_basis_size(); jbf++ )
    //                        {
    //                            // calc scalar part of the forces
    //                            double_complex scalar_part = main_two_factor *
    //                                    kpoint.band_occupancy(ibnd + ispn * ctx_.num_fv_states()) * kpoint.weight() *
    //                                    D_aug_mtrx(ibf, jbf) *
    //                                    std::conj(bp_phi_chunk(offs + jbf, ibnd_loc));

    //                            // multiply scalar part by gradient components
    //                            for(int comp: {0,1,2}) forces(comp,ia) += (scalar_part * bp_grad_phi_chunk[comp](offs + ibf, ibnd_loc)).real();
    //                        }
    //                    }
    //                }
    //            }
    //        }
    //    }
    //}

    void symmetrize_forces(mdarray<double,2>& unsym_forces, mdarray<double,2>& sym_forces );

public:
    Forces_PS(Simulation_context &ctx__,
              Density& density__,
              Potential& potential__,
              K_point_set& kset__)
        : ctx_(ctx__)
        , density_(density__)
        , potential_(potential__)
        , kset_(kset__)
    {
        local_forces_     = mdarray<double,2>(3, ctx_.unit_cell().num_atoms());
        ultrasoft_forces_ = mdarray<double,2>(3, ctx_.unit_cell().num_atoms());
        nonlocal_forces_  = mdarray<double,2>(3, ctx_.unit_cell().num_atoms());
        nlcc_forces_      = mdarray<double,2>(3, ctx_.unit_cell().num_atoms());
        ewald_forces_     = mdarray<double,2>(3, ctx_.unit_cell().num_atoms());
    }

    void calc_local_forces(mdarray<double,2>& forces);

    void calc_ultrasoft_forces(mdarray<double,2>& forces);

    void calc_nonlocal_forces(mdarray<double,2>& forces);

    void calc_nlcc_forces(mdarray<double,2>& forces);

    void calc_ewald_forces(mdarray<double,2>& forces);

    void calc_forces_contributions();

    mdarray<double,2> const& local_forces()
    {
        return local_forces_;
    }

    mdarray<double,2> const& ultrasoft_forces()
    {
        return ultrasoft_forces_;
    }

    mdarray<double,2> const& nonlocal_forces()
    {
        return nonlocal_forces_;
    }

    mdarray<double,2> const& nlcc_forces()
    {
        return nlcc_forces_;
    }

    mdarray<double,2> const& ewald_forces()
    {
        return ewald_forces_;
    }

    mdarray<double,2> sum_forces();

    void sum_forces(mdarray<double,2>& inout_total_forces);

};

}

#endif /* SRC_FORCES_PS_H_ */
