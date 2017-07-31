/*
 * Non_local_functor.h
 *
 *  Created on: Mar 13, 2017
 *      Author: isivkov
 */

#ifndef __NON_LOCAL_FUNCTOR_H__
#define __NON_LOCAL_FUNCTOR_H__

#include "../simulation_context.h"
#include "../periodic_function.h"
#include "../augmentation_operator.h"
#include "../Beta_projectors/beta_projectors.h"
#include "../potential.h"
#include "../density.h"

namespace sirius {

template <typename T, int N>
class Non_local_functor
{
  private:
    Simulation_context& ctx_;
    Beta_projectors_base<N>& bp_base_;

  public:

    Non_local_functor(Simulation_context& ctx__,
                      Beta_projectors_base<N>& bp_base__)
        : ctx_(ctx__)
        , bp_base_(bp_base__)
    {}

    /// Dimension of the beta-projector array.
    static const int N_ = N;

    /// collect summation result in an array
    void add_k_point_contribution(K_point& kpoint__, mdarray<double, 2>& collect_res__)
    {
        Unit_cell& unit_cell = ctx_.unit_cell();

        Beta_projectors& bp = kpoint__.beta_projectors();

        auto& bp_chunks = bp.beta_projector_chunks();

        double main_two_factor = -2.0;

        bp_base_.prepare();
        bp.prepare();

        for (int icnk = 0; icnk < bp_chunks.num_chunks(); icnk++) {
            /* generate chunk for inner product of beta */
            bp.generate(icnk);

            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                double spin_factor = (ispn == 0 ? 1.0 : -1.0);

                /* total number of occupied bands for this spin */
                int nbnd = kpoint__.num_occupied_bands(ispn);

                /* inner product of beta and WF */
                auto bp_phi_chunk = bp.inner<T>(icnk, kpoint__.spinor_wave_functions(ispn), 0, nbnd);

                for (int x = 0; x < N; x++) {
                    /* generate chunk for inner product of beta gradient */
                    bp_base_.generate(icnk, x);

                    /* inner product of beta gradient and WF */
                    auto bp_base_phi_chunk = bp_base_.inner<T>(icnk, kpoint__.spinor_wave_functions(ispn), 0, nbnd);

                    splindex<block> spl_nbnd(nbnd, kpoint__.comm().size(), kpoint__.comm().rank());

                    int nbnd_loc = spl_nbnd.local_size();

                    int bnd_offset = spl_nbnd.global_offset();

                    #pragma omp parallel for
                    for (int ia_chunk = 0; ia_chunk < bp_chunks(icnk).num_atoms_; ia_chunk++) {
                        int ia   = bp_chunks(icnk).desc_(beta_desc_idx::ia, ia_chunk);
                        int offs = bp_chunks(icnk).desc_(beta_desc_idx::offset, ia_chunk);
                        int nbf  = bp_chunks(icnk).desc_(beta_desc_idx::nbf, ia_chunk);
                        int iat  = unit_cell.atom(ia).type_id();

                        // TODO store this to array before (for speed optimization)
                        auto D_aug_mtrx = [&](int i, int j, int ibnd)
                        {
                            double dij = 0.0;

                            switch (ctx_.num_spins())
                            {
                                case 1:
                                    dij = unit_cell.atom(ia).d_mtrx(i, j, 0);
                                    break;

                                case 2:
                                    dij =  (unit_cell.atom(ia).d_mtrx(i, j, 0) + spin_factor * unit_cell.atom(ia).d_mtrx(i, j, 1));
                                    break;

                                default:
                                    TERMINATE("Error in calc_ultrasoft_forces: Non-collinear not implemented");
                                    break;
                            }

                            if (unit_cell.atom(ia).type().pp_desc().augment) {
                                dij -= kpoint__.band_energy(ibnd + ispn * ctx_.num_fv_states()) * ctx_.augmentation_op(iat).q_mtrx(i, j);
                            }

                            return dij;
                        };

                        /* iterate over mpi-distributed bands */
                        for (int ibnd_loc = 0; ibnd_loc < nbnd_loc; ibnd_loc++) {
                            int ibnd = spl_nbnd[ibnd_loc];

                            for (int ibf = 0; ibf < unit_cell.atom(ia).type().mt_lo_basis_size(); ibf++) {
                                for (int jbf = 0; jbf < unit_cell.atom(ia).type().mt_lo_basis_size(); jbf++) {
                                    /* calculate scalar part of the forces */
                                    double_complex scalar_part = main_two_factor * kpoint__.band_occupancy(ibnd + ispn * ctx_.num_fv_states()) *
                                            kpoint__.weight() * D_aug_mtrx(ibf, jbf, ibnd) *
                                            std::conj(bp_phi_chunk(offs + jbf, ibnd));

                                    /* multiply scalar part by gradient components */
                                    collect_res__(x, ia) += (scalar_part * bp_base_phi_chunk(offs + ibf, ibnd)).real();
                                }
                            }
                        }
                    }
                }
            }
        }

        bp.dismiss();
        bp_base_.dismiss();
    }


    /// collect summation result in an array
    void add_k_point_contribution_nl(K_point& kpoint__, mdarray<double, 2>& collect_res__)
    {
        Unit_cell& unit_cell = ctx_.unit_cell();

        Beta_projectors& bp = kpoint__.beta_projectors();

        auto& bp_chunks = bp.beta_projector_chunks();

        double main_two_factor = -2.0;

        bp_base_.prepare();
        bp.prepare();

        for (int icnk = 0; icnk < bp_chunks.num_chunks(); icnk++) {
            /* generate chunk for inner product of beta */
            bp.generate(icnk);

            // store <beta|psi> for spin up and down
            matrix<T> bp_phi[2];

            for(int ispn = 0; ispn < ctx_.num_spins(); ispn++){
                int nbnd = kpoint__.num_occupied_bands(ispn);
                bp_phi[0] = std::move( bp.inner<T>(icnk, kpoint__.spinor_wave_functions(ispn), 0, nbnd) );
            }

            for (int x = 0; x < N; x++) {
                /* generate chunk for inner product of beta gradient */
                bp_base_.generate(icnk, x);

                for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                    int spin_bnd_offset = ctx_.num_mag_dims() == 1 ? ispn * ctx_.num_fv_states() : 0 ;
                    double spin_factor = (ispn == 0 ? 1.0 : -1.0);

                    /* inner product of beta gradient and WF */
                    auto bp_base_phi_chunk = bp_base_.inner<T>(icnk, kpoint__.spinor_wave_functions(ispn), 0, nbnd);

                    splindex<block> spl_nbnd(nbnd, kpoint__.comm().size(), kpoint__.comm().rank());

                    int nbnd_loc = spl_nbnd.local_size();

                    int bnd_offset = spl_nbnd.global_offset();

                    #pragma omp parallel for
                    for (int ia_chunk = 0; ia_chunk < bp_chunks(icnk).num_atoms_; ia_chunk++) {
                        int ia   = bp_chunks(icnk).desc_(beta_desc_idx::ia, ia_chunk);
                        int offs = bp_chunks(icnk).desc_(beta_desc_idx::offset, ia_chunk);
                        int nbf  = bp_chunks(icnk).desc_(beta_desc_idx::nbf, ia_chunk);
                        int iat  = unit_cell.atom(ia).type_id();

                        // TODO store this to array before (for speed optimization)
                        auto D_aug_mtrx = [&](int i, int j, int ibnd)
                        {
                            double dij = 0.0;

                            switch (ctx_.num_spins())
                            {
                                case 1:
                                    dij = unit_cell.atom(ia).d_mtrx(i, j, 0);
                                    break;

                                case 2:
                                    dij =  (unit_cell.atom(ia).d_mtrx(i, j, 0) + spin_factor * unit_cell.atom(ia).d_mtrx(i, j, 1));
                                    break;

                                default:
                                    TERMINATE("Error in non_local_functor, D_aug_mtrx. ");
                                    break;
                            }

                            if (unit_cell.atom(ia).type().pp_desc().augment) {
                                dij -= kpoint__.band_energy(ibnd + spin_bnd_offset) * ctx_.augmentation_op(iat).q_mtrx(i, j);
                            }

                            return dij;
                        };

                        /* iterate over mpi-distributed bands */
                        for (int ibnd_loc = 0; ibnd_loc < nbnd_loc; ibnd_loc++) {
                            int ibnd = spl_nbnd[ibnd_loc];

                            for (int ibf = 0; ibf < unit_cell.atom(ia).type().mt_lo_basis_size(); ibf++) {
                                for (int jbf = 0; jbf < unit_cell.atom(ia).type().mt_lo_basis_size(); jbf++) {
                                    /* calculate scalar part of the forces */
                                    double_complex scalar_part = main_two_factor * kpoint__.band_occupancy(spin_bnd_offset) *
                                            kpoint__.weight() * D_aug_mtrx(ibf, jbf, ibnd) *
                                            std::conj(bp_phi_chunk(offs + jbf, ibnd));

                                    /* multiply scalar part by gradient components */
                                    collect_res__(x, ia) += (scalar_part * bp_base_phi_chunk(offs + ibf, ibnd)).real();
                                }
                            }
                        }
                    }
                }
            }
        }

        bp.dismiss();
        bp_base_.dismiss();
    }
};

}

#endif /* __NON_LOCAL_FUNCTOR_H__ */
