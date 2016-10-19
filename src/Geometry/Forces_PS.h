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

    //---------------------------------------------------------------
    //---------------------------------------------------------------
    template<typename T>
    void add_k_point_contribution_to_nonlocal(K_point& kpoint, mdarray<double,2>& forces)
    {
        Unit_cell &unit_cell = ctx_.unit_cell();

        Beta_projectors &bp = kpoint.beta_projectors();

        Beta_projectors_gradient bp_grad(bp);

        for (int icnk = 0; icnk < bp.num_beta_chunks(); icnk++)
        {
            // generate chunk for inner product of beta gradient
            bp_grad.generate(icnk);

            // generate chunk for inner product of beta
            bp.generate(icnk);

            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
            {
                int nbnd = kpoint.num_occupied_bands(ispn);

                // inner product of beta gradient and WF
                bp_grad.inner<T>(icnk, kpoint.spinor_wave_functions(ispn), 0, nbnd);

                // get inner product
                std::array<matrix<T>, 3> bp_grad_phi_chunk = bp_grad.beta_phi(icnk, nbnd);

                // inner product of beta and WF
                bp.inner<T>(icnk, kpoint.spinor_wave_functions(ispn), 0, nbnd);

                // get inner product
                matrix<T> bp_phi_chunk = bp.beta_phi(icnk, nbnd);

                /* total number of occupied bands for this spin */
                int nbnd = kpoint.num_occupied_bands(ispn);

                splindex<block> spl_nbnd(nbnd, kpoint.comm().size(), kpoint.comm().rank());

                int nbnd_loc = spl_nbnd.local_size();

                // omp
                for(int ia_chunk = 0; ia_chunk < bp.beta_chunk(icnk).num_atoms_; ia_chunk++)
                {
                    int ia = bp.beta_chunk(icnk).desc_(3, ia_chunk);
                    int offs = bp.beta_chunk(icnk).desc_(1, ia_chunk);

                    // mpi
                    // TODO make in smart way with matrix multiplication
                    for (int ibnd_loc = 0; i < nbnd_loc; ibnd_loc++)
                    {
                        int ibnd = spl_nbnd[ibnd_loc];

                        auto D_aug_mtrx = [&](int i, int j)
                                {
                                    return unit_cell.atom(ia).d_mtrx(i, j) - kpoint.band_energy(ibnd) *
                                            ctx_.augmentation_op(ia).q_mtrx(i, j);
                                }

                        for(int ibf = 0; ibf < unit_cell.atom(ia).type().mt_lo_basis_size(); ibf++ )
                        {
                            for(int jbf = 0; jbf <= ibf; ibf++ )
                            {
                                double diag_fact = ibf == jbf ? 1.0 : 2.0 ;

                                // calc scalar part of the forces
                                double scalar_part = 2 * diag_fact *
                                        kpoint.band_occupancy(ibnd + ispn * ctx_.num_fv_states()) *
                                        D_aug_mtrx(ibf, jbf) *
                                        std::conj(bp_phi_chunk(offs + ibf, ibnd));

                                // multiply scalar part by gradient components
                                for(comp: {0,1,2}) forces(comp,ia) += scalar_part * bp_grad_phi_chunk[comp](offs + jbf, ibnd);
                            }
                        }
                    }
                }
            }
        }

    }

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
