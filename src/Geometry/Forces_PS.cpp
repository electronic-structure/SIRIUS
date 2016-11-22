/*
 * Forces_PS.cpp
 *
 *  Created on: Oct 6, 2016
 *      Author: isivkov
 */

#include "Forces_PS.h"

//#include "../Beta_gradient/Beta_projectors_gradient.h"

#include "../k_set.h"

namespace sirius
{

//---------------------------------------------------------------
//---------------------------------------------------------------
mdarray<double,2> Forces_PS::calc_local_forces() const
{
    // get main arrays
    const Periodic_function<double>* valence_rho = density_.rho();

    const mdarray<double, 2>& vloc_radial_integrals = potential_.get_vloc_radial_integrals();

    // other
    Unit_cell &unit_cell = ctx_.unit_cell();

    splindex<block> spl_ngv(valence_rho->gvec().num_gvec(), ctx_.comm().size(), ctx_.comm().rank());

    //mdarray<double_complex, 2> vloc_G_comp(unit_cell.num_atoms(), spl_ngv.local_size() );

    mdarray<double,2> forces(3, unit_cell.num_atoms());

    forces.zero();

    double fact = valence_rho->gvec().reduced() ? 2.0 : 1.0 ;

    // here the calculations are in lattice vectors space
    #pragma omp parallel for
    for (int ia = 0; ia < unit_cell.num_atoms(); ia++)
    {
        Atom &atom = unit_cell.atom(ia);

        int iat = atom.type_id();

        for (int igloc = 0; igloc < spl_ngv.local_size(); igloc++)
        {
            int ig = spl_ngv[igloc];

            int igs = valence_rho->gvec().shell(ig);

            // fractional form for calculation of scalar product with atomic position
            // since atomic positions are stored in fractional coords
            vector3d<int> gvec = valence_rho->gvec().gvec(ig);

            // cartesian form for getting cartesian force components
            vector3d<double> gvec_cart = valence_rho->gvec().gvec_cart(ig);

            // store conj(rho_G) * 4 * pi
            double_complex g_dependent_prefactor =  fact * std::conj( valence_rho->f_pw_local(igloc) ) * fourpi;

            // scalar part of a force without multipying by G-vector
            double_complex z = vloc_radial_integrals(iat, igs) * g_dependent_prefactor *
                    std::exp(double_complex(0.0, - twopi * (gvec * atom.position())));

            // get force components multiplying by cartesian G-vector ( -image part goes from formula)
            forces(0, ia) += - (gvec_cart[0] * z).imag();
            forces(1, ia) += - (gvec_cart[1] * z).imag();
            forces(2, ia) += - (gvec_cart[2] * z).imag();
        }
    }

    ctx_.comm().allreduce(&forces(0,0),forces.size());

    return std::move(forces);
}




//---------------------------------------------------------------
//---------------------------------------------------------------
mdarray<double,2> Forces_PS::calc_ultrasoft_forces() const
{
    // get main arrays
    const mdarray<double_complex, 4> &density_matrix = density_.density_matrix();

    const Periodic_function<double> *veff_full = potential_.effective_potential();

    //Periodic_function<double> **magnetization = potential_.effective_magnetic_field();

    // other
    Unit_cell &unit_cell = ctx_.unit_cell();

    splindex<block> spl_ngv(veff_full->gvec().num_gvec(), ctx_.comm().size(), ctx_.comm().rank());

    mdarray<double,2> forces(3, unit_cell.num_atoms());

    forces.zero();

    double reduce_g_fact = veff_full->gvec().reduced() ? 2.0 : 1.0 ;

    // iterate over atoms
    #pragma omp parallel for
    for (int ia = 0; ia < unit_cell.num_atoms(); ia++)
    {
        Atom &atom = unit_cell.atom(ia);

        int iat = atom.type_id();

        for (int igloc = 0; igloc < spl_ngv.local_size(); igloc++)
        {
            int ig = spl_ngv[igloc];

            // fractional form for calculation of scalar product with atomic position
            // since atomic positions are stored in fractional coords
            vector3d<int> gvec = veff_full->gvec().gvec(ig);

            // cartesian form for getting cartesian force components
            vector3d<double> gvec_cart = veff_full->gvec().gvec_cart(ig);

            // scalar part of a force without multipying by G-vector and Qij
            // omega * V_conj(G) * exp(-i G Rn)
            double_complex g_atom_part =  reduce_g_fact * ctx_.unit_cell().omega() * std::conj(veff_full->f_pw_local(ig)) *
                    std::exp(double_complex(0.0, - twopi * (gvec * atom.position())));

            const Augmentation_operator &aug_op = ctx_.augmentation_op(iat);

            // iterate over trangle matrix Qij
            for (int ib2 = 0; ib2 < atom.type().indexb().size(); ib2++)
            {
                for(int ib1 = 0; ib1 <= ib2; ib1++)
                {
                    int iqij = (ib2 * (ib2 + 1)) / 2 + ib1;

                    double diag_fact = ib1 == ib2 ? 1.0 : 2.0;

                    //  [omega * V_conj(G) * exp(-i G Rn) ] * rho_ij * Qij(G)
                    double_complex z = diag_fact * g_atom_part * density_matrix(ib1,ib2,0,ia).real() *
                            double_complex( aug_op.q_pw( iqij , 2*igloc ), aug_op.q_pw( iqij , 2*igloc + 1 ) );

                    // get force components multiplying by cartesian G-vector ( -image part goes from formula)
                    forces(0, ia) -=  (gvec_cart[0] * z).imag();
                    forces(1, ia) -=  (gvec_cart[1] * z).imag();
                    forces(2, ia) -=  (gvec_cart[2] * z).imag();
                }
            }
        }
    }

    ctx_.comm().allreduce(&forces(0,0),forces.size());

    return std::move(forces);
}




//---------------------------------------------------------------
//---------------------------------------------------------------
mdarray<double,2> Forces_PS::calc_nonlocal_forces(K_set& kset) const
{
    Unit_cell &unit_cell = ctx_.unit_cell();

    mdarray<double,2> forces(3, unit_cell.num_atoms());

    forces.zero();

    for(int ikp=0; ikp < kset.num_kpoints(); ikp++)
    {
        K_point *kp = kset.k_point(ikp);

        add_k_point_contribution_to_nonlocal<double_complex>(*kp, forces);
    }

    ctx_.comm().allreduce(&forces(0,0),forces.size());

    //return std::move(forces);
    return std::move( symmetrize_forces(forces) );
}




//---------------------------------------------------------------
//---------------------------------------------------------------
mdarray<double, 2> Forces_PS::symmetrize_forces(mdarray<double,2>& forces) const
{
    mdarray<double, 2> symm_lat_forces(forces.size(0), forces.size(1));

    symm_lat_forces.zero();

    matrix3d<double> const& lattice_vectors = ctx_.unit_cell().symmetry().lattice_vectors();
    matrix3d<double> const& inverse_lattice_vectors = ctx_.unit_cell().symmetry().inverse_lattice_vectors();

    #pragma omp parallel for
    for(int ia = 0; ia < (int)forces.size(1); ia++)
    {
        vector3d<double> cart_force(&forces(0,ia));

        vector3d<double> lat_force = inverse_lattice_vectors * (cart_force / (double)ctx_.unit_cell().symmetry().num_mag_sym());

        for (int isym = 0; isym < ctx_.unit_cell().symmetry().num_mag_sym(); isym++)
        {
            int ja = ctx_.unit_cell().symmetry().sym_table(ia,isym);

            auto &R = ctx_.unit_cell().symmetry().magnetic_group_symmetry(isym).spg_op.R;

            vector3d<double> rot_force = lattice_vectors * ( R * lat_force );

            #pragma omp atomic update
            symm_lat_forces(0, ja) += rot_force[0];

            #pragma omp atomic update
            symm_lat_forces(1, ja) += rot_force[1];

            #pragma omp atomic update
            symm_lat_forces(2, ja) += rot_force[2];
        }
    }

    return std::move(symm_lat_forces);
}

}
