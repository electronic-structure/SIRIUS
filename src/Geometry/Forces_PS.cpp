/*
 * Forces_PS.cpp
 *
 *  Created on: Oct 6, 2016
 *      Author: isivkov
 */

#include "Forces_PS.h"


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

        for (int ia = 0; ia < unit_cell.num_atoms(); ia++)
        {
            Atom &atom = unit_cell.atom(ia);

            int iat = atom.type_id();

            // scalar part of a force without multipying by G-vector
            double_complex z = vloc_radial_integrals(iat, igs) * g_dependent_prefactor *
                    std::exp(double_complex(0.0, - twopi * (gvec * atom.position())));

            // get force components multiplying by cartesian G-vector ( -image part goes from formula)
            #pragma omp atomic
            forces(0, ia) += - (gvec_cart[0] * z).imag();

            #pragma omp atomic
            forces(1, ia) += - (gvec_cart[1] * z).imag();

            #pragma omp atomic
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

    // here the calculations are in lattice vectors space
    #pragma omp parallel for
    for (int igloc = 0; igloc < spl_ngv.local_size(); igloc++)
    {
        int ig = spl_ngv[igloc];

        // fractional form for calculation of scalar product with atomic position
        // since atomic positions are stored in fractional coords
        vector3d<int> gvec = veff_full->gvec().gvec(ig);

        // cartesian form for getting cartesian force components
        vector3d<double> gvec_cart = veff_full->gvec().gvec_cart(ig);

        // store conjugate of g component of veff
        double_complex veff_of_g = reduce_g_fact * std::conj(veff_full->f_pw_local(ig));

        // iterate over atoms
        for (int ia = 0; ia < unit_cell.num_atoms(); ia++)
        {
            Atom &atom = unit_cell.atom(ia);

            int iat = atom.type_id();

            // scalar part of a force without multipying by G-vector and Qij
            double_complex g_atom_part =  ctx_.unit_cell().omega() * veff_of_g * std::exp(double_complex(0.0, - twopi * (gvec * atom.position())));

            const Augmentation_operator &aug_op = ctx_.augmentation_op(iat);

            // iterate over trangle matrix Qij
            for (int ib2 = 0; ib2 < atom.type().indexb().size(); ib2++)
            {
                for(int ib1 = 0; ib1 <= ib2; ib1++)
                {
                    int iqij = (ib2 * (ib2 + 1)) / 2 + ib1;

                    double diag_fact = ib1 == ib2 ? 1.0 : 2.0;

                    // scalar part of force
                    double_complex z = diag_fact * density_matrix(ib1,ib2,0,ia) * g_atom_part *
                            double_complex( aug_op.q_pw( iqij , 2*igloc ), aug_op.q_pw( iqij , 2*igloc + 1 ) );

                    // get force components multiplying by cartesian G-vector ( -image part goes from formula)
                    #pragma omp atomic
                    forces(0, ia) += - (gvec_cart[0] * z).imag();

                    #pragma omp atomic
                    forces(1, ia) += - (gvec_cart[1] * z).imag();

                    #pragma omp atomic
                    forces(2, ia) += - (gvec_cart[2] * z).imag();
                }
            }
        }
    }

    ctx_.comm().allreduce(&forces(0,0),forces.size());


    return std::move(forces);
}

}
