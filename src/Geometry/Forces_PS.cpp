/*
 * Forces_PS.cpp
 *
 *  Created on: Oct 6, 2016
 *      Author: isivkov
 */

#include "Forces_PS.h"


namespace sirius
{

mdarray<double,2> Forces_PS::calc_local_forces(const Periodic_function<double>& valence_rho, const mdarray<double, 2>& vloc_radial_integrals) const
{
    Unit_cell &unit_cell = ctx_.unit_cell();

    double fourpi_3o2 = fourpi * std::sqrt(fourpi);

    splindex<block> spl_ngv(valence_rho.gvec().num_gvec(), ctx_.comm().size(), ctx_.comm().rank());

    //mdarray<double_complex, 2> vloc_G_comp(unit_cell.num_atoms(), spl_ngv.local_size() );

    mdarray<double,2> forces(3, unit_cell.num_atoms());

    forces.zero();

    // here the calculations are in lattice vectors space
    #pragma omp parallel for
    for (int igloc = 0; igloc < spl_ngv.local_size(); igloc++)
    {
        int ig = spl_ngv[igloc];

        int igs = valence_rho.gvec().shell(ig);



        // fractional form for calculation of scalar product with atomic position
        // since atomic positions are stored in fractional coords
        vector3d<int> gvec = valence_rho.gvec().gvec(ig);

        // cartesian form for getting cartesian force components
        vector3d<double> gvec_cart = valence_rho.gvec().gvec_cart(ig);

        // store conj(rho_G) * 4 * pi
        double_complex g_dependent_prefactor =  std::conj( valence_rho.f_pw_local(igloc) ) * fourpi ;

        for (int ia = 0; ia < unit_cell.num_atoms(); ia++)
        {
            Atom &atom = unit_cell.atom(ia);

            int iat = atom.type_id();

            // scalar part of a force without multipying by G-vector
            double_complex z = vloc_radial_integrals(iat, igs) * g_dependent_prefactor *
                    std::exp(double_complex(0.0, - twopi * (gvec * atom.position())));

            // get force components multiplying by cartesian G-vector ( -image part goes from formula)
            forces(0, ia) += - ((double)gvec[0] * z).imag();
            forces(1, ia) += - ((double)gvec[1] * z).imag();
            forces(2, ia) += - ((double)gvec[2] * z).imag();
        }
    }

    for (int iat = 0; iat < unit_cell.num_atom_types(); iat++)
    {
        for(int igs=0; igs<valence_rho.gvec().num_shells(); igs++)
        {
            std::cout<<vloc_radial_integrals(iat, igs)<<std::endl;
        }
    }

    ctx_.comm().allreduce(&forces(0,0),forces.size());

    return std::move(forces);
}

}
