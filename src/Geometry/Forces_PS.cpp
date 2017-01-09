/*
 * Forces_PS.cpp
 *
 *  Created on: Oct 6, 2016
 *      Author: isivkov
 */

#include "Forces_PS.h"
#include "../k_point_set.h"

namespace sirius
{

//---------------------------------------------------------------
//---------------------------------------------------------------
void Forces_PS::calc_local_forces(mdarray<double,2>& forces)
{
    PROFILE("sirius::Forces_PS::calc_local_forces");

    // get main arrays
    const Periodic_function<double>* valence_rho = density_.rho();

    const mdarray<double, 2>& vloc_radial_integrals = potential_.get_vloc_radial_integrals();

    // other
    Unit_cell &unit_cell = ctx_.unit_cell();

    Gvec const& gvecs = ctx_.gvec();

    int gvec_count = gvecs.gvec_count(ctx_.comm().rank());
    int gvec_offset = gvecs.gvec_offset(ctx_.comm().rank());

    //mdarray<double_complex, 2> vloc_G_comp(unit_cell.num_atoms(), spl_ngv.local_size() );

    if (forces.size(0) != 3 || (int)forces.size(1) != unit_cell.num_atoms()) {
        TERMINATE("forces array has wrong number of elements");
    }

    forces.zero();

    double fact = valence_rho->gvec().reduced() ? 2.0 : 1.0 ;

    // here the calculations are in lattice vectors space
    #pragma omp parallel for
    for (int ia = 0; ia < unit_cell.num_atoms(); ia++)
    {
        Atom &atom = unit_cell.atom(ia);

        int iat = atom.type_id();

        // mpi distributed
        for (int igloc = 0; igloc < gvec_count; igloc++)
        {
            int ig = gvec_offset + igloc;

            int igs = gvecs.shell(ig);

            // fractional form for calculation of scalar product with atomic position
            // since atomic positions are stored in fractional coords
            sddk::vector3d<int> gvec = gvecs.gvec(ig);

            // cartesian form for getting cartesian force components
            sddk::vector3d<double> gvec_cart = gvecs.gvec_cart(ig);

            // scalar part of a force without multipying by G-vector
            double_complex z = fact * fourpi * vloc_radial_integrals(iat, igs) * std::conj( valence_rho->f_pw(ig) ) *
                    std::exp(double_complex(0.0, - twopi * (gvec * atom.position())));

            // get force components multiplying by cartesian G-vector ( -image part goes from formula)
            forces(0, ia) -= (gvec_cart[0] * z).imag();
            forces(1, ia) -= (gvec_cart[1] * z).imag();
            forces(2, ia) -= (gvec_cart[2] * z).imag();
        }
    }

    ctx_.comm().allreduce(&forces(0,0), static_cast<int>(forces.size()));
}



//---------------------------------------------------------------
//---------------------------------------------------------------
void Forces_PS::calc_nlcc_forces(mdarray<double,2>& forces)
{
    PROFILE("sirius::Forces_PS::calc_nlcc_force");

    // get main arrays
    Periodic_function<double>* xc_pot = potential_.xc_potential();

    // check because it is not allocated in dft loop
    if ( !xc_pot->is_f_pw_allocated() )
    {
        xc_pot->allocate_pw();
    }

    // transform from real space to reciprocal
    xc_pot->fft().prepare(xc_pot->gvec().partition() );
    xc_pot->fft_transform(-1);
    xc_pot->fft().dismiss( );

    const mdarray<double, 2>&  rho_core_radial_integrals = density_.rho_pseudo_core_radial_integrals();

    Unit_cell &unit_cell = ctx_.unit_cell();

    Gvec const& gvecs = ctx_.gvec();

    int gvec_count = gvecs.gvec_count(ctx_.comm().rank());
    int gvec_offset = gvecs.gvec_offset(ctx_.comm().rank());

    forces.zero();

    double fact = gvecs.reduced() ? 2.0 : 1.0 ;

    // here the calculations are in lattice vectors space
    #pragma omp parallel for
    for (int ia = 0; ia < unit_cell.num_atoms(); ia++)
    {
        Atom &atom = unit_cell.atom(ia);

        int iat = atom.type_id();

        // mpi distributed
        for (int igloc = 0; igloc < gvec_count; igloc++)
        {
            int ig = gvec_offset + igloc;

            int igs = gvecs.shell(ig);

            // fractional form for calculation of scalar product with atomic position
            // since atomic positions are stored in fractional coords
            sddk::vector3d<int> gvec = gvecs.gvec(ig);

            // cartesian form for getting cartesian force components
            sddk::vector3d<double> gvec_cart = gvecs.gvec_cart(ig);

            // scalar part of a force without multipying by G-vector
            double_complex z = fact * fourpi * rho_core_radial_integrals(iat, igs) * std::conj( xc_pot->f_pw(ig) ) *
                    std::exp(double_complex(0.0, - twopi * (gvec * atom.position())));

            // get force components multiplying by cartesian G-vector ( -image part goes from formula)
            forces(0, ia) -= (gvec_cart[0] * z).imag();
            forces(1, ia) -= (gvec_cart[1] * z).imag();
            forces(2, ia) -= (gvec_cart[2] * z).imag();
        }
    }

    ctx_.comm().allreduce(&forces(0,0), static_cast<int>(forces.size()));

}


//---------------------------------------------------------------
//---------------------------------------------------------------
void Forces_PS::calc_ultrasoft_forces(mdarray<double,2>& forces)
{
    PROFILE("sirius::Forces_PS::calc_ultrasoft_forces");

    // get main arrays
    const mdarray<double_complex, 4> &density_matrix = density_.density_matrix();

    const Periodic_function<double> *veff_full = potential_.effective_potential();

    // other
    Unit_cell &unit_cell = ctx_.unit_cell();

    Gvec const& gvecs = ctx_.gvec();

    int gvec_count = gvecs.gvec_count(ctx_.comm().rank());
    int gvec_offset = gvecs.gvec_offset(ctx_.comm().rank());

    forces.zero();

    double reduce_g_fact = veff_full->gvec().reduced() ? 2.0 : 1.0 ;

    // iterate over atoms
    #pragma omp parallel for
    for (int ia = 0; ia < unit_cell.num_atoms(); ia++)
    {
        Atom &atom = unit_cell.atom(ia);

        int iat = atom.type_id();

        for (int igloc = 0; igloc < gvec_count; igloc++)
        {
            int ig = gvec_offset + igloc;

            // fractional form for calculation of scalar product with atomic position
            // since atomic positions are stored in fractional coords
            sddk::vector3d<int> gvec = gvecs.gvec(ig);

            // cartesian form for getting cartesian force components
            sddk::vector3d<double> gvec_cart = gvecs.gvec_cart(ig);

            // scalar part of a force without multipying by G-vector and Qij
            // omega * V_conj(G) * exp(-i G Rn)
            double_complex g_atom_part =  reduce_g_fact * ctx_.unit_cell().omega() * std::conj(veff_full->f_pw(ig)) *
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

    ctx_.comm().allreduce(&forces(0,0), static_cast<int>(forces.size()));
}



//---------------------------------------------------------------
//---------------------------------------------------------------
void Forces_PS::calc_nonlocal_forces(mdarray<double,2>& forces)
{
    PROFILE("sirius::Forces_PS::calc_nonlocal_forces");

    mdarray<double, 2> unsym_forces( forces.size(0), forces.size(1));

    unsym_forces.zero();
    forces.zero();

    auto& spl_num_kp = kset_.spl_num_kpoints();

    for(int ikploc=0; ikploc < spl_num_kp.local_size() ; ikploc++)
    {
        K_point *kp = kset_.k_point(spl_num_kp[ikploc]);

        add_k_point_contribution_to_nonlocal2<double_complex>(*kp, unsym_forces);
    }

    ctx_.comm().allreduce(&unsym_forces(0,0), static_cast<int>(unsym_forces.size()));

    symmetrize_forces(unsym_forces, forces);
}


//-----------------------------------------
// for omp reduction
//------------------------------------------
template<typename T>
void init_mdarray2d(mdarray<T,2> &priv, mdarray<T,2> &orig )
{
    priv = mdarray<double,2>(orig.size(0),orig.size(1)); priv.zero();
}

template<typename T>
void add_mdarray2d(mdarray<T,2> &in, mdarray<T,2> &out)
{
    for(size_t i = 0; i < in.size(1); i++ ) {
        for(size_t j = 0; j < in.size(0); j++ ) {
            out(j,i) += in(j,i);
        }
    }
}

//---------------------------------------------------------------
//---------------------------------------------------------------
void Forces_PS::calc_ewald_forces(mdarray<double,2>& forces)
{
    PROFILE("sirius::Forces_PS::calc_ewald_forces");

    Unit_cell &unit_cell = ctx_.unit_cell();

    forces.zero();

    #pragma omp declare reduction( + : mdarray<double,2> : add_mdarray2d(omp_in, omp_out))  initializer( init_mdarray2d(omp_priv, omp_orig) )


    // 1 / ( 2 sigma^2 )
    double alpha = 1.5;

    double prefac = (ctx_.gvec().reduced() ? 4.0 : 2.0) * (twopi / unit_cell.omega());

    //mpi
    #pragma omp parallel for reduction( + : forces )
    for (int igloc = 0; igloc < ctx_.gvec_count(); igloc++)
    {
        int ig = ctx_.gvec_offset() + igloc;

        if( ig == 0 )
        {
            continue;
        }

        double g2 = std::pow(ctx_.gvec().shell_len(ctx_.gvec().shell(ig)), 2);

        // cartesian form for getting cartesian force components
        sddk::vector3d<double> gvec_cart = ctx_.gvec().gvec_cart(ig);

        double_complex rho(0, 0);

        for (int ja = 0; ja < unit_cell.num_atoms(); ja++)
        {
            rho += ctx_.gvec_phase_factor(ig, ja) * static_cast<double>(unit_cell.atom(ja).zn());
        }

        rho = std::conj(rho);

        for (int ja = 0; ja < unit_cell.num_atoms(); ja++)
        {
            double scalar_part = prefac * (rho * ctx_.gvec_phase_factor(ig, ja)).imag() *
                    static_cast<double>(unit_cell.atom(ja).zn()) * std::exp(-g2 / (4 * alpha) ) / g2;

            for(int x: {0,1,2})
            {
                forces(x,ja) += scalar_part * gvec_cart[x];
            }
        }
    }

    ctx_.comm().allreduce(&forces(0,0), static_cast<int>(forces.size()));


    double invpi = 1. / pi;

    #pragma omp parallel for
    for (int ia = 0; ia < unit_cell.num_atoms(); ia++)
    {
        for (int i = 1; i < unit_cell.num_nearest_neighbours(ia); i++)
        {
            int ja = unit_cell.nearest_neighbour(i, ia).atom_id;

            double d = unit_cell.nearest_neighbour(i, ia).distance;

            double d2 = d*d;

            sddk::vector3d<double> t = unit_cell.lattice_vectors() * unit_cell.nearest_neighbour(i, ia).translation;

            double scalar_part = static_cast<double>(unit_cell.atom(ia).zn() * unit_cell.atom(ja).zn()) / d2 *
                          ( gsl_sf_erfc(std::sqrt(alpha) * d) / d  +  2.0 * std::sqrt(alpha * invpi ) * std::exp( - d2 * alpha ) );


            for(int x: {0,1,2})
            {
                forces(x,ia) += scalar_part * t[x];
            }
        }
    }
}


//---------------------------------------------------------------
//---------------------------------------------------------------
void Forces_PS::symmetrize_forces(mdarray<double,2>& unsym_forces, mdarray<double,2>& sym_forces )
{
    sddk::matrix3d<double> const& lattice_vectors = ctx_.unit_cell().symmetry().lattice_vectors();
    sddk::matrix3d<double> const& inverse_lattice_vectors = ctx_.unit_cell().symmetry().inverse_lattice_vectors();

    sym_forces.zero();

    #pragma omp parallel for
    for(int ia = 0; ia < (int)unsym_forces.size(1); ia++)
    {
        sddk::vector3d<double> cart_force(&unsym_forces(0,ia) );

        sddk::vector3d<double> lat_force = inverse_lattice_vectors * (cart_force / (double)ctx_.unit_cell().symmetry().num_mag_sym());

        for (int isym = 0; isym < ctx_.unit_cell().symmetry().num_mag_sym(); isym++)
        {
            int ja = ctx_.unit_cell().symmetry().sym_table(ia,isym);

            auto &R = ctx_.unit_cell().symmetry().magnetic_group_symmetry(isym).spg_op.R;

            sddk::vector3d<double> rot_force = lattice_vectors * ( R * lat_force );

            #pragma omp atomic update
            sym_forces(0, ja) += rot_force[0];

            #pragma omp atomic update
            sym_forces(1, ja) += rot_force[1];

            #pragma omp atomic update
            sym_forces(2, ja) += rot_force[2];
        }
    }
}


//---------------------------------------------------------------
//---------------------------------------------------------------
void Forces_PS::calc_forces_contributions()
{
    calc_local_forces(local_forces_);
    calc_ultrasoft_forces(ultrasoft_forces_);
    calc_nonlocal_forces(nonlocal_forces_);
    calc_nlcc_forces(nlcc_forces_);
    calc_ewald_forces(ewald_forces_);
}


//---------------------------------------------------------------
//---------------------------------------------------------------
mdarray<double,2> Forces_PS::sum_forces()
{
    mdarray<double,2> total_forces(3, ctx_.unit_cell().num_atoms());

    sum_forces(total_forces);

    return std::move(total_forces);
}



void Forces_PS::sum_forces(mdarray<double,2>& inout_total_forces)
{
    if(inout_total_forces.size() != local_forces_.size())
    {
        TERMINATE("ERROR: Passed total forces array has wrong length!");
    }

    mdarray<double,2> total_forces(3, ctx_.unit_cell().num_atoms());

    #pragma omp parallel for
    for(size_t i = 0; i < inout_total_forces.size(); i++ ) {
        total_forces[i] = local_forces_[i] + ultrasoft_forces_[i] + nonlocal_forces_[i] + nlcc_forces_[i] + ewald_forces_[i];
    }

    symmetrize_forces(total_forces, inout_total_forces);
}

}
