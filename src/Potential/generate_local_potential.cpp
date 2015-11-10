#include "potential.h"

namespace sirius {

void Potential::generate_local_potential()
{
    PROFILE();

    Timer t("sirius::Potential::generate_local_potential");

    auto rl = ctx_.reciprocal_lattice();

    mdarray<double, 2> vloc_radial_integrals(unit_cell_.num_atom_types(), ctx_.gvec().num_shells());

    /* split G-shells between MPI ranks */
    splindex<block> spl_gshells(ctx_.gvec().num_shells(), comm_.size(), comm_.rank());

    #pragma omp parallel
    {
        /* splines for all atom types */
        std::vector< Spline<double> > sa(unit_cell_.num_atom_types());
        
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) sa[iat] = Spline<double>(unit_cell_.atom_type(iat)->radial_grid());
    
        #pragma omp for
        for (int igsloc = 0; igsloc < (int)spl_gshells.local_size(); igsloc++)
        {
            int igs = (int)spl_gshells[igsloc];

            for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
            {
                auto atom_type = unit_cell_.atom_type(iat);

                if (igs == 0)
                {
                    for (int ir = 0; ir < atom_type->num_mt_points(); ir++) 
                    {
                        double x = atom_type->radial_grid(ir);
                        sa[iat][ir] = (x * atom_type->uspp().vloc[ir] + atom_type->zn()) * x;
                    }
                    vloc_radial_integrals(iat, igs) = sa[iat].interpolate().integrate(0);
                }
                else
                {
                    double g = ctx_.gvec().shell_len(igs);
                    double g2 = std::pow(g, 2);
                    for (int ir = 0; ir < atom_type->num_mt_points(); ir++) 
                    {
                        double x = atom_type->radial_grid(ir);
                        sa[iat][ir] = (x * atom_type->uspp().vloc[ir] + atom_type->zn() * gsl_sf_erf(x)) * sin(g * x);
                    }
                    vloc_radial_integrals(iat, igs) = (sa[iat].interpolate().integrate(0) / g - atom_type->zn() * exp(-g2 / 4) / g2);
                }
            }
        }
    }

    int ld = unit_cell_.num_atom_types();
    comm_.allgather(vloc_radial_integrals.at<CPU>(), static_cast<int>(ld * spl_gshells.global_offset()), 
                    static_cast<int>(ld * spl_gshells.local_size()));

    auto v = rl->make_periodic_function(vloc_radial_integrals, ctx_.gvec().num_gvec());
    STOP();
    //fft_->input(ctx_.gvec().num_gvec_loc(), ctx_.gvec().index_map(), &v[ctx_.gvec().gvec_offset()]); 
    //fft_->transform(1, ctx_.gvec().z_sticks_coord());
    //fft_->output(&local_potential_->f_it(0));
}

};
