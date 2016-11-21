#include "potential.h"

namespace sirius {

void Potential::generate_local_potential()
{
    PROFILE_WITH_TIMER("sirius::Potential::generate_local_potential");

    vloc_radial_integrals_ = mdarray<double, 2>(unit_cell_.num_atom_types(), ctx_.gvec().num_shells());

    /* split G-shells between MPI ranks */
    splindex<block> spl_gshells(ctx_.gvec().num_shells(), comm_.size(), comm_.rank());

    #pragma omp parallel
    {
        /* splines for all atom types */
        std::vector< Spline<double> > sa(unit_cell_.num_atom_types());
        
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
            sa[iat] = Spline<double>(unit_cell_.atom_type(iat).radial_grid());
    
        #pragma omp for
        for (int igsloc = 0; igsloc < spl_gshells.local_size(); igsloc++)
        {
            int igs = spl_gshells[igsloc];

            for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
            {
                auto& atom_type = unit_cell_.atom_type(iat);

                if (igs == 0)
                {
                    for (int ir = 0; ir < atom_type.num_mt_points(); ir++) 
                    {
                        double x = atom_type.radial_grid(ir);
                        sa[iat][ir] = (x * atom_type.pp_desc().vloc[ir] + atom_type.zn()) * x;
                    }
                    vloc_radial_integrals_(iat, igs) = sa[iat].interpolate().integrate(0);
                }
                else
                {
                    double g = ctx_.gvec().shell_len(igs);
                    double g2 = std::pow(g, 2);
                    for (int ir = 0; ir < atom_type.num_mt_points(); ir++) 
                    {
                        double x = atom_type.radial_grid(ir);
                        sa[iat][ir] = (x * atom_type.pp_desc().vloc[ir] + atom_type.zn() * gsl_sf_erf(x)) * std::sin(g * x);
                    }
                    vloc_radial_integrals_(iat, igs) = (sa[iat].interpolate().integrate(0) / g - atom_type.zn() * std::exp(-g2 / 4) / g2);
                }
            }
        }
    }

    int ld = unit_cell_.num_atom_types();
    comm_.allgather(vloc_radial_integrals_.at<CPU>(), ld * spl_gshells.global_offset(), ld * spl_gshells.local_size());

    auto v = unit_cell_.make_periodic_function(vloc_radial_integrals_, ctx_.gvec());
    ctx_.fft().prepare(ctx_.gvec().partition());
    ctx_.fft().transform<1>(ctx_.gvec().partition(), &v[ctx_.gvec().partition().gvec_offset_fft()]);
    ctx_.fft().output(&local_potential_->f_rg(0));
    ctx_.fft().dismiss();
}

};
