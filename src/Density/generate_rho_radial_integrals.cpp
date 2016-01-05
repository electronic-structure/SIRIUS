#include "density.h"

namespace sirius
{

/** type = 0: full-potential radial integrals \n
 *  type = 1: pseudopotential valence density integrals \n
 *  type = 2: pseudopotential code density integrals
 */
mdarray<double, 2> Density::generate_rho_radial_integrals(int type__)
{
    Timer t("sirius::Density::generate_rho_radial_integrals");

    mdarray<double, 2> rho_radial_integrals(unit_cell_.num_atom_types(), ctx_.gvec().num_shells());

    /* split G-shells between MPI ranks */
    splindex<block> spl_gshells(ctx_.gvec().num_shells(), ctx_.comm().size(), ctx_.comm().rank());

    if (type__ == 4)
    {
        /* rho[r_] := Z*b^3/8/Pi*Exp[-b*r]
         * Integrate[Sin[G*r]*rho[r]*r*r/(G*r), {r, 0, \[Infinity]}, Assumptions -> {G >= 0, b > 0}]
         * Out[] = (b^4 Z)/(4 (b^2 + G^2)^2 \[Pi])
         */
        double b = 4;
        for (int igs = 0; igs < ctx_.gvec().num_shells(); igs++)
        {
            double G = ctx_.gvec().shell_len(igs);

            for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) 
            {
                rho_radial_integrals(iat, igs) = std::pow(b, 4) * unit_cell_.atom_type(iat).zn() / (4 * std::pow(std::pow(b, 2) + std::pow(G, 2), 2) * pi);
            }
        }
        return rho_radial_integrals;
    }
    #pragma omp parallel
    {
        /* splines for all atom types */
        std::vector< Spline<double> > sa(unit_cell_.num_atom_types());
        
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) 
        {
            /* full potential radial integrals requre a free atom grid */
            if (type__ == 0) 
            {
                sa[iat] = Spline<double>(unit_cell_.atom_type(iat).free_atom_radial_grid());
            }
            else
            {
                sa[iat] = Spline<double>(unit_cell_.atom_type(iat).radial_grid());
            }
        }
        
        /* spherical Bessel functions */
        sbessel_pw<double> jl(unit_cell_, 0);

        #pragma omp for
        for (int igsloc = 0; igsloc < (int)spl_gshells.local_size(); igsloc++)
        {
            int igs = (int)spl_gshells[igsloc];

            /* for pseudopotential valence or core charge density */
            if (type__ == 1 || type__ == 2) jl.load(ctx_.gvec().shell_len(igs));

            for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
            {
                auto& atom_type = unit_cell_.atom_type(iat);

                if (type__ == 0)
                {
                    if (igs == 0)
                    {
                        for (int ir = 0; ir < sa[iat].num_points(); ir++) sa[iat][ir] = atom_type.free_atom_density(ir);
                        rho_radial_integrals(iat, igs) = sa[iat].interpolate().integrate(2);
                    }
                    else
                    {
                        double G = ctx_.gvec().shell_len(igs);
                        for (int ir = 0; ir < sa[iat].num_points(); ir++) 
                        {
                            sa[iat][ir] = atom_type.free_atom_density(ir) *
                                          std::sin(G * atom_type.free_atom_radial_grid(ir)) / G;
                        }
                        rho_radial_integrals(iat, igs) = sa[iat].interpolate().integrate(1);
                    }
                }

                if (type__ == 1)
                {
                    for (int ir = 0; ir < sa[iat].num_points(); ir++) 
                        sa[iat][ir] = jl(ir, 0, iat) * atom_type.uspp().total_charge_density[ir];
                    rho_radial_integrals(iat, igs) = sa[iat].interpolate().integrate(0) / fourpi;
                }

                if (type__ == 2)
                {
                    for (int ir = 0; ir < sa[iat].num_points(); ir++) 
                        sa[iat][ir] = jl(ir, 0, iat) * atom_type.uspp().core_charge_density[ir];
                    rho_radial_integrals(iat, igs) = sa[iat].interpolate().integrate(2);
                }
            }
        }
    }

    int ld = unit_cell_.num_atom_types();
    ctx_.comm().allgather(rho_radial_integrals.at<CPU>(), static_cast<int>(ld * spl_gshells.global_offset()), 
                          static_cast<int>(ld * spl_gshells.local_size()));

    return rho_radial_integrals;
}

};
