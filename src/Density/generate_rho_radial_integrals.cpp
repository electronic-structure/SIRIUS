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

    auto rl = parameters_.reciprocal_lattice();
    auto uc = parameters_.unit_cell();

    mdarray<double, 2> rho_radial_integrals(uc->num_atom_types(), rl->num_gvec_shells_inner());

    /* split G-shells between MPI ranks */
    splindex<block> spl_gshells(rl->num_gvec_shells_inner(), parameters_.comm().size(), parameters_.comm().rank());

    if (type__ == 5)
    {
        #ifdef _PRINT_OBJECT_CHECKSUM_
        for (int iat = 0; iat < uc->num_atom_types(); iat++)
        {
            DUMP("iat: %i, checksum(radial_grid): %18.10f %18.10f", iat, uc->atom_type(iat)->radial_grid().x().checksum(),
                                                                    uc->atom_type(iat)->radial_grid().dx().checksum());
        }
        #endif
        #ifdef _PRINT_OBJECT_HASH_
        for (int iat = 0; iat < uc->num_atom_types(); iat++)
        {
            DUMP("iat: %i, hash(radial_grid): %16llX", iat, uc->atom_type(iat)->radial_grid().hash());
            DUMP("iat: %i, hash(free_atom_radial_grid): %16llX", iat, uc->atom_type(iat)->free_atom_radial_grid().hash());
        }
        #endif

        double b = 8;
        for (int igs = 0; igs < rl->num_gvec_shells_inner(); igs++)
        {
            double G = rl->gvec_shell_len(igs);

            for (int iat = 0; iat < uc->num_atom_types(); iat++) 
            {
                int Z = uc->atom_type(iat)->zn();
                double R = uc->atom_type(iat)->mt_radius();
                if (igs != 0)
                {
                    rho_radial_integrals(iat, igs) = (std::pow(b,3)*Z*((std::pow(G,5)*(G*(2*b + (std::pow(b,2) + std::pow(G,2))*R)*std::cos(G*R) +
                                                     (std::pow(b,2) - std::pow(G,2) + b*(std::pow(b,2) + std::pow(G,2))*R)*std::sin(G*R)))/
                                                     std::pow(std::pow(b,2) + std::pow(G,2),2) + (G*(3 + b*R)*(G*R*(6 - std::pow(G,2)*std::pow(R,2))*std::cos(G*R) +
                                                     3*(-2 + std::pow(G,2)*std::pow(R,2))*std::sin(G*R)))/std::pow(R,2) +
                                                     ((2 + b*R)*((24 - 12*std::pow(G,2)*std::pow(R,2) + std::pow(G,4)*std::pow(R,4))*std::cos(G*R) -
                                                     4*(6 + G*R*(-6 + std::pow(G,2)*std::pow(R,2))*std::sin(G*R))))/std::pow(R,3)))/
                                                     (8.*std::exp(b*R)*std::pow(G,6)*pi);
                }
                else
                {
                    rho_radial_integrals(iat, igs) = ((60 + 60*b*R + 30*std::pow(b,2)*std::pow(R,2) + 8*std::pow(b,3)*std::pow(R,3) +
                                                      std::pow(b,4)*std::pow(R,4))*Z)/(240.*std::exp(b*R)*pi);
                }
            }
        }
        #ifdef _PRINT_OBJECT_CHECKSUM_
        DUMP("checksum(rho_radial_integrals): %18.10f", rho_radial_integrals.checksum());
        #endif
        return rho_radial_integrals;
    }

    if (type__ == 4)
    {
        /* rho[r_] := Z*b^3/8/Pi*Exp[-b*r]
         * Integrate[Sin[G*r]*rho[r]*r*r/(G*r), {r, 0, \[Infinity]}, Assumptions -> {G >= 0, b > 0}]
         * Out[] = (b^4 Z)/(4 (b^2 + G^2)^2 \[Pi])
         */
        double b = 4;
        for (int igs = 0; igs < rl->num_gvec_shells_inner(); igs++)
        {
            double G = rl->gvec_shell_len(igs);

            for (int iat = 0; iat < uc->num_atom_types(); iat++) 
            {
                rho_radial_integrals(iat, igs) = std::pow(b, 4) * uc->atom_type(iat)->zn() / (4 * std::pow(std::pow(b, 2) + std::pow(G, 2), 2) * pi);
            }
        }
        return rho_radial_integrals;
    }
    #pragma omp parallel
    {
        /* splines for all atom types */
        std::vector< Spline<double> > sa(uc->num_atom_types());
        
        for (int iat = 0; iat < uc->num_atom_types(); iat++) 
        {
            /* full potential radial integrals requre a free atom grid */
            if (type__ == 0) 
            {
                sa[iat] = Spline<double>(uc->atom_type(iat)->free_atom_radial_grid());
            }
            else
            {
                sa[iat] = Spline<double>(uc->atom_type(iat)->radial_grid());
            }
        }
        
        /* spherical Bessel functions */
        sbessel_pw<double> jl(uc, 0);

        #pragma omp for
        for (int igsloc = 0; igsloc < (int)spl_gshells.local_size(); igsloc++)
        {
            int igs = (int)spl_gshells[igsloc];

            /* for pseudopotential valence or core charge density */
            if (type__ == 1 || type__ == 2) jl.load(rl->gvec_shell_len(igs));

            for (int iat = 0; iat < uc->num_atom_types(); iat++)
            {
                auto atom_type = uc->atom_type(iat);

                if (type__ == 0)
                {
                    if (igs == 0)
                    {
                        for (int ir = 0; ir < sa[iat].num_points(); ir++) sa[iat][ir] = atom_type->free_atom_density(ir);
                        rho_radial_integrals(iat, igs) = sa[iat].interpolate().integrate(2);
                    }
                    else
                    {
                        double G = rl->gvec_shell_len(igs);
                        for (int ir = 0; ir < sa[iat].num_points(); ir++) 
                        {
                            sa[iat][ir] = atom_type->free_atom_density(ir) *
                                          sin(G * atom_type->free_atom_radial_grid(ir)) / G;
                        }
                        rho_radial_integrals(iat, igs) = sa[iat].interpolate().integrate(1);
                    }
                }

                if (type__ == 1)
                {
                    for (int ir = 0; ir < sa[iat].num_points(); ir++) 
                        sa[iat][ir] = jl(ir, 0, iat) * atom_type->uspp().total_charge_density[ir];
                    rho_radial_integrals(iat, igs) = sa[iat].interpolate().integrate(0) / fourpi;
                }

                if (type__ == 2)
                {
                    for (int ir = 0; ir < sa[iat].num_points(); ir++) 
                        sa[iat][ir] = jl(ir, 0, iat) * atom_type->uspp().core_charge_density[ir];
                    rho_radial_integrals(iat, igs) = sa[iat].interpolate().integrate(2);
                }
            }
        }
    }

    int ld = uc->num_atom_types();
    parameters_.comm().allgather(rho_radial_integrals.at<CPU>(), static_cast<int>(ld * spl_gshells.global_offset()), 
                                 static_cast<int>(ld * spl_gshells.local_size()));

    return rho_radial_integrals;
}

};
