// Copyright (c) 2013-2016 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that 
// the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the 
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions 
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file atom_symmetry_class.cpp
 *   
 *  \brief Contains remaining implementation of sirius::Atom_symmetry_class class.
 */

#include "atom_symmetry_class.h"
#include "eigenproblem.h"

namespace sirius {

void Atom_symmetry_class::generate_aw_radial_functions(relativity_t rel__)
{
    int nmtp = atom_type_.num_mt_points();

    Radial_solver solver(atom_type_.zn(), spherical_potential_, atom_type_.radial_grid());

    #pragma omp parallel default(shared)
    {
        Spline<double> s(atom_type_.radial_grid());
        
        std::vector<double> p;
        std::vector<double> rdudr;
        std::array<double, 2> uderiv;
        
        #pragma omp for schedule(dynamic, 1)
        for (int l = 0; l < num_aw_descriptors(); l++) {
            for (int order = 0; order < (int)aw_descriptor(l).size(); order++) {
                auto rsd = aw_descriptor(l)[order];

                int idxrf = atom_type_.indexr().index_by_l_order(l, order);

                solver.solve(rel__, rsd.dme, rsd.l, rsd.enu, p, rdudr, uderiv);

                /* normalize */
                for (int ir = 0; ir < nmtp; ir++) {
                    s[ir] = std::pow(p[ir], 2);
                }
                double norm = 1.0 / std::sqrt(s.interpolate().integrate(0));

                for (int ir = 0; ir < nmtp; ir++) {
                    radial_functions_(ir, idxrf, 0) = p[ir] * norm;
                    radial_functions_(ir, idxrf, 1) = rdudr[ir] * norm;
                }
                aw_surface_derivatives_(order, l, 0) = norm * p.back() / atom_type_.mt_radius();
                for (int i: {0, 1}) {
                    aw_surface_derivatives_(order, l, i + 1) = uderiv[i] * norm;
                }

                /* orthogonalize to previous radial functions */
                for (int order1 = 0; order1 < order; order1++) {
                    int idxrf1 = atom_type_.indexr().index_by_l_order(l, order1);

                    for (int ir = 0; ir < nmtp; ir++) {
                        s[ir] = radial_functions_(ir, idxrf, 0) * radial_functions_(ir, idxrf1, 0);
                    }
                    
                    /* <u_{\nu'}|u_{\nu}> */
                    double ovlp = s.interpolate().integrate(0);

                    for (int ir = 0; ir < nmtp; ir++) {
                        radial_functions_(ir, idxrf, 0) -= radial_functions_(ir, idxrf1, 0) * ovlp;
                        radial_functions_(ir, idxrf, 1) -= radial_functions_(ir, idxrf1, 1) * ovlp;
                    }
                    for (int i: {0, 1, 2}) {
                        aw_surface_derivatives_(order, l, i) -= aw_surface_derivatives_(order1, l, i) * ovlp;
                    }
                }

                /* normalize again */
                for (int ir = 0; ir < nmtp; ir++) {
                    s[ir] = std::pow(radial_functions_(ir, idxrf, 0), 2);
                }
                norm = s.interpolate().integrate(0);

                if (std::abs(norm) < 1e-10) {
                    TERMINATE("aw radial functions are linearly dependent");
                }

                norm = 1.0 / std::sqrt(norm);

                for (int ir = 0; ir < nmtp; ir++) {
                    radial_functions_(ir, idxrf, 0) *= norm;
                    radial_functions_(ir, idxrf, 1) *= norm;
                }
                for (int i: {0, 1, 2}) {
                    aw_surface_derivatives_(order, l, i) *= norm;
                }
            }
            /* divide by r */
            for (int order = 0; order < (int)aw_descriptor(l).size(); order++) {
                int idxrf = atom_type_.indexr().index_by_l_order(l, order);
                for (int ir = 0; ir < nmtp; ir++) {
                    radial_functions_(ir, idxrf, 0) *= atom_type_.radial_grid().x_inv(ir);
                }
            }
        }
    }
}

void Atom_symmetry_class::generate_lo_radial_functions(relativity_t rel__)
{
    int nmtp = atom_type_.num_mt_points();

    Radial_solver solver(atom_type_.zn(), spherical_potential_, atom_type_.radial_grid());
    
    #pragma omp parallel default(shared)
    {
        Spline<double> s(atom_type_.radial_grid());
        
        double a[3][3];

        #pragma omp for schedule(dynamic, 1)
        for (int idxlo = 0; idxlo < num_lo_descriptors(); idxlo++) {
            /* number of radial solutions */
            int num_rs = static_cast<int>(lo_descriptor(idxlo).rsd_set.size());
            assert(num_rs <= 3);

            std::vector<std::vector<double>> p(num_rs);
            std::vector<std::vector<double>> rdudr(num_rs);
            std::array<double, 2> uderiv;

            for (int order = 0; order < num_rs; order++) {
                auto rsd = lo_descriptor(idxlo).rsd_set[order];
                
                solver.solve(rel__, rsd.dme, rsd.l, rsd.enu, p[order], rdudr[order], uderiv);

                /* find norm of the radial solution */
                for (int ir = 0; ir < nmtp; ir++) {
                    s[ir] = std::pow(p[order][ir], 2);
                }
                double norm = 1.0 / std::sqrt(s.interpolate().integrate(0));

                /* normalize radial solution and divide by r */
                for (int ir = 0; ir < nmtp; ir++) {
                    p[order][ir] *= (norm * atom_type_.radial_grid().x_inv(ir));
                    /* don't divide rdudr by r */
                    rdudr[order][ir] *= norm;
                }
                uderiv[0] *= norm;
                uderiv[1] *= norm;

                /* matrix of derivatives */
                a[order][0] = p[order].back();
                a[order][1] = uderiv[0];
                a[order][2] = uderiv[1];
            }

            double b[] = {0, 0, 0};
            b[num_rs - 1] = 1.0;

            int info = linalg<CPU>::gesv(num_rs, 1, &a[0][0], 3, b, 3);

            if (info) {
                std::stringstream s;
                s << "gesv returned " << info;
                TERMINATE(s);
            }
            
            /* index of local orbital radial function */
            int idxrf = atom_type_.indexr().index_by_idxlo(idxlo);
            /* take linear combination of radial solutions */
            for (int order = 0; order < num_rs; order++) {
                for (int ir = 0; ir < nmtp; ir++) {
                    radial_functions_(ir, idxrf, 0) += b[order] * p[order][ir];
                    radial_functions_(ir, idxrf, 1) += b[order] * rdudr[order][ir];
                }
            }

            /* find norm of constructed local orbital */
            for (int ir = 0; ir < nmtp; ir++) {
                s[ir] = std::pow(radial_functions_(ir, idxrf, 0), 2);
            }
            double norm = 1.0 / std::sqrt(s.interpolate().integrate(2));

            /* normalize */
            for (int ir = 0; ir < nmtp; ir++) {
                radial_functions_(ir, idxrf, 0) *= norm;
                radial_functions_(ir, idxrf, 1) *= norm;
            }
            
            if (std::abs(radial_functions_(nmtp - 1, idxrf, 0)) > 1e-10) {
                std::stringstream s;
                s << "local orbital " << idxlo << " is not zero at MT boundary" << std::endl 
                  << "  atom symmetry class id : " << id() << " (" << atom_type().symbol() << ")" << std::endl
                  << "  value : " << radial_functions_(nmtp - 1, idxrf, 0) << std::endl
                  << "  number of MT points: " << nmtp << std::endl
                  << "  MT radius: " << atom_type_.radial_grid().last() << std::endl
                  << "  b_coeffs: ";
                for (int j = 0; j < num_rs; j++) {
                    s << b[j] << " ";
                }
                WARNING(s);
            }
        }
    }
    
    #if (__VERIFICATION > 0)
    if (num_lo_descriptors() > 0) check_lo_linear_independence(0.0001);
    #endif

    //if (verbosity_level > 0) dump_lo();
}

std::vector<int> Atom_symmetry_class::check_lo_linear_independence(double tol__)
{
    int nmtp = atom_type_.num_mt_points();
    
    Spline<double> s(atom_type_.radial_grid());
    mdarray<double, 2> loprod(num_lo_descriptors(), num_lo_descriptors());
    loprod.zero();
    for (int idxlo1 = 0; idxlo1 < num_lo_descriptors(); idxlo1++) {
        
        int idxrf1 = atom_type_.indexr().index_by_idxlo(idxlo1);
        
        for (int idxlo2 = 0; idxlo2 < num_lo_descriptors(); idxlo2++) {
            
            int idxrf2 = atom_type_.indexr().index_by_idxlo(idxlo2);

            if (lo_descriptor(idxlo1).l == lo_descriptor(idxlo2).l) {
            
                for (int ir = 0; ir < nmtp; ir++) {
                    s[ir] = radial_functions_(ir, idxrf1, 0) * radial_functions_(ir, idxrf2, 0);
                }
                loprod(idxlo1, idxlo2) = s.interpolate().integrate(2);
            }
        }
    }
        
    mdarray<double, 2> ovlp(num_lo_descriptors(), num_lo_descriptors());
    loprod >> ovlp;

    Eigenproblem_lapack stdevp;

    std::vector<double> loprod_eval(num_lo_descriptors());
    mdarray<double, 2> loprod_evec(num_lo_descriptors(), num_lo_descriptors());

    stdevp.solve(num_lo_descriptors(), loprod.at<CPU>(), loprod.ld(), &loprod_eval[0], 
                 loprod_evec.at<CPU>(), loprod_evec.ld());

    if (std::abs(loprod_eval[0]) < tol__) {
        printf("\n");
        printf("local orbitals for atom symmetry class %i are almost linearly dependent\n", id_);
        printf("local orbitals overlap matrix:\n");
        for (int i = 0; i < num_lo_descriptors(); i++) {
            for (int j = 0; j < num_lo_descriptors(); j++) {
                printf("%12.6f", ovlp(i, j));
            }
            printf("\n");
        }
        printf("overlap matrix eigen-values:\n");
        for (int i = 0; i < num_lo_descriptors(); i++) {
            printf("%12.6f", loprod_eval[i]);
        }
        printf("\n");
        printf("smallest eigenvalue: %20.16f\n", loprod_eval[0]);
    }

    std::vector<int> inc(num_lo_descriptors(), 0);

    /* try all local orbitals */
    for (int i = 0; i < num_lo_descriptors(); i++) {
        inc[i] = 1;

        std::vector<int> ilo;
        for (int j = 0; j < num_lo_descriptors(); j++) {
            if (inc[j] == 1) {
                ilo.push_back(j);
            }
        }

        std::vector<double> eval(ilo.size());
        mdarray<double, 2> evec(ilo.size(), ilo.size());
        mdarray<double, 2> tmp(ilo.size(), ilo.size());
        for (size_t j1 = 0; j1 < ilo.size(); j1++) {
            for (size_t j2 = 0; j2 < ilo.size(); j2++) {
                tmp(j1, j2) = ovlp(ilo[j1], ilo[j2]);
            }
        }

        stdevp.solve(static_cast<int>(ilo.size()), tmp.at<CPU>(), tmp.ld(), &eval[0], evec.at<CPU>(), evec.ld());

        if (eval[0] < tol__) {
            printf("local orbital %i can be removed\n", i);
            inc[i] = 0;
        }
    }
    return inc;
}

void Atom_symmetry_class::dump_lo()
{
    std::stringstream s;
    s << "local_orbitals_" << id_ << ".dat";
    FILE* fout = fopen(s.str().c_str(), "w");

    for (int ir = 0; ir <atom_type_.num_mt_points(); ir++)
    {
        fprintf(fout, "%f ", atom_type_.radial_grid(ir));
        for (int idxlo = 0; idxlo < num_lo_descriptors(); idxlo++)
        {
            int idxrf = atom_type_.indexr().index_by_idxlo(idxlo);
            fprintf(fout, "%f ", radial_functions_(ir, idxrf, 0));
        }
        fprintf(fout, "\n");
    }
    fclose(fout);
    
    s.str("");
    s << "local_orbitals_h_" << id_ << ".dat";
    fout = fopen(s.str().c_str(), "w");

    for (int ir = 0; ir <atom_type_.num_mt_points(); ir++)
    {
        fprintf(fout, "%f ", atom_type_.radial_grid(ir));
        for (int idxlo = 0; idxlo < num_lo_descriptors(); idxlo++)
        {
            int idxrf = atom_type_.indexr().index_by_idxlo(idxlo);
            fprintf(fout, "%f ", radial_functions_(ir, idxrf, 1));
        }
        fprintf(fout, "\n");
    }
    fclose(fout);
}

void Atom_symmetry_class::initialize()
{
    aw_surface_derivatives_ = mdarray<double, 3>(atom_type_.max_aw_order(), atom_type_.num_aw_descriptors(), 3);

    radial_functions_ = mdarray<double, 3>(atom_type_.num_mt_points(), atom_type_.mt_radial_basis_size(), 2);

    h_spherical_integrals_ = mdarray<double, 2>(atom_type_.mt_radial_basis_size(), atom_type_.mt_radial_basis_size());
    h_spherical_integrals_.zero();
    
    o_radial_integrals_ = mdarray<double, 3>(atom_type_.indexr().lmax() + 1, atom_type_.indexr().max_num_rf(), 
                                             atom_type_.indexr().max_num_rf());
    o_radial_integrals_.zero();
    
    so_radial_integrals_ = mdarray<double, 3>(atom_type_.indexr().lmax() + 1, atom_type_.indexr().max_num_rf(), 
                                              atom_type_.indexr().max_num_rf());
    so_radial_integrals_.zero();

    /* copy descriptors because enu is defferent between atom classes */
    aw_descriptors_.resize(atom_type_.num_aw_descriptors());
    for (int i = 0; i < num_aw_descriptors(); i++) {
        aw_descriptors_[i] = atom_type_.aw_descriptor(i);
    }

    lo_descriptors_.resize(atom_type_.num_lo_descriptors());
    for (int i = 0; i < num_lo_descriptors(); i++) {
        lo_descriptors_[i] = atom_type_.lo_descriptor(i);
    }
    
    core_charge_density_.resize(atom_type_.num_mt_points());
    std::memset(&core_charge_density_[0], 0, atom_type_.num_mt_points() * sizeof(double));
}

void Atom_symmetry_class::set_spherical_potential(std::vector<double> const& vs__)
{
    if (atom_type_.num_mt_points() != (int)vs__.size()) {
        TERMINATE("wrong size of effective potential array");
    }

    spherical_potential_ = vs__;

    //HDF5_tree fout("mt_potential.h5", true);
    //fout.write("potential", spherical_potential_);

    ///* write spherical potential */
    //std::stringstream sstr;
    //sstr << "mt_spheric_potential_" << id_ << ".dat";
    //FILE* fout = fopen(sstr.str().c_str(), "w");

    //for (int ir = 0; ir < atom_type_.num_mt_points(); ir++) {
    //    double r = atom_type_.radial_grid(ir);
    //    fprintf(fout, "%20.10f %20.10f \n", r, spherical_potential_[ir] + atom_type_.zn() / r);
    //}
    //fclose(fout);
}

void Atom_symmetry_class::find_enu(relativity_t rel__)
{
    runtime::Timer t("sirius::Atom_symmetry_class::find_enu");

    std::vector<radial_solution_descriptor*> rs_with_auto_enu;
    
    /* find which aw functions need auto enu */
    for (int l = 0; l < num_aw_descriptors(); l++) {
        for (size_t order = 0; order < aw_descriptor(l).size(); order++) {
            auto& rsd = aw_descriptor(l)[order];
            if (rsd.auto_enu) {
                rs_with_auto_enu.push_back(&rsd);
            }
        }
    }

    /* find which lo functions need auto enu */
    for (int idxlo = 0; idxlo < num_lo_descriptors(); idxlo++) {
        /* number of radial solutions */
        size_t num_rs = lo_descriptor(idxlo).rsd_set.size();

        for (size_t order = 0; order < num_rs; order++) {
            auto& rsd = lo_descriptor(idxlo).rsd_set[order];
            if (rsd.auto_enu) {
                rs_with_auto_enu.push_back(&rsd);
            }
        }
    }

    #pragma omp parallel for
    for (size_t i = 0; i < rs_with_auto_enu.size(); i++) {
        auto rsd = rs_with_auto_enu[i];
        rsd->enu = Enu_finder(rel__, atom_type_.zn(), rsd->n, rsd->l, atom_type_.radial_grid(), spherical_potential_, rsd->enu).enu();
    }
}

void Atom_symmetry_class::generate_radial_functions(relativity_t rel__)
{
    runtime::Timer t("sirius::Atom_symmetry_class::generate_radial_functions");

    radial_functions_.zero();

    find_enu(rel__);

    generate_aw_radial_functions(rel__);

    generate_lo_radial_functions(rel__);

    #ifdef __PRINT_OBJECT_CHECKSUM
    DUMP("checksum(spherical_potential): %18.10f", mdarray<double, 1>(spherical_potential_.data(), atom_type_.num_mt_points()).checksum());
    DUMP("checksum(radial_functions): %18.10f", radial_functions_.checksum());
    #endif
    
    //** if (verbosity_level > 0)
    //** {
    //**     std::stringstream s;
    //**     s << "radial_functions_" << id_ << ".dat";
    //**     FILE* fout = fopen(s.str().c_str(), "w");

    //**     for (int ir = 0; ir <atom_type_.num_mt_points(); ir++)
    //**     {
    //**         fprintf(fout, "%f ", atom_type_.radial_grid(ir));
    //**         for (int idxrf = 0; idxrf < atom_type_.indexr().size(); idxrf++)
    //**         {
    //**             fprintf(fout, "%f ", radial_functions_(ir, idxrf, 0));
    //**         }
    //**         fprintf(fout, "\n");
    //**     }
    //**     fclose(fout);
    //** }
    //** STOP();
}

void Atom_symmetry_class::sync_radial_functions(Communicator const& comm__, int const rank__)
{
    /* don't broadcast Hamiltonian radial functions, because they are used locally */
    int size = (int)(radial_functions_.size(0) * radial_functions_.size(1));
    comm__.bcast(radial_functions_.at<CPU>(), size, rank__);
    comm__.bcast(aw_surface_derivatives_.at<CPU>(), (int)aw_surface_derivatives_.size(), rank__);
    // TODO: sync enu to pass to Exciting / Elk
}

void Atom_symmetry_class::sync_radial_integrals(Communicator const& comm__, int const rank__)
{
    comm__.bcast(h_spherical_integrals_.at<CPU>(), (int)h_spherical_integrals_.size(), rank__);
    comm__.bcast(o_radial_integrals_.at<CPU>(), (int)o_radial_integrals_.size(), rank__);
    comm__.bcast(so_radial_integrals_.at<CPU>(), (int)so_radial_integrals_.size(), rank__);
}

void Atom_symmetry_class::sync_core_charge_density(Communicator const& comm__, int const rank__)
{
    assert(core_charge_density_.size() != 0);
    
    comm__.bcast(&core_charge_density_[0], atom_type_.radial_grid().num_points(), rank__);
    comm__.bcast(&core_leakage_, 1, rank__);
    comm__.bcast(&core_eval_sum_, 1, rank__);
}

void Atom_symmetry_class::generate_radial_integrals(relativity_t rel__)
{
    runtime::Timer t("sirius::Atom_symmetry_class::generate_radial_integrals");

    int nmtp = atom_type_.num_mt_points();

    double sq_alpha_half = 0.5 * std::pow(speed_of_light, -2);
    if (rel__ == relativity_t::none) {
        sq_alpha_half = 0;
    }

    h_spherical_integrals_.zero();
    #pragma omp parallel default(shared)
    {
        Spline<double> s(atom_type_.radial_grid()); 
        #pragma omp for
        for (int i1 = 0; i1 < atom_type_.mt_radial_basis_size(); i1++) {
            for (int i2 = 0; i2 < atom_type_.mt_radial_basis_size(); i2++) {
                /* for spherical part of potential integrals are diagonal in l */
                if (atom_type_.indexr(i1).l == atom_type_.indexr(i2).l) {
                    int ll = atom_type_.indexr(i1).l * (atom_type_.indexr(i1).l + 1);
                    for (int ir = 0; ir < nmtp; ir++) {
                        double Minv = 1.0 / (1 - spherical_potential_[ir] * sq_alpha_half);
                        /* u_1(r) * u_2(r) */
                        double t0 = radial_functions_(ir, i1, 0) * radial_functions_(ir, i2, 0);
                        /* r*u'_1(r) * r*u'_2(r) */
                        double t1 = radial_functions_(ir, i1, 1) * radial_functions_(ir, i2, 1);
                        s[ir] = 0.5 * t1 * Minv + t0 * (0.5 * ll * Minv + spherical_potential_[ir] * std::pow(atom_type_.radial_grid(ir), 2));
                    }
                    h_spherical_integrals_(i1, i2) = s.interpolate().integrate(0) / y00;
                }
            }
        }
    }

    o_radial_integrals_.zero();
    #pragma omp parallel default(shared)
    {
        Spline<double> s(atom_type_.radial_grid()); 
        #pragma omp for
        for (int l = 0; l <= atom_type_.indexr().lmax(); l++) {
            int nrf = atom_type_.indexr().num_rf(l);

            for (int order1 = 0; order1 < nrf; order1++) {
                int idxrf1 = atom_type_.indexr().index_by_l_order(l, order1);
                for (int order2 = 0; order2 < nrf; order2++) {
                    int idxrf2 = atom_type_.indexr().index_by_l_order(l, order2); 
                    if (order1 == order2) {
                        o_radial_integrals_(l, order1, order2) = 1.0;
                    } else {
                        for (int ir = 0; ir < nmtp; ir++) {
                            s[ir] = radial_functions_(ir, idxrf1, 0) * radial_functions_(ir, idxrf2, 0);
                        }
                        o_radial_integrals_(l, order1, order2) = s.interpolate().integrate(2);
                    }
                }
            }
        }
    }

    if (false) // TODO: if it's slow, compute only when spin-orbit is turned on
    {
        double soc = std::pow(2 * speed_of_light, -2);

        Spline<double> s(atom_type_.radial_grid()); 
        Spline<double> s1(atom_type_.radial_grid()); 
        Spline<double> ve(atom_type_.radial_grid()); 
        
        for (int i = 0; i < nmtp; i++) ve[i] = spherical_potential_[i] + atom_type_.zn() / atom_type_.radial_grid(i);
        ve.interpolate();

        so_radial_integrals_.zero();
        for (int l = 0; l <= atom_type_.indexr().lmax(); l++)
        {
            int nrf = atom_type_.indexr().num_rf(l);

            for (int order1 = 0; order1 < nrf; order1++)
            {
                int idxrf1 = atom_type_.indexr().index_by_l_order(l, order1);
                for (int order2 = 0; order2 < nrf; order2++)
                {
                    int idxrf2 = atom_type_.indexr().index_by_l_order(l, order2);

                    for (int ir = 0; ir < nmtp; ir++)
                    {
                        double M = 1.0 - 2 * soc * spherical_potential_[ir];
                        /* first part <f| dVe / dr |f'> */
                        s[ir] = radial_functions_(ir, idxrf1, 0) * radial_functions_(ir, idxrf2, 0) * 
                                soc * ve.deriv(1, ir) / pow(M, 2);

                        /* second part <f| d(z/r) / dr |f'> */
                        s1[ir] = radial_functions_(ir, idxrf1, 0) * radial_functions_(ir, idxrf2, 0) *
                                 soc * atom_type_.zn() / pow(M, 2);
                    }
                    s.interpolate();
                    s1.interpolate();

                    so_radial_integrals_(l, order1, order2) = s.integrate(1) + s1.integrate(-1);
                }
            }
        }
    }
}

void Atom_symmetry_class::write_enu(runtime::pstdout& pout) const
{
    pout.printf("Atom : %s, class id : %i\n", atom_type_.symbol().c_str(), id_); 
    pout.printf("augmented waves\n");
    for (int l = 0; l < num_aw_descriptors(); l++) {
        for (size_t order = 0; order < aw_descriptor(l).size(); order++) {
            auto& rsd = aw_descriptor(l)[order];
            if (rsd.auto_enu) {
                pout.printf("n = %2i   l = %2i   order = %i   enu = %12.6f\n", rsd.n, rsd.l, order, rsd.enu);
            }
        }
    }

    pout.printf("local orbitals\n");
    for (int idxlo = 0; idxlo < num_lo_descriptors(); idxlo++) {
        for (size_t order = 0; order < lo_descriptor(idxlo).rsd_set.size(); order++) {
            auto& rsd = lo_descriptor(idxlo).rsd_set[order];
            if (rsd.auto_enu) {
                pout.printf("n = %2i   l = %2i   order = %i   enu = %12.6f\n", rsd.n, rsd.l, order, rsd.enu);
            }
        }
    }
    pout.printf("\n");
}

void Atom_symmetry_class::generate_core_charge_density(relativity_t core_rel__)
{
    runtime::Timer t("sirius::Atom_symmetry_class::generate_core_charge_density");

    /* nothing to do */
    if (atom_type_.num_core_electrons() == 0.0) {
        return;
    }

    int nmtp = atom_type_.num_mt_points();

    std::vector<double> free_atom_grid(nmtp);
    for (int i = 0; i < nmtp; i++) {
        free_atom_grid[i] = atom_type_.radial_grid(i);
    }

    /* extend radial grid */
    double x = atom_type_.radial_grid(nmtp - 1);
    double dx = atom_type_.radial_grid().dx(nmtp - 2);
    while (x < 30.0 + atom_type_.zn() / 4.0) {
        x += dx;
        free_atom_grid.push_back(x);
        dx *= 1.025;
    }
    Radial_grid rgrid(free_atom_grid);

    /* interpolate spherical potential inside muffin-tin */
    Spline<double> svmt(atom_type_.radial_grid());
    /* remove nucleus contribution from Vmt */
    for (int ir = 0; ir < nmtp; ir++) {
        svmt[ir] = spherical_potential_[ir] + atom_type_.zn() * atom_type_.radial_grid().x_inv(ir);
    }
    svmt.interpolate();
    /* fit tail to alpha/r + beta */
    double alpha = -(std::pow(atom_type_.mt_radius(), 2) * svmt.deriv(1, nmtp - 1) + atom_type_.zn());
    double beta = svmt[nmtp - 1] - (atom_type_.zn() + alpha) / atom_type_.mt_radius();

    /* cook an effective potential from muffin-tin part and a tail */
    std::vector<double> veff(rgrid.num_points());
    for (int ir = 0; ir < nmtp; ir++) {
        veff[ir] = spherical_potential_[ir];
    }
    /* simple tail alpha/r + beta */
    for (int ir = nmtp; ir < rgrid.num_points(); ir++) {
        veff[ir] = alpha * rgrid.x_inv(ir) + beta;
    }

    //== /* write spherical potential */
    //== std::stringstream sstr;
    //== sstr << "spheric_potential_" << id_ << ".dat";
    //== FILE* fout = fopen(sstr.str().c_str(), "w");

    //== for (int ir = 0; ir < rgrid.num_points(); ir++)
    //== {
    //==     fprintf(fout, "%18.10f %18.10f\n", rgrid[ir], veff[ir]);
    //== }
    //== fclose(fout);
    //== STOP();

    /* charge density */
    Spline<double> rho(rgrid);

    /* atomic level energies */
    std::vector<double> level_energy(atom_type_.num_atomic_levels());

    for (int ist = 0; ist < atom_type_.num_atomic_levels(); ist++) {
        level_energy[ist] = -1.0 * atom_type_.zn() / 2 / std::pow(double(atom_type_.atomic_level(ist).n), 2);
    }

    #pragma omp parallel default(shared)
    {
        std::vector<double> rho_t(rho.num_points());
        std::memset(&rho_t[0], 0, rho.num_points() * sizeof(double));

        #pragma omp for
        for (int ist = 0; ist < atom_type_.num_atomic_levels(); ist++) {
            if (atom_type_.atomic_level(ist).core) {
                Bound_state bs(core_rel__, atom_type_.zn(), atom_type_.atomic_level(ist).n, atom_type_.atomic_level(ist).l,
                               atom_type_.atomic_level(ist).k, rgrid, veff, level_energy[ist]);

                auto& rho = bs.rho();
                for (int i = 0; i < rgrid.num_points(); i++) {
                    rho_t[i] += atom_type_.atomic_level(ist).occupancy * rho[i] / fourpi;
                }

                level_energy[ist] = bs.enu();
            }
        }

        #pragma omp critical
        for (int i = 0; i < rho.num_points(); i++) {
            rho[i] += rho_t[i];
        }
    }

    for (int ir = 0; ir < atom_type_.num_mt_points(); ir++) {
        core_charge_density_[ir] = rho[ir];
    }

    /* interpolate muffin-tin part of core density */
    Spline<double> rho_mt(atom_type_.radial_grid(), core_charge_density_);

    /* compute core leakage */
    core_leakage_ = fourpi * (rho.interpolate().integrate(2) - rho_mt.integrate(2));

    /* compute eigen-value sum of core states */
    core_eval_sum_ = 0.0;
    for (int ist = 0; ist < atom_type_.num_atomic_levels(); ist++) {
        if (atom_type_.atomic_level(ist).core) {
            core_eval_sum_ += level_energy[ist] * atom_type_.atomic_level(ist).occupancy;
        }
    }
}

}
