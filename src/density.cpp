// Copyright (c) 2013-2014 Anton Kozhevnikov, Thomas Schulthess
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

/** \file density.cpp
 *   
 *  \brief Contains remaining implementation of sirius::Density class.
 */

#include <thread>
#include <mutex>
#include "density.h"

namespace sirius {

Density::Density(Global& parameters__) : parameters_(parameters__), gaunt_coefs_(NULL)
{
    fft_ = parameters_.fft();

    rho_ = new Periodic_function<double>(parameters_, parameters_.lmmax_rho(), parameters_.reciprocal_lattice()->num_gvec(), parameters_.comm());

    /* core density of the pseudopotential method */
    if (parameters_.esm_type() == ultrasoft_pseudopotential || 
        parameters_.esm_type() == norm_conserving_pseudopotential)
    {
        rho_pseudo_core_ = new Periodic_function<double>(parameters_, 0, 0, parameters_.comm());
        rho_pseudo_core_->allocate(false, true);
        rho_pseudo_core_->zero();

        generate_pseudo_core_charge_density();
    }

    for (int i = 0; i < parameters_.num_mag_dims(); i++)
    {
        magnetization_[i] = new Periodic_function<double>(parameters_, parameters_.lmmax_rho(), 0, parameters_.comm());
    }

    dmat_spins_.clear();
    dmat_spins_.push_back(std::pair<int, int>(0, 0));
    dmat_spins_.push_back(std::pair<int, int>(1, 1));
    dmat_spins_.push_back(std::pair<int, int>(0, 1));
    
    switch (parameters_.esm_type())
    {
        case full_potential_lapwlo:
        {
            gaunt_coefs_ = new Gaunt_coefficients<double_complex>(parameters_.lmax_apw(), parameters_.lmax_rho(), 
                                                                  parameters_.lmax_apw());
            break;
        }
        case full_potential_pwlo:
        {
            gaunt_coefs_ = new Gaunt_coefficients<double_complex>(parameters_.lmax_pw(), parameters_.lmax_rho(), 
                                                                  parameters_.lmax_pw());
            break;
        }
        case ultrasoft_pseudopotential:
        case norm_conserving_pseudopotential:
        {
            break;
        }
    }

    l_by_lm_ = Utils::l_by_lm(parameters_.lmax_rho());

    high_freq_mixer_ = new Linear_mixer<double_complex>((parameters_.fft()->num_gvec() - parameters_.fft_coarse()->num_gvec()),
                                                        parameters_.iip_.mixer_input_section_.beta_,
                                                        parameters_.comm());

    if (parameters_.iip_.mixer_input_section_.type_ == "linear")
    {
        low_freq_mixer_ = new Linear_mixer<double_complex>(parameters_.fft_coarse()->num_gvec(),
                                                           parameters_.iip_.mixer_input_section_.beta_,
                                                           parameters_.comm());
    }
    else if (parameters_.iip_.mixer_input_section_.type_ == "broyden2")
    {
        low_freq_mixer_ = new Broyden_mixer<double_complex>(parameters_.fft_coarse()->num_gvec(),
                                                            parameters_.iip_.mixer_input_section_.max_history_,
                                                            parameters_.iip_.mixer_input_section_.beta_,
                                                            parameters_.comm());
    } 
    else if (parameters_.iip_.mixer_input_section_.type_ == "broyden_modified")
    {
        std::vector<double> weights(parameters_.fft_coarse()->num_gvec());
        weights[0] = 0;
        for (int ig = 1; ig < parameters_.fft_coarse()->num_gvec(); ig++)
            weights[ig] = fourpi * parameters_.unit_cell()->omega() / std::pow(parameters_.fft_coarse()->gvec_len(ig), 2);

        low_freq_mixer_ = new Broyden_modified_mixer<double_complex>(parameters_.fft_coarse()->num_gvec(),
                                                                     parameters_.iip_.mixer_input_section_.max_history_,
                                                                     parameters_.iip_.mixer_input_section_.beta_,
                                                                     weights,
                                                                     parameters_.comm());
    }
    else
    {
        TERMINATE("Wrong mixer type");
    }
}

Density::~Density()
{
    delete rho_;
    for (int j = 0; j < parameters_.num_mag_dims(); j++) delete magnetization_[j];

    if (parameters_.esm_type() == ultrasoft_pseudopotential ||
        parameters_.esm_type() == norm_conserving_pseudopotential) delete rho_pseudo_core_;
    
    if (gaunt_coefs_) delete gaunt_coefs_;
    
    delete low_freq_mixer_;
    delete high_freq_mixer_;
}

void Density::set_charge_density_ptr(double* rhomt, double* rhoir)
{
    if (parameters_.esm_type() == full_potential_lapwlo || parameters_.esm_type() == full_potential_pwlo)
        rho_->set_mt_ptr(rhomt);
    rho_->set_it_ptr(rhoir);
}

void Density::set_magnetization_ptr(double* magmt, double* magir)
{
    if (parameters_.num_mag_dims() == 0) return;
    assert(parameters_.num_spins() == 2);

    // set temporary array wrapper
    mdarray<double, 4> magmt_tmp(magmt, parameters_.lmmax_rho(), parameters_.unit_cell()->max_num_mt_points(), 
                                 parameters_.unit_cell()->num_atoms(), parameters_.num_mag_dims());
    mdarray<double, 2> magir_tmp(magir, fft_->size(), parameters_.num_mag_dims());
    
    if (parameters_.num_mag_dims() == 1)
    {
        // z component is the first and only one
        magnetization_[0]->set_mt_ptr(&magmt_tmp(0, 0, 0, 0));
        magnetization_[0]->set_it_ptr(&magir_tmp(0, 0));
    }

    if (parameters_.num_mag_dims() == 3)
    {
        // z component is the first
        magnetization_[0]->set_mt_ptr(&magmt_tmp(0, 0, 0, 2));
        magnetization_[0]->set_it_ptr(&magir_tmp(0, 2));
        // x component is the second
        magnetization_[1]->set_mt_ptr(&magmt_tmp(0, 0, 0, 0));
        magnetization_[1]->set_it_ptr(&magir_tmp(0, 0));
        // y component is the third
        magnetization_[2]->set_mt_ptr(&magmt_tmp(0, 0, 0, 1));
        magnetization_[2]->set_it_ptr(&magir_tmp(0, 1));
    }
}
    
void Density::zero()
{
    rho_->zero();
    for (int i = 0; i < parameters_.num_mag_dims(); i++) magnetization_[i]->zero();
}

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

void Density::initial_density()
{
    Timer t("sirius::Density::initial_density");

    zero();
    
    auto rl = parameters_.reciprocal_lattice();
    auto uc = parameters_.unit_cell();

    if (uc->full_potential())
    {
        /* initialize smooth density of free atoms */
        for (int iat = 0; iat < uc->num_atom_types(); iat++) uc->atom_type(iat)->init_free_atom(true);

        /* compute radial integrals */
        auto rho_radial_integrals = generate_rho_radial_integrals(0);

        /* compute contribution from free atoms to the interstitial density */
        std::vector<double_complex> v = rl->make_periodic_function(rho_radial_integrals, rl->num_gvec());

        /* set plane-wave coefficients of the charge density */
        memcpy(&rho_->f_pw(0), &v[0], rl->num_gvec() * sizeof(double_complex));

        /* convert charge deisnty to real space mesh */
        fft_->input(fft_->num_gvec(), fft_->index_map(), &rho_->f_pw(0));
        fft_->transform(1);
        fft_->output(&rho_->f_it<global>(0));

        /* remove possible negative noise */
        for (int ir = 0; ir < fft_->size(); ir++)
        {
            if (rho_->f_it<global>(ir) < 0) rho_->f_it<global>(ir) = 0;
        }

        int ngv_loc = (int)rl->spl_num_gvec().local_size();

        /* mapping between G-shell (global index) and a list of G-vectors (local index) */
        std::map<int, std::vector<int> > gsh_map;

        for (int igloc = 0; igloc < ngv_loc; igloc++)
        {
            /* global index of the G-vector */
            int ig = (int)rl->spl_num_gvec(igloc);
            /* index of the G-vector shell */
            int igsh = rl->gvec_shell(ig);
            if (gsh_map.count(igsh) == 0) gsh_map[igsh] = std::vector<int>();
            gsh_map[igsh].push_back(igloc);
        }

        /* list of G-shells for the curent MPI rank */
        std::vector<std::pair<int, std::vector<int> > > gsh_list;
        for (auto& i: gsh_map) gsh_list.push_back(std::pair<int, std::vector<int> >(i.first, i.second));

        int lmax = parameters_.lmax_rho();
        int lmmax = Utils::lmmax(lmax);
        
        sbessel_approx sba(uc, lmax, rl->gvec_shell_len(1), rl->gvec_shell_len(rl->num_gvec_shells_inner() - 1), 1e-6);
        
        std::vector<double> gvec_len(gsh_list.size());
        for (int i = 0; i < (int)gsh_list.size(); i++)
        {
            gvec_len[i] = rl->gvec_shell_len(gsh_list[i].first);
        }
        sba.approximate(gvec_len);

        auto l_by_lm = Utils::l_by_lm(lmax);

        std::vector<double_complex> zil(lmax + 1);
        for (int l = 0; l <= lmax; l++) zil[l] = pow(double_complex(0, 1), l);

        Timer t3("sirius::Density::initial_density|znulm");

        mdarray<double_complex, 3> znulm(sba.nqnu_max(), lmmax, uc->num_atoms());
        znulm.zero();
        
        #pragma omp parallel for
        for (int ia = 0; ia < uc->num_atoms(); ia++)
        {
            int iat = uc->atom(ia)->type_id();

            /* loop over local fraction of G-shells */
            for (int i = 0; i < (int)gsh_list.size(); i++)
            {
                auto& gv = gsh_list[i].second;
                
                /* loop over G-vectors */
                for (int igloc: gv)
                {
                    /* global index of the G-vector */
                    int ig = rl->spl_num_gvec(igloc);

                    double_complex z1 = rl->gvec_phase_factor<local>(igloc, ia) * v[ig] * fourpi; 

                    for (int lm = 0; lm < lmmax; lm++)
                    {
                        int l = l_by_lm[lm];
                        
                        /* number of expansion coefficients */
                        int nqnu = sba.nqnu(l, iat);

                        double_complex z2 = z1 * zil[l] * rl->gvec_ylm(lm, igloc);
                    
                        for (int iq = 0; iq < nqnu; iq++) znulm(iq, lm, ia) += z2 * sba.coeff(iq, i, l, iat);
                    }
                }
            }
        }
        parameters_.comm().allreduce(znulm.at<CPU>(), (int)znulm.size());
        t3.stop();

        Timer t4("sirius::Density::initial_density|rholm");
        
        SHT sht(lmax);

        for (int ialoc = 0; ialoc < (int)parameters_.unit_cell()->spl_num_atoms().local_size(); ialoc++)
        {
            int ia = parameters_.unit_cell()->spl_num_atoms(ialoc);
            int iat = uc->atom(ia)->type_id();

            Spheric_function<spectral, double_complex> rhoylm(lmmax, uc->atom(ia)->radial_grid());
            rhoylm.zero();
            #pragma omp parallel for
            for (int lm = 0; lm < lmmax; lm++)
            {
                int l = l_by_lm[lm];
                for (int iq = 0; iq < sba.nqnu(l, iat); iq++)
                {
                    double qnu = sba.qnu(iq, l, iat);

                    for (int ir = 0; ir < uc->atom(ia)->num_mt_points(); ir++)
                    {
                        double x = uc->atom(ia)->radial_grid(ir);
                        rhoylm(lm, ir) += znulm(iq, lm, ia) * gsl_sf_bessel_jl(l, x * qnu);
                    }
                }
            }
            for (int ir = 0; ir < uc->atom(ia)->num_mt_points(); ir++)
            {
                double x = uc->atom(ia)->radial_grid(ir);
                rhoylm(0, ir) += (v[0] - uc->atom(ia)->type()->free_atom_density(x)) / y00;
            }
            sht.convert(rhoylm, rho_->f_mt(ialoc));
        }
        
        t4.stop();

        /* initialize density of free atoms (not smoothed) */
        for (int iat = 0; iat < uc->num_atom_types(); iat++) uc->atom_type(iat)->init_free_atom(false);

        for (int ia = 0; ia < uc->num_atoms(); ia++)
        {
            auto p = uc->spl_num_atoms().location(ia);
            
            if (p.second == parameters_.comm().rank())
            {
                /* add density of a free atom */
                for (int ir = 0; ir < uc->atom(ia)->num_mt_points(); ir++)
                {
                    double x = uc->atom(ia)->type()->radial_grid(ir);
                    rho_->f_mt<local>(0, ir, (int)p.first) += uc->atom(ia)->type()->free_atom_density(x) / y00;
                }
            }
        }

        /* initialize the magnetization */
        if (parameters_.num_mag_dims())
        {
            for (int ialoc = 0; ialoc < (int)uc->spl_num_atoms().local_size(); ialoc++)
            {
                int ia = (int)uc->spl_num_atoms(ialoc);
                vector3d<double> v = uc->atom(ia)->vector_field();
                double len = v.length();

                int nmtp = uc->atom(ia)->type()->num_mt_points();
                Spline<double> rho(uc->atom(ia)->type()->radial_grid());
                double R = uc->atom(ia)->type()->mt_radius();
                for (int ir = 0; ir < nmtp; ir++)
                {
                    double x = uc->atom(ia)->type()->radial_grid(ir);
                    rho[ir] = rho_->f_mt<local>(0, ir, ialoc) * y00 * (1 - 3 * std::pow(x / R, 2) + 2 * std::pow(x / R, 3));
                }

                /* maximum magnetization which can be achieved if we smooth density towards MT boundary */
                double q = fourpi * rho.interpolate().integrate(2);
                
                /* if very strong initial magnetization is given */
                if (q < len)
                {
                    /* renormalize starting magnetization */
                    for (int x = 0; x < 3; x++) v[x] *= (q / len);

                    len = q;
                }

                if (len > 1e-8)
                {
                    for (int ir = 0; ir < nmtp; ir++)
                        magnetization_[0]->f_mt<local>(0, ir, ialoc) = rho[ir] * v[2] / q / y00;

                    if (parameters_.num_mag_dims() == 3)
                    {
                        for (int ir = 0; ir < nmtp; ir++)
                        {
                            magnetization_[1]->f_mt<local>(0, ir, ialoc) = rho[ir] * v[0] / q / y00;
                            magnetization_[2]->f_mt<local>(0, ir, ialoc) = rho[ir] * v[1] / q / y00;
                        }
                    }
                }
            }
        }
    }

    if (parameters_.esm_type() == ultrasoft_pseudopotential ||
        parameters_.esm_type() == norm_conserving_pseudopotential) 
    {
        auto rho_radial_integrals = generate_rho_radial_integrals(1);

        std::vector<double_complex> v = rl->make_periodic_function(rho_radial_integrals, rl->num_gvec());

        memcpy(&rho_->f_pw(0), &v[0], rl->num_gvec() * sizeof(double_complex));

        double charge = real(rho_->f_pw(0) * uc->omega());
        if (std::abs(charge - uc->num_valence_electrons()) > 1e-6)
        {
            std::stringstream s;
            s << "wrong initial charge density" << std::endl
              << "  integral of the density : " << real(rho_->f_pw(0) * uc->omega()) << std::endl
              << "  target number of electrons : " << uc->num_valence_electrons();
            warning_global(__FILE__, __LINE__, s);
        }

        fft_->input(fft_->num_gvec(), fft_->index_map(), &rho_->f_pw(0));
        fft_->transform(1);
        fft_->output(&rho_->f_it<global>(0));
        
        /* remove possible negative noise */
        for (int ir = 0; ir < fft_->size(); ir++)
        {
            rho_->f_it<global>(ir) = rho_->f_it<global>(ir) *  uc->num_valence_electrons() / charge;
            if (rho_->f_it<global>(ir) < 0) rho_->f_it<global>(ir) = 0;
        }

        //== FILE* fout = fopen("unit_cell.xsf", "w");
        //== fprintf(fout, "CRYSTAL\n");
        //== fprintf(fout, "PRIMVEC\n");
        //== auto& lv = uc->lattice_vectors();
        //== for (int i = 0; i < 3; i++)
        //== {
        //==     fprintf(fout, "%18.12f %18.12f %18.12f\n", lv(0, i), lv(1, i), lv(2, i));
        //== }
        //== fprintf(fout, "CONVVEC\n");
        //== for (int i = 0; i < 3; i++)
        //== {
        //==     fprintf(fout, "%18.12f %18.12f %18.12f\n", lv(0, i), lv(1, i), lv(2, i));
        //== }
        //== fprintf(fout, "PRIMCOORD\n");
        //== fprintf(fout, "%i 1\n", uc->num_atoms());
        //== for (int ia = 0; ia < uc->num_atoms(); ia++)
        //== {
        //==     auto pos = uc->get_cartesian_coordinates(uc->atom(ia)->position());
        //==     fprintf(fout, "%i %18.12f %18.12f %18.12f\n", uc->atom(ia)->zn(), pos[0], pos[1], pos[2]);
        //== }
        //== fclose(fout);


        //== /* initialize the magnetization */
        //== if (parameters_.num_mag_dims())
        //== {
        //==     for (int ia = 0; ia < uc->num_atoms(); ia++)
        //==     {
        //==         vector3d<double> v = uc->atom(ia)->vector_field();
        //==         double len = v.length();

        //==         for (int j0 = 0; j0 < fft_->size(0); j0++)
        //==         {
        //==             for (int j1 = 0; j1 < fft_->size(1); j1++)
        //==             {
        //==                 for (int j2 = 0; j2 < fft_->size(2); j2++)
        //==                 {
        //==                     /* get real space fractional coordinate */
        //==                     vector3d<double> v0(double(j0) / fft_->size(0), double(j1) / fft_->size(1), double(j2) / fft_->size(2));
        //==                     /* index of real space point */
        //==                     int ir = static_cast<int>(j0 + j1 * fft_->size(0) + j2 * fft_->size(0) * fft_->size(1));

        //==                     for (int t0 = -1; t0 <= 1; t0++)
        //==                     {
        //==                         for (int t1 = -1; t1 <= 1; t1++)
        //==                         {
        //==                             for (int t2 = -1; t2 <= 1; t2++)
        //==                             {
        //==                                 vector3d<double> v1 = v0 - (uc->atom(ia)->position() + vector3d<double>(t0, t1, t2));
        //==                                 auto r = uc->get_cartesian_coordinates(vector3d<double>(v1));
        //==                                 if (r.length() <= 2.0)
        //==                                 {
        //==                                     magnetization_[0]->f_it<global>(ir) = 1.0;
        //==                                 }
        //==                             }
        //==                         }
        //==                     }
        //==                 }
        //==             }
        //==         }
        //==     }
        //== }


        //== mdarray<double, 3> rho_grid(&rho_->f_it<global>(0), fft_->size(0), fft_->size(1), fft_->size(2));
        //== mdarray<double, 4> pos_grid(3, fft_->size(0), fft_->size(1), fft_->size(2));

        //== mdarray<double, 4> mag_grid(3, fft_->size(0), fft_->size(1), fft_->size(2));
        //== mag_grid.zero();

        //== // loop over 3D array (real space)
        //== for (int j0 = 0; j0 < fft_->size(0); j0++)
        //== {
        //==     for (int j1 = 0; j1 < fft_->size(1); j1++)
        //==     {
        //==         for (int j2 = 0; j2 < fft_->size(2); j2++)
        //==         {
        //==             int ir = static_cast<int>(j0 + j1 * fft_->size(0) + j2 * fft_->size(0) * fft_->size(1));
        //==             // get real space fractional coordinate
        //==             double frv[] = {double(j0) / fft_->size(0), 
        //==                             double(j1) / fft_->size(1), 
        //==                             double(j2) / fft_->size(2)};
        //==             vector3d<double> rv = parameters_.unit_cell()->get_cartesian_coordinates(vector3d<double>(frv));
        //==             for (int x = 0; x < 3; x++) pos_grid(x, j0, j1, j2) = rv[x];
        //==             if (parameters_.num_mag_dims() == 1) mag_grid(2, j0, j1, j2) = magnetization_[0]->f_it<global>(ir);
        //==             if (parameters_.num_mag_dims() == 3) 
        //==             {
        //==                 mag_grid(0, j0, j1, j2) = magnetization_[1]->f_it<global>(ir);
        //==                 mag_grid(1, j0, j1, j2) = magnetization_[2]->f_it<global>(ir);
        //==             }
        //==         }
        //==     }
        //== }

        //== HDF5_tree h5_rho("rho.hdf5", true);
        //== h5_rho.write("rho", rho_grid);
        //== h5_rho.write("pos", pos_grid);
        //== h5_rho.write("mag", mag_grid);

        //== FILE* fout = fopen("rho.xdmf", "w");
        //== //== fprintf(fout, "<?xml version=\"1.0\" ?>\n"
        //== //==               "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\">\n"
        //== //==               "<Xdmf>\n"
        //== //==               "  <Domain Name=\"name1\">\n"
        //== //==               "    <Grid Name=\"fft_fine_grid\" Collection=\"Unknown\">\n"
        //== //==               "      <Topology TopologyType=\"3DSMesh\" NumberOfElements=\" %i %i %i \"/>\n"
        //== //==               "      <Geometry GeometryType=\"XYZ\">\n"
        //== //==               "        <DataItem Dimensions=\"%i %i %i 3\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">rho.hdf5:/pos</DataItem>\n"
        //== //==               "      </Geometry>\n"
        //== //==               "      <Attribute\n"
        //== //==               "           AttributeType=\"Scalar\"\n"
        //== //==               "           Center=\"Node\"\n"
        //== //==               "           Name=\"rho\">\n"
        //== //==               "          <DataItem\n"
        //== //==               "             NumberType=\"Float\"\n"
        //== //==               "             Precision=\"8\"\n"
        //== //==               "             Dimensions=\"%i %i %i\"\n"
        //== //==               "             Format=\"HDF\">\n"
        //== //==               "             rho.hdf5:/rho\n"
        //== //==               "          </DataItem>\n"
        //== //==               "        </Attribute>\n"
        //== //==               "    </Grid>\n"
        //== //==               "  </Domain>\n"
        //== //==               "</Xdmf>\n", fft_->size(0), fft_->size(1), fft_->size(2), fft_->size(0), fft_->size(1), fft_->size(2), fft_->size(0), fft_->size(1), fft_->size(2));
        //== fprintf(fout, "<?xml version=\"1.0\" ?>\n"
        //==               "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\">\n"
        //==               "<Xdmf>\n"
        //==               "  <Domain Name=\"name1\">\n"
        //==               "    <Grid Name=\"fft_fine_grid\" Collection=\"Unknown\">\n"
        //==               "      <Topology TopologyType=\"3DSMesh\" NumberOfElements=\" %i %i %i \"/>\n"
        //==               "      <Geometry GeometryType=\"XYZ\">\n"
        //==               "        <DataItem Dimensions=\"%i %i %i 3\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">rho.hdf5:/pos</DataItem>\n"
        //==               "      </Geometry>\n"
        //==               "      <Attribute\n"
        //==               "           AttributeType=\"Vector\"\n"
        //==               "           Center=\"Node\"\n"
        //==               "           Name=\"mag\">\n"
        //==               "          <DataItem\n"
        //==               "             NumberType=\"Float\"\n"
        //==               "             Precision=\"8\"\n"
        //==               "             Dimensions=\"%i %i %i 3\"\n"
        //==               "             Format=\"HDF\">\n"
        //==               "             rho.hdf5:/mag\n"
        //==               "          </DataItem>\n"
        //==               "        </Attribute>\n"
        //==               "    </Grid>\n"
        //==               "  </Domain>\n"
        //==               "</Xdmf>\n", fft_->size(0), fft_->size(1), fft_->size(2), fft_->size(0), fft_->size(1), fft_->size(2), fft_->size(0), fft_->size(1), fft_->size(2));
        //== fclose(fout);
        
        fft_->input(&rho_->f_it<global>(0));
        fft_->transform(-1);
        fft_->output(fft_->num_gvec(), fft_->index_map(), &rho_->f_pw(0));
    }

    rho_->sync(true, true);
    for (int i = 0; i < parameters_.num_mag_dims(); i++) magnetization_[i]->sync(true, true);

    if (uc->full_potential())
    {
        /* check initial charge */
        std::vector<double> nel_mt;
        double nel_it;
        double nel = rho_->integrate(nel_mt, nel_it);
        if (uc->num_electrons() > 1e-8 && std::abs(nel - uc->num_electrons()) / uc->num_electrons()  > 1e-3)
        {
            std::stringstream s;
            s << "wrong initial charge density" << std::endl
              << "  integral of the density : " << nel << std::endl
              << "  target number of electrons : " << uc->num_electrons();
            warning_global(__FILE__, __LINE__, s);
        }
    }
}

void Density::add_kpoint_contribution_mt(K_point* kp, std::vector< std::pair<int, double> >& occupied_bands, 
                                         mdarray<double_complex, 4>& mt_complex_density_matrix)
{
    Timer t("sirius::Density::add_kpoint_contribution_mt");
    
    if (occupied_bands.size() == 0) return;
   
    mdarray<double_complex, 3> wf1(parameters_.unit_cell()->max_mt_basis_size(), (int)occupied_bands.size(), parameters_.num_spins());
    mdarray<double_complex, 3> wf2(parameters_.unit_cell()->max_mt_basis_size(), (int)occupied_bands.size(), parameters_.num_spins());

    for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    {
        int offset_wf = parameters_.unit_cell()->atom(ia)->offset_wf();
        int mt_basis_size = parameters_.unit_cell()->atom(ia)->type()->mt_basis_size();
        
        for (int i = 0; i < (int)occupied_bands.size(); i++)
        {
            for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
            {
                for (int j = 0; j < mt_basis_size; j++)
                {
                    wf1(j, i, ispn) = conj(kp->spinor_wave_function(offset_wf + j, occupied_bands[i].first, ispn));
                    wf2(j, i, ispn) = kp->spinor_wave_function(offset_wf + j, occupied_bands[i].first, ispn) * occupied_bands[i].second;
                }
            }
        }

        for (int j = 0; j < (int)mt_complex_density_matrix.size(2); j++)
        {
            linalg<CPU>::gemm(0, 1, mt_basis_size, mt_basis_size, (int)occupied_bands.size(), complex_one, 
                              &wf1(0, 0, dmat_spins_[j].first), wf1.ld(), 
                              &wf2(0, 0, dmat_spins_[j].second), wf2.ld(), complex_one, 
                              &mt_complex_density_matrix(0, 0, j, ia), mt_complex_density_matrix.ld());
        }
    }
}

template <int num_mag_dims> 
void Density::reduce_zdens(Atom_type* atom_type, int ialoc, mdarray<double_complex, 4>& zdens, mdarray<double, 3>& mt_density_matrix)
{
    mt_density_matrix.zero();
    
    #pragma omp parallel for default(shared)
    for (int idxrf2 = 0; idxrf2 < atom_type->mt_radial_basis_size(); idxrf2++)
    {
        int l2 = atom_type->indexr(idxrf2).l;
        for (int idxrf1 = 0; idxrf1 <= idxrf2; idxrf1++)
        {
            int offs = idxrf2 * (idxrf2 + 1) / 2 + idxrf1;
            int l1 = atom_type->indexr(idxrf1).l;

            int xi2 = atom_type->indexb().index_by_idxrf(idxrf2);
            for (int lm2 = Utils::lm_by_l_m(l2, -l2); lm2 <= Utils::lm_by_l_m(l2, l2); lm2++, xi2++)
            {
                int xi1 = atom_type->indexb().index_by_idxrf(idxrf1);
                for (int lm1 = Utils::lm_by_l_m(l1, -l1); lm1 <= Utils::lm_by_l_m(l1, l1); lm1++, xi1++)
                {
                    for (int k = 0; k < gaunt_coefs_->num_gaunt(lm1, lm2); k++)
                    {
                        int lm3 = gaunt_coefs_->gaunt(lm1, lm2, k).lm3;
                        double_complex gc = gaunt_coefs_->gaunt(lm1, lm2, k).coef;
                        switch (num_mag_dims)
                        {
                            case 3:
                            {
                                mt_density_matrix(lm3, offs, 2) += 2.0 * real(zdens(xi1, xi2, 2, ialoc) * gc); 
                                mt_density_matrix(lm3, offs, 3) -= 2.0 * imag(zdens(xi1, xi2, 2, ialoc) * gc);
                            }
                            case 1:
                            {
                                mt_density_matrix(lm3, offs, 1) += real(zdens(xi1, xi2, 1, ialoc) * gc);
                            }
                            case 0:
                            {
                                mt_density_matrix(lm3, offs, 0) += real(zdens(xi1, xi2, 0, ialoc) * gc);
                            }
                        }
                    }
                }
            }
        }
    }
}

std::vector< std::pair<int, double> > Density::get_occupied_bands_list(Band* band, K_point* kp)
{
    std::vector< std::pair<int, double> > bands;
    for (int jsub = 0; jsub < kp->num_sub_bands(); jsub++)
    {
        int j = kp->idxbandglob(jsub);
        double wo = kp->band_occupancy(j) * kp->weight();
        if (wo > 1e-14) bands.push_back(std::pair<int, double>(jsub, wo));
    }
    return bands;
}

void Density::add_kpoint_contribution_pp(K_point* kp__, 
                                         std::vector< std::pair<int, double> >& occupied_bands__, 
                                         mdarray<double_complex, 4>& pp_complex_density_matrix__)
{
    Timer t("sirius::Density::add_kpoint_contribution_pp");

    int nbnd = num_occupied_bands(kp__);

    if (nbnd == 0) return;

    auto uc = parameters_.unit_cell();

    dmatrix<double_complex> beta_psi(uc->mt_basis_size(), nbnd, kp__->blacs_grid());

    /* compute <beta|Psi> */
    Timer t1("sirius::Density::add_kpoint_contribution_pp|beta_psi");
    linalg<CPU>::gemm(2, 0, uc->mt_basis_size(), nbnd, kp__->num_gkvec(), complex_one, 
                      kp__->beta_pw_panel(), kp__->fv_states_panel(), complex_zero, beta_psi);
    t1.stop();

    splindex<block> sub_spl_col(beta_psi.num_cols_local(), kp__->num_ranks_row(), kp__->rank_row());

    mdarray<double_complex, 2> beta_psi_slice(uc->mt_basis_size(), sub_spl_col.local_size());
    beta_psi.gather(beta_psi_slice);

    #pragma omp parallel
    {
        /* auxiliary arrays */
        mdarray<double_complex, 2> bp1(uc->max_mt_basis_size(), (int)sub_spl_col.local_size());
        mdarray<double_complex, 2> bp2(uc->max_mt_basis_size(), (int)sub_spl_col.local_size());
        #pragma omp for
        for (int ia = 0; ia < uc->num_atoms(); ia++)
        {   
            /* number of beta functions for a given atom */
            int nbf = uc->atom(ia)->mt_basis_size();

            for (int i = 0; i < (int)sub_spl_col.local_size(); i++)
            {
                int j = beta_psi.icol((int)sub_spl_col[i]);
                for (int xi = 0; xi < nbf; xi++)
                {
                    bp1(xi, i) = beta_psi_slice(uc->atom(ia)->offset_lo() + xi, i);
                    bp2(xi, i) = conj(bp1(xi, i)) * kp__->band_occupancy(j) * kp__->weight();
                }
            }

            linalg<CPU>::gemm(0, 1, nbf, nbf, (int)sub_spl_col.local_size(), complex_one, &bp1(0, 0), bp1.ld(),
                              &bp2(0, 0), bp2.ld(), complex_one, &pp_complex_density_matrix__(0, 0, 0, ia), 
                              pp_complex_density_matrix__.ld());
        }
    }
}

#ifdef _GPU_
extern "C" void copy_beta_psi_gpu(int nbf,
                                  int nloc,
                                  double_complex const* beta_psi,
                                  int beta_psi_ld,
                                  double const* wo,
                                  double_complex* beta_psi_wo,
                                  int beta_psi_wo_ld,
                                  int stream_id);

void Density::add_kpoint_contribution_pp_gpu(K_point* kp__,
                                             std::vector< std::pair<int, double> >& occupied_bands__, 
                                             mdarray<double_complex, 4>& pp_complex_density_matrix__)
{
    Timer t("sirius::Density::add_kpoint_contribution_pp_gpu", kp__->comm());

    auto& psi = kp__->fv_states_panel();
    
    mdarray<double, 1> wo(psi.num_cols_local());
    int nloc = 0;
    for (int jloc = 0; jloc < psi.num_cols_local(); jloc++)
    {
        int j = psi.icol(jloc);
        double d = kp__->band_occupancy(j) * kp__->weight();
        if (d > 1e-14) wo(nloc++) = d;
    }
    if (!nloc) return;

    wo.allocate_on_device();
    wo.copy_to_device();

    auto uc = parameters_.unit_cell();
    
    /* allocate space for <beta|psi> array */
    int nbf_max = 0;
    for (int ib = 0; ib < uc->num_beta_chunks(); ib++)
        nbf_max  = std::max(nbf_max, uc->beta_chunk(ib).num_beta_);

    mdarray<double_complex, 1> beta_psi_tmp(nbf_max * nloc);
    beta_psi_tmp.allocate_on_device();

    matrix<double_complex> beta_gk(nullptr, kp__->num_gkvec_row(), nbf_max);
    beta_gk.allocate_on_device();

    matrix<double_complex> psi_occ(&psi(0, 0), kp__->num_gkvec_row(), nloc);
    psi_occ.allocate_on_device();
    psi_occ.copy_to_device();

    mdarray<double_complex, 3> tmp(nullptr, uc->max_mt_basis_size(), nloc, Platform::max_num_threads());
    tmp.allocate_on_device();

    for (int ib = 0; ib < uc->num_beta_chunks(); ib++)
    {
        /* wrapper for <beta|psi> with required dimensions */
        matrix<double_complex> beta_psi(beta_psi_tmp.at<CPU>(), beta_psi_tmp.at<GPU>(), uc->beta_chunk(ib).num_beta_, nloc);

        kp__->generate_beta_gk(uc->beta_chunk(ib).num_atoms_, uc->beta_chunk(ib).atom_pos_, uc->beta_chunk(ib).desc_, beta_gk);
        kp__->generate_beta_phi(uc->beta_chunk(ib).num_beta_, psi_occ, nloc, 0, beta_gk, beta_psi);

        double_complex alpha(1, 0);

        #pragma omp parallel for
        for (int i = 0; i < uc->beta_chunk(ib).num_atoms_; i++)
        {
            /* number of beta functions for a given atom */
            int nbf = uc->beta_chunk(ib).desc_(0, i);
            int ofs = uc->beta_chunk(ib).desc_(1, i);
            int ia = uc->beta_chunk(ib).desc_(3, i);
            int thread_id = Platform::thread_id();
            
            copy_beta_psi_gpu(nbf,
                              nloc,
                              beta_psi.at<GPU>(ofs, 0),
                              beta_psi.ld(),
                              wo.at<GPU>(),
                              tmp.at<GPU>(0, 0, thread_id),
                              tmp.ld(),
                              thread_id);
            
            linalg<GPU>::gemm(0, 1, nbf, nbf, nloc, &alpha, beta_psi.at<GPU>(ofs, 0), beta_psi.ld(),
                              tmp.at<GPU>(0, 0, thread_id), tmp.ld(), &alpha, 
                              pp_complex_density_matrix__.at<GPU>(0, 0, 0, ia), pp_complex_density_matrix__.ld(), thread_id);
        }
        cuda_device_synchronize();
    }

    tmp.deallocate_on_device();
    psi_occ.deallocate_on_device();
}
#endif

#ifdef _GPU_
extern "C" void update_it_density_matrix_gpu(int fft_size, 
                                             int nfft_max, 
                                             int num_spins, 
                                             int num_mag_dims, 
                                             void* psi_it, 
                                             double* wt, 
                                             void* it_density_matrix);
#endif

void Density::add_kpoint_contribution_it(K_point* kp, std::vector< std::pair<int, double> >& occupied_bands)
{
    Timer t("sirius::Density::add_kpoint_contribution_it");
    
    if (occupied_bands.size() == 0) return;
    
    /* index of the occupied bands */
    int idx_band = 0;
    std::mutex idx_band_mutex;

    int num_fft_threads = -1;
    switch (parameters_.processing_unit())
    {
        case CPU:
        {
            num_fft_threads = fft_->num_fft_threads();
            break;
        }
        case GPU:
        {
            num_fft_threads = std::min(fft_->num_fft_threads() + 1, Platform::max_num_threads());
            break;
        }
    }

    mdarray<double, 3> it_density_matrix(fft_->size(), parameters_.num_mag_dims() + 1, num_fft_threads);
    it_density_matrix.zero();
    
    #ifdef _GPU_
    mdarray<double, 2> it_density_matrix_gpu;
    /* last thread is doing cuFFT */
    if (parameters_.processing_unit() == GPU && num_fft_threads > 1)
    {
        it_density_matrix_gpu = mdarray<double, 2>(&it_density_matrix(0, 0, num_fft_threads - 1), fft_->size(), parameters_.num_mag_dims() + 1);
        it_density_matrix_gpu.allocate_on_device();
        it_density_matrix_gpu.zero_on_device();
    }
    auto fft_gpu = parameters_.fft_gpu();
    #endif

    std::vector<std::thread> fft_threads;

    auto fft = parameters_.fft();
    int num_spins = parameters_.num_spins();
    int num_mag_dims = parameters_.num_mag_dims();
    double omega = parameters_.unit_cell()->omega();

    for (int thread_id = 0; thread_id < num_fft_threads; thread_id++)
    {
        if (thread_id == (num_fft_threads - 1) && num_fft_threads > 1 && parameters_.processing_unit() == GPU)
        {
            #ifdef _GPU_
            fft_threads.push_back(std::thread([thread_id, kp, fft_gpu, &idx_band, &idx_band_mutex, num_spins, num_mag_dims, 
                                               omega, &occupied_bands, &it_density_matrix_gpu]()
            {
                Timer t("sirius::Density::add_kpoint_contribution_it|gpu");

                int wf_pw_offset = kp->wf_pw_offset();
                
                /* move fft index to GPU */
                mdarray<int, 1> fft_index(kp->fft_index(), kp->num_gkvec());
                fft_index.allocate_on_device();
                fft_index.copy_to_device();

                int nfft_max = fft_gpu->num_fft();
 
                /* allocate work area array */
                mdarray<char, 1> work_area(nullptr, fft_gpu->work_area_size());
                work_area.allocate_on_device();
                fft_gpu->set_work_area_ptr(work_area.at<GPU>());
                
                /* allocate space for plane-wave expansion coefficients */
                mdarray<double_complex, 2> psi_pw_gpu(nullptr, kp->num_gkvec(), nfft_max); 
                psi_pw_gpu.allocate_on_device();
                
                /* allocate space for spinor components */
                mdarray<double_complex, 3> psi_it_gpu(nullptr, fft_gpu->size(), nfft_max, num_spins);
                psi_it_gpu.allocate_on_device();
                
                /* allocate space for weights */
                mdarray<double, 1> w(nfft_max);
                w.allocate_on_device();

                bool done = false;

                while (!done)
                {
                    idx_band_mutex.lock();
                    int i = idx_band;
                    if (idx_band + nfft_max > (int)occupied_bands.size()) 
                    {
                        done = true;
                    }
                    else
                    {
                        idx_band += nfft_max;
                    }
                    idx_band_mutex.unlock();

                    if (!done)
                    {
                        for (int ispn = 0; ispn < num_spins; ispn++)
                        {
                            /* copy PW coefficients to GPU */
                            for (int j = 0; j < nfft_max; j++)
                            {
                                w(j) = occupied_bands[i + j].second / omega;

                                cublas_set_vector(kp->num_gkvec(), sizeof(double_complex), 
                                                  &kp->spinor_wave_function(wf_pw_offset, occupied_bands[i + j].first, ispn), 1, 
                                                  psi_pw_gpu.at<GPU>(0, j), 1);
                            }
                            w.copy_to_device();
                            
                            /* set PW coefficients into proper positions inside FFT buffer */
                            fft_gpu->batch_load(kp->num_gkvec(), fft_index.at<GPU>(), psi_pw_gpu.at<GPU>(0, 0), 
                                                psi_it_gpu.at<GPU>(0, 0, ispn));

                            /* execute batch FFT */
                            fft_gpu->transform(1, psi_it_gpu.at<GPU>(0, 0, ispn));
                        }

                        update_it_density_matrix_gpu(fft_gpu->size(), nfft_max, num_spins, num_mag_dims, 
                                                     psi_it_gpu.at<GPU>(), w.at<GPU>(),
                                                     it_density_matrix_gpu.at<GPU>(0, 0));
                    }
                }
            }));
            #else
            TERMINATE_NO_GPU
            #endif
        }
        else
        {
            fft_threads.push_back(std::thread([thread_id, kp, fft, &idx_band, &idx_band_mutex, num_spins, num_mag_dims, 
                                               omega, &occupied_bands, &it_density_matrix]()
            {
                bool done = false;

                int wf_pw_offset = kp->wf_pw_offset();
                
                mdarray<double_complex, 2> psi_it(fft->size(), num_spins);

                while (!done)
                {
                    // increment the band index
                    idx_band_mutex.lock();
                    int i = idx_band;
                    if (idx_band + 1 > (int)occupied_bands.size()) 
                    {
                        done = true;
                    }
                    else
                    {
                        idx_band++;
                    }
                    idx_band_mutex.unlock();

                    if (!done)
                    {
                        for (int ispn = 0; ispn < num_spins; ispn++)
                        {
                            fft->input(kp->num_gkvec(), kp->fft_index(), 
                                       &kp->spinor_wave_function(wf_pw_offset, occupied_bands[i].first, ispn), thread_id);
                            fft->transform(1, thread_id);
                            fft->output(&psi_it(0, ispn), thread_id);
                        }
                        double w = occupied_bands[i].second / omega;
                       
                        switch (num_mag_dims)
                        {
                            case 3:
                            {
                                for (int ir = 0; ir < fft->size(); ir++)
                                {
                                    double_complex z = psi_it(ir, 0) * conj(psi_it(ir, 1)) * w;
                                    it_density_matrix(ir, 2, thread_id) += 2.0 * real(z);
                                    it_density_matrix(ir, 3, thread_id) -= 2.0 * imag(z);
                                }
                            }
                            case 1:
                            {
                                for (int ir = 0; ir < fft->size(); ir++)
                                    it_density_matrix(ir, 1, thread_id) += real(psi_it(ir, 1) * conj(psi_it(ir, 1))) * w;
                            }
                            case 0:
                            {
                                for (int ir = 0; ir < fft->size(); ir++)
                                    it_density_matrix(ir, 0, thread_id) += real(psi_it(ir, 0) * conj(psi_it(ir, 0))) * w;
                            }
                        }
                    }
                }
            }));
        }
    }

    for (auto& thread: fft_threads) thread.join();

    if (idx_band != (int)occupied_bands.size()) 
    {
        std::stringstream s;
        s << "not all FFTs are executed" << std::endl
          << " number of wave-functions : " << occupied_bands.size() << ", number of executed FFTs : " << idx_band;
        error_local(__FILE__, __LINE__, s);
    }

    #ifdef _GPU_
    if (parameters_.processing_unit() == GPU && num_fft_threads > 1)
    {
        it_density_matrix_gpu.copy_to_host();
        it_density_matrix_gpu.deallocate_on_device();
    }
    #endif

    /* switch from real density matrix to density and magnetization */
    switch (parameters_.num_mag_dims())
    {
        case 3:
        {
            for (int i = 0; i < num_fft_threads; i++)
            {
                for (int ir = 0; ir < fft_->size(); ir++)
                {
                    magnetization_[1]->f_it<global>(ir) += it_density_matrix(ir, 2, i);
                    magnetization_[2]->f_it<global>(ir) += it_density_matrix(ir, 3, i);
                }
            }
        }
        case 1:
        {
            for (int i = 0; i < num_fft_threads; i++)
            {
                for (int ir = 0; ir < fft_->size(); ir++)
                {
                    rho_->f_it<global>(ir) += (it_density_matrix(ir, 0, i) + it_density_matrix(ir, 1, i));
                    magnetization_[0]->f_it<global>(ir) += (it_density_matrix(ir, 0, i) - it_density_matrix(ir, 1, i));
                }
            }
            break;
        }
        case 0:
        {
            for (int i = 0; i < num_fft_threads; i++)
            {
                for (int ir = 0; ir < fft_->size(); ir++) rho_->f_it<global>(ir) += it_density_matrix(ir, 0, i);
            }
        }
    }
}

void Density::add_q_contribution_to_valence_density(K_set& ks)
{
    Timer t("sirius::Density::add_q_contribution_to_valence_density");

    /* If we have ud and du spin blocks, don't compute one of them (du in this implementation)
     * because density matrix is symmetric.
     */
    int num_zdmat = (parameters_.num_mag_dims() == 3) ? 3 : (parameters_.num_mag_dims() + 1);

    auto uc = parameters_.unit_cell();

    /* complex density matrix */
    mdarray<double_complex, 4> pp_complex_density_matrix(uc->max_mt_basis_size(), uc->max_mt_basis_size(),
                                                         num_zdmat, uc->num_atoms());
    pp_complex_density_matrix.zero();
    
    /* add k-point contribution */
    for (int ikloc = 0; ikloc < (int)ks.spl_num_kpoints().local_size(); ikloc++)
    {
        int ik = (int)ks.spl_num_kpoints(ikloc);
        auto occupied_bands = get_occupied_bands_list(ks.band(), ks[ik]);

        add_kpoint_contribution_pp(ks[ik], occupied_bands, pp_complex_density_matrix);
    }
    parameters_.comm().allreduce(pp_complex_density_matrix.at<CPU>(), (int)pp_complex_density_matrix.size());

    auto rl = parameters_.reciprocal_lattice();

    std::vector<double_complex> f_pw(rl->num_gvec(), complex_zero);

    //== int max_num_atoms = 0;
    //== for (int iat = 0; iat < uc->num_atom_types(); iat++)
    //==     max_num_atoms = std::max(max_num_atoms, uc->atom_type(iat)->num_atoms());

    /* split local fraction of G-vectors between threads */
    splindex<block> spl_ngv_loc(rl->spl_num_gvec().local_size(), Platform::max_num_threads(), 0);

    for (int iat = 0; iat < uc->num_atom_types(); iat++)
    {
        auto atom_type = uc->atom_type(iat);
        int nbf = atom_type->mt_basis_size();

        mdarray<double_complex, 2> d_mtrx_packed(atom_type->num_atoms(), nbf * (nbf + 1) / 2);
        for (int i = 0; i < atom_type->num_atoms(); i++)
        {
            int ia = atom_type->atom_id(i);

            for (int xi2 = 0; xi2 < nbf; xi2++)
            {
                for (int xi1 = 0; xi1 <= xi2; xi1++)
                {
                    d_mtrx_packed(i, xi2 * (xi2 + 1) / 2 + xi1) = pp_complex_density_matrix(xi2, xi1, 0, ia);
                }
            }
        }
        #pragma omp parallel
        {
            mdarray<double_complex, 2> phase_factors(spl_ngv_loc.local_size(), atom_type->num_atoms());

            mdarray<double_complex, 2> d_mtrx_pw(spl_ngv_loc.local_size(), nbf * (nbf + 1) / 2);
    
            int thread_id = Platform::thread_id();

            for (int i = 0; i < atom_type->num_atoms(); i++)
            {
                int ia = atom_type->atom_id(i);

                for (int igloc_t = 0; igloc_t < (int)spl_ngv_loc.local_size(thread_id); igloc_t++)
                {
                    int igloc = (int)spl_ngv_loc.global_index(igloc_t, thread_id);
                    phase_factors(igloc_t, i) = conj(rl->gvec_phase_factor<local>(igloc, ia));
                }
            }

            linalg<CPU>::gemm(0, 0, (int)spl_ngv_loc.local_size(thread_id), nbf * (nbf + 1) / 2, atom_type->num_atoms(),
                              &phase_factors(0, 0), phase_factors.ld(), &d_mtrx_packed(0, 0), d_mtrx_packed.ld(), 
                              &d_mtrx_pw(0, 0), d_mtrx_pw.ld());

            for (int xi2 = 0; xi2 < nbf; xi2++)
            {
                int idx12 = xi2 * (xi2 + 1) / 2;

                /* add diagonal term */
                for (int igloc_t = 0; igloc_t < (int)spl_ngv_loc.local_size(thread_id); igloc_t++)
                {
                    int igloc = (int)spl_ngv_loc.global_index(igloc_t, thread_id);
                    /* D_{xi2,xi2} * Q(G)_{xi2, xi2} */
                    f_pw[rl->spl_num_gvec(igloc)] += d_mtrx_pw(igloc_t, idx12 + xi2) * 
                                                     atom_type->uspp().q_pw(igloc, idx12 + xi2);

                }
                /* add non-diagonal terms */
                for (int xi1 = 0; xi1 < xi2; xi1++, idx12++)
                {
                    for (int igloc_t = 0; igloc_t < (int)spl_ngv_loc.local_size(thread_id); igloc_t++)
                    {
                        int igloc = (int)spl_ngv_loc.global_index(igloc_t, thread_id);
                        /* D_{xi2,xi1} * Q(G)_{xi1, xi2} */
                        f_pw[rl->spl_num_gvec(igloc)] += 2 * real(d_mtrx_pw(igloc_t, idx12) * 
                                                                  atom_type->uspp().q_pw(igloc, idx12));
                    }
                }
            }
        }
    }
    
    parameters_.comm().allgather(&f_pw[0], (int)rl->spl_num_gvec().global_offset(), (int)rl->spl_num_gvec().local_size());

    //fft_->input(rl->num_gvec(), rl->fft_index(), &f_pw[0]);
    //fft_->transform(1);
    //for (int ir = 0; ir < fft_->size(); ir++) rho_->f_it<global>(ir) += real(fft_->buffer(ir));

    for (int ig = 0; ig < rl->num_gvec(); ig++) rho_->f_pw(ig) += f_pw[ig];
}

#ifdef _GPU_

extern "C" void sum_q_pw_d_mtrx_pw_gpu(int num_gvec_loc,
                                       int num_beta,
                                       void* q_pw_t,
                                       void* dm_g,
                                       void* rho_pw);

extern "C" void generate_d_mtrx_pw_gpu(int num_atoms,
                                       int num_gvec_loc,
                                       int num_beta,
                                       double* atom_pos,
                                       int* gvec,
                                       void* d_mtrx_packed,
                                       void* d_mtrx_pw);

void Density::add_q_contribution_to_valence_density_gpu(K_set& ks)
{
    Timer t("sirius::Density::add_q_contribution_to_valence_density_gpu");

    /* If we have ud and du spin blocks, don't compute one of them (du in this implementation)
     * because density matrix is symmetric.
     */
    int num_zdmat = (parameters_.num_mag_dims() == 3) ? 3 : (parameters_.num_mag_dims() + 1);

    auto uc = parameters_.unit_cell();

    /* complex density matrix */
    mdarray<double_complex, 4> pp_complex_density_matrix(uc->max_mt_basis_size(), 
                                                         uc->max_mt_basis_size(),
                                                         num_zdmat, uc->num_atoms());
    pp_complex_density_matrix.allocate_on_device();
    pp_complex_density_matrix.zero_on_device();
    
    /* add k-point contribution */
    for (int ikloc = 0; ikloc < (int)ks.spl_num_kpoints().local_size(); ikloc++)
    {
        int ik = ks.spl_num_kpoints(ikloc);
        std::vector< std::pair<int, double> > occupied_bands = get_occupied_bands_list(ks.band(), ks[ik]);

        add_kpoint_contribution_pp_gpu(ks[ik], occupied_bands, pp_complex_density_matrix);
    }
    pp_complex_density_matrix.copy_to_host();
    pp_complex_density_matrix.deallocate_on_device();

    //parameters_.comm().allreduce(pp_complex_density_matrix.at<CPU>(), (int)pp_complex_density_matrix.size());
    parameters_.mpi_grid().communicator(1 << _dim_k_ | 1 << _dim_col_).allreduce(pp_complex_density_matrix.at<CPU>(), 
                                                                                 (int)pp_complex_density_matrix.size());

    auto rl = parameters_.reciprocal_lattice();

    for (int iat = 0; iat < uc->num_atom_types(); iat++)
    {
         auto type = uc->atom_type(iat);
         type->uspp().q_pw.allocate_on_device();
         type->uspp().q_pw.copy_to_device();
    }

    mdarray<int, 2> gvec(3, rl->spl_num_gvec().local_size());
    for (int igloc = 0; igloc < (int)rl->spl_num_gvec().local_size(); igloc++)
    {
        for (int x = 0; x < 3; x++) gvec(x, igloc) = rl->gvec(rl->spl_num_gvec(igloc))[x];
    }
    gvec.allocate_on_device();
    gvec.copy_to_device();

    std::vector<double_complex> rho_pw(rl->num_gvec(), double_complex(0, 0));
    mdarray<double_complex, 1> rho_pw_gpu(&rho_pw[rl->spl_num_gvec().global_offset()], rl->spl_num_gvec().local_size());
    rho_pw_gpu.allocate_on_device();
    rho_pw_gpu.zero_on_device();

    for (int iat = 0; iat < uc->num_atom_types(); iat++)
    {
        auto type = uc->atom_type(iat);
        int nbf = type->mt_basis_size();

        mdarray<double_complex, 2> d_mtrx_packed(type->num_atoms(), nbf * (nbf + 1) / 2);
        mdarray<double, 2> atom_pos(type->num_atoms(), 3);
        for (int i = 0; i < type->num_atoms(); i++)
        {
            int ia = type->atom_id(i);

            for (int xi2 = 0; xi2 < nbf; xi2++)
            {
                for (int xi1 = 0; xi1 <= xi2; xi1++)
                {
                    d_mtrx_packed(i, xi2 * (xi2 + 1) / 2 + xi1) = pp_complex_density_matrix(xi2, xi1, 0, ia);
                }
            }
            for (int x = 0; x < 3; x++) atom_pos(i, x) = uc->atom(ia)->position(x);
        }
        d_mtrx_packed.allocate_on_device();
        d_mtrx_packed.copy_to_device();
        atom_pos.allocate_on_device();
        atom_pos.copy_to_device();

        mdarray<double_complex, 2> d_mtrx_pw(nullptr, rl->spl_num_gvec().local_size(), nbf * (nbf + 1) / 2);
        d_mtrx_pw.allocate_on_device();
        d_mtrx_pw.zero_on_device();

        generate_d_mtrx_pw_gpu(type->num_atoms(),
                               (int)rl->spl_num_gvec().local_size(),
                               nbf,
                               atom_pos.at<GPU>(),
                               gvec.at<GPU>(),
                               d_mtrx_packed.at<GPU>(),
                               d_mtrx_pw.at<GPU>());

        sum_q_pw_d_mtrx_pw_gpu((int)rl->spl_num_gvec().local_size(), 
                               nbf,
                               type->uspp().q_pw.at<GPU>(),
                               d_mtrx_pw.at<GPU>(),
                               rho_pw_gpu.at<GPU>());
    }

    rho_pw_gpu.copy_to_host();

    parameters_.comm().allgather(&rho_pw[0], (int)rl->spl_num_gvec().global_offset(), (int)rl->spl_num_gvec().local_size());
    
    for (int ig = 0; ig < rl->num_gvec(); ig++) rho_->f_pw(ig) += rho_pw[ig];
    
    //fft_->input(rl->num_gvec(), rl->fft_index(), &rho_pw[0]);
    //fft_->transform(1);
    //for (int ir = 0; ir < fft_->size(); ir++) rho_->f_it<global>(ir) += real(fft_->buffer(ir));
    
    for (int iat = 0; iat < uc->num_atom_types(); iat++) uc->atom_type(iat)->uspp().q_pw.deallocate_on_device();
}
#endif

void Density::generate_valence_density_mt(K_set& ks)
{
    Timer t("sirius::Density::generate_valence_density_mt");

    /* if we have ud and du spin blocks, don't compute one of them (du in this implementation)
       because density matrix is symmetric */
    int num_zdmat = (parameters_.num_mag_dims() == 3) ? 3 : (parameters_.num_mag_dims() + 1);

    // complex density matrix
    mdarray<double_complex, 4> mt_complex_density_matrix(parameters_.unit_cell()->max_mt_basis_size(), 
                                                    parameters_.unit_cell()->max_mt_basis_size(),
                                                    num_zdmat, parameters_.unit_cell()->num_atoms());
    mt_complex_density_matrix.zero();
    
    /* add k-point contribution */
    for (int ikloc = 0; ikloc < (int)ks.spl_num_kpoints().local_size(); ikloc++)
    {
        int ik = ks.spl_num_kpoints(ikloc);
        std::vector< std::pair<int, double> > occupied_bands = get_occupied_bands_list(ks.band(), ks[ik]);
        add_kpoint_contribution_mt(ks[ik], occupied_bands, mt_complex_density_matrix);
    }
    
    mdarray<double_complex, 4> mt_complex_density_matrix_loc(parameters_.unit_cell()->max_mt_basis_size(), 
                                                             parameters_.unit_cell()->max_mt_basis_size(),
                                                             num_zdmat, parameters_.unit_cell()->spl_num_atoms().local_size(0));
   
    for (int j = 0; j < num_zdmat; j++)
    {
        for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
        {
            int ialoc = (int)parameters_.unit_cell()->spl_num_atoms().local_index(ia);
            int rank = parameters_.unit_cell()->spl_num_atoms().local_rank(ia);

           parameters_.comm().reduce(&mt_complex_density_matrix(0, 0, j, ia), &mt_complex_density_matrix_loc(0, 0, j, ialoc),
                                     parameters_.unit_cell()->max_mt_basis_size() * parameters_.unit_cell()->max_mt_basis_size(),
                                     rank);
        }
    }
   
    // compute occupation matrix
    if (parameters_.uj_correction())
    {
        Timer* t3 = new Timer("sirius::Density::generate:om");
        
        mdarray<double_complex, 4> occupation_matrix(16, 16, 2, 2); 
        
        for (int ialoc = 0; ialoc < (int)parameters_.unit_cell()->spl_num_atoms().local_size(); ialoc++)
        {
            int ia = parameters_.unit_cell()->spl_num_atoms(ialoc);
            Atom_type* type = parameters_.unit_cell()->atom(ia)->type();
            
            occupation_matrix.zero();
            for (int l = 0; l <= 3; l++)
            {
                int num_rf = type->indexr().num_rf(l);

                for (int j = 0; j < num_zdmat; j++)
                {
                    for (int order2 = 0; order2 < num_rf; order2++)
                    {
                    for (int lm2 = Utils::lm_by_l_m(l, -l); lm2 <= Utils::lm_by_l_m(l, l); lm2++)
                    {
                        for (int order1 = 0; order1 < num_rf; order1++)
                        {
                        for (int lm1 = Utils::lm_by_l_m(l, -l); lm1 <= Utils::lm_by_l_m(l, l); lm1++)
                        {
                            occupation_matrix(lm1, lm2, dmat_spins_[j].first, dmat_spins_[j].second) +=
                                mt_complex_density_matrix_loc(type->indexb_by_lm_order(lm1, order1),
                                                              type->indexb_by_lm_order(lm2, order2), j, ialoc) *
                                parameters_.unit_cell()->atom(ia)->symmetry_class()->o_radial_integral(l, order1, order2);
                        }
                        }
                    }
                    }
                }
            }
        
            // restore the du block
            for (int lm1 = 0; lm1 < 16; lm1++)
            {
                for (int lm2 = 0; lm2 < 16; lm2++)
                    occupation_matrix(lm2, lm1, 1, 0) = conj(occupation_matrix(lm1, lm2, 0, 1));
            }

            parameters_.unit_cell()->atom(ia)->set_occupation_matrix(&occupation_matrix(0, 0, 0, 0));
        }

        for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
        {
            int rank = parameters_.unit_cell()->spl_num_atoms().local_rank(ia);
            parameters_.unit_cell()->atom(ia)->sync_occupation_matrix(parameters_.comm(), rank);
        }

        delete t3;
    }

    int max_num_rf_pairs = parameters_.unit_cell()->max_mt_radial_basis_size() * 
                           (parameters_.unit_cell()->max_mt_radial_basis_size() + 1) / 2;
    
    // real density matrix
    mdarray<double, 3> mt_density_matrix(parameters_.lmmax_rho(), max_num_rf_pairs, parameters_.num_mag_dims() + 1);
    
    mdarray<double, 2> rf_pairs(parameters_.unit_cell()->max_num_mt_points(), max_num_rf_pairs);
    mdarray<double, 3> dlm(parameters_.lmmax_rho(), parameters_.unit_cell()->max_num_mt_points(), 
                           parameters_.num_mag_dims() + 1);
    for (int ialoc = 0; ialoc < (int)parameters_.unit_cell()->spl_num_atoms().local_size(); ialoc++)
    {
        int ia = (int)parameters_.unit_cell()->spl_num_atoms(ialoc);
        Atom_type* atom_type = parameters_.unit_cell()->atom(ia)->type();

        int nmtp = atom_type->num_mt_points();
        int num_rf_pairs = atom_type->mt_radial_basis_size() * (atom_type->mt_radial_basis_size() + 1) / 2;
        
        Timer t1("sirius::Density::generate|sum_zdens");
        switch (parameters_.num_mag_dims())
        {
            case 3:
            {
                reduce_zdens<3>(atom_type, ialoc, mt_complex_density_matrix_loc, mt_density_matrix);
                break;
            }
            case 1:
            {
                reduce_zdens<1>(atom_type, ialoc, mt_complex_density_matrix_loc, mt_density_matrix);
                break;
            }
            case 0:
            {
                reduce_zdens<0>(atom_type, ialoc, mt_complex_density_matrix_loc, mt_density_matrix);
                break;
            }
        }
        t1.stop();
        
        Timer t2("sirius::Density::generate|expand_lm");
        // collect radial functions
        for (int idxrf2 = 0; idxrf2 < atom_type->mt_radial_basis_size(); idxrf2++)
        {
            int offs = idxrf2 * (idxrf2 + 1) / 2;
            for (int idxrf1 = 0; idxrf1 <= idxrf2; idxrf1++)
            {
                // off-diagonal pairs are taken two times: d_{12}*f_1*f_2 + d_{21}*f_2*f_1 = d_{12}*2*f_1*f_2
                int n = (idxrf1 == idxrf2) ? 1 : 2; 
                for (int ir = 0; ir < parameters_.unit_cell()->atom(ia)->type()->num_mt_points(); ir++)
                {
                    rf_pairs(ir, offs + idxrf1) = n * parameters_.unit_cell()->atom(ia)->symmetry_class()->radial_function(ir, idxrf1) * 
                                                      parameters_.unit_cell()->atom(ia)->symmetry_class()->radial_function(ir, idxrf2); 
                }
            }
        }
        for (int j = 0; j < parameters_.num_mag_dims() + 1; j++)
        {
            linalg<CPU>::gemm(0, 1, parameters_.lmmax_rho(), nmtp, num_rf_pairs, 
                              &mt_density_matrix(0, 0, j), mt_density_matrix.ld(), 
                              &rf_pairs(0, 0), rf_pairs.ld(), &dlm(0, 0, j), dlm.ld());
        }

        int sz = parameters_.lmmax_rho() * nmtp * (int)sizeof(double);
        switch (parameters_.num_mag_dims())
        {
            case 3:
            {
                memcpy(&magnetization_[1]->f_mt<local>(0, 0, ialoc), &dlm(0, 0, 2), sz); 
                memcpy(&magnetization_[2]->f_mt<local>(0, 0, ialoc), &dlm(0, 0, 3), sz);
            }
            case 1:
            {
                for (int ir = 0; ir < nmtp; ir++)
                {
                    for (int lm = 0; lm < parameters_.lmmax_rho(); lm++)
                    {
                        rho_->f_mt<local>(lm, ir, ialoc) = dlm(lm, ir, 0) + dlm(lm, ir, 1);
                        magnetization_[0]->f_mt<local>(lm, ir, ialoc) = dlm(lm, ir, 0) - dlm(lm, ir, 1);
                    }
                }
                break;
            }
            case 0:
            {
                memcpy(&rho_->f_mt<local>(0, 0, ialoc), &dlm(0, 0, 0), sz);
            }
        }
        t2.stop();
    }
}

void Density::generate_valence_density_it(K_set& ks)
{
    Timer t("sirius::Density::generate_valence_density_it");

    /* add k-point contribution */
    for (int ikloc = 0; ikloc < (int)ks.spl_num_kpoints().local_size(); ikloc++)
    {
        int ik = ks.spl_num_kpoints(ikloc);
        auto occupied_bands = get_occupied_bands_list(ks.band(), ks[ik]);
        add_kpoint_contribution_it(ks[ik], occupied_bands);
    }
    
    /* reduce arrays; assume that each rank did it's own fraction of the density */
    parameters_.comm().allreduce(&rho_->f_it<global>(0), fft_->size()); 
    for (int j = 0; j < parameters_.num_mag_dims(); j++)
        parameters_.comm().allreduce(&magnetization_[j]->f_it<global>(0), fft_->size()); 
}

double Density::core_leakage()
{
    double sum = 0.0;
    for (int ic = 0; ic < parameters_.unit_cell()->num_atom_symmetry_classes(); ic++)
    {
        sum += parameters_.unit_cell()->atom_symmetry_class(ic)->core_leakage() * 
               parameters_.unit_cell()->atom_symmetry_class(ic)->num_atoms();
    }
    return sum;
}

double Density::core_leakage(int ic)
{
    return parameters_.unit_cell()->atom_symmetry_class(ic)->core_leakage();
}

void Density::generate_core_charge_density()
{
    Timer t("sirius::Density::generate_core_charge_density");

    for (int icloc = 0; icloc < (int)parameters_.unit_cell()->spl_num_atom_symmetry_classes().local_size(); icloc++)
    {
        int ic = parameters_.unit_cell()->spl_num_atom_symmetry_classes(icloc);
        parameters_.unit_cell()->atom_symmetry_class(ic)->generate_core_charge_density();
    }

    for (int ic = 0; ic < parameters_.unit_cell()->num_atom_symmetry_classes(); ic++)
    {
        int rank = parameters_.unit_cell()->spl_num_atom_symmetry_classes().local_rank(ic);
        parameters_.unit_cell()->atom_symmetry_class(ic)->sync_core_charge_density(parameters_.comm(), rank);
    }
}

void Density::generate_pseudo_core_charge_density()
{
    Timer t("sirius::Density::generate_pseudo_core_charge_density");

    auto rl = parameters_.reciprocal_lattice();
    auto rho_core_radial_integrals = generate_rho_radial_integrals(2);

    std::vector<double_complex> v = rl->make_periodic_function(rho_core_radial_integrals, rl->num_gvec());

    fft_->input(fft_->num_gvec(), fft_->index_map(), &v[0]);
    fft_->transform(1);
    fft_->output(&rho_pseudo_core_->f_it<global>(0));
}

void Density::generate_valence(K_set& ks__)
{
    Timer t("sirius::Density::generate_valence");
    
    double wt = 0.0;
    double ot = 0.0;
    for (int ik = 0; ik < ks__.num_kpoints(); ik++)
    {
        wt += ks__[ik]->weight();
        for (int j = 0; j < parameters_.num_bands(); j++) ot += ks__[ik]->weight() * ks__[ik]->band_occupancy(j);
    }

    if (fabs(wt - 1.0) > 1e-12) error_local(__FILE__, __LINE__, "K_point weights don't sum to one");

    if (fabs(ot - parameters_.unit_cell()->num_valence_electrons()) > 1e-8)
    {
        std::stringstream s;
        s << "wrong occupancies" << std::endl
          << "  computed : " << ot << std::endl
          << "  required : " << parameters_.unit_cell()->num_valence_electrons() << std::endl
          << "  difference : " << fabs(ot - parameters_.unit_cell()->num_valence_electrons());
        warning_local(__FILE__, __LINE__, s);
    }

    /* zero density and magnetization */
    zero();

    /* interstitial part is independent of basis type */
    generate_valence_density_it(ks__);

    /* for muffin-tin part */
    switch (parameters_.esm_type())
    {
        case full_potential_lapwlo:
        {
            generate_valence_density_mt(ks__);
            break;
        }
        case full_potential_pwlo:
        {
            STOP();
        }
        default:
        {
            break;
        }
    }
    
    /* get rho(G) */
    fft_->input(&rho_->f_it<global>(0));
    fft_->transform(-1);
    fft_->output(fft_->num_gvec(), fft_->index_map(), &rho_->f_pw(0));

    if (parameters_.esm_type() == ultrasoft_pseudopotential ||
        parameters_.esm_type() == norm_conserving_pseudopotential)
    {
        augment(ks__);
    }
}

void Density::augment(K_set& ks__)
{
    switch (parameters_.esm_type())
    {
        case ultrasoft_pseudopotential:
        {
            switch (parameters_.processing_unit())
            {
                case CPU:
                {
                    add_q_contribution_to_valence_density(ks__);
                    break;
                }
                #ifdef _GPU_
                case GPU:
                {
                    add_q_contribution_to_valence_density_gpu(ks__);
                    break;
                }
                #endif
                default:
                {
                    error_local(__FILE__, __LINE__, "wrong processing unit");
                }
            }
            break;
        }
        default:
        {
            break;
        }
    }
}

void Density::generate(K_set& ks__)
{
    Timer t("sirius::Density::generate");
    
    //== double wt = 0.0;
    //== double ot = 0.0;
    //== for (int ik = 0; ik < ks__.num_kpoints(); ik++)
    //== {
    //==     wt += ks__[ik]->weight();
    //==     for (int j = 0; j < parameters_.num_bands(); j++) ot += ks__[ik]->weight() * ks__[ik]->band_occupancy(j);
    //== }

    //== if (fabs(wt - 1.0) > 1e-12) error_local(__FILE__, __LINE__, "K_point weights don't sum to one");

    //== if (fabs(ot - parameters_.unit_cell()->num_valence_electrons()) > 1e-8)
    //== {
    //==     std::stringstream s;
    //==     s << "wrong occupancies" << std::endl
    //==       << "  computed : " << ot << std::endl
    //==       << "  required : " << parameters_.unit_cell()->num_valence_electrons() << std::endl
    //==       << "  difference : " << fabs(ot - parameters_.unit_cell()->num_valence_electrons());
    //==     warning_local(__FILE__, __LINE__, s);
    //== }

    //== /* zero density and magnetization */
    //== zero();

    //== /* interstitial part is independent of basis type */
    //== generate_valence_density_it(ks__);

    //== switch (parameters_.esm_type())
    //== {
    //==     case full_potential_lapwlo:
    //==     {
    //==         /* muffin-tin part */
    //==         generate_valence_density_mt(ks__);
    //==         break;
    //==     }
    //==     case full_potential_pwlo:
    //==     {
    //==         switch (parameters_.processing_unit())
    //==         {
    //==             STOP();
    //==             case CPU:
    //==             {
    //==                 break;
    //==             }
    //==             #ifdef _GPU_
    //==             case GPU:
    //==             {
    //==                 break;
    //==             }
    //==             #endif
    //==             default:
    //==             {
    //==                 error_local(__FILE__, __LINE__, "wrong processing unit");
    //==             }
    //==         }
    //==         break;
    //==     }
    //==     case ultrasoft_pseudopotential:
    //==     {
    //==         switch (parameters_.processing_unit())
    //==         {
    //==             case CPU:
    //==             {
    //==                 add_q_contribution_to_valence_density(ks__);
    //==                 break;
    //==             }
    //==             #ifdef _GPU_
    //==             case GPU:
    //==             {
    //==                 add_q_contribution_to_valence_density_gpu(ks__);
    //==                 break;
    //==             }
    //==             #endif
    //==             default:
    //==             {
    //==                 error_local(__FILE__, __LINE__, "wrong processing unit");
    //==             }
    //==         }
    //==         break;
    //==     }
    //==     case norm_conserving_pseudopotential:
    //==     {
    //==         break;
    //==     }
    //== }

    generate_valence(ks__);

    if (parameters_.unit_cell()->full_potential())
    {
        generate_core_charge_density();

        /* add core contribution */
        for (int ialoc = 0; ialoc < (int)parameters_.unit_cell()->spl_num_atoms().local_size(); ialoc++)
        {
            int ia = parameters_.unit_cell()->spl_num_atoms(ialoc);
            for (int ir = 0; ir < parameters_.unit_cell()->atom(ia)->num_mt_points(); ir++)
                rho_->f_mt<local>(0, ir, ialoc) += parameters_.unit_cell()->atom(ia)->symmetry_class()->core_charge_density(ir) / y00;
        }

        /* synchronize muffin-tin part (interstitial is already syncronized with allreduce) */
        rho_->sync(true, false);
        for (int j = 0; j < parameters_.num_mag_dims(); j++) magnetization_[j]->sync(true, false);
    }
    
    double nel = 0;
    if (parameters_.esm_type() == full_potential_lapwlo ||
        parameters_.esm_type() == full_potential_pwlo)
    {
        std::vector<double> nel_mt;
        double nel_it;
        nel = rho_->integrate(nel_mt, nel_it);
    }
    if (parameters_.esm_type() == ultrasoft_pseudopotential ||
         parameters_.esm_type() == norm_conserving_pseudopotential)
    {
        nel = real(rho_->f_pw(0)) * parameters_.unit_cell()->omega();
    }

    if (fabs(nel - parameters_.unit_cell()->num_electrons()) > 1e-5)
    {
        std::stringstream s;
        s << "wrong charge density after k-point summation" << std::endl
          << "obtained value : " << nel << std::endl 
          << "target value : " << parameters_.unit_cell()->num_electrons() << std::endl
          << "difference : " << fabs(nel - parameters_.unit_cell()->num_electrons()) << std::endl;
        if (parameters_.unit_cell()->full_potential())
        {
            s << "total core leakage : " << core_leakage();
            for (int ic = 0; ic < parameters_.unit_cell()->num_atom_symmetry_classes(); ic++) 
                s << std::endl << "  atom class : " << ic << ", core leakage : " << core_leakage(ic);
        }
        warning_global(__FILE__, __LINE__, s);
    }

    //if (debug_level > 1) check_density_continuity_at_mt();
}

//void Density::check_density_continuity_at_mt()
//{
//    // generate plane-wave coefficients of the potential in the interstitial region
//    parameters_.fft().input(&rho_->f_it<global>(0));
//    parameters_.fft().transform(-1);
//    parameters_.fft().output(parameters_.num_gvec(), parameters_.fft_index(), &rho_->f_pw(0));
//    
//    SHT sht(parameters_.lmax_rho());
//
//    double diff = 0.0;
//    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//    {
//        for (int itp = 0; itp < sht.num_points(); itp++)
//        {
//            double vc[3];
//            for (int x = 0; x < 3; x++) vc[x] = sht.coord(x, itp) * parameters_.atom(ia)->mt_radius();
//
//            double val_it = 0.0;
//            for (int ig = 0; ig < parameters_.num_gvec(); ig++) 
//            {
//                double vgc[3];
//                parameters_.get_coordinates<cartesian, reciprocal>(parameters_.gvec(ig), vgc);
//                val_it += real(rho_->f_pw(ig) * exp(double_complex(0.0, Utils::scalar_product(vc, vgc))));
//            }
//
//            double val_mt = 0.0;
//            for (int lm = 0; lm < parameters_.lmmax_rho(); lm++)
//                val_mt += rho_->f_rlm(lm, parameters_.atom(ia)->num_mt_points() - 1, ia) * sht.rlm_backward(lm, itp);
//
//            diff += fabs(val_it - val_mt);
//        }
//    }
//    printf("Total and average charge difference at MT boundary : %.12f %.12f\n", diff, diff / parameters_.num_atoms() / sht.num_points());
//}


void Density::save()
{
    if (parameters_.comm().rank() == 0)
    {
        HDF5_tree fout(storage_file_name, false);
        rho_->hdf5_write(fout["density"]);
        for (int j = 0; j < parameters_.num_mag_dims(); j++)
            magnetization_[j]->hdf5_write(fout["magnetization"].create_node(j));
    }
    parameters_.comm().barrier();
}

void Density::load()
{
    HDF5_tree fout(storage_file_name, false);
    rho_->hdf5_read(fout["density"]);
    for (int j = 0; j < parameters_.num_mag_dims(); j++)
        magnetization_[j]->hdf5_read(fout["magnetization"][j]);
}

void Density::generate_pw_coefs()
{
    fft_->input(&rho_->f_it<global>(0));
    fft_->transform(-1);

    fft_->output(fft_->num_gvec(), fft_->index_map(), &rho_->f_pw(0));
}

}
