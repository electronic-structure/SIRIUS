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

/** \file dft_ground_state.cpp
 *
 *  \brief Contains remaining implementation of sirius::DFT_ground_state class.
 */

#include "dft_ground_state.h"

//#include <tbb/task_scheduler_init.h>
//#include <omp.h>

namespace sirius {

double DFT_ground_state::ewald_energy()
{
    runtime::Timer t("sirius::DFT_ground_state::ewald_energy");

    double alpha = 1.5;
    
    double ewald_g = 0;

    #pragma omp parallel
    {
        double ewald_g_pt = 0;

        #pragma omp for
        for (int igloc = 0; igloc < ctx_.gvec_count(); igloc++) {
            int ig = ctx_.gvec_offset() + igloc;

            double g2 = std::pow(ctx_.gvec().shell_len(ctx_.gvec().shell(ig)), 2);

            double_complex rho(0, 0);

            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                rho += ctx_.gvec_phase_factor(ig, ia) * static_cast<double>(unit_cell_.atom(ia).zn());
            }

            if (ig) {
                ewald_g_pt += std::pow(std::abs(rho), 2) * std::exp(-g2 / 4 / alpha) / g2;
            } else {
                ewald_g_pt -= std::pow(unit_cell_.num_electrons(), 2) / alpha / 4; // constant term in QE comments
            }

            if (ctx_.gvec().reduced() && ig) {
                rho = 0;
                for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                    rho += std::conj(ctx_.gvec_phase_factor(ig, ia)) * static_cast<double>(unit_cell_.atom(ia).zn());
                }

                ewald_g_pt += std::pow(std::abs(rho), 2) * std::exp(-g2 / 4 / alpha) / g2;
            }
        }

        #pragma omp critical
        ewald_g += ewald_g_pt;
    }
    ctx_.comm().allreduce(&ewald_g, 1);
    ewald_g *= (twopi / unit_cell_.omega());

    /* remove self-interaction */
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        ewald_g -= std::sqrt(alpha / pi) * std::pow(unit_cell_.atom(ia).zn(), 2);
    }

    double ewald_r = 0;
    #pragma omp parallel
    {
        double ewald_r_pt = 0;

        #pragma omp for
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            for (int i = 1; i < unit_cell_.num_nearest_neighbours(ia); i++) {
                int ja = unit_cell_.nearest_neighbour(i, ia).atom_id;
                double d = unit_cell_.nearest_neighbour(i, ia).distance;
                ewald_r_pt += 0.5 * unit_cell_.atom(ia).zn() * unit_cell_.atom(ja).zn() *
                              gsl_sf_erfc(std::sqrt(alpha) * d) / d;
            }
        }

        #pragma omp critical
        ewald_r += ewald_r_pt;
    }

    return (ewald_g + ewald_r);
}


void DFT_ground_state::move_atoms(int istep)
{
    STOP();

    //mdarray<double, 2> atom_force(3, unit_cell_.num_atoms());
    //forces(atom_force);
    //#if (__VERBOSITY > 0)
    //if (ctx_.comm().rank() == 0)
    //{
    //    printf("\n");
    //    printf("Atomic forces\n");
    //    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
    //    {
    //        printf("ia : %i, force : %12.6f %12.6f %12.6f\n", ia, atom_force(0, ia), atom_force(1, ia), atom_force(2, ia));
    //    }
    //}
    //#endif

    //for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
    //{
    //    vector3d<double> pos = unit_cell_.atom(ia).position();
    //    
    //    vector3d<double> f(atom_force(0, ia), atom_force(1, ia), atom_force(2, ia));
    //    vector3d<double> forcef = unit_cell_.get_fractional_coordinates(f);

    //    for (int x = 0; x < 3; x++) pos[x] += forcef[x];
    //    
    //    unit_cell_.atom(ia).set_position(pos);
    //}
}




mdarray<double,2 > DFT_ground_state::forces()
{
    //STOP();

    mdarray<double,2 > loc_forces = forces_->calc_local_forces( );

    //Force::total_force(ctx_, potential_, density_, kset_, forces__);

    std::cout<<"===== Forces: local contribution =====" << std::endl;

    for(int ia=0; ia < unit_cell_.num_atoms(); ia++)
    {
        std::cout<< loc_forces(0,ia) <<"   "<< loc_forces(1,ia) << "   " << loc_forces(2,ia) << std::endl;
    }

    mdarray<double,2 > us_forces = forces_->calc_ultrasoft_forces();

    std::cout<<"===== Forces: ultrasoft contribution =====" << std::endl;

    for(int ia=0; ia < unit_cell_.num_atoms(); ia++)
    {
        std::cout<< us_forces(0,ia) <<"   "<< us_forces(1,ia) << "   " << us_forces(2,ia) << std::endl;
    }

    return std::move(loc_forces);
}



int DFT_ground_state::find(double potential_tol, double energy_tol, int num_dft_iter)
{
    runtime::Timer t("sirius::DFT_ground_state::scf_loop");
    
    double eold{0}, rms{0};

    if (ctx_.full_potential()) {
        potential_.mixer_init();
    } else {
        density_.mixer_init();
    }

    int result{-1};

//    tbb::task_scheduler_init tbb_init(omp_get_num_threads());

    for (int iter = 0; iter < num_dft_iter; iter++) {
        runtime::Timer t1("sirius::DFT_ground_state::scf_loop|iteration");

        /* find new wave-functions */
        band_.solve_for_kset(kset_, potential_, true);
        /* find band occupancies */
        kset_.find_band_occupancies();
        /* generate new density from the occupied wave-functions */
        density_.generate(kset_);
        /* compute new total energy for a new density */
        double etot = total_energy();
        /* symmetrize density and magnetization */
        if (use_symmetry_) {
            symmetrize(density_.rho(), density_.magnetization(0), density_.magnetization(1),
                       density_.magnetization(2));
        }
        /* set new tolerance of iterative solver */
        if (!ctx_.full_potential()) {
            rms = density_.mix();
            double tol = std::max(1e-12, 0.1 * density_.dr2() / ctx_.unit_cell().num_valence_electrons());
            if (ctx_.comm().rank() == 0) {
                printf("dr2: %18.10f, tol: %18.10f\n",  density_.dr2(), tol);
            }
            ctx_.set_iterative_solver_tolerance(std::min(ctx_.iterative_solver_tolerance(), tol));
        }

        if (ctx_.esm_type() == electronic_structure_method_t::paw_pseudopotential) {
            density_.generate_paw_loc_density();
        }

        //== if (ctx_.num_mag_dims())
        //== {
        //==     for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
        //==     {
        //==         vector3d<double> mag(0, 0, 0);

        //==         for (int j0 = 0; j0 < ctx_.fft().grid().size(0); j0++)
        //==         {
        //==             for (int j1 = 0; j1 < ctx_.fft().grid().size(1); j1++)
        //==             {
        //==                 for (int j2 = 0; j2 < ctx_.fft().local_size_z(); j2++)
        //==                 {
        //==                     /* get real space fractional coordinate */
        //==                     auto v0 = vector3d<double>(double(j0) / ctx_.fft().grid().size(0), 
        //==                                                double(j1) / ctx_.fft().grid().size(1), 
        //==                                                double(ctx_.fft().offset_z() + j2) / ctx_.fft().grid().size(2));
        //==                     /* index of real space point */
        //==                     int ir = ctx_.fft().grid().index_by_coord(j0, j1, j2);

        //==                     for (int t0 = -1; t0 <= 1; t0++)
        //==                     {
        //==                         for (int t1 = -1; t1 <= 1; t1++)
        //==                         {
        //==                             for (int t2 = -1; t2 <= 1; t2++)
        //==                             {
        //==                                 vector3d<double> v1 = v0 - (unit_cell_.atom(ia).position() + vector3d<double>(t0, t1, t2));
        //==                                 auto r = unit_cell_.get_cartesian_coordinates(v1);
        //==                                 auto a = r.length();

        //==                                 if (a <= 2.0)
        //==                                 {
        //==                                     mag[2] += density_.magnetization(0)->f_rg(ir);
        //==                                 }
        //==                             }
        //==                         }
        //==                     }
        //==                 }
        //==             }
        //==         }
        //==         for (int x: {0, 1, 2}) mag[x] *= (unit_cell_.omega() / ctx_.fft().size());
        //==         printf("atom: %i, mag: %f %f %f\n", ia, mag[0], mag[1], mag[2]);
        //==     }
        //== }

        /* compute new potential */
        generate_effective_potential();

        /* symmetrize potential and effective magnetic field */
        if (use_symmetry_) {
            symmetrize(potential_.effective_potential(), potential_.effective_magnetic_field(0),
                       potential_.effective_magnetic_field(1), potential_.effective_magnetic_field(2));
        }

        if (ctx_.full_potential()) {
            rms = potential_.mix();
            double tol = std::max(1e-12, rms);
            if (ctx_.comm().rank() == 0) {
                printf("tol: %18.10f\n", tol);
            }
            ctx_.set_iterative_solver_tolerance(std::min(ctx_.iterative_solver_tolerance(), tol));
        }


        /* write some information */
        print_info();

        if (ctx_.comm().rank() == 0) {
            printf("iteration : %3i, RMS %18.12f, energy difference : %12.6f\n", iter, rms, etot - eold);
        }
        
        if (std::abs(eold - etot) < energy_tol && rms < potential_tol) {
            result = iter;
            break;
        }

        eold = etot;
    }
    
    ctx_.create_storage_file();
    potential_.save();
    density_.save();

//    tbb_init.terminate();

    return result;
}



void DFT_ground_state::relax_atom_positions()
{
    STOP();

    //for (int i = 0; i < 5; i++)
    //{
    //    scf_loop(1e-4, 1e-4, 100);
    //    move_atoms(i);
    //    update();
    //    //ctx_.print_info();
    //}
}



void DFT_ground_state::print_info()
{
    double evalsum1 = kset_.valence_eval_sum();
    double evalsum2 = core_eval_sum();
    double ekin = energy_kin();
    double evxc = energy_vxc();
    double eexc = energy_exc();
    double ebxc = energy_bxc();
    double evha = energy_vha();
    double etot = total_energy();
    double gap = kset_.band_gap() * ha2ev;
    double ef = kset_.energy_fermi();
    double core_leak = density_.core_leakage();
    double enuc = energy_enuc();

    double one_elec_en = evalsum1 - (evxc + evha);

    if (ctx_.esm_type() == electronic_structure_method_t::paw_pseudopotential) {
        one_elec_en -= potential_.PAW_one_elec_energy();
    }

    std::vector<double> mt_charge;
    double it_charge;
    double total_charge = density_.rho()->integrate(mt_charge, it_charge); 
    
    double total_mag[3];
    std::vector<double> mt_mag[3];
    double it_mag[3];
    for (int j = 0; j < ctx_.num_mag_dims(); j++) {
        total_mag[j] = density_.magnetization(j)->integrate(mt_mag[j], it_mag[j]);
    }
    
    if (ctx_.comm().rank() == 0) {
        if (ctx_.full_potential()) {
            double total_core_leakage = 0.0;
            printf("\n");
            printf("Charges and magnetic moments\n");
            for (int i = 0; i < 80; i++) printf("-");
            printf("\n"); 
            printf("atom      charge    core leakage");
            if (ctx_.num_mag_dims()) printf("              moment                |moment|");
            printf("\n");
            for (int i = 0; i < 80; i++) printf("-");
            printf("\n"); 

            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                double core_leakage = unit_cell_.atom(ia).symmetry_class().core_leakage();
                total_core_leakage += core_leakage;
                printf("%4i  %10.6f  %10.8e", ia, mt_charge[ia], core_leakage);
                if (ctx_.num_mag_dims()) {
                    vector3d<double> v;
                    v[2] = mt_mag[0][ia];
                    if (ctx_.num_mag_dims() == 3) {
                        v[0] = mt_mag[1][ia];
                        v[1] = mt_mag[2][ia];
                    }
                    printf("  [%8.4f, %8.4f, %8.4f]  %10.6f", v[0], v[1], v[2], v.length());
                }
                printf("\n");
            }
            
            printf("\n");
            printf("interstitial charge   : %10.6f\n", it_charge);
            if (ctx_.num_mag_dims()) {
                vector3d<double> v;
                v[2] = it_mag[0];
                if (ctx_.num_mag_dims() == 3) {
                    v[0] = it_mag[1];
                    v[1] = it_mag[2];
                }
                printf("interstitial moment   : [%8.4f, %8.4f, %8.4f], magnitude : %10.6f\n", 
                       v[0], v[1], v[2], v.length());
            }
            
            printf("\n");
            printf("total charge          : %10.6f\n", total_charge);
            printf("total core leakage    : %10.8e\n", total_core_leakage);
            if (ctx_.num_mag_dims()) {
                vector3d<double> v;
                v[2] = total_mag[0];
                if (ctx_.num_mag_dims() == 3) {
                    v[0] = total_mag[1];
                    v[1] = total_mag[2];
                }
                printf("total moment          : [%8.4f, %8.4f, %8.4f], magnitude : %10.6f\n", 
                       v[0], v[1], v[2], v.length());
            }
        }
        printf("\n");
        printf("Energy\n");
        for (int i = 0; i < 80; i++) printf("-");
        printf("\n"); 

        printf("valence_eval_sum          : %18.8f\n", evalsum1);
        if (ctx_.full_potential()) {
            printf("core_eval_sum             : %18.8f\n", evalsum2);
            printf("kinetic energy            : %18.8f\n", ekin);
            printf("enuc                      : %18.8f\n", enuc);
        }
        printf("<rho|V^{XC}>              : %18.8f\n", evxc);
        printf("<rho|E^{XC}>              : %18.8f\n", eexc);
        printf("<mag|B^{XC}>              : %18.8f\n", ebxc);
        printf("<rho|V^{H}>               : %18.8f\n", evha);
        if (!ctx_.full_potential()) {
            printf("one-electron contribution : %18.8f\n", one_elec_en); // eband + deband in QE
            printf("hartree contribution      : %18.8f\n", 0.5 * evha);
            printf("xc contribution           : %18.8f\n", eexc);
            printf("ewald contribution        : %18.8f\n", ewald_energy_);
            printf("PAW contribution          : %18.8f\n", potential_.PAW_total_energy());
        }
        printf("Total energy              : %18.8f\n", etot);

        printf("\n");
        printf("band gap (eV) : %18.8f\n", gap);
        printf("Efermi        : %18.8f\n", ef);
        printf("\n");
        if (ctx_.full_potential()) {
            printf("core leakage : %18.8f\n", core_leak);
        }
    }
}



void DFT_ground_state::initialize_subspace()
{
    PROFILE_WITH_TIMER("sirius::DFT_ground_state::initialize_subspace");

    int nq = 20;
    int lmax = 4;
    /* this is the regular grid in reciprocal space in the range [0, |G+k|_max ] */
    Radial_grid qgrid(linear_grid, nq, 0, ctx_.gk_cutoff());

    /* interpolate I_{\alpha,n}(q) = <j_{l_n}(q*x) | wf_{n,l_n}(x) > with splines */
    std::vector< std::vector< Spline<double> > > rad_int(unit_cell_.num_atom_types());
    
    /* spherical Bessel functions jl(qx) for atom types */
    mdarray<Spherical_Bessel_functions, 2> jl(nq, unit_cell_.num_atom_types());

    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        auto& atom_type = unit_cell_.atom_type(iat);
        /* create jl(qx) */
        #pragma omp parallel for
        for (int iq = 0; iq < nq; iq++) {
            jl(iq, iat) = Spherical_Bessel_functions(lmax, atom_type.radial_grid(), qgrid[iq]);
        }

        rad_int[iat].resize(atom_type.uspp().atomic_pseudo_wfs_.size());
        /* loop over all pseudo wave-functions */
        for (size_t i = 0; i < atom_type.uspp().atomic_pseudo_wfs_.size(); i++) {
            rad_int[iat][i] = Spline<double>(qgrid);
            
            /* interpolate atomic_pseudo_wfs(r) */
            Spline<double> wf(atom_type.radial_grid());
            for (int ir = 0; ir < atom_type.num_mt_points(); ir++) {
                wf[ir] = atom_type.uspp().atomic_pseudo_wfs_[i].second[ir];
            }
            wf.interpolate();
            
            int l = atom_type.uspp().atomic_pseudo_wfs_[i].first;
            #pragma omp parallel for
            for (int iq = 0; iq < nq; iq++) {
                rad_int[iat][i][iq] = sirius::inner(jl(iq, iat)[l], wf, 1);
            }

            rad_int[iat][i].interpolate();
        }
    }

    /* get the total number of atomic-centered orbitals */
    int N{0};
    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        auto& atom_type = unit_cell_.atom_type(iat);
        int n{0};
        for (auto& wf: atom_type.uspp().atomic_pseudo_wfs_) {
            n += (2 * wf.first + 1);
        }
        N += atom_type.num_atoms() * n;
    }
    printf("number of atomic orbitals: %i\n", N);

    for (int ikloc = 0; ikloc < kset_.spl_num_kpoints().local_size(); ikloc++) {
        int ik = kset_.spl_num_kpoints(ikloc);
        auto kp = kset_[ik];
        
        if (ctx_.gamma_point()) {
            band_.initialize_subspace<double>(kp, potential_.effective_potential(),
                                              potential_.effective_magnetic_field(), N, lmax, rad_int);
        } else {
            band_.initialize_subspace<double_complex>(kp, potential_.effective_potential(),
                                                      potential_.effective_magnetic_field(), N, lmax, rad_int);
        }
    }

    kset_.find_band_occupancies();

    /* reset the energies for the iterative solver to do at least two steps */
    for (int ik = 0; ik < kset_.num_kpoints(); ik++) {
        for (int i = 0; i < ctx_.num_bands(); i++) {
            kset_[ik]->band_energy(i) = 0;
        }
    }
}

}
