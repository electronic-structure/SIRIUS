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

namespace sirius {

double DFT_ground_state::energy_enuc()
{
    double enuc = 0.0;
    if (parameters_.full_potential())
    {
        for (size_t ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++)
        {
            int ia = unit_cell_.spl_num_atoms((int)ialoc);
            int zn = unit_cell_.atom(ia)->type()->zn();
            enuc -= 0.5 * zn * potential_->vh_el(ia) * y00;
        }
        ctx_.comm().allreduce(&enuc, 1);
    }
    
    return enuc;
}

double DFT_ground_state::core_eval_sum()
{
    double sum = 0.0;
    for (int ic = 0; ic < unit_cell_.num_atom_symmetry_classes(); ic++)
    {
        sum += unit_cell_.atom_symmetry_class(ic)->core_eval_sum() * 
               unit_cell_.atom_symmetry_class(ic)->num_atoms();
    }
    return sum;
}

void DFT_ground_state::move_atoms(int istep)
{
    mdarray<double, 2> atom_force(3, unit_cell_.num_atoms());
    forces(atom_force);
    if (verbosity_level >= 6 && ctx_.comm().rank() == 0)
    {
        printf("\n");
        printf("Atomic forces\n");
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
        {
            printf("ia : %i, force : %12.6f %12.6f %12.6f\n", ia, atom_force(0, ia), atom_force(1, ia), atom_force(2, ia));
        }
    }

    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
    {
        vector3d<double> pos = unit_cell_.atom(ia)->position();
        
        vector3d<double> f(atom_force(0, ia), atom_force(1, ia), atom_force(2, ia));
        vector3d<double> forcef = unit_cell_.get_fractional_coordinates(f);

        for (int x = 0; x < 3; x++) pos[x] += forcef[x];
        
        unit_cell_.atom(ia)->set_position(pos);
    }
}

//void DFT_ground_state::update()
//{
//    parameters_.update();
//    potential_->update();
//    kset_->update();
//}

void DFT_ground_state::forces(mdarray<double, 2>& forces__)
{
    Force::total_force(ctx_, potential_, density_, kset_, forces__);
}

void DFT_ground_state::scf_loop(double potential_tol, double energy_tol, int num_dft_iter)
{
    Timer t("sirius::DFT_ground_state::scf_loop");
    
    if (!parameters_.full_potential()) density_->mixer_init();

    double eold = 0.0;
    double rms;

    generate_effective_potential();
    if (parameters_.full_potential()) potential_->mixer_init();

    for (int iter = 0; iter < num_dft_iter; iter++)
    {
        Timer t1("sirius::DFT_ground_state::scf_loop|iteration");

        /* find new wave-functions */
        kset_->find_eigen_states(potential_, true);
        kset_->find_band_occupancies();

        /* generate new density from the occupied wave-functions */
        density_->generate(*kset_);

        if (use_symmetry_) symmetrize_density();

        if (!parameters_.full_potential()) rms = density_->mix();

        /* compute new potential */
        generate_effective_potential();

        if (parameters_.full_potential()) rms = potential_->mix();

        ctx_.comm().bcast(&rms, 1, 0);

        /* compute new total energy for a new density */
        double etot = total_energy();
        
        /* write some information */
        print_info();

        if (ctx_.comm().rank() == 0)
        {
            printf("iteration : %3i, potential RMS %12.6f, energy difference : %12.6f\n", iter, rms, etot - eold);
        }
        
        if (std::abs(eold - etot) < energy_tol && rms < potential_tol) break;

        eold = etot;
    }
    
    ctx_.create_storage_file();
    potential_->save();
    density_->save();
}

void DFT_ground_state::relax_atom_positions()
{
    STOP();

    //for (int i = 0; i < 5; i++)
    //{
    //    scf_loop(1e-4, 1e-4, 100);
    //    move_atoms(i);
    //    update();
    //    //parameters_.print_info();
    //}
}

void DFT_ground_state::print_info()
{
    double evalsum1 = kset_->valence_eval_sum();
    double evalsum2 = core_eval_sum();
    double ekin = energy_kin();
    double evxc = energy_vxc();
    double eexc = energy_exc();
    double ebxc = energy_bxc();
    double evha = energy_vha();
    double etot = total_energy();
    double gap = kset_->band_gap() * ha2ev;
    double ef = kset_->energy_fermi();
    double core_leak = density_->core_leakage();
    double enuc = energy_enuc();

    std::vector<double> mt_charge;
    double it_charge;
    double total_charge = density_->rho()->integrate(mt_charge, it_charge); 
    
    double total_mag[3];
    std::vector<double> mt_mag[3];
    double it_mag[3];
    for (int j = 0; j < parameters_.num_mag_dims(); j++) 
        total_mag[j] = density_->magnetization(j)->integrate(mt_mag[j], it_mag[j]);
    
    if (ctx_.comm().rank() == 0)
    {
        if (parameters_.full_potential())
        {
            double total_core_leakage = 0.0;
            printf("\n");
            printf("Charges and magnetic moments\n");
            for (int i = 0; i < 80; i++) printf("-");
            printf("\n"); 
            printf("atom      charge    core leakage");
            if (parameters_.num_mag_dims()) printf("              moment                |moment|");
            printf("\n");
            for (int i = 0; i < 80; i++) printf("-");
            printf("\n"); 

            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
            {
                double core_leakage = unit_cell_.atom(ia)->symmetry_class()->core_leakage();
                total_core_leakage += core_leakage;
                printf("%4i  %10.6f  %10.8e", ia, mt_charge[ia], core_leakage);
                if (parameters_.num_mag_dims())
                {
                    vector3d<double> v;
                    v[2] = mt_mag[0][ia];
                    if (parameters_.num_mag_dims() == 3)
                    {
                        v[0] = mt_mag[1][ia];
                        v[1] = mt_mag[2][ia];
                    }
                    printf("  [%8.4f, %8.4f, %8.4f]  %10.6f", v[0], v[1], v[2], v.length());
                }
                printf("\n");
            }
            
            printf("\n");
            printf("interstitial charge   : %10.6f\n", it_charge);
            if (parameters_.num_mag_dims())
            {
                vector3d<double> v;
                v[2] = it_mag[0];
                if (parameters_.num_mag_dims() == 3)
                {
                    v[0] = it_mag[1];
                    v[1] = it_mag[2];
                }
                printf("interstitial moment   : [%8.4f, %8.4f, %8.4f], magnitude : %10.6f\n", 
                       v[0], v[1], v[2], v.length());
            }
            
            printf("\n");
            printf("total charge          : %10.6f\n", total_charge);
            printf("total core leakage    : %10.8e\n", total_core_leakage);
            if (parameters_.num_mag_dims())
            {
                vector3d<double> v;
                v[2] = total_mag[0];
                if (parameters_.num_mag_dims() == 3)
                {
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
        if (parameters_.full_potential())
        {
            printf("core_eval_sum             : %18.8f\n", evalsum2);
            printf("kinetic energy            : %18.8f\n", ekin);
            printf("enuc                      : %18.8f\n", enuc);
        }
        printf("<rho|V^{XC}>              : %18.8f\n", evxc);
        printf("<rho|E^{XC}>              : %18.8f\n", eexc);
        printf("<mag|B^{XC}>              : %18.8f\n", ebxc);
        printf("<rho|V^{H}>               : %18.8f\n", evha);
        if (parameters_.esm_type() == ultrasoft_pseudopotential)
        {
            printf("one-electron contribution : %18.8f\n", evalsum1 - (evxc + evha)); // eband + deband in QE
            printf("hartree contribution      : %18.8f\n", 0.5 * evha);
            printf("xc contribution           : %18.8f\n", eexc);
            printf("ewald contribution        : %18.8f\n", ewald_energy_);
        }
        printf("Total energy              : %18.8f\n", etot);

        printf("\n");
        printf("band gap (eV) : %18.8f\n", gap);
        printf("Efermi        : %18.8f\n", ef);
        printf("\n");
        if (parameters_.full_potential()) printf("core leakage : %18.8f\n", core_leak);
    }
}

}
