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

/** \file k_set.cpp
 *
 *  \brief Contains remaining implementation of sirius::K_set class.
 */

#include "k_set.h"

namespace sirius {

void K_set::initialize()
{
    /* distribute k-points along the 1-st dimension of the MPI grid */
    spl_num_kpoints_ = splindex<block>(num_kpoints(), comm_k_.size(), comm_k_.rank());

    for (int ikloc = 0; ikloc < (int)spl_num_kpoints_.local_size(); ikloc++)
        kpoints_[spl_num_kpoints_[ikloc]]->initialize();

    if (verbosity_level >= 2) print_info();
}

void K_set::update()
{
    for (int ikloc = 0; ikloc < (int)spl_num_kpoints_.local_size(); ikloc++)
        kpoints_[spl_num_kpoints_[ikloc]]->update();
}

void K_set::sync_band_energies()
{
    mdarray<double, 2> band_energies(parameters_.num_bands(), num_kpoints());

    for (int ikloc = 0; ikloc < (int)spl_num_kpoints_.local_size(); ikloc++)
    {
        int ik = (int)spl_num_kpoints_[ikloc];
        kpoints_[ik]->get_band_energies(&band_energies(0, ik));
    }
    comm_k_.allgather(band_energies.at<CPU>(), 
                      static_cast<int>(parameters_.num_bands() * spl_num_kpoints_.global_offset()),
                      static_cast<int>(parameters_.num_bands() * spl_num_kpoints_.local_size()));

    for (int ik = 0; ik < num_kpoints(); ik++) kpoints_[ik]->set_band_energies(&band_energies(0, ik));
}

void K_set::find_eigen_states(Potential* potential, bool precompute)
{
    Timer t("sirius::K_set::find_eigen_states", comm_k_);
    
    if (precompute && unit_cell_.full_potential())
    {
        potential->generate_pw_coefs();
        potential->update_atomic_potential();
        unit_cell_.generate_radial_functions();
        unit_cell_.generate_radial_integrals();
    }

    /* solve secular equation and generate wave functions */
    for (int ikloc = 0; ikloc < (int)spl_num_kpoints().local_size(); ikloc++)
    {
        int ik = spl_num_kpoints(ikloc);
        if (use_second_variation)
        {
            band_->solve_fv(kpoints_[ik], potential->effective_potential());
            kpoints_[ik]->generate_fv_states();
            band_->solve_sv(kpoints_[ik], potential->effective_magnetic_field());
        }
        else
        {
            band_->solve_fd(kpoints_[ik], potential->effective_potential(), potential->effective_magnetic_field());
        }
        kpoints_[ik]->generate_spinor_wave_functions();
    }

    /* synchronize eigen-values */
    sync_band_energies();

    if (comm_k_.rank() == 0 && blacs_grid_.comm().rank() == 0 && verbosity_level >= 5)
    {
        printf("Lowest band energies\n");
        for (int ik = 0; ik < num_kpoints(); ik++)
        {
            printf("ik : %2i, ", ik); 
            if (parameters_.num_mag_dims() != 1)
            {
                for (int j = 0; j < std::min(10, parameters_.num_bands()); j++) 
                    printf("%12.6f", kpoints_[ik]->band_energy(j));
            }
            else
            {
                for (int j = 0; j < std::min(10, parameters_.num_fv_states()); j++) 
                    printf("%12.6f", kpoints_[ik]->band_energy(j));
                printf("\n         ");
                for (int j = 0; j < std::min(10, parameters_.num_fv_states()); j++) 
                    printf("%12.6f", kpoints_[ik]->band_energy(parameters_.num_fv_states() + j));
            }
            printf("\n");
        }

        //== FILE* fout = fopen("eval.txt", "w");
        //== for (int ik = 0; ik < num_kpoints(); ik++)
        //== {
        //==     fprintf(fout, "ik : %2i\n", ik); 
        //==     for (int j = 0; j < parameters_.num_bands(); j++) 
        //==         fprintf(fout, "%4i: %18.10f\n", j, kpoints_[ik]->band_energy(j));
        //== }
        //== fclose(fout);
    }
}

double K_set::valence_eval_sum()
{
    double eval_sum = 0.0;

    for (int ik = 0; ik < num_kpoints(); ik++)
    {
        double wk = kpoints_[ik]->weight();
        for (int j = 0; j < parameters_.num_bands(); j++)
            eval_sum += wk * kpoints_[ik]->band_energy(j) * kpoints_[ik]->band_occupancy(j);
    }

    return eval_sum;
}

void K_set::find_band_occupancies()
{
    Timer t("sirius::K_set::find_band_occupancies");

    double ef = 0.15;

    double de = 0.1;

    int s = 1;
    int sp;

    double ne = 0.0;

    mdarray<double, 2> bnd_occ(parameters_.num_bands(), num_kpoints());
    
    // TODO: safe way not to get stuck here
    while (true)
    {
        ne = 0.0;
        for (int ik = 0; ik < num_kpoints(); ik++)
        {
            for (int j = 0; j < parameters_.num_bands(); j++)
            {
                bnd_occ(j, ik) = Utils::gaussian_smearing(kpoints_[ik]->band_energy(j) - ef, parameters_.smearing_width()) * 
                                 parameters_.max_occupancy();
                ne += bnd_occ(j, ik) * kpoints_[ik]->weight();
            }
        }

        if (fabs(ne - unit_cell_.num_valence_electrons()) < 1e-11) break;

        sp = s;
        s = (ne > unit_cell_.num_valence_electrons()) ? -1 : 1;

        de = s * fabs(de);

        (s != sp) ? de *= 0.5 : de *= 1.25; 
        
        ef += de;
    } 
    energy_fermi_ = ef;

    for (int ik = 0; ik < num_kpoints(); ik++) kpoints_[ik]->set_band_occupancies(&bnd_occ(0, ik));

    band_gap_ = 0.0;
    
    int nve = static_cast<int>(unit_cell_.num_valence_electrons() + 1e-12);
    if (parameters_.num_spins() == 2 || 
        (fabs(nve - unit_cell_.num_valence_electrons()) < 1e-12 && nve % 2 == 0))
    {
        /* find band gap */
        std::vector< std::pair<double, double> > eband;
        std::pair<double, double> eminmax;

        for (int j = 0; j < parameters_.num_bands(); j++)
        {
            eminmax.first = 1e10;
            eminmax.second = -1e10;

            for (int ik = 0; ik < num_kpoints(); ik++)
            {
                eminmax.first = std::min(eminmax.first, kpoints_[ik]->band_energy(j));
                eminmax.second = std::max(eminmax.second, kpoints_[ik]->band_energy(j));
            }

            eband.push_back(eminmax);
        }
        
        std::sort(eband.begin(), eband.end());

        int ist = nve;
        if (parameters_.num_spins() == 1) ist /= 2; 

        if (eband[ist].first > eband[ist - 1].second) band_gap_ = eband[ist].first - eband[ist - 1].second;
    }
}

void K_set::print_info()
{
    if (comm_k_.rank() == 0 && blacs_grid_.comm().rank() == 0)
    {
        printf("\n");
        printf("total number of k-points : %i\n", num_kpoints());
        for (int i = 0; i < 80; i++) printf("-");
        printf("\n");
        printf("  ik                vk                    weight  num_gkvec");
        if (unit_cell_.full_potential()) printf("   gklo_basis_size");
        printf("\n");
        for (int i = 0; i < 80; i++) printf("-");
        printf("\n");
    }

    if (blacs_grid_.comm().rank() == 0)
    {
        pstdout pout(comm_k_);
        for (int ikloc = 0; ikloc < (int)spl_num_kpoints().local_size(); ikloc++)
        {
            int ik = spl_num_kpoints(ikloc);
            pout.printf("%4i   %8.4f %8.4f %8.4f   %12.6f     %6i", 
                        ik, kpoints_[ik]->vk()[0], kpoints_[ik]->vk()[1], kpoints_[ik]->vk()[2], 
                        kpoints_[ik]->weight(), kpoints_[ik]->num_gkvec());

            if (unit_cell_.full_potential()) pout.printf("            %6i", kpoints_[ik]->gklo_basis_size());
            
            pout.printf("\n");
        }
    }
}

void K_set::save()
{
    warning_local(__FILE__, __LINE__, "fix me");
    STOP();

    //==if (comm_.rank() == 0)
    //=={
    //==    HDF5_tree fout(storage_file_name, false);
    //==    fout.create_node("K_set");
    //==    fout["K_set"].write("num_kpoints", num_kpoints());
    //==}
    //==comm_.barrier();
    //==
    //==if (parameters_.mpi_grid().side(1 << _dim_k_ | 1 << _dim_col_))
    //=={
    //==    for (int ik = 0; ik < num_kpoints(); ik++)
    //==    {
    //==        int rank = spl_num_kpoints_.local_rank(ik);
    //==        
    //==        if (parameters_.mpi_grid().coordinate(_dim_k_) == rank) kpoints_[ik]->save(ik);
    //==        
    //==        parameters_.mpi_grid().barrier(1 << _dim_k_ | 1 << _dim_col_);
    //==    }
    //==}
}

/// \todo check parameters of saved data in a separate function
void K_set::load()
{
    STOP();

    //== HDF5_tree fin(storage_file_name, false);

    //== int num_kpoints_in;
    //== fin["K_set"].read("num_kpoints", &num_kpoints_in);

    //== std::vector<int> ikidx(num_kpoints(), -1); 
    //== // read available k-points
    //== double vk_in[3];
    //== for (int jk = 0; jk < num_kpoints_in; jk++)
    //== {
    //==     fin["K_set"][jk].read("coordinates", vk_in, 3);
    //==     for (int ik = 0; ik < num_kpoints(); ik++)
    //==     {
    //==         vector3d<double> dvk; 
    //==         for (int x = 0; x < 3; x++) dvk[x] = vk_in[x] - kpoints_[ik]->vk(x);
    //==         if (dvk.length() < 1e-12)
    //==         {
    //==             ikidx[ik] = jk;
    //==             break;
    //==         }
    //==     }
    //== }

    //== for (int ik = 0; ik < num_kpoints(); ik++)
    //== {
    //==     int rank = spl_num_kpoints_.local_rank(ik);
    //==     
    //==     if (comm_.rank() == rank) kpoints_[ik]->load(fin["K_set"], ikidx[ik]);
    //== }
}

//== void K_set::save_wave_functions()
//== {
//==     if (Platform::mpi_rank() == 0)
//==     {
//==         HDF5_tree fout(storage_file_name, false);
//==         fout["parameters"].write("num_kpoints", num_kpoints());
//==         fout["parameters"].write("num_bands", parameters_.num_bands());
//==         fout["parameters"].write("num_spins", parameters_.num_spins());
//==     }
//== 
//==     if (parameters_.mpi_grid().side(1 << _dim_k_ | 1 << _dim_col_))
//==     {
//==         for (int ik = 0; ik < num_kpoints(); ik++)
//==         {
//==             int rank = spl_num_kpoints_.location(_splindex_rank_, ik);
//==             
//==             if (parameters_.mpi_grid().coordinate(_dim_k_) == rank) kpoints_[ik]->save_wave_functions(ik);
//==             
//==             parameters_.mpi_grid().barrier(1 << _dim_k_ | 1 << _dim_col_);
//==         }
//==     }
//== }
//== 
//== void K_set::load_wave_functions()
//== {
//==     HDF5_tree fin(storage_file_name, false);
//==     int num_spins;
//==     fin["parameters"].read("num_spins", &num_spins);
//==     if (num_spins != parameters_.num_spins()) error_local(__FILE__, __LINE__, "wrong number of spins");
//== 
//==     int num_bands;
//==     fin["parameters"].read("num_bands", &num_bands);
//==     if (num_bands != parameters_.num_bands()) error_local(__FILE__, __LINE__, "wrong number of bands");
//==     
//==     int num_kpoints_in;
//==     fin["parameters"].read("num_kpoints", &num_kpoints_in);
//== 
//==     // ==================================================================
//==     // index of current k-points in the hdf5 file, which (in general) may 
//==     // contain a different set of k-points 
//==     // ==================================================================
//==     std::vector<int> ikidx(num_kpoints(), -1); 
//==     // read available k-points
//==     double vk_in[3];
//==     for (int jk = 0; jk < num_kpoints_in; jk++)
//==     {
//==         fin["kpoints"][jk].read("coordinates", vk_in, 3);
//==         for (int ik = 0; ik < num_kpoints(); ik++)
//==         {
//==             vector3d<double> dvk; 
//==             for (int x = 0; x < 3; x++) dvk[x] = vk_in[x] - kpoints_[ik]->vk(x);
//==             if (dvk.length() < 1e-12)
//==             {
//==                 ikidx[ik] = jk;
//==                 break;
//==             }
//==         }
//==     }
//== 
//==     for (int ik = 0; ik < num_kpoints(); ik++)
//==     {
//==         int rank = spl_num_kpoints_.location(_splindex_rank_, ik);
//==         
//==         if (parameters_.mpi_grid().coordinate(0) == rank) kpoints_[ik]->load_wave_functions(ikidx[ik]);
//==     }
//== }

int K_set::max_num_gkvec()
{
    int max_num_gkvec_ = 0;
    for (size_t ikloc = 0; ikloc < spl_num_kpoints_.local_size(); ikloc++)
    {
        auto ik = spl_num_kpoints_[ikloc];
        max_num_gkvec_ = std::max(max_num_gkvec_, kpoints_[ik]->num_gkvec());
    }
    comm_k_.allreduce<int, op_max>(&max_num_gkvec_, 1);
    return max_num_gkvec_;
}

//== void K_set::fixed_band_occupancies()
//== {
//==     Timer t("sirius::K_set::fixed_band_occupancies");
//== 
//==     if (parameters_.num_mag_dims() != 1) error_local(__FILE__, __LINE__, "works only for collinear magnetism");
//== 
//==     double n_up = (parameters_.num_valence_electrons() + parameters_.fixed_moment()) / 2.0;
//==     double n_dn = (parameters_.num_valence_electrons() - parameters_.fixed_moment()) / 2.0;
//==     
//==     mdarray<double, 2> bnd_occ(parameters_.num_bands(), num_kpoints());
//==     bnd_occ.zero();
//== 
//==     int j = 0;
//==     while (n_up > 0)
//==     {
//==         for (int ik = 0; ik < num_kpoints(); ik++) bnd_occ(j, ik) = std::min(double(parameters_.max_occupancy()), n_up);
//==         j++;
//==         n_up -= parameters_.max_occupancy();
//==     }
//==             
//==     j = parameters_.num_fv_states();
//==     while (n_dn > 0)
//==     {
//==         for (int ik = 0; ik < num_kpoints(); ik++) bnd_occ(j, ik) = std::min(double(parameters_.max_occupancy()), n_dn);
//==         j++;
//==         n_dn -= parameters_.max_occupancy();
//==     }
//==             
//==     for (int ik = 0; ik < num_kpoints(); ik++) kpoints_[ik]->set_band_occupancies(&bnd_occ(0, ik));
//== 
//==     double gap = 0.0;
//==     
//==     int nve = int(parameters_.num_valence_electrons() + 1e-12);
//==     if ((parameters_.num_spins() == 2) || 
//==         ((fabs(nve - parameters_.num_valence_electrons()) < 1e-12) && nve % 2 == 0))
//==     {
//==         // find band gap
//==         std::vector< std::pair<double, double> > eband;
//==         std::pair<double, double> eminmax;
//== 
//==         for (int j = 0; j < parameters_.num_bands(); j++)
//==         {
//==             eminmax.first = 1e10;
//==             eminmax.second = -1e10;
//== 
//==             for (int ik = 0; ik < num_kpoints(); ik++)
//==             {
//==                 eminmax.first = std::min(eminmax.first, kpoints_[ik]->band_energy(j));
//==                 eminmax.second = std::max(eminmax.second, kpoints_[ik]->band_energy(j));
//==             }
//== 
//==             eband.push_back(eminmax);
//==         }
//==         
//==         std::sort(eband.begin(), eband.end());
//== 
//==         int ist = nve;
//==         if (parameters_.num_spins() == 1) ist /= 2; 
//== 
//==         if (eband[ist].first > eband[ist - 1].second) gap = eband[ist].first - eband[ist - 1].second;
//== 
//==         band_gap_ = gap;
//==     }
//==     
//==     if (Platform::mpi_rank() == 0 && verbosity_level >= 5)
//==     {
//==         printf("Lowest band occupancies\n");
//==         for (int ik = 0; ik < num_kpoints(); ik++)
//==         {
//==             printf("ik : %2i, ", ik); 
//==             if (parameters_.num_mag_dims() != 1)
//==             {
//==                 for (int j = 0; j < std::min(10, parameters_.num_bands()); j++) 
//==                     printf("%12.6f", kpoints_[ik]->band_occupancy(j));
//==             }
//==             else
//==             {
//==                 for (int j = 0; j < std::min(10, parameters_.num_fv_states()); j++) 
//==                     printf("%12.6f", kpoints_[ik]->band_occupancy(j));
//==                 printf("\n         ");
//==                 for (int j = 0; j < std::min(10, parameters_.num_fv_states()); j++) 
//==                     printf("%12.6f", kpoints_[ik]->band_occupancy(parameters_.num_fv_states() + j));
//==             }
//==             printf("\n");
//==         }
//==     }
//== }

}
