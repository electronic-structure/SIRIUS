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

/** \file k_point_set.h
 *
 *  \brief Contains declaration and partial implementation of sirius::K_point_set class.
 */

#ifndef __K_POINT_SET_H__
#define __K_POINT_SET_H__

#include "k_point.h"
#include "geometry3d.hpp"

namespace sirius {

struct kq
{
    // index of reduced k+q vector
    int jk;

    // vector which reduces k+q to first BZ
    vector3d<int> K;
};

/// Set of k-points.
class K_point_set
{
    private:

        Simulation_context& ctx_;

        std::vector<std::unique_ptr<K_point>> kpoints_;

        splindex<chunk> spl_num_kpoints_;

        double energy_fermi_{0};

        double band_gap_{0};

        Unit_cell& unit_cell_;

        Communicator const& comm_k_;

        K_point_set(K_point_set& src) = delete;

    public:

        K_point_set(Simulation_context& ctx__)
                  : ctx_(ctx__)
                  , unit_cell_(ctx__.unit_cell())
                  , comm_k_(ctx__.comm_k())
        {
            PROFILE("sirius::K_point_set::K_point_set");
        }

        K_point_set(Simulation_context& ctx__,
                    vector3d<int> k_grid__,
                    vector3d<int> k_shift__,
                    int use_symmetry__)
                  : ctx_(ctx__)
                  , unit_cell_(ctx__.unit_cell())
                  , comm_k_(ctx_.comm_k())
        {
            PROFILE("sirius::K_point_set::K_point_set");

            int nk;
            mdarray<double, 2> kp;
            std::vector<double> wk;
            if (use_symmetry__) {
                nk = unit_cell_.symmetry().get_irreducible_reciprocal_mesh(k_grid__, k_shift__, kp, wk);
            } else {
                nk = k_grid__[0] * k_grid__[1] * k_grid__[2];
                wk = std::vector<double>(nk, 1.0 / nk);
                kp = mdarray<double, 2>(3, nk);

                int ik = 0;
                for (int i0 = 0; i0 < k_grid__[0]; i0++) {
                    for (int i1 = 0; i1 < k_grid__[1]; i1++) {
                        for (int i2 = 0; i2 < k_grid__[2]; i2++) {
                            kp(0, ik) = double(i0 + k_shift__[0] / 2.0) / k_grid__[0];
                            kp(1, ik) = double(i1 + k_shift__[1] / 2.0) / k_grid__[1];
                            kp(2, ik) = double(i2 + k_shift__[2] / 2.0) / k_grid__[2];
                            ik++;
                        }
                    }
                }
            }

            //if (use_symmetry__)
            //{
            //    mdarray<int, 2> kmap(parameters_.unit_cell()->symmetry()->num_sym_op(), nk);
            //    for (int ik = 0; ik < nk; ik++)
            //    {
            //        for (int isym = 0; isym < parameters_.unit_cell()->symmetry()->num_sym_op(); isym++)
            //        {
            //            auto vk_rot = matrix3d<double>(transpose(parameters_.unit_cell()->symmetry()->rot_mtrx(isym))) *
            //                          vector3d<double>(vk(0, ik), vk(1, ik), vk(2, ik));
            //            for (int x = 0; x < 3; x++)
            //            {
            //                if (vk_rot[x] < 0) vk_rot[x] += 1;
            //                if (vk_rot[x] < 0 || vk_rot[x] >= 1) TERMINATE("wrong rotated k-point");
            //            }

            //            for (int jk = 0; jk < nk; jk++)
            //            {
            //                if (std::abs(vk_rot[0] - vk(0, jk)) < 1e-10 &&
            //                    std::abs(vk_rot[1] - vk(1, jk)) < 1e-10 &&
            //                    std::abs(vk_rot[2] - vk(2, jk)) < 1e-10)
            //                {
            //                    kmap(isym, ik) = jk;
            //                }
            //            }
            //        }
            //    }

            //    //== std::cout << "sym.table" << std::endl;
            //    //== for (int isym = 0; isym < parameters_.unit_cell()->symmetry().num_sym_op(); isym++)
            //    //== {
            //    //==     printf("sym: %2i, ", isym);
            //    //==     for (int ik = 0; ik < nk; ik++) printf(" %2i", kmap(isym, ik));
            //    //==     printf("\n");
            //    //== }

            //    std::vector<int> flag(nk, 1);
            //    for (int ik = 0; ik < nk; ik++)
            //    {
            //        if (flag[ik])
            //        {
            //            int ndeg = 0;
            //            for (int isym = 0; isym < parameters_.unit_cell()->symmetry()->num_sym_op(); isym++)
            //            {
            //                if (flag[kmap(isym, ik)])
            //                {
            //                    flag[kmap(isym, ik)] = 0;
            //                    ndeg++;
            //                }
            //            }
            //            add_kpoint(&vk(0, ik), double(ndeg) / nk);
            //        }
            //    }
            //}
            //else
            //{
            //    for (int ik = 0; ik < nk; ik++) add_kpoint(&vk(0, ik), wk[ik]);
            //}

            for (int ik = 0; ik < nk; ik++) {
                add_kpoint(&kp(0, ik), wk[ik]);
            }
            initialize();
        }

        K_point_set(Simulation_context& ctx__, std::vector<vector3d<double>> vec__)
                  : ctx_(ctx__)
                  , unit_cell_(ctx__.unit_cell())
                  , comm_k_(ctx__.comm_k())
        {
            PROFILE("sirius::K_point_set::K_point_set");
            for (auto& v: vec__) {
                add_kpoint(&v[0], 1.0);
            }
            initialize();
        }

        /// Initialize the k-point set
        void initialize(std::vector<int> counts)
        {
            /* distribute k-points along the 1-st dimension of the MPI grid */
            if (counts.empty()) {
                splindex<block> spl_tmp(num_kpoints(), comm_k_.size(), comm_k_.rank());
                spl_num_kpoints_ = splindex<chunk>(num_kpoints(), comm_k_.size(), comm_k_.rank(), spl_tmp.counts());
            } else {
                spl_num_kpoints_ = splindex<chunk>(num_kpoints(), comm_k_.size(), comm_k_.rank(), counts);
            }

            for (int ikloc = 0; ikloc < spl_num_kpoints_.local_size(); ikloc++) {
                kpoints_[spl_num_kpoints_[ikloc]]->initialize();
            }

            if (ctx_.control().verbosity_ > 0) {
                print_info();
            }

            if (ctx_.comm().rank() == 0 && ctx_.control().print_memory_usage_) {
                MEMORY_USAGE_INFO();
            }
        }

        void initialize()
        {
            initialize(std::vector<int>());
        }

        /// Get a list of band energies for a given k-point index.
        std::vector<double> get_band_energies(int ik__, int ispn__)
        {
            std::vector<double> bnd_e(ctx_.num_bands());
            for (int j = 0; j < ctx_.num_bands(); j++) {
                bnd_e[j] = (*this)[ik__]->band_energy(j, ispn__);
            }
            return std::move(bnd_e);
        }

        /// Find Fermi energy and band occupation numbers
        void find_band_occupancies();

        /// Return sum of valence eigen-values
        double valence_eval_sum()
        {
            double eval_sum{0};

            for (int ik = 0; ik < num_kpoints(); ik++) {
                double wk = kpoints_[ik]->weight();
                for (int j = 0; j < ctx_.num_bands(); j++) {
                    for (int ispn = 0; ispn < ctx_.num_spin_dims(); ispn++) {
                        eval_sum += wk * kpoints_[ik]->band_energy(j, ispn) * kpoints_[ik]->band_occupancy(j, ispn);
                    }
                }
            }

            return eval_sum;
        }

        void print_info();

        void sync_band_energies();

        void save();

        void load();

        int max_num_gkvec()
        {
            int max_num_gkvec{0};
            for (int ikloc = 0; ikloc < spl_num_kpoints_.local_size(); ikloc++) {
                auto ik = spl_num_kpoints_[ikloc];
                max_num_gkvec = std::max(max_num_gkvec, kpoints_[ik]->num_gkvec());
            }
            comm_k_.allreduce<int, mpi_op_t::max>(&max_num_gkvec, 1);
            return max_num_gkvec;
        }

        void add_kpoint(double* vk__, double weight__)
        {
            PROFILE("sirius::K_point_set::add_kpoint");
            kpoints_.push_back(std::unique_ptr<K_point>(new K_point(ctx_, vk__, weight__)));
        }

        void add_kpoints(mdarray<double, 2>& kpoints__, double* weights__)
        {
            PROFILE("sirius::K_point_set::add_kpoints");
            for (size_t ik = 0; ik < kpoints__.size(1); ik++) {
                add_kpoint(&kpoints__(0, ik), weights__[ik]);
            }
        }

        inline K_point* operator[](int i)
        {
            assert(i >= 0 && i < (int)kpoints_.size());

            return kpoints_[i].get();
        }

        inline int num_kpoints() const
        {
            return static_cast<int>(kpoints_.size());
        }

        inline splindex<chunk> const& spl_num_kpoints() const
        {
            return spl_num_kpoints_;
        }

        inline int spl_num_kpoints(int ikloc) const
        {
            return spl_num_kpoints_[ikloc];
        }

        inline double energy_fermi() const
        {
            return energy_fermi_;
        }

        inline double band_gap() const
        {
            return band_gap_;
        }

        /// Find index of k-point.
        inline int find_kpoint(vector3d<double> vk__)
        {
            for (int ik = 0; ik < num_kpoints(); ik++) {
                if ((kpoints_[ik]->vk() - vk__).length() < 1e-12) {
                    return ik;
                }
            }
            return -1;
        }

        //void generate_Gq_matrix_elements(vector3d<double> vq)
        //{
        //    std::vector<kq> kpq(num_kpoints());
        //    for (int ik = 0; ik < num_kpoints(); ik++)
        //    {
        //        // reduce k+q to first BZ: k+q=k"+K; k"=k+q-K
        //        std::pair< vector3d<double>, vector3d<int> > vkqr = reduce_coordinates(kpoints_[ik]->vk() + vq);
        //
        //        if ((kpq[ik].jk = find_kpoint(vkqr.first)) == -1)
        //            TERMINATE("index of reduced k+q point is not found");

        //        kpq[ik].K = vkqr.second;
        //    }
        //}

        inline Communicator const& comm() const
        {
            return comm_k_;
        }
};

inline void K_point_set::sync_band_energies()
{
    PROFILE("sirius::K_point_set::sync_band_energies");

    mdarray<double, 3> band_energies(ctx_.num_bands(), ctx_.num_spin_dims(), num_kpoints());

    for (int ikloc = 0; ikloc < spl_num_kpoints_.local_size(); ikloc++) {
        int ik = spl_num_kpoints_[ikloc];
        for (int ispn = 0; ispn < ctx_.num_spin_dims(); ispn++) {
            for (int j = 0; j < ctx_.num_bands(); j++) {
                band_energies(j, ispn, ik) = kpoints_[ik]->band_energy(j, ispn);
            }
        }
    }
    comm_k_.allgather(band_energies.at<CPU>(),
                      ctx_.num_bands() * ctx_.num_spin_dims() * spl_num_kpoints_.global_offset(),
                      ctx_.num_bands() * ctx_.num_spin_dims() * spl_num_kpoints_.local_size());

    for (int ik = 0; ik < num_kpoints(); ik++) {
        for (int ispn = 0; ispn < ctx_.num_spin_dims(); ispn++) {
            for (int j = 0; j < ctx_.num_bands(); j++) {
                kpoints_[ik]->band_energy(j, ispn) = band_energies(j, ispn, ik);
            }
        }
    }
}

inline void K_point_set::find_band_occupancies()
{
    PROFILE("sirius::K_point_set::find_band_occupancies");

    double ef{0};
    double de{0.1};

    int s{1};
    int sp;

    mdarray<double, 3> bnd_occ(ctx_.num_bands(), ctx_.num_spin_dims(), num_kpoints());

    double ne{0};

    int step{0};
    /* calculate occupations */
    while (std::abs(ne - unit_cell_.num_valence_electrons()) >= 1e-11) {
        /* update Efermi */
        ef += de;
        /* compute total number of electrons */
        ne = 0.0;
        for (int ik = 0; ik < num_kpoints(); ik++) {
            for (int ispn = 0; ispn < ctx_.num_spin_dims(); ispn++) {
                for (int j = 0; j < ctx_.num_bands(); j++) {
                    bnd_occ(j, ispn, ik) = Utils::gaussian_smearing(kpoints_[ik]->band_energy(j, ispn) - ef, ctx_.smearing_width()) *
                                           ctx_.max_occupancy();
                    ne += bnd_occ(j, ispn, ik) * kpoints_[ik]->weight();
                }
            }
        }

        sp = s;
        s = (ne > unit_cell_.num_valence_electrons()) ? -1 : 1;
        /* reduce de step if we change the direction, otherwise increase the step */
        de = (s != sp) ? (-de * 0.5) : (de * 1.25);

        if (step > 10000) {
            std::stringstream s;
            s << "search of band occupancies failed after 10000 steps";
            TERMINATE(s);

            //ef = 0;

            //for (int ik = 0; ik < num_kpoints(); ik++) {
            //    ne = unit_cell_.num_valence_electrons();
            //    for (int j = 0; j < ctx_.num_bands(); j++) {
            //        bnd_occ(j, ik) = std::min(ne, static_cast<double>(ctx_.max_occupancy()));
            //        ne = std::max(ne - ctx_.max_occupancy(), 0.0);
            //    }
            //}

            //break;
        }
        step++;
    }

    energy_fermi_ = ef;

    for (int ik = 0; ik < num_kpoints(); ik++) {
        for (int ispn = 0; ispn < ctx_.num_spin_dims(); ispn++) {
            for (int j = 0; j < ctx_.num_bands(); j++) {
                kpoints_[ik]->band_occupancy(j, ispn) = bnd_occ(j, ispn, ik);
            }
        }
    }

    band_gap_ = 0.0;

    int nve = static_cast<int>(unit_cell_.num_valence_electrons() + 1e-12);
    if (ctx_.num_spins() == 2 ||
        (std::abs(nve - unit_cell_.num_valence_electrons()) < 1e-12 && nve % 2 == 0)) {
        /* find band gap */
        std::vector<std::pair<double, double>> eband;
        std::pair<double, double> eminmax;

        for (int ispn = 0; ispn < ctx_.num_spin_dims(); ispn++) {
            for (int j = 0; j < ctx_.num_bands(); j++) {
                eminmax.first = 1e10;
                eminmax.second = -1e10;

                for (int ik = 0; ik < num_kpoints(); ik++) {
                    eminmax.first = std::min(eminmax.first, kpoints_[ik]->band_energy(j, ispn));
                    eminmax.second = std::max(eminmax.second, kpoints_[ik]->band_energy(j, ispn));
                }

                eband.push_back(eminmax);
            }
        }

        std::sort(eband.begin(), eband.end());

        int ist = nve;
        if (ctx_.num_spins() == 1) {
            ist /= 2;
        }

        if (eband[ist].first > eband[ist - 1].second) {
            band_gap_ = eband[ist].first - eband[ist - 1].second;
        }
    }
}

inline void K_point_set::print_info()
{
    if (comm_k_.rank() == 0 && ctx_.comm_band().rank() == 0) {
        printf("\n");
        printf("total number of k-points : %i\n", num_kpoints());
        for (int i = 0; i < 80; i++) {
            printf("-");
        }
        printf("\n");
        printf("  ik                vk                    weight  num_gkvec");
        if (ctx_.full_potential()) {
            printf("   gklo_basis_size");
        }
        printf("\n");
        for (int i = 0; i < 80; i++) {
            printf("-");
        }
        printf("\n");
    }

    if (ctx_.comm_band().rank() == 0) {
        runtime::pstdout pout(comm_k_);
        for (int ikloc = 0; ikloc < spl_num_kpoints().local_size(); ikloc++) {
            int ik = spl_num_kpoints(ikloc);
            pout.printf("%4i   %8.4f %8.4f %8.4f   %12.6f     %6i",
                        ik, kpoints_[ik]->vk()[0], kpoints_[ik]->vk()[1], kpoints_[ik]->vk()[2],
                        kpoints_[ik]->weight(), kpoints_[ik]->num_gkvec());

            if (ctx_.full_potential()) {
                pout.printf("            %6i", kpoints_[ik]->gklo_basis_size());
            }

            pout.printf("\n");
        }
    }
}

inline void K_point_set::save()
{
    TERMINATE("fix me");
    STOP();

    //==if (comm_.rank() == 0)
    //=={
    //==    HDF5_tree fout(storage_file_name, false);
    //==    fout.create_node("K_point_set");
    //==    fout["K_point_set"].write("num_kpoints", num_kpoints());
    //==}
    //==comm_.barrier();
    //==
    //==if (ctx_.mpi_grid().side(1 << _dim_k_ | 1 << _dim_col_))
    //=={
    //==    for (int ik = 0; ik < num_kpoints(); ik++)
    //==    {
    //==        int rank = spl_num_kpoints_.local_rank(ik);
    //==
    //==        if (ctx_.mpi_grid().coordinate(_dim_k_) == rank) kpoints_[ik]->save(ik);
    //==
    //==        ctx_.mpi_grid().barrier(1 << _dim_k_ | 1 << _dim_col_);
    //==    }
    //==}
}

/// \todo check parameters of saved data in a separate function
inline void K_point_set::load()
{
    STOP();

    //== HDF5_tree fin(storage_file_name, false);

    //== int num_kpoints_in;
    //== fin["K_point_set"].read("num_kpoints", &num_kpoints_in);

    //== std::vector<int> ikidx(num_kpoints(), -1);
    //== // read available k-points
    //== double vk_in[3];
    //== for (int jk = 0; jk < num_kpoints_in; jk++)
    //== {
    //==     fin["K_point_set"][jk].read("coordinates", vk_in, 3);
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
    //==     if (comm_.rank() == rank) kpoints_[ik]->load(fin["K_point_set"], ikidx[ik]);
    //== }
}

//== void K_point_set::save_wave_functions()
//== {
//==     if (Platform::mpi_rank() == 0)
//==     {
//==         HDF5_tree fout(storage_file_name, false);
//==         fout["parameters"].write("num_kpoints", num_kpoints());
//==         fout["parameters"].write("num_bands", ctx_.num_bands());
//==         fout["parameters"].write("num_spins", ctx_.num_spins());
//==     }
//==
//==     if (ctx_.mpi_grid().side(1 << _dim_k_ | 1 << _dim_col_))
//==     {
//==         for (int ik = 0; ik < num_kpoints(); ik++)
//==         {
//==             int rank = spl_num_kpoints_.location(_splindex_rank_, ik);
//==
//==             if (ctx_.mpi_grid().coordinate(_dim_k_) == rank) kpoints_[ik]->save_wave_functions(ik);
//==
//==             ctx_.mpi_grid().barrier(1 << _dim_k_ | 1 << _dim_col_);
//==         }
//==     }
//== }
//==
//== void K_point_set::load_wave_functions()
//== {
//==     HDF5_tree fin(storage_file_name, false);
//==     int num_spins;
//==     fin["parameters"].read("num_spins", &num_spins);
//==     if (num_spins != ctx_.num_spins()) error_local(__FILE__, __LINE__, "wrong number of spins");
//==
//==     int num_bands;
//==     fin["parameters"].read("num_bands", &num_bands);
//==     if (num_bands != ctx_.num_bands()) error_local(__FILE__, __LINE__, "wrong number of bands");
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
//==         if (ctx_.mpi_grid().coordinate(0) == rank) kpoints_[ik]->load_wave_functions(ikidx[ik]);
//==     }
//== }

//== void K_point_set::fixed_band_occupancies()
//== {
//==     Timer t("sirius::K_point_set::fixed_band_occupancies");
//==
//==     if (ctx_.num_mag_dims() != 1) error_local(__FILE__, __LINE__, "works only for collinear magnetism");
//==
//==     double n_up = (ctx_.num_valence_electrons() + ctx_.fixed_moment()) / 2.0;
//==     double n_dn = (ctx_.num_valence_electrons() - ctx_.fixed_moment()) / 2.0;
//==
//==     mdarray<double, 2> bnd_occ(ctx_.num_bands(), num_kpoints());
//==     bnd_occ.zero();
//==
//==     int j = 0;
//==     while (n_up > 0)
//==     {
//==         for (int ik = 0; ik < num_kpoints(); ik++) bnd_occ(j, ik) = std::min(double(ctx_.max_occupancy()), n_up);
//==         j++;
//==         n_up -= ctx_.max_occupancy();
//==     }
//==
//==     j = ctx_.num_fv_states();
//==     while (n_dn > 0)
//==     {
//==         for (int ik = 0; ik < num_kpoints(); ik++) bnd_occ(j, ik) = std::min(double(ctx_.max_occupancy()), n_dn);
//==         j++;
//==         n_dn -= ctx_.max_occupancy();
//==     }
//==
//==     for (int ik = 0; ik < num_kpoints(); ik++) kpoints_[ik]->set_band_occupancies(&bnd_occ(0, ik));
//==
//==     double gap = 0.0;
//==
//==     int nve = int(ctx_.num_valence_electrons() + 1e-12);
//==     if ((ctx_.num_spins() == 2) ||
//==         ((fabs(nve - ctx_.num_valence_electrons()) < 1e-12) && nve % 2 == 0))
//==     {
//==         // find band gap
//==         std::vector< std::pair<double, double> > eband;
//==         std::pair<double, double> eminmax;
//==
//==         for (int j = 0; j < ctx_.num_bands(); j++)
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
//==         if (ctx_.num_spins() == 1) ist /= 2;
//==
//==         if (eband[ist].first > eband[ist - 1].second) gap = eband[ist].first - eband[ist - 1].second;
//==
//==         band_gap_ = gap;
//==     }
//== }
};

#endif // __K_POINT_SET_H__
