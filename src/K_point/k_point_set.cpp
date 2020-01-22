// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
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

#include <limits>
#include "K_point/k_point.hpp"
#include "K_point/k_point_set.hpp"

namespace sirius {

void K_point_set::sync_band_energies()
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
    comm().allgather(band_energies.at(memory_t::host),
                     ctx_.num_bands() * ctx_.num_spin_dims() * spl_num_kpoints_.global_offset(),
                     ctx_.num_bands() * ctx_.num_spin_dims() * spl_num_kpoints_.local_size());

    for (int ik = 0; ik < num_kpoints(); ik++) {
        for (int ispn = 0; ispn < ctx_.num_spin_dims(); ispn++) {
            for (int j = 0; j < ctx_.num_bands(); j++) {
                kpoints_[ik]->band_energy(j, ispn, band_energies(j, ispn, ik));
            }
        }
    }
}

void K_point_set::create_k_mesh(vector3d<int> k_grid__, vector3d<int> k_shift__, int use_symmetry__)
{
    PROFILE("sirius::K_point_set::create_k_mesh");

    int nk;
    mdarray<double, 2> kp;
    std::vector<double> wk;
    if (use_symmetry__) {
        auto result = get_irreducible_reciprocal_mesh(unit_cell_.symmetry(), k_grid__, k_shift__);
        nk          = std::get<0>(result);
        wk          = std::get<1>(result);
        auto tmp    = std::get<2>(result);
        kp          = mdarray<double, 2>(3, nk);
        for (int i = 0; i < nk; i++) {
            for (int x : {0, 1, 2}) {
                kp(x, i) = tmp[i][x];
            }
        }
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

    for (int ik = 0; ik < nk; ik++) {
        add_kpoint(&kp(0, ik), wk[ik]);
    }

    initialize();
}

void K_point_set::initialize(std::vector<int> const& counts)
{
    PROFILE("sirius::K_point_set::initialize");
    /* distribute k-points along the 1-st dimension of the MPI grid */
    if (counts.empty()) {
        splindex<splindex_t::block> spl_tmp(num_kpoints(), comm().size(), comm().rank());
        spl_num_kpoints_ = splindex<splindex_t::chunk>(num_kpoints(), comm().size(), comm().rank(), spl_tmp.counts());
    } else {
        spl_num_kpoints_ = splindex<splindex_t::chunk>(num_kpoints(), comm().size(), comm().rank(), counts);
    }

    for (int ikloc = 0; ikloc < spl_num_kpoints_.local_size(); ikloc++) {
        kpoints_[spl_num_kpoints_[ikloc]]->initialize();
    }

    if (ctx_.control().verbosity_ > 0) {
        print_info();
    }
    ctx_.print_memory_usage(__FILE__, __LINE__);
}

void K_point_set::sync_band_occupancies()
{
    int nranks = comm().size();
    int pid    = comm().rank();
    std::vector<int> nk_per_rank(nranks);
    for (int i = 0; i < nranks; ++i) {
        nk_per_rank[i] = spl_num_kpoints_.local_size(i);
    }
    int ns        = 1;
    int num_bands = ctx_.num_bands();
    if (ctx_.num_spin_dims() > 1)
        ns = 2;

    std::vector<int> offsets(nranks);
    std::vector<int> sizes(nranks);
    int offset = 0;
    for (int i = 0; i < nranks; ++i) {

        offsets[i] = offset;
        int lsize  = nk_per_rank[i] * num_bands * ns;
        sizes[i]   = lsize;
        offset += lsize;
    }
    int size = offset;

    std::vector<double> tmp(sizes[pid]);
    for (int i = 0; i < sizes[pid]; ++i) {
        int gi   = offsets[pid] + i;
        int k    = gi / (num_bands * ns);
        int spin = (gi % (num_bands * ns)) / num_bands;
        int n    = (gi % (num_bands * ns)) % num_bands;
        tmp[i]   = kpoints_[k]->band_occupancy(n, spin);
    }

    std::vector<double> occupancies(size);
    comm().allgather(tmp.data(), sizes[pid], occupancies.data(), sizes.data(), offsets.data());

    for (int i = 0; i < size; ++i) {
        int k    = i / (num_bands * ns);
        int spin = (i % (num_bands * ns)) / num_bands;
        int n    = (i % (num_bands * ns)) % num_bands;
        kpoints_[k]->band_occupancy(n, spin, occupancies[i]);
    }
}

void K_point_set::find_band_occupancies()
{
    PROFILE("sirius::K_point_set::find_band_occupancies");

    double ef{0};
    double de{0.1};

    int s{1};
    int sp;

    mdarray<double, 3> bnd_occ(ctx_.num_bands(), ctx_.num_spin_dims(), num_kpoints());

    double ne{0};

    /* target number of electrons */
    double ne_target = unit_cell_.num_valence_electrons() - ctx_.parameters_input().extra_charge_;

    if (std::abs(ctx_.num_fv_states() * double(ctx_.max_occupancy()) - ne_target) < 1e-10) {
        // this is an insulator, skip search for band occupancies
        this->band_gap_ = -1;

        // determine fermi energy as max occupied band energy.
        double efermi = std::numeric_limits<double>::min();
        for (int ik = 0; ik < num_kpoints(); ik++) {
            for (int ispn = 0; ispn < ctx_.num_spin_dims(); ispn++) {
                for (int j = 0; j < ctx_.num_bands(); j++) {
                    efermi = std::max(efermi, kpoints_[ik]->band_energy(j, ispn));
                }
            }
        }
        energy_fermi_ = efermi;
        return;
    }

    int step{0};
    /* calculate occupations */
    while (std::abs(ne - ne_target) >= 1e-11) {
        /* update Efermi */
        ef += de;
        /* compute total number of electrons */
        ne = 0.0;
        for (int ik = 0; ik < num_kpoints(); ik++) {
            for (int ispn = 0; ispn < ctx_.num_spin_dims(); ispn++) {
                for (int j = 0; j < ctx_.num_bands(); j++) {
                    bnd_occ(j, ispn, ik) =
                        smearing::gaussian(kpoints_[ik]->band_energy(j, ispn) - ef, ctx_.smearing_width()) *
                        ctx_.max_occupancy();
                    ne += bnd_occ(j, ispn, ik) * kpoints_[ik]->weight();
                }
            }
        }

        sp = s;
        s  = (ne > ne_target) ? -1 : 1;
        /* reduce de step if we change the direction, otherwise increase the step */
        de = (s != sp) ? (-de * 0.5) : (de * 1.25);

        if (step > 10000) {
            std::stringstream s;
            s << "search of band occupancies failed after 10000 steps";
            TERMINATE(s);
        }
        step++;
    }

    energy_fermi_ = ef;

    for (int ik = 0; ik < num_kpoints(); ik++) {
        for (int ispn = 0; ispn < ctx_.num_spin_dims(); ispn++) {
            for (int j = 0; j < ctx_.num_bands(); j++) {
                kpoints_[ik]->band_occupancy(j, ispn, bnd_occ(j, ispn, ik));
            }
        }
    }

    band_gap_ = 0.0;

    int nve = static_cast<int>(ne_target + 1e-12);
    if (ctx_.num_spins() == 2 || (std::abs(nve - ne_target) < 1e-12 && nve % 2 == 0)) {
        /* find band gap */
        std::vector<std::pair<double, double>> eband;
        std::pair<double, double> eminmax;

        for (int ispn = 0; ispn < ctx_.num_spin_dims(); ispn++) {
            for (int j = 0; j < ctx_.num_bands(); j++) {
                eminmax.first  = 1e10;
                eminmax.second = -1e10;

                for (int ik = 0; ik < num_kpoints(); ik++) {
                    eminmax.first  = std::min(eminmax.first, kpoints_[ik]->band_energy(j, ispn));
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

void K_point_set::print_info()
{
    if (ctx_.comm().rank() == 0) {
        std::printf("\n");
        std::printf("total number of k-points : %i\n", num_kpoints());
        for (int i = 0; i < 80; i++) {
            std::printf("-");
        }
        std::printf("\n");
        std::printf("  ik                vk                    weight  num_gkvec");
        if (ctx_.full_potential()) {
            std::printf("   gklo_basis_size");
        }
        std::printf("\n");
        for (int i = 0; i < 80; i++) {
            std::printf("-");
        }
        std::printf("\n");
    }

    if (ctx_.comm_band().rank() == 0) {
        pstdout pout(comm());
        for (int ikloc = 0; ikloc < spl_num_kpoints().local_size(); ikloc++) {
            int ik = spl_num_kpoints(ikloc);
            pout.printf("%4i   %8.4f %8.4f %8.4f   %12.6f     %6i", ik, kpoints_[ik]->vk()[0], kpoints_[ik]->vk()[1],
                        kpoints_[ik]->vk()[2], kpoints_[ik]->weight(), kpoints_[ik]->num_gkvec());

            if (ctx_.full_potential()) {
                pout.printf("            %6i", kpoints_[ik]->gklo_basis_size());
            }

            pout.printf("\n");
        }
    }
}

void K_point_set::save(std::string const& name__) const
{
    if (ctx_.comm().rank() == 0) {
        if (!utils::file_exists(name__)) {
            HDF5_tree(name__, hdf5_access_t::truncate);
        }
        HDF5_tree fout(name__, hdf5_access_t::read_write);
        fout.create_node("K_point_set");
        fout["K_point_set"].write("num_kpoints", num_kpoints());
    }
    ctx_.comm().barrier();
    for (int ik = 0; ik < num_kpoints(); ik++) {
        /* check if this ranks stores the k-point */
        if (ctx_.comm_k().rank() == spl_num_kpoints_.local_rank(ik)) {
            kpoints_[ik]->save(name__, ik);
        }
        /* wait for all */
        ctx_.comm().barrier();
    }
}

/// \todo check parameters of saved data in a separate function
void K_point_set::load()
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

} // namespace sirius
