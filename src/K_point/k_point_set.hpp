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

/** \file k_point_set.hpp
 *
 *  \brief Contains declaration and partial implementation of sirius::K_point_set class.
 */

#ifndef __K_POINT_SET_HPP__
#define __K_POINT_SET_HPP__

#include "k_point.hpp"
#include "geometry3d.hpp"
#include "smearing.hpp"
#include "Symmetry/get_irreducible_reciprocal_mesh.hpp"

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
    /// Context of a simulation.
    Simulation_context& ctx_;

    /// List of k-points.
    std::vector<std::unique_ptr<K_point>> kpoints_;

    /// Split index of k-points.
    splindex<splindex_t::chunk> spl_num_kpoints_;

    double energy_fermi_{0};

    double band_gap_{0};

    Unit_cell& unit_cell_;

    K_point_set(K_point_set& src) = delete;

    void create_k_mesh(vector3d<int> k_grid__,
                       vector3d<int> k_shift__,
                       int           use_symmetry__);

  public:
    K_point_set(Simulation_context& ctx__)
        : ctx_(ctx__)
        , unit_cell_(ctx__.unit_cell())
    {
    }

    K_point_set(Simulation_context& ctx__,
                std::vector<int>    k_grid__,
                std::vector<int>    k_shift__,
                int                 use_symmetry__)
        : ctx_(ctx__)
        , unit_cell_(ctx__.unit_cell())
    {
        create_k_mesh(k_grid__, k_shift__, use_symmetry__);
    }

    K_point_set(Simulation_context& ctx__,
                vector3d<int>       k_grid__,
                vector3d<int>       k_shift__,
                int                 use_symmetry__)
        : ctx_(ctx__)
        , unit_cell_(ctx__.unit_cell())
    {
        create_k_mesh(k_grid__, k_shift__, use_symmetry__);
    }

    K_point_set(Simulation_context& ctx__, std::vector<vector3d<double>> const& vec__)
        : ctx_(ctx__)
        , unit_cell_(ctx__.unit_cell())
    {
        for (auto& v : vec__) {
            add_kpoint(&v[0], 1.0);
        }
        initialize();
    }

    K_point_set(Simulation_context& ctx__, std::initializer_list<std::initializer_list<double>> vec__)
        : K_point_set(ctx__, std::vector<vector3d<double>>(vec__.begin(), vec__.end()))
    {
    }

    /// Initialize the k-point set
    void initialize(std::vector<int> const& counts = {});

    /// Update k-points after moving atoms or changing the lattice vectors.
    void update()
    {
        /* update k-points */
        for (int ikloc = 0; ikloc < spl_num_kpoints().local_size(); ikloc++) {
            int ik = spl_num_kpoints(ikloc);
            kpoints_[ik]->update();
        }
    }

    /// Get a list of band energies for a given k-point index.
    std::vector<double> get_band_energies(int ik__, int ispn__) const
    {
        std::vector<double> bnd_e(ctx_.num_bands());
        for (int j = 0; j < ctx_.num_bands(); j++) {
            bnd_e[j] = (*this)[ik__]->band_energy(j, ispn__);
        }
        return bnd_e;
    }

    /// Sync band occupations numbers
    void sync_band_occupancies();

    /// Find Fermi energy and band occupation numbers
    void find_band_occupancies();

    /// Return sum of valence eigen-values
    double valence_eval_sum() const
    {
        double eval_sum{0};

        for (int ik = 0; ik < num_kpoints(); ik++) {
            const auto& kp = kpoints_[ik];
            double wk = kp->weight();
            for (int j = 0; j < ctx_.num_bands(); j++) {
                for (int ispn = 0; ispn < ctx_.num_spin_dims(); ispn++) {
                    eval_sum += wk * kp->band_energy(j, ispn) * kp->band_occupancy(j, ispn);
                }
            }
        }

        return eval_sum;
    }

    void print_info();

    void sync_band_energies();

    /// Save k-point set to HDF5 file.
    void save(std::string const& name__) const;

    void load();

    /// Return maximum number of G+k vectors among all k-points.
    int max_num_gkvec() const
    {
        int max_num_gkvec{0};
        for (int ikloc = 0; ikloc < spl_num_kpoints_.local_size(); ikloc++) {
            auto ik       = spl_num_kpoints_[ikloc];
            max_num_gkvec = std::max(max_num_gkvec, kpoints_[ik]->num_gkvec());
        }
        comm().allreduce<int, mpi_op_t::max>(&max_num_gkvec, 1);
        return max_num_gkvec;
    }

    void add_kpoint(double const* vk__, double weight__)
    {
        PROFILE("sirius::K_point_set::add_kpoint");
        kpoints_.push_back(std::unique_ptr<K_point>(new K_point(ctx_, vk__, weight__)));
    }

    void add_kpoints(mdarray<double, 2> const& kpoints__, double const* weights__)
    {
        PROFILE("sirius::K_point_set::add_kpoints");
        for (int ik = 0; ik < (int)kpoints__.size(1); ik++) {
            add_kpoint(&kpoints__(0, ik), weights__[ik]);
        }
    }

    inline K_point* operator[](int i)
    {
        assert(i >= 0 && i < (int)kpoints_.size());
        return kpoints_[i].get();
    }

    inline K_point* operator[](int i) const
    {
        assert(i >= 0 && i < (int)kpoints_.size());

        return kpoints_[i].get();
    }

    inline int num_kpoints() const
    {
        return static_cast<int>(kpoints_.size());
    }

    inline splindex<splindex_t::chunk> const& spl_num_kpoints() const
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
        return ctx_.comm_k();
    }

    inline Simulation_context& ctx()
    {
        return ctx_;
    }

    const Unit_cell& unit_cell()
    {
        return unit_cell_;
    }

    /// Send G+k vectors of k-point jk to a given rank.
    /** Other ranks receive an empty Gvec placeholder */
    inline Gvec send_recv_gkvec(int jk__, int rank__)
    {
        /* rank in the k-point communicator */
        int my_rank = comm().rank();

        /* rank that stores jk */
        int jrank = spl_num_kpoints().local_rank(jk__);

        /* placeholder for G+k vectors of kpoint jk */
        Gvec gkvec(ctx_.comm_band());

        /* if this rank stores the k-point, then send it */
        if (jrank == my_rank) {
            auto kp = kpoints_[jk__].get();
            kp->gkvec().send_recv(comm(), jrank, rank__, gkvec);
        }
        /* this rank receives the k-point */
        if (rank__ == my_rank) {
            gkvec.send_recv(comm(), jrank, rank__, gkvec);
        }
        return gkvec;
    }
};


}; // namespace sirius

#endif // __K_POINT_SET_H__
