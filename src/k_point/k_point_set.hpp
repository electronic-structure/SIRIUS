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
#include "dft/smearing.hpp"

namespace sirius {

enum class sync_band_t
{
    energy,
    occupancy
};

/// Set of k-points.
class K_point_set
{
  private:
    /// Context of a simulation.
    Simulation_context& ctx_;

    /// List of k-points.
    std::vector<std::unique_ptr<K_point<double>>> kpoints_;

#ifdef USE_FP32
    /// List of k-points in fp32 type, most calculation and assertion in this class only rely on fp64 type kpoints_
    std::vector<std::unique_ptr<K_point<float>>> kpoints_float_;

    /// bool variable to store last access of kpoints using fp32 or fp64 type
    bool access_fp64{true};
#endif

    /// Split index of k-points.
    splindex<splindex_t::chunk> spl_num_kpoints_;

    /// Fermi energy which is searched in find_band_occupancies().
    double energy_fermi_{0};

    /// Band gap found by find_band_occupancies().
    double band_gap_{0};

    /// Copy constuctor is not allowed.
    K_point_set(K_point_set& src) = delete;

    /// Create regular grid of k-points.
    void create_k_mesh(vector3d<int> k_grid__, vector3d<int> k_shift__, int use_symmetry__);

    bool initialized_{false};

  public:
    /// Create empty k-point set.
    K_point_set(Simulation_context& ctx__)
        : ctx_(ctx__)
    {
    }

    /// Create a regular mesh of k-points.
    K_point_set(Simulation_context& ctx__, vector3d<int> k_grid__, vector3d<int> k_shift__, int use_symmetry__)
        : ctx_(ctx__)
    {
        create_k_mesh(k_grid__, k_shift__, use_symmetry__);
    }

    /// Create k-point set from a list of vectors.
    K_point_set(Simulation_context& ctx__, std::vector<std::array<double, 3>> vec__)
        : ctx_(ctx__)
    {
        for (auto& v : vec__) {
            add_kpoint(&v[0], 1.0);
        }
        initialize();
    }

    /// Create k-point set from a list of vectors.
    K_point_set(Simulation_context& ctx__, std::initializer_list<std::array<double, 3>> vec__)
        : K_point_set(ctx__, std::vector<std::array<double, 3>>(vec__.begin(), vec__.end()))
    {
    }

    /// Initialize the k-point set
    void initialize(std::vector<int> const& counts = {});

    /// Sync band energies or occupancies between all MPI ranks.
    template <sync_band_t what>
    void sync_band();

    /// Find Fermi energy and band occupation numbers.
    void find_band_occupancies();

    /// Print basic info to the standard output.
    void print_info();

    /// Save k-point set to HDF5 file.
    void save(std::string const& name__) const;

    void load();

    /// Return sum of valence eigen-values.
    double valence_eval_sum() const;

    /// Return entropy contribution from smearing.
    double entropy_sum() const;

    /// Update k-points after moving atoms or changing the lattice vectors.
    void update()
    {
        /* update k-points */
        for (int ikloc = 0; ikloc < spl_num_kpoints().local_size(); ikloc++) {
            int ik = spl_num_kpoints(ikloc);
            kpoints_[ik]->update();
#ifdef USE_FP32
            kpoints_float_[ik]->update();
#endif
        }
    }

    /// Get a list of band energies for a given k-point index.
    std::vector<double> get_band_energies(int ik__, int ispn__) const
    {
        std::vector<double> bnd_e(ctx_.num_bands());
        for (int j = 0; j < ctx_.num_bands(); j++) {
#ifdef USE_FP32
            if (access_fp64) {
#endif
                bnd_e[j] = (*this).get<double>(ik__)->band_energy(j, ispn__);
#ifdef USE_FP32
            } else {
                bnd_e[j] = (*this).get<float>(ik__)->band_energy(j, ispn__);
            }
#endif
        }
        return bnd_e;
    }

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

    /// Add k-point to the set.
    void add_kpoint(double const* vk__, double weight__)
    {
        int id = static_cast<int>(kpoints_.size());
        kpoints_.push_back(std::unique_ptr<K_point<double>>(new K_point<double>(ctx_, vk__, weight__, id)));
#ifdef USE_FP32
        kpoints_float_.push_back(std::unique_ptr<K_point<float>>(new K_point<float>(ctx_, vk__, static_cast<float>(weight__), id)));
#endif
    }

    /// Add multiple k-points to the set.
    void add_kpoints(sddk::mdarray<double, 2> const& kpoints__, double const* weights__)
    {
        for (int ik = 0; ik < (int)kpoints__.size(1); ik++) {
            add_kpoint(&kpoints__(0, ik), weights__[ik]);
        }
    }

    template <typename T>
    inline K_point<T>* get(int i)
    {
        assert(i >= 0 && i < (int)kpoints_.size());

#ifdef USE_FP32
        if(std::is_same<T, double>::value) {
            access_fp64 = true;
#endif
            return kpoints_[i].get();
#ifdef USE_FP32
        } else {
            access_fp64 = false;
            return kpoints_float_[i].get();
        }
#endif
    }

    template <typename T>
    inline K_point<T>* get(int i) const
    {
        assert(i >= 0 && i < (int)kpoints_.size());

#ifdef USE_FP32
        if(std::is_same<T, double>::value) {
            access_fp64 = true;
#endif
            return kpoints_[i].get();
#ifdef USE_FP32
        } else {
            access_fp64 = false;
            return kpoints_float_[i].get();
        }
#endif
    }

    /// Return total number of k-points.
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
        return ctx_.unit_cell();
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
#ifdef USE_FP32
            auto kp_float = kpoints_float_[jk__].get();
            kp_float->gkvec().send_recv(comm(), jrank, rank__, gkvec);
#endif
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
