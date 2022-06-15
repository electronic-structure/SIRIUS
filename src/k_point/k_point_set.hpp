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

#if defined(USE_FP32)
    /// List of k-points in fp32 type, most calculation and assertion in this class only rely on fp64 type kpoints_
    std::vector<std::unique_ptr<K_point<float>>> kpoints_float_;
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

    /// Return sum of valence eigen-values store in Kpoint<T>.
    template <typename T>
    double valence_eval_sum() const;

    /// Return entropy contribution from smearing store in Kpoint<T>.
    template <typename T>
    double entropy_sum() const;
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
    template <typename T, sync_band_t what>
    void sync_band();

    /// Find Fermi energy and band occupation numbers.
    template <typename T>
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
#if defined(USE_FP32)
            kpoints_float_[ik]->update();
#endif
        }
    }

    /// Get a list of band energies for a given k-point index.
    template <typename T>
    std::vector<double> get_band_energies(int ik__, int ispn__) const;

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
        kpoints_.push_back(std::unique_ptr<K_point<double>>(new K_point<double>(ctx_, vk__, weight__)));
#ifdef USE_FP32
        kpoints_float_.push_back(std::unique_ptr<K_point<float>>(new K_point<float>(ctx_, vk__, weight__)));
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
    inline K_point<T>* get(int ik__) const;

    template <typename T>
    inline K_point<T>* get(int ik__)
    {
        return const_cast<K_point<T>*>(static_cast<K_point_set const&>(*this).get<T>(ik__));
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
    inline Gvec get_gkvec(int jk__, int rank__)
    {
        /* rank in the k-point communicator */
        int my_rank = comm().rank();

        /* rank that stores jk */
        int jrank = spl_num_kpoints().local_rank(jk__);

        /* need this to pass communicator */
        Gvec gkvec(ctx_.comm_band());

        Gvec const* gvptr{nullptr};
        /* if this rank stores the k-point, then send it */
        if (my_rank == jrank) {
            gvptr = &kpoints_[jk__].get()->gkvec();
        } else {
            gvptr = &gkvec;
        }
        return send_recv(comm(), *gvptr, jrank, rank__);
    }
};

template<>
inline K_point<double>* K_point_set::get<double>(int ik__) const
{
    assert(ik__ >= 0 && ik__ < (int)kpoints_.size());
    return kpoints_[ik__].get();
}

template<>
inline K_point<float>* K_point_set::get<float>(int ik__) const
{
#if defined(USE_FP32)
    assert(ik__ >= 0 && ik__ < (int)kpoints_float_.size());
    return kpoints_float_[ik__].get();
#else
    RTE_THROW("not compiled with FP32 support");
    return nullptr; // make compiler happy
#endif
}

template <typename T>
std::vector<double> K_point_set::get_band_energies(int ik__, int ispn__) const
{
    std::vector<double> bnd_e(ctx_.num_bands());
    for (int j = 0; j < ctx_.num_bands(); j++) {
        bnd_e[j] = (*this).get<T>(ik__)->band_energy(j, ispn__);
    }
    return bnd_e;
}

}; // namespace sirius

#endif // __K_POINT_SET_H__
