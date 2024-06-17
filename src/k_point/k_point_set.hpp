/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

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

#if defined(SIRIUS_USE_FP32)
    /// List of k-points in fp32 type, most calculation and assertion in this class only rely on fp64 type kpoints_
    std::vector<std::unique_ptr<K_point<float>>> kpoints_float_;
#endif

    /// Split index of k-points.
    splindex_chunk<kp_index_t> spl_num_kpoints_;

    /// Fermi energy which is searched in find_band_occupancies().
    double energy_fermi_{0};

    /// Band gap found by find_band_occupancies().
    double band_gap_{0};

    /// Copy constuctor is not allowed.
    K_point_set(K_point_set& src) = delete;

    /// Create regular grid of k-points.
    void
    create_k_mesh(r3::vector<int> k_grid__, r3::vector<int> k_shift__, int use_symmetry__);

    bool initialized_{false};

    /// Return sum of valence eigen-values store in Kpoint<T>.
    template <typename T>
    double
    valence_eval_sum() const;

    /// Return entropy contribution from smearing store in Kpoint<T>.
    template <typename T>
    double
    entropy_sum() const;

  public:
    /// Create empty k-point set.
    K_point_set(Simulation_context& ctx__)
        : ctx_(ctx__)
    {
    }

    /// Create a regular mesh of k-points.
    K_point_set(Simulation_context& ctx__, r3::vector<int> k_grid__, r3::vector<int> k_shift__, int use_symmetry__)
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
    void
    initialize(std::vector<int> const& counts = {});

    /// Sync band energies or occupancies between all MPI ranks.
    template <typename T, sync_band_t what>
    void
    sync_band();

    /// Find Fermi energy and band occupation numbers.
    template <typename T>
    void
    find_band_occupancies();

    /// Print basic info to the standard output.
    void
    print_info();

    /// Save k-point set to HDF5 file.
    void
    save(std::string const& name__) const;

    void
    load();

    /// Return sum of valence eigen-values.
    double
    valence_eval_sum() const;

    /// Return entropy contribution from smearing.
    double
    entropy_sum() const;

    inline auto const&
    spl_num_kpoints() const
    {
        return spl_num_kpoints_;
    }

    inline auto const&
    comm() const
    {
        return ctx_.comm_k();
    }

    /// Update k-points after moving atoms or changing the lattice vectors.
    void
    update()
    {
        /* update k-points */
        for (auto it : spl_num_kpoints_) {
            kpoints_[it.i]->update();
#if defined(SIRIUS_USE_FP32)
            kpoints_float_[it.i]->update();
#endif
        }
    }

    /// Get a list of band energies for a given k-point index.
    template <typename T>
    auto
    get_band_energies(int ik__, int ispn__) const
    {
        std::vector<double> bnd_e(ctx_.num_bands());
        for (int j = 0; j < ctx_.num_bands(); j++) {
            bnd_e[j] = (*this).get<T>(ik__)->band_energy(j, ispn__);
        }
        return bnd_e;
    }

    /// Return maximum number of G+k vectors among all k-points.
    int
    max_num_gkvec() const
    {
        int max_num_gkvec{0};
        for (auto it : spl_num_kpoints_) {
            max_num_gkvec = std::max(max_num_gkvec, kpoints_[it.i]->num_gkvec());
        }
        comm().allreduce<int, mpi::op_t::max>(&max_num_gkvec, 1);
        return max_num_gkvec;
    }

    /// Add k-point to the set.
    void
    add_kpoint(r3::vector<double> vk__, double weight__)
    {
        kpoints_.push_back(std::make_unique<K_point<double>>(ctx_, vk__, weight__));
#ifdef SIRIUS_USE_FP32
        kpoints_float_.push_back(std::make_unique<K_point<float>>(ctx_, vk__, weight__));
#endif
    }

    /// Add multiple k-points to the set.
    void
    add_kpoints(mdarray<double, 2> const& kpoints__, double const* weights__)
    {
        for (int ik = 0; ik < (int)kpoints__.size(1); ik++) {
            add_kpoint(&kpoints__(0, ik), weights__[ik]);
        }
    }

    template <typename T>
    inline K_point<T>*
    get(int ik__) const;

    template <typename T>
    inline K_point<T>*
    get(int ik__)
    {
        return const_cast<K_point<T>*>(static_cast<K_point_set const&>(*this).get<T>(ik__));
    }

    /// Return total number of k-points.
    inline int
    num_kpoints() const
    {
        return static_cast<int>(kpoints_.size());
    }

    inline auto
    spl_num_kpoints(kp_index_t::local ikloc__) const
    {
        return spl_num_kpoints_.global_index(ikloc__);
    }

    inline double
    energy_fermi() const
    {
        return energy_fermi_;
    }

    inline void
    set_energy_fermi(double energy_fermi__)
    {
        this->energy_fermi_ = energy_fermi__;
    }

    inline double
    band_gap() const
    {
        return band_gap_;
    }

    /// Find index of k-point.
    inline int
    find_kpoint(r3::vector<double> vk__)
    {
        for (int ik = 0; ik < num_kpoints(); ik++) {
            if ((kpoints_[ik]->vk() - vk__).length() < 1e-12) {
                return ik;
            }
        }
        return -1;
    }

    inline auto&
    ctx()
    {
        return ctx_;
    }

    const auto&
    unit_cell()
    {
        return ctx_.unit_cell();
    }

    /// Send G+k vectors of k-point jk to a given rank.
    /** Other ranks receive an empty Gvec placeholder */
    inline fft::Gvec
    get_gkvec(kp_index_t::global jk__, int rank__)
    {
        /* rank in the k-point communicator */
        int my_rank = comm().rank();

        /* rank that stores jk */
        int jrank = spl_num_kpoints().location(jk__).ib;

        /* need this to pass communicator */
        fft::Gvec gkvec(ctx_.comm_band());

        fft::Gvec const* gvptr{nullptr};
        /* if this rank stores the k-point, then send it */
        if (my_rank == jrank) {
            gvptr = &kpoints_[jk__].get()->gkvec();
        } else {
            gvptr = &gkvec;
        }
        return send_recv(comm(), *gvptr, jrank, rank__);
    }
};

template <>
inline K_point<double>*
K_point_set::get<double>(int ik__) const
{
    RTE_ASSERT(ik__ >= 0 && ik__ < (int)kpoints_.size());
    return kpoints_[ik__].get();
}

template <>
inline K_point<float>*
K_point_set::get<float>(int ik__) const
{
#if defined(SIRIUS_USE_FP32)
    RTE_ASSERT(ik__ >= 0 && ik__ < (int)kpoints_float_.size());
    return kpoints_float_[ik__].get();
#else
    RTE_THROW("not compiled with FP32 support");
    return nullptr; // make compiler happy
#endif
}

}; // namespace sirius

#endif // __K_POINT_SET_H__
