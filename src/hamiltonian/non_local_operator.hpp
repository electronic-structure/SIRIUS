/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file non_local_operator.hpp
 *
 *  \brief Contains declaration of sirius::Non_local_operator class.
 */

#ifndef __NON_LOCAL_OPERATOR_HPP__
#define __NON_LOCAL_OPERATOR_HPP__

#include "core/omp.hpp"
#include "core/rte/rte.hpp"
#include "non_local_operator_base.hpp"
#include "context/simulation_context.hpp"
#include "hubbard/hubbard_matrix.hpp"
#include <utility>
#include <vector>
#include <complex>

namespace sirius {
/* forward declaration */
template <typename T>
class Beta_projectors;
template <typename T>
class Beta_projectors_base;

template <typename T>
class D_operator : public Non_local_operator<T>
{
  private:
    void
    initialize();

  public:
    D_operator(Simulation_context const& ctx_);
};

template <typename T>
class Q_operator : public Non_local_operator<T>
{
  private:
    void
    initialize();

  public:
    Q_operator(Simulation_context const& ctx__);
};

template <typename T>
class U_operator
{
  private:
    Simulation_context const& ctx_;
    // sddk::mdarray<std::complex<T>, 3> um_;
    std::array<la::dmatrix<std::complex<T>>, 4> um_;
    std::vector<int> offset_;
    std::vector<std::pair<int, int>> atomic_orbitals_;
    int nhwf_;
    r3::vector<double> vk_;

  public:
    U_operator(Simulation_context const& ctx__, Hubbard_matrix const& um1__, std::array<double, 3> vk__);
    ~U_operator() = default;

    inline auto
    atomic_orbitals() const
    {
        return atomic_orbitals_;
    }

    inline auto
    atomic_orbitals(const int idx__) const
    {
        return atomic_orbitals_[idx__];
    }
    inline auto
    nhwf() const
    {
        return nhwf_;
    }

    inline auto
    offset(int ia__) const
    {
        return offset_[ia__];
    }

    std::complex<T> const&
    operator()(int m1, int m2, int j) const
    {
        return um_[j](m1, m2);
    }

    std::complex<T> const*
    at(memory_t mem__, const int idx1, const int idx2, const int idx3) const
    {
        return um_[idx3].at(mem__, idx1, idx2);
    }

    int
    find_orbital_index(const int ia__, const int n__, const int l__) const;

    matrix<std::complex<T>> const&
    mat(int i) const
    {
        return um_[i];
    }
};

/** \tparam T  Precision of the wave-functions.
 *  \tparam F  Type of the subspace.
 *
 *  \param [in]  spins          Range of the spin index.
 *  \param [in]  N              Starting index of the wave-functions.
 *  \param [in]  n              Number of wave-functions to which D and Q are applied.
 *  \param [in]  beta           Beta-projector generator
 *  \param [in]  beta_coeffs    Beta-projector coefficients
 *  \param [in]  phi            Wave-functions.
 *  \param [in]  d_op           Pointer to D-operator.
 *  \param [out] hphi           Resulting |beta>D<beta|phi>
 *  \param [in]  q_op           Pointer to Q-operator.
 *  \param [out] sphi           Resulting |beta>Q<beta|phi>
 **/
template <typename T, typename F>
void
apply_non_local_D_Q(memory_t mem__, wf::spin_range spins__, wf::band_range br__, Beta_projector_generator<T>& beta__,
                    beta_projectors_coeffs_t<T>& beta_coeffs__, wf::Wave_functions<T> const& phi__,
                    D_operator<T> const* d_op__, wf::Wave_functions<T>* hphi__, Q_operator<T> const* q_op__,
                    wf::Wave_functions<T>* sphi__)
{
    if (is_device_memory(mem__)) {
        RTE_ASSERT(beta__.pu() == device_t::GPU);
    }

    auto& ctx = beta__.ctx();

    for (int i = 0; i < beta__.num_chunks(); i++) {
        /* generate beta-projectors for a block of atoms */
        beta__.generate(beta_coeffs__, i);

        for (auto s = spins__.begin(); s != spins__.end(); s++) {
            auto sp       = phi__.actual_spin_index(s);
            auto beta_phi = inner_prod_beta<F>(ctx.spla_context(), mem__, ctx.host_memory_t(),
                                               is_device_memory(mem__), /* copy result back to gpu if true */
                                               beta_coeffs__, phi__, sp, br__);

            if (hphi__ && d_op__) {
                /* apply diagonal spin blocks */
                d_op__->apply(mem__, i, s.get(), *hphi__, br__, beta_coeffs__, beta_phi);
                if (!d_op__->is_diag() && hphi__->num_md() == wf::num_mag_dims(3)) {
                    /* apply non-diagonal spin blocks */
                    /* xor 3 operator will map 0 to 3 and 1 to 2 */
                    d_op__->apply(mem__, i, s.get() ^ 3, *hphi__, br__, beta_coeffs__, beta_phi);
                }
            }

            if (sphi__ && q_op__) {
                /* apply Q operator (diagonal in spin) */
                q_op__->apply(mem__, i, s.get(), *sphi__, br__, beta_coeffs__, beta_phi);
                if (!q_op__->is_diag() && sphi__->num_md() == wf::num_mag_dims(3)) {
                    q_op__->apply(mem__, i, s.get() ^ 3, *sphi__, br__, beta_coeffs__, beta_phi);
                }
            }
        }
    }
}

/// Compute |sphi> = (1 + Q)|phi>
template <typename T, typename F>
void
apply_S_operator(memory_t mem__, wf::spin_range spins__, wf::band_range br__, Beta_projector_generator<T>& beta__,
                 beta_projectors_coeffs_t<T>& beta_coeffs__, wf::Wave_functions<T> const& phi__,
                 Q_operator<T> const* q_op__, wf::Wave_functions<T>& sphi__)
{
    for (auto s = spins__.begin(); s != spins__.end(); s++) {
        wf::copy(mem__, phi__, s, br__, sphi__, s, br__);
    }

    if (q_op__) {
        apply_non_local_D_Q<T, F>(mem__, spins__, br__, beta__, beta_coeffs__, phi__, nullptr, nullptr, q_op__,
                                  &sphi__);
    }
}

/** Apply Hubbard U correction
 * \tparam T  Precision type of wave-functions (flat or double).
 * \param [in]  hub_wf   Hubbard atomic wave-functions.
 * \param [in]  phi      Set of wave-functions to which Hubbard correction is applied.
 * \param [out] hphi     Output wave-functions to which the result is added.
 */
template <typename T>
void
apply_U_operator(Simulation_context& ctx__, wf::spin_range spins__, wf::band_range br__,
                 wf::Wave_functions<T> const& hub_wf__, wf::Wave_functions<T> const& phi__, U_operator<T> const& um__,
                 wf::Wave_functions<T>& hphi__);
/// Apply strain derivative of S-operator to all scalar functions.
void
apply_S_operator_strain_deriv(memory_t mem__, int comp__, Beta_projector_generator<double>& bp__,
                              beta_projectors_coeffs_t<double>& bp_coeffs__,
                              Beta_projector_generator<double>& bp_strain_deriv__,
                              beta_projectors_coeffs_t<double>& bp_strain_deriv_coeffs__,
                              wf::Wave_functions<double>& phi__, Q_operator<double>& q_op__,
                              wf::Wave_functions<double>& ds_phi__);
} // namespace sirius

#endif
