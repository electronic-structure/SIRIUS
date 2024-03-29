/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __ULTRASOFT_PRECOND_HPP__
#define __ULTRASOFT_PRECOND_HPP__

#include "adaptor.hpp"
#include "k_point/k_point_set.hpp"
#include "context/simulation_context.hpp"
#include <nlcglib/interface.hpp>
#include "hamiltonian/non_local_operator.hpp"
#include "preconditioner/ultrasoft_precond_k.hpp"
#include "adaptor.hpp"
#include <stdexcept>
#include <memory>
#include <complex>

#ifdef SIRIUS_NLCGLIB

namespace sirius {

class UltrasoftPrecond : public nlcglib::UltrasoftPrecondBase
{
  private:
    using key_t     = std::pair<int, int>;
    using numeric_t = std::complex<double>;
    using op_t      = Ultrasoft_preconditioner<numeric_t>;

  public:
    using buffer_t = nlcglib::MatrixBaseZ::buffer_t;

  public:
    UltrasoftPrecond(const K_point_set& kset, Simulation_context& ctx, const Q_operator<double>& q_op);

    virtual void
    apply(const key_t& key, buffer_t& out, buffer_t& in) const override;
    virtual std::vector<std::pair<int, int>>
    get_keys() const override;

  private:
    std::map<key_t, std::shared_ptr<op_t>> data_;
};

inline UltrasoftPrecond::UltrasoftPrecond(K_point_set const& kset, Simulation_context& ctx,
                                          Q_operator<double> const& q_op)
{
    for (auto it : kset.spl_num_kpoints()) {
        auto& kp = *kset.get<double>(it.i);
        for (int ispn = 0; ispn < ctx.num_spins(); ++ispn) {
            key_t key{it.i.get(), ispn};
            data_[key] = std::make_shared<op_t>(ctx, q_op, ispn, kp.beta_projectors(), kp.gkvec());
        }
    }
}

inline void
UltrasoftPrecond::apply(const key_t& key, buffer_t& out, buffer_t& in) const
{
    auto& op       = data_.at(key);
    auto array_out = make_matrix_view(out);
    auto array_in  = make_matrix_view(in);
    memory_t pm    = out.memtype == nlcglib::memory_type::host ? memory_t::host : memory_t::device;
    op->apply(array_out, array_in, pm);
}

inline std::vector<std::pair<int, int>>
UltrasoftPrecond::get_keys() const
{
    std::vector<key_t> keys;
    for (auto& elem : data_) {
        keys.push_back(elem.first);
    }
    return keys;
}

} // namespace sirius

#endif /* SIRIUS_NLCGLIB */
#endif /* __ULTRASOFT_PRECOND_HPP__ */
