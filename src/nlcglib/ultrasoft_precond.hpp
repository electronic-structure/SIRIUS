// Copyright (c) 2023 Simon Pintarelli, Anton Kozhevnikov, Thomas Schulthess
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
#include "SDDK/memory.hpp"
#include <memory>
#include <complex>

#ifdef SIRIUS_NLCGLIB

namespace sirius {

class UltrasoftPrecond : public nlcglib::UltrasoftPrecondBase
{
  private:
    using key_t = std::pair<int, int>;
    using numeric_t = std::complex<double>;
    using op_t = Ultrasoft_preconditioner<numeric_t>;

  public:
    using buffer_t = nlcglib::MatrixBaseZ::buffer_t;

  public:
    UltrasoftPrecond(const K_point_set& kset, Simulation_context& ctx, const Q_operator<double>& q_op);

    virtual void apply(const key_t& key, buffer_t& out, buffer_t& in) const override;
    virtual std::vector<std::pair<int, int>> get_keys() const override;

  private:
    std::map<key_t, std::shared_ptr<op_t>> data;
};

inline UltrasoftPrecond::UltrasoftPrecond(const K_point_set& kset, Simulation_context& ctx, const Q_operator<double>& q_op)
{
    int nk = kset.spl_num_kpoints().local_size();
    for (int ik_loc = 0; ik_loc < nk; ++ik_loc) {
        int ik   = kset.spl_num_kpoints(ik_loc);
        auto& kp = *kset.get<double>(ik);
        for (int ispn = 0; ispn < ctx.num_spins(); ++ispn) {
            key_t key{ik, ispn};
            data[key] = std::make_shared<op_t>(ctx, q_op, ispn, kp.beta_projectors(), kp.gkvec());
        }
    }
}

inline void UltrasoftPrecond::apply(const key_t& key, buffer_t& out, buffer_t& in) const
{
    auto& op    = data.at(key);
    auto array_out = make_matrix_view(out);
    auto array_in  = make_matrix_view(in);
    sddk::memory_t pm = out.memtype == nlcglib::memory_type::host ? sddk::memory_t::host : sddk::memory_t::device;
    op->apply(array_out, array_in, pm);
}

inline std::vector<std::pair<int,int>> UltrasoftPrecond::get_keys() const
{
    std::vector<key_t> keys;
    for (auto& elem : data) {
        keys.push_back(elem.first);
    }
    return keys;
}

}  // sirius

#endif /* SIRIUS_NLCGLIB */
#endif /* __ULTRASOFT_PRECOND_HPP__ */
