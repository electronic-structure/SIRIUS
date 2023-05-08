#ifndef OVERLAP_H
#define OVERLAP_H

#ifdef SIRIUS_NLCGLIB
#include "hamiltonian/inverse_overlap.hpp"
#include "k_point/k_point_set.hpp"
#include "context/simulation_context.hpp"
#include <nlcglib/interface.hpp>

#include "hamiltonian/non_local_operator.hpp"
#include "adaptor.hpp"
#include <stdexcept>
#include "SDDK/memory.hpp"
#include <memory>
#include <complex>

namespace sirius {

using inverseS = InverseS_k<std::complex<double>>;
using S = S_k<std::complex<double>>;

template<class op_t>
class Overlap_operators : public nlcglib::OverlapBase
{
  private:
    using key_t = std::pair<int, int>;
  public:
    Overlap_operators(const K_point_set& kset, Simulation_context& ctx, const Q_operator<double>& q_op);
    // virtual void apply(nlcglib::MatrixBaseZ& out, const nlcglib::MatrixBaseZ& in) const override;
    /// return a functor for nlcglib at given key
    virtual void apply(const key_t& key, nlcglib::MatrixBaseZ::buffer_t& out, nlcglib::MatrixBaseZ::buffer_t& in) const override;
    virtual std::vector<std::pair<int, int>>  get_keys() const override;

  private:
    std::map<key_t, std::shared_ptr<op_t>> data;
};

template<class op_t>
Overlap_operators<op_t>::Overlap_operators(const K_point_set& kset, Simulation_context& ctx, const Q_operator<double>& q_op)
{
    int nk = kset.spl_num_kpoints().local_size();
    for (int ik_loc = 0; ik_loc < nk; ++ik_loc) {
        int ik   = kset.spl_num_kpoints(ik_loc);
        auto& kp = *kset.get<double>(ik);
        for (int ispn = 0; ispn < ctx.num_spins(); ++ispn) {
            key_t key{ik, ispn};
            data[key] = std::make_shared<op_t>(ctx, q_op, kp.beta_projectors(), ispn);
        }
    }
}

template <class op_t>
void
Overlap_operators<op_t>::apply(const key_t& key, nlcglib::MatrixBaseZ::buffer_t& out, nlcglib::MatrixBaseZ::buffer_t& in) const
{
    auto& op = data.at(key);
    auto array_out = make_matrix_view(out);
    auto array_in  = make_matrix_view(in);
    // TODO: make sure the processing unit is correct
    sddk::memory_t pm = out.memtype == nlcglib::memory_type::host ? sddk::memory_t::host : sddk::memory_t::device;
    op->apply(array_out, array_in, pm);
}

template <class op_t>
std::vector<std::pair<int,int>> Overlap_operators<op_t>::get_keys() const
{
    std::vector<key_t> keys;
    for (auto& elem : data) {
        keys.push_back(elem.first);
    }
    return keys;
}


} // namespace sirius
#endif /* __NLCGLIB */

#endif /* OVERLAP_H */
