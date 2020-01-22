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

/** \file test_fv_states.hpp
 *
 *  \brief Test orthonormalisation of first-variational states.
 */

#include "k_point.hpp"

namespace sirius {

void K_point::test_fv_states()
{
    PROFILE("sirius::K_point::test_fv_states");

    STOP();

    // Wave_functions<true> o_fv(wf_size(), ctx_.num_fv_states(), ctx_.cyclic_block_size(), ctx_.blacs_grid(),
    // ctx_.blacs_grid_slice()); o_fv.set_num_swapped(ctx_.num_fv_states());

    // for (int i = 0; i < o_fv.spl_num_col().local_size(); i++)
    //{
    //    std::memset(o_fv[i], 0, wf_size() * sizeof(double_complex));
    //    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
    //    {
    //        int offset_wf = unit_cell_.atom(ia).offset_wf();
    //        auto& type = unit_cell_.atom(ia).type();
    //        auto& symmetry_class = unit_cell_.atom(ia).symmetry_class();

    //        for (int l = 0; l <= ctx_.lmax_apw(); l++)
    //        {
    //            int ordmax = type.indexr().num_rf(l);
    //            for (int io1 = 0; io1 < ordmax; io1++)
    //            {
    //                for (int io2 = 0; io2 < ordmax; io2++)
    //                {
    //                    for (int m = -l; m <= l; m++)
    //                    {
    //                        o_fv[i][offset_wf + type.indexb_by_l_m_order(l, m, io2)] +=
    //                            fv_states<true>()[i][offset_wf + type.indexb_by_l_m_order(l, m, io1)] *
    //                            symmetry_class.o_radial_integral(l, io1, io2);
    //                    }
    //                }
    //            }
    //        }
    //    }
    //    ctx_.fft().transform<1>(gkvec().partition(), &fv_states<true>()[i][unit_cell_.mt_basis_size()]);
    //    for (int ir = 0; ir < ctx_.fft().local_size(); ir++) ctx_.fft().buffer(ir) *=
    //    ctx_.step_function().theta_r(ir); ctx_.fft().transform<-1>(gkvec().partition(),
    //    &o_fv[i][unit_cell_.mt_basis_size()]);
    //}
    // o_fv.swap_backward(0, ctx_.num_fv_states());

    // dmatrix<double_complex> ovlp(ctx_.num_fv_states(), ctx_.num_fv_states(), ctx_.blacs_grid(),
    //                             ctx_.cyclic_block_size(), ctx_.cyclic_block_size());
    //
    // linalg<device_t::CPU>::gemm(2, 0, ctx_.num_fv_states(), ctx_.num_fv_states(), wf_size(), double_complex(1, 0),
    //                  fv_states<true>().prime(), o_fv.prime(), double_complex(0, 0), ovlp);
    //
    // double max_err = 0;
    // for (int i = 0; i < ovlp.num_cols_local(); i++)
    //{
    //    for (int j = 0; j < ovlp.num_rows_local(); j++)
    //    {
    //        if (ovlp.icol(i) == ovlp.irow(j)) ovlp(j, i) -= 1;
    //        max_err = std::max(max_err, std::abs(ovlp(j, i)));
    //    }
    //}

    // comm().allreduce<double, op_max>(&max_err, 1);

    // if (comm().rank() == 0)
    //{
    //    printf("k-point: %f %f %f, maximum error of fv_states overlap : %18.10e\n",
    //        vk_[0], vk_[1], vk_[2], max_err);
    //}
}

} // namespace sirius
