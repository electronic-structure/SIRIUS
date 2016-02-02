#include "k_point.h"

namespace sirius {

void K_point::test_fv_states()
{
    PROFILE();

    Wave_functions<true> o_fv(wf_size(), ctx_.num_fv_states(), ctx_.cyclic_block_size(), ctx_.blacs_grid(), ctx_.blacs_grid_slice());
    o_fv.set_num_swapped(ctx_.num_fv_states());

    for (int i = 0; i < o_fv.spl_num_swapped().local_size(); i++)
    {
        std::memset(o_fv[i], 0, wf_size() * sizeof(double_complex));
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
        {
            int offset_wf = unit_cell_.atom(ia).offset_wf();
            auto& type = unit_cell_.atom(ia).type();
            auto& symmetry_class = unit_cell_.atom(ia).symmetry_class();

            for (int l = 0; l <= ctx_.lmax_apw(); l++)
            {
                int ordmax = type.indexr().num_rf(l);
                for (int io1 = 0; io1 < ordmax; io1++)
                {
                    for (int io2 = 0; io2 < ordmax; io2++)
                    {
                        for (int m = -l; m <= l; m++)
                        {
                            o_fv[i][offset_wf + type.indexb_by_l_m_order(l, m, io2)] +=
                                fv_states<true>()[i][offset_wf + type.indexb_by_l_m_order(l, m, io1)] *
                                symmetry_class.o_radial_integral(l, io1, io2);
                        }
                    }
                }
            }
        }
        ctx_.fft().transform<1>(gkvec_, &fv_states<true>()[i][unit_cell_.mt_basis_size()]);
        for (int ir = 0; ir < ctx_.fft().local_size(); ir++) ctx_.fft().buffer(ir) *= ctx_.step_function().theta_r(ir);
        ctx_.fft().transform<-1>(gkvec_, &o_fv[i][unit_cell_.mt_basis_size()]);
    }
    o_fv.swap_backward(0, ctx_.num_fv_states());

    dmatrix<double_complex> ovlp(ctx_.num_fv_states(), ctx_.num_fv_states(), ctx_.blacs_grid(),
                                 ctx_.cyclic_block_size(), ctx_.cyclic_block_size());
    
    linalg<CPU>::gemm(2, 0, ctx_.num_fv_states(), ctx_.num_fv_states(), wf_size(), double_complex(1, 0),
                      fv_states<true>().coeffs(), o_fv.coeffs(), double_complex(0, 0), ovlp);
    
    double max_err = 0;
    for (int i = 0; i < ovlp.num_cols_local(); i++)
    {
        for (int j = 0; j < ovlp.num_rows_local(); j++)
        {
            if (ovlp.icol(i) == ovlp.irow(j)) ovlp(j, i) -= 1;
            max_err = std::max(max_err, std::abs(ovlp(j, i)));
        }
    }

    comm().allreduce<double, op_max>(&max_err, 1);

    if (comm().rank() == 0)
    {
        printf("k-point: %f %f %f, maximum error of fv_states overlap : %18.10e\n", 
            vk_[0], vk_[1], vk_[2], max_err);
    }
}

}
