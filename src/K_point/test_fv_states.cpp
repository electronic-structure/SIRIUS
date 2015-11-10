#include "k_point.h"

namespace sirius {

void K_point::test_fv_states()
{
    PROFILE();

    dmatrix<double_complex> o_fv_slice(wf_size(), parameters_.num_fv_states(), blacs_grid_slice_, 1, 1);
    assert(o_fv_slice.num_cols_local() == fv_states_slice_.num_cols_local());
    o_fv_slice.zero();

    for (int i = 0; i < o_fv_slice.num_cols_local(); i++)
    {
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
        {
            int offset_wf = unit_cell_.atom(ia)->offset_wf();
            Atom_type* type = unit_cell_.atom(ia)->type();
            Atom_symmetry_class* symmetry_class = unit_cell_.atom(ia)->symmetry_class();

            for (int l = 0; l <= parameters_.lmax_apw(); l++)
            {
                int ordmax = type->indexr().num_rf(l);
                for (int io1 = 0; io1 < ordmax; io1++)
                {
                    for (int io2 = 0; io2 < ordmax; io2++)
                    {
                        for (int m = -l; m <= l; m++)
                        {
                            o_fv_slice(offset_wf + type->indexb_by_l_m_order(l, m, io2), i) +=
                                fv_states_slice_(offset_wf + type->indexb_by_l_m_order(l, m, io1), i) *
                                symmetry_class->o_radial_integral(l, io1, io2);
                        }
                    }
                }
            }
        }
        STOP();
        //ctx_.fft(0)->input(num_gkvec(), gkvec_.index_map(), &fv_states_slice_(unit_cell_.mt_basis_size(), i));
        //ctx_.fft(0)->transform(1, gkvec_.z_sticks_coord());
        //for (int ir = 0; ir < ctx_.fft(0)->size(); ir++) ctx_.fft(0)->buffer(ir) *= ctx_.step_function()->theta_r(ir);
        //ctx_.fft(0)->transform(-1, gkvec_.z_sticks_coord());
        //ctx_.fft(0)->output(num_gkvec(), gkvec_.index_map(), &o_fv_slice(unit_cell_.mt_basis_size(), i));
    }

    dmatrix<double_complex> o_fv(wf_size(), parameters_.num_fv_states(), blacs_grid_,
                                 parameters_.cyclic_block_size(), parameters_.cyclic_block_size());

    /* change from slice storage to 2d block cyclic */
    linalg<CPU>::gemr2d(wf_size(), parameters_.num_fv_states(), o_fv_slice, 0, 0, o_fv, 0, 0, blacs_grid_.context());

    dmatrix<double_complex> ovlp(parameters_.num_fv_states(), parameters_.num_fv_states(), blacs_grid_,
                                 parameters_.cyclic_block_size(), parameters_.cyclic_block_size());
    
    linalg<CPU>::gemm(2, 0, parameters_.num_fv_states(), parameters_.num_fv_states(), wf_size(), double_complex(1, 0),
                      fv_states_, o_fv, double_complex(0, 0), ovlp);
    
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
