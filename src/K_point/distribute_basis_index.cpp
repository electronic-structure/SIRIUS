#include "k_point.h"

namespace sirius {

void K_point::distribute_basis_index()
{
    PROFILE();

    if (ctx_.full_potential())
    {
        /* distribute Gk+lo basis between rows */
        splindex<block_cyclic> spl_row(gklo_basis_size(), num_ranks_row_, rank_row_, ctx_.cyclic_block_size());
        gklo_basis_descriptors_row_.resize(spl_row.local_size());
        for (int i = 0; i < spl_row.local_size(); i++)
            gklo_basis_descriptors_row_[i] = gklo_basis_descriptors_[spl_row[i]];

        /* distribute Gk+lo basis between columns */
        splindex<block_cyclic> spl_col(gklo_basis_size(), num_ranks_col_, rank_col_, ctx_.cyclic_block_size());
        gklo_basis_descriptors_col_.resize(spl_col.local_size());
        for (int i = 0; i < spl_col.local_size(); i++)
            gklo_basis_descriptors_col_[i] = gklo_basis_descriptors_[spl_col[i]];

        #ifdef __SCALAPACK
        int bs = ctx_.cyclic_block_size();
        int nr = linalg_base::numroc(gklo_basis_size(), bs, rank_row(), 0, num_ranks_row());
        
        if (nr != gklo_basis_size_row()) TERMINATE("numroc returned a different local row size");

        int nc = linalg_base::numroc(gklo_basis_size(), bs, rank_col(), 0, num_ranks_col());
        
        if (nc != gklo_basis_size_col()) TERMINATE("numroc returned a different local column size");
        #endif

        /* get number of column G+k vectors */
        num_gkvec_col_ = 0;
        for (int i = 0; i < gklo_basis_size_col(); i++)
        {
            if (gklo_basis_descriptor_col(i).ig != -1) num_gkvec_col_++;
        }
    }
    else
    {
        /* split G+k vectors between all available ranks and keep the split index */
        gklo_basis_descriptors_row_.resize(num_gkvec_loc());
        
        for (int i = 0; i < num_gkvec_loc(); i++) {
            gklo_basis_descriptors_row_[i] = gklo_basis_descriptors_[gkvec_.gvec_offset(comm_.rank()) + i];
        }
    }

    /* get the number of row G+k-vectors */
    num_gkvec_row_ = 0;
    for (int i = 0; i < gklo_basis_size_row(); i++)
    {
        if (gklo_basis_descriptor_row(i).ig != -1) num_gkvec_row_++;
    }
}

};
