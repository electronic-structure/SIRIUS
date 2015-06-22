#include "k_point.h"

namespace sirius {

void K_point::distribute_basis_index()
{
    if (parameters_.wave_function_distribution() == block_cyclic_2d)
    {
        /* distribute Gk+lo basis between rows */
        splindex<block_cyclic> spl_row(gklo_basis_size(), num_ranks_row_, rank_row_, parameters_.cyclic_block_size());
        gklo_basis_descriptors_row_.resize(spl_row.local_size());
        for (int i = 0; i < (int)spl_row.local_size(); i++)
            gklo_basis_descriptors_row_[i] = gklo_basis_descriptors_[spl_row[i]];

        /* distribute Gk+lo basis between columns */
        splindex<block_cyclic> spl_col(gklo_basis_size(), num_ranks_col_, rank_col_, parameters_.cyclic_block_size());
        gklo_basis_descriptors_col_.resize(spl_col.local_size());
        for (int i = 0; i < (int)spl_col.local_size(); i++)
            gklo_basis_descriptors_col_[i] = gklo_basis_descriptors_[spl_col[i]];

        #ifdef __SCALAPACK
        int bs = parameters_.cyclic_block_size();
        int nr = linalg_base::numroc(gklo_basis_size(), bs, rank_row(), 0, num_ranks_row());
        
        if (nr != gklo_basis_size_row()) error_local(__FILE__, __LINE__, "numroc returned a different local row size");

        int nc = linalg_base::numroc(gklo_basis_size(), bs, rank_col(), 0, num_ranks_col());
        
        if (nc != gklo_basis_size_col()) error_local(__FILE__, __LINE__, "numroc returned a different local column size");
        #endif

        /* get number of column G+k vectors */
        num_gkvec_col_ = 0;
        for (int i = 0; i < gklo_basis_size_col(); i++)
        {
            if (gklo_basis_descriptor_col(i).igk != -1) num_gkvec_col_++;
        }
    }

    if (parameters_.wave_function_distribution() == slab)
    {
        /* split G+k vectors between all available ranks and keep the split index */
        spl_gkvec_ = splindex<block>(gklo_basis_size(), comm_.size(), comm_.rank());
        gklo_basis_descriptors_row_.resize(spl_gkvec_.local_size());
        for (int i = 0; i < (int)spl_gkvec_.local_size(); i++)
            gklo_basis_descriptors_row_[i] = gklo_basis_descriptors_[spl_gkvec_[i]];

    }

    /* get the number of row G+k-vectors */
    num_gkvec_row_ = 0;
    for (int i = 0; i < gklo_basis_size_row(); i++)
    {
        if (gklo_basis_descriptor_row(i).igk != -1) num_gkvec_row_++;
    }
}

};
