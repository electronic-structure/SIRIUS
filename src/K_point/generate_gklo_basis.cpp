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

/** \file generate_gklo_basis.cpp
 *
 *  \brief Generate LAPW basis of G+k augmented plane waves and local orbitals.
 */

#include "K_point/k_point.hpp"

namespace sirius {

void K_point::generate_gklo_basis()
{
    /* find local number of row G+k vectors */
    splindex<splindex_t::block_cyclic> spl_ngk_row(num_gkvec(), num_ranks_row_, rank_row_, ctx_.cyclic_block_size());
    num_gkvec_row_ = spl_ngk_row.local_size();

    igk_row_.resize(num_gkvec_row_);
    for (int i = 0; i < num_gkvec_row_; i++) {
        igk_row_[i] = spl_ngk_row[i];
    }

    /* find local number of column G+k vectors */
    splindex<splindex_t::block_cyclic> spl_ngk_col(num_gkvec(), num_ranks_col_, rank_col_, ctx_.cyclic_block_size());
    num_gkvec_col_ = spl_ngk_col.local_size();

    igk_col_.resize(num_gkvec_col_);
    for (int i = 0; i < num_gkvec_col_; i++) {
        igk_col_[i] = spl_ngk_col[i];
    }

    /* mapping between local and global G+k vecotor indices */
    igk_loc_.resize(num_gkvec_loc());
    for (int i = 0; i < num_gkvec_loc(); i++) {
        igk_loc_[i] = gkvec().offset() + i;
    }

    if (ctx_.full_potential()) {
        splindex<splindex_t::block_cyclic> spl_nlo_row(num_gkvec() + unit_cell_.mt_lo_basis_size(), num_ranks_row_,
                                                       rank_row_, ctx_.cyclic_block_size());
        splindex<splindex_t::block_cyclic> spl_nlo_col(num_gkvec() + unit_cell_.mt_lo_basis_size(), num_ranks_col_,
                                                       rank_col_, ctx_.cyclic_block_size());

        lo_basis_descriptor lo_desc;

        int idx{0};
        /* local orbital basis functions */
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            auto& atom = unit_cell_.atom(ia);
            auto& type = atom.type();

            int lo_index_offset = type.mt_aw_basis_size();

            for (int j = 0; j < type.mt_lo_basis_size(); j++) {
                int l         = type.indexb(lo_index_offset + j).l;
                int lm        = type.indexb(lo_index_offset + j).lm;
                int order     = type.indexb(lo_index_offset + j).order;
                int idxrf     = type.indexb(lo_index_offset + j).idxrf;
                lo_desc.ia    = static_cast<uint16_t>(ia);
                lo_desc.l     = static_cast<uint8_t>(l);
                lo_desc.lm    = static_cast<uint16_t>(lm);
                lo_desc.order = static_cast<uint8_t>(order);
                lo_desc.idxrf = static_cast<uint8_t>(idxrf);

                if (spl_nlo_row.local_rank(num_gkvec() + idx) == rank_row_) {
                    lo_basis_descriptors_row_.push_back(lo_desc);
                }
                if (spl_nlo_col.local_rank(num_gkvec() + idx) == rank_col_) {
                    lo_basis_descriptors_col_.push_back(lo_desc);
                }

                idx++;
            }
        }
        assert(idx == unit_cell_.mt_lo_basis_size());

        atom_lo_cols_.clear();
        atom_lo_cols_.resize(unit_cell_.num_atoms());
        for (int i = 0; i < num_lo_col(); i++) {
            int ia = lo_basis_descriptor_col(i).ia;
            atom_lo_cols_[ia].push_back(i);
        }

        atom_lo_rows_.clear();
        atom_lo_rows_.resize(unit_cell_.num_atoms());
        for (int i = 0; i < num_lo_row(); i++) {
            int ia = lo_basis_descriptor_row(i).ia;
            atom_lo_rows_[ia].push_back(i);
        }
    }
}

} // namespace sirius
