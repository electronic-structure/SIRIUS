//
// Created by mathieut on 7/23/19.
//

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

/** \file set_lapw_h_o.hpp
 *
 *  \brief Contains functions of LAPW Hamiltonian and overlap setup.
 */


#include "hamiltonian.hpp"

namespace sirius {

void Hamiltonian_k::set_fv_h_o(dmatrix<double_complex> &h__, dmatrix<double_complex> &o__) const 
{
    PROFILE("sirius::Hamiltonian_k::set_fv_h_o");

    /* alias to unit cell */
    auto& uc = H0_.ctx().unit_cell();
    /* alias to k-point */
    auto& kp = this->kp();
    /* split atoms in blocks */
    int num_atoms_in_block = 2 * omp_get_max_threads();
    int nblk = uc.num_atoms() / num_atoms_in_block + std::min(1, uc.num_atoms() % num_atoms_in_block);
    /* maximum number of apw coefficients in the block of atoms */
    int max_mt_aw = num_atoms_in_block * uc.max_mt_aw_basis_size();
    /* current processing unit */
    auto pu = H0_.ctx().processing_unit();

    mdarray<double_complex, 3> alm_row;
    mdarray<double_complex, 3> alm_col;
    mdarray<double_complex, 3> halm_col;

    h__.zero();
    o__.zero();
    switch (pu) { // TODO: replace with allocations from memory pool
        case device_t::GPU: {
            alm_row = mdarray<double_complex, 3>(kp.num_gkvec_row(), max_mt_aw, 2, memory_t::host_pinned);
            alm_col = mdarray<double_complex, 3>(kp.num_gkvec_col(), max_mt_aw, 2, memory_t::host_pinned);
            halm_col = mdarray<double_complex, 3>(kp.num_gkvec_col(), max_mt_aw, 2, memory_t::host_pinned);
            alm_row.allocate(memory_t::device);
            alm_col.allocate(memory_t::device);
            halm_col.allocate(memory_t::device);
            h__.allocate(memory_t::device).zero(memory_t::device);
            o__.allocate(memory_t::device).zero(memory_t::device);
            break;
        }
        case device_t::CPU: {
            alm_row = mdarray<double_complex, 3>(kp.num_gkvec_row(), max_mt_aw, 1);
            alm_col = mdarray<double_complex, 3>(kp.num_gkvec_col(), max_mt_aw, 1);
            halm_col = mdarray<double_complex, 3>(kp.num_gkvec_col(), max_mt_aw, 1);
            break;
        }
    }

    /* offsets for matching coefficients of individual atoms in the AW block */
    std::vector<int> offsets(uc.num_atoms());

    utils::timer t1("sirius::Hamiltonian::set_fv_h_o|zgemm");
    /* loop over blocks of atoms */
    for (int iblk = 0; iblk < nblk; iblk++) {
        /* number of matching AW coefficients in the block */
        int num_mt_aw{0};
        int ia_begin = iblk * num_atoms_in_block;
        int ia_end = std::min(uc.num_atoms(), (iblk + 1) * num_atoms_in_block);
        for (int ia = ia_begin; ia < ia_end; ia++) {
            offsets[ia] = num_mt_aw;
            num_mt_aw += uc.atom(ia).type().mt_aw_basis_size();
        }

        int s = (pu == device_t::GPU) ? (iblk % 2) : 0;

        if (H0_.ctx().control().print_checksum_) {
            alm_row.zero();
            alm_col.zero();
            halm_col.zero();
        }

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp for
            for (int ia = ia_begin; ia < ia_end; ia++) {
                auto& atom = uc.atom(ia);
                auto& type = atom.type();
                int naw = type.mt_aw_basis_size();

                mdarray<double_complex, 2> alm_row_atom;
                mdarray<double_complex, 2> alm_col_atom;
                mdarray<double_complex, 2> halm_col_atom;

                switch (pu) {
                    case device_t::CPU: {
                        alm_row_atom = mdarray<double_complex, 2>(alm_row.at(memory_t::host, 0, offsets[ia], s),
                                                                  kp.num_gkvec_row(), naw);

                        alm_col_atom = mdarray<double_complex, 2>(alm_col.at(memory_t::host, 0, offsets[ia], s),
                                                                  kp.num_gkvec_col(), naw);

                        halm_col_atom = mdarray<double_complex, 2>(halm_col.at(memory_t::host, 0, offsets[ia], s),
                                                                   kp.num_gkvec_col(), naw);
                        break;
                    }
                    case device_t::GPU: {
                        alm_row_atom = mdarray<double_complex, 2>(alm_row.at(memory_t::host, 0, offsets[ia], s),
                                                                  alm_row.at(memory_t::device, 0, offsets[ia], s),
                                                                  kp.num_gkvec_row(), naw);

                        alm_col_atom = mdarray<double_complex, 2>(alm_col.at(memory_t::host, 0, offsets[ia], s),
                                                                  alm_col.at(memory_t::device, 0, offsets[ia], s),
                                                                  kp.num_gkvec_col(), naw);

                        halm_col_atom = mdarray<double_complex, 2>(halm_col.at(memory_t::host, 0, offsets[ia], s),
                                                                   halm_col.at(memory_t::device, 0, offsets[ia], s),
                                                                   kp.num_gkvec_col(), naw);
                        break;
                    }
                }

                kp.alm_coeffs_col().generate<false>(atom, alm_col_atom);
                /* can't copy alm to device how as it might be modified by the iora */

                H0_.apply_hmt_to_apw<spin_block_t::nm>(atom, kp.num_gkvec_col(), alm_col_atom, halm_col_atom);
                if (pu == device_t::GPU) {
                    halm_col_atom.copy_to(memory_t::device, stream_id(tid));
                }

                kp.alm_coeffs_row().generate<true>(atom, alm_row_atom);
                if (pu == device_t::GPU) {
                     alm_row_atom.copy_to(memory_t::device, stream_id(tid));
                }

                /* setup apw-lo and lo-apw blocks */
                set_fv_h_o_apw_lo(atom, ia, alm_row_atom, alm_col_atom, h__, o__);

                /* finally, modify alm coefficients for iora */
                if (H0_.ctx().valence_relativity() == relativity_t::iora) {
                    // TODO: check if we can modify alm_col with IORA eralier and then not apply it in set_fv_h_o_apw_lo()
                    H0_.add_o1mt_to_apw(atom, kp.num_gkvec_col(), alm_col_atom);
                }

                if (pu == device_t::GPU) {
                    alm_col_atom.copy_to(memory_t::device, stream_id(tid));
                }
            }
            acc::sync_stream(stream_id(tid));
        }
        acc::sync_stream(stream_id(omp_get_max_threads()));

        if (H0_.ctx().control().print_checksum_) {
            double_complex z1 = alm_row.checksum();
            double_complex z2 = alm_col.checksum();
            double_complex z3 = halm_col.checksum();
            utils::print_checksum("alm_row", z1);
            utils::print_checksum("alm_col", z2);
            utils::print_checksum("halm_col", z3);
        }
        switch (pu) {
            case device_t::CPU: {
                linalg<device_t::CPU>::gemm(0, 1, kp.num_gkvec_row(), kp.num_gkvec_col(), num_mt_aw,
                                            linalg_const<double_complex>::one(),
                                            alm_row.at(memory_t::host), alm_row.ld(),
                                            alm_col.at(memory_t::host), alm_col.ld(),
                                            linalg_const<double_complex>::one(),
                                            o__.at(memory_t::host), o__.ld());

                linalg<device_t::CPU>::gemm(0, 1, kp.num_gkvec_row(), kp.num_gkvec_col(), num_mt_aw,
                                            linalg_const<double_complex>::one(),
                                            alm_row.at(memory_t::host), alm_row.ld(),
                                            halm_col.at(memory_t::host), halm_col.ld(),
                                            linalg_const<double_complex>::one(),
                                            h__.at(memory_t::host), h__.ld());
                break;
            }
            case device_t::GPU: {
#if defined(__GPU)
                linalg<device_t::GPU>::gemm(0, 1, kp.num_gkvec_row(), kp.num_gkvec_col(), num_mt_aw,
                                           &linalg_const<double_complex>::one(),
                                           alm_row.at(memory_t::device, 0, 0, s), alm_row.ld(),
                                           alm_col.at(memory_t::device, 0, 0, s), alm_col.ld(),
                                           &linalg_const<double_complex>::one(),
                                           o__.at(memory_t::device), o__.ld(), omp_get_max_threads());

                linalg<device_t::GPU>::gemm(0, 1, kp.num_gkvec_row(), kp.num_gkvec_col(), num_mt_aw,
                                           &linalg_const<double_complex>::one(),
                                           alm_row.at(memory_t::device, 0, 0, s), alm_row.ld(),
                                           halm_col.at(memory_t::device, 0, 0, s), halm_col.ld(),
                                           &linalg_const<double_complex>::one(),
                                           h__.at(memory_t::device), h__.ld());
#endif
                break;
            }
        }
    }
    if (pu == device_t::GPU) {
        acc::copyout(h__.at(memory_t::host), h__.ld(), h__.at(memory_t::device), h__.ld(), kp.num_gkvec_row(), kp.num_gkvec_col());
        acc::copyout(o__.at(memory_t::host), o__.ld(), o__.at(memory_t::device), o__.ld(), kp.num_gkvec_row(), kp.num_gkvec_col());
        h__.deallocate(memory_t::device);
        o__.deallocate(memory_t::device);
    }
    double tval = t1.stop();
    if (kp.comm().rank() == 0 && H0_.ctx().control().print_performance_) {
        kp.message(1, __func__, "effective zgemm performance: %12.6f GFlops",
               2 * 8e-9 * kp.num_gkvec() * kp.num_gkvec() * uc.mt_aw_basis_size() / tval);
    }

    /* add interstitial contributon */
    set_fv_h_o_it(h__, o__);

    /* setup lo-lo block */
    set_fv_h_o_lo_lo(h__, o__);
}

/* alm_row comes in already conjugated */
void Hamiltonian_k::set_fv_h_o_apw_lo(Atom const& atom__, int ia__, mdarray<double_complex, 2>& alm_row__,
                                      mdarray<double_complex, 2>& alm_col__, mdarray<double_complex, 2>& h__,
                                      mdarray<double_complex, 2>& o__) const
{
    auto& type = atom__.type();
    /* apw-lo block */
    for (int i = 0; i < kp().num_atom_lo_cols(ia__); i++) {
        int icol = kp().lo_col(ia__, i);
        /* local orbital indices */
        int l = kp().lo_basis_descriptor_col(icol).l;
        int lm = kp().lo_basis_descriptor_col(icol).lm;
        int idxrf = kp().lo_basis_descriptor_col(icol).idxrf;
        int order = kp().lo_basis_descriptor_col(icol).order;
        /* loop over apw components and update H */
        for (int j1 = 0; j1 < type.mt_aw_basis_size(); j1++) {
            int lm1 = type.indexb(j1).lm;
            int idxrf1 = type.indexb(j1).idxrf;

            auto zsum = atom__.radial_integrals_sum_L3<spin_block_t::nm>(idxrf, idxrf1,
                H0_.gaunt_coefs().gaunt_vector(lm1, lm));

            if (std::abs(zsum) > 1e-14) {
                for (int igkloc = 0; igkloc < kp().num_gkvec_row(); igkloc++) {
                    h__(igkloc, kp().num_gkvec_col() + icol) += zsum * alm_row__(igkloc, j1);
                }
            }
        }
        /* update O */
        for (int order1 = 0; order1 < type.aw_order(l); order1++) {
            int xi1 = type.indexb().index_by_lm_order(lm, order1);
            double ori = atom__.symmetry_class().o_radial_integral(l, order1, order);
            if (H0_.ctx().valence_relativity() == relativity_t::iora) {
                int idxrf1 = type.indexr().index_by_l_order(l, order1);
                ori += atom__.symmetry_class().o1_radial_integral(idxrf1, idxrf);
            }

            for (int igkloc = 0; igkloc < kp().num_gkvec_row(); igkloc++) {
                o__(igkloc, kp().num_gkvec_col() + icol) += ori * alm_row__(igkloc, xi1);
            }
        }
    }

    std::vector<double_complex> ztmp(kp().num_gkvec_col());
    /* lo-apw block */
    for (int i = 0; i < kp().num_atom_lo_rows(ia__); i++) {
        int irow = kp().lo_row(ia__, i);
        /* local orbital indices */
        int l = kp().lo_basis_descriptor_row(irow).l;
        int lm = kp().lo_basis_descriptor_row(irow).lm;
        int idxrf = kp().lo_basis_descriptor_row(irow).idxrf;
        int order = kp().lo_basis_descriptor_row(irow).order;

        std::fill(ztmp.begin(), ztmp.end(), 0);

        /* loop over apw components */
        for (int j1 = 0; j1 < type.mt_aw_basis_size(); j1++) {
            int lm1 = type.indexb(j1).lm;
            int idxrf1 = type.indexb(j1).idxrf;

            auto zsum = atom__.radial_integrals_sum_L3<spin_block_t::nm>(idxrf1, idxrf,
                H0_.gaunt_coefs().gaunt_vector(lm, lm1));

            if (std::abs(zsum) > 1e-14) {
                for (int igkloc = 0; igkloc < kp().num_gkvec_col(); igkloc++) {
                    ztmp[igkloc] += zsum * alm_col__(igkloc, j1);
                }
            }
        }

        for (int igkloc = 0; igkloc < kp().num_gkvec_col(); igkloc++) {
            h__(irow + kp().num_gkvec_row(), igkloc) += ztmp[igkloc];
        }

        for (int order1 = 0; order1 < type.aw_order(l); order1++) {
            int xi1 = type.indexb().index_by_lm_order(lm, order1);
            double ori = atom__.symmetry_class().o_radial_integral(l, order, order1);
            if (H0_.ctx().valence_relativity() == relativity_t::iora) {
                int idxrf1 = type.indexr().index_by_l_order(l, order1);
                ori += atom__.symmetry_class().o1_radial_integral(idxrf, idxrf1);
            }

            for (int igkloc = 0; igkloc < kp().num_gkvec_col(); igkloc++) {
                o__(irow + kp().num_gkvec_row(), igkloc) += ori * alm_col__(igkloc, xi1);
            }
        }
    }
}

void Hamiltonian_k::set_fv_h_o_lo_lo(dmatrix<double_complex>& h__, dmatrix<double_complex>& o__) const 
{
    PROFILE("sirius::Hamiltonian_k::set_fv_h_o_lo_lo");

    auto& kp = this->kp();

    /* lo-lo block */
    #pragma omp parallel for default(shared)
    for (int icol = 0; icol < kp.num_lo_col(); icol++) {
        int ia = kp.lo_basis_descriptor_col(icol).ia;
        int lm2 = kp.lo_basis_descriptor_col(icol).lm;
        int idxrf2 = kp.lo_basis_descriptor_col(icol).idxrf;

        for (int irow = 0; irow < kp.num_lo_row(); irow++) {
            /* lo-lo block is diagonal in atom index */
            if (ia == kp.lo_basis_descriptor_row(irow).ia) {
                auto& atom = H0_.ctx().unit_cell().atom(ia);
                int lm1 = kp.lo_basis_descriptor_row(irow).lm;
                int idxrf1 = kp.lo_basis_descriptor_row(irow).idxrf;

                h__(kp.num_gkvec_row() + irow, kp.num_gkvec_col() + icol) +=
                    atom.radial_integrals_sum_L3<spin_block_t::nm>(idxrf1, idxrf2,
                        H0_.gaunt_coefs().gaunt_vector(lm1, lm2));

                if (lm1 == lm2) {
                    int l = kp.lo_basis_descriptor_row(irow).l;
                    int order1 = kp.lo_basis_descriptor_row(irow).order;
                    int order2 = kp.lo_basis_descriptor_col(icol).order;
                    o__(kp.num_gkvec_row() + irow, kp.num_gkvec_col() + icol) +=
                            atom.symmetry_class().o_radial_integral(l, order1, order2);
                    if (H0_.ctx().valence_relativity() == relativity_t::iora) {
                        int idxrf1 = atom.type().indexr().index_by_l_order(l, order1);
                        int idxrf2 = atom.type().indexr().index_by_l_order(l, order2);
                        o__(kp.num_gkvec_row() + irow, kp.num_gkvec_col() + icol) +=
                            atom.symmetry_class().o1_radial_integral(idxrf1, idxrf2);
                    }
                }
            }
        }
    }
}

void Hamiltonian_k::set_fv_h_o_it(dmatrix<double_complex>& h__, dmatrix<double_complex>& o__) const 
{
    PROFILE("sirius::Hamiltonian_k::set_fv_h_o_it");

    double sq_alpha_half = 0.5 * std::pow(speed_of_light, -2);

    auto& kp = this->kp();

    #pragma omp parallel for default(shared)
    for (int igk_col = 0; igk_col < kp.num_gkvec_col(); igk_col++) {
        int ig_col = kp.igk_col(igk_col);
        /* fractional coordinates of G vectors */
        auto gvec_col = kp.gkvec().gvec(ig_col);
        /* Cartesian coordinates of G+k vectors */
        auto gkvec_col_cart = kp.gkvec().gkvec_cart<index_domain_t::global>(ig_col);
        for (int igk_row = 0; igk_row < kp.num_gkvec_row(); igk_row++) {
            int ig_row = kp.igk_row(igk_row);
            auto gvec_row = kp.gkvec().gvec(ig_row);
            auto gkvec_row_cart = kp.gkvec().gkvec_cart<index_domain_t::global>(ig_row);
            int ig12 = H0().ctx().gvec().index_g12(gvec_row, gvec_col);
            /* pw kinetic energy */
            double t1 = 0.5 * geometry3d::dot(gkvec_row_cart, gkvec_col_cart);

            h__(igk_row, igk_col) += H0().potential().veff_pw(ig12);
            o__(igk_row, igk_col) += H0().ctx().theta_pw(ig12);

            if (H0().ctx().valence_relativity() == relativity_t::none) {
                h__(igk_row, igk_col) += t1 * H0().ctx().theta_pw(ig12);
            }
            if (H0().ctx().valence_relativity() == relativity_t::zora) {
                h__(igk_row, igk_col) += t1 * H0().potential().rm_inv_pw(ig12);
            }
            if (H0().ctx().valence_relativity() == relativity_t::iora) {
                h__(igk_row, igk_col) += t1 * H0().potential().rm_inv_pw(ig12);
                o__(igk_row, igk_col) += t1 * sq_alpha_half * H0().potential().rm2_inv_pw(ig12);
            }
        }
    }
}

}
