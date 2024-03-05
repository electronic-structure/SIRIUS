// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
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

/** \file gvec.cpp
 *
 *  \brief Contains the implementation of Gvec class.
 *
 */

#include "symmetry/lattice.hpp"
#include "gvec.hpp"
#include "core/serializer.hpp"

namespace sirius {

namespace fft {

r3::vector<int>
Gvec::gvec_by_full_index(uint32_t idx__) const
{
    /* index of the z coordinate of G-vector: first 12 bits */
    uint32_t j = idx__ & 0xFFF;
    /* index of z-column: last 20 bits */
    uint32_t i = idx__ >> 12;
    RTE_ASSERT(i < (uint32_t)z_columns_.size());
    RTE_ASSERT(j < (uint32_t)z_columns_[i].z.size());
    int x = z_columns_[i].x;
    int y = z_columns_[i].y;
    int z = z_columns_[i].z[j];
    return r3::vector<int>(x, y, z);
}

void
Gvec::find_z_columns(double Gmax__, fft::Grid const& fft_box__)
{
    PROFILE("fft::Gvec::find_z_columns");

    mdarray<int, 2> non_zero_columns({index_range(fft_box__.limits(0).first, fft_box__.limits(0).second + 1),
                                      index_range(fft_box__.limits(1).first, fft_box__.limits(1).second + 1)});
    non_zero_columns.zero();

    num_gvec_ = 0;

    auto add_new_column = [&](int i, int j) {
        if (non_zero_columns(i, j)) {
            return;
        }

        std::vector<int> zcol;

        /* in general case take z in [0, Nz) */
        int zmax = fft_box__[2] - 1;
        /* in case of G-vector reduction take z in [0, Nz/2] for {x=0,y=0} stick */
        if (reduce_gvec_ && !i && !j) {
            zmax = fft_box__.limits(2).second;
        }
        /* loop over z-coordinates of FFT grid */
        for (int iz = 0; iz <= zmax; iz++) {
            /* get z-coordinate of G-vector */
            int k = fft_box__.freq_by_coord<2>(iz);
            /* take G+k */
            auto vgk = r3::dot(lattice_vectors_, (r3::vector<double>(i, j, k) + vk_));
            /* add z-coordinate of G-vector to the list */
            if (vgk.length() <= Gmax__) {
                zcol.push_back(k);
            }
        }

        /* add column to the list */
        if (zcol.size()) {
            z_columns_.push_back(z_column_descriptor(i, j, zcol));
            num_gvec_ += static_cast<int>(zcol.size());

            non_zero_columns(i, j) = 1;
            if (reduce_gvec_) {
                int mi = -i;
                int mj = -j;
                if (mi >= fft_box__.limits(0).first && mi <= fft_box__.limits(0).second &&
                    mj >= fft_box__.limits(1).first && mj <= fft_box__.limits(1).second) {
                    non_zero_columns(mi, mj) = 1;
                }
            }
        }
    };

    PROFILE_START("fft::Gvec::find_z_columns|add");
    /* copy column order from previous G-vector set */
    if (gvec_base_) {
        for (int icol = 0; icol < gvec_base_->num_zcol(); icol++) {
            int i = gvec_base_->zcol(icol).x;
            int j = gvec_base_->zcol(icol).y;
            add_new_column(i, j);
        }
    }

    /* check all z-columns and add if within sphere. Only allow non-negative x-indices for reduced case */
    for (int i = reduce_gvec_ ? 0 : fft_box__.limits(0).first; i <= fft_box__.limits(0).second; i++) {
        for (int j = fft_box__.limits(1).first; j <= fft_box__.limits(1).second; j++) {
            add_new_column(i, j);
        }
    }
    PROFILE_STOP("fft::Gvec::find_z_columns|add");

    if (!gvec_base_) {
        /* put column with {x, y} = {0, 0} to the beginning */
        for (size_t i = 0; i < z_columns_.size(); i++) {
            if (z_columns_[i].x == 0 && z_columns_[i].y == 0) {
                std::swap(z_columns_[i], z_columns_[0]);
                break;
            }
        }
    }

    PROFILE_START("fft::Gvec::find_z_columns|sym");
    /* now we have to remove edge G-vectors that don't form a complete shell */
    if (bare_gvec_) {
        auto lat_sym = sirius::find_lat_sym(this->unit_cell_lattice_vectors(), this->sym_tol_);

        std::fill(non_zero_columns.at(memory_t::host), non_zero_columns.at(memory_t::host) + non_zero_columns.size(),
                  -1);
        for (int i = 0; i < static_cast<int>(z_columns_.size()); i++) {
            non_zero_columns(z_columns_[i].x, z_columns_[i].y) = i;
        }

        std::vector<z_column_descriptor> z_columns_tmp;

        auto remove_incomplete = [this, &z_columns_tmp, &lat_sym, &non_zero_columns](int i) // i - index of column
        {
            int z_min = (2 << 20);
            int z_max = -(2 << 20);
            for (int iz = 0; iz < static_cast<int>(z_columns_[i].z.size()); iz++) {
                z_min = std::min(z_min, z_columns_[i].z[iz]);
                z_max = std::max(z_max, z_columns_[i].z[iz]);
            }
            std::vector<int> z;
            for (int iz = 0; iz < static_cast<int>(z_columns_[i].z.size()); iz++) {
                bool found_for_all_sym{true};
                /* check only first or last z coordinate inside z column */
                if (z_columns_[i].z[iz] == z_min || z_columns_[i].z[iz] == z_max) {
                    r3::vector<int> G(z_columns_[i].x, z_columns_[i].y, z_columns_[i].z[iz]);
                    for (auto& R : lat_sym) {
                        /* apply lattice symmeetry operation to a G-vector */
                        auto G1 = r3::dot(G, R);
                        if (reduce_gvec_) {
                            if (G1[0] == 0 && G1[1] == 0) {
                                G1[2] = std::abs(G1[2]);
                            }
                        }
                        int i1 = non_zero_columns(G1[0], G1[1]);
                        if (i1 == -1) {
                            G1 = G1 * (-1);
                            i1 = non_zero_columns(G1[0], G1[1]);
                            if (i1 == -1) {
                                std::stringstream s;
                                s << "index of z-column is not found" << std::endl
                                  << "  G : " << G << std::endl
                                  << "  G1 : " << G1;
                                RTE_THROW(s);
                            }
                        }

                        bool found        = (std::find(z_columns_[i1].z.begin(), z_columns_[i1].z.end(), G1[2]) !=
                                      std::end(z_columns_[i1].z));
                        found_for_all_sym = found_for_all_sym && found;
                    } // R
                }
                if (found_for_all_sym) {
                    z.push_back(z_columns_[i].z[iz]);
                }
            } // iz
            if (z.size()) {
                z_columns_tmp.push_back(z_column_descriptor(z_columns_[i].x, z_columns_[i].y, z));
            }
        };

        for (int i = 0; i < static_cast<int>(z_columns_.size()); i++) {
            remove_incomplete(i);
        }
        z_columns_ = z_columns_tmp;

        num_gvec_ = 0;
        for (auto& zc : z_columns_) {
            num_gvec_ += static_cast<int>(zc.z.size());
        }
    }
    PROFILE_STOP("fft::Gvec::find_z_columns|sym");

    PROFILE_START("fft::Gvec::find_z_columns|sort");
    /* sort z-columns starting from the second or skip num_zcol of base distribution */
    int n = (gvec_base_) ? gvec_base_->num_zcol() : 1;
    std::sort(z_columns_.begin() + n, z_columns_.end(),
              [](z_column_descriptor const& a, z_column_descriptor const& b) { return a.z.size() > b.z.size(); });
    PROFILE_STOP("fft::Gvec::find_z_columns|sort");
}

void
Gvec::distribute_z_columns()
{
    gvec_distr_ = mpi::block_data_descriptor(comm().size());
    zcol_distr_ = mpi::block_data_descriptor(comm().size());
    /* local number of z-columns for each rank */
    std::vector<std::vector<z_column_descriptor>> zcols_local(comm().size());

    /* use already existing distribution of base G-vector set */
    if (gvec_base_) {
        for (int rank = 0; rank < comm().size(); rank++) {
            for (int i = 0; i < gvec_base_->zcol_count(rank); i++) {
                int icol = gvec_base_->zcol_offset(rank) + i;
                /* assign column to the found rank */
                zcols_local[rank].push_back(z_columns_[icol]);
                /* count local number of z-columns */
                zcol_distr_.counts[rank] += 1;
                /* count local number of G-vectors */
                gvec_distr_.counts[rank] += static_cast<int>(z_columns_[icol].z.size());
            }
        }
    }

    int n = (gvec_base_) ? gvec_base_->num_zcol() : 0;

    std::vector<int> ranks;
    for (int i = n; i < static_cast<int>(z_columns_.size()); i++) {
        /* initialize the list of ranks to 0,1,2,... */
        if (ranks.empty()) {
            ranks.resize(comm().size());
            std::iota(ranks.begin(), ranks.end(), 0);
        }
        /* find rank with minimum number of G-vectors */
        auto rank_with_min_gvec = std::min_element(ranks.begin(), ranks.end(), [this](const int& a, const int& b) {
            return gvec_distr_.counts[a] < gvec_distr_.counts[b];
        });

        /* assign column to the found rank */
        zcols_local[*rank_with_min_gvec].push_back(z_columns_[i]);
        /* count local number of z-columns */
        zcol_distr_.counts[*rank_with_min_gvec] += 1;
        /* count local number of G-vectors */
        gvec_distr_.counts[*rank_with_min_gvec] += static_cast<int>(z_columns_[i].z.size());
        /* exclude this rank from the search */
        ranks.erase(rank_with_min_gvec);
    }
    gvec_distr_.calc_offsets();
    zcol_distr_.calc_offsets();

    /* save new ordering of z-columns */
    z_columns_.clear();
    for (int rank = 0; rank < comm().size(); rank++) {
        z_columns_.insert(z_columns_.end(), zcols_local[rank].begin(), zcols_local[rank].end());
    }

    /* sanity check */
    int ng{0};
    for (int rank = 0; rank < comm().size(); rank++) {
        ng += gvec_distr_.counts[rank];
    }
    if (ng != num_gvec_) {
        RTE_THROW("wrong number of G-vectors");
    }
    this->offset_         = this->offset(this->comm().rank());
    this->count_          = this->count(this->comm().rank());
    this->num_zcol_local_ = this->zcol_distr_.counts[this->comm().rank()];
}

void
Gvec::find_gvec_shells()
{
    if (!bare_gvec_) {
        return;
    }

    PROFILE("fft::Gvec::find_gvec_shells");

    auto lat_sym = sirius::find_lat_sym(this->unit_cell_lattice_vectors(), this->sym_tol_);

    num_gvec_shells_ = 0;
    gvec_shell_      = mdarray<int, 1>({num_gvec_}, mdarray_label("gvec_shell_"));

    std::fill(&gvec_shell_[0], &gvec_shell_[0] + num_gvec_, -1);

    /* find G-vector shells using symmetry consideration */
    for (int ig = 0; ig < num_gvec_; ig++) {
        /* if the shell for this vector is not yet found */
        if (gvec_shell_[ig] == -1) {
            auto G = gvec(gvec_index_t::global(ig));
            for (auto& R : lat_sym) {
                auto G1  = r3::dot(G, R);
                auto ig1 = index_by_gvec(G1);
                if (ig1 == -1) {
                    G1  = G1 * (-1);
                    ig1 = index_by_gvec(G1);
                    if (ig1 == -1) {
                        RTE_THROW("symmetry-related G-vector is not found");
                    }
                }
                if (gvec_shell_[ig1] == -1) {
                    gvec_shell_[ig1] = num_gvec_shells_;
                } else {
                    if (gvec_shell_[ig1] != num_gvec_shells_) {
                        auto gc  = r3::dot(lattice_vectors_, G);
                        auto gc1 = r3::dot(lattice_vectors_, G1);
                        std::stringstream s;
                        s << "Error in G-vector shell index" << std::endl
                          << "  G : " << G << std::endl
                          << "  rotated G : " << G1 << std::endl
                          << "  current shell index : " << num_gvec_shells_ << std::endl
                          << "  rotated G shell index : " << gvec_shell_[ig1] << std::endl
                          << "  length of G : " << gc.length() << std::endl
                          << "  length of rotated G : " << gc1.length() << std::endl
                          << "  length difference : " << std::abs(gc.length() - gc1.length());
                        RTE_THROW(s);
                    }
                }
            }
            num_gvec_shells_++;
        }
    }
    for (int ig = 0; ig < num_gvec_; ig++) {
        if (gvec_shell_[ig] == -1) {
            RTE_THROW("wrong G-vector shell");
        }
    }

    gvec_shell_len_ = mdarray<double, 1>({num_gvec_shells_}, mdarray_label("gvec_shell_len_"));
    std::fill(&gvec_shell_len_[0], &gvec_shell_len_[0] + num_gvec_shells_, 0);

    std::vector<int> ngv_sh(num_gvec_shells_, 0);

    for (int ig = 0; ig < num_gvec_; ig++) {
        auto g   = gvec_cart(gvec_index_t::global(ig)).length();
        int igsh = gvec_shell_[ig];
        gvec_shell_len_[igsh] += g;
        ngv_sh[igsh]++;
    }
    for (int i = 0; i < num_gvec_shells_; i++) {
        gvec_shell_len_[i] /= ngv_sh[i];
    }

    /* map from global index of G-shell to a list of local G-vectors */
    std::map<int, std::vector<int>> gshmap;
    for (int igloc = 0; igloc < this->count(); igloc++) {
        int igsh = this->shell(this->offset() + igloc);
        if (gshmap.count(igsh) == 0) {
            gshmap[igsh] = std::vector<int>();
        }
        gshmap[igsh].push_back(igloc);
    }

    num_gvec_shells_local_ = 0;
    gvec_shell_idx_local_.resize(this->count());
    gvec_shell_len_local_.clear();
    for (auto it = gshmap.begin(); it != gshmap.end(); ++it) {
        int igsh = it->first;
        gvec_shell_len_local_.push_back(this->shell_len(igsh));
        for (auto igloc : it->second) {
            gvec_shell_idx_local_[igloc] = num_gvec_shells_local_;
        }
        num_gvec_shells_local_++;
    }
}

void
Gvec::init_gvec_local()
{
    gvec_  = mdarray<int, 2>({3, count()}, mdarray_label("gvec_"));
    gkvec_ = mdarray<double, 2>({3, count()}, mdarray_label("gkvec_"));

    for (int igloc = 0; igloc < count(); igloc++) {
        int ig = offset() + igloc;
        auto G = gvec_by_full_index(gvec_full_index_(ig));
        for (int x : {0, 1, 2}) {
            gvec_(x, igloc)  = G[x];
            gkvec_(x, igloc) = G[x] + vk_[x];
        }
    }
}

void
Gvec::init_gvec_cart_local()
{
    gvec_cart_  = mdarray<double, 2>({3, count()}, mdarray_label("gvec_cart_"));
    gkvec_cart_ = mdarray<double, 2>({3, count()}, mdarray_label("gkvec_cart_"));
    /* this arrays are allocated with GPU- friendly data layout */
    gvec_tp_  = mdarray<double, 2>({count(), 2}, mdarray_label("gvec_tp_"));
    gkvec_tp_ = mdarray<double, 2>({count(), 2}, mdarray_label("gvec_tp_"));
    if (bare_gvec_) {
        gvec_len_ = mdarray<double, 1>({count()}, mdarray_label("gvec_len_"));
    }

    for (int igloc = 0; igloc < count(); igloc++) {
        auto gc  = r3::dot(lattice_vectors_, r3::vector<int>(&gvec_(0, igloc)));
        auto gkc = r3::dot(lattice_vectors_, r3::vector<double>(&gkvec_(0, igloc)));
        for (int x : {0, 1, 2}) {
            gvec_cart_(x, igloc)  = gc[x];
            gkvec_cart_(x, igloc) = gkc[x];
        }
        if (bare_gvec_) {
            gvec_len_(igloc) = gvec_shell_len_(gvec_shell_(this->offset() + igloc));
        }
        auto gs            = r3::spherical_coordinates(gc);
        gvec_tp_(igloc, 0) = gs[1];
        gvec_tp_(igloc, 1) = gs[2];

        auto gks            = r3::spherical_coordinates(gkc);
        gkvec_tp_(igloc, 0) = gks[1];
        gkvec_tp_(igloc, 1) = gks[2];
    }
}

void
Gvec::init(fft::Grid const& fft_grid)
{
    PROFILE("fft::Gvec::init");

    find_z_columns(Gmax_, fft_grid);

    distribute_z_columns();

    gvec_index_by_xy_ = mdarray<int, 3>({2, index_range(fft_grid.limits(0).first, fft_grid.limits(0).second + 1),
                                         index_range(fft_grid.limits(1).first, fft_grid.limits(1).second + 1)},
                                        mdarray_label("Gvec.gvec_index_by_xy_"));
    std::fill(gvec_index_by_xy_.at(memory_t::host), gvec_index_by_xy_.at(memory_t::host) + gvec_index_by_xy_.size(),
              -1);

    /* build the full G-vector index and reverse mapping */
    gvec_full_index_ = mdarray<uint32_t, 1>({num_gvec_});
    int ig{0};
    for (size_t i = 0; i < z_columns_.size(); i++) {
        /* starting G-vector index for a z-stick */
        gvec_index_by_xy_(0, z_columns_[i].x, z_columns_[i].y) = ig;
        /* pack size of a z-stick and column index in one number */
        gvec_index_by_xy_(1, z_columns_[i].x, z_columns_[i].y) = static_cast<int>((z_columns_[i].z.size() << 20) + i);
        for (size_t j = 0; j < z_columns_[i].z.size(); j++) {
            gvec_full_index_[ig++] = static_cast<uint32_t>((i << 12) + j);
        }
    }
    if (ig != num_gvec_) {
        RTE_THROW("wrong G-vector count");
    }

    for (int ig = 0; ig < num_gvec_; ig++) {
        auto gv = gvec(gvec_index_t::global(ig));
        if (index_by_gvec(gv) != ig) {
            std::stringstream s;
            s << "wrong G-vector index: ig=" << ig << " gv=" << gv << " index_by_gvec(gv)=" << index_by_gvec(gv);
            RTE_THROW(s);
        }
    }

    /* first G-vector must be (0, 0, 0); never remove this check!!! */
    auto g0 = gvec_by_full_index(gvec_full_index_(0));
    if (g0[0] || g0[1] || g0[2]) {
        RTE_THROW("first G-vector is not zero");
    }

    find_gvec_shells();

    if (gvec_base_) {
        /* the size of the mapping is equal to the local number of G-vectors in the base set */
        gvec_base_mapping_ = mdarray<int, 1>({gvec_base_->count()}, mdarray_label("gvec_base_mapping_"));
        /* loop over local G-vectors of a base set */
        for (int igloc = 0; igloc < gvec_base_->count(); igloc++) {
            /* G-vector in lattice coordinates */
            auto G = gvec_base_->gvec(gvec_index_t::local(igloc));
            /* global index of G-vector in the current set */
            int ig = index_by_gvec(G);
            /* the same MPI rank must store this G-vector */
            ig -= offset();
            if (ig >= 0 && ig < count()) {
                gvec_base_mapping_(igloc) = ig;
            } else {
                std::stringstream s;
                s << "local G-vector index is not found" << std::endl
                  << " G-vector: " << G << std::endl
                  << " G-vector index in base distribution : " << gvec_base_->offset() + igloc << std::endl
                  << " G-vector index in base distribution (by G-vector): " << gvec_base_->index_by_gvec(G) << std::endl
                  << " G-vector index in new distribution : " << index_by_gvec(G) << std::endl
                  << " offset in G-vector index for this rank: " << offset() << std::endl
                  << " local number of G-vectors for this rank: " << count();
                RTE_THROW(s);
            }
        }
    }
    // TODO: add a check for gvec_base (there is already a test for this).
    init_gvec_local();
    init_gvec_cart_local();
}

std::pair<int, bool>
Gvec::index_g12_safe(r3::vector<int> const& g1__, r3::vector<int> const& g2__) const
{
    auto v  = g1__ - g2__;
    int idx = index_by_gvec(v);
    bool conj{false};
    if (idx < 0) {
        idx  = index_by_gvec(v * (-1));
        conj = true;
    }
    if (idx < 0 || idx >= num_gvec()) {
        std::stringstream s;
        s << "wrong index of G-G' vector" << std::endl
          << "  G: " << g1__ << std::endl
          << "  G': " << g2__ << std::endl
          << "  G - G': " << v << std::endl
          << " idx: " << idx;
        RTE_THROW(s);
    }
    return std::make_pair(idx, conj);
}

int
Gvec::index_by_gvec(r3::vector<int> const& G__) const
{
    /* reduced G-vector set does not have negative z for x=y=0 */
    if (reduced() && G__[0] == 0 && G__[1] == 0 && G__[2] < 0) {
        return -1;
    }
    int ig0 = gvec_index_by_xy_(0, G__[0], G__[1]);
    if (ig0 == -1) {
        return -1;
    }
    /* index of the column */
    int icol = gvec_index_by_xy_(1, G__[0], G__[1]) & 0xFFFFF;
    /* quick exit if z is out of bounds */
    if (G__[2] < z_columns_[icol].z_min || G__[2] > z_columns_[icol].z_max) {
        return -1;
    }
    /* size of the column */
    int col_size = gvec_index_by_xy_(1, G__[0], G__[1]) >> 20;

    /* three possible options for the z-column location

          frequency                ... -4, -3, -2, -1, 0, 1, 2, 3, 4 ...
       -----------------------------------------------------------------------------
          G-vec ordering
       #1 (all negative)           ___  0   1   2   3 __________________
       #2 (negative and positive)  ____________ 3   4  0  1  2 _________
       #3 (all positive)           _____________________  0  1  2  3 ___

       Remember how FFT frequencies are stored: first positive frequences, then negative in the reverse order

       subtract first z-coordinate in column from the current z-coordinate of G-vector: in case #1 or #3 this
       already gives a proper offset, in case #2 storage of FFT frequencies must be taken into account
    */
    int z0 = G__[2] - z_columns_[icol].z[0];
    /* calculate proper offset */
    int offs = (z0 >= 0) ? z0 : z0 + col_size;
    /* full index */
    int ig = ig0 + offs;
    RTE_ASSERT(ig >= 0 && ig < num_gvec());
    return ig;
}

Gvec
send_recv(mpi::Communicator const& comm__, Gvec const& gv_src__, int source__, int dest__)
{
    serializer s;

    if (comm__.rank() == source__) {
        ::sirius::fft::serialize(s, gv_src__);
    }

    s.send_recv(comm__, source__, dest__);

    Gvec gv(gv_src__.comm());

    if (comm__.rank() == dest__) {
        ::sirius::fft::deserialize(s, gv);
    }
    return gv;
}

void
Gvec_fft::build_fft_distr()
{
    /* calculate distribution of G-vectors and z-columns for the FFT communicator */
    gvec_distr_fft_ = mpi::block_data_descriptor(comm_fft().size());

    for (int rank = 0; rank < comm_fft().size(); rank++) {
        for (int i = 0; i < comm_ortho_fft().size(); i++) {
            /* fine-grained rank */
            int r = rank_map_(rank, i);
            gvec_distr_fft_.counts[rank] += gvec().count(r);
        }
    }
    for (int i = 0; i < comm_ortho_fft().size(); i++) {
        /* fine-grained rank */
        int r = rank_map_(comm_fft().rank(), i);
        num_zcol_local_ += gvec().zcol_count(r);
    }
    /* get offsets of G-vectors */
    gvec_distr_fft_.calc_offsets();
}

void
Gvec_fft::pile_gvec()
{
    /* build a table of {offset, count} values for G-vectors in the swapped distribution;
     * we are preparing to swap plane-wave coefficients from a default slab distribution to a FFT-friendly
     * distribution
     * +--------------+      +----+----+----+
     * |    :    :    |      |    |    |    |
     * +--------------+      |....|....|....|
     * |    :    :    |  ->  |    |    |    |
     * +--------------+      |....|....|....|
     * |    :    :    |      |    |    |    |
     * +--------------+      +----+----+----+
     *
     * i.e. we will make G-vector slabs more fat (pile-of-slabs) and at the same time reshulffle wave-functions
     * between columns of the 2D MPI grid */
    gvec_fft_slab_ = mpi::block_data_descriptor(comm_ortho_fft_.size());
    for (int i = 0; i < comm_ortho_fft_.size(); i++) {
        gvec_fft_slab_.counts[i] = gvec().count(rank_map_(comm_fft_.rank(), i));
    }
    gvec_fft_slab_.calc_offsets();

    RTE_ASSERT(gvec_fft_slab_.offsets.back() + gvec_fft_slab_.counts.back() ==
               gvec_distr_fft_.counts[comm_fft().rank()]);

    gvec_array_       = mdarray<int, 2>({3, this->count()});
    gkvec_cart_array_ = mdarray<double, 2>({3, this->count()});
    for (int i = 0; i < comm_ortho_fft_.size(); i++) {
        for (int j = 0; j < comm_fft_.size(); j++) {
            int r = rank_map_(j, i);
            /* get array of G-vectors of rank r */
            auto gv = this->gvec_.gvec_local(r);

            if (j == comm_fft_.rank()) {
                for (int ig = 0; ig < gvec_fft_slab_.counts[i]; ig++) {
                    for (int x : {0, 1, 2}) {
                        gvec_array_(x, gvec_fft_slab_.offsets[i] + ig) = gv(x, ig);
                    }
                }
            }
        }
    }
    update_gkvec_cart();
}

Gvec_fft::Gvec_fft(Gvec const& gvec__, mpi::Communicator const& comm_fft__, mpi::Communicator const& comm_ortho_fft__)
    : gvec_(gvec__)
    , comm_fft_(comm_fft__)
    , comm_ortho_fft_(comm_ortho_fft__)
{
    if (comm_fft_.size() * comm_ortho_fft_.size() != gvec_.comm().size()) {
        std::stringstream s;
        s << "wrong size of communicators" << std::endl
          << "  comm_fft_.size()       = " << comm_fft_.size() << std::endl
          << "  comm_ortho_fft_.size() = " << comm_ortho_fft_.size() << std::endl
          << "  gvec_.comm().size()    = " << gvec_.comm().size();
        RTE_THROW(s);
    }
    rank_map_ = mdarray<int, 2>({comm_fft_.size(), comm_ortho_fft_.size()});
    rank_map_.zero();
    /* get a global rank */
    rank_map_(comm_fft_.rank(), comm_ortho_fft_.rank()) = gvec_.comm().rank();
    gvec_.comm().allreduce(&rank_map_(0, 0), gvec_.comm().size());

    build_fft_distr();

    pile_gvec();
}

Gvec_shells::Gvec_shells(Gvec const& gvec__)
    : comm_(gvec__.comm())
    , gvec_(gvec__)
{
    PROFILE("fft::Gvec_shells");

    a2a_send_ = mpi::block_data_descriptor(comm_.size());
    a2a_recv_ = mpi::block_data_descriptor(comm_.size());

    /* split G-vector shells between ranks in cyclic order */
    spl_num_gsh_ = splindex_block_cyclic<>(gvec_.num_shells(), n_blocks(comm_.size()), block_id(comm_.rank()), 1);

    /* each rank sends a fraction of its local G-vectors to other ranks */
    /* count this fraction */
    for (int igloc = 0; igloc < gvec_.count(); igloc++) {
        int ig   = gvec_.offset() + igloc;
        int igsh = gvec_.shell(ig);
        a2a_send_.counts[spl_num_gsh_.location(igsh).ib]++;
    }
    a2a_send_.calc_offsets();
    /* sanity check: total number of elements to send is equal to the local number of G-vector */
    if (a2a_send_.size() != gvec_.count()) {
        RTE_THROW("wrong number of G-vectors");
    }
    /* count the number of elements to receive */
    for (int r = 0; r < comm_.size(); r++) {
        for (int igloc = 0; igloc < gvec_.count(r); igloc++) {
            /* index of G-vector in the original distribution */
            int ig = gvec_.offset(r) + igloc;
            /* index of the G-vector shell */
            int igsh = gvec_.shell(ig);
            if (spl_num_gsh_.location(igsh).ib == comm_.rank()) {
                a2a_recv_.counts[r]++;
            }
        }
    }
    a2a_recv_.calc_offsets();
    /* sanity check: sum of local sizes in the remapped order is equal to the total number of G-vectors */
    int ng = gvec_count_remapped();
    comm_.allreduce(&ng, 1);
    if (ng != gvec_.num_gvec()) {
        RTE_THROW("wrong number of G-vectors");
    }

    /* local set of G-vectors in the remapped order */
    gvec_remapped_       = mdarray<int, 2>({3, gvec_count_remapped()}, mdarray_label("gvec_remapped_"));
    gvec_shell_remapped_ = mdarray<int, 1>({gvec_count_remapped()}, mdarray_label("gvec_shell_remapped_"));
    std::vector<int> counts(comm_.size(), 0);
    for (int r = 0; r < comm_.size(); r++) {
        for (int igloc = 0; igloc < gvec_.count(r); igloc++) {
            int ig   = gvec_.offset(r) + igloc;
            int igsh = gvec_.shell(ig);
            auto G   = gvec_.gvec(gvec_index_t::global(ig));
            if (spl_num_gsh_.location(igsh).ib == comm_.rank()) {
                for (int x : {0, 1, 2}) {
                    gvec_remapped_(x, a2a_recv_.offsets[r] + counts[r]) = G[x];
                }
                gvec_shell_remapped_(a2a_recv_.offsets[r] + counts[r]) = igsh;
                counts[r]++;
            }
        }
    }
    for (int ig = 0; ig < gvec_count_remapped(); ig++) {
        idx_gvec_[gvec_remapped(ig)] = ig;
    }
    /* sanity check */
    for (int igloc = 0; igloc < this->gvec_count_remapped(); igloc++) {
        auto G = this->gvec_remapped(igloc);
        if (this->index_by_gvec(G) != igloc) {
            RTE_THROW("Wrong remapped index of G-vector");
        }
        int igsh = this->gvec_shell_remapped(igloc);
        if (igsh != this->gvec().shell(G)) {
            RTE_THROW("Wrong remapped shell of G-vector");
        }
    }
}

void
serialize(serializer& s__, Gvec const& gv__)
{
    serialize(s__, gv__.vk_);
    serialize(s__, gv__.Gmax_);
    serialize(s__, gv__.lattice_vectors_);
    serialize(s__, gv__.reduce_gvec_);
    serialize(s__, gv__.bare_gvec_);
    serialize(s__, gv__.num_gvec_);
    serialize(s__, gv__.num_gvec_shells_);
    serialize(s__, gv__.gvec_full_index_);
    serialize(s__, gv__.gvec_shell_);
    serialize(s__, gv__.gvec_shell_len_);
    serialize(s__, gv__.gvec_index_by_xy_);
    serialize(s__, gv__.z_columns_);
    serialize(s__, gv__.gvec_distr_);
    serialize(s__, gv__.zcol_distr_);
    serialize(s__, gv__.gvec_base_mapping_);
    serialize(s__, gv__.offset_);
    serialize(s__, gv__.count_);
}

void
deserialize(serializer& s__, Gvec& gv__)
{
    deserialize(s__, gv__.vk_);
    deserialize(s__, gv__.Gmax_);
    deserialize(s__, gv__.lattice_vectors_);
    deserialize(s__, gv__.reduce_gvec_);
    deserialize(s__, gv__.bare_gvec_);
    deserialize(s__, gv__.num_gvec_);
    deserialize(s__, gv__.num_gvec_shells_);
    deserialize(s__, gv__.gvec_full_index_);
    deserialize(s__, gv__.gvec_shell_);
    deserialize(s__, gv__.gvec_shell_len_);
    deserialize(s__, gv__.gvec_index_by_xy_);
    deserialize(s__, gv__.z_columns_);
    deserialize(s__, gv__.gvec_distr_);
    deserialize(s__, gv__.zcol_distr_);
    deserialize(s__, gv__.gvec_base_mapping_);
    deserialize(s__, gv__.offset_);
    deserialize(s__, gv__.count_);
}

} // namespace fft

} // namespace sirius
