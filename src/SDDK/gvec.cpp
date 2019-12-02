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
#include "Symmetry/find_lat_sym.hpp"
#include "gvec.hpp"

namespace sddk {

vector3d<int> sddk::Gvec::gvec_by_full_index(uint32_t idx__) const
{
    /* index of the z coordinate of G-vector: first 12 bits */
    uint32_t j = idx__ & 0xFFF;
    /* index of z-column: last 20 bits */
    uint32_t i = idx__ >> 12;
    assert(i < (uint32_t)z_columns_.size());
    assert(j < (uint32_t)z_columns_[i].z.size());
    int x = z_columns_[i].x;
    int y = z_columns_[i].y;
    int z = z_columns_[i].z[j];
    return vector3d<int>(x, y, z);
}

void sddk::Gvec::find_z_columns(double Gmax__, const FFT3D_grid& fft_box__)
{
    mdarray<int, 2> non_zero_columns(fft_box__.limits(0), fft_box__.limits(1));
    non_zero_columns.zero();

    num_gvec_ = 0;

    auto add_new_column = [&](int i, int j) {
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
            auto vgk = lattice_vectors_ * (vector3d<double>(i, j, k) + vk_);
            /* add z-coordinate of G-vector to the list */
            if (vgk.length() <= Gmax__) {
                zcol.push_back(k);
            }
        }

        /* add column to the list */
        if (zcol.size() && !non_zero_columns(i, j)) {
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

    /* copy column order from previous G-vector set */
    if (gvec_base_) {
        for (int icol = 0; icol < gvec_base_->num_zcol(); icol++) {
            int i = gvec_base_->zcol(icol).x;
            int j = gvec_base_->zcol(icol).y;
            add_new_column(i, j);
        }
    }

    // Check all z-columns and add if within sphere. Only allow non-negative x-indices for reduced case
    for (int i = reduce_gvec_? 0 : fft_box__.limits(0).first; i <= fft_box__.limits(0).second; i++) {
        for (int j = fft_box__.limits(1).first; j <= fft_box__.limits(1).second; j++) {
            add_new_column(i, j);
        }
    }

    if (!gvec_base_) {
        /* put column with {x, y} = {0, 0} to the beginning */
        for (size_t i = 0; i < z_columns_.size(); i++) {
            if (z_columns_[i].x == 0 && z_columns_[i].y == 0) {
                std::swap(z_columns_[i], z_columns_[0]);
                break;
            }
        }
    }

    /* sort z-columns starting from the second or skip num_zcol of base distribution */
    int n = (gvec_base_) ? gvec_base_->num_zcol() : 1;
    std::sort(z_columns_.begin() + n, z_columns_.end(),
              [](z_column_descriptor const& a, z_column_descriptor const& b) { return a.z.size() > b.z.size(); });
}

void Gvec::distribute_z_columns()
{
    gvec_distr_ = block_data_descriptor(comm().size());
    zcol_distr_ = block_data_descriptor(comm().size());
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
        throw std::runtime_error("wrong number of G-vectors");
    }
}

void Gvec::find_gvec_shells()
{
    if (!bare_gvec_) {
        return;
    }

    PROFILE("sddk::Gvec::find_gvec_shells");

    auto lat_sym = sirius::find_lat_sym(lattice_vectors_, 1e-6);

    num_gvec_shells_ = 0;
    gvec_shell_      = mdarray<int, 1>(num_gvec_);

    std::fill(&gvec_shell_[0], &gvec_shell_[0] + num_gvec_, -1);

    /* find G-vector shells using symmetry consideration */
    for (int ig = 0; ig < num_gvec_; ig++) {
        if (gvec_shell_[ig] == -1) {
            auto G = gvec(ig);
            for (size_t isym = 0; isym < lat_sym.size(); isym++) {
                auto R   = lat_sym[isym];
                auto G1  = R * G;
                auto ig1 = index_by_gvec(G1);
                if (ig1 == -1) {
                    auto G1 = R * (G * (-1));
                    ig1     = index_by_gvec(G1);
                }
                if (ig1 >= 0) {
                    gvec_shell_[ig1] = num_gvec_shells_;
                }
            }
            num_gvec_shells_++;
        }
    }
    for (int ig = 0; ig < num_gvec_; ig++) {
        if (gvec_shell_[ig] == -1) {
            throw std::runtime_error("[sddk::Gvec] wrong G-vector shell");
        }
    }

    gvec_shell_len_ = mdarray<double, 1>(num_gvec_shells_);
    std::fill(&gvec_shell_len_[0], &gvec_shell_len_[0] + num_gvec_shells_, -1);

    for (int ig = 0; ig < num_gvec_; ig++) {
        auto g   = gvec_cart<index_domain_t::global>(ig).length();
        int igsh = gvec_shell_[ig];
        if (gvec_shell_len_[igsh] < 0) {
            gvec_shell_len_[igsh] = g;
        } else {
            /* lattice symmetries were found wih 1e-6 tolererance for the metric tensor,
               so tolerance on length should be square root of that */
            if (std::abs(gvec_shell_len_[igsh] - g) > 1e-3) {
                std::stringstream s;
                s << "[sddk::Gvec] wrong G-vector length\n"
                  << "  length of G-shell : " << gvec_shell_len_[igsh] << "\n"
                  << "  length of current G-vector: " << g << "\n"
                  << "  index of G-vector: " << ig << "\n"
                  << "  index of G-shell: " << igsh << "\n"
                  << "  length difference: " << std::abs(gvec_shell_len_[igsh] - g);
                throw std::runtime_error(s.str());
            }
        }
    }

    // TODO: maybe, make an average G-shell length.

    /* list of pairs (length, index of G-vector) */
    std::vector<std::pair<uint64_t, int>> tmp(num_gvec_);
    #pragma omp parallel for schedule(static)
    for (int ig = 0; ig < num_gvec(); ig++) {
        /* make some reasonable roundoff */
        uint64_t len = static_cast<uint64_t>(gvec_shell_len_[gvec_shell_[ig]] * 1e10);
        tmp[ig]      = std::make_pair(len, ig);
    }
    /* sort by first element in pair (length) */
    std::sort(tmp.begin(), tmp.end());

    /* index of the first shell */
    gvec_shell_(tmp[0].second) = 0;
    num_gvec_shells_           = 1;
    /* temporary vector to store G-shell radius */
    std::vector<double> tmp_len;
    /* radius of the first shell */
    tmp_len.push_back(static_cast<double>(tmp[0].first) * 1e-10);
    for (int ig = 1; ig < num_gvec_; ig++) {
        /* if this G-vector has a different length */
        if (tmp[ig].first != tmp[ig - 1].first) {
            /* increment number of shells */
            num_gvec_shells_++;
            /* save the radius of the new shell */
            tmp_len.push_back(static_cast<double>(tmp[ig].first) * 1e-10);
        }
        /* assign the index of the current shell */
        gvec_shell_(tmp[ig].second) = num_gvec_shells_ - 1;
    }
    gvec_shell_len_ = mdarray<double, 1>(num_gvec_shells_);
    std::copy(tmp_len.begin(), tmp_len.end(), gvec_shell_len_.at(memory_t::host));

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
        for (auto igloc: it->second) {
            gvec_shell_idx_local_[igloc] = num_gvec_shells_local_;
        }
        num_gvec_shells_local_++;
    }
}

void Gvec::init_gvec_cart()
{
    gvec_cart_  = mdarray<double, 2>(3, count());
    gkvec_cart_ = mdarray<double, 2>(3, count());

    for (int igloc = 0; igloc < count(); igloc++) {
        int ig   = offset() + igloc;
        auto G   = gvec_by_full_index(gvec_full_index_(ig));
        auto gc  = lattice_vectors_ * vector3d<double>(G[0], G[1], G[2]);
        auto gkc = lattice_vectors_ * (vector3d<double>(G[0], G[1], G[2]) + vk_);
        for (int x : {0, 1, 2}) {
            gvec_cart_(x, igloc)  = gc[x];
            gkvec_cart_(x, igloc) = gkc[x];
        }
    }
}

void Gvec::init(FFT3D_grid const& fft_grid)
{
    PROFILE("sddk::Gvec::init");

    find_z_columns(Gmax_, fft_grid);

    distribute_z_columns();

    gvec_index_by_xy_ =
        mdarray<int, 3>(2, fft_grid.limits(0), fft_grid.limits(1), memory_t::host, "Gvec.gvec_index_by_xy_");
    std::fill(gvec_index_by_xy_.at(memory_t::host), gvec_index_by_xy_.at(memory_t::host) + gvec_index_by_xy_.size(),
              -1);

    /* build the full G-vector index and reverse mapping */
    gvec_full_index_ = mdarray<uint32_t, 1>(num_gvec_);
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
        throw std::runtime_error("wrong G-vector count");
    }
    for (int ig = 0; ig < num_gvec_; ig++) {
        auto gv = gvec(ig);
        if (index_by_gvec(gv) != ig) {
            std::stringstream s;
            s << "wrong G-vector index: ig=" << ig << " gv=" << gv << " index_by_gvec(gv)=" << index_by_gvec(gv);
            throw std::runtime_error(s.str());
        }
    }

    /* first G-vector must be (0, 0, 0); never reomove this check!!! */
    auto g0 = gvec_by_full_index(gvec_full_index_(0));
    if (g0[0] || g0[1] || g0[2]) {
        throw std::runtime_error("first G-vector is not zero");
    }

    init_gvec_cart();

    find_gvec_shells();

    if (gvec_base_) {
        /* the size of the mapping is equal to the local number of G-vectors in the base set */
        gvec_base_mapping_ = mdarray<int, 1>(gvec_base_->count());
        /* loop over local G-vectors of a base set */
        for (int igloc = 0; igloc < gvec_base_->count(); igloc++) {
            /* G-vector in lattice coordinates */
            auto G = gvec_base_->gvec(gvec_base_->offset() + igloc);
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
                throw std::runtime_error(s.str());
            }
        }
    }
    // TODO: add a check for gvec_base (there is already a test for this).
}

Gvec& Gvec::operator=(Gvec&& src__)
{
    if (this != &src__) {
        vk_                = src__.vk_;
        Gmax_              = src__.Gmax_;
        lattice_vectors_   = src__.lattice_vectors_;
        reduce_gvec_       = src__.reduce_gvec_;
        bare_gvec_         = src__.bare_gvec_;
        num_gvec_          = src__.num_gvec_;
        gvec_full_index_   = std::move(src__.gvec_full_index_);
        gvec_shell_        = std::move(src__.gvec_shell_);
        num_gvec_shells_   = std::move(src__.num_gvec_shells_);
        gvec_shell_len_    = std::move(src__.gvec_shell_len_);
        gvec_index_by_xy_  = std::move(src__.gvec_index_by_xy_);
        z_columns_         = std::move(src__.z_columns_);
        gvec_distr_        = std::move(src__.gvec_distr_);
        zcol_distr_        = std::move(src__.zcol_distr_);
        gvec_base_mapping_ = std::move(src__.gvec_base_mapping_);
    }
    return *this;
}

std::pair<int, bool> Gvec::index_g12_safe(vector3d<int> const& g1__, vector3d<int> const& g2__) const
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
        throw std::runtime_error(s.str());
    }
    return std::make_pair(idx, conj);
}

int Gvec::index_by_gvec(vector3d<int> const& G__) const
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
    assert(ig >= 0 && ig < num_gvec());
    return ig;
}

void Gvec::pack(serializer& s__) const
{
    serialize(s__, vk_);
    serialize(s__, Gmax_);
    serialize(s__, lattice_vectors_);
    serialize(s__, reduce_gvec_);
    serialize(s__, bare_gvec_);
    serialize(s__, num_gvec_);
    serialize(s__, num_gvec_shells_);
    serialize(s__, gvec_full_index_);
    serialize(s__, gvec_shell_);
    serialize(s__, gvec_shell_len_);
    serialize(s__, gvec_index_by_xy_);
    serialize(s__, z_columns_);
    serialize(s__, gvec_distr_);
    serialize(s__, zcol_distr_);
    serialize(s__, gvec_base_mapping_);
}

void Gvec::unpack(serializer& s__, Gvec& gv__) const
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
}

void Gvec::send_recv(Communicator const& comm__, int source__, int dest__, Gvec& gv__) const
{
    serializer s;

    if (comm__.rank() == source__) {
        this->pack(s);
    }

    s.send_recv(comm__, source__, dest__);

    if (comm__.rank() == dest__) {
        this->unpack(s, gv__);
    }
}

void Gvec_partition::build_fft_distr()
{
    /* calculate distribution of G-vectors and z-columns for the FFT communicator */
    gvec_distr_fft_ = block_data_descriptor(fft_comm().size());
    zcol_distr_fft_ = block_data_descriptor(fft_comm().size());

    for (int rank = 0; rank < fft_comm().size(); rank++) {
        for (int i = 0; i < comm_ortho_fft().size(); i++) {
            /* fine-grained rank */
            int r = rank_map_(rank, i);
            gvec_distr_fft_.counts[rank] += gvec().gvec_count(r);
            zcol_distr_fft_.counts[rank] += gvec().zcol_count(r);
        }
    }
    /* get offsets of z-columns */
    zcol_distr_fft_.calc_offsets();
    /* get offsets of G-vectors */
    gvec_distr_fft_.calc_offsets();
}

void Gvec_partition::calc_offsets()
{
    zcol_offs_ = mdarray<int, 1>(gvec().num_zcol(), memory_t::host, "Gvec_partition.zcol_offs_");
    for (int rank = 0; rank < fft_comm().size(); rank++) {
        int offs{0};
        /* loop over local number of z-columns */
        for (int i = 0; i < zcol_count_fft(rank); i++) {
            /* global index of z-column */
            int icol         = idx_zcol_[zcol_distr_fft_.offsets[rank] + i];
            zcol_offs_[icol] = offs;
            offs += static_cast<int>(gvec().zcol(icol).z.size());
        }
        assert(offs == gvec_distr_fft_.counts[rank]);
    }
}

void Gvec_partition::pile_gvec()
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
    gvec_fft_slab_ = block_data_descriptor(comm_ortho_fft_.size());
    for (int i = 0; i < comm_ortho_fft_.size(); i++) {
        gvec_fft_slab_.counts[i] = gvec().gvec_count(rank_map_(fft_comm_.rank(), i));
    }
    gvec_fft_slab_.calc_offsets();

    assert(gvec_fft_slab_.offsets.back() + gvec_fft_slab_.counts.back() == gvec_distr_fft_.counts[fft_comm().rank()]);
}

Gvec_partition::Gvec_partition(Gvec const& gvec__, Communicator const& fft_comm__, Communicator const& comm_ortho_fft__)
    : gvec_(gvec__)
    , fft_comm_(fft_comm__)
    , comm_ortho_fft_(comm_ortho_fft__)
{
    if (fft_comm_.size() * comm_ortho_fft_.size() != gvec_.comm().size()) {
        std::stringstream s;
        s << "[sddk::Gvec] wrong size of communicators" << std::endl
          << "  fft_comm_.size()       = " << fft_comm_.size() << std::endl
          << "  comm_ortho_fft_.size() = " << comm_ortho_fft_.size() << std::endl
          << "  gvec_.comm().size()    = " << gvec_.comm().size();
        throw std::runtime_error(s.str());
    }
    rank_map_ = mdarray<int, 2>(fft_comm_.size(), comm_ortho_fft_.size());
    rank_map_.zero();
    /* get a global rank */
    rank_map_(fft_comm_.rank(), comm_ortho_fft_.rank()) = gvec_.comm().rank();
    gvec_.comm().allreduce(&rank_map_(0, 0), gvec_.comm().size());

    build_fft_distr();

    idx_zcol_ = mdarray<int, 1>(gvec().num_zcol());
    int icol{0};
    for (int rank = 0; rank < fft_comm().size(); rank++) {
        for (int i = 0; i < comm_ortho_fft().size(); i++) {
            for (int k = 0; k < gvec_.zcol_count(rank_map_(rank, i)); k++) {
                idx_zcol_(icol) = gvec_.zcol_offset(rank_map_(rank, i)) + k;
                icol++;
            }
        }
        assert(icol == zcol_distr_fft_.counts[rank] + zcol_distr_fft_.offsets[rank]);
    }
    assert(icol == gvec().num_zcol());

    idx_gvec_ = mdarray<int, 1>(gvec_count_fft());
    int ig{0};
    for (int i = 0; i < comm_ortho_fft_.size(); i++) {
        for (int k = 0; k < gvec_.gvec_count(rank_map_(fft_comm_.rank(), i)); k++) {
            idx_gvec_[ig] = gvec_.gvec_offset(rank_map_(fft_comm_.rank(), i)) + k;
            ig++;
        }
    }

    calc_offsets();
    pile_gvec();
}

mdarray<int, 2> Gvec_partition::get_gvec() const
{
    mdarray<int, 2> gv(3, gvec_count_fft());
    for (int i = 0; i < gvec_count_fft(); i++) {
        int ig = idx_gvec_[i];
        auto G = gvec_.gvec(ig);
        for (int x: {0, 1, 2}) {
            gv(x, i) = G[x];
        }
    }
    return gv;
}

void Gvec_partition::gather_pw_fft(std::complex<double>* f_pw_local__, std::complex<double>* f_pw_fft__) const
{
    int rank = gvec().comm().rank();
    /* collect scattered PW coefficients */
    comm_ortho_fft().allgather(f_pw_local__, gvec().gvec_count(rank), f_pw_fft__, gvec_fft_slab().counts.data(),
                               gvec_fft_slab().offsets.data());
}

void Gvec_partition::gather_pw_global(std::complex<double>* f_pw_fft__, std::complex<double>* f_pw_global__) const
{
    for (int ig = 0; ig < gvec().count(); ig++) {
        /* position inside fft buffer */
        int ig1                             = gvec_fft_slab().offsets[comm_ortho_fft().rank()] + ig;
        f_pw_global__[gvec().offset() + ig] = f_pw_fft__[ig1];
    }
    gvec().comm().allgather(&f_pw_global__[0], gvec().offset(), gvec().count());
}

Gvec_shells::Gvec_shells(Gvec const& gvec__)
    : comm_(gvec__.comm())
    , gvec_(gvec__)
{
    PROFILE("sddk::Gvec_shells");

    a2a_send = block_data_descriptor(comm_.size());
    a2a_recv = block_data_descriptor(comm_.size());

    /* split G-vector shells between ranks in cyclic order */
    spl_num_gsh = splindex<splindex_t::block_cyclic>(gvec_.num_shells(), comm_.size(), comm_.rank(), 1);

    /* each rank sends a fraction of its local G-vectors to other ranks */
    /* count this fraction */
    for (int igloc = 0; igloc < gvec_.count(); igloc++) {
        int ig   = gvec_.offset() + igloc;
        int igsh = gvec_.shell(ig);
        a2a_send.counts[spl_num_gsh.local_rank(igsh)]++;
    }
    a2a_send.calc_offsets();
    /* sanity check: total number of elements to send is equal to the local number of G-vector */
    if (a2a_send.size() != gvec_.count()) {
        throw std::runtime_error("wrong number of G-vectors");
    }
    /* count the number of elements to receive */
    for (int r = 0; r < comm_.size(); r++) {
        for (int igloc = 0; igloc < gvec_.gvec_count(r); igloc++) {
            /* index of G-vector in the original distribution */
            int ig = gvec_.gvec_offset(r) + igloc;
            /* index of the G-vector shell */
            int igsh = gvec_.shell(ig);
            if (spl_num_gsh.local_rank(igsh) == comm_.rank()) {
                a2a_recv.counts[r]++;
            }
        }
    }
    a2a_recv.calc_offsets();
    /* sanity check: sum of local sizes in the remapped order is equal to the total number of G-vectors */
    int ng = gvec_count_remapped();
    comm_.allreduce(&ng, 1);
    if (ng != gvec_.num_gvec()) {
        throw std::runtime_error("wrong number of G-vectors");
    }

    /* local set of G-vectors in the remapped order */
    gvec_remapped_       = mdarray<int, 2>(3, gvec_count_remapped());
    gvec_shell_remapped_ = mdarray<int, 1>(gvec_count_remapped());
    std::vector<int> counts(comm_.size(), 0);
    for (int r = 0; r < comm_.size(); r++) {
        for (int igloc = 0; igloc < gvec_.gvec_count(r); igloc++) {
            int ig   = gvec_.gvec_offset(r) + igloc;
            int igsh = gvec_.shell(ig);
            auto G   = gvec_.gvec(ig);
            if (spl_num_gsh.local_rank(igsh) == comm_.rank()) {
                for (int x : {0, 1, 2}) {
                    gvec_remapped_(x, a2a_recv.offsets[r] + counts[r]) = G[x];
                }
                gvec_shell_remapped_(a2a_recv.offsets[r] + counts[r]) = igsh;
                counts[r]++;
            }
        }
    }
    for (int ig = 0; ig < gvec_count_remapped(); ig++) {
        idx_gvec[gvec_remapped(ig)] = ig;
    }
}

} // namespace sddk
