#ifndef __GVEC_H__
#define __GVEC_H__

#include <algorithm>
#include "descriptors.h"
#include "fft_grid.h"
#include "splindex.h"

namespace sirius {

class Gvec
{
    private:

        vector3d<double> q_;
        
        FFT_grid fft_grid_;

        matrix3d<double> lattice_vectors_;

        bool reduce_gvec_;

        /// Total number of G-vectors.
        int num_gvec_;

        /// Local number of G-vectors for FFT communicator.
        int num_gvec_fft_;
        
        /// Offset (in the global index) of the local fraction of G-vectors for FFT communicator.
        int offset_gvec_fft_;
        
        /// Mapping between G-vector index [0:num_gvec_) and a full index.
        /** Full index is used to store x,y,z coordinates in a packed form in a single integer number. */
        mdarray<int, 1> gvec_full_index_;
    
        /// Index of the shell to which the given G-vector belongs.
        mdarray<int, 1> gvec_shell_;
        
        /// Position in the local slab of FFT buffer by local G-vec index.
        mdarray<int, 1> index_map_;

        mdarray<int, 2> z_columns_pos_;

        int num_gvec_shells_;

        mdarray<double, 1> gvec_shell_len_;

        mdarray<int, 3> index_by_gvec_;

        /// Global list of non-zero z-columns.
        std::vector<z_column_descriptor> z_columns_;

        block_data_descriptor zcol_fft_distr_;

        block_data_descriptor gvec_fft_distr_;

        block_data_descriptor gvec_distr_;

        Gvec(Gvec const& src__) = delete;

        Gvec& operator=(Gvec const& src__) = delete;

    public:

        Gvec()
        {
        }

        Gvec(vector3d<double> q__,
             matrix3d<double> const& M__,
             double Gmax__,
             FFT_grid const& fft_grid__,
             Communicator const& comm__,
             int comm_size_factor__,
             bool build_reverse_mapping__,
             bool reduce_gvec__)
            : q_(q__),
              fft_grid_(fft_grid__),
              lattice_vectors_(M__),
              reduce_gvec_(reduce_gvec__)
        {
            mdarray<int, 2> non_zero_columns(fft_grid_.limits(0), fft_grid_.limits(1));
            non_zero_columns.zero();

            num_gvec_ = 0;
            for (int i = fft_grid_.limits(0).first; i <= fft_grid_.limits(0).second; i++)
            {
                for (int j = fft_grid_.limits(1).first; j <= fft_grid_.limits(1).second; j++)
                {
                    std::vector<int> z;

                    int zmin = fft_grid_.size(2);
                    if (reduce_gvec_ && !i && !j) zmin = fft_grid_.limits(2).second;
                    
                    for (int iz = 0; iz < zmin; iz++)
                    {
                        /* get z-coordinate of G-vector */
                        int k = (iz > fft_grid_.limits(2).second) ? iz - fft_grid_.size(2) : iz;
                        /* take G+q */
                        auto gq = lattice_vectors_ * (vector3d<double>(i, j, k) + q__);

                        if (gq.length() <= Gmax__) z.push_back(k);
                    }
                    
                    if (z.size() && !non_zero_columns(i, j))
                    {
                        z_columns_.push_back(z_column_descriptor(i, j, z));
                        num_gvec_ += static_cast<int>(z.size());

                        non_zero_columns(i, j) = 1;
                        if (reduce_gvec__) non_zero_columns(-i, -j) = 1;
                    }
                }
            }
            
            /* put column with {x, y} = {0, 0} to the beginning */
            for (size_t i = 0; i < z_columns_.size(); i++)
            {
                if (z_columns_[i].x == 0 && z_columns_[i].y == 0)
                {
                    std::swap(z_columns_[i], z_columns_[0]);
                    break;
                }
            }
            
            /* sort z-columns starting from the second */
            std::sort(z_columns_.begin() + 1, z_columns_.end(),
                      [](z_column_descriptor const& a, z_column_descriptor const& b)
                      {
                          return a.z.size() > b.z.size();
                      });
            
            int num_ranks = comm__.size() * comm_size_factor__;

            gvec_distr_ = block_data_descriptor(num_ranks);
            std::vector< std::vector<z_column_descriptor> > zcols_local(num_ranks);

            std::vector<int> ranks;
            for (size_t i = 0; i < z_columns_.size(); i++)
            {
                /* initialize the list of ranks to 0,1,2,... */
                if (ranks.empty())
                {
                    ranks.resize(num_ranks);
                    std::iota(ranks.begin(), ranks.end(), 0);
                }
                auto rank_with_min_gvec = std::min_element(ranks.begin(), ranks.end(), 
                                                           [this](const int& a, const int& b)
                                                           {
                                                               return gvec_distr_.counts[a] < gvec_distr_.counts[b];
                                                           });

                /* assign column to the current rank */
                zcols_local[*rank_with_min_gvec].push_back(z_columns_[i]);
                /* count local number of G-vectors */
                gvec_distr_.counts[*rank_with_min_gvec] += static_cast<int>(z_columns_[i].z.size());
                /* exclude this rank from the search */
                ranks.erase(rank_with_min_gvec);
            }
            gvec_distr_.calc_offsets();

            /* save new ordering of z-columns */
            z_columns_.clear();
            for (int rank = 0; rank < num_ranks; rank++)
            {
                z_columns_.insert(z_columns_.end(), zcols_local[rank].begin(), zcols_local[rank].end());
            }

            z_columns_pos_ = mdarray<int, 2>(2, z_columns_.size());
            for (size_t i = 0; i < z_columns_.size(); i++)
            {
                int x = z_columns_[i].x;
                int y = z_columns_[i].y;
                if (x < 0) x += fft_grid_.size(0);
                if (y < 0) y += fft_grid_.size(1);
                z_columns_pos_(0, i) = x;
                z_columns_pos_(1, i) = y;
            }

            /* calculate distribution of G-vectors and z-columns for FFT communicator */
            gvec_fft_distr_ = block_data_descriptor(comm__.size());
            zcol_fft_distr_ = block_data_descriptor(comm__.size());
            for (int rank = 0; rank < comm__.size(); rank++)
            {
                for (int i = 0; i < comm_size_factor__; i++)
                {
                    int r = rank * comm_size_factor__ + i;
                    gvec_fft_distr_.counts[rank] += gvec_distr_.counts[r];
                    zcol_fft_distr_.counts[rank] += static_cast<int>(zcols_local[r].size());
                }
            }
            /* get offsets of z-columns */
            zcol_fft_distr_.calc_offsets();
            /* get offsets of G-vectors */
            gvec_fft_distr_.calc_offsets();
            /* get local number of G-vectors for a given rank */
            num_gvec_fft_ = gvec_fft_distr_.counts[comm__.rank()];
            /* get offset of G-vectors for a given rank */
            offset_gvec_fft_ = gvec_fft_distr_.offsets[comm__.rank()];
            
            /* calculate offsets of z-columns inside each local buffer of PW coefficients */
            for (int rank = 0; rank < comm__.size(); rank++)
            {
                int offs = 0;
                for (int i = 0; i < zcol_fft_distr_.counts[rank]; i++)
                {
                    int icol = zcol_fft_distr_.offsets[rank] + i;
                    z_columns_[icol].offset = offs;
                    offs += static_cast<int>(z_columns_[icol].z.size());
                }
                assert(offs == gvec_fft_distr_.counts[rank]);
            }

            gvec_full_index_ = mdarray<int, 1>(num_gvec_);
            int ig = 0;
            for (size_t i = 0; i < z_columns_.size(); i++)
            {
                for (size_t j = 0; j < z_columns_[i].z.size(); j++)
                {
                    gvec_full_index_(ig++) = static_cast<int>((i << 12) + j);
                }
            }

            auto g0 = gvec_by_full_index(gvec_full_index_(0));
            if (g0[0] || g0[1] || g0[2]) TERMINATE("first G-vector is not zero");

            if (comm__.size() == 1)
            {
                index_map_ = mdarray<int, 1>(num_gvec_);
                for (int ig = 0; ig < num_gvec_; ig++)
                {
                    auto G = gvec_by_full_index(gvec_full_index_(ig));
                    index_map_(ig) = fft_grid_.index_by_gvec(G[0], G[1], G[2]);
                }
            }

            std::map<size_t, std::vector<int> > gsh;
            for (int ig = 0; ig < num_gvec_; ig++)
            {
                auto G = gvec_by_full_index(gvec_full_index_(ig));

                /* take G+q */
                auto gq = M__ * (vector3d<double>(G[0], G[1], G[2]) + q__);

                size_t len = size_t(gq.length() * 1e10);

                if (!gsh.count(len)) gsh[len] = std::vector<int>();
                
                gsh[len].push_back(ig);
            }
            num_gvec_shells_ = (int)gsh.size();
            gvec_shell_ = mdarray<int, 1>(num_gvec_);
            gvec_shell_len_ = mdarray<double, 1>(num_gvec_shells_);
            
            int n = 0;
            for (auto it = gsh.begin(); it != gsh.end(); it++)
            {
                gvec_shell_len_(n) = double(it->first) * 1e-10;
                for (int ig: it->second) gvec_shell_(ig) = n;
                n++;
            }

            if (build_reverse_mapping__)
            {
                index_by_gvec_ = mdarray<int, 3>(fft_grid_.limits(0), fft_grid_.limits(1), fft_grid_.limits(2));

                std::fill(index_by_gvec_.at<CPU>(), index_by_gvec_.at<CPU>() + fft_grid_.size(), -1);

                for (int ig = 0; ig < num_gvec_; ig++)
                {
                    auto G = gvec_by_full_index(gvec_full_index_(ig));
                    index_by_gvec_(G[0], G[1], G[2]) = ig;
                }
            }
        }

        Gvec& operator=(Gvec&& src__) = default;

        /// Return number of G-vectors within the cutoff.
        inline int num_gvec() const
        {
            return num_gvec_;
        }

        /// Return local number of G-vectors for the FFT communicator.
        inline int num_gvec_fft() const
        {
            return num_gvec_fft_;
        }

        /// Offset (in the global index) of G-vectors distributed between ranks of FFT communicator.
        inline int offset_gvec_fft() const
        {
            return offset_gvec_fft_;
        }
        
        /// Number of G-vectors for a fine-grained distribution.
        inline int num_gvec(int rank__) const
        {
            assert((size_t)rank__ < gvec_distr_.counts.size());
            return gvec_distr_.counts[rank__];
        }

        /// Offset (in the global index) of G-vectors for a fine-grained distribution.
        inline int offset_gvec(int rank__) const
        {
            return gvec_distr_.offsets[rank__];
        }
        
        /// Return number of G-vector shells.
        inline int num_shells() const
        {
            return num_gvec_shells_;
        }
        
        /// Return corresponding G-vector for an index in the range [0, num_gvec).
        inline vector3d<int> gvec_by_full_index(int idx__) const
        {
            int j = idx__ & 0xFFF;
            int i = idx__ >> 12;
            assert(i < (int)z_columns_.size());
            assert(j < (int)z_columns_[i].z.size());
            int x = z_columns_[i].x;
            int y = z_columns_[i].y;
            int z = z_columns_[i].z[j];
            return fft_grid_.gvec_by_coord(x, y, z);
        }

        /// Return corresponding G-vector for an index in the range [0, num_gvec).
        inline vector3d<int> operator[](int ig__) const
        {
            assert(ig__ >= 0 && ig__ < num_gvec_);
            return gvec_by_full_index(gvec_full_index_(ig__));
        }

        inline vector3d<double> gvec_shifted(int ig__) const
        {
            auto G = gvec_by_full_index(gvec_full_index_(ig__));
            return (vector3d<double>(G[0], G[1], G[2]) + q_);
        }

        /// Return G-vector in Cartesian coordinates.
        inline vector3d<double> cart(int ig__) const
        {
            auto G = gvec_by_full_index(gvec_full_index_(ig__));
            return lattice_vectors_ * vector3d<double>(G[0], G[1], G[2]);
        }

        /// Return G+q-vector in Cartesian coordinates.
        inline vector3d<double> cart_shifted(int ig__) const
        {
            auto G = gvec_by_full_index(gvec_full_index_(ig__));
            return lattice_vectors_ * (vector3d<double>(G[0], G[1], G[2]) + q_);
        }

        inline int shell(int ig__) const
        {
            return gvec_shell_(ig__);
        }

        inline double shell_len(int igs__) const
        {
            return gvec_shell_len_(igs__);
        }

        inline double gvec_len(int ig__) const
        {
            return gvec_shell_len_(gvec_shell_(ig__));
        }

        inline int index_g12(vector3d<int> const& g1__, vector3d<int> const& g2__) const
        {
            auto v = g1__ - g2__;
            int idx = index_by_gvec(v);
            assert(idx >= 0 && idx < num_gvec());
            return idx;
        }

        inline int index_g12_safe(int ig1__, int ig2__) const
        {
            STOP();
            return 0;
        }

        inline int index_by_gvec(vector3d<int>& G__) const
        {
            return index_by_gvec_(G__[0], G__[1], G__[2]);
        }

        inline std::vector<z_column_descriptor> const& z_columns() const
        {
            return z_columns_;
        }

        inline block_data_descriptor const& zcol_fft_distr() const
        {
            return zcol_fft_distr_;
        }

        inline bool reduced() const
        {
            return reduce_gvec_;
        }

        inline mdarray<int,1>& index_map()
        {
            return index_map_;
        }

        inline mdarray<int, 2>& z_columns_pos()
        {
            return z_columns_pos_;
        }

        inline mdarray<int, 2> const& z_columns_pos() const
        {
            return z_columns_pos_;
        }
};

};

#endif
