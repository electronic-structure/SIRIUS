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
        
        //FFT3D* fft_;
        FFT_grid fft_grid_;

        matrix3d<double> lattice_vectors_;

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
        mdarray<int, 1> index_map_local_to_local_;

        //std::vector<int> gvec_counts_;

        //std::vector<int> gvec_offsets_;

        int num_gvec_shells_;

        mdarray<double, 1> gvec_shell_len_;

        mdarray<int, 3> index_by_gvec_;

        /// Coordinates (x, y) of non-zero z-sticks.
        std::vector< std::pair<int, int> > z_sticks_coord_;
        
        /// Global list of non-zero z-columns.
        std::vector<z_column_descriptor> z_columns_;

        /// Map G-vectors to the 3D buffer index (in range [0:Nx*Ny*N_z_loc_-1]) or to the packed index of z-sticks.
        bool map_to_3d_idx_;

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
             bool build_reverse_mapping__)
            : fft_grid_(fft_grid__),
              lattice_vectors_(M__),
              map_to_3d_idx_(false)
        {
            num_gvec_ = 0;
            for (int i = 0; i < fft_grid_.size(0); i++)
            {
                for (int j = 0; j < fft_grid_.size(1); j++)
                {
                    std::vector<int> z;
                    
                    for (int k = 0; k < fft_grid_.size(2); k++)
                    {
                        auto G = fft_grid_.gvec_by_coord(i, j, k);
                       
                        /* take G+q */
                        auto gq = lattice_vectors_ * (vector3d<double>(G[0], G[1], G[2]) + q__);

                        if (gq.length() <= Gmax__) z.push_back(k);
                    }

                    if (z.size())
                    {
                        z_columns_.push_back(z_column_descriptor(i, j, z));
                        num_gvec_ += static_cast<int>(z.size());
                    }
                }
            }
            
            /* sort z-columns */
            std::sort(z_columns_.begin(), z_columns_.end(),
                      [](z_column_descriptor const& a, z_column_descriptor const& b)
                      {
                          return a.z.size() > b.z.size();
                      });
            
            //== if (comm__.rank()== 0)
            //== {
            //==     FILE* fout = fopen("z_cols_sorted.dat", "w");
            //==     for (size_t i = 0; i < z_columns_.size(); i++)
            //==     {
            //==         fprintf(fout, "%li %li %li\n", i, z_columns_[i].z.size(), z_columns_[z_columns_.size()-i-1].z.size());
            //==     }
            //==     fclose(fout);
            //== }

            int num_ranks = comm__.size() * comm_size_factor__;

            gvec_distr_ = block_data_descriptor(num_ranks);
            std::vector< std::vector<z_column_descriptor> > zcols_local(num_ranks);

            std::vector<int> ranks;
            for (int i = 0; i < (int)z_columns_.size(); i++)
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

            //if (comm__.rank() == 0)
            //{
            //    for (int i = 0; i < num_ranks; i++)
            //    {
            //        printf("rank: %i, num_cols: %li, num_gvec: %i\n", i, z_columns_distr_[i].size(), gvec_counts[i]);
            //    }
            //}
            
            /* reorder z-columns */
            z_columns_.clear();
            for (int rank = 0; rank < num_ranks; rank++)
            {
                z_columns_.insert(z_columns_.end(), zcols_local[rank].begin(), zcols_local[rank].end());
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


            //if (comm__.rank() == 0)
            //{
            //    for (int i = 0; i < comm__.size(); i++)
            //    {
            //        printf("rank: %i, num_gvec: %i\n", i, gvec_fft_distr_.counts[i]);
            //    }
            //}
            









            //zcol_distr_ = block_data_descriptor(comm__.size());
            //gvec_distr_ = block_data_descriptor(comm__.size());

            //std::vector<int> ranks;
            //
            //for (size_t i = 0; i < z_columns_.size(); i++)
            //{
            //    /* initialize the list of ranks to 0,1,2,... */
            //    if (ranks.empty())
            //    {
            //        ranks.resize(comm__.size());
            //        std::iota(ranks.begin(), ranks.end(), 0);
            //    }
            //    auto rank_with_min_gvec = std::min_element(ranks.begin(), ranks.end(), 
            //                                               [this](const int& a, const int& b)
            //                                                     {
            //                                                        return gvec_distr_.counts[a] < gvec_distr_.counts[b];
            //                                                     });

            //    /* offset in the local part of PW coefficients */
            //    z_columns_[i].offset = gvec_distr_.counts[*rank_with_min_gvec];
            //    /* assign column to the current rank */
            //    z_columns_local_[*rank_with_min_gvec].push_back(z_columns_[i]);
            //    /* count local number of columns */
            //    zcol_distr_.counts[*rank_with_min_gvec]++;
            //    /* count local number of G-vectors */
            //    gvec_distr_.counts[*rank_with_min_gvec] += static_cast<int>(z_columns_[i].z.size());
            //    /* exclude this rank from the search */
            //    ranks.erase(rank_with_min_gvec);
            //}

            //if (comm__.rank()== 0)
            //{
            //    FILE* fout = fopen("z_cols_distr.dat", "w");
            //    for (size_t i = 0; i < z_columns_.size(); i++)
            //    {
            //        fprintf(fout, "%li %li %li\n", i, z_columns_[i].z.size(), z_columns_[z_columns_.size()-i-1].z.size());
            //    }
            //    fclose(fout);
            //}




            ///* find local number of G-vectors for each slab of FFT buffer;
            // * at the same time, find the non-zero z-sticks */
            //std::vector< vector3d<int> > pos;
            //for (int k = 0; k < fft_->local_size_z(); k++)
            //{
            //    for (int j = 0; j < fft_->size(1); j++)
            //    {
            //        for (int i = 0; i < fft_->size(0); i++)
            //        {
            //            auto G = fft_->gvec_by_grid_pos(i, j, k + fft_->offset_z());
            //           
            //            /* take G+q */
            //            auto gq = lattice_vectors_ * (vector3d<double>(G[0], G[1], G[2]) + q__);

            //            if (gq.length() <= Gmax__)
            //            {
            //                pos.push_back(vector3d<int>(i, j, k));
            //                non_zero_z_sticks(i, j) = 1;
            //            }
            //        }
            //    }
            //}
            ///* get total number of G-vectors */
            //num_gvec_loc_ = (int)pos.size();
            //num_gvec_ = num_gvec_loc_;
            //fft_->comm().allreduce(&num_gvec_, 1);
            ///* get the full map of non-zero z-sticks */
            //fft_->comm().allreduce<int, op_max>(non_zero_z_sticks.at<CPU>(), (int)non_zero_z_sticks.size());
            
            ///* build a linear index of xy coordinates of non-zero z-sticks */
            //mdarray<int, 2> xy_idx(fft_->size(0), fft_->size(1));
            //for (int x = 0; x < fft_->size(0); x++)
            //{
            //    for (int y = 0; y < fft_->size(1); y++)
            //    {
            //        if (non_zero_z_columns(x, y))
            //        {
            //            xy_idx(x, y) = (int)z_sticks_coord_.size();
            //            z_sticks_coord_.push_back({x, y});
            //        }
            //    }
            //}

            gvec_full_index_          = mdarray<int, 1>(num_gvec_);
            //index_map_local_to_local_ = mdarray<int, 1>(num_gvec_loc_);

            //gvec_counts_  = std::vector<int>(fft_->comm().size(), 0);
            //gvec_offsets_ = std::vector<int>(fft_->comm().size(), 0);

            /* get local sizes from all ranks */
            //gvec_counts_[fft_->comm().rank()] = num_gvec_loc();
            //fft_->comm().allreduce(&gvec_counts_[0], fft_->comm().size());
            
            ///* compute offsets in global G-vector index */
            //for (int i = 1; i < fft_->comm().size(); i++) 
            //    gvec_offsets_[i] = gvec_offsets_[i - 1] + gvec_counts_[i - 1]; 

            //== for (int igloc = 0; igloc < num_gvec_loc_; igloc++)
            //== {
            //==     auto p = pos[igloc];

            //==     if (!fft_->parallel())
            //==     {
            //==         /* map G-vector to a position in 3D FFT buffer */ 
            //==         index_map_local_to_local_(igloc) = p[0] + p[1] * fft_->size(0) + p[2] * fft_->size(0) * fft_->size(1);
            //==     }
            //==     else
            //==     {
            //==         /* map G-vector to the {z, xy} storage of non-zero z-sticks;
            //==          * this kind of storage is needed for mpi_alltoall */
            //==         index_map_local_to_local_(igloc) = p[2] + fft_->local_size_z() * xy_idx(p[0], p[1]);
            //==     }
            //==     
            //==     /* this is only one way to pack coordinates into single integer */
            //==     gvec_full_index_(gvec_offset() + igloc) = 
            //==         p[0] + p[1] * fft_->size(0) + (p[2] + fft_->offset_z()) * fft_->size(0) * fft_->size(1);
            //== }

            //== fft_->comm().allgather(&gvec_full_index_(0), gvec_offset(), num_gvec_loc()); 

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
                index_by_gvec_ = mdarray<int, 3>(mdarray_index_descriptor(fft_grid_.limits(0).first, fft_grid_.limits(0).second),
                                                 mdarray_index_descriptor(fft_grid_.limits(1).first, fft_grid_.limits(1).second),
                                                 mdarray_index_descriptor(fft_grid_.limits(2).first, fft_grid_.limits(2).second));
                //memset(index_by_gvec_.at<CPU>(), 0xFF, index_by_gvec_.size() * sizeof(int));
                std::fill(index_by_gvec_.at<CPU>(), index_by_gvec_.at<CPU>() + fft_grid_.size(), -1);

                for (int ig = 0; ig < num_gvec_; ig++)
                {
                    auto G = gvec_by_full_index(gvec_full_index_(ig));
                    index_by_gvec_(G[0], G[1], G[2]) = ig;
                }
            }

            if (build_reverse_mapping__ && false)
            {
                int num_gvec_reduced_ = 0;
                mdarray<int, 1> flg(num_gvec_);
                flg.zero();

                for (int ig = 0; ig < num_gvec_; ig++)
                {
                    if (!flg(ig))
                    {
                        flg(ig) = 1;
                        num_gvec_reduced_++;
                    }

                    auto G = gvec_by_full_index(gvec_full_index_(ig));
                    auto mG = G * (-1);
                    int igm = index_by_gvec_(mG[0], mG[1], mG[2]);
                    flg(igm) = 1;
                }
                printf("num_gvec_ : %i, num_gvec_reduced_ : %i\n", num_gvec_, num_gvec_reduced_);
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

        /// Return G-vector in Cartesian coordinates.
        inline vector3d<double> cart(int ig__) const
        {
            auto gv = gvec_by_full_index(gvec_full_index_(ig__));
            return lattice_vectors_ * vector3d<double>(gv[0], gv[1], gv[2]);
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

        inline int const* index_map() const
        {
            return nullptr;
            //return (num_gvec_loc() == 0) ? nullptr : &index_map_local_to_local_(0);
        }

        inline int index_by_gvec(vector3d<int>& G__) const
        {
            return index_by_gvec_(G__[0], G__[1], G__[2]);
        }

        //inline std::vector<int> const& counts() const
        //{
        //    return gvec_counts_;
        //}

        //inline std::vector< std::pair<int, int> > const& z_sticks_coord() const
        //{
        //    return z_sticks_coord_;
        //}

        inline std::vector<z_column_descriptor> const& z_columns() const
        {
            return z_columns_;
        }

        inline block_data_descriptor const& zcol_fft_distr() const
        {
            return zcol_fft_distr_;
        }
};

};

#endif
