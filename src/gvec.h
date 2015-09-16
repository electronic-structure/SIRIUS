#include "fft3d.h"

namespace sirius {

class Gvec
{
    private:
        
        FFT3D_CPU* fft_;

        matrix3d<double> lattice_vectors_;

        /// Total number of G-vectors.
        int num_gvec_;

        /// Local number of G-vectors for a given slab.
        int num_gvec_loc_;
        
        /// Mapping between G-vector index [0:num_gvec_) and a full index.
        /** Full index is used to store x,y,z coordinates in a packed form in a single integer number. */
        mdarray<int, 1> gvec_full_index_;
    
        /// Index of the shell to which the given G-vector belongs.
        mdarray<int, 1> gvec_shell_;
        
        /// Position in the local slab of FFT buffer by local G-vec index.
        mdarray<int, 1> index_map_local_to_local_;

        std::vector<int> gvec_counts_;

        std::vector<int> gvec_offsets_;

        int num_gvec_shells_;

        mdarray<double, 1> gvec_shell_len_;

        mdarray<int, 3> index_by_gvec_;

        /// Coordinates (x, y) of non-zero z-sticks.
        std::vector< std::pair<int, int> > z_sticks_coord_;
        
        /// Map G-vectors to the 3D buffer index (in range [0:Nx*Ny*N_z_loc_-1]) or to the packed index of z-sticks.
        bool map_to_3d_idx_;

        Gvec(Gvec const& src__) = delete;

        Gvec& operator=(Gvec const& src__) = delete;

    public:

        Gvec() : fft_(nullptr)
        {
        }

        Gvec(vector3d<double> q__,
             double Gmax__,
             matrix3d<double> const& M__,
             FFT3D_CPU* fft__,
             bool build_reverse_mapping__)
            : fft_(fft__),
              lattice_vectors_(M__),
              map_to_3d_idx_(false)
        {
            mdarray<int, 2> non_zero_z_sticks(fft_->size(0), fft_->size(1));
            non_zero_z_sticks.zero();

            /* find local number of G-vectors for each slab of FFT buffer;
             * at the same time, find the non-zero z-sticks */
            std::vector< vector3d<int> > pos;
            for (int k = 0; k < fft_->local_size_z(); k++)
            {
                for (int j = 0; j < fft_->size(1); j++)
                {
                    for (int i = 0; i < fft_->size(0); i++)
                    {
                        auto G = fft_->gvec_by_grid_pos(i, j, k + fft_->offset_z());
                       
                        /* take G+q */
                        auto gq = lattice_vectors_ * (vector3d<double>(G[0], G[1], G[2]) + q__);

                        if (gq.length() <= Gmax__)
                        {
                            pos.push_back(vector3d<int>(i, j, k));
                            non_zero_z_sticks(i, j) = 1;
                        }
                    }
                }
            }
            /* get total number of G-vectors */
            num_gvec_loc_ = (int)pos.size();
            num_gvec_ = num_gvec_loc_;
            fft_->comm().allreduce(&num_gvec_, 1);
            /* get the full map of non-zero z-sticks */
            fft_->comm().allreduce<int, op_max>(non_zero_z_sticks.at<CPU>(), (int)non_zero_z_sticks.size());
            
            /* build a linear index of xy coordinates of non-zero z-sticks */
            mdarray<int, 2> xy_idx(fft_->size(0), fft_->size(1));
            for (int x = 0; x < fft_->size(0); x++)
            {
                for (int y = 0; y < fft_->size(1); y++)
                {
                    if (non_zero_z_sticks(x, y))
                    {
                        xy_idx(x, y) = (int)z_sticks_coord_.size();
                        z_sticks_coord_.push_back({x, y});
                    }
                }
            }

            gvec_full_index_          = mdarray<int, 1>(num_gvec_);
            index_map_local_to_local_ = mdarray<int, 1>(num_gvec_loc_);

            gvec_counts_  = std::vector<int>(fft_->comm().size(), 0);
            gvec_offsets_ = std::vector<int>(fft_->comm().size(), 0);

            /* get local sizes from all ranks */
            gvec_counts_[fft_->comm().rank()] = num_gvec_loc();
            fft_->comm().allreduce(&gvec_counts_[0], fft_->comm().size());
            
            /* compute offsets in global G-vector index */
            for (int i = 1; i < fft_->comm().size(); i++) 
                gvec_offsets_[i] = gvec_offsets_[i - 1] + gvec_counts_[i - 1]; 

            for (int igloc = 0; igloc < num_gvec_loc_; igloc++)
            {
                auto p = pos[igloc];

                if (!fft_->parallel())
                {
                    /* map G-vector to a position in 3D FFT buffer */ 
                    index_map_local_to_local_(igloc) = p[0] + p[1] * fft_->size(0) + p[2] * fft_->size(0) * fft_->size(1);
                }
                else
                {
                    /* map G-vector to the {z, xy} storage of non-zero z-sticks;
                     * this kind of storage is needed for mpi_alltoall */
                    index_map_local_to_local_(igloc) = p[2] + fft_->local_size_z() * xy_idx(p[0], p[1]);
                }
                
                /* this is only one way to pack coordinates into single integer */
                gvec_full_index_(gvec_offset() + igloc) = 
                    p[0] + p[1] * fft_->size(0) + (p[2] + fft_->offset_z()) * fft_->size(0) * fft_->size(1);
            }

            fft_->comm().allgather(&gvec_full_index_(0), gvec_offset(), num_gvec_loc()); 

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
                index_by_gvec_ = mdarray<int, 3>(mdarray_index_descriptor(fft_->grid_limits(0).first, fft_->grid_limits(0).second),
                                                 mdarray_index_descriptor(fft_->grid_limits(1).first, fft_->grid_limits(1).second),
                                                 mdarray_index_descriptor(fft_->grid_limits(2).first, fft_->grid_limits(2).second));
                memset(index_by_gvec_.at<CPU>(), 0xFF, index_by_gvec_.size() * sizeof(int));

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

        /// Return local number of G-vectors for the z-slab of FFT buffer.
        inline int num_gvec_loc() const
        {
            return num_gvec_loc_;
        }
        
        /// Offset of G-vector global index.
        inline int gvec_offset() const
        {
            return gvec_offsets_[fft_->comm().rank()];
        }

        /// Return number of G-vector shells.
        inline int num_shells() const
        {
            return num_gvec_shells_;
        }

        inline vector3d<int> gvec_by_full_index(int idx__) const // TODO: use bit masks and bit shifts
        {
            int k = idx__ / (fft_->size(0) * fft_->size(1));
            idx__ -= k * fft_->size(0) * fft_->size(1);
            int j = idx__ / fft_->size(0);
            int i = idx__ -  j * fft_->size(0);
            return fft_->gvec_by_grid_pos(i, j, k);
        }

        inline vector3d<int> operator[](int ig__) const
        {
            assert(ig__ >= 0 && ig__ < num_gvec_);
            return gvec_by_full_index(gvec_full_index_(ig__));
        }

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
            return (num_gvec_loc() == 0) ? nullptr : &index_map_local_to_local_(0);
        }

        inline int index_by_gvec(vector3d<int>& G__) const
        {
            return index_by_gvec_(G__[0], G__[1], G__[2]);
        }

        inline std::vector<int> const& counts() const
        {
            return gvec_counts_;
        }

        std::vector< std::pair<int, int> > const& z_sticks_coord() const
        {
            return z_sticks_coord_;
        }
};

};
