// Copyright (c) 2013-2014 Anton Kozhevnikov, Thomas Schulthess
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

/** \file fft3d_cpu.hpp
 *   
 *  \brief Contains CPU specialization.
 */

/// CPU specialization of FFT3D class.
/** FFT convention:
 *  \f[
 *      f({\bf r}) = \sum_{{\bf G}} e^{i{\bf G}{\bf r}} f({\bf G})
 *  \f]
 *  is a \em backward transformation from a set of pw coefficients to a function.  
 *
 *  \f[
 *      f({\bf G}) = \frac{1}{\Omega} \int e^{-i{\bf G}{\bf r}} f({\bf r}) d {\bf r} = 
 *          \frac{1}{N} \sum_{{\bf r}_j} e^{-i{\bf G}{\bf r}_j} f({\bf r}_j)
 *  \f]
 *  is a \em forward transformation from a function to a set of coefficients. 
*/
template<> 
class FFT3D<CPU>
{
    private:

        /// Number of working threads inside each FFT.
        int num_fft_workers_;
        
        /// Number of threads doing individual FFTs.
        int num_fft_threads_;

        Communicator comm_;

        mdarray<double_complex, 2> data_slab_;
        mdarray<double_complex, 1> data_slice_;
        splindex<block> spl_z_;

        /// Size of each dimension.
        int grid_size_[3];

        int local_size_;

        int local_size_z_;

        int offset_z_;

        /// Reciprocal space range
        std::pair<int, int> grid_limits_[3];
        
        /// Backward transformation plan for each thread
        std::vector<fftw_plan> plan_backward_;

        std::vector<fftw_plan> plan_backward_z_;

        std::vector<fftw_plan> plan_backward_xy_;
        
        /// Forward transformation plan for each thread
        std::vector<fftw_plan> plan_forward_;
    
        /// In/out buffer for each thread
        std::vector<double_complex*> fftw_buffer_;

        std::vector<double_complex*> fftw_buffer_z_;
        std::vector<double_complex*> fftw_buffer_xy_;
        
        /// Execute backward transformation.
        inline void backward(int thread_id = 0)
        {    
            fftw_execute(plan_backward_[thread_id]);
        }
        
        /// Execute forward transformation.
        inline void forward(int thread_id = 0)
        {    
            fftw_execute(plan_forward_[thread_id]);
            double norm = 1.0 / size();
            for (int i = 0; i < local_size(); i++) fftw_buffer_[thread_id][i] *= norm;
        }

        /// Find smallest optimal grid size starting from n.
        int find_grid_size(int n)
        {
            while (true)
            {
                int m = n;
                for (int k = 2; k <= 5; k++)
                {
                    while (m % k == 0) m /= k;
                }
                if (m == 1) 
                {
                    return n;
                }
                else 
                {
                    n++;
                }
            }
        } 
        
    public:

        FFT3D(vector3d<int> dims__,
              int num_fft_threads__,
              int num_fft_workers__,
              Communicator const& comm__)
            : num_fft_workers_(num_fft_workers__),
              num_fft_threads_(num_fft_threads__),
              comm_(comm__)
        {
            PROFILE();

            Timer t("sirius::FFT3D<CPU>::FFT3D");
            for (int i = 0; i < 3; i++)
            {
                grid_size_[i] = find_grid_size(dims__[i]);
                
                grid_limits_[i].second = grid_size_[i] / 2;
                grid_limits_[i].first = grid_limits_[i].second - grid_size_[i] + 1;
            }

            fftw_plan_with_nthreads(num_fft_workers_);

            if (comm_.size() > 1 && num_fft_threads_ != 1) TERMINATE("distributed FFT can't be used inside multiple threads");

            size_t alloc_local_size = 0;
            if (comm_.size() > 1)
            {
                #ifdef __FFTW_MPI
                ptrdiff_t sz, offs;
                alloc_local_size = fftw_mpi_local_size_3d(size(2), size(1), size(0), comm__.mpi_comm(), &sz, &offs);

                local_size_z_ = (int)sz;
                offset_z_ = (int)offs;
                #else
                TERMINATE("not compiled with MPI support");
                #endif
            }
            else
            {
                alloc_local_size = size();
                local_size_z_ = size(2);
                offset_z_ = 0;
            }

            /* split z-direction */
            spl_z_ = splindex<block>(size(2), comm_.size(), comm_.rank());
            assert(spl_z_.local_size() == local_size_z_);

            data_slab_  = mdarray<double_complex, 2>(local_size_z_, size(0) * size(1));
            data_slice_ = mdarray<double_complex, 1>(size(2) * splindex_base::block_size(size(0) * size(1), comm_.size()));

            fftw_buffer_   = std::vector<double_complex*>(num_fft_threads_);
            plan_backward_ = std::vector<fftw_plan>(num_fft_threads_);
            plan_forward_  = std::vector<fftw_plan>(num_fft_threads_);

            for (int i = 0; i < num_fft_threads_; i++)
            {
                fftw_buffer_[i] = (double_complex*)fftw_malloc(alloc_local_size * sizeof(double_complex));

                if (comm_.size() > 1)
                {
                    #ifdef __FFTW_MPI
                    plan_backward_[i] = fftw_mpi_plan_dft_3d(size(2), size(1), size(0), 
                                                             (fftw_complex*)fftw_buffer_[i], 
                                                             (fftw_complex*)fftw_buffer_[i],
                                                             comm_.mpi_comm(), FFTW_BACKWARD, FFTW_ESTIMATE);

                    plan_forward_[i] = fftw_mpi_plan_dft_3d(size(2), size(1), size(0), 
                                                            (fftw_complex*)fftw_buffer_[i], 
                                                            (fftw_complex*)fftw_buffer_[i],
                                                            comm_.mpi_comm(), FFTW_FORWARD, FFTW_ESTIMATE);

                    #else
                    TERMINATE("not compiled with MPI support");
                    #endif
                }
                else
                {
                    plan_backward_[i] = fftw_plan_dft_3d(size(2), size(1), size(0), 
                                                         (fftw_complex*)fftw_buffer_[i], 
                                                         (fftw_complex*)fftw_buffer_[i], FFTW_BACKWARD, FFTW_ESTIMATE);

                    plan_forward_[i] = fftw_plan_dft_3d(size(2), size(1), size(0), 
                                                        (fftw_complex*)fftw_buffer_[i], 
                                                        (fftw_complex*)fftw_buffer_[i], FFTW_FORWARD, FFTW_ESTIMATE);
                }
            }
            fftw_plan_with_nthreads(1);



            fftw_buffer_z_    = std::vector<double_complex*>(num_fft_workers_);
            fftw_buffer_xy_   = std::vector<double_complex*>(num_fft_workers_);
            plan_backward_z_  = std::vector<fftw_plan>(num_fft_workers_);
            plan_backward_xy_ = std::vector<fftw_plan>(num_fft_workers_);

            for (int i = 0; i < num_fft_workers_; i++)
            {
                fftw_buffer_z_[i] = (double_complex*)fftw_malloc(size(2) * sizeof(double_complex));
                fftw_buffer_xy_[i] = (double_complex*)fftw_malloc(size(0) * size(1) * sizeof(double_complex));

                plan_backward_z_[i] = fftw_plan_dft_1d(size(2), (fftw_complex*)fftw_buffer_z_[i], 
                                                       (fftw_complex*)fftw_buffer_z_[i], FFTW_BACKWARD, FFTW_ESTIMATE);
                
                plan_backward_xy_[i] = fftw_plan_dft_2d(size(1), size(0), (fftw_complex*)fftw_buffer_xy_[i], 
                                                        (fftw_complex*)fftw_buffer_xy_[i], FFTW_BACKWARD, FFTW_ESTIMATE);
            }
        }

        ~FFT3D()
        {
            for (int i = 0; i < num_fft_threads_; i++)
            {
                fftw_free(fftw_buffer_[i]);
                fftw_destroy_plan(plan_backward_[i]);
                fftw_destroy_plan(plan_forward_[i]);
            }

            for (int i = 0; i < num_fft_workers_; i++)
            {
                fftw_free(fftw_buffer_z_[i]);
                fftw_free(fftw_buffer_xy_[i]);
                fftw_destroy_plan(plan_backward_z_[i]);
                fftw_destroy_plan(plan_backward_xy_[i]);
            }
        }

        /// Execute the transformation for a given thread.
        inline void transform(int direction, int thread_id = 0)
        {
            assert(thread_id < num_fft_threads_);

            switch (direction)
            {
                case 1:
                {
                    backward(thread_id);
                    break;
                }
                case -1:
                {
                    forward(thread_id);
                    break;
                }
                default:
                {
                    error_local(__FILE__, __LINE__, "wrong FFT direction");
                }
            }
        }

        void transform_custom(int direction, mdarray<int, 2> const& xy_mask__)
        {
            assert(direction == 1);

            int size_xy = size(0) * size(1);

            Timer t0("fft|load_col");
            int n = 0;
            for (int x = 0; x < size(0); x++)
            {
                for (int y = 0; y < size(1); y++)
                {
                    if (xy_mask__(x, y))
                    {
                        for (int z = 0; z < local_size_z_; z++)
                        {
                            data_slab_(z, n) = fftw_buffer_[0][x + y * size(0) + z * size_xy];
                        }
                        n++;
                    }
                }
            }
            t0.stop();

            Timer t1("fft|swap_col");

            splindex<block> spl_n(n, comm_.size(), comm_.rank());

            std::vector<int> sendcounts(comm_.size());
            std::vector<int> sdispls(comm_.size());
            std::vector<int> recvcounts(comm_.size());
            std::vector<int> rdispls(comm_.size());

            for (int rank = 0; rank < comm_.size(); rank++)
            {
                sendcounts[rank] = (int)spl_z_.local_size() * (int)spl_n.local_size(rank);
                sdispls[rank]    = (int)spl_z_.local_size() * (int)spl_n.global_offset(rank);

                recvcounts[rank] = (int)spl_n.local_size() * (int)spl_z_.local_size(rank);
                rdispls[rank]    = (int)spl_n.local_size() * (int)spl_z_.global_offset(rank);
            }

            comm_.alltoall(data_slab_.at<CPU>(), &sendcounts[0], &sdispls[0], 
                           data_slice_.at<CPU>(), &recvcounts[0], &rdispls[0]);

            t1.stop();

            Timer t2("fft|transform_z");
            #pragma omp parallel num_threads(num_fft_workers_)
            {
                int tid = omp_get_thread_num();
                
                #pragma omp for
                for (int i = 0; i < (int)spl_n.local_size(); i++)
                {
                    for (int rank = 0; rank < comm_.size(); rank++)
                    {
                        int lsz = (int)spl_z_.local_size(rank);

                        memcpy(&fftw_buffer_z_[tid][spl_z_.global_offset(rank)], 
                               &data_slice_(spl_z_.global_offset(rank) * spl_n.local_size() + i * lsz),
                               lsz * sizeof(double_complex));
                    }

                    fftw_execute(plan_backward_z_[tid]);

                    for (int rank = 0; rank < comm_.size(); rank++)
                    {
                        int lsz = (int)spl_z_.local_size(rank);

                        memcpy(&data_slice_(spl_z_.global_offset(rank) * spl_n.local_size() + i * lsz),
                               &fftw_buffer_z_[tid][spl_z_.global_offset(rank)], 
                               lsz * sizeof(double_complex));
                    }
                }
            }
            t2.stop();

            Timer t3("fft|unload_and_swap_col");
            comm_.alltoall(data_slice_.at<CPU>(), &recvcounts[0], &rdispls[0], 
                           data_slab_.at<CPU>(), &sendcounts[0], &sdispls[0]);

            n = 0;
            for (int x = 0; x < size(0); x++)
            {
                for (int y = 0; y < size(1); y++)
                {
                    if (xy_mask__(x, y))
                    {
                        for (int z = 0; z < local_size_z_; z++)
                        {
                            fftw_buffer_[0][x + y * size(0) + z * size_xy] = data_slab_(z, n);
                        }
                        n++;
                    }
                }
            }
            t3.stop();

            Timer t4("fft|transform_xy");
            #pragma omp parallel num_threads(num_fft_workers_)
            {
                int tid = omp_get_thread_num();
                
                #pragma omp for
                for (int i = 0; i < local_size_z_; i++)
                {
                    memcpy(fftw_buffer_xy_[tid], &fftw_buffer_[0][i * size_xy], sizeof(fftw_complex) * size_xy);
                    fftw_execute(plan_backward_xy_[tid]);
                    memcpy(&fftw_buffer_[0][i * size_xy], fftw_buffer_xy_[tid], sizeof(fftw_complex) * size_xy);
                }
            }
            t4.stop();
        }

        template<typename T>
        inline void input(int n, int const* map, T* data, int thread_id = 0)
        {
            assert(thread_id < num_fft_threads_);
            
            memset(fftw_buffer_[thread_id], 0, local_size() * sizeof(double_complex));
            for (int i = 0; i < n; i++) fftw_buffer_[thread_id][map[i]] = data[i];
        }

        template <typename T>
        inline void input(T* data__, int thread_id__ = 0)
        {
            assert(thread_id__ < num_fft_threads_);
            
            for (int i = 0; i < local_size(); i++) fftw_buffer_[thread_id__][i] = data__[i];
        }
        
        inline void output(double* data, int thread_id = 0)
        {
            assert(thread_id < num_fft_threads_);

            for (int i = 0; i < local_size(); i++) data[i] = std::real(fftw_buffer_[thread_id][i]);
        }
        
        inline void output(double_complex* data, int thread_id = 0)
        {
            assert(thread_id < num_fft_threads_);

            memcpy(data, fftw_buffer_[thread_id], local_size() * sizeof(double_complex));
        }
        
        inline void output(int n, int const* map, double_complex* data, int thread_id = 0)
        {
            assert(thread_id < num_fft_threads_);

            for (int i = 0; i < n; i++) data[i] = fftw_buffer_[thread_id][map[i]];
        }

        inline void output(int n, int const* map, double_complex* data, int thread_id, double alpha)
        {
            assert(thread_id < num_fft_threads_);

            for (int i = 0; i < n; i++) data[i] += alpha * fftw_buffer_[thread_id][map[i]];
        }
        
        inline const std::pair<int, int>& grid_limits(int idim)
        {
            return grid_limits_[idim];
        }

        /// Size of a given dimension.
        inline int size(int d) const
        {
            assert(d >= 0 && d < 3);
            return grid_size_[d]; 
        }

        /// Total size of the FFT grid.
        inline int size() const
        {
            return grid_size_[0] * grid_size_[1] * grid_size_[2]; 
        }

        inline int local_size() const
        {
            return grid_size_[0] * grid_size_[1] * local_size_z_;
        }

        inline int local_size_z() const
        {
            return local_size_z_;
        }

        inline int offset_z() const
        {
            return offset_z_;
        }

        /// Return linear index of a plane-wave harmonic with fractional coordinates (i0, i1, i2) inside fft buffer.
        inline int index(int i0, int i1, int i2) const
        {
            if (i0 < 0) i0 += grid_size_[0];
            if (i1 < 0) i1 += grid_size_[1];
            if (i2 < 0) i2 += grid_size_[2];

            return (i0 + i1 * grid_size_[0] + i2 * grid_size_[0] * grid_size_[1]);
        }

        /// Direct access to the fft buffer
        inline double_complex& buffer(int i, int thread_id = 0)
        {
            return fftw_buffer_[thread_id][i];
        }
        
        vector3d<int> grid_size() const
        {
            return vector3d<int>(grid_size_[0], grid_size_[1], grid_size_[2]);
        }

        inline vector3d<int> gvec_by_grid_pos(int i0__, int i1__, int i2__) const
        {
            if (i0__ > grid_limits_[0].second) i0__ -= grid_size_[0];
            if (i1__ > grid_limits_[1].second) i1__ -= grid_size_[1];
            if (i2__ > grid_limits_[2].second) i2__ -= grid_size_[2];

            return vector3d<int>(i0__, i1__, i2__);
        }

        Communicator const& comm() const
        {
            return comm_;
        }

        inline bool parallel() const
        {
            return (comm_.size() != 1);
        }
};

class Gvec
{
    private:
        
        FFT3D<CPU>* fft_;

        matrix3d<double> lattice_vectors_;

        /// Total number of G-vectors.
        int num_gvec_;

        /// Local number of G-vectors for a given slab.
        int num_gvec_loc_;
        
        /// Offset of local part of G-vectors in the global index array.
        int gvec_offset_;
        
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

        mdarray<int, 2> xy_mask_;

        Gvec(Gvec const& src__) = delete;

        Gvec& operator=(Gvec const& src__) = delete;

    public:

        Gvec() : fft_(nullptr)
        {
        }

        Gvec(vector3d<double> q__,
             double Gmax__,
             matrix3d<double> const& M__,
             FFT3D<CPU>* fft__,
             bool build_reverse_mapping__)
            : fft_(fft__),
              lattice_vectors_(M__)
        {
            xy_mask_ = mdarray<int, 2>(fft_->size(0), fft_->size(1));
            xy_mask_.zero();

            /* find local number of G-vectors for each slab of FFT buffer;
             * at the same time, create the xy mask that tells which columns in 3d buffer are zero when
             * wave-functions are loaded into buffer */
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
                            xy_mask_(i, j) = 1;
                        }
                    }
                }
            }
            fft_->comm().allreduce<int, op_max>(xy_mask_.at<CPU>(), (int)xy_mask_.size());

            /* get total number of G-vectors */
            num_gvec_loc_ = (int)pos.size();
            num_gvec_ = num_gvec_loc_;
            fft_->comm().allreduce(&num_gvec_, 1);

            gvec_full_index_ = mdarray<int, 1>(num_gvec_);
            index_map_local_to_local_ = mdarray<int, 1>(num_gvec_loc_);

            gvec_counts_ = std::vector<int>(fft_->comm().size(), 0);
            gvec_offsets_ = std::vector<int>(fft_->comm().size(), 0);

            /* get local sizes from all ranks */
            gvec_counts_[fft_->comm().rank()] = num_gvec_loc_;
            fft_->comm().allreduce(&gvec_counts_[0], fft_->comm().size());

            for (int i = 1; i < fft_->comm().size(); i++) 
                gvec_offsets_[i] = gvec_offsets_[i - 1] + gvec_counts_[i - 1]; 
            gvec_offset_ = gvec_offsets_[fft_->comm().rank()];

            for (int igloc = 0; igloc < num_gvec_loc_; igloc++)
            {
                auto p = pos[igloc];
                index_map_local_to_local_(igloc) = p[0] + p[1] * fft_->size(0) + p[2] * fft_->size(0) * fft_->size(1);
                
                /* this is only one way to pack coordinates into single integer */
                gvec_full_index_(gvec_offset_ + igloc) = 
                    p[0] + p[1] * fft_->size(0) + (p[2] + fft_->offset_z()) * fft_->size(0) * fft_->size(1);
            }

            fft_->comm().allgather(&gvec_full_index_(0), gvec_offset_, num_gvec_loc_); 

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
        }

        Gvec& operator=(Gvec&& src__) = default;

        /// Return number of G-vectors within the cutoff.
        inline int num_gvec() const
        {
            return num_gvec_;
        }

        inline int num_gvec_loc() const
        {
            return num_gvec_loc_;
        }

        inline int gvec_offset() const
        {
            return gvec_offset_;
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

        mdarray<int, 2> const& xy_mask() const
        {
            return xy_mask_;
        }
};

