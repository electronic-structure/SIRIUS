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

/** \file mixer.h
 *   
 *   \brief Contains definition and implementation of sirius::Mixer, sirius::Linear_mixer, sirius::Broyden1 and 
 *          sirius::Broyden2 classes.
 */    

#ifndef __MIXER_H__
#define __MIXER_H__

namespace sirius {

/// Abstract mixer
template <typename T>
class Mixer
{
    protected:

        /// Size of the vector which is global to (in other words, shared between) all MPI ranks.
        int shared_vector_size_;
        
        /// Size of the vector which is local to MPI rank.
        int local_vector_size_;

        /// Split shared vector size beteen all MPI ranks.
        splindex<block, int> spl_shared_size_;

        /// Local size of shared vector.
        int spl_shared_local_size_{0};

        /// Local number of vector elements.
        /** The local number of elements is a sum of local vector size and local size of shared vector. */
        int local_size_;
        
        /// Total number of vector elements.
        /** The total number of vector elements is the sum of shared vector size and all local vector sizes. */
        size_t total_size_;

        /// Maximum number of stored vectors.
        int max_history_;

        /// Linear mixing factor
        double beta_;
        
        /// Number of times mixer was called so far.
        int count_{0};

        /// Weights of vector elements.
        /** Weights are used in Broyden-type mixers when the inner product of residuals is computed */
        mdarray<double, 1> weights_;
        
        /// Storage for the input (unmixed) data.
        mdarray<T, 1> input_buffer_;
        
        /// History of previous vectors.
        mdarray<T, 2> vectors_;

        /// Output buffer for the shared (global) part of the vector.
        mdarray<T, 1> output_buffer_;
        
        /// Base communicator.
        Communicator const& comm_;

        /// Residual sum of squares.
        double rss_{0};

        /// Return position in the list of mixed vectors for the given mixing step.
        inline int idx_hist(int step__) const
        {
            assert(step__ >= 0);
            return step__ % max_history_;
        }

        /// Compute RMS deviation between current vector and input vector.
        double rms_deviation() const
        {
            double rms{0};
            int ipos = idx_hist(count_); 

            for (int i = 0; i < local_size_; i++) {
                rms += std::pow(std::abs(vectors_(i, ipos) - input_buffer_(i)), 2);
            }

            comm_.allreduce(&rms, 1);
            rms = std::sqrt(rms / double(total_size_));
            return rms;
        }
        
        /// Mix input buffer and previous vector and store result in the current vector.
        void mix_linear(double beta__)
        {
            int ipos = idx_hist(count_); 
            int ipos1 = idx_hist(count_ - 1); 

            for (int i = 0; i < local_size_; i++) {
                vectors_(i, ipos) = beta__ * input_buffer_(i) + (1 - beta__) * vectors_(i, ipos1);
            }

            /* collect shared data */
            comm_.allgather(&vectors_(0, ipos), output_buffer_.template at<CPU>(), spl_shared_size_.global_offset(), 
                            spl_shared_size_.local_size());
        }

    public:

        Mixer(int                 shared_vector_size__,
              int                 local_vector_size__,
              int                 max_history__,
              double              beta__,
              Communicator const& comm__)
            : shared_vector_size_(shared_vector_size__)
            , local_vector_size_(local_vector_size__)
            , max_history_(max_history__)
            , beta_(beta__)
            , comm_(comm__)
        {
            assert(shared_vector_size__ >= 0);
            assert(local_vector_size__ >= 0);

            unsigned long long n = local_vector_size__;
            comm_.allreduce(&n, 1);
            total_size_ = n + shared_vector_size_;

            spl_shared_size_ = splindex<block>(shared_vector_size_, comm_.size(), comm_.rank());
            if (shared_vector_size_) {
                spl_shared_local_size_ = spl_shared_size_.local_size();
            }
            local_size_ = spl_shared_local_size_ + local_vector_size_;
            if (local_size_ == 0) {
                TERMINATE("Ratio between gk_cutoff and pw_cutoff is exactly 2\n");
            }
            /* allocate input buffer */
            input_buffer_ = mdarray<T, 1>(local_size_, memory_t::host, "Mixer::input_buffer_");
            /* allocate output bffer */
            output_buffer_ = mdarray<T, 1>(shared_vector_size_, memory_t::host, "Mixer::output_buffer_");
            /* allocate storage for previous vectors */
            vectors_ = mdarray<T, 2>(local_size_, max_history_, memory_t::host, "Mixer::vectors_");
            /* allocate weights */
            weights_ = mdarray<double, 1>(local_size_, memory_t::host, "Mixer::weights_");
            weights_.zero();
        }

        virtual ~Mixer()
        {
        }

        void input_shared(int idx__, T value__, double w__ = 1.0)
        {
            assert(idx__ >= 0 && idx__ < shared_vector_size_);

            auto offs_and_rank = spl_shared_size_.location(idx__);
            if (offs_and_rank.rank == comm_.rank()) {
                input_buffer_(offs_and_rank.local_index) = value__;
                weights_(offs_and_rank.local_index) = w__;
            }
        }

        void input_local(int idx__, T value__, double w__ = 1.0)
        {
            assert(idx__ >= 0 && idx__ < local_vector_size_);

            input_buffer_(spl_shared_local_size_ + idx__) = value__;
            weights_(spl_shared_local_size_ + idx__) = w__;
        }

        inline T output_shared(int idx) const
        {
            return output_buffer_(idx);
        }

        inline T output_local(int idx) const
        {
            int ipos = idx_hist(count_);
            return vectors_(spl_shared_local_size_ + idx, ipos);
        }
        
        /// Initialize the mixer.
        /** Copy content of the input buffer into first vector of the mixing history. */
        inline void initialize()
        {
            std::memcpy(&vectors_(0, 0), &input_buffer_(0), local_size_ * sizeof(T));
        }

        inline double beta() const
        {
            return beta_;
        }

        inline double rss() const
        {
            return rss_;
        }
            
        virtual double mix() = 0;
};

/// Primitive linear mixer.
template <typename T>
class Linear_mixer: public Mixer<T>
{
    private:
        
        double beta0_;

    public:
        
        /// Constructor
        Linear_mixer(int                 shared_vector_size__,
                     int                 local_vector_size__,
                     double              beta0__,
                     Communicator const& comm__) 
            : Mixer<T>(shared_vector_size__, local_vector_size__, 2, beta0__, comm__),
              beta0_(beta0__)
        {
        }

        double mix()
        {
            double rms = this->rms_deviation();
            this->count_++;
            this->mix_linear(this->beta_);
            return rms;
        }
};

/// Broyden mixer.
/** First version of the Broyden mixer, which requres inversion of the Jacobian matrix.
 *  Reference paper: "Robust acceleration of self consistent field calculations for 
 *  density functional theory", Baarman K, Eirola T, Havu V., J Chem Phys. 134, 134109 (2011)
 */
template <typename T>
class Broyden1: public Mixer<T>
{
    private:

        mdarray<T, 2> residuals_;

    public:

        Broyden1(int                 shared_vector_size__,
                 int                 local_vector_size__,
                 int                 max_history__,
                 double              beta__,
                 Communicator const& comm__) 
            : Mixer<T>(shared_vector_size__, local_vector_size__, max_history__, beta__, comm__)
        {
            residuals_ = mdarray<T, 2>(this->local_size_, max_history__);
        }

        double mix()
        {
            PROFILE("sirius::Broyden1::mix");

            /* current position in history */
            int ipos = this->idx_hist(this->count_);

            /* compute residual square sum */
            this->rss_ = 0;
            for (int i = 0; i < this->local_size_; i++) {
                residuals_(i, ipos) = this->input_buffer_(i) - this->vectors_(i, ipos);
                this->rss_ += std::pow(std::abs(residuals_(i, ipos)), 2) * this->weights_(i);
            }
            this->comm_.allreduce(&this->rss_, 1);

            /* exit if the vector has converged */
            if (this->rss_ < 1e-16) {
                int i1 = this->idx_hist(this->count_);
                /* copy input to output */
                for (int i = 0; i < this->local_size_; i++) {
                    this->vectors_(i, i1) = this->input_buffer_(i);
                }

                this->comm_.allgather(&this->vectors_(0, i1), this->output_buffer_.template at<CPU>(),
                                      this->spl_shared_size_.global_offset(), this->spl_shared_size_.local_size());
                return 0.0;
            }

            double rms = this->rms_deviation();
            
            /* number of previous vectors */
            int N = std::min(this->count_, this->max_history_ - 1);
            
            /* new vector will be stored in the input buffer */
            this->input_buffer_.zero();

            if (N > 0) {
                mdarray<double, 2> S(N, N);
                S.zero();
                for (int j1 = 0; j1 < N; j1++) {
                    int i1 = this->idx_hist(this->count_ - j1);
                    int i2 = this->idx_hist(this->count_ - j1 - 1);
                    for (int j2 = 0; j2 <= j1; j2++) {
                        int i3 = this->idx_hist(this->count_ - j2);
                        int i4 = this->idx_hist(this->count_ - j2 - 1);
                        for (int i = 0; i < this->local_size_; i++) {
                            T dr1 = residuals_(i, i1) - residuals_(i, i2);
                            T dr2 = residuals_(i, i3) - residuals_(i, i4);

                            S(j1, j2) += std::real(std::conj(dr1) * dr2) * this->weights_(i);
                        }
                        S(j2, j1) = S(j1, j2);
                    }
                }
                this->comm_.allreduce(S.at<CPU>(), (int)S.size());

                // printf("[mixer] S matrix\n");
                // for (int i = 0; i < N; i++)
                // {
                //     for (int j = 0; j < N; j++) printf("%18.10f ", S(i, j));
                //     printf("\n");
                // }

                /* invert matrix */
                linalg<CPU>::syinv(N, S);
                /* restore lower triangular part */
                for (int j1 = 0; j1 < N; j1++) {
                    for (int j2 = 0; j2 < j1; j2++) {
                        S(j1, j2) = S(j2, j1);
                    }
                }

                // printf("[mixer] S^{-1} matrix\n");
                // for (int i = 0; i < N; i++)
                // {
                //     for (int j = 0; j < N; j++) printf("%18.10f ", S(i, j));
                //     printf("\n");
                // }

                mdarray<double, 1> c(N);
                c.zero();
                for (int j = 0; j < N; j++) {
                    int i1 = this->idx_hist(this->count_ - j);
                    int i2 = this->idx_hist(this->count_ - j - 1);
                    for (int i = 0; i < this->local_size_; i++) {
                        T dr = residuals_(i, i1) - residuals_(i, i2);
                        c(j) += std::real(std::conj(dr) * residuals_(i, ipos)) * this->weights_(i);
                    }
                }
                this->comm_.allreduce(c.at<CPU>(), (int)c.size());

                for (int j = 0; j < N; j++) {
                    double gamma = 0;
                    for (int i = 0; i < N; i++) {
                        gamma += c(i) * S(i, j);
                    }

                    int i1 = this->idx_hist(this->count_ - j);
                    int i2 = this->idx_hist(this->count_ - j - 1);
                
                    for (int i = 0; i < this->local_size_; i++) {
                        T dr = residuals_(i, i1) - residuals_(i, i2);
                        T dv = this->vectors_(i, i1) - this->vectors_(i, i2);

                        this->input_buffer_(i) -= gamma * (dr * this->beta_ + dv);
                    }
                }
            }

            int i1 = this->idx_hist(this->count_ + 1);
            /* linear part */
            for (int i = 0; i < this->local_size_; i++) {
                this->vectors_(i, i1) = this->vectors_(i, ipos) + this->beta_ * residuals_(i, ipos) + this->input_buffer_(i);
            }

            this->comm_.allgather(&this->vectors_(0, i1), this->output_buffer_.template at<CPU>(),
                                  this->spl_shared_size_.global_offset(), this->spl_shared_size_.local_size());
            
            /* increment the history step */
            this->count_++;

            return rms;
        }
};

/// Broyden mixer.
/** Second version of the Broyden mixer, which doesn't requre inversion of the Jacobian matrix.
 *  Reference paper: "Robust acceleration of self consistent field calculations for 
 *  density functional theory", Baarman K, Eirola T, Havu V., J Chem Phys. 134, 134109 (2011)
 */
template <typename T>
class Broyden2: public Mixer<T>
{
    private:
        
        double beta0_;
        double linear_mix_rms_tol_;

        mdarray<T, 2> residuals_;
    
    public:

        Broyden2(int                 shared_vector_size__,
                 int                 local_vector_size__,
                 int                 max_history__,
                 double              beta__,
                 double              beta0__,
                 double              linear_mix_rms_tol__,
                 Communicator const& comm__) 
            : Mixer<T>(shared_vector_size__, local_vector_size__, max_history__, beta__, comm__),
              beta0_(beta0__),
              linear_mix_rms_tol_(linear_mix_rms_tol__)
        {
            residuals_ = mdarray<T, 2>(this->local_size_, max_history__);
        }

        double mix()
        {
            PROFILE("sirius::Broyden2::mix");

            /* current position in history */
            int ipos = this->idx_hist(this->count_);

            /* compute residual square sum */
            this->rss_ = 0;
            for (int i = 0; i < this->local_size_; i++) 
            {
                /* curent residual f_k = x_k - g(x_k) */
                residuals_(i, ipos) = this->vectors_(i, ipos) - this->input_buffer_(i);
                this->rss_ += std::pow(std::abs(residuals_(i, ipos)), 2) * this->weights_(i);
            }
            this->comm_.allreduce(&this->rss_, 1);

            /* exit if the vector has converged */
            if (this->rss_ < 1e-11) {
                return 0.0;
            }

            double rms = this->rms_deviation();

            /* increment the history step */
            this->count_++;

            /* at this point we have min(count_, max_history_) residuals and vectors from the previous iterations */
            int N = std::min(this->count_, this->max_history_);

            if ((linear_mix_rms_tol_ > 0 && rms < linear_mix_rms_tol_ && N > 1) || 
                (linear_mix_rms_tol_ <= 0 && this->count_ > this->max_history_)) {
                mdarray<long double, 2> S(N, N);
                S.zero();
                /* S = F^T * F, where F is the matrix of residual vectors */
                for (int j1 = 0; j1 < N; j1++) {
                    int i1 = this->idx_hist(this->count_ - N + j1);
                    for (int j2 = 0; j2 <= j1; j2++) {
                        int i2 = this->idx_hist(this->count_ - N + j2);
                        for (int i = 0; i < this->local_size_; i++) {
                            S(j1, j2) += std::real(std::conj(residuals_(i, i1)) * residuals_(i, i2));
                        }
                        S(j2, j1) = S(j1, j2);
                    }
                }
                this->comm_.allreduce(S.at<CPU>(), (int)S.size());
                for (int j1 = 0; j1 < N; j1++) { 
                    for (int j2 = 0; j2 < N; j2++) {
                        S(j1, j2) /= this->total_size_;
                    }
                }
               
                mdarray<long double, 2> gamma_k(2 * N, N);
                gamma_k.zero();
                /* initial gamma_0 */
                for (int i = 0; i < N; i++) gamma_k(i, i) = 0.25;

                std::vector<long double> v1(N);
                std::vector<long double> v2(2 * N);
                
                /* update gamma_k by recursion */
                for (int k = 0; k < N - 1; k++) {
                    /* denominator df_k^{T} S df_k */
                    long double d = S(k, k) + S(k + 1, k + 1) - S(k, k + 1) - S(k + 1, k);
                    /* nominator */
                    std::memset(&v1[0], 0, N * sizeof(long double));
                    for (int j = 0; j < N; j++) {
                        v1[j] = S(k + 1, j) - S(k, j);
                    }

                    std::memset(&v2[0], 0, 2 * N * sizeof(long double));
                    for (int j = 0; j < 2 * N; j++) {
                        v2[j] = -(gamma_k(j, k + 1) - gamma_k(j, k));
                    }
                    v2[N + k] -= 1;
                    v2[N + k + 1] += 1;

                    for (int j1 = 0; j1 < N; j1++) {
                        for (int j2 = 0; j2 < 2 * N; j2++) {
                            gamma_k(j2, j1) += v2[j2] * v1[j1] / d;
                        }
                    }
                }
 
                std::memset(&v2[0], 0, 2 * N * sizeof(long double));
                for (int j = 0; j < 2 * N; j++) {
                    v2[j] = -gamma_k(j, N - 1);
                }
                v2[2 * N - 1] += 1;
                
                /* store new vector in the input buffer */
                this->input_buffer_.zero();

                /* make linear combination of vectors and residuals; this is the update vector \tilda x */
                for (int j = 0; j < N; j++) {
                    int i1 = this->idx_hist(this->count_ - N + j);
                    for (int i = 0; i < this->local_size_; i++) {
                        this->input_buffer_(i) += ((double)v2[j] * residuals_(i, i1) + (double)v2[j + N] * this->vectors_(i, i1));
                    }
                }
                /* mix last vector with the update vector \tilda x */
                this->mix_linear(this->beta_);
            }
            else {
                this->mix_linear(beta0_);
            }

            return rms;
        }
};

template <typename T>
inline std::unique_ptr<Mixer<T>> Mixer_factory(std::string  const& type__,
                                               int                 shared_size__,
                                               int                 local_size__,
                                               Mixer_input         mix_cfg__,
                                               Communicator const& comm__)
{
    std::unique_ptr<Mixer<T>> mixer;

    if (type__ == "linear") {
        mixer = std::unique_ptr<Mixer<T>>(new Linear_mixer<T>(shared_size__, local_size__, mix_cfg__.beta_, comm__));
    } else if (type__ == "broyden1") {
        mixer = std::unique_ptr<Mixer<T>>(new Broyden1<T>(shared_size__, local_size__, mix_cfg__.max_history_, mix_cfg__.beta_,
                                                          comm__));
    }
    else if (type__ == "broyden2") {
        mixer = std::unique_ptr<Mixer<T>>(new Broyden2<T>(shared_size__, local_size__, mix_cfg__.max_history_, mix_cfg__.beta_,
                                                          mix_cfg__.beta0_, mix_cfg__.linear_mix_rms_tol_,
                                                          comm__));
    } else {
        TERMINATE("wrong type of mixer");
    }
    return std::move(mixer);
}

}

#endif // __MIXER_H__

