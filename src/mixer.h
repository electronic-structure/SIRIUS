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

/** \file mixer.h
 *   
 *   \brief Contains definition and implementation of sirius::Mixer, sirius::Linear_mixer, sirius::Broyden1 and 
 *          sirius::Broyden2 classes.
 */    

#ifndef __MIXER_H__
#define __MIXER_H__

namespace sirius
{

/// Abstract mixer
template <typename T>
class Mixer
{
    protected:
        
        /// Size of the mixed vectors.
        size_t size_;
        
        /// Split size of the vectors beteen all MPI ranks.
        splindex<block, size_t> spl_size_;
        
        /// Maximum number of stored vectors.
        int max_history_;

        /// Linear mixing factor
        double beta_;
        
        /// Number of times mixer was called so far.
        int count_;
        
        /// Temporary storage for the input data.
        mdarray<T, 1> input_buffer_;
        
        /// History of previous vectors.
        mdarray<T, 2> vectors_;

        /// Output buffer for the whole vector.
        mdarray<T, 1> output_buffer_;
        
        /// Base communicator.
        Communicator const& comm_;

        /// Residual sum of squares.
        double rss_;

        /// Return position in the list of mixed vectors for the given mixing step.
        inline int idx_hist(int step__) const
        {
            assert(step__ >= 0);
            return step__ % max_history_;
        }

        /// Compute RMS deviation between current vector and input vector.
        double rms_deviation() const
        {
            double rms = 0.0;
            if (size_ != 0)
            {
                int ipos = idx_hist(count_); 

                for (size_t i = 0; i < spl_size_.local_size(); i++)
                    rms += std::pow(std::abs(vectors_(i, ipos) - input_buffer_(i)), 2);

                comm_.allreduce(&rms, 1);
                rms = std::sqrt(rms / double(size_));
            }
            return rms;
        }
        
        /// Mix input buffer and previous vector and store result in the current vector.
        void mix_linear(double beta__)
        {
            int i1 = idx_hist(count_); 
            int i2 = idx_hist(count_ - 1); 

            for (size_t i = 0; i < spl_size_.local_size(); i++)
                vectors_(i, i1) = beta__ * input_buffer_(i) + (1 - beta__) * vectors_(i, i2);

            comm_.allgather(&vectors_(0, i1), output_buffer_.template at<CPU>(), (int)spl_size_.global_offset(), 
                            (int)spl_size_.local_size());
        }

    public:

        Mixer(size_t size__, int max_history__, double beta__, Communicator const& comm__)
            : size_(size__), 
              max_history_(max_history__), 
              beta_(beta__), 
              count_(0),
              comm_(comm__),
              rss_(0)
        {
            spl_size_ = splindex<block, size_t>(size_, comm_.size(), comm_.rank());
            /* allocate input buffer (local size) */
            input_buffer_ = mdarray<T, 1>(spl_size_.local_size());
            /* allocate output bffer (global size) */
            output_buffer_ = mdarray<T, 1>(size_);
            /* allocate storage for previous vectors (local size) */
            vectors_ = mdarray<T, 2>(spl_size_.local_size(), max_history_);
        }

        virtual ~Mixer()
        {
        }

        void input(size_t idx, T value)
        {
            assert(idx < size_t(1 << 31));
            assert(idx >= 0 && idx < size_);

            auto offs_and_rank = spl_size_.location(idx);
            if (offs_and_rank.second == comm_.rank()) input_buffer_(offs_and_rank.first) = value;
        }

        inline T const* output_buffer() const
        {
            return output_buffer_.template at<CPU>();
        }

        inline T output_buffer(int idx) const
        {
            return output_buffer_(idx);
        }

        inline void initialize()
        {
            memcpy(&vectors_(0, 0), &input_buffer_(0), spl_size_.local_size() * sizeof(T));
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
        Linear_mixer(size_t size__, double beta0__, Communicator const& comm__) 
            : Mixer<T>(size__, 2, beta0__, comm__),
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

        std::vector<double> weights_;

    public:

        Broyden1(size_t size__,
                 int max_history__,
                 double beta__,
                 std::vector<double>& weights__,
                 Communicator const& comm__) 
            : Mixer<T>(size__, max_history__, beta__, comm__)
        {
            residuals_ = mdarray<T, 2>(this->spl_size_.local_size(), max_history__);
            weights_ = weights__;
        }

        double mix()
        {
            runtime::Timer t("sirius::Broyden1::mix");
            
            //== /* weights as a functor */
            //== struct w_functor
            //== {
            //==     std::vector<double> const& weights_;
            //==     typedef double (w_functor::*fptr_t)(size_t);
            //==     fptr_t fptr_;
            //==     w_functor(std::vector<double> const& weights__) : weights_(weights__)
            //==     {
            //==         fptr_ = (weights_.size()) ? (&w_functor::f1) : (&w_functor::f2);
            //==     }
            //==     inline double f1(size_t idx__)
            //==     {
            //==         return weights_[idx__];
            //==     }
            //==     inline double f2(size_t idx__)
            //==     {
            //==         return 1.0;
            //==     }
            //==     inline double operator()(size_t idx__)
            //==     {
            //==         return (this->*fptr_)(idx__);
            //==     }
            //== };
            //== w_functor w(weights_);
            
            /* weights as a lambda function */
            auto w = [this](size_t idx)
            {
                return (this->weights_.size()) ? weights_[idx] : 1.0;
            };

            /* current position in history */
            int ipos = this->idx_hist(this->count_);

            /* compute residual square sum */
            this->rss_ = 0;
            for (size_t i = 0; i < this->spl_size_.local_size(); i++) 
            {
                residuals_(i, ipos) = this->input_buffer_(i) - this->vectors_(i, ipos);
                this->rss_ += std::pow(std::abs(residuals_(i, ipos)), 2) * w(this->spl_size_[i]);
            }
            this->comm_.allreduce(&this->rss_, 1);

            /* exit if the vector has converged */
            if (this->rss_ < 1e-11) return 0.0;

            double rms = this->rms_deviation();
            
            /* number of previous vectors */
            int N = std::min(this->count_, this->max_history_ - 1);
            
            if (N > 0)
            {
                mdarray<double, 2> S(N, N);
                S.zero();
                for (int j1 = 0; j1 < N; j1++)
                {
                    int i1 = this->idx_hist(this->count_ - j1);
                    int i2 = this->idx_hist(this->count_ - j1 - 1);
                    for (int j2 = 0; j2 <= j1; j2++)
                    {
                        int i3 = this->idx_hist(this->count_ - j2);
                        int i4 = this->idx_hist(this->count_ - j2 - 1);
                        for (size_t i = 0; i < this->spl_size_.local_size(); i++) 
                        {
                            T dr1 = residuals_(i, i1) - residuals_(i, i2);
                            T dr2 = residuals_(i, i3) - residuals_(i, i4);

                            S(j1, j2) += type_wrapper<double>::sift(type_wrapper<T>::conjugate(dr1) * dr2) * w(this->spl_size_[i]);
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
                for (int j1 = 0; j1 < N; j1++)
                {
                    for (int j2 = 0; j2 < j1; j2++) S(j1, j2) = S(j2, j1);
                }

                // printf("[mixer] S^{-1} matrix\n");
                // for (int i = 0; i < N; i++)
                // {
                //     for (int j = 0; j < N; j++) printf("%18.10f ", S(i, j));
                //     printf("\n");
                // }

                mdarray<double, 1> c(N);
                c.zero();
                for (int j = 0; j < N; j++)
                {
                    int i1 = this->idx_hist(this->count_ - j);
                    int i2 = this->idx_hist(this->count_ - j - 1);
                    for (size_t i = 0; i < this->spl_size_.local_size(); i++) 
                    {
                        T dr = residuals_(i, i1) - residuals_(i, i2);
                        c(j) += type_wrapper<double>::sift(type_wrapper<T>::conjugate(dr) * residuals_(i, ipos) * w(this->spl_size_[i]));
                    }
                }
                this->comm_.allreduce(c.at<CPU>(), (int)c.size());

                /* store new vector in the input buffer */
                this->input_buffer_.zero();

                for (int j = 0; j < N; j++)
                {
                    double gamma = 0;
                    for (int i = 0; i < N; i++) gamma += c(i) * S(i, j);

                    int i1 = this->idx_hist(this->count_ - j);
                    int i2 = this->idx_hist(this->count_ - j - 1);
                
                    for (size_t i = 0; i < this->spl_size_.local_size(); i++)
                    {
                        T dr = residuals_(i, i1) - residuals_(i, i2);
                        T dv = this->vectors_(i, i1) - this->vectors_(i, i2);

                        this->input_buffer_(i) -= gamma * (dr * this->beta_ + dv);
                    }
                }
            }
            int i1 = this->idx_hist(this->count_ + 1);
            /* linear part */
            for (size_t i = 0; i < this->spl_size_.local_size(); i++)
            {
                this->vectors_(i, i1) = this->vectors_(i, ipos) + this->beta_ * residuals_(i, ipos) + this->input_buffer_(i);
            }

            this->comm_.allgather(&this->vectors_(0, i1), this->output_buffer_.template at<CPU>(),
                                  (int)this->spl_size_.global_offset(), (int)this->spl_size_.local_size());
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

        std::vector<double> weights_;

        mdarray<T, 2> residuals_;
    
    public:

        Broyden2(size_t size__,
                 int max_history__,
                 double beta__,
                 std::vector<double>& weights__,
                 Communicator const& comm__) 
            : Mixer<T>(size__, max_history__, beta__, comm__)
        {
            weights_ = weights__;
            residuals_ = mdarray<T, 2>(this->spl_size_.local_size(), max_history__);
        }

        double mix()
        {
            runtime::Timer t("sirius::Broyden2::mix");

            /* weights as a lambda function */
            auto w = [this](size_t idx)
            {
                return (this->weights_.size()) ? weights_[idx] : 1.0;
            };

            /* current position in history */
            int ipos = this->idx_hist(this->count_);

            /* compute residual square sum */
            this->rss_ = 0;
            for (size_t i = 0; i < this->spl_size_.local_size(); i++) 
            {
                /* curent residual f_k = x_k - g(x_k) */
                residuals_(i, ipos) = this->vectors_(i, ipos) - this->input_buffer_(i);
                this->rss_ += std::pow(std::abs(residuals_(i, ipos)), 2) * w(this->spl_size_[i]);
            }
            this->comm_.allreduce(&this->rss_, 1);

            /* exit if the vector has converged */
            if (this->rss_ < 1e-11) return 0.0;

            double rms = this->rms_deviation();

            /* increment the history step */
            this->count_++;

            /* at this point we have min(count_, max_history_) residuals and vectors from the previous iterations */
            int N = std::min(this->count_, this->max_history_);

            if (N > 1)
            {
                mdarray<long double, 2> S(N, N);
                S.zero();
                /* S = F^T * F, where F is the matrix of residual vectors */
                for (int j1 = 0; j1 < N; j1++)
                {
                    int i1 = this->idx_hist(this->count_ - N + j1);
                    for (int j2 = 0; j2 <= j1; j2++)
                    {
                        int i2 = this->idx_hist(this->count_ - N + j2);
                        for (size_t i = 0; i < this->spl_size_.local_size(); i++) 
                        {
                            S(j1, j2) += type_wrapper<double>::sift(type_wrapper<T>::conjugate(residuals_(i, i1)) * residuals_(i, i2));
                        }
                        S(j2, j1) = S(j1, j2);
                    }
                }
                this->comm_.allreduce(S.at<CPU>(), (int)S.size());
                for (int j1 = 0; j1 < N; j1++)
                { 
                    for (int j2 = 0; j2 < N; j2++) S(j1, j2) /= this->size_;
                }
               
                mdarray<long double, 2> gamma_k(2 * N, N);
                gamma_k.zero();
                /* initial gamma_0 */
                for (int i = 0; i < N; i++) gamma_k(i, i) = 0.25;

                std::vector<long double> v1(N);
                std::vector<long double> v2(2 * N);
                
                /* update gamma_k by recursion */
                for (int k = 0; k < N - 1; k++)
                {
                    /* denominator df_k^{T} S df_k */
                    long double d = S(k, k) + S(k + 1, k + 1) - S(k, k + 1) - S(k + 1, k);
                    /* nominator */
                    memset(&v1[0], 0, N * sizeof(long double));
                    for (int j = 0; j < N; j++) v1[j] = S(k + 1, j) - S(k, j);

                    memset(&v2[0], 0, 2 * N * sizeof(long double));
                    for (int j = 0; j < 2 * N; j++) v2[j] = -(gamma_k(j, k + 1) - gamma_k(j, k));
                    v2[N + k] -= 1;
                    v2[N + k + 1] += 1;

                    for (int j1 = 0; j1 < N; j1++)
                    {
                        for (int j2 = 0; j2 < 2 * N; j2++) gamma_k(j2, j1) += v2[j2] * v1[j1] / d;
                    }
                }
 
                std::memset(&v2[0], 0, 2 * N * sizeof(long double));
                for (int j = 0; j < 2 * N; j++) v2[j] = -gamma_k(j, N - 1);
                v2[2 * N - 1] += 1;
                
                /* store new vector in the input buffer */
                this->input_buffer_.zero();

                /* make linear combination of vectors and residuals; this is the update vector \tilda x */
                for (int j = 0; j < N; j++)
                {
                    int i1 = this->idx_hist(this->count_ - N + j);
                    for (size_t i = 0; i < this->spl_size_.local_size(); i++) 
                        this->input_buffer_(i) += ((double)v2[j] * residuals_(i, i1) + (double)v2[j + N] * this->vectors_(i, i1));
                }
                /* mix last vector with the update vector \tilda x */
                this->mix_linear(this->beta_);
            }
            else
            {
                this->mix_linear(0.05);
            }

            return rms;
        }
};

}

#endif // __MIXER_H__

