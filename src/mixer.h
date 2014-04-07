#ifndef __MIXER_H__
#define __MIXER_H__

/** \file mixer.h
    
    \brief Contains definition of and implementation of sirius::Mixer, sirius::Linear_mixer and sirius::Broyden_mixer clases.
*/    
namespace sirius
{

/// Abstract mixer
class Mixer
{
    protected:
        
        /// size of the mixed vectors
        size_t size_;
        
        /// split the size of the vectors beteen all MPI ranks
        splindex<block> spl_size_;
        
        /// maximum number of stored vectors
        int max_history_;

        /// mixing factor
        double beta_;
        
        /// number of times mixer was called so far
        int count_;
        
        /// temporary storage for the input data
        mdarray<double, 1> input_buffer_;
        
        /// history of previous vectors
        mdarray<double, 2> vectors_;

        /// output buffer for the whole vector
        mdarray<double, 1> output_buffer_;

        /// Return position in the list of vectors for the given mixing step.
        inline int offset(int step)
        {
            return step % max_history_;
        }

        double rms_deviation()
        {
            double rms = 0.0;
            for (int i = 0; i < spl_size_.local_size(); i++)
            {
                rms += pow(vectors_(i, offset(count_)) - vectors_(i, offset(count_ - 1)), 2);
            }
            Platform::allreduce(&rms, 1);
            rms = sqrt(rms / double(size_));
            return rms;
        }

        void mix_linear(double beta__)
        {
            for (int i = 0; i < spl_size_.local_size(); i++)
                vectors_(i, offset(count_)) = beta__ * input_buffer_(i) + (1 - beta__) * vectors_(i, offset(count_ - 1));

            Platform::allgather(&vectors_(0, offset(count_)), output_buffer_.ptr(), spl_size_.global_offset(), 
                                spl_size_.local_size());
        }

    public:

        Mixer(size_t size__, int max_history__, double beta__) 
            : size_(size__), 
              max_history_(max_history__), 
              beta_(beta__), 
              count_(0)
        {
            spl_size_ = splindex<block>((int)size_, Platform::num_mpi_ranks(), Platform::mpi_rank());
            // allocate input buffer (local size)
            input_buffer_.set_dimensions(spl_size_.local_size());
            input_buffer_.allocate();
            // allocate output bffer (global size)
            output_buffer_.set_dimensions(size_);
            output_buffer_.allocate();
            // allocate storage for previous vectors (local size)
            vectors_.set_dimensions(spl_size_.local_size(), max_history_);
            vectors_.allocate();
        }

        virtual ~Mixer()
        {
        }

        void input(size_t idx, double value)
        {
            assert(idx < size_t(1 << 31));

            auto offs_and_rank = spl_size_.location((int)idx);
            if (offs_and_rank.second == Platform::mpi_rank()) input_buffer_(offs_and_rank.first) = value;
        }

        inline double* output_buffer()
        {
            return output_buffer_.ptr();
        }

        inline void initialize()
        {
            memcpy(&vectors_(0, 0), &input_buffer_(0), spl_size_.local_size() * sizeof(double));
        }

        inline double beta()
        {
            return beta_;
        }
            
        virtual double mix() = 0;
};

/// Primitive linear adaptive mixer
class Linear_mixer: public Mixer
{
    private:
        
        /// previous root mean square
        double rms_prev_;

        double beta0_;

    public:
        
        /// Constructor
        Linear_mixer(size_t size__, double beta0__) : Mixer(size__, 2, beta0__), rms_prev_(0), beta0_(beta0__)
        {
        }

        double mix()
        {
            count_++;

            mix_linear(beta_);
            
            double rms = rms_deviation();

            //if (rms < rms_prev_) 
            //{
            //    beta_ *= 1.1;
            //}
            //else 
            //{
            //    beta_ = beta0_;
            //}
            beta_ = std::min(beta_, 0.9);

            rms_prev_ = rms;
            
            return rms;
        }
};

/// Broyden mixer
/** Reference paper: "Robust acceleration of self consistent field calculations for 
 *  density functional theory", Baarman K, Eirola T, Havu V., J Chem Phys. 134, 134109 (2011)
 */
class Broyden_mixer: public Mixer
{
    private:

        mdarray<double, 2> residuals_;
    
    public:

        Broyden_mixer(size_t size__, int max_history__, double beta__) : Mixer(size__, max_history__, beta__)
        {
            residuals_.set_dimensions(spl_size_.local_size(), max_history__);
            residuals_.allocate();
        }

        double mix()
        {
            Timer t("sirius::Broyden_mixer::mix");

            // curent residual f_k = x_k - g(x_k)
            for (int i = 0; i < spl_size_.local_size(); i++) 
                residuals_(i, offset(count_)) = vectors_(i, offset(count_)) - input_buffer_(i);

            count_++;

            // at this point we have min(count_, max_history_) residuals and vectors from the previous iterations
            int N = std::min(count_, max_history_);

            //if (N > 1)
            if (count_ > max_history_)
            {
                mdarray<long double, 2> S(N, N);
                S.zero();
                // S = F^T * F, where F is the matrix of residual vectors
                for (int j1 = 0; j1 < N; j1++)
                { 
                    for (int j2 = 0; j2 <= j1; j2++)
                    {
                        for (int i = 0; i < spl_size_.local_size(); i++) 
                        {
                            S(j1, j2) += residuals_(i, offset(count_ - N + j1)) * residuals_(i, offset(count_ - N + j2));
                        }
                        S(j2, j1) = S(j1, j2);
                    }
                }
                Platform::allreduce(S.ptr(), (int)S.size());
                for (int j1 = 0; j1 < N; j1++)
                { 
                    for (int j2 = 0; j2 < N; j2++) S(j1, j2) /= size_;
                }
               
                mdarray<long double, 2> gamma_k(2 * N, N);
                gamma_k.zero();
                // initial gamma_0
                for (int i = 0; i < N; i++) gamma_k(i, i) = 0.25;

                std::vector<long double> v1(N);
                std::vector<long double> v2(2 * N);
                
                // update gamma_k by recursion
                for (int k = 0; k < N - 1; k++)
                {
                    // denominator df_k^{T} S df_k
                    long double d = S(k, k) + S(k + 1, k + 1) - S(k, k + 1) - S(k + 1, k);
                    // nominator
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
 
                memset(&v2[0], 0, 2 * N * sizeof(long double));
                for (int j = 0; j < 2 * N; j++) v2[j] = -gamma_k(j, N - 1);
                v2[2 * N - 1] += 1;
                
                // use input_buffer as a temporary storage 
                input_buffer_.zero();

                // make linear combination of vectors and residuals; this is the update vector \tilda x
                for (int j = 0; j < N; j++)
                {
                    for (int i = 0; i < spl_size_.local_size(); i++) 
                    {
                        input_buffer_(i) += ((double)v2[j] * residuals_(i, offset(count_ - N + j)) + 
                                             (double)v2[j + N] * vectors_(i, offset(count_ - N + j)));
                    }
                }
            }
            
            // mix last vector with the update vector \tilda x
            if (count_ > max_history_)
            {
                mix_linear(beta_);
            }
            else
            {
                mix_linear(beta_ / 10.0);
            }

            return rms_deviation();
        }
};

class Pulay_mixer: public Mixer
{
    private:

        mdarray<double, 2> residuals_;
    
    public:

        Pulay_mixer(size_t size__, int max_history__, double beta__) : Mixer(size__, max_history__, beta__)
        {
            residuals_.set_dimensions(spl_size_.local_size(), max_history__);
            residuals_.allocate();
        }

        double mix()
        {
            Timer t("sirius::Pulay_mixer::mix");

            for (int i = 0; i < spl_size_.local_size(); i++) 
                residuals_(i, offset(count_)) = input_buffer_(i) - vectors_(i, offset(count_));

            count_++;

            int N = std::min(count_, max_history_);

            //if (count_ > max_history_)
            if (N > 1)
            {
                mdarray<long double, 2> S(N, N);
                S.zero();

                for (int j1 = 0; j1 < N - 1; j1++)
                { 
                    for (int j2 = 0; j2 <= j1; j2++)
                    {
                        for (int i = 0; i < spl_size_.local_size(); i++) 
                        {
                            S(j1, j2) += std::pow(residuals_(i, offset(count_ - N + j1)) * residuals_(i, offset(count_ - N + j2)), 2);
                        }
                        S(j2, j1) = S(j1, j2);
                    }
                }
                Platform::allreduce(S.ptr(), (int)S.size());
                for (int j1 = 0; j1 < N - 1; j1++)
                { 
                    for (int j2 = 0; j2 < N - 1; j2++) S(j1, j2) = std::sqrt(S(j1, j2) / size_);
                }
                for (int j = 0; j < N - 1; j++) S(N - 1, j) = S(j, N - 1) = 1;

                mdarray<double, 2> s(N, N);
                s.zero();
                for (int j = 0; j < N; j++)
                {
                    for (int i = 0; i < N; i++) s(i, j) = (double)S(i, j);
                }

                linalg<lapack>::invert_ge(s.ptr(), N);

                memset(&vectors_(0, offset(count_)), 0, spl_size_.local_size() * sizeof(double));

                for (int j = 0; j < N - 1; j++)
                {
                    for (int i = 0; i < spl_size_.local_size(); i++)
                    {
                        vectors_(i, offset(count_)) += s(j, N - 1) * (vectors_(i, offset(count_ - N + j)) + 
                                                                      residuals_(i, offset(count_ - N + j)));
                        //vectors_(i, offset(count_)) += s(j, N - 1) * vectors_(i, offset(count_ - N + j));
                    }
                }

                Platform::allgather(&vectors_(0, offset(count_)), output_buffer_.ptr(), spl_size_.global_offset(), 
                                              spl_size_.local_size());
            }
            else
            {
                mix_linear(beta_);
            }

            return rms_deviation();
        }
};

class Adaptive_mixer: public Mixer
{
    private:

        //mdarray<double, 2> residuals_;
    
    public:

        Adaptive_mixer(size_t size__, int max_history__, double beta__) : Mixer(size__, max_history__, beta__)
        {
            // residuals_.set_dimensions(spl_size_.local_size(), max_history__);
            // residuals_.allocate();
        }

        double mix()
        {
            Timer t("sirius::Adaptive_mixer::mix");

            //== for (int i = 0; i < spl_size_.local_size(); i++) 
            //==     residuals_(i, offset(count_)) = input_buffer_(i) - vectors_(i, offset(count_));

            count_++;

            int N = std::min(count_, max_history_);

            if (N > 1)
            {
                for (int j = 0; j <= 10; j++)
                {
                    //==double k0 = (1 - beta_) * double(j) / 10;
                    //==double k1 = (1 - beta_) * double(10 - j) / 10;
                    for (int i = 0; i < spl_size_.local_size(); i++)
                    {
                        vectors_(i, offset(count_)) = 0.5 * (1 - beta_) * vectors_(i, offset(count_ - 2)) + 
                                                      0.5 * (1 - beta_) * vectors_(i, offset(count_ - 1)) + 
                                                      beta_ * input_buffer_(i); 
                    }
                    //==double rms = rms_deviation();
                    //==if (Platform::mpi_rank() == 0) std::cout << " j = " << j << ", rms = " << rms << std::endl;

                    Platform::allgather(&vectors_(0, offset(count_)), output_buffer_.ptr(), spl_size_.global_offset(), 
                                        spl_size_.local_size());
                }
               
            }
            else
            {
                mix_linear(beta_);
            }

            return rms_deviation();
        }
};

}

#endif // __MIXER_H__

