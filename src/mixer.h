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

        void mix_linear()
        {
            double b = (count_ == 1) ? 0.1 : beta_;

            for (int i = 0; i < spl_size_.local_size(); i++)
                vectors_(i, offset(count_)) = b * input_buffer_(i) + (1 - b) * vectors_(i, offset(count_ - 1));

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

            mix_linear();
            
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

            if (N > 1)
            {
                mdarray<double, 2> S(N, N);
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
               
                mdarray<double, 2> gamma_k(2 * max_history_, max_history_);
                gamma_k.zero();
                // initial gamma_0
                for (int i = 0; i < max_history_; i++) gamma_k(i, i) = 0.25;

                std::vector<double> v1(max_history_);
                std::vector<double> v2(max_history_ * 2);
                
                // update gamma_k by recursion
                for (int k = 0; k < N - 1; k++)
                {
                    // denominator df_k^{T} S df_k
                    double d = S(k, k) + S(k + 1, k + 1) - S(k, k + 1) - S(k + 1, k);
                    // nominator
                    memset(&v1[0], 0, max_history_ * sizeof(double));
                    for (int j = 0; j < N; j++) v1[j] = S(k + 1, j) - S(k, j);

                    memset(&v2[0], 0, 2 * max_history_ * sizeof(double));
                    for (int j = 0; j < 2 * max_history_; j++) v2[j] = -(gamma_k(j, k + 1) - gamma_k(j, k));
                    v2[max_history_ + k] -= 1;
                    v2[max_history_ + k + 1] += 1;

                    for (int j1 = 0; j1 < max_history_; j1++)
                    {
                        for (int j2 = 0; j2 < 2 * max_history_; j2++) gamma_k(j2, j1) += v2[j2] * v1[j1] / d;
                    }
                }
 
                memset(&v2[0], 0, 2 * max_history_ * sizeof(double));
                for (int j = 0; j < 2 * max_history_; j++) v2[j] = -gamma_k(j, N - 1);
                v2[max_history_ + N - 1] += 1;
                
                // use input_buffer as a temporary storage 
                input_buffer_.zero();

                // make linear combination of vectors and residuals; this is the update vector \tilda x
                for (int j = 0; j < N; j++)
                {
                    for (int i = 0; i < spl_size_.local_size(); i++) 
                    {
                        input_buffer_(i) += (v2[j] * residuals_(i, offset(count_ - N + j)) + 
                                             v2[j + max_history_] * vectors_(i, offset(count_ - N + j)));
                    }
                }
            }
            
            // mix last vector with the update vector \tilda x
            mix_linear();

            return rms_deviation();
        }
};

};

#endif // __MIXER_H__

