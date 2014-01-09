#ifndef __MIXER_H__
#define __MIXER_H__

/** \file mixer.h
    
    \brief Contains definition of and implementation of sirius::Mixer, sirius::Linear_mixer and sirius::Broyden_mixer 
           clases.
*/    
namespace sirius
{

//= template <typename T> class mixer
//= {
//=     protected:
//=         
//=         double beta_;
//= 
//=         double rms_prev_;
//= 
//=         mdarray<T, 2> mixer_data_;
//= 
//=         bool initialized_;
//= 
//=     public:
//=         virtual ~mixer()
//=         {
//=         }
//= 
//=         virtual void load() = 0;
//= 
//=         virtual double mix() = 0;
//= };
//= 
//= class density_mixer: public mixer<double>
//= {
//=     private:
//= 
//=         int num_mag_dims_;
//=         Periodic_function<double>* rho_;
//=         Periodic_function<double>* mag_[3];
//= 
//=     public:
//= 
//=         density_mixer(Periodic_function<double>* rho__, Periodic_function<double>* mag__[3], int num_mag_dims__)
//=         {
//=             rho_ = rho__;
//=             size_t mixer_size = rho_->size();
//=             num_mag_dims_ = num_mag_dims__;
//=             for (int i = 0; i < num_mag_dims_; i++) 
//=             {
//=                 mag_[i] = mag__[i];
//=                 mixer_size += mag_[i]->size();
//=             }
//= 
//=             this->mixer_data_.set_dimensions((int)mixer_size, 2);
//=             this->mixer_data_.allocate();
//=             this->mixer_data_.zero();
//=             this->beta_ = 0.1;
//=             this->rms_prev_ = 0;
//=             this->initialized_ = false;
//=         }
//= 
//=         void load()
//=         {
//=             int p = 1;
//=             if (!this->initialized_)
//=             {
//=                 p = 0;
//=                 this->initialized_ = true;
//=             }
//=             
//=             size_t n = rho_->pack(&this->mixer_data_(0, p));
//=             for (int i = 0; i < num_mag_dims_; i++) n += mag_[i]->pack(&this->mixer_data_((int)n, p));
//=         }
//= 
//=         double mix()
//=         {
//=             load();
//= 
//=             double rms = 0.0;
//=             for (int n = 0; n < this->mixer_data_.size(0); n++)
//=             {
//=                 this->mixer_data_(n, 0) = (1 - this->beta_) * this->mixer_data_(n, 0) + this->beta_ * this->mixer_data_(n, 1);
//=                 rms += pow(this->mixer_data_(n, 0) - this->mixer_data_(n, 1), 2);
//=             }
//=             rms = sqrt(rms / this->mixer_data_.size(0));
//= 
//=             int n = (int)rho_->unpack(&this->mixer_data_(0, 0));
//=             for (int i = 0; i < num_mag_dims_; i++) n += (int)mag_[i]->unpack(&this->mixer_data_(n, 0));
//=             
//=             if (rms < this->rms_prev_) 
//=             {
//=                 this->beta_ *= 1.1;
//=             }
//=             else 
//=             {
//=                 this->beta_ = 0.1;
//=             }
//=             this->beta_ = std::min(this->beta_, 0.9);
//= 
//=             this->rms_prev_ = rms;
//=             
//=             return rms;
//=         }
//= 
//=         inline double beta()
//=         {
//=             return beta_;
//=         }
//= };



/// Abstract mixer
class Mixer
{
    protected:
        
        /// size of the mixed vectors
        size_t size_;
        
        /// maximum number of stored vectors
        int max_history_;

        /// mixing factor
        double beta_;
        
        /// number of times mixer was called so far
        int count_;

        std::vector<double> input_buffer_;

        mdarray<double, 2> vectors_;

        /// Return position in the list of vectors for the given mixing step.
        inline int offset(int step)
        {
            return step % max_history_;
        }

        double rms_deviation()
        {
            double rms = 0.0;
            for (size_t i = 0; i < size_; i++)
            {
                rms += pow(vectors_((int)i, offset(count_)) - vectors_((int)i, offset(count_ - 1)), 2);
            }
            rms = sqrt(rms / double(size_));
            return rms;
        }

        void mix_linear()
        {
            for (size_t i = 0; i < size_; i++)
            {
                vectors_((int)i, offset(count_)) = beta_ * input_buffer_[i] + (1 - beta_) * vectors_((int)i, offset(count_ - 1));
            }
        }

    public:

        Mixer(size_t size__, int max_history__, double beta__) 
            : size_(size__), 
              max_history_(max_history__), 
              beta_(beta__), 
              count_(0)
        {
            input_buffer_.resize(size_);
            vectors_.set_dimensions((int)size_, max_history_);
            vectors_.allocate();
        }

        virtual ~Mixer()
        {
        }

        inline double* input_buffer()
        {
            return &input_buffer_[0];
        }

        inline double* output_buffer()
        {
            return &vectors_(0, offset(count_));
        }

        inline void initialize()
        {
            memcpy(&vectors_(0, 0), &input_buffer_[0], size_ * sizeof(double));
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

    public:
        
        /// Constructor
        Linear_mixer(size_t size__) : Mixer(size__, 2, 0.1), rms_prev_(0)
        {
        }

        double mix()
        {
            count_++;

            mix_linear();
            
            double rms = rms_deviation();

            if (rms < rms_prev_) 
            {
                beta_ *= 1.1;
            }
            else 
            {
                beta_ = 0.1;
            }
            beta_ = std::min(beta_, 0.9);

            rms_prev_ = rms;
            
            return rms;
        }
};

/// Broyden mixer
/** Reference paper: "Robust acceleration of self consistent field calculations for density functional theory", 
    Baarman K, Eirola T, Havu V., J Chem Phys. 134, 134109 (2011)
*/
class Broyden_mixer: public Mixer
{
    private:

        mdarray<double, 2> residuals_;
    
    public:

        Broyden_mixer(size_t size__, int max_history__, double beta__) : Mixer(size__, max_history__, beta__)
        {
            residuals_.set_dimensions((int)size__, max_history__);
            residuals_.allocate();
        }

        double mix()
        {
            Timer t("sirius::Broyden_mixer::mix");

            // curent residual f_k = x_k - g(x_k)
            for (size_t i = 0; i < size_; i++) residuals_((int)i, offset(count_)) = vectors_((int)i, offset(count_)) - input_buffer_[i];

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
                        for (size_t i = 0; i < size_; i++) 
                        {
                            S(j1, j2) += residuals_((int)i, offset(count_ - N + j1)) * residuals_((int)i, offset(count_ - N + j2));
                        }
                        S(j2, j1) = S(j1, j2);
                    }
                }
               
                mdarray<double, 2> gamma_k(2 * max_history_, max_history_);
                gamma_k.zero();
                // initial gamma_0
                for (int i = 0; i < max_history_; i++) gamma_k(i, i) = 0.5;

                std::vector<double> v1(max_history_);
                std::vector<double> v2(max_history_ * 2);
                
                // update gamma_k by recursion
                for (int k = 0; k < N - 1; k++)
                {
                    // denominator df_k^{T} S df_k
                    double d = S(k, k) + S(k + 1, k + 1) - S(k, k + 1) - S(k + 1, k);
                    // nominator
                    memset(&v1[0], 0, max_history_ * sizeof(int));
                    for (int j = 0; j < N; j++) v1[j] = S(k + 1, j) - S(k, j);

                    memset(&v2[0], 0, 2 * max_history_ * sizeof(int));
                    for (int j = 0; j < 2 * max_history_; j++) v2[j] = -(gamma_k(j, k + 1) - gamma_k(j, k));
                    v2[max_history_ + k] -= 1;
                    v2[max_history_ + k + 1] += 1;

                    for (int j1 = 0; j1 < max_history_; j1++)
                    {
                        for (int j2 = 0; j2 < 2 * max_history_; j2++) gamma_k(j2, j1) += v2[j2] * v1[j1] / d;
                    }
                }
 
                memset(&v2[0], 0, 2 * max_history_ * sizeof(int));
                for (int j = 0; j < 2 * max_history_; j++) v2[j] = -gamma_k(j, N - 1);
                v2[max_history_ + N - 1] += 1;
                
                // use input_buffer as a temporary storage 
                memset(&input_buffer_[0], 0, size_ * sizeof(double));

                // make linear combination of vectors and residuals; this is the update vector \tilda x
                for (int j = 0; j < N; j++)
                {
                    for (size_t i = 0; i < size_; i++) 
                    {
                        input_buffer_[i] += (v2[j] * residuals_((int)i, offset(count_ - N + j)) + 
                                             v2[j + max_history_] * vectors_((int)i, offset(count_ - N + j)));
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

