
namespace sirius
{

template <typename T> class mixer
{
    protected:
        
        double beta_;

        double rms_prev_;

        mdarray<T, 2> mixer_data_;

        bool initialized_;

    public:
        virtual ~mixer()
        {
        }

        virtual void load() = 0;

        virtual double mix() = 0;
};

class density_mixer: public mixer<double>
{
    private:

        int num_mag_dims_;
        Periodic_function<double>* rho_;
        Periodic_function<double>* mag_[3];

    public:

        density_mixer(Periodic_function<double>* rho__, Periodic_function<double>* mag__[3], int num_mag_dims__)
        {
            rho_ = rho__;
            size_t mixer_size = rho_->size();
            num_mag_dims_ = num_mag_dims__;
            for (int i = 0; i < num_mag_dims_; i++) 
            {
                mag_[i] = mag__[i];
                mixer_size += mag_[i]->size();
            }

            this->mixer_data_.set_dimensions((int)mixer_size, 2);
            this->mixer_data_.allocate();
            this->mixer_data_.zero();
            this->beta_ = 0.1;
            this->rms_prev_ = 0;
            this->initialized_ = false;
        }

        void load()
        {
            int p = 1;
            if (!this->initialized_)
            {
                p = 0;
                this->initialized_ = true;
            }
            
            size_t n = rho_->pack(&this->mixer_data_(0, p));
            for (int i = 0; i < num_mag_dims_; i++) n += mag_[i]->pack(&this->mixer_data_((int)n, p));
        }

        double mix()
        {
            load();

            double rms = 0.0;
            for (int n = 0; n < this->mixer_data_.size(0); n++)
            {
                this->mixer_data_(n, 0) = (1 - this->beta_) * this->mixer_data_(n, 0) + this->beta_ * this->mixer_data_(n, 1);
                rms += pow(this->mixer_data_(n, 0) - this->mixer_data_(n, 1), 2);
            }
            rms = sqrt(rms / this->mixer_data_.size(0));

            int n = (int)rho_->unpack(&this->mixer_data_(0, 0));
            for (int i = 0; i < num_mag_dims_; i++) n += (int)mag_[i]->unpack(&this->mixer_data_(n, 0));
            
            if (rms < this->rms_prev_) 
            {
                this->beta_ *= 1.1;
            }
            else 
            {
                this->beta_ = 0.1;
            }
            this->beta_ = std::min(this->beta_, 0.9);

            this->rms_prev_ = rms;
            
            return rms;
        }

        inline double beta()
        {
            return beta_;
        }
};


};
