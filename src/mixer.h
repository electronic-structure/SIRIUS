
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

template <typename T> class periodic_function_mixer: public mixer<T>
{
    private:

        PeriodicFunction<T>* pf_;

    public:

        periodic_function_mixer(PeriodicFunction<T>* pf__, double beta__) : pf_(pf__)
        {
            this->mixer_data_.set_dimensions((int)pf_->size(), 2);
            this->mixer_data_.allocate();
            this->mixer_data_.zero();
            this->beta_ = beta__;
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
            
            pf_->pack(&this->mixer_data_(0, p));
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
            pf_->unpack(&this->mixer_data_(0, 0));
            rms = sqrt(rms / this->mixer_data_.size(0));
            
            if (rms < this->rms_prev_) 
            {
                this->beta_ *= 1.1;
            }
            else 
            {
                this->beta_ = 0.1;
            }
            this->beta_ = std::min(this->beta_, 0.9);

            printf("[periodic_function_mixer]\n");
            printf("beta = %f\n", this->beta_);

            this->rms_prev_ = rms;
            
            return rms;
        }

        

};


};
