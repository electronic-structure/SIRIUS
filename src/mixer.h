
namespace sirius
{

template <typename T> class mixer
{
    protected:
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

        periodic_function_mixer(PeriodicFunction<T>* pf__) : pf_(pf__)
        {
            
            this->mixer_data_.set_dimensions((int)pf_->size(), 2);
            this->mixer_data_.allocate();
            this->mixer_data_.zero();
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
            double beta = 0.5;
            double rms = 0.0;
            for (int n = 0; n < this->mixer_data_.size(0); n++)
            {
                this->mixer_data_(n, 0) = (1 - beta) * this->mixer_data_(n, 0) + beta * this->mixer_data_(n, 1);
                rms += pow(this->mixer_data_(n, 0) - this->mixer_data_(n, 1), 2);
            }
            pf_->unpack(&this->mixer_data_(0, 0));
            return sqrt(rms / this->mixer_data_.size(0));
        }

        

};


};
