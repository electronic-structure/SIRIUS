
namespace sirius
{

template <typename T> class mixer_base
{
    protected:
        mdarray<T, 2> mixer_data_;

    public:




};

template <typename T> class periodic_function_mixer: public mixer_base<T>
{
    public:
        periodic_function_mixer(PeriodicFunction<T>* func)
        {
            
            int mixer_size = func->size();
            this->mixer_data_.set_dimensions(mixer_size, 2);
            this->mixer_data_.allocate();
            this->mixer_data_.zero();
        }

};









};
