namespace sirius
{

template <typename T>
class Spheric_function_vector
{
    private:
        std::vector< Spheric_function<T>* > vec_;

    public:
        Spheric_function_vector(T* ptr, SHT& sht, Radial_grid& radial_grid, int nd)
        {
            vec_.resize(nd);
            for (int i = 0; i < nd; i++) 
                vec_[i] = new Spheric_function<T>(&ptr[i * sht.num_points() * radial_grid.num_mt_points()], sht, radial_grid);
        }
        ~Spheric_function_vector()
        {
            for (int i = 0; i < (int)vec_.size(); i++) delete vec_[i];
        }

        inline Spheric_function<T>& operator[](const int idx)
        {
            return *vec_[idx];
        }

};

}
