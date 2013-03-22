#ifndef __SBESSEL_PW_H__
#define __SBESSEL_PW_H__

namespace sirius
{

template <typename T> class sbessel_pw
{
    private:

        Global& parameters_;

        mdarray<Spline<T>*, 2> sjl_; 

    public:

        sbessel_pw(Global& parameters__) : parameters_(parameters__) 
        {
            sjl_.set_dimensions(parameters_.lmax_pot() + 1, parameters_.num_atom_types());
            sjl_.allocate();

            for (int iat = 0; iat < parameters_.num_atom_types(); iat++)
            {
                for (int l = 0; l <= parameters_.lmax_pot(); l++)
                {
                    sjl_(l, iat) = new Spline<T>(parameters_.atom_type(iat)->num_mt_points(),
                                                 parameters_.atom_type(iat)->radial_grid());
                }
            }
        }

        void load(double q)
        {
            std::vector<double> jl(parameters_.lmax_pot() + 1);
            for (int iat = 0; iat < parameters_.num_atom_types(); iat++)
            {
                for (int ir = 0; ir < parameters_.atom_type(iat)->num_mt_points(); ir++)
                {
                    double x = parameters_.atom_type(iat)->radial_grid(ir) * q;
                    gsl_sf_bessel_jl_array(parameters_.lmax_pot(), x, &jl[0]);
                    for (int l = 0; l <= parameters_.lmax_pot(); l++) (*sjl_(l, iat))[ir]= jl[l];
                }
            }
        }

        void interpolate(double q)
        {
            load(q);
            
            for (int iat = 0; iat < parameters_.num_atom_types(); iat++)
            {
                for (int l = 0; l <= parameters_.lmax_pot(); l++) sjl_(l, iat)->interpolate();
            }
        }

        inline T operator()(int ir, int l, int iat)
        {
            return (*sjl_(l, iat))[ir];
        }

        inline Spline<T>* operator()(int l, int iat)
        {
            return sjl_(l, iat);
        }

        ~sbessel_pw()
        {
            for (int iat = 0; iat < parameters_.num_atom_types(); iat++)
            {
                for (int l = 0; l <= parameters_.lmax_pot(); l++) delete sjl_(l, iat);
            }
            sjl_.deallocate();
        }
};

};

#endif
