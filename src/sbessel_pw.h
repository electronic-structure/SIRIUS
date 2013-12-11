#ifndef __SBESSEL_PW_H__
#define __SBESSEL_PW_H__

namespace sirius
{

template <typename T> class sbessel_pw
{
    private:

        Global& parameters_;

        int lmax_;

        mdarray<Spline<T>*, 2> sjl_; 

    public:

        sbessel_pw(Global& parameters__, int lmax__) : parameters_(parameters__), lmax_(lmax__)
        {
            sjl_.set_dimensions(lmax_ + 1, parameters_.unit_cell()->num_atom_types());
            sjl_.allocate();

            for (int iat = 0; iat < parameters_.unit_cell()->num_atom_types(); iat++)
            {
                for (int l = 0; l <= lmax_; l++)
                {
                    sjl_(l, iat) = new Spline<T>(parameters_.unit_cell()->atom_type(iat)->num_mt_points(),
                                                 parameters_.unit_cell()->atom_type(iat)->radial_grid());
                }
            }
        }
        
        ~sbessel_pw()
        {
            for (int iat = 0; iat < parameters_.unit_cell()->num_atom_types(); iat++)
            {
                for (int l = 0; l <= lmax_; l++) delete sjl_(l, iat);
            }
            sjl_.deallocate();
        }

        void load(double q)
        {
            std::vector<double> jl(lmax_+ 1);
            for (int iat = 0; iat < parameters_.unit_cell()->num_atom_types(); iat++)
            {
                for (int ir = 0; ir < parameters_.unit_cell()->atom_type(iat)->num_mt_points(); ir++)
                {
                    double x = parameters_.unit_cell()->atom_type(iat)->radial_grid(ir) * q;
                    gsl_sf_bessel_jl_array(lmax_, x, &jl[0]);
                    for (int l = 0; l <= lmax_; l++) (*sjl_(l, iat))[ir] = jl[l];
                }
            }
        }

        void interpolate(double q)
        {
            load(q);
            
            for (int iat = 0; iat < parameters_.unit_cell()->num_atom_types(); iat++)
            {
                for (int l = 0; l <= lmax_; l++) sjl_(l, iat)->interpolate();
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
};

};

#endif
