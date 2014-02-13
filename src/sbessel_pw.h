#ifndef __SBESSEL_PW_H__
#define __SBESSEL_PW_H__

#include <gsl/gsl_sf_bessel.h>

namespace sirius
{

template <typename T> class sbessel_pw
{
    private:

        Unit_cell* unit_cell_;

        int lmax_;

        mdarray<Spline<T>*, 2> sjl_; 

    public:

        sbessel_pw(Unit_cell* unit_cell__, int lmax__) : unit_cell_(unit_cell__), lmax_(lmax__)
        {
            sjl_.set_dimensions(lmax_ + 1, unit_cell_->num_atom_types());
            sjl_.allocate();

            for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
            {
                for (int l = 0; l <= lmax_; l++)
                {
                    sjl_(l, iat) = new Spline<T>(unit_cell_->atom_type(iat)->num_mt_points(),
                                                 unit_cell_->atom_type(iat)->radial_grid());
                }
            }
        }
        
        ~sbessel_pw()
        {
            for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
            {
                for (int l = 0; l <= lmax_; l++) delete sjl_(l, iat);
            }
            sjl_.deallocate();
        }

        void load(double q)
        {
            std::vector<double> jl(lmax_+ 1);
            for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
            {
                for (int ir = 0; ir < unit_cell_->atom_type(iat)->num_mt_points(); ir++)
                {
                    double x = unit_cell_->atom_type(iat)->radial_grid(ir) * q;
                    gsl_sf_bessel_jl_array(lmax_, x, &jl[0]);
                    for (int l = 0; l <= lmax_; l++) (*sjl_(l, iat))[ir] = jl[l];
                }
            }
        }

        void interpolate(double q)
        {
            load(q);
            
            for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
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
