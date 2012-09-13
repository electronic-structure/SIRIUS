#ifndef __ATOM_H__
#define __ATOM_H__

namespace sirius {

class Atom
{
    private:
    
        AtomType* type_;

        AtomSymmetryClass* symmetry_class_;
        
        /// position in fractional coordinates
        double position_[3];
        
        double vector_field_[3];

        mdarray<double,3> h_radial_integrals_;
        mdarray<double,4> o_radial_integrals_;
    
    public:
    
        Atom(AtomType* type__,
             double* position__, 
             double* vector_field__) : type_(type__),
                                       symmetry_class_(NULL)
        {
            assert(type__);
                
            for (int i = 0; i < 3; i++)
            {
                position_[i] = position__[i];
                vector_field_[i] = vector_field__[i];
            }
        }

        inline AtomType* type()
        {
            return type_;
        }

        inline int type_id()
        {
            return type_->id();
        }

        inline void get_position(double* position__)
        {
            for (int i = 0; i < 3; i++)
                position__[i] = position_[i];
        }
        
        inline double* position()
        {
            return position_;
        }

        inline int symmetry_class_id()
        {
            if (symmetry_class_) 
                return symmetry_class_->id();
            else
                return -1;
        }

        inline void set_symmetry_class(AtomSymmetryClass* symmetry_class__)
        {
            symmetry_class_ = symmetry_class__;
        }

        void init(int lmax)
        {
            assert(symmetry_class_);
            
            int lmmax = lmmax_by_lmax(lmax);

            h_radial_integrals_.set_dimensions(lmmax, type_->indexr().size(), type_->indexr().size());
            h_radial_integrals_.allocate();
        }

        /*!
            \brief Generate radial integrals used to setup Hamiltonian and overlap matrices

            Hamiltonian operator has the following representation inside muffin-tins:
            \f[
                \hat H=-\frac{1}{2}\nabla^2 + \sum_{\ell m} V_{\ell m}(r) R_{\ell m}(\hat {\bf r}) =
                  \underbrace{-\frac{1}{2} \nabla^2+V_{00}(r)R_{00}}_{H_{s}(r)} +\sum_{\ell=1} \sum_{m=-\ell}^{\ell} 
                   V_{\ell m}(r) R_{\ell m}(\hat {\bf r}) = \sum_{\ell m} \widetilde V_{\ell m}(r) R_{\ell m}(\hat {\bf r})
            \f]
            where
            \f[
                \widetilde V_{\ell m}(r)=\left\{ \begin{array}{ll} 
                  \frac{H_{s}(r)}{R_{00}} & \ell = 0 \\ 
                  V_{\ell m}(r) & \ell > 0 \end{array} \right.
            \f]
        */
        void generate_radial_integrals(double lmax, double* veff_)
        {
            Timer t("sirius::Atom::generate_radial_integrals");
            
            int lmmax = lmmax_by_lmax(lmax);
            int nmtp = type_->num_mt_points();

            mdarray<double,2> veff(veff_, lmmax, nmtp);

            Spline<double> s(nmtp, type_->radial_grid());
            
            h_radial_integrals_.zero();

            // TODO: check if integration with explicit r^2 is faster
            for (int i1 = 0; i1 < type_->indexr().size(); i1++)
                for (int i2 = 0; i2 < type_->indexr().size(); i2++)
                {
                    // for spherical part of potential integrals are diagonal in l
                    if (type_->indexr(i1).l == type_->indexr(i2).l)
                    {
                        for (int ir = 0; ir < nmtp; ir++)
                            s[ir] = symmetry_class_->radial_function(ir, i1, 0) * symmetry_class_->radial_function(ir, i2, 1);
                        s.interpolate();
                        h_radial_integrals_(0, i1, i2) = s.integrate(2) / y00;
                    }
                    
                    // non-spherical terms
                    for (int lm = 1; lm < lmmax; lm++)
                    {
                        for (int ir = 0; ir < nmtp; ir++)
                            s[ir] = symmetry_class_->radial_function(ir, i1, 0) * symmetry_class_->radial_function(ir, i2, 0) * veff(lm, ir);
                        s.interpolate();
                        h_radial_integrals_(lm, i1, i2) = s.integrate(2);
                    }
                }

                      


        }
};

};

#endif // __ATOM_H__
