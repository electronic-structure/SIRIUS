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
    
    
    public:
    
        Atom(AtomType* _type,
             double* _position, 
             double* _vector_field) : type_(_type),
                                      symmetry_class_(NULL)
        {
            assert(_type != NULL);
                
            for (int i = 0; i < 3; i++)
            {
                position_[i] = _position[i];
                vector_field_[i] = _vector_field[i];
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

        inline void set_symmetry_class(AtomSymmetryClass* _symmetry_class)
        {
            symmetry_class_ = _symmetry_class;
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
            int lmmax = lmmax_by_lmax(lmax);
            mdarray<double,2> veff(veff_, lmmax, type_->num_mt_points());  


        }
};

};

#endif // __ATOM_H__
