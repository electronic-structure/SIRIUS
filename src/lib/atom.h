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
        mdarray<double,3> o_radial_integrals_;

        int offset_aw_;
    
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

        inline AtomSymmetryClass* symmetry_class()
        {
            return symmetry_class_;
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
            if (symmetry_class()) 
                return symmetry_class()->id();
            else
                return -1;
        }

        inline void set_symmetry_class(AtomSymmetryClass* symmetry_class__)
        {
            symmetry_class_ = symmetry_class__;
        }

        void init(int lmax, int offset_aw__)
        {
            assert(symmetry_class());
            assert(lmax >= 0);
            assert(offset_aw__ >= 0);
            
            offset_aw_ = offset_aw__;
            int lmmax = lmmax_by_lmax(lmax);

            h_radial_integrals_.set_dimensions(lmmax, type()->indexr().size(), type()->indexr().size());
            h_radial_integrals_.allocate();

            o_radial_integrals_.set_dimensions(type()->num_aw_descriptors(), type()->indexr().max_num_rf(), type()->indexr().max_num_rf());
            o_radial_integrals_.allocate();
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
            int nmtp = type()->num_mt_points();

            mdarray<double,2> veff(veff_, lmmax, nmtp);

            Spline<double> s(nmtp, type()->radial_grid());
            
            h_radial_integrals_.zero();

            // TODO: check if integration with explicit r^2 is faster
            for (int i1 = 0; i1 < type()->indexr().size(); i1++)
                for (int i2 = 0; i2 < type()->indexr().size(); i2++)
                {
                    // for spherical part of potential integrals are diagonal in l
                    if (type()->indexr(i1).l == type()->indexr(i2).l)
                    {
                        for (int ir = 0; ir < nmtp; ir++)
                            s[ir] = symmetry_class()->radial_function(ir, i1, 0) * symmetry_class()->radial_function(ir, i2, 1);
                        s.interpolate();
                        h_radial_integrals_(0, i1, i2) = s.integrate(2) / y00;
                    }
                    
                    // non-spherical terms
                    for (int lm = 1; lm < lmmax; lm++)
                    {
                        for (int ir = 0; ir < nmtp; ir++)
                            s[ir] = symmetry_class()->radial_function(ir, i1, 0) * symmetry_class()->radial_function(ir, i2, 0) * veff(lm, ir);
                        s.interpolate();
                        h_radial_integrals_(lm, i1, i2) = s.integrate(2);
                    }
                }

            o_radial_integrals_.zero();
            for (int l = 0; l < type()->num_aw_descriptors(); l++)
                for (int order1 = 0; order1 < type()->indexr().num_rf(l); order1++)
                {
                    int idxrf1 = type()->indexr().index_by_l_order(l, order1);
                    for (int order2 = 0; order2 < type()->indexr().num_rf(l); order2++)
                    {
                        int idxrf2 = type()->indexr().index_by_l_order(l, order2);
                        
                        for (int ir = 0; ir < nmtp; ir++)
                            s[ir] = symmetry_class()->radial_function(ir, idxrf1, 0) * symmetry_class()->radial_function(ir, idxrf2, 0);
                        s.interpolate();
                        o_radial_integrals_(l, order1, order2) = s.integrate(2);
                    }
                }
        }

        inline int offset_aw()
        {
            return offset_aw_;  
        }

        inline double h_radial_integral(int lm, int idxrf1, int idxrf2)
        {
            return h_radial_integrals_(lm, idxrf1, idxrf2);
        }
};

};

#endif // __ATOM_H__
