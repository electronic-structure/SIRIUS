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

        /// MT potential
        mdarray<double,2> veff_;

        /// radial integrals of the Hamiltonian 
        mdarray<double,3> h_radial_integrals_;
        
        /// MT magnetic field
        mdarray<double,2> beff_[3];

        /// radial integrals of the effective magnetic field
        mdarray<double,4> b_radial_integrals_;

        /// number of magnetic dimensions
        int num_mag_dims_;
        
        /// maximum l for potential and magnetic field 
        int lmax_pot_;

        /// offset in the array of matching coefficients and in the array of wave-functions
        int offset_aw_;

        /// offset in the block of local orbitals in Hamiltonian and overlap matrices and in the array of wave-functions
        int offset_lo_;

        /// offset in the wave-function array 
        int offset_wf_;
    
    public:
    
        Atom(AtomType* type__, double* position__, double* vector_field__) : type_(type__),
                                                                             symmetry_class_(NULL),
                                                                             offset_aw_(-1),
                                                                             offset_lo_(-1),
                                                                             offset_wf_(-1)
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

        inline void get_position(double position__[3])
        {
            for (int i = 0; i < 3; i++)
                position__[i] = position_[i];
        }
        
        inline double* position()
        {
            return position_;
        }
        
        inline double position(int i)
        {
            return position_[i];
        }
        
        inline double* vector_field()
        {
            return vector_field_;
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

        void init(int lmax_pot__, int num_mag_dims__, int offset_aw__, int offset_lo__, int offset_wf__)
        {
            assert(symmetry_class());
            assert(lmax_pot__ >= 0);
            assert(offset_aw__ >= 0);
            
            offset_aw_ = offset_aw__;
            offset_lo_ = offset_lo__;
            offset_wf_ = offset_wf__;

            lmax_pot_ = lmax_pot__;
            num_mag_dims_ = num_mag_dims__;

            int lmmax = lmmax_by_lmax(lmax_pot_);

            h_radial_integrals_.set_dimensions(lmmax, type()->indexr().size(), type()->indexr().size());
            h_radial_integrals_.allocate();
            
            veff_.set_dimensions(lmmax, type()->num_mt_points());
            
            b_radial_integrals_.set_dimensions(lmmax, type()->indexr().size(), type()->indexr().size(), num_mag_dims_);
            b_radial_integrals_.allocate();
            
            for (int j = 0; j < num_mag_dims_; j++)
                beff_[j].set_dimensions(lmmax, type()->num_mt_points());
        }

        void set_nonspherical_potential(double* veff__, double** beff__)
        {
            veff_.set_ptr(veff__);
            
            for (int j = 0; j < num_mag_dims_; j++)
                beff_[j].set_ptr(beff__[j]);
        }

        /// Generate radial Hamiltonian and effective magnetic field integrals

        /** Hamiltonian operator has the following representation inside muffin-tins:
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
        void generate_radial_integrals()
        {
            Timer t("sirius::Atom::generate_radial_integrals");
            
            int lmmax = lmmax_by_lmax(lmax_pot_);
            int nmtp = type()->num_mt_points();

            h_radial_integrals_.zero();
            if (num_mag_dims_) b_radial_integrals_.zero();
            
            // copy spherical integrals
            for (int i2 = 0; i2 < type()->indexr().size(); i2++)
                for (int i1 = 0; i1 < type()->indexr().size(); i1++)
                    h_radial_integrals_(0, i1, i2) = symmetry_class()->h_spherical_integral(i1, i2);

            #pragma omp parallel default(shared)
            {
                Spline<double> s(nmtp, type()->radial_grid());
                std::vector<double> v(nmtp);

                #pragma omp for
                for (int lm = 1; lm < lmmax; lm++)
                {
                    for (int i2 = 0; i2 < type()->indexr().size(); i2++)
                    {
                        for (int ir = 0; ir < nmtp; ir++)
                            v[ir] = symmetry_class()->radial_function(ir, i2) * veff_(lm, ir);
                        
                        for (int i1 = 0; i1 <= i2; i1++)
                        {
                            for (int ir = 0; ir < nmtp; ir++)
                                s[ir] = symmetry_class()->radial_function(ir, i1) * v[ir];
                            
                            s.interpolate();
                            h_radial_integrals_(lm, i1, i2) = h_radial_integrals_(lm, i2, i1) = s.integrate(2);
                        }
                    }
                }
            }

            for (int j = 0; j < num_mag_dims_; j++)
            {
                #pragma omp parallel default(shared)
                {
                    Spline<double> s(nmtp, type()->radial_grid());
                    std::vector<double> v(nmtp);

                    #pragma omp for
                    for (int lm = 0; lm < lmmax; lm++)
                    {
                        for (int i2 = 0; i2 < type()->indexr().size(); i2++)
                        {
                            for (int ir = 0; ir < nmtp; ir++)
                                v[ir] = symmetry_class()->radial_function(ir, i2) * beff_[j](lm, ir);
                            
                            for (int i1 = 0; i1 <= i2; i1++)
                            {
                                for (int ir = 0; ir < nmtp; ir++)
                                    s[ir] = symmetry_class()->radial_function(ir, i1) * v[ir];
                                
                                s.interpolate();
                                b_radial_integrals_(lm, i1, i2, j) = b_radial_integrals_(lm, i2, i1, j) = s.integrate(2);
                            }
                        }
                    }
                }
            }
       }

        inline int offset_aw()
        {
            assert(offset_aw_ >= 0);

            return offset_aw_;  
        }
        
        inline int offset_lo()
        {
            assert(offset_lo_ >= 0);

            return offset_lo_;  
        }
        
        inline int offset_wf()
        {
            assert(offset_wf_ >= 0);

            return offset_wf_;  
        }

        inline double h_radial_integral(int lm, int idxrf1, int idxrf2)
        {
            return h_radial_integrals_(lm, idxrf1, idxrf2);
        }
        
        inline double* h_radial_integral(int idxrf1, int idxrf2)
        {
            return &h_radial_integrals_(0, idxrf1, idxrf2);
        }
        
        inline double* b_radial_integral(int idxrf1, int idxrf2, int x)
        {
            return &b_radial_integrals_(0, idxrf1, idxrf2, x);
        }
        
        inline int num_mt_points()
        {
            return type_->num_mt_points();
        }
};

};

#endif // __ATOM_H__
