#ifndef __ATOM_H__
#define __ATOM_H__

#include <vector>
#include "gaunt.h"
#include "atom_type.h"
#include "atom_symmetry_class.h"
#include "splindex.h"

/** \file atom.h
    
    \brief Data and methods specific to each atom in the unit cell.
*/
namespace sirius {

// TODO: return index of atom type, don't bother with atom_type_index_by_id

class Atom
{
    private:
    
        /// type of the given atom
        Atom_type* type_;

        /// symmetry class of the given atom
        Atom_symmetry_class* symmetry_class_;
        
        /// position in fractional coordinates
        vector3d<double> position_;
       
        /// vector field associated with the current site
        vector3d<double> vector_field_;

        /// MT potential
        mdarray<double, 2> veff_;

        /// radial integrals of the Hamiltonian 
        mdarray<double, 3> h_radial_integrals_;
        
        /// MT magnetic field
        mdarray<double, 2> beff_[3];

        /// radial integrals of the effective magnetic field
        mdarray<double, 4> b_radial_integrals_;

        /// number of magnetic dimensions
        int num_mag_dims_;
        
        /// maximum l for potential and magnetic field 
        int lmax_pot_;

        /// offset in the array of matching coefficients
        int offset_aw_;

        /// offset in the block of local orbitals of the Hamiltonian and overlap matrices and in the eigen-vectors
        int offset_lo_;

        /// offset in the wave-function array 
        int offset_wf_;

        /// unsymmetrized (sampled over IBZ) occupation matrix of the L(S)DA+U method
        mdarray<double_complex, 4> occupation_matrix_;
        
        /// U,J correction matrix of the L(S)DA+U method
        mdarray<double_complex, 4> uj_correction_matrix_;

        /// true if UJ correction is applied for the current atom
        bool apply_uj_correction_;

        /// orbital quantum number for UJ correction
        int uj_correction_l_;

        /// D_{ij} matrix of the pseudo-potential method
        mdarray<double_complex, 2> d_mtrx_;
    
    public:
    
        /// Constructor
        Atom(Atom_type* type__, double* position__, double* vector_field__);
        
        void init(int lmax_pot__, int num_mag_dims__, int offset_aw__, int offset_lo__, int offset_wf__);

        void init();
        
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
        void generate_radial_integrals(MPI_Comm& comm);
        void generate_radial_integrals();
        
        inline Atom_type* type()
        {
            return type_;
        }

        inline Atom_symmetry_class* symmetry_class()
        {
            return symmetry_class_;
        }

        inline int type_id()
        {
            return type_->id();
        }

        inline vector3d<double> position()
        {
            return position_;
        }

        inline void set_position(vector3d<double> position__)
        {
            position_ = position__;
        }
        
        inline double position(int i)
        {
            return position_[i];
        }
        
        inline vector3d<double> vector_field()
        {
            return vector_field_;
        }

        inline int symmetry_class_id()
        {
            if (symmetry_class()) 
            {
                return symmetry_class()->id();
            }
            else
            {
                return -1;
            }
        }

        inline void set_symmetry_class(Atom_symmetry_class* symmetry_class__)
        {
            symmetry_class_ = symmetry_class__;
        }

        void set_nonspherical_potential(double* veff__, double* beff__[3])
        {
            veff_.set_ptr(veff__);
            
            for (int j = 0; j < 3; j++) beff_[j].set_ptr(beff__[j]);
        }

        void sync_radial_integrals(int rank)
        {
            Platform::bcast(h_radial_integrals_.get_ptr(), (int)h_radial_integrals_.size(), rank);
            if (num_mag_dims_) Platform::bcast(b_radial_integrals_.get_ptr(), (int)b_radial_integrals_.size(), rank);
        }

        void sync_occupation_matrix(int rank)
        {
            Platform::bcast(occupation_matrix_.get_ptr(), (int)occupation_matrix_.size(), rank);
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

        inline double* h_radial_integrals(int idxrf1, int idxrf2)
        {
            return &h_radial_integrals_(0, idxrf1, idxrf2);
        }
        
        inline double* b_radial_integrals(int idxrf1, int idxrf2, int x)
        {
            return &b_radial_integrals_(0, idxrf1, idxrf2, x);
        }
        
        template <spin_block_t sblock>
        inline double_complex hb_radial_integrals_sum_L3(int idxrf1, int idxrf2, std::vector<gaunt_L3<double_complex> >& gnt)
        {
            double_complex zsum(0, 0);

            for (int i = 0; i < (int)gnt.size(); i++)
            {
                switch (sblock)
                {
                    case nm:
                    {
                        zsum += gnt[i].coef * h_radial_integrals_(gnt[i].lm3, idxrf1, idxrf2);
                        break;
                    }
                    case uu:
                    {
                        zsum += gnt[i].coef * (h_radial_integrals_(gnt[i].lm3, idxrf1, idxrf2) + 
                                               b_radial_integrals_(gnt[i].lm3, idxrf1, idxrf2, 0));
                        break;
                    }
                    case dd:
                    {
                        zsum += gnt[i].coef * (h_radial_integrals_(gnt[i].lm3, idxrf1, idxrf2) -
                                               b_radial_integrals_(gnt[i].lm3, idxrf1, idxrf2, 0));
                        break;
                    }
                    case ud:
                    {
                        zsum += gnt[i].coef * double_complex(b_radial_integrals_(gnt[i].lm3, idxrf1, idxrf2, 1), 
                                                       -b_radial_integrals_(gnt[i].lm3, idxrf1, idxrf2, 2));
                        break;
                    }
                    case du:
                    {
                        zsum += gnt[i].coef * double_complex(b_radial_integrals_(gnt[i].lm3, idxrf1, idxrf2, 1), 
                                                        b_radial_integrals_(gnt[i].lm3, idxrf1, idxrf2, 2));
                        break;
                    }
                }

            }
            return zsum;
        }

        inline int num_mt_points()
        {
            return type_->num_mt_points();
        }

        inline Radial_grid& radial_grid()
        {
            return type_->radial_grid();
        }

        inline double mt_radius()
        {
            return type_->mt_radius();
        }

        inline int zn()
        {
            return type_->zn();
        }

        inline void set_occupation_matrix(const double_complex* source)
        {
            memcpy(occupation_matrix_.get_ptr(), source, 16 * 16 * 2 * 2 * sizeof(double_complex));
            apply_uj_correction_ = false;
        }
        
        inline void get_occupation_matrix(double_complex* destination)
        {
            memcpy(destination, occupation_matrix_.get_ptr(), 16 * 16 * 2 * 2 * sizeof(double_complex));
        }

        inline void set_uj_correction_matrix(const int l, const double_complex* source)
        {
            uj_correction_l_ = l;
            memcpy(uj_correction_matrix_.get_ptr(), source, 16 * 16 * 2 * 2 * sizeof(double_complex));
            apply_uj_correction_ = true;
        }

        inline bool apply_uj_correction()
        {
            return apply_uj_correction_;
        }

        inline int uj_correction_l()
        {
            return uj_correction_l_;
        }

        inline double_complex uj_correction_matrix(int lm1, int lm2, int ispn1, int ispn2)
        {
             return uj_correction_matrix_(lm1, lm2, ispn1, ispn2);
        }

        inline double_complex& d_mtrx(int xi1, int xi2)
        {
            return d_mtrx_(xi1, xi2);
        }
        
        inline mdarray<double_complex, 2>& d_mtrx()
        {
            return d_mtrx_;
        }
};

};

#endif // __ATOM_H__
