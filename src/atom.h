// Copyright (c) 2013-2014 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that 
// the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the 
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions 
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file atom.h
 *   
 *  \brief Contains declaration and partial implementation of sirius::Atom class.
 */

#ifndef __ATOM_H__
#define __ATOM_H__

#include <vector>
#include "gaunt.h"
#include "atom_type.h"
#include "atom_symmetry_class.h"
#include "splindex.h"
#include "spheric_function.h"

namespace sirius {

/// Data and methods specific to the actual atom in the unit cell.
class Atom
{
    private:
    
        /// Type of the given atom.
        Atom_type* type_;

        /// Symmetry class of the given atom.
        Atom_symmetry_class* symmetry_class_;
        
        /// Position in fractional coordinates.
        vector3d<double> position_;
       
        /// Vector field associated with the current site.
        vector3d<double> vector_field_;

        /// Muffin-tin potential.
        mdarray<double, 2> veff_;

        /// Radial integrals of the Hamiltonian.
        mdarray<double, 3> h_radial_integrals_;
        
        /// Muffin-tin magnetic field.
        mdarray<double, 2> beff_[3];

        /// Radial integrals of the effective magnetic field.
        mdarray<double, 4> b_radial_integrals_;

        /// Number of magnetic dimensions.
        int num_mag_dims_;
        
        /// Maximum l for potential and magnetic field.
        int lmax_pot_;

        /// Offset in the array of matching coefficients.
        int offset_aw_;

        /// Offset in the block of local orbitals of the Hamiltonian and overlap matrices and in the eigen-vectors.
        int offset_lo_;

        /// Offset in the wave-function array.
        int offset_wf_;

        /// Unsymmetrized (sampled over IBZ) occupation matrix of the L(S)DA+U method.
        mdarray<double_complex, 4> occupation_matrix_;
        
        /// U,J correction matrix of the L(S)DA+U method
        mdarray<double_complex, 4> uj_correction_matrix_;

        /// True if UJ correction is applied for the current atom.
        bool apply_uj_correction_;

        /// Orbital quantum number for UJ correction.
        int uj_correction_l_;

        /// D_{ij} matrix of the pseudo-potential method.
        mdarray<double_complex, 2> d_mtrx_;
    
    public:
    
        /// Constructor.
        Atom(Atom_type* type__, double* position__, double* vector_field__);
        
        /// Initialize atom.
        void init(int lmax_pot__, int num_mag_dims__, int offset_aw__, int offset_lo__, int offset_wf__);

        /// Generate radial Hamiltonian and effective magnetic field integrals
        /** Hamiltonian operator has the following representation inside muffin-tins:
         *  \f[
         *      \hat H=-\frac{1}{2}\nabla^2 + \sum_{\ell m} V_{\ell m}(r) R_{\ell m}(\hat {\bf r}) =
         *        \underbrace{-\frac{1}{2} \nabla^2+V_{00}(r)R_{00}}_{H_{s}(r)} +\sum_{\ell=1} \sum_{m=-\ell}^{\ell} 
         *         V_{\ell m}(r) R_{\ell m}(\hat {\bf r}) = \sum_{\ell m} \widetilde V_{\ell m}(r) R_{\ell m}(\hat {\bf r})
         *  \f]
         *  where
         *  \f[
         *      \widetilde V_{\ell m}(r)=\left\{ \begin{array}{ll}
         *        \frac{H_{s}(r)}{R_{00}} & \ell = 0 \\
         *        V_{\ell m}(r) & \ell > 0 \end{array} \right.
         *  \f]
         */
        void generate_radial_integrals(MPI_Comm& comm);
        
        /// Return pointer to corresponding atom type class.
        inline Atom_type* type()
        {
            return type_;
        }

        /// Return pointer to corresponding atom symmetry class.
        inline Atom_symmetry_class* symmetry_class()
        {
            return symmetry_class_;
        }

        /// Return atom type id.
        inline int type_id()
        {
            return type_->id();
        }
        
        /// Return atom position in fractional coordinates.
        inline vector3d<double> position()
        {
            return position_;
        }
        
        /// Set atom position in fractional coordinates.
        inline void set_position(vector3d<double> position__)
        {
            position_ = position__;
        }
        
        /// Return specific coordinate of position.
        inline double position(int i)
        {
            return position_[i];
        }
        
        /// Return vector field.
        inline vector3d<double> vector_field()
        {
            return vector_field_;
        }
        
        /// Return id of the symmetry class.
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

        /// Set symmetry class of the atom.
        inline void set_symmetry_class(Atom_symmetry_class* symmetry_class__)
        {
            symmetry_class_ = symmetry_class__;
        }
        
        /// Set muffin-tin potential and magnetic field.
        void set_nonspherical_potential(double* veff__, double* beff__[3])
        {
            veff_.set_ptr(veff__);
            
            for (int j = 0; j < 3; j++) beff_[j].set_ptr(beff__[j]);
        }

        void sync_radial_integrals(int rank)
        {
            Platform::bcast(h_radial_integrals_.ptr(), (int)h_radial_integrals_.size(), rank);
            if (num_mag_dims_) Platform::bcast(b_radial_integrals_.ptr(), (int)b_radial_integrals_.size(), rank);
        }

        void sync_occupation_matrix(int rank)
        {
            Platform::bcast(occupation_matrix_.ptr(), (int)occupation_matrix_.size(), rank);
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

        inline double radial_grid(int idx)
        {
            return type_->radial_grid(idx);
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
            memcpy(occupation_matrix_.ptr(), source, 16 * 16 * 2 * 2 * sizeof(double_complex));
            apply_uj_correction_ = false;
        }
        
        inline void get_occupation_matrix(double_complex* destination)
        {
            memcpy(destination, occupation_matrix_.ptr(), 16 * 16 * 2 * 2 * sizeof(double_complex));
        }

        inline void set_uj_correction_matrix(const int l, const double_complex* source)
        {
            uj_correction_l_ = l;
            memcpy(uj_correction_matrix_.ptr(), source, 16 * 16 * 2 * 2 * sizeof(double_complex));
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

        inline int mt_basis_size()
        {
            return type_->mt_basis_size();
        }

        inline int mt_aw_basis_size()
        {
            return type_->mt_aw_basis_size();
        }

        inline int mt_lo_basis_size()
        {
            return type_->mt_lo_basis_size();
        }
};

};

#endif // __ATOM_H__
