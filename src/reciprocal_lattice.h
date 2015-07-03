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

/** \file reciprocal_lattice.h
 *
 *  \brief Contains definition and partial implementation of sirius::Reciprocal_lattice class. 
 */

#ifndef __RECIPROCAL_LATTICE_H__
#define __RECIPROCAL_LATTICE_H__

#include "unit_cell.h"
#include "fft3d.h"
#include "sbessel_pw.h"

namespace sirius {

/// Reciprocal lattice of the crystal.
class Reciprocal_lattice
{
    private:

        /// Corresponding Unit_cell class instance. 
        Unit_cell const& unit_cell_;

        /// Type of electronic structure method.
        electronic_structure_method_t esm_type_;
        
        /// Reciprocal lattice vectors in column order.
        /** The following convention is used:
         *  \f[
         *    \vec a_{i} \vec b_{j} = 2 \pi \delta_{ij}
         *  \f]
         *  or in matrix notation
         *  \f[
         *    {\bf A} {\bf B}^{T} = 2 \pi {\bf I}
         *  \f]
         */
        matrix3d<double> reciprocal_lattice_vectors_;
        
        /// Inverse matrix or reciprocal vectors.
        matrix3d<double> inverse_reciprocal_lattice_vectors_;
        
        /// FFT wrapper for dense grid.
        FFT3D<CPU>* fft_;

        Gvec const& gvec_;

        /// Cached values of G-vector phase factors 
        mdarray<double_complex, 2> gvec_phase_factors_;

        Communicator const& comm_;

        void init(int lmax);

        void fix_q_radial_functions(mdarray<double, 4>& qrf);

        void generate_q_radial_integrals(int lmax, mdarray<double, 4>& qrf, mdarray<double, 4>& qri);

        void generate_q_pw(int lmax, mdarray<double, 4>& qri);

    public:
        
        Reciprocal_lattice(Unit_cell const& unit_cell__, 
                           electronic_structure_method_t esm_type__,
                           FFT3D<CPU>* fft__,
                           Gvec const& gvec__,
                           int lmax__,
                           Communicator const& comm__);

        ~Reciprocal_lattice();
  
        /// Make periodic function out of form factors
        /** Return vector of plane-wave coefficients */
        std::vector<double_complex> make_periodic_function(mdarray<double, 2>& ffac, int ngv) const;
        
        /// Phase factors \f$ e^{i {\bf G} {\bf r}_{\alpha}} \f$
        inline double_complex gvec_phase_factor(int ig__, int ia__) const
        {
            return std::exp(double_complex(0.0, twopi * (vector3d<int>(fft_->gvec(ig__)) * unit_cell_.atom(ia__)->position())));
        }
       
        //== /// Return length of G-vector.
        //== inline double gvec_len(int ig__) const
        //== {
        //==     return fft_->gvec_len(ig__);
        //== }
        
        inline int gvec_index(vector3d<int> gvec__) const
        {
            return fft_->gvec_index(gvec__);
        }

        /// FFT index for a given G-vector index
        inline int index_map(int ig__) const
        {
            return fft_->index_map(ig__);
        }

        /// Pointer to FFT index array
        inline int const* index_map() const
        {
            return fft_->index_map();
        }

        /// Number of G-vector shells within plane-wave cutoff
        inline int num_gvec_shells_inner() const
        {
            return fft_->num_gvec_shells_inner();
        }

        inline int num_gvec_shells_total() const
        {
            return fft_->num_gvec_shells_total();
        }

        /// Index of G-vector shell
        inline int gvec_shell(int ig__) const
        {
            return fft_->gvec_shell(ig__);
        }

        inline double gvec_shell_len(int igs__) const
        {
            return fft_->gvec_shell_len(igs__);
        }
        
        /// Return global index of G1-G2 vector
        inline int index_g12(int ig1__, int ig2__) const
        {
            vector3d<int> v = fft_->gvec(ig1__) - fft_->gvec(ig2__);
            return fft_->gvec_index(v);
        }
        
        inline int index_g12_safe(int ig1__, int ig2__) const
        {
            vector3d<int> v = fft_->gvec(ig1__) - fft_->gvec(ig2__);
            if (v[0] >= fft_->grid_limits(0).first && v[0] <= fft_->grid_limits(0).second &&
                v[1] >= fft_->grid_limits(1).first && v[1] <= fft_->grid_limits(1).second &&
                v[2] >= fft_->grid_limits(2).first && v[2] <= fft_->grid_limits(2).second)
            {
                return fft_->gvec_index(v);
            }
            else
            {
                return -1;
            }
        }

        //== inline splindex<block> const& spl_num_gvec() const
        //== {
        //==     return spl_num_gvec_;
        //== }
        //== 
        //== inline int spl_num_gvec(int igloc__) const
        //== {
        //==     return static_cast<int>(spl_num_gvec_[igloc__]);
        //== }
        
        //== inline double_complex gvec_ylm(int lm, int igloc) const
        //== {
        //==     return gvec_ylm_(lm, igloc);
        //== }

        void write_periodic_function()
        {
            //== mdarray<double, 3> vloc_3d_map(&vloc_it[0], fft_->size(0), fft_->size(1), fft_->size(2));
            //== int nx = fft_->size(0);
            //== int ny = fft_->size(1);
            //== int nz = fft_->size(2);

            //== auto p = parameters_.unit_cell()->unit_cell_parameters();

            //== FILE* fout = fopen("potential.ted", "w");
            //== fprintf(fout, "%s\n", parameters_.unit_cell()->chemical_formula().c_str());
            //== fprintf(fout, "%16.10f %16.10f %16.10f  %16.10f %16.10f %16.10f\n", p.a, p.b, p.c, p.alpha, p.beta, p.gamma);
            //== fprintf(fout, "%i %i %i\n", nx + 1, ny + 1, nz + 1);
            //== for (int i0 = 0; i0 <= nx; i0++)
            //== {
            //==     for (int i1 = 0; i1 <= ny; i1++)
            //==     {
            //==         for (int i2 = 0; i2 <= nz; i2++)
            //==         {
            //==             fprintf(fout, "%14.8f\n", vloc_3d_map(i0 % nx, i1 % ny, i2 % nz));
            //==         }
            //==     }
            //== }
            //== fclose(fout);
        }

        matrix3d<double> const& reciprocal_lattice_vectors() const
        {
            return reciprocal_lattice_vectors_;
        }
};

};

#endif

