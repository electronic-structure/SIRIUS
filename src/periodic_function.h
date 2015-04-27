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

/** \file periodic_function.h
 *   
 *  \brief Contains declaration and partial implementation of sirius::Periodic_function class.
 */

#ifndef __PERIODIC_FUNCTION_H__
#define __PERIODIC_FUNCTION_H__

#include "mdarray.h"
#include "spheric_function.h"
#include "mixer.h"

// TODO: this implementation is better, however the distinction between local and global periodic functions is
//       still not very clear
namespace sirius
{

/// Representation of the periodical function on the muffin-tin geometry.
/** Inside each muffin-tin the spherical expansion is used:
 *   \f[
 *       f({\bf r}) = \sum_{\ell m} f_{\ell m}(r) Y_{\ell m}(\hat {\bf r})
 *   \f]
 *   or
 *   \f[
 *       f({\bf r}) = \sum_{\ell m} f_{\ell m}(r) R_{\ell m}(\hat {\bf r})
 *   \f]
 *   In the interstitial region function is stored on the real-space grid or as a Fourier series:
 *   \f[
 *       f({\bf r}) = \sum_{{\bf G}} f({\bf G}) e^{i{\bf G}{\bf r}}
 *   \f]
 *
 *   The following terminology is used to describe the distribution of the function:
 *       - global function: the whole copy of the function is stored on each MPI rank. Ranks should take care about the
 *         syncronization of the data.
 *       - local function: the function is distributed across the MPI ranks. 
 *
 *   \note In order to check if the function is defined as global or as distributed, check the f_mt_ and f_it_ pointers.
 *         If the function is global, the pointers should not be null.
 */
template<typename T> 
class Periodic_function
{ 
    protected:

        Periodic_function(const Periodic_function<T>& src);

        Periodic_function<T>& operator=(const Periodic_function<T>& src);

    private:
        
        typedef typename type_wrapper<T>::complex_t complex_t; 
        
        Unit_cell* unit_cell_;

        Step_function* step_function_;

        /// Alias for FFT driver.
        FFT3D<CPU>* fft_;

        electronic_structure_method_t esm_type_;

        /// Local part of muffin-tin functions.
        mdarray<Spheric_function<spectral, T>, 1> f_mt_local_;
        
        /// global muffin-tin array 
        mdarray<T, 3> f_mt_;

        /// local part of interstitial array
        mdarray<T, 1> f_it_local_;
        
        /// global interstitial array
        mdarray<T, 1> f_it_;

        /// plane-wave expansion coefficients
        mdarray<complex_t, 1> f_pw_;

        int angular_domain_size_;
        
        /// number of plane-wave expansion coefficients
        int num_gvec_;

        Communicator comm_;

        splindex<block> spl_fft_size_;

        /// Set pointer to local part of muffin-tin functions
        void set_local_mt_ptr()
        {
            for (int ialoc = 0; ialoc < (int)unit_cell_->spl_num_atoms().local_size(); ialoc++)
            {
                int ia = unit_cell_->spl_num_atoms(ialoc);
                f_mt_local_(ialoc) = Spheric_function<spectral, T>(&f_mt_(0, 0, ia), angular_domain_size_, unit_cell_->atom(ia)->radial_grid());
            }
        }
        
        /// Set pointer to local part of interstitial array
        void set_local_it_ptr()
        {
            f_it_local_ = mdarray<T, 1>(&f_it_(spl_fft_size_.global_offset()), spl_fft_size_.local_size());
        }

    public:

        /// Constructor
        Periodic_function(Global& parameters__, int angular_domain_size, int num_gvec, Communicator const& comm__);
        
        /// Destructor
        ~Periodic_function();
        
        /// Allocate memory
        void allocate(bool allocate_global_mt, bool allocate_global_it);

        /// Zero the function.
        void zero();
        
        /// Syncronize global function.
        void sync(bool sync_mt, bool sync_it);

        /// Copy from source
        //void copy(Periodic_function<T>* src);
        inline void copy_to_global_ptr(T* f_mt__, T* f_it__);

        /// Add the function
        void add(Periodic_function<T>* g);

        T integrate(std::vector<T>& mt_val, T& it_val);

        template <index_domain_t index_domain>
        inline T& f_mt(int idx0, int idx1, int ia);
        
        /** \todo write and read distributed functions */
        void hdf5_write(HDF5_tree h5f);

        void hdf5_read(HDF5_tree h5f);

        size_t size();

        size_t pack(size_t offset, Mixer<double>* mixer);
        
        size_t unpack(T const* array);
       
        /// Set the global pointer to the muffin-tin part
        void set_mt_ptr(T* mt_ptr__)
        {
            f_mt_ = mdarray<T, 3>(mt_ptr__, angular_domain_size_, unit_cell_->max_num_mt_points(), unit_cell_->num_atoms());
            set_local_mt_ptr();
        }

        /// Set the global pointer to the interstitial part
        void set_it_ptr(T* it_ptr__)
        {
            f_it_ = mdarray<T, 1>(it_ptr__, fft_->size());
            set_local_it_ptr();
        }

        inline Spheric_function<spectral, T>& f_mt(int ialoc)
        {
            return f_mt_local_(ialoc);
        }

        template <index_domain_t index_domain>
        inline T& f_it(int ir)
        {
            switch (index_domain)
            {
                case local:
                {
                    return f_it_local_(ir);
                }
                case global:
                {
                    return f_it_(ir);
                }
            }
        }
        
        inline complex_t& f_pw(int ig)
        {
            return f_pw_(ig);
        }

        double value(Global& parameters_, vector3d<double>& vc)
        {
            int ja, jr;
            double dr, tp[2];
        
            if (unit_cell_->is_point_in_mt(vc, ja, jr, dr, tp)) 
            {
                int lmax = Utils::lmax_by_lmmax(angular_domain_size_);
                std::vector<double> rlm(angular_domain_size_);
                SHT::spherical_harmonics(lmax, tp[0], tp[1], &rlm[0]);
                double p = 0.0;
                for (int lm = 0; lm < angular_domain_size_; lm++)
                {
                    double d = (f_mt_(lm, jr + 1, ja) - f_mt_(lm, jr, ja)) / 
                               (unit_cell_->atom(ja)->type()->radial_grid(jr + 1) - unit_cell_->atom(ja)->type()->radial_grid(jr));
        
                    p += rlm[lm] * (f_mt_(lm, jr, ja) + d * dr);
                }
                return p;
            }
            else
            {
                double p = 0.0;
                for (int ig = 0; ig < num_gvec_; ig++)
                {
                    vector3d<double> vgc = parameters_.reciprocal_lattice()->gvec_cart(ig);
                    p += real(f_pw_(ig) * exp(double_complex(0.0, vc * vgc)));
                }
                return p;
            }
        }

        int64_t hash()
        {
            int64_t h = Utils::hash(&f_it_(0), fft_->size() * sizeof(T));
            h += Utils::hash(&f_pw_(0), num_gvec_ * sizeof(double_complex), h);
            return h;
        }

        void fft_transform(int direction__)
        {
            switch (direction__)
            {
                case 1:
                {
                    fft_->input(fft_->num_gvec(), fft_->index_map(), &f_pw(0));
                    fft_->transform(1);
                    fft_->output(&f_it<global>(0));
                    break;
                }
                case -1:
                {
                    fft_->input(&f_it<global>(0));
                    fft_->transform(-1);
                    fft_->output(fft_->num_gvec(), fft_->index_map(), &f_pw(0));
                    break;
                }
                default:
                {
                    TERMINATE("wrong fft direction");
                }
            }
        }
        
        mdarray<T, 3>& f_mt()
        {
            return f_mt_;
        }

        mdarray<T, 1>& f_it()
        {
            return f_it_;
        }
};

#include "periodic_function.hpp"

};

#endif // __PERIODIC_FUNCTION_H__

