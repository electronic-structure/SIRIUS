// Copyright (c) 2013-2016 Anton Kozhevnikov, Thomas Schulthess
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

#include "simulation_context.h"
#include "mdarray.h"
#include "spheric_function.h"
#include "smooth_periodic_function.h"
#include "mixer.h"

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
 */
template<typename T> 
class Periodic_function: public Smooth_periodic_function<T>
{ 
    protected:

        /* forbid copy constructor */
        Periodic_function(const Periodic_function<T>& src) = delete;
        
        /* forbid assigment operator */
        Periodic_function<T>& operator=(const Periodic_function<T>& src) = delete;

    private:
       
        /// Complex counterpart for a given type T.
        typedef typename type_wrapper<T>::complex_t complex_t; 

        Simulation_parameters const& parameters_;
        
        Unit_cell const& unit_cell_;

        Step_function const& step_function_;

        Communicator const& comm_;

        /// Local part of muffin-tin functions.
        mdarray<Spheric_function<spectral, T>, 1> f_mt_local_;
        
        /// Global muffin-tin array 
        mdarray<T, 3> f_mt_;

        /// Plane-wave expansion coefficients
        mdarray<complex_t, 1> f_pw_;

        /// Angular domain size.
        int angular_domain_size_;
        
        /// Set pointer to local part of muffin-tin functions
        void set_local_mt_ptr()
        {
            for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++)
            {
                int ia = unit_cell_.spl_num_atoms(ialoc);
                f_mt_local_(ialoc) = Spheric_function<spectral, T>(&f_mt_(0, 0, ia), angular_domain_size_, unit_cell_.atom(ia).radial_grid());
            }
        }
        
    public:

        /// Constructor
        Periodic_function(Simulation_context& ctx__,
                          int angular_domain_size__,
                          int allocate_pw__)
            : Smooth_periodic_function<T>(ctx__.fft(), ctx__.gvec_fft_distr()),
              parameters_(ctx__),
              unit_cell_(ctx__.unit_cell()), 
              step_function_(ctx__.step_function()),
              comm_(ctx__.comm()),
              angular_domain_size_(angular_domain_size__)
        {
            if (allocate_pw__)
            {
                f_pw_ = mdarray<double_complex, 1>(ctx__.gvec().num_gvec());
                this->f_pw_local_ = mdarray<double_complex, 1>(&f_pw_[this->gvec_fft_distr().offset_gvec_fft()], this->fft_->local_size());
            }
        
            if (parameters_.full_potential())
                f_mt_local_ = mdarray<Spheric_function<spectral, T>, 1>(unit_cell_.spl_num_atoms().local_size());
        }
        
        /// Allocate memory for muffin-tin part.
        void allocate_mt(bool allocate_global__)
        {
            if (parameters_.full_potential())
            {
                if (allocate_global__)
                {
                    f_mt_ = mdarray<T, 3>(angular_domain_size_, unit_cell_.max_num_mt_points(), unit_cell_.num_atoms());
                    set_local_mt_ptr();
                }
                else
                {
                    for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++) 
                    {
                        int ia = unit_cell_.spl_num_atoms(ialoc);
                        f_mt_local_(ialoc) = Spheric_function<spectral, T>(angular_domain_size_, unit_cell_.atom(ia).radial_grid());
                    }
                }
            }
        }

        /// Syncronize global muffin-tin array.
        void sync_mt()
        {
            runtime::Timer t("sirius::Periodic_function::sync_mt");
            assert(f_mt_.size() != 0); 

            int ld = angular_domain_size_ * unit_cell_.max_num_mt_points(); 
            comm_.allgather(&f_mt_(0, 0, 0), ld * unit_cell_.spl_num_atoms().global_offset(), ld * unit_cell_.spl_num_atoms().local_size());
        }

        /// Zero the function.
        void zero()
        {
            f_mt_.zero();
            this->f_rg_.zero();
            f_pw_.zero();
            if (parameters_.full_potential())
            {
                for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++) f_mt_local_(ialoc).zero();
            }
        }
        
        /// Copy from source
        //void copy(Periodic_function<T>* src);
        inline void copy_to_global_ptr(T* f_mt__, T* f_it__);

        /// Add the function
        void add(Periodic_function<T>* g);

        T integrate(std::vector<T>& mt_val, T& it_val);

        template <index_domain_t index_domain>
        inline T& f_mt(int idx0, int ir, int ia)
        {
            switch (index_domain)
            {
                case local:
                {
                    return f_mt_local_(ia)(idx0, ir);
                }
                case global:
                {
                    return f_mt_(idx0, ir, ia);
                }
            }
        }
        
        /** \todo write and read distributed functions */
        void hdf5_write(HDF5_tree h5f);

        void hdf5_read(HDF5_tree h5f);

        size_t size();

        size_t pack(size_t offset, Mixer<double>* mixer);
        
        size_t unpack(T const* array);
       
        /// Set the global pointer to the muffin-tin part
        void set_mt_ptr(T* mt_ptr__)
        {
            f_mt_ = mdarray<T, 3>(mt_ptr__, angular_domain_size_, unit_cell_.max_num_mt_points(), unit_cell_.num_atoms());
            set_local_mt_ptr();
        }

        /// Set the global pointer to the interstitial part
        void set_rg_ptr(T* rg_ptr__)
        {
            this->f_rg_ = mdarray<T, 1>(rg_ptr__, this->fft_->local_size());
        }

        inline Spheric_function<spectral, T>& f_mt(int ialoc__)
        {
            return f_mt_local_(ialoc__);
        }

        inline Spheric_function<spectral, T> const& f_mt(int ialoc__) const
        {
            return f_mt_local_(ialoc__);
        }

        inline complex_t& f_pw(int ig__)
        {
            return f_pw_(ig__);
        }

        inline complex_t& f_pw(vector3d<int> const& G__)
        {
            return f_pw_(this->gvec_fft_distr().gvec().index_by_gvec(G__));
        }

        //double value(vector3d<double>& vc)
        //{
        //    int ja, jr;
        //    double dr, tp[2];
        //
        //    if (unit_cell_.is_point_in_mt(vc, ja, jr, dr, tp)) 
        //    {
        //        int lmax = Utils::lmax_by_lmmax(angular_domain_size_);
        //        std::vector<double> rlm(angular_domain_size_);
        //        SHT::spherical_harmonics(lmax, tp[0], tp[1], &rlm[0]);
        //        double p = 0.0;
        //        for (int lm = 0; lm < angular_domain_size_; lm++)
        //        {
        //            double d = (f_mt_(lm, jr + 1, ja) - f_mt_(lm, jr, ja)) / 
        //                       (unit_cell_.atom(ja).type().radial_grid(jr + 1) - unit_cell_.atom(ja).type().radial_grid(jr));
        //
        //            p += rlm[lm] * (f_mt_(lm, jr, ja) + d * dr);
        //        }
        //        return p;
        //    }
        //    else
        //    {
        //        double p = 0.0;
        //        for (int ig = 0; ig < num_gvec_; ig++)
        //        {
        //            vector3d<double> vgc = gvec_->cart(ig);
        //            p += std::real(f_pw_(ig) * std::exp(double_complex(0.0, vc * vgc)));
        //        }
        //        return p;
        //    }
        //}

        inline T checksum_rg() const
        {
            T cs = this->f_rg_.checksum();
            this->fft_->comm().allreduce(&cs, 1);
            return cs;
        }

        inline complex_t checksum_pw() const
        {
            return f_pw_.checksum();
        }

        //int64_t hash()
        //{
        //    STOP();
        //    int64_t h = this->f_rg_.hash();
        //    h += f_pw_.hash();
        //    return h;
        //}

        void fft_transform(int direction__)
        {
            Smooth_periodic_function<T>::fft_transform(direction__);

            runtime::Timer t("sirius::Periodic_function::fft_transform|comm");
            this->fft_->comm().allgather(&f_pw_(0), this->gvec_fft_distr().offset_gvec_fft(), this->gvec_fft_distr().num_gvec_fft());
        }
        
        mdarray<T, 3>& f_mt()
        {
            return f_mt_;
        }

        /// Compute inner product <f|g>
        T inner(Periodic_function<T> const* g__)
        {
            runtime::Timer t("sirius::Periodic_function::inner");
        
            assert(this->fft_ == g__->fft_);
            assert(&step_function_ == &g__->step_function_);
            assert(&unit_cell_ == &g__->unit_cell_);
            assert(&comm_ == &g__->comm_);
            
            T result = 0.0;
            T ri = 0.0;
            
            if (!parameters_.full_potential())
            {
                #pragma omp parallel
                {
                    T ri_t = 0;
                    
                    #pragma omp for
                    for (int irloc = 0; irloc < this->fft_->local_size(); irloc++)
                        ri_t += type_wrapper<T>::conjugate(this->f_rg(irloc)) * g__->f_rg(irloc);
        
                    #pragma omp critical
                    ri += ri_t;
                }
            }
            else
            {
                for (int irloc = 0; irloc < this->fft_->local_size(); irloc++)
                {
                    ri += type_wrapper<T>::conjugate(this->f_rg(irloc)) * g__->f_rg(irloc) * 
                          this->step_function_.theta_r(irloc);
                }
            }
                    
            ri *= (unit_cell_.omega() / this->fft_->size());
            this->fft_->comm().allreduce(&ri, 1);
            
            if (parameters_.full_potential())
            {
                for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++)
                    result += sirius::inner(f_mt(ialoc), g__->f_mt(ialoc));
                comm_.allreduce(&result, 1);
            }
        
            return result + ri;
        }

};

#include "periodic_function.hpp"

};

#endif // __PERIODIC_FUNCTION_H__

