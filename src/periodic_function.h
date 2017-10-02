// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
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
#include "spheric_function.h"
#include "smooth_periodic_function.h"

namespace sirius {

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
    private:
       
        /// Complex counterpart for a given type T.
        typedef typename type_wrapper<T>::complex_t complex_t;

        Simulation_context const& ctx_;
        
        Unit_cell const& unit_cell_;

        Step_function const& step_function_;

        Communicator const& comm_;

        /// Local part of muffin-tin functions.
        mdarray<Spheric_function<spectral, T>, 1> f_mt_local_;
        
        /// Global muffin-tin array 
        mdarray<T, 3> f_mt_;

        Gvec const& gvec_;

        /// Size of the muffin-tin functions angular domain size.
        int angular_domain_size_;
        
        /// Set pointer to local part of muffin-tin functions
        void set_local_mt_ptr()
        {
            for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++) {
                int ia = unit_cell_.spl_num_atoms(ialoc);
                f_mt_local_(ialoc) = Spheric_function<spectral, T>(&f_mt_(0, 0, ia), angular_domain_size_,
                                                                   unit_cell_.atom(ia).radial_grid());
            }
        }
        
        /* forbid copy constructor */
        Periodic_function(const Periodic_function<T>& src) = delete;
        
        /* forbid assigment operator */
        Periodic_function<T>& operator=(const Periodic_function<T>& src) = delete;

    public:

        /// Constructor
        Periodic_function(Simulation_context& ctx__,
                          int angular_domain_size__)
            : Smooth_periodic_function<T>(ctx__.fft(), ctx__.gvec())
            , ctx_(ctx__)
            , unit_cell_(ctx__.unit_cell())
            , step_function_(ctx__.step_function())
            , comm_(ctx__.comm())
            , gvec_(ctx__.gvec())
            , angular_domain_size_(angular_domain_size__)
        {
            if (ctx_.full_potential()) {
                f_mt_local_ = mdarray<Spheric_function<spectral, T>, 1>(unit_cell_.spl_num_atoms().local_size());
            }
        }
        
        /// Allocate memory for muffin-tin part.
        void allocate_mt(bool allocate_global__)
        {
            if (ctx_.full_potential()) {
                if (allocate_global__) {
                    f_mt_ = mdarray<T, 3>(angular_domain_size_, unit_cell_.max_num_mt_points(), unit_cell_.num_atoms(), memory_t::host, "f_mt_");
                    set_local_mt_ptr();
                } else {
                    for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++) {
                        int ia = unit_cell_.spl_num_atoms(ialoc);
                        f_mt_local_(ialoc) = Spheric_function<spectral, T>(angular_domain_size_, unit_cell_.atom(ia).radial_grid());
                    }
                }
            }
        }

        /// Syncronize global muffin-tin array.
        void sync_mt()
        {
            PROFILE("sirius::Periodic_function::sync_mt");
            assert(f_mt_.size() != 0); 

            int ld = angular_domain_size_ * unit_cell_.max_num_mt_points(); 
            comm_.allgather(&f_mt_(0, 0, 0),
                            ld * unit_cell_.spl_num_atoms().global_offset(),
                            ld * unit_cell_.spl_num_atoms().local_size());
        }

        /// Zero the function.
        void zero()
        {
            f_mt_.zero();
            this->f_rg_.zero();
            this->f_pw_local_.zero();
            if (ctx_.full_potential()) {
                for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++) {
                    f_mt_local_(ialoc).zero();
                }
            }
        }
        
        inline void copy_to_global_ptr(T* f_mt__, T* f_it__) const
        {
            std::memcpy(f_it__, this->f_rg_.template at<CPU>(), this->fft_->local_size() * sizeof(T));

            if (ctx_.full_potential()) {
                mdarray<T, 3> f_mt(f_mt__, angular_domain_size_, unit_cell_.max_num_mt_points(), unit_cell_.num_atoms());
                for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++) {
                    int ia = unit_cell_.spl_num_atoms(ialoc);
                    std::memcpy(&f_mt(0, 0, ia), &f_mt_local_(ialoc)(0, 0), f_mt_local_(ialoc).size() * sizeof(T));
                }
                int ld = angular_domain_size_ * unit_cell_.max_num_mt_points();
                comm_.allgather(f_mt__,
                                ld * unit_cell_.spl_num_atoms().global_offset(),
                                ld * unit_cell_.spl_num_atoms().local_size());
            }
        }

        using Smooth_periodic_function<T>::add;

        /// Add the function
        void add(Periodic_function<T>* g)
        {
            PROFILE("sirius::Periodic_function::add");
            /* add regular-grid part */
            Smooth_periodic_function<T>::add(*g);
            /* add muffin-tin part */
            if (ctx_.full_potential()) {
                for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++)
                    f_mt_local_(ialoc) += g->f_mt(ialoc);
            }
        }

        T integrate(std::vector<T>& mt_val, T& it_val)
        {
            PROFILE("sirius::Periodic_function::integrate");

            it_val = 0;
            
            if (!ctx_.full_potential()) {
                #pragma omp parallel
                {
                    T it_val_t = 0;
                    
                    #pragma omp for schedule(static)
                    for (int irloc = 0; irloc < this->fft_->local_size(); irloc++) {
                        it_val_t += this->f_rg_(irloc);
                    }

                    #pragma omp critical
                    it_val += it_val_t;
                }
            } else {
                for (int irloc = 0; irloc < this->fft_->local_size(); irloc++) {
                    it_val += this->f_rg_(irloc) * step_function_.theta_r(irloc);
                }
            }
            it_val *= (unit_cell_.omega() / this->fft_->size());
            this->fft_->comm().allreduce(&it_val, 1);
            T total = it_val;
            
            if (ctx_.full_potential()) {
                mt_val = std::vector<T>(unit_cell_.num_atoms(), 0);

                for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++) {
                    int ia = unit_cell_.spl_num_atoms(ialoc);
                    mt_val[ia] = f_mt_local_(ialoc).component(0).integrate(2) * fourpi * y00;
                }
                
                comm_.allreduce(&mt_val[0], unit_cell_.num_atoms());
                for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                    total += mt_val[ia];
                }
            }

            return total;
        }

        template <index_domain_t index_domain>
        inline T& f_mt(int idx0, int ir, int ia)
        {
            switch (index_domain) {
                case index_domain_t::local: {
                    return f_mt_local_(ia)(idx0, ir);
                }
                case index_domain_t::global: {
                    return f_mt_(idx0, ir, ia);
                }
            }
        }

        template <index_domain_t index_domain>
        inline T const& f_mt(int idx0, int ir, int ia) const
        {
            switch (index_domain) {
                case index_domain_t::local: {
                    return f_mt_local_(ia)(idx0, ir);
                }
                case index_domain_t::global: {
                    return f_mt_(idx0, ir, ia);
                }
            }
        }
        
        /** \todo write and read distributed functions */
        void hdf5_write(std::string storage_file_name__, std::string path__)
        {
            //STOP();
            //if (ctx_.full_potential()) {
            //    h5f.write("f_mt", f_mt_);
            //}
            auto v = this->gather_f_pw();
            if (ctx_.comm().rank() == 0) {
                HDF5_tree fout(storage_file_name, false);
                fout[path__].write("f_pw", reinterpret_cast<double*>(v.data()), static_cast<int>(v.size() * 2));
            }
        }

        void hdf5_read(HDF5_tree h5f__, mdarray<int, 2>& gvec__)
        {
            //if (ctx_.full_potential()) {
            //    h5f.read("f_mt", f_mt_);
            //}
            std::vector<double_complex> v(gvec_.num_gvec());
            h5f__.read("f_pw", reinterpret_cast<double*>(v.data()), static_cast<int>(v.size() * 2));

            std::map<vector3d<int>, int> local_gvec_mapping;

            for (int igloc = 0; igloc < gvec_.count(); igloc++) {
                int ig = gvec_.offset() + igloc;
                auto G = gvec_.gvec(ig);
                local_gvec_mapping[G] = igloc;
            }

            for (int ig = 0; ig < gvec_.num_gvec(); ig++) {
                vector3d<int> G(&gvec__(0, ig));
                if (local_gvec_mapping.count(G) != 0) {
                    this->f_pw_local_[local_gvec_mapping[G]] = v[ig];
                }
            }
        }

        /// Set the global pointer to the muffin-tin part
        void set_mt_ptr(T* mt_ptr__)
        {
            f_mt_ = mdarray<T, 3>(mt_ptr__, angular_domain_size_, unit_cell_.max_num_mt_points(), unit_cell_.num_atoms(), "f_mt_");
            set_local_mt_ptr();
        }

        /// Set the pointer to the interstitial part
        void set_rg_ptr(T* rg_ptr__)
        {
            this->f_rg_ = mdarray<T, 1>(rg_ptr__, this->fft_->local_size());
        }

        inline Spheric_function<spectral, T> const& f_mt(int ialoc__) const
        {
            return f_mt_local_(ialoc__);
        }

        double value(vector3d<double>& vc)
        {
            int ja{-1}, jr{-1};
            double dr{0}, tp[2];
        
            if (unit_cell_.is_point_in_mt(vc, ja, jr, dr, tp)) {
                int lmax = Utils::lmax_by_lmmax(angular_domain_size_);
                std::vector<double> rlm(angular_domain_size_);
                SHT::spherical_harmonics(lmax, tp[0], tp[1], &rlm[0]);
                double p{0};
                for (int lm = 0; lm < angular_domain_size_; lm++) {
                    double d = (f_mt_(lm, jr + 1, ja) - f_mt_(lm, jr, ja)) / unit_cell_.atom(ja).type().radial_grid().dx(jr);
        
                    p += rlm[lm] * (f_mt_(lm, jr, ja) + d * dr);
                }
                return p;
            } else {
                STOP();
                double p{0};
                //for (int ig = 0; ig < gvec_.num_gvec(); ig++) {
                //    vector3d<double> vgc = gvec_.gvec_cart(ig);
                //    p += std::real(f_pw_(ig) * std::exp(double_complex(0.0, vc * vgc)));
                //}
                return p;
            }
        }

        mdarray<T, 3>& f_mt()
        {
            return f_mt_;
        }

        /// Compute inner product <f|g>
        T inner(Periodic_function<T> const* g__) const
        {
            PROFILE("sirius::Periodic_function::inner");
        
            assert(this->fft_ == g__->fft_);
            assert(&step_function_ == &g__->step_function_);
            assert(&unit_cell_ == &g__->unit_cell_);
            assert(&comm_ == &g__->comm_);
            
            T result_rg{0};
            
            if (!ctx_.full_potential()) {
                Smooth_periodic_function<T> const& tmp = *g__;
                result_rg = Smooth_periodic_function<T>::inner(tmp);
                //#pragma omp parallel
                //{
                //    T rt{0};
                //    
                //    #pragma omp for schedule(static)
                //    for (int irloc = 0; irloc < this->fft_->local_size(); irloc++) {
                //        rt += std::conj(this->f_rg(irloc)) * g__->f_rg(irloc);
                //    }        
                //    #pragma omp critical
                //    result_rg += rt;
                //}
            } else {
                for (int irloc = 0; irloc < this->fft_->local_size(); irloc++) {
                    result_rg += type_wrapper<T>::bypass(std::conj(this->f_rg(irloc))) * g__->f_rg(irloc) * 
                        this->step_function_.theta_r(irloc);
                }
                result_rg *= (unit_cell_.omega() / this->fft_->size());
                this->fft_->comm().allreduce(&result_rg, 1);
            }


            T result_mt{0};
            if (ctx_.full_potential()) {
                for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++) {
                    auto r = sirius::inner(f_mt(ialoc), g__->f_mt(ialoc));
                    result_mt += r;
                }
                comm_.allreduce(&result_mt, 1);
            }
        
            return result_mt + result_rg;
        }
};

}

#endif // __PERIODIC_FUNCTION_H__

