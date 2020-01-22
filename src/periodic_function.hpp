// Copyright (c) 2013-2019 Anton Kozhevnikov, Thomas Schulthess
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

/** \file periodic_function.hpp
 *
 *  \brief Contains declaration and partial implementation of sirius::Periodic_function class.
 */

#ifndef __PERIODIC_FUNCTION_HPP__
#define __PERIODIC_FUNCTION_HPP__

#include "simulation_context.hpp"
#include "spheric_function.hpp"
#include "smooth_periodic_function.hpp"
#include "utils/profiler.hpp"

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
template <typename T>
class Periodic_function : public Smooth_periodic_function<T>
{
  private:

    Simulation_context const& ctx_;

    Unit_cell const& unit_cell_;

    Communicator const& comm_;

    /// Local part of muffin-tin functions.
    sddk::mdarray<Spheric_function<function_domain_t::spectral, T>, 1> f_mt_local_;

    /// Global muffin-tin array
    sddk::mdarray<T, 3> f_mt_;

    Gvec const& gvec_;

    /// Size of the muffin-tin functions angular domain size.
    int angular_domain_size_;

    /// Set pointer to local part of muffin-tin functions
    void set_local_mt_ptr()
    {
        for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++) {
            int ia             = unit_cell_.spl_num_atoms(ialoc);
            f_mt_local_(ialoc) = Spheric_function<function_domain_t::spectral, T>(&f_mt_(0, 0, ia), angular_domain_size_,
                                                                                  unit_cell_.atom(ia).radial_grid());
        }
    }

    /* forbid copy constructor */
    Periodic_function(const Periodic_function<T>& src) = delete;

    /* forbid assigment operator */
    Periodic_function<T>& operator=(const Periodic_function<T>& src) = delete;

  public:
    /// Constructor
    Periodic_function(Simulation_context& ctx__, int angular_domain_size__)
        : Smooth_periodic_function<T>(ctx__.spfft(), ctx__.gvec_partition())
        , ctx_(ctx__)
        , unit_cell_(ctx__.unit_cell())
        , comm_(ctx__.comm())
        , gvec_(ctx__.gvec())
        , angular_domain_size_(angular_domain_size__)
    {
        if (ctx_.full_potential()) {
            f_mt_local_ = mdarray<Spheric_function<function_domain_t::spectral, T>, 1>(unit_cell_.spl_num_atoms().local_size());
        }
    }

    Periodic_function(Simulation_context& ctx__, int angular_domain_size__, bool allocate_global__)
        : Periodic_function(ctx__, angular_domain_size__)
    {
      this->allocate_mt(allocate_global__);
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
                    int ia             = unit_cell_.spl_num_atoms(ialoc);
                    f_mt_local_(ialoc) = Spheric_function<function_domain_t::spectral, T>(angular_domain_size_, unit_cell_.atom(ia).radial_grid());
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

    /// Copy the values of the function to the external location.
    inline void copy_to(T* f_mt__, T* f_rg__, bool is_local_rg__) const
    {
        int offs = (is_local_rg__) ? 0 : this->spfft_->dim_x() * this->spfft_->dim_y() * this->spfft_->local_z_offset();
        std::copy(this->f_rg_.at(memory_t::host), this->f_rg_.at(memory_t::host) + this->spfft_->local_slice_size(),
                  f_rg__ + offs);
        if (!is_local_rg__) {
            sddk::Communicator(this->spfft_->communicator()).allgather(f_rg__, offs, this->spfft_->local_slice_size());
        }

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
    void add(Periodic_function<T> const& g__)
    {
        PROFILE("sirius::Periodic_function::add");
        /* add regular-grid part */
        Smooth_periodic_function<T>::add(g__);
        /* add muffin-tin part */
        if (ctx_.full_potential()) {
            for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++) {
                f_mt_local_(ialoc) += g__.f_mt(ialoc);
            }
        }
    }

    /// Return total integral, interstitial contribution and muffin-tin contributions.
    inline std::tuple<T, T, std::vector<T>>
    integrate() const
    {
        PROFILE("sirius::Periodic_function::integrate");

        T it_val = 0;

        if (!ctx_.full_potential()) {
            #pragma omp parallel for schedule(static) reduction(+:it_val)
            for (int irloc = 0; irloc < this->spfft_->local_slice_size(); irloc++) {
                it_val += this->f_rg_(irloc);
            }
        } else {
            #pragma omp parallel for schedule(static) reduction(+:it_val)
            for (int irloc = 0; irloc < this->spfft_->local_slice_size(); irloc++) {
                it_val += this->f_rg_(irloc) * ctx_.theta(irloc);
            }
        }
        it_val *= (unit_cell_.omega() / spfft_grid_size(this->spfft()));
        Communicator(this->spfft_->communicator()).allreduce(&it_val, 1);
        T total = it_val;

        std::vector<T> mt_val;
        if (ctx_.full_potential()) {
            mt_val = std::vector<T>(unit_cell_.num_atoms(), 0);

            for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++) {
                int ia     = unit_cell_.spl_num_atoms(ialoc);
                mt_val[ia] = f_mt_local_(ialoc).component(0).integrate(2) * fourpi * y00;
            }

            comm_.allreduce(&mt_val[0], unit_cell_.num_atoms());
            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                total += mt_val[ia];
            }
        }

        return std::make_tuple(total, it_val, mt_val);
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
        auto v = this->gather_f_pw();
        if (ctx_.comm().rank() == 0) {
            HDF5_tree fout(storage_file_name, hdf5_access_t::read_write);
            fout[path__].write("f_pw", reinterpret_cast<double*>(v.data()), static_cast<int>(v.size() * 2));
            if (ctx_.full_potential()) {
                fout[path__].write("f_mt", f_mt_);
            }
        }
    }

    void hdf5_read(HDF5_tree h5f__, mdarray<int, 2>& gvec__)
    {
        std::vector<double_complex> v(gvec_.num_gvec());
        h5f__.read("f_pw", reinterpret_cast<double*>(v.data()), static_cast<int>(v.size() * 2));

        std::map<vector3d<int>, int> local_gvec_mapping;

        for (int igloc = 0; igloc < gvec_.count(); igloc++) {
            int  ig               = gvec_.offset() + igloc;
            auto G                = gvec_.gvec(ig);
            local_gvec_mapping[G] = igloc;
        }

        for (int ig = 0; ig < gvec_.num_gvec(); ig++) {
            vector3d<int> G(&gvec__(0, ig));
            if (local_gvec_mapping.count(G) != 0) {
                this->f_pw_local_[local_gvec_mapping[G]] = v[ig];
            }
        }

        if (ctx_.full_potential()) {
            h5f__.read("f_mt", f_mt_);
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
        this->f_rg_ = mdarray<T, 1>(rg_ptr__, this->spfft_->local_slice_size());
    }

    inline Spheric_function<function_domain_t::spectral, T> const& f_mt(int ialoc__) const
    {
        return f_mt_local_(ialoc__);
    }

    inline Spheric_function<function_domain_t::spectral, T> & f_mt(int ialoc__)
    {
        return f_mt_local_(ialoc__);
    }

    double value_rg(vector3d<double> const& vc)
    {
        double p{0};
        for (int igloc = 0; igloc < gvec_.count(); igloc++) {
            vector3d<double> vgc = gvec_.gvec_cart<index_domain_t::local>(igloc);
            p += std::real(this->f_pw_local_(igloc) * std::exp(double_complex(0.0, dot(vc, vgc))));
        }
        gvec_.comm().allreduce(&p, 1);
        return p;
    }

    double value(vector3d<double> const& vc)
    {
        int    ja{-1}, jr{-1};
        double dr{0}, tp[2];

        if (unit_cell_.is_point_in_mt(vc, ja, jr, dr, tp)) {
            int lmax = utils::lmax(angular_domain_size_);
            std::vector<double> rlm(angular_domain_size_);
            sf::spherical_harmonics(lmax, tp[0], tp[1], &rlm[0]);
            double p{0};
            for (int lm = 0; lm < angular_domain_size_; lm++) {
                double d = (f_mt_(lm, jr + 1, ja) - f_mt_(lm, jr, ja)) / unit_cell_.atom(ja).type().radial_grid().dx(jr);

                p += rlm[lm] * (f_mt_(lm, jr, ja) + d * dr);
            }
            return p;
        } else {
            return value_rg(vc);
        }
    }

    mdarray<T, 3>& f_mt()
    {
        return f_mt_;
    }

    const mdarray<T, 3>& f_mt() const
    {
        return f_mt_;
    }

    inline Simulation_context const& ctx() const
    {
        return ctx_;
    }
};

template <typename T>
inline T inner_local(Periodic_function<T> const& f__, Periodic_function<T> const& g__)
{
    PROFILE("sirius::Periodic_function|inner_local");

    assert(&f__.ctx() == &g__.ctx());

    T result_rg{0};

    if (!f__.ctx().full_potential()) {
        result_rg = sirius::inner_local(static_cast<Smooth_periodic_function<T> const&>(f__),
                                        static_cast<Smooth_periodic_function<T> const&>(g__));
    } else {
        result_rg = sirius::inner_local(static_cast<Smooth_periodic_function<T> const&>(f__),
                                        static_cast<Smooth_periodic_function<T> const&>(g__),
                                        [&](int ir) { return f__.ctx().theta(ir); });
    }

    T result_mt{0};
    if (f__.ctx().full_potential()) {
        for (int ialoc = 0; ialoc < f__.ctx().unit_cell().spl_num_atoms().local_size(); ialoc++) {
            auto r = sirius::inner(f__.f_mt(ialoc), g__.f_mt(ialoc));
            result_mt += r;
        }
    }

    return result_mt + result_rg;
}

template <typename T>
inline T inner(Periodic_function<T> const& f__, Periodic_function<T> const& g__)
{
    PROFILE("sirius::Periodic_function|inner");

    T result = inner_local(f__, g__);
    f__.ctx().comm().allreduce(&result, 1);

    return result;
}

} // namespace sirius

#endif // __PERIODIC_FUNCTION_HPP__
