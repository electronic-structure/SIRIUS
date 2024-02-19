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

#include "context/simulation_context.hpp"
#include "spheric_function.hpp"
#include "spheric_function_set.hpp"
#include "smooth_periodic_function.hpp"
#include "core/profiler.hpp"

namespace sirius {

template <typename T>
struct periodic_function_integrate_t
{
    T total{0};
    T rg{0};
    std::vector<T> mt;
};

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
class Periodic_function
{
  private:
    /// Simulation contex.
    Simulation_context const& ctx_;

    /// Alias to unit cell.
    Unit_cell const& unit_cell_;

    mpi::Communicator const& comm_;

    /// Regular space grid component of the periodic function.
    Smooth_periodic_function<T> rg_component_;

    /// Muffin-tin part of the periodic function.
    Spheric_function_set<T, atom_index_t> mt_component_;

    /// Alias to G-vectors.
    fft::Gvec const& gvec_;

    /* forbid copy constructor */
    Periodic_function(const Periodic_function<T>& src) = delete;

    /* forbid assignment operator */
    Periodic_function<T>&
    operator=(const Periodic_function<T>& src) = delete;

  public:
    /// Constructor for real-space FFT grid only (PP-PW case).
    Periodic_function(Simulation_context const& ctx__, smooth_periodic_function_ptr_t<T> const* rg_ptr__ = nullptr)
        : ctx_(ctx__)
        , unit_cell_(ctx__.unit_cell())
        , comm_(ctx__.comm())
        , rg_component_(ctx__.spfft<real_type<T>>(), ctx__.gvec_fft_sptr(), rg_ptr__)
        , gvec_(ctx__.gvec())
    {
    }

    /// Constructor for interstitial and muffin-tin parts (FP-LAPW case).
    Periodic_function(Simulation_context const& ctx__, std::function<lmax_t(int)> lmax__,
                      splindex_block<atom_index_t> const* spl_atoms__   = nullptr,
                      smooth_periodic_function_ptr_t<T> const* rg_ptr__ = nullptr,
                      spheric_function_set_ptr_t<T> const* mt_ptr__     = nullptr)
        : ctx_(ctx__)
        , unit_cell_(ctx__.unit_cell())
        , comm_(ctx__.comm())
        , rg_component_(ctx__.spfft<real_type<T>>(), ctx__.gvec_fft_sptr(), rg_ptr__)
        , mt_component_("MT component of Periodic_function", ctx__.unit_cell(), lmax__, spl_atoms__, mt_ptr__)
        , gvec_(ctx__.gvec())
    {
    }

    /// Zero the function.
    void
    zero()
    {
        mt_component_.zero();
        rg_component_.zero();
    }

    /// Add the function
    Periodic_function<T>&
    operator+=(Periodic_function<T> const& g__)
    {
        PROFILE("sirius::Periodic_function::add");
        /* add regular-grid part */
        this->rg_component_ += g__.rg();
        /* add muffin-tin part */
        if (ctx_.full_potential()) {
            this->mt_component_ += g__.mt();
        }
        return *this;
    }

    Periodic_function<T>&
    operator*=(T alpha__)
    {
        PROFILE("sirius::Periodic_function::scale");
        /* add regular-grid part */
        this->rg_component_ *= alpha__;
        /* add muffin-tin part */
        if (ctx_.full_potential()) {
            for (auto it : unit_cell_.spl_num_atoms()) {
                this->mt_component_[it.i] *= alpha__;
            }
        }
        return *this;
    }

    /// Return total integral, interstitial contribution and muffin-tin contributions.
    inline auto
    integrate() const
    {
        PROFILE("sirius::Periodic_function::integrate");

        periodic_function_integrate_t<T> result;

        if (!ctx_.full_potential()) {
            //#pragma omp parallel for schedule(static) reduction(+:it_val)
            for (int irloc = 0; irloc < this->rg().spfft().local_slice_size(); irloc++) {
                result.rg += this->rg().value(irloc);
            }
        } else {
            //#pragma omp parallel for schedule(static) reduction(+:it_val)
            for (int irloc = 0; irloc < this->rg().spfft().local_slice_size(); irloc++) {
                result.rg += this->rg().value(irloc) * ctx_.theta(irloc);
            }
        }
        result.rg *= (unit_cell_.omega() / fft::spfft_grid_size(this->rg().spfft()));
        mpi::Communicator(this->rg().spfft().communicator()).allreduce(&result.rg, 1);
        result.total = result.rg;

        if (ctx_.full_potential()) {
            result.mt = std::vector<T>(unit_cell_.num_atoms(), 0);

            for (auto it : unit_cell_.spl_num_atoms()) {
                result.mt[it.i] = mt_component_[it.i].component(0).integrate(2) * fourpi * y00;
            }

            comm_.allreduce(result.mt.data(), unit_cell_.num_atoms());
            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                result.total += result.mt[ia];
            }
        }

        return result;
    }

    /** \todo write and read distributed functions */
    void
    hdf5_write(std::string file_name__, std::string path__) const
    {
        auto v = this->rg().gather_f_pw();
        if (ctx_.comm().rank() == 0) {
            HDF5_tree fout(file_name__, hdf5_access_t::read_write);
            fout[path__].write("f_pw", reinterpret_cast<T*>(v.data()), static_cast<int>(v.size() * 2));
            if (ctx_.full_potential()) {
                for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                    fout[path__].write("f_mt_" + std::to_string(ia), mt_component_[ia].at(memory_t::host),
                                       mt_component_[ia].size());
                }
            }
        }
    }

    void
    hdf5_read(std::string file_name__, std::string path__, mdarray<int, 2> const& gvec__)
    {
        HDF5_tree h5f(file_name__, hdf5_access_t::read_only);

        /* read the PW coeffs. */
        std::vector<std::complex<T>> v(gvec_.num_gvec());
        h5f[path__].read("f_pw", reinterpret_cast<T*>(v.data()), static_cast<int>(v.size() * 2));

        mdarray<int, 1> igmap({gvec_.count()});
        for (int ig = 0; ig < gvec_.num_gvec(); ig++) {
            r3::vector<int> G(&gvec__(0, ig));
            /* locl index in a new (current) layout */
            auto igloc = gvec_.index_by_gvec(G) - gvec_.offset();
            /* only one rank will store the G-vector index ig */
            if (igloc >= 0 && igloc < gvec_.count()) {
                igmap[igloc] = ig;
            }
        }
        for (int igloc = 0; igloc < gvec_.count(); igloc++) {
            this->rg().f_pw_local(igloc) = v[igmap[igloc]];
        }

        if (ctx_.full_potential()) {
            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                h5f[path__].read("f_mt_" + std::to_string(ia), mt_component_[ia].at(memory_t::host),
                                 mt_component_[ia].size());
            }
        }
    }

    T
    value_rg(r3::vector<T> const& vc) const
    {
        T p{0};
        for (int igloc = 0; igloc < gvec_.count(); igloc++) {
            auto vgc = gvec_.gvec_cart<index_domain_t::local>(igloc);
            p += std::real(this->rg().f_pw_local(igloc) * std::exp(std::complex<T>(0.0, dot(vc, vgc))));
        }
        gvec_.comm().allreduce(&p, 1);
        return p;
    }

    T
    value(r3::vector<T> const& vc)
    {
        int ja{-1}, jr{-1};
        T dr{0}, tp[2];

        if (unit_cell_.is_point_in_mt(vc, ja, jr, dr, tp)) {
            auto& frlm = mt_component_[ja];
            int lmax   = sf::lmax(frlm.angular_domain_size());
            std::vector<T> rlm(frlm.angular_domain_size());
            sf::spherical_harmonics(lmax, tp[0], tp[1], &rlm[0]);
            T p{0};
            for (int lm = 0; lm < frlm.angular_domain_size(); lm++) {
                T d = (frlm(lm, jr + 1) - frlm(lm, jr)) / unit_cell_.atom(ja).type().radial_grid().dx(jr);

                p += rlm[lm] * (frlm(lm, jr) + d * dr);
            }
            return p;
        } else {
            return value_rg(vc);
        }
    }

    inline auto const&
    ctx() const
    {
        return ctx_;
    }

    /// Return reference to regular space grid component.
    auto&
    rg()
    {
        return rg_component_;
    }

    /// Return const reference to regular space grid component.
    auto const&
    rg() const
    {
        return rg_component_;
    }

    /// Return reference to spherical functions component.
    auto&
    mt()
    {
        return mt_component_;
    }

    /// Return const reference to spherical functions component.
    auto const&
    mt() const
    {
        return mt_component_;
    }
};

template <typename T>
inline T
inner(Periodic_function<T> const& f__, Periodic_function<T> const& g__)
{
    PROFILE("sirius::inner::pf");
    if (f__.ctx().full_potential()) {
        auto result = sirius::inner_local(f__.rg(), g__.rg(), [&](int ir) { return f__.ctx().theta(ir); });
        f__.ctx().comm_fft().allreduce(&result, 1);
        result += inner(f__.mt(), g__.mt());
        return result;
    } else {
        return inner(f__.rg(), g__.rg());
    }
}

/// Copy values of the function to the external location.
template <typename T>
inline void
copy(Periodic_function<T> const& src__, periodic_function_ptr_t<T> dest__)
{
    copy(src__.rg(), dest__.rg);
    if (src__.ctx().full_potential()) {
        copy(src__.mt(), dest__.mt);
    }
}

/// Copy the values of the function from the external location.
template <typename T>
inline void
copy(periodic_function_ptr_t<T> const src__, Periodic_function<T>& dest__)
{
    copy(src__.rg, dest__.rg());
    if (dest__.ctx().full_potential()) {
        copy(src__.mt, dest__.mt());
    }
}

} // namespace sirius

#endif // __PERIODIC_FUNCTION_HPP__
