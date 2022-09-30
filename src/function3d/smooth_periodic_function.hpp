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

/** \file smooth_periodic_function.hpp
 *
 *  \brief Contains declaration and implementation of sirius::Smooth_periodic_function and
 *         sirius::Smooth_periodic_function_gradient classes.
 */

#include "SDDK/fft.hpp"
#include "SDDK/gvec.hpp"
#include "utils/utils.hpp"
#include "utils/profiler.hpp"
#include "SDDK/type_definition.hpp"
#include "SDDK/fft.hpp"

#ifndef __SMOOTH_PERIODIC_FUNCTION_HPP__
#define __SMOOTH_PERIODIC_FUNCTION_HPP__

namespace sirius {

/// Representation of a smooth (Fourier-transformable) periodic function.
/** The class is designed to handle periodic functions such as density or potential, defined on a regular FFT grid.
 *  The following functionality is provided:
 *    - access to real-space values
 *    - access to plane-wave coefficients
 *    - distribution of plane-wave coefficients over entire communicator
 *    - Fourier transformation using FFT communicator
 *    - gather PW coefficients into global array
 */
template <typename T>
class Smooth_periodic_function
{
  protected:
    /// FFT driver.
    spfft_transform_type<T>* spfft_{nullptr};

    /// Distribution of G-vectors.
    std::shared_ptr<sddk::Gvec_fft> gvecp_{nullptr};

    /// Function on the regular real-space grid.
    sddk::mdarray<T, 1> f_rg_;

    /// Local set of plane-wave expansion coefficients.
    sddk::mdarray<complex_type<T>, 1> f_pw_local_;

    /// Storage of the PW coefficients for the FFT transformation.
    sddk::mdarray<complex_type<T>, 1> f_pw_fft_;

    /// Gather plane-wave coefficients for the subsequent FFT call.
    inline void gather_f_pw_fft()
    {
        gvecp_->gather_pw_fft(f_pw_local_.at(sddk::memory_t::host), f_pw_fft_.at(sddk::memory_t::host));
    }

    Smooth_periodic_function(Smooth_periodic_function<T> const& src__) = delete;
    Smooth_periodic_function<T>& operator=(Smooth_periodic_function<T> const& src__) = delete;

  public:
    /// Default constructor.
    Smooth_periodic_function()
    {
    }

    /// Constructor.
    Smooth_periodic_function(spfft_transform_type<T>& spfft__, std::shared_ptr<sddk::Gvec_fft> gvecp__, sddk::memory_pool* mp__ = nullptr)
        : spfft_(&spfft__)
        , gvecp_(gvecp__)
    {
        if (mp__) {
            f_rg_ = sddk::mdarray<T, 1>(spfft_->local_slice_size(), *mp__, "Smooth_periodic_function.f_rg_");
        } else {
            f_rg_ = sddk::mdarray<T, 1>(spfft_->local_slice_size(), sddk::memory_t::host,
                                        "Smooth_periodic_function.f_rg_");
        }
        f_rg_.zero();

        if (mp__) {
            f_pw_local_ = sddk::mdarray<complex_type<T>, 1>(gvecp_->gvec().count(), *mp__,
                                                           "Smooth_periodic_function.f_pw_local_");
        } else {
            f_pw_local_ = sddk::mdarray<complex_type<T>, 1>(gvecp_->gvec().count(), sddk::memory_t::host,
                                                       "Smooth_periodic_function.f_pw_local_");
        }
        f_pw_local_.zero();
        if (gvecp_->comm_ortho_fft().size() != 1) {
            if (mp__) {
                f_pw_fft_ = sddk::mdarray<complex_type<T>, 1>(gvecp_->gvec_count_fft(), *mp__,
                                                             "Smooth_periodic_function.f_pw_fft_");
            } else {
                f_pw_fft_ = sddk::mdarray<complex_type<T>, 1>(gvecp_->gvec_count_fft(), sddk::memory_t::host,
                                                             "Smooth_periodic_function.f_pw_fft_");
            }
            f_pw_fft_.zero();
        } else {
            /* alias to f_pw_local array */
            f_pw_fft_ = sddk::mdarray<complex_type<T>, 1>(&f_pw_local_[0], gvecp_->gvec().count());
        }
    }
    Smooth_periodic_function(Smooth_periodic_function<T>&& src__) = default;
    Smooth_periodic_function<T>& operator=(Smooth_periodic_function<T>&& src__) = default;

    /// Zero the values on the regular real-space grid.
    inline void zero()
    {
        f_rg_.zero();
    }

    inline T& f_rg(int ir__)
    {
        return const_cast<T&>(static_cast<Smooth_periodic_function<T> const&>(*this).f_rg(ir__));
    }

    inline T const& f_rg(int ir__) const
    {
        return f_rg_(ir__);
    }

    inline sddk::mdarray<T, 1>& f_rg()
    {
        return f_rg_;
    }

    inline sddk::mdarray<T, 1> const& f_rg() const
    {
        return f_rg_;
    }

    inline complex_type<T>& f_pw_local(int ig__)
    {
        return f_pw_local_(ig__);
    }

    inline complex_type<T> const& f_pw_local(int ig__) const
    {
        return f_pw_local_(ig__);
    }

    inline sddk::mdarray<complex_type<T>, 1>& f_pw_local()
    {
      return f_pw_local_;
    }

    inline const sddk::mdarray<complex_type<T>, 1>& f_pw_local() const
    {
      return f_pw_local_;
    }

    inline complex_type<T>& f_pw_fft(int ig__)
    {
        return f_pw_fft_(ig__);
    }

    /// Return plane-wave coefficient for G=0 component.
    inline complex_type<T> f_0() const
    {
        complex_type<T> z;
        if (gvecp_->gvec().comm().rank() == 0) {
            z = f_pw_local_(0);
        }
        gvecp_->gvec().comm().bcast(&z, 1, 0);
        return z;
    }

    spfft_transform_type<T>& spfft()
    {
        assert(spfft_ != nullptr);
        return *spfft_;
    }

    spfft_transform_type<T> const& spfft() const
    {
        assert(spfft_ != nullptr);
        return *spfft_;
    }

    sddk::Gvec const& gvec() const
    {
        assert(gvecp_ != nullptr);
        return gvecp_->gvec();
    }

    auto gvec_fft() const
    {
        return gvecp_;
    }

    void fft_transform(int direction__)
    {
        PROFILE("sirius::Smooth_periodic_function::fft_transform");

        assert(gvecp_ != nullptr);

        auto frg_ptr = (spfft_->local_slice_size() == 0) ? nullptr : &f_rg_[0];

#if defined(USE_FP32)
        using precision_type = typename std::conditional<std::is_same<real_type<T>, double>::value, double, float>::type;
#else
        using precision_type = double;
#endif 
        switch (direction__) {
            case 1: {
                if (gvecp_->comm_ortho_fft().size() != 1) {
                    gather_f_pw_fft();
                }
                spfft_->backward(reinterpret_cast<precision_type const*>(f_pw_fft_.at(sddk::memory_t::host)), SPFFT_PU_HOST);
                spfft_output(*spfft_, frg_ptr);
                break;
            }
            case -1: {
                spfft_input(*spfft_, frg_ptr);
                spfft_->forward(SPFFT_PU_HOST, reinterpret_cast<precision_type*>(f_pw_fft_.at(sddk::memory_t::host)),
                                SPFFT_FULL_SCALING);
                if (gvecp_->comm_ortho_fft().size() != 1) {
                    int count  = gvecp_->gvec_fft_slab().counts[gvecp_->comm_ortho_fft().rank()];
                    int offset = gvecp_->gvec_fft_slab().offsets[gvecp_->comm_ortho_fft().rank()];
                    std::memcpy(f_pw_local_.at(sddk::memory_t::host), f_pw_fft_.at(sddk::memory_t::host, offset),
                                count * sizeof(complex_type<T>));
                }
                break;
            }
            default: {
                throw std::runtime_error("wrong FFT direction");
            }
        }
    }

    inline std::vector<complex_type<T>> gather_f_pw()
    {
        PROFILE("sirius::Smooth_periodic_function::gather_f_pw");

        std::vector<complex_type<T>> fpw(gvecp_->gvec().num_gvec());
        gvec().comm().allgather(&f_pw_local_[0], fpw.data(), gvec().count(), gvec().offset());

        return fpw;
    }

    inline void scatter_f_pw(std::vector<complex_type<T>> const& f_pw__)
    {
        std::copy(&f_pw__[gvecp_->gvec().offset()], &f_pw__[gvecp_->gvec().offset()] + gvecp_->gvec().count(),
                  &f_pw_local_(0));
    }

    void add(Smooth_periodic_function<T> const& g__)
    {
        #pragma omp parallel for schedule(static)
        for (int irloc = 0; irloc < this->spfft_->local_slice_size(); irloc++) {
            this->f_rg_(irloc) += g__.f_rg(irloc);
        }
    }

    inline T checksum_rg() const
    {
        T cs = this->f_rg_.checksum();
        sddk::Communicator(this->spfft_->communicator()).allreduce(&cs, 1);
        return cs;
    }

    inline complex_type<T> checksum_pw() const
    {
        complex_type<T> cs = this->f_pw_local_.checksum();
        this->gvecp_->gvec().comm().allreduce(&cs, 1);
        return cs;
    }

    inline uint64_t hash_f_pw() const
    {
        auto h = f_pw_local_.hash();
        gvecp_->gvec().comm().bcast(&h, 1, 0);

        for (int r = 1; r < gvecp_->gvec().comm().size(); r++) {
            h = f_pw_local_.hash(h);
            gvecp_->gvec().comm().bcast(&h, 1, r);
        }
        return h;
    }

    inline uint64_t hash_f_rg() const
    {
        auto comm = sddk::Communicator(spfft_->communicator());

        uint64_t h;
        for (int r = 0; r < comm.size(); r++) {
            if (r == 0) {
                h = f_rg_.hash();
            } else {
                h = f_rg_.hash(h);
            }
            comm.bcast(&h, 1, r);
        }
        return h;
    }
};

/// Vector of the smooth periodic functions.
template <typename T>
class Smooth_periodic_vector_function : public std::array<Smooth_periodic_function<T>, 3>
{
  private:
    /// FFT driver.
    spfft_transform_type<T>* spfft_{nullptr};

    /// Distribution of G-vectors.
    std::shared_ptr<sddk::Gvec_fft> gvecp_{nullptr};

    Smooth_periodic_vector_function(Smooth_periodic_vector_function<T> const& src__) = delete;
    Smooth_periodic_vector_function<T>& operator=(Smooth_periodic_vector_function<T> const& src__) = delete;

  public:
    /// Default constructor does nothing.
    Smooth_periodic_vector_function()
    {
    }

    Smooth_periodic_vector_function(spfft_transform_type<T>& spfft__, std::shared_ptr<sddk::Gvec_fft> gvecp__)
        : spfft_(&spfft__)
        , gvecp_(gvecp__)
    {
        for (int x : {0, 1, 2}) {
            (*this)[x] = Smooth_periodic_function<T>(spfft__, gvecp__);
        }
    }
    Smooth_periodic_vector_function(Smooth_periodic_vector_function<T>&& src__) = default;
    Smooth_periodic_vector_function<T>& operator=(Smooth_periodic_vector_function<T>&& src__) = default;

    spfft::Transform& spfft() const
    {
        assert(spfft_ != nullptr);
        return *spfft_;
    }

    auto gvec_fft() const
    {
        assert(gvecp_ != nullptr);
        return gvecp_;
    }
};


/// Gradient of the function in the plane-wave domain.
/** Input functions is expected in the plane wave domain, output function is also in the plane-wave domain */
template <typename T>
inline Smooth_periodic_vector_function<T> gradient(Smooth_periodic_function<T>& f__)
{
    PROFILE("sirius::gradient");

    Smooth_periodic_vector_function<T> g(f__.spfft(), f__.gvec_fft());

    #pragma omp parallel for schedule(static)
    for (int igloc = 0; igloc < f__.gvec().count(); igloc++) {
        auto G = f__.gvec().template gvec_cart<sddk::index_domain_t::local>(igloc);
        for (int x : {0, 1, 2}) {
            g[x].f_pw_local(igloc) = f__.f_pw_local(igloc) * std::complex<T>(0, G[x]);
        }
    }
    return g;
}

/// Divergence of the vecor function.
/** Input and output functions are in plane-wave domain */
template <typename T>
inline Smooth_periodic_function<T> divergence(Smooth_periodic_vector_function<T>& g__)
{
    PROFILE("sirius::divergence");

    /* resulting scalar function */
    Smooth_periodic_function<T> f(g__.spfft(), g__.gvec_fft());
    f.zero();
    for (int x : {0, 1, 2}) {
        for (int igloc = 0; igloc < f.gvec().count(); igloc++) {
            auto G = f.gvec().template gvec_cart<sddk::index_domain_t::local>(igloc);
            f.f_pw_local(igloc) += g__[x].f_pw_local(igloc) * std::complex<T>(0, G[x]);
        }
    }

    return f;
}

/// Laplacian of the function in the plane-wave domain.
template <typename T>
inline Smooth_periodic_function<T> laplacian(Smooth_periodic_function<T>& f__)
{
    PROFILE("sirius::laplacian");

    Smooth_periodic_function<T> g(f__.spfft(), f__.gvec_fft());

    #pragma omp parallel for schedule(static)
    for (int igloc = 0; igloc < f__.gvec().count(); igloc++) {
        auto G              = f__.gvec().template gvec_cart<sddk::index_domain_t::local>(igloc);
        g.f_pw_local(igloc) = f__.f_pw_local(igloc) * std::complex<T>(-std::pow(G.length(), 2), 0);
    }

    return g;
}

template <typename T>
inline Smooth_periodic_function<T>
dot(Smooth_periodic_vector_function<T>& vf__, Smooth_periodic_vector_function<T>& vg__)

{
    PROFILE("sirius::dot");

    Smooth_periodic_function<T> result(vf__.spfft(), vf__.gvec_fft());

    #pragma omp parallel for schedule(static)
    for (int ir = 0; ir < vf__.spfft().local_slice_size(); ir++) {
        T d{0};
        for (int x : {0, 1, 2}) {
            d += vf__[x].f_rg(ir) * vg__[x].f_rg(ir);
        }
        result.f_rg(ir) = d;
    }

    return result;
}

/// Compute inner product <f|g>
template <typename T, typename F>
inline T
inner_local(Smooth_periodic_function<T> const& f__, Smooth_periodic_function<T> const& g__, F&& theta__)
{
    assert(&f__.spfft() == &g__.spfft());

    T result_rg{0};

    //#pragma omp parallel for schedule(static) reduction(+:result_rg)
    for (int irloc = 0; irloc < f__.spfft().local_slice_size(); irloc++) {
        result_rg += utils::conj(f__.f_rg(irloc)) * g__.f_rg(irloc) * theta__(irloc);
    }

    result_rg *= (f__.gvec().omega() / spfft_grid_size(f__.spfft()));

    return result_rg;
}
template <typename T, typename F>
inline T
inner(Smooth_periodic_function<T> const& f__, Smooth_periodic_function<T> const& g__, F&& theta__)
{
    PROFILE("sirius::inner");

    T result_rg = inner_local(f__, g__, std::forward<F>(theta__));
    sddk::Communicator(f__.spfft().communicator()).allreduce(&result_rg, 1);

    return result_rg;
}

/// Compute inner product <f|g>
template <typename T>
inline T
inner(Smooth_periodic_function<T> const& f__, Smooth_periodic_function<T> const& g__)
{
    return inner(f__, g__, [](int ir){return 1;});
}

template <typename T>
inline T
inner_local(Smooth_periodic_function<T> const& f__, Smooth_periodic_function<T> const& g__)
{
    return inner_local(f__, g__, [](int ir){return 1;});
}

} // namespace sirius

#endif // __SMOOTH_PERIODIC_FUNCTION_HPP__
