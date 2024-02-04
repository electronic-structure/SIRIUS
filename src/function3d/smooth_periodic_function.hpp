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

#include "core/typedefs.hpp"
#include "core/fft/fft.hpp"
#include "core/fft/gvec.hpp"
#include "core/profiler.hpp"

#ifndef __SMOOTH_PERIODIC_FUNCTION_HPP__
#define __SMOOTH_PERIODIC_FUNCTION_HPP__

namespace sirius {

template <typename T>
inline void
check_smooth_periodic_function_ptr(smooth_periodic_function_ptr_t<T> const& ptr__,
                                   fft::spfft_transform_type<T> const& spfft__)
{
    if (spfft__.dim_x() != ptr__.size_x) {
        std::stringstream s;
        s << "x-dimensions don't match" << std::endl
          << "  spfft__.dim_x() : " << spfft__.dim_x() << std::endl
          << "  ptr__.size_x : " << ptr__.size_x;
        RTE_THROW(s);
    }
    if (spfft__.dim_y() != ptr__.size_y) {
        std::stringstream s;
        s << "y-dimensions don't match" << std::endl
          << "  spfft__.dim_y() : " << spfft__.dim_y() << std::endl
          << "  ptr__.size_y : " << ptr__.size_y;
        RTE_THROW(s);
    }
    if (ptr__.offset_z < 0) { /* global FFT buffer */
        if (spfft__.dim_z() != ptr__.size_z) {
            std::stringstream s;
            s << "global z-dimensions don't match" << std::endl
              << "  spfft__.dim_z() : " << spfft__.dim_z() << std::endl
              << "  ptr__.size_z : " << ptr__.size_z;
            RTE_THROW(s);
        }
    } else { /* local FFT buffer */
        if ((spfft__.local_z_length() != ptr__.size_z) || (spfft__.local_z_offset() != ptr__.offset_z)) {
            RTE_THROW("local z-dimensions don't match");
        }
    }
}

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
    fft::spfft_transform_type<T>* spfft_{nullptr};

    /// Distribution of G-vectors.
    std::shared_ptr<fft::Gvec_fft> gvecp_{nullptr};

    /// Function on the regular real-space grid.
    mdarray<T, 1> f_rg_;

    /// Local set of plane-wave expansion coefficients.
    mdarray<std::complex<real_type<T>>, 1> f_pw_local_;

    /// Storage of the PW coefficients for the FFT transformation.
    mdarray<std::complex<real_type<T>>, 1> f_pw_fft_;

    /// Gather plane-wave coefficients for the subsequent FFT call.
    inline void
    gather_f_pw_fft()
    {
        gvecp_->gather_pw_fft(f_pw_local_.at(memory_t::host), f_pw_fft_.at(memory_t::host));
    }

    template <typename F>
    friend void
    copy(Smooth_periodic_function<F> const& src__, Smooth_periodic_function<F>& dest__);

    template <typename F>
    friend void
    scale(F alpha__, Smooth_periodic_function<F>& x__);

    template <typename F>
    friend void
    axpy(F alpha__, Smooth_periodic_function<F> const& x__, Smooth_periodic_function<F>& y__);

    Smooth_periodic_function(Smooth_periodic_function<T> const& src__) = delete;
    Smooth_periodic_function<T>&
    operator=(Smooth_periodic_function<T> const& src__) = delete;

  public:
    /// Default constructor.
    Smooth_periodic_function()
    {
    }

    /// Constructor.
    Smooth_periodic_function(fft::spfft_transform_type<T> const& spfft__, std::shared_ptr<fft::Gvec_fft> gvecp__,
                             smooth_periodic_function_ptr_t<T> const* sptr__ = nullptr)
        : spfft_{const_cast<fft::spfft_transform_type<T>*>(&spfft__)}
        , gvecp_{gvecp__}
    {
        auto& mp = get_memory_pool(memory_t::host);
        /* wrap external pointer */
        if (sptr__) {
            check_smooth_periodic_function_ptr(*sptr__, spfft__);

            if (!sptr__->ptr) {
                RTE_THROW("Input pointer is null");
            }
            /* true if input external buffer points to local part of FFT grid */
            bool is_local_rg = (sptr__->offset_z >= 0);

            int offs = (is_local_rg) ? 0 : spfft__.dim_x() * spfft__.dim_y() * spfft__.local_z_offset();
            /* wrap the pointer */
            f_rg_ = mdarray<T, 1>({fft::spfft_grid_size_local(spfft__)}, &sptr__->ptr[offs],
                                  mdarray_label("Smooth_periodic_function.f_rg_"));

        } else {
            f_rg_ = mdarray<T, 1>({fft::spfft_grid_size_local(spfft__)}, mp,
                                  mdarray_label("Smooth_periodic_function.f_rg_"));
        }
        f_rg_.zero();

        f_pw_local_ = mdarray<std::complex<real_type<T>>, 1>({gvecp_->gvec().count()}, mp,
                                                             mdarray_label("Smooth_periodic_function.f_pw_local_"));
        f_pw_local_.zero();
        if (gvecp_->comm_ortho_fft().size() != 1) {
            f_pw_fft_ = mdarray<std::complex<real_type<T>>, 1>({gvecp_->count()}, mp,
                                                               mdarray_label("Smooth_periodic_function.f_pw_fft_"));
            f_pw_fft_.zero();
        } else {
            /* alias to f_pw_local array */
            f_pw_fft_ = mdarray<std::complex<real_type<T>>, 1>({gvecp_->gvec().count()}, &f_pw_local_[0]);
        }
    }
    Smooth_periodic_function(Smooth_periodic_function<T>&& src__) = default;
    Smooth_periodic_function<T>&
    operator=(Smooth_periodic_function<T>&& src__) = default;

    /// Zero the values on the regular real-space grid and plane-wave coefficients.
    inline void
    zero()
    {
        f_rg_.zero();
        f_pw_local_.zero();
    }

    inline T const&
    value(int ir__) const
    {
        return f_rg_(ir__);
    }

    inline T&
    value(int ir__)
    {
        return const_cast<T&>(static_cast<Smooth_periodic_function<T> const&>(*this).value(ir__));
    }

    inline auto
    values() -> mdarray<T, 1>&
    {
        return f_rg_;
    }

    inline auto
    values() const -> const mdarray<T, 1>&
    {
        return f_rg_;
    }

    inline auto
    f_pw_local(int ig__) -> std::complex<real_type<T>>&
    {
        return f_pw_local_(ig__);
    }

    inline auto
    f_pw_local(int ig__) const -> const std::complex<real_type<T>>&
    {
        return f_pw_local_(ig__);
    }

    inline auto
    f_pw_local() -> mdarray<std::complex<real_type<T>>, 1>&
    {
        return f_pw_local_;
    }

    inline auto
    f_pw_local() const -> const mdarray<std::complex<real_type<T>>, 1>&
    {
        return f_pw_local_;
    }

    inline auto&
    f_pw_fft(int ig__)
    {
        return f_pw_fft_(ig__);
    }

    /// Return plane-wave coefficient for G=0 component.
    inline auto
    f_0() const
    {
        std::complex<real_type<T>> z;
        if (gvecp_->gvec().comm().rank() == 0) {
            z = f_pw_local_(0);
        }
        gvecp_->gvec().comm().bcast(&z, 1, 0);
        return z;
    }

    auto&
    spfft()
    {
        RTE_ASSERT(spfft_ != nullptr);
        return *spfft_;
    }

    auto const&
    spfft() const
    {
        RTE_ASSERT(spfft_ != nullptr);
        return *spfft_;
    }

    auto&
    gvec() const
    {
        RTE_ASSERT(gvecp_ != nullptr);
        return gvecp_->gvec();
    }

    auto
    gvec_fft() const
    {
        return gvecp_;
    }

    void
    fft_transform(int direction__)
    {
        PROFILE("sirius::Smooth_periodic_function::fft_transform");

        RTE_ASSERT(gvecp_ != nullptr);

        auto frg_ptr = (spfft_->local_slice_size() == 0) ? nullptr : &f_rg_[0];

        switch (direction__) {
            case 1: {
                if (gvecp_->comm_ortho_fft().size() != 1) {
                    gather_f_pw_fft();
                }
                spfft_->backward(reinterpret_cast<real_type<T> const*>(f_pw_fft_.at(memory_t::host)), SPFFT_PU_HOST);
                fft::spfft_output(*spfft_, frg_ptr);
                break;
            }
            case -1: {
                fft::spfft_input(*spfft_, frg_ptr);
                spfft_->forward(SPFFT_PU_HOST, reinterpret_cast<real_type<T>*>(f_pw_fft_.at(memory_t::host)),
                                SPFFT_FULL_SCALING);
                if (gvecp_->comm_ortho_fft().size() != 1) {
                    int count  = gvecp_->gvec_slab().counts[gvecp_->comm_ortho_fft().rank()];
                    int offset = gvecp_->gvec_slab().offsets[gvecp_->comm_ortho_fft().rank()];
                    std::memcpy(f_pw_local_.at(memory_t::host), f_pw_fft_.at(memory_t::host, offset),
                                count * sizeof(std::complex<real_type<T>>));
                }
                break;
            }
            default: {
                throw std::runtime_error("wrong FFT direction");
            }
        }
    }

    inline auto
    gather_f_pw() const
    {
        PROFILE("sirius::Smooth_periodic_function::gather_f_pw");

        std::vector<std::complex<real_type<T>>> fpw(gvecp_->gvec().num_gvec());
        gvec().comm().allgather(&f_pw_local_[0], fpw.data(), gvec().count(), gvec().offset());

        return fpw;
    }

    inline void
    scatter_f_pw(std::vector<std::complex<real_type<T>>> const& f_pw__)
    {
        std::copy(&f_pw__[gvecp_->gvec().offset()], &f_pw__[gvecp_->gvec().offset()] + gvecp_->gvec().count(),
                  &f_pw_local_(0));
    }

    Smooth_periodic_function<T>&
    operator+=(Smooth_periodic_function<T> const& rhs__)
    {
        #pragma omp parallel
        {
            #pragma omp for schedule(static) nowait
            for (int irloc = 0; irloc < this->spfft_->local_slice_size(); irloc++) {
                this->f_rg_(irloc) += rhs__.value(irloc);
            }
            #pragma omp for schedule(static) nowait
            for (int igloc = 0; igloc < this->gvecp_->gvec().count(); igloc++) {
                this->f_pw_local_(igloc) += rhs__.f_pw_local(igloc);
            }
        }
        return *this;
    }

    Smooth_periodic_function<T>&
    operator*=(T alpha__)
    {
        #pragma omp parallel
        {
            #pragma omp for schedule(static) nowait
            for (int irloc = 0; irloc < this->spfft_->local_slice_size(); irloc++) {
                this->f_rg_(irloc) *= alpha__;
            }
            #pragma omp for schedule(static) nowait
            for (int igloc = 0; igloc < this->gvecp_->gvec().count(); igloc++) {
                this->f_pw_local_(igloc) *= alpha__;
            }
        }
        return *this;
    }

    inline T
    checksum_rg() const
    {
        T cs = this->f_rg_.checksum();
        mpi::Communicator(this->spfft_->communicator()).allreduce(&cs, 1);
        return cs;
    }

    inline auto
    checksum_pw() const
    {
        auto cs = this->f_pw_local_.checksum();
        this->gvecp_->gvec().comm().allreduce(&cs, 1);
        return cs;
    }

    inline uint64_t
    hash_f_pw() const
    {
        auto h = f_pw_local_.hash();
        gvecp_->gvec().comm().bcast(&h, 1, 0);

        for (int r = 1; r < gvecp_->gvec().comm().size(); r++) {
            h = f_pw_local_.hash(h);
            gvecp_->gvec().comm().bcast(&h, 1, r);
        }
        return h;
    }

    inline uint64_t
    hash_f_rg() const
    {
        auto comm = mpi::Communicator(spfft_->communicator());

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
    fft::spfft_transform_type<T>* spfft_{nullptr};

    /// Distribution of G-vectors.
    std::shared_ptr<fft::Gvec_fft> gvecp_{nullptr};

    Smooth_periodic_vector_function(Smooth_periodic_vector_function<T> const& src__) = delete;
    Smooth_periodic_vector_function<T>&
    operator=(Smooth_periodic_vector_function<T> const& src__) = delete;

  public:
    /// Default constructor does nothing.
    Smooth_periodic_vector_function()
    {
    }

    Smooth_periodic_vector_function(fft::spfft_transform_type<T>& spfft__, std::shared_ptr<fft::Gvec_fft> gvecp__)
        : spfft_(&spfft__)
        , gvecp_(gvecp__)
    {
        for (int x : {0, 1, 2}) {
            (*this)[x] = Smooth_periodic_function<T>(spfft__, gvecp__);
        }
    }
    Smooth_periodic_vector_function(Smooth_periodic_vector_function<T>&& src__) = default;
    Smooth_periodic_vector_function<T>&
    operator=(Smooth_periodic_vector_function<T>&& src__) = default;

    spfft::Transform&
    spfft() const
    {
        RTE_ASSERT(spfft_ != nullptr);
        return *spfft_;
    }

    auto
    gvec_fft() const
    {
        RTE_ASSERT(gvecp_ != nullptr);
        return gvecp_;
    }
};

template <typename T>
inline Smooth_periodic_function<T>
to_rg(Smooth_periodic_function<T>&& f__)
{
    f__.fft_transform(1);
    return std::move(f__);
}

template <typename T>
inline Smooth_periodic_vector_function<T>
to_rg(Smooth_periodic_vector_function<T>&& f__)
{
    for (int x : {0, 1, 2}) {
        f__[x].fft_transform(1);
    }
    return std::move(f__);
}

/// Gradient of the function in the plane-wave domain.
/** Input functions is expected in the plane wave domain, output function is also in the plane-wave domain */
template <typename T>
inline Smooth_periodic_vector_function<T>
gradient(Smooth_periodic_function<T>& f__)
{
    PROFILE("sirius::gradient");

    Smooth_periodic_vector_function<T> g(f__.spfft(), f__.gvec_fft());

    #pragma omp parallel for schedule(static)
    for (int igloc = 0; igloc < f__.gvec().count(); igloc++) {
        auto G = f__.gvec().template gvec_cart<index_domain_t::local>(igloc);
        for (int x : {0, 1, 2}) {
            g[x].f_pw_local(igloc) = f__.f_pw_local(igloc) * std::complex<real_type<T>>(0, G[x]);
        }
    }
    return g;
}

/// Divergence of the vecor function.
/** Input and output functions are in plane-wave domain */
template <typename T>
inline Smooth_periodic_function<T>
divergence(Smooth_periodic_vector_function<T>& g__)
{
    PROFILE("sirius::divergence");

    /* resulting scalar function */
    Smooth_periodic_function<T> f(g__.spfft(), g__.gvec_fft());
    f.zero();
    for (int x : {0, 1, 2}) {
        for (int igloc = 0; igloc < f.gvec().count(); igloc++) {
            auto G = f.gvec().template gvec_cart<index_domain_t::local>(igloc);
            f.f_pw_local(igloc) += g__[x].f_pw_local(igloc) * std::complex<real_type<T>>(0, G[x]);
        }
    }
    return f;
}

/// Laplacian of the function in the plane-wave domain.
template <typename T>
inline Smooth_periodic_function<T>
laplacian(Smooth_periodic_function<T>& f__)
{
    PROFILE("sirius::laplacian");

    Smooth_periodic_function<T> g(f__.spfft(), f__.gvec_fft());

    #pragma omp parallel for schedule(static)
    for (int igloc = 0; igloc < f__.gvec().count(); igloc++) {
        auto G              = f__.gvec().template gvec_cart<index_domain_t::local>(igloc);
        g.f_pw_local(igloc) = f__.f_pw_local(igloc) * std::complex<real_type<T>>(-std::pow(G.length(), 2), 0);
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
            d += vf__[x].value(ir) * vg__[x].value(ir);
        }
        result.value(ir) = d;
    }

    return result;
}

/// Compute local contribution to inner product <f|g>
template <typename T, typename F>
inline T
inner_local(Smooth_periodic_function<T> const& f__, Smooth_periodic_function<T> const& g__, F&& theta__)
{
    RTE_ASSERT(&f__.spfft() == &g__.spfft());

    T result_rg{0};

    //#pragma omp parallel for schedule(static) reduction(+:result_rg)
    for (int irloc = 0; irloc < f__.spfft().local_slice_size(); irloc++) {
        result_rg += conj(f__.value(irloc)) * g__.value(irloc) * theta__(irloc);
    }

    result_rg *= (f__.gvec().omega() / fft::spfft_grid_size(f__.spfft()));

    return result_rg;
}

template <typename T>
inline T
inner_local(Smooth_periodic_function<T> const& f__, Smooth_periodic_function<T> const& g__)
{
    return inner_local(f__, g__, [](int ir) { return 1; });
}

template <typename T, typename F>
inline T
inner(Smooth_periodic_function<T> const& f__, Smooth_periodic_function<T> const& g__, F&& theta__)
{
    PROFILE("sirius::inner::spf");

    T result_rg = inner_local(f__, g__, std::forward<F>(theta__));
    mpi::Communicator(f__.spfft().communicator()).allreduce(&result_rg, 1);

    return result_rg;
}

/// Compute inner product <f|g>
template <typename T>
inline T
inner(Smooth_periodic_function<T> const& f__, Smooth_periodic_function<T> const& g__)
{
    return inner(f__, g__, [](int ir) { return 1; });
}

/// Copy real-space values from the function to external pointer.
template <typename T>
inline void
copy(Smooth_periodic_function<T> const& src__, smooth_periodic_function_ptr_t<T> dest__)
{
    auto& spfft = src__.spfft();
    check_smooth_periodic_function_ptr(dest__, spfft);

    if (!dest__.ptr) {
        RTE_THROW("Output pointer is null");
    }
    /* true if input external buffer points to local part of FFT grid */
    bool is_local_rg = (dest__.offset_z >= 0);

    int offs = (is_local_rg) ? 0 : spfft.dim_x() * spfft.dim_y() * spfft.local_z_offset();

    /* copy local fraction of real-space points to local or global array */
    std::copy(src__.values().at(memory_t::host), src__.values().at(memory_t::host) + spfft.local_slice_size(),
              dest__.ptr + offs);

    /* if output buffer stores the global data array */
    if (!is_local_rg) {
        mpi::Communicator(spfft.communicator()).allgather(dest__.ptr, spfft.local_slice_size(), offs);
    }
}

/// Copy real-space values from the external pointer to function.
template <typename T>
inline void
copy(smooth_periodic_function_ptr_t<T> const src__, Smooth_periodic_function<T>& dest__)
{
    auto& spfft = dest__.spfft();
    check_smooth_periodic_function_ptr(src__, spfft);

    if (!src__.ptr) {
        RTE_THROW("Input pointer is null");
    }
    /* true if input external buffer points to local part of FFT grid */
    bool is_local_rg = (src__.offset_z >= 0);

    int offs = (is_local_rg) ? 0 : spfft.dim_x() * spfft.dim_y() * spfft.local_z_offset();

    /* copy local fraction of real-space points to local or global array */
    std::copy(src__.ptr + offs, src__.ptr + offs + spfft.local_slice_size(), dest__.values().at(memory_t::host));
}

template <typename T>
inline void
copy(Smooth_periodic_function<T> const& src__, Smooth_periodic_function<T>& dest__)
{
    copy(src__.f_rg_, dest__.f_rg_);
    copy(src__.f_pw_local_, dest__.f_pw_local_);
}

template <typename T>
inline void
scale(T alpha__, Smooth_periodic_function<T>& x__)
{
    for (size_t i = 0; i < x__.f_rg_.size(); i++) {
        x__.f_rg_[i] *= alpha__;
    }
    for (size_t i = 0; i < x__.f_pw_local_.size(); i++) {
        x__.f_pw_local_[i] *= alpha__;
    }
}

template <typename T>
inline void
axpy(T alpha__, Smooth_periodic_function<T> const& x__, Smooth_periodic_function<T>& y__)
{
    for (size_t i = 0; i < x__.f_rg_.size(); i++) {
        y__.f_rg_[i] += x__.f_rg_[i] * alpha__;
    }
    for (size_t i = 0; i < x__.f_pw_local_.size(); i++) {
        y__.f_pw_local_[i] += x__.f_pw_local_[i] * alpha__;
    }
}

} // namespace sirius

#endif // __SMOOTH_PERIODIC_FUNCTION_HPP__
