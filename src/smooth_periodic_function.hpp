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

/** \file smooth_periodic_function.hpp
 *
 *  \brief Contains declaration and implementation of sirius::Smooth_periodic_function and
 *         sirius::Smooth_periodic_function_gradient classes.
 */

#ifndef __SMOOTH_PERIODIC_FUNCTION_HPP__
#define __SMOOTH_PERIODIC_FUNCTION_HPP__

namespace sirius {

/// Representation of a smooth (Fourier-transformable) periodic function.
/** The class is designed to handle periodic functions such as density or potential, defined on a regular FFT grid.
 *  The following functionality is expected:
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
    FFT3D* fft_{nullptr};

    /// Distribution of G-vectors.
    Gvec_partition const* gvecp_{nullptr};

    /// Function on the regular real-space grid.
    mdarray<T, 1> f_rg_;

    /// Local set of plane-wave expansion coefficients.
    mdarray<double_complex, 1> f_pw_local_;

    /// Storage of the PW coefficients for the FFT transformation.
    mdarray<double_complex, 1> f_pw_fft_;

    /// Gather plane-wave coefficients for the subsequent FFT call.
    inline void gather_f_pw_fft()
    {
        gvecp_->gather_pw_fft(f_pw_local_.at(memory_t::host), f_pw_fft_.at(memory_t::host));
    }

  public:
    /// Default constructor.
    Smooth_periodic_function()
    {
    }

    Smooth_periodic_function(FFT3D& fft__, Gvec_partition const& gvecp__)
        : fft_(&fft__)
        , gvecp_(&gvecp__)
    {
        f_rg_ = mdarray<T, 1>(fft_->local_size(), memory_t::host, "Smooth_periodic_function.f_rg_");
        f_rg_.zero();

        f_pw_fft_ = mdarray<double_complex, 1>(gvecp_->gvec_count_fft(), memory_t::host,
                                               "Smooth_periodic_function.f_pw_fft_");
        f_pw_fft_.zero();

        f_pw_local_ = mdarray<double_complex, 1>(gvecp_->gvec().count(), memory_t::host,
                                                 "Smooth_periodic_function.f_pw_local_");
        f_pw_local_.zero();
    }

    inline void zero()
    {
        f_rg_.zero();
    }

    inline mdarray<double_complex, 1>& pw_array()
    {
        return f_pw_local_;
    }

    inline mdarray<T, 1>& rg_array()
    {
        return f_rg_;
    }

    inline T& f_rg(int ir__)
    {
        return const_cast<T&>(static_cast<Smooth_periodic_function<T> const&>(*this).f_rg(ir__));
    }

    inline T const& f_rg(int ir__) const
    {
        return f_rg_(ir__);
    }

    inline mdarray<T, 1>& f_rg()
    {
        return f_rg_;
    }

    inline mdarray<T, 1> const& f_rg() const
    {
        return f_rg_;
    }

    inline double_complex& f_pw_local(int ig__)
    {
        return f_pw_local_(ig__);
    }

    inline double_complex const& f_pw_local(int ig__) const
    {
        return f_pw_local_(ig__);
    }

    inline double_complex& f_pw_fft(int ig__)
    {
        return f_pw_fft_(ig__);
    }

    /// Return plane-wave coefficient for G=0 component.
    inline double_complex f_0() const
    {
        double_complex z;
        if (gvecp_->gvec().comm().rank() == 0) {
            z = f_pw_local_(0);
        }
        gvecp_->gvec().comm().bcast(&z, 1, 0);
        return z;
    }

    FFT3D& fft()
    {
        assert(fft_ != nullptr);
        return *fft_;
    }

    FFT3D const& fft() const
    {
        assert(fft_ != nullptr);
        return *fft_;
    }

    Gvec const& gvec() const
    {
        assert(gvecp_ != nullptr);
        return gvecp_->gvec();
    }

    Gvec_partition const& gvec_partition() const
    {
        return *gvecp_;
    }

    void fft_transform(int direction__)
    {
        PROFILE("sirius::Smooth_periodic_function::fft_transform");

        assert(gvecp_ != nullptr);

        switch (direction__) {
            case 1: {
                gather_f_pw_fft();
                fft_->transform<1>(f_pw_fft_.at(memory_t::host));
                fft_->output(f_rg_.at(memory_t::host));
                break;
            }
            case -1: {
                fft_->input(f_rg_.at(memory_t::host));
                fft_->transform<-1>(f_pw_fft_.at(memory_t::host));
                int count  = gvecp_->gvec_fft_slab().counts[gvecp_->comm_ortho_fft().rank()];
                int offset = gvecp_->gvec_fft_slab().offsets[gvecp_->comm_ortho_fft().rank()];
                std::memcpy(f_pw_local_.at(memory_t::host), f_pw_fft_.at(memory_t::host, offset),
                            count * sizeof(double_complex));
                break;
            }
            default: {
                TERMINATE("wrong fft direction");
            }
        }
    }

    inline std::vector<double_complex> gather_f_pw()
    {
        PROFILE("sirius::Smooth_periodic_function::gather_f_pw");

        std::vector<double_complex> fpw(gvecp_->gvec().num_gvec());
        gvec().comm().allgather(&f_pw_local_[0], fpw.data(), gvec().offset(), gvec().count());

        return std::move(fpw);
    }

    inline void scatter_f_pw(std::vector<double_complex> const& f_pw__)
    {
        std::copy(&f_pw__[gvecp_->gvec().offset()], &f_pw__[gvecp_->gvec().offset()] + gvecp_->gvec().count(),
                  &f_pw_local_(0));
    }

    void add(Smooth_periodic_function<T> const& g__)
    {
        #pragma omp parallel for schedule(static)
        for (int irloc = 0; irloc < this->fft_->local_size(); irloc++) {
            this->f_rg_(irloc) += g__.f_rg(irloc);
        }
    }

    inline T checksum_rg() const
    {
        T cs = this->f_rg_.checksum();
        this->fft_->comm().allreduce(&cs, 1);
        return cs;
    }

    inline double_complex checksum_pw() const
    {
        double_complex cs = this->f_pw_local_.checksum();
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
        auto h = f_rg_.hash();
        fft_->comm().bcast(&h, 1, 0);

        for (int r = 1; r < fft_->comm().size(); r++) {
            h = f_rg_.hash(h);
            fft_->comm().bcast(&h, 1, r);
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
    FFT3D* fft_{nullptr};

    /// Distribution of G-vectors.
    Gvec_partition const* gvecp_{nullptr};

  public:
    Smooth_periodic_vector_function()
    {
    }

    Smooth_periodic_vector_function(FFT3D& fft__, Gvec_partition const& gvecp__)
        : fft_(&fft__)
        , gvecp_(&gvecp__)
    {
        for (int x : {0, 1, 2}) {
            (*this)[x] = Smooth_periodic_function<T>(fft__, gvecp__);
        }
    }

    FFT3D& fft() const
    {
        assert(fft_ != nullptr);
        return *fft_;
    }

    Gvec_partition const& gvec_partition() const
    {
        assert(gvecp_ != nullptr);
        return *gvecp_;
    }
};

/// Gradient of the function in the plane-wave domain.
/** Input functions is expected in the plane wave domain, output function is also in the plane-wave domain */
inline Smooth_periodic_vector_function<double> gradient(Smooth_periodic_function<double>& f__)
{
    utils::timer t1("sirius::gradient");

    Smooth_periodic_vector_function<double> g(f__.fft(), f__.gvec_partition());

    #pragma omp parallel for schedule(static)
    for (int igloc = 0; igloc < f__.gvec().count(); igloc++) {
        auto G = f__.gvec().gvec_cart<index_domain_t::local>(igloc);
        for (int x : {0, 1, 2}) {
            g[x].f_pw_local(igloc) = f__.f_pw_local(igloc) * double_complex(0, G[x]);
        }
    }
    return std::move(g);
}


/// Divergence of the vecor function.
/** Input is expected in the plane-wave domain, output is in real-space domain. */
inline Smooth_periodic_function<double> divergence(Smooth_periodic_vector_function<double>& g__)
{
    utils::timer t1("sirius::divergence");

    Smooth_periodic_function<double> f(g__.fft(), g__.gvec_partition());
    f.zero();
    Smooth_periodic_function<double> g_tmp(g__.fft(), g__.gvec_partition());
    for (int x : {0, 1, 2}) {
        for (int igloc = 0; igloc < f.gvec().count(); igloc++) {
            auto G = f.gvec().gvec_cart<index_domain_t::local>(igloc);
            g_tmp.f_pw_local(igloc) = g__[x].f_pw_local(igloc) * double_complex(0, G[x]);
        }
        g_tmp.fft_transform(1);
        for (int ir = 0; ir < f.fft().local_size(); ir++) {
            f.f_rg(ir) += g_tmp.f_rg(ir);
        }
    }
    return std::move(f);
}

/// Laplacian of the function in the plane-wave domain.
inline Smooth_periodic_function<double> laplacian(Smooth_periodic_function<double>& f__)
{
    utils::timer t1("sirius::laplacian");

    Smooth_periodic_function<double> g(f__.fft(), f__.gvec_partition());

    #pragma omp parallel for schedule(static)
    for (int igloc = 0; igloc < f__.gvec().count(); igloc++) {
        auto G              = f__.gvec().gvec_cart<index_domain_t::local>(igloc);
        g.f_pw_local(igloc) = f__.f_pw_local(igloc) * double_complex(-std::pow(G.length(), 2), 0);
    }

    return std::move(g);
}

template <typename T>
inline Smooth_periodic_function<T> dot(Smooth_periodic_vector_function<T>& vf__,
                                       Smooth_periodic_vector_function<T>& vg__)

{
    utils::timer t1("sirius::dot");

    Smooth_periodic_function<T> result(vf__.fft(), vf__.gvec_partition());

    #pragma omp parallel for schedule(static)
    for (int ir = 0; ir < vf__.fft().local_size(); ir++) {
        double d{0};
        for (int x : {0, 1, 2}) {
            d += vf__[x].f_rg(ir) * vg__[x].f_rg(ir);
        }
        result.f_rg(ir) = d;
    }

    return std::move(result);
}

/// Compute inner product <f|g>
template <typename T>
T inner(Smooth_periodic_function<T> const& f__, Smooth_periodic_function<T> const& g__)
{
    utils::timer t1("sirius::Smooth_periodic_function|inner");

    assert(&f__.fft() == &g__.fft());

    T result_rg{0};

    #pragma omp parallel for schedule(static) reduction(+:result_rg)
    for (int irloc = 0; irloc < f__.fft().local_size(); irloc++) {
        // result_rg += type_wrapper<T>::bypass(std::conj(f__.f_rg(irloc)) * g__.f_rg(irloc));
        result_rg += utils::conj(f__.f_rg(irloc)) * g__.f_rg(irloc);
    }

    result_rg *= (f__.gvec().omega() / f__.fft().size());

    f__.fft().comm().allreduce(&result_rg, 1);

    return result_rg;
}

} // namespace sirius

#endif // __SMOOTH_PERIODIC_FUNCTION_HPP__
