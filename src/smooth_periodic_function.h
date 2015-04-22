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

/** \file smooth_periodic_function.h
 *   
 *  \brief Contains declaration and implementation of sirius::Smooth_periodic_function and 
 *         sirius::Smooth_periodic_function_gradient classes.
 */

#include "reciprocal_lattice.h"

namespace sirius {

/// Smooth periodic function on the real-space mesh or plane-wave domain.
/** Main purpose of this class is to provide a storage and representation of a smooth (Fourier-transformable)
 *  periodic function. The function is stored as a set of values on the regular grid (function_domain_t = spatial) or
 *  a set of plane-wave coefficients (function_domain_t = spectral).
 */
template <function_domain_t domain_t, typename T = double_complex>
class Smooth_periodic_function
{
    private:
        
        mdarray<T, 1> data_;

        FFT3D<CPU>* fft_;

    public:

        Smooth_periodic_function() : fft_(nullptr)
        {
        }
        
        Smooth_periodic_function(T* ptr__, FFT3D<CPU>* fft__) : fft_(fft__)
        {
            if (domain_t == spatial)
            {
                data_ = mdarray<T, 1>(ptr__, fft_->size());
            }
        }

        Smooth_periodic_function(FFT3D<CPU>* fft__) : fft_(fft__)
        {
            switch (domain_t)
            {
                case spectral:
                {
                    data_ = mdarray<T, 1>(fft_->num_gvec());
                    break;
                }
                case spatial:
                {
                    data_ = mdarray<T, 1>(fft_->size());
                    break;
                }
            }
        }

        Smooth_periodic_function(FFT3D<CPU>* fft__, size_t size__) : fft_(fft__)
        {
            if (domain_t == spatial) data_ = mdarray<T, 1>(size__);
        }

        inline T& operator()(const int64_t idx__)
        {
            return data_(idx__);
        }

        inline size_t size()
        {
            return data_.size(0);
        }

        inline void zero()
        {
            data_.zero();
        }

        inline FFT3D<CPU>* fft()
        {
            return fft_;
        }
};

/// Transform funciton from real-space grid to plane-wave harmonics. 
template<typename T>
Smooth_periodic_function<spectral> transform(Smooth_periodic_function<spatial, T>& f)
{
    auto fft = f.fft();

    Smooth_periodic_function<spectral> g(fft);
        
    fft->input(&f(0));
    fft->transform(-1);
    fft->output(fft->num_gvec(), fft->index_map(), &g(0));

    return g;
}

/// Transform function from plane-wace domain to real-space grid.
template<typename T>
Smooth_periodic_function<spatial, T> transform(Smooth_periodic_function<spectral>& f)
{
    auto fft = f.fft();

    Smooth_periodic_function<spatial, T> g(fft);

    fft->input(fft->num_gvec(), fft->index_map(), &f(0));
    fft->transform(1);
    fft->output(&g(0));
    
    return g; 
}

template<typename T>
Smooth_periodic_function<spatial, T> transform(Smooth_periodic_function<spectral>& f, splindex<block>& spl_fft_size__)
{
    auto fft = f.fft();

    Smooth_periodic_function<spatial, T> g(fft, spl_fft_size__.local_size());

    fft->input(fft->num_gvec(), fft->index_map(), &f(0));
    fft->transform(1);
    for (int i = 0; i < (int)spl_fft_size__.local_size(); i++) 
        g(i) = type_wrapper<T>::sift(fft->buffer((int)spl_fft_size__.global_offset() + i));

    return g; 
}

inline Smooth_periodic_function<spectral> laplacian(Smooth_periodic_function<spectral>& f)
{
    auto fft = f.fft();

    Smooth_periodic_function<spectral> g(fft);

    for (int ig = 0; ig < fft->num_gvec(); ig++)
    {
        auto G = fft->gvec_cart(ig);
        g(ig) = f(ig) * double_complex(-std::pow(G.length(), 2), 0);
    }
    return g;
}

/// Gradient of the smooth periodic function.
template<function_domain_t domaint_t, typename T = double_complex>
class Smooth_periodic_function_gradient
{
    private:

        std::array<Smooth_periodic_function<domaint_t, T>, 3> grad_;

        FFT3D<CPU>* fft_;

    public:

        Smooth_periodic_function_gradient() : fft_(nullptr)
        {
        }

        Smooth_periodic_function_gradient(FFT3D<CPU>* fft__) : fft_(fft__)
        {
        }

        Smooth_periodic_function<domaint_t, T>& operator[](const int idx__)
        {
            return grad_[idx__];
        }

        inline FFT3D<CPU>* fft()
        {
            return fft_;
        }
};
        
inline Smooth_periodic_function_gradient<spectral> gradient(Smooth_periodic_function<spectral>& f)
{
    auto fft = f.fft();

    Smooth_periodic_function_gradient<spectral> g(fft);

    for (int x = 0; x < 3; x++) g[x] = Smooth_periodic_function<spectral>(fft);

    for (int ig = 0; ig < fft->num_gvec(); ig++)
    {
        auto G = fft->gvec_cart(ig);
        for (int x = 0; x < 3; x++) g[x](ig) = f(ig) * double_complex(0, G[x]); 
    }
    return g;
}

template <typename T>
Smooth_periodic_function<spatial, T> operator*(Smooth_periodic_function_gradient<spatial, T>& f, 
                                               Smooth_periodic_function_gradient<spatial, T>& g)


{
    size_t size = f[0].size();

    for (int x = 0; x < 3; x++)
    {
        if (f[x].size() != size || g[x].size() != size) error_local(__FILE__, __LINE__, "wrong size");
    }

    Smooth_periodic_function<spatial, T> result(f.fft(), size);
    result.zero();

    for (int x = 0; x < 3; x++)
    {
        for (int ir = 0; ir < (int)size; ir++)
        {
            result(ir) += f[x](ir) * g[x](ir);
        }
    }

    return result;
}

}
