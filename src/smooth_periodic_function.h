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

/** \file smooth_periodic_function.h
 *   
 *  \brief Contains declaration and implementation of sirius::Smooth_periodic_function and 
 *         sirius::Smooth_periodic_function_gradient classes.
 */

#ifndef __SMOOTH_PERIODIC_FUNCTION_H__
#define __SMOOTH_PERIODIC_FUNCTION_H__

namespace sirius {

/// Smooth periodic function on the regular real-space grid or in plane-wave domain.
/** Main purpose of this class is to provide a storage and representation of a smooth (Fourier-transformable)
 *  periodic function.
 */
template <typename T>
class Smooth_periodic_function
{
    protected:

        /// FFT driver.
        FFT3D* fft_{nullptr};

        /// Distribution of G-vectors.
        Gvec_FFT_distribution const* gvec_fft_distr_{nullptr};
        
        /// Function on the regular real-space grid.
        mdarray<T, 1> f_rg_;
        
        /// Local set of plane-wave expansion coefficients.
        mdarray<double_complex, 1> f_pw_local_;

    public:

        Smooth_periodic_function() 
        {
        }

        Smooth_periodic_function(FFT3D& fft__)
            : fft_(&fft__)
        {
            f_rg_ = mdarray<T, 1>(fft_->local_size());
        }

        Smooth_periodic_function(FFT3D& fft__, Gvec_FFT_distribution const& gvec_fft_distr__)
            : fft_(&fft__),
              gvec_fft_distr_(&gvec_fft_distr__)
        {
            f_rg_ = mdarray<T, 1>(fft_->local_size());
            f_pw_local_ = mdarray<double_complex, 1>(gvec_fft_distr_->num_gvec_fft());
        }

        inline T& f_rg(int ir__)
        {
            return f_rg_(ir__);
        }

        inline T const& f_rg(int ir__) const
        {
            return f_rg_(ir__);
        }
        
        inline double_complex& f_pw_local(int ig__)
        {
            return f_pw_local_(ig__);
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

        Gvec_FFT_distribution const& gvec_fft_distr() const
        {
            assert(gvec_fft_distr_ != nullptr);
            return *gvec_fft_distr_;
        }

        void fft_transform(int direction__)
        {
            runtime::Timer t("sirius::Smooth_periodic_function::fft_transform");

            assert(gvec_fft_distr_ != nullptr);

            switch (direction__)
            {
                case 1:
                {
                    fft_->transform<1>(gvec_fft_distr(), &f_pw_local_(0));
                    fft_->output(&f_rg_(0));
                    break;
                }
                case -1:
                {
                    fft_->input(&f_rg_(0));
                    fft_->transform<-1>(gvec_fft_distr(), &f_pw_local_(0));
                    break;
                }
                default:
                {
                    TERMINATE("wrong fft direction");
                }
            }
        }
};

/// Gradient of the smooth periodic function.
template<typename T>
class Smooth_periodic_function_gradient
{
    private:
        
        Smooth_periodic_function<T>* f_;

        std::array<Smooth_periodic_function<T>, 3> grad_f_;

    public:

        Smooth_periodic_function_gradient() : f_(nullptr)
        {
        }

        Smooth_periodic_function_gradient(Smooth_periodic_function<T>& f__) : f_(&f__)
        {
            for (int x: {0, 1, 2}) grad_f_[x] = Smooth_periodic_function<T>(f_->fft(), f_->gvec_fft_distr());
        }

        Smooth_periodic_function<T>& operator[](const int idx__)
        {
            return grad_f_[idx__];
        }

        Smooth_periodic_function<T>& f()
        {
            assert(f_ != nullptr);

            return *f_;
        }
};

/// Gradient of the function in the plane-wave domain.
inline Smooth_periodic_function_gradient<double> gradient(Smooth_periodic_function<double>& f__)
{
    Smooth_periodic_function_gradient<double> g(f__);

    #pragma omp parallel for
    for (int igloc = 0; igloc < f__.gvec_fft_distr().num_gvec_fft(); igloc++)
    {
        int ig = f__.gvec_fft_distr().offset_gvec_fft() + igloc;

        auto G = f__.gvec_fft_distr().gvec().cart(ig);
        for (int x: {0, 1, 2}) g[x].f_pw_local(igloc) = f__.f_pw_local(igloc) * double_complex(0, G[x]);
    }
    return std::move(g);
}

/// Laplacian of the function in the plane-wave domain.
inline Smooth_periodic_function<double> laplacian(Smooth_periodic_function<double>& f__)
{
    Smooth_periodic_function<double> g(f__.fft(), f__.gvec_fft_distr());
    
    #pragma omp parallel for
    for (int igloc = 0; igloc < f__.gvec_fft_distr().num_gvec_fft(); igloc++)
    {
        int ig = f__.gvec_fft_distr().offset_gvec_fft() + igloc;

        auto G = f__.gvec_fft_distr().gvec().cart(ig);
        g.f_pw_local(igloc) = f__.f_pw_local(igloc) * double_complex(-std::pow(G.length(), 2), 0);
    }

    return std::move(g);
}

template <typename T>
Smooth_periodic_function<T> operator*(Smooth_periodic_function_gradient<T>& grad_f__, 
                                      Smooth_periodic_function_gradient<T>& grad_g__)

{
    assert(&grad_f__.f().fft() == &grad_g__.f().fft());
    assert(&grad_f__.f().gvec_fft_distr() == &grad_g__.f().gvec_fft_distr());
    
    Smooth_periodic_function<T> result(grad_f__.f().fft());

    #pragma omp parallel for
    for (int ir = 0; ir < grad_f__.f().fft().local_size(); ir++)
    {
        double d = 0;
        for (int x: {0, 1, 2})
        {
            d += grad_f__[x].f_rg(ir) * grad_g__[x].f_rg(ir);
        }
        result.f_rg(ir) = d;
    }

    return std::move(result);
}

}

#endif // __SMOOTH_PERIODIC_FUNCTION_H__
