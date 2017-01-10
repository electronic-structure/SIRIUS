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
        Gvec const* gvec_{nullptr};
        
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

        Smooth_periodic_function(FFT3D& fft__, Gvec const& gvec__)
            : fft_(&fft__),
              gvec_(&gvec__)
        {
            f_rg_ = mdarray<T, 1>(fft_->local_size());
            f_pw_local_ = mdarray<double_complex, 1>(gvec_->partition().gvec_count_fft());
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

        inline const double_complex& f_pw_local(int ig__) const
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

        Gvec const& gvec() const
        {
            assert(gvec_ != nullptr);
            return *gvec_;
        }

        void fft_transform(int direction__)
        {
            PROFILE("sirius::Smooth_periodic_function::fft_transform");

            assert(gvec_ != nullptr);

            switch (direction__) {
                case 1: {
                    fft_->transform<1>(gvec_->partition(), &f_pw_local_(0));
                    fft_->output(&f_rg_(0));
                    break;
                }
                case -1: {
                    fft_->input(&f_rg_(0));
                    fft_->transform<-1>(gvec_->partition(), &f_pw_local_(0));
                    break;
                }
                default: {
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
            for (int x: {0, 1, 2}) grad_f_[x] = Smooth_periodic_function<T>(f_->fft(), f_->gvec());
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
    for (int igloc = 0; igloc < f__.gvec().partition().gvec_count_fft(); igloc++)
    {
        int ig = f__.gvec().partition().gvec_offset_fft() + igloc;

        auto G = f__.gvec().gvec_cart(ig);
        for (int x: {0, 1, 2}) g[x].f_pw_local(igloc) = f__.f_pw_local(igloc) * double_complex(0, G[x]);
    }
    return std::move(g);
}

/// Laplacian of the function in the plane-wave domain.
inline Smooth_periodic_function<double> laplacian(Smooth_periodic_function<double>& f__)
{
    Smooth_periodic_function<double> g(f__.fft(), f__.gvec());
    
    #pragma omp parallel for
    for (int igloc = 0; igloc < f__.gvec().partition().gvec_count_fft(); igloc++)
    {
        int ig = f__.gvec().partition().gvec_offset_fft() + igloc;

        auto G = f__.gvec().gvec_cart(ig);
        g.f_pw_local(igloc) = f__.f_pw_local(igloc) * double_complex(-std::pow(G.length(), 2), 0);
    }

    return std::move(g);
}

template <typename T>
Smooth_periodic_function<T> operator*(Smooth_periodic_function_gradient<T>& grad_f__, 
                                      Smooth_periodic_function_gradient<T>& grad_g__)

{
    assert(&grad_f__.f().fft() == &grad_g__.f().fft());
    assert(&grad_f__.f().gvec() == &grad_g__.f().gvec());
    
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

/// Experimental features.
namespace experimental {

/// Representation of a smooth (Fourier-transformable) periodic function.
/** The class is designed to handle periodic functions such as density or potential, defined on a regular-space grid.
 *  The following functionality is expected:
 *    - access to real-space values
 *    - access to plane-wave coefficients
 *    - distribution of plane-wave coefficients over entire communicator
 *    - Fourier transformation using FFT communicator
 *    - gather PW coefficients into global array
 *  In some cases the PW coefficients are not necessary and only the real-space values are stored.
 */
template <typename T>
class Smooth_periodic_function
{
    protected:

        /// FFT driver.
        FFT3D* fft_{nullptr};

        /// Distribution of G-vectors.
        Gvec const* gvec_{nullptr};

        Communicator const* comm_{nullptr};

        std::unique_ptr<Communicator> comm_local_{nullptr};
        
        /// Function on the regular real-space grid.
        mdarray<T, 1> f_rg_;
        
        /// Local set of plane-wave expansion coefficients.
        mdarray<double_complex, 1> f_pw_local_;

        /// Storage of the PW coefficients for the FFT transformation.
        mdarray<double_complex, 1> f_pw_fft_;

        /// Distribution of G-vectors inside FFT slab.
        block_data_descriptor gvec_fft_slab_;
        
        /* copy constructor is not allowed */
        Smooth_periodic_function(Smooth_periodic_function<T> const& src__) = delete;
        /* assigment is not allowed */
        Smooth_periodic_function<T>& operator=(Smooth_periodic_function<T> const& src__) = delete;

    public:
        
        /// Default constructor.
        Smooth_periodic_function() 
        {
        }

        Smooth_periodic_function(FFT3D& fft__, Gvec const& gvec__, Communicator const& comm__)
            : fft_(&fft__)
            , gvec_(&gvec__)
            , comm_(&comm__)
        {
            f_rg_ = mdarray<T, 1>(fft_->local_size());
            f_pw_fft_ = mdarray<double_complex, 1>(gvec_->partition().gvec_count_fft());
            f_pw_local_ = mdarray<double_complex, 1>(gvec_->gvec_count(comm_->rank()));

            /* find orthogonal communicator to the FFT communicator */
            int n = comm_->size() / fft_->comm().size();

            comm_local_ = std::unique_ptr<Communicator>(new Communicator());

            MPI_Comm_split(comm_->mpi_comm(), fft_->comm().rank(), comm_->rank() % n, &comm_local_->mpi_comm());

            assert(comm_local_->size() == n);

            gvec_fft_slab_ = block_data_descriptor(comm_local_->size());
            for (int i = 0; i < comm_local_->size(); i++) {
                gvec_fft_slab_.counts[i] = gvec_->gvec_count(fft_->comm().rank() * comm_local_->size() + i);
            }
            gvec_fft_slab_.calc_offsets();
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

        inline double_complex& f_pw_fft(int ig__)
        {
            return f_pw_fft_(ig__);
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
            assert(gvec_ != nullptr);
            return *gvec_;
        }

        void fft_transform(int direction__)
        {
            PROFILE("sirius::Smooth_periodic_function::fft_transform");

            assert(gvec_ != nullptr);

            switch (direction__) {
                case 1: {
                    int rank = fft_->comm().rank() * comm_local_->size() + comm_local_->rank();
                    /* collect scattered PW coefficients */
                    comm_local_->allgather(f_pw_local_.at<CPU>(),
                                           gvec_->gvec_count(rank),
                                           f_pw_fft_.at<CPU>(),
                                           gvec_fft_slab_.counts.data(), 
                                           gvec_fft_slab_.offsets.data());

                    fft_->transform<1>(gvec_->partition(), f_pw_fft_.at<CPU>());
                    fft_->output(f_rg_.template at<CPU>());
                    break;
                }
                case -1: {
                    fft_->input(f_rg_.template at<CPU>());
                    fft_->transform<-1>(gvec_->partition(), f_pw_fft_.at<CPU>());
                    int count = gvec_fft_slab_.counts[comm_local_->rank()];
                    int offset = gvec_fft_slab_.offsets[comm_local_->rank()];
                    std::memcpy(f_pw_local_.at<CPU>(), f_pw_fft_.at<CPU>(offset), count * sizeof(T));
                    break;
                }
                default: {
                    TERMINATE("wrong fft direction");
                }
            }
        }
};

} // namespace experimental

} // namespace sirius

#endif // __SMOOTH_PERIODIC_FUNCTION_H__
