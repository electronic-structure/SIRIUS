#include "reciprocal_lattice.h"

namespace sirius {

template <function_domain_t domain_t, typename T = double_complex>
class Smooth_periodic_function
{
    private:
        
        mdarray<T, 1> data_;

        Reciprocal_lattice* reciprocal_lattice_;

    public:

        Smooth_periodic_function() : reciprocal_lattice_(nullptr)
        {
        }
        
        Smooth_periodic_function(T* ptr__, Reciprocal_lattice* reciprocal_lattice__) : reciprocal_lattice_(reciprocal_lattice__)
        {
            if (domain_t == spatial) data_.set_dimensions(reciprocal_lattice_->fft()->size());
            data_.set_ptr(ptr__);
        }

        Smooth_periodic_function(Reciprocal_lattice* reciprocal_lattice__) : reciprocal_lattice_(reciprocal_lattice__)
        {
            switch (domain_t)
            {
                case spectral:
                {
                    data_.set_dimensions(reciprocal_lattice_->num_gvec());
                    break;
                }
                case spatial:
                {
                    data_.set_dimensions(reciprocal_lattice_->fft()->size());
                    break;
                }
            }
            data_.allocate();
        }

        Smooth_periodic_function(Reciprocal_lattice* reciprocal_lattice__, size_t size) : reciprocal_lattice_(reciprocal_lattice__)
        {
            if (domain_t == spatial) data_.set_dimensions(size);
            data_.allocate();
        }

        inline T& operator()(const int idx)
        {
            return data_(idx);
        }

        Reciprocal_lattice* reciprocal_lattice()
        {
            return reciprocal_lattice_;
        }

        inline size_t size()
        {
            return data_.size(0);
        }

        inline void zero()
        {
            data_.zero();
        }
};

/// Transform funciton from real-space grid to plane-wave harmonics. 
template<typename T>
Smooth_periodic_function<spectral> transform(Smooth_periodic_function<spatial, T>& f)
{
    auto rl = f.reciprocal_lattice();

    Smooth_periodic_function<spectral> g(rl);
        
    rl->fft()->input(&f(0));
    rl->fft()->transform(-1);
    rl->fft()->output(rl->num_gvec(), rl->fft_index(), &g(0));

    return g;
}

/// Transform function from plane-wace domain to real-space grid.
template<typename T, index_domain_t index_domain>
Smooth_periodic_function<spatial, T> transform(Smooth_periodic_function<spectral>& f)
{
    Reciprocal_lattice* rl = f.reciprocal_lattice();

    int size = (index_domain == global) ? rl->fft()->size() : rl->fft()->local_size();
    int offset = (index_domain == global) ? 0 : rl->fft()->global_offset();

    Smooth_periodic_function<spatial, T> g(rl, size);

    rl->fft()->input(rl->num_gvec(), rl->fft_index(), &f(0));
    rl->fft()->transform(1);
    for (int i = 0; i < size; i++) g(i) = type_wrapper<T>::sift(rl->fft()->buffer(offset + i));
    
    return g; 
}

Smooth_periodic_function<spectral> laplacian(Smooth_periodic_function<spectral>& f)
{
    Reciprocal_lattice* rl = f.reciprocal_lattice();

    Smooth_periodic_function<spectral> g(rl);

    for (int ig = 0; ig < rl->num_gvec(); ig++)
    {
        auto G = rl->gvec_cart(ig);
        g(ig) = f(ig) * double_complex(-pow(G.length(), 2), 0);
    }
    return g;
}

template<function_domain_t domaint_t, typename T = double_complex>
class Smooth_periodic_function_gradient
{
    private:

        std::array<Smooth_periodic_function<domaint_t, T>, 3> grad_;

        Reciprocal_lattice* reciprocal_lattice_;

    public:

        Smooth_periodic_function_gradient() : reciprocal_lattice_(nullptr)
        {
        }

        Smooth_periodic_function_gradient(Reciprocal_lattice* reciprocal_lattice__) : reciprocal_lattice_(reciprocal_lattice__)
        {
        }

        Smooth_periodic_function<domaint_t, T>& operator[](const int idx)
        {
            return grad_[idx];
        }

        inline Reciprocal_lattice* reciprocal_lattice()
        {
            return reciprocal_lattice_;
        }
};
        
Smooth_periodic_function_gradient<spectral> gradient(Smooth_periodic_function<spectral>& f)
{
    Reciprocal_lattice* rl = f.reciprocal_lattice();

    Smooth_periodic_function_gradient<spectral> g(rl);

    for (int x = 0; x < 3; x++) g[x] = Smooth_periodic_function<spectral>(rl);

    for (int ig = 0; ig < rl->num_gvec(); ig++)
    {
        auto G = rl->gvec_cart(ig);
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

    Smooth_periodic_function<spatial, T> result(f.reciprocal_lattice(), size);
    result.zero();

    for (int x = 0; x < 3; x++)
    {
        for (int ir = 0; ir < size; ir++)
        {
            result(ir) += f[x](ir) * g[x](ir);
        }
    }

    return result;
}

}
