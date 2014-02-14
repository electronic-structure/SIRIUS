#ifndef __SPHERIC_FUNCTION_GRADIENT_H__
#define __SPHERIC_FUNCTION_GRADIENT_H__

namespace sirius
{

template <typename T>
class Spheric_function_gradient
{
    private:

        Spheric_function<T>* grad_[3];

        // forbid copy constructor
        Spheric_function_gradient(const Spheric_function_gradient& src);

        // forbid assigment operator
        Spheric_function_gradient& operator=(const Spheric_function_gradient& src);

        void gradient(Spheric_function<double_complex>& f);
        
        void gradient(Spheric_function<double>& f);

    public:

        Spheric_function_gradient(Spheric_function<T>& f)
        {
            grad_[0] = new Spheric_function<T>(f, false);
            grad_[1] = new Spheric_function<T>(f, false);
            grad_[2] = new Spheric_function<T>(f, false);
            gradient(f);
        }

        Spheric_function<T>& operator[](const int idx)
        {
            return *(grad_[idx]);
        }

        ~Spheric_function_gradient()
        {
            delete grad_[0];
            delete grad_[1];
            delete grad_[2];
        }
};

#include "spheric_function_gradient.hpp"

}

#endif // __SPHERIC_FUNCTION_GRADIENT_H__
