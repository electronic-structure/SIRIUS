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

/** \file spline.hpp
 *   
 *  \brief Contains remaining implementaiton of sirius::Spline class.
 */

template <typename T> 
Spline<T>& Spline<T>::interpolate()
{
    int np = num_points();

    std::vector<T> diag_main(np);
    std::vector<T> diag_lower(np - 1);
    std::vector<T> diag_upper(np - 1);
    std::vector<T> m(np);
    std::vector<T> dy(np - 1);
    
    /* derivative of y */
    for (int i = 0; i < np - 1; i++) dy[i] = (a[i + 1] - a[i]) / radial_grid_.dx(i);
    
    /* setup "B" vector of AX=B equation */
    for (int i = 0; i < np - 2; i++) m[i + 1] = (dy[i + 1] - dy[i]) * 6.0;
    
    m[0] = -m[1];
    m[np - 1] = -m[np - 2];
    
    /* main diagonal of "A" matrix */
    for (int i = 0; i < np - 2; i++) diag_main[i + 1] = 2 * (radial_grid_.dx(i) + radial_grid_.dx(i + 1));
    double h0 = radial_grid_.dx(0);
    double h1 = radial_grid_.dx(1);
    double h2 = radial_grid_.dx(np - 2);
    double h3 = radial_grid_.dx(np - 3);
    diag_main[0] = (h1 / h0) * h1 - h0;
    diag_main[np - 1] = (h3 / h2) * h3 - h2;
    
    /* subdiagonals of "A" matrix */
    for (int i = 0; i < np - 1; i++)
    {
        diag_upper[i] = radial_grid_.dx(i);
        diag_lower[i] = radial_grid_.dx(i);
    }
    diag_upper[0] = -(h1 * (1 + h1 / h0) + diag_main[1]);
    diag_lower[np - 2] = -(h3 * (1 + h3 / h2) + diag_main[np - 2]); 

    /* solve tridiagonal system */
    int info = linalg<CPU>::gtsv(np, 1, &diag_lower[0], &diag_main[0], &diag_upper[0], &m[0], np);

    if (info)
    {
        std::stringstream s;
        s << "gtsv returned " << info;
        error_local(__FILE__, __LINE__, s);
    }
    
    b.resize(np - 1);
    c.resize(np - 1);
    d.resize(np - 1);

    for (int i = 0; i < np - 1; i++)
    {
        c[i] = m[i] / 2.0;
        T t = (m[i + 1] - m[i]) / 6.0;
        b[i] = dy[i] - (c[i] + t) * radial_grid_.dx(i);
        d[i] = t / radial_grid_.dx(i);
    }
    return *this;
}

template <typename T> 
template <typename U>
T Spline<T>::integrate(Spline<T>* f, Spline<U>* g, int m, int num_points)
{
    if (f->radial_grid_.num_points() != g->radial_grid_.num_points()) 
        error_local(__FILE__, __LINE__, "radial grids don't match");
 
    T result = 0;

    switch (m)
    {
        case 0:
        {
            for (int i = 0; i < num_points - 1; i++)
            {
                double dx = f->radial_grid_.dx(i);
                
                T faga = f->a[i] * g->a[i];
                T fdgd = f->d[i] * g->d[i];

                T k1 = f->a[i] * g->b[i] + f->b[i] * g->a[i];
                T k2 = f->c[i] * g->a[i] + f->b[i] * g->b[i] + f->a[i] * g->c[i];
                T k3 = f->a[i] * g->d[i] + f->b[i] * g->c[i] + f->c[i] * g->b[i] + f->d[i] * g->a[i];
                T k4 = f->b[i] * g->d[i] + f->c[i] * g->c[i] + f->d[i] * g->b[i];
                T k5 = f->c[i] * g->d[i] + f->d[i] * g->c[i];

                result += dx * (faga + 
                          dx * (k1 / 2.0 + 
                          dx * (k2 / 3.0 + 
                          dx * (k3 / 4.0 + 
                          dx * (k4 / 5.0 + 
                          dx * (k5 / 6.0 + 
                          dx * fdgd / 7.0))))));
            }
            break;

        }
        case 1:
        {
            for (int i = 0; i < num_points - 1; i++)
            {
                double x0 = f->radial_grid_[i];
                double dx = f->radial_grid_.dx(i);
                
                T faga = f->a[i] * g->a[i];
                T fdgd = f->d[i] * g->d[i];

                T k1 = f->a[i] * g->b[i] + f->b[i] * g->a[i];
                T k2 = f->c[i] * g->a[i] + f->b[i] * g->b[i] + f->a[i] * g->c[i];
                T k3 = f->a[i] * g->d[i] + f->b[i] * g->c[i] + f->c[i] * g->b[i] + f->d[i] * g->a[i];
                T k4 = f->b[i] * g->d[i] + f->c[i] * g->c[i] + f->d[i] * g->b[i];
                T k5 = f->c[i] * g->d[i] + f->d[i] * g->c[i];

                result += dx * ((faga * x0) + 
                          dx * ((faga + k1 * x0) / 2.0 + 
                          dx * ((k1 + k2 * x0) / 3.0 + 
                          dx * ((k2 + k3 * x0) / 4.0 + 
                          dx * ((k3 + k4 * x0) / 5.0 + 
                          dx * ((k4 + k5 * x0) / 6.0 + 
                          dx * ((k5 + fdgd * x0) / 7.0 +
                          dx * fdgd / 8.0)))))));

            }
            break;
        }
        case 2:
        {
            for (int i = 0; i < num_points - 1; i++)
            {
                double x0 = f->radial_grid_[i];
                double dx = f->radial_grid_.dx(i);
                
                T a0b0 = f->a[i] * g->a[i];
                T a3b3 = f->d[i] * g->d[i];
                
                T k1 = f->d[i] * g->b[i] + f->c[i] * g->c[i] + f->b[i] * g->d[i];
                T k2 = f->d[i] * g->a[i] + f->c[i] * g->b[i] + f->b[i] * g->c[i] + f->a[i] * g->d[i];
                T k3 = f->c[i] * g->a[i] + f->b[i] * g->b[i] + f->a[i] * g->c[i];
                T k4 = f->d[i] * g->c[i] + f->c[i] * g->d[i];
                T k5 = f->b[i] * g->a[i] + f->a[i] * g->b[i];

                result += dx * ((a0b0 * x0 * x0) + 
                          dx * ((x0 * (2.0 * a0b0 + x0 * k5)) / 2.0 +
                          dx * ((a0b0 + x0 * (2.0 * k5 + k3 * x0)) / 3.0 + 
                          dx * ((k5 + x0 * (2.0 * k3 + k2 * x0)) / 4.0 +
                          dx * ((k3 + x0 * (2.0 * k2 + k1 * x0)) / 5.0 + 
                          dx * ((k2 + x0 * (2.0 * k1 + k4 * x0)) / 6.0 + 
                          dx * ((k1 + x0 * (2.0 * k4 + a3b3 * x0)) / 7.0 + 
                          dx * ((k4 + 2.0 * a3b3 * x0) / 8.0 + 
                          dx * a3b3 / 9.0)))))))); 
            }
            break;
        }
        default:
        {
            error_local(__FILE__, __LINE__, "wrong m for r^m prefactor"); 
        }
    }

    return result;
}

template <typename T> 
template <typename U>
T Spline<T>::integrate(Spline<T>* f, Spline<U>* g, int m)
{
    if (f->num_points() != g->num_points()) error_local(__FILE__, __LINE__, "number of points doesn't match");

    return Spline<T>::integrate(f, g, m, f->num_points()); 
}

template <typename T>
T Spline<T>::integrate(std::vector<T>& g, int m)
{
    g = std::vector<T>(num_points());

    g[0] = 0.0;

    switch (m)
    {
        case 0:
        {
            double t = 1.0 / 3.0;
            for (int i = 0; i < num_points() - 1; i++)
            {
                double dx = radial_grid_.dx(i);
                g[i + 1] = g[i] + (((d[i] * dx * 0.25 + c[i] * t) * dx + b[i] * 0.5) * dx + a[i]) * dx;
            }
            break;
        }
        case 2:
        {
            for (int i = 0; i < num_points() - 1; i++)
            {
                double x0 = radial_grid_[i];
                double x1 = radial_grid_[i + 1];
                double dx = radial_grid_.dx(i);
                T a0 = a[i];
                T a1 = b[i];
                T a2 = c[i];
                T a3 = d[i];

                double x0_2 = x0 * x0;
                double x0_3 = x0_2 * x0;
                double x1_2 = x1 * x1;
                double x1_3 = x1_2 * x1;

                g[i + 1] = g[i] + (20.0 * a0 * (x1_3 - x0_3) + 5.0 * a1 * (x0 * x0_3 + x1_3 * (3.0 * dx - x0)) - 
                           dx * dx * dx * (-2.0 * a2 * (x0_2 + 3.0 * x0 * x1 + 6.0 * x1_2) - 
                           a3 * dx * (x0_2 + 4.0 * x0 * x1 + 10.0 * x1_2))) / 60.0;
            }
            break;
        }
        case -1:
        {
            for (int i = 0; i < num_points() - 1; i++)
            {
                double x0 = radial_grid_[i];
                double x1 = radial_grid_[i + 1];
                T a0 = a[i];
                T a1 = b[i];
                T a2 = c[i];
                T a3 = d[i];

                // obtained with the following Mathematica code:
                //   FullSimplify[Integrate[x^(-1)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}],
                //                          Assumptions->{Element[{x0,x1},Reals],x1>x0>0}]
                g[i + 1] = g[i] + (-((x0 - x1) * (6.0 * a1 - 9.0 * a2 * x0 + 11.0 * a3 * pow(x0, 2) + 
                           3.0 * a2 * x1 - 7.0 * a3 * x0 * x1 + 2.0 * a3 * pow(x1, 2))) / 6.0 + 
                           (-a0 + x0 * (a1 - a2 * x0 + a3 * pow(x0, 2))) * log(x0 / x1));
            }
            break;
        }
        case -2:
        {
            for (int i = 0; i < num_points() - 1; i++)
            {
                double x0 = radial_grid_[i];
                double x1 = radial_grid_[i + 1];
                T a0 = a[i];
                T a1 = b[i];
                T a2 = c[i];
                T a3 = d[i];

                // obtained with the following Mathematica code:
                //   FullSimplify[Integrate[x^(-2)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}],
                //                          Assumptions->{Element[{x0,x1},Reals],x1>x0>0}]
                g[i + 1] = g[i] + (((x0 - x1) * (-2.0 * a0 + x0 * (2.0 * a1 - 2.0 * a2 * (x0 + x1) + 
                           a3 * (2.0 * pow(x0, 2) + 5.0 * x0 * x1 - pow(x1, 2)))) + 
                           2.0 * x0 * (a1 + x0 * (-2.0 * a2 + 3.0 * a3 * x0)) * x1 * log(x1 / x0)) / 
                           (2.0 * x0 * x1));
            }
            break;
        }
        case -3:
        {
            for (int i = 0; i < num_points() - 1; i++)
            {
                double x0 = radial_grid_[i];
                double x1 = radial_grid_[i + 1];
                T a0 = a[i];
                T a1 = b[i];
                T a2 = c[i];
                T a3 = d[i];

                // obtained with the following Mathematica code:
                //   FullSimplify[Integrate[x^(-3)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}],
                //                          Assumptions->{Element[{x0,x1},Reals],x1>x0>0}]
                g[i + 1] = g[i] + (-((x0 - x1) * (a0 * (x0 + x1) + x0 * (a1 * (-x0 + x1) + 
                           x0 * (a2 * x0 - a3 * pow(x0, 2) - 3.0 * a2 * x1 + 5.0 * a3 * x0 * x1 + 
                           2.0 * a3 * pow(x1, 2)))) + 2.0 * pow(x0, 2) * (a2 - 3.0 * a3 * x0) * pow(x1, 2) * 
                           log(x0 / x1)) / (2.0 * pow(x0, 2) * pow(x1, 2)));
            }
            break;
        }
        case -4:
        {
            for (int i = 0; i < num_points() - 1; i++)
            {
                double x0 = radial_grid_[i];
                double x1 = radial_grid_[i + 1];
                T a0 = a[i];
                T a1 = b[i];
                T a2 = c[i];
                T a3 = d[i];

                // obtained with the following Mathematica code:
                //   FullSimplify[Integrate[x^(-4)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}],
                //                          Assumptions->{Element[{x0,x1},Reals],x1>x0>0}]
                g[i + 1] = g[i] + ((2.0 * a0 * (-pow(x0, 3) + pow(x1, 3)) + 
                           x0 * (x0 - x1) * (a1 * (x0 - x1) * (2.0 * x0 + x1) + 
                           x0 * (-2.0 * a2 * pow(x0 - x1, 2) + a3 * x0 * (2.0 * pow(x0, 2) - 7.0 * x0 * x1 + 
                           11.0 * pow(x1, 2)))) + 6.0 * a3 * pow(x0, 3) * pow(x1, 3) * log(x1 / x0)) / 
                           (6.0 * pow(x0, 3) * pow(x1, 3)));
            }
            break;
        }
        default:
        {
            for (int i = 0; i < num_points() - 1; i++)
            {
                double x0 = radial_grid_[i];
                double x1 = radial_grid_[i + 1];
                T a0 = a[i];
                T a1 = b[i];
                T a2 = c[i];
                T a3 = d[i];

                // obtained with the following Mathematica code:
                //   FullSimplify[Integrate[x^(m)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}], 
                //                          Assumptions->{Element[{x0,x1},Reals],x1>x0>0}]
                g[i + 1] = g[i] + (pow(x0, 1 + m) * (-(a0 * double((2 + m) * (3 + m) * (4 + m))) + 
                           x0 * (a1 * double((3 + m) * (4 + m)) - 2.0 * a2 * double(4 + m) * x0 + 
                           6.0 * a3 * pow(x0, 2)))) / double((1 + m) * (2 + m) * (3 + m) * (4 + m)) + 
                           pow(x1, 1 + m) * ((a0 - x0 * (a1 + x0 * (-a2 + a3 * x0))) / double(1 + m) + 
                           ((a1 + x0 * (-2.0 * a2 + 3.0 * a3 * x0)) * x1) / double(2 + m) + 
                           ((a2 - 3.0 * a3 * x0) * pow(x1, 2)) / double(3 + m) + 
                           (a3 * pow(x1, 3)) / double(4 + m));
            }
            break;
        }
    }
    
    return g[num_points() - 1];
}

template <typename T>
void Spline<T>::get_coefs(T* array, int lda)
{
    for (int i = 0; i < num_points() - 1; i++)
    {
        array[0 * lda + i] = a[i];
        array[1 * lda + i] = b[i];
        array[2 * lda + i] = c[i];
        array[3 * lda + i] = d[i];
    }
    array[num_points() - 1] = a[num_points() - 1];
}

