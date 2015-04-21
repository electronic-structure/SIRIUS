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

//template <typename T> 
//template <typename U>
//T Spline<T>::integrate(Spline<T>* f, Spline<U>* g, int m, int num_points)
//{
//    STOP();
//
//    //if (f->radial_grid_.num_points() != g->radial_grid_.num_points()) 
//    //    error_local(__FILE__, __LINE__, "radial grids don't match");
// 
//    T result = 0;
//
//    //switch (m)
//    //{
//    //    case 0:
//    //    {
//    //        for (int i = 0; i < num_points - 1; i++)
//    //        {
//    //            double dx = f->radial_grid_.dx(i);
//    //            
//    //            T faga = f->a[i] * g->a[i];
//    //            T fdgd = f->d[i] * g->d[i];
//
//    //            T k1 = f->a[i] * g->b[i] + f->b[i] * g->a[i];
//    //            T k2 = f->c[i] * g->a[i] + f->b[i] * g->b[i] + f->a[i] * g->c[i];
//    //            T k3 = f->a[i] * g->d[i] + f->b[i] * g->c[i] + f->c[i] * g->b[i] + f->d[i] * g->a[i];
//    //            T k4 = f->b[i] * g->d[i] + f->c[i] * g->c[i] + f->d[i] * g->b[i];
//    //            T k5 = f->c[i] * g->d[i] + f->d[i] * g->c[i];
//
//    //            result += dx * (faga + 
//    //                      dx * (k1 / 2.0 + 
//    //                      dx * (k2 / 3.0 + 
//    //                      dx * (k3 / 4.0 + 
//    //                      dx * (k4 / 5.0 + 
//    //                      dx * (k5 / 6.0 + 
//    //                      dx * fdgd / 7.0))))));
//    //        }
//    //        break;
//
//    //    }
//    //    case 1:
//    //    {
//    //        for (int i = 0; i < num_points - 1; i++)
//    //        {
//    //            double x0 = f->radial_grid_[i];
//    //            double dx = f->radial_grid_.dx(i);
//    //            
//    //            T faga = f->a[i] * g->a[i];
//    //            T fdgd = f->d[i] * g->d[i];
//
//    //            T k1 = f->a[i] * g->b[i] + f->b[i] * g->a[i];
//    //            T k2 = f->c[i] * g->a[i] + f->b[i] * g->b[i] + f->a[i] * g->c[i];
//    //            T k3 = f->a[i] * g->d[i] + f->b[i] * g->c[i] + f->c[i] * g->b[i] + f->d[i] * g->a[i];
//    //            T k4 = f->b[i] * g->d[i] + f->c[i] * g->c[i] + f->d[i] * g->b[i];
//    //            T k5 = f->c[i] * g->d[i] + f->d[i] * g->c[i];
//
//    //            result += dx * ((faga * x0) + 
//    //                      dx * ((faga + k1 * x0) / 2.0 + 
//    //                      dx * ((k1 + k2 * x0) / 3.0 + 
//    //                      dx * ((k2 + k3 * x0) / 4.0 + 
//    //                      dx * ((k3 + k4 * x0) / 5.0 + 
//    //                      dx * ((k4 + k5 * x0) / 6.0 + 
//    //                      dx * ((k5 + fdgd * x0) / 7.0 +
//    //                      dx * fdgd / 8.0)))))));
//
//    //        }
//    //        break;
//    //    }
//    //    case 2:
//    //    {
//    //        for (int i = 0; i < num_points - 1; i++)
//    //        {
//    //            double x0 = f->radial_grid_[i];
//    //            double dx = f->radial_grid_.dx(i);
//    //            
//    //            T a0b0 = f->a[i] * g->a[i];
//    //            T a3b3 = f->d[i] * g->d[i];
//    //            
//    //            T k1 = f->d[i] * g->b[i] + f->c[i] * g->c[i] + f->b[i] * g->d[i];
//    //            T k2 = f->d[i] * g->a[i] + f->c[i] * g->b[i] + f->b[i] * g->c[i] + f->a[i] * g->d[i];
//    //            T k3 = f->c[i] * g->a[i] + f->b[i] * g->b[i] + f->a[i] * g->c[i];
//    //            T k4 = f->d[i] * g->c[i] + f->c[i] * g->d[i];
//    //            T k5 = f->b[i] * g->a[i] + f->a[i] * g->b[i];
//
//    //            result += dx * ((a0b0 * x0 * x0) + 
//    //                      dx * ((x0 * (2.0 * a0b0 + x0 * k5)) / 2.0 +
//    //                      dx * ((a0b0 + x0 * (2.0 * k5 + k3 * x0)) / 3.0 + 
//    //                      dx * ((k5 + x0 * (2.0 * k3 + k2 * x0)) / 4.0 +
//    //                      dx * ((k3 + x0 * (2.0 * k2 + k1 * x0)) / 5.0 + 
//    //                      dx * ((k2 + x0 * (2.0 * k1 + k4 * x0)) / 6.0 + 
//    //                      dx * ((k1 + x0 * (2.0 * k4 + a3b3 * x0)) / 7.0 + 
//    //                      dx * ((k4 + 2.0 * a3b3 * x0) / 8.0 + 
//    //                      dx * a3b3 / 9.0)))))))); 
//    //        }
//    //        break;
//    //    }
//    //    default:
//    //    {
//    //        error_local(__FILE__, __LINE__, "wrong m for r^m prefactor"); 
//    //    }
//    //}
//
//    return result;
//}
//
//template <typename T> 
//template <typename U>
//T Spline<T>::integrate(Spline<T>* f, Spline<U>* g, int m)
//{
//    if (f->num_points() != g->num_points()) error_local(__FILE__, __LINE__, "number of points doesn't match");
//
//    return Spline<T>::integrate(f, g, m, f->num_points()); 
//}

template <typename T>
T Spline<T>::integrate(std::vector<T>& g__, int m__) const
{
    g__ = std::vector<T>(num_points());

    g__[0] = 0.0;

    switch (m__)
    {
        case 0:
        {
            double t = 1.0 / 3.0;
            for (int i = 0; i < num_points() - 1; i++)
            {
                double dx = radial_grid_->dx(i);
                g__[i + 1] = g__[i] + (((coefs_(i, 3) * dx * 0.25 + coefs_(i, 2) * t) * dx + coefs_(i, 1) * 0.5) * dx + coefs_(i, 0)) * dx;
            }
            break;
        }
        case 2:
        {
            for (int i = 0; i < num_points() - 1; i++)
            {
                double x0 = (*radial_grid_)[i];
                double x1 = (*radial_grid_)[i + 1];
                double dx = radial_grid_->dx(i);
                T a0 = coefs_(i, 0);
                T a1 = coefs_(i, 1);
                T a2 = coefs_(i, 2);
                T a3 = coefs_(i, 3);

                double x0_2 = x0 * x0;
                double x0_3 = x0_2 * x0;
                double x1_2 = x1 * x1;
                double x1_3 = x1_2 * x1;

                g__[i + 1] = g__[i] + (20.0 * a0 * (x1_3 - x0_3) + 5.0 * a1 * (x0 * x0_3 + x1_3 * (3.0 * dx - x0)) - 
                             dx * dx * dx * (-2.0 * a2 * (x0_2 + 3.0 * x0 * x1 + 6.0 * x1_2) - 
                             a3 * dx * (x0_2 + 4.0 * x0 * x1 + 10.0 * x1_2))) / 60.0;
            }
            break;
        }
        case -1:
        {
            for (int i = 0; i < num_points() - 1; i++)
            {
                double x0 = (*radial_grid_)[i];
                double x1 = (*radial_grid_)[i + 1];
                T a0 = coefs_(i, 0);
                T a1 = coefs_(i, 1);
                T a2 = coefs_(i, 2);
                T a3 = coefs_(i, 3);

                // obtained with the following Mathematica code:
                //   FullSimplify[Integrate[x^(-1)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}],
                //                          Assumptions->{Element[{x0,x1},Reals],x1>x0>0}]
                g__[i + 1] = g__[i] + (-((x0 - x1) * (6.0 * a1 - 9.0 * a2 * x0 + 11.0 * a3 * std::pow(x0, 2) + 
                             3.0 * a2 * x1 - 7.0 * a3 * x0 * x1 + 2.0 * a3 * std::pow(x1, 2))) / 6.0 + 
                             (-a0 + x0 * (a1 - a2 * x0 + a3 * std::pow(x0, 2))) * std::log(x0 / x1));
            }
            break;
        }
        case -2:
        {
            for (int i = 0; i < num_points() - 1; i++)
            {
                double x0 = (*radial_grid_)[i];
                double x1 = (*radial_grid_)[i + 1];
                T a0 = coefs_(i, 0);
                T a1 = coefs_(i, 1);
                T a2 = coefs_(i, 2);
                T a3 = coefs_(i, 3);

                // obtained with the following Mathematica code:
                //   FullSimplify[Integrate[x^(-2)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}],
                //                          Assumptions->{Element[{x0,x1},Reals],x1>x0>0}]
                g__[i + 1] = g__[i] + (((x0 - x1) * (-2.0 * a0 + x0 * (2.0 * a1 - 2.0 * a2 * (x0 + x1) + 
                             a3 * (2.0 * std::pow(x0, 2) + 5.0 * x0 * x1 - std::pow(x1, 2)))) + 
                             2.0 * x0 * (a1 + x0 * (-2.0 * a2 + 3.0 * a3 * x0)) * x1 * std::log(x1 / x0)) / 
                             (2.0 * x0 * x1));
            }
            break;
        }
        case -3:
        {
            for (int i = 0; i < num_points() - 1; i++)
            {
                double x0 = (*radial_grid_)[i];
                double x1 = (*radial_grid_)[i + 1];
                T a0 = coefs_(i, 0);
                T a1 = coefs_(i, 1);
                T a2 = coefs_(i, 2);
                T a3 = coefs_(i, 3);

                // obtained with the following Mathematica code:
                //   FullSimplify[Integrate[x^(-3)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}],
                //                          Assumptions->{Element[{x0,x1},Reals],x1>x0>0}]
                g__[i + 1] = g__[i] + (-((x0 - x1) * (a0 * (x0 + x1) + x0 * (a1 * (-x0 + x1) + 
                             x0 * (a2 * x0 - a3 * std::pow(x0, 2) - 3.0 * a2 * x1 + 5.0 * a3 * x0 * x1 + 
                             2.0 * a3 * std::pow(x1, 2)))) + 2.0 * std::pow(x0, 2) * (a2 - 3.0 * a3 * x0) * std::pow(x1, 2) * 
                             std::log(x0 / x1)) / (2.0 * std::pow(x0, 2) * std::pow(x1, 2)));
            }
            break;
        }
        case -4:
        {
            for (int i = 0; i < num_points() - 1; i++)
            {
                double x0 = (*radial_grid_)[i];
                double x1 = (*radial_grid_)[i + 1];
                T a0 = coefs_(i, 0);
                T a1 = coefs_(i, 1);
                T a2 = coefs_(i, 2);
                T a3 = coefs_(i, 3);

                // obtained with the following Mathematica code:
                //   FullSimplify[Integrate[x^(-4)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}],
                //                          Assumptions->{Element[{x0,x1},Reals],x1>x0>0}]
                g__[i + 1] = g__[i] + ((2.0 * a0 * (-std::pow(x0, 3) + std::pow(x1, 3)) + 
                             x0 * (x0 - x1) * (a1 * (x0 - x1) * (2.0 * x0 + x1) + 
                             x0 * (-2.0 * a2 * std::pow(x0 - x1, 2) + a3 * x0 * (2.0 * std::pow(x0, 2) - 7.0 * x0 * x1 + 
                             11.0 * std::pow(x1, 2)))) + 6.0 * a3 * std::pow(x0 * x1, 3) * std::log(x1 / x0)) / 
                             (6.0 * std::pow(x0 * x1, 3)));
            }
            break;
        }
        default:
        {
            for (int i = 0; i < num_points() - 1; i++)
            {
                double x0 = (*radial_grid_)[i];
                double x1 = (*radial_grid_)[i + 1];
                T a0 = coefs_(i, 0);
                T a1 = coefs_(i, 1);
                T a2 = coefs_(i, 2);
                T a3 = coefs_(i, 3);

                // obtained with the following Mathematica code:
                //   FullSimplify[Integrate[x^(m)*(a0+a1*(x-x0)+a2*(x-x0)^2+a3*(x-x0)^3),{x,x0,x1}], 
                //                          Assumptions->{Element[{x0,x1},Reals],x1>x0>0}]
                g__[i + 1] = g__[i] + (std::pow(x0, 1 + m__) * (-(a0 * double((2 + m__) * (3 + m__) * (4 + m__))) + 
                             x0 * (a1 * double((3 + m__) * (4 + m__)) - 2.0 * a2 * double(4 + m__) * x0 + 
                             6.0 * a3 * std::pow(x0, 2)))) / double((1 + m__) * (2 + m__) * (3 + m__) * (4 + m__)) + 
                             std::pow(x1, 1 + m__) * ((a0 - x0 * (a1 + x0 * (-a2 + a3 * x0))) / double(1 + m__) + 
                             ((a1 + x0 * (-2.0 * a2 + 3.0 * a3 * x0)) * x1) / double(2 + m__) + 
                             ((a2 - 3.0 * a3 * x0) * std::pow(x1, 2)) / double(3 + m__) + 
                             (a3 * std::pow(x1, 3)) / double(4 + m__));
            }
            break;
        }
    }
    
    return g__[num_points() - 1];
}

