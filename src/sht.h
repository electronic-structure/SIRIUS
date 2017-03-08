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

/** \file sht.h
 *   
 *  \brief Contains declaration and particular implementation of sirius::SHT class.
 */

#ifndef __SHT_H__
#define __SHT_H__

#include <math.h>
#include <stddef.h>
#include <gsl/gsl_sf_coupling.h>
#include <gsl/gsl_sf_legendre.h>
#include <string.h>
#include <vector>
#include "typedefs.h"
#include "utils.h"
#include "linalg.hpp"
#include "LebedevLaikov.h"

namespace sirius
{

/// Spherical harmonics transformation.
class SHT // TODO: better name
{
    private:

        int lmax_;

        int lmmax_;

        int num_points_;

        mdarray<double, 2> coord_;

        mdarray<double, 2> tp_;

        std::vector<double> w_;

        /// backward transformation from Ylm to spherical coordinates
        mdarray<double_complex, 2> ylm_backward_;
        
        /// forward transformation from spherical coordinates to Ylm
        mdarray<double_complex, 2> ylm_forward_;
        
        /// backward transformation from Rlm to spherical coordinates
        mdarray<double, 2> rlm_backward_;

        /// forward transformation from spherical coordinates to Rlm
        mdarray<double, 2> rlm_forward_;

        int mesh_type_;

    public:
        
        /// Default constructor.
        SHT(int lmax__) 
            : lmax_(lmax__), mesh_type_(0)
        {
            lmmax_ = (lmax_ + 1) * (lmax_ + 1);
            
            if (mesh_type_ == 0) num_points_ = Lebedev_Laikov_npoint(2 * lmax_);
            if (mesh_type_ == 1) num_points_ = lmmax_;
            
            std::vector<double> x(num_points_);
            std::vector<double> y(num_points_);
            std::vector<double> z(num_points_);
        
            coord_ = mdarray<double, 2>(3, num_points_);
        
            tp_ = mdarray<double, 2>(2, num_points_);
        
            w_.resize(num_points_);
        
            if (mesh_type_ == 0) Lebedev_Laikov_sphere(num_points_, &x[0], &y[0], &z[0], &w_[0]);
            if (mesh_type_ == 1) uniform_coverage();
        
            ylm_backward_ = mdarray<double_complex, 2>(lmmax_, num_points_);
        
            ylm_forward_ = mdarray<double_complex, 2>(num_points_, lmmax_);
        
            rlm_backward_ = mdarray<double, 2>(lmmax_, num_points_);
        
            rlm_forward_ = mdarray<double, 2>(num_points_, lmmax_);
        
            for (int itp = 0; itp < num_points_; itp++)
            {
                if (mesh_type_ == 0)
                {
                    coord_(0, itp) = x[itp];
                    coord_(1, itp) = y[itp];
                    coord_(2, itp) = z[itp];
                    
                    auto vs = spherical_coordinates(vector3d<double>(x[itp], y[itp], z[itp]));
                    spherical_harmonics(lmax_, vs[1], vs[2], &ylm_backward_(0, itp));
                    spherical_harmonics(lmax_, vs[1], vs[2], &rlm_backward_(0, itp));
                    for (int lm = 0; lm < lmmax_; lm++)
                    {
                        ylm_forward_(itp, lm) = conj(ylm_backward_(lm, itp)) * w_[itp] * fourpi;
                        rlm_forward_(itp, lm) = rlm_backward_(lm, itp) * w_[itp] * fourpi;
                    }
                }
                if (mesh_type_ == 1)
                {
                    double t = tp_(0, itp);
                    double p = tp_(1, itp);
        
                    coord_(0, itp) = sin(t) * cos(p);
                    coord_(1, itp) = sin(t) * sin(p);
                    coord_(2, itp) = cos(t);
        
                    spherical_harmonics(lmax_, t, p, &ylm_backward_(0, itp));
                    spherical_harmonics(lmax_, t, p, &rlm_backward_(0, itp));
        
                    for (int lm = 0; lm < lmmax_; lm++)
                    {
                        ylm_forward_(lm, itp) = ylm_backward_(lm, itp);
                        rlm_forward_(lm, itp) = rlm_backward_(lm, itp);
                    }
                }
            }
        
            if (mesh_type_ == 1)
            {
                linalg<CPU>::geinv(lmmax_, ylm_forward_);
                linalg<CPU>::geinv(lmmax_, rlm_forward_);
            }
            
            #if (__VERIFICATION > 0)
            {
                double dr = 0;
                double dy = 0;
        
                for (int lm = 0; lm < lmmax_; lm++)
                {
                    for (int lm1 = 0; lm1 < lmmax_; lm1++)
                    {
                        double t = 0;
                        double_complex zt(0, 0);
                        for (int itp = 0; itp < num_points_; itp++)
                        {
                            zt += ylm_forward_(itp, lm) * ylm_backward_(lm1, itp);
                            t += rlm_forward_(itp, lm) * rlm_backward_(lm1, itp);
                        }
                        
                        if (lm == lm1) 
                        {
                            zt -= 1.0;
                            t -= 1.0;
                        }
                        dr += std::abs(t);
                        dy += std::abs(zt);
                    }
                }
                dr = dr / lmmax_ / lmmax_;
                dy = dy / lmmax_ / lmmax_;
        
                if (dr > 1e-15 || dy > 1e-15)
                {
                    std::stringstream s;
                    s << "spherical mesh error is too big" << std::endl
                      << "  real spherical integration error " << dr << std::endl
                      << "  complex spherical integration error " << dy;
                    WARNING(s.str())
                }
        
                std::vector<double> flm(lmmax_);
                std::vector<double> ftp(num_points_);
                for (int lm = 0; lm < lmmax_; lm++)
                {
                    std::memset(&flm[0], 0, lmmax_ * sizeof(double));
                    flm[lm] = 1.0;
                    backward_transform(lmmax_, &flm[0], 1, lmmax_, &ftp[0]);
                    forward_transform(&ftp[0], 1, lmmax_, lmmax_, &flm[0]);
                    flm[lm] -= 1.0;
        
                    double t = 0.0;
                    for (int lm1 = 0; lm1 < lmmax_; lm1++) t += std::abs(flm[lm1]);
        
                    t /= lmmax_;
        
                    if (t > 1e-15) 
                    {
                        std::stringstream s;
                        s << "test of backward / forward real SHT failed" << std::endl
                          << "  total error " << t;
                        WARNING(s.str());
                    }
                }
            }
            #endif
        }

        /// Perform a backward transformation from spherical harmonics to spherical coordinates.
        /** \f[
         *      f(\theta, \phi, r) = \sum_{\ell m} f_{\ell m}(r) Y_{\ell m}(\theta, \phi)
         *  \f]
         *
         *  \param [in] ld Size of leading dimension of flm.
         *  \param [in] flm Raw pointer to \f$ f_{\ell m}(r) \f$.
         *  \param [in] nr Number of radial points.
         *  \param [in] lmmax Maximum number of lm- harmonics to take into sum.
         *  \param [out] ftp Raw pointer to \f$ f(\theta, \phi, r) \f$.
         */
        template <typename T>
        void backward_transform(int ld, T const* flm, int nr, int lmmax, T* ftp);
        
        /// Perform a forward transformation from spherical coordinates to spherical harmonics.
        /** \f[
         *      f_{\ell m}(r) = \iint  f(\theta, \phi, r) Y_{\ell m}^{*}(\theta, \phi) \sin \theta d\phi d\theta = 
         *        \sum_{i} f(\theta_i, \phi_i, r) Y_{\ell m}^{*}(\theta_i, \phi_i) w_i
         *  \f]
         *
         *  \param [in] ftp Raw pointer to \f$ f(\theta, \phi, r) \f$.
         *  \param [in] nr Number of radial points.
         *  \param [in] lmmax Maximum number of lm- coefficients to generate.
         *  \param [in] ld Size of leading dimension of flm.
         *  \param [out] flm Raw pointer to \f$ f_{\ell m}(r) \f$.
         */
        template <typename T>
        void forward_transform(T const* ftp, int nr, int lmmax, int ld, T* flm);
        
        /// Convert form Rlm to Ylm representation.
        static void convert(int lmax__, double const* f_rlm__, double_complex* f_ylm__);

        /// Convert from Ylm to Rlm representation.
        static void convert(int lmax__, double_complex const* f_ylm__, double* f_rlm__);

        //void rlm_forward_iterative_transform(double *ftp__, int lmmax, int ncol, double* flm)
        //{
        //    Timer t("sirius::SHT::rlm_forward_iterative_transform");
        //    
        //    assert(lmmax <= lmmax_);

        //    mdarray<double, 2> ftp(ftp__, num_points_, ncol);
        //    mdarray<double, 2> ftp1(num_points_, ncol);
        //    
        //    blas<cpu>::gemm(1, 0, lmmax, ncol, num_points_, 1.0, &rlm_forward_(0, 0), num_points_, &ftp(0, 0), num_points_, 0.0, 
        //                    flm, lmmax);
        //    
        //    for (int i = 0; i < 2; i++)
        //    {
        //        rlm_backward_transform(flm, lmmax, ncol, &ftp1(0, 0));
        //        double tdiff = 0.0;
        //        for (int ir = 0; ir < ncol; ir++)
        //        {
        //            for (int itp = 0; itp < num_points_; itp++) 
        //            {
        //                ftp1(itp, ir) = ftp(itp, ir) - ftp1(itp, ir);
        //                //tdiff += fabs(ftp1(itp, ir));
        //            }
        //        }
        //        
        //        for (int itp = 0; itp < num_points_; itp++) 
        //        {
        //            tdiff += fabs(ftp1(itp, ncol - 1));
        //        }
        //        std::cout << "iter : " << i << " avg. MT diff = " << tdiff / num_points_ << std::endl;
        //        blas<cpu>::gemm(1, 0, lmmax, ncol, num_points_, 1.0, &rlm_forward_(0, 0), num_points_, &ftp1(0, 0), num_points_, 1.0, 
        //                        flm, lmmax);
        //    }
        //}

        /// Transform Cartesian coordinates [x,y,z] to spherical coordinates [r,theta,phi]
        static vector3d<double> spherical_coordinates(vector3d<double> vc);

        /// Generate complex spherical harmonics Ylm
        static void spherical_harmonics(int lmax, double theta, double phi, double_complex* ylm);
        
        /// Generate real spherical harmonics Rlm
        /** Mathematica code:
         *  \verbatim
         *  R[l_, m_, th_, ph_] := 
         *   If[m > 0, std::sqrt[2]*ComplexExpand[Re[SphericalHarmonicY[l, m, th, ph]]], 
         *   If[m < 0, std::sqrt[2]*ComplexExpand[Im[SphericalHarmonicY[l, m, th, ph]]], 
         *   If[m == 0, ComplexExpand[Re[SphericalHarmonicY[l, 0, th, ph]]]]]]
         *  \endverbatim
         */
        static void spherical_harmonics(int lmax, double theta, double phi, double* rlm);
                        
        /// Compute element of the transformation matrix from complex to real spherical harmonics. 
        /** Real spherical harmonic can be written as a linear combination of complex harmonics:
         *
         *    \f[
         *        R_{\ell m}(\theta, \phi) = \sum_{m'} a^{\ell}_{m' m}Y_{\ell m'}(\theta, \phi)
         *    \f]
         *    where 
         *    \f[
         *        a^{\ell}_{m' m} = \langle Y_{\ell m'} | R_{\ell m} \rangle
         *    \f]
         *
         *    Transformation from real to complex spherical harmonics is conjugate transpose:
         *    
         *    \f[
         *        Y_{\ell m}(\theta, \phi) = \sum_{m'} a^{\ell*}_{m m'}R_{\ell m'}(\theta, \phi)
         *    \f]
         *
         *    Mathematica code:
         *    \verbatim
         *    b[m1_, m2_] := 
         *     If[m1 == 0, 1, 
         *     If[m1 < 0 && m2 < 0, -I/std::sqrt[2], 
         *     If[m1 > 0 && m2 < 0, (-1)^m1*I/std::sqrt[2], 
         *     If[m1 < 0 && m2 > 0, (-1)^m2/std::sqrt[2], 
         *     If[m1 > 0 && m2 > 0, 1/std::sqrt[2]]]]]]
         *    
         *    a[m1_, m2_] := If[Abs[m1] == Abs[m2], b[m1, m2], 0]
         *    
         *    R[l_, m_, t_, p_] := Sum[a[m1, m]*SphericalHarmonicY[l, m1, t, p], {m1, -l, l}]
         *    \endverbatim
         */
        static inline double_complex ylm_dot_rlm(int l, int m1, int m2)
        {
            const double isqrt2 = 0.70710678118654752440;

            assert(l >= 0 && std::abs(m1) <= l && std::abs(m2) <= l);

            if (!((m1 == m2) || (m1 == -m2))) {
                return double_complex(0, 0);
            }

            if (m1 == 0) {
                return double_complex(1, 0);
            }

            if (m1 < 0) {
                if (m2 < 0) {
                    return -double_complex(0, isqrt2);
                } else {
                    return std::pow(-1.0, m2) * double_complex(isqrt2, 0);
                }
            } else {
                if (m2 < 0) {
                    return pow(-1.0, m1) * double_complex(0, isqrt2);
                } else {
                    return double_complex(isqrt2, 0);
                }
            }
        }

        static inline double_complex rlm_dot_ylm(int l, int m1, int m2)
        {
            return std::conj(ylm_dot_rlm(l, m2, m1));
        }

        /// Gaunt coefficent of three complex spherical harmonics.
        /** 
         *  \f[
         *    \langle Y_{\ell_1 m_1} | Y_{\ell_2 m_2} | Y_{\ell_3 m_3} \rangle
         *  \f]
         */
        static double gaunt_ylm(int l1, int l2, int l3, int m1, int m2, int m3);

        /// Gaunt coefficent of three real spherical harmonics.
        /** 
         *  \f[
         *    \langle R_{\ell_1 m_1} | R_{\ell_2 m_2} | R_{\ell_3 m_3} \rangle
         *  \f]
         */
        static double gaunt_rlm(int l1, int l2, int l3, int m1, int m2, int m3);

        /// Gaunt coefficent of two complex and one real spherical harmonics.
        /** 
         *  \f[
         *    \langle Y_{\ell_1 m_1} | R_{\ell_2 m_2} | Y_{\ell_3 m_3} \rangle
         *  \f]
         */
        static double_complex gaunt_hybrid(int l1, int l2, int l3, int m1, int m2, int m3);

        void uniform_coverage();

        /// Return Clebsch-Gordan coefficient.
        /** Clebsch-Gordan coefficients arise when two angular momenta are combined into a
         *  total angular momentum. 
         */
        static inline double clebsch_gordan(int l1, int l2, int l3, int m1, int m2, int m3)
        {
            assert(l1 >= 0);
            assert(l2 >= 0);
            assert(l3 >= 0);
            assert(m1 >= -l1 && m1 <= l1);
            assert(m2 >= -l2 && m2 <= l2);
            assert(m3 >= -l3 && m3 <= l3);

            return pow(-1, l1 - l2 + m3) * sqrt(double(2 * l3 + 1)) * 
                   gsl_sf_coupling_3j(2 * l1, 2 * l2, 2 * l3, 2 * m1, 2 * m2, -2 * m3);
        }

        inline double_complex ylm_backward(int lm,  int itp)
        {
            return ylm_backward_(lm, itp);
        }
        
        inline double rlm_backward(int lm,  int itp)
        {
            return rlm_backward_(lm, itp);
        }

        inline double coord(int x, int itp)
        {
            return coord_(x, itp);
        }

        inline int num_points()
        {
            return num_points_;
        }

        inline int lmax()
        {
            return lmax_;
        }

        inline int lmmax()
        {
            return lmmax_;
        }

        static void wigner_d_matrix(int l, double beta, mdarray<double, 2>& d_mtrx__)
        {
            long double cos_b2 = std::cos((long double)beta / 2.0L);
            long double sin_b2 = std::sin((long double)beta / 2.0L);
            
            for (int m1 = -l; m1 <= l; m1++)
            {
                for (int m2 = -l; m2 <= l; m2++)
                {
                    long double d = 0;
                    for (int j = 0; j <= std::min(l + m1, l - m2); j++)
                    {
                        if ((l - m2 - j) >= 0 && (l + m1 - j) >= 0 && (j + m2 - m1) >= 0)
                        {
                            long double g = (std::sqrt(Utils::factorial(l + m1)) / Utils::factorial(l - m2 - j)) *
                                            (std::sqrt(Utils::factorial(l - m1)) / Utils::factorial(l + m1 - j)) * 
                                            (std::sqrt(Utils::factorial(l - m2)) / Utils::factorial(j + m2 - m1)) * 
                                            (std::sqrt(Utils::factorial(l + m2)) / Utils::factorial(j));
                            d += g * std::pow(-1, j) * std::pow(cos_b2, 2 * l + m1 - m2 - 2 * j) * std::pow(sin_b2, 2 * j + m2 - m1);
                        }
                    }
                    d_mtrx__(m1 + l, m2 + l) = (double)d;
                }
            }
        }

        static void rotation_matrix_l(int l, vector3d<double> euler_angles, int proper_rotation, 
                                      double_complex* rot_mtrx__, int ld)
        {
            mdarray<double_complex, 2> rot_mtrx(rot_mtrx__, ld, 2 * l + 1);

            mdarray<double, 2> d_mtrx(2 * l + 1, 2 * l + 1);
            wigner_d_matrix(l, euler_angles[1], d_mtrx);

            for (int m1 = -l; m1 <= l; m1++)
            {
                for (int m2 = -l; m2 <= l; m2++)
                {
                    rot_mtrx(m1 + l, m2 + l) = std::exp(double_complex(0, -euler_angles[0] * m1 - euler_angles[2] * m2)) * 
                                               d_mtrx(m1 + l, m2 + l) * std::pow(proper_rotation, l);
                }
            }
        }

        static void rotation_matrix_l(int l, vector3d<double> euler_angles, int proper_rotation, 
                                      double* rot_mtrx__, int ld)
        {
            mdarray<double, 2> rot_mtrx_rlm(rot_mtrx__, ld, 2 * l + 1);
            mdarray<double_complex, 2> rot_mtrx_ylm(2 * l + 1, 2 * l + 1);

            mdarray<double, 2> d_mtrx(2 * l + 1, 2 * l + 1);
            wigner_d_matrix(l, euler_angles[1], d_mtrx);

            for (int m1 = -l; m1 <= l; m1++)
            {
                for (int m2 = -l; m2 <= l; m2++)
                {
                    rot_mtrx_ylm(m1 + l, m2 + l) = std::exp(double_complex(0, -euler_angles[0] * m1 - euler_angles[2] * m2)) * 
                                                   d_mtrx(m1 + l, m2 + l) * std::pow(proper_rotation, l);
                }
            }
            for (int m1 = -l; m1 <= l; m1++)
            {
                auto i13 = (m1 == 0) ? std::vector<int>({0}) : std::vector<int>({-m1, m1});

                for (int m2 = -l; m2 <= l; m2++)
                {
                    auto i24 = (m2 == 0) ? std::vector<int>({0}) : std::vector<int>({-m2, m2});

                    for (int m3: i13)
                    {
                        for (int m4: i24)
                        {
                            rot_mtrx_rlm(m1 + l, m2 + l) += std::real(rlm_dot_ylm(l, m1, m3) *
                                                                      rot_mtrx_ylm(m3 + l, m4 + l) *
                                                                      ylm_dot_rlm(l, m4, m2));
                        }
                    }
                }
            }
        }

        static void rotation_matrix(int lmax, vector3d<double> euler_angles, int proper_rotation, 
                                    mdarray<double_complex, 2>& rotm)
        {
            rotm.zero();

            for (int l = 0; l <= lmax; l++)
            {
                rotation_matrix_l(l, euler_angles, proper_rotation, &rotm(l * l, l * l), rotm.ld());
            }
        }

        static void rotation_matrix(int lmax, vector3d<double> euler_angles, int proper_rotation, 
                                    mdarray<double, 2>& rotm)
        {
            rotm.zero();

            for (int l = 0; l <= lmax; l++)
            {
                rotation_matrix_l(l, euler_angles, proper_rotation, &rotm(l * l, l * l), rotm.ld());
            }
        }

        /// Compute derivative of real-spherical harmonic with respect to theta angle.
        static double dRlm_dtheta(int lm, double theta, double phi)
        {
            switch (lm) {
                case 0: return 0;
                
                case 1: return -(std::sqrt(3/pi)*std::cos(theta)*std::sin(phi))/2.;
                
                case 2: return -(std::sqrt(3/pi)*std::sin(theta))/2.;
                
                case 3: return -(std::sqrt(3/pi)*std::cos(phi)*std::cos(theta))/2.;
                
                case 4: return -(std::sqrt(15/pi)*std::cos(phi)*std::cos(theta)*std::sin(phi)*std::sin(theta));
                
                case 5: return -(std::sqrt(15/pi)*std::cos(2*theta)*std::sin(phi))/2.;
                
                case 6: return (-3*std::sqrt(5/pi)*std::cos(theta)*std::sin(theta))/2.;
                
                case 7: return -(std::sqrt(15/pi)*std::cos(phi)*std::cos(2*theta))/2.;
                
                case 8: return (std::sqrt(15/pi)*std::cos(2*phi)*std::sin(2*theta))/4.;
                
                case 9: return (-3*std::sqrt(35/(2.*pi))*std::cos(theta)*std::sin(3*phi)*std::pow(std::sin(theta),2))/4.;
                
                case 10: return (std::sqrt(105/pi)*std::sin(2*phi)*(std::sin(theta) - 3*std::sin(3*theta)))/16.;
                
                case 11: return (std::sqrt(21/(2.*pi))*std::cos(theta)*(7 - 15*std::cos(2*theta))*std::sin(phi))/8.;
                
                case 12: return (-3*std::sqrt(7/pi)*(3 + 5*std::cos(2*theta))*std::sin(theta))/8.;
                
                case 13: return (std::sqrt(21/(2.*pi))*std::cos(phi)*std::cos(theta)*(7 - 15*std::cos(2*theta)))/8.;
                
                case 14: return (std::sqrt(105/pi)*std::cos(2*phi)*(1 + 3*std::cos(2*theta))*std::sin(theta))/8.;
                
                case 15: return (-3*std::sqrt(35/(2.*pi))*std::cos(3*phi)*std::cos(theta)*std::pow(std::sin(theta),2))/4.;
                
                case 16: return (-3*std::sqrt(35/pi)*std::cos(theta)*std::sin(4*phi)*std::pow(std::sin(theta),3))/4.;
                
                case 17: return (-3*std::sqrt(35/(2.*pi))*(1 + 2*std::cos(2*theta))*std::sin(3*phi)*std::pow(std::sin(theta),2))/4.;
                
                case 18: return (3*std::sqrt(5/pi)*(1 - 7*std::cos(2*theta))*std::sin(2*phi)*std::sin(2*theta))/8.;
                
                case 19: return (-3*std::sqrt(5/(2.*pi))*(std::cos(2*theta) + 7*std::cos(4*theta))*std::sin(phi))/8.;
                
                case 20: return (15*std::cos(theta)*(3 - 7*std::pow(std::cos(theta),2))*std::sin(theta))/(4.*std::sqrt(pi));
                
                case 21: return (-3*std::sqrt(5/(2.*pi))*std::cos(phi)*(std::cos(2*theta) + 7*std::cos(4*theta)))/8.;
                
                case 22: return (3*std::sqrt(5/pi)*std::cos(2*phi)*(-2*std::sin(2*theta) + 7*std::sin(4*theta)))/16.;
                
                case 23: return (-3*std::sqrt(35/(2.*pi))*std::cos(3*phi)*(1 + 2*std::cos(2*theta))*std::pow(std::sin(theta),2))/4.;
                
                case 24: return (3*std::sqrt(35/pi)*std::cos(4*phi)*std::cos(theta)*std::pow(std::sin(theta),3))/4.;

                default: {
                    TERMINATE_NOT_IMPLEMENTED
                }
            }
            return 0; // make compiler happy
        }
        
        ///  Compute derivative of real-spherical harmonic with respect to phi angle and divide by sin(theta).
        static double dRlm_dphi_sin_theta(int lm, double theta, double phi)
        {
            switch (lm) {
                case 0: return 0;
                
                case 1: return -(std::sqrt(3/pi)*std::cos(phi))/2.;
                
                case 2: return 0;
                
                case 3: return (std::sqrt(3/pi)*std::sin(phi))/2.;
                
                case 4: return -(std::sqrt(15/pi)*std::cos(2*phi)*std::sin(theta))/2.;
                
                case 5: return -(std::sqrt(15/pi)*std::cos(phi)*std::cos(theta))/2.;
                
                case 6: return 0;
                
                case 7: return (std::sqrt(15/pi)*std::cos(theta)*std::sin(phi))/2.;
                
                case 8: return -(std::sqrt(15/pi)*std::cos(phi)*std::sin(phi)*std::sin(theta));
                
                case 9: return (-3*std::sqrt(35/(2.*pi))*std::cos(3*phi)*std::pow(std::sin(theta),2))/4.;
                
                case 10: return -(std::sqrt(105/pi)*std::cos(2*phi)*std::sin(2*theta))/4.;
                
                case 11: return -(std::sqrt(21/(2.*pi))*std::cos(phi)*(3 + 5*std::cos(2*theta)))/8.;
                
                case 12: return 0;
                
                case 13: return (std::sqrt(21/(2.*pi))*(3 + 5*std::cos(2*theta))*std::sin(phi))/8.;
                
                case 14: return -(std::sqrt(105/pi)*std::cos(phi)*std::cos(theta)*std::sin(phi)*std::sin(theta));
                
                case 15: return (3*std::sqrt(35/(2.*pi))*std::sin(3*phi)*std::pow(std::sin(theta),2))/4.;
                
                case 16: return (-3*std::sqrt(35/pi)*std::cos(4*phi)*std::pow(std::sin(theta),3))/4.;
                
                case 17: return (-9*std::sqrt(35/(2.*pi))*std::cos(3*phi)*std::cos(theta)*std::pow(std::sin(theta),2))/4.;
                
                case 18: return (-3*std::sqrt(5/pi)*std::cos(2*phi)*(3*std::sin(theta) + 7*std::sin(3*theta)))/16.;
                
                case 19: return (-3*std::sqrt(5/(2.*pi))*std::cos(phi)*(9*std::cos(theta) + 7*std::cos(3*theta)))/16.;
                
                case 20: return 0;
                
                case 21: return (3*std::sqrt(5/(2.*pi))*std::cos(theta)*(1 + 7*std::cos(2*theta))*std::sin(phi))/8.;
                
                case 22: return (-3*std::sqrt(5/pi)*std::sin(2*phi)*(3*std::sin(theta) + 7*std::sin(3*theta)))/16.;
                
                case 23: return (9*std::sqrt(35/(2.*pi))*std::cos(theta)*std::sin(3*phi)*std::pow(std::sin(theta),2))/4.;
                
                case 24: return (-3*std::sqrt(35/pi)*std::sin(4*phi)*std::pow(std::sin(theta),3))/4.;

                default: {
                    TERMINATE_NOT_IMPLEMENTED
                }
            }
            return 0; // make compiler happy
        }
};

template <>
inline void SHT::backward_transform<double>(int ld, double const* flm, int nr, int lmmax, double* ftp)
{
    assert(lmmax <= lmmax_);
    assert(ld >= lmmax);
    linalg<CPU>::gemm(1, 0, num_points_, nr, lmmax, &rlm_backward_(0, 0), lmmax_, flm, ld, ftp, num_points_);
}

template <>
inline void SHT::backward_transform<double_complex>(int ld, double_complex const* flm, int nr, int lmmax, double_complex* ftp)
{
    assert(lmmax <= lmmax_);
    assert(ld >= lmmax);
    linalg<CPU>::gemm(1, 0, num_points_, nr, lmmax, &ylm_backward_(0, 0), lmmax_, flm, ld, ftp, num_points_);
}

template <>
inline void SHT::forward_transform<double>(double const* ftp, int nr, int lmmax, int ld, double* flm)
{
    assert(lmmax <= lmmax_);
    assert(ld >= lmmax);
    linalg<CPU>::gemm(1, 0, lmmax, nr, num_points_, &rlm_forward_(0, 0), num_points_, ftp, num_points_, flm, ld);
}

template <>
inline void SHT::forward_transform<double_complex>(double_complex const* ftp, int nr, int lmmax, int ld, double_complex* flm)
{
    assert(lmmax <= lmmax_);
    assert(ld >= lmmax);
    linalg<CPU>::gemm(1, 0, lmmax, nr, num_points_, &ylm_forward_(0, 0), num_points_, ftp, num_points_, flm, ld);
}

inline double SHT::gaunt_ylm(int l1, int l2, int l3, int m1, int m2, int m3)
{
    assert(l1 >= 0);
    assert(l2 >= 0);
    assert(l3 >= 0);
    assert(m1 >= -l1 && m1 <= l1);
    assert(m2 >= -l2 && m2 <= l2);
    assert(m3 >= -l3 && m3 <= l3);
    
    return std::pow(-1.0, std::abs(m1)) * std::sqrt(double(2 * l1 + 1) * double(2 * l2 + 1) * double(2 * l3 + 1) / fourpi) * 
           gsl_sf_coupling_3j(2 * l1, 2 * l2, 2 * l3, 0, 0, 0) *
           gsl_sf_coupling_3j(2 * l1, 2 * l2, 2 * l3, -2 * m1, 2 * m2, 2 * m3);
}

inline double SHT::gaunt_rlm(int l1, int l2, int l3, int m1, int m2, int m3)
{
    assert(l1 >= 0);
    assert(l2 >= 0);
    assert(l3 >= 0);
    assert(m1 >= -l1 && m1 <= l1);
    assert(m2 >= -l2 && m2 <= l2);
    assert(m3 >= -l3 && m3 <= l3);
    
    double d = 0;
    for (int k1 = -l1; k1 <= l1; k1++)
    {
        for (int k2 = -l2; k2 <= l2; k2++)
        {
            for (int k3 = -l3; k3 <= l3; k3++)
            {
                d += std::real(std::conj(SHT::ylm_dot_rlm(l1, k1, m1)) *
                                         SHT::ylm_dot_rlm(l2, k2, m2) *
                                         SHT::ylm_dot_rlm(l3, k3, m3)) * SHT::gaunt_ylm(l1, l2, l3, k1, k2, k3);
            }
        }
    }
    return d;
}

inline double_complex SHT::gaunt_hybrid(int l1, int l2, int l3, int m1, int m2, int m3)
{
    assert(l1 >= 0);
    assert(l2 >= 0);
    assert(l3 >= 0);
    assert(m1 >= -l1 && m1 <= l1);
    assert(m2 >= -l2 && m2 <= l2);
    assert(m3 >= -l3 && m3 <= l3);

    if (m2 == 0) 
    {
        return double_complex(gaunt_ylm(l1, l2, l3, m1, m2, m3), 0.0);
    }
    else 
    {
        return (ylm_dot_rlm(l2, m2, m2) * gaunt_ylm(l1, l2, l3, m1, m2, m3) +  
                ylm_dot_rlm(l2, -m2, m2) * gaunt_ylm(l1, l2, l3, m1, -m2, m3));
    }
}


inline vector3d<double> SHT::spherical_coordinates(vector3d<double> vc)
{
    vector3d<double> vs;

    double eps = 1e-12;

    vs[0] = vc.length();

    if (vs[0] <= eps)
    {
        vs[1] = 0.0;
        vs[2] = 0.0;
    } 
    else
    {
        vs[1] = std::acos(vc[2] / vs[0]); // theta = cos^{-1}(z/r)

        if (std::abs(vc[0]) > eps || std::abs(vc[1]) > eps)
        {
            vs[2] = std::atan2(vc[1], vc[0]); // phi = tan^{-1}(y/x)
            if (vs[2] < 0.0) vs[2] += twopi;
        }
        else
        {
            vs[2] = 0.0;
        }
    }

    return vs;
}

inline void SHT::spherical_harmonics(int lmax, double theta, double phi, double_complex* ylm)
{
    double x = std::cos(theta);
    std::vector<double> result_array(lmax + 1);

    for (int l = 0; l <= lmax; l++)
    {
        for (int m = 0; m <= l; m++)
        {
            double_complex z = std::exp(double_complex(0.0, m * phi)); 
            ylm[Utils::lm_by_l_m(l, m)] = gsl_sf_legendre_sphPlm(l, m, x) * z;
            if (m % 2) 
            {
                ylm[Utils::lm_by_l_m(l, -m)] = -std::conj(ylm[Utils::lm_by_l_m(l, m)]);
            }
            else
            {
                ylm[Utils::lm_by_l_m(l, -m)] = std::conj(ylm[Utils::lm_by_l_m(l, m)]);        
            }
        }
    }
}

inline void SHT::spherical_harmonics(int lmax, double theta, double phi, double* rlm)
{
    int lmmax = (lmax + 1) * (lmax + 1);
    std::vector<double_complex> ylm(lmmax);
    spherical_harmonics(lmax, theta, phi, &ylm[0]);
    
    double t = std::sqrt(2.0);
    
    rlm[0] = y00;

    for (int l = 1; l <= lmax; l++)
    {
        for (int m = -l; m < 0; m++) 
            rlm[Utils::lm_by_l_m(l, m)] = t * ylm[Utils::lm_by_l_m(l, m)].imag();
        
        rlm[Utils::lm_by_l_m(l, 0)] = ylm[Utils::lm_by_l_m(l, 0)].real();
         
        for (int m = 1; m <= l; m++) 
            rlm[Utils::lm_by_l_m(l, m)] = t * ylm[Utils::lm_by_l_m(l, m)].real();
    }
}
                
inline void SHT::uniform_coverage()
{
    tp_(0, 0) = pi;
    tp_(1, 0) = 0;

    for (int k = 1; k < num_points_ - 1; k++)
    {
        double hk = -1.0 + double(2 * k) / double(num_points_ - 1);
        tp_(0, k) = acos(hk);
        double t = tp_(1, k - 1) + 3.80925122745582 / sqrt(double(num_points_)) / sqrt(1 - hk * hk);
        tp_(1, k) = fmod(t, twopi);
    }
    
    tp_(0, num_points_ - 1) = 0;
    tp_(1, num_points_ - 1) = 0;
}

inline void SHT::convert(int lmax__, double const* f_rlm__, double_complex* f_ylm__)
{
    int lm = 0;
    for (int l = 0; l <= lmax__; l++)
    {
        for (int m = -l; m <= l; m++)
        {
            if (m == 0)
            {
                f_ylm__[lm] = f_rlm__[lm];
            }
            else 
            {
                int lm1 = Utils::lm_by_l_m(l, -m);
                f_ylm__[lm] = ylm_dot_rlm(l, m, m) * f_rlm__[lm] + ylm_dot_rlm(l, m, -m) * f_rlm__[lm1];
            }
            lm++;
        }
    }
}

inline void SHT::convert(int lmax__, double_complex const* f_ylm__, double* f_rlm__)
{
    int lm = 0;
    for (int l = 0; l <= lmax__; l++)
    {
        for (int m = -l; m <= l; m++)
        {
            if (m == 0)
            {
                f_rlm__[lm] = std::real(f_ylm__[lm]);
            }
            else 
            {
                int lm1 = Utils::lm_by_l_m(l, -m);
                f_rlm__[lm] = std::real(rlm_dot_ylm(l, m, m) * f_ylm__[lm] + rlm_dot_ylm(l, m, -m) * f_ylm__[lm1]);
            }
            lm++;
        }
    }
}

};

#endif // __SHT_H__
