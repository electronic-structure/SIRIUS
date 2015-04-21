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

/** \file sht.h
 *   
 *  \brief Contains declaration and particular implementation of sirius::SHT class.
 */

#ifndef __SHT_H__
#define __SHT_H__

#include <math.h>
#include <gsl/gsl_sf_coupling.h>
#include <gsl/gsl_sf_legendre.h>
#include <string.h>
#include <vector>
#include "typedefs.h"
#include "utils.h"
#include "constants.h"
#include "linalg.h"
#include "LebedevLaikov.h"
#include "radial_grid.h"
#include "spheric_function.h"

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
        SHT(int lmax_);
       
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
        void backward_transform(int ld, T* flm, int nr, int lmmax, T* ftp);
        
        /// Perform a forward transformation from spherical coordinates to spherical harmonics.
        /** 
         *  \f[
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
        void forward_transform(T* ftp, int nr, int lmmax, int ld, T* flm);
        
        /// Convert from Ylm to Rlm representation.
        Spheric_function<spectral, double> convert(Spheric_function<spectral, double_complex>& f);
        
        /// Convert form Rlm to Ylm representation.
        Spheric_function<spectral, double_complex> convert(Spheric_function<spectral, double>& f);
        
        /// Convert from Ylm to Rlm representation.
        void convert(Spheric_function<spectral, double_complex>& f, Spheric_function<spectral, double>& g);

        template <typename T>
        Spheric_function<spectral, T> transform(Spheric_function<spatial, T>& f)
        {
            Spheric_function<spectral, T> g(lmmax(), f.radial_grid());
            
            forward_transform(&f(0, 0), f.radial_grid().num_points(), lmmax(), lmmax(), &g(0, 0));

            return std::move(g);
        }
        
        template <typename T>
        Spheric_function<spatial, T> transform(Spheric_function<spectral, T>& f)
        {
            Spheric_function<spatial, T> g(num_points(), f.radial_grid());
            
            backward_transform(f.angular_domain_size(), &f(0, 0), f.radial_grid().num_points(), 
                               std::min(lmmax(), f.angular_domain_size()), &g(0, 0));

            return std::move(g);
        }
        
        template <typename T>
        void transform(Spheric_function<spatial, T>& f, Spheric_function<spectral, T>&g)
        {
            assert(f.radial_grid().hash() == g.radial_grid().hash());

            forward_transform(&f(0, 0), f.radial_grid().num_points(), std::min(g.angular_domain_size(), lmmax()), 
                              g.angular_domain_size(), &g(0, 0));
        }

        
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
         *   If[m > 0, Sqrt[2]*ComplexExpand[Re[SphericalHarmonicY[l, m, th, ph]]], 
         *   If[m < 0, Sqrt[2]*ComplexExpand[Im[SphericalHarmonicY[l, m, th, ph]]], 
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
         *     If[m1 < 0 && m2 < 0, -I/Sqrt[2], 
         *     If[m1 > 0 && m2 < 0, (-1)^m1*I/Sqrt[2], 
         *     If[m1 < 0 && m2 > 0, (-1)^m2/Sqrt[2], 
         *     If[m1 > 0 && m2 > 0, 1/Sqrt[2]]]]]]
         *    
         *    a[m1_, m2_] := If[Abs[m1] == Abs[m2], b[m1, m2], 0]
         *    
         *    R[l_, m_, t_, p_] := Sum[a[m1, m]*SphericalHarmonicY[l, m1, t, p], {m1, -l, l}]
         *    \endverbatim
         */
        static double_complex ylm_dot_rlm(int l, int m1, int m2);
        
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

        static inline double_complex rlm_dot_ylm(int l, int m1, int m2)
        {
            return conj(ylm_dot_rlm(l, m2, m1));
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

        static void wigner_d_matrix(int l, double beta, double* d_mtrx__, int ld)
        {
            mdarray<double, 2> d_mtrx(d_mtrx__, ld, 2 * l + 1);

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
                            d += g * pow(-1, j) * std::pow(cos_b2, 2 * l + m1 - m2 - 2 * j) * std::pow(sin_b2, 2 * j + m2 - m1);
                        }
                    }
                    d_mtrx(m1 + l, m2 + l) = (double)d;
                }
            }
        }

        static void rotation_matrix_l(int l, vector3d<double> euler_angles, int proper_rotation, 
                                      double_complex* rot_mtrx__, int ld)
        {
            mdarray<double_complex, 2> rot_mtrx(rot_mtrx__, ld, 2 * l + 1);

            mdarray<double, 2> d_mtrx(2 * l + 1, 2 * l + 1);
            wigner_d_matrix(l, euler_angles[1], &d_mtrx(0, 0), 2 * l + 1);

            double p = (proper_rotation == -1) ? pow(-1.0, l) : 1.0; 
            for (int m1 = -l; m1 <= l; m1++)
            {
                for (int m2 = -l; m2 <= l; m2++)
                {
                    rot_mtrx(m1 + l, m2 + l) = exp(double_complex(0, -euler_angles[0] * m1 - euler_angles[2] * m2)) * 
                                               d_mtrx(m1 + l, m2 + l) * p;
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
};

};

#endif // __SHT_H__
