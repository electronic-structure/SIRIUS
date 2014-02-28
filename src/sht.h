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

namespace sirius
{

class SHT
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
        
        /// Default constructor
        SHT(int lmax_);
        
        template <typename T>
        inline void backward_transform(T* flm, int lmmax, int ncol, T* ftp);
        
        template <typename T>
        inline void forward_transform(T* ftp, int lmmax, int ncol, T* flm);
        
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
        static void spherical_coordinates(vector3d<double> vc, double* vs);

        /// Generate complex spherical harmonics Ylm
        static void spherical_harmonics(int lmax, double theta, double phi, double_complex* ylm);
        
        /// Generate real spherical harmonics Rlm
        /** Mathematica code:
            \verbatim
            R[l_, m_, th_, ph_] := 
             If[m > 0, Sqrt[2]*ComplexExpand[Re[SphericalHarmonicY[l, m, th, ph]]], 
             If[m < 0, Sqrt[2]*ComplexExpand[Im[SphericalHarmonicY[l, m, th, ph]]], 
             If[m == 0, ComplexExpand[Re[SphericalHarmonicY[l, 0, th, ph]]]]]]
            \endverbatim
        */
        static void spherical_harmonics(int lmax, double theta, double phi, double* rlm);
                        
        /// Compute element of the transformation matrix from complex to real spherical harmonics. 
        /** Real spherical harmonic can be written as a linear combination of complex harmonics:

            \f[
                R_{\ell m}(\theta, \phi) = \sum_{m'} a^{\ell}_{m' m}Y_{\ell m'}(\theta, \phi)
            \f]
            where 
            \f[
                a^{\ell}_{m' m} = \langle Y_{\ell m'} | R_{\ell m} \rangle
            \f]
        
            Transformation from real to complex spherical harmonics is conjugate transpose:
            
            \f[
                Y_{\ell m}(\theta, \phi) = \sum_{m'} a^{\ell*}_{m m'}R_{\ell m'}(\theta, \phi)
            \f]
        
            Mathematica code:
            \verbatim
            b[m1_, m2_] := 
             If[m1 == 0, 1, 
             If[m1 < 0 && m2 < 0, -I/Sqrt[2], 
             If[m1 > 0 && m2 < 0, (-1)^m1*I/Sqrt[2], 
             If[m1 < 0 && m2 > 0, (-1)^m2/Sqrt[2], 
             If[m1 > 0 && m2 > 0, 1/Sqrt[2]]]]]]
            
            a[m1_, m2_] := If[Abs[m1] == Abs[m2], b[m1, m2], 0]
            
            R[l_, m_, t_, p_] := Sum[a[m1, m]*SphericalHarmonicY[l, m1, t, p], {m1, -l, l}]
            \endverbatim
        */
        static double_complex ylm_dot_rlm(int l, int m1, int m2);
        
        /// Return real or complex Gaunt coefficent.
        template <typename T>
        static inline T gaunt(int l1, int l2, int l3, int m1, int m2, int m3);
        
        void uniform_coverage();

        /// Return Clebsch-Gordan coefficient.
        /** Clebsch-Gordan coefficients arise when two angular momenta are combined into a
            total angular momentum. */
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

#include "sht.hpp"

};

#endif // __SHT_H__
