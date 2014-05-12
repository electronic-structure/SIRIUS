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

namespace sirius
{

template <typename T> 
class Spheric_function1
{
    private:

        /// Spheric function values.
        mdarray<T, 2> data_;
        
        /// Radial grid.
        Radial_grid radial_grid_;
        
        int angular_domain_size_;

        int angular_domain_idx_;

        int radial_domain_idx_;

    public:

        Spheric_function1()
        {
        }

        Spheric_function1(Radial_grid& radial_grid__, int angular_domain_size__) 
            : radial_grid_(radial_grid__), 
              angular_domain_size_(angular_domain_size__), 
              angular_domain_idx_(1),
              radial_domain_idx_(0)
        {
            data_.set_dimensions(radial_grid_.num_points(), angular_domain_size_);
            data_.allocate();
        }
        
        Spheric_function1(int angular_domain_size__, Radial_grid& radial_grid__) 
            : radial_grid_(radial_grid__), 
              angular_domain_size_(angular_domain_size__), 
              angular_domain_idx_(0),
              radial_domain_idx_(1)
        {
            data_.set_dimensions(angular_domain_size_, radial_grid_.num_points());
            data_.allocate();
        }

        Spheric_function1(T* ptr, int angular_domain_size__, Radial_grid& radial_grid__) 
            : radial_grid_(radial_grid__), 
              angular_domain_size_(angular_domain_size__), 
              angular_domain_idx_(0),
              radial_domain_idx_(1)
        {
            data_.set_dimensions(angular_domain_size_, radial_grid_.num_points());
            data_.set_ptr(ptr);
        }

        inline int angular_domain_size()
        {
            return angular_domain_size_;
        }

        inline int angular_domain_idx()
        {
            return angular_domain_idx_;
        }

        inline int radial_domain_idx()
        {
            return radial_domain_idx_;
        }

        inline Radial_grid& radial_grid()
        {
            return radial_grid_;
        }

        inline T& operator()(const int64_t i0, const int64_t i1) 
        {
            return data_(i0, i1);
        }

        void zero()
        {
            data_.zero();
        }

        void allocate()
        {
            data_.allocate();
        }

        void set_ptr(T* ptr)
        {
            data_.set_ptr(ptr);
        }
};


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

        Spheric_function1<double> convert(Spheric_function1<double_complex>& f)
        {
            Spheric_function1<double> g;

            int lmax = Utils::lmax_by_lmmax(f.angular_domain_size());

            /* cache transformation arrays */
            std::vector<double_complex> tpp(f.angular_domain_size());
            std::vector<double_complex> tpm(f.angular_domain_size());
            for (int l = 0; l <= lmax; l++)
            {
                for (int m = -l; m <= l; m++) 
                {
                    int lm = Utils::lm_by_l_m(l, m);
                    tpp[lm] = rlm_dot_ylm(l, m, m);
                    tpm[lm] = rlm_dot_ylm(l, m, -m);
                }
            }

            /* radial index is first */
            if (f.radial_domain_idx() == 0)
            {
                g = Spheric_function1<double>(f.radial_grid(), f.angular_domain_size());
                int lm = 0;
                for (int l = 0; l <= lmax; l++)
                {
                    for (int m = -l; m <= l; m++)
                    {
                        if (m == 0)
                        {
                            for (int ir = 0; ir < f.radial_grid().num_points(); ir++) 
                                g(ir, lm) = real(f(ir, lm));
                        }
                        else 
                        {
                            int lm1 = Utils::lm_by_l_m(l, -m);
                            for (int ir = 0; ir < f.radial_grid().num_points(); ir++)
                                g(ir, lm) = real(tpp[lm] * f(ir, lm) + tpm[lm] * f(ir, lm1));
                        }
                        lm++;
                    }
                }
            }
            else
            {
                g = Spheric_function1<double>(f.angular_domain_size(), f.radial_grid());
                for (int ir = 0; ir < f.radial_grid().num_points(); ir++)
                {
                    int lm = 0;
                    for (int l = 0; l <= lmax; l++)
                    {
                        for (int m = -l; m <= l; m++)
                        {
                            if (m == 0)
                            {
                                g(lm, ir) = real(f(lm, ir));
                            }
                            else 
                            {
                                int lm1 = Utils::lm_by_l_m(l, -m);
                                g(lm, ir) = real(tpp[lm] * f(lm, ir) + tpm[lm] * f(lm1, ir));
                            }
                            lm++;
                        }
                    }
                }
            }

            return g;
        }

        Spheric_function1<double_complex> convert(Spheric_function1<double> f)
        {
            Spheric_function1<double_complex> g;
            
            int lmax = Utils::lmax_by_lmmax(f.angular_domain_size());

            /* cache transformation arrays */
            std::vector<double_complex> tpp(f.angular_domain_size());
            std::vector<double_complex> tpm(f.angular_domain_size());
            for (int l = 0; l <= lmax; l++)
            {
                for (int m = -l; m <= l; m++) 
                {
                    int lm = Utils::lm_by_l_m(l, m);
                    tpp[lm] = ylm_dot_rlm(l, m, m);
                    tpm[lm] = ylm_dot_rlm(l, m, -m);
                }
            }

            /* radial index is first */
            if (f.radial_domain_idx() == 0)
            {
                g = Spheric_function1<double_complex>(f.radial_grid(), f.angular_domain_size());

                int lm = 0;
                for (int l = 0; l <= lmax; l++)
                {
                    for (int m = -l; m <= l; m++)
                    {
                        if (m == 0)
                        {
                            for (int ir = 0; ir < f.radial_grid().num_points(); ir++) g(ir, lm) = f(ir, lm);
                        }
                        else 
                        {
                            int lm1 = Utils::lm_by_l_m(l, -m);
                            for (int ir = 0; ir < f.radial_grid().num_points(); ir++)
                                g(ir, lm) = tpp[lm] * f(ir, lm) + tpm[lm] * f(ir, lm1);
                        }
                        lm++;
                    }
                }
            }
            else
            {
                g = Spheric_function1<double_complex>(f.angular_domain_size(), f.radial_grid());
                for (int ir = 0; ir < f.radial_grid().num_points(); ir++)
                {
                    int lm = 0;
                    for (int l = 0; l <= lmax; l++)
                    {
                        for (int m = -l; m <= l; m++)
                        {
                            if (m == 0)
                            {
                                g(lm, ir) = f(lm, ir);
                            }
                            else 
                            {
                                int lm1 = Utils::lm_by_l_m(l, -m);
                                g(lm, ir) = tpp[lm] * f(lm, ir) + tpm[lm] * f(lm1, ir);
                            }
                            lm++;
                        }
                    }
                }
            }

            return g;
        }

        template <int direction, typename T>
        Spheric_function1<T> transform(Spheric_function1<T>& f)
        {
            Spheric_function1<T> g;

            switch (direction)
            {
                /* forward transform, f(t, p) -> g(l, m) */
                case 1:
                {
                    g = Spheric_function1<T>(lmmax(), f.radial_grid());
                    forward_transform(&f(0, 0), lmmax(), f.radial_grid().num_points(), &g(0, 0));
                    break;
                }
                /* backward transform, f(l, m) -> g(t, p) */
                case -1:
                {
                    g = Spheric_function1<T>(num_points(), f.radial_grid());
                    backward_transform(&f(0, 0), lmmax(), f.radial_grid().num_points(), &g(0, 0));
                    break;
                }
                default:
                {
                    error_local(__FILE__, __LINE__, "Wrong direction of transformation");
                }
            }
            return g;
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
