
namespace sirius
{

class SHT
{
    private:

        int lmax_;

        int lmmax_;

        int num_points_;

        mdarray<double,2> coord_;

        std::vector<double> w_;

        /// backward transformation from Ylm to spherical coordinates
        mdarray<complex16,2> ylm_backward_;
        
        /// forward transformation from spherical coordinates to Ylm
        mdarray<complex16,2> ylm_forward_;
        
        /// backward transformation from Rlm to spherical coordinates
        mdarray<double,2> rlm_backward_;

        /// forward transformation from spherical coordinates to Rlm
        mdarray<double,2> rlm_forward_;

    public:
        
        SHT()
        {

        }

        void set_lmax(int lmax__)
        {
            lmax_ = lmax__;
            lmmax_ = (lmax_ + 1) * (lmax_ + 1);

            num_points_ = Lebedev_Laikov_npoint(2 * lmax_);

            std::vector<double> x(num_points_);
            std::vector<double> y(num_points_);
            std::vector<double> z(num_points_);

            coord_.set_dimensions(3, num_points_);
            coord_.allocate();

            w_.resize(num_points_);

            Lebedev_Laikov_sphere(num_points_, &x[0], &y[0], &z[0], &w_[0]);

            ylm_backward_.set_dimensions(lmmax_, num_points_);
            ylm_backward_.allocate();

            ylm_forward_.set_dimensions(num_points_, lmmax_);
            ylm_forward_.allocate();

            rlm_backward_.set_dimensions(lmmax_, num_points_);
            rlm_backward_.allocate();

            rlm_forward_.set_dimensions(num_points_, lmmax_);
            rlm_forward_.allocate();

            for (int itp = 0; itp < num_points_; itp++)
            {
                coord_(0, itp) = x[itp];
                coord_(1, itp) = y[itp];
                coord_(2, itp) = z[itp];
                
                double vs[3];

                spherical_coordinates(&coord_(0, itp), vs);

                spherical_harmonics(lmax_, vs[1], vs[2], &ylm_backward_(0, itp));
                spherical_harmonics(lmax_, vs[1], vs[2], &rlm_backward_(0, itp));

                for (int lm = 0; lm < lmmax_; lm++)
                {
                    ylm_forward_(itp, lm) = conj(ylm_backward_(lm, itp)) * w_[itp] * fourpi;
                    rlm_forward_(itp, lm) = rlm_backward_(lm, itp) * w_[itp] * fourpi;
                }
            }

            double dr = 0.0;
            double dy = 0.0;

            for (int lm = 0; lm < lmmax_; lm++)
            {
                for (int lm1 = 0; lm1 < lmmax_; lm1++)
                {
                    double t = 0.0;
                    complex16 zt(0.0, 0.0);
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
                    dr += fabs(t);
                    dy += abs(zt);
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
                error(__FILE__, __LINE__, s);
            }

            std::vector<double> flm(lmmax_);
            std::vector<double> ftp(num_points_);
            for (int lm = 0; lm < lmmax_; lm++)
            {
                memset(&flm[0], 0, lmmax_ * sizeof(double));
                flm[lm] = 1.0;
                rlm_backward_transform(&flm[0], lmmax_, 1, &ftp[0]);
                rlm_forward_transform(&ftp[0], lmmax_, 1, &flm[0]);
                flm[lm] -= 1.0;

                double t = 0.0;
                for (int lm1 = 0; lm1 < lmmax_; lm1++)
                    t += fabs(flm[lm1]);

                t /= lmmax_;

                if (t > 1e-15) 
                {
                    std::stringstream s;
                    s << "test of backward / forward real SHT failed" << std::endl
                      << "  total error " << t;
                    error(__FILE__, __LINE__, s);
                }
            }
        }

        /*!
            \brief Transform from Rlm to spherical coordinates 
        */
        void rlm_backward_transform(double* flm, int lmmax, int ncol, double* ftp)
        {
            assert(lmmax <= lmmax_);

            gemm<cpu>(1, 0, num_points_, ncol, lmmax, 1.0, &rlm_backward_(0, 0), lmmax_, flm, lmmax, 0.0, ftp, num_points_);
        }

   
        /*
            \brief Transform from spherical coordinates to Rlm
        */
        void rlm_forward_transform(double *ftp, int lmmax, int ncol, double* flm)
        {
            assert(lmmax <= lmmax_);
            
            gemm<cpu>(1, 0, lmmax, ncol, num_points_, 1.0, &rlm_forward_(0, 0), num_points_, ftp, num_points_, 0.0, flm, lmmax);
        }

        /*!
            \brief Transform Cartesian coordinates [x,y,z] to spherical coordinates [r,theta,phi]
        */
        static inline void spherical_coordinates(double* vc, double* vs)
        {
            double eps = 1e-12;
        
            vs[0] = vector_length(vc);
        
            if (vs[0] <= eps)
            {
                vs[1] = 0.0;
                vs[2] = 0.0;
            } 
            else
            {
                vs[1] = acos(vc[2] / vs[0]); // theta = cos^{-1}(z/r)
        
                if (fabs(vc[0]) > eps || fabs(vc[1]) > eps)
                {
                    vs[2] = atan2(vc[1], vc[0]); // phi = tan^{-1}(y/x)
                    if (vs[2] < 0.0) vs[2] += twopi;
                }
                else
                    vs[2] = 0.0;
            }
        }

        /*!
            \brief Generate complex spherical harmonics Ylm
        */
        static inline void spherical_harmonics(int lmax, double theta, double phi, complex16* ylm)
        {
            double x = cos(theta);
            std::vector<double> result_array(lmax + 1);
        
            for (int m = 0; m <= lmax; m++)
            {
                complex16 z = exp(complex16(0.0, m * phi)); 
                
                gsl_sf_legendre_sphPlm_array(lmax, m, x, &result_array[0]);
        
                for (int l = m; l <= lmax; l++)
                {
                    ylm[lm_by_l_m(l, m)] = result_array[l - m] * z;
                    if (m % 2) 
                        ylm[lm_by_l_m(l, -m)] = -conj(ylm[lm_by_l_m(l, m)]);
                    else
                        ylm[lm_by_l_m(l, -m)] = conj(ylm[lm_by_l_m(l, m)]);        
                }
            }
        }
        
        /*!
            \brief Generate real spherical harmonics Rlm

            Mathematica code:
            \verbatim
            R[l_, m_, th_, ph_] := 
             If[m > 0, Sqrt[2]*ComplexExpand[Re[SphericalHarmonicY[l, m, th, ph]]], 
             If[m < 0, Sqrt[2]*ComplexExpand[Im[SphericalHarmonicY[l, m, th, ph]]], 
             If[m == 0, ComplexExpand[Re[SphericalHarmonicY[l, 0, th, ph]]]]]]
            \endverbatim
        */
        static inline void spherical_harmonics(int lmax, double theta, double phi, double* rlm)
        {
            int lmmax = (lmax + 1) * (lmax + 1);
            std::vector<complex16> ylm(lmmax);
            spherical_harmonics(lmax, theta, phi, &ylm[0]);
            
            double t = sqrt(2.0);
            
            rlm[0] = y00;
        
            for (int l = 1; l <= lmax; l++)
            {
                for (int m = -l; m < 0; m++) 
                    rlm[lm_by_l_m(l, m)] = t * imag(ylm[lm_by_l_m(l, m)]);
                
                rlm[lm_by_l_m(l, 0)] = real(ylm[lm_by_l_m(l, 0)]);
                 
                for (int m = 1; m <= l; m++) 
                    rlm[lm_by_l_m(l, m)] = t * real(ylm[lm_by_l_m(l, m)]);
            }
        }
                        
        /*!
            \brief Compute element of the transformation matrix from complex to real spherical harmonics. 
        
            Real spherical harmonic can be written as a linear combination of complex harmonics:

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
        static inline complex16 ylm_dot_rlm(int l, int m1, int m2)
        {
            assert(l >= 0 && abs(m1) <= l && abs(m2) <= l);
        
            if (abs(m1) != abs(m2)) return complex16(0.0, 0.0);
        
            if (m1 == 0) return complex16(1.0, 0.0);
        
            if (m1 < 0)
            {
                if (m2 < 0) return -zi / sqrt(2.0);
                else return pow(-1.0, m2) / sqrt(2.0);
            }
            else
            {
                if (m2 < 0) return pow(-1.0, m1) * zi / sqrt(2.0);
                else return complex16(1.0 / sqrt(2.0), 0.0);
            }
        }
        
        static inline void convert_frlm_to_fylm(int lmax, double* frlm, complex16* fylm)
        {
            int lmmax = (lmax + 1) * (lmax + 1);
        
            memset(fylm, 0, lmmax * sizeof(complex16));
        
            int lm = 0;
            for (int l = 0; l <= lmax; l++)
                for (int m = -l; m <= l; m++)
                {
                    if (m == 0) fylm[lm] = frlm[lm];
                    else fylm[lm] = ylm_dot_rlm(l, m, m) * frlm[lm] + ylm_dot_rlm(l, m, -m) * frlm[lm_by_l_m(l, -m)];
                    lm++;
                }
        }
        
        static inline void convert_fylm_to_frlm(int lmax, complex16* fylm, double* frlm)
        {
            int lmmax = (lmax + 1) * (lmax + 1);
        
            memset(frlm, 0, lmmax * sizeof(double));
        
            int lm = 0;
            for (int l = 0; l <= lmax; l++)
                for (int m = -l; m <= l; m++)
                {
                    if (m == 0) frlm[lm] = real(fylm[lm]);
                    else frlm[lm] = real(conj(ylm_dot_rlm(l, m, m)) * fylm[lm] + conj(ylm_dot_rlm(l, -m, m)) * fylm[lm_by_l_m(l, -m)]);
                    lm++;
                }
        }

        inline int num_points()
        {
            return num_points_;
        }

        static inline double gaunt(int l1, int l2, int l3, int m1, int m2, int m3)
        {
            assert(l1 >= 0);
            assert(l2 >= 0);
            assert(l3 >= 0);
            assert(m1 >= -l1 && m1 <= l1);
            assert(m2 >= -l2 && m2 <= l2);
            assert(m3 >= -l3 && m3 <= l3);

            return pow(-1.0, abs(m1)) * sqrt(double(2 * l1 + 1) * double(2 * l2 + 1) * double(2 * l3 + 1) / fourpi) * 
                   gsl_sf_coupling_3j(2 * l1, 2 * l2, 2 * l3, 0, 0, 0) *
                   gsl_sf_coupling_3j(2 * l1, 2 * l2, 2 * l3, -2 * m1, 2 * m2, 2 * m3);
        }

        static inline complex16 complex_gaunt(int l1, int l2, int l3, int m1, int m2, int m3)
        {
            assert(l1 >= 0);
            assert(l2 >= 0);
            assert(l3 >= 0);
            assert(m1 >= -l1 && m1 <= l1);
            assert(m2 >= -l2 && m2 <= l2);
            assert(m3 >= -l3 && m3 <= l3);

            if (m2 == 0) return complex16(gaunt(l1, l2, l3, m1, m2, m3), 0.0);
            else 
            {
                return ylm_dot_rlm(l2, m2, m2) *  gaunt(l1, l2, l3, m1, m2, m3) +  ylm_dot_rlm(l2, -m2, m2) *  gaunt(l1, l2, l3, m1, -m2, m3);
            }
        }
};

};


