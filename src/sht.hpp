template<> inline void SHT::backward_transform<double>(double* flm, int lmmax, int ncol, double* ftp)
{
    assert(lmmax <= lmmax_);
    blas<cpu>::gemm(1, 0, num_points_, ncol, lmmax, &rlm_backward_(0, 0), lmmax_, flm, lmmax, ftp, num_points_);
}

template<> inline void SHT::backward_transform<double_complex>(double_complex* flm, int lmmax, int ncol, double_complex* ftp)
{
    assert(lmmax <= lmmax_);
    blas<cpu>::gemm(1, 0, num_points_, ncol, lmmax, &ylm_backward_(0, 0), lmmax_, flm, lmmax, ftp, num_points_);
}

template<> inline void SHT::forward_transform<double>(double* ftp, int lmmax, int ncol, double* flm)
{
    assert(lmmax <= lmmax_);
    blas<cpu>::gemm(1, 0, lmmax, ncol, num_points_, &rlm_forward_(0, 0), num_points_, ftp, num_points_, flm, lmmax);
}

template<> inline void SHT::forward_transform<double_complex>(double_complex* ftp, int lmmax, int ncol, double_complex* flm)
{
    assert(lmmax <= lmmax_);
    blas<cpu>::gemm(1, 0, lmmax, ncol, num_points_, &ylm_forward_(0, 0), num_points_, ftp, num_points_, flm, lmmax);
}

/** Specialization for real Gaunt coefficients between three complex spherical harmonics
    \f[ 
        \langle Y_{\ell_1 m_1} | Y_{\ell_2 m_2} | Y_{\ell_3 m_3} \rangle
    \f]
*/
template<> inline double SHT::gaunt<double>(int l1, int l2, int l3, int m1, int m2, int m3)
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

/** Specialization for complex Gaunt coefficients between two complex and one real spherical harmonics
    \f[ 
        \langle Y_{\ell_1 m_1} | R_{\ell_2 m_2} | Y_{\ell_3 m_3} \rangle
    \f]
*/
template<> inline double_complex SHT::gaunt<double_complex>(int l1, int l2, int l3, int m1, int m2, int m3)
{
    assert(l1 >= 0);
    assert(l2 >= 0);
    assert(l3 >= 0);
    assert(m1 >= -l1 && m1 <= l1);
    assert(m2 >= -l2 && m2 <= l2);
    assert(m3 >= -l3 && m3 <= l3);

    if (m2 == 0) 
    {
        return double_complex(gaunt<double>(l1, l2, l3, m1, m2, m3), 0.0);
    }
    else 
    {
        return (ylm_dot_rlm(l2, m2, m2) * gaunt<double>(l1, l2, l3, m1, m2, m3) +  
                ylm_dot_rlm(l2, -m2, m2) * gaunt<double>(l1, l2, l3, m1, -m2, m3));
    }
}

template <int direction, typename T>
Spheric_function<T> SHT::transform(Spheric_function<T>& f)
{
    Spheric_function<T> g;

    switch (direction)
    {
        /* forward transform, f(t, p) -> g(l, m) */
        case 1:
        {
            g = Spheric_function<T>(lmmax(), f.radial_grid());
            break;
        }
        /* backward transform, f(l, m) -> g(t, p) */
        case -1:
        {
            g = Spheric_function<T>(num_points(), f.radial_grid());
            break;
        }
        default:
        {
            error_local(__FILE__, __LINE__, "Wrong direction of transformation");
        }
    }
    transform<direction, T>(f, g);
    return g;
}

template <int direction, typename T>
void SHT::transform(Spheric_function<T>& f, Spheric_function<T>& g)
{
    switch (direction)
    {
        /* forward transform, f(t, p) -> g(l, m) */
        case 1:
        {
            forward_transform(&f(0, 0), lmmax(), f.radial_grid().num_points(), &g(0, 0));
            break;
        }
        /* backward transform, f(l, m) -> g(t, p) */
        case -1:
        {
            backward_transform(&f(0, 0), f.angular_domain_size(), f.radial_grid().num_points(), &g(0, 0));
            break;
        }
        default:
        {
            error_local(__FILE__, __LINE__, "Wrong direction of transformation");
        }
    }
}

