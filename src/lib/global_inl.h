inline double vector_length(double* v)
{
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

inline double vector_scalar_product(double* a, double* b)
{
    return (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
}

template <typename Ta, typename Tb>
inline double scalar_product(Ta* a, Tb* b)
{
    return (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
}

inline int compare_doubles(const void* a, const void* b)
{
    if (*(double*)a > *(double*)b)
        return 1;
    else if (*(double*)a < *(double*)b)
        return -1;
    else
        return 0;
}

inline int lm_by_l_m(int l, int m)
{
    return (l * l + l + m);
}

inline int l_by_lm(int lm)
{
    return int(sqrt(double(lm)));
}

/*!
    \brief Transform Cartesian coordinates [x,y,z] to spherical coordinates [r,theta,phi]
*/
inline void spherical_coordinates(double* vc, double* vs)
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

inline void spherical_harmonics(int lmax, double theta, double phi, complex16* ylm)
{
    double x = cos(theta);
    std::vector<double> result_array(lmax + 1);
    //std::vector<complex16> expimphi(lmax + 1);

    for (int m = 0; m <= lmax; m++)
    {
        complex16 z = exp(complex16(0.0, m * phi)); 
        
        gsl_sf_legendre_sphPlm_array(lmax, m, x, &result_array[0]);

        int i = 0;
        for (int l = m; l <= lmax; l++)
        {
            ylm[lm_by_l_m(l, m)] = result_array[i++] * z;
            if (m % 2) 
                ylm[lm_by_l_m(l, -m)] = -conj(ylm[lm_by_l_m(l, m)]);
            else
                ylm[lm_by_l_m(l, -m)] = conj(ylm[lm_by_l_m(l, m)]);        
        }
    }
}

