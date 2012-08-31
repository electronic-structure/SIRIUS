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

/*!
    \brief Compute element of the transformation matrix from complex to real spherical harmonics. 

    Real spherical harmonic can be written as a linear combination of complex harmonics:
    \f[
        R_{\ell m}(\theta, \phi) = \sum_{m'} a^{\ell}_{m' m}Y_{\ell m'}(\theta, \phi)
    \f]
    so 
    \f[
        a^{\ell}_{m' m} = \langle Y_{\ell m'} | R_{\ell m} \rangle
    \f]
    which gives the name for the function.

    Transformation from real to complex spherical harmonics is conjugate transpose.

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
inline complex16 ylm_dot_rlm(int l, int m1, int m2)
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

template <typename T>
inline void convert_frlm_to_fylm(int lmax, T* frlm, complex16* fylm)
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

