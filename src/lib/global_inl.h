inline double vector_length(double* v)
{
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
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

inline int lmmax_by_lmax(int lmax)
{
    return (lmax + 1) * (lmax + 1);
}

inline int lm_by_l_m(int l, int m)
{
    return (l * l + l + m);
}

inline int l_by_lm(int lm)
{
    static std::vector<int> l_values;
    static const int lmax = 50;
    
    if (!l_values.size())
    {
        l_values.resize(lmmax_by_lmax(lmax));
                
        int lm = 0;
        for (int l = 0; l <= lmax; l++)
            for (int m = -l; m <= l; m++, lm++)
                l_values[lm] = l;
    }
    assert(lm < lmmax_by_lmax(lmax));
    
    return l_values[lm];
}

double fermi_dirac_distribution(double e)
{
    double kT = 0.001;
    return (1.0 / (exp(e / kT) + 1.0));
}
