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

inline int lm_by_l_m(int l, int m)
{
    return (l * l + l + m);
}

inline int l_by_lm(int lm)
{
    return int(sqrt(double(lm)));
}

inline int lmmax_by_lmax(int lmax)
{
    return (lmax + 1) * (lmax + 1);
}
