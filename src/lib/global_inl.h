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

