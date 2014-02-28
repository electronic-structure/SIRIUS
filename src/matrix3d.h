#ifndef __MATRIX3D_H__
#define __MATRIX3D_H__

#include <string.h>
#include "typedefs.h"
#include "error_handling.h"

template <typename T>
class matrix3d
{
    private:

        T mtrx_[3][3];

    public:
        
        matrix3d()
        {
            memset(&mtrx_[0][0], 0, 9 * sizeof(T));
        }
        
        matrix3d(T mtrx__[3][3])
        {
            memcpy(&mtrx_[0][0], &mtrx__[0][0], 9 * sizeof(T));
        }

        matrix3d(const matrix3d<T>& src)
        {
            memcpy(&mtrx_[0][0], &src.mtrx_[0][0], 9 * sizeof(T));
        }

        matrix3d<T>& operator=(const matrix3d<T>& rhs)
        {
            if (this != &rhs) memcpy(&this->mtrx_[0][0], &rhs.mtrx_[0][0], 9 * sizeof(T));
            return *this;
        }

        inline T& operator()(const int i, const int j)
        {
            return mtrx_[i][j];
        }

        inline matrix3d<T> operator*(matrix3d<T> b)
        {
            matrix3d<T> c;
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    for (int k = 0; k < 3; k++) c(i, j) += (*this)(i, k) * b(k, j);
                }
            }
            return c;
        }

        inline matrix3d<T> operator*(int p)
        {
            matrix3d<T> c;
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++) c(i, j) = (*this)(i, j) * p;
            }
            return c;
        }

        inline T det()
        {
            return (mtrx_[0][2] * (mtrx_[1][0] * mtrx_[2][1] - mtrx_[1][1] * mtrx_[2][0]) + 
                    mtrx_[0][1] * (mtrx_[1][2] * mtrx_[2][0] - mtrx_[1][0] * mtrx_[2][2]) + 
                    mtrx_[0][0] * (mtrx_[1][1] * mtrx_[2][2] - mtrx_[1][2] * mtrx_[2][1]));
        }

};

template <typename T>
matrix3d<T> transpose(matrix3d<T> src)
{
    matrix3d<T> mtrx;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++) mtrx(i, j) = src(j, i);
    }
    return mtrx;
}

template <typename T>
matrix3d<T> inverse(matrix3d<T> src)
{
    matrix3d<T> mtrx;
    
    T t1 = src.det();
    
    if (type_wrapper<T>::abs(t1) < 1e-10) error_local(__FILE__, __LINE__, "matix is degenerate");
    
    t1 = 1.0 / t1;

    mtrx(0, 0) = t1 * (src(1, 1) * src(2, 2) - src(1, 2) * src(2, 1));
    mtrx(0, 1) = t1 * (src(0, 2) * src(2, 1) - src(0, 1) * src(2, 2));
    mtrx(0, 2) = t1 * (src(0, 1) * src(1, 2) - src(0, 2) * src(1, 1));
    mtrx(1, 0) = t1 * (src(1, 2) * src(2, 0) - src(1, 0) * src(2, 2));
    mtrx(1, 1) = t1 * (src(0, 0) * src(2, 2) - src(0, 2) * src(2, 0));
    mtrx(1, 2) = t1 * (src(0, 2) * src(1, 0) - src(0, 0) * src(1, 2));
    mtrx(2, 0) = t1 * (src(1, 0) * src(2, 1) - src(1, 1) * src(2, 0));
    mtrx(2, 1) = t1 * (src(0, 1) * src(2, 0) - src(0, 0) * src(2, 1));
    mtrx(2, 2) = t1 * (src(0, 0) * src(1, 1) - src(0, 1) * src(1, 0));

    return mtrx;
}
    
    




#endif // __MATRIX3D_H__
