#ifndef __VECTOR3D_H__
#define __VECTOR3D_H__

/// Simple implementation of 3d vector
template <typename T> class vector3d
{
    private:

        T vec_[3];

    public:
        
        /// Construct zero vector
        vector3d()
        {
            vec_[0] = vec_[1] = vec_[2] = 0;
        }

        /// Construct vector with the same values
        vector3d(T v0)
        {
            vec_[0] = vec_[1] = vec_[2] = v0;
        }

        /// Construct arbitrary vector
        vector3d(T x, T y, T z)
        {
            vec_[0] = x;
            vec_[1] = y;
            vec_[2] = z;
        }

        /// Construct vector from pointer
        vector3d(T* ptr)
        {
            for (int i = 0; i < 3; i++) vec_[i] = ptr[i];
        }

        /// Access vector elements
        inline T& operator[](const int i)
        {
            assert(i >= 0 && i <= 2);
            return vec_[i];
        }

        /// Return vector length
        inline double length()
        {
            return sqrt(vec_[0] * vec_[0] + vec_[1] * vec_[1] + vec_[2] * vec_[2]);
        }

        inline vector3d<T> operator+(const vector3d<T>& b)
        {
            vector3d<T> a = *this;
            for (int x = 0; x < 3; x++) a[x] += b.vec_[x];
            return a;
        }

        inline vector3d<T> operator-(const vector3d<T>& b)
        {
            vector3d<T> a = *this;
            for (int x = 0; x < 3; x++) a[x] -= b.vec_[x];
            return a;
        }
};

#endif // __VECTOR3D_H__

