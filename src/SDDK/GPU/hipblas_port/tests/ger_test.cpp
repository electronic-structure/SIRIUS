#ifdef __CUDA
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#else
#include <hip/hip_runtime_api.h>
#include <hip/hip_complex.h>
#include <hipblas.h>
#include "hipblas_port.h"
#endif
#include <vector>
#include "gtest/gtest.h"
// #define CATCH_CONFIG_MAIN
// #include "catch.hpp"

using testing::Types;
#ifdef __CUDA
#define GPU_PREFIX(val) cuda##val
#else
#define GPU_PREFIX(val) hip##val
#endif

#ifdef __CUDA
#define BLAS_PREFIX(val) cu##val
#else
#define BLAS_PREFIX(val) hip##val
#endif

#ifdef __CUDA
#define GPU_PREFIX_CAPS(val) CU##val
#else
#define GPU_PREFIX_CAPS(val) HIP##val
#endif

template<typename T>
struct create_real {
    template<typename U>
    static inline T eval(const U& val) {
        return T(val);
    }
};

template<>
struct create_real<BLAS_PREFIX(FloatComplex)> {
    template<typename U>
    static inline BLAS_PREFIX(FloatComplex) eval(const U& val) {
        BLAS_PREFIX(FloatComplex) c;
        c.x = val;
        c.y = 0;
        return c;
    }
};

template<>
struct create_real<BLAS_PREFIX(DoubleComplex)> {
    template<typename U>
    static inline BLAS_PREFIX(DoubleComplex) eval(const U& val) {
        BLAS_PREFIX(DoubleComplex) c;
        c.x = val;
        c.y = 0;
        return c;
    }
};

template <typename T>
struct create_complex
{
    template <typename U1, typename U2>
    static inline T eval(const U1& val1, const U2& val2)
    {
        T c;
        c.x = val1;
        c.y = val2;
        return c;
    }
};

template<typename T>
inline double get_real_double(const T& val) {
    return double(val);
}

template<>
inline double get_real_double<BLAS_PREFIX(FloatComplex)>(const BLAS_PREFIX(FloatComplex)& val) {
    return double(val.x);
}

template<>
inline double get_real_double<BLAS_PREFIX(DoubleComplex)>(const BLAS_PREFIX(DoubleComplex)& val) {
    return double(val.x);
}

inline BLAS_PREFIX(blasStatus_t) call_ger(BLAS_PREFIX(blasHandle_t) handle, int m, int n, const float* alpha, const float* x, int incx,
                                     const float* y, int incy, float* A, int lda)
{

#ifdef __CUDA
    return BLAS_PREFIX(blasSger)(handle, m, n, alpha, x, incx, y, incy, A, lda);
#else
    return BLAS_PREFIX(blas_port_Sger)(handle, m, n, alpha, x, incx, y, incy, A, lda);
#endif
}

inline BLAS_PREFIX(blasStatus_t) call_ger(BLAS_PREFIX(blasHandle_t) handle, int m, int n, const double* alpha, const double* x,
                                     int incx, const double* y, int incy, double* A, int lda)
{

#ifdef __CUDA
    return BLAS_PREFIX(blasDger)(handle, m, n, alpha, x, incx, y, incy, A, lda);
#else
    return BLAS_PREFIX(blas_port_Dger)(handle, m, n, alpha, x, incx, y, incy, A, lda);
#endif
}

inline BLAS_PREFIX(blasStatus_t) call_ger(BLAS_PREFIX(blasHandle_t) handle, int m, int n, const BLAS_PREFIX(FloatComplex)* alpha,
                                     const BLAS_PREFIX(FloatComplex)* x, int incx, const BLAS_PREFIX(FloatComplex)* y, int incy,
                                     BLAS_PREFIX(FloatComplex)* A, int lda)
{

#ifdef __CUDA
    return BLAS_PREFIX(blasCgeru)(handle, m, n, alpha, x, incx, y, incy, A, lda);
#else
    return BLAS_PREFIX(blas_port_Cgeru)(handle, m, n, alpha, x, incx, y, incy, A, lda);
#endif
}

inline BLAS_PREFIX(blasStatus_t) call_ger(BLAS_PREFIX(blasHandle_t) handle, int m, int n, const BLAS_PREFIX(DoubleComplex)* alpha,
                                     const BLAS_PREFIX(DoubleComplex)* x, int incx, const BLAS_PREFIX(DoubleComplex)* y, int incy,
                                     BLAS_PREFIX(DoubleComplex)* A, int lda)
{

#ifdef __CUDA
    return BLAS_PREFIX(blasZgeru)(handle, m, n, alpha, x, incx, y, incy, A, lda);
#else
    return BLAS_PREFIX(blas_port_Zgeru)(handle, m, n, alpha, x, incx, y, incy, A, lda);
#endif
}

template <typename T>
class GerTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        /*     1 4
         * A = 2 5
         *     3 6
         *     * *
         */
        A = {create_real<T>::eval(1), create_real<T>::eval(2), create_real<T>::eval(3), create_real<T>::eval(-10000),
             create_real<T>::eval(4), create_real<T>::eval(5), create_real<T>::eval(6), create_real<T>::eval(-20000)};
        GPU_PREFIX(Malloc)((void**)&A_device, A.size() * sizeof(T));
        GPU_PREFIX(Memcpy)(A_device, A.data(), A.size() * sizeof(T), GPU_PREFIX(MemcpyHostToDevice));

        /*     1
         * x = *
         *     2
         *     *
         *     3
         *     *
         */
        x = {create_real<T>::eval(1), create_real<T>::eval(-10000),
                                  create_real<T>::eval(2), create_real<T>::eval(-10000),
                                  create_real<T>::eval(3), create_real<T>::eval(-10000)};
        GPU_PREFIX(Malloc)((void**)&x_device, x.size() * sizeof(T));
        GPU_PREFIX(Memcpy)(x_device, x.data(), x.size() * sizeof(T), GPU_PREFIX(MemcpyHostToDevice));

        /*     7
         * y = *
         *     8
         *     *
         */
        y = {create_real<T>::eval(7), create_real<T>::eval(-10000),
                                  create_real<T>::eval(8), create_real<T>::eval(-10000)};
        GPU_PREFIX(Malloc)((void**)&y_device, y.size() * sizeof(T));
        GPU_PREFIX(Memcpy)(y_device, y.data(), y.size() * sizeof(T), GPU_PREFIX(MemcpyHostToDevice));


        BLAS_PREFIX(blasCreate)(&handle);

        alpha = create_real<T>::eval(2);

        A_result.resize(A.size());
    }

    void TearDown() override
    {
        GPU_PREFIX(Free)(A_device);
        GPU_PREFIX(Free)(x_device);
        GPU_PREFIX(Free)(y_device);
        BLAS_PREFIX(blasDestroy)(handle);
    }

    std::vector<T> A, x, y, A_result;
    T* A_device;
    T* x_device;
    T* y_device;
    T alpha;
    BLAS_PREFIX(blasHandle_t) handle;
    using value_type = T;
};



typedef Types<float, double, BLAS_PREFIX(FloatComplex), BLAS_PREFIX(DoubleComplex)> GemvValueTypes;

TYPED_TEST_CASE(GerTest, GemvValueTypes);

TYPED_TEST(GerTest, NON_SQUARED) {
    BLAS_PREFIX(blasStatus_t) status =
        call_ger(this->handle, 3, 2, &(this->alpha), this->x_device, 2, this->y_device, 2, this->A_device, 4);
    EXPECT_TRUE(status == GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(Memcpy)(this->A_result.data(), this->A_device, this->A_result.size() * sizeof(typename TestFixture::value_type), GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->A_result[0]), 15.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->A_result[1]), 30.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->A_result[2]), 45.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->A_result[4]), 20.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->A_result[5]), 37.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->A_result[6]), 54.);
}
