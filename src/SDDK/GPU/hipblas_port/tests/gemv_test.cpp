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

inline BLAS_PREFIX(blasStatus_t) call_gemv(BLAS_PREFIX(blasHandle_t) handle, BLAS_PREFIX(blasOperation_t) trans, int m, int n, const float* alpha,
                                   const float* A, int lda, const float* x, int incx, const float* beta, float* y,
                                   int incy) {
#ifdef __CUDA
    return BLAS_PREFIX(blasSgemv)(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
#else
    return BLAS_PREFIX(blas_port_Sgemv)(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
#endif
}

inline BLAS_PREFIX(blasStatus_t) call_gemv(BLAS_PREFIX(blasHandle_t) handle, BLAS_PREFIX(blasOperation_t) trans, int m, int n, const double* alpha,
                                   const double* A, int lda, const double* x, int incx, const double* beta, double* y,
                                   int incy) {
#ifdef __CUDA
    return BLAS_PREFIX(blasDgemv)(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
#else
    return BLAS_PREFIX(blas_port_Dgemv)(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
#endif
}

inline BLAS_PREFIX(blasStatus_t) call_gemv(BLAS_PREFIX(blasHandle_t) handle, BLAS_PREFIX(blasOperation_t) trans, int m, int n,
                               const BLAS_PREFIX(FloatComplex)* alpha, const BLAS_PREFIX(FloatComplex)* A, int lda,
                               const BLAS_PREFIX(FloatComplex)* x, int incx, const BLAS_PREFIX(FloatComplex)* beta, BLAS_PREFIX(FloatComplex)* y,
                               int incy)
{
#ifdef __CUDA
    return BLAS_PREFIX(blasCgemv)(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
#else
    return BLAS_PREFIX(blas_port_Cgemv)(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
#endif
}

inline BLAS_PREFIX(blasStatus_t) call_gemv(BLAS_PREFIX(blasHandle_t) handle, BLAS_PREFIX(blasOperation_t) trans, int m, int n,
                               const BLAS_PREFIX(DoubleComplex)* alpha, const BLAS_PREFIX(DoubleComplex)* A, int lda,
                               const BLAS_PREFIX(DoubleComplex)* x, int incx, const BLAS_PREFIX(DoubleComplex)* beta, BLAS_PREFIX(DoubleComplex)* y,
                               int incy)
{
#ifdef __CUDA
    return BLAS_PREFIX(blasZgemv)(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
#else
    return BLAS_PREFIX(blas_port_Zgemv)(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
#endif
}

template <typename T>
class GemvRealTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        /*     1 4
         * A = 2 5
         *     3 6
         *     * *
         */
        A = {create_real<T>::eval(1),      create_real<T>::eval(2),     create_real<T>::eval(3),
                                  create_real<T>::eval(-10000), create_real<T>::eval(4),     create_real<T>::eval(5),
                                  create_real<T>::eval(6),      create_real<T>::eval(-20000)};
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

        /*     1
         * y = *
         *     1
         *     *
         *     1
         *     *
         */
        y = {create_real<T>::eval(1), create_real<T>::eval(-10000),
                                  create_real<T>::eval(1), create_real<T>::eval(-10000),
                                  create_real<T>::eval(1), create_real<T>::eval(-10000)};
        GPU_PREFIX(Malloc)((void**)&y_device, y.size() * sizeof(T));
        GPU_PREFIX(Memcpy)(y_device, y.data(), y.size() * sizeof(T), GPU_PREFIX(MemcpyHostToDevice));


        BLAS_PREFIX(blasCreate)(&handle);

        alpha = create_real<T>::eval(1);
        beta = create_real<T>::eval(2);

        y_result.resize(y.size());
    }

    void TearDown() override
    {
        GPU_PREFIX(Free)(A_device);
        GPU_PREFIX(Free)(x_device);
        GPU_PREFIX(Free)(y_device);
        BLAS_PREFIX(blasDestroy)(handle);
    }

    std::vector<T> A, x, y, y_result;
    T* A_device;
    T* x_device;
    T* y_device;
    T alpha, beta;
    BLAS_PREFIX(blasHandle_t) handle;
    using value_type = T;
};



typedef Types<float, double, BLAS_PREFIX(FloatComplex), BLAS_PREFIX(DoubleComplex)> GemvValueTypes;

TYPED_TEST_CASE(GemvRealTest, GemvValueTypes);

TYPED_TEST(GemvRealTest, OP_NONE) {
    BLAS_PREFIX(blasStatus_t) status =
        call_gemv(this->handle, GPU_PREFIX_CAPS(BLAS_OP_N), 3, 2, &(this->alpha), this->A_device, 4, this->x_device, 2, &(this->beta), this->y_device, 2);
    EXPECT_TRUE(status == GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(Memcpy)(this->y_result.data(), this->y_device, this->y_result.size() * sizeof(typename TestFixture::value_type), GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->y_result[0]), 11.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->y_result[2]), 14.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->y_result[4]), 17.);
}

TYPED_TEST(GemvRealTest, OP_T) {
    BLAS_PREFIX(blasStatus_t) status =
        call_gemv(this->handle, GPU_PREFIX_CAPS(BLAS_OP_T), 3, 2, &(this->alpha), this->A_device, 4, this->x_device, 2, &(this->beta), this->y_device, 2);
    EXPECT_TRUE(status == GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(Memcpy)(this->y_result.data(), this->y_device, this->y_result.size() * sizeof(typename TestFixture::value_type), GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->y_result[0]), 16.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->y_result[2]), 34.);
}

TYPED_TEST(GemvRealTest, OP_C) {
    BLAS_PREFIX(blasStatus_t) status =
        call_gemv(this->handle, GPU_PREFIX_CAPS(BLAS_OP_C), 3, 2, &(this->alpha), this->A_device, 4, this->x_device, 2, &(this->beta), this->y_device, 2);
    EXPECT_TRUE(status == GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(Memcpy)(this->y_result.data(), this->y_device, this->y_result.size() * sizeof(typename TestFixture::value_type), GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->y_result[0]), 16.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->y_result[2]), 34.);
}


template <typename T>
class GemvComplexTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        /*     1 4
         * A = 2 5
         *     3 6
         *     * *
         */
        A = {create_complex<T>::eval(1, 1), create_complex<T>::eval(1, 2), create_complex<T>::eval(1, 3), create_complex<T>::eval(1, -10000), create_complex<T>::eval(1, 4), create_complex<T>::eval(1, 5), create_complex<T>::eval(1, 6), create_complex<T>::eval(1, -20000)};
        GPU_PREFIX(Malloc)((void**)&A_device, A.size() * sizeof(T));
        GPU_PREFIX(Memcpy)(A_device, A.data(), A.size() * sizeof(T), GPU_PREFIX(MemcpyHostToDevice));

        /*     1
         * x = *
         *     2
         *     *
         *     3
         *     *
         */
        x = {create_complex<T>::eval(1, 1), create_complex<T>::eval(1, -10000), create_complex<T>::eval(1, 2), create_complex<T>::eval(1, -10000), create_complex<T>::eval(1, 3), create_complex<T>::eval(1, -10000)};
        GPU_PREFIX(Malloc)((void**)&x_device, x.size() * sizeof(T));
        GPU_PREFIX(Memcpy)(x_device, x.data(), x.size() * sizeof(T), GPU_PREFIX(MemcpyHostToDevice));

        /*     1
         * y = *
         *     1
         *     *
         *     1
         *     *
         */
        y = {create_complex<T>::eval(1, 1), create_complex<T>::eval(1, -10000), create_complex<T>::eval(1, 1), create_complex<T>::eval(1, -10000), create_complex<T>::eval(1, 1), create_complex<T>::eval(1, -10000)};
        GPU_PREFIX(Malloc)((void**)&y_device, y.size() * sizeof(T));
        GPU_PREFIX(Memcpy)(y_device, y.data(), y.size() * sizeof(T), GPU_PREFIX(MemcpyHostToDevice));

        BLAS_PREFIX(blasCreate)(&handle);

        alpha = create_real<T>::eval(1);
        beta  = create_real<T>::eval(2);

        y_result.resize(y.size());
    }

    void TearDown() override
    {
        GPU_PREFIX(Free)(A_device);
        GPU_PREFIX(Free)(x_device);
        GPU_PREFIX(Free)(y_device);
        BLAS_PREFIX(blasDestroy)(handle);
    }

    std::vector<T> A, x, y, y_result;
    T* A_device;
    T* x_device;
    T* y_device;
    T alpha, beta;
    BLAS_PREFIX(blasHandle_t) handle;
    using value_type = T;
};

typedef Types<BLAS_PREFIX(FloatComplex), BLAS_PREFIX(DoubleComplex)> GemvComplexValueTypes;

TYPED_TEST_CASE(GemvComplexTest, GemvComplexValueTypes);

TYPED_TEST(GemvComplexTest, OP_NONE) {
    BLAS_PREFIX(blasStatus_t) status =
        call_gemv(this->handle, GPU_PREFIX_CAPS(BLAS_OP_N), 3, 2, &(this->alpha), this->A_device, 4, this->x_device, 2, &(this->beta), this->y_device, 2);
    EXPECT_TRUE(status == GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(Memcpy)(this->y_result.data(), this->y_device, this->y_result.size() * sizeof(typename TestFixture::value_type), GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ((this->y_result[0]).y, 10.);
    EXPECT_DOUBLE_EQ((this->y_result[2]).y, 12.);
    EXPECT_DOUBLE_EQ((this->y_result[4]).y, 14.);
}

TYPED_TEST(GemvComplexTest, OP_T) {
    BLAS_PREFIX(blasStatus_t) status =
        call_gemv(this->handle, GPU_PREFIX_CAPS(BLAS_OP_T), 3, 2, &(this->alpha), this->A_device, 4, this->x_device, 2, &(this->beta), this->y_device, 2);
    EXPECT_TRUE(status == GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(Memcpy)(this->y_result.data(), this->y_device, this->y_result.size() * sizeof(typename TestFixture::value_type), GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ((this->y_result[0]).y, 14.);
    EXPECT_DOUBLE_EQ((this->y_result[2]).y, 23.);
}

TYPED_TEST(GemvComplexTest, OP_C) {
    BLAS_PREFIX(blasStatus_t) status =
        call_gemv(this->handle, GPU_PREFIX_CAPS(BLAS_OP_C), 3, 2, &(this->alpha), this->A_device, 4, this->x_device, 2, &(this->beta), this->y_device, 2);
    EXPECT_TRUE(status == GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(Memcpy)(this->y_result.data(), this->y_device, this->y_result.size() * sizeof(typename TestFixture::value_type), GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ((this->y_result[0]).y, 2.);
    EXPECT_DOUBLE_EQ((this->y_result[2]).y, -7.);
}


