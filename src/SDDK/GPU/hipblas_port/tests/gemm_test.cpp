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

inline BLAS_PREFIX(blasStatus_t) call_gemm(BLAS_PREFIX(blasHandle_t) handle, BLAS_PREFIX(blasOperation_t) transa, BLAS_PREFIX(blasOperation_t) transb,
                                      int m, int n, int k, const float* alpha, const float* A, int lda, const float* B,
                                      int ldb, const float* beta, float* C, int ldc)
{
#ifdef __CUDA
    return BLAS_PREFIX(blasSgemm)(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
    return BLAS_PREFIX(blas_port_Sgemm)(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

inline BLAS_PREFIX(blasStatus_t)
    call_gemm(BLAS_PREFIX(blasHandle_t) handle, BLAS_PREFIX(blasOperation_t) transa,
              BLAS_PREFIX(blasOperation_t) transb, int m, int n, int k, const double* alpha, const double* A, int lda,
              const double* B, int ldb, const double* beta, double* C, int ldc)
{
#ifdef __CUDA
    return BLAS_PREFIX(blasDgemm)(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
    return BLAS_PREFIX(blas_port_Dgemm)(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

inline BLAS_PREFIX(blasStatus_t)
    call_gemm(BLAS_PREFIX(blasHandle_t) handle, BLAS_PREFIX(blasOperation_t) transa,
              BLAS_PREFIX(blasOperation_t) transb, int m, int n, int k, const BLAS_PREFIX(FloatComplex) * alpha,
              const BLAS_PREFIX(FloatComplex) * A, int lda, const BLAS_PREFIX(FloatComplex) * B, int ldb,
              const BLAS_PREFIX(FloatComplex) * beta, BLAS_PREFIX(FloatComplex) * C, int ldc)
{
#ifdef __CUDA
    return BLAS_PREFIX(blasCgemm)(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
    return BLAS_PREFIX(blas_port_Cgemm)(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

inline BLAS_PREFIX(blasStatus_t)
    call_gemm(BLAS_PREFIX(blasHandle_t) handle, BLAS_PREFIX(blasOperation_t) transa,
              BLAS_PREFIX(blasOperation_t) transb, int m, int n, int k, const BLAS_PREFIX(DoubleComplex) * alpha,
              const BLAS_PREFIX(DoubleComplex) * A, int lda, const BLAS_PREFIX(DoubleComplex) * B, int ldb,
              const BLAS_PREFIX(DoubleComplex) * beta, BLAS_PREFIX(DoubleComplex) * C, int ldc)
{
#ifdef __CUDA
    return BLAS_PREFIX(blasZgemm)(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
    return BLAS_PREFIX(blas_port_Zgemm)(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

template <typename T>
class GemmRealTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        /*     1 3
         * A = 2 4
         *     * *
         */
        A = {create_real<T>::eval(1), create_real<T>::eval(2), create_real<T>::eval(-10000),
             create_real<T>::eval(3), create_real<T>::eval(4), create_real<T>::eval(-20000)};
        GPU_PREFIX(Malloc)((void**)&A_device, A.size() * sizeof(T));
        GPU_PREFIX(Memcpy)(A_device, A.data(), A.size() * sizeof(T), GPU_PREFIX(MemcpyHostToDevice));

        /*     1 3
         * B = 2 4
         *     * *
         */
        B = {create_real<T>::eval(1), create_real<T>::eval(2), create_real<T>::eval(-10000),
             create_real<T>::eval(3), create_real<T>::eval(4), create_real<T>::eval(-20000)};
        GPU_PREFIX(Malloc)((void**)&B_device, B.size() * sizeof(T));
        GPU_PREFIX(Memcpy)(B_device, B.data(), B.size() * sizeof(T), GPU_PREFIX(MemcpyHostToDevice));

        /*     1 3
         * C = 2 4
         *     * *
         */
        C = {create_real<T>::eval(1), create_real<T>::eval(2), create_real<T>::eval(-10000),
             create_real<T>::eval(3), create_real<T>::eval(4), create_real<T>::eval(-20000)};
        GPU_PREFIX(Malloc)((void**)&C_device, C.size() * sizeof(T));
        GPU_PREFIX(Memcpy)(C_device, C.data(), C.size() * sizeof(T), GPU_PREFIX(MemcpyHostToDevice));

        BLAS_PREFIX(blasCreate)(&handle);

        alpha = create_real<T>::eval(1);
        beta  = create_real<T>::eval(2);

        C_result.resize(C.size());
    }

    void TearDown() override
    {
        GPU_PREFIX(Free)(A_device);
        GPU_PREFIX(Free)(B_device);
        GPU_PREFIX(Free)(C_device);
        BLAS_PREFIX(blasDestroy)(handle);
    }

    std::vector<T> A, B, C, C_result;
    T* A_device;
    T* B_device;
    T* C_device;
    T alpha, beta;
    BLAS_PREFIX(blasHandle_t) handle;
    using value_type = T;
};

typedef Types<float, double, BLAS_PREFIX(FloatComplex), BLAS_PREFIX(DoubleComplex)> GemmRealTypes;

TYPED_TEST_CASE(GemmRealTest, GemmRealTypes);

TYPED_TEST(GemmRealTest, AN_BN)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_gemm(this->handle, GPU_PREFIX_CAPS(BLAS_OP_N), GPU_PREFIX_CAPS(BLAS_OP_N), 2, 2, 2, &(this->alpha),
                       this->A_device, 3, this->B_device, 3, &(this->beta), this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[0]), 9.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[1]), 14.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[3]), 21.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[4]), 30.);
}

TYPED_TEST(GemmRealTest, AT_BN)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_gemm(this->handle, GPU_PREFIX_CAPS(BLAS_OP_T), GPU_PREFIX_CAPS(BLAS_OP_N), 2, 2, 2, &(this->alpha),
                       this->A_device, 3, this->B_device, 3, &(this->beta), this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[0]), 7.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[1]), 15.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[3]), 17.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[4]), 33.);
}

TYPED_TEST(GemmRealTest, AN_BT)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_gemm(this->handle, GPU_PREFIX_CAPS(BLAS_OP_N), GPU_PREFIX_CAPS(BLAS_OP_T), 2, 2, 2, &(this->alpha),
                       this->A_device, 3, this->B_device, 3, &(this->beta), this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[0]), 12.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[1]), 18.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[3]), 20.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[4]), 28.);
}

TYPED_TEST(GemmRealTest, AT_BT)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_gemm(this->handle, GPU_PREFIX_CAPS(BLAS_OP_T), GPU_PREFIX_CAPS(BLAS_OP_T), 2, 2, 2, &(this->alpha),
                       this->A_device, 3, this->B_device, 3, &(this->beta), this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[0]), 9.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[1]), 19.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[3]), 16.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[4]), 30.);
}

TYPED_TEST(GemmRealTest, AC_BC)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_gemm(this->handle, GPU_PREFIX_CAPS(BLAS_OP_C), GPU_PREFIX_CAPS(BLAS_OP_C), 2, 2, 2, &(this->alpha),
                       this->A_device, 3, this->B_device, 3, &(this->beta), this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[0]), 9.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[1]), 19.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[3]), 16.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[4]), 30.);
}

template <typename T>
class GemmComplexTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        /*     1 3
         * A = 2 4
         *     * *
         */
        A = {create_complex<T>::eval(1, 1), create_complex<T>::eval(1, 2), create_complex<T>::eval(1, -10000),
             create_complex<T>::eval(1, 3), create_complex<T>::eval(1, 4), create_complex<T>::eval(1, -20000)};
        GPU_PREFIX(Malloc)((void**)&A_device, A.size() * sizeof(T));
        GPU_PREFIX(Memcpy)(A_device, A.data(), A.size() * sizeof(T), GPU_PREFIX(MemcpyHostToDevice));

        /*     1 3
         * B = 2 4
         *     * *
         */
        B = {create_complex<T>::eval(1, 1), create_complex<T>::eval(1, 2), create_complex<T>::eval(1, -10000),
             create_complex<T>::eval(1, 3), create_complex<T>::eval(1, 4), create_complex<T>::eval(1, -20000)};
        GPU_PREFIX(Malloc)((void**)&B_device, B.size() * sizeof(T));
        GPU_PREFIX(Memcpy)(B_device, B.data(), B.size() * sizeof(T), GPU_PREFIX(MemcpyHostToDevice));

        /*     1 3
         * C = 2 4
         *     * *
         */
        C = {create_complex<T>::eval(1, 1), create_complex<T>::eval(1, 2), create_complex<T>::eval(1, -10000),
             create_complex<T>::eval(1, 3), create_complex<T>::eval(1, 4), create_complex<T>::eval(1, -20000)};
        GPU_PREFIX(Malloc)((void**)&C_device, C.size() * sizeof(T));
        GPU_PREFIX(Memcpy)(C_device, C.data(), C.size() * sizeof(T), GPU_PREFIX(MemcpyHostToDevice));

        BLAS_PREFIX(blasCreate)(&handle);

        alpha = create_complex<T>::eval(1, 0);
        beta  = create_complex<T>::eval(2, 0);

        C_result.resize(C.size());
    }

    void TearDown() override
    {
        GPU_PREFIX(Free)(A_device);
        GPU_PREFIX(Free)(B_device);
        GPU_PREFIX(Free)(C_device);
        BLAS_PREFIX(blasDestroy)(handle);
    }

    std::vector<T> A, B, C, C_result;
    T* A_device;
    T* B_device;
    T* C_device;
    T alpha, beta;
    BLAS_PREFIX(blasHandle_t) handle;
    using value_type = T;
};

typedef Types<BLAS_PREFIX(FloatComplex), BLAS_PREFIX(DoubleComplex)> GemmComplexTypes;

TYPED_TEST_CASE(GemmComplexTest, GemmComplexTypes);

TYPED_TEST(GemmComplexTest, AN_BN)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_gemm(this->handle, GPU_PREFIX_CAPS(BLAS_OP_N), GPU_PREFIX_CAPS(BLAS_OP_N), 2, 2, 2, &(this->alpha),
                       this->A_device, 3, this->B_device, 3, &(this->beta), this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ((this->C_result[0]).x, -3.);
    EXPECT_DOUBLE_EQ((this->C_result[1]).x, -6.);
    EXPECT_DOUBLE_EQ((this->C_result[3]).x, -11.);
    EXPECT_DOUBLE_EQ((this->C_result[4]).x, -18.);

    EXPECT_DOUBLE_EQ((this->C_result[0]).y, 9.);
    EXPECT_DOUBLE_EQ((this->C_result[1]).y, 13.);
    EXPECT_DOUBLE_EQ((this->C_result[3]).y, 17.);
    EXPECT_DOUBLE_EQ((this->C_result[4]).y, 21.);
}

TYPED_TEST(GemmComplexTest, AT_BN)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_gemm(this->handle, GPU_PREFIX_CAPS(BLAS_OP_T), GPU_PREFIX_CAPS(BLAS_OP_N), 2, 2, 2, &(this->alpha),
                       this->A_device, 3, this->B_device, 3, &(this->beta), this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ((this->C_result[0]).x, -1.);
    EXPECT_DOUBLE_EQ((this->C_result[1]).x, -7.);
    EXPECT_DOUBLE_EQ((this->C_result[3]).x, -7.);
    EXPECT_DOUBLE_EQ((this->C_result[4]).x, -21.);

    EXPECT_DOUBLE_EQ((this->C_result[0]).y, 8.);
    EXPECT_DOUBLE_EQ((this->C_result[1]).y, 14.);
    EXPECT_DOUBLE_EQ((this->C_result[3]).y, 16.);
    EXPECT_DOUBLE_EQ((this->C_result[4]).y, 22.);
}

TYPED_TEST(GemmComplexTest, AN_BT)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_gemm(this->handle, GPU_PREFIX_CAPS(BLAS_OP_N), GPU_PREFIX_CAPS(BLAS_OP_T), 2, 2, 2, &(this->alpha),
                       this->A_device, 3, this->B_device, 3, &(this->beta), this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ((this->C_result[0]).x, -6.);
    EXPECT_DOUBLE_EQ((this->C_result[1]).x, -10.);
    EXPECT_DOUBLE_EQ((this->C_result[3]).x, -10.);
    EXPECT_DOUBLE_EQ((this->C_result[4]).x, -16.);

    EXPECT_DOUBLE_EQ((this->C_result[0]).y, 10.);
    EXPECT_DOUBLE_EQ((this->C_result[1]).y, 14.);
    EXPECT_DOUBLE_EQ((this->C_result[3]).y, 16.);
    EXPECT_DOUBLE_EQ((this->C_result[4]).y, 20.);
}

TYPED_TEST(GemmComplexTest, AT_BT)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_gemm(this->handle, GPU_PREFIX_CAPS(BLAS_OP_T), GPU_PREFIX_CAPS(BLAS_OP_T), 2, 2, 2, &(this->alpha),
                       this->A_device, 3, this->B_device, 3, &(this->beta), this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ((this->C_result[0]).x, -3.);
    EXPECT_DOUBLE_EQ((this->C_result[1]).x, -11.);
    EXPECT_DOUBLE_EQ((this->C_result[3]).x, -6.);
    EXPECT_DOUBLE_EQ((this->C_result[4]).x, -18.);

    EXPECT_DOUBLE_EQ((this->C_result[0]).y, 9.);
    EXPECT_DOUBLE_EQ((this->C_result[1]).y, 15.);
    EXPECT_DOUBLE_EQ((this->C_result[3]).y, 15.);
    EXPECT_DOUBLE_EQ((this->C_result[4]).y, 21.);
}

TYPED_TEST(GemmComplexTest, AC_BC)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_gemm(this->handle, GPU_PREFIX_CAPS(BLAS_OP_C), GPU_PREFIX_CAPS(BLAS_OP_C), 2, 2, 2, &(this->alpha),
                       this->A_device, 3, this->B_device, 3, &(this->beta), this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ((this->C_result[0]).x, -3.);
    EXPECT_DOUBLE_EQ((this->C_result[1]).x, -11.);
    EXPECT_DOUBLE_EQ((this->C_result[3]).x, -6.);
    EXPECT_DOUBLE_EQ((this->C_result[4]).x, -18.);

    EXPECT_DOUBLE_EQ((this->C_result[0]).y, -5.);
    EXPECT_DOUBLE_EQ((this->C_result[1]).y, -7.);
    EXPECT_DOUBLE_EQ((this->C_result[3]).y, -3.);
    EXPECT_DOUBLE_EQ((this->C_result[4]).y, -5.);
}

/*
 * Non-squared test
 */

template <typename T>
class GemmRealNonSquaredTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        /*     1 3 5
         * A = 2 4 6
         *     * * *
         */
        A = {create_real<T>::eval(1), create_real<T>::eval(2), create_real<T>::eval(-10000),
             create_real<T>::eval(3), create_real<T>::eval(4), create_real<T>::eval(-10000),
             create_real<T>::eval(5), create_real<T>::eval(6), create_real<T>::eval(-20000)};
        GPU_PREFIX(Malloc)((void**)&A_device, A.size() * sizeof(T));
        GPU_PREFIX(Memcpy)(A_device, A.data(), A.size() * sizeof(T), GPU_PREFIX(MemcpyHostToDevice));

        /*     1 4
         * B = 2 5
         *     3 6
         *     * *
         */
        B = {create_real<T>::eval(1), create_real<T>::eval(2), create_real<T>::eval(3), create_real<T>::eval(-10000),
             create_real<T>::eval(4), create_real<T>::eval(5), create_real<T>::eval(6), create_real<T>::eval(-20000)};
        GPU_PREFIX(Malloc)((void**)&B_device, B.size() * sizeof(T));
        GPU_PREFIX(Memcpy)(B_device, B.data(), B.size() * sizeof(T), GPU_PREFIX(MemcpyHostToDevice));

        /*     * * *
         * C = * * *
         *     * * *
         */
        C = {create_real<T>::eval(1), create_real<T>::eval(2), create_real<T>::eval(-10000),
             create_real<T>::eval(3), create_real<T>::eval(4), create_real<T>::eval(-20000),
             create_real<T>::eval(3), create_real<T>::eval(4), create_real<T>::eval(-20000)};
        GPU_PREFIX(Malloc)((void**)&C_device, C.size() * sizeof(T));
        GPU_PREFIX(Memcpy)(C_device, C.data(), C.size() * sizeof(T), GPU_PREFIX(MemcpyHostToDevice));

        BLAS_PREFIX(blasCreate)(&handle);

        alpha = create_real<T>::eval(1);
        beta  = create_real<T>::eval(0);

        C_result.resize(C.size());
    }

    void TearDown() override
    {
        GPU_PREFIX(Free)(A_device);
        GPU_PREFIX(Free)(B_device);
        GPU_PREFIX(Free)(C_device);
        BLAS_PREFIX(blasDestroy)(handle);
    }

    std::vector<T> A, B, C, C_result;
    T* A_device;
    T* B_device;
    T* C_device;
    T alpha, beta;
    BLAS_PREFIX(blasHandle_t) handle;
    using value_type = T;
};

TYPED_TEST_CASE(GemmRealNonSquaredTest, GemmRealTypes);

TYPED_TEST(GemmRealNonSquaredTest, AN_BN)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_gemm(this->handle, GPU_PREFIX_CAPS(BLAS_OP_N), GPU_PREFIX_CAPS(BLAS_OP_N), 2, 2, 3, &(this->alpha),
                       this->A_device, 3, this->B_device, 4, &(this->beta), this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[0]), 22.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[1]), 28.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[3]), 49.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[4]), 64.);
}

TYPED_TEST(GemmRealNonSquaredTest, AT_BT)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_gemm(this->handle, GPU_PREFIX_CAPS(BLAS_OP_T), GPU_PREFIX_CAPS(BLAS_OP_T), 3, 3, 2, &(this->alpha),
                       this->A_device, 3, this->B_device, 4, &(this->beta), this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[0]), 9.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[1]), 19.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[2]), 29.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[3]), 12.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[4]), 26.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[5]), 40.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[6]), 15.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[7]), 33.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[8]), 51.);
}

/*
 * Non-squared test 2
 */

template <typename T>
class GemmRealNonSquaredTest2 : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        /*     1 3 5
         * A = 2 4 6
         *     * * *
         */
        A = {create_real<T>::eval(1), create_real<T>::eval(2), create_real<T>::eval(-10000),
             create_real<T>::eval(3), create_real<T>::eval(4), create_real<T>::eval(-10000),
             create_real<T>::eval(5), create_real<T>::eval(6), create_real<T>::eval(-20000)};
        GPU_PREFIX(Malloc)((void**)&A_device, A.size() * sizeof(T));
        GPU_PREFIX(Memcpy)(A_device, A.data(), A.size() * sizeof(T), GPU_PREFIX(MemcpyHostToDevice));

        /*     1 3 5
         * B = 2 4 6
         */
        B = {create_real<T>::eval(1), create_real<T>::eval(2), create_real<T>::eval(3),
             create_real<T>::eval(4), create_real<T>::eval(5), create_real<T>::eval(6)};
        GPU_PREFIX(Malloc)((void**)&B_device, B.size() * sizeof(T));
        GPU_PREFIX(Memcpy)(B_device, B.data(), B.size() * sizeof(T), GPU_PREFIX(MemcpyHostToDevice));

        /*     * * *
         * C = * * *
         *     * * *
         */
        C.resize(9, create_real<T>::eval(-1000));
        GPU_PREFIX(Malloc)((void**)&C_device, C.size() * sizeof(T));
        GPU_PREFIX(Memcpy)(C_device, C.data(), C.size() * sizeof(T), GPU_PREFIX(MemcpyHostToDevice));

        BLAS_PREFIX(blasCreate)(&handle);

        alpha = create_real<T>::eval(1);
        beta  = create_real<T>::eval(0);

        C_result.resize(C.size());
    }

    void TearDown() override
    {
        GPU_PREFIX(Free)(A_device);
        GPU_PREFIX(Free)(B_device);
        GPU_PREFIX(Free)(C_device);
        BLAS_PREFIX(blasDestroy)(handle);
    }

    std::vector<T> A, B, C, C_result;
    T* A_device;
    T* B_device;
    T* C_device;
    T alpha, beta;
    BLAS_PREFIX(blasHandle_t) handle;
    using value_type = T;
};

TYPED_TEST_CASE(GemmRealNonSquaredTest2, GemmRealTypes);

TYPED_TEST(GemmRealNonSquaredTest2, AN_BT)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_gemm(this->handle, GPU_PREFIX_CAPS(BLAS_OP_N), GPU_PREFIX_CAPS(BLAS_OP_T), 2, 2, 3, &(this->alpha),
                       this->A_device, 3, this->B_device, 2, &(this->beta), this->C_device, 2);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[0]), 35.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[1]), 44.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[2]), 44.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[3]), 56.);
}

TYPED_TEST(GemmRealNonSquaredTest2, AT_BN)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_gemm(this->handle, GPU_PREFIX_CAPS(BLAS_OP_T), GPU_PREFIX_CAPS(BLAS_OP_N), 3, 3, 2, &(this->alpha),
                       this->A_device, 3, this->B_device, 2, &(this->beta), this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[0]), 5.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[1]), 11.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[2]), 17.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[3]), 11.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[4]), 25.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[5]), 39.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[6]), 17.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[7]), 39.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[8]), 61.);
}

template <typename T>
class GemmSiriusTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        /*
         * B * A
         */
        m      = 412;
        n      = 15;
        k      = 15;
        lda    = m;
        ldb    = k;
        ldc    = m;
        size_A = k * lda;
        size_B = n * ldb;
        size_C = n * ldc;

        A.resize(size_A, create_real<T>::eval(-1000));
        B.resize(size_B, create_real<T>::eval(-1000));
        C.resize(size_C, create_real<T>::eval(-1000));

        for (int i = 0; i < size_A; ++i) {
            A[i] = create_real<T>::eval(i + 1);
        }

        for (int i = 0; i < size_B; ++i) {
            B[i] = create_real<T>::eval(i + 1);
        }
        GPU_PREFIX(Malloc)((void**)&A_device, A.size() * sizeof(T));
        GPU_PREFIX(Memcpy)(A_device, A.data(), A.size() * sizeof(T), GPU_PREFIX(MemcpyHostToDevice));

        GPU_PREFIX(Malloc)((void**)&B_device, B.size() * sizeof(T));
        GPU_PREFIX(Memcpy)(B_device, B.data(), B.size() * sizeof(T), GPU_PREFIX(MemcpyHostToDevice));

        GPU_PREFIX(Malloc)((void**)&C_device, C.size() * sizeof(T));
        GPU_PREFIX(Memcpy)(C_device, C.data(), C.size() * sizeof(T), GPU_PREFIX(MemcpyHostToDevice));

        BLAS_PREFIX(blasCreate)(&handle);

        alpha = create_real<T>::eval(1);
        beta  = create_real<T>::eval(0);

        C_result.resize(C.size());
    }

    void TearDown() override
    {
        GPU_PREFIX(Free)(A_device);
        GPU_PREFIX(Free)(B_device);
        GPU_PREFIX(Free)(C_device);
        BLAS_PREFIX(blasDestroy)(handle);
    }

    int m, n, k, ldb, lda, ldc, size_B, size_A, size_C;
    std::vector<T> A, B, C, C_result;
    T* A_device;
    T* B_device;
    T* C_device;
    T alpha;
    T beta;
    BLAS_PREFIX(blasHandle_t) handle;
    using value_type = T;
};

TYPED_TEST_CASE(GemmSiriusTest, GemmRealTypes);

TYPED_TEST(GemmSiriusTest, SIRIUS_N_N)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_gemm(this->handle, GPU_PREFIX_CAPS(BLAS_OP_N), GPU_PREFIX_CAPS(BLAS_OP_N), this->m, this->n, this->k,
                       &(this->alpha), this->A_device, this->lda, this->B_device, this->ldb, &(this->beta),
                       this->C_device, this->ldc);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[0]), 461560.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[411]), 510880.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[414]), 1111375.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[825]), 1760380.);
}

