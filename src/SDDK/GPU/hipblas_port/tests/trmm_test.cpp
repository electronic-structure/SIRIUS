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

template <typename T>
struct create_real
{
    template <typename U>
    static inline T eval(const U& val)
    {
        return T(val);
    }
};

template <>
struct create_real<BLAS_PREFIX(FloatComplex)>
{
    template <typename U>
    static inline BLAS_PREFIX(FloatComplex) eval(const U& val)
    {
        BLAS_PREFIX(FloatComplex) c;
        c.x = val;
        c.y = 0;
        return c;
    }
};

template <>
struct create_real<BLAS_PREFIX(DoubleComplex)>
{
    template <typename U>
    static inline BLAS_PREFIX(DoubleComplex) eval(const U& val)
    {
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
template <typename T>
inline double get_real_double(const T& val)
{
    return double(val);
}

template <>
inline double get_real_double<BLAS_PREFIX(FloatComplex)>(const BLAS_PREFIX(FloatComplex) & val)
{
    return double(val.x);
}

template <>
inline double get_real_double<BLAS_PREFIX(DoubleComplex)>(const BLAS_PREFIX(DoubleComplex) & val)
{
    return double(val.x);
}

inline BLAS_PREFIX(blasStatus_t)
    call_trmm(BLAS_PREFIX(blasHandle_t) handle, BLAS_PREFIX(blasSideMode_t) side, BLAS_PREFIX(blasFillMode_t) uplo,
              BLAS_PREFIX(blasOperation_t) trans, BLAS_PREFIX(blasDiagType_t) diag, int m, int n, const float* alpha,
              const float* A, int lda, const float* B, int ldb, float* C, int ldc)
{
#ifdef __CUDA
    return BLAS_PREFIX(blasStrmm)(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
#else
    return BLAS_PREFIX(blas_port_Strmm)(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
#endif
}

inline BLAS_PREFIX(blasStatus_t)
    call_trmm(BLAS_PREFIX(blasHandle_t) handle, BLAS_PREFIX(blasSideMode_t) side, BLAS_PREFIX(blasFillMode_t) uplo,
              BLAS_PREFIX(blasOperation_t) trans, BLAS_PREFIX(blasDiagType_t) diag, int m, int n, const double* alpha,
              const double* A, int lda, const double* B, int ldb, double* C, int ldc)
{
#ifdef __CUDA
    return BLAS_PREFIX(blasDtrmm)(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
#else
    return BLAS_PREFIX(blas_port_Dtrmm)(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
#endif
}

inline BLAS_PREFIX(blasStatus_t)
    call_trmm(BLAS_PREFIX(blasHandle_t) handle, BLAS_PREFIX(blasSideMode_t) side, BLAS_PREFIX(blasFillMode_t) uplo,
              BLAS_PREFIX(blasOperation_t) trans, BLAS_PREFIX(blasDiagType_t) diag, int m, int n,
              const BLAS_PREFIX(FloatComplex) * alpha, const BLAS_PREFIX(FloatComplex) * A, int lda,
              const BLAS_PREFIX(FloatComplex) * B, int ldb, BLAS_PREFIX(FloatComplex) * C, int ldc)
{
#ifdef __CUDA
    return BLAS_PREFIX(blasCtrmm)(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
#else
    return BLAS_PREFIX(blas_port_Ctrmm)(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
#endif
}

inline BLAS_PREFIX(blasStatus_t)
    call_trmm(BLAS_PREFIX(blasHandle_t) handle, BLAS_PREFIX(blasSideMode_t) side, BLAS_PREFIX(blasFillMode_t) uplo,
              BLAS_PREFIX(blasOperation_t) trans, BLAS_PREFIX(blasDiagType_t) diag, int m, int n,
              const BLAS_PREFIX(DoubleComplex) * alpha, const BLAS_PREFIX(DoubleComplex) * A, int lda,
              const BLAS_PREFIX(DoubleComplex) * B, int ldb, BLAS_PREFIX(DoubleComplex) * C, int ldc)
{
#ifdef __CUDA
    return BLAS_PREFIX(blasZtrmm)(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
#else
    return BLAS_PREFIX(blas_port_Ztrmm)(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
#endif
}

template <typename T>
class TrmmRealTest : public ::testing::Test
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
    T alpha;
    BLAS_PREFIX(blasHandle_t) handle;
    using value_type = T;
};

// typedef Types<float> TrmmRealTypes;
typedef Types<float, double, BLAS_PREFIX(FloatComplex), BLAS_PREFIX(DoubleComplex)> TrmmRealTypes;

TYPED_TEST_CASE(TrmmRealTest, TrmmRealTypes);

TYPED_TEST(TrmmRealTest, LEFT_FULL_NU_N)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_trmm(this->handle, GPU_PREFIX_CAPS(BLAS_SIDE_LEFT), GPU_PREFIX_CAPS(BLAS_FILL_MODE_FULL),
                       GPU_PREFIX_CAPS(BLAS_OP_N), GPU_PREFIX_CAPS(BLAS_DIAG_NON_UNIT), 2, 2, &(this->alpha),
                       this->A_device, 3, this->B_device, 3, this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[0]), 7.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[1]), 10.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[3]), 15.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[4]), 22.);
}

TYPED_TEST(TrmmRealTest, LEFT_LOWER_NU_N)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_trmm(this->handle, GPU_PREFIX_CAPS(BLAS_SIDE_LEFT), GPU_PREFIX_CAPS(BLAS_FILL_MODE_LOWER),
                       GPU_PREFIX_CAPS(BLAS_OP_N), GPU_PREFIX_CAPS(BLAS_DIAG_NON_UNIT), 2, 2, &(this->alpha),
                       this->A_device, 3, this->B_device, 3, this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[0]), 1.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[1]), 10.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[3]), 3.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[4]), 22.);
}

TYPED_TEST(TrmmRealTest, LEFT_UPPER_NU_N)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_trmm(this->handle, GPU_PREFIX_CAPS(BLAS_SIDE_LEFT), GPU_PREFIX_CAPS(BLAS_FILL_MODE_UPPER),
                       GPU_PREFIX_CAPS(BLAS_OP_N), GPU_PREFIX_CAPS(BLAS_DIAG_NON_UNIT), 2, 2, &(this->alpha),
                       this->A_device, 3, this->B_device, 3, this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[0]), 7.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[1]), 8.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[3]), 15.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[4]), 16.);
}

TYPED_TEST(TrmmRealTest, LEFT_UPPER_U_N)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_trmm(this->handle, GPU_PREFIX_CAPS(BLAS_SIDE_LEFT), GPU_PREFIX_CAPS(BLAS_FILL_MODE_UPPER),
                       GPU_PREFIX_CAPS(BLAS_OP_N), GPU_PREFIX_CAPS(BLAS_DIAG_UNIT), 2, 2, &(this->alpha),
                       this->A_device, 3, this->B_device, 3, this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[0]), 7.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[1]), 2.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[3]), 15.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[4]), 4.);
}

TYPED_TEST(TrmmRealTest, LEFT_LOWER_U_N)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_trmm(this->handle, GPU_PREFIX_CAPS(BLAS_SIDE_LEFT), GPU_PREFIX_CAPS(BLAS_FILL_MODE_LOWER),
                       GPU_PREFIX_CAPS(BLAS_OP_N), GPU_PREFIX_CAPS(BLAS_DIAG_UNIT), 2, 2, &(this->alpha),
                       this->A_device, 3, this->B_device, 3, this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[0]), 1.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[1]), 4.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[3]), 3.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[4]), 10.);
}

TYPED_TEST(TrmmRealTest, LEFT_UPPER_U_T)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_trmm(this->handle, GPU_PREFIX_CAPS(BLAS_SIDE_LEFT), GPU_PREFIX_CAPS(BLAS_FILL_MODE_UPPER),
                       GPU_PREFIX_CAPS(BLAS_OP_T), GPU_PREFIX_CAPS(BLAS_DIAG_UNIT), 2, 2, &(this->alpha),
                       this->A_device, 3, this->B_device, 3, this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[0]), 1.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[1]), 5.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[3]), 3.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[4]), 13.);
}

TYPED_TEST(TrmmRealTest, LEFT_LOWER_U_T)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_trmm(this->handle, GPU_PREFIX_CAPS(BLAS_SIDE_LEFT), GPU_PREFIX_CAPS(BLAS_FILL_MODE_LOWER),
                       GPU_PREFIX_CAPS(BLAS_OP_T), GPU_PREFIX_CAPS(BLAS_DIAG_UNIT), 2, 2, &(this->alpha),
                       this->A_device, 3, this->B_device, 3, this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[0]), 5.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[1]), 2.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[3]), 11.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[4]), 4.);
}

TYPED_TEST(TrmmRealTest, LEFT_UPPER_NU_T)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_trmm(this->handle, GPU_PREFIX_CAPS(BLAS_SIDE_LEFT), GPU_PREFIX_CAPS(BLAS_FILL_MODE_UPPER),
                       GPU_PREFIX_CAPS(BLAS_OP_T), GPU_PREFIX_CAPS(BLAS_DIAG_NON_UNIT), 2, 2, &(this->alpha),
                       this->A_device, 3, this->B_device, 3, this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[0]), 1.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[1]), 11.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[3]), 3.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[4]), 25.);
}

TYPED_TEST(TrmmRealTest, LEFT_LOWER_NU_T)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_trmm(this->handle, GPU_PREFIX_CAPS(BLAS_SIDE_LEFT), GPU_PREFIX_CAPS(BLAS_FILL_MODE_LOWER),
                       GPU_PREFIX_CAPS(BLAS_OP_T), GPU_PREFIX_CAPS(BLAS_DIAG_NON_UNIT), 2, 2, &(this->alpha),
                       this->A_device, 3, this->B_device, 3, this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[0]), 5.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[1]), 8.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[3]), 11.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[4]), 16.);
}

TYPED_TEST(TrmmRealTest, LEFT_FULL_NU_T)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_trmm(this->handle, GPU_PREFIX_CAPS(BLAS_SIDE_LEFT), GPU_PREFIX_CAPS(BLAS_FILL_MODE_FULL),
                       GPU_PREFIX_CAPS(BLAS_OP_T), GPU_PREFIX_CAPS(BLAS_DIAG_UNIT), 2, 2, &(this->alpha),
                       this->A_device, 3, this->B_device, 3, this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[0]), 5.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[1]), 11.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[3]), 11.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[4]), 25.);
}

/*
 * RIGHT SIDE
 */
TYPED_TEST(TrmmRealTest, RIGHT_FULL_NU_N)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_trmm(this->handle, GPU_PREFIX_CAPS(BLAS_SIDE_RIGHT), GPU_PREFIX_CAPS(BLAS_FILL_MODE_FULL),
                       GPU_PREFIX_CAPS(BLAS_OP_N), GPU_PREFIX_CAPS(BLAS_DIAG_NON_UNIT), 2, 2, &(this->alpha),
                       this->A_device, 3, this->B_device, 3, this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[0]), 7.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[1]), 10.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[3]), 15.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[4]), 22.);
}

TYPED_TEST(TrmmRealTest, RIGHT_LOWER_NU_N)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_trmm(this->handle, GPU_PREFIX_CAPS(BLAS_SIDE_RIGHT), GPU_PREFIX_CAPS(BLAS_FILL_MODE_LOWER),
                       GPU_PREFIX_CAPS(BLAS_OP_N), GPU_PREFIX_CAPS(BLAS_DIAG_NON_UNIT), 2, 2, &(this->alpha),
                       this->A_device, 3, this->B_device, 3, this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[0]), 7.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[1]), 10.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[3]), 12.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[4]), 16.);
}

TYPED_TEST(TrmmRealTest, RIGHT_UPPER_NU_N)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_trmm(this->handle, GPU_PREFIX_CAPS(BLAS_SIDE_RIGHT), GPU_PREFIX_CAPS(BLAS_FILL_MODE_UPPER),
                       GPU_PREFIX_CAPS(BLAS_OP_N), GPU_PREFIX_CAPS(BLAS_DIAG_NON_UNIT), 2, 2, &(this->alpha),
                       this->A_device, 3, this->B_device, 3, this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[0]), 1.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[1]), 2.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[3]), 15.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[4]), 22.);
}

TYPED_TEST(TrmmRealTest, RIGHT_UPPER_U_N)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_trmm(this->handle, GPU_PREFIX_CAPS(BLAS_SIDE_RIGHT), GPU_PREFIX_CAPS(BLAS_FILL_MODE_UPPER),
                       GPU_PREFIX_CAPS(BLAS_OP_N), GPU_PREFIX_CAPS(BLAS_DIAG_UNIT), 2, 2, &(this->alpha),
                       this->A_device, 3, this->B_device, 3, this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[0]), 1.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[1]), 2.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[3]), 6.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[4]), 10.);
}

TYPED_TEST(TrmmRealTest, RIGHT_LOWER_U_N)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_trmm(this->handle, GPU_PREFIX_CAPS(BLAS_SIDE_RIGHT), GPU_PREFIX_CAPS(BLAS_FILL_MODE_LOWER),
                       GPU_PREFIX_CAPS(BLAS_OP_N), GPU_PREFIX_CAPS(BLAS_DIAG_UNIT), 2, 2, &(this->alpha),
                       this->A_device, 3, this->B_device, 3, this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[0]), 7.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[1]), 10.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[3]), 3.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[4]), 4.);
}

TYPED_TEST(TrmmRealTest, RIGHT_UPPER_U_T)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_trmm(this->handle, GPU_PREFIX_CAPS(BLAS_SIDE_RIGHT), GPU_PREFIX_CAPS(BLAS_FILL_MODE_UPPER),
                       GPU_PREFIX_CAPS(BLAS_OP_T), GPU_PREFIX_CAPS(BLAS_DIAG_UNIT), 2, 2, &(this->alpha),
                       this->A_device, 3, this->B_device, 3, this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[0]), 10.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[1]), 14.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[3]), 3.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[4]), 4.);
}

TYPED_TEST(TrmmRealTest, RIGHT_LOWER_U_T)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_trmm(this->handle, GPU_PREFIX_CAPS(BLAS_SIDE_RIGHT), GPU_PREFIX_CAPS(BLAS_FILL_MODE_LOWER),
                       GPU_PREFIX_CAPS(BLAS_OP_T), GPU_PREFIX_CAPS(BLAS_DIAG_UNIT), 2, 2, &(this->alpha),
                       this->A_device, 3, this->B_device, 3, this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[0]), 1.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[1]), 2.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[3]), 5.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[4]), 8.);
}

TYPED_TEST(TrmmRealTest, RIGHT_UPPER_NU_T)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_trmm(this->handle, GPU_PREFIX_CAPS(BLAS_SIDE_RIGHT), GPU_PREFIX_CAPS(BLAS_FILL_MODE_UPPER),
                       GPU_PREFIX_CAPS(BLAS_OP_T), GPU_PREFIX_CAPS(BLAS_DIAG_NON_UNIT), 2, 2, &(this->alpha),
                       this->A_device, 3, this->B_device, 3, this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[0]), 10.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[1]), 14.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[3]), 12.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[4]), 16.);
}

TYPED_TEST(TrmmRealTest, RIGHT_LOWER_NU_T)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_trmm(this->handle, GPU_PREFIX_CAPS(BLAS_SIDE_RIGHT), GPU_PREFIX_CAPS(BLAS_FILL_MODE_LOWER),
                       GPU_PREFIX_CAPS(BLAS_OP_T), GPU_PREFIX_CAPS(BLAS_DIAG_NON_UNIT), 2, 2, &(this->alpha),
                       this->A_device, 3, this->B_device, 3, this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[0]), 1.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[1]), 2.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[3]), 14.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[4]), 20.);
}

TYPED_TEST(TrmmRealTest, RIGHT_FULL_NU_T)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_trmm(this->handle, GPU_PREFIX_CAPS(BLAS_SIDE_RIGHT), GPU_PREFIX_CAPS(BLAS_FILL_MODE_FULL),
                       GPU_PREFIX_CAPS(BLAS_OP_T), GPU_PREFIX_CAPS(BLAS_DIAG_UNIT), 2, 2, &(this->alpha),
                       this->A_device, 3, this->B_device, 3, this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[0]), 10.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[1]), 14.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[3]), 14.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[4]), 20.);
}

/**************************
 * Complex only
 **************************/

template <typename T>
class TrmmComplexLeftTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        /*     1 3
         * A = 2 4
         *     * *
         */
        A = {create_complex<T>::eval(1, 1), create_complex<T>::eval(2, 2), create_real<T>::eval(-10000),
             create_complex<T>::eval(3, 3), create_complex<T>::eval(4, 4), create_real<T>::eval(-20000)};
        GPU_PREFIX(Malloc)((void**)&A_device, A.size() * sizeof(T));
        GPU_PREFIX(Memcpy)(A_device, A.data(), A.size() * sizeof(T), GPU_PREFIX(MemcpyHostToDevice));

        /*     1 3 5
         * B = 2 4 6
         *     * * *
         */
        B = {create_complex<T>::eval(1, 1), create_complex<T>::eval(2, 2), create_real<T>::eval(-10000),
             create_complex<T>::eval(3, 3), create_complex<T>::eval(4, 4), create_real<T>::eval(-20000),
             create_complex<T>::eval(5, 5), create_complex<T>::eval(6, 6), create_real<T>::eval(-20000)};
        GPU_PREFIX(Malloc)((void**)&B_device, B.size() * sizeof(T));
        GPU_PREFIX(Memcpy)(B_device, B.data(), B.size() * sizeof(T), GPU_PREFIX(MemcpyHostToDevice));

        /*     * * *
         * C = * * *
         *     * * *
         */
        C = {create_real<T>::eval(-1000), create_real<T>::eval(-1000), create_real<T>::eval(-10000),
             create_real<T>::eval(-1000), create_real<T>::eval(-1000), create_real<T>::eval(-20000),
             create_real<T>::eval(-1000), create_real<T>::eval(-1000), create_real<T>::eval(-20000)};
        GPU_PREFIX(Malloc)((void**)&C_device, C.size() * sizeof(T));
        GPU_PREFIX(Memcpy)(C_device, C.data(), C.size() * sizeof(T), GPU_PREFIX(MemcpyHostToDevice));

        BLAS_PREFIX(blasCreate)(&handle);

        alpha = create_real<T>::eval(1);

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
    T alpha;
    BLAS_PREFIX(blasHandle_t) handle;
    using value_type = T;
};

typedef Types<BLAS_PREFIX(FloatComplex), BLAS_PREFIX(DoubleComplex)> TrmmComplexTypes;

TYPED_TEST_CASE(TrmmComplexLeftTest, TrmmComplexTypes);

TYPED_TEST(TrmmComplexLeftTest, COMPLEX_LEFT_FULL_N)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_trmm(this->handle, GPU_PREFIX_CAPS(BLAS_SIDE_LEFT), GPU_PREFIX_CAPS(BLAS_FILL_MODE_FULL),
                       GPU_PREFIX_CAPS(BLAS_OP_N), GPU_PREFIX_CAPS(BLAS_DIAG_NON_UNIT), 2, 3, &(this->alpha),
                       this->A_device, 3, this->B_device, 3, this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ((this->C_result[0]).x, 0.);
    EXPECT_DOUBLE_EQ((this->C_result[0]).y, 14.);

    EXPECT_DOUBLE_EQ((this->C_result[1]).x, 0.);
    EXPECT_DOUBLE_EQ((this->C_result[1]).y, 20.);

    EXPECT_DOUBLE_EQ((this->C_result[3]).x, 0.);
    EXPECT_DOUBLE_EQ((this->C_result[3]).y, 30.);

    EXPECT_DOUBLE_EQ((this->C_result[4]).x, 0.);
    EXPECT_DOUBLE_EQ((this->C_result[4]).y, 44.);

    EXPECT_DOUBLE_EQ((this->C_result[6]).x, 0.);
    EXPECT_DOUBLE_EQ((this->C_result[6]).y, 46.);

    EXPECT_DOUBLE_EQ((this->C_result[7]).x, 0.);
    EXPECT_DOUBLE_EQ((this->C_result[7]).y, 68.);
}

TYPED_TEST(TrmmComplexLeftTest, COMPLEX_LEFT_FULL_C)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_trmm(this->handle, GPU_PREFIX_CAPS(BLAS_SIDE_LEFT), GPU_PREFIX_CAPS(BLAS_FILL_MODE_FULL),
                       GPU_PREFIX_CAPS(BLAS_OP_C), GPU_PREFIX_CAPS(BLAS_DIAG_NON_UNIT), 2, 3, &(this->alpha),
                       this->A_device, 3, this->B_device, 3, this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ((this->C_result[0]).x, 10.);
    EXPECT_DOUBLE_EQ((this->C_result[0]).y, 0.);

    EXPECT_DOUBLE_EQ((this->C_result[1]).x, 22.);
    EXPECT_DOUBLE_EQ((this->C_result[1]).y, 0.);

    EXPECT_DOUBLE_EQ((this->C_result[3]).x, 22.);
    EXPECT_DOUBLE_EQ((this->C_result[3]).y, 0.);

    EXPECT_DOUBLE_EQ((this->C_result[4]).x, 50.);
    EXPECT_DOUBLE_EQ((this->C_result[4]).y, 0.);

    EXPECT_DOUBLE_EQ((this->C_result[6]).x, 34.);
    EXPECT_DOUBLE_EQ((this->C_result[6]).y, 0.);

    EXPECT_DOUBLE_EQ((this->C_result[7]).x, 78.);
    EXPECT_DOUBLE_EQ((this->C_result[7]).y, 0.);
}

template <typename T>
class TrmmComplexRightTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        /*     1 4 7
         * A = 2 5 8
         *     3 6 9
         *     * * *
         */
        A = {
            create_complex<T>::eval(1, 1), create_complex<T>::eval(2, 2), create_complex<T>::eval(3, 3),
            create_real<T>::eval(-10000),  create_complex<T>::eval(4, 4), create_complex<T>::eval(5, 5),
            create_complex<T>::eval(6, 6), create_real<T>::eval(-20000),  create_complex<T>::eval(7, 7),
            create_complex<T>::eval(8, 8), create_complex<T>::eval(9, 9), create_real<T>::eval(-30000),
        };
        GPU_PREFIX(Malloc)((void**)&A_device, A.size() * sizeof(T));
        GPU_PREFIX(Memcpy)(A_device, A.data(), A.size() * sizeof(T), GPU_PREFIX(MemcpyHostToDevice));

        /*     1 3 5
         * B = 2 4 6
         *     * * *
         */
        B = {create_complex<T>::eval(1, 1), create_complex<T>::eval(2, 2), create_real<T>::eval(-10000),
             create_complex<T>::eval(3, 3), create_complex<T>::eval(4, 4), create_real<T>::eval(-20000),
             create_complex<T>::eval(5, 5), create_complex<T>::eval(6, 6), create_real<T>::eval(-20000)};
        GPU_PREFIX(Malloc)((void**)&B_device, B.size() * sizeof(T));
        GPU_PREFIX(Memcpy)(B_device, B.data(), B.size() * sizeof(T), GPU_PREFIX(MemcpyHostToDevice));

        /*     * * *
         * C = * * *
         *     * * *
         */
        C = {create_real<T>::eval(-1000), create_real<T>::eval(-1000), create_real<T>::eval(-10000),
             create_real<T>::eval(-1000), create_real<T>::eval(-1000), create_real<T>::eval(-20000),
             create_real<T>::eval(-1000), create_real<T>::eval(-1000), create_real<T>::eval(-20000)};
        GPU_PREFIX(Malloc)((void**)&C_device, C.size() * sizeof(T));
        GPU_PREFIX(Memcpy)(C_device, C.data(), C.size() * sizeof(T), GPU_PREFIX(MemcpyHostToDevice));

        BLAS_PREFIX(blasCreate)(&handle);

        alpha = create_real<T>::eval(1);

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
    T alpha;
    BLAS_PREFIX(blasHandle_t) handle;
    using value_type = T;
};

TYPED_TEST_CASE(TrmmComplexRightTest, TrmmComplexTypes);

TYPED_TEST(TrmmComplexRightTest, COMPLEX_RIGHT_FULL_N)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_trmm(this->handle, GPU_PREFIX_CAPS(BLAS_SIDE_RIGHT), GPU_PREFIX_CAPS(BLAS_FILL_MODE_FULL),
                       GPU_PREFIX_CAPS(BLAS_OP_N), GPU_PREFIX_CAPS(BLAS_DIAG_NON_UNIT), 2, 3, &(this->alpha),
                       this->A_device, 4, this->B_device, 3, this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ((this->C_result[0]).x, 0.);
    EXPECT_DOUBLE_EQ((this->C_result[0]).y, 44.);

    EXPECT_DOUBLE_EQ((this->C_result[1]).x, 0.);
    EXPECT_DOUBLE_EQ((this->C_result[1]).y, 56.);

    EXPECT_DOUBLE_EQ((this->C_result[3]).x, 0.);
    EXPECT_DOUBLE_EQ((this->C_result[3]).y, 98.);

    EXPECT_DOUBLE_EQ((this->C_result[4]).x, 0.);
    EXPECT_DOUBLE_EQ((this->C_result[4]).y, 128.);

    EXPECT_DOUBLE_EQ((this->C_result[6]).x, 0.);
    EXPECT_DOUBLE_EQ((this->C_result[6]).y, 152.);

    EXPECT_DOUBLE_EQ((this->C_result[7]).x, 0.);
    EXPECT_DOUBLE_EQ((this->C_result[7]).y, 200.);
}

TYPED_TEST(TrmmComplexRightTest, COMPLEX_RIGHT_FULL_C)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_trmm(this->handle, GPU_PREFIX_CAPS(BLAS_SIDE_RIGHT), GPU_PREFIX_CAPS(BLAS_FILL_MODE_FULL),
                       GPU_PREFIX_CAPS(BLAS_OP_C), GPU_PREFIX_CAPS(BLAS_DIAG_NON_UNIT), 2, 3, &(this->alpha),
                       this->A_device, 4, this->B_device, 3, this->C_device, 3);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ((this->C_result[0]).x, 96.);
    EXPECT_DOUBLE_EQ((this->C_result[0]).y, 0.);

    EXPECT_DOUBLE_EQ((this->C_result[1]).x, 120.);
    EXPECT_DOUBLE_EQ((this->C_result[1]).y, 0.);

    EXPECT_DOUBLE_EQ((this->C_result[3]).x, 114.);
    EXPECT_DOUBLE_EQ((this->C_result[3]).y, 0.);

    EXPECT_DOUBLE_EQ((this->C_result[4]).x, 144.);
    EXPECT_DOUBLE_EQ((this->C_result[4]).y, 0.);

    EXPECT_DOUBLE_EQ((this->C_result[6]).x, 132.);
    EXPECT_DOUBLE_EQ((this->C_result[6]).y, 0.);

    EXPECT_DOUBLE_EQ((this->C_result[7]).x, 168.);
    EXPECT_DOUBLE_EQ((this->C_result[7]).y, 0.);
}

template <typename T>
class TrmmSiriusTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        /*
         * B * A
         */
        m      = 412;
        n      = 15;
        ldb    = m;
        lda    = 60;
        size_B = ldb * n;
        size_A = n * lda;

        A.resize(size_A, create_real<T>::eval(-1000));
        B.resize(size_B, create_real<T>::eval(-1000));
        C.resize(size_B, create_real<T>::eval(-1000));

        {
            int i = 1;
            for (int col = 0; col < n; ++col) {
                for (int row = 0; row < n; ++row) {
                    if (col >= row)
                        A[row + col * lda] = create_real<T>::eval(i);

                    ++i;
                }
            }
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

        C_result.resize(C.size());
    }

    void TearDown() override
    {
        GPU_PREFIX(Free)(A_device);
        GPU_PREFIX(Free)(B_device);
        GPU_PREFIX(Free)(C_device);
        BLAS_PREFIX(blasDestroy)(handle);
    }

    int m, n, ldb, lda, size_B, size_A;
    std::vector<T> A, B, C, C_result;
    T* A_device;
    T* B_device;
    T* C_device;
    T alpha;
    BLAS_PREFIX(blasHandle_t) handle;
    using value_type = T;
};

TYPED_TEST_CASE(TrmmSiriusTest, TrmmRealTypes);

TYPED_TEST(TrmmSiriusTest, LEFT_LOWER_NU_N)
{
    BLAS_PREFIX(blasStatus_t)
    status = call_trmm(this->handle, GPU_PREFIX_CAPS(BLAS_SIDE_RIGHT), GPU_PREFIX_CAPS(BLAS_FILL_MODE_UPPER),
                       GPU_PREFIX_CAPS(BLAS_OP_N), GPU_PREFIX_CAPS(BLAS_DIAG_NON_UNIT), this->m, this->n,
                       &(this->alpha), this->A_device, this->lda, this->B_device, this->ldb, this->C_device, this->ldb);
    ASSERT_EQ(status, GPU_PREFIX_CAPS(BLAS_STATUS_SUCCESS));
    GPU_PREFIX(DeviceSynchronize());
    ASSERT_EQ(GPU_PREFIX(GetLastError)(), GPU_PREFIX(Success));
    GPU_PREFIX(Memcpy)
    (this->C_result.data(), this->C_device, this->C_result.size() * sizeof(typename TestFixture::value_type),
     GPU_PREFIX(MemcpyDeviceToHost));

    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[0]), 1.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[411]), 412.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[412]), 7037.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[413]), 7070.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[414]), 7103.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[824]), 40472.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[825]), 40568.);
    EXPECT_DOUBLE_EQ(get_real_double<typename TestFixture::value_type>(this->C_result[826]), 40664.);
}

