#ifndef _PORT_HELPER_FUNC_H_
#define _PORT_HELPER_FUNC_H_

#include <hip/hip_complex.h>
#include "rocblas-types.h"

namespace {

/*
 * Check if real only and cmp value
 */
template<typename T, typename U>
__host__ __device__ inline bool rb_port_cmp_and_real_only(const T& a, const U& val) { return a == val; }

template<typename T>
__host__ __device__ inline bool rb_port_cmp_and_real_only(const hipDoubleComplex& a, const T& val) { return a.x == val && a.y == 0; }

template<typename T>
__host__ __device__ inline bool rb_port_cmp_and_real_only(const hipFloatComplex& a, const T& val) { return a.x == val && a.y == 0; }

/*
 * Conjugate helper functions
 */
template<rocblas_operation OP, typename T>
struct ConjOp {
    __host__ __device__ static inline T eval(const T& val) { return val; }
};

template<>
struct ConjOp<rocblas_operation_conjugate_transpose, hipDoubleComplex> {
    __host__ __device__ static inline hipDoubleComplex eval(const hipDoubleComplex& val) {
        return hipDoubleComplex(val.x, -val.y);
    }
};

template<>
struct ConjOp<rocblas_operation_conjugate_transpose, hipFloatComplex> {
    __host__ __device__ static inline hipFloatComplex eval(const hipFloatComplex& val) {
        return hipFloatComplex(val.x, -val.y);
    }
};

/*
 * Swap of leading dimension / increment for transposed matrices
 */
template<rocblas_operation OP>
struct MatrixDim {
    __host__ __device__ static inline rocblas_int ld(const rocblas_int& ld, const rocblas_int& inc) { return inc; }
    __host__ __device__ static inline rocblas_int inc(const rocblas_int& ld, const rocblas_int& inc) { return ld; }
};

template<>
struct MatrixDim<rocblas_operation_none> {
    __host__ __device__ static inline rocblas_int ld(const rocblas_int& ld, const rocblas_int& inc) { return ld; }
    __host__ __device__ static inline rocblas_int inc(const rocblas_int& ld, const rocblas_int& inc) { return inc; }
};


}

#endif
