/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */
#ifndef _PORT_HIP_ROC_TRANSLATION_
#define _PORT_HIP_ROC_TRANSLATION_

#include <hipblas.h>

namespace {

rocblas_operation_ hipOperationToHCCOperation(hipblasOperation_t op)
{
    switch (op) {
        case HIPBLAS_OP_N:
            return rocblas_operation_none;
        case HIPBLAS_OP_T:
            return rocblas_operation_transpose;
        case HIPBLAS_OP_C:
            return rocblas_operation_conjugate_transpose;
        default:
            throw "Non existent OP";
    }
}

hipblasOperation_t HCCOperationToHIPOperation(rocblas_operation_ op)
{
    switch (op) {
        case rocblas_operation_none:
            return HIPBLAS_OP_N;
        case rocblas_operation_transpose:
            return HIPBLAS_OP_T;
        case rocblas_operation_conjugate_transpose:
            return HIPBLAS_OP_C;
        default:
            throw "Non existent OP";
    }
}

rocblas_fill_ hipFillToHCCFill(hipblasFillMode_t fill)
{
    switch (fill) {
        case HIPBLAS_FILL_MODE_UPPER:
            return rocblas_fill_upper;
        case HIPBLAS_FILL_MODE_LOWER:
            return rocblas_fill_lower;
        case HIPBLAS_FILL_MODE_FULL:
            return rocblas_fill_full;
        default:
            throw "Non existent FILL";
    }
}

hipblasFillMode_t HCCFillToHIPFill(rocblas_fill_ fill)
{
    switch (fill) {
        case rocblas_fill_upper:
            return HIPBLAS_FILL_MODE_UPPER;
        case rocblas_fill_lower:
            return HIPBLAS_FILL_MODE_LOWER;
        case rocblas_fill_full:
            return HIPBLAS_FILL_MODE_FULL;
        default:
            throw "Non existent FILL";
    }
}

rocblas_diagonal_ hipDiagonalToHCCDiagonal(hipblasDiagType_t diagonal)
{
    switch (diagonal) {
        case HIPBLAS_DIAG_NON_UNIT:
            return rocblas_diagonal_non_unit;
        case HIPBLAS_DIAG_UNIT:
            return rocblas_diagonal_unit;
        default:
            throw "Non existent DIAGONAL";
    }
}

hipblasDiagType_t HCCDiagonalToHIPDiagonal(rocblas_diagonal_ diagonal)
{
    switch (diagonal) {
        case rocblas_diagonal_non_unit:
            return HIPBLAS_DIAG_NON_UNIT;
        case rocblas_diagonal_unit:
            return HIPBLAS_DIAG_UNIT;
        default:
            throw "Non existent DIAGONAL";
    }
}

rocblas_side_ hipSideToHCCSide(hipblasSideMode_t side)
{
    switch (side) {
        case HIPBLAS_SIDE_LEFT:
            return rocblas_side_left;
        case HIPBLAS_SIDE_RIGHT:
            return rocblas_side_right;
        case HIPBLAS_SIDE_BOTH:
            return rocblas_side_both;
        default:
            throw "Non existent SIDE";
    }
}

hipblasSideMode_t HCCSideToHIPSide(rocblas_side_ side)
{
    switch (side) {
        case rocblas_side_left:
            return HIPBLAS_SIDE_LEFT;
        case rocblas_side_right:
            return HIPBLAS_SIDE_RIGHT;
        case rocblas_side_both:
            return HIPBLAS_SIDE_BOTH;
        default:
            throw "Non existent SIDE";
    }
}

rocblas_pointer_mode HIPPointerModeToRocblasPointerMode(hipblasPointerMode_t mode)
{
    switch (mode) {
        case HIPBLAS_POINTER_MODE_HOST:
            return rocblas_pointer_mode_host;

        case HIPBLAS_POINTER_MODE_DEVICE:
            return rocblas_pointer_mode_device;

        default:
            throw "Non existent PointerMode";
    }
}

hipblasPointerMode_t RocblasPointerModeToHIPPointerMode(rocblas_pointer_mode mode)
{
    switch (mode) {
        case rocblas_pointer_mode_host:
            return HIPBLAS_POINTER_MODE_HOST;

        case rocblas_pointer_mode_device:
            return HIPBLAS_POINTER_MODE_DEVICE;

        default:
            throw "Non existent PointerMode";
    }
}

rocblas_datatype HIPDatatypeToRocblasDatatype(hipblasDatatype_t type)
{
    switch (type) {
        case HIPBLAS_R_16F:
            return rocblas_datatype_f16_r;

        case HIPBLAS_R_32F:
            return rocblas_datatype_f32_r;

        case HIPBLAS_R_64F:
            return rocblas_datatype_f64_r;

        case HIPBLAS_C_16F:
            return rocblas_datatype_f16_c;

        case HIPBLAS_C_32F:
            return rocblas_datatype_f32_c;

        case HIPBLAS_C_64F:
            return rocblas_datatype_f64_c;

        default:
            throw "Non existant DataType";
    }
}

hipblasDatatype_t RocblasDatatypeToHIPDatatype(rocblas_datatype type)
{
    switch (type) {
        case rocblas_datatype_f16_r:
            return HIPBLAS_R_16F;

        case rocblas_datatype_f32_r:
            return HIPBLAS_R_32F;

        case rocblas_datatype_f64_r:
            return HIPBLAS_R_64F;

        case rocblas_datatype_f16_c:
            return HIPBLAS_C_16F;

        case rocblas_datatype_f32_c:
            return HIPBLAS_C_32F;

        case rocblas_datatype_f64_c:
            return HIPBLAS_C_64F;

        default:
            throw "Non existant DataType";
    }
}

rocblas_gemm_algo HIPGemmAlgoToRocblasGemmAlgo(hipblasGemmAlgo_t algo)
{
    switch (algo) {
        case HIPBLAS_GEMM_DEFAULT:
            return rocblas_gemm_algo_standard;

        default:
            throw "Non existant GemmAlgo";
    }
}

hipblasGemmAlgo_t RocblasGemmAlgoToHIPGemmAlgo(rocblas_gemm_algo algo)
{
    switch (algo) {
        case rocblas_gemm_algo_standard:
            return HIPBLAS_GEMM_DEFAULT;

        default:
            throw "Non existant GemmAlgo";
    }
}

hipblasStatus_t rocBLASStatusToHIPStatus(rocblas_status_ error)
{
    switch (error) {
        case rocblas_status_success:
            return HIPBLAS_STATUS_SUCCESS;
        case rocblas_status_invalid_handle:
            return HIPBLAS_STATUS_NOT_INITIALIZED;
        case rocblas_status_not_implemented:
            return HIPBLAS_STATUS_NOT_SUPPORTED;
        case rocblas_status_invalid_pointer:
            return HIPBLAS_STATUS_INVALID_VALUE;
        case rocblas_status_invalid_size:
            return HIPBLAS_STATUS_INVALID_VALUE;
        case rocblas_status_memory_error:
            return HIPBLAS_STATUS_ALLOC_FAILED;
        case rocblas_status_internal_error:
            return HIPBLAS_STATUS_INTERNAL_ERROR;
        default:
            throw "Unimplemented status";
    }
}

}

#endif
