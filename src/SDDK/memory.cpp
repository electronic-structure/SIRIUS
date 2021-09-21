/** \file memory.cpp
*
*  \brief Definitions.
*/

#include "memory.hpp"

#if defined(SIRIUS_GPU)
void copy_gpu(float* to_device_ptr__, const double* from_device_ptr__, int n__)
{
    copy_double_to_float_gpu(to_device_ptr__, from_device_ptr__, n__);
}

void copy_gpu(double* to_device_ptr__, const float* from_device_ptr__, int n__)
{
    copy_float_to_double_gpu(to_device_ptr__, from_device_ptr__, n__);
}

void copy_gpu(std::complex<float>* to_device_ptr__, const std::complex<double>* from_device_ptr__, int n__)
{
    copy_double_to_float_complex_gpu(to_device_ptr__, from_device_ptr__, n__);
}

void copy_gpu(std::complex<double>* to_device_ptr__, const std::complex<float>* from_device_ptr__, int n__)
{
    copy_float_to_double_complex_gpu(to_device_ptr__, from_device_ptr__, n__);
}
#endif