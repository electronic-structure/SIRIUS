#ifndef __TYPEDEFS_H__
#define __TYPEDEFS_H__

#include <complex>
#include <stdint.h>

typedef std::complex<double> complex16;

typedef double real8;

typedef int32_t int4;

enum spin_block {nm, uu, ud, dd};

enum implementation {cpu, gpu};

enum lattice_type {direct, reciprocal};

enum coordinates_type {cartesian, fractional};

/*enum data_type {real_data_type, complex_data_type};

template <typename T> class data_type_wrapper
{
    public:
    
        data_type_wrapper();
        data_type type_;

        inline data_type operator()(void)
        {
            return type_;
        }
};

template<> data_type_wrapper<double>::data_type_wrapper() : type_(real_data_type) {}
template<> data_type_wrapper<complex16>::data_type_wrapper() : type_(complex_data_type) {}
*/

#endif // __TYPEDEFS_H__
