#ifndef __TYPEDEFS_H__
#define __TYPEDEFS_H__

#include <complex>
#include <stdint.h>

typedef std::complex<double> complex16;

typedef double real8;

typedef int32_t int4;

enum spin_block {nm, uu, ud, dd}; // add _t to type names

enum implementation {cpu, gpu};

enum lattice_type {direct, reciprocal};

enum coordinates_type {cartesian, fractional};

//
// wrapper for basis data types
//
template <typename T> class data_type_wrapper; // rename to primitive_type_wrapper

template<> class data_type_wrapper<double>
{
    public:
        typedef std::complex<double> complex_type_;
        typedef double real_type_;
        inline bool real() 
        {
            return true;
        }
        inline complex_type_ zone()
        {
            return complex_type_(1.0, 0.0);
        }
        
        inline complex_type_ zzero()
        {
            return complex_type_(0.0, 0.0);
        }

        inline complex_type_ zi()
        {
            return complex_type_(0.0, 1.0);
        }

        static inline double conjugate(double& v)
        {
            return v;
        }
};

template<> class data_type_wrapper<float>
{
    public:
        typedef std::complex<float> complex_type_;
        typedef float real_type_;
        inline bool real() 
        {
            return true;
        }
};

template<> class data_type_wrapper< std::complex<double> >
{
    public:
        typedef std::complex<double> complex_type_;
        typedef double real_type_;
        
        static inline bool real() 
        {
            return false;
        }

        static inline complex_type_ zi()
        {
            return complex_type_(0.0, 1.0);
        }
};

template<> class data_type_wrapper< std::complex<float> >
{
    public:
        typedef std::complex<float> complex_type_;
        typedef float real_type_;
        inline bool real() 
        {
            return false;
        }
};

#endif // __TYPEDEFS_H__
