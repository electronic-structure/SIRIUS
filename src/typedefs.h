#ifndef __TYPEDEFS_H__
#define __TYPEDEFS_H__

typedef std::complex<double> complex16;

typedef double real8;

typedef int32_t int4;

enum spin_block {nm, uu, ud, dd, du}; // TODO: add _t to type names

enum processing_unit_t {cpu, gpu};

enum lattice_t {direct, reciprocal};

enum coordinates_t {cartesian, fractional};

enum mpi_op_t {op_sum, op_max};

enum linalg_t {lapack, scalapack, elpa, magma};

enum splindex_t {block, block_cyclic};

/// Wrapper for primitive data types
template <typename T> class primitive_type_wrapper;

template<> class primitive_type_wrapper<double>
{
    public:
        typedef std::complex<double> complex_t;
        typedef double real_t;
        
        static hid_t hdf5_type_id()
        {
            return H5T_NATIVE_DOUBLE;
        }
        
        static inline double conjugate(double& v)
        {
            return v;
        }

        static inline std::complex<double> conjugate(std::complex<double>& v)
        {
            return conj(v);
        }
        
        static inline double sift(std::complex<double> v)
        {
            return real(v);
        }
        
        static MPI_Datatype mpi_type_id()
        {
            return MPI_DOUBLE;
        }

        /*static bool is_complex()
        {
            return false;
        }*/
};

template<> class primitive_type_wrapper<float>
{
    public:
        typedef std::complex<float> complex_t;
        typedef float real_t;
};

template<> class primitive_type_wrapper< std::complex<double> >
{
    public:
        typedef std::complex<double> complex_t;
        typedef double real_t;
        
        static inline std::complex<double> conjugate(std::complex<double>& v)
        {
            return conj(v);
        }
        
        static inline std::complex<double> sift(std::complex<double> v)
        {
            return v;
        }
        
        static MPI_Datatype mpi_type_id()
        {
            return MPI_COMPLEX16;
        }

        /*static bool is_complex()
        {
            return true;
        }*/
};

template<> class primitive_type_wrapper< std::complex<float> >
{
    public:
        typedef std::complex<float> complex_t;
        typedef float real_t;
};

template<> class primitive_type_wrapper<int>
{
    public:
        static hid_t hdf5_type_id()
        {
            return H5T_NATIVE_INT;
        }

        static MPI_Datatype mpi_type_id()
        {
            return MPI_INT;
        }

        /*static bool is_complex()
        {
            return false;
        }*/
};


#endif // __TYPEDEFS_H__
