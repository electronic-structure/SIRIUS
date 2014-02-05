#ifndef __TYPEDEFS_H__
#define __TYPEDEFS_H__

#include <hdf5.h>
#include <mpi.h>
#include <assert.h>
#include <complex>

/** \file typedefs.h

    \brief Contains typedefs, enums and primitive_type_wrapper class
*/

typedef std::complex<double> complex16;

typedef double real8;

enum spin_block_t {nm, uu, ud, dd, du};

enum processing_unit_t {cpu, gpu};

enum lattice_t {direct, reciprocal};

enum coordinates_t {cartesian, fractional};

enum mpi_op_t {op_sum, op_max};

enum linalg_t {lapack, scalapack, elpa, magma};

enum splindex_t {block, block_cyclic};

/** Types of electronic structure methods
    
    The following types are supported:
        - \b full_potential_apwlo: full potential linearized augmented plane waves with local orbitals
        - \b full_potential_pwlo: full potential plane waves with local orbitals (heavily experimental and not completely implemented)
        - \b ultrasoft_pseudopotential: ultrasoft pseudopotential with plane wave basis
*/
enum electronic_structure_method_t {full_potential_lapwlo, full_potential_pwlo, ultrasoft_pseudopotential};

enum index_domain_t {global, local};

enum argument_t {arg_lm, arg_tp, arg_radial};

/// Types of radial grid
enum radial_grid_t {linear_grid, exponential_grid, linear_exponential_grid, pow_grid, hyperbolic_grid, incremental_grid};

/// type of local orbitals
/** lo_rs - local orbital, composed of radial solutions
    lo_cp - confined polynomial local orbital */
enum local_orbital_t {lo_rs, lo_cp};

/// Wrapper for data types
template <typename T> class type_wrapper
{
    public:
        static bool is_primitive()
        {
            return false;
        }
};

template<> class type_wrapper<double>
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

        static inline double sift(std::complex<double> v)
        {
            return real(v);
        }
        
        static MPI_Datatype mpi_type_id()
        {
            return MPI_DOUBLE;
        }

        static bool is_complex()
        {
            return false;
        }
        
        static bool is_real()
        {
            return true;
        }

        static inline double abs(double val)
        {
            return fabs(val);
        }

        static bool is_primitive()
        {
            return true;
        }
};

template<> class type_wrapper<float>
{
    public:
        typedef std::complex<float> complex_t;
        typedef float real_t;
};

template<> class type_wrapper< std::complex<double> >
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
        
        static hid_t hdf5_type_id()
        {
            return H5T_NATIVE_LDOUBLE;
        }
        
        static MPI_Datatype mpi_type_id()
        {
            return MPI_COMPLEX16;
        }

        static bool is_complex()
        {
            return true;
        }
        
        static bool is_real()
        {
            return false;
        }
        
        static inline double abs(std::complex<double> val)
        {
            return std::abs(val);
        }

        static bool is_primitive()
        {
            return true;
        }
};

/*template<> class type_wrapper< std::complex<float> >
{
    public:
        typedef std::complex<float> complex_t;
        typedef float real_t;
};*/

template<> class type_wrapper<int>
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

        static bool is_primitive()
        {
            return true;
        }

        /*static bool is_complex()
        {
            return false;
        }*/
};

template<> class type_wrapper<char>
{
    public:
        /*static hid_t hdf5_type_id()
        {
            return H5T_NATIVE_INT;
        }*/

        static MPI_Datatype mpi_type_id()
        {
            return MPI_CHAR;
        }

        /*static bool is_complex()
        {
            return false;
        }*/
};

#endif // __TYPEDEFS_H__
