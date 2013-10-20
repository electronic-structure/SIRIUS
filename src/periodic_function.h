
// TODO: this implementation is better, however the distinction between local and global periodic functions is
//       still not very clear
namespace sirius
{

/// Representation of the periodical function on the muffin-tin geometry
/** Inside each muffin-tin the spherical expansion is used:
    \f[
        f({\bf r}) = \sum_{\ell m} f_{\ell m}(r) Y_{\ell m}(\hat {\bf r})
    \f]
    or
    \f[
        f({\bf r}) = \sum_{\ell m} f_{\ell m}(r) R_{\ell m}(\hat {\bf r})
    \f]
    In the interstitial region function is stored on the real-space grid or as a Fourier series:
    \f[
        f({\bf r}) = \sum_{{\bf G}} f({\bf G}) e^{i{\bf G}{\bf r}}
    \f]

    The following terminology is used to describe the distribution of the function:
        - global function: the whole copy of the function is stored at each MPI rank. Ranks should take care about the
          syncronization of the data.
        - local function: the function is distributed across the MPI ranks. 

    \note In order to check if the function is defined as global or as distributed, check the f_mt_ and f_it_ pointers.
          If the function is global, the pointers should not be null.
*/
template<typename T> class Periodic_function
{ 
    protected:

        Periodic_function(const Periodic_function<T>& src);

        Periodic_function<T>& operator=(const Periodic_function<T>& src);

    private:
        
        typedef typename primitive_type_wrapper<T>::complex_t complex_t; 
        
        Global& parameters_;

        /// local part of muffin-tin functions 
        mdarray<Spheric_function<T>*, 1> f_mt_local_;
        
        /// global muffin tin array 
        mdarray<T, 3> f_mt_;

        /// interstitial values defined on the FFT grid
        mdarray<T, 1> f_it_local_;
        
        /// global interstitial array
        mdarray<T, 1> f_it_;

        /// plane-wave expansion coefficients
        mdarray<complex_t, 1> f_pw_;

        int num_gvec_;

        void set_local_mt_ptr()
        {
            for (int ialoc = 0; ialoc < parameters_.spl_num_atoms().local_size(); ialoc++)
            {
                int ia = parameters_.spl_num_atoms(ialoc);
                f_mt_local_(ialoc)->set_ptr(&f_mt_(0, 0, ia));
            }
        }

        void set_local_it_ptr()
        {
            f_it_local_.set_ptr(&f_it_(parameters_.spl_fft_size().global_offset()));
        }

    public:

        /// Constructor
        Periodic_function(Global& parameters__, int angular_domain_size, int num_gvec);
        
        /// Destructor
        ~Periodic_function();
        
        /// Allocate memory
        void allocate(bool allocate_global_mt, bool allocate_global_it);

        /// Zero the function.
        void zero();
        
        /// Syncronize global function.
        void sync(bool sync_mt, bool sync_it);

        /// Copy from source
        void copy(Periodic_function<T>* src);

        /// Add the function
        void add(Periodic_function<T>* g);

        T integrate(std::vector<T>& mt_val, T& it_val);

        T integrate(int flg);
        
        template <index_domain_t index_domain>
        inline T& f_mt(int idx0, int idx1, int ia);
        
        /** \todo write and read distributed functions */
        void hdf5_write(HDF5_tree h5f);

        void hdf5_read(HDF5_tree h5f);

        size_t size();

        size_t pack(T* array);
        
        size_t unpack(T* array);
       
        /// Set the global pointer to the muffin-tin part
        void set_mt_ptr(T* mt_ptr)
        {
            f_mt_.set_ptr(mt_ptr);
            set_local_mt_ptr();
        }

        /// Set the global pointer to the interstitial part
        void set_it_ptr(T* it_ptr)
        {
            f_it_.set_ptr(it_ptr);
            set_local_it_ptr();
        }

        inline Spheric_function<T>& f_mt(int ialoc)
        {
            return *f_mt_local_(ialoc);
        }

        template <index_domain_t index_domain>
        inline T& f_it(int ir)
        {
            switch (index_domain)
            {
                case local:
                {
                    return f_it_local_(ir);
                }
                case global:
                {
                    return f_it_(ir);
                }
            }
        }
        
        inline complex_t& f_pw(int ig)
        {
            return f_pw_(ig);
        }
};

#include "periodic_function.hpp"

};
