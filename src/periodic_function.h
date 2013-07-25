
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
*/
template<typename T> class PeriodicFunction
{ 
    protected:

        PeriodicFunction(const PeriodicFunction<T>& src);

        PeriodicFunction<T>& operator=(const PeriodicFunction<T>& src);

    private:
        
        typedef typename primitive_type_wrapper<T>::complex_t complex_t; 
        
        Global& parameters_;

        /// local part of muffin-tin functions 
        mdarray< mt_function<T>*, 1> f_mt_local_;
        
        /// global muffin tin array 
        mdarray<T, 3> f_mt_;

        /// interstitial values defined on the FFT grid
        mdarray<T, 1> f_it_local_;
        
        /// global interstitial array
        mdarray<T, 1> f_it_;

        /// plane-wave expansion coefficients
        mdarray<complex_t, 1> f_pw_;

    public:

        /// Constructor
        PeriodicFunction(Global& parameters__, Argument arg0, Argument arg1, int num_gvec = 0) : 
            parameters_(parameters__)
        {
            f_mt_.set_dimensions(arg0.size(), arg1.size(), parameters_.num_atoms());
            f_mt_local_.set_dimensions(parameters_.spl_num_atoms().local_size());
            f_mt_local_.allocate();
            for (int ialoc = 0; ialoc < parameters_.spl_num_atoms().local_size(); ialoc++)
                f_mt_local_(ialoc) = new mt_function<T>(NULL, arg0, arg1);
            
            f_it_.set_dimensions(parameters_.fft().size());
            f_it_local_.set_dimensions(parameters_.spl_fft_size().local_size());

            f_pw_.set_dimensions(num_gvec);
            f_pw_.allocate();
        }

        ~PeriodicFunction()
        {
            for (int ialoc = 0; ialoc < parameters_.spl_num_atoms().local_size(); ialoc++) delete f_mt_local_(ialoc);
        }
        
        void set_local_mt_ptr()
        {
            for (int ialoc = 0; ialoc < parameters_.spl_num_atoms().local_size(); ialoc++)
            {
                int ia = parameters_.spl_num_atoms(ialoc);
                f_mt_local_(ialoc)->set_ptr(&f_mt_(0, 0, ia));
            }
        }
        
        void set_mt_ptr(T* mt_ptr)
        {
            f_mt_.set_ptr(mt_ptr);
            set_local_mt_ptr();
        }

        void set_local_it_ptr()
        {
            f_it_local_.set_ptr(&f_it_(parameters_.spl_fft_size().global_offset()));
        }
        
        void set_it_ptr(T* it_ptr)
        {
            f_it_.set_ptr(it_ptr);
            set_local_it_ptr();
        }

        void allocate(bool allocate_global) 
        {
            if (allocate_global)
            {
                f_it_.allocate();
                set_local_it_ptr();
                f_mt_.allocate();
                set_local_mt_ptr();
            }
            else
            {
                f_it_local_.allocate();
                for (int ialoc = 0; ialoc < parameters_.spl_num_atoms().local_size(); ialoc++)
                    f_mt_local_(ialoc)->allocate();
            }
        }


        //void deallocate(int flags = rlm_component | ylm_component | pw_component | it_component)
        //{
        //    if (flags & rlm_component) f_rlm_.deallocate();
        //    if (flags & ylm_component) f_ylm_.deallocate();
        //    if (flags & pw_component) f_pw_.deallocate();
        //    if (flags & it_component) f_it_.deallocate();
        //}

        void zero()
        {
            f_mt_.zero();
            f_it_.zero();
            f_pw_.zero();
            for (int ialoc = 0; ialoc < parameters_.spl_num_atoms().local_size(); ialoc++) f_mt_local_(ialoc)->zero();
            f_it_local_.zero();
        }
        
        template <index_domain_t index_domain>
        inline T& f_mt(int idx0, int idx1, int ia)
        {
            switch (index_domain)
            {
                case local:
                {
                    return (*f_mt_local_(ia))(idx0, idx1);
                }
                case global:
                {
                    return f_mt_(idx0, idx1, ia);
                }
            }

        }

        inline mt_function<T>* f_mt(int ialoc)
        {
            return f_mt_local_(ialoc);
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
       
        inline void sync()
        {
            Timer t("sirius::PeriodicFunction::sync");

            if (f_it_.get_ptr() == NULL || f_mt_.get_ptr() == NULL)
                error(__FILE__, __LINE__, "global arrays are not allocated");

            Platform::allgather(&f_it_(0), parameters_.spl_fft_size().global_offset(), 
                                parameters_.spl_fft_size().local_size());

            Platform::allgather(&f_mt_(0, 0, 0), 
                                f_mt_.size(0) * f_mt_.size(1) * parameters_.spl_num_atoms().global_offset(), 
                                f_mt_.size(0) * f_mt_.size(1) * parameters_.spl_num_atoms().local_size());
                             

            //for (int rank = 0; rank < Platform::num_mpi_ranks(); rank++)
            //{
            //    int offset = parameters_.spl_fft_size().global_index(0, rank);
            //    Platform::bcast(&f_it_(offset), parameters_.spl_fft_size().local_size(), rank);
            //}

            //for (int ia = 0; ia < parameters_.num_atoms(); ia++)
            //{
            //    int rank = parameters_.spl_num_atoms().location(_splindex_rank_, ia);
            //    Platform::bcast(&f_mt_(0, 0, ia), (int)f_mt_.size(0) * f_mt_.size(1), rank);
            //}
        }

        inline void add(PeriodicFunction<T>* g)
        {
            for (int irloc = 0; irloc < parameters_.spl_fft_size().local_size(); irloc++)
                f_it_local_(irloc) += g->f_it<local>(irloc);
            
            for (int ialoc = 0; ialoc < parameters_.spl_num_atoms().local_size(); ialoc++)
                f_mt_local_(ialoc)->add(g->f_mt(ialoc));
        }

        inline T integrate(std::vector<T>& mt_val, T& it_val)
        {
            it_val = 0.0;
            mt_val.resize(parameters_.num_atoms());
            memset(&mt_val[0], 0, parameters_.num_atoms() * sizeof(T));

            for (int irloc = 0; irloc < parameters_.spl_fft_size().local_size(); irloc++)
            {
                int ir = parameters_.spl_fft_size(irloc);
                it_val += f_it_local_(irloc) * parameters_.step_function(ir);
            }
            it_val *= (parameters_.omega() / parameters_.fft().size());

            Platform::allreduce(&it_val, 1);

            for (int ialoc = 0; ialoc < parameters_.spl_num_atoms().local_size(); ialoc++)
            {
                int ia = parameters_.spl_num_atoms(ialoc);
                int nmtp = parameters_.atom(ia)->num_mt_points();
                
                Spline<T> s(nmtp, parameters_.atom(ia)->type()->radial_grid());
                for (int ir = 0; ir < nmtp; ir++) s[ir] = f_mt<local>(0, ir, ialoc);
                mt_val[ia] = s.interpolate().integrate(2) * fourpi * y00;
            }
            
            Platform::allreduce(&mt_val[0], parameters_.num_atoms());

            T total = it_val;
            for (int ia = 0; ia < parameters_.num_atoms(); ia++) total += mt_val[ia];

            return total;
        }

        //T integrate(int flg)
        //{
        //    std::vector<T> mt_val;
        //    T it_val;

        //    return integrate(flg, mt_val, it_val);
        //}
        
        //TODO: this is more complicated if the function is distributed
        void hdf5_write(hdf5_tree h5f)
        {
            //h5f.write("lmax", &lmax_); 
            //h5f.write("lmmax", &lmmax_); 
            h5f.write("f_rlm", f_mt_);
            h5f.write("f_it", f_it_);
        }

        void hdf5_read(hdf5_tree h5f)
        {
            h5f.read("f_rlm", f_mt_);
            h5f.read("f_it", f_it_);
        }

        int size()
        {

            return 0;
        }
};

template <typename T>
T inner(Global& parameters_, PeriodicFunction<T>* f1, PeriodicFunction<T>* f2)
{
    T result = 0.0;

    for (int irloc = 0; irloc < parameters_.spl_fft_size().local_size(); irloc++)
    {
        int ir = parameters_.spl_fft_size(irloc);
        result += primitive_type_wrapper<T>::conjugate(f1->template f_it<local>(irloc)) * f2->template f_it<local>(irloc) * 
                  parameters_.step_function(ir);
    }
            
    result *= (parameters_.omega() / parameters_.fft().size());
    
    for (int ialoc = 0; ialoc < parameters_.spl_num_atoms().local_size(); ialoc++)
    {
        int ia =  parameters_.spl_num_atoms(ialoc);
        result += inner(parameters_.atom(ia)->type()->radial_grid(), f1->f_mt(ialoc), f2->f_mt(ialoc));
    }

    Platform::allreduce(&result, 1);

    return result;
}

};
