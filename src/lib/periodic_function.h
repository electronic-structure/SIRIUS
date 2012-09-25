
namespace sirius
{

const int rlm_component = 1 << 0;
const int ylm_component = 1 << 1;
const int pw_component = 1 << 2;
const int it_component = 1 << 3;

template <typename T> class data_type_wrapper;

template<> class data_type_wrapper<double>
{
    public:
        typedef std::complex<double> complex_type_;
        inline bool real() 
        {
            return true;
        }
};

template<> class data_type_wrapper<float>
{
    public:
        typedef std::complex<float> complex_type_;
        inline bool real() 
        {
            return true;
        }
};

template<> class data_type_wrapper< std::complex<double> >
{
    public:
        typedef std::complex<double> complex_type_;
        inline bool real() 
        {
            return false;
        }
};

template<> class data_type_wrapper< std::complex<float> >
{
    public:
        typedef std::complex<float> complex_type_;
        inline bool real() 
        {
            return false;
        }
};

/*!
    \brief Representation of the periodical function on the muffin-tin geometry

    Inside each muffin-tin the spherical expansion is used:
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
    private:
        
        typedef typename data_type_wrapper<T>::complex_type_ complex_type_; 
        
        data_type_wrapper<T> data_type_;
        
        int lmax_;
        int lmmax_;

        int max_num_mt_points_;

        int num_atoms_;

        int num_gvec_;

        int num_it_points_;

        /// real spherical harmonic expansion coefficients
        mdarray<T,3> f_rlm_;
        
        /// complex spherical harmonic expansion coefficients
        mdarray<complex_type_,3> f_ylm_;
        
        /// interstitial values defined on the FFT grid
        mdarray<T,1> f_it_;
        
        /// plane-wave expansion coefficients
        mdarray<complex_type_,1> f_pw_;
    
    public:

        void convert_to_ylm()
        {
            // check source
            if (!f_rlm_.get_ptr()) error(__FILE__, __LINE__, "f_rlm array is empty");
            
            // allocate target
            if (!f_ylm_.get_ptr()) f_ylm_.allocate();

            f_ylm_.zero();

            for (int ia = 0; ia < num_atoms_; ia++)
                for (int ir = 0; ir < f_rlm_.size(1); ir++)
                    SHT::convert_frlm_to_fylm(lmax_, &f_rlm_(0, ir, ia), &f_ylm_(0, ir, ia));      

        }
        
        void convert_to_rlm()
        {
            // check source  
            if (!f_ylm_.get_ptr()) error(__FILE__, __LINE__, "f_ylm array is empty");

            // allocate target
            if (!f_rlm_.get_ptr()) f_rlm_.allocate();
            
            f_rlm_.zero();

            for (int ia = 0; ia < num_atoms_; ia++)
                for (int ir = 0; ir < f_rlm_.size(1); ir++)
                    SHT::convert_fylm_to_frlm(lmax_, &f_ylm_(0, ir, ia), &f_rlm_(0, ir, ia));      

        }

        void set_dimensions(int lmax__, int max_num_mt_points__, int num_atoms__, int num_it_points__, int num_gvec__)
        {
            lmax_ = lmax__;
            lmmax_ = (lmax_ + 1) * (lmax_ + 1);
            max_num_mt_points_ = max_num_mt_points__;
            num_atoms_ = num_atoms__;
            num_it_points_ = num_it_points__;
            num_gvec_ = num_gvec__;

            f_rlm_.set_dimensions(lmmax_, max_num_mt_points__, num_atoms__);
            f_ylm_.set_dimensions(lmmax_, max_num_mt_points__, num_atoms__);
            f_it_.set_dimensions(num_it_points__);
            f_pw_.set_dimensions(num_gvec__);
            
            /*if (data_type_.real()) f_rlm_.allocate();
            else f_ylm_.allocate();
            
            f_it_.allocate();
            f_pw_.allocate();*/
        }

        void allocate(int flags = rlm_component | ylm_component | pw_component | it_component)
        {
            if (flags & rlm_component) f_rlm_.allocate();
            if (flags & ylm_component) f_ylm_.allocate();
            if (flags & pw_component) f_pw_.allocate();
            if (flags & it_component) f_it_.allocate();
        }

        void set_rlm_ptr(T* f_rlm__)
        {
            f_rlm_.set_ptr(f_rlm__);
        }
        
        void set_ylm_ptr(T* f_ylm__)
        {
            f_ylm_.set_ptr(f_ylm__);
        }
        
        void set_pw_ptr(T* f_pw__)
        {
            f_pw_.set_ptr(f_pw__);
        }

        void set_it_ptr(T* f_it__)
        {
            f_it_.set_ptr(f_it__);
        }

        void deallocate()
        {
            f_rlm_.deallocate();
            f_ylm_.deallocate();
            f_it_.deallocate();
            f_pw_.deallocate();
        }

        void allocate_ylm()
        {
            f_ylm_.allocate();
        }

        void deallocate_ylm()
        {
            f_ylm_.deallocate();
        }

        void zero()
        {
            if (f_rlm_.get_ptr()) f_rlm_.zero();
            if (f_ylm_.get_ptr()) f_ylm_.zero();
            if (f_it_.get_ptr()) f_it_.zero();
            if (f_pw_.get_ptr()) f_pw_.zero();
        }
        
        inline T& f_rlm(int lm, int ir, int ia)
        {
            assert(f_rlm_.get_ptr());

            return f_rlm_(lm, ir, ia);
        }
        
        inline complex_type_& f_ylm(int lm, int ir, int ia)
        {
            assert(f_ylm_.get_ptr());

            return f_ylm_(lm, ir, ia);
        }

        inline T& f_it(int ir)
        {
            return f_it_(ir);
        }
        
        inline T* f_it()
        {
            return &f_it_(0);
        }

        inline complex_type_& f_pw(int ig)
        {
            return f_pw_(ig);
        }

        inline complex_type_* f_pw()
        {
            return &f_pw_(0);
        }
        
        inline void add(PeriodicFunction<T>& rhs, int flg)
        {
            assert(lmax_ == rhs.lmax_);
            assert(max_num_mt_points_ == rhs.max_num_mt_points_);
            assert(num_atoms_ == rhs.num_atoms_);
            assert(num_gvec_ == rhs.num_gvec_);
            assert(num_it_points_ == rhs.num_it_points_);

            if (flg & rlm_component)
            {
                assert(f_rlm_.get_ptr());
                assert(rhs.f_rlm_.get_ptr());
           
                for (int ia = 0; ia < num_atoms_; ia++)
                    for (int ir = 0; ir < max_num_mt_points_; ir++)
                        for (int lm = 0; lm < lmmax_; lm++)
                            f_rlm_(lm, ir, ia) += rhs.f_rlm_(lm, ir, ia);
            }

            if (flg & ylm_component)
            {
                assert(f_ylm_.get_ptr());
                assert(rhs.f_ylm_.get_ptr());
           
                for (int ia = 0; ia < num_atoms_; ia++)
                    for (int ir = 0; ir < max_num_mt_points_; ir++)
                        for (int lm = 0; lm < lmmax_; lm++)
                            f_ylm_(lm, ir, ia) += rhs.f_ylm_(lm, ir, ia);
            } 

            if (flg & pw_component)
            {
                assert(f_pw_.get_ptr());
                assert(rhs.f_pw_.get_ptr());
                
                for (int ig = 0; ig < num_gvec_; ig++)
                    f_pw_(ig) += rhs.f_pw_(ig);
            }

            if (flg & it_component)
            {
                assert(f_it_.get_ptr());
                assert(rhs.f_it_.get_ptr());
                
                for (int ir = 0; ir < num_it_points_; ir++)
                    f_it_(ir) += rhs.f_it_(ir);
            }
        }

        inline void inner(const PeriodicFunction<T>& f)
        {
        }

};

};
