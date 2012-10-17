
namespace sirius
{

const int rlm_component = 1 << 0;
const int ylm_component = 1 << 1;
const int pw_component = 1 << 2;
const int it_component = 1 << 3;


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
    protected:

        PeriodicFunction(const PeriodicFunction<T>& src);

        PeriodicFunction<T>& operator=(const PeriodicFunction<T>& src);

    private:
        
        Global& parameters_;
        
        typedef typename data_type_wrapper<T>::complex_type_ complex_type_; 
        
        data_type_wrapper<T> data_type_;
        
        /// maximum angular momentum quantum number
        int lmax_;

        /// maxim number of Ylm or Rlm components
        int lmmax_;

        /// real spherical harmonic expansion coefficients
        mdarray<T,3> f_rlm_;
        
        /// complex spherical harmonic expansion coefficients
        mdarray<complex_type_,3> f_ylm_;
        
        /// interstitial values defined on the FFT grid
        mdarray<T,1> f_it_;
        
        /// plane-wave expansion coefficients
        mdarray<complex_type_,1> f_pw_;

    public:

        PeriodicFunction(Global& parameters__, int lmax__) : parameters_(parameters__), 
                                                             lmax_(lmax__)
        {
            lmmax_ = lmmax_by_lmax(lmax_);
            
            f_rlm_.set_dimensions(lmmax_, parameters_.max_num_mt_points(), parameters_.num_atoms());
            f_ylm_.set_dimensions(lmmax_, parameters_.max_num_mt_points(), parameters_.num_atoms());
            f_it_.set_dimensions(parameters_.fft().size());
            f_pw_.set_dimensions(parameters_.num_gvec());
         }

        void convert_to_ylm()
        {
            // check source
            if (!f_rlm_.get_ptr()) error(__FILE__, __LINE__, "f_rlm array is empty");
            
            // check target
            if (!f_ylm_.get_ptr()) error(__FILE__, __LINE__, "f_ylm array is empty");
            
            for (int ia = 0; ia < parameters_.num_atoms(); ia++)
                for (int ir = 0; ir < parameters_.atom(ia)->type()->num_mt_points(); ir++)
                    SHT::convert_frlm_to_fylm(lmax_, &f_rlm_(0, ir, ia), &f_ylm_(0, ir, ia));      

        }
        
        void convert_to_rlm()
        {
            // check source  
            if (!f_ylm_.get_ptr()) error(__FILE__, __LINE__, "f_ylm array is empty");

            // check target
            if (!f_rlm_.get_ptr()) error(__FILE__, __LINE__, "f_rlm array is empty");
            
            for (int ia = 0; ia < parameters_.num_atoms(); ia++)
                for (int ir = 0; ir < parameters_.atom(ia)->type()->num_mt_points(); ir++)
                    SHT::convert_fylm_to_frlm(lmax_, &f_ylm_(0, ir, ia), &f_rlm_(0, ir, ia));      

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
        
        void set_it_ptr(T* f_it__)
        {
            f_it_.set_ptr(f_it__);
        }

        void deallocate(int flags = rlm_component | ylm_component | pw_component | it_component)
        {
            if (flags & rlm_component) f_rlm_.deallocate();
            if (flags & ylm_component) f_ylm_.deallocate();
            if (flags & pw_component) f_pw_.deallocate();
            if (flags & it_component) f_it_.deallocate();
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
            assert(f_it_.get_ptr());

            return f_it_(ir);
        }
        
        inline complex_type_& f_pw(int ig)
        {
            assert(f_pw_.get_ptr());

            return f_pw_(ig);
        }
       
        inline complex_type_* f_ylm()
        {
            return f_ylm_.get_ptr();
        }
        
        inline complex_type_* f_pw()
        {
            return f_pw_.get_ptr();
        }
 
        inline T* f_it()
        {
            return f_it_.get_ptr();
        }

        
        inline void add(PeriodicFunction<T>* rhs, int flg)
        {
            assert(lmax_ == rhs->lmax_);
            assert(parameters_.max_num_mt_points() == rhs->parameters_.max_num_mt_points());
            assert(parameters_.num_atoms() == rhs->parameters_.num_atoms());
            assert(parameters_.num_gvec() == rhs->parameters_.num_gvec());
            assert(parameters_.fft().size() == rhs->parameters_.fft().size());

            if (flg & rlm_component)
            {
                assert(f_rlm_.get_ptr());
                assert(rhs->f_rlm_.get_ptr());
           
                for (int ia = 0; ia < parameters_.num_atoms(); ia++)
                    for (int ir = 0; ir < parameters_.atom(ia)->type()->num_mt_points(); ir++)
                        for (int lm = 0; lm < lmmax_; lm++)
                            f_rlm_(lm, ir, ia) += rhs->f_rlm_(lm, ir, ia);
            }

            if (flg & ylm_component)
            {
                assert(f_ylm_.get_ptr());
                assert(rhs->f_ylm_.get_ptr());
           
                for (int ia = 0; ia < parameters_.num_atoms(); ia++)
                    for (int ir = 0; ir < parameters_.atom(ia)->type()->num_mt_points(); ir++)
                        for (int lm = 0; lm < lmmax_; lm++)
                            f_ylm_(lm, ir, ia) += rhs->f_ylm_(lm, ir, ia);
            } 

            if (flg & pw_component)
            {
                assert(f_pw_.get_ptr());
                assert(rhs->f_pw_.get_ptr());
                
                for (int ig = 0; ig < parameters_.num_gvec(); ig++)
                    f_pw_(ig) += rhs->f_pw_(ig);
            }

            if (flg & it_component)
            {
                assert(f_it_.get_ptr());
                assert(rhs->f_it_.get_ptr());
                
                for (int ir = 0; ir < parameters_.fft().size(); ir++)
                    f_it_(ir) += rhs->f_it_(ir);
            }
        }

        inline void inner(const PeriodicFunction<T>& f)
        {
        }

        T integrate(int flg, std::vector<T>& mt_val, T& it_val)
        {
            it_val = 0;
            mt_val.resize(parameters_.num_atoms());
            memset(&mt_val[0], 0, parameters_.num_atoms() * sizeof(T));

            if (flg & it_component)
            {
                double dv = parameters_.omega() / parameters_.fft().size();
                for (int ir = 0; ir < parameters_.fft().size(); ir++)
                    it_val += f_it_(ir) * parameters_.step_function(ir) * dv;
            }

            if (flg & rlm_component)
            {
                for (int ia = 0; ia < parameters_.num_atoms(); ia++)
                {
                    int nmtp = parameters_.atom(ia)->type()->num_mt_points();
                    Spline<T> s(nmtp, parameters_.atom(ia)->type()->radial_grid());
                    for (int ir = 0; ir < nmtp; ir++)
                        s[ir] = f_rlm_(0, ir, ia);
                    s.interpolate();
                    mt_val[ia] = s.integrate(2) * fourpi * y00;
                }
            }

            T total = it_val;
            for (int ia = 0; ia < parameters_.num_atoms(); ia++)
                total += mt_val[ia];

            return total;
        }

        T integrate(int flg)
        {
            std::vector<T> mt_val;
            T it_val;

            return integrate(flg, mt_val, it_val);
        }
};

};
