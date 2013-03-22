
// TODO: this is one of the most ugly parts of the code; need to think how to improve the implementation
namespace sirius
{

const int rlm_component = 1 << 0;
const int ylm_component = 1 << 1;
const int pw_component = 1 << 2;
const int it_component = 1 << 3;

enum index_order_t {angular_radial, radial_angular};

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
template<typename T, index_order_t index_order = angular_radial> class PeriodicFunction
{ 
    protected:

        PeriodicFunction(const PeriodicFunction<T>& src);

        PeriodicFunction<T>& operator=(const PeriodicFunction<T>& src);

    private:
        
        Global& parameters_;
        
        typedef typename primitive_type_wrapper<T>::complex_t complex_t; 
        
        /// maximum angular momentum quantum number
        int lmax_;

        /// maxim number of Ylm or Rlm components
        int lmmax_;

        /// real spherical harmonic expansion coefficients
        mdarray<T, 3> f_rlm_;
        
        /// complex spherical harmonic expansion coefficients
        mdarray<complex_t, 3> f_ylm_;
        
        /// interstitial values defined on the FFT grid
        mdarray<T, 1> f_it_;
        
        /// plane-wave expansion coefficients
        mdarray<complex_t, 1> f_pw_;

        SHT sht_;

        template <int split_f, int split_g>
        inline void add_it(T* f_it__, T* g_it__)
        {
            if (split_f == 1 || split_g == 1)
            {
                for (int irloc = 0; irloc < parameters_.spl_fft_size().local_size(); irloc++)
                {
                    int ir = parameters_.spl_fft_size(irloc);
                    int irf = (split_f == 0) ? ir : irloc;
                    int irg = (split_g == 0) ? ir : irloc;
                    f_it__[irf] += g_it__[irg];
                }
            }
            else
            {
                for (int ir = 0; ir < parameters_.fft().size(); ir++) f_it__[ir] += g_it__[ir];
            }
        }

        template <int split_f, int split_g>
        inline void add_mt_rlm(mdarray<T, 3>& f_rlm__, mdarray<T, 3>& g_rlm__)
        {
            for (int ialoc = 0; ialoc < parameters_.spl_num_atoms().local_size(); ialoc++)
            {
                int ia = parameters_.spl_num_atoms(ialoc);
                int iaf = (split_f == 0) ? ia : ialoc;
                int iag = (split_g == 0) ? ia : ialoc;
                
                for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
                {
                    for (int lm = 0; lm < lmmax_; lm++) f_rlm__(lm, ir, iaf) += g_rlm__(lm, ir, iag);
                }
            }
        }
        
        template <int split_f, int split_g> 
        inline T prod_it(T* f_it__, T* g_it__)
        {
            T result = 0;

            for (int irloc = 0; irloc < parameters_.spl_fft_size().local_size(); irloc++)
            {
                int ir = parameters_.spl_fft_size(irloc);
                int irf = (split_f == 0) ? ir : irloc;
                int irg = (split_g == 0) ? ir : irloc;
                result += primitive_type_wrapper<T>::conjugate(f_it__[irf]) * g_it__[irg] * 
                          parameters_.step_function(ir);
            }
            
            Platform::allreduce(&result, 1);

            result *= (parameters_.omega() / parameters_.fft().size());

            return result;
        }

        template <int split_f, int split_g>
        inline T prod_mt_rlm(int lmmax, mdarray<T, 3>& f_rlm__, mdarray<T, 3>& g_rlm__)
        {
            T result = 0;

            for (int ialoc = 0; ialoc < parameters_.spl_num_atoms().local_size(); ialoc++)
            {
                int ia = parameters_.spl_num_atoms(ialoc);
                int iaf = (split_f == 0) ? ia : ialoc;
                int iag = (split_g == 0) ? ia : ialoc;

                int nmtp = parameters_.atom(ia)->num_mt_points();
                Spline<T> s(nmtp, parameters_.atom(ia)->type()->radial_grid());

                for (int ir = 0; ir < nmtp; ir++)
                {
                    for (int lm = 0; lm < lmmax; lm++)
                    {
                        s[ir] += primitive_type_wrapper<T>::conjugate(f_rlm__(lm, ir, iaf)) * g_rlm__(lm, ir, iag);
                    }
                }
                s.interpolate();

                result += s.integrate(2);
            }
            
            Platform::allreduce(&result, 1);

            return result;
        }

    public:

        PeriodicFunction(Global& parameters__, int lmax__) : parameters_(parameters__), lmax_(lmax__)
        {
            lmmax_ = Utils::lmmax_by_lmax(lmax_);

            sht_.set_lmax(lmax_);
            
            switch (index_order)
            {
                case angular_radial:
                {
                    f_rlm_.set_dimensions(lmmax_, parameters_.max_num_mt_points(), parameters_.num_atoms());
                    f_ylm_.set_dimensions(lmmax_, parameters_.max_num_mt_points(), parameters_.num_atoms());
                    break;
                }
                case radial_angular:
                {
                    f_rlm_.set_dimensions(parameters_.max_num_mt_points(), lmmax_, parameters_.num_atoms());
                    f_ylm_.set_dimensions(parameters_.max_num_mt_points(), lmmax_, parameters_.num_atoms());
                    break;
                }
            }

            f_it_.set_dimensions(parameters_.fft().size());
            f_pw_.set_dimensions(parameters_.num_gvec());
        }

        void split(int flags)
        {
            int spl_num_atoms = parameters_.spl_num_atoms().local_size();
            if (flags & rlm_component) f_rlm_.set_dimensions(lmmax_, parameters_.max_num_mt_points(), spl_num_atoms);
            if (flags & ylm_component) f_ylm_.set_dimensions(lmmax_, parameters_.max_num_mt_points(), spl_num_atoms);
            if (flags & it_component) f_it_.set_dimensions(parameters_.spl_fft_size().local_size());
        }

        void allocate(int flags) 
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

        void zero(int flags = rlm_component | ylm_component | pw_component | it_component)
        {
            if (f_rlm_.get_ptr() && (flags & rlm_component)) f_rlm_.zero();
            if (f_ylm_.get_ptr() && (flags & ylm_component)) f_ylm_.zero();
            if (f_it_.get_ptr() && (flags & it_component)) f_it_.zero();
            if (f_pw_.get_ptr() && (flags & pw_component)) f_pw_.zero();
        }
        
        inline T& f_rlm(int idx0, int idx1, int ia)
        {
            assert(f_rlm_.get_ptr());

            return f_rlm_(idx0, idx1, ia);
        }
        
        inline complex_t& f_ylm(int idx0, int idx1, int ia)
        {
            assert(f_ylm_.get_ptr());

            return f_ylm_(idx0, idx1, ia);
        }

        inline T& f_it(int ir)
        {
            assert(f_it_.get_ptr());

            return f_it_(ir);
        }
        
        inline complex_t& f_pw(int ig)
        {
            assert(f_pw_.get_ptr());

            return f_pw_(ig);
        }
       
        inline complex_t* f_ylm()
        {
            return f_ylm_.get_ptr();
        }
        
        inline T* f_rlm()
        {
            return f_rlm_.get_ptr();
        }
        
        inline complex_t* f_pw()
        {
            return f_pw_.get_ptr();
        }
 
        inline T* f_it()
        {
            return f_it_.get_ptr();
        }

        inline void sync(int flg)
        {
            Timer t("sirius::PeriodicFunction::sync");

            if (flg & it_component)
            {
                std::vector<T> buff(parameters_.fft().size(), 0); 

                for (int irloc = 0; irloc < parameters_.spl_fft_size().local_size(); irloc++)
                {
                    int ir = parameters_.spl_fft_size(irloc);
                    buff[ir] = f_it_(ir);
                }

                Platform::allreduce(&buff[0], (int)buff.size());
                for (int ir = 0; ir < (int)buff.size(); ir++) f_it_(ir) = buff[ir];
            }

            if (flg & rlm_component)
            {
                for (int ia = 0; ia < parameters_.num_atoms(); ia++)
                {
                    int rank = parameters_.spl_num_atoms().location(1, ia);
                    Platform::bcast(&f_rlm_(0, 0, ia), (int)f_rlm_.size(0) * f_rlm_.size(1), rank);
                }
            }
        }

        inline void add(PeriodicFunction<T>* g, int flg)
        {
            assert(lmax_ == g->lmax_);
            assert(parameters_.max_num_mt_points() == g->parameters_.max_num_mt_points());
            assert(parameters_.num_atoms() == g->parameters_.num_atoms());
            assert(parameters_.num_gvec() == g->parameters_.num_gvec());
            assert(parameters_.fft().size() == g->parameters_.fft().size());

            if (flg & it_component)
            {
                bool f_split = (f_it_.size(0) == parameters_.spl_fft_size().local_size());
                bool g_split = (g->f_it_.size(0) == parameters_.spl_fft_size().local_size());

                if (!f_split && !g_split) add_it<0, 0>(f_it(), g->f_it());
                if (!f_split && g_split) add_it<0, 1>(f_it(), g->f_it());
                if (f_split && !g_split) add_it<1, 0>(f_it(), g->f_it());
                if (f_split && g_split) add_it<1, 1>(f_it(), g->f_it());
            }

            if (flg & rlm_component)
            {
                bool f_split = (f_rlm_.size(2) == parameters_.spl_num_atoms().local_size());
                bool g_split = (g->f_rlm_.size(2) == parameters_.spl_num_atoms().local_size());

                if (!f_split && !g_split) add_mt_rlm<0, 0>(f_rlm_, g->f_rlm_);
                if (!f_split && g_split) add_mt_rlm<0, 1>(f_rlm_, g->f_rlm_);
                if (f_split && !g_split) add_mt_rlm<1, 0>(f_rlm_, g->f_rlm_);
                if (f_split && g_split) add_mt_rlm<1, 1>(f_rlm_, g->f_rlm_);
            }
        }

        /// Computes the inner product <f|g>, where f is "this" and g is the argument
        inline T inner(PeriodicFunction<T>* g, int flg)
        {
            // put asserts here

            int lmmax = std::min(lmmax_, g->lmmax_);

            T result = 0;

            if (flg & it_component)
            {
                bool f_split = (f_it_.size(0) == parameters_.spl_fft_size().local_size());
                bool g_split = (g->f_it_.size(0) == parameters_.spl_fft_size().local_size());

                if (!f_split && !g_split) result = prod_it<0, 0>(f_it(), g->f_it());
                if (!f_split && g_split) result = prod_it<0, 1>(f_it(), g->f_it());
                if (f_split && !g_split) result = prod_it<1, 0>(f_it(), g->f_it());
                if (f_split && g_split) result = prod_it<1, 1>(f_it(), g->f_it());
            }

            if (flg & rlm_component)
            {
                bool f_split = (f_rlm_.size(2) == parameters_.spl_num_atoms().local_size());
                bool g_split = (g->f_rlm_.size(2) == parameters_.spl_num_atoms().local_size());

                if (!f_split && !g_split) result += prod_mt_rlm<0, 0>(lmmax, f_rlm_, g->f_rlm_);
                if (!f_split && g_split) result += prod_mt_rlm<0, 1>(lmmax, f_rlm_, g->f_rlm_);
                if (f_split && !g_split) result += prod_mt_rlm<1, 0>(lmmax, f_rlm_, g->f_rlm_);
                if (f_split && g_split) result += prod_mt_rlm<1, 1>(lmmax, f_rlm_, g->f_rlm_);
            }

            return result;
        }

        inline T inner(PeriodicFunction<T>* g, int flg, T& mt_value, T& it_value)
        {
            // put asserts here

            int lmmax = std::min(lmmax_, g->lmmax_);

            T result = 0;

            if (flg & it_component)
            {
                bool f_split = (f_it_.size(0) == parameters_.spl_fft_size().local_size());
                bool g_split = (g->f_it_.size(0) == parameters_.spl_fft_size().local_size());

                if (!f_split && !g_split) result = prod_it<0, 0>(f_it(), g->f_it());
                if (!f_split && g_split) result = prod_it<0, 1>(f_it(), g->f_it());
                if (f_split && !g_split) result = prod_it<1, 0>(f_it(), g->f_it());
                if (f_split && g_split) result = prod_it<1, 1>(f_it(), g->f_it());

                it_value = result;
            }

            if (flg & rlm_component)
            {
                bool f_split = (f_rlm_.size(2) == parameters_.spl_num_atoms().local_size());
                bool g_split = (g->f_rlm_.size(2) == parameters_.spl_num_atoms().local_size());

                if (!f_split && !g_split) result += prod_mt_rlm<0, 0>(lmmax, f_rlm_, g->f_rlm_);
                if (!f_split && g_split) result += prod_mt_rlm<0, 1>(lmmax, f_rlm_, g->f_rlm_);
                if (f_split && !g_split) result += prod_mt_rlm<1, 0>(lmmax, f_rlm_, g->f_rlm_);
                if (f_split && g_split) result += prod_mt_rlm<1, 1>(lmmax, f_rlm_, g->f_rlm_);

                mt_value = result - it_value;
            }

            return result;
        }
        
        inline T integrate(int flg, std::vector<T>& mt_val, T& it_val)
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
                    for (int ir = 0; ir < nmtp; ir++) s[ir] = f_rlm_(0, ir, ia);
                    s.interpolate();
                    mt_val[ia] = s.integrate(2) * fourpi * y00;
                }
            }

            T total = it_val;
            for (int ia = 0; ia < parameters_.num_atoms(); ia++) total += mt_val[ia];

            return total;
        }

        T integrate(int flg)
        {
            std::vector<T> mt_val;
            T it_val;

            return integrate(flg, mt_val, it_val);
        }

        void hdf5_write(hdf5_tree h5f)
        {
            h5f.write("lmax", &lmax_); 
            h5f.write("lmmax", &lmmax_); 
            h5f.write("f_rlm", f_rlm_);
            h5f.write("f_it", f_it_);
        }

        void hdf5_read(hdf5_tree h5f)
        {
            h5f.read("f_rlm", f_rlm_);
            h5f.read("f_it", f_it_);
        }
};

};
