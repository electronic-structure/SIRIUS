
// TODO: this is one of the most ugly parts of the code; need to think how to improve the implementation
namespace sirius
{

class Argument
{
    public:

        argument_t type_;
        
        dimension d_;

        Argument()
        {
        }

        Argument(argument_t type__, dimension d__) : type_(type__), d_(d__)
        {
        }

        int size()
        {
            return d_.size();
        }
};

/// Compare the function arguments
/** Return 0 if arguments are different. Return 1 if types are equal. Return 2 if types and sizes are equal */
int operator ==(Argument& a, Argument& b)
{
    int result = 0;
    if (a.type_ == b.type_) result = 1;
    if (result && (a.d_.size() == b.d_.size())) result = 2;
    return result;
}

template <typename T> class mt_function
{
    protected:
        
        mdarray<T, 2> data_;

        Argument arguments_[2];

    public:
        
        mt_function(Argument arg0, Argument arg1)
        {
            this->arguments_[0] = arg0;
            this->arguments_[1] = arg1;
            this->data_.set_dimensions(arg0.d_, arg1.d_);
            this->data_.allocate();
        }
        
        mt_function(T* ptr, Argument arg0, Argument arg1)
        {
            this->arguments_[0] = arg0;
            this->arguments_[1] = arg1;
            this->data_.set_dimensions(arg0.d_, arg1.d_);
            this->data_.set_ptr(ptr);
        }

        Argument& argument(int i)
        {
            assert(i >= 0 && i < 2);
            return arguments_[i];
        }

        void allocate()
        {
            this->data_.allocate();
        }

        void set_ptr(T* ptr)
        {
            this->data_.set_ptr(ptr);
        }
        
        int size(int i)
        {
            assert(i >= 0 && i < 2);
            return arguments_[i].d_.size();
        }

        int argument_idx(argument_t type)
        {
            for (int i = 0; i < 2; i++) 
            {
                if (arguments_[i].type_ == type) return i;
            }

            return -1;
        }

        void zero()
        {
            data_.zero();
        }
    
        inline T& operator()(const int i0, const int i1) 
        {
            return data_(i0, i1);
        }

        /// Convert between Rlm and Ylm domains
        template <typename U>
        void sh_convert(mt_function<U>* f)
        {
            int radial_domain_idx = this->argument_idx(arg_radial);

            if (radial_domain_idx == -1) error(__FILE__, __LINE__, "no radial argument");
            
            // check radial arguments
            if ((this->argument(radial_domain_idx) == f->argument(radial_domain_idx)) != 2)
            {
                error(__FILE__, __LINE__, "wrong radial arguments");
            }

            // number of radial points
            int nr = this->size(radial_domain_idx);
            
            // angular domain index is the opposite
            int angular_domain_idx = (radial_domain_idx == 0) ? 1 : 0;

            if (!(this->argument(angular_domain_idx).type_ == arg_rlm ||
                  this->argument(angular_domain_idx).type_ == arg_ylm))
            {
                error(__FILE__, __LINE__, "wrong angular argument of initial function");
            }
            
            if (!(f->argument(angular_domain_idx).type_ == arg_rlm ||
                  f->argument(angular_domain_idx).type_ == arg_ylm))
            {
                error(__FILE__, __LINE__, "wrong angular argument of final function");
            }
            
            if (this->size(angular_domain_idx) != f->size(angular_domain_idx))
            {
                error(__FILE__, __LINE__, "wrong size of angular arguments");
            }

            int lmmax = this->size(angular_domain_idx);
            int lmax = Utils::lmax_by_lmmax(lmmax);

            // cache transformation arrays
            std::vector<complex16> tpp(lmmax);
            std::vector<complex16> tpm(lmmax);
            for (int l = 0; l <= lmax; l++)
            {
                for (int m = -l; m <= l; m++) 
                {
                    int lm = Utils::lm_by_l_m(l, m);
                    if (this->argument(angular_domain_idx).type_ == arg_rlm &&
                        f->argument(angular_domain_idx).type_ == arg_ylm)
                    {
                        tpp[lm] = SHT::ylm_dot_rlm(l, m, m);
                        tpm[lm] = SHT::ylm_dot_rlm(l, m, -m);
                    }
                    if (this->argument(angular_domain_idx).type_ == arg_ylm && 
                        f->argument(angular_domain_idx).type_ == arg_rlm)
                    {
                        tpp[lm] = SHT::rlm_dot_ylm(l, m, m);
                        tpm[lm] = SHT::rlm_dot_ylm(l, m, -m);
                    }
                }
            }

            // radial index is first
            if (radial_domain_idx == 0)
            {
                int lm = 0;
                for (int l = 0; l <= lmax; l++)
                {
                    for (int m = -l; m <= l; m++)
                    {
                        if (m == 0)
                        {
                            for (int ir = 0; ir < nr; ir++) (*f)(ir, lm) = primitive_type_wrapper<U>::sift(this->data_(ir, lm));
                        }
                        else 
                        {
                            int lm1 = Utils::lm_by_l_m(l, -m);
                            for (int ir = 0; ir < nr; ir++)
                            {
                                (*f)(ir, lm) = primitive_type_wrapper<U>::sift(tpp[lm] * this->data_(ir, lm) + 
                                                                               tpm[lm] * this->data_(ir, lm1));
                            }
                        }
                        lm++;
                    }
                }
            }
            else
            {
                for (int ir = 0; ir < nr; ir++)
                {
                    int lm = 0;
                    for (int l = 0; l <= lmax; l++)
                    {
                        for (int m = -l; m <= l; m++)
                        {
                            if (m == 0)
                            {
                                (*f)(lm, ir) = primitive_type_wrapper<U>::sift(this->data_(lm, ir));
                            }
                            else 
                            {
                                int lm1 = Utils::lm_by_l_m(l, -m);
                                (*f)(lm, ir) = primitive_type_wrapper<U>::sift(tpp[lm] * this->data_(lm, ir) + 
                                                                               tpm[lm] * this->data_(lm1, ir));
                            }
                            lm++;
                        }
                    }
                }
            }
        }

        void sh_transform(SHT* sht, mt_function<T>* f)
        {
            // check radial arguments
            if ((this->argument(1) == f->argument(1)) != 2) error(__FILE__, __LINE__, "wrong radial arguments");

            if ((this->argument(0).type_ == arg_rlm || this->argument(0).type_ == arg_ylm) && 
                f->argument(0).type_ == arg_tp)
            {
                if (this->size(0) > sht->lmmax()) error(__FILE__, __LINE__, "wrong lm size");
                if (f->size(0) != sht->num_points()) error(__FILE__, __LINE__, "wrong tp size");
                
                sht->backward_transform(&this->data_(0, 0), this->size(0), this->size(1), &(*f)(0, 0));
            }
            
            if (this->argument(0).type_ == arg_tp && 
                (f->argument(0).type_ == arg_rlm || f->argument(0).type_ == arg_ylm))
            {
                if (this->size(0) != sht->num_points()) error(__FILE__, __LINE__, "wrong tp size");
                if (f->size(0) > sht->lmmax()) error(__FILE__, __LINE__, "wrong lm size");
                
                sht->forward_transform(&this->data_(0, 0), f->size(0), f->size(1), &(*f)(0, 0));
            }
        }

        void add(mt_function<T>* f)
        {
            for (int i1 = 0; i1 < size(1); i1++)
            {
                for (int i0 = 0; i0 < size(0); i0++) this->data_(i0, i1) += (*f)(i0, i1);
            }
        }
};

void gradient(RadialGrid& r, mt_function<complex16>& f, mt_function<complex16>* gx, mt_function<complex16>* gy,
              mt_function<complex16>* gz)
{
    // TODO: in principle, gradient has lmax+1 harmonics, or it may be computed up to a smaller number of harmonics
    
    mt_function<complex16>* g[] = {gx, gy, gz};

    int radial_domain_idx = f.argument_idx(arg_radial);

    if (radial_domain_idx == -1) error(__FILE__, __LINE__, "no radial argument");

    for (int i = 0; i < 3; i++)
    {
        // check radial arguments
        if ((f.argument(radial_domain_idx) == g[i]->argument(radial_domain_idx)) != 2)
        {
            error(__FILE__, __LINE__, "wrong radial arguments");
        }
    }

    // number of radial points
    int nr = f.size(radial_domain_idx);
            
    // angular domain index is the opposite
    int angular_domain_idx = (radial_domain_idx == 0) ? 1 : 0;

    if (f.argument(angular_domain_idx).type_ != arg_ylm)
    {
        error(__FILE__, __LINE__, "wrong angular argument of initial function");
    }

    for (int i = 0; i < 3; i++)
    {
        if ((f.argument(angular_domain_idx) == g[i]->argument(angular_domain_idx)) != 2)
        {
            error(__FILE__, __LINE__, "wrong angular argument of final function");
        }    
    }

    for (int i = 0; i < 3; i++) g[i]->zero();

    int lmmax = f.size(angular_domain_idx);
    int lmax = Utils::lmax_by_lmmax(lmmax);

    if (radial_domain_idx == 0)
    {
        Spline<complex16> s(nr, r);
        for (int l = 0; l <= lmax; l++)
        {
            double d1 = sqrt(double(l + 1) / double(2 * l + 3));
            double d2 = sqrt(double(l) / double(2 * l - 1));

            for (int m = -l; m <= l; m++)
            {
                int lm = Utils::lm_by_l_m(l, m);
                for (int ir = 0; ir < nr; ir++) s[ir] = f(ir, lm);
                s.interpolate();

                for (int mu = -1; mu <= 1; mu++)
                {
                    int j = (mu + 2) % 3; // map -1,0,1 to 1,2,0

                    if ((l + 1) <= lmax && abs(m + mu) <= l + 1)
                    {
                        int lm1 = Utils::lm_by_l_m(l + 1, m + mu); 
                        double d = d1 * SHT::clebsch_gordan(l, 1, l + 1, m, mu, m + mu);
                        for (int ir = 0; ir < nr; ir++)
                        {
                            (*g[j])(ir, lm1) += (s.deriv(1, ir) - f(ir, lm) * r.rinv(ir) * double(l)) * d;  
                        }
                    }
                    if ((l - 1) >= 0 && abs(m + mu) <= l - 1)
                    {
                        int lm1 = Utils::lm_by_l_m(l - 1, m + mu); 
                        double d = d2 * SHT::clebsch_gordan(l, 1, l - 1, m, mu, m + mu); 
                        for (int ir = 0; ir < nr; ir++)
                        {
                            (*g[j])(ir, lm1) -= (s.deriv(1, ir) + f(ir, lm) * r.rinv(ir) * double(l + 1)) * d;
                        }
                    }
                }
            }
        }

        complex16 d1(1.0 / sqrt(2.0), 0);
        complex16 d2(0, 1.0 / sqrt(2.0));
        for (int lm = 0; lm < lmmax; lm++)
        {
            for (int ir = 0; ir < nr; ir++)
            {
                complex16 g_p = (*g[0])(ir, lm);
                complex16 g_m = (*g[1])(ir, lm);
                (*g[0])(ir, lm) = d1 * (g_m - g_p);
                (*g[1])(ir, lm) = d2 * (g_m + g_p);
            }
        }
    }

}

// TODO: generalize for different lmax
template <typename T>
T inner(RadialGrid& r, mt_function<T>* f1, mt_function<T>* f2)
{
    for (int i = 0; i < 2; i++)
    {
        if ((f1->argument(i) == f2->argument(i)) != 2) error(__FILE__, __LINE__, "wrong arguments");
    }
    int radial_domain_idx = f1->argument_idx(arg_radial);
    int angular_domain_idx = (radial_domain_idx == 0) ? 1 : 0;
    
    Spline<T> s(f1->size(radial_domain_idx), r);

    if (radial_domain_idx == 0)
    {
        for (int lm = 0; lm < f1->size(angular_domain_idx); lm++)
        {
            for (int ir = 0; ir < f1->size(radial_domain_idx); ir++)
                s[ir] += primitive_type_wrapper<T>::conjugate((*f1)(ir, lm)) * (*f2)(ir, lm);
        }       
    }
    else
    {
        for (int ir = 0; ir < f1->size(radial_domain_idx); ir++)
        {
            for (int lm = 0; lm < f1->size(angular_domain_idx); lm++)
                s[ir] += primitive_type_wrapper<T>::conjugate((*f1)(lm, ir)) * (*f2)(lm, ir);
        }
    }
    return s.interpolate().integrate(2);
}

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
            //assert(lmax_ == g->lmax_);
            //assert(parameters_.max_num_mt_points() == g->parameters_.max_num_mt_points());
            //assert(parameters_.num_atoms() == g->parameters_.num_atoms());
            //assert(parameters_.num_gvec() == g->parameters_.num_gvec());
            //assert(parameters_.fft().size() == g->parameters_.fft().size());

            //if (flg & it_component)
            //{
            //    bool f_split = (f_it_.size(0) == parameters_.spl_fft_size().local_size());
            //    bool g_split = (g->f_it_.size(0) == parameters_.spl_fft_size().local_size());

            //    if (!f_split && !g_split) add_it<0, 0>(f_it(), g->f_it());
            //    if (!f_split && g_split) add_it<0, 1>(f_it(), g->f_it());
            //    if (f_split && !g_split) add_it<1, 0>(f_it(), g->f_it());
            //    if (f_split && g_split) add_it<1, 1>(f_it(), g->f_it());
            //}

            //if (flg & rlm_component)
            //{
            //    bool f_split = (f_rlm_.size(2) == parameters_.spl_num_atoms().local_size());
            //    bool g_split = (g->f_rlm_.size(2) == parameters_.spl_num_atoms().local_size());

            //    if (!f_split && !g_split) add_mt_rlm<0, 0>(f_rlm_, g->f_rlm_);
            //    if (!f_split && g_split) add_mt_rlm<0, 1>(f_rlm_, g->f_rlm_);
            //    if (f_split && !g_split) add_mt_rlm<1, 0>(f_rlm_, g->f_rlm_);
            //    if (f_split && g_split) add_mt_rlm<1, 1>(f_rlm_, g->f_rlm_);
            //}

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
