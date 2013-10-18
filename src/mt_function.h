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

/// Compare function arguments
/** Return 0 if the arguments are different. \n
    Return 1 if only the argument types are equal. \n 
    Return 2 if the argument types and sizes are equal */
int operator ==(Argument& a, Argument& b)
{
    int result = 0;
    if (a.type_ == b.type_) result = 1;
    if (result && (a.d_.size() == b.d_.size())) result = 2;
    return result;
}

/// Muffin-tin function representation
template <typename T> class MT_function
{
    protected:
       
        /// function data
        mdarray<T, 2> data_;

        /// function arguments
        Argument arguments_[2];

    public:
        
        /// Constructor for the new function
        MT_function(Argument arg0, Argument arg1)
        {
            this->arguments_[0] = arg0;
            this->arguments_[1] = arg1;
            this->data_.set_dimensions(arg0.d_, arg1.d_);
            this->data_.allocate();
        }
        
        /// Constructor for the existing function (wrapper for the existing array)
        MT_function(T* ptr, Argument arg0, Argument arg1)
        {
            this->arguments_[0] = arg0;
            this->arguments_[1] = arg1;
            this->data_.set_dimensions(arg0.d_, arg1.d_);
            this->data_.set_ptr(ptr);
        }
        
        /// Analogue of the copy constructor for the different function type
        template <typename U>
        MT_function(MT_function<U>* f, bool copy)
        {
            this->arguments_[0] = f->argument(0);
            this->arguments_[1] = f->argument(1);
            this->data_.set_dimensions(this->argument(0).d_, this->argument(1).d_);
            this->data_.allocate();
            
            if (copy)
            {
                if (typeid(T) != typeid(U))
                {
                    f->sh_convert(this);
                }
                else
                {
                    memcpy(this->data_.get_ptr(), &(*f)(0, 0), this->data_.size() * sizeof(T));
                }
            }
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

        int size()
        {
            return size(0) * size(1);
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
        void sh_convert(MT_function<U>* f)
        {
            int radial_domain_idx = this->argument_idx(arg_radial);

            if (radial_domain_idx == -1) error_local(__FILE__, __LINE__, "no radial argument");
            
            // check radial arguments
            if ((this->argument(radial_domain_idx) == f->argument(radial_domain_idx)) != 2)
            {
                error_local(__FILE__, __LINE__, "wrong radial arguments");
            }

            // number of radial points
            int nr = this->size(radial_domain_idx);
            
            // angular domain index is the opposite
            int angular_domain_idx = (radial_domain_idx == 0) ? 1 : 0;

            if ((this->argument(angular_domain_idx) == f->argument(angular_domain_idx)) != 2)
            {
                error_local(__FILE__, __LINE__, "wrong angular argument of initial function");
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
                    if (primitive_type_wrapper<T>::is_real() && primitive_type_wrapper<U>::is_complex())
                    {
                        tpp[lm] = SHT::ylm_dot_rlm(l, m, m);
                        tpm[lm] = SHT::ylm_dot_rlm(l, m, -m);
                    }
                    if (primitive_type_wrapper<T>::is_complex() && primitive_type_wrapper<U>::is_real())
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

        void sh_transform(SHT* sht, MT_function<T>* f)
        {
            // check radial arguments
            if ((this->argument(1) == f->argument(1)) != 2) error_local(__FILE__, __LINE__, "wrong radial arguments");

            if (this->argument(0).type_ == arg_lm && f->argument(0).type_ == arg_tp)
            {
                if (this->size(0) > sht->lmmax()) error_local(__FILE__, __LINE__, "wrong lm size");
                if (f->size(0) != sht->num_points()) error_local(__FILE__, __LINE__, "wrong tp size");
                
                sht->backward_transform(&this->data_(0, 0), this->size(0), this->size(1), &(*f)(0, 0));
            }
            
            if (this->argument(0).type_ == arg_tp && f->argument(0).type_ == arg_lm)
            {
                if (this->size(0) != sht->num_points()) error_local(__FILE__, __LINE__, "wrong tp size");
                if (f->size(0) > sht->lmmax()) error_local(__FILE__, __LINE__, "wrong lm size");
                
                sht->forward_transform(&this->data_(0, 0), f->size(0), f->size(1), &(*f)(0, 0));
            }
        }

        void add(MT_function<T>* f)
        {
            for (int i1 = 0; i1 < size(1); i1++)
            {
                for (int i0 = 0; i0 < size(0); i0++) this->data_(i0, i1) += (*f)(i0, i1);
            }
        }
        
        void copy(MT_function<T>* f)
        {
            for (int i1 = 0; i1 < size(1); i1++)
            {
                for (int i0 = 0; i0 < size(0); i0++) this->data_(i0, i1) = (*f)(i0, i1);
            }
        }
};

template <typename T> class mt_functions
{
    private:

        mdarray<MT_function<T>*, 1> f_;

    public:

        mt_functions(Argument arg0, Argument arg1, int nf)
        {
            f_.set_dimensions(nf);
            f_.allocate();
            for (int i = 0; i < nf; i++) f_(i) = new MT_function<T>(arg0, arg1);
        }

        ~mt_functions()
        {
            for (int i = 0; i < f_.size(0); i++) delete f_(i);
        }

        T& operator()(int i0, int i1, int i2)
        {
            return (*f_(i2))(i0, i1);
        }

        void zero()
        {
            for (int i = 0; i < f_.size(0); i++) f_(i)->zero();
        }

        MT_function<T>* operator()(int i)
        {
            return f_(i);
        }
};

template <typename T>
class MT_function_vector3d
{
    private:
        MT_function<T>* vec_[3];

    public:
        MT_function_vector3d()
        {
            vec_[0] = vec_[1] = vec_[2] = NULL;
        }
        
            

};

void gradient(Radial_grid& r, MT_function<complex16>* f, MT_function<complex16>* gx, MT_function<complex16>* gy,
              MT_function<complex16>* gz)
{
    // TODO: in principle, gradient has lmax+1 harmonics, or it may be computed up to a smaller number of harmonics
    
    MT_function<complex16>* g[] = {gx, gy, gz};

    int radial_domain_idx = f->argument_idx(arg_radial);

    if (radial_domain_idx == -1) error_local(__FILE__, __LINE__, "no radial argument");

    for (int i = 0; i < 3; i++)
    {
        // check radial arguments
        if ((f->argument(radial_domain_idx) == g[i]->argument(radial_domain_idx)) != 2)
        {
            error_local(__FILE__, __LINE__, "wrong radial arguments");
        }
    }

    // number of radial points
    int nr = f->size(radial_domain_idx);
            
    // angular domain index is the opposite
    int angular_domain_idx = (radial_domain_idx == 0) ? 1 : 0;

    if (f->argument(angular_domain_idx).type_ != arg_lm)
    {
        error_local(__FILE__, __LINE__, "wrong angular argument of initial function");
    }

    for (int i = 0; i < 3; i++)
    {
        if ((f->argument(angular_domain_idx) == g[i]->argument(angular_domain_idx)) != 2)
        {
            error_local(__FILE__, __LINE__, "wrong angular argument of final function");
        }    
    }

    for (int i = 0; i < 3; i++) g[i]->zero();

    int lmmax = f->size(angular_domain_idx);
    int lmax = Utils::lmax_by_lmmax(lmmax);

    Spline<complex16> s(nr, r);

    for (int l = 0; l <= lmax; l++)
    {
        double d1 = sqrt(double(l + 1) / double(2 * l + 3));
        double d2 = sqrt(double(l) / double(2 * l - 1));

        for (int m = -l; m <= l; m++)
        {
            int lm = Utils::lm_by_l_m(l, m);
            if (radial_domain_idx == 0)
            {
                for (int ir = 0; ir < nr; ir++) s[ir] = (*f)(ir, lm);
            }
            else
            {
                for (int ir = 0; ir < nr; ir++) s[ir] = (*f)(lm, ir);
            }
            s.interpolate();

            for (int mu = -1; mu <= 1; mu++)
            {
                int j = (mu + 2) % 3; // map -1,0,1 to 1,2,0

                if ((l + 1) <= lmax && abs(m + mu) <= l + 1)
                {
                    int lm1 = Utils::lm_by_l_m(l + 1, m + mu); 
                    double d = d1 * SHT::clebsch_gordan(l, 1, l + 1, m, mu, m + mu);
                    if (radial_domain_idx == 0)
                    {
                        for (int ir = 0; ir < nr; ir++)
                            (*g[j])(ir, lm1) += (s.deriv(1, ir) - (*f)(ir, lm) * r.rinv(ir) * double(l)) * d;  
                    }
                    else
                    {
                        for (int ir = 0; ir < nr; ir++)
                            (*g[j])(lm1, ir) += (s.deriv(1, ir) - (*f)(lm, ir) * r.rinv(ir) * double(l)) * d;  
                    }
                }
                if ((l - 1) >= 0 && abs(m + mu) <= l - 1)
                {
                    int lm1 = Utils::lm_by_l_m(l - 1, m + mu); 
                    double d = d2 * SHT::clebsch_gordan(l, 1, l - 1, m, mu, m + mu); 
                    if (radial_domain_idx == 0)
                    {
                        for (int ir = 0; ir < nr; ir++)
                            (*g[j])(ir, lm1) -= (s.deriv(1, ir) + (*f)(ir, lm) * r.rinv(ir) * double(l + 1)) * d;
                    }
                    else
                    {
                        for (int ir = 0; ir < nr; ir++)
                            (*g[j])(lm1, ir) -= (s.deriv(1, ir) + (*f)(lm, ir) * r.rinv(ir) * double(l + 1)) * d;
                    }
                }
            }
        }
    }

    complex16 d1(1.0 / sqrt(2.0), 0);
    complex16 d2(0, 1.0 / sqrt(2.0));

    if (radial_domain_idx == 0)
    {
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
    else
    {
        for (int ir = 0; ir < nr; ir++)
        {
            for (int lm = 0; lm < lmmax; lm++)
            {
                complex16 g_p = (*g[0])(lm, ir);
                complex16 g_m = (*g[1])(lm, ir);
                (*g[0])(lm, ir) = d1 * (g_m - g_p);
                (*g[1])(lm, ir) = d2 * (g_m + g_p);
            }
        }
    }
}

void gradient(Radial_grid& r, MT_function<double>* f, MT_function<double>* gx, MT_function<double>* gy,
              MT_function<double>* gz)
{
    MT_function<double>* g[] = {gx, gy, gz};

    MT_function<complex16>* zf = new MT_function<complex16>(f, true);
    MT_function<complex16>* zg[3];
    for (int i = 0; i < 3; i++) zg[i] = new MT_function<complex16>(g[i], false);

    gradient(r, zf, zg[0], zg[1], zg[2]);

    for (int i = 0; i < 3; i++)
    {
        zg[i]->sh_convert(g[i]);
        delete zg[i];
    }
    delete zf;
}

// TODO: generalize for different lmax
template <typename T>
T inner(Radial_grid& r, MT_function<T>* f1, MT_function<T>* f2)
{
    for (int i = 0; i < 2; i++)
    {
        if ((f1->argument(i) == f2->argument(i)) != 2) error_local(__FILE__, __LINE__, "wrong arguments");
    }
    int radial_domain_idx = f1->argument_idx(arg_radial);
    int angular_domain_idx = (radial_domain_idx == 0) ? 1 : 0;
    
    Spline<T> s(r.num_mt_points(), r);

    if (radial_domain_idx == 0)
    {
        for (int lm = 0; lm < f1->size(angular_domain_idx); lm++)
        {
            for (int ir = 0; ir < r.num_mt_points(); ir++)
                s[ir] += primitive_type_wrapper<T>::conjugate((*f1)(ir, lm)) * (*f2)(ir, lm);
        }       
    }
    else
    {
        for (int ir = 0; ir < r.num_mt_points(); ir++)
        {
            for (int lm = 0; lm < f1->size(angular_domain_idx); lm++)
                s[ir] += primitive_type_wrapper<T>::conjugate((*f1)(lm, ir)) * (*f2)(lm, ir);
        }
    }
    return s.interpolate().integrate(2);
}

};
