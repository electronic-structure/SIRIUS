template <typename T>
template <typename U>
void Spheric_function<T>::sh_convert(Spheric_function<U>& f)
{
    // check radial arguments
    if (radial_domain_idx_ != f.radial_domain_idx_ || &radial_grid_ != &f.radial_grid_)
    {
        error_local(__FILE__, __LINE__, "wrong radial arguments");
    }

    // check angular arguments
    if (angular_domain_idx_ != f.angular_domain_idx_ || angular_domain_size_ != f.angular_domain_size_)
    {
        error_local(__FILE__, __LINE__, "wrong angular argumens");
    }
    
    int lmax = Utils::lmax_by_lmmax(angular_domain_size_);

    // cache transformation arrays
    std::vector<complex16> tpp(angular_domain_size_);
    std::vector<complex16> tpm(angular_domain_size_);
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
    if (radial_domain_idx_ == 0)
    {
        int lm = 0;
        for (int l = 0; l <= lmax; l++)
        {
            for (int m = -l; m <= l; m++)
            {
                if (m == 0)
                {
                    for (int ir = 0; ir < radial_domain_size_; ir++) f(ir, lm) = primitive_type_wrapper<U>::sift(this->data_(ir, lm));
                }
                else 
                {
                    int lm1 = Utils::lm_by_l_m(l, -m);
                    for (int ir = 0; ir < radial_domain_size_; ir++)
                    {
                        f(ir, lm) = primitive_type_wrapper<U>::sift(tpp[lm] * this->data_(ir, lm) + 
                                                                    tpm[lm] * this->data_(ir, lm1));
                    }
                }
                lm++;
            }
        }
    }
    else
    {
        for (int ir = 0; ir < radial_domain_size_; ir++)
        {
            int lm = 0;
            for (int l = 0; l <= lmax; l++)
            {
                for (int m = -l; m <= l; m++)
                {
                    if (m == 0)
                    {
                        f(lm, ir) = primitive_type_wrapper<U>::sift(this->data_(lm, ir));
                    }
                    else 
                    {
                        int lm1 = Utils::lm_by_l_m(l, -m);
                        f(lm, ir) = primitive_type_wrapper<U>::sift(tpp[lm] * this->data_(lm, ir) + 
                                                                    tpm[lm] * this->data_(lm1, ir));
                    }
                    lm++;
                }
            }
        }
    }
}

template <typename T>
void Spheric_function<T>::sh_transform(Spheric_function<T>& f)
{
    // check radial arguments
    if (radial_domain_idx_ != f.radial_domain_idx_ || &radial_grid_ != &f.radial_grid_)
    {
        error_local(__FILE__, __LINE__, "wrong radial arguments");
    }
    if (radial_domain_idx_ != 1)
    {
        error_local(__FILE__, __LINE__, "radial argument must be second");
    }
    if ((sht_ == NULL && f.sht_ == NULL) || (sht_ != NULL && f.sht_ != NULL))
    {
        error_local(__FILE__, __LINE__, "wrong anguler arguments");
    }
        
    if (sht_ == NULL)
    {
        if (data_.size(0) != f.sht_->lmmax()) error_local(__FILE__, __LINE__, "wrong lm size");
        if (f.data_.size(0) != f.sht_->num_points()) error_local(__FILE__, __LINE__, "wrong tp size");
        
        f.sht_->backward_transform(&data_(0, 0), angular_domain_size_, radial_domain_size_, &f(0, 0));
    }
    
    if (sht_)
    {
        if (data_.size(0) != sht_->num_points()) error_local(__FILE__, __LINE__, "wrong tp size");
        if (f.data_.size(0) != sht_->lmmax()) error_local(__FILE__, __LINE__, "wrong lm size");
        
        sht_->forward_transform(&data_(0, 0), f.angular_domain_size_, radial_domain_size_, &f(0, 0));
    }
}

template <typename T>
T inner(Spheric_function<T>& f1, Spheric_function<T>& f2)
{
    if ((f1.angular_domain_idx() != f2.angular_domain_idx()) || (f1.angular_domain_size() != f2.angular_domain_size()))
    {
        error_local(__FILE__, __LINE__, "wrong angular arguments");
    }
    if ((f1.radial_domain_idx() != f2.radial_domain_idx()) || (&f1.radial_grid() != &f2.radial_grid()))
    {
        error_local(__FILE__, __LINE__, "wrong radial arguments");
    }
    Spline<T> s(f1.radial_domain_size(), f1.radial_grid());

    if (f1.radial_domain_idx() == 0)
    {
        for (int lm = 0; lm < f1.angular_domain_size(); lm++)
        {
            for (int ir = 0; ir < f1.radial_domain_size(); ir++)
                s[ir] += primitive_type_wrapper<T>::conjugate(f1(ir, lm)) * f2(ir, lm);
        }       
    }
    else
    {
        for (int ir = 0; ir < f1.radial_domain_size(); ir++)
        {
            for (int lm = 0; lm < f1.angular_domain_size(); lm++)
                s[ir] += primitive_type_wrapper<T>::conjugate(f1(lm, ir)) * f2(lm, ir);
        }
    }
    return s.interpolate().integrate(2);
}

