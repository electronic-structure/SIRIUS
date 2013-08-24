template <typename T>
Periodic_function<T>::Periodic_function(Global& parameters__, Argument arg0, Argument arg1, int num_gvec = 0) : 
    parameters_(parameters__)
{
    f_mt_.set_dimensions(arg0.size(), arg1.size(), parameters_.num_atoms());
    f_mt_local_.set_dimensions(parameters_.spl_num_atoms().local_size());
    f_mt_local_.allocate();
    for (int ialoc = 0; ialoc < parameters_.spl_num_atoms().local_size(); ialoc++)
        f_mt_local_(ialoc) = new MT_function<T>(NULL, arg0, arg1);
    
    f_it_.set_dimensions(parameters_.fft().size());
    f_it_local_.set_dimensions(parameters_.spl_fft_size().local_size());

    f_pw_.set_dimensions(num_gvec);
    f_pw_.allocate();
}

template <typename T>
Periodic_function<T>::~Periodic_function()
{
    for (int ialoc = 0; ialoc < parameters_.spl_num_atoms().local_size(); ialoc++) delete f_mt_local_(ialoc);
}

template <typename T>
void Periodic_function<T>::allocate(bool allocate_global) 
{
    // allocate global array if interstial part requires plane-wave coefficients (need FFT buffer)
    if (f_pw_.size(0) || allocate_global)
    {
        f_it_.allocate();
        set_local_it_ptr();
    }
    else
    {
        f_it_local_.allocate();
    }

    if (allocate_global)
    {
        f_mt_.allocate();
        set_local_mt_ptr();
    }
    else
    {
        for (int ialoc = 0; ialoc < parameters_.spl_num_atoms().local_size(); ialoc++)
            f_mt_local_(ialoc)->allocate();
    }
}

template <typename T>
void Periodic_function<T>::zero()
{
    if (f_mt_.get_ptr()) f_mt_.zero();
    if (f_it_.get_ptr()) f_it_.zero();
    if (f_pw_.get_ptr()) f_pw_.zero();
    for (int ialoc = 0; ialoc < parameters_.spl_num_atoms().local_size(); ialoc++) f_mt_local_(ialoc)->zero();
    f_it_local_.zero();
}

template <typename T> template <index_domain_t index_domain>
inline T& Periodic_function<T>::f_mt(int idx0, int idx1, int ia)
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

template <typename T>
inline void Periodic_function<T>::sync()
{
    Timer t("sirius::Periodic_function::sync");

    if (f_it_.get_ptr() != NULL)
    {
        Platform::allgather(&f_it_(0), parameters_.spl_fft_size().global_offset(), 
                            parameters_.spl_fft_size().local_size());
    }
    
    if (f_mt_.get_ptr() != NULL)
    {
        Platform::allgather(&f_mt_(0, 0, 0), 
                            f_mt_.size(0) * f_mt_.size(1) * parameters_.spl_num_atoms().global_offset(), 
                            f_mt_.size(0) * f_mt_.size(1) * parameters_.spl_num_atoms().local_size());
    }
}

template <typename T>
inline void Periodic_function<T>::add(Periodic_function<T>* g)
{
    for (int irloc = 0; irloc < parameters_.spl_fft_size().local_size(); irloc++)
        f_it_local_(irloc) += g->f_it<local>(irloc);
    
    for (int ialoc = 0; ialoc < parameters_.spl_num_atoms().local_size(); ialoc++)
        f_mt_local_(ialoc)->add(g->f_mt(ialoc));
}

template <typename T>
inline T Periodic_function<T>::integrate(std::vector<T>& mt_val, T& it_val)
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

template <typename T>
T Periodic_function<T>::integrate(int flg)
{
    std::vector<T> mt_val;
    T it_val;

    return integrate(flg, mt_val, it_val);
}

template <typename T>
void Periodic_function<T>::hdf5_write(hdf5_tree h5f)
{
    h5f.write("f_mt", f_mt_);
    h5f.write("f_it", f_it_);
}

template <typename T>
void Periodic_function<T>::hdf5_read(hdf5_tree h5f)
{
    h5f.read("f_mt", f_mt_);
    h5f.read("f_it", f_it_);
}

template <typename T>
size_t Periodic_function<T>::size()
{
    size_t size = f_it_local_.size();
    for (int ialoc = 0; ialoc < parameters_.spl_num_atoms().local_size(); ialoc++)
        size += f_mt_local_(ialoc)->size();
    return size;
}

template <typename T>
void Periodic_function<T>::pack(T* array)
{
    int n = 0;
    for (int ialoc = 0; ialoc < parameters_.spl_num_atoms().local_size(); ialoc++)
    {
        for (int i1 = 0; i1 < f_mt_local_(ialoc)->size(1); i1++)
        {
            for (int i0 = 0; i0 < f_mt_local_(ialoc)->size(0); i0++)
            {
                array[n++] = (*f_mt_local_(ialoc))(i0, i1);
            }
        }
    }

    for (int irloc = 0; irloc < parameters_.spl_fft_size().local_size(); irloc++)
        array[n++] = f_it_local_(irloc);
}

template <typename T>
void Periodic_function<T>::unpack(T* array)
{
    int n = 0;
    for (int ialoc = 0; ialoc < parameters_.spl_num_atoms().local_size(); ialoc++)
    {
        for (int i1 = 0; i1 < f_mt_local_(ialoc)->size(1); i1++)
        {
            for (int i0 = 0; i0 < f_mt_local_(ialoc)->size(0); i0++)
            {
                (*f_mt_local_(ialoc))(i0, i1) = array[n++];
            }
        }
    }

    for (int irloc = 0; irloc < parameters_.spl_fft_size().local_size(); irloc++)
        f_it_local_(irloc) = array[n++];
}

template <typename T>
T inner(Global& parameters_, Periodic_function<T>* f1, Periodic_function<T>* f2)
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
