template <typename T>
Periodic_function<T>::Periodic_function(Global& parameters_, int angular_domain_size__, int num_gvec__ = 0) 
    : unit_cell_(parameters_.unit_cell()), 
      step_function_(parameters_.step_function()),
      fft_(parameters_.reciprocal_lattice()->fft()),
      esm_type_(parameters_.esm_type()),
      angular_domain_size_(angular_domain_size__),
      num_gvec_(num_gvec__)
{
    if (unit_cell_->full_potential())
    {
        f_mt_.set_dimensions(angular_domain_size_, unit_cell_->max_num_mt_points(), unit_cell_->num_atoms());
        f_mt_local_.set_dimensions(unit_cell_->spl_num_atoms().local_size());
        f_mt_local_.allocate();
        for (int ialoc = 0; ialoc < unit_cell_->spl_num_atoms().local_size(); ialoc++)
        {
            int ia = unit_cell_->spl_num_atoms(ialoc);
            f_mt_local_(ialoc) = new Spheric_function<T>(NULL, angular_domain_size_, unit_cell_->atom(ia)->radial_grid());
        }
    }
    
    f_it_.set_dimensions(fft_->size());
    f_it_local_.set_dimensions(fft_->local_size());

    f_pw_.set_dimensions(num_gvec_);
    f_pw_.allocate();
}

template <typename T>
Periodic_function<T>::~Periodic_function()
{
    if (unit_cell_->full_potential())
    {
        for (int ialoc = 0; ialoc < unit_cell_->spl_num_atoms().local_size(); ialoc++) delete f_mt_local_(ialoc);
    }
}

template <typename T>
void Periodic_function<T>::allocate(bool allocate_global_mt, bool allocate_global_it) 
{
    if (allocate_global_it)
    {
        f_it_.allocate();
        set_local_it_ptr();
    }
    else
    {   
        if (num_gvec_) error_global(__FILE__, __LINE__, "Function requires global array for interstitial storage");
        f_it_local_.allocate();
    }

    if (unit_cell_->full_potential())
    {
        if (allocate_global_mt)
        {
            f_mt_.allocate();
            set_local_mt_ptr();
        }
        else
        {
            for (int ialoc = 0; ialoc < unit_cell_->spl_num_atoms().local_size(); ialoc++) f_mt_local_(ialoc)->allocate();
        }
    }
}

template <typename T>
void Periodic_function<T>::zero()
{
    if (f_mt_.get_ptr()) f_mt_.zero();
    if (f_it_.get_ptr()) f_it_.zero();
    if (f_pw_.get_ptr()) f_pw_.zero();
    if (unit_cell_->full_potential())
    {
        for (int ialoc = 0; ialoc < unit_cell_->spl_num_atoms().local_size(); ialoc++) f_mt_local_(ialoc)->zero();
    }
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
inline void Periodic_function<T>::sync(bool sync_mt, bool sync_it)
{
    Timer t("sirius::Periodic_function::sync");

    if (f_it_.get_ptr() != NULL && sync_it)
    {
        Platform::allgather(&f_it_(0), fft_->global_offset(), fft_->local_size());
    }
    
    if (f_mt_.get_ptr() != NULL && sync_mt)
    {
        Platform::allgather(&f_mt_(0, 0, 0), 
                            f_mt_.size(0) * f_mt_.size(1) * unit_cell_->spl_num_atoms().global_offset(), 
                            f_mt_.size(0) * f_mt_.size(1) * unit_cell_->spl_num_atoms().local_size());
    }
}

template <typename T>
inline void Periodic_function<T>::copy(Periodic_function<T>* src)
{
    for (int irloc = 0; irloc < fft_->local_size(); irloc++)
        f_it_local_(irloc) = src->f_it<local>(irloc);

    if (unit_cell_->full_potential())
    {
        for (int ialoc = 0; ialoc < unit_cell_->spl_num_atoms().local_size(); ialoc++)
            f_mt_local_(ialoc)->copy(src->f_mt(ialoc));
    }
}

template <typename T>
inline void Periodic_function<T>::add(Periodic_function<T>* g)
{
    for (int irloc = 0; irloc < fft_->local_size(); irloc++)
        f_it_local_(irloc) += g->f_it<local>(irloc);
    
    if (unit_cell_->full_potential())
    {
        for (int ialoc = 0; ialoc < unit_cell_->spl_num_atoms().local_size(); ialoc++)
            f_mt_local_(ialoc)->add(g->f_mt(ialoc));
    }
}

template <typename T>
inline T Periodic_function<T>::integrate(std::vector<T>& mt_val, T& it_val)
{
    it_val = 0.0;
    
    if (step_function_ == NULL)
    {
        for (int irloc = 0; irloc < fft_->local_size(); irloc++) it_val += f_it_local_(irloc);
    }
    else
    {
        for (int irloc = 0; irloc < fft_->local_size(); irloc++)
        {
            int ir = fft_->global_index(irloc);
            it_val += f_it_local_(irloc) * step_function_->theta_it(ir);
        }
    }
    it_val *= (unit_cell_->omega() / fft_->size());
    Platform::allreduce(&it_val, 1);
    T total = it_val;
    
    if (unit_cell_->full_potential())
    {
        mt_val.resize(unit_cell_->num_atoms());
        memset(&mt_val[0], 0, unit_cell_->num_atoms() * sizeof(T));

        for (int ialoc = 0; ialoc < unit_cell_->spl_num_atoms().local_size(); ialoc++)
        {
            int ia = unit_cell_->spl_num_atoms(ialoc);
            int nmtp = unit_cell_->atom(ia)->num_mt_points();
            
            Spline<T> s(nmtp, unit_cell_->atom(ia)->type()->radial_grid());
            for (int ir = 0; ir < nmtp; ir++) s[ir] = f_mt<local>(0, ir, ialoc);
            mt_val[ia] = s.interpolate().integrate(2) * fourpi * y00;
        }
        
        Platform::allreduce(&mt_val[0], unit_cell_->num_atoms());
        for (int ia = 0; ia < unit_cell_->num_atoms(); ia++) total += mt_val[ia];
    }

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
void Periodic_function<T>::hdf5_write(HDF5_tree h5f)
{
    if (unit_cell_->full_potential()) h5f.write_mdarray("f_mt", f_mt_);
    h5f.write_mdarray("f_it", f_it_);
}

template <typename T>
void Periodic_function<T>::hdf5_read(HDF5_tree h5f)
{
    if (unit_cell_->full_potential()) h5f.read_mdarray("f_mt", f_mt_);
    h5f.read_mdarray("f_it", f_it_);
}

template <typename T>
size_t Periodic_function<T>::size()
{
    size_t size = f_it_.size();
    if (unit_cell_->full_potential())
    {
        for (int ic = 0; ic < unit_cell_->num_atom_symmetry_classes(); ic++)
        {
            size += angular_domain_size_ * unit_cell_->atom_symmetry_class(ic)->atom_type()->num_mt_points() * 
                    unit_cell_->atom_symmetry_class(ic)->num_atoms();
        }
    }
    
    return size;
}

template <typename T>
size_t Periodic_function<T>::pack(T* array)
{
    size_t n = 0;
    
    if (unit_cell_->full_potential()) 
    {
        for (int ia = 0; ia < unit_cell_->num_atoms(); ia++)
        {
            for (int i1 = 0; i1 < unit_cell_->atom(ia)->num_mt_points(); i1++)
            {
                for (int i0 = 0; i0 < angular_domain_size_; i0++) array[n++] = f_mt_(i0, i1, ia);
            }
        }
    }

    for (int ir = 0; ir < fft_->size(); ir++) array[n++] = f_it_(ir);

    return n;
}

template <typename T>
size_t Periodic_function<T>::unpack(T* array)
{
    size_t n = 0;

    if (unit_cell_->full_potential()) 
    {
        for (int ia = 0; ia < unit_cell_->num_atoms(); ia++)
        {
            for (int i1 = 0; i1 < unit_cell_->atom(ia)->num_mt_points(); i1++)
            {
                for (int i0 = 0; i0 < angular_domain_size_; i0++) f_mt_(i0, i1, ia) = array[n++];
            }
        }
    }

    for (int ir = 0; ir < fft_->size(); ir++) f_it_(ir) = array[n++];

    return n;
}

template <typename T>
T inner(Global& parameters_, Periodic_function<T>* f1, Periodic_function<T>* f2)
{
    auto fft_ = parameters_.reciprocal_lattice()->fft();

    T result = 0.0;

    if (parameters_.step_function() == NULL)
    {
        for (int irloc = 0; irloc < fft_->local_size(); irloc++)
            result += primitive_type_wrapper<T>::conjugate(f1->template f_it<local>(irloc)) * f2->template f_it<local>(irloc);
    }
    else
    {
        for (int irloc = 0; irloc < fft_->local_size(); irloc++)
        {
            int ir = fft_->global_index(irloc);
            result += primitive_type_wrapper<T>::conjugate(f1->template f_it<local>(irloc)) * f2->template f_it<local>(irloc) * 
                      parameters_.step_function(ir);
        }
    }
            
    result *= (parameters_.unit_cell()->omega() / fft_->size());
    
    if (parameters_.unit_cell()->full_potential())
    {
        for (int ialoc = 0; ialoc < parameters_.unit_cell()->spl_num_atoms().local_size(); ialoc++)
            result += inner(f1->f_mt(ialoc), f2->f_mt(ialoc));
    }

    Platform::allreduce(&result, 1);

    return result;
}
