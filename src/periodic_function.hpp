// Copyright (c) 2013-2014 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that 
// the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the 
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions 
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file periodic_function.hpp
 *   
 *  \brief Contains templated implementation of sirius::Periodic_function class.
 */

template <typename T>
Periodic_function<T>::Periodic_function(Global& parameters_,
                                        int angular_domain_size__,
                                        int num_gvec__,
                                        Communicator const& comm__)
    : unit_cell_(parameters_.unit_cell()), 
      step_function_(parameters_.step_function()),
      fft_(parameters_.reciprocal_lattice()->fft()),
      esm_type_(parameters_.esm_type()),
      angular_domain_size_(angular_domain_size__),
      num_gvec_(num_gvec__),
      comm_(comm__)
{
    if (unit_cell_->full_potential())
    {
        f_mt_ = mdarray<T, 3>(nullptr, angular_domain_size_, unit_cell_->max_num_mt_points(), unit_cell_->num_atoms());
        f_mt_local_ = mdarray<Spheric_function<spectral, T>, 1>(unit_cell_->spl_num_atoms().local_size());
        for (int ialoc = 0; ialoc < unit_cell_->spl_num_atoms().local_size(); ialoc++)
        {
            int ia = unit_cell_->spl_num_atoms(ialoc);
            f_mt_local_(ialoc) = Spheric_function<spectral, T>(NULL, angular_domain_size_, unit_cell_->atom(ia)->radial_grid());
        }
    }
    
    f_pw_ = mdarray<double_complex, 1>(num_gvec_);
    f_it_ = mdarray<T, 1>(nullptr, fft_->size());
}

template <typename T>
Periodic_function<T>::~Periodic_function()
{
}

template <typename T>
void Periodic_function<T>::allocate(bool allocate_global_mt, bool allocate_global_it) 
{
    if (allocate_global_it)
    {
        f_it_ = mdarray<T, 1>(fft_->size());
        f_it_local_ = mdarray<T, 1>(nullptr, fft_->local_size());
        set_local_it_ptr();
    }
    else
    {   
        if (num_gvec_) error_global(__FILE__, __LINE__, "Function requires global array for interstitial storage");
        f_it_local_ = mdarray<T, 1>(fft_->local_size());
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
            for (int ialoc = 0; ialoc < unit_cell_->spl_num_atoms().local_size(); ialoc++) f_mt_local_(ialoc).allocate();
        }
    }
}

template <typename T>
void Periodic_function<T>::zero()
{
    if (f_mt_.ptr()) f_mt_.zero();
    if (f_it_.ptr()) f_it_.zero();
    if (f_pw_.ptr()) f_pw_.zero();
    if (unit_cell_->full_potential())
    {
        for (int ialoc = 0; ialoc < unit_cell_->spl_num_atoms().local_size(); ialoc++) f_mt_local_(ialoc).zero();
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
            return f_mt_local_(ia)(idx0, idx1);
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

    if (f_it_.ptr() != NULL && sync_it)
    {
        comm_.allgather(&f_it_(0), fft_->global_offset(), fft_->local_size());
    }
    
    if (f_mt_.ptr() != NULL && sync_mt)
    {
        comm_.allgather(&f_mt_(0, 0, 0), 
                        (int)(f_mt_.size(0) * f_mt_.size(1) * unit_cell_->spl_num_atoms().global_offset()), 
                        (int)(f_mt_.size(0) * f_mt_.size(1) * unit_cell_->spl_num_atoms().local_size()));
    }
}

template <typename T>
inline void Periodic_function<T>::copy_to_global_ptr(T* f_mt__, T* f_it__)
{
    comm_.allgather(f_it_local_.ptr(), f_it__, fft_->global_offset(), fft_->local_size());

    if (unit_cell_->full_potential()) 
    {
        mdarray<T, 3> f_mt(f_mt__, angular_domain_size_, unit_cell_->max_num_mt_points(), unit_cell_->num_atoms());
        for (int ialoc = 0; ialoc < unit_cell_->spl_num_atoms().local_size(); ialoc++)
        {
            int ia = unit_cell_->spl_num_atoms(ialoc);
            memcpy(&f_mt(0, 0, ia), &f_mt_local_(ialoc)(0, 0), f_mt_local_(ialoc).size() * sizeof(T));
        }
        int ld = angular_domain_size_ * unit_cell_->max_num_mt_points();
        comm_.allgather(f_mt__, static_cast<int>(ld * unit_cell_->spl_num_atoms().global_offset()),
                        static_cast<int>(ld * unit_cell_->spl_num_atoms().local_size()));
    }
    


    //==for (int irloc = 0; irloc < fft_->local_size(); irloc++)
    //==    f_it_local_(irloc) = src->f_it<local>(irloc);

    //==if (unit_cell_->full_potential())
    //=={
    //==    for (int ialoc = 0; ialoc < unit_cell_->spl_num_atoms().local_size(); ialoc++)
    //==        f_mt_local_(ialoc).copy(src->f_mt(ialoc));
    //==}
}

template <typename T>
inline void Periodic_function<T>::add(Periodic_function<T>* g)
{
    for (int irloc = 0; irloc < fft_->local_size(); irloc++)
        f_it_local_(irloc) += g->f_it<local>(irloc);
    
    if (unit_cell_->full_potential())
    {
        for (int ialoc = 0; ialoc < unit_cell_->spl_num_atoms().local_size(); ialoc++)
            f_mt_local_(ialoc) += g->f_mt(ialoc);
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
    comm_.allreduce(&it_val, 1);
    T total = it_val;
    
    if (unit_cell_->full_potential())
    {
        mt_val.resize(unit_cell_->num_atoms());
        memset(&mt_val[0], 0, unit_cell_->num_atoms() * sizeof(T));

        for (int ialoc = 0; ialoc < unit_cell_->spl_num_atoms().local_size(); ialoc++)
        {
            int ia = unit_cell_->spl_num_atoms(ialoc);
            int nmtp = unit_cell_->atom(ia)->num_mt_points();
            
            Spline<T> s(unit_cell_->atom(ia)->type()->radial_grid());
            for (int ir = 0; ir < nmtp; ir++) s[ir] = f_mt<local>(0, ir, ialoc);
            mt_val[ia] = s.interpolate().integrate(2) * fourpi * y00;
        }
        
        comm_.allreduce(&mt_val[0], unit_cell_->num_atoms());
        for (int ia = 0; ia < unit_cell_->num_atoms(); ia++) total += mt_val[ia];
    }

    return total;
}

template <typename T>
void Periodic_function<T>::hdf5_write(HDF5_tree h5f)
{
    if (unit_cell_->full_potential()) h5f.write("f_mt", f_mt_);
    h5f.write("f_it", f_it_);
    if (num_gvec_) h5f.write("f_pw", f_pw_);
}

template <typename T>
void Periodic_function<T>::hdf5_read(HDF5_tree h5f)
{
    if (unit_cell_->full_potential()) h5f.read_mdarray("f_mt", f_mt_);
    h5f.read_mdarray("f_it", f_it_);
    if (num_gvec_) h5f.read_mdarray("f_pw", f_pw_);
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
size_t Periodic_function<T>::pack(size_t offset, Mixer* mixer)
{
    size_t n = 0;
    
    if (unit_cell_->full_potential()) 
    {
        for (int ia = 0; ia < unit_cell_->num_atoms(); ia++)
        {
            for (int i1 = 0; i1 < unit_cell_->atom(ia)->num_mt_points(); i1++)
            {
                for (int i0 = 0; i0 < angular_domain_size_; i0++) mixer->input(offset + n++, f_mt_(i0, i1, ia));
            }
        }
    }

    for (int ir = 0; ir < fft_->size(); ir++) mixer->input(offset + n++, f_it_(ir));

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
            result += type_wrapper<T>::conjugate(f1->template f_it<local>(irloc)) * f2->template f_it<local>(irloc);
    }
    else
    {
        for (int irloc = 0; irloc < fft_->local_size(); irloc++)
        {
            int ir = fft_->global_index(irloc);
            result += type_wrapper<T>::conjugate(f1->template f_it<local>(irloc)) * f2->template f_it<local>(irloc) * 
                      parameters_.step_function(ir);
        }
    }
            
    result *= (parameters_.unit_cell()->omega() / fft_->size());
    
    if (parameters_.unit_cell()->full_potential())
    {
        for (int ialoc = 0; ialoc < parameters_.unit_cell()->spl_num_atoms().local_size(); ialoc++)
            result += inner(f1->f_mt(ialoc), f2->f_mt(ialoc));
    }

    parameters_.comm().allreduce(&result, 1);

    return result;
}
