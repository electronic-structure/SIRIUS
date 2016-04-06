// Copyright (c) 2013-2016 Anton Kozhevnikov, Thomas Schulthess
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
inline void Periodic_function<T>::copy_to_global_ptr(T* f_mt__, T* f_it__)
{
    STOP();

    //comm_.allgather(f_it_local_.template at<CPU>(), f_it__, (int)spl_fft_size_.global_offset(), (int)spl_fft_size_.local_size());

    //if (parameters_.full_potential()) 
    //{
    //    mdarray<T, 3> f_mt(f_mt__, angular_domain_size_, unit_cell_.max_num_mt_points(), unit_cell_.num_atoms());
    //    for (int ialoc = 0; ialoc < (int)unit_cell_.spl_num_atoms().local_size(); ialoc++)
    //    {
    //        int ia = unit_cell_.spl_num_atoms(ialoc);
    //        memcpy(&f_mt(0, 0, ia), &f_mt_local_(ialoc)(0, 0), f_mt_local_(ialoc).size() * sizeof(T));
    //    }
    //    int ld = angular_domain_size_ * unit_cell_.max_num_mt_points();
    //    comm_.allgather(f_mt__, static_cast<int>(ld * unit_cell_.spl_num_atoms().global_offset()),
    //                    static_cast<int>(ld * unit_cell_.spl_num_atoms().local_size()));
    //}
}

template <typename T>
inline void Periodic_function<T>::add(Periodic_function<T>* g)
{
    runtime::Timer t("sirius::Periodic_function::add");

    #pragma omp parallel for
    for (int irloc = 0; irloc < fft_.local_size(); irloc++)
        f_rg_(irloc) += g->f_rg(irloc);
    
    if (parameters_.full_potential())
    {
        for (int ialoc = 0; ialoc < (int)unit_cell_.spl_num_atoms().local_size(); ialoc++)
            f_mt_local_(ialoc) += g->f_mt(ialoc);
    }
}

template <typename T>
inline T Periodic_function<T>::integrate(std::vector<T>& mt_val, T& it_val)
{
    runtime::Timer t("sirius::Periodic_function::integrate");

    it_val = 0.0;
    
    if (!parameters_.full_potential())
    {
        #pragma omp parallel
        {
            T it_val_t = 0;
            
            #pragma omp for
            for (int irloc = 0; irloc < fft_.local_size(); irloc++) it_val_t += f_rg_(irloc);

            #pragma omp critical
            it_val += it_val_t;
        }
    }
    else
    {
        for (int irloc = 0; irloc < fft_.local_size(); irloc++)
        {
            it_val += f_rg_(irloc) * step_function_.theta_r(irloc);
        }
    }
    it_val *= (unit_cell_.omega() / fft_.size());
    fft_.comm().allreduce(&it_val, 1);
    T total = it_val;
    
    if (parameters_.full_potential())
    {
        mt_val.resize(unit_cell_.num_atoms());
        memset(&mt_val[0], 0, unit_cell_.num_atoms() * sizeof(T));

        for (int ialoc = 0; ialoc < (int)unit_cell_.spl_num_atoms().local_size(); ialoc++)
        {
            int ia = unit_cell_.spl_num_atoms(ialoc);
            int nmtp = unit_cell_.atom(ia).num_mt_points();
            
            Spline<T> s(unit_cell_.atom(ia).type().radial_grid());
            for (int ir = 0; ir < nmtp; ir++) s[ir] = f_mt<local>(0, ir, ialoc);
            mt_val[ia] = s.interpolate().integrate(2) * fourpi * y00;
        }
        
        comm_.allreduce(&mt_val[0], unit_cell_.num_atoms());
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) total += mt_val[ia];
    }

    return total;
}

template <typename T>
void Periodic_function<T>::hdf5_write(HDF5_tree h5f)
{
    if (parameters_.full_potential()) h5f.write("f_mt", f_mt_);
    h5f.write("f_rg", f_rg_);
    if (num_gvec_) h5f.write("f_pw", f_pw_);
}

template <typename T>
void Periodic_function<T>::hdf5_read(HDF5_tree h5f)
{
    if (parameters_.full_potential()) h5f.read_mdarray("f_mt", f_mt_);
    h5f.read_mdarray("f_rg", f_rg_);
    if (num_gvec_) h5f.read_mdarray("f_pw", f_pw_);
}

template <typename T>
size_t Periodic_function<T>::size()
{
    size_t size = fft_.local_size();
    if (parameters_.full_potential())
    {
        for (int ic = 0; ic < unit_cell_.num_atom_symmetry_classes(); ic++)
        {
            size += angular_domain_size_ * unit_cell_.atom_symmetry_class(ic).atom_type().num_mt_points() * 
                    unit_cell_.atom_symmetry_class(ic).num_atoms();
        }
    }
    return size;
}

template <typename T>
size_t Periodic_function<T>::pack(size_t offset__, Mixer<double>* mixer__)
{
    runtime::Timer t("sirius::Periodic_function::pack");

    size_t n = 0;
    
    if (parameters_.full_potential()) 
    {
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
        {
            for (int i1 = 0; i1 < unit_cell_.atom(ia).num_mt_points(); i1++)
            {
                for (int i0 = 0; i0 < angular_domain_size_; i0++) mixer__->input(offset__ + n++, f_mt_(i0, i1, ia));
            }
        }
    }

    for (int ir = 0; ir < fft_.local_size(); ir++) mixer__->input(offset__ + n++, f_rg_(ir));

    return n;
}

template <typename T>
size_t Periodic_function<T>::unpack(T const* array__)
{
    runtime::Timer t("sirius::Periodic_function::unpack");

    size_t n = 0;

    if (parameters_.full_potential()) 
    {
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
        {
            for (int i1 = 0; i1 < unit_cell_.atom(ia).num_mt_points(); i1++)
            {
                for (int i0 = 0; i0 < angular_domain_size_; i0++) f_mt_(i0, i1, ia) = array__[n++];
            }
        }
    }

    for (int ir = 0; ir < fft_.local_size(); ir++) f_rg_(ir) = array__[n++];

    return n;
}

