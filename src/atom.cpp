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

/** \file atom.cpp
 *   
 *  \brief Contains remaining implementation of sirius::Atom class.
 */

#include "atom.h"

namespace sirius {

Atom::Atom(Atom_type* type__, double* position__, double* vector_field__) 
    : type_(type__), 
      symmetry_class_(NULL), 
      offset_aw_(-1), 
      offset_lo_(-1), 
      offset_wf_(-1), 
      apply_uj_correction_(false), 
      uj_correction_l_(-1)
{
    assert(type__);
        
    for (int x = 0; x < 3; x++)
    {
        if (position__[x] < 0 || position__[x] >= 1)
        {
            std::stringstream s;
            s << "Wrong atomic position for atom " << type__->label() << ": " << position__[0] << " " << position__[1] << " " << position__[2];
            TERMINATE(s);
        }
        position_[x] = position__[x];
        vector_field_[x] = vector_field__[x];
    }
}

void Atom::init(int lmax_pot__, int num_mag_dims__, int offset_aw__, int offset_lo__, int offset_wf__)
{
    assert(lmax_pot__ >= -1);
    assert(offset_aw__ >= 0);
    
    offset_aw_ = offset_aw__;
    offset_lo_ = offset_lo__;
    offset_wf_ = offset_wf__;

    lmax_pot_ = lmax_pot__;
    num_mag_dims_ = num_mag_dims__;

    if (type()->esm_type() == full_potential_lapwlo || type()->esm_type() == full_potential_pwlo)
    {
        int lmmax = Utils::lmmax(lmax_pot_);

        h_radial_integrals_ = mdarray<double, 3>(lmmax, type()->indexr().size(), type()->indexr().size());
        
        b_radial_integrals_ = mdarray<double, 4>(lmmax, type()->indexr().size(), type()->indexr().size(), num_mag_dims_);
        
        occupation_matrix_ = mdarray<double_complex, 4>(16, 16, 2, 2);
        
        uj_correction_matrix_ = mdarray<double_complex, 4>(16, 16, 2, 2);
    }

    if (type()->esm_type() == ultrasoft_pseudopotential ||
        type()->esm_type() == norm_conserving_pseudopotential)
    {
        int nbf = type()->mt_lo_basis_size();
        d_mtrx_ = matrix<double_complex>(nbf, nbf);

        /* initialize d-matrix */
        for (int xi2 = 0; xi2 < nbf; xi2++)
        {
            int lm2 = type()->indexb(xi2).lm;
            int idxrf2 = type()->indexb(xi2).idxrf;
            for (int xi1 = 0; xi1 <= xi2; xi1++)
            {
                int lm1 = type()->indexb(xi1).lm;
                int idxrf1 = type()->indexb(xi1).idxrf;

                d_mtrx_(xi1, xi2) = (lm1 == lm2) ? type()->uspp().d_mtrx_ion(idxrf1, idxrf2) : 0.0;
                d_mtrx_(xi2, xi1) = conj(d_mtrx_(xi1, xi2));
            }
        }
    }
}

extern "C" void spline_inner_product_gpu_v3(int const* idx_ri__,
                                            int num_ri__,
                                            int num_points__,
                                            double const* x__,
                                            double const* dx__,
                                            double const* f__, 
                                            double const* g__,
                                            double* result__);

void Atom::generate_radial_integrals(processing_unit_t pu__, Communicator const& comm__)
{
    Timer t("sirius::Atom::generate_radial_integrals");
    
    int lmmax = Utils::lmmax(lmax_pot_);
    int nmtp = type()->num_mt_points();
    int nrf = type()->indexr().size();

    if (comm__.size() != 1) TERMINATE("not yet mpi parallel");

    splindex<block> spl_lm(lmmax, comm__.size(), comm__.rank());

    std::vector<int> l_by_lm = Utils::l_by_lm(lmax_pot_);

    h_radial_integrals_.zero();
    if (num_mag_dims_) b_radial_integrals_.zero();
    
    //Timer t1("sirius::Atom::generate_radial_integrals|interp");
    /* interpolate radial functions */
    std::vector< Spline<double> > rf_spline(nrf);
    #pragma omp parallel for
    for (int i = 0; i < nrf; i++)
    {
        rf_spline[i] = Spline<double>(type()->radial_grid());
        for (int ir = 0; ir < nmtp; ir++) rf_spline[i][ir] = symmetry_class()->radial_function(ir, i);
        rf_spline[i].interpolate();
    }

    /* interpolate potential multiplied by a radial function */
    std::vector< Spline<double> > vrf_spline(lmmax * nrf * (1 + num_mag_dims_));
    #pragma omp parallel for
    for (int i = 0; i < nrf; i++)
    {
        for (int lm = 0; lm < lmmax; lm++)
        {
            vrf_spline[lm + lmmax * i] = Spline<double>(type()->radial_grid());

            for (int ir = 0; ir < nmtp; ir++)
                vrf_spline[lm + lmmax * i][ir] = symmetry_class()->radial_function(ir, i) * veff_(lm, ir);

            //vrf_spline[lm + lmmax * i].interpolate();
        }
        for (int j = 0; j < num_mag_dims_; j++)
        {
            int offs = (j + 1) * lmmax * nrf;
            for (int lm = 0; lm < lmmax; lm++)
            {
                vrf_spline[lm + lmmax * i + offs] = Spline<double>(type()->radial_grid());

                for (int ir = 0; ir < nmtp; ir++)
                    vrf_spline[lm + lmmax * i + offs][ir] = symmetry_class()->radial_function(ir, i) * beff_[j](lm, ir);
                
                //vrf_spline[lm + lmmax * i + offs].interpolate();
            }
        }
    }
    //t1.stop();

    /* flag the non-zero radial integrals */
    std::vector< std::pair<int, int> > non_zero_elements;

    for (int lm = 0; lm < lmmax; lm++)
    {
        int l = l_by_lm[lm];

        for (int i2 = 0; i2 < type()->indexr().size(); i2++)
        {
            int l2 = type()->indexr(i2).l;
            
            for (int i1 = 0; i1 <= i2; i1++)
            {
                int l1 = type()->indexr(i1).l;
                if ((l + l1 + l2) % 2 == 0)
                {
                    if (lm) non_zero_elements.push_back(std::pair<int, int>(i2, lm + lmmax * i1));
                    for (int j = 0; j < num_mag_dims_; j++)
                    {
                        int offs = (j + 1) * lmmax * type()->indexr().size();
                        non_zero_elements.push_back(std::pair<int, int>(i2, lm + lmmax * i1 + offs));
                    }
                }
            }
        }
    }
    mdarray<int, 2> idx_ri(2, non_zero_elements.size());
    for (int j = 0; j < (int)non_zero_elements.size(); j++)
    {
        idx_ri(0, j) = non_zero_elements[j].first;
        idx_ri(1, j) = non_zero_elements[j].second;
    }

    Timer t2("sirius::Atom::generate_radial_integrals|inner");

    mdarray<double, 1> result(non_zero_elements.size());

    if (pu__ == GPU)
    {
        #ifdef _GPU_
        idx_ri.allocate_on_device();
        idx_ri.copy_to_device();

        mdarray<double, 3> rf_coef(nmtp, 4, nrf);
        mdarray<double, 3> vrf_coef(nullptr, nmtp, 4, lmmax * nrf * (num_mag_dims_ + 1)); 
        vrf_coef.allocate(1);
        vrf_coef.allocate_on_device();

        #pragma omp parallel
        {
            int tid = Platform::thread_id();
            #pragma omp for
            for (int i = 0; i < lmmax * nrf * (num_mag_dims_ + 1); i++)
            {
                vrf_spline[i].interpolate();
                memcpy(vrf_coef.at<CPU>(0, 0, i), vrf_spline[i].coefs().at<CPU>(), nmtp * 4 * sizeof(double));
                cuda_async_copy_to_device(vrf_coef.at<GPU>(0, 0, i), vrf_coef.at<CPU>(0, 0, i), nmtp * 4 *sizeof(double), tid);
            }
        }

        //vrf_coef.copy_to_device();

        for (int i = 0; i < nrf; i++)
            memcpy(rf_coef.at<CPU>(0, 0, i), rf_spline[i].coefs().at<CPU>(), nmtp * 4 * sizeof(double));
        rf_coef.allocate_on_device();
        rf_coef.copy_to_device();

        auto& rgrid = type()->radial_grid();
        mdarray<double, 1> x(nmtp);
        mdarray<double, 1> dx(nmtp - 1);
        rgrid.x() >> x;
        rgrid.dx() >> dx;
        x.allocate_on_device();
        x.copy_to_device();
        dx.allocate_on_device();
        dx.copy_to_device();

        result.allocate_on_device();
        Timer t3("spline_inner_gpu");
        spline_inner_product_gpu_v3(idx_ri.at<GPU>(), (int)non_zero_elements.size(), nmtp, x.at<GPU>(), 
                                    dx.at<GPU>(), rf_coef.at<GPU>(), vrf_coef.at<GPU>(), result.at<GPU>());
        cuda_device_synchronize();
        double tval = t3.stop();
        DUMP("spline GPU integration performance: %12.6f GFlops", 1e-9 * double(non_zero_elements.size()) * nmtp * 85 / tval);
        result.copy_to_host();
        result.deallocate_on_device();
        dx.deallocate_on_device();
        x.deallocate_on_device();
        vrf_coef.deallocate_on_device();
        rf_coef.deallocate_on_device();
        idx_ri.deallocate_on_device();
        #else
        TERMINATE_NO_GPU();
        #endif
    }
    if (pu__ == CPU)
    {
        #pragma omp parallel for
        for (int i = 0; i < lmmax * nrf * (num_mag_dims_ + 1); i++) vrf_spline[i].interpolate();

        #pragma omp parallel for
        for (int j = 0; j < (int)non_zero_elements.size(); j++)
            result(j) = inner(rf_spline[idx_ri(0, j)], vrf_spline[idx_ri(1, j)], 2);
    }
    double tval = t2.stop();
    DUMP("spline integration performance: %12.6f GFlops", 1e-9 * double(non_zero_elements.size()) * nmtp * 85 / tval);
    
    int n = 0;
    for (int lm = 0; lm < lmmax; lm++)
    {
        int l = l_by_lm[lm];

        for (int i2 = 0; i2 < type()->indexr().size(); i2++)
        {
            int l2 = type()->indexr(i2).l;
            
            for (int i1 = 0; i1 <= i2; i1++)
            {
                int l1 = type()->indexr(i1).l;
                if ((l + l1 + l2) % 2 == 0)
                {
                    if (lm)
                    {
                        h_radial_integrals_(lm, i1, i2) = h_radial_integrals_(lm, i2, i1) = result(n++);
                    }
                    else
                    {
                        h_radial_integrals_(0, i1, i2) = symmetry_class()->h_spherical_integral(i1, i2);
                        h_radial_integrals_(0, i2, i1) = symmetry_class()->h_spherical_integral(i2, i1);
                    }
                    for (int j = 0; j < num_mag_dims_; j++)
                    {
                        b_radial_integrals_(lm, i1, i2, j) = b_radial_integrals_(lm, i2, i1, j) = result(n++);
                    }
                }
            }
        }
    }
    
    //== #pragma omp parallel default(shared)
    //== {
    //==     /* potential or magnetic field times a radial function */
    //==     std::vector< Spline<double> > vrf_spline(1 + num_mag_dims_);
    //==     for (int i = 0; i < 1 + num_mag_dims_; i++) vrf_spline[i] = Spline<double>(type()->radial_grid());

    //==     for (int lm_loc = 0; lm_loc < (int)spl_lm.local_size(); lm_loc++)
    //==     {
    //==         int lm = (int)spl_lm[lm_loc];
    //==         int l = l_by_lm[lm];

    //==         #pragma omp for
    //==         for (int i2 = 0; i2 < type()->indexr().size(); i2++)
    //==         {
    //==             int l2 = type()->indexr(i2).l;
    //==             
    //==             /* multiply potential by a radial function */
    //==             for (int ir = 0; ir < nmtp; ir++) 
    //==                 vrf_spline[0][ir] = symmetry_class()->radial_function(ir, i2) * veff_(lm, ir);
    //==             vrf_spline[0].interpolate();
    //==             /* multiply magnetic field by a radial function */
    //==             for (int j = 0; j < num_mag_dims_; j++)
    //==             {
    //==                 for (int ir = 0; ir < nmtp; ir++) 
    //==                     vrf_spline[1 + j][ir] = symmetry_class()->radial_function(ir, i2) * beff_[j](lm, ir);
    //==                 vrf_spline[1 + j].interpolate();
    //==             }
    //==             
    //==             for (int i1 = 0; i1 <= i2; i1++)
    //==             {
    //==                 int l1 = type()->indexr(i1).l;
    //==                 if ((l + l1 + l2) % 2 == 0)
    //==                 {
    //==                     if (lm)
    //==                     {
    //==                         double a = inner(rf_spline[i1], vrf_spline[0], 2);
    //==                         if (std::abs(h_radial_integrals_(lm, i1, i2)-a)>1e-10) 
    //==                         {
    //==                             std::cout << i1 << " " << i2 << " " << lm << std::endl;
    //==                             STOP();

    //==                         }
    //==                         h_radial_integrals_(lm, i1, i2) = h_radial_integrals_(lm, i2, i1) = 
    //==                             inner(rf_spline[i1], vrf_spline[0], 2);
    //==                     }
    //==                     else
    //==                     {
    //==                         h_radial_integrals_(0, i1, i2) = symmetry_class()->h_spherical_integral(i1, i2);
    //==                         h_radial_integrals_(0, i2, i1) = symmetry_class()->h_spherical_integral(i2, i1);
    //==                     }
    //==                     for (int j = 0; j < num_mag_dims_; j++)
    //==                     {
    //==                         double a = inner(rf_spline[i1], vrf_spline[1+j], 2);
    //==                         if (std::abs( b_radial_integrals_(lm, i1, i2, j) - a)>1e-10)
    //==                         {
    //==                             std::cout << i1 << " " << i2 << " " << lm << std::endl;
    //==                             std::cout << "a=" << a <<" b_radial_integrals_(lm, i1, i2, j)="<<  b_radial_integrals_(lm, i1, i2, j)<<std::endl;
    //==                             STOP();
    //==                         }
    //==                         b_radial_integrals_(lm, i1, i2, j) = b_radial_integrals_(lm, i2, i1, j) = 
    //==                             inner(rf_spline[i1], vrf_spline[1 + j], 2);
    //==                     }
    //==                 }
    //==             }
    //==         }
    //==     }
    //== }

    //== comm__.reduce(h_radial_integrals_.at<CPU>(), (int)h_radial_integrals_.size(), 0);
    //== if (num_mag_dims_) comm__.reduce(b_radial_integrals_.at<CPU>(), (int)b_radial_integrals_.size(), 0);

    #ifdef _PRINT_OBJECT_HASH_
    DUMP("hash(veff): %16llX", veff_.hash());
    DUMP("hash(h_radial_integrals): %16llX", h_radial_integrals_.hash());
    #endif
    #ifdef _PRINT_OBJECT_CHECKSUM_
    DUMP("checksum(h_radial_integrals): %18.10f", h_radial_integrals_.checksum());
    #endif
}

}

