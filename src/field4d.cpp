// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
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

/** \file field4d.cpp
 *
 *  \brief Implementation of sirius::Field4D class.
 */

#include "field4d.hpp"
#include "periodic_function.hpp"
#include "Mixer/mixer.hpp"
#include "Mixer/mixer_functions.hpp"
#include "Mixer/mixer_factory.hpp"
#include "Symmetry/symmetrize.hpp"

namespace sirius {

void Field4D::symmetrize(Periodic_function<double>* f__, Periodic_function<double>* gz__,
                         Periodic_function<double>* gx__, Periodic_function<double>* gy__)
{
    PROFILE("sirius::Field4D::symmetrize");

    /* quick exit: the only symmetry operation is identity */
    if (ctx_.unit_cell().symmetry().num_mag_sym() == 1) {
        return;
    }

    auto& comm = ctx_.comm();

    auto& remap_gvec = ctx_.remap_gvec();

    if (ctx_.control().print_hash_) {
        auto h = f__->hash_f_pw();
        if (ctx_.comm().rank() == 0) {
            utils::print_hash("f_unsymmetrized(G)", h);
        }
    }

    symmetrize_function(ctx_.unit_cell().symmetry(), remap_gvec, ctx_.sym_phase_factors(), &f__->f_pw_local(0));

    if (ctx_.control().print_hash_) {
        auto h = f__->hash_f_pw();
        if (ctx_.comm().rank() == 0) {
            utils::print_hash("f_symmetrized(G)", h);
        }
    }

    /* symmetrize PW components */
    switch (ctx_.num_mag_dims()) {
        case 1: {
            symmetrize_vector_function(ctx_.unit_cell().symmetry(), remap_gvec, ctx_.sym_phase_factors(),
                                       &gz__->f_pw_local(0));
            break;
        }
        case 3: {
            if (ctx_.control().print_hash_) {
                auto h1 = gx__->hash_f_pw();
                auto h2 = gy__->hash_f_pw();
                auto h3 = gz__->hash_f_pw();
                if (ctx_.comm().rank() == 0) {
                    utils::print_hash("fx_unsymmetrized(G)", h1);
                    utils::print_hash("fy_unsymmetrized(G)", h2);
                    utils::print_hash("fz_unsymmetrized(G)", h3);
                }
            }

            symmetrize_vector_function(ctx_.unit_cell().symmetry(), remap_gvec, ctx_.sym_phase_factors(),
                                       &gx__->f_pw_local(0), &gy__->f_pw_local(0), &gz__->f_pw_local(0));

            if (ctx_.control().print_hash_) {
                auto h1 = gx__->hash_f_pw();
                auto h2 = gy__->hash_f_pw();
                auto h3 = gz__->hash_f_pw();
                if (ctx_.comm().rank() == 0) {
                    utils::print_hash("fx_symmetrized(G)", h1);
                    utils::print_hash("fy_symmetrized(G)", h2);
                    utils::print_hash("fz_symmetrized(G)", h3);
                }
            }
            break;
        }
    }

    if (ctx_.full_potential()) {
        /* symmetrize MT components */
        symmetrize_function(ctx_.unit_cell().symmetry(), comm, f__->f_mt());
        switch (ctx_.num_mag_dims()) {
            case 1: {
                symmetrize_vector_function(ctx_.unit_cell().symmetry(), comm, gz__->f_mt());
                break;
            }
            case 3: {
                symmetrize_vector_function(ctx_.unit_cell().symmetry(), comm, gx__->f_mt(), gy__->f_mt(), gz__->f_mt());
                break;
            }
        }
    }
}

sirius::Field4D::Field4D(Simulation_context& ctx__, int lmmax__)
    : lmmax_(lmmax__)
    , ctx_(ctx__)
{
    for (int i = 0; i < ctx_.num_mag_dims() + 1; i++) {
        components_[i] = std::unique_ptr<Periodic_function<double>>(new Periodic_function<double>(ctx_, lmmax__));
        /* allocate global MT array */
        components_[i]->allocate_mt(true);
    }
}

Periodic_function<double> &sirius::Field4D::scalar() 
{
    return *(components_[0]);
}

const Periodic_function<double> &sirius::Field4D::scalar() const
{
    return *(components_[0]);
}

void sirius::Field4D::zero()
{
    for (int i = 0; i < ctx_.num_mag_dims() + 1; i++) {
        component(i).zero();
    }
}

void sirius::Field4D::fft_transform(int direction__)
{
    for (int i = 0; i < ctx_.num_mag_dims() + 1; i++) {
        component(i).fft_transform(direction__);
    }
}

void sirius::Field4D::mixer_input()
{
    mixer_->set_input<0>(component(0));
    if (ctx_.num_mag_dims() > 0)
        mixer_->set_input<1>(component(1));
    if (ctx_.num_mag_dims() > 1)
        mixer_->set_input<2>(component(2));
    if (ctx_.num_mag_dims() > 2)
        mixer_->set_input<3>(component(3));
}

void sirius::Field4D::mixer_output()
{
    mixer_->get_output<0>(component(0));
    if (ctx_.num_mag_dims() > 0)
        mixer_->get_output<1>(component(1));
    if (ctx_.num_mag_dims() > 1)
        mixer_->get_output<2>(component(2));
    if (ctx_.num_mag_dims() > 2)
        mixer_->get_output<3>(component(3));

    /* split real-space points between available ranks */
    splindex<splindex_t::block> spl_np(ctx_.spfft().local_slice_size(), ctx_.comm_ortho_fft().size(),
                                       ctx_.comm_ortho_fft().rank());

    for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
        ctx_.comm_ortho_fft().allgather(&component(j).f_rg(0), spl_np.global_offset(), spl_np.local_size());
        component(j).sync_mt();
    }
}

void sirius::Field4D::mixer_init(Mixer_input mixer_cfg__)
{
    auto func_prop = mixer::full_potential_periodic_function_property(false);
    auto density_prop = mixer::density_function_property(true);

    mixer_ = Mixer_factory<Periodic_function<double>, Periodic_function<double>, Periodic_function<double>,
                           Periodic_function<double>, mdarray<double_complex, 4>>(
        mixer_cfg__, ctx_.comm(), func_prop, func_prop, func_prop, func_prop, density_prop);
    mixer_->initialize_function<0>(component(0), ctx_, lmmax_, true);
    if (ctx_.num_mag_dims() > 0)
        mixer_->initialize_function<1>(component(1), ctx_, lmmax_, true);
    if (ctx_.num_mag_dims() > 1)
        mixer_->initialize_function<2>(component(2), ctx_, lmmax_, true);
    if (ctx_.num_mag_dims() > 2)
        mixer_->initialize_function<3>(component(3), ctx_, lmmax_, true);
}

double sirius::Field4D::mix(double rss_min__)
{
    mixer_input();
    double rms = mixer_->mix(rss_min__);
    mixer_output();
    return rms;
}

} // namespace sirius
