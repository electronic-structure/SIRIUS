#include "field4d.hpp"
#include "periodic_function.hpp"
#include "mixer.hpp"
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

    sirius::Field4D::Field4D(Simulation_context &ctx__, int lmmax__)
            : ctx_(ctx__)
    {
        for (int i = 0; i < ctx_.num_mag_dims() + 1; i++) {
            components_[i] = std::unique_ptr<Periodic_function<double>>(new Periodic_function<double>(ctx_, lmmax__));
            /* allocate global MT array */
            components_[i]->allocate_mt(true);
        }
    }

    Periodic_function<double> &sirius::Field4D::scalar() {
        return *(components_[0]);
    }

    const Periodic_function<double> &sirius::Field4D::scalar() const {
        return *(components_[0]);
    }

    void sirius::Field4D::zero() {
        for (int i = 0; i < ctx_.num_mag_dims() + 1; i++) {
            component(i).zero();
        }
    }

    void sirius::Field4D::fft_transform(int direction__) {
        for (int i = 0; i < ctx_.num_mag_dims() + 1; i++) {
            component(i).fft_transform(direction__);
        }
    }

    void sirius::Field4D::mixer_input() {
        /* split real-space points between available ranks */
        splindex<splindex_t::block> spl_np(ctx_.fft().local_size(), ctx_.comm_ortho_fft().size(), ctx_.comm_ortho_fft().rank());

        int k{0};

        for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
            for (int ialoc = 0; ialoc < ctx_.unit_cell().spl_num_atoms().local_size(); ialoc++) {
                for (int i = 0; i < static_cast<int>(component(j).f_mt(ialoc).size()); i++) {
                    mixer_->input_local(k++, component(j).f_mt(ialoc)[i]);
                }
            }
            //for (int i = 0; i < ctx_.fft().local_size(); i++) {
            for (int i = 0; i < spl_np.local_size(); i++) {
                mixer_->input_local(k++, component(j).f_rg(spl_np[i]));
            }
        }
    }

    void sirius::Field4D::mixer_output() {
        /* split real-space points between available ranks */
        splindex<splindex_t::block> spl_np(ctx_.fft().local_size(), ctx_.comm_ortho_fft().size(), ctx_.comm_ortho_fft().rank());

        int k{0};

        for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
            for (int ialoc = 0; ialoc < ctx_.unit_cell().spl_num_atoms().local_size(); ialoc++) {
                auto& f_mt = const_cast<Spheric_function<function_domain_t::spectral, double>&>(component(j).f_mt(ialoc));
                for (int i = 0; i < static_cast<int>(component(j).f_mt(ialoc).size()); i++) {
                    f_mt[i] = mixer_->output_local(k++);
                }
            }
            //for (int i = 0; i < ctx_.fft().local_size(); i++) {
            for (int i = 0; i < spl_np.local_size(); i++) {
                component(j).f_rg(spl_np[i]) = mixer_->output_local(k++);
            }
            ctx_.comm_ortho_fft().allgather(&component(j).f_rg(0), spl_np.global_offset(), spl_np.local_size());
            component(j).sync_mt();
        }
    }

    void sirius::Field4D::mixer_init(Mixer_input mixer_cfg__) {
        int sz{0};
        for (int ialoc = 0; ialoc < ctx_.unit_cell().spl_num_atoms().local_size(); ialoc++) {
            sz += static_cast<int>(scalar().f_mt(ialoc).size());
        }
        sz += ctx_.fft().local_size();

        mixer_ = Mixer_factory<double>(0, (ctx_.num_mag_dims() + 1) * sz, mixer_cfg__, ctx_.comm());
        mixer_input();
        mixer_->initialize();
    }

    double sirius::Field4D::mix(double rss_min__) {
        mixer_input();
        double rms = mixer_->mix(rss_min__);
        mixer_output();
        return rms;
    }

    Mixer<double> &sirius::Field4D::mixer() {
        return *mixer_;
    }
} // namespace sirius
