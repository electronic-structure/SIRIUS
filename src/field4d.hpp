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

/** \file field4d.hpp
 *
 *  \brief Base class for Density and Potential.
 */

#ifndef __FIELD4D_HPP__
#define __FIELD4D_HPP__


namespace sirius {
// TODO: add symmetrize_scalar, symmetrize_vector
class Field4D
{
  private:
      /// Four components of the field: scalar, vector_z, vector_x, vector_y
      std::array<std::unique_ptr<Periodic_function<double>>, 4> components_;
  protected:
    Simulation_context& ctx_;

    void symmetrize(Periodic_function<double>* f__,
                    Periodic_function<double>* gz__,
                    Periodic_function<double>* gx__,
                    Periodic_function<double>* gy__)
    {
        PROFILE("sirius::Field4D::symmetrize");

        auto& comm = ctx_.comm();

        auto& remap_gvec = ctx_.remap_gvec();

        if (ctx_.control().print_hash_) {
            auto h = f__->hash_f_pw();
            if (ctx_.comm().rank() == 0) {
                print_hash("f_unsymmetrized(G)", h);
            }
        }

        ctx_.unit_cell().symmetry().symmetrize_function(&f__->f_pw_local(0), remap_gvec, ctx_.sym_phase_factors());

        if (ctx_.control().print_hash_) {
            auto h = f__->hash_f_pw();
            if (ctx_.comm().rank() == 0) {
                print_hash("f_symmetrized(G)", h);
            }
        }

        /* symmetrize PW components */
        switch (ctx_.num_mag_dims()) {
            case 1: {
                ctx_.unit_cell().symmetry().symmetrize_vector_function(&gz__->f_pw_local(0), remap_gvec, ctx_.sym_phase_factors());
                break;
            }
            case 3: {
                if (ctx_.control().print_hash_) {
                    auto h1 = gx__->hash_f_pw();
                    auto h2 = gy__->hash_f_pw();
                    auto h3 = gz__->hash_f_pw();
                    if (ctx_.comm().rank() == 0) {
                        print_hash("fx_unsymmetrized(G)", h1);
                        print_hash("fy_unsymmetrized(G)", h2);
                        print_hash("fz_unsymmetrized(G)", h3);
                    }
                }

                ctx_.unit_cell().symmetry().symmetrize_vector_function(&gx__->f_pw_local(0),
                                                                       &gy__->f_pw_local(0),
                                                                       &gz__->f_pw_local(0),
                                                                       remap_gvec,
                                                                       ctx_.sym_phase_factors());

                if (ctx_.control().print_hash_) {
                    auto h1 = gx__->hash_f_pw();
                    auto h2 = gy__->hash_f_pw();
                    auto h3 = gz__->hash_f_pw();
                    if (ctx_.comm().rank() == 0) {
                        print_hash("fx_symmetrized(G)", h1);
                        print_hash("fy_symmetrized(G)", h2);
                        print_hash("fz_symmetrized(G)", h3);
                    }
                }
                break;
            }
        }

        if (ctx_.full_potential()) {
            /* symmetrize MT components */
            ctx_.unit_cell().symmetry().symmetrize_function(f__->f_mt(), comm);
            switch (ctx_.num_mag_dims()) {
                case 1: {
                    ctx_.unit_cell().symmetry().symmetrize_vector_function(gz__->f_mt(), comm);
                    break;
                }
                case 3: {
                    ctx_.unit_cell().symmetry().symmetrize_vector_function(gx__->f_mt(), gy__->f_mt(), gz__->f_mt(), comm);
                    break;
                }
            }
        }
    }

  public:
    Field4D(Simulation_context& ctx__, int lmmax__)
        : ctx_(ctx__)
    {
        for (int i = 0; i < ctx_.num_mag_dims() + 1; i++) {
            components_[i] = std::unique_ptr<Periodic_function<double>>(new Periodic_function<double>(ctx_, lmmax__));
        }
    }

    Periodic_function<double>& scalar()
    {
        return *(components_[0]);
    }

    Periodic_function<double> const& scalar() const
    {
        return *(components_[0]);
    }

    Periodic_function<double>& vector(int i)
    {
        assert(i >= 0 && i <= 2);
        return *(components_[i + 1]);
    }

    Periodic_function<double> const& vector(int i) const
    {
        assert(i >= 0 && i <= 2);
        return *(components_[i + 1]);
    }

    Periodic_function<double>& component(int i)
    {
        assert(i >= 0 && i <= 3);
        return *(components_[i]);
    }

    Periodic_function<double> const& component(int i) const
    {
        assert(i >= 0 && i <= 3);
        return *(components_[i]);
    }

    void allocate()
    {
        for (int i = 0; i < ctx_.num_mag_dims() + 1; i++) {
            components_[i]->allocate_mt(true);
        }
    }
};

}

#endif
