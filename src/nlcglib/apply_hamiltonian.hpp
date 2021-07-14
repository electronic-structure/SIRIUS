#ifndef APPLY_HAMILTONIAN_H
#define APPLY_HAMILTONIAN_H

#include "potential/potential.hpp"
#include "hamiltonian/hamiltonian.hpp"
#include "density/density.hpp"
#include "SDDK/wave_functions.hpp"
#include <memory>
#include <complex>

namespace sirius {

void apply_hamiltonian(Hamiltonian0& H0, K_point& kp, Wave_functions<double>& wf_out, Wave_functions<double>& wf,
                       std::shared_ptr<Wave_functions<double>>& swf)
{
    /////////////////////////////////////////////////////////////
    // // TODO: Hubbard needs manual call to copy to device // //
    /////////////////////////////////////////////////////////////

    int num_wf = wf.num_wf();
    int num_sc = wf.num_sc();
    if (num_wf != wf_out.num_wf() || wf_out.num_sc() != num_sc) {
        throw std::runtime_error("Hamiltonian::apply_ref (python bindings): num_sc or num_wf do not match");
    }
    auto H    = H0(kp);
    auto& ctx = H0.ctx();
// #ifdef SIRIUS_GPU
//     if (is_device_memory(ctx.preferred_memory_t())) {
//         auto& mpd = ctx.mem_pool(memory_t::device);
//         for (int ispn = 0; ispn < num_sc; ++ispn) {
//             wf_out.pw_coeffs(ispn).allocate(mpd);
//             wf.pw_coeffs(ispn).allocate(mpd);
//             wf.pw_coeffs(ispn).copy_to(memory_t::device, 0, num_wf);
//         }
//     }
// #endif
    /* apply H to all wave functions */
    int N = 0;
    int n = num_wf;
    for (int ispn_step = 0; ispn_step < ctx.num_spinors(); ispn_step++) {
        // sping_range: 2 for non-collinear magnetism, otherwise ispn_step
        auto spin_range = sddk::spin_range((ctx.num_mag_dims() == 3) ? 2 : ispn_step);
        H.apply_h_s<std::complex<double>>(spin_range, N, n, wf, &wf_out, swf.get());
    }
// #ifdef SIRIUS_GPU
//     if (is_device_memory(ctx.preferred_memory_t())) {
//         for (int ispn = 0; ispn < num_sc; ++ispn) {
//             wf_out.pw_coeffs(ispn).copy_to(memory_t::host, 0, n);
//             if (swf) {
//                 swf->pw_coeffs(ispn).copy_to(memory_t::host, 0, n);
//             }
//         }
//     }
// #endif // SIRIUS_GPU
}


}  // sirius

#endif /* APPLY_HAMILTONIAN_H */
