#ifndef APPLY_HAMILTONIAN_H
#define APPLY_HAMILTONIAN_H

#include "potential/potential.hpp"
#include "hamiltonian/hamiltonian.hpp"
#include "density/density.hpp"
#include "SDDK/wave_functions.hpp"
#include <memory>
#include <complex>

namespace sirius {

template<class T>
void apply_hamiltonian(Hamiltonian0<T>& H0, K_point<T>& kp, sddk::Wave_functions<T>& wf_out, sddk::Wave_functions<T>& wf,
                       std::shared_ptr<sddk::Wave_functions<T>>& swf)
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

    /* apply H to all wave functions */
    int N = 0;
    int n = num_wf;
    for (int ispn_step = 0; ispn_step < ctx.num_spinors(); ispn_step++) {
        // sping_range: 2 for non-colinear magnetism, otherwise ispn_step
        auto spin_range = sddk::spin_range((ctx.num_mag_dims() == 3) ? 2 : ispn_step);
        H.template apply_h_s<std::complex<double>>(spin_range, N, n, wf, &wf_out, swf.get());
    }
}


}  // sirius

#endif /* APPLY_HAMILTONIAN_H */
