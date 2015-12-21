#include "density.h"

namespace sirius {

void Density::generate_pseudo_core_charge_density()
{
    Timer t("sirius::Density::generate_pseudo_core_charge_density");

    auto rl = ctx_.reciprocal_lattice();
    auto rho_core_radial_integrals = generate_rho_radial_integrals(2);

    std::vector<double_complex> v = rl->make_periodic_function(rho_core_radial_integrals, ctx_.gvec().num_gvec());
    ctx_.fft(0)->allocate_workspace();
    ctx_.fft(0)->transform<1>(ctx_.gvec(), &v[ctx_.gvec().offset_gvec_fft()]);
    ctx_.fft(0)->output(&rho_pseudo_core_->f_it(0));
    ctx_.fft(0)->deallocate_workspace();
}

};
