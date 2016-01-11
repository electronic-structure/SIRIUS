#include "density.h"

namespace sirius {

void Density::generate_pseudo_core_charge_density()
{
    runtime::Timer t("sirius::Density::generate_pseudo_core_charge_density");

    auto rho_core_radial_integrals = generate_rho_radial_integrals(2);

    std::vector<double_complex> v = unit_cell_.make_periodic_function(rho_core_radial_integrals, ctx_.gvec());
    ctx_.fft(0)->prepare();
    ctx_.fft(0)->transform<1>(ctx_.gvec(), &v[ctx_.gvec().offset_gvec_fft()]);
    ctx_.fft(0)->output(&rho_pseudo_core_->f_rg(0));
    ctx_.fft(0)->dismiss();
}

};
