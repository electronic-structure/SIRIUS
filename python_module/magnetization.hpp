#include <list>
#include <vector>
#include <tuple>
#include "Density/density.hpp"
#include "simulation_context.hpp"

namespace sirius {

std::vector<double> magnetization(Density& density)
{
    std::vector<double> lm(3, 0.0);
    auto result = density.get_magnetisation();

    for (int i = 0; i < 3; ++i)
        lm[i] = std::get<0>(result)[i];

    return lm;
}

} // namespace sirius
