#include <list>
#include <vector>
#include <tuple>
#include "Density/density.hpp"
#include "simulation_context.hpp"

namespace sirius {

std::tuple<std::list<std::vector<double>>, std::list<double>> magnetization(Density& density, Simulation_context& ctx)
{
    std::list<std::vector<double>> lvec;
    std::list<double> lm;

    for (int j=0; j < ctx.num_mag_dims(); ++j) {
        auto result = density.magnetization(j).integrate();
        lvec.push_back(std::get<2>(result));
        lm.push_back(std::get<1>(result));
    }

    return std::make_tuple(lvec, lm);
}

}  // sirius
