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
        std::vector<double> vec(3);
        double m;
        density.magnetization(j).integrate(vec, m);
        lvec.push_back(vec);
        lm.push_back(m);
    }

    return std::make_tuple(lvec, lm);
}

}  // sirius
