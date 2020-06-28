#include <list>
#include <vector>
#include <tuple>
#include <sstream>
#include "density/density.hpp"
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

std::string sprint_magnetization(K_point_set& kset, const Density& density)
{
    auto& ctx       = kset.ctx();
    auto& unit_cell = kset.unit_cell();

    auto result_mag = density.get_magnetisation();
    // auto total_mag  = std::get<0>(result_mag);
    // auto it_mag     = std::get<1>(result_mag);
    auto mt_mag = std::get<2>(result_mag);
    std::stringstream sstream;

    char buffer[20000];

    if (ctx.num_mag_dims()) {
        std::sprintf(buffer, "atom              moment                |moment|");
        sstream << buffer;
        std::sprintf(buffer ,"\n");
        sstream << buffer;
        for (int i = 0; i < 80; i++) {
            std::sprintf(buffer, "-");
            sstream << buffer;
        }
        std::sprintf(buffer, "\n");
        sstream << buffer;

        for (int ia = 0; ia < unit_cell.num_atoms(); ia++) {
            vector3d<double> v(mt_mag[ia]);
            std::sprintf(buffer, "%4i  [%8.4f, %8.4f, %8.4f]  %10.6f", ia, v[0], v[1], v[2], v.length());
            sstream << buffer;
            std::sprintf(buffer, "\n");
            sstream << buffer;
        }

        std::sprintf(buffer, "\n");
        sstream << buffer;
    }

    return sstream.str();
}

} // namespace sirius
