#include "atom_type.h"

namespace sirius {

void Atom_type::set_radial_grid(int num_points, double const* points)
{
    if (num_mt_points_ == 0) TERMINATE("number of muffin-tin points is zero");
    if (num_points < 0 && points == nullptr)
    {
        /* create default exponential grid */
        radial_grid_ = Radial_grid(exponential_grid, num_mt_points_, radial_grid_origin_, mt_radius_); 
    }
    else
    {
        assert(num_points == num_mt_points_);
        radial_grid_ = Radial_grid(num_points, points);
    }
    if (parameters_.processing_unit() == GPU)
    {
        #ifdef __GPU
        radial_grid_.copy_to_device();
        #endif
    }
}

}
