#include "atom_type.h"

namespace sirius {

Atom_type::Atom_type(Simulation_parameters const& parameters__,
                     const char* symbol__, 
                     const char* name__, 
                     int zn__, 
                     double mass__, 
                     std::vector<atomic_level_descriptor>& levels__,
                     radial_grid_t grid_type__) 
    : parameters_(parameters__),
      symbol_(std::string(symbol__)), 
      name_(std::string(name__)), 
      zn_(zn__), 
      mass_(mass__), 
      mt_radius_(2.0), 
      num_mt_points_(2000 + zn__ * 50), 
      atomic_levels_(levels__), 
      offset_lo_(-1),
      initialized_(false)
{
    radial_grid_ = Radial_grid(grid_type__, num_mt_points_, 1e-6 / zn_, 20.0 + 0.25 * zn_); 
}

Atom_type::Atom_type(Simulation_parameters const& parameters__,
                     const int id__, 
                     const std::string label__, 
                     const std::string file_name__)
    : parameters_(parameters__),
      id_(id__), 
      label_(label__),
      zn_(0), 
      mass_(0), 
      num_mt_points_(0), 
      offset_lo_(-1),
      file_name_(file_name__),
      initialized_(false)
{
}

Atom_type::~Atom_type()
{
}

}
