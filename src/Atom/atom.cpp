#include "atom.h"

namespace sirius {


Atom::Atom(Atom_type const& type__, vector3d<double> position__, vector3d<double> vector_field__) 
    : type_(type__),
      symmetry_class_(nullptr),
      position_(position__),
      vector_field_(vector_field__),
      offset_aw_(-1),
      offset_lo_(-1),
      offset_wf_(-1),
      apply_uj_correction_(false),
      uj_correction_l_(-1)
{
    for (int x: {0, 1, 2})
    {
        if (position_[x] < 0 || position_[x] >= 1)
        {
            std::stringstream s;
            s << "Wrong atomic position for atom " << type__.label() << ": " << position_[0] << " " << position_[1] << " " << position_[2];
            TERMINATE(s);
        }
    }
}

}
