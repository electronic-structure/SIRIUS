#ifndef _HUBBARD_HPP_
#define _HUBBARD_HPP_

class hubbard {

// Apply Hubbard correction in the colinear case
private:

  Simulation_context& ctx_;

  Unit_cell& unit_cell_;

  Communicator const& comm_;

  Wavefunctions phi;

  int hubbard_lmax_{0};

  int number_of_hubbard_orbitals_{0};

  mdarray<double_complex, 4> occupancy_number_;

  std::vector<int> offset;

  double hubbard_energy_{0.0};
  double hubbard_energy_dc_contribution_{0.0};
  double hubbard_energy_noflip_{0.0};
  double hubbard_energy_flip_{0.0};

  mdarray<double_complex, 5> hubbard_potential_;

  /// type of hubbard correction to be considered.
  int approximation_{0};


public :

  void set_hubbard_correction(const int approx)
  {
    approximation_ = approx;
  }

  template <typename T> void calculate_hubbard_potential_and_energy()
  {
    this->hubbard_energy_ = 0.0;
    this->hubbard_energy_dc_contribution_ = 0.0;
    this->hubbard_energy_noflip_ = 0.0;
    this->hubbard_energy_flip_ = 0.0;
    // the hubbard potential has the same structure than the occupation
    // numbers
    this->hubbard_potential_.zero();

    if (ctx_.num_mag_dims() != 3) {
      calculate_hubbard_potential_and_energy_colinear_case<T>();
    } else {
      calculate_hubbard_potential_and_energy_non_colinear_case<double_complex>();
    }
    // I need a reduction over k points here.
  }


#include "Potential/hubbard_generate_atomic_orbitals.hpp"
#include "Potential/hubbard_potential_energy.hpp"
#include "Potential/apply_hubbard_potential.hpp"

};
}
#endif
