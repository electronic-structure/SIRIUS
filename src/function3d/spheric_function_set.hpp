#ifndef __SPHERIC_FUNCTION_SET_HPP__
#define __SPHERIC_FUNCTION_SET_HPP__

#include "unit_cell/unit_cell.hpp"

namespace sirius {

template <typename T>
class Spheric_function_set
{
  private:
    Unit_cell const* unit_cell_{nullptr};
    std::vector<int> atoms_;
    std::vector<Spheric_function<function_domain_t::spectral, T>> func_;

  public:
    Spheric_function_set()
    {
    }

    Spheric_function_set(Unit_cell const& unit_cell__, std::vector<int> atoms__, std::function<int(int)> lmax__)
        : unit_cell_{&unit_cell__}
        , atoms_{atoms__}
    {
        func_.resize(unit_cell__.num_atoms());
        for (int ia : atoms_) {
            func_[ia] = Spheric_function<function_domain_t::spectral, T>(utils::lmmax(lmax__(ia)), unit_cell_->atom(ia).radial_grid());
        }
    }

    auto const& atoms() const
    {
        return atoms_;
    }

    auto& operator[](int ia__)
    {
        return func_[ia__];
    }

    auto const& unit_cell() const
    {
        return *unit_cell_;
    }
};

}

#endif
