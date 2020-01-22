// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file field4d.hpp
 *
 *  \brief Base class for sirius::Density and sirius::Potential.
 */

#ifndef __FIELD4D_HPP__
#define __FIELD4D_HPP__

#include <memory>
#include <array>
#include <cassert>
#include <stdexcept>
#include "SDDK/memory.hpp"
#include "typedefs.hpp"

namespace sirius {

/// Four-component function consisting of scalar and vector parts.
/** This class is used to represents density/magnetisation and potential/magentic filed of the system.
 */

// forward declarations
class Simulation_context;
template<class> class Periodic_function;
//class Mixer_input;

class Field4D
{
  private:
    /// Four components of the field: scalar, vector_z, vector_x, vector_y
    std::array<std::unique_ptr<Periodic_function<double>>, 4> components_;

  protected:
    int lmmax_;

    Simulation_context& ctx_;

    void symmetrize(Periodic_function<double>* f__,
                    Periodic_function<double>* gz__,
                    Periodic_function<double>* gx__,
                    Periodic_function<double>* gy__);

  public:
    Field4D(Simulation_context& ctx__, int lmmax__);

    /// Return scalar part of the field.
    Periodic_function<double>& scalar();

    /// Return scalar part of the field.
    Periodic_function<double> const& scalar() const;

    /// Return component of the vector part of the field.
    Periodic_function<double>& vector(int i)
    {
        assert(i >= 0 && i <= 2);
        return *(components_[i + 1]);
    }

    /// Return component of the vector part of the field.
    Periodic_function<double> const& vector(int i) const
    {
        assert(i >= 0 && i <= 2);
        return *(components_[i + 1]);
    }

    Periodic_function<double>& component(int i)
    {
        assert(i >= 0 && i <= 3);
        return *(components_[i]);
    }

    /// Throws error in case of invalid access.
    Periodic_function<double>& component_raise(int i)
    {
        if (components_[i] == nullptr) {
            throw std::runtime_error("invalid access");
        }
        return *(components_[i]);
    }

    Periodic_function<double> const& component(int i) const
    {
        assert(i >= 0 && i <= 3);
        return *(components_[i]);
    }

    void zero();

    void fft_transform(int direction__);

    Simulation_context& ctx()
    {
        return ctx_;
    }
};

} // namespace sirius

#endif
