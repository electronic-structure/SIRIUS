// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
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

/** \file simulation_context.h
 *
 *  \brief Contains definition and implementation of Simulation_parameters and Simulation_context classes.
 */

#ifndef __SIMULATION_CONTEXT_H__
#define __SIMULATION_CONTEXT_H__

#include "simulation_context_base.h"
#include "mpi_grid.hpp"
#include "step_function.h"
#include "version.h"
#include "augmentation_operator.h"

namespace sirius {

/// Simulation context is a set of parameters and objects describing a single simulation.
/** The order of initialization of the simulation context is the following: first, the default parameter
 *  values are set in the constructor, then (optionally) import() method is called and the parameters are
 *  overwritten with the those from the input file, and finally, the user sets the values with set_...() metods.
 *  Then the unit cell can be populated and the context can be initialized. */
class Simulation_context : public Simulation_context_base
{
  private:
    /// Step function is used in full-potential methods.
    std::unique_ptr<Step_function> step_function_;

    std::vector<Augmentation_operator> augmentation_op_;

    /* copy constructor is forbidden */
    Simulation_context(Simulation_context const&) = delete;

  public:
    Simulation_context()
        : Simulation_context_base(mpi_comm_world())
    {
    }

    Simulation_context(std::string const& str__)
        : Simulation_context_base(str__, mpi_comm_world())
    {
    }

    Simulation_context(Communicator const& comm__)
        : Simulation_context_base(comm__)
    {
    }

    Simulation_context(std::string const& str__, Communicator const& comm__)
        : Simulation_context_base(str__, comm__)
    {
    }

    /// Initialize the similation (can only be called once).
    void initialize()
    {
        PROFILE("sirius::Simulation_context::initialize");

        Simulation_context_base::initialize();

        if (full_potential()) {
            step_function_ = std::unique_ptr<Step_function>(new Step_function(*this));
        }

        if (!full_potential()) {

            if (comm().rank() == 0 && control().print_memory_usage_) {
                MEMORY_USAGE_INFO();
            }

            /* create augmentation operator Q_{xi,xi'}(G) here */
            for (int iat = 0; iat < unit_cell().num_atom_types(); iat++) {
                augmentation_op_.push_back(std::move(Augmentation_operator(*this, iat)));

                if (comm().rank() == 0 && control().print_memory_usage_) {
                    MEMORY_USAGE_INFO();
                }
            }
        }

        if (comm().rank() == 0 && control().print_memory_usage_) {
            MEMORY_USAGE_INFO();
        }
    }

    Step_function const& step_function() const
    {
        return *step_function_;
    }

    inline Augmentation_operator const& augmentation_op(int iat__) const
    {
        return augmentation_op_[iat__];
    }

    inline Augmentation_operator& augmentation_op(int iat__)
    {
        return augmentation_op_[iat__];
    }
};

} // namespace sirius

#endif
