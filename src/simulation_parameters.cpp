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

/** \file simulation_parameters.cpp
 *
 *  \brief Contains implementation of sirius::Simulation_parameters class.
 */

#include "simulation_parameters.hpp"
#include "SDDK/communicator.hpp"

namespace sirius {

void Simulation_parameters::import(std::string const& str__)
{
    if (str__.size() == 0) {
        return;
    }

    json dict = utils::read_json_from_file_or_string(str__);

    /* read unit cell */
    unit_cell_input_.read(dict);
    /* read parameters of mixer */
    mixer_input_.read(dict);
    /* read parameters of iterative solver */
    iterative_solver_input_.read(dict);
    /* read controls */
    control_input_.read(dict);
    /* read parameters */
    parameters_input_.read(dict);
    /* read settings */
    settings_input_.read(dict);
    /* read hubbard parameters */
    hubbard_input_.read(dict);
}

void Simulation_parameters::import(json const& dict)
{
    if (dict.size() == 0) {
        return;
    }
    /* read unit cell */
    unit_cell_input_.read(dict);
    /* read parameters of mixer */
    mixer_input_.read(dict);
    /* read parameters of iterative solver */
    iterative_solver_input_.read(dict);
    /* read controls */
    control_input_.read(dict);
    /* read parameters */
    parameters_input_.read(dict);
    /* read settings */
    settings_input_.read(dict);
    /* read hubbard parameters */
    hubbard_input_.read(dict);
}

void Simulation_parameters::import(cmd_args const& args__)
{
    control_input_.processing_unit_ = args__.value("control.processing_unit", control_input_.processing_unit_);
    control_input_.mpi_grid_dims_   = args__.value("control.mpi_grid_dims", control_input_.mpi_grid_dims_);
    control_input_.std_evp_solver_name_ =
        args__.value("control.std_evp_solver_name", control_input_.std_evp_solver_name_);
    control_input_.gen_evp_solver_name_ =
        args__.value("control.gen_evp_solver_name", control_input_.gen_evp_solver_name_);
    control_input_.fft_mode_     = args__.value("control.fft_mode", control_input_.fft_mode_);
    control_input_.memory_usage_ = args__.value("control.memory_usage", control_input_.memory_usage_);
    control_input_.verbosity_    = args__.value("control.verbosity", control_input_.verbosity_);
    control_input_.verification_ = args__.value("control.verification", control_input_.verification_);

    parameters_input_.ngridk_      = args__.value("parameters.ngridk", parameters_input_.ngridk_);
    parameters_input_.gamma_point_ = args__.value("parameters.gamma_point", parameters_input_.gamma_point_);
    parameters_input_.pw_cutoff_   = args__.value("parameters.pw_cutoff", parameters_input_.pw_cutoff_);

    iterative_solver_input_.orthogonalize_ =
        args__.value("iterative_solver.orthogonalize", iterative_solver_input_.orthogonalize_);
}

void Simulation_parameters::set_core_relativity(std::string name__)
{
    parameters_input_.core_relativity_ = name__;

    std::map<std::string, relativity_t> const m = {{"none", relativity_t::none}, {"dirac", relativity_t::dirac}};

    if (m.count(name__) == 0) {
        std::stringstream s;
        s << "wrong type of core relativity: " << name__;
        TERMINATE(s);
    }
    core_relativity_ = m.at(name__);
}

void Simulation_parameters::set_valence_relativity(std::string name__)
{
    parameters_input_.valence_relativity_ = name__;

    std::map<std::string, relativity_t> const m = {{"none", relativity_t::none},
                                                   {"zora", relativity_t::zora},
                                                   {"iora", relativity_t::iora},
                                                   {"koelling_harmon", relativity_t::koelling_harmon}};

    if (m.count(name__) == 0) {
        std::stringstream s;
        s << "wrong type of valence relativity: " << name__;
        TERMINATE(s);
    }
    valence_relativity_ = m.at(name__);
}

void Simulation_parameters::set_processing_unit(std::string name__)
{
    std::transform(name__.begin(), name__.end(), name__.begin(), ::tolower);

    /* set the default value */
    if (name__ == "") {
        if (acc::num_devices() > 0) {
            name__ = "gpu";
        } else {
            name__ = "cpu";
        }
    }
    control_input_.processing_unit_ = name__;
    if (name__ == "cpu") {
        this->set_processing_unit(device_t::CPU);
    } else if (name__ == "gpu") {
        this->set_processing_unit(device_t::GPU);
    } else {
        std::stringstream s;
        s << "wrong processing unit name: " << name__;
        TERMINATE(s);
    }
}

void Simulation_parameters::set_processing_unit(device_t pu__)
{
    if (acc::num_devices() == 0) {
        processing_unit_                = device_t::CPU;
        control_input_.processing_unit_ = "cpu";
    } else {
        processing_unit_ = pu__;
        if (pu__ == device_t::CPU) {
            control_input_.processing_unit_ = "cpu";
        } else if (pu__ == device_t::GPU) {
            control_input_.processing_unit_ = "gpu";
        } else {
            std::stringstream s;
            s << "wrong processing unit type";
            TERMINATE(s);
        }
    }
}

void Simulation_parameters::print_options() const
{
    json const& dict = get_options_dictionary();

    if (Communicator::world().rank() == 0) {
        std::cout << "the sirius library or the mini apps can be initialized through the interface" << std::endl;
        std::cout << "using the api directly or through a json dictionary. The following contains " << std::endl;
        std::cout << "a description of all the runtime options, that can be used directly to      " << std::endl;
        std::cout << "initialize sirius.                                                          " << std::endl;

        for (auto& el : dict.items()) {
            std::cout << "============================================================================\n";
            std::cout << "                                                                              ";
            std::cout << "                      section : " << el.key() << "                             \n";
            std::cout << "                                                                            \n";
            std::cout << "============================================================================\n";

            for (size_t s = 0; s < dict[el.key()].size(); s++) {
                std::cout << "name of the option : " << dict[el.key()][s]["name"].get<std::string>() << std::endl;
                std::cout << "description : " << dict[el.key()][s]["description"].get<std::string>() << std::endl;
                if (dict[el.key()][s].count("possible_values")) {
                    const auto& v = dict[el.key()][s]["description"].get<std::vector<std::string>>();
                    std::cout << "possible values : " << v[0];
                    for (size_t st = 1; st < v.size(); st++)
                        std::cout << " " << v[st];
                }
                std::cout << "default value : " << dict[el.key()]["default_values"].get<std::string>() << std::endl;
            }
        }
    }
    Communicator::world().barrier();
}

void Simulation_parameters::electronic_structure_method(std::string name__)
{
    parameters_input_.electronic_structure_method_ = name__;

    std::map<std::string, electronic_structure_method_t> m = {
        {"full_potential_lapwlo", electronic_structure_method_t::full_potential_lapwlo},
        {"pseudopotential", electronic_structure_method_t::pseudopotential}};

    if (m.count(name__) == 0) {
        std::stringstream s;
        s << "wrong type of electronic structure method: " << name__;
        TERMINATE(s);
    }
    electronic_structure_method_ = m[name__];
}
} // namespace sirius
