// Copyright (c) 2013-2021 Anton Kozhevnikov, Thomas Schulthess
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
#include "core/mpi/communicator.hpp"
#include "context/input_schema.hpp"

#include <unordered_set>
#include <iterator>
#include <sstream>

namespace sirius {
template <typename T>
static std::ostringstream
option_print_vector__(const std::vector<T>& vec)
{
    std::ostringstream out;
    if (!vec.empty()) {
        out << "[" << vec[0];
        for (auto i = 0u; i < vec.size(); i++) {
            out << ", " << vec[i];
        }
        out << "]";
    }
    return out;
}
/// Compose JSON dictionary with default parameters based on input schema.
/** Traverse the JSON schema and add nodes with default parameters to the output dictionary. The nodes without
 *  default parameters are ignored. Still, user has a possibility to add the missing nodes later by providing a
 *  corresponding input JSON dictionary. See compose_json() function. */
void
compose_default_json(nlohmann::json const& schema__, nlohmann::json& output__)
{
    for (auto it : schema__.items()) {
        auto key = it.key();
        /* this is a final node with the description of the data type */
        if (it.value().contains("type") && it.value()["type"] != "object") {
            /* check if default parameter is present */
            if (it.value().contains("default")) {
                output__[key] = it.value()["default"];
            }
        } else { /* otherwise continue to traverse the schema */
            if (!output__.contains(key)) {
                output__[key] = nlohmann::json{};
            }
            if (it.value().contains("properties")) {
                compose_default_json(it.value()["properties"], output__[key]);
            }
        }
    }
}

/// Append the input dictionary to the existing dictionary.
/** Use JSON schema to traverse the existing dictionary and add on top the values from the input dictionary. In this
 *  way we can add missing nodes which were not defined in the existing dictionary. */
void
compose_json(nlohmann::json const& schema__, nlohmann::json const& in__, nlohmann::json& inout__)
{
    std::unordered_set<std::string> visited;

    for (auto it : in__.items()) {
        visited.insert(it.key());
    }

    for (auto it : schema__.items()) {
        auto key = it.key();

        // Remove visited items.
        auto found = visited.find(key);
        if (found != visited.end()) {
            visited.erase(found);
        }

        /* this is a final node with the description of the data type */
        if (it.value().contains("type") && it.value()["type"] != "object") {
            if (in__.contains(key)) {
                /* copy the new input */
                inout__[key] = in__[key];
            }
        } else { /* otherwise continue to traverse the schema */
            /* not simple data type : a section with parameter, a dictionary map, etc.*/
            if (it.value().contains("properties")) {
                compose_json(it.value()["properties"], in__.contains(key) ? in__[key] : nlohmann::json{}, inout__[key]);
            } else if (in__.contains(key)) {
                inout__[key] = in__[key];
            } else if (!inout__.contains(key)) {
                inout__[key] = nlohmann::json();
            }
        }
    }

    // Emit warnings about keys that were set but unused.
    if (!visited.empty()) {
        std::stringstream ss;
        ss << "The following configuration parameters were not recognized and ignored: ";
        std::copy(visited.begin(), visited.end(), std::ostream_iterator<std::string>(ss, " "));
        RTE_WARNING(ss)
    }
}

Config::Config()
{
    /* initialize JSON dictionary with default parameters */
    compose_default_json(sirius::input_schema["properties"], this->dict_);
}

void
Config::import(nlohmann::json const& in__)
{
    /* overwrite the parameters by the values from the input dictionary */
    compose_json(sirius::input_schema["properties"], in__, this->dict_);
}

/// Get all possible options for initializing sirius. It is a json dictionary.
nlohmann::json const&
get_options_dictionary()
{
    if (input_schema.size() == 0) {
        RTE_THROW("Dictionary not initialized");
    }
    return input_schema;
}

/// Get all possible options of a given input section. It is a json dictionary.
nlohmann::json const&
get_section_options(std::string const& section__)
{
    if (input_schema.size() == 0) {
        RTE_THROW("Dictionary not initialized");
    }
    return input_schema["properties"][section__]["properties"];
}

void
Simulation_parameters::import(std::string const& str__)
{
    auto json = read_json_from_file_or_string(str__);
    import(json);
}

void
Simulation_parameters::import(nlohmann::json const& dict__)
{
    cfg_.import(dict__);
}

void
Simulation_parameters::import(cmd_args const& args__)
{
    cfg_.control().processing_unit(args__.value("control.processing_unit", cfg_.control().processing_unit()));
    cfg_.control().mpi_grid_dims(args__.value("control.mpi_grid_dims", cfg_.control().mpi_grid_dims()));
    cfg_.control().std_evp_solver_name(
            args__.value("control.std_evp_solver_name", cfg_.control().std_evp_solver_name()));
    cfg_.control().gen_evp_solver_name(
            args__.value("control.gen_evp_solver_name", cfg_.control().gen_evp_solver_name()));
    cfg_.control().fft_mode(args__.value("control.fft_mode", cfg_.control().fft_mode()));
    cfg_.control().verbosity(args__.value("control.verbosity", cfg_.control().verbosity()));
    cfg_.control().verification(args__.value("control.verification", cfg_.control().verification()));

    cfg_.parameters().ngridk(args__.value("parameters.ngridk", cfg_.parameters().ngridk()));
    cfg_.parameters().gamma_point(args__.value("parameters.gamma_point", cfg_.parameters().gamma_point()));
    cfg_.parameters().pw_cutoff(args__.value("parameters.pw_cutoff", cfg_.parameters().pw_cutoff()));
    cfg_.parameters().gk_cutoff(args__.value("parameters.gk_cutoff", cfg_.parameters().gk_cutoff()));

    cfg_.iterative_solver().early_restart(
            args__.value("iterative_solver.early_restart", cfg_.iterative_solver().early_restart()));
    cfg_.mixer().beta(args__.value("mixer.beta", cfg_.mixer().beta()));
    cfg_.mixer().type(args__.value("mixer.type", cfg_.mixer().type()));
}

void
Simulation_parameters::core_relativity(std::string name__)
{
    cfg_.parameters().core_relativity(name__);
    core_relativity_ = get_relativity_t(name__);
}

void
Simulation_parameters::valence_relativity(std::string name__)
{
    cfg_.parameters().valence_relativity(name__);
    valence_relativity_ = get_relativity_t(name__);
}

void
Simulation_parameters::processing_unit(std::string name__)
{
    /* set the default value */
    if (name__ == "auto") {
        if (acc::num_devices() > 0) {
            name__ = "gpu";
        } else {
            name__ = "cpu";
        }
    }
    cfg_.control().processing_unit(name__);
    processing_unit_ = get_device_t(name__);
}

void
Simulation_parameters::smearing(std::string name__)
{
    cfg_.parameters().smearing(name__);
    smearing_ = smearing::get_smearing_t(name__);
}

void
Simulation_parameters::electronic_structure_method(std::string name__)
{
    cfg_.parameters().electronic_structure_method(name__);

    std::map<std::string, electronic_structure_method_t> m = {
            {"full_potential_lapwlo", electronic_structure_method_t::full_potential_lapwlo},
            {"pseudopotential", electronic_structure_method_t::pseudopotential}};

    if (m.count(name__) == 0) {
        std::stringstream s;
        s << "wrong type of electronic structure method: " << name__;
        RTE_THROW(s);
    }
    electronic_structure_method_ = m[name__];
}
} // namespace sirius
