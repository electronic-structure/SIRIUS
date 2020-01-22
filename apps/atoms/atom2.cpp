// Copyright (c) 2013-2016 Anton Kozhevnikov, Thomas Schulthess
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

#include <sirius.h>
#include <Unit_cell/free_atom.hpp>

using namespace sirius;

int main(int argn, char **argv)
{
    sirius::initialize(true);

    /* handle command line arguments */
    cmd_args args;
    args.register_key("--symbol=", "{string} symbol of a chemical element");
    args.register_key("--type=", "{lo1, lo2, lo3, LO1, LO2} type of local orbital basis");
    args.register_key("--core=", "{double} cutoff for core states: energy (in Ha, if <0), radius (in a.u. if >0)");
    args.register_key("--order=", "{int} order of augmentation");
    args.register_key("--apw_enu=", "{double} default value for APW linearization energies");
    args.register_key("--auto_enu", "allow search of APW linearization energies");
    args.register_key("--xml", "xml output for Exciting code");
    args.register_key("--rel", "use scalar-relativistic solver");
    args.parse_args(argn, argv);

    if (argn == 1 || args.exist("help")) {
        std::printf("\n");
        std::printf("Atom (L)APW+lo basis generation.\n");
        std::printf("\n");
        std::printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        std::printf("\n");
        std::printf("Definition of the local orbital types:\n");
        std::printf("  lo  : 2nd order local orbitals composed of u(E) and udot(E),\n");
        std::printf("        where E is the energy of the bound-state level {n,l}\n");
        std::printf("  LO  : 3rd order local orbitals composed of u(E), udot(E) and u(E1),\n");
        std::printf("        where E and E1 are the energies of the bound-state levels {n,l} and {n+1,l}\n");
        std::printf("\n");
        std::printf("Examples:\n");
        std::printf("\n");
        std::printf("  generate default basis for lithium:\n");
        std::printf("    ./atom --symbol=Li\n");
        std::printf("\n");
        std::printf("  generate high precision basis for titanium:\n");
        std::printf("    ./atom --type=lo+LO --symbol=Ti\n");
        std::printf("\n");
        std::printf("  make all states of iron to be valence:\n");
        std::printf("    ./atom --core=-1000 --symbol=Fe\n");
        std::printf("\n");
        return 0;
    }

    auto symbol = args.value<std::string>("symbol");

    double core_cutoff = args.value<double>("core", -10.0);

    std::string lo_type = args.value<std::string>("type", "lo1");

    int apw_order = args.value<int>("order", 2);

    double apw_enu = args.value<double>("apw_enu", 0.15);

    bool auto_enu = args.exist("auto_enu");

    bool write_to_xml = args.exist("xml");

    bool rel = args.exist("rel");

    Free_atom fa(symbol);

    fa.ground_state(1e-6, 1e-6, rel);

    std::string recipe("{ \
        \"lo\" : [ {\"n\" : 0, \"o\" : 0}, {\"n\" : 0, \"o\" : 1}] \
    }");
    fa.generate_local_orbitals(recipe);

    auto& v = fa.free_atom_potential();
    FILE* fout = fopen("v.dat", "w");
    for (int ir = 0; ir < v.num_points(); ir++) {
        double x = v[ir];
        fprintf(fout, "%18.12f %18.12f\n", x, v(ir));
    }
    fclose(fout);

    sirius::finalize();
}
