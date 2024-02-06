# Utility to generate full-potential atomic species files

Utility is doing the following steps:
  - find ground state of an atom using spherical approximation for the effective potential
  - split atomic states into "core" and "valence"
  - generate atomic JSON file containing description of the (L)APW+lo bassis

Run `pyton atoms.py` to generate `atoms.json` and the batch script `run.x`.

Run `./atom` to generate json input file for a specific atom.

Run batch script `./run.x` to generate species files for all atoms. Optionaly, redirect the
stderr stream to a file (`./run.x 2> err.dat`) to save the absoule total energy error with respect to NIST data.

WARNING! NIST total energy is computed with Vosko-Wilk-Nusair correlation energy (availbale from LibXC as XC_LDA_C_VWN).

# Other files in this directory
 - `atoms.in` file with atomic configurations for each of the elements
 - `atoms.py` Python script that takes `atoms.in` and generates `atomic_data.hpp` header file
   and `atoms.json` JSON dictionary
 - `atomic_conf.hpp.in` template for CMake to read `atoms.json` and create `atomic_conf.hpp` header file
