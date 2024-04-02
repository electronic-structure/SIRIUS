A collection of small tests to check different types of calculations.

To submit tests manually on Piz Daint:

```bash
export MKL_NUM_THREADS=12
export OMP_NUM_THREADS=12
export CRAY_CUDA_MPS=0
export MPIRUN="srun -N4 -n4 -c12 --unbuffered --hint=nomultithread"
salloc -N4 -C gpu --time=60:00 -A csstaff

./run_tests_parallel_k_gpu.x
# or
./run_tests_parallel_gpu.x
```
| Folder | Structure | K-points | Potential type | XC type | Magnetism | Remarks |
|-|-|-|-|-|-|-|
| test01 | SrVO3   | \[2, 2, 2\] | USPP           | LDA (PZ)  | non-magnetic  | - ground state <br> - forces <br> - stress |
| test02 | He atom | \[1, 1, 1\] | full-potential | LDA (VWN) | non-magnetic  | test of iterative solver|
| test03 | Fe      | \[4, 4, 4\] | PAW            | GGA (PBE) | FM collinear  | |
| test04 | LiF     | \[4, 4, 4\] | PAW            | LDA (PZ)  | non-magnetic  | - ground state <br> - forces <br> - stress |
| test05 | NiO     | \[2, 2, 2\] | USPP           | LDA (PZ)  | AFM collinear | - ground state <br> - forces <br> - stress |
| test06 | Fe      | \[2, 2, 2\] | USPP           | LDA (PZ)  | FM collinear  | - ground state <br> - forces <br> - stress |
| test07 | Ni      | \[4, 4, 4\] | USPP           | GGA (BPE) | FM collinear  | - ground state <br> - forces <br> - stress |
| test08 | Si      | \[1, 1, 1\] | USPP           | LDA (PZ)  | non-magnetic  | - Gamma point treatment is off <br> - complex wave-functions <br> - forces <br> - stress |
| test09 | Ni      | \[2, 2, 2\] | USPP           | LDA(PZ)   | non-collinear | - ground state <br> - forces <br> - stress |
| test10 | Au      | \[2, 2, 2\] | NC <br> with SO correction   | LDA (PZ) | non-collinear | - test of SO correction <br> - no symmetry |
| test11 | Au      | \[2, 2, 2\] | USPP <br> with SO correction | LDA (PZ) | non-collinear | - test of SO correction <br> - no symmetry |
| test12 | C       | \[2, 2, 2\] | full-potential | LDA (PZ)  | non-magnetic  | |
| test14 | SrVO3   | \[2, 2, 2\] | USPP           | GGA (PBE) | non-magnetic  | - ground state <br> - forces <br> - stress |
| test15 | LiF     | \[1, 1, 1\] | PAW            | LDA (PZ)  | non-magnetic  | - Gamma-point calculation <br> - low symmetry <br> - forces <br> - stress |
| test16 | NiO     | \[2, 2, 2\] | full-potential | LDA (PZ)  | AFM collinear | |
| test17 | NiO     | \[2, 2, 2\] | full-potential | GGA (PBE) | non-magnetic  | - forces |
| test18 | YN      | \[2, 2, 2\] | full-potential | LDA (PZ)  | non-magnetic  | - IORA treatment of valence states |
| test19 | Fe      | \[4, 4, 4\] | full-potential | LDA (PW)  | FM collinear  | |
| test20 | H2O     | \[1, 1, 1\] | full-potential | LDA (VWN) | non-magnetic  | - water molecule in a box <br> - no relativity for core and valence |
| test21 | FeSi    | \[2, 2, 2\] | NC             | GGA (PBE) | FM collinear  | - Fermi-Dirac smearing |
| test22 | NiO     | \[4, 4, 4\] | USPP <br> Hubbard correction | GGA (PBE) | AFM collinear | - Hubbard local-U correction <br> - simplified treatment |
| test23 | H atom  | \[2, 2, 2\] | NC             | LDA (PZ) | non-magnetic |  - ground state <br> - forces <br> - stress |
| test24 | NiO     | \[4, 4, 4\] | USPP <br> Hubbard correction | GGA (PBE) | AFM collinear | - Hubbard U+V correction |
| test25 | NiO     | \[4, 4, 4\] | USPP <br> Hubbard correction | GGA (PBE) | AFM collinear | - Hubbard local-U corection <br> - full orthogonalization of atomic orbitals <br> - forces |
| test26 | NiO     | \[4, 4, 4\] | USPP <br> Hubbard correction | GGA (PBE) | AFM collinear | - Hubbard local-U correction <br> - full orthogonalization of atomic orbitals <br> - forces |
| test27 | LiCoO2  | \[2, 2, 2\] | USPP <br> Hubbard correction | GGA (PBE-sol) | non-magnetic | - Hubbard U+V correction <br> - full orthogonalization of atomic orbitals <br> - full treatment of Hubbard correction |
| test28 | LiCoO2  | \[2, 2, 2\] | USPP <br> Hubbard correction | GGA (PBE-sol) | non-magnetic | - Hubbard U+V correction <br> - full orthogonalization of atomic orbitals <br> - simplified treatment of Hubbard correction |
| test29 | NiO     | \[2, 2, 2\] | - Ni: USPP <br> - O: PAW <br> Hubbard correction |  GGA (PBE-sol) | AFM collinear | - Hubbard U+V correction |
| test30 | NiO     | \[2, 2, 2\] | USPP <br> Hubbard correction | GGA (PBE) | non-magnetic | - Constrained Hubbard potential <br> - full orthogonalization of atomic orbitals |
| test31 | H       | \[2, 2, 2\] | full-potential | LDA (PZ) | non-magnetic | - test of Koelling-Harmon radial solver |
| test32 | SrVO3   | \[2, 2, 2\] | USPP, PAW & NC | GGA (PBE)| non-magnetic | testing the parsing of UPF v2 files with pugixml |
