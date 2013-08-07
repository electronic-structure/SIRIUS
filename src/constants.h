#ifndef __CONSTANTS_H__
#define __CONSTANTS_H__

#include "atomic_conf.h"
#include "atomic_symb.h"

// NIST value for the inverse fine structure (http://physics.nist.gov/cuu/Constants/index.html)
const double speed_of_light = 137.035999074; 

const double pi = 3.1415926535897932385;

const double twopi = 6.2831853071795864769;

const double fourpi = 12.566370614359172954;

const double y00 = 0.28209479177387814347;

const double ha2ev = 27.21138505;

const double pw_cutoff_default = 16.0; 

const double aw_cutoff_default = 7.0;

const int lmax_apw_default = 8;

const int lmax_rho_default = 7;

const int lmax_pot_default = 7;

const char* storage_file_name = "sirius.h5";

const int _dim_k_ = 0;

const int _dim_row_ = 1;

const int _dim_col_ = 2;

#endif // __CONSTANTS_H__

