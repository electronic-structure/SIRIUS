#ifndef __SIRIUS_H__
#define __SIRIUS_H__

#define stop(info_string) { info_string; std::cout << std::endl << "  line " << __LINE__ << " of file " \
  << __FILE__ << std::endl; throw std::runtime_error("Abort execution");}

#include <vector>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include "config.h"
#include "mdarray.h"
#include "linalg.h"
#include "radial_grid.h"
#include "spline.h"
#include "radial_solver.h"

#endif // __SIRIUS_H__
