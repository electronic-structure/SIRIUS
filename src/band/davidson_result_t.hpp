#ifndef __DAVIDSON_RESULT_T_HPP__
#define __DAVIDSON_RESULT_T_HPP__

namespace sirius {

/// Result of Davidson solver.
struct davidson_result_t {
    int niter;
    sddk::mdarray<double, 2> eval;
    bool converged;
    int num_unconverged[2];
};

}

#endif
