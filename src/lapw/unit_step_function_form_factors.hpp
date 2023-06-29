#ifndef __UNIT_STEP_FUNCTION_FORM_FACTORS_HPP__
#define __UNIT_STEP_FUNCTION_FORM_FACTORS_HPP__

namespace sirius {

/// Utility function to generate LAPW unit step function.
inline double
unit_step_function_form_factors(double R__, double g__)
{
    if (g__ < 1e-12) {
        return std::pow(R__, 3) / 3.0;
    } else {
        return (std::sin(g__ * R__) - g__ * R__ * std::cos(g__ * R__)) / std::pow(g__, 3);
    }
}

}

#endif
