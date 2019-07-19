#include "memory.hpp"
#include "linalg.hpp"
#include "wave_functions.hpp"

#if defined(__GPU)
extern "C" void residuals_aux_gpu(int num_gvec_loc__,
                                  int num_res_local__,
                                  int* res_idx__,
                                  double* eval__,
                                  double_complex const* hpsi__,
                                  double_complex const* opsi__,
                                  double const* h_diag__,
                                  double const* o_diag__,
                                  double_complex* res__,
                                  double* res_norm__,
                                  double* p_norm__,
                                  int gkvec_reduced__,
                                  int mpi_rank__);

extern "C" void compute_residuals_gpu(double_complex* hpsi__,
                                      double_complex* opsi__,
                                      double_complex* res__,
                                      int num_gvec_loc__,
                                      int num_bands__,
                                      double* eval__);

extern "C" void apply_preconditioner_gpu(double_complex* res__,
                                         int num_rows_loc__,
                                         int num_bands__,
                                         double* eval__,
                                         const double* h_diag__,
                                         const double* o_diag__);

extern "C" void make_real_g0_gpu(double_complex* res__,
                                 int ld__,
                                 int n__);
#endif

using namespace sddk;

namespace sirius {

template <typename T>
int
residuals(memory_t mem_type__, linalg_t la_type__, int ispn__, int N__, int num_bands__, mdarray<double, 1>& eval__,
          dmatrix<T>& evec__, Wave_functions& hphi__, Wave_functions& ophi__, Wave_functions& hpsi__,
          Wave_functions& opsi__, Wave_functions& res__, mdarray<double, 2> const& h_diag__,
          mdarray<double, 2> const& o_diag__, bool estimate_eval__, double norm_tolerance__,
          std::function<bool(int, int)> is_converged__);

}
