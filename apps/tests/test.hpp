#include <sirius.h>

using namespace sirius;

class Measurement: public std::vector<double>
{
  public:

    double average() const
    {
        double d = 0;
        for (size_t i = 0; i < this->size(); i++) {
            d += (*this)[i];
        }
        d /= static_cast<double>(this->size());
        return d;
    }

    double sigma() const
    {
        double avg = average();
        double variance = 0;
        for (size_t i = 0; i < this->size(); i++) {
            variance += std::pow((*this)[i] - avg, 2);
        }
        variance /= static_cast<double>(this->size());
        return std::sqrt(variance);
    }
};

template <typename T>
dmatrix<T> random_symmetric(int N__, int bs__, BLACS_grid const& blacs_grid__)
{
    dmatrix<T> A(N__, N__, blacs_grid__, bs__, bs__);
    dmatrix<T> B(N__, N__, blacs_grid__, bs__, bs__);
    for (int j = 0; j < A.num_cols_local(); j++) {
        for (int i = 0; i < A.num_rows_local(); i++) {
            A(i, j) = utils::random<T>();
        }
    }

#ifdef __SCALAPACK
    linalg(linalg_t::scalapack).tranc(N__, N__, A, 0, 0, B, 0, 0);
#else
    for (int i = 0; i < N__; i++) {
        for (int j = 0; j < N__; j++) {
            B(i, j) = utils::conj(A(j, i));
        }
    }
#endif

    for (int j = 0; j < A.num_cols_local(); j++) {
        for (int i = 0; i < A.num_rows_local(); i++) {
            A(i, j) = 0.5 * (A(i, j) + B(i, j));
        }
    }

    for (int i = 0; i < N__; i++) {
        A.set(i, i, 50.0);
    }

    return A;
}

template <typename T>
dmatrix<T> random_positive_definite(int N__, int bs__, BLACS_grid const& blacs_grid__)
{
    double p = 1.0 / N__;
    dmatrix<T> A(N__, N__, blacs_grid__, bs__, bs__);
    dmatrix<T> B(N__, N__, blacs_grid__, bs__, bs__);
    for (int j = 0; j < A.num_cols_local(); j++) {
        for (int i = 0; i < A.num_rows_local(); i++) {
            A(i, j) = p * utils::random<T>();
        }
    }

#ifdef __SCALAPACK
    linalg(linalg_t::scalapack).tranc(N__, N__, A, 0, 0, B, 0, 0);
#else
    for (int i = 0; i < N__; i++) {
        for (int j = 0; j < N__; j++) {
          B(i, j) = utils::conj(A(j, i));
        }
    }
#endif
    linalg(linalg_t::scalapack).gemm('C', 'N', N__, N__, N__, &linalg_const<T>::one(), A, 0, 0, A, 0, 0,
        &linalg_const<T>::zero(), B, 0, 0);

    for (int i = 0; i < N__; i++) {
        B.set(i, i, N__);
    }

    return B;
}

