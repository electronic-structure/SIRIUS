#ifndef NLCGLIB_ADAPTOR_H
#define NLCGLIB_ADAPTOR_H

#include <memory>
#include <nlcglib/interface.hpp>
#include <cmath>

#include "k_point/k_point_set.hpp"
#include "density/density.hpp"
#include "potential/potential.hpp"
#include "SDDK/wave_functions.hpp"

namespace sirius {

class Matrix : public nlcglib::MatrixBaseZ
{
  public:
    Matrix(const std::vector<buffer_t>& data, const std::vector<kindex_t>& indices, MPI_Comm mpi_comm = MPI_COMM_SELF)
        : data(data)
        , indices(indices)
        , mpi_comm(mpi_comm)
    {
    }

    Matrix(std::vector<buffer_t>&& data, std::vector<kindex_t>&& indices, MPI_Comm mpi_comm = MPI_COMM_SELF)
        : data{std::forward<std::vector<buffer_t>>(data)}
        , indices{std::forward<std::vector<kindex_t>>(indices)}
        , mpi_comm(mpi_comm)
    { /* empty */
    }

    buffer_t get(int i) override;
    const buffer_t get(int i) const override;

    int size() const override
    {
        return data.size();
    };

    MPI_Comm mpicomm(int i) const override
    {
        return data[i].mpi_comm;
    }

    MPI_Comm mpicomm() const override
    {
        return mpi_comm;
    }

    kindex_t kpoint_index(int i) const override
    {
        return indices[i];
    }

  private:
    std::vector<buffer_t> data;
    std::vector<kindex_t> indices;
    MPI_Comm mpi_comm;
};


/// TODO: Array1d owns data...
class Array1d : public nlcglib::VectorBaseZ
{
  public:
    Array1d(const std::vector<std::vector<double>>& data, const std::vector<kindex_t>& indices, MPI_Comm mpi_comm = MPI_COMM_SELF)
        : data(data)
        , indices(indices)
        , mpi_comm(mpi_comm)
    {
    }

    Array1d(std::vector<std::vector<double>>&& data, std::vector<kindex_t>&& indices, MPI_Comm mpi_comm = MPI_COMM_SELF)
        : data{std::forward<decltype(data)>(data)}
        , indices{std::forward<decltype(indices)>(indices)}
        , mpi_comm(mpi_comm)
    {
    }

    buffer_t get(int i) override;
    const buffer_t get(int i) const override;

    int size() const override
    {
        return data.size();
    };

    MPI_Comm mpicomm(int i) const override
    {
        // this object is never distributed
        return MPI_COMM_SELF;
    }

    MPI_Comm mpicomm() const override
    {
        return mpi_comm;
    }

    kindex_t kpoint_index(int i) const override
    {
        assert(i < static_cast<int>(indices.size()));
        return indices[i];
    }

  private:
    std::vector<std::vector<double>> data;
    std::vector<kindex_t> indices;
    MPI_Comm mpi_comm;
};

class Scalar : public nlcglib::ScalarBaseZ
{
  public:
    Scalar(const std::vector<double>& data__, const std::vector<kindex_t>& indices__,
           MPI_Comm mpi_comm = MPI_COMM_SELF)
        : data(data__)
        , indices(indices__)
        , mpi_comm(mpi_comm)
    {
    }

    Scalar(std::vector<double>&& data__, std::vector<kindex_t>&& indices__,
           MPI_Comm mpi_comm = MPI_COMM_SELF)
        : data{std::forward<decltype(data)>(data__)}
        , indices{std::forward<decltype(indices)>(indices__)}
        , mpi_comm(mpi_comm)
    {
    }

    buffer_t get(int i) override
    {
        return data[i];
    }

    const buffer_t get(int i) const override
    {
        return data[i];
    }

    int size() const override
    {
        return data.size();
    };

    MPI_Comm mpicomm(int i) const override
    {
        // this object is never distributed
        return MPI_COMM_SELF;
    }

    MPI_Comm mpicomm() const override
    {
        return mpi_comm;
    }

    kindex_t kpoint_index(int i) const override
    {
        return indices[i];
    }

  private:
    std::vector<double> data;
    std::vector<kindex_t> indices;
    MPI_Comm mpi_comm;
};

/// Kohn-Sham energy
class Energy : public nlcglib::EnergyBase
{
  public:
    Energy(K_point_set& kset, Density& density, Potential& potential);

    void set_occupation_numbers(const std::vector<std::vector<double>>& fn) override;
    void set_wfct(nlcglib::MatrixBaseZ& vector) override;
    int nelectrons() override;
    int occupancy() override;
    void compute() override;
    double get_total_energy() override;
    std::shared_ptr<nlcglib::MatrixBaseZ> get_hphi() override;
    std::shared_ptr<nlcglib::MatrixBaseZ> get_sphi() override;
    std::shared_ptr<nlcglib::MatrixBaseZ> get_C(nlcglib::memory_type) override;
    std::shared_ptr<nlcglib::VectorBaseZ> get_fn() override;
    void set_fn(const std::vector<std::vector<double>>& fn) override;
    std::shared_ptr<nlcglib::VectorBaseZ> get_ek() override;
    std::shared_ptr<nlcglib::VectorBaseZ> get_gkvec_ekin() override;
    std::shared_ptr<nlcglib::ScalarBaseZ> get_kpoint_weights() override;
    void print_info() const override;

  private:
    K_point_set& kset;
    Density& density;
    Potential& potential;
    std::vector<std::shared_ptr<sddk::Wave_functions>> hphis;
    std::vector<std::shared_ptr<sddk::Wave_functions>> sphis;
    std::vector<std::shared_ptr<sddk::Wave_functions>> cphis;
    double etot{std::nan("1")};
};

} // namespace sirius

#endif /* NLCGLIB_ADAPTOR_H */
