/**
 * @file periodic_optimizer.hpp
 * @author Moritz Gubler (moritz.gubler@unibas.ch)
 * @brief Implementation of the vc-sqnm method. More informations about the algorithm can be found here: https://arxiv.org/abs/2206.07339
 * @date 2022-07-13
 * 
 */

#ifndef PERIODIC_OPTIMIZER_HPP
#define PERIODIC_OPTIMIZER_HPP
#include <Eigen/Dense>
#include <iostream>
#include "sqnm.hpp"


namespace PES_optimizer{

  class periodic_optimizer
  {
    private:
    Eigen::Matrix3d initial_latttice;
    Eigen::Matrix3d initial_latttice_inv;
    Eigen::Matrix3d lattice_transformer;
    Eigen::Matrix3d lattice_transformer_inv;
    std::unique_ptr<sqnm_space::SQNM> opt;
    int nat;
    int ndim;
    bool opt_lattice;
    double initial_step_size = 1;
    int n_hist_max = 10;
    double w = 2.0;
    double f_std_deviation = 0.0;

    public:

    double get_w(){
      return w;
    }

    int get_n_hist_max(){
      return n_hist_max;
    }

    double get_initial_step_size(){
      return initial_step_size;
    }

    /**
     * @brief Construct a new periodic optimizer::periodic optimizer object for free or fixed cell optimization with default parameters.
     * 
     * @param nat number of atoms
     */
    periodic_optimizer(int &nat)
    {
      this->nat = nat;
      this->ndim = 3*nat;
      this->opt_lattice = false;
      this->opt = std::make_unique<sqnm_space::SQNM>(ndim, n_hist_max, initial_step_size);
    }

    /**
     * @brief Construct a new periodic optimizer::periodic optimizer object for free or fixed cell optimization with custom parameters.
     * 
     * @param nat number of atoms
     * @param initial_step_size initial step size. default is 1.0. For systems with hard bonds (e.g. C-C) use a value between and 1.0 and
     * 2.5. If a system only contains weaker bonds a value up to 5.0 may speed up the convergence.
     * @param nhist_max Maximal number of steps that will be stored in the history list. Use a value between 3 and 20. Must be <= than 3*nat.
     * @param alpha0 Lower limit on the step size. 1.e-2 is the default.
     * @param eps_subsp Lower limit on linear dependencies of basis vectors in history list. Default 1.e-4.
     */
    periodic_optimizer(int &nat, double initial_step_size, int nhist_max, double alpha0, double eps_subsp)
    {
      this->nat = nat;
      this->ndim = 3*nat;
      this->initial_step_size = initial_step_size;
      this->n_hist_max = nhist_max;
      this->opt_lattice = false;
      this->opt = std::make_unique<sqnm_space::SQNM>(ndim, n_hist_max, initial_step_size, alpha0, eps_subsp);
    }

    /**
     * @brief Construct a new periodic optimizer::periodic optimizer object for variable cell shape optimization with default parameters.
     * 
     * @param nat number of atoms
     * @param lat_a first lattice vector
     * @param lat_b second lattice vector
     * @param lat_c third lattice vector
     */
    periodic_optimizer(int &nat, Eigen::Vector3d &lat_a, Eigen::Vector3d &lat_b, Eigen::Vector3d &lat_c)
    {
      setupInitialLattice(nat, lat_a, lat_b, lat_c);
      this->opt = std::make_unique<sqnm_space::SQNM>(ndim, n_hist_max, initial_step_size);
    }

    /**
     * @brief Construct a new periodic optimizer::periodic optimizer object for variable cell shape optimization with custom parameters.
     * 
     * @param nat number of atoms
     * @param lat_a first lattice vector
     * @param lat_b second lattice vector
     * @param lat_c third lattice vector
    * @param initial_step_size initial step size. default is 1.0. For systems with hard bonds (e.g. C-C) use a value between and 1.0 and
    * 2.5. If a system only contains weaker bonds a value up to 5.0 may speed up the convergence.
    * @param nhist_max Maximal number of steps that will be stored in the history list. Use a value between 3 and 20. Must be <= than 3*nat + 9.
    * @param lattice_weight weight / size of the supercell that is used to transform lattice derivatives. Use a value between 1 and 2. Default is 2.
    * @param alpha0 Lower limit on the step size. 1.e-2 is the default.
    * @param eps_subsp Lower limit on linear dependencies of basis vectors in history list. Default 1.e-4.
    */
    periodic_optimizer(int &nat, Eigen::Vector3d &lat_a, Eigen::Vector3d &lat_b, Eigen::Vector3d &lat_c, double initial_step_size, int nhist_max, double lattice_weight, double alpha0, double eps_subsp)
    {
      this->w = lattice_weight;
      this->n_hist_max = nhist_max;
      this->initial_step_size = initial_step_size;
      setupInitialLattice(nat, lat_a, lat_b, lat_c);
      this->opt = std::make_unique<sqnm_space::SQNM>(ndim, n_hist_max, initial_step_size, alpha0, eps_subsp);
    }

    /**
     * @brief Calculates new atomic coordinates that are closer to the local minimum. Fixed cell optimization. This function should be used the following way:
     * 1. calculate energies and forces at positions r.
     * 2. call the step function to update positions r.
     * 3. repeat.
     * 
     * @param r Input: atomic coordinates, dimension(3, nat). Output: improved coordinates that are calculated based on forces from this and previous iterations.
     * @param energy Potential energy of the system in it's current state
     * @param f Forces of the system in it's current state. dimension(3, nat)
     */
    void step(Eigen::MatrixXd &r, double &energy, Eigen::MatrixXd &f){
      if (opt_lattice)
      {
        std::cout << "The fixed cell step function was called even though the object was created for vc-relaxation. returning" << "\n";
        return;
      }
      check_forces(f);
      Eigen::VectorXd pos_all = Eigen::Map<Eigen::VectorXd>(r.data(), 3*nat);
      Eigen::VectorXd force_all = - Eigen::Map<Eigen::VectorXd>(f.data(), 3*nat);
      pos_all += opt->step(pos_all, energy, force_all);
      r = Eigen::Map<Eigen::MatrixXd>(pos_all.data(), 3, nat);
    }

    /**
     * @brief Calculates new atomic coordinates that are closer to the local minimum. Variable cell shape optimization. This function should be used the following way:
     * 1. calculate energies, forces and stress tensor at positions r and lattice vectors a, b, c.
     * 2. call the step function to update positions r and lattice vectors.
     * 3. repeat.
     * 
     * @param r Input: atomic coordinates, dimension(3, nat). Output: improved coordinates that are calculated based on forces from this and previous iterations.
     * @param energy Potential energy of the system in it's current state
     * @param f Forces of the system in it's current state. dimension(3, nat)
     * @param lat_a first lattice vector
     * @param lat_b second lattice vector
     * @param lat_c third lattice vector
     * @param stress stress tensor of the system in it' current state.
     */
    void step(Eigen::MatrixXd &r, double &energy, Eigen::MatrixXd &f, Eigen::Vector3d &lat_a, Eigen::Vector3d &lat_b, Eigen::Vector3d &lat_c, Eigen::Matrix3d &stress){
        if (! opt_lattice)
      {
        std::cout << "The vc step function was called even though the object was created for fixed cell relaxation. returning" << "\n";
        return;
      }
      check_forces(f);
      Eigen::Matrix3d alat;
      Eigen::MatrixXd alat_tilde;
      alat.col(0) = lat_a;
      alat.col(1) = lat_b;
      alat.col(2) = lat_c;

      //  calculate transformed coordinates
      Eigen::MatrixXd q(3, nat);
      Eigen::MatrixXd dq(3, nat);
      q = initial_latttice * alat.inverse() * r;
      dq = - alat * this->initial_latttice_inv * f;

      //cout << "transform lattice vectors" << endl;
      // transform lattice vectors
      alat_tilde = alat * lattice_transformer;
      Eigen::MatrixXd dalat = calc_lattice_derivatices(stress, alat) * lattice_transformer_inv;
      Eigen::VectorXd qall = combine_matrices(q, alat_tilde);
      Eigen::VectorXd dqall = combine_matrices(dq, dalat);
      
      //cout << "update coordinates" << endl;
      qall += this->opt->step(qall, energy, dqall);

      split_matrices(q, alat_tilde, qall);
      alat = alat_tilde * lattice_transformer_inv;
      r = alat * this->initial_latttice_inv * q;
      lat_a = alat.col(0);
      lat_b = alat.col(1);
      lat_c = alat.col(2);
    }

    void check_forces(Eigen::MatrixXd forces)
    {
      double fnoise = forces.rowwise().sum().norm() / sqrt(3 * this->nat);
      if (this->f_std_deviation == 0)
      {
        this->f_std_deviation = fnoise;
      } else {
        this->f_std_deviation = .8 * this->f_std_deviation + .2 * fnoise;
      }
      if (this->f_std_deviation > 0.2 * forces.cwiseAbs().maxCoeff()) {
        std::cerr << "Noise in force is larger than 0.2 times the larges force component. Convergence cannot be guaranteed.";
      }
    }

    /**
     * @brief Estimates a lower bound of the energy of the local minimum
     * 
     * @return double Lower bound estimate
     */
    double lower_bound()
    {
      return this->opt->lower_bound();
    }

    private:
    Eigen::VectorXd combine_matrices(Eigen::MatrixXd a, Eigen::MatrixXd b){
      Eigen::VectorXd result(this->ndim);
      for (int i = 0; i < 3*nat; i++)
      {
        result(i) = a(i);
      }
      int j = 0;
      for (int i = 3*nat; i < ndim; i++)
      {
        result(i) = b(j);
        ++j; 
      }
      return result;
    }

    void split_matrices(Eigen::MatrixXd &a, Eigen::MatrixXd &b, Eigen::VectorXd &c){
      for (int i = 0; i < 3*nat; i++)
      {
        a(i) = c(i);
      }
      int j = 0;
      for (int i = 3*nat; i < ndim; i++)
      {
        b(j) = c(i);
        j++;
      }
    }

    Eigen::Matrix3d calc_lattice_derivatices(Eigen::Matrix3d &stress, Eigen::Matrix3d &alat){
      Eigen::Matrix3d da = - stress * alat.inverse().transpose() * alat.determinant();
      return da;
    }

    void setupInitialLattice(int &nat, Eigen::Vector3d &lat_a, Eigen::Vector3d &lat_b, Eigen::Vector3d &lat_c)
    {
      this->nat = nat;
      this->ndim = 3*nat + 9;
      this->initial_latttice.col(0) = lat_a;
      this->initial_latttice.col(1) = lat_b;
      this->initial_latttice.col(2) = lat_c;
      this->initial_latttice_inv = initial_latttice.inverse();
      lattice_transformer.setZero();
      for (int i = 0; i < 3; i++)
      {
        lattice_transformer(i, i) = 1.0 / (initial_latttice.col(i).norm());
      }
      lattice_transformer = lattice_transformer * (w * sqrt(nat));
      lattice_transformer_inv = lattice_transformer.inverse();
      this->opt_lattice = true;
    }
  };
}
#endif
