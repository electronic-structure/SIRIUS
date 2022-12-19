/**
 * @file sqnm.hpp
 * @author Moritz Gubler (moritz.gubler@unibas.ch)
 * @brief Implementation of the SQNM method. More informations about the algorithm can be found here: https://aip.scitation.org/doi/10.1063/1.4905665
 * @date 2022-07-13
 * 
 */

#ifndef SQNM_HPP
#define SQNM_HPP
#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include "historylist.hpp"

namespace sqnm_space
{
  class SQNM {
    private:
    int ndim;
    int nhistx;
    double eps_subsp = 1.e-3;
    double alpha0 = 1.e-2;
    std::unique_ptr<hlist_space::HistoryList> xlist;
    std::unique_ptr<hlist_space::HistoryList> flist;
    double alpha;
    Eigen::VectorXd dir_of_descent;
    double prev_f;
    Eigen::VectorXd prev_df_dx;
    Eigen::VectorXd expected_positions;
    Eigen::MatrixXd h_subsp;
    Eigen::MatrixXd h_evec_subsp;
    Eigen::MatrixXd h_evec;
    Eigen::VectorXd h_eval;
    Eigen::VectorXd res;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> esolve;
    Eigen::VectorXd res_temp;
    int nhist = 0;
    bool estimate_step_size = false;

    public:
    /**
     * @brief Construct a new SQNM::SQNM object using default parameters
     * 
     * @param ndim_ number of dimensions of target function
     * @param nhistx_ Maximal number of steps that will be stored in the history list. Use a value between 3 and 20. Must be <= than ndim_.
     * @param alpha_ initial step size. default is 1.0. For systems with hard bonds (e.g. C-C) use a value between and 1.0 and 2.5
     * Should be approximately the inverse of the largest eigenvalue of the Hessian matrix.
     * If alpha is negative, the inial step size is estimated using the mechanism from section 6.4 of the
     * vc-sqnm paper: https://arxiv.org/abs/2206.07339
     * beta will then be equal to minus alpha. Good choices for beta are 0.1 in hartee / bohr^2 and
     * 0.001 in eV / A^2
     */
    SQNM(int ndim_, int nhistx_, double alpha_) {
      ndim = ndim_;
      nhistx = nhistx_;
      if (alpha_ <= 0)
      {
        this->estimate_step_size = true;
        this->alpha = -alpha_;
      } else
      {
        alpha = alpha_;
      }

      xlist = std::make_unique<hlist_space::HistoryList>(ndim, nhistx);
      flist = std::make_unique<hlist_space::HistoryList>(ndim, nhistx);
    }

    /**
     * @brief Construct a new SQNM::SQNM object using custom parameters.
     * 
     * @param ndim_ number of dimensions of target function
     * @param nhistx_ Maximal number of steps that will be stored in the history list. Use a value between 3 and 20. Must be <= than ndim_.
     * @param alpha_ initial step size. default is 1.0. For systems with hard bonds (e.g. C-C) use a value between and 1.0 and 2.5.
     * Should be approximately the inverse of the largest eigenvalue of the Hessian matrix.
     * If alpha is negative, the inial step size is estimated using the mechanism from section 6.4 of the
     * vc-sqnm paper: https://arxiv.org/abs/2206.07339
     * beta will then be equal to minus alpha. Good choices for beta are 0.1 in hartee / bohr^2 and
     * 0.001 in eV / A^2
     * @param alpha0_  * @param alpha0 Lower limit on the step size. 1.e-2 is the default.
     * @param eps_subsp_ Lower limit on linear dependencies of basis vectors in history list. Default 1.e-4.
     */
    SQNM(int ndim_, int nhistx_, double alpha_, double alpha0_, double eps_subsp_) {
      ndim = ndim_;
      nhistx = nhistx_;
      if (alpha_ <= 0)
      {
        this->estimate_step_size = true;
        this->alpha = -alpha_;
      } else
      {
        alpha = alpha_;
      }
      xlist = std::make_unique<hlist_space::HistoryList>(ndim, nhistx);
      flist = std::make_unique<hlist_space::HistoryList>(ndim, nhistx);
      alpha0 = alpha0_;
      eps_subsp = eps_subsp_;
    }

    /**
     * @brief Calculates new coordinates that are closer to local minimum that the current coordinates. This function should be used the following way:
     * 1. calculate f(x) and the derivative.
     * 2. call the step function.
     * 3. add return value of step function to x.
     * 4. repeat.
     * 
     * @param x Current position vector
     * @param f_of_x value of the target function evaluated at position x.
     * @param df_dx derivative of the target function evaluated at x.
     * @return VectorXd displacent that can be added to x in order to get new improved coordinates.
     */
    Eigen::VectorXd step(Eigen::VectorXd &x, double &f_of_x, Eigen::VectorXd &df_dx) {

      // check if forces are zero. If so zero is returned because a local minimum has already been found.
      if (df_dx.norm() <= 10.0e-13)
      {
        this->dir_of_descent.setZero(ndim);
        return this->dir_of_descent;
      }

      nhist = xlist->add(x);
      flist->add(df_dx);
      if (nhist == 0) { // initial and first step
        this->dir_of_descent = - alpha * df_dx;
      } else 
      {
        // check if positions have been changed and print a warning if they were.
        if ((x - expected_positions).norm() > 10e-8)
        {
          std::cerr << "SQNM was not called with positions that were expected. If this was not done on purpose, it is probably a bug.\n";
          std::cerr << "Were atoms that left the simulation box put back into the cell? This is not allowed.\n";
        }

        if (this->estimate_step_size)
        {
          double prev_df_squared = std::pow(prev_df_dx.norm(), 2);
          double l1 = (f_of_x - prev_f + alpha * prev_df_squared) / (0.5 * alpha * alpha * prev_df_squared);
          double l2 = (df_dx - prev_df_dx).norm() / (alpha * prev_df_dx.norm());
          alpha = 1.0 / std::max(l1, l2);
          std::cout << "Automatic initial step size guess: " << alpha << '\n';
          this->estimate_step_size = false;
        } else
        {
          double gainratio = calc_gainratio(f_of_x);
          adjust_stepsize(gainratio);
        }

        Eigen::MatrixXd S = calc_ovrlp();
        esolve.compute(S);
        Eigen::VectorXd S_eval = esolve.eigenvalues();
        Eigen::MatrixXd S_evec = esolve.eigenvectors();

        // compute eq 8
        int dim_subsp = 0;
        for (int i = 0; i < S_eval.size(); i++){
          if (S_eval(i) / S_eval(nhist-1) > eps_subsp)
          {
            dim_subsp+=1;
          }
        }
        // remove dimensions from subspace
        for (int i = 0; i < dim_subsp; i++)
        {
          S_evec.col(i) = S_evec.col(nhist - dim_subsp + i);
          S_eval(i) = S_eval(nhist - dim_subsp + i);
        }
        
        Eigen::MatrixXd dr_subsp(ndim, dim_subsp);
        dr_subsp.setZero();
        for (int i = 0; i < dim_subsp; i++) {
          for (int ihist = 0; ihist < nhist; ihist++){
            dr_subsp.col(i) += S_evec(ihist, i) * xlist->normalized_difflist.col(ihist);
          }
          dr_subsp.col(i) /= sqrt(S_eval(i));
        }

        // compute eq. 11
        Eigen::MatrixXd df_subsp(ndim, dim_subsp);
        df_subsp.setZero();
        for (int i = 0; i < dim_subsp; i++) {
          for (int ihist = 0; ihist < nhist; ihist++){
            df_subsp.col(i) += S_evec(ihist, i) * flist->difflist.col(ihist) / xlist->difflist.col(ihist).norm();
          }
          df_subsp.col(i) /= sqrt(S_eval(i));
        }
        // compute eq. 13
        h_subsp = .5 * (df_subsp.transpose() * dr_subsp + dr_subsp.transpose() * df_subsp);
        esolve.compute(h_subsp);
        h_eval = esolve.eigenvalues();
        h_evec_subsp = esolve.eigenvectors();

        // compute eq. 15
        h_evec.resize(ndim, dim_subsp);
        h_evec.setZero();
        for (int i = 0; i < dim_subsp; i++){
          for (int k = 0; k < dim_subsp; k++){
            h_evec.col(i) += h_evec_subsp(k, i) * dr_subsp.col(k);
          }
        }

        // compute residues (eq. 20)
        res.resize(dim_subsp);
        for (int j = 0; j < dim_subsp; j++){
          res_temp = - h_eval(j) * h_evec.col(j);
          for (int k = 0; k < dim_subsp; k++){
            res_temp += h_evec_subsp(k, j) * df_subsp.col(k);
          }
          res(j) = res_temp.norm();
        }

        // modify eigenvalues (eq. 18)
        for (int idim = 0; idim < dim_subsp; idim++){
          h_eval(idim) = sqrt(pow(h_eval(idim), 2) + pow(res(idim), 2));
        }
        
        // decompose gradient (eq. 16)
        dir_of_descent = df_dx;
        for (int i = 0; i < dim_subsp; i++){
          dir_of_descent -= h_evec.col(i).dot(df_dx) * h_evec.col(i);
        }
        dir_of_descent *= alpha;

        // appy preconditioning to subspace gradient (eq. 21)
        for (int idim = 0; idim < dim_subsp; idim++)
        {
          dir_of_descent += (df_dx.dot(h_evec.col(idim)) / h_eval(idim)) * h_evec.col(idim);
        }
        dir_of_descent *= -1.0;
        
      }
      expected_positions = x + dir_of_descent;
      prev_f = f_of_x;
      prev_df_dx = df_dx;
      return this->dir_of_descent;
    }

    /**
     * @brief Estimates a lower bound of the energy of the local minimum
     * 
     * @return double Lower bound estimate
     */
    double lower_bound()
    {
      if (this->nhist == 0) {
        std::cout << "No lower bound estimate can be given yet.";
        return 0.0;
      }
      return this->prev_f - 0.5 * this->prev_df_dx.dot(this->prev_df_dx) / this->h_eval(0);
    }

    private:
    double calc_gainratio(double &f){
      double gr = (f - prev_f) / ( .5 * this->dir_of_descent.dot(prev_df_dx));
      return gr;
    }

    void adjust_stepsize(double &gainratio){
      if ( gainratio < 0.5 ) alpha = std::max(alpha * 0.65, alpha0);
      else if(gainratio > 1.05) alpha = alpha * 1.05;
    }

    Eigen::MatrixXd calc_ovrlp(){
      Eigen::MatrixXd S = xlist->normalized_difflist.block(0,0, ndim, nhist).transpose() * xlist->normalized_difflist.block(0,0, ndim, nhist);
      return S;
    }
  };
}
#endif