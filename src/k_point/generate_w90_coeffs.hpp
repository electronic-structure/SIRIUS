#ifndef __GENERATE_W90_COEFFS_HPP__
#define __GENERATE_W90_COEFFS_HPP__

#include "k_point_set.hpp"

/// Generate the necessary data for the W90 input.
/** Wave-functions:
 * \f[
 *  \psi_{n{\bf k}} ({\bf r}) = \sum_{\bf G} e^{i({\bf G+k}){\bf r}} C_{n{\bf k}}({\bf G})
 * \f]
 *
 *  Matrix elements:
 *  \f[
 *  M_{nn'} = \int e^{-i{\bf qr}}  \psi_{n{\bf k}}^{*} ({\bf r})  \psi_{n'{\bf k+q}} ({\bf r}) d{\bf r}
 *  \f]
 *
 *
 *
 */
void generate_w90_coeffs(K_point_set const& k_set__)
{

// phase1: k-point exchange
// each MPI rank sores the local set of k-points
// for each k-point we have a list of q vectors to compute k+q. In general we assume that the number
// of q-points nq(k) is nefferent for each k
// The easy way to implement send/recieve of k-points is through brute-force broadcast:
// each MPI rank broadcasts one-by-one each of its local k-points. Everyone listens and recieves the data;
// only MPI ranks that need the broadcasted point as k+q are storing it in the local array. Yes, there is
// some overhead in moving data between the MPI ranks, but this can be optimized later.
//
// phase1 is not required intially for the sequential code
//
// phase2: construnction of the k+q wave-functions and bringin them to the order of G+k G-vectors
//
// we are going to compute <u_{i,k} | S exp{-iqr} | u_{j,k+q}>
// where S = 1 + \sum_{\alpha} \sum_{\xi, \xi'} |beta_{\xi}^{\alpha} Q_{\xi,\xi'}^{\alpha} <beta_{\xi'}^{\alpha}|
//
// the inner product splits into following contributions:
// <u_{i,k} | 1 + |beta>Q<beta|  u_{j,k+q}> = <u_{i,k} | exp^{-iqr} | u_{j,k+q}> +
// <u_{i,k} | exp^{-iqr} |beta>Q<beta|  u_{j,k+q}>
//
// we will need: |u_{j,k+q}> in the order of G+k vectors
//               <beta_{\xi'}^{\alpha}|  u_{j,k+q}> computed at k+q
//
// we can then apply the Q matrix to <beta_{\xi'}^{\alpha}|  u_{j,k+q}> and compute 1st and 2nd contributions
// as two matrix multiplications.
//



}

#endif
