/** \file generate_w90_coeffs.hpp
 *
 *  \brief Interface to W90 library.
 */
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
 *  \f{eqnarray*}{
 *  M_{nn'} &= \int e^{-i{\bf qr}}  \psi_{n{\bf k}}^{*} ({\bf r})  \psi_{n'{\bf k+q}} ({\bf r}) d{\bf r} =
 *    \sum_{\bf G} e^{-i({\bf G+k}){\bf r}} C_{n{\bf k}}^{*}({\bf G})
 *    \sum_{\bf G'} e^{i({\bf G'+k+q}){\bf r}} C_{n{\bf k+q}}({\bf G'}) e^{-i{\bf qr}} = \\
 *    &= \sum_{\bf GG'} \int e^{i({\bf G'-G}){\bf r}} d{\bf r}  C_{n{\bf k}}^{*}({\bf G}) C_{n{\bf k+q}}({\bf G'}) =
 *    \sum_{\bf G}  C_{n{\bf k}}^{*}({\bf G}) C_{n{\bf k+q}}({\bf G})
 *  \f}
 *
 *  Let's rewrite \f$ {\bf k + q} = {\bf \tilde G} + {\bf \tilde k} \f$. Now, through the property of plane-wave
 *  expansion coefficients \f$ C_{n{\bf k+q}}({\bf G}) = C_{n{\bf \tilde k}}({\bf G + \tilde G}) \f$ it follows that
 *  \f[
 *    M_{nn'} = \sum_{\bf G} C_{n{\bf k}}^{*}({\bf G}) C_{n{\bf \tilde k}}({\bf G + \tilde G})
 *  \f]
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
// we are going to compute <psi_{n,k} | S exp{-iqr} | psi_{n',k+q}>
// where S = 1 + \sum_{\alpha} \sum_{\xi, \xi'} |beta_{\xi}^{\alpha} Q_{\xi,\xi'}^{\alpha} <beta_{\xi'}^{\alpha}|
//
// the inner product splits into following contributions:
// <psi_{n,k} | 1 + |beta>Q<beta|  psi_{n',k+q}> = <psi_{n,k} | exp^{-iqr} | psi_{n',k+q}> +
// <psi_{n,k} | exp^{-iqr} |beta>Q<beta|  psi_{n',k+q}>
//
// we will need: |psi_{n',k+q}> in the order of G+k vectors
//               <beta_{\xi'}^{\alpha}|  psi_{n',k+q}> computed at k+q
//
// we can then apply the Q matrix to <beta_{\xi'}^{\alpha}|  psi_{j,k+q}> and compute 1st and 2nd contributions
// as two matrix multiplications.
//
//
// For the ultrasoft contribution (2nd term):
//   construct the matrix of <beta_{\xi'}^{\alpha}| psi_{n',k'}>, where k'+G'=k+q for all local k-points;
//   exchange information between MPI ranks as is done for the wave-functions
//
//
// 1st step: get a list of q-vectors for each k-point and a G' vector that bring k+q back into 1st Brilloun zone
//
// 2nd step: compute <beta_{\xi'}^{\alpha}|  psi_{j,k+q}>; check how this is done in the Beta_projector class;
// Q-operator can be applied here. Look how this is done in Non_local_operator::apply();
// (look for Beta_projectors_base::inner() function; understand the "chunks" of beta-projectors
//
// 3nd step: copy wave-function at k+q (k') into an auxiliary wave-function object of G+k order and see how
// the G+k+q index can be reshuffled. Check the implementation of G-vector class which handles all the G- and G+k-
// indices
//
// 4th step: allocate resulting matrix M_{nn'}, compute contribution from C*C (1st part) using wf::inner() function;
// compute contribution from ultrasoft part using a matrix-matrix multiplication
//
// 5th step: parallelize over k-points
//
// 6ts step: parallelize over G+k vectors and k-points

}

#endif
