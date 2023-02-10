/** \file generate_w90_coeffs.hpp
 *
 *  \brief Interface to W90 library.
 */
#ifndef __GENERATE_W90_COEFFS_HPP__
#define __GENERATE_W90_COEFFS_HPP__

#include "k_point_set.hpp"
#include "k_point_set.cpp"


//extern "C"{
//void wannier_setup_(const char*, int32_t*, int32_t*,
//                    const double*, const double*, double*, int32_t*,//care! arg (4,5) changed with const
//                    int32_t*, char(*) [3], double*, bool*, bool*,
//                    int32_t*, int32_t*, int32_t*, int32_t*, int32_t*,
//                    double*, int32_t*, int32_t*, int32_t*, double*,
//                    double*, double*, int32_t*, int32_t*, double*,
//                    size_t, size_t);
//
//void wannier_run_(const char*, int32_t*, int32_t*,
//                  double*, double*, double*, int32_t*,
//                  int32_t*, int32_t*, int32_t*, char(*) [3],
//                  double*, bool*, std::complex<double>*, std::complex<double>*, double*,
//                  std::complex<double>*, std::complex<double>*, bool*, double*,
//                  double*, double*,
//                  size_t, size_t);
//
//}

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
void generate_w90_coeffs(sirius::K_point_set& k_set__)
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
// this is the library equivalent step of producing nnkp file from w90

    std::cout << "\n\n\nwannierization!!!!\n\n\n";
    
    //parameters needed for wannier_setup_
    size_t length_seedname = 256;              //aux variable for the length of a string 
    int32_t num_kpts;                          //input        
    int32_t num_bands_tot;                     //input      
    int32_t num_atoms;                         //input         
    size_t length_atomic_symbol = 3;           //aux, as expected from wannier90 lib          
    bool gamma_only;                           //input                   
    bool spinors;                              //input                    
    int32_t num_bands;                         //output                  
    int32_t num_wann;                          //output                      
    int32_t nntot;                             //output
    int32_t num_nnmax = 12;                    //aux variable for max number of neighbors             
                                               //fixed, as in pw2wannier or in wannier90 docs

    //initializing input variables from local variables
    num_kpts = k_set__.num_kpoints();                              
    num_bands_tot = k_set__.get<double>(0)->spinor_wave_functions().num_wf();      
    num_atoms = k_set__.ctx().unit_cell().num_atoms();                      
    gamma_only = k_set__.ctx().gamma_point();
    spinors = false;//right now, generate_wave_functions only works with noncolin (from SIRIUS docs)!
    //WARNING we need to compare with .win file!!!


    //non-scalar variables - definition + space allocation
    char seedname[length_seedname];                                         //input  
    //sddk::mdarray<int32_t ,1> mp_grid(3);                                 //input  
    //sddk::mdarray<double,2> real_lattice(3,3);                            //input       
    //sddk::mdarray<double,2> recip_lattice(3,3);                           //input         
    sddk::mdarray<double,2> kpt_lattice(3,num_kpts);                        //input             
    char atomic_symbol[num_atoms][3];                                       //input               
    sddk::mdarray<double,2> atoms_cart(3,num_atoms);                        //input
    sddk::mdarray<int,2> nnlist(num_kpts,num_nnmax);                        //output                   
    sddk::mdarray<int32_t ,3> nncell(3,num_kpts,num_nnmax);                 //output                 
    sddk::mdarray<double,2> proj_site(3,num_bands_tot);                     //output                    
    sddk::mdarray<int32_t ,1> proj_l(num_bands_tot);                        //output               
    sddk::mdarray<int32_t ,1> proj_m(num_bands_tot);                        //output                   
    sddk::mdarray<int32_t ,1> proj_radial(num_bands_tot);                   //output                   
    sddk::mdarray<double,2> proj_z(3,num_bands_tot);                        //output                   
    sddk::mdarray<double,2> proj_x(3,num_bands_tot);                        //output                    
    sddk::mdarray<double,1> proj_zona(num_bands_tot);                       //output                   
    sddk::mdarray<int32_t ,1> exclude_bands(num_bands_tot);                 //output                   
    sddk::mdarray<int32_t ,1> proj_s(num_bands_tot);                        //output - optional        
    sddk::mdarray<double,2> proj_s_qaxis(3,num_bands_tot);                  //output - optional        
    //end non-scalar variables


    //initializing non-scalar variables
    std::string aux = "diamond.lib"; 
    strcpy(seedname, aux.c_str());
    length_seedname = aux.length();
   
    //copying the k fractional coordinates to a contiguous array
    for(int ik=0; ik < num_kpts; ik++){
	    for(int icoor=0; icoor<3; icoor++){
	    kpt_lattice(icoor, ik) = k_set__.get<double>(ik)->vk()[icoor]; 
	    }
    }
    //initializing atomic_symbol and atomic_cart
    for(int iat=0; iat<num_atoms; iat++){
        std::fill(atomic_symbol[iat],atomic_symbol[iat]+3,' ');//check!!!!!!
        std::strcpy(atomic_symbol[iat], k_set__.ctx().unit_cell().atom(iat).type().label().c_str());
        
	//position is saved in fractional coordinates, we need cartesian for wannier_setup_
        auto* frac_coord = &k_set__.ctx().unit_cell().atom(iat).position();
        auto cart_coord = k_set__.ctx().unit_cell().get_cartesian_coordinates(*frac_coord);
        for (int icoor=0; icoor<3; icoor++){
	    atoms_cart(icoor,iat) = cart_coord[icoor];
	    }
    }
    //end parameters needed for wannier_setup_
    
    sirius::wannier_setup_(seedname,
                   k_set__.ctx().cfg().parameters().ngridk().data(),                // input              
                   &num_kpts,                                                       // input                      
                   &(k_set__.ctx().unit_cell().lattice_vectors()(0,0)),             // input        
                   &(k_set__.ctx().unit_cell().reciprocal_lattice_vectors()(0,0)),  // input    
                   kpt_lattice.at(sddk::memory_t::host),                            // input            
                   &num_bands_tot,                                                  // input                     
                   &num_atoms,                                                      // input            
                   atomic_symbol,                                                   // input                
                   atoms_cart.at(sddk::memory_t::host),                             // input           
                   &gamma_only,                                                     // input                 
                   &spinors,                                                        // input                 
                   &nntot,                                                          // output                  
                   nnlist.at(sddk::memory_t::host),                                 // output           
                   nncell.at(sddk::memory_t::host),                                 // output                 
                   &num_bands,                                                      // output                         
                   &num_wann,                                                       // output                    
                   proj_site.at(sddk::memory_t::host),                              // output               
                   proj_l.at(sddk::memory_t::host),                                 // output                        
                   proj_m.at(sddk::memory_t::host),                                 // output                           
                   proj_radial.at(sddk::memory_t::host),                            // output                             
                   proj_z.at(sddk::memory_t::host),                                 // output                 
                   proj_x.at(sddk::memory_t::host),                                 // output                    
                   proj_zona.at(sddk::memory_t::host),                              // output          
                   exclude_bands.at(sddk::memory_t::host),                          // output               
                   proj_s.at(sddk::memory_t::host),                                 // output                 
                   proj_s_qaxis.at(sddk::memory_t::host),                           // output               
                   length_seedname,                                                 // aux-length of a string                
                   length_atomic_symbol);                                           // aux-length of a string                     




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
