/** \file generate_w90_coeffs.hpp
 *
 *  \brief Interface to W90 library.
 */
#ifdef SIRIUS_WANNIER90

#include <limits>
#include "dft/smearing.hpp"
#include "k_point/k_point.hpp"
#include "k_point/k_point_set.hpp"
#include "symmetry/get_irreducible_reciprocal_mesh.hpp"
#include "hamiltonian/non_local_operator.hpp"
#include "linalg/inverse_sqrt.hpp"
#include "generate_w90_coeffs.hpp"

namespace sirius {

void 
read_nnkp(const int& num_kpts, int& num_wann, int& nntot, sddk::mdarray<int, 2>& nnlist, 
    sddk::mdarray<int32_t, 3>& nncell, sddk::mdarray<int32_t, 1>& exclude_bands)
{
    std::ifstream readNNKP;
    readNNKP.open("silicon.nnkp");
    std::string line;
    //read file
    std::vector<std::string> file_content;
    while( std::getline(readNNKP,line) ){
        file_content.push_back(line);
    for(auto ch : line)
    {
        std::cout << ch << std::endl;
    }    
    }
    std::cout << "endl!\n";

    //read num_wann
    std::string string_to_check = "begin projections";
    auto iterator = std::find_if(file_content.begin(), file_content.end(), 
                                        [&, string_to_check](std::string& iter_file)
                                          {return ( string_to_check == iter_file ); });
    num_wann = std::atoi((*(iterator+1)).c_str());
    std::cout << "num_wann:" << num_wann;
    //read nnlist and nncell
    string_to_check = "begin nnkpts";
    iterator = std::find_if(file_content.begin(), file_content.end(), 
                                        [&, string_to_check](std::string& iter_file)
                                          {return ( string_to_check == iter_file ); });
    iterator++;
    nntot = std::atoi((*(iterator)).c_str());
    std::cout << "nntot:" << nntot;
    iterator++;
    int aux_int;
    std::cout << std::endl;
    for(int ik=0; ik<num_kpts; ik++){
        for(int ib=0; ib<nntot; ib++){
            std::stringstream split_line;
            split_line << *iterator;
            std::cout << split_line.str() << std::endl;
            split_line >> aux_int;
            assert(aux_int == ik+1);
            split_line >> nnlist(ik,ib);
            split_line >> nncell(0,ik,ib) >> nncell(1,ik,ib) >> nncell(2,ik,ib);
            iterator++;
            //std::cout << std::setw(6) << ik+1;
            //std::cout << std::setw(6) <<  nnlist(ik,ib);
            //std::cout << std::setw(7) << nncell(0,ik,ib);
            //std::cout << std::setw(4) << nncell(1,ik,ib);
            //std::cout << std::setw(4) << nncell(2,ik,ib);
            //std::cout << std::endl;
            //std::cout << std::endl;
        }
    }
}

/*
 * This function creates a file with extension ".amn" that can eventually be read by wannier90
 * to set the matrix Amn (not needed if we want to use the library)
*/
void
write_Amn(sddk::mdarray<std::complex<double>, 3> const& Amn, 
          int const& num_kpts, int const& num_bands, int const& num_wann)
{
    std::ofstream writeAmn;
    writeAmn.open("sirius.amn");
    std::string line;
    writeAmn << "#produced in sirius" << std::endl;
    writeAmn << std::setw(10) << num_bands;
    writeAmn << std::setw(10) << num_kpts;
    writeAmn << std::setw(10) << num_wann;
    writeAmn << std::endl;

    for (int ik = 0; ik < num_kpts; ik++) {
        for (int n = 0; n < num_wann; n++) {
            for (int m = 0; m < num_bands; m++) {
                writeAmn << std::fixed << std::setw(5) << m + 1;
                writeAmn << std::fixed << std::setw(5) << n + 1;
                writeAmn << std::fixed << std::setw(5) << ik + 1;
                writeAmn << std::fixed << std::setprecision(12) << std::setw(18) << Amn(m, n, ik).real();
                writeAmn << std::fixed << std::setprecision(12) << std::setw(18) << Amn(m, n, ik).imag();
                // writeAmn << std::fixed << std::setprecision(12) << std::setw(18) << abs(Amn(m, n, ik));
                writeAmn << std::endl;
            }
        }
    }
}

/*
 * This function creates a file with extension ".mmn" that can eventually be read by wannier90
 * to set the matrix Mmn (not needed if we want to use the library)
*/
void 
write_Mmn(sddk::mdarray<std::complex<double>, 4> const& M, 
          sddk::mdarray<int, 2> const& nnlist, sddk::mdarray<int32_t, 3> const& nncell,
          int const& num_kpts, int const& num_neighbors, int const& num_bands)
{
    std::ofstream writeMmn;
    writeMmn.open("sirius.mmn");
    writeMmn << "#produced in sirius" << std::endl;
    writeMmn << std::setw(10) << num_bands;
    writeMmn << std::setw(10) << num_kpts;
    writeMmn << std::setw(10) << num_neighbors;
    writeMmn << std::endl;
    for (int ik = 0; ik < num_kpts; ik++) {
        for (int ib = 0; ib < num_neighbors; ib++) {
            writeMmn << std::setw(5) << ik + 1;
            writeMmn << std::setw(5) << nnlist(ik, ib);
            writeMmn << std::setw(5) << nncell(0, ik, ib);
            writeMmn << std::setw(5) << nncell(1, ik, ib);
            writeMmn << std::setw(5) << nncell(2, ik, ib);
            writeMmn << std::endl;
            for (int n = 0; n < num_bands; n++) {
                for (int m = 0; m < num_bands; m++) {
                    writeMmn << std::fixed << std::setprecision(12) << std::setw(18) << M(m, n, ib, ik).real();
                    writeMmn << std::fixed << std::setprecision(12) << std::setw(18) << M(m, n, ib, ik).imag();
                    // writeMmn << std::fixed << std::setprecision(12) << std::setw(18) << abs(M(m, n, ib, ik));
                    writeMmn << std::endl;
                }
            }
        }
    }
    writeMmn.close();
}

/*
 * This function creates a file with extension ".eig" that can eventually be read by wannier90
 * to pass the energy eigenvalues if we need a window (not needed if we want to use the library)
*/
void
write_eig(sddk::mdarray<double, 2> const& eigval, 
          int const& num_bands, int const& num_kpts) 
{
    std::ofstream writeEig;
    writeEig.open("sirius.eig");
    for (int ik = 0; ik < num_kpts; ik++) {
        for (int iband = 0; iband < num_bands; iband++) {
            writeEig << std::setw(5) << iband + 1;
            writeEig << std::setw(5) << ik + 1;
            writeEig << std::fixed << std::setprecision(12) << std::setw(18) << eigval(iband, ik);
            writeEig << std::endl;
        }
    }
    writeEig.close();
}

/*
 * This function generates the Full Brillouin zone starting from the Irreducible wedge. 
 * The equation to satisfy is:
 * \f[
 *     {\bf k}_{fbz} + {\bf G} = R.{\bf k}_{ibz} 
 * \f] 
 */
void
from_irreduciblewedge_to_fullbrillouinzone(K_point_set& kset_ibz, K_point_set& kset_fbz, std::vector<k_info>& k_temp)
{
    PROFILE_START("sirius::K_point_set::generate_w90_coeffs::unfold_fbz");
    // Apply symmetry to all points of the IBZ. Save indices of ibz, fbz, sym
    for (int ik = 0; ik < kset_ibz.num_kpoints(); ik++) {
        for (int isym = 0; isym < kset_ibz.ctx().unit_cell().symmetry().size(); isym++) {
            auto& R = kset_ibz.ctx().unit_cell().symmetry()[isym].spg_op.R; // point symmetry rotation in crystal coordinates

            auto Rk = r3::dot(kset_ibz.get<double>(ik)->vk(), R);
            auto Rk_reduced = r3::reduce_coordinates(Rk);
            bool found = ( std::find_if(k_temp.begin(), k_temp.end(), 
                                        //[](){return false;}
                                        [&, Rk_reduced](k_info const& k)
                                          {return ( (k.fbz-Rk_reduced.first).length() < 1.e-08 ); }
                                       ) != k_temp.end() );
            if (!found) {
                k_info new_kpt;
                new_kpt.ibz     = kset_ibz.get<double>(ik)->vk();
                new_kpt.ik_ibz  = ik;
                new_kpt.fbz     = Rk_reduced.first;
                new_kpt.G       = r3::vector<double>({(double)Rk_reduced.second[0],(double)Rk_reduced.second[1],(double)Rk_reduced.second[2]});
                new_kpt.R       = kset_fbz.ctx().unit_cell().symmetry()[isym].spg_op.R;
                new_kpt.invR    = kset_fbz.ctx().unit_cell().symmetry()[isym].spg_op.invR;
                new_kpt.t       = kset_fbz.ctx().unit_cell().symmetry()[isym].spg_op.t;

                assert ( ( ( new_kpt.fbz - ( r3::dot( new_kpt.ibz, new_kpt.R ) - new_kpt.G ) ).length() < 1.e-08 ) );

                k_temp.push_back(new_kpt);
            }
        }// end isym
    }// end ik

    // remove additional G vector from k_temp. 
    for (int ik = 0; ik < (int)k_temp.size(); ik++) {
        if (k_temp[ik].G.length() > 1.e-08) {
            k_temp[ik].fbz += k_temp[ik].G;
            k_temp[ik].G = {0,0,0};
        }        
        assert ( ( ( k_temp[ik].fbz - ( r3::dot( k_temp[ik].ibz, k_temp[ik].R ) ) ).length() < 1.e-08 ) );
    }
    
    for (int ik = 0; ik < (int)k_temp.size(); ik++) {
        kset_fbz.add_kpoint(k_temp[ik].fbz, 1.);
    }
    kset_fbz.initialize(); 

    PROFILE_STOP("sirius::K_point_set::generate_w90_coeffs::unfold_fbz");
}

/*
 * This function generates the Full Brillouin zone starting from the Irreducible wedge. 
 * The equation to satisfy is:
 * \f[
 *     \psi_{n, R \bf k}({\bf G}) = e^{-i {\bf \tau}\cdot ({R\bf k}+{\bf G})}\psi_{n, \bf k} (R^{-1} {\bf G}) 
 * \f] 
 */
void 
rotate_wavefunctions( K_point_set& kset_ibz, K_point_set& kset_fbz, std::vector<k_info> const& k_temp,
                      int const& num_bands, std::vector<int> const& band_index_tot)
{
    PROFILE_START("sirius::K_point_set::generate_w90_coeffs::unfold_wfs");
    int num_bands_tot = kset_ibz.ctx().num_bands();
    std::complex<double> imtwopi = std::complex<double>(0., twopi);
    std::complex<double> exp1, exp2;
    srand(time(NULL));
    for (int ik = 0; ik < kset_fbz.num_kpoints(); ik++) {
        int src_rank  = kset_ibz.spl_num_kpoints().local_rank(k_temp[ik].ik_ibz);
        int dest_rank = kset_fbz.spl_num_kpoints().local_rank(ik);

        //send gvec
        auto gvec_IBZ = std::make_shared<fft::Gvec>(static_cast<fft::Gvec>(kset_ibz.get_gkvec(k_temp[ik].ik_ibz, dest_rank)));

        //send wf
        auto wf_IBZ = sddk::mdarray<std::complex<double>, 2>(gvec_IBZ->num_gvec(), num_bands_tot);
        int tag = src_rank + kset_fbz.num_kpoints()*dest_rank;
        mpi::Request req;
        if (kset_fbz.ctx().comm_k().rank() == src_rank) {
            req = kset_fbz.ctx().comm_k().isend(kset_ibz.get<double>(k_temp[ik].ik_ibz)->spinor_wave_functions().at(
                                sddk::memory_t::host, 0, wf::spin_index(0), wf::band_index(0)),
                                kset_ibz.get<double>(k_temp[ik].ik_ibz)->gkvec().num_gvec() * num_bands_tot, dest_rank, tag);
        }
        if (kset_fbz.ctx().comm_k().rank() == dest_rank) {
            kset_fbz.ctx().comm_k().recv(&wf_IBZ(0, 0), gvec_IBZ->num_gvec() * num_bands_tot, src_rank, tag);
        }

        //rotate wf
        if (kset_fbz.ctx().comm_k().rank() == dest_rank) {   
            //kset_fbz.get<double>(ik)->spinor_wave_functions_ = std::make_unique<wf::Wave_functions<double>>(
            //    kset_fbz.get<double>(ik)->gkvec_, wf::num_mag_dims(0), wf::num_bands(num_bands_tot), kset_fbz.ctx().host_memory_t());
                
            //kset_fbz.get<double>(ik)->spinor_wave_functions_->zero(sddk::memory_t::host);

            std::complex<double> exp1 = exp(-imtwopi * r3::dot(kset_fbz.get<double>(ik)->vk(), k_temp[ik].t)); 
            r3::vector<int> invRG;
            for (int ig = 0; ig < kset_fbz.get<double>(ik)->gkvec().num_gvec(); ig++) {
                // WARNING!! I suppose always that ik2ig[ik]=0 so i don't have it in the equation.
                invRG = r3::dot(kset_fbz.get<double>(ik)->gkvec().gvec<sddk::index_domain_t::local>(ig), k_temp[ik].invR);
                exp2 =
                    exp(-imtwopi * r3::dot(kset_fbz.get<double>(ik)->gkvec().gvec<sddk::index_domain_t::local>(ig), k_temp[ik].t));
                int ig_ = gvec_IBZ->index_by_gvec(invRG);
                assert ( ( ig_ != -1 ) );

                for (int iband = 0; iband < num_bands; iband++) {
                    kset_fbz.get<double>(ik)->spinor_wave_functions().pw_coeffs(ig, wf::spin_index(0),
                                                                             wf::band_index(iband)) =
                        exp1 * exp2 * wf_IBZ(ig_, band_index_tot[iband]) +
                        std::complex<double>(rand() % 1000, rand() % 1000) * 1.e-08; // needed to not get stuck on local
                                                                                     // minima. not working with 1.e-09
                }
            }
            for (int iband = 0; iband < num_bands; iband++) {
                kset_fbz.get<double>(ik)->band_energy(iband,0, kset_ibz.get<double>(k_temp[ik].ik_ibz)->band_energy(band_index_tot[iband], 0) );             
            }
        }
        if(src_rank == kset_fbz.ctx().comm_k().rank()){
            req.wait();
        }
    } // end ik loop
    PROFILE_STOP("sirius::K_point_set::generate_w90_coeffs::unfold_wfs");
}



/*
 * This function calculates the projection of the Bloch functions over an initial guess for the Wannier functions. 
 * The matrix A has matrix elements:
 * \f[
 *                    A_{mn}({\bf k})   = \langle u_{m \bf k}|\hat{S}|w_{n \bf k}\rangle
 * \f]
 * where u is the periodic part of the Bloch function and w is the initial guess.
 * Here we set as initial guesses the atomic orbitals of the pseudopotential.
*/
void 
calculate_Amn(K_point_set& kset_fbz, int const& num_bands, int const& num_wann, sddk::mdarray<std::complex<double>, 3>& A)
{

    A.zero();
    la::dmatrix<std::complex<double>> Ak(num_bands, num_wann); // matrix at the actual k point

    std::vector<int> atoms(kset_fbz.ctx().unit_cell().num_atoms());
    std::iota(atoms.begin(), atoms.end(), 0); // we need to understand which orbitals to pick up, I am using every here
    int num_atomic_wf = kset_fbz.ctx().unit_cell().num_ps_atomic_wf().first;

    std::unique_ptr<wf::Wave_functions<double>> Swf_k;
    // sddk::mdarray<std::complex<double>, 3> psidotpsi(num_bands, num_bands, num_kpts); // sirius2wannier
    // sddk::mdarray<std::complex<double>, 3> atdotat(num_wann, num_wann, num_kpts);     // sirius2wannier
    // psidotpsi.zero();
    // atdotat.zero();
    std::cout << "Calculating Amn...\n";
    auto mem = kset_fbz.ctx().processing_unit_memory_t();

    PROFILE_START("sirius::K_point_set::generate_w90_coeffs::calculate_Amn");

    for (int ikloc = 0; ikloc < kset_fbz.spl_num_kpoints().local_size(); ikloc++) {
        int ik = kset_fbz.spl_num_kpoints(ikloc);

        // calculate atomic orbitals + orthogonalization
        auto q_op = (kset_fbz.ctx().unit_cell().augment())
                        ? std::make_unique<Q_operator<double>>(kset_fbz.ctx())
                        : nullptr;
        // kset_fbz.kpoints_[ik]->beta_projectors().prepare();

        Swf_k = std::make_unique<wf::Wave_functions<double>>(kset_fbz.get<double>(ik)->get_gkvec(),
                                                             wf::num_mag_dims(0),
                                                             wf::num_bands(num_bands), kset_fbz.ctx().host_memory_t());

        auto bp_gen    = kset_fbz.get<double>(ik)->beta_projectors().make_generator();
        auto bp_coeffs = bp_gen.prepare();

        apply_S_operator<double, std::complex<double>>(mem, wf::spin_range(0), wf::band_range(0, num_bands), bp_gen,
                                                       bp_coeffs, (kset_fbz.get<double>(ik)->spinor_wave_functions()),
                                                       q_op.get(), *Swf_k);

        kset_fbz.get<double>(ik)->generate_atomic_wave_functions(
            atoms, [&](int iat) { return &kset_fbz.ctx().unit_cell().atom_type(iat).indexb_wfs(); },
            kset_fbz.ctx().ps_atomic_wf_ri(), kset_fbz.get<double>(ik)->atomic_wave_functions());

        /*
         *  Pick up only needed atomic functions, with their proper linear combinations
        //define index in atomic_wave_functions for atom iat
        std::vector<int> offset(kset_fbz.ctx().unit_cell().num_atoms());
        offset[0]=0;
        for(int i=1; i<kset_fbz.ctx().unit_cell().num_atoms(); i++){
            offset[i] = offset[i-1] + kset_fbz.ctx_.unit_cell().atom_type(i-1).indexb_wfs()->size();
        }
       //reconstruct map i-th wann func -> atom, l, m
        std::vector<std::array<int,3>> atoms_info(num_wann);

        auto needed_atomic_wf = std::make_unique<wf::Wave_functions<double>>(
                               kset_fbz.kpoints_[ik]->gkvec_, wf::num_mag_dims(0), wf::num_bands(num_wann),
       ctx_.host_memory_t());

        for(int iw=0; iw<num_wann; iw++)
        {
            int iat__=-1;
            for(int iat=0; iat<kset_fbz.ctx().unit_cell().num_atoms(); iat++){
                //calculate norm of center_w - atomic_position to decide which atom is the correct one
                auto& frac = this->unit_cell().atom(iat).position();
                r3::vector<double> diff = {center_w(0,iw)-frac[0], center_w(1,iw)-frac[1], center_w(2,iw)-frac[2] }
                if(diff.length() < 1.e-08){
                    iat__ = iat;
                    break;
                }
            }
            if(iat__==-1){
                std::cout <<"\n\n\nWARNING!! Could not find center_w: " << center_w(0,iw) << "  " << center_w(1,iw);
                std::cout <<"  " << center_w(2,iw) << std::endl << std::endl;
            }

            atoms_info[iw][0] = offset[iat__];
            atoms_info[iw][1] = proj_l(iw);
            atoms_info[iw][2] = proj_m(iw);
        }//end definition of atoms_info
        */

        // TODO: what is going on here?
        // it this code is taken from generate_hubbard_orbitals() then it should be reused
        // no code repetition is allowed
        // also, why do we need orthogonalized atomic orbitals for the initial guess?

        // ORTHOGONALIZING -CHECK HUBBARD FUNCTION
        apply_S_operator<double, std::complex<double>>(mem, wf::spin_range(0), wf::band_range(0, num_wann), bp_gen,
                                                       bp_coeffs, kset_fbz.get<double>(ik)->atomic_wave_functions(),
                                                       q_op.get(), kset_fbz.get<double>(ik)->atomic_wave_functions_S());

        int BS = kset_fbz.ctx().cyclic_block_size();
        la::dmatrix<std::complex<double>> ovlp(num_wann, num_wann, kset_fbz.ctx().blacs_grid(), BS, BS);
        wf::inner(kset_fbz.ctx().spla_context(), mem, wf::spin_range(0),
                  kset_fbz.get<double>(ik)->atomic_wave_functions(), wf::band_range(0, num_wann),
                  kset_fbz.get<double>(ik)->atomic_wave_functions_S(), wf::band_range(0, num_wann), ovlp, 0, 0);

        auto B = std::get<0>(inverse_sqrt(ovlp, num_wann));
        wf::transform(kset_fbz.ctx().spla_context(), mem, *B, 0, 0, 1.0,
                      kset_fbz.get<double>(ik)->atomic_wave_functions(), wf::spin_index(0), wf::band_range(0, num_wann),
                      0.0, kset_fbz.get<double>(ik)->atomic_wave_functions_S(), wf::spin_index(0),
                      wf::band_range(0, num_wann));
        wf::copy(mem, kset_fbz.get<double>(ik)->atomic_wave_functions_S(), wf::spin_index(0),
                 wf::band_range(0, num_wann), kset_fbz.get<double>(ik)->atomic_wave_functions(), wf::spin_index(0),
                 wf::band_range(0, num_wann));
        apply_S_operator<double, std::complex<double>>(mem, wf::spin_range(0), wf::band_range(0, num_wann), bp_gen,
                                                       bp_coeffs, kset_fbz.get<double>(ik)->atomic_wave_functions(),
                                                       q_op.get(), kset_fbz.get<double>(ik)->atomic_wave_functions_S());
        // END of the orthogonalization.

        wf::inner(kset_fbz.ctx().spla_context(), mem, wf::spin_range(0), kset_fbz.get<double>(ik)->spinor_wave_functions(),
                  wf::band_range(0, num_bands), kset_fbz.get<double>(ik)->atomic_wave_functions_S(),
                  wf::band_range(0, num_wann), Ak, 0, 0);
        // already in the correct way, we just copy in the bigger array. (alternative:: create dmatrix with an index
        // as multiindex to avoid copies) note!! we need +1 to copy the last element
        std::copy(Ak.begin(), Ak.end(),
                  A.at(sddk::memory_t::host, 0, 0, ik));

        std::cout << "Calculated Amn in rank " << kset_fbz.ctx().comm().rank() << " ik: " << ik << std::endl;
    } // end ik loop for Amn


    for (int ik = 0; ik < kset_fbz.num_kpoints(); ik++) {
        int local_rank = kset_fbz.spl_num_kpoints().local_rank(ik);
        kset_fbz.ctx().comm_k().bcast(A.at(sddk::memory_t::host, 0, 0, ik), num_bands * num_wann, local_rank);
    }
    PROFILE_STOP("sirius::K_point_set::generate_w90_coeffs::calculate_Amn");

}

/*
 * This function uses MPI to send the wavefunction at k+b to the node that holds k. 
 * All the wavefunctions and G vectors will be hold in vector structures, so that when calculating M each node is independent,
 * as the information has already been passed.
 */
void 
send_receive_kpb(std::vector<std::shared_ptr<fft::Gvec>>& gvec_kpb, std::vector<sddk::mdarray<std::complex<double>, 2>>& wf_kpb, K_point_set& kset_fbz, 
                std::vector<int>& ikpb_index, int const& nntot, sddk::mdarray<int,2> const& nnlist, int const& num_bands)
{
    PROFILE_START("sirius::K_point_set::generate_w90_coeffs::send_k+b");
    int index=-1; //to keep track of the index to use
    bool found;

    mpi::Request req;
    for (int ik = 0; ik < kset_fbz.num_kpoints(); ik++) {
        for (int ib = 0; ib < nntot; ib++) {
            int ikpb = nnlist(ik,ib)-1;
            int src_rank = kset_fbz.spl_num_kpoints().local_rank(ikpb);
            int dest_rank = kset_fbz.spl_num_kpoints().local_rank(ik);
            
            int tag = src_rank + kset_fbz.num_kpoints()*kset_fbz.num_kpoints()*dest_rank;
            if(kset_fbz.ctx().comm_k().rank() == dest_rank){
                found = ikpb_index[ikpb] != -1;//std::find(ikpb2ik_.begin(), ikpb2ik_.end(), ikpb) != ikpb2ik_.end(); //false if ikpb is not in ikpb2ik_ 
                req = kset_fbz.ctx().comm_k().isend(&found, 1, src_rank, tag);
            }
            if(kset_fbz.ctx().comm_k().rank() == src_rank){
                kset_fbz.ctx().comm_k().recv(&found, 1, dest_rank, tag);   
            }
            if(kset_fbz.ctx().comm_k().rank() == dest_rank){
                req.wait();
            }

            if(found){
                continue;
            }

            tag = src_rank + kset_fbz.num_kpoints()*dest_rank;
            
            auto temp = std::make_shared<fft::Gvec>( static_cast<fft::Gvec>( kset_fbz.get_gkvec(ikpb, dest_rank) ) );
            
            if (kset_fbz.ctx().comm_k().rank() == src_rank) {
                req = kset_fbz.ctx().comm_k().isend(kset_fbz.get<double>(ikpb)->spinor_wave_functions().at(
                                        sddk::memory_t::host, 0, wf::spin_index(0), wf::band_index(0)),
                                        temp->num_gvec() * num_bands, dest_rank, tag);
            }
            if (kset_fbz.ctx().comm_k().rank() == dest_rank) {
                index++;
                gvec_kpb.push_back(temp);
                wf_kpb.push_back(sddk::mdarray<std::complex<double>, 2>(gvec_kpb[index]->num_gvec(), num_bands));
                kset_fbz.ctx().comm_k().recv(&wf_kpb[index](0, 0), gvec_kpb[index]->num_gvec()*num_bands, src_rank, tag);
                ikpb_index[ikpb] = index;
            }
            if(kset_fbz.ctx().comm_k().rank() == src_rank){
                req.wait();
            }
        }//end ib
    }//end ik
    PROFILE_STOP("sirius::K_point_set::generate_w90_coeffs::send_k+b");

}

/*
 * This function calculates the projection of the periodic part of the Bloch functions at k over the periodic part of the Bloch function at k+b. 
 * The matrix M has matrix elements:
 * \f[
 *                    M_{mn}({\bf k},{\bf b})   = \langle u_{m, \bf k}|\hat{S}|u_{n, \bf k+b}\rangle
 * \f]
 * where u is the periodic part of the Bloch function. The set of neighbors k+b for each k is calculated with wannier_setup.
*/
void 
calculate_Mmn(sddk::mdarray<std::complex<double>,4>& M, K_point_set& kset_fbz, 
          int const& num_bands, std::vector<std::shared_ptr<fft::Gvec>> const& gvec_kpb, 
          std::vector<sddk::mdarray<std::complex<double>, 2>> const& wf_kpb,
          std::vector<int> const& ikpb_index, int const& nntot, sddk::mdarray<int,2> const& nnlist, 
          sddk::mdarray<int,3> const& nncell)
{
    PROFILE("sirius::K_point_set::generate_w90_coeffs::calculate_Mmn");
    la::dmatrix<std::complex<double>> Mbk(num_bands, num_bands);
    Mbk.zero();
    auto mem = kset_fbz.ctx().processing_unit_memory_t();

    for (int ikloc = 0; ikloc < kset_fbz.spl_num_kpoints().local_size(); ikloc++) {
        int ik = kset_fbz.spl_num_kpoints(ikloc);
        std::cout << "Calculating Mmn. ik = " << ik << std::endl;
        auto q_op = (kset_fbz.unit_cell().augment())
                        ? std::make_unique<Q_operator<double>>(kset_fbz.get<double>(ik)->ctx())
                        : nullptr;
        auto bp_gen    = kset_fbz.get<double>(ik)->beta_projectors().make_generator();
        auto bp_coeffs = bp_gen.prepare();
        auto Swf_k = std::make_unique<wf::Wave_functions<double>>(kset_fbz.get<double>(ik)->get_gkvec(), wf::num_mag_dims(0),
                                                         wf::num_bands(num_bands), kset_fbz.ctx().host_memory_t());
        apply_S_operator<double, std::complex<double>>(mem, wf::spin_range(0), wf::band_range(0, num_bands), bp_gen,
                                                            bp_coeffs, (kset_fbz.get<double>(ik)->spinor_wave_functions()),
                                                            q_op.get(), *Swf_k);

        for (int ib = 0; ib < nntot; ib++){
            int ikpb      = nnlist(ik, ib) - 1;
            auto index_ikpb = ikpb_index[ikpb];
            assert((index_ikpb != -1));
                
            std::unique_ptr<wf::Wave_functions<double>> aux_psi_kpb = std::make_unique<wf::Wave_functions<double>>(
                        kset_fbz.get<double>(ik)->get_gkvec(), wf::num_mag_dims(0), wf::num_bands(num_bands), kset_fbz.ctx().host_memory_t());
            aux_psi_kpb->zero(sddk::memory_t::host);
            r3::vector<int> G;
            for (int ig = 0; ig < kset_fbz.get<double>(ik)->gkvec().num_gvec(); ig++) {
                // compute the total vector to use to get the index in kpb
                G = kset_fbz.get<double>(ik)->gkvec().gvec<sddk::index_domain_t::local>(ig);
                G += r3::vector<int>(nncell(0, ik, ib), nncell(1, ik, ib), nncell(2, ik, ib));
                int ig_ = gvec_kpb[index_ikpb]->index_by_gvec(G); // kpoints_[ikpb]->gkvec_->index_by_gvec(G);
                if (ig_ == -1) {
                    continue;
                }
                for (int iband = 0; iband < num_bands; iband++) {
                    aux_psi_kpb->pw_coeffs(ig, wf::spin_index(0), wf::band_index(iband)) = wf_kpb[index_ikpb](ig_, iband);
                }
            } // end ig

            wf::inner(kset_fbz.ctx().spla_context(), mem, wf::spin_range(0), *aux_psi_kpb, wf::band_range(0, num_bands), *Swf_k,
                      wf::band_range(0, num_bands), Mbk, 0, 0);
            for (int n = 0; n < num_bands; n++) {
                for (int m = 0; m < num_bands; m++) {
                    M(m, n, ib, ik) = std::conj(Mbk(n, m));
                }
            }
        }
    }     // end ik
    std::cout << "Mmn calculated.\n";
    std::cout << "starting broadcast...\n";
    for (int ik = 0; ik < kset_fbz.num_kpoints(); ik++) {
        int local_rank = kset_fbz.spl_num_kpoints().local_rank(ik);
        kset_fbz.ctx().comm_k().bcast(M.at(sddk::memory_t::host, 0, 0, 0, ik), num_bands * num_bands * nntot,
                                      local_rank);
    }

}


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
void
K_point_set::generate_w90_coeffs() // sirius::K_point_set& k_set__)
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
    //
    // 2nd step: compute <beta_{\xi'}^{\alpha}|  psi_{j,k+q}>; check how this is done in the Beta_projector class;
    // Q-operator can be applied here. Look how this is done in Non_local_operator::apply();
    // (look for Beta_projectors_base::inner() function; understand the "chunks" of beta-projectors
    //
    // 3nd step: copy wave-function at k+q (k') into an auxiliary wave-function object of G+k order and see how
    // the G+k+q index can be reshuffled. Check the implementation of G-vector class which handles all the G- and G+k-
    // indice
    //
    // 4th step: allocate resulting matrix M_{nn'}, compute contribution from C*C (1st part) using wf::inner() function;
    // compute contribution from ultrasoft part using a matrix-matrix multiplication
    //
    // 5th step: parallelize over k-points
    //
    // 6ts step: parallelize over G+k vectors and k-points
    PROFILE("sirius::K_point_set::generate_w90_coeffs");
    std::cout << "\n\n\nWannierization!!!!\n\n\n";

    K_point_set& kset_ibz = *this;
    K_point_set kset_fbz(this->ctx());
    std::vector<k_info> k_temp;    
    from_irreduciblewedge_to_fullbrillouinzone(*this, kset_fbz, k_temp); 

    auto ngridk = this->ctx().cfg().parameters().ngridk();
    assert( ((int)k_temp.size() == ngridk[0] * ngridk[1] * ngridk[2]) );

    int num_bands_tot = this->ctx().num_bands();
 
    /*
     * Set all variables for wannier_setup and call the function
     */
 
    // scalar variables definition
    size_t length_seedname = 100;    // aux variable for the length of a string
    int32_t num_kpts;                // input
    // int32_t num_bands_tot;        // input
    int32_t num_atoms;               // input
    size_t length_atomic_symbol = 3; // aux, as expected from wannier90 lib
    fortran_bool gamma_only;         // input
    fortran_bool spinors;            // input
    int32_t num_bands;               // output
    int32_t num_wann;                // output
    int32_t nntot;                   // output
    int32_t num_nnmax = 12;          // aux variable for max number of neighbors
                                     // fixed, as in pw2wannier or in wannier90 docs

    // scalar variables initialization
    num_kpts = kset_fbz.num_kpoints();
    // num_bands_tot = this->get<double>(spl_num_kpoints_[0])->spinor_wave_functions().num_wf();
    num_atoms  = this->ctx().unit_cell().num_atoms();
    gamma_only = this->ctx().gamma_point();
    spinors    = false; // right now, generate_wave_functions only works with colin!
    // WARNING we need to compare with .win file!!!

    // non-scalar variables definition + space allocation
    char seedname[length_seedname];                           // input
    sddk::mdarray<int32_t, 1> mp_grid(3);                     // input
    sddk::mdarray<double, 2> real_lattice(3, 3);              // input BOHR!
    sddk::mdarray<double, 2> recip_lattice(3, 3);             // input BOHR^{-1}!
    sddk::mdarray<double, 2> kpt_lattice(3, num_kpts);        // input
    char atomic_symbol[num_atoms][3];                         // input
    sddk::mdarray<double, 2> atoms_cart(3, num_atoms);        // input
    sddk::mdarray<int, 2> nnlist(num_kpts, num_nnmax);        // output
    sddk::mdarray<int32_t, 3> nncell(3, num_kpts, num_nnmax); // output
    sddk::mdarray<double, 2> proj_site(3, num_bands_tot);     // output
    sddk::mdarray<int32_t, 1> proj_l(num_bands_tot);          // output
    sddk::mdarray<int32_t, 1> proj_m(num_bands_tot);          // output
    sddk::mdarray<int32_t, 1> proj_radial(num_bands_tot);     // output
    sddk::mdarray<double, 2> proj_z(3, num_bands_tot);        // output
    sddk::mdarray<double, 2> proj_x(3, num_bands_tot);        // output
    sddk::mdarray<double, 1> proj_zona(num_bands_tot);        // output
    sddk::mdarray<int32_t, 1> exclude_bands(num_bands_tot);   // output
    sddk::mdarray<int32_t, 1> proj_s(num_bands_tot);          // output - optional
    sddk::mdarray<double, 2> proj_s_qaxis(3, num_bands_tot);  // output - optional

    // non-scalar variables initialization
    std::string aux = "silicon";
    strcpy(seedname, aux.c_str());
    length_seedname = aux.length();

    for (int ivec = 0; ivec < 3; ivec++) {
        for (int icoor = 0; icoor < 3; icoor++) {
            real_lattice(ivec, icoor)  = ctx().unit_cell().lattice_vectors()(icoor, ivec) * bohr_radius;
            recip_lattice(ivec, icoor) = ctx().unit_cell().reciprocal_lattice_vectors()(icoor, ivec) / bohr_radius;
        }
    }


    for (int ik = 0; ik < num_kpts; ik++) {
        for (int ix : {0, 1, 2}) {
            kpt_lattice(ix, ik) = kset_fbz.kpoints_[ik]->vk_[ix];
        }
    }

    for (int iat = 0; iat < num_atoms; iat++) {
        std::fill(atomic_symbol[iat], atomic_symbol[iat] + 3, ' ');
        std::strcpy(atomic_symbol[iat], this->ctx().unit_cell().atom(iat).type().label().c_str());
        // position is saved in fractional coordinates, we need cartesian for wannier_setup_
        auto frac_coord = this->unit_cell().atom(iat).position();
        auto cart_coord = this->ctx().unit_cell().get_cartesian_coordinates(frac_coord);
        for (int icoor = 0; icoor < 3; icoor++) {
            atoms_cart(icoor, iat) = cart_coord[icoor] * bohr_radius;
        }
    }

    /*
     * Call wannier_setup_ from wannier library. This calculates two important arrays:
     * nnlist(ik,ib) is the index of the neighbor ib of the vector at index ik
     * nncell(ix,ik,ib) is the ix-th coordinate of the G vector that brings back the vector defined in nnlist(ik,ib)
     * nntot is the total number of neighbors.
     * to the first Brillouin zone. Eq. to hold:
     *             kpoints_[nnlist(ik,ib)] = kpoints_[ik] + (neighbor b) - nncell(.,ik,ib)
     */
    std::cout << "I am process " << ctx().comm().rank() << " and I go inside the wannier_setup\n";

    PROFILE_START("sirius::K_point_set::generate_w90_coeffs::wannier_setup");
    if (ctx().comm().rank() == 0) {
        std::cout << "starting wannier_setup_\n";
/*        wannier_setup_(seedname,
                       this->ctx().cfg().parameters().ngridk().data(), // input
                       &num_kpts,                                      // input
                       real_lattice.at(sddk::memory_t::host),          // input
                       recip_lattice.at(sddk::memory_t::host),         // input
                       kpt_lattice.at(sddk::memory_t::host),           // input
                       &num_bands_tot,                                 // input
                       &num_atoms,                                     // input
                       atomic_symbol,                                  // input
                       atoms_cart.at(sddk::memory_t::host),            // input
                       &gamma_only,                                    // input
                       &spinors,                                       // input
                       &nntot,                                         // output
                       nnlist.at(sddk::memory_t::host),                // output
                       nncell.at(sddk::memory_t::host),                // output
                       &num_bands,                                     // output
                       &num_wann,                                      // output
                       proj_site.at(sddk::memory_t::host),             // output
                       proj_l.at(sddk::memory_t::host),                // output
                       proj_m.at(sddk::memory_t::host),                // output
                       proj_radial.at(sddk::memory_t::host),           // output
                       proj_z.at(sddk::memory_t::host),                // output
                       proj_x.at(sddk::memory_t::host),                // output
                       proj_zona.at(sddk::memory_t::host),             // output
                       exclude_bands.at(sddk::memory_t::host),         // output
                       proj_s.at(sddk::memory_t::host),                // output
                       proj_s_qaxis.at(sddk::memory_t::host),          // output
                       length_seedname,                                // aux-length of a string
                       length_atomic_symbol);                          // aux-length of a string
*/
    read_nnkp(const int& num_kpts, int& num_wann, int& nntot, sddk::mdarray<int, 2>& nnlist, 
    sddk::mdarray<int32_t, 3>& nncell, sddk::mdarray<int32_t, 1>& exclude_bands)

    }
    ctx().comm().bcast(&nntot, 1, 0);
    ctx().comm().bcast(nnlist.at(sddk::memory_t::host), num_kpts * num_nnmax, 0);
    ctx().comm().bcast(nncell.at(sddk::memory_t::host), 3 * num_kpts * num_nnmax, 0);
    ctx().comm().bcast(&num_bands, 1, 0);
    ctx().comm().bcast(&num_wann, 1, 0);
    ctx().comm().bcast(exclude_bands.at(sddk::memory_t::host), num_bands_tot, 0);

    std::cout << "\n\n\n\n\n\n";
    std::cout << "wannier_setup succeeded. rank " << ctx().comm().rank() << "\n";

    PROFILE_STOP("sirius::K_point_set::generate_w90_coeffs::wannier_setup");

    std::vector<int> band_index_tot;//band_index_tot[iband] gives the index of iband in the full band vector
    for(int iband=0; iband<exclude_bands.size(); iband++){
        int band_fortran = iband+1;
        //bool is_excluded = (std::find(exclude_bands.at(sddk::memory_t::host),
        //                    exclude_bands.at(sddk::memory_t::host)+exclude_bands.size(),
        //                    iband+1)
        //                    != exclude_bands.at(sddk::memory_t::host)+exclude_bands.size()
        //                   );// true if the value iband+1 is in exclude_bands

        bool is_excluded = ( std::find_if(exclude_bands.at(sddk::memory_t::host),
                                          exclude_bands.at(sddk::memory_t::host)+exclude_bands.size(), 
                                          [&, band_fortran](int const& band_excluded)
                                          {return ( band_excluded == band_fortran );  }
                                         ) != exclude_bands.at(sddk::memory_t::host)+exclude_bands.size() );

        if(!is_excluded){
            band_index_tot.push_back(iband);
        }
    }

    rotate_wavefunctions(*this, kset_fbz, k_temp, num_bands, band_index_tot);

    num_wann = ctx_.unit_cell().num_ps_atomic_wf().first;

    sddk::mdarray<std::complex<double>, 3> A(num_bands, num_wann, kset_fbz.num_kpoints());
    A.zero();

    calculate_Amn(kset_fbz, num_bands, num_wann, A);
 
    if (ctx().comm().rank() == 0) {
        write_Amn(A, num_kpts, num_bands, num_wann);
    }

    std::vector< std::shared_ptr<fft::Gvec> > gvec_kpb;
    std::vector< sddk::mdarray<std::complex<double>, 2> > wf_kpb;
    std::vector<int> ikpb_index(kset_fbz.num_kpoints(),-1);

    send_receive_kpb(gvec_kpb, wf_kpb, kset_fbz, ikpb_index, nntot, nnlist, num_bands);


    sddk::mdarray<std::complex<double>, 4> M(num_bands, num_bands, nntot, kset_fbz.num_kpoints());
    M.zero();
 
    calculate_Mmn(M, kset_fbz, num_bands, gvec_kpb, wf_kpb, ikpb_index, nntot, nnlist, nncell);

    if (ctx().comm().rank() == 0) {
        write_Mmn(M, nnlist, nncell, num_kpts, nntot, num_bands);
    }

    // Initialize eigval with the value of the energy dispersion

    sddk::mdarray<double, 2> eigval(num_bands, num_kpts); // input

    for (int ik = 0; ik < num_kpts; ik++) {
        int local_rank = kset_fbz.spl_num_kpoints().local_rank(ik);
        if (kset_fbz.ctx().comm_k().rank() == local_rank) {
            for (int iband = 0; iband < num_bands; iband++) {
                eigval(iband, ik) = kset_fbz.get<double>(ik)->band_energy(iband, 0) * ha2ev; // sirius saves energy in
                                                                                             // Hartree, we need it in eV
            }
        }
        kset_fbz.ctx().comm_k().bcast(eigval.at(sddk::memory_t::host, 0, ik), num_bands, local_rank);//TODO: remove
    }
/*
    if (kset_fbz.ctx().comm_k().rank() == 0) {
        std::cout << "Starting wannier_run..." << std::endl;

        // compute wannier orbitals
        // define additional arguments
        sddk::mdarray<std::complex<double>, 3> U_matrix(num_wann, num_wann, num_kpts); // output
        sddk::mdarray<std::complex<double>, 3> U_dis(num_bands, num_wann, num_kpts);   // output
        sddk::mdarray<fortran_bool, 2> lwindow(num_bands, num_kpts);                   // output
        sddk::mdarray<double, 2> wannier_centres(3, num_wann);                         // output
        sddk::mdarray<double, 1> wannier_spreads(num_wann);                            // output
        sddk::mdarray<double, 1> spread_loc(3);                                        // output-op

        write_eig(eigval, num_bands, num_kpts);

        U_matrix.zero();
        U_dis.zero();
        lwindow.zero();
        wannier_centres.zero();
        wannier_spreads.zero();
        spread_loc.zero();

        PROFILE_START("sirius::K_point_set::generate_w90_coeffs::wannier_run");

        wannier_run_(seedname, this->ctx().cfg().parameters().ngridk().data(), &num_kpts,
                     real_lattice.at(sddk::memory_t::host), recip_lattice.at(sddk::memory_t::host),
                     kpt_lattice.at(sddk::memory_t::host), &num_bands, &num_wann, &nntot, &num_atoms, atomic_symbol,
                     atoms_cart.at(sddk::memory_t::host), &gamma_only, M.at(sddk::memory_t::host),
                     A.at(sddk::memory_t::host), eigval.at(sddk::memory_t::host), U_matrix.at(sddk::memory_t::host),
                     U_dis.at(sddk::memory_t::host), lwindow.at(sddk::memory_t::host),
                     wannier_centres.at(sddk::memory_t::host), wannier_spreads.at(sddk::memory_t::host),
                     spread_loc.at(sddk::memory_t::host), length_seedname, length_atomic_symbol);
        std::cout << "Wannier_run succeeded. " << std::endl;
    }
*/
    PROFILE_STOP("sirius::K_point_set::generate_w90_coeffs::wannier_run");
}

}
#endif // SIRIUS_WANNIER90
