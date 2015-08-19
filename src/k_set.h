// Copyright (c) 2013-2014 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that 
// the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the 
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions 
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file k_set.h
 *
 *  \brief Contains declaration and partial implementation of sirius::K_set class.
 */

#ifndef __K_SET_H__
#define __K_SET_H__

#include "band.h"
#include "potential.h"
#include "k_point.h"
#include "blacs_grid.h"
#include "vector3d.h"

namespace sirius 
{

struct kq
{
    // index of reduced k+q vector
    int jk;

    // vector which reduces k+q to first BZ
    vector3d<int> K;
};

/// Set of k-points.
class K_set
{
    private:
    
        Simulation_context& ctx_;

        Simulation_parameters const& parameters_;

        Band* band_;

        std::vector<K_point*> kpoints_;

        splindex<block> spl_num_kpoints_;

        double energy_fermi_;

        double band_gap_;

        Unit_cell& unit_cell_;

        Communicator comm_k_;

        BLACS_grid const& blacs_grid_;

        /// 1D BLACS grid for a "slab" data distribution.
        /** This grid is used to distribute G+k vector index and keep a whole band index */
        BLACS_grid blacs_grid_slab_;
        
        /// 1D BLACS grid for a "slice" data distribution.
        /** This grid is used to distribute band index and keep a whole G+k vector index */
        BLACS_grid blacs_grid_slice_;

        void init()
        {
            band_ = new Band(ctx_, blacs_grid_);
        }

    public:

        K_set(Simulation_context& ctx__,
              Communicator const& comm_k__,
              BLACS_grid const& blacs_grid__)
            : ctx_(ctx__),
              parameters_(ctx__.parameters()),
              unit_cell_(ctx__.unit_cell()),
              comm_k_(comm_k__),
              blacs_grid_(blacs_grid__),
              blacs_grid_slab_(blacs_grid_.comm(), blacs_grid_.comm().size(), 1),
              blacs_grid_slice_(blacs_grid_.comm(), 1, blacs_grid_.comm().size())
        {
            init();
        }

        K_set(Simulation_context& ctx__,
              Communicator const& comm_k__,
              BLACS_grid const& blacs_grid__,
              vector3d<int> k_grid__,
              vector3d<int> k_shift__,
              int use_symmetry__) 
            : ctx_(ctx__),
              parameters_(ctx__.parameters()),
              unit_cell_(ctx__.unit_cell()),
              comm_k_(comm_k__),
              blacs_grid_(blacs_grid__),
              blacs_grid_slab_(blacs_grid_.comm(), blacs_grid_.comm().size(), 1),
              blacs_grid_slice_(blacs_grid_.comm(), 1, blacs_grid_.comm().size())
        {
            init();

            int nk;
            mdarray<double, 2> kp;
            std::vector<double> wk;
            if (use_symmetry__)
            {
                nk = unit_cell_.symmetry()->get_irreducible_reciprocal_mesh(k_grid__, k_shift__, kp, wk);
            }
            else
            {
                nk = k_grid__[0] * k_grid__[1] * k_grid__[2];
                wk = std::vector<double>(nk, 1.0 / nk);
                kp = mdarray<double, 2>(3, nk);

                int ik = 0;
                for (int i0 = 0; i0 < k_grid__[0]; i0++)
                {
                    for (int i1 = 0; i1 < k_grid__[1]; i1++)
                    {
                        for (int i2 = 0; i2 < k_grid__[2]; i2++)
                        {
                            kp(0, ik) = double(i0 + k_shift__[0] / 2.0) / k_grid__[0];
                            kp(1, ik) = double(i1 + k_shift__[1] / 2.0) / k_grid__[1];
                            kp(2, ik) = double(i2 + k_shift__[2] / 2.0) / k_grid__[2];
                            ik++;
                        }
                    }
                }
            }

            //if (use_symmetry__)
            //{
            //    mdarray<int, 2> kmap(parameters_.unit_cell()->symmetry()->num_sym_op(), nk);
            //    for (int ik = 0; ik < nk; ik++)
            //    {
            //        for (int isym = 0; isym < parameters_.unit_cell()->symmetry()->num_sym_op(); isym++)
            //        {
            //            auto vk_rot = matrix3d<double>(transpose(parameters_.unit_cell()->symmetry()->rot_mtrx(isym))) * 
            //                          vector3d<double>(vk(0, ik), vk(1, ik), vk(2, ik));
            //            for (int x = 0; x < 3; x++)
            //            {
            //                if (vk_rot[x] < 0) vk_rot[x] += 1;
            //                if (vk_rot[x] < 0 || vk_rot[x] >= 1) TERMINATE("wrong rotated k-point");
            //            }

            //            for (int jk = 0; jk < nk; jk++)
            //            {
            //                if (std::abs(vk_rot[0] - vk(0, jk)) < 1e-10 &&
            //                    std::abs(vk_rot[1] - vk(1, jk)) < 1e-10 &&
            //                    std::abs(vk_rot[2] - vk(2, jk)) < 1e-10)
            //                {
            //                    kmap(isym, ik) = jk;
            //                }
            //            }
            //        }
            //    }

            //    //== std::cout << "sym.table" << std::endl;
            //    //== for (int isym = 0; isym < parameters_.unit_cell()->symmetry().num_sym_op(); isym++)
            //    //== {
            //    //==     printf("sym: %2i, ", isym); 
            //    //==     for (int ik = 0; ik < nk; ik++) printf(" %2i", kmap(isym, ik));
            //    //==     printf("\n");
            //    //== }

            //    std::vector<int> flag(nk, 1);
            //    for (int ik = 0; ik < nk; ik++)
            //    {
            //        if (flag[ik])
            //        {
            //            int ndeg = 0;
            //            for (int isym = 0; isym < parameters_.unit_cell()->symmetry()->num_sym_op(); isym++)
            //            {
            //                if (flag[kmap(isym, ik)])
            //                {
            //                    flag[kmap(isym, ik)] = 0;
            //                    ndeg++;
            //                }
            //            }
            //            add_kpoint(&vk(0, ik), double(ndeg) / nk);
            //        }
            //    }
            //}
            //else
            //{
            //    for (int ik = 0; ik < nk; ik++) add_kpoint(&vk(0, ik), wk[ik]);
            //}

            for (int ik = 0; ik < nk; ik++) add_kpoint(&kp(0, ik), wk[ik]);
        }

        ~K_set()
        {
            clear();
            delete band_;
        }
        
        /// Initialize the k-point set
        void initialize();

        //void update();
        
        /// Solve \f$ \hat H \psi = E \psi \f$ and find eigen-states of the Hamiltonian
        void find_eigen_states(Potential* potential, bool precompute);

        /// Find Fermi energy and band occupation numbers
        void find_band_occupancies();

        /// Return sum of valence eigen-values
        double valence_eval_sum();

        void print_info();

        void sync_band_energies();
        
        void save();

        void load();

        int max_num_gkvec();

        void force(mdarray<double, 2>& forcek);
        
        void add_kpoint(double* vk__, double weight__)
        {
            kpoints_.push_back(new K_point(ctx_, vk__, weight__, blacs_grid_, blacs_grid_slab_, blacs_grid_slice_));
        }

        void add_kpoints(mdarray<double, 2>& kpoints__, double* weights__)
        {
            for (int ik = 0; ik < (int)kpoints__.size(1); ik++) add_kpoint(&kpoints__(0, ik), weights__[ik]);
        }

        inline K_point* operator[](int i)
        {
            assert(i >= 0 && i < (int)kpoints_.size());
            
            return kpoints_[i];
        }

        void clear()
        {
            for (int ik = 0; ik < (int)kpoints_.size(); ik++) delete kpoints_[ik];
            
            kpoints_.clear();
        }
        
        inline int num_kpoints()
        {
            return (int)kpoints_.size();
        }

        inline splindex<block>& spl_num_kpoints()
        {
            return spl_num_kpoints_;
        }
        
        inline int spl_num_kpoints(int ikloc)
        {
            return static_cast<int>(spl_num_kpoints_[ikloc]);
        }

        void set_band_occupancies(int ik, double* band_occupancies)
        {
            kpoints_[ik]->set_band_occupancies(band_occupancies);
        }
        
        void get_band_energies(int ik, double* band_energies)
        {
            kpoints_[ik]->get_band_energies(band_energies);
        }
        
        void get_band_occupancies(int ik, double* band_occupancies)
        {
            kpoints_[ik]->get_band_occupancies(band_occupancies);
        }

        Band* band()
        {
            return band_;
        }

        inline double energy_fermi()
        {
            return energy_fermi_;
        }

        inline double band_gap()
        {
            return band_gap_;
        }

        /// Find index of k-point.
        inline int find_kpoint(vector3d<double> vk)
        {
            for (int ik = 0; ik < num_kpoints(); ik++) 
            {
                if ((kpoints_[ik]->vk() - vk).length() < 1e-12) return ik;
            }
            return -1;
        }

        void generate_Gq_matrix_elements(vector3d<double> vq)
        {
            std::vector<kq> kpq(num_kpoints());
            for (int ik = 0; ik < num_kpoints(); ik++)
            {
                // reduce k+q to first BZ: k+q=k"+K; k"=k+q-K
                std::pair< vector3d<double>, vector3d<int> > vkqr = Utils::reduce_coordinates(kpoints_[ik]->vk() + vq);
                
                if ((kpq[ik].jk = find_kpoint(vkqr.first)) == -1) 
                    error_local(__FILE__, __LINE__, "index of reduced k+q point is not found");

                kpq[ik].K = vkqr.second;
            }
        }
};

};

#endif // __K_SET_H__

