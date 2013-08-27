// This file is part of SIRIUS
//
// Copyright (c) 2013 Anton Kozhevnikov, Thomas Schulthess
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

#ifndef __K_SET_H__
#define __K_SET_H__

/** \file k_set.h

    \brief Set of k-points
*/

namespace sirius 
{

class K_set
{
    private:
    
        Global& parameters_;

        Band* band_;

        std::vector<K_point*> kpoints_;

        splindex<block> spl_num_kpoints_;

    public:

        K_set(Global& parameters__) : parameters_(parameters__)
        {
            band_ = new Band(parameters_);
        }

        ~K_set()
        {
            clear();
            delete band_;
        }
        
        /// Initialize the k-point set
        void initialize();

        void update();
        
        /// Find eigen-states
        void find_eigen_states(Potential* potential, bool precompute);

        /// Find Fermi energy and band occupation numbers
        void find_band_occupancies();

        /// Return sum of valence eigen-values
        double valence_eval_sum();

        void print_info();

        void sync_band_energies();
        
        void save_wave_functions();

        void load_wave_functions();
        
        int max_num_gkvec();

        void force(mdarray<double, 2>& forcek);
        
        void add_kpoint(double* vk__, double weight__)
        {
            kpoints_.push_back(new K_point(parameters_, vk__, weight__));
        }

        void add_kpoints(mdarray<double, 2>& kpoints__, double* weights__)
        {
            for (int ik = 0; ik < kpoints__.size(1); ik++) add_kpoint(&kpoints__(0, ik), weights__[ik]);
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
            return spl_num_kpoints_[ikloc];
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
};

#include "k_set.hpp"

};

#endif // __K_SET_H__

