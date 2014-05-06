#ifndef __GLOBAL_H__
#define __GLOBAL_H__

/** \file global.h

    \brief Global variables 
*/

/** \page stdvarname Standard variable names
    
    Below is the list of standard names for some of the loop variables:
    
    l - index of orbital quantum number \n
    m - index of azimutal quantum nuber \n
    lm - combined index of (l,m) quantum numbers \n
    ia - index of atom \n
    ic - index of atom class \n
    iat - index of atom type \n
    ir - index of r-point \n
    ig - index of G-vector \n
    idxlo - index of local orbital \n
    idxrf - index of radial function \n

    The loc suffix is added to the variable to indicate that it runs over local fraction of elements for the given
    MPI rank. Typical code looks like this:
    
    \code{.cpp}
        // zero array
        memset(&mt_val[0], 0, parameters_.num_atoms() * sizeof(T));
        
        // loop over local fraction of atoms
        for (int ialoc = 0; ialoc < parameters_.spl_num_atoms().local_size(); ialoc++)
        {
            // get global index of atom
            int ia = parameters_.spl_num_atoms(ialoc);

            int nmtp = parameters_.atom(ia)->num_mt_points();
           
            // integrate spherical part of the function
            Spline<T> s(nmtp, parameters_.atom(ia)->type()->radial_grid());
            for (int ir = 0; ir < nmtp; ir++) s[ir] = f_mt<local>(0, ir, ialoc);
            mt_val[ia] = s.interpolate().integrate(2) * fourpi * y00;
        }

        // simple array synchronization
        Platform::allreduce(&mt_val[0], parameters_.num_atoms());
    \endcode
*/

#include <vector>
#include "version.h"
#include "mpi_grid.h"
#include "splindex.h"
#include "step_function.h"
#include "error_handling.h"

namespace sirius {

/// Global variables and methods
class Global
{
    private:

        /// true if class was initialized
        bool initialized_;
    
        /// maximum l for APW functions
        int lmax_apw_;
        
        /// maximum l for plane waves
        int lmax_pw_;
        
        /// maximum l for density
        int lmax_rho_;
        
        /// maximum l for potential
        int lmax_pot_;

        /// cutoff for augmented-wave functions
        double aw_cutoff_;

        /// cutoff for plane-waves
        double pw_cutoff_;

        /// cutoff for |G+k| plane-waves
        double gk_cutoff_;
        
        /// number of first-variational states
        int num_fv_states_;

        /// number of bands (= number of spinor states)
        int num_bands_;
       
        /// number of spin componensts (1 or 2)
        int num_spins_;

        /// number of dimensions of the magnetization and effective magnetic field (0, 1 or 3)
        int num_mag_dims_;

        /// true if spin-orbit correction is applied
        bool so_correction_;
       
        /// true if UJ correction is applied
        bool uj_correction_;

        /// general purpose synchronization flag
        int sync_flag_;

        /// MPI grid dimensions
        std::vector<int> mpi_grid_dims_;
        
        /// MPI grid
        MPI_grid mpi_grid_;

        /// block size for block-cyclic data distribution  
        int cyclic_block_size_;
        
        /// starting time of the program
        timeval start_time_;

        ev_solver_t std_evp_solver_type_;

        standard_evp* std_evp_solver_; 

        ev_solver_t gen_evp_solver_type_;

        generalized_evp* gen_evp_solver_;
        
        /// type of the processing unit
        processing_unit_t processing_unit_;

        /// block-cyclic distribution of the first-variational states along columns of the MPI grid
        splindex<block_cyclic> spl_fv_states_;
        
        /// additional splitting of the first-variational states along rows of the MPI grid
        splindex<block> sub_spl_fv_states_;

        /// block-cyclic distribution of the spinor wave-functions along columns of the MPI grid
        splindex<block_cyclic> spl_spinor_wf_;
       
        /// additional splitting of spinor wave-functions along rows of the MPI grid
        splindex<block> sub_spl_spinor_wf_;

        /// BLACS communication context for an eigen-value solver and related operations
        int blacs_context_;
        
        /// smearing function width
        double smearing_width_;
        
        electronic_structure_method_t esm_type_;
        
        /// step function is used in full-potential methods
        Step_function* step_function_;

        Reciprocal_lattice* reciprocal_lattice_;

        Unit_cell* unit_cell_;

        double iterative_solver_tolerance_;

        /// read from the input file if it exists
        void read_input();

        std::string start_time(const char* fmt);

    public:
    
        Global() 
            : initialized_(false), 
              lmax_apw_(lmax_apw_default), 
              lmax_pw_(-1), 
              lmax_rho_(lmax_rho_default), 
              lmax_pot_(lmax_pot_default), 
              aw_cutoff_(aw_cutoff_default), 
              pw_cutoff_(pw_cutoff_default), 
              gk_cutoff_(pw_cutoff_default / 4.0), 
              num_fv_states_(-1), 
              num_spins_(1), 
              num_mag_dims_(0), 
              so_correction_(false), 
              uj_correction_(false), 
              cyclic_block_size_(64), 
              std_evp_solver_type_(ev_lapack),
              std_evp_solver_(NULL),
              gen_evp_solver_type_(ev_lapack),
              gen_evp_solver_(NULL),
              #ifdef _GPU_
              processing_unit_(gpu),
              #else
              processing_unit_(cpu),
              #endif
              blacs_context_(-1),
              smearing_width_(0.001), 
              esm_type_(full_potential_lapwlo),
              step_function_(NULL),
              reciprocal_lattice_(NULL),
              iterative_solver_tolerance_(0.01)
        {
            /* get the starting time */
            gettimeofday(&start_time_, NULL);

            /* read initial data from sirius.json */
            read_input();

            /* create new empty unit cell */
            unit_cell_ = new Unit_cell(esm_type_);

            #ifdef _SCALAPACK_
            if (linalg<scalapack>::cyclic_block_size() <= 0) // TODO: get rid of this
                linalg<scalapack>::set_cyclic_block_size(cyclic_block_size_);

            splindex<block_cyclic>::set_cyclic_block_size(linalg<scalapack>::cyclic_block_size());
            #endif
        }
            
        ~Global()
        {
            clear();
            delete unit_cell_;
        }

        void set_lmax_apw(int lmax_apw__)
        {
            lmax_apw_ = lmax_apw__;
        }

        void set_lmax_rho(int lmax_rho__)
        {
            lmax_rho_ = lmax_rho__;
        }

        void set_lmax_pot(int lmax_pot__)
        {
            lmax_pot_ = lmax_pot__;
        }

        void set_num_spins(int num_spins__)
        {
            num_spins_ = num_spins__;
        }

        void set_num_mag_dims(int num_mag_dims__)
        {
            num_mag_dims_ = num_mag_dims__;
        }

        inline int lmax_apw()
        {
            return lmax_apw_;
        }

        inline int lmmax_apw()
        {
            return Utils::lmmax(lmax_apw_);
        }
        
        inline int lmax_pw()
        {
            return lmax_pw_;
        }

        inline int lmmax_pw()
        {
            return Utils::lmmax(lmax_pw_);
        }
        
        inline int lmax_rho()
        {
            return lmax_rho_;
        }

        inline int lmmax_rho()
        {
            return Utils::lmmax(lmax_rho_);
        }
        
        inline int lmax_pot()
        {
            return lmax_pot_;
        }

        inline int lmax_beta()
        {
            return unit_cell_->lmax_beta();
        }

        inline int lmmax_pot()
        {
            return Utils::lmmax(lmax_pot_);
        }

        inline double aw_cutoff()
        {
            return aw_cutoff_;
        }

        inline void set_aw_cutoff(double aw_cutoff__)
        {
            aw_cutoff_ = aw_cutoff__;
        }

        /// Return plane-wave cutoff for G-vectors.
        inline double pw_cutoff()
        {
            return pw_cutoff_;
        }

        /// Set plane-wave cutoff.
        inline void set_pw_cutoff(double pw_cutoff__)
        {
            pw_cutoff_ = pw_cutoff__;
        }

        inline double gk_cutoff()
        {
            return gk_cutoff_;
        }

        inline void set_gk_cutoff(double gk_cutoff__)
        {
            gk_cutoff_ = gk_cutoff__;
        }

        inline int num_fv_states()
        {
            return num_fv_states_;
        }

        inline int num_bands()
        {
            return num_bands_;
        }
        
        inline int num_spins()
        {
            assert(num_spins_ == 1 || num_spins_ == 2);
            
            return num_spins_;
        }

        inline int num_mag_dims()
        {
            assert(num_mag_dims_ == 0 || num_mag_dims_ == 1 || num_mag_dims_ == 3);
            
            return num_mag_dims_;
        }

        inline int max_occupancy()
        {
            return (2 / num_spins());
        }
        
        inline void set_so_correction(bool so_correction__)
        {
            so_correction_ = so_correction__; 
        }

        inline void set_uj_correction(bool uj_correction__)
        {
            uj_correction_ = uj_correction__; 
        }

        inline bool so_correction()
        {
            return so_correction_;
        }
        
        inline bool uj_correction()
        {
            return uj_correction_;
        }

        inline void set_sync_flag(int flg)
        {
            sync_flag_ = flg;
            Platform::allreduce<op_max>(&sync_flag_, 1);
        }

        inline int sync_flag()
        {
            return sync_flag_;
        }
        
        inline MPI_grid& mpi_grid()
        {
            return mpi_grid_;
        }

        inline bool initialized()
        {
            return initialized_;
        }

        inline processing_unit_t processing_unit()
        {
            return processing_unit_;
        }

        inline double smearing_width()
        {
            return smearing_width_;
        }

        bool need_sv()
        {
            if (num_spins() == 2 || uj_correction() || so_correction()) return true;
            return false;
        }
        
        inline splindex<block_cyclic>& spl_fv_states()
        {
            return spl_fv_states_;
        }
        
        inline int spl_fv_states(int icol_loc)
        {
            return spl_fv_states_[icol_loc];
        }

        inline splindex<block_cyclic>& spl_spinor_wf()
        {
            return spl_spinor_wf_;
        }

        inline splindex<block>& sub_spl_spinor_wf()
        {
            return sub_spl_spinor_wf_;
        }
        
        inline int spl_spinor_wf(int jloc)
        {
            return spl_spinor_wf_[jloc];
        }
        
        inline int num_sub_bands()
        {
            return sub_spl_spinor_wf_.local_size();
        }

        inline splindex<block>& sub_spl_fv_states()
        {
            return sub_spl_fv_states_;
        }
        
        inline int sub_spl_fv_states(int idx)
        {
            return sub_spl_fv_states_[idx];
        }

        inline int idxbandglob(int sub_index)
        {
            return spl_spinor_wf_[sub_spl_spinor_wf_[sub_index]];
        }
        
        inline int idxbandloc(int sub_index)
        {
            return sub_spl_spinor_wf_[sub_index];
        }

        /// Initialize the global variables
        void initialize();

        /// Clear global variables
        void clear();

        void print_info();

        void write_json_output();

        void create_storage_file();

        void update(); // TODO: better way to update unit cell after relaxation
       
        inline int blacs_context()
        {
            return blacs_context_;
        }

        inline electronic_structure_method_t esm_type()
        {
            return esm_type_;
        }

        inline Step_function* step_function()
        {
            return step_function_;
        }

        inline double step_function(int ir)
        {
            return step_function_->theta_it(ir);
        }

        inline Reciprocal_lattice* reciprocal_lattice()
        {
            return reciprocal_lattice_;
        }

        inline Unit_cell* unit_cell()
        {
            return unit_cell_;
        }
        
        struct mixer_input_section
        {
            double beta_;
            double gamma_;
            std::string type_;
            int max_history_;

            mixer_input_section() : beta_(0.9), gamma_(1.0), type_("broyden"), max_history_(5)
            {
            }

            void read(JSON_tree parser)
            {
                if (parser.exist("mixer"))
                {
                    JSON_tree section = parser["mixer"];
                    beta_ = section["beta"].get(beta_);
                    gamma_ = section["gamma"].get(gamma_);
                    max_history_ = section["max_history"].get(max_history_);
                    type_ = section["type"].get(type_);
                }
            }
        } mixer_input_section_;
        
        /// Parse unit cell input section.
        /** The following part of the input file is parsed:
         *  \code{.json}
         *      "unit_cell" : {
         *          "lattice_vectors" : [
         *              [a1_x, a1_y, a1_z],
         *              [a2_x, a2_y, a2_z],
         *              [a3_x, a3_y, a3_z]
         *          ],
         *
         *          "lattice_vectors_scale" : scale,
         *
         *          "atom_types" : [label_A, label_B, ...],
         *
         *          "atom_files" : {
         *              label_A : file_A, 
         *              label_B : file_B,
         *              ...
         *          },
         *
         *          "atoms" : {
         *              label_A: [
         *                  coordinates_A_1, 
         *                  coordinates_A_2,
         *                  ...
         *              ],
         *              label_B : [
         *                  coordinates_B_1,
         *                  coordinates_B_2,
         *                  ...
         *              ]
         *          }
         *      }
         *  \endcode
         */
        struct unit_cell_input_section
        {
            double lattice_vectors_[3][3];

            std::vector<std::string> labels_;
            std::map<std::string, std::string> atom_files_;
            std::vector< std::vector< std::vector<double> > > coordinates_;

            void read(JSON_tree parser)
            {
                std::vector<double> a0, a1, a2;
                parser["unit_cell"]["lattice_vectors"][0] >> a0;
                parser["unit_cell"]["lattice_vectors"][1] >> a1;
                parser["unit_cell"]["lattice_vectors"][2] >> a2;

                double scale = parser["unit_cell"]["lattice_vectors_scale"].get(1.0);

                for (int x = 0; x < 3; x++)
                {
                    lattice_vectors_[0][x] = a0[x] * scale;
                    lattice_vectors_[1][x] = a1[x] * scale;
                    lattice_vectors_[2][x] = a2[x] * scale;
                }

                labels_.clear();
                coordinates_.clear();
                
                for (int iat = 0; iat < (int)parser["unit_cell"]["atom_types"].size(); iat++)
                {
                    std::string label;
                    parser["unit_cell"]["atom_types"][iat] >> label;
                    for (int i = 0; i < (int)labels_.size(); i++)
                    {
                        if (labels_[i] == label) 
                            error_global(__FILE__, __LINE__, "atom type with such label is already in list");
                    }
                    labels_.push_back(label);
                }
                
                if (parser["unit_cell"].exist("atom_files"))
                {
                    for (int iat = 0; iat < (int)labels_.size(); iat++)
                    {
                        if (parser["unit_cell"]["atom_files"].exist(labels_[iat]))
                        {
                            std::string fname;
                            parser["unit_cell"]["atom_files"][labels_[iat]] >> fname;
                            atom_files_[labels_[iat]] = fname;
                        }
                        else
                        {
                            atom_files_[labels_[iat]] = "";
                        }
                    }
                }
                
                for (int iat = 0; iat < (int)labels_.size(); iat++)
                {
                    coordinates_.push_back(std::vector< std::vector<double> >());
                    for (int ia = 0; ia < parser["unit_cell"]["atoms"][labels_[iat]].size(); ia++)
                    {
                        std::vector<double> v;
                        parser["unit_cell"]["atoms"][labels_[iat]][ia] >> v;

                        if (!(v.size() == 3 || v.size() == 6)) error_global(__FILE__, __LINE__, "wrong coordinates size");
                        if (v.size() == 3) v.resize(6, 0.0);

                        coordinates_[iat].push_back(v);
                    }
                }
            }

        } unit_cell_input_section_;

        void read_unit_cell_input();

        inline double iterative_solver_tolerance()
        {
            return iterative_solver_tolerance_;
        }

        inline void set_iterative_solver_tolerance(double tol)
        {
            iterative_solver_tolerance_ = tol;
        }

        #ifdef _SCALAPACK_
        int create_blacs_context();
        #endif

        inline standard_evp* std_evp_solver()
        {
            return std_evp_solver_;
        }

        inline generalized_evp* gen_evp_solver()
        {
            return gen_evp_solver_;
        }
};

};

#endif // __GLOBAL_H__
