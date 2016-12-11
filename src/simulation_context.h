// Copyright (c) 2013-2016 Anton Kozhevnikov, Thomas Schulthess
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

/** \file simulation_context.h
 *   
 *  \brief Contains definition and implementation of Simulation_parameters and Simulation_context classes.
 */

#ifndef __SIMULATION_CONTEXT_H__
#define __SIMULATION_CONTEXT_H__

#include "simulation_parameters.h"
#include "mpi_grid.hpp"
#include "step_function.h"
#include "real_space_prj.h"
#include "version.h"
#include "augmentation_operator.h"

#ifdef __GPU
extern "C" void generate_phase_factors_gpu(int num_gvec_loc__,
                                           int num_atoms__,
                                           int const* gvec__,
                                           double const* atom_pos__,
                                           cuDoubleComplex* phase_factors__);
#endif

namespace sirius {

/// Simulation context is a set of parameters and objects describing a single simulation. 
/** The order of initialization of the simulation context is the following: first, the default parameter 
 *  values are set in set_defaults() method, then (optionally) import() method is called and the parameters are 
 *  overwritten with the those from the input file, and finally, the user sets the values with set_...() metods.
 *  Then the atom types and atoms can be added to the simulation and the context can be initialized with the 
 *  corresponding method. */
class Simulation_context: public Simulation_parameters
{
    private:

        /// Communicator for this simulation.
        Communicator const& comm_;

        /// Unit cell of the simulation.
        Unit_cell unit_cell_;

        /// MPI grid for this simulation.
        std::unique_ptr<MPI_grid> mpi_grid_;
        
        /// 2D MPI grid for the FFT driver.
        std::unique_ptr<MPI_grid> mpi_grid_fft_;

        /// 2D MPI grid for the FFT driver of the Hloc operation.
        std::unique_ptr<MPI_grid> mpi_grid_fft_vloc_;

        /// 2D BLACS grid for distributed linear algebra operations.
        std::unique_ptr<BLACS_grid> blacs_grid_;

        /// 1D BLACS grid for a "slice" data distribution of full-potential wave-functions.
        /** This grid is used to distribute band index and keep a whole wave-function */
        std::unique_ptr<BLACS_grid> blacs_grid_slice_;

        std::unique_ptr<FFT3D> fft_;

        std::unique_ptr<FFT3D> fft_coarse_;

        /// Step function is used in full-potential methods.
        std::unique_ptr<Step_function> step_function_;

        Gvec gvec_;

        Gvec gvec_coarse_;

        std::vector<Augmentation_operator> augmentation_op_;

        std::unique_ptr<Real_space_prj> real_space_prj_;

        /// Creation time of the context.
        timeval start_time_;

        std::string start_time_tag_;

        ev_solver_t std_evp_solver_type_{ev_lapack};

        ev_solver_t gen_evp_solver_type_{ev_lapack};

        mdarray<double_complex, 3> phase_factors_;
        
        #ifdef __GPU
        mdarray<int, 2> gvec_coord_;
        std::vector<mdarray<double, 2>> atom_coord_;
        #endif

        double time_active_;
        
        bool initialized_{false};

        void init_fft();

        Simulation_context(Simulation_context const&) = delete;

        void init()
        {
            gettimeofday(&start_time_, NULL);
            
            tm const* ptm = localtime(&start_time_.tv_sec); 
            char buf[100];
            strftime(buf, sizeof(buf), "%Y%m%d%H%M%S", ptm);
            start_time_tag_ = std::string(buf);
        }

    public:
        
        Simulation_context(std::string const& fname__,
                           Communicator const& comm__)
            : comm_(comm__),
              unit_cell_(*this, comm_)
        {
            PROFILE("sirius::Simulation_context::Simulation_context");
            init();
            import(fname__);
            unit_cell_.import(unit_cell_input_section_);
        }

        Simulation_context(Communicator const& comm__)
            : comm_(comm__),
              unit_cell_(*this, comm_)
        {
            PROFILE("sirius::Simulation_context::Simulation_context");
            init();
        }
        
        ~Simulation_context()
        {
            PROFILE("sirius::Simulation_context::~Simulation_context");

            //time_active_ += runtime::wtime();

            //if (mpi_comm_world().rank() == 0 && initialized_) {
            //    printf("Simulation_context active time: %.4f sec.\n", time_active_);
            //}
        }

        /// Initialize the similation (can only be called once).
        void initialize();

        void print_info();

        Unit_cell& unit_cell()
        {
            return unit_cell_;
        }

        Unit_cell const& unit_cell() const
        {
            return unit_cell_;
        }

        Step_function const& step_function() const
        {
            return *step_function_;
        }

        inline FFT3D& fft() const
        {
            return *fft_;
        }

        inline FFT3D& fft_coarse() const
        {
            return *fft_coarse_;
        }

        Gvec const& gvec() const
        {
            return gvec_;
        }

        Gvec const& gvec_coarse() const
        {
            return gvec_coarse_;
        }

        Communicator const& comm() const
        {
            return comm_;
        }

        MPI_grid const& mpi_grid() const
        {
            return *mpi_grid_;
        }

        MPI_grid const& mpi_grid_fft() const
        {
            return *mpi_grid_fft_;
        }

        MPI_grid const& mpi_grid_fft_vloc() const
        {
            return *mpi_grid_fft_vloc_;
        }

        BLACS_grid const& blacs_grid() const
        {
            return *blacs_grid_;
        }

        BLACS_grid const& blacs_grid_slice() const
        {
            return *blacs_grid_slice_;
        }

        inline int num_fv_states() const
        {
            return num_fv_states_;
        }

        inline int num_bands() const
        {
            return num_spins() * num_fv_states_;
        }
        
        void create_storage_file() const
        {
            if (comm_.rank() == 0) {
                /* create new hdf5 file */
                HDF5_tree fout(storage_file_name, true);
                fout.create_node("parameters");
                fout.create_node("effective_potential");
                fout.create_node("effective_magnetic_field");
                fout.create_node("density");
                fout.create_node("magnetization");
                
                fout["parameters"].write("num_spins", num_spins());
                fout["parameters"].write("num_mag_dims", num_mag_dims());
                fout["parameters"].write("num_bands", num_bands());

                mdarray<int, 2> gv(3, gvec_.num_gvec());
                for (int ig = 0; ig < gvec_.num_gvec(); ig++) {
                    auto G = gvec_.gvec(ig);
                    for (int x: {0, 1, 2}) gv(x, ig) = G[x];
                }
                fout["parameters"].write("num_gvec", gvec_.num_gvec());
                fout["parameters"].write("gvec", gv);
            }
            comm_.barrier();
        }

        Real_space_prj& real_space_prj() const
        {
            return *real_space_prj_;
        }

        inline std::string const& start_time_tag() const
        {
            return start_time_tag_;
        }

        inline void set_iterative_solver_tolerance(double tolerance__)
        {
            iterative_solver_input_section_.energy_tolerance_ = tolerance__;
        }

        inline double iterative_solver_tolerance() const
        {
            return iterative_solver_input_section_.energy_tolerance_;
        }

        inline ev_solver_t std_evp_solver_type() const
        {
            return std_evp_solver_type_;
        }
    
        inline ev_solver_t gen_evp_solver_type() const
        {
            return gen_evp_solver_type_;
        }

        inline Augmentation_operator const& augmentation_op(int iat__) const
        {
            return augmentation_op_[iat__];
        }

        /// Phase factors \f$ e^{i {\bf G} {\bf r}_{\alpha}} \f$
        inline double_complex gvec_phase_factor(int ig__, int ia__) const
        {
            auto G = gvec_.gvec(ig__);
            return phase_factors_(0, G[0], ia__) * phase_factors_(1, G[1], ia__) * phase_factors_(2, G[2], ia__);
        }

        inline int gvec_count() const
        {
            return gvec_.gvec_count(comm_.rank());
        }

        inline int gvec_offset() const
        {
            return gvec_.gvec_offset(comm_.rank());
        }

        #ifdef __GPU
        inline mdarray<int, 2> const& gvec_coord() const
        {
            return gvec_coord_;
        }

        inline mdarray<double, 2> const& atom_coord(int iat__) const
        {
            return atom_coord_[iat__];
        }
        #endif

        inline void generate_phase_factors(int iat__, mdarray<double_complex, 2>& phase_factors__) const
        {
            int na = unit_cell_.atom_type(iat__).num_atoms();
            switch (processing_unit_) {
                case CPU: {
                    #pragma omp parallel for
                    for (int igloc = 0; igloc < gvec_count(); igloc++) {
                        int ig = gvec_offset() + igloc;
                        for (int i = 0; i < na; i++) {
                            int ia = unit_cell_.atom_type(iat__).atom_id(i);
                            phase_factors__(igloc, i) = gvec_phase_factor(ig, ia);
                        }
                    }
                    break;
                }
                case GPU: {
                    #ifdef __GPU
                    generate_phase_factors_gpu(gvec_count(), na, gvec_coord().at<GPU>(), atom_coord(iat__).at<GPU>(),
                                               phase_factors__.at<GPU>());
                    #else
                    TERMINATE_NO_GPU
                    #endif
                    break;
                }
            }
        }
};

inline void Simulation_context::init_fft()
{
    auto rlv = unit_cell_.reciprocal_lattice_vectors();

    auto& comm = mpi_grid_->communicator(1 << _mpi_dim_k_row_ | 1 << _mpi_dim_k_col_);

    if (!(control_input_section_.fft_mode_ == "serial" || control_input_section_.fft_mode_ == "parallel")) {
        TERMINATE("wrong FFT mode");
    }

    mpi_grid_fft_ = std::unique_ptr<MPI_grid>(new MPI_grid({mpi_grid_->dimension_size(_mpi_dim_k_row_),
                                                            mpi_grid_->dimension_size(_mpi_dim_k_col_)}, comm));

    if (control_input_section_.fft_mode_ == "serial") {
        /* serial FFT in Hloc */
        mpi_grid_fft_vloc_ = std::unique_ptr<MPI_grid>(new MPI_grid({1, comm.size()}, comm));
    } else {
        /* parallel FFT in Hloc */
        mpi_grid_fft_vloc_ = std::unique_ptr<MPI_grid>(new MPI_grid({mpi_grid_->dimension_size(_mpi_dim_k_row_),
                                                                     mpi_grid_->dimension_size(_mpi_dim_k_col_)}, comm));
    }

    /* create FFT driver for dense mesh (density and potential) */
    fft_ = std::unique_ptr<FFT3D>(new FFT3D(Utils::find_translations(pw_cutoff(), rlv),
                                            mpi_grid_fft_->communicator(1 << 0),
                                            processing_unit(), 0.9));

    /* create a list of G-vectors for dense FFT grid */
    /* G-vectors are divided between all available MPI ranks (comm_.size()) */
    gvec_ = Gvec({0, 0, 0}, rlv, pw_cutoff(), fft_->grid(),
                 comm_.size(), fft_->comm(), control_input_section_.reduce_gvec_);

    //double gpu_workload = (mpi_grid_fft_vloc_->communicator(1 << 0).size() == 1) ? 1.0 : 0.9;
    double gpu_workload = (mpi_grid_fft_vloc_->communicator().size() == 1) ? 1.0 : 0.9;
    /* create FFT driver for coarse mesh */
    fft_coarse_ = std::unique_ptr<FFT3D>(new FFT3D(Utils::find_translations(2 * gk_cutoff(), rlv),
                                                   mpi_grid_fft_vloc_->communicator(1 << 0),
                                                   processing_unit(), gpu_workload));

    /* create a list of G-vectors for corase FFT grid */
    gvec_coarse_ = Gvec({0, 0, 0}, rlv, gk_cutoff() * 2, fft_coarse_->grid(),
                        comm_.size(), fft_coarse_->comm(), control_input_section_.reduce_gvec_);
}

inline void Simulation_context::initialize()
{
    PROFILE("sirius::Simulation_context::initialize");

    /* can't initialize twice */
    if (initialized_) {
        TERMINATE("Simulation context is already initialized.");
    }
    
    /* get processing unit */
    std::string pu = control_input_section_.processing_unit_;
    if (pu == "") {
        #ifdef __GPU
        pu = "gpu";
        #else
        pu = "cpu";
        #endif
    }
    if (pu == "cpu") {
        processing_unit_ = CPU;
    } else {
        if (pu == "gpu") {
            processing_unit_ = GPU;
        } else {
            TERMINATE("wrong processing unit");
        }
    }

    /* check if we can use a GPU device */
    if (processing_unit() == GPU) {
        #ifndef __GPU
        TERMINATE_NO_GPU
        #endif
    }
    
    /* check MPI grid dimensions and set a default grid if needed */
    if (!control_input_section_.mpi_grid_dims_.size()) {
        control_input_section_.mpi_grid_dims_ = {comm_.size()};
    }
    
    /* can't use reduced G-vectors in LAPW code */
    if (full_potential()) {
        control_input_section_.reduce_gvec_ = false;
    }

    if (!iterative_solver_input_section_.type_.size()) {
        if (full_potential()) {
            iterative_solver_input_section_.type_ = "exact";
        } else {
            iterative_solver_input_section_.type_ = "davidson";
        }
    }

    /* setup MPI grid */
    mpi_grid_ = std::unique_ptr<MPI_grid>(new MPI_grid(control_input_section_.mpi_grid_dims_, comm_));

    blacs_grid_ = std::unique_ptr<BLACS_grid>(new BLACS_grid(mpi_grid_->communicator(1 << _mpi_dim_k_row_ | 1 << _mpi_dim_k_col_), 
                                                             mpi_grid_->dimension_size(_mpi_dim_k_row_), mpi_grid_->dimension_size(_mpi_dim_k_col_)));
    
    blacs_grid_slice_ = std::unique_ptr<BLACS_grid>(new BLACS_grid(blacs_grid_->comm(), 1, blacs_grid_->comm().size()));

    /* initialize variables, related to the unit cell */
    unit_cell_.initialize();

    /* find the cutoff for G+k vectors (derived from rgkmax (aw_cutoff here) and maximum MT radius) */
    if (full_potential()) {
        set_gk_cutoff(aw_cutoff() / unit_cell_.min_mt_radius());
    }

    if (esm_type() == electronic_structure_method_t::pseudopotential) {
        lmax_rho_ = unit_cell_.lmax() * 2;
        lmax_pot_ = unit_cell_.lmax() * 2;
    }

    /* initialize FFT interface */
    init_fft();

    if (control().print_memory_usage_) {
        MEMORY_USAGE_INFO();
    }

    //if (comm_.rank() == 0)
    //{
    //    unit_cell_.write_cif();
    //    unit_cell_.write_json();
    //}

    if (unit_cell_.num_atoms() != 0) {
        unit_cell_.symmetry().check_gvec_symmetry(gvec_, comm_);
        if (!full_potential()) {
            unit_cell_.symmetry().check_gvec_symmetry(gvec_coarse_, comm_);
        }
    }

    auto& fft_grid = fft().grid();
    std::pair<int, int> limits(0, 0);
    for (int x: {0, 1, 2}) {
        limits.first = std::min(limits.first, fft_grid.limits(x).first); 
        limits.second = std::max(limits.second, fft_grid.limits(x).second); 
    }

    phase_factors_ = mdarray<double_complex, 3>(3, limits, unit_cell().num_atoms());

    #pragma omp parallel for
    for (int i = limits.first; i <= limits.second; i++) {
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            auto pos = unit_cell_.atom(ia).position();
            for (int x: {0, 1, 2}) {
                phase_factors_(x, i, ia) = std::exp(double_complex(0.0, twopi * (i * pos[x])));
            }
        }
    }
    
    if (full_potential()) {
        step_function_ = std::unique_ptr<Step_function>(new Step_function(unit_cell_, fft_.get(), gvec_, comm_));
    }

    if (iterative_solver_input_section().real_space_prj_) {
        STOP();
        //real_space_prj_ = new Real_space_prj(unit_cell_, comm_, iterative_solver_input_section().R_mask_scale_,
        //                                     iterative_solver_input_section().mask_alpha_,
        //                                     gk_cutoff(), num_fft_streams(), num_fft_workers());
    }

    /* take 10% of empty non-magnetic states */
    if (num_fv_states_ < 0) {
        num_fv_states_ = static_cast<int>(1e-8 + unit_cell_.num_valence_electrons() / 2.0) +
                                          std::max(10, static_cast<int>(0.1 * unit_cell_.num_valence_electrons()));
    }
    
    if (num_fv_states() < static_cast<int>(unit_cell_.num_valence_electrons() / 2.0)) {
        std::stringstream s;
        s << "not enough first-variational states : " << num_fv_states();
        TERMINATE(s);
    }

    if (cyclic_block_size() < 0) {
        double a = std::min(std::log2(double(num_fv_states_) / blacs_grid_->num_ranks_col()),
                            std::log2(double(num_fv_states_) / blacs_grid_->num_ranks_row()));
        if (a < 1) {
            control_input_section_.cyclic_block_size_ = 2;
        } else {
            control_input_section_.cyclic_block_size_ = static_cast<int>(std::min(128.0, std::pow(2.0, static_cast<int>(a))) + 1e-12);
        }
    }
    
    std::string evsn[] = {std_evp_solver_name(), gen_evp_solver_name()};

    if (mpi_grid_->size(1 << _mpi_dim_k_row_ | 1 << _mpi_dim_k_col_) == 1) {
        if (evsn[0] == "") {
            #if defined(__GPU) && defined(__MAGMA)
            evsn[0] = "magma";
            #else
            evsn[0] = "lapack";
            #endif
        }
        if (evsn[1] == "") {
            #if defined(__GPU) && defined(__MAGMA)
            evsn[1] = "magma";
            #else
            evsn[1] = "lapack";
            #endif
        }
    } else {
        if (evsn[0] == "") {
            #ifdef __SCALAPACK
            evsn[0] = "scalapack";
            #endif
            #ifdef __ELPA
            evsn[0] = "elpa1";
            #endif
        }
        if (evsn[1] == "") {
            #ifdef __SCALAPACK
            evsn[1] = "scalapack";
            #endif
            #ifdef __ELPA
            evsn[1] = "elpa1";
            #endif
        }
    }

    ev_solver_t* evst[] = {&std_evp_solver_type_, &gen_evp_solver_type_};

    std::map<std::string, ev_solver_t> str_to_ev_solver_t;

    str_to_ev_solver_t["lapack"]    = ev_lapack;
    str_to_ev_solver_t["scalapack"] = ev_scalapack;
    str_to_ev_solver_t["elpa1"]     = ev_elpa1;
    str_to_ev_solver_t["elpa2"]     = ev_elpa2;
    str_to_ev_solver_t["magma"]     = ev_magma;
    str_to_ev_solver_t["plasma"]    = ev_plasma;
    str_to_ev_solver_t["rs_cpu"]    = ev_rs_cpu;
    str_to_ev_solver_t["rs_gpu"]    = ev_rs_gpu;

    for (int i: {0, 1}) {
        auto name = evsn[i];

        if (str_to_ev_solver_t.count(name) == 0) {
            std::stringstream s;
            s << "wrong eigen value solver " << name;
            TERMINATE(s);
        }
        *evst[i] = str_to_ev_solver_t[name];
    }

    if (control().verbosity_ > 0 && comm_.rank() == 0) {
        print_info();
    }

    if (esm_type() == electronic_structure_method_t::pseudopotential) {
        /* create augmentation operator Q_{xi,xi'}(G) here */
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            augmentation_op_.push_back(std::move(Augmentation_operator(comm_, unit_cell_.atom_type(iat), gvec_, unit_cell_.omega())));
        }
    }
    
    if (processing_unit() == GPU) {
        #ifdef __GPU
        gvec_coord_ = mdarray<int, 2>(gvec_count(), 3, memory_t::host | memory_t::device, "gvec_coord_");
        for (int igloc = 0; igloc < gvec_count(); igloc++) {
            int ig = gvec_offset() + igloc;
            auto G = gvec_.gvec(ig);
            for (int x: {0, 1, 2}) {
                gvec_coord_(igloc, x) = G[x];
            }
        }
        gvec_coord_.copy_to_device();

        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            int nat = unit_cell_.atom_type(iat).num_atoms();
            atom_coord_.push_back(std::move(mdarray<double, 2>(nat, 3, memory_t::host | memory_t::device)));
            for (int i = 0; i < nat; i++) {
                int ia = unit_cell_.atom_type(iat).atom_id(i);
                for (int x: {0, 1, 2}) {
                    atom_coord_.back()(i, x) = unit_cell_.atom(ia).position()[x];
                }
            }
            atom_coord_.back().copy_to_device();
        }
        #endif
    }
    
    //time_active_ = -runtime::wtime();

    initialized_ = true;
}

inline void Simulation_context::print_info()
{
    tm const* ptm = localtime(&start_time_.tv_sec); 
    char buf[100];
    strftime(buf, sizeof(buf), "%a, %e %b %Y %H:%M:%S", ptm);

    printf("\n");
    printf("SIRIUS version : %2i.%02i\n", major_version, minor_version);
    printf("git hash       : %s\n", git_hash);
    printf("build date     : %s\n", build_date);
    printf("start time     : %s\n", buf);
    printf("\n");
    printf("number of MPI ranks           : %i\n", comm_.size());
    printf("MPI grid                      :");
    for (int i = 0; i < mpi_grid_->num_dimensions(); i++) {
        printf(" %i", mpi_grid_->dimension_size(i));
    }
    printf("\n");
    printf("maximum number of OMP threads : %i\n", omp_get_max_threads());
    printf("number of independent FFTs    : %i\n", mpi_grid_fft_->dimension_size(1));
    printf("FFT comm size                 : %i\n", mpi_grid_fft_->dimension_size(0));

    printf("\n");
    printf("FFT context for density and potential\n");
    printf("=====================================\n");
    printf("  plane wave cutoff                     : %f\n", pw_cutoff());
    auto fft_grid = fft_->grid();
    printf("  grid size                             : %i %i %i   total : %i\n", fft_grid.size(0),
                                                                                fft_grid.size(1),
                                                                                fft_grid.size(2),
                                                                                fft_grid.size());
    printf("  grid limits                           : %i %i   %i %i   %i %i\n", fft_grid.limits(0).first,
                                                                                fft_grid.limits(0).second,
                                                                                fft_grid.limits(1).first,
                                                                                fft_grid.limits(1).second,
                                                                                fft_grid.limits(2).first,
                                                                                fft_grid.limits(2).second);
    printf("  number of G-vectors within the cutoff : %i\n", gvec_.num_gvec());
    printf("  local number of G-vectors             : %i\n", gvec_.gvec_count(0));
    printf("  number of G-shells                    : %i\n", gvec_.num_shells());
    printf("\n");
    printf("FFT context for coarse grid\n");
    printf("===========================\n");
    printf("  plane wave cutoff                     : %f\n", 2 * gk_cutoff());
    fft_grid = fft_coarse_->grid();
    printf("  grid size                             : %i %i %i   total : %i\n", fft_grid.size(0),
                                                                                fft_grid.size(1),
                                                                                fft_grid.size(2),
                                                                                fft_grid.size());
    printf("  grid limits                           : %i %i   %i %i   %i %i\n", fft_grid.limits(0).first,
                                                                                fft_grid.limits(0).second,
                                                                                fft_grid.limits(1).first,
                                                                                fft_grid.limits(1).second,
                                                                                fft_grid.limits(2).first,
                                                                                fft_grid.limits(2).second);
    printf("  number of G-vectors within the cutoff : %i\n", gvec_coarse_.num_gvec());
    printf("  number of G-shells                    : %i\n", gvec_coarse_.num_shells());
    printf("\n");

    unit_cell_.print_info(control().verbosity_);
    for (int i = 0; i < unit_cell_.num_atom_types(); i++) {
        unit_cell_.atom_type(i).print_info();
    }

    printf("\n");
    printf("total nuclear charge               : %i\n", unit_cell_.total_nuclear_charge());
    printf("number of core electrons           : %f\n", unit_cell_.num_core_electrons());
    printf("number of valence electrons        : %f\n", unit_cell_.num_valence_electrons());
    printf("total number of electrons          : %f\n", unit_cell_.num_electrons());
    printf("total number of aw basis functions : %i\n", unit_cell_.mt_aw_basis_size());
    printf("total number of lo basis functions : %i\n", unit_cell_.mt_lo_basis_size());
    printf("number of first-variational states : %i\n", num_fv_states());
    printf("number of bands                    : %i\n", num_bands());
    printf("number of spins                    : %i\n", num_spins());
    printf("number of magnetic dimensions      : %i\n", num_mag_dims());
    printf("lmax_apw                           : %i\n", lmax_apw());
    printf("lmax_pw                            : %i\n", lmax_pw());
    printf("lmax_rho                           : %i\n", lmax_rho());
    printf("lmax_pot                           : %i\n", lmax_pot());
    printf("lmax_rf                            : %i\n", unit_cell_.lmax());
    printf("smearing width                     : %f\n", smearing_width());
    printf("cyclic block size                  : %i\n", cyclic_block_size());
    printf("|G+k| cutoff                       : %f\n", gk_cutoff());

    std::string reln[] = {"valence relativity                 : ",
                          "core relativity                    : "};

    relativity_t relt[] = {valence_relativity_, core_relativity_};
    for (int i = 0; i < 2; i++) {
        printf("%s", reln[i].c_str());
        switch (relt[i]) {
            case relativity_t::none: {
                printf("none\n");
                break;
            }
            case relativity_t::koelling_harmon: {
                printf("Koelling-Harmon\n");
                break;
            }
            case relativity_t::zora: {
                printf("zora\n");
                break;
            }
            case relativity_t::iora: {
                printf("iora\n");
                break;
            }
            case relativity_t::dirac: {
                printf("Dirac\n");
                break;
            }
        }
    }

    std::string evsn[] = {"standard eigen-value solver        : ",
                          "generalized eigen-value solver     : "};

    ev_solver_t evst[] = {std_evp_solver_type_, gen_evp_solver_type_};
    for (int i = 0; i < 2; i++) {
        printf("%s", evsn[i].c_str());
        switch (evst[i]) {
            case ev_lapack: {
                printf("LAPACK\n");
                break;
            }
            #ifdef __SCALAPACK
            case ev_scalapack: {
                printf("ScaLAPACK\n");
                break;
            }
            case ev_elpa1: {
                printf("ELPA1\n");
                break;
            }
            case ev_elpa2: {
                printf("ELPA2\n");
                break;
            }
            case ev_rs_gpu: {
                printf("RS_gpu\n");
                break;
            }
            case ev_rs_cpu: {
                printf("RS_cpu\n");
                break;
            }
            #endif
            case ev_magma: {
                printf("MAGMA\n");
                break;
            }
            case ev_plasma: {
                printf("PLASMA\n");
                break;
            }
            default: {
                TERMINATE("wrong eigen-value solver");
            }
        }
    }

    printf("processing unit                    : ");
    switch (processing_unit()) {
        case CPU: {
            printf("CPU\n");
            break;
        }
        case GPU: {
            printf("GPU\n");
            break;
        }
    }
    if (processing_unit() == GPU) {
        #ifdef __GPU
        cuda_device_info();
        #endif
    }
   
    int i{1};
    printf("\n");
    printf("XC functionals\n");
    printf("==============\n");
    for (auto& xc_label: xc_functionals()) {
        XC_functional xc(xc_label, num_spins());
        printf("%i) %s: %s\n", i, xc_label.c_str(), xc.name().c_str());
        printf("%s\n", xc.refs().c_str());
        i++;
    }
}

};

#endif
