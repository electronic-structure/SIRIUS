// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
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

/** \file simulation_context_base.h
 *   
 *  \brief Contains definition and implementation of Simulation_context_base class.
 */

#ifndef __SIMULATION_CONTEXT_BASE_H__
#define __SIMULATION_CONTEXT_BASE_H__

#include "version.h"
#include "simulation_parameters.h"
#include "mpi_grid.hpp"
#include "radial_integrals.h"

#ifdef __GPU
extern "C" void generate_phase_factors_gpu(int num_gvec_loc__,
                                           int num_atoms__,
                                           int const* gvec__,
                                           double const* atom_pos__,
                                           double_complex* phase_factors__);
#endif

namespace sirius {

/// Base class for Simulation_context.
class Simulation_context_base: public Simulation_parameters
{
    private:

        /// Communicator for this simulation.
        Communicator const& comm_;
        
        /// Unit cell of the simulation.
        Unit_cell unit_cell_;

        /// MPI grid for this simulation.
        std::unique_ptr<MPI_grid> mpi_grid_;

        /// 2D BLACS grid for distributed linear algebra operations.
        std::unique_ptr<BLACS_grid> blacs_grid_;

        /// Fine-grained FFT for density and potential.
        /** This is the FFT driver to transform periodic functions such as density and potential on the fine-grained 
         *  FFT grid. The transformation is parallel. */
        std::unique_ptr<FFT3D> fft_;
        
        /// Coarse-grained FFT for application of local potential and density summation.
        std::unique_ptr<FFT3D> fft_coarse_;

        /// G-vectors within the Gmax cutoff.
        Gvec gvec_;
        
        /// G-vectors within the 2 * |Gmax^{WF}| cutoff.
        Gvec gvec_coarse_;

        /// Creation time of the parameters.
        timeval start_time_;

        std::string start_time_tag_;

        ev_solver_t std_evp_solver_type_{ev_lapack};

        ev_solver_t gen_evp_solver_type_{ev_lapack};

        mdarray<double_complex, 3> phase_factors_;
        
        mdarray<int, 2> gvec_coord_;

        std::vector<mdarray<double, 2>> atom_coord_;
        
        mdarray<char, 1> memory_buffer_;

        std::unique_ptr<Radial_integrals_beta<false>> beta_ri_;

        std::unique_ptr<Radial_integrals_beta<true>> beta_ri_djl_;

        double time_active_;
        
        bool initialized_{false};

        void init_fft();

        /* copy constructor is forbidden */
        Simulation_context_base(Simulation_context_base const&) = delete;

        void start()
        {
            gettimeofday(&start_time_, NULL);

            tm const* ptm = localtime(&start_time_.tv_sec); 
            char buf[100];
            strftime(buf, sizeof(buf), "%Y%m%d%H%M%S", ptm);
            start_time_tag_ = std::string(buf);
        }

    public:

        Simulation_context_base(std::string const& fname__,
                                Communicator const& comm__)
            : comm_(comm__)
            , unit_cell_(*this, comm_)
        {
            start();
            import(fname__);
            unit_cell_.import(unit_cell_input_);
        }

        Simulation_context_base(Communicator const& comm__)
            : comm_(comm__)
            , unit_cell_(*this, comm_)
        {
            start();
        }

        ~Simulation_context_base()
        {
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

        BLACS_grid const& blacs_grid() const
        {
            return *blacs_grid_;
        }

        /// Total communicator of the simulation.
        Communicator const& comm() const
        {
            return comm_;
        }

        /// Communicator between k-points.
        Communicator const& comm_k() const
        {
            /* 3rd dimension of the MPI grid is used for k-point distribution */
            return mpi_grid_->communicator(1 << 2);
        }

        /// Band and BLACS grid communicator.
        Communicator const& comm_band() const
        {
            /* 1st and 2nd dimensions of the MPI grid are used for parallelization inside k-point */
            return mpi_grid_->communicator(1 << 0 | 1 << 1);
        }

        /// Communicator of the dense FFT grid.
        Communicator const& comm_fft() const
        {
            /* 1st dimension of MPI grid is used */
            return mpi_grid_->communicator(1 << 0);
        }

        /// Communicator of the coarse FFT grid.
        Communicator const& comm_fft_coarse() const
        {
            if (control().fft_mode_ == "serial") {
                return mpi_comm_self();
            } else {
                return comm_fft();
            }
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

        inline std::string const& start_time_tag() const
        {
            return start_time_tag_;
        }

        inline ev_solver_t std_evp_solver_type() const
        {
            return std_evp_solver_type_;
        }
    
        inline ev_solver_t gen_evp_solver_type() const
        {
            return gen_evp_solver_type_;
        }

        /// Phase factors \f$ e^{i {\bf G} {\bf r}_{\alpha}} \f$
        inline double_complex gvec_phase_factor(vector3d<int> G__, int ia__) const
        {
            return phase_factors_(0, G__[0], ia__) *
                   phase_factors_(1, G__[1], ia__) *
                   phase_factors_(2, G__[2], ia__);
        }

        /// Phase factors \f$ e^{i {\bf G} {\bf r}_{\alpha}} \f$
        inline double_complex gvec_phase_factor(int ig__, int ia__) const
        {
            return gvec_phase_factor(gvec_.gvec(ig__), ia__);
        }

        inline mdarray<int, 2> const& gvec_coord() const
        {
            return gvec_coord_;
        }

        inline mdarray<double, 2> const& atom_coord(int iat__) const
        {
            return atom_coord_[iat__];
        }

        /// Generate phase factors \f$ e^{i {\bf G} {\bf r}_{\alpha}} \f$ for all atoms of a given type.
        inline void generate_phase_factors(int iat__, mdarray<double_complex, 2>& phase_factors__) const
        {
            PROFILE("sirius::Simulation_context_base::generate_phase_factors");

            int na = unit_cell_.atom_type(iat__).num_atoms();
            switch (processing_unit_) {
                case CPU: {
                    #pragma omp parallel for
                    for (int igloc = 0; igloc < gvec_.count(); igloc++) {
                        int ig = gvec_.offset() + igloc;
                        for (int i = 0; i < na; i++) {
                            int ia = unit_cell_.atom_type(iat__).atom_id(i);
                            phase_factors__(igloc, i) = gvec_phase_factor(ig, ia);
                        }
                    }
                    break;
                }
                case GPU: {
                    #ifdef __GPU
                    generate_phase_factors_gpu(gvec_.count(), na, gvec_coord().at<GPU>(), atom_coord(iat__).at<GPU>(),
                                               phase_factors__.at<GPU>());
                    #else
                    TERMINATE_NO_GPU
                    #endif
                    break;
                }
            }
        }

        /// Make periodic function out of form factors.
        /** Return vector of plane-wave coefficients */
        template <index_domain_t index_domain>
        inline std::vector<double_complex> make_periodic_function(std::function<double(int, double)> form_factors__) const
        {
            PROFILE("sirius::Simulation_context_base::make_periodic_function");

            double fourpi_omega = fourpi / unit_cell_.omega();

            int ngv = (index_domain == index_domain_t::local) ? gvec_.count() : gvec().num_gvec();
            std::vector<double_complex> f_pw(ngv, double_complex(0, 0));

            #pragma omp parallel for schedule(static)
            for (int igloc = 0; igloc < gvec_.count(); igloc++) {
                /* global index of G-vector */
                int ig = gvec_.offset() + igloc;
                double g = gvec().gvec_len(ig);

                int j = (index_domain == index_domain_t::local) ? igloc : ig;
                for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                    int iat = unit_cell_.atom(ia).type_id();
                    f_pw[j] += fourpi_omega * std::conj(gvec_phase_factor(ig, ia)) * form_factors__(iat, g);
                }
            }

            if (index_domain == index_domain_t::global) {
                comm_.allgather(&f_pw[0], gvec_.offset(), gvec_.count());
            }

            return std::move(f_pw);
        }

        /// Return pointer to already allocated temporary memory buffer.
        /** Buffer can only grow in size. The requested buffer length is in bytes. */
        inline void* memory_buffer(size_t size__)
        {
            /* reallocate if needed */
            if (memory_buffer_.size() < size__) {
                memory_buffer_ = mdarray<char, 1>(size__);
            }
            return memory_buffer_.at<CPU>();
        }

        Radial_integrals_beta<false> const& beta_ri() const
        {
            return *beta_ri_;
        }

        Radial_integrals_beta<true> const& beta_ri_djl() const
        {
            return *beta_ri_djl_;
        }
};

inline void Simulation_context_base::init_fft()
{
    auto rlv = unit_cell_.reciprocal_lattice_vectors();

    if (!(control().fft_mode_ == "serial" || control().fft_mode_ == "parallel")) {
        TERMINATE("wrong FFT mode");
    }

    /* create FFT driver for dense mesh (density and potential) */
    fft_ = std::unique_ptr<FFT3D>(new FFT3D(find_translations(pw_cutoff(), rlv), comm_fft(), processing_unit())); 

    /* create a list of G-vectors for dense FFT grid; G-vectors are divided between all available MPI ranks.*/
    gvec_ = Gvec(rlv, pw_cutoff(), comm(), comm_fft(), control().reduce_gvec_);

    /* prepare fine-grained FFT driver for the entire simulation */
    fft_->prepare(gvec_.partition());

    /* create FFT driver for coarse mesh */
    auto fft_coarse_grid = FFT3D_grid(find_translations(2 * gk_cutoff(), rlv));
    auto pu = (fft_coarse_grid.size() < std::pow(64, 3)) ? CPU : processing_unit(); 
    if (full_potential()) {
        pu = processing_unit();
    }
    fft_coarse_ = std::unique_ptr<FFT3D>(new FFT3D(fft_coarse_grid, comm_fft_coarse(), pu));

    /* create a list of G-vectors for corase FFT grid */
    gvec_coarse_ = Gvec(rlv, gk_cutoff() * 2, comm(), comm_fft_coarse(), control().reduce_gvec_);
}

inline void Simulation_context_base::initialize()
{
    PROFILE("sirius::Simulation_context_base::initialize");

    /* can't initialize twice */
    if (initialized_) {
        TERMINATE("Simulation parameters are already initialized.");
    }
    /* Gamma-point calculation and non-collinear magnetism are not compatible */
    if (num_mag_dims() == 3) {
        set_gamma_point(false);
    }

    set_esm_type(parameters_input().esm_);
    set_core_relativity(parameters_input().core_relativity_);
    set_valence_relativity(parameters_input().valence_relativity_);

    /* get processing unit */
    std::string pu = control().processing_unit_;
    if (pu == "") {
        #ifdef __GPU
        pu = "gpu";
        #else
        pu = "cpu";
        #endif
    }
    set_processing_unit(pu);

    /* check if we can use a GPU device */
    if (processing_unit() == GPU) {
        #ifndef __GPU
        TERMINATE_NO_GPU
        #endif
    }
    
    /* check MPI grid dimensions and set a default grid if needed */
    if (!control().mpi_grid_dims_.size()) {
        set_mpi_grid_dims({1, 1});
    }
    if (control().mpi_grid_dims_.size() != 2) {
        TERMINATE("wrong MPI grid");
    }
    
    int npr = control_input_.mpi_grid_dims_[0];
    int npc = control_input_.mpi_grid_dims_[1];
    int npb = npr * npc;
    int npk = comm_.size() / npb;
    if (npk * npb != comm_.size()) {
        std::stringstream s;
        s << "Can't divide " << comm_.size() << " ranks into groups of size " << npb;
        TERMINATE(s);
    }

    /* setup MPI grid */
    mpi_grid_ = std::unique_ptr<MPI_grid>(new MPI_grid({npr, npc, npk}, comm_));
    
    /* setup BLACS grid */
    blacs_grid_ = std::unique_ptr<BLACS_grid>(new BLACS_grid(comm_band(), npr, npc));

    /* can't use reduced G-vectors in LAPW code */
    if (full_potential()) {
        control_input_.reduce_gvec_ = false;
    }

    if (!iterative_solver_input_.type_.size()) {
        if (full_potential()) {
            iterative_solver_input_.type_ = "exact";
        } else {
            iterative_solver_input_.type_ = "davidson";
        }
    }

    /* initialize variables, related to the unit cell */
    unit_cell_.initialize();

    /* find the cutoff for G+k vectors (derived from rgkmax (aw_cutoff here) and minimum MT radius) */
    if (full_potential()) {
        set_gk_cutoff(aw_cutoff() / unit_cell_.min_mt_radius());
    }

    if (!full_potential()) {
        set_lmax_rho(unit_cell_.lmax() * 2);
        set_lmax_pot(unit_cell_.lmax() * 2);
        set_lmax_apw(-1);
    }

    /* initialize FFT interface */
    init_fft();

    if (comm_.rank() == 0 && control().print_memory_usage_) {
        MEMORY_USAGE_INFO();
    }

    if (unit_cell_.num_atoms() != 0) {
        unit_cell_.symmetry().check_gvec_symmetry(gvec_, comm_);
        if (!full_potential()) {
            unit_cell_.symmetry().check_gvec_symmetry(gvec_coarse_, comm_);
        }
    }

    auto& fft_grid = fft().grid();
    std::pair<int, int> limits(0, 0);
    for (int x: {0, 1, 2}) {
        limits.first  = std::min(limits.first,  fft_grid.limits(x).first); 
        limits.second = std::max(limits.second, fft_grid.limits(x).second); 
    }

    phase_factors_ = mdarray<double_complex, 3>(3, limits, unit_cell().num_atoms(), memory_t::host, "phase_factors_");

    #pragma omp parallel for
    for (int i = limits.first; i <= limits.second; i++) {
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            auto pos = unit_cell_.atom(ia).position();
            for (int x: {0, 1, 2}) {
                phase_factors_(x, i, ia) = std::exp(double_complex(0.0, twopi * (i * pos[x])));
            }
        }
    }
    
    /* take 10% of empty non-magnetic states */
    if (num_fv_states() < 0) {
        set_num_fv_states(static_cast<int>(unit_cell_.num_valence_electrons() / 2.0) +
                                           std::max(10, static_cast<int>(0.1 * unit_cell_.num_valence_electrons())));
    }
    
    if (num_fv_states() < static_cast<int>(unit_cell_.num_valence_electrons() / 2.0)) {
        std::stringstream s;
        s << "not enough first-variational states : " << num_fv_states();
        TERMINATE(s);
    }
    
    /* setup the cyclic block size */
    if (cyclic_block_size() < 0) {
        double a = std::min(std::log2(double(num_fv_states()) / blacs_grid_->num_ranks_col()),
                            std::log2(double(num_fv_states()) / blacs_grid_->num_ranks_row()));
        if (a < 1) {
            control_input_.cyclic_block_size_ = 2;
        } else {
            control_input_.cyclic_block_size_ = static_cast<int>(std::min(128.0, std::pow(2.0, static_cast<int>(a))) + 1e-12);
        }
    }
    
    std::string evsn[] = {std_evp_solver_name(), gen_evp_solver_name()};

    if (comm_band().size() == 1) {
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

    if (processing_unit() == GPU) {
        gvec_coord_ = mdarray<int, 2>(gvec_.count(), 3, memory_t::host | memory_t::device, "gvec_coord_");
        for (int igloc = 0; igloc < gvec_.count(); igloc++) {
            int ig = gvec_.offset() + igloc;
            auto G = gvec_.gvec(ig);
            for (int x: {0, 1, 2}) {
                gvec_coord_(igloc, x) = G[x];
            }
        }
        gvec_coord_.copy<memory_t::host, memory_t::device>();

        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            int nat = unit_cell_.atom_type(iat).num_atoms();
            atom_coord_.push_back(std::move(mdarray<double, 2>(nat, 3, memory_t::host | memory_t::device)));
            for (int i = 0; i < nat; i++) {
                int ia = unit_cell_.atom_type(iat).atom_id(i);
                for (int x: {0, 1, 2}) {
                    atom_coord_.back()(i, x) = unit_cell_.atom(ia).position()[x];
                }
            }
            atom_coord_.back().copy<memory_t::host, memory_t::device>();
        }
    }
    
    if (!full_potential()) {
        beta_ri_ = std::unique_ptr<Radial_integrals_beta<false>>(new Radial_integrals_beta<false>(unit_cell(), gk_cutoff(), 20));

        beta_ri_djl_ = std::unique_ptr<Radial_integrals_beta<true>>(new Radial_integrals_beta<true>(unit_cell(), gk_cutoff(), 20));
    }

    //time_active_ = -runtime::wtime();

    if (control().verbosity_ > 0 && comm_.rank() == 0) {
        print_info();
    }

    if (comm_.rank() == 0 && control().print_memory_usage_) {
        MEMORY_USAGE_INFO();
    }

    initialized_ = true;
}

inline void Simulation_context_base::print_info()
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

    printf("\n");
    printf("FFT context for density and potential\n");
    printf("=====================================\n");
    printf("  comm size                             : %i\n", comm_fft().size());
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
    printf("  comm size                             : %i\n", comm_fft_coarse().size());
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
        acc::print_device_info(0);
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

} // namespace

#endif
