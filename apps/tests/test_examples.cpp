#include <sirius.h>

using namespace sirius;

void test1()
{
/* reciprocal lattice vectors in 
    inverse atomic units */
matrix3d<double> M = {{1, 0, 0},
                      {0, 1, 0},
                      {0, 0, 1}};
/* G-vector cutoff radius in
    inverse atomic units */
double Gmax = 10;
/* create a list of G-vectors;
   last boolean parameter switches
   off the reduction of G-vectors by 
   inversion symmetry */
Gvec gvec(M, Gmax, Communicator::world(), false);
/* loop over local number of G-vectors
   for current MPI rank */
for (int j = 0; j < gvec.count(); j++) {
    /* get global index of G-vector */
    int ig = gvec.offset() + j;
    /* get lattice coordinates */
    auto G = gvec.gvec(ig);
    /* get index of G-vector by lattice coordinates */
    int jg = gvec.index_by_gvec(G);
    /* check for correctness */
    if (ig != jg) {
        throw std::runtime_error("wrong index");
    }
}


}

void test2()
{

///* reciprocal lattice vectors in 
//    inverse atomic units */
//matrix3d<double> M = {{1, 0, 0},
//                      {0, 1, 0},
//                      {0, 0, 1}};
///* G-vector cutoff radius in
//    inverse atomic units */
//double Gmax = 10;
///* create a list of G-vectors;
//   last boolean parameter switches
//   off the reduction of G-vectors by 
//   inversion symmetry */
//Gvec gvec(M, Gmax, Communicator::world(), false);
///* dimensions of the FFT box */
//std::array<int, 3> dims = {20, 20, 20};
///* create parallel FFT driver with CPU backend */
//FFT3D fft(dims, Communicator::world(), device_t::CPU);
///* create G-vector partition; second communicator 
//   is used in remappting data for FFT */
//Gvec_partition gvp(gvec, fft.comm(), Communicator::self());
///* create data buffer with local number of G-vectors
//   and fill with random numbers */
//mdarray<double_complex, 1> f(gvp.gvec_count_fft());
//f = [](int64_t){
//  return utils::random<double_complex>();
//};
///* prepare FFT driver for a given G-vector partition */
//fft.prepare(gvp);
///* transform to real-space domain */
//fft.transform<1>(f.at(memory_t::host));
///* now the fft buffer contains the real space values */
//for (int j0 = 0; j0 < fft.size(0); j0++) {
//    for (int j1 = 0; j1 < fft.size(1); j1++) {
//        for (int j2 = 0; j2 < fft.local_size_z(); j2++) {
//            int idx = fft.index_by_coord(j0, j1, j2);
//            /* get the value at (j0, j1, j2) point of the grid */
//            auto val = fft.buffer(idx);
//        }
//    }
//}
////for (int i = 0; i < fft.local_size(); i++) {
////    /* get the value */
////    auto val = fft.buffer(i);
////}
///* dismiss the FFT driver */
//fft.dismiss();
}

void test3()
{

/* reciprocal lattice vectors in 
    inverse atomic units */
matrix3d<double> M = {{1, 0, 0},
                      {0, 1, 0},
                      {0, 0, 1}};
/* G-vector cutoff radius in
    inverse atomic units */
double Gmax = 10;
/* create a list of G-vectors;
   last boolean parameter switches
   off the reduction of G-vectors by 
   inversion symmetry */
Gvec gvec(M, Gmax, Communicator::world(), false);
/* create G-vector partition; second communicator 
   is used in remappting data for FFT */
Gvec_partition gvp(gvec, Communicator::world(), Communicator::self());
/* number of wave-functions */
int N = 100;
/* create scalar wave-functions for N bands */
Wave_functions wf(gvp, N, memory_t::host);
/* spin index for scalar wave-functions */
int ispn = 0;
/* fill with random numbers */
wf.pw_coeffs(ispn).prime() = [](int64_t, int64_t){
    return utils::random<double_complex>();
};
/* create a 2x2 BLACS grid */
BLACS_grid grid(Communicator::world(), 2, 2);
/* cyclic block size */
int bs = 16;
/* create a distributed overlap matrix */
dmatrix<double_complex> o(N, N, grid, bs, bs);
/* create temporary wave-functions */
Wave_functions tmp(gvp, N, memory_t::host);
/* orthogonalize wave-functions */
orthogonalize<double_complex, 0, 0>(memory_t::host, linalg_t::blas, ispn, {&wf},
                                    0, N, o, tmp);
/* compute overlap */
inner(memory_t::host, linalg_t::blas, ispn, wf, 0, N, wf, 0, N, o, 0, 0);
/* get the diagonal of the matrix */
auto d = o.get_diag(N);
/* check diagonal */
for (int i = 0; i < N; i++) {
    if (std::abs(d[i] - 1.0) > 1e-10) {
        throw std::runtime_error("wrong overlap");
    }
}
}

void test4()
{
///* dimensions of the FFT box */
//std::array<int, 3> dims = {20, 20, 20};
///* reciprocal lattice vectors in 
//    inverse atomic units */
//matrix3d<double> M = {{1, 0, 0},
//                      {0, 1, 0},
//                      {0, 0, 1}};
///* G-vector cutoff radius in
//    inverse atomic units */
//double Gmax = 10;
///* create a list of G-vectors;
//   last boolean parameter switches
//   off the reduction of G-vectors by 
//   inversion symmetry */
//Gvec gvec(M, Gmax, Communicator::world(), false);
///* create sequential FFT driver with CPU backend */
//FFT3D fft(dims, Communicator::self(), device_t::CPU);
///* potential on a real-space grid */
//std::vector<double> v(fft.local_size(), 1);
///* create G-vector partition; second communicator 
//   is used in remappting wave-functions */
//Gvec_partition gvp(gvec, fft.comm(), Communicator::world());
///* prepare FFT driver */
//fft.prepare(gvp);
///* number of wave-functions */
//int N = 100;
///* create scalar wave-functions for N bands */
//Wave_functions wf(gvp, N, memory_t::host);
///* spin index for scalar wave-functions */
//int ispn = 0;
///* fill with random numbers */
//wf.pw_coeffs(ispn).prime() = [](int64_t, int64_t){
//    return utils::random<double_complex>();
//};
///* resulting |v*wf> */
//Wave_functions vwf(gvp, N, memory_t::host);
///* remap wave-functions */
//wf.pw_coeffs(ispn).remap_forward(N, 0, nullptr);
///* prepare the target wave-functions */
//vwf.pw_coeffs(ispn).set_num_extra(N, 0, nullptr);
///* loop over local number of bands */
//for (int i = 0; i < wf.pw_coeffs(ispn).spl_num_col().local_size(); i++) {
//    /* transform to real-space */
//    fft.transform<1>(wf.pw_coeffs(ispn).extra().at(memory_t::host, 0, i));
//    /* multiply by potential */
//    for (int j = 0; j < fft.local_size(); j++) {
//        fft.buffer(j) *= v[j];
//    }
//    /* transform to reciprocal space */
//    fft.transform<-1>(vwf.pw_coeffs(ispn).extra().at(memory_t::host, 0, i));
//}
///* remap to default "slab" storage */
//vwf.pw_coeffs(ispn).remap_backward(N, 0);
///* dismiss the FFT driver */
//fft.dismiss();

}

void test5()
{
/* create simulation context */
Simulation_context ctx("{\"parameters\" : "
    "{\"electronic_structure_method\":"
    "\"pseudopotential\"}}",
     Communicator::world());
/* lattice constant */
double a{5};
/* set lattice vectors */
ctx.unit_cell().set_lattice_vectors({{a,0,0}, 
                                     {0,a,0}, 
                                     {0,0,a}});
/* add atom type */
ctx.unit_cell().add_atom_type("H");
/* get created atom type */
auto& atype = ctx.unit_cell().atom_type(0);
/* set charge */
atype.zn(1);
/* set radial grid */
atype.set_radial_grid(radial_grid_t::lin_exp,
                      1000, 0, 2, 6);
/* create beta radial function */
std::vector<double> beta(atype.num_mt_points());
for (int i = 0; i < atype.num_mt_points(); i++) {
    double x = atype.radial_grid(i);
    beta[i] = std::exp(-x) * (4 - x * x);
}
/* add radial function for l=0 */
atype.add_beta_radial_function(0, beta);
/* add atom */
ctx.unit_cell().add_atom("H", {0, 0, 0});
/* initialize the context */
ctx.initialize();
/* get FFT driver */
//auto& fft = ctx.spfft();
/* get G-vectors */
//auto& gvec = ctx.gvec();
}

int main(int argn, char** argv)
{
    cmd_args args;

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(1);
    test1();
    test2();
    test3();
    test4();
    test5();
    sirius::finalize();
}
