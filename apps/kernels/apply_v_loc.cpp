#include <sirius.h>
#include <thread>

using namespace sirius;

void apply_v_loc(double alat, double wf_cutoff, int num_bands, int num_fft_threads, int num_fft_workers)
{
    Communicator comm(MPI_COMM_WORLD);

    double a1[] = {alat, 0, 0};
    double a2[] = {0, alat, 0};
    double a3[] = {0, 0, alat};

    Simulation_parameters p;
    Unit_cell uc(p, comm);
    uc.set_lattice_vectors(a1, a2, a3);

    auto& rlv = uc.reciprocal_lattice_vectors();
    auto dims = Utils::find_translation_limits(2 * wf_cutoff, rlv);

    FFT3D<CPU> fft(dims, num_fft_threads, num_fft_workers);
    fft.init_gvec(2 * wf_cutoff, rlv);

    int num_gkvec = 0;
    for (int ig = 0; ig < fft.num_gvec(); ig++)
    {
        if (fft.gvec_len(ig) <= wf_cutoff) num_gkvec++;
    }

    if (comm.rank() == 0)
    {
        printf("FFT dimensions: %i %i %i\n", fft.size(0), fft.size(1), fft.size(2));
        printf("num_gkvec: %i\n", num_gkvec);
        printf("num_fft_threads: %i\n", num_fft_threads);
        printf("num_fft_workers: %i\n", num_fft_workers);
    }

    splindex<block> spl_num_bands(num_bands, comm.size(), comm.rank());
    int nbnd = (int)spl_num_bands.local_size();

    matrix<double_complex> phi(num_gkvec, nbnd);
    matrix<double_complex> hphi;
    if (false)
    {
        hphi = matrix<double_complex>(&phi(0, 0), num_gkvec, nbnd);
    }
    else
    {
        hphi = matrix<double_complex>(num_gkvec, nbnd);
    }

    for (int i = 0; i < nbnd; i++) 
    {
        for (int ig = 0; ig < num_gkvec; ig++) phi(ig, i) = type_wrapper<double_complex>::random();
    }

    std::atomic_int ibnd;
    ibnd.store(0);

    //std::mutex ibnd_mutex;

    Timer t("apply_v_loc", comm);
    std::vector<std::thread> fft_threads;
    for (int thread_id = 0; thread_id < num_fft_threads; thread_id++)
    {
        fft_threads.push_back(std::thread([thread_id, nbnd, &ibnd, &fft, &phi, &hphi, num_gkvec]()
        {
            while (true)
            {
                int i = ibnd++;
                if (i >= nbnd) return;

                //== ibnd_mutex.lock();
                //== int i = ibnd;
                //== if (ibnd + 1 > num_phi) 
                //== {
                //==     ibnd_mutex.unlock();
                //==     return;
                //== }
                //== else
                //== {
                //==     ibnd++;
                //== }
                //== ibnd_mutex.unlock();
                
                fft.input(num_gkvec, fft.index_map(), &phi(0, i), thread_id);
                fft.transform(1, thread_id);

                for (int ir = 0; ir < fft.size(); ir++) fft.buffer(ir, thread_id) += 1.0;

                fft.transform(-1, thread_id);
                fft.output(num_gkvec, fft.index_map(), &hphi(0, i), thread_id);
            }
        }));
    }
    for (auto& thread: fft_threads) thread.join();
    double tval = t.stop();

    if (comm.rank() == 0)
    {
        printf("\n");
        printf("time: %f (sec.), performance: %f, (FFT/sec.)\n", tval, 2 * num_bands / tval);
    }
}

int main(int argn, char **argv)
{
    cmd_args args;
    args.register_key("--help", "print this help and exit");
    args.register_key("--alat=", "{double} lattice constant");
    args.register_key("--pw_cutoff=", "{double} plane-wave cutoff [a.u.^-1]");
    args.register_key("--wf_cutoff=", "{double} wave-function cutoff [a.u.^-1]");
    args.register_key("--num_bands=", "{int} number of bands");
    args.register_key("--num_fft_threads=", "{int} number of threads for independent FFTs");
    args.register_key("--num_fft_workers=", "{int} number of threads working on the same FFT");

    args.parse_args(argn, argv);
    if (args.exist("help"))
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        exit(0);
    }

    double alat = 7.0;
    double pw_cutoff = 16.0;
    double wf_cutoff = 5.0;
    int num_bands = 10;
    int num_fft_threads = omp_get_max_threads();
    int num_fft_workers = 1;

    alat = args.value<double>("alat", alat);
    pw_cutoff = args.value<double>("pw_cutoff", pw_cutoff);
    wf_cutoff = args.value<double>("wf_cutoff", wf_cutoff);
    num_bands = args.value<int>("num_bands", num_bands);
    num_fft_threads = args.value<int>("num_fft_threads", num_fft_threads);
    num_fft_workers = args.value<int>("num_fft_workers", num_fft_workers);

    Platform::initialize(1);
    
    for (int i = 0; i < 10; i++)
    {
        apply_v_loc(alat, wf_cutoff, num_bands, num_fft_threads, num_fft_workers);
    }

    Timer::print();

    Platform::finalize();
}
