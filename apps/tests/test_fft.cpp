#include <sirius.h>
#include <thread>
#include <mutex>

using namespace sirius;

void test_fft_omp(vector3d<int>& dims__)
{
    printf("test of threaded FFTs (OMP version)\n");

    int max_num_threads = Platform::max_num_threads();
    std::vector< std::pair<int, int> > threads_conf;

    for (int i = 1; i <= max_num_threads; i++)
    {
        for (int j = 1; j <= max_num_threads; j++)
        {
            if (i * j < 2 * max_num_threads) threads_conf.push_back(std::pair<int, int>(i, j));
        }
    }

    Communicator comm(MPI_COMM_WORLD);

    matrix3d<double> reciprocal_lattice_vectors;
    for (int i = 0; i < 3; i++) reciprocal_lattice_vectors(i, i) = 1.0;

    for (int k = 0; k < (int)threads_conf.size(); k++)
    {
        printf("number of fft threads: %i, number of fft workers: %i\n", threads_conf[k].first, threads_conf[k].second);

        FFT3D<CPU> fft(dims__, threads_conf[k].first, threads_conf[k].second, MPI_COMM_SELF);
        Gvec gvec(vector3d<double>(0, 0, 0), 20.0, reciprocal_lattice_vectors, &fft, true);

        int num_phi = 160;
        mdarray<double_complex, 2> phi(gvec.num_gvec(), num_phi);
        for (int i = 0; i < num_phi; i++)
        {
            for (int ig = 0; ig < gvec.num_gvec(); ig++) phi(ig, i) = type_wrapper<double_complex>::random();
        }

        Timer t("fft_loop");
        #pragma omp parallel num_threads(threads_conf[k].first)
        {
            int thread_id = Platform::thread_id();
            
            #pragma omp for
            for (int i = 0; i < num_phi; i++)
            {
                fft.input(gvec.num_gvec(), gvec.index_map(), &phi(0, i), thread_id);
                fft.transform(1, thread_id);

                for (int ir = 0; ir < fft.size(); ir++) fft.buffer(ir, thread_id) += 1.0;

                fft.transform(-1, thread_id);
                fft.output(gvec.num_gvec(), gvec.index_map(), &phi(0, i), thread_id);
            }
        }
        double tval = t.stop();

        printf("performance: %f, (FFT/sec.)\n", 2 * num_phi / tval);
        comm.barrier();
    }
}

void test_fft_pthread(vector3d<int>& dims__)
{
    printf("test of threaded FFTs (pthread version)\n");

    int max_num_threads = Platform::max_num_threads();
    std::vector< std::pair<int, int> > threads_conf;

    for (int i = 1; i <= max_num_threads; i++)
    {
        for (int j = 1; j <= max_num_threads; j++)
        {
            if (i * j < 2 * max_num_threads) threads_conf.push_back(std::pair<int, int>(i, j));
        }
    }

    Communicator comm(MPI_COMM_WORLD);

    matrix3d<double> reciprocal_lattice_vectors;
    for (int i = 0; i < 3; i++) reciprocal_lattice_vectors(i, i) = 1.0;

    for (int k = 0; k < (int)threads_conf.size(); k++)
    {
        printf("number of fft threads: %i, number of fft workers: %i\n", threads_conf[k].first, threads_conf[k].second);

        FFT3D<CPU> fft(dims__, threads_conf[k].first, threads_conf[k].second, MPI_COMM_SELF);
        Gvec gvec(vector3d<double>(0, 0, 0), 20.0, reciprocal_lattice_vectors, &fft, true);

        int num_phi = 160;
        mdarray<double_complex, 2> phi(gvec.num_gvec(), num_phi);
        for (int i = 0; i < num_phi; i++)
        {
            for (int ig = 0; ig < gvec.num_gvec(); ig++) phi(ig, i) = type_wrapper<double_complex>::random();
        }

        std::atomic_int ibnd;
        ibnd.store(0);

        Timer t("fft_loop");
        std::vector<std::thread> fft_threads;
        for (int thread_id = 0; thread_id < threads_conf[k].first; thread_id++)
        {
            fft_threads.push_back(std::thread([thread_id, num_phi, &ibnd, &fft, &phi, &gvec]()
            {
                while (true)
                {
                    int i = ibnd++;
                    if (i >= num_phi) return;
                    
                    fft.input(gvec.num_gvec(), gvec.index_map(), &phi(0, i), thread_id);
                    fft.transform(1, thread_id);

                    for (int ir = 0; ir < fft.size(); ir++) fft.buffer(ir, thread_id) += 1.0;

                    fft.transform(-1, thread_id);
                    fft.output(gvec.num_gvec(), gvec.index_map(), &phi(0, i), thread_id);
                }
            }));
        }
        for (auto& thread: fft_threads) thread.join();
        double tval = t.stop();

        printf("performance: %f, (FFT/sec.)\n", 2 * num_phi / tval);
        comm.barrier();
    }
}

void test_fft_pthread_mutex(vector3d<int>& dims__)
{
    printf("\n");
    printf("test of threaded FFTs (pthread version, mutex lock)\n");

    int max_num_threads = Platform::max_num_threads();
    std::vector< std::pair<int, int> > threads_conf;

    for (int i = 1; i <= max_num_threads; i++)
    {
        for (int j = 1; j <= max_num_threads; j++)
        {
            if (i * j < 2 * max_num_threads) threads_conf.push_back(std::pair<int, int>(i, j));
        }
    }

    Communicator comm(MPI_COMM_WORLD);

    matrix3d<double> reciprocal_lattice_vectors;
    for (int i = 0; i < 3; i++) reciprocal_lattice_vectors(i, i) = 1.0;

    for (int k = 0; k < (int)threads_conf.size(); k++)
    {
        printf("number of fft threads: %i, number of fft workers: %i\n", threads_conf[k].first, threads_conf[k].second);

        FFT3D<CPU> fft(dims__, threads_conf[k].first, threads_conf[k].second, MPI_COMM_SELF);
        Gvec gvec(vector3d<double>(0, 0, 0), 20.0, reciprocal_lattice_vectors, &fft, true);

        int num_phi = 160;
        mdarray<double_complex, 2> phi(gvec.num_gvec(), num_phi);
        for (int i = 0; i < num_phi; i++)
        {
            for (int ig = 0; ig < gvec.num_gvec(); ig++) phi(ig, i) = type_wrapper<double_complex>::random();
        }

        int ibnd = 0;
        std::mutex ibnd_mutex;

        Timer t("fft_loop");
        std::vector<std::thread> fft_threads;
        for (int thread_id = 0; thread_id < threads_conf[k].first; thread_id++)
        {
            fft_threads.push_back(std::thread([thread_id, num_phi, &ibnd, &ibnd_mutex, &fft, &phi, &gvec]()
            {
                while (true)
                {
                    ibnd_mutex.lock();
                    int i = ibnd;
                    if (ibnd + 1 > num_phi) 
                    {
                        ibnd_mutex.unlock();
                        return;
                    }
                    else
                    {
                        ibnd++;
                    }
                    ibnd_mutex.unlock();
                    
                    fft.input(gvec.num_gvec(), gvec.index_map(), &phi(0, i), thread_id);
                    fft.transform(1, thread_id);

                    for (int ir = 0; ir < fft.size(); ir++) fft.buffer(ir, thread_id) += 1.0;

                    fft.transform(-1, thread_id);
                    fft.output(gvec.num_gvec(), gvec.index_map(), &phi(0, i), thread_id);
                }
            }));
        }
        for (auto& thread: fft_threads) thread.join();
        double tval = t.stop();

        printf("performance: %f, (FFT/sec.)\n", 2 * num_phi / tval);
        comm.barrier();
    }
}

int main(int argn, char **argv)
{
    cmd_args args;
    args.register_key("--dims=", "{vector3d<int>} FFT dimensions");

    args.parse_args(argn, argv);
    if (argn == 1)
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        exit(0);
    }

    vector3d<int> dims = args.value< vector3d<int> >("dims");

    Platform::initialize(1);

    test_fft_omp(dims);
    test_fft_pthread(dims);
    //test_fft_pthread_mutex(dims);

    Platform::finalize();
}
