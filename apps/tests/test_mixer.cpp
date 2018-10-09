#include <sirius.h>
#include <numeric>

using namespace sirius;

std::vector<double> get_values(int N__)
{
    std::vector<double> a(N__);
    std::generate(a.begin(), a.end(), [](){return type_wrapper<double>::random();});
    double norm = std::accumulate(a.begin(), a.end(), 0.0);
    std::transform(a.begin(), a.end(), a.begin(), [&](double v){return v / norm;});
    return std::move(a);
}

void test1_mixer(int N, Mixer<double>& mixer)
{
    auto a = get_values(N);
    for (int i = 0; i < N; i++) {
        mixer.input_shared(i, a[i]);
    }
    mixer.initialize();

    auto b = get_values(N);
    for (int i = 0; i < N; i++) {
        mixer.input_shared(i, b[i]);
    }

    double rms = mixer.mix(1e-16);
    std::cout << "rms = " << rms << "\n";
    std::vector<double> c(N);

    for (int i = 0; i < N; i++) {
        c[i] = mixer.output_shared(i);
    }
    double norm = std::accumulate(c.begin(), c.end(), 0.0);
    std::cout << "norm of mixed vector = " << norm << "\n";

    for (int i = 0; i < N; i++) {
        if (c[i] != a[i]) {
            std::stringstream s;
            s << "a = " << a[i] << ", b = " << b[i] << ", c = " << c[i];
            TERMINATE(s);
        }
    }
}

int main(int argn, char** argv)
{
    cmd_args args;

    args.parse_args(argn, argv);
    if (args.exist("help"))
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(1);

    int N = 10;

    printf("testing linear mixer\n");
    Mixer_input mix_cfg;
    mix_cfg.type_ = "linear";
    mix_cfg.beta_ = 0.0;
    auto mixer = Mixer_factory<double>(N, 0, mix_cfg, Communicator::world());
    test1_mixer(N, *mixer);

    printf("testing broyden1 mixer\n");
    mix_cfg.type_ = "broyden1";
    mix_cfg.beta_ = 0.0;
    mixer = Mixer_factory<double>(N, 0, mix_cfg, Communicator::world());
    test1_mixer(N, *mixer);

    printf("testing broyden2 mixer\n");
    mix_cfg.type_ = "broyden2";
    mix_cfg.beta_ = 0.0;
    mixer = Mixer_factory<double>(N, 0, mix_cfg, Communicator::world());
    test1_mixer(N, *mixer);

    sirius::finalize();
}
