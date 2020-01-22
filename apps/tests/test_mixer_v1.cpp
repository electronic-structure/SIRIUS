#include <sirius.h>
#include <numeric>

using namespace sirius;

void black_box(std::vector<double> const& v0, std::vector<double>& vout)
{
    for (size_t i = 0; i < v0.size(); i++) {
        double d = v0[i] - vout[i];
        vout[i] += d * std::exp(-std::abs(d));
    }
}

std::vector<double> get_values(int N__)
{
    std::vector<double> a(N__);
    std::generate(a.begin(), a.end(), [](){return utils::random<double>();});
    //double norm = std::accumulate(a.begin(), a.end(), 0.0);
    //std::transform(a.begin(), a.end(), a.begin(), [&](double v){return v / norm;});
    return a;
}

//void test1_mixer(int N, Mixer<double>& mixer)
//{
//    auto v0 = get_values(N);
//
//    auto a = get_values(N);
//
//    for (int i = 0; i < N; i++) {
//        mixer.input_local(i, a[i]);
//    }
//    mixer.initialize();
//
//    for (int i = 0; i < 30; i++) {
//        black_box(v0, a);
//        for (int i = 0; i < N; i++) {
//            mixer.input_local(i, a[i]);
//        }
//        auto rms = mixer.mix(0);
//        if (rms < 1e-12) {
//            break;
//        }
//        //auto rss = mixer.rss();
//        std::cout << "rms=" << rms << "\n";
//        for (int i = 0; i < N; i++) {
//            a[i] = mixer.output_local(i);
//        }
//    }
//    double diff{0};
//    for (size_t i = 0; i < v0.size(); i++) {
//        diff += std::abs(v0[i] - a[i]);
//    }
//    std::cout << "diff: " << diff << "\n";
//}

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

    //int N = 1000;

    //printf("testing linear mixer\n");
    //Mixer_input mix_cfg;
    //mix_cfg.type_ = "linear";
    //mix_cfg.beta_ = 0.5;
    //auto mixer = Mixer_factory<double>(0, N, mix_cfg, Communicator::world());
    //test1_mixer(N, *mixer);

    //printf("testing broyden1 mixer\n");
    //mix_cfg.type_ = "broyden1";
    //mix_cfg.beta_ = 0.5;
    //mixer = Mixer_factory<double>(0, N, mix_cfg, Communicator::world());
    //test1_mixer(N, *mixer);

    //printf("testing broyden2 mixer\n");
    //mix_cfg.type_ = "broyden2";
    //mix_cfg.beta_ = 1;
    //mixer = Mixer_factory<double>(0, N, mix_cfg, Communicator::world());
    //test1_mixer(N, *mixer);

    sirius::finalize();
}
