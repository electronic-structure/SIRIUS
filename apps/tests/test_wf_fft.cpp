#include <sirius.hpp>
#include "SDDK/wave_functions.hpp"
#include "mpi/mpi_grid.hpp"

using namespace sirius;

void test_wf_fft()
{
    //sddk::MPI_grid mpi_grid({2, 3}, sddk::Communicator::world());
    mpi::Grid mpi_grid({1, 1}, mpi::Communicator::world());

    /* creation of simple G+k vector set */
    auto gkvec = sddk::gkvec_factory(8.0, mpi_grid.communicator());
    std::cout << "num_gvec=" << gkvec->num_gvec() << std::endl;
    /* creation of G+k set for FFTt */
    auto gkvec_fft = std::make_shared<sddk::Gvec_fft>(*gkvec, mpi_grid.communicator(1 << 0), mpi_grid.communicator(1 << 1));

    /* get the FFT box boundaries */
    auto fft_grid = fft::get_min_grid(8.0, gkvec->lattice_vectors());

    std::vector<int> num_mt_coeffs({10, 20, 30, 10, 20});

    wf::Wave_functions<double> wf(gkvec, num_mt_coeffs, wf::num_mag_dims(1), wf::num_bands(10), sddk::memory_t::host);
    wf::Wave_functions<double> wf_ref(gkvec, num_mt_coeffs, wf::num_mag_dims(1), wf::num_bands(10), sddk::memory_t::host);

    for (int ispn = 0; ispn < 2; ispn++) {
        for (int i = 0; i < 10; i++) {
            for (int ig = 0; ig < gkvec->count(); ig++) {
                wf.pw_coeffs(ig, wf::spin_index(ispn), wf::band_index(i)) =
                    wf_ref.pw_coeffs(ig, wf::spin_index(ispn), wf::band_index(i)) =
                        utils::random<std::complex<double>>();
            }
        }
    }
    auto mg = wf.memory_guard(sddk::memory_t::device, wf::copy_to::device);

    auto pu = sddk::device_t::CPU;

    auto spfft_pu = pu == sddk::device_t::CPU ? SPFFT_PU_HOST : SPFFT_PU_GPU;
    auto spl_z = split_fft_z(fft_grid[2], gkvec_fft->comm_fft());

    /* create spfft buffer for coarse transform */
    auto spfft_grid = std::unique_ptr<spfft::Grid>(new spfft::Grid(
            fft_grid[0], fft_grid[1], fft_grid[2], gkvec_fft->zcol_count_fft(),
            spl_z.local_size(), spfft_pu, -1, gkvec_fft->comm_fft().native(), SPFFT_EXCH_DEFAULT));

    const auto fft_type = gkvec->reduced() ? SPFFT_TRANS_R2C : SPFFT_TRANS_C2C;

    /* create actual transform object */
    auto spfft_transform = std::make_unique<spfft::Transform>(spfft_grid->create_transform(
        spfft_pu, fft_type, fft_grid[0], fft_grid[1], fft_grid[2],
        spl_z.local_size(), gkvec_fft->gvec_count_fft(), SPFFT_INDEX_TRIPLETS,
        gkvec_fft->gvec_array().at(sddk::memory_t::host)));

    std::array<wf::Wave_functions_fft<double>, 2> wf1;
    for (int ispn = 0; ispn < 2; ispn++) {
        wf1[ispn] = wf::Wave_functions_fft<double>(gkvec_fft, wf, wf::spin_index(ispn), wf::band_range(0,10),
                wf::shuffle_to::fft_layout);
    }

    for (int ispn = 0; ispn < 2; ispn++) {

        wf::Wave_functions_fft<double> wf_fft(gkvec_fft, wf, wf::spin_index(ispn), wf::band_range(0,10),
                wf::shuffle_to::fft_layout | wf::shuffle_to::wf_layout);

        for (int i = 0; i < wf_fft.num_wf_local(); i++) {
            spfft_transform->backward(wf_fft.pw_coeffs_spfft(sddk::memory_t::host, wf::band_index(i)), spfft_pu);
            spfft_transform->forward(spfft_pu, wf_fft.pw_coeffs_spfft(sddk::memory_t::host, wf::band_index(i)), SPFFT_FULL_SCALING);
        }
    }

    for (int ispn = 0; ispn < 2; ispn++) {
        for (int i = 0; i < 10; i++) {
            for (int ig = 0; ig < gkvec->count(); ig++) {
                if (std::abs(wf.pw_coeffs(ig, wf::spin_index(ispn), wf::band_index(i)) -
                             wf_ref.pw_coeffs(ig, wf::spin_index(ispn), wf::band_index(i))) > 1e-10) {
                    std::cout << "Error!" << std::endl;
                }
            }
        }
    }
}


int main(int argn, char** argv)
{
    sirius::initialize(1);
    test_wf_fft();
    sirius::finalize();
}
