#include "sddk.hpp"

using namespace sddk;

using ftn_int = int32_t;
using ftn_double = double;
using ftn_double_complex = std::complex<double>;

std::vector<void*> sddk_objects;

inline Communicator const& map_fcomm(ftn_int fcomm__)
{
    static std::map<int, std::unique_ptr<Communicator>> fcomm_map;
    if (!fcomm_map.count(fcomm__)) {
        fcomm_map[fcomm__] = std::unique_ptr<Communicator>(new Communicator(MPI_Comm_f2c(fcomm__)));
    }

    auto& comm = *fcomm_map[fcomm__];
    return comm;
}

inline int get_next_free_object_id()
{
    for (int i = 0; i < static_cast<int>(sddk_objects.size()); i++) {
        if (sddk_objects[i] == nullptr) {
            return i;
        }
    }
    sddk_objects.push_back(nullptr);
    return static_cast<int>(sddk_objects.size() - 1);
}

extern "C" {

void sddk_init();

/// Create FFT grid.
void sddk_create_fft_grid(ftn_int* dims__,
                          ftn_int* fft_grid_id__) 
{
    *fft_grid_id__ = get_next_free_object_id();
    sddk_objects[*fft_grid_id__] = new FFT3D_grid({dims__[0], dims__[1], dims__[2]});
}

/// Delete FFT grid.
void sddk_delete_fft_grid(ftn_int* fft_grid_id__)
{
    delete reinterpret_cast<FFT3D_grid*>(sddk_objects[*fft_grid_id__]);
    sddk_objects[*fft_grid_id__] = nullptr;
}

/// Create list of G-vectors.
void sddk_create_gvec(ftn_double* vk__,
                      ftn_double* b1__,
                      ftn_double* b2__,
                      ftn_double* b3__,
                      ftn_double* gmax__,
                      ftn_int*    reduce_gvec__,
                      ftn_int*    fcomm__,
                      ftn_int*    fcomm_fft__,
                      ftn_int*    gvec_id__)
{
    auto& comm = map_fcomm(*fcomm__);
    auto& comm_fft = map_fcomm(*fcomm_fft__);

    bool reduce_gvec = (*reduce_gvec__ == 0) ? false : true;

    matrix3d<double> lat_vec;
    for (int x: {0, 1, 2}) {
        lat_vec(x, 0) = b1__[x];
        lat_vec(x, 1) = b2__[x];
        lat_vec(x, 2) = b3__[x];
    }

    *gvec_id__ = get_next_free_object_id();
    sddk_objects[*gvec_id__] = new Gvec({vk__[0], vk__[1], vk__[2]},
                                        lat_vec,
                                        *gmax__, 
                                        comm,
                                        comm_fft,
                                        reduce_gvec);
}

/// Delete list of G-vectors.
void sddk_delete_gvec(ftn_int* gvec_id__)
{
    delete reinterpret_cast<Gvec*>(sddk_objects[*gvec_id__]);
    sddk_objects[*gvec_id__] = nullptr;
}

/// Create FFT driver.
void sddk_create_fft(ftn_int* fft_grid_id__,
                     ftn_int* fcomm__,
                     ftn_int* fft_id__)
{
    auto& comm = map_fcomm(*fcomm__);
    auto& fft_grid = *reinterpret_cast<FFT3D_grid*>(sddk_objects[*fft_grid_id__]);
    
    *fft_id__ = get_next_free_object_id();
    sddk_objects[*fft_id__] = new FFT3D(fft_grid, comm, device_t::CPU);
}

/// Delete fft driver.
void sddk_delete_fft(ftn_int* fft_id__)
{
    delete reinterpret_cast<FFT3D*>(sddk_objects[*fft_id__]);
    sddk_objects[*fft_id__] = nullptr;
}

/// Get total number of G-vectors.
void sddk_get_num_gvec(ftn_int* gvec_id__, ftn_int* num_gvec__)
{

}

/// Get local number of G-vectors in the fine-graind distribution.
void sddk_get_gvec_count(ftn_int* gvec_id__,
                         ftn_int* rank__,
                         ftn_int* gvec_count__)
{
    *gvec_count__ = reinterpret_cast<Gvec*>(sddk_objects[*gvec_id__])->gvec_count(*rank__);
}

/// Get index offset of G-vectors in the fine-graind distribution.
void sddk_get_gvec_offset(ftn_int* gvec_id__,
                          ftn_int* rank__,
                          ftn_int* gvec_offset__)
{
    *gvec_offset__ = reinterpret_cast<Gvec*>(sddk_objects[*gvec_id__])->gvec_offset(*rank__);
}

/// Get local number of G-vectors for the FFT.
void sddk_get_gvec_count_fft(ftn_int* gvec_id__,
                             ftn_int* gvec_count__)
{
    *gvec_count__ = reinterpret_cast<Gvec*>(sddk_objects[*gvec_id__])->partition().gvec_count_fft();
}

/// Get index offset of G-vectors for the FFT.
void sddk_get_gvec_offset_fft(ftn_int* gvec_id__,
                              ftn_int* gvec_offset__)
{
    *gvec_offset__ = reinterpret_cast<Gvec*>(sddk_objects[*gvec_id__])->partition().gvec_offset_fft();
}

void sddk_fft(ftn_int*            fft_id__,
              ftn_int*            gvec_id__,
              ftn_int*            direction__,
              ftn_double_complex* data__)
{
    auto gv = reinterpret_cast<Gvec*>(sddk_objects[*gvec_id__]);
    switch (*direction__) {
        case 1: {
            reinterpret_cast<FFT3D*>(sddk_objects[*fft_id__])->transform<1>(gv->partition(), data__);
            break;
        }
        case -1: {
            reinterpret_cast<FFT3D*>(sddk_objects[*fft_id__])->transform<-1>(gv->partition(), data__);
            break;
        }
        default: {
            TERMINATE("wrong FFT direction");
        }
    }
}

void sddk_fft_prepare(ftn_int* fft_id__,
                      ftn_int* gvec_id__)
{
    auto gv = reinterpret_cast<Gvec*>(sddk_objects[*gvec_id__]);
    reinterpret_cast<FFT3D*>(sddk_objects[*fft_id__])->prepare(gv->partition());
}

void sddk_fft_dismiss(ftn_int* fft_id__)
{
    reinterpret_cast<FFT3D*>(sddk_objects[*fft_id__])->dismiss();
}

void sddk_print_timers()
{
    timer::print();
}


}
