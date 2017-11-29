#include "sddk.hpp"

using namespace sddk;

using ftn_int            = int32_t;
using ftn_double         = double;
using ftn_double_complex = std::complex<double>;

/// List of all allocated objects.
std::vector<void*> sddk_objects;

/// Mapping between object id and its class name.
std::map<int, std::string> sddk_objects_class_name;

/// Mapping between Fortran and SIRIUS MPI communicators.
inline Communicator const& map_fcomm(ftn_int fcomm__)
{
    static std::map<int, std::unique_ptr<Communicator>> fcomm_map;
    if (!fcomm_map.count(fcomm__)) {
        fcomm_map[fcomm__] = std::unique_ptr<Communicator>(new Communicator(MPI_Comm_f2c(fcomm__)));
    }

    auto& comm = *fcomm_map[fcomm__];
    return comm;
}

/// Get a free slot int the list of sddk objects.
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

void sddk_init()
{
    /* something useful can be done here */
}

/// Delete allocated object.
void sddk_delete_object(ftn_int* object_id__)
{
    int id = *object_id__;
    void* ptr = sddk_objects[id];

    if (sddk_objects_class_name[id] == "FFT3D_grid") {
        delete reinterpret_cast<FFT3D_grid*>(ptr);
    } else if (sddk_objects_class_name[id] == "Gvec") {
        delete reinterpret_cast<Gvec*>(ptr);
    } else if (sddk_objects_class_name[id] == "FFT3D") {
        delete reinterpret_cast<FFT3D*>(ptr);
    } else if (sddk_objects_class_name[id] == "Wave_functions") {
        delete reinterpret_cast<Wave_functions*>(ptr);
    } else {
        std::stringstream s;
        s << "wrong class name (" << sddk_objects_class_name[id] << ") for object id " << id;
        throw std::runtime_error(s.str());
    }
    sddk_objects[id] = nullptr;
    sddk_objects_class_name[id] = "";
}

/// Create FFT grid.
void sddk_create_fft_grid(ftn_int* dims__,
                          ftn_int* new_object_id__) 
{
    int id = get_next_free_object_id();
    sddk_objects[id] = new FFT3D_grid({dims__[0], dims__[1], dims__[2]});
    sddk_objects_class_name[id] = "FFT3D_grid";
    *new_object_id__ = id;
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
                      ftn_int*    new_object_id__)
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

    int id = get_next_free_object_id();
    sddk_objects[id] = new Gvec({vk__[0], vk__[1], vk__[2]}, lat_vec, *gmax__, comm, comm_fft, reduce_gvec);
    sddk_objects_class_name[id] = "Gvec";
    *new_object_id__ = id;
}

/// Create FFT driver.
void sddk_create_fft(ftn_int* fft_grid_id__,
                     ftn_int* fcomm__,
                     ftn_int* new_object_id__)
{
    auto& comm = map_fcomm(*fcomm__);
    auto& fft_grid = *reinterpret_cast<FFT3D_grid*>(sddk_objects[*fft_grid_id__]);
    
    int id = get_next_free_object_id();
    sddk_objects[id] = new FFT3D(fft_grid, comm, device_t::CPU);
    sddk_objects_class_name[id] = "FFT3D";
    *new_object_id__ = id;
}

/// Create wave functions.
void sddk_create_wave_functions(ftn_int* gkvec_id__,
                                ftn_int* num_wf__,
                                ftn_int* new_object_id__)
{
    int id = get_next_free_object_id();
    auto& gkvec = *reinterpret_cast<Gvec*>(sddk_objects[*gkvec_id__]);

    TERMINATE("pass number of spins");
    sddk_objects[id] = new Wave_functions(gkvec, *num_wf__, 1);
    sddk_objects_class_name[id] = "Wave_functions";
    *new_object_id__ = id;
    //auto& wf = *reinterpret_cast<Wave_functions*>(sddk_objects[id]);
    //wf.pw_coeffs().prime(0, 0) = double_complex(12.13, 14.15);
    //wf.pw_coeffs().prime(0, 1) = double_complex(1, 2);
}

void sddk_remap_wave_functions_forward(ftn_int* wf_id__, ftn_int* n__, ftn_int* idx0__)
{
    //auto& wf = *reinterpret_cast<Wave_functions*>(sddk_objects[*wf_id__]);
    STOP();
    //wf.pw_coeffs().remap_forward(CPU, wf.gkvec().partition().gvec_fft_slab(), *n__, *idx0__ - 1);
}

void sddk_remap_wave_functions_backward(ftn_int* wf_id__, ftn_int* n__, ftn_int* idx0__)
{
    //auto& wf = *reinterpret_cast<Wave_functions*>(sddk_objects[*wf_id__]);
    STOP();
    //wf.pw_coeffs().remap_backward(CPU, wf.gkvec().partition().gvec_fft_slab(), *n__, *idx0__ - 1);
}

void sddk_get_num_wave_functions(ftn_int* wf_id__, ftn_int* num_wf__)
{
    auto& wf = *reinterpret_cast<Wave_functions*>(sddk_objects[*wf_id__]);
    *num_wf__ = wf.num_wf();
}

void sddk_get_num_wave_functions_local(ftn_int* wf_id__, ftn_int* num_wf__)
{
    STOP();
    //auto& wf = *reinterpret_cast<Wave_functions*>(sddk_objects[*wf_id__]);
    //*num_wf__ = wf.pw_coeffs().spl_num_col().local_size();
}

void sddk_get_wave_functions_prime_ld(ftn_int* wf_id__, ftn_int* ld__)
{
    auto& wf = *reinterpret_cast<Wave_functions*>(sddk_objects[*wf_id__]);
    *ld__ = wf.pw_coeffs().prime().ld();
}

void sddk_get_wave_functions_extra_ld(ftn_int* wf_id__, ftn_int* ld__)
{
    auto& wf = *reinterpret_cast<Wave_functions*>(sddk_objects[*wf_id__]);
    *ld__ = wf.pw_coeffs().extra().ld();
}

void sddk_get_wave_functions_prime_ptr(ftn_int* wf_id__,
                                       ftn_double_complex** ptr__)
{
    auto& wf = *reinterpret_cast<Wave_functions*>(sddk_objects[*wf_id__]);
    *ptr__ = wf.pw_coeffs().prime().at<CPU>();
}

void sddk_get_wave_functions_extra_ptr(ftn_int* wf_id__,
                                       ftn_double_complex** ptr__)
{
    auto& wf = *reinterpret_cast<Wave_functions*>(sddk_objects[*wf_id__]);
    *ptr__ = wf.pw_coeffs().extra().at<CPU>();
}

/// Get total number of G-vectors.
void sddk_get_num_gvec(ftn_int* gvec_id__, ftn_int* num_gvec__)
{
    *num_gvec__ = reinterpret_cast<Gvec*>(sddk_objects[*gvec_id__])->num_gvec();
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
              ftn_int*            direction__,
              ftn_double_complex* data__)
{
    switch (*direction__) {
        case 1: {
            reinterpret_cast<FFT3D*>(sddk_objects[*fft_id__])->transform<1>(data__);
            break;
        }
        case -1: {
            reinterpret_cast<FFT3D*>(sddk_objects[*fft_id__])->transform<-1>(data__);
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
